"""The common fit/plan/repair path used by experiments, application and learning."""
from dataclasses import dataclass
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from .io import read,write,append,plain
from .operations import fit_bundle,apply_policy,apply_plan,available,candidate_mask,numeric
from .profile import target_association
from .planner import schema
from .scoring import SourceScorer,fit_predictor,predict
from . import __version__

CANDIDATE_COLUMNS=['row_id','column','observed_value','candidate_status','candidate_basis','suggested_value',
    'suggestion_status','suggestion_source','selected_policy','actual_value','treatment_status']


def payload(x,column,bundle,task,include_task,best_plan):
    s=bundle['state'];values=numeric(x[column]);mask=candidate_mask(x,s)
    missing=int(values.isna().sum());outliers=int((mask & values.notna()).sum())
    result={'background':task.get('background',''),'column':column,
        'evidence':{'rows':len(x),'missing_count':missing,'iqr_nonmissing_candidates':outliers,
            'observed_min':plain(values.min()),'observed_max':plain(values.max()),'iqr_lower':s['lower'],
            'iqr_upper':s['upper'],'candidate_status':'suspected','fact_summary':f'缺失={missing};IQR={outliers};'},
        'other_field_associations_to_repair_field':[
            {'column':c,**target_association(x.loc[~mask,c],values.loc[~mask],'regression')} for c in x if c!=column],
        'imputer':{'type':'fixed conditional gradient-boosted regression',
            'trusted_training_rows':len(s['trusted_row_ids']),'predictor_columns':s['feature_columns']},
        'downstream_type':task['type']}
    if include_task:result['task']={k:v for k,v in task.items() if k!='background'}
    if len(best_plan)>1:result['current_plan']=best_plan
    return result


def feedback(column,proposal,evaluation,raw,best,accepted):
    result={'previous_strategy':proposal[column],'previous_scores':evaluation['scores'],
        'baseline_scores':raw['scores'],'accepted':accepted,'best_strategy':best['plan'][column],
        'best_scores':best['scores'],'candidate_report_retained':True}
    if len(proposal)>1:result.update(previous_plan=proposal,best_plan=best['plan'])
    return result


@dataclass
class FittedPipeline:
    bundles: dict
    plan: dict
    predictor: dict
    features: list
    metadata: dict

    def transform(self,x):return apply_plan(x[self.features],self.plan,self.bundles)[0]

    def predict(self,x):return predict(self.predictor,self.transform(x))

    def report(self,x):
        x=x[self.features];actual=self.transform(x);rows=[]
        for column,bundle in self.bundles.items():
            mask=candidate_mask(x,bundle['state']);actions=available(bundle)
            suggestion='model_repair' if 'model_repair' in actions else 'median_repair' if 'median_repair' in actions else None
            suggested=apply_policy(x,suggestion,bundle)[0] if suggestion else None
            for row in x.index[mask]:
                old,new=x.loc[row,column],actual.loc[row,column]
                changed=not np.isclose(old,new,rtol=0,atol=1e-9,equal_nan=True)
                policy=self.plan[column];flag=policy=='flag_only' or policy.endswith('_with_flag')
                rows.append(plain({'row_id':row,'column':column,'observed_value':old,
                    'candidate_status':'suspected','candidate_basis':'missing_or_nonfinite' if pd.isna(numeric(x[column]).loc[row]) else 'training_IQR',
                    'suggested_value':suggested.loc[row,column] if suggested is not None else None,
                    'suggestion_status':'estimated' if suggestion else 'unavailable','suggestion_source':suggestion,
                    'selected_policy':policy,'actual_value':new,
                    'treatment_status':'modified' if changed else 'flagged' if flag else 'not_applied'}))
        return rows

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(self,path)


def load_pipeline(path):return joblib.load(path)


def fit_pipeline(x,y,task,config,planner,output_dir,reference=None,groups=None,scorer=None,metadata=None):
    """Fit on training features/labels only. Reference is scoring-only injection truth."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Completed pipeline exists; load it or select a new output directory')
    if not x.index.is_unique or task['target'] in x:raise ValueError('Unique row identities and target-free features are required')
    if not x.index.equals(y.index):raise ValueError('Feature and label indices differ')
    meta={**(metadata or {}),'task':task['name'],'system_version':__version__};columns=config['repair_columns']
    source={'x':x,'y':y,'task':task,'reference':reference if reference is not None else x,
        'groups':groups,'seed':config.get('seed',42),'config':config,'metadata':meta}
    scorer=scorer or SourceScorer(out/'scorer',source)
    bundle_path=scorer.directory/'outer_bundles.joblib'
    if not bundle_path.exists():
        saved={c:fit_bundle(x,c,config['imputer']) for c in columns};joblib.dump(saved,bundle_path)
        write(bundle_path.with_suffix('.json'),{'fitted_imputers':sum(b['model'] is not None for b in saved.values())})
    bundles=joblib.load(bundle_path);actions={c:set(available(bundles[c])) for c in columns}
    for i in range(len(scorer.fold_splits)):
        inner=scorer.bundles(i)
        for c in columns:actions[c]&=set(available(inner[c]))
    ordered={c:[p for p in available(bundles[c]) if p in actions[c]] for c in columns}
    keep=dict.fromkeys(columns,'keep');raw=scorer.evaluate(keep)
    probe={c:next(a for a in ['model_repair_with_flag','median_repair_with_flag','flag_only','keep'] if a in ordered[c]) for c in columns}
    initial=scorer.evaluate(probe);accepted=initial['utility']>raw['utility']+1e-12
    best=initial if accepted else raw;history=[]
    write(out/'probe.json',{'plan':probe,'evaluation':initial,'accepted':accepted,'source':'fixed_code_exploration_not_LM_action'})
    last_proposal,last_evaluation,last_accepted=probe,initial,accepted
    for column in columns:
        fb=feedback(column,last_proposal,last_evaluation,raw,best,last_accepted)
        for iteration in range(2):
            context={'column':column,'round':iteration,'payload':payload(x,column,bundles[column],task,config.get('include_task',True),best['plan']),
                'feedback':fb,'schema':schema(ordered[column]),'best_plan':best['plan'],
                'raw_utility':raw['utility'],'known_best_utility':best['utility'],
                'scorer_dir':os.path.relpath(scorer.directory,out),'metadata':meta}
            append(out/'contexts.jsonl',context)
            record=planner.generate(context,out/'calls'/column/str(iteration))
            if not record['format_valid']:
                history.append({'column':column,'round':iteration,'accepted':False,'error':record.get('error'),'format_valid':False})
                break
            proposal={**best['plan'],column:record['policy']};evaluation=scorer.evaluate(proposal)
            previous_best=best['utility'];accepted=evaluation['utility']>previous_best+1e-12
            if accepted:best=evaluation
            row={'column':column,'round':iteration,'plan':proposal,'evaluation':evaluation,'accepted':accepted,
                'known_best_utility':previous_best,'format_valid':True,'facts_match':record['facts_match']}
            history.append(row)
            append(out/'trajectories.jsonl',{'messages':record['messages']+[{'role':'assistant','content':record['raw_output']}],
                'metadata':{**meta,'column':column,'round':iteration,'model_generated':record['model_generated'],
                    'format_valid':True,'facts_match':record['facts_match'],'utility':evaluation['utility'],
                    'raw_utility':raw['utility'],'known_best_utility':previous_best,'policy':record['policy'],
                    'planner':planner.config,'origin':str(Path('calls')/column/str(iteration)/'response.json')}})
            fb=feedback(column,proposal,evaluation,raw,best,accepted)
            last_proposal,last_evaluation,last_accepted=proposal,evaluation,accepted
    # Freeze every decision before fitting/scoring the outer predictor.
    write(out/'selection.json',{'selected_plan':best['plan'],'inner_scores':best['scores'],'raw_scores':raw['scores'],'history':history})
    transformed,events=apply_plan(x,best['plan'],bundles)
    predictor=fit_predictor(transformed,y,task,config['model'])
    result=FittedPipeline(bundles,best['plan'],predictor,list(x.columns),meta)
    result.save(out/'pipeline.joblib');write(out/'events.json',events)
    report=result.report(x);write(out/'candidates.json',report)
    pd.DataFrame(report,columns=CANDIDATE_COLUMNS).to_csv(out/'candidates.csv',index=False)
    transformed.to_csv(out/'processed_training.csv',index_label='row_id')
    write(out/'complete.json',{'status':'complete','metadata':meta,'plan':best['plan'],
        'new_inner_predictor_fits':scorer.new_fits,'outer_predictor_fits':1,'candidate_cells':len(report)})
    return result
