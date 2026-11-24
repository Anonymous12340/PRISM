"""Formal E2 execution: frozen tables, prefix budgets, training-only selection."""
from copy import deepcopy
import datetime
from pathlib import Path
import os
import random
import subprocess
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import e2_library as lib
from .e2_data import config,prepare
from .e2_pilot import Scorer,Engine,run_method,key,METHODS,RAW,run_native_command,native_time_limit
from .e1 import metrics,fit_preprocessor
from .e2_operations import normalize,apply_code,apply_classical_clean
from .scoring import splits,fit_predictor,predict,fit_encoder,encode
from .operations import fit_bundle
from .profile import profile_table
from .io import read,write,plain


class FormalScorer(Scorer):
    def __init__(self,c,unit):
        super().__init__(c,unit);self._evidence={}
        models=[]
        for operation in self.menu:
            if operation['kind']!='model_repair':continue
            column=operation['column'];parts=[self.x.iloc[f['train']][column] for f in self.folds] or [self.x[column]]
            enough=True
            for values in parts:
                q1,q3=values.quantile([.25,.75]);trusted=values.notna()&values.between(q1-1.5*(q3-q1),q3+1.5*(q3-q1))
                if trusted.sum()<30:enough=False;break
            if enough:models.append(operation)
        candidates=models or [o for o in self.menu if o['kind']=='fill']
        self.probe={'kind':'plan','ids':[candidates[0]['id']] if candidates else []}
        self.native_seconds_limit=c['native_timeout']

    def formal_evidence(self,include_task):
        if include_task in self._evidence:return deepcopy(self._evidence[include_task])
        prof=profile_table(self.x);columns=[]
        for item in prof['column_profiles']:
            row={k:item[k] for k in ['name','kind','missing_count','unique_count']}
            for k in ['quantiles','iqr_outliers']:
                if k in item:row[k]=item[k]
            if 'top_values' in item:row['top_values']=item['top_values'][:3]
            columns.append(row)
        result={'rows':len(self.x),'columns':columns,'samples':prof['sample_rows'][:5],
            'task_type':self.task['type'],'date_encoding':self.source.get('date_columns',{}),
            'constraints':'All target/identifier/leakage columns were excluded by the common dataset specification.'}
        if include_task:
            result['task']=self.task;result['background']=self.source['pair']['background']
            correlations={}
            for col in self.x.select_dtypes(include=np.number):
                r=self.x[col].corr(self.y,method='spearman')
                if np.isfinite(r):correlations[col]=float(r)
            result['training_only_target_spearman']=correlations
        # No-task arm has neither the target description nor a background that names it.
        self._evidence[include_task]=plain(result);return deepcopy(self._evidence[include_task])

    def transform(self,spec,train,valid,y,groups,slot,where):
        if spec['kind']!='plan':
            if spec['kind'] in ['classical_clean','clean_then_openfe'] and not len(train.select_dtypes(include=np.number).columns):
                return self._categorical_clean(spec,train,valid,y,groups,where)
            return super().transform(spec,train,valid,y,groups,slot,where)
        operations=[self.ops[i] for i in spec['ids']];bundles={}
        for op in operations:
            if op['kind']!='model_repair':continue
            col=op['column'];path=self.unit/'repair_bundles'/(str(slot)+'_'+key(col)+'.joblib')
            if not path.exists():
                value=fit_bundle(train,col);path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(value,path)
                write(path.with_suffix('.json'),{'column':col,'training_row_ids':train.index.tolist(),'fitted_imputers':int(value['model'] is not None)})
            bundles[col]=joblib.load(path)
        state,a,keep=lib.fit_plan(train,operations,bundles);b=lib.apply_plan(valid,state)
        where.mkdir(parents=True,exist_ok=True);joblib.dump({'kind':'plan','steps':state},where/'transform_state.joblib')
        pd.testing.assert_frame_equal(b,lib.apply_plan(valid,joblib.load(where/'transform_state.joblib')['steps']))
        if len(valid)>1:pd.testing.assert_frame_equal(b.iloc[:2],lib.apply_plan(valid.iloc[:2],state))
        write(where/'transform_check.json',{'passed':True,'training_rows_before':len(train),'training_rows_after':len(a),
            'training_row_ids':a.index.tolist(),'validation_row_ids':b.index.tolist(),'features':list(a),
            'state_saved':True,'batch_invariance_checked':True,'test_opened':False})
        write(where/'operation_log.json',{'operations':operations,'deleted_training_row_ids':train.index[~keep].tolist(),
            'evaluation_rows_deleted':0,'new_features':[v for v in a if v not in train],
            'imputer_components_fit_on_original_training_fold':bool(bundles)})
        return a,b,np.asarray(y)[keep]

    def _categorical_clean(self,spec,train,valid,y,groups,where):
        from sklearn.ensemble import IsolationForest
        encoder=fit_encoder(train);matrix=encode(train,encoder)
        forest=IsolationForest(n_estimators=100,contamination=.05,random_state=0,n_jobs=2)
        keep=forest.fit_predict(np.nan_to_num(matrix))==1;a=train.loc[keep].copy();b=valid.copy()
        state={'encoder':encoder,'forest':forest,'numeric':[],'training_keep':keep}
        if spec['kind']=='clean_then_openfe':
            a,b,native=self.native(a,b,np.asarray(y)[keep],np.asarray(groups)[keep],'openfe',where)
            state={'cleaning':state,'native':native}
        where.mkdir(parents=True,exist_ok=True);joblib.dump(state,where/'transform_state.joblib')
        write(where/'transform_check.json',{'passed':True,'training_rows_before':len(train),'training_rows_after':len(a),
            'training_row_ids':a.index.tolist(),'validation_row_ids':b.index.tolist(),'features':list(a),
            'state_saved':True,'batch_invariance_checked':False,'test_opened':False,
            'categorical_only_note':'IF on common train-fitted one-hot; no numeric columns require KNN.'})
        return a,b,np.asarray(y)[keep]


def unit_path(c,name,seed,fold):return Path(c['output_dir'])/'units'/name/str(seed)/str(fold)


def prepare_unit(c,name,seed,fold):
    if c.get('planning_scope')=='complete_development_only' and str(fold)!='final':
        raise ValueError('Main E2 permits only complete-development planning, not outer folds')
    unit=unit_path(c,name,seed,fold)
    if (unit/'source.joblib').exists():return unit
    item=next(d for d in c['datasets'] if d['name']==name)
    base=Path(c['data_dir'])/'prepared'/name/str(seed);development=joblib.load(base/'development.joblib')
    if str(fold)=='final':source=development
    else:
        positions=read(base/'folds.json')[int(fold)-1];a,b=positions['train'],positions['validation'];source=deepcopy(development)
        for k in ['x','y','groups']:source[k]=development[k].iloc[a]
        unit.mkdir(parents=True,exist_ok=True)
        joblib.dump({'x':development['x'].iloc[b],'y':development['y'].iloc[b]},unit/'outer_validation.joblib')
    if c.get('method_revision')=='native-risk-unified-v1':
        source=deepcopy(source)
        hints=c.get('field_metadata',{}).get(name,{}).get('numeric_type_hints',[])
        source['numeric_type_hints']=[v for v in hints if v in source['x'] and not pd.api.types.is_numeric_dtype(source['x'][v])]
        source['seed']=seed
    unit.mkdir(parents=True,exist_ok=True);joblib.dump(source,unit/'source.joblib')
    inner=splits(source['x'],source['y'],source['task'],c['inner_folds'],seed,source['groups'])
    write(unit/'inner_splits.json',[{'train':a.tolist(),'validation':b.tolist(),
        'train_row_ids':source['x'].index[a].tolist(),'validation_row_ids':source['x'].index[b].tolist()} for a,b in inner])
    if c.get('method_revision')=='native-risk-unified-v1':
        from .e2_parameterized import registry
        operations=registry(source['x'],source.get('date_columns'))
    else:operations=lib.menu(source['x'],source.get('date_columns'),c['menu_limit'])
    write(unit/'menu.json',operations)
    return unit


def selections(c,scorer,engine):
    unified=c.get('method_revision')=='native-risk-unified-v1'
    if unified:
        from .e2_unified import run_planner
        run_planner(c,scorer,engine,'ours')
        run_planner(c,scorer,engine,'without_task')
    for method in [m for m in METHODS if m not in ['one_shot','random_order']]:
        if unified and method in ['ours','without_task','no_lm_search']:continue
        folder=scorer.unit/'methods'/method
        if (folder/'failure.json').exists() and not (folder/'selection.json').exists() and all(
            (folder/'checkpoints'/f'{budget:03}.json').exists() for budget in c['budgets']):
            # A recorded resource failure is a frozen fallback, not an invitation
            # to rerun the same baseline whenever another method is recovered.
            continue
        try:run_method(c,scorer,engine,method)
        except Exception:
            write(scorer.unit/'methods'/method/'failure.json',{'error':traceback.format_exc(),'stage':'planning'})
            if c.get('fail_on_method_errors'):raise
    for budget in c['budgets']:
        for method in METHODS:
            folder=scorer.unit/'methods'/method
            target=folder/'checkpoints'/f'{budget:03}.json'
            if target.exists():continue
            try:
                if method=='one_shot':
                    ours=read(scorer.unit/'methods/ours/selection.json');first=ours['first_choice']
                    selected={'method':method,'spec':first['spec'],'inner_scores':first['scores'],'logical_calls':1,
                        'valid_proposals':int(ours['proposals'][0]['executable']),'status':'passed' if ours['proposals'][0]['executable'] else 'fallback'}
                elif method=='random_order':
                    ours=read(scorer.unit/'methods/ours/checkpoints'/f'{budget:03}.json');spec=deepcopy(ours['spec'])
                    if spec['kind']=='native_joint':
                        tail=spec['ids'][1:];random.Random(c['seed']).shuffle(tail);spec['ids']=spec['ids'][:1]+tail
                    else:random.Random(c['seed']).shuffle(spec['ids'])
                    score=scorer.inner(spec)
                    selected={'method':method,'spec':spec,'inner_scores':score['scores'],'logical_calls':0,'valid_proposals':0,'status':'passed'}
                else:
                    original=read(folder/'selection.json')
                    if original['logical_calls']:raise ValueError('Iterative method missing true prefix checkpoint')
                    selected=deepcopy(original)
                write(target,{**selected,'budget':budget,'selected_before_outer_labels':True})
            except Exception:
                error=traceback.format_exc();write(folder/'checkpoint_failures'/f'{budget:03}.json',{'error':error})
                raw=scorer.inner(RAW)
                write(target,{'method':method,'budget':budget,'spec':RAW,'inner_scores':raw['scores'],
                    'logical_calls':sum(1 for p in (folder/'calls').glob('*.json') if int(p.stem)<=budget),
                    'valid_proposals':0,'status':'fallback','fallback_reason':'Planning/checkpoint failure; original method did not complete',
                    'selected_before_outer_labels':True})
    expected=[scorer.unit/'methods'/m/'checkpoints'/f'{b:03}.json' for m in METHODS for b in c['budgets']]
    write(scorer.unit/'all_selections_frozen.json',{'method_budgets':45,'missing':[str(p.relative_to(scorer.unit)) for p in expected if not p.exists()],
        'sealed_test_opened':False})


def _run_unit(c,name,seed,fold):
    unit=prepare_unit(c,name,seed,fold);cc={**c,'seed':seed,'strict_replay_existing_calls':True}
    if c.get('method_revision')=='native-risk-unified-v1':
        from .e2_unified import UnifiedScorer
        scorer=UnifiedScorer(cc,unit)
    else:scorer=FormalScorer(cc,unit)
    if c.get('method_revision')=='native-risk-unified-v1':
        from .e2_unified import MethodEngine, method_weight_manifest
        write(unit/'method_weight_manifest.json',method_weight_manifest(cc))
        engine=MethodEngine(cc)
    else:engine=Engine(cc)
    selections(cc,scorer,engine)
    if str(fold)=='final':
        fit_final(cc,scorer);return
    validation=joblib.load(unit/'outer_validation.joblib')
    for method in METHODS:
        for budget in c['budgets']:
            folder=unit/'methods'/method/'budgets'/str(budget)
            if (folder/'result.json').exists():continue
            try:
                selected=read(unit/'methods'/method/'checkpoints'/f'{budget:03}.json');spec=selected['spec']
                members=spec['members'] if spec['kind']=='ensemble' else [spec]
                pred=[scorer.outer_member(v,validation) for v in members]
                result={'dataset':name,'seed':seed,'fold':int(fold),'method':method,'budget':budget,'status':selected['status'],
                    'logical_calls':selected['logical_calls'],'valid_proposals':selected['valid_proposals'],'sealed_test_opened':False}
                for i,model in enumerate(['xgboost','linear']):
                    prediction=np.mean([p[i] for p in pred],axis=0);result[model]=metrics(validation['y'],prediction,scorer.task)
                    folder.mkdir(parents=True,exist_ok=True);pd.DataFrame({'row_id':validation['x'].index,'y_true':validation['y'],
                        'prediction':prediction}).to_csv(folder/(model+'_predictions.csv'),index=False)
                write(folder/'result.json',result)
            except Exception:write(folder/'result.json',{'dataset':name,'seed':seed,'fold':fold,'method':method,'budget':budget,
                'status':'blocked','error':traceback.format_exc(),'sealed_test_opened':False})
    if (unit/'type_recovery_pending.json').exists():verify_type_recovery(c,unit)


def run_unit(c,name,seed,fold):
    if c.get('method_revision')!='native-risk-unified-v1':
        return _run_unit(c,name,seed,fold)
    # A newly idle card may help the fixed queue. One unit still has one writer.
    import fcntl
    unit=unit_path(c,name,seed,fold);unit.mkdir(parents=True,exist_ok=True)
    with (unit/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if str(fold)=='final' and (unit/'final_registry.json').exists():return
        write(unit/'execution_ownership.json',{'pid':os.getpid(),'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'dataset':name,'seed':seed,'fold':str(fold),'exclusive_unit_writer':True})
        return _run_unit(c,name,seed,fold)


def verify_type_recovery(c,unit):
    """A repaired baseline cannot silently replace a valid prefix or another arm."""
    pending=read(unit/'type_recovery_pending.json');archive=Path(c['output_dir'])/pending['archive']
    for rel,previous in read(archive/'unaffected_results.json').items():
        if read(unit/rel)!=previous:raise ValueError('Type recovery changed an unrelated method result: '+rel)
    for rel,meta in read(archive/'unaffected_predictions.json').items():
        p=unit/rel
        if p.stat().st_size!=meta['size'] or p.stat().st_mtime_ns!=meta['mtime_ns']:
            raise ValueError('Type recovery changed an unrelated prediction file: '+rel)
    folder=unit/'methods/llmfe'
    if (folder/'failure.json').exists():raise ValueError('Type recovery retains an unresolved method failure')
    selected=read(folder/'selection.json')
    if selected['logical_calls']!=20:raise ValueError('Recovered LLM-FE did not finish the frozen 20-round trajectory')
    for p in (archive/'calls').glob('*.json'):
        before,after=read(p),read(folder/'calls'/p.name)
        for name in ['request','raw_output','prompt_token_ids','completion_token_ids']:
            if before.get(name)!=after.get(name):raise ValueError('Type recovery altered an existing request/output: '+p.name)
    for budget in c['budgets']:
        result=read(folder/'budgets'/str(budget)/'result.json')
        if result['status'] not in ['passed','fallback']:raise ValueError('Recovered budget evaluation incomplete')
        old=archive/'checkpoints'/f'{budget:03}.json'
        if old.exists() and read(old).get('status')=='passed':
            previous,current=read(old),read(folder/'checkpoints'/f'{budget:03}.json')
            if any(previous[k]!=current[k] for k in ['spec','inner_scores']):raise ValueError('Type recovery changed an already valid prefix checkpoint')
            old_result=archive/'budgets'/str(budget)/'result.json'
            if old_result.exists() and read(old_result)['status']=='passed':
                if any(read(old_result)[m]!=result[m] for m in ['xgboost','linear']):raise ValueError('Type recovery changed an already valid prefix score')
    write(unit/'type_recovery_verified.json',{'passed':True,'old_requests_preserved':pending['old_call_count'],
        'unrelated_results_preserved':pending['unaffected_results'],'unrelated_predictions_preserved':pending['unaffected_predictions'],
        'logical_rounds':20,'new_model_calls_for_check':0,'test_loaded':False})


def final_consistency_fallback(c,error):
    """Only a disclosed, pre-test invalid-plan rejection can retain raw data."""
    if not isinstance(error,ValueError) or not str(error).startswith('Candidate creates consistency violations:'):
        return False
    revision=Path(c.get('output_dir',''))/'final_fit_failure_revision.json'
    return revision.exists() and read(revision).get('consistency_violation_action')=='mark_failed_and_retain_raw'


def fit_final(c,scorer,only_method=None):
    if only_method is not None and only_method not in METHODS:raise ValueError('Unknown final-fit method')
    previous=read(scorer.unit/'final_registry.json') if only_method is not None else []
    def fit_member(member):
        folder=scorer.unit/'final_models'/key(member)
        if (folder/'bundle.joblib').exists() and (folder/'fit.json').exists():return
        a,b,y=scorer.transform(member,scorer.x,scorer.x.iloc[:2],scorer.y,scorer.source['groups'],'outer',folder)
        model=fit_predictor(a,y,scorer.task,c['model']);encoder=fit_encoder(a)
        estimator=LogisticRegression(C=1.,max_iter=2000,random_state=0) if scorer.task['type']=='classification' else LinearRegression()
        linear=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),estimator).fit(encode(a,encoder),y)
        joblib.dump({'primary':model,'linear':linear,'encoder':encoder,'spec':member},folder/'bundle.joblib')
        replayed=apply_final(c,folder,scorer.x.iloc[:2]);pd.testing.assert_frame_equal(b,replayed)
        saved=joblib.load(folder/'bundle.joblib')
        np.testing.assert_array_equal(predict(model,b),predict(saved['primary'],replayed))
        np.testing.assert_array_equal(linear.predict(encode(b,encoder)),saved['linear'].predict(encode(replayed,saved['encoder'])))
        write(folder/'fit.json',{'predictor_fits':2,'training_row_ids':a.index.tolist(),'sealed_test_opened':False})
    registry=[]
    for method in METHODS:
        if only_method is not None and method!=only_method:
            registry.extend(r for r in previous if r['method']==method);continue
        for budget in c['budgets']:
            selected=read(scorer.unit/'methods'/method/'checkpoints'/f'{budget:03}.json');spec=selected['spec'];status=selected['status']
            members=spec['members'] if spec['kind']=='ensemble' else [spec]
            try:
                for member in members:fit_member(member)
            except Exception as exc:
                write(scorer.unit/'methods'/method/'final_fit_failures'/f'{budget:03}.json',
                    {'spec':spec,'error':traceback.format_exc(),'before_test_access':True,'fallback_spec':RAW})
                if c.get('fail_on_final_fit_errors') and not final_consistency_fallback(c,exc):raise
                spec=deepcopy(RAW);status='fallback';fit_member(spec)
            row={'method':method,'budget':budget,'spec':spec,'status':status}
            if spec!=selected['spec']:
                row.update(requested_spec=selected['spec'],fallback_reason='Selected plan failed final training fit; retained original data.',
                           failed_plan_not_counted_as_success=True)
            registry.append(row)
    write(scorer.unit/'final_registry.json',registry)


def apply_final(c,folder,frame):
    bundle=joblib.load(folder/'bundle.joblib');kind=bundle['spec']['kind'];state=joblib.load(folder/'transform_state.joblib')
    if isinstance(state,dict) and state.get('kind')=='native_joint':
        from .e2_unified import native_apply
        return native_apply(c,state,frame,'saved_application')
    if kind=='plan':return lib.apply_plan(frame,state['steps'])
    if kind=='code':
        result=frame
        for code,s in zip(bundle['spec']['codes'],state):result=apply_code(result,code,s)
        return result
    if kind=='standard':
        matrix=state.transform(frame);return pd.DataFrame(matrix,index=frame.index,columns=[f'e{i}' for i in range(matrix.shape[1])])
    result=normalize(frame)
    if kind in ['classical_clean','clean_then_openfe']:
        cleaning=state['cleaning'] if kind=='clean_then_openfe' else state
        if cleaning['numeric']:result=apply_classical_clean(result,cleaning)
        if kind=='classical_clean':return result
    native_method='openfe' if kind=='clean_then_openfe' else kind
    native=folder/'native';input_csv=native/'final_application.csv';output_csv=native/'final_application_features.csv'
    result.to_csv(input_csv,index_label='__row_id')
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
    with (native/'apply.log').open('w') as log:
        command=[c['classical_python'],'-m','taskdq.e2_classical','--folder',str(native),'--method',native_method,
            '--apply-csv',str(input_csv),'--output-csv',str(output_csv)]
        returncode=run_native_command(command,env,log,native_time_limit(c,native_method))
        if returncode:raise subprocess.CalledProcessError(returncode,command)
    new=pd.read_csv(output_csv,float_precision='round_trip').set_index('__row_id')
    if not new.index.equals(result.index):raise ValueError('Native final application row order changed')
    for col in read(native/native_method/'generation.json')['categorical']:
        new[col]=new[col].astype('string').astype(object)
    return pd.concat([result,new],axis=1)


def final_test(c):
    if (Path(c['output_dir'])/'development_review_hold.json').exists():
        raise ValueError('Development-result review stopped dispatch; sealed test evaluation is not permitted')
    if (Path(c['output_dir'])/'native_repair_pending.json').exists():
        raise ValueError('Native baseline repair must finish before any sealed test access')
    if (Path(c['output_dir'])/'autofeat_repair_pending.json').exists():
        raise ValueError('Native AutoFeat repair must finish before any sealed test access')
    # Every final fit and selection across the whole preregistered matrix must
    # exist before ANY sealed test labels are loaded.
    for marker in (Path(c['output_dir'])/'units').glob('*/*/*/type_recovery_pending.json'):
        verify_type_recovery(c,marker.parent)
    registries={};expected={(m,b) for m in METHODS for b in c['budgets']}
    for item in c['datasets']:
        for seed in c['seeds']:
            unit=unit_path(c,item['name'],seed,'final');registry=read(unit/'final_registry.json')
            if len(registry)!=len(expected) or {(r['method'],r['budget']) for r in registry}!=expected:
                raise ValueError('Incomplete or duplicate final selection matrix')
            if c.get('planning_scope')=='complete_development_only':
                frozen=read(unit/'all_selections_frozen.json')
                if frozen['missing'] or frozen['method_budgets']!=len(expected):raise ValueError('Selections are not frozen')
                for method,budget in expected:
                    if not (unit/'methods'/method/'checkpoints'/f'{budget:03}.json').exists():
                        raise ValueError('Missing final prefix selection')
            for row in registry:
                if row['method'] in c.get('require_native_success_before_test',[]) and row['status']!='passed':
                    raise ValueError('Native baseline has an unresolved fallback before sealed test: '+row['method'])
                if c.get('require_autofeat_success_before_test') and row['method']=='autofeat' and row['status']!='passed':
                    raise ValueError('Native AutoFeat has an unresolved fallback before sealed test')
                spec=row['spec'];members=spec['members'] if spec['kind']=='ensemble' else [spec]
                if any(not (unit/'final_models'/key(s)/'bundle.joblib').exists() for s in members):raise ValueError('Missing final model')
                if c.get('planning_scope')=='complete_development_only' and any(
                    not (unit/'final_models'/key(s)/'fit.json').exists() for s in members):
                    raise ValueError('Missing verified final fit receipt')
            registries[(item['name'],seed)]=registry
    write(Path(c['output_dir'])/'all_final_models_frozen.json',{'datasets':len(c['datasets']),'seeds':c['seeds'],
        'method_budget_selections':len(registries)*45,'test_loaded':False})
    for item in c['datasets']:
        for seed in c['seeds']:
            unit=unit_path(c,item['name'],seed,'final');directory=unit/'test_results'
            if (directory/'complete.json').exists():continue
            if (directory/'access.json').exists():raise ValueError('Prior partial test access; inspect the recorded failure before recovery')
            write(directory/'access.json',{'status':'opening','method_selection_complete':True,'model_fits_complete':True})
            test=joblib.load(Path(c['data_dir'])/'prepared'/item['name']/str(seed)/'sealed_test.joblib')
            cache={}
            for row in registries[(item['name'],seed)]:
                spec=row['spec'];members=spec['members'] if spec['kind']=='ensemble' else [spec]
                for member in members:
                    token=key(member)
                    if token in cache:continue
                    folder=unit/'final_models'/token;bundle=joblib.load(folder/'bundle.joblib');x=apply_final(c,folder,test['x'])
                    cache[token]=(predict(bundle['primary'],x),
                        bundle['linear'].predict_proba(encode(x,bundle['encoder']))[:,1] if item['type']=='classification'
                        else bundle['linear'].predict(encode(x,bundle['encoder'])))
                result={'dataset':item['name'],'type':item['type'],'seed':seed,'method':row['method'],'budget':row['budget'],
                    'status':row['status'],'test_evaluated':True,'target_dtype':str(np.asarray(test['y']).dtype),
                    'prediction_dtypes':{}}
                target=directory/row['method']/str(row['budget']);target.mkdir(parents=True,exist_ok=True)
                for i,model in enumerate(['xgboost','linear']):
                    pred=np.mean([cache[key(s)][i] for s in members],axis=0);result[model]=metrics(test['y'],pred,item)
                    result['prediction_dtypes'][model]=str(pred.dtype)
                    pd.DataFrame({'row_id':test['x'].index,'y_true':test['y'],'prediction':pred}).to_csv(target/(model+'_predictions.csv'),index=False)
                write(target/'result.json',result)
            write(directory/'complete.json',{'method_budget_results':45,'test_source_loads':1,'new_model_fits':0,'lm_calls':0})


def report(c):
    out=Path(c['output_dir']);flat=[];statuses=[]
    for item in c['datasets']:
        for seed in c['seeds']:
            unit=unit_path(c,item['name'],seed,'final')
            for p in (unit/'test_results').glob('*/*/result.json'):
                r=read(p);statuses.append({k:r[k] for k in ['dataset','seed','method','budget','status']})
                for model in ['xgboost','linear']:
                    for metric,value in r[model].items():
                        flat.append({k:r[k] for k in ['dataset','type','seed','method','budget','status']}|
                            {'downstream':model,'metric':metric,'value':value})
    out.mkdir(parents=True,exist_ok=True)
    if not flat:
        write(out/'report_status.json',{'formal_results_available':False,'test_evaluated':False});return
    df=pd.DataFrame(flat);df.to_csv(out/'test_results.csv',index=False)
    summary=df.groupby(['dataset','type','method','budget','downstream','metric'],as_index=False).agg(
        mean=('value','mean'),sd=('value','std'),seeds=('seed','nunique'),fallback_seeds=('status',lambda x:sum(v=='fallback' for v in x)))
    summary.loc[summary.seeds!=3,['mean','sd']]=np.nan
    summary.to_csv(out/'test_summary.csv',index=False);pd.DataFrame(statuses).to_csv(out/'method_coverage.csv',index=False)
    compare=[];ranks=[]
    for model in ['xgboost','linear']:
        for budget in c['budgets']:
            primary=summary[(summary.budget==budget)&(summary.downstream==model)&(summary.seeds==3)&
                (((summary.type=='classification')&(summary.metric=='auc'))|((summary.type=='regression')&(summary.metric=='rmse')))].copy()
            primary['utility']=np.where(primary.type=='classification',primary['mean'],-primary['mean'])
            for coverage in ['all_registered_with_marked_fallback','common_all_methods_success_only']:
                eligible=[name for name,g in primary.groupby('dataset') if len(g)==len(METHODS) and
                    (coverage!='common_all_methods_success_only' or not g.fallback_seeds.any())]
                scored=primary[primary.dataset.isin(eligible)].copy()
                scored['rank']=scored.groupby('dataset').utility.rank(ascending=False,method='average')
                for method,g in scored.groupby('method'):
                    ranks.append({'downstream':model,'budget':budget,'coverage':coverage,'method':method,
                        'datasets':len(eligible),'mean_rank':float(g['rank'].mean())})
                pivot=scored.pivot(index='dataset',columns='method',values='utility')
                for reference in [m for m in METHODS if m!='ours']:
                    if 'ours' not in pivot or reference not in pivot:continue
                    delta=pivot['ours']-pivot[reference]
                    compare.append({'downstream':model,'budget':budget,'coverage':coverage,'reference':reference,'datasets':len(delta),
                        'wins':int((delta>1e-9).sum()),'ties':int((delta.abs()<=1e-9).sum()),'losses':int((delta< -1e-9).sum())})
    pd.DataFrame(ranks).to_csv(out/'average_ranks.csv',index=False)
    pd.DataFrame(compare).to_csv(out/'wins_ties_losses.csv',index=False)
    calls=[read(p) for p in (out/'units').glob('*/*/*/physical_calls/*.json')]
    cached_metadata=[read(p) for p in (out/'units').glob('*/*/*/lm_cache/*.json')]
    fits=[read(p) for p in (out/'units').glob('*/*/*/inner_models/*/*/fit.json')]
    fits += [read(p) for p in (out/'units').glob('*/*/*/outer_models/*/fit.json')]
    fits += [read(p) for p in (out/'units').glob('*/*/*/final_models/*/fit.json')]
    imputers=[read(p) for p in (out/'units').glob('*/*/*/repair_bundles/*.json')]
    native=[read(p) for p in (out/'units').glob('*/*/*/**/generation.json')]
    logical=sum(read(p).get('logical_calls',0) for p in (out/'units').glob('*/*/*/methods/*/selection.json')
        if p.parent.name not in ['one_shot','random_order'])
    cost={'physical_lm_calls':len(calls),'iterative_logical_calls':logical,
        'completed_input_tokens':sum(r.get('input_tokens',0) for r in calls),
        'completed_output_tokens':sum(r.get('output_tokens',0) for r in calls),
        'completed_generation_seconds':sum(r.get('seconds',0) for r in calls),
        'peak_gpu_allocated_bytes':max([r.get('peak_allocated_bytes',0) for r in cached_metadata] or [0]),
        'token_counts_complete':all(r['status']=='completed' for r in calls),
        'common_predictor_and_linear_fits':sum(r['predictor_fits'] for r in fits),
        'conditional_imputer_fits':sum(r['fitted_imputers'] for r in imputers),'native_generation_jobs':len(native),
        'native_internal_estimator_fits':'not individually instrumented','sft_grpo_updates':0}
    empty_native=[]
    for p in (out/'units').glob('*/*/final/final_models/*/native/autofeat/generation.json'):
        r=read(p)
        if r.get('status')=='passed' and r.get('native',{}).get('numeric_input_columns')==[]:
            empty_native.append('/'.join(p.relative_to(out/'units').parts[:3]))
    cost['autofeat_no_numeric_input_units']=sorted(set(empty_native))
    recovery=out/'autofeat_review_20261105'
    if (recovery/'initial_launch.json').exists():
        archived=[read(p) for p in (recovery/'before_repair').glob('**/generation.json')]
        probe_file=recovery/'probe_summary.json';probes=read(probe_file)['receipts'] if probe_file.exists() else []
        repair_receipts=[read(p) for p in (recovery/'repaired').glob('*/*/final/receipt.json')]
        reused_units={r['unit'] for r in repair_receipts if r.get('reused_probe_native_fit')}
        # Reused probe outputs already occur in active unit records; count the
        # physical work once. Archived failed attempts remain real costs.
        extra_probes=[r for r in probes if r['source_unit'] not in reused_units]
        cost.update(native_generation_jobs_active=len(native),native_generation_jobs_archived=len(archived),
            native_probe_jobs=len(probes),native_probe_jobs_reused=len(reused_units),
            native_generation_jobs=len(native)+len(archived)+len(extra_probes),
            native_generation_seconds_measured=sum(r.get('seconds',0) for r in native+archived)+
                sum(r.get('wall_seconds',0) for r in extra_probes),
            native_archived_resource_failure_seconds=sum(r.get('seconds',0) for r in archived),
            cost_scope='Includes preserved resource failures; identical probe fits reused by active units count once.')
        started=datetime.datetime.fromisoformat(read(recovery/'initial_launch.json')['utc']).timestamp()
        cost['end_to_end_wall_seconds_at_report']=time.time()-started
        cost['end_to_end_start_utc']=read(recovery/'initial_launch.json')['utc']
    native_recovery=out/'native_review_20261106'
    if native_recovery.exists():
        archived=[read(p) for p in (native_recovery/'before_repair').glob('**/generation.json')]
        probes=[{**read(p),'artifact_directory':str(p.parent)} for location in ['probes','probes_parallel5']
            for p in (native_recovery/location).glob('*/receipt.json')]
        receipts=[read(p) for p in (native_recovery/'repaired').glob('**/receipt.json')]
        reused={r.get('reused_probe_path') or str(native_recovery/'probes'/(r['unit'].replace('/','_')+'_'+r['method']))
            for r in receipts if r.get('reused_probe_native_fit')}
        extra=[r for r in probes if r['artifact_directory'] not in reused]
        cost['native_generation_jobs']=cost.get('native_generation_jobs',len(native))+len(archived)+len(extra)
        cost['native_generation_seconds_measured']=cost.get('native_generation_seconds_measured',sum(r.get('seconds',0) for r in native))+\
            sum(r.get('seconds',0) for r in archived)+sum(r.get('wall_seconds',0) for r in extra)
        cost['native_recovery_20261106']={'archived_jobs':len(archived),'diagnostic_jobs':len(probes),
            'reused_diagnostic_jobs':len(reused),'extra_diagnostic_jobs':len(extra),
            'archived_seconds':sum(r.get('seconds',0) for r in archived),'repair_arms':len(receipts),
            'failed_diagnostics':sum(r['status']!='completed' for r in probes)}
        archived_fits=[read(p) for p in (native_recovery/'before_repair').glob('**/fit.json')]
        cost['common_predictor_and_linear_fits']+=sum(r.get('predictor_fits',0) for r in archived_fits)
        cost['cost_scope']='Includes native failures and both recovery phases; diagnostic fits reused by active units count once.'
    current_repairs=out/'engineering_repairs'
    if current_repairs.exists():
        archived=[read(p) for p in current_repairs.glob('**/generation.json')]
        cost['native_archived_engineering_attempts']=len(archived)
        cost['native_generation_jobs']+=len(archived)
        cost['native_generation_seconds_measured']=cost.get('native_generation_seconds_measured',sum(r.get('seconds',0) for r in native))+sum(r.get('seconds',0) for r in archived)
        cost['current_engineering_failure_seconds']=sum(r.get('seconds',0) for r in archived)
    if (out/'native_runtime_revision.json').exists():
        cost['native_runtime_revision']=read(out/'native_runtime_revision.json')
    if (out/'execution_overlap_revision.json').exists():
        cost['execution_overlap_revision']=read(out/'execution_overlap_revision.json')
    write(out/'costs.json',cost)
    audit=[];review_fold='final' if c.get('planning_scope')=='complete_development_only' else 1
    for item in c['datasets']:
        p=unit_path(c,item['name'],42,review_fold)/'methods/ours/calls/1.json'
        if p.exists():
            r=read(p);audit.append({'dataset':item['name'],'seed':42,'fold':review_fold,'round':1,'reason':r.get('answer',{}).get('reason'),
                'operations':r.get('answer',{}).get('operations'),'reason_valid':None,'reviewer_note':None})
    write(out/'reason_review_queue.json',{'sampling':f'One first proposal per dataset, seed42 fold={review_fold}; fixed before scores. Failed/missing responses retained in coverage.',
        'expected':len(c['datasets']),'available':len(audit),'reviewed':False,'items':audit})
    for model in ['xgboost','linear']:
        for budget in c['budgets']:
            rows=[]
            for item in c['datasets']:
                metric='auc' if item['type']=='classification' else 'rmse';values=[item['name']+' '+metric]
                for method in METHODS:
                    selected=summary[(summary.dataset==item['name'])&(summary.method==method)&(summary.budget==budget)&
                        (summary.downstream==model)&(summary.metric==metric)]
                    if len(selected)!=1 or selected.iloc[0].seeds!=3:values.append('未完整完成');continue
                    r=selected.iloc[0];values.append(f'{r["mean"]:.6f} ± {r["sd"]:.6f}'+('（含回退）' if r.fallback_seeds else ''))
                rows.append(values)
            pd.DataFrame(rows,columns=['dataset_metric']+METHODS).to_csv(out/f'{model}_N{budget}_table.csv',index=False)
    write(out/'report_status.json',{'formal_results_available':True,'recorded_method_budget_seeds':len(statuses),
        'expected':len(c['datasets'])*len(c['seeds'])*len(METHODS)*len(c['budgets']),
        'test_evaluated':True,'new_fits_for_report':0,'new_lm_calls_for_report':0})


def run(config_path,stage,dataset=None,seed=None,fold=None):
    c=config(config_path)
    if c.get('planning_scope')=='complete_development_only' and stage not in ['report']:
        ready=read(Path(c['project_root'])/c['revision_readiness'])
        needed='source_verification_complete' if stage=='prepare' else 'ready_for_main_e2'
        if not ready.get(needed):raise ValueError('Revised OpenML E2 is not ready: '+needed)
    if stage=='prepare':return prepare(c)
    if stage=='report':report(c);return {'reported':True}
    if stage=='final-test':final_test(c);report(c);return {'test_evaluated':True}
    if dataset not in {d['name'] for d in c['datasets']} or seed not in c['seeds']:raise ValueError('Specify a registered dataset and data seed')
    if stage=='fit-final':fold='final'
    elif fold not in range(1,6):raise ValueError('Specify outer fold 1 through 5')
    run_unit(c,dataset,seed,fold);return {'stage':stage,'dataset':dataset,'seed':seed,'fold':fold}
