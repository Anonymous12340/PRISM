"""Config-driven preparation, cross-validation, application and one final test."""
from pathlib import Path
from copy import deepcopy
import importlib.metadata
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from . import __version__
from .io import read,write,append,records
from .config import load_config
from .operations import candidate_mask
from .scoring import SourceScorer,splits,metrics
from .planner import Planner
from .pipeline import fit_pipeline,load_pipeline,CANDIDATE_COLUMNS


def prepared_dir(c):return Path(c.get('prepared_dir',Path(c['output_dir'])/'prepared'))


def labels(frame,task):
    s=frame[task['target']]
    if task['type']=='regression':
        y=pd.to_numeric(s,errors='raise').astype(float)
        if not np.isfinite(y).all():raise ValueError('Regression labels must be finite')
        return y
    positive=task.get('positive_values',[1]);negative=task.get('negative_values',[0])
    if set(positive)&set(negative) or not s.isin(positive+negative).all():raise ValueError('Binary labels need explicit disjoint positive/negative values')
    return s.isin(positive).astype(int)


def dataset_spec(d):return {k:v for k,v in d.items() if k!='csv'}


def prepare(c):
    folder=prepared_dir(c);manifest=folder/'manifest.json'
    expected={'schema_version':1,'datasets':[dataset_spec(d) for d in c['datasets']],
        'test_size':c['test_size'],'split_seed':c['split_seed']}
    if manifest.exists():
        if read(manifest)['spec']!=expected:raise ValueError('Prepared split differs; use another prepared_dir')
        return folder
    if folder.exists() and any(folder.iterdir()):raise ValueError('Incomplete preparation exists; inspect or choose another directory')
    counts={}
    for d in c['datasets']:
        frame=pd.read_csv(d['csv'])
        if d.get('row_id'):frame=frame.set_index(d['row_id'])
        if not frame.index.is_unique:raise ValueError('Row IDs must be unique')
        for task in d['tasks']:labels(frame,task)
        for column in d['repair_columns']:pd.to_numeric(frame[column],errors='raise')
        positions=np.arange(len(frame))
        if d.get('group'):
            if frame[d['group']].isna().any():raise ValueError('Group IDs cannot be missing')
            dev,test=next(GroupShuffleSplit(1,test_size=c['test_size'],random_state=c['split_seed']).split(frame,groups=frame[d['group']]))
            assert set(frame.iloc[dev][d['group']]).isdisjoint(frame.iloc[test][d['group']])
        else:
            first=d['tasks'][0];strata=labels(frame,first) if first['type']=='classification' else None
            dev,test=train_test_split(positions,test_size=c['test_size'],random_state=c['split_seed'],stratify=strata)
        target=folder/d['name'];target.mkdir(parents=True,exist_ok=True)
        joblib.dump(frame.iloc[dev],target/'development.joblib');joblib.dump(frame.iloc[test],target/'sealed_test.joblib')
        write(target/'split.json',{'development_ids':frame.index[dev].tolist(),'sealed_test_ids':frame.index[test].tolist()})
        counts[d['name']]={'development_rows':len(dev),'sealed_rows':len(test)}
    write(manifest,{'spec':expected,'counts':counts,'sealed_test_evaluated':False})
    return folder


def inject(frame,columns,condition,seed):
    dirty=frame.copy();truth=pd.DataFrame(False,index=frame.index,columns=columns)
    if condition['type'] is None:return dirty,None
    rng=np.random.default_rng(seed)
    for column in columns:
        eligible=np.flatnonzero(frame[column].notna().to_numpy());selected=rng.permutation(eligible)[:round(len(eligible)*condition['rate'])]
        if condition['type']=='missing':dirty.iloc[selected,dirty.columns.get_loc(column)]=np.nan
        else:
            q1,q3=frame[column].quantile([.25,.75]);scale=q3-q1
            if not scale>0:raise ValueError('Outlier injection needs a nonzero IQR')
            dirty[column]=dirty[column].astype(float)
            dirty.iloc[selected,dirty.columns.get_loc(column)]=rng.choice([q1-10*scale,q3+10*scale],size=len(selected))
        truth.iloc[selected,truth.columns.get_loc(column)]=True
    return dirty,truth


def repair_metrics(clean,dirty,model,truth):
    if truth is None:return {'detection_f1':None,'repair_mae':None,'remaining_missing':None,'injected_cells':0}
    actual=model.transform(dirty);tp=fp=fn=0;errors=[];missing=0;injected=0
    for column in truth:
        true=truth[column].to_numpy(bool);found=candidate_mask(dirty,model.bundles[column]['state']).to_numpy(bool)
        tp+=int((true&found).sum());fp+=int((~true&found).sum());fn+=int((true&~found).sum())
        original=clean.loc[truth[column],column].to_numpy(float);repaired=actual.loc[truth[column],column].to_numpy(float)
        injected+=len(original);missing+=int((~np.isfinite(repaired)).sum());errors.extend(np.abs(original-repaired).tolist())
    return {'detection_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        'repair_mae':float(np.mean(errors)) if errors and not missing else None,
        'remaining_missing':missing,'injected_cells':injected}


def unit_config(c,d,seed,variant):
    return {k:deepcopy(c[k]) for k in ['model','imputer','inner_folds','rounds']}|{
        'repair_columns':d['repair_columns'],'seed':seed,'include_task':variant.get('include_task',True)}


def run(c,resume=False):
    out=Path(c['output_dir']);manifest=out/'run.json'
    if manifest.exists():
        if read(manifest)['config']!=c:raise ValueError('Frozen run configuration differs; choose another output directory')
        if not resume:raise ValueError('Run already exists; use --resume to reuse frozen units')
    data=prepare(c);out.mkdir(parents=True,exist_ok=True)
    write(manifest,{'schema_version':1,'system_version':__version__,'mode':c['mode'],'config':c,
        'test_opened':False,'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','joblib']}})
    planners={v['name']:Planner(v['planner'],out/'model_cache') for v in c['variants']};all_rows=[]
    for d in c['datasets']:
        full=joblib.load(data/d['name']/'development.joblib')
        for task0 in d['tasks']:
            task={**task0,'background':d.get('background','')};y=labels(full,task)
            group=full[d['group']] if d.get('group') else None
            for seed in c['seeds']:
                for fold,(a,b) in enumerate(splits(full[d['features']],y,task,c['folds'],seed,group),1):
                    for condition in c['conditions']:
                        base=out/'units'/d['name']/task['name']/str(seed)/str(fold)/condition['name']
                        clean=full.iloc[a][d['features']];dirty,truth=inject(clean,d['repair_columns'],condition,seed+1000*fold)
                        for variant in c['variants']:
                            unit=base/'variants'/variant['name'];result_path=unit/'result.json'
                            if result_path.exists():
                                if not (unit/'pipeline.joblib').exists():raise FileNotFoundError(unit/'pipeline.joblib')
                                all_rows.append(read(result_path));continue
                            cfg=unit_config(c,d,seed+fold,variant)
                            meta={'dataset':d['name'],'task':task['name'],'seed':seed,'fold':fold,'condition':condition['name'],
                                'variant':variant['name'],'mode':c['mode']}
                            source={'x':dirty,'reference':clean,'y':y.iloc[a],'task':task,'seed':seed+fold,
                                'groups':group.iloc[a] if group is not None else None,'config':cfg,'metadata':meta}
                            scorer=SourceScorer(base/'scorer',source)
                            if (unit/'complete.json').exists():model=load_pipeline(unit/'pipeline.joblib')
                            else:model=fit_pipeline(dirty,y.iloc[a],task,cfg,planners[variant['name']],unit,
                                reference=clean,groups=source['groups'],scorer=scorer,metadata=meta)
                            if truth is not None:truth.to_csv(base/'injection_truth.csv',index_label='row_id')
                            # Held-out development rows/labels enter only after selection.json exists.
                            assert (unit/'selection.json').exists()
                            score=metrics(y.iloc[b],model.predict(full.iloc[b][d['features']]),task)
                            row={**meta,**score,**repair_metrics(clean,dirty,model,truth),'selected_plan':model.plan,
                                'model_file':os.path.relpath(unit/'pipeline.joblib',out),'test_opened':False}
                            write(result_path,row);all_rows.append(row)
                            print(f"completed {d['name']}/{task['name']}/{seed}/{fold}/{condition['name']}/{variant['name']}",flush=True)
    write(out/'results.json',all_rows);pd.DataFrame(all_rows).to_csv(out/'results.csv',index=False)
    summary=summarize(out)
    cached=[read(p) for p in (out/'model_cache').glob('*.json')]
    write(out/'costs.json',{'physical_lm_calls':sum(r['physical_call'] for r in cached),
        'inner_predictor_models':len(list((out/'units').glob('*/*/*/*/*/scorer/models/*/*.joblib'))),
        'auxiliary_imputer_models':sum(read(p)['fitted_imputers'] for p in (out/'units').glob('*/*/*/*/*/scorer/bundles/*.json'))+
            sum(read(p)['fitted_imputers'] for p in (out/'units').glob('*/*/*/*/*/scorer/outer_bundles.json')),
        'outer_predictor_models':len(all_rows),'test_opened':False})
    write(out/'state.json',{'status':'completed','mode':c['mode'],'test_opened':False,'units':len(all_rows)})
    return summary


def summarize(out):
    out=Path(out);frame=pd.DataFrame(read(out/'results.json'));keys=['dataset','task','condition','variant']
    metric_names=[m for m in ['auc','f1','rmse','detection_f1','repair_mae','remaining_missing'] if m in frame]
    rows=[];seed_rows=[]
    for key,group in frame.groupby(keys,dropna=False):
        for metric in metric_names:
            if metric in ('auc','f1','rmse') and group[metric].isna().all():continue
            per_seed=[]
            for seed,samples in group.groupby('seed'):
                vals=samples[metric];value=float(vals.mean()) if vals.notna().all() else None
                seed_rows.append(dict(zip(keys,key))|{'seed':int(seed),'metric':metric,'value':value});per_seed.append(value)
            complete=all(v is not None for v in per_seed)
            rows.append(dict(zip(keys,key))|{'metric':metric,'mean':float(np.mean(per_seed)) if complete else None,
                'std':float(np.std(per_seed,ddof=1)) if complete and len(per_seed)>1 else None,'data_seeds':len(per_seed)})
    pd.DataFrame(seed_rows).to_csv(out/'seed_scores.csv',index=False);pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    write(out/'summary.json',{'system_version':__version__,'test_opened':False,'metrics':rows,
        'note':'Synthetic demo results are not research claims. Missing repair estimates are not dropped.'})
    return rows


def final_test(c,variant_name):
    data=prepared_dir(c);marker=data/'final_test_use.json'
    if marker.exists():raise ValueError('This sealed split has already entered final evaluation')
    variant=next(v for v in c['variants'] if v['name']==variant_name)
    out=Path(c['output_dir'])/'final_test'/variant_name;planner=Planner(variant['planner'],out/'model_cache');frozen=[]
    # Plan all tasks using development data before loading any sealed data file.
    for d in c['datasets']:
        frame=joblib.load(data/d['name']/'development.joblib')
        for task0 in d['tasks']:
            task={**task0,'background':d.get('background','')};unit=out/d['name']/task['name']
            model=fit_pipeline(frame[d['features']],labels(frame,task),task,unit_config(c,d,c['split_seed'],variant),planner,unit,
                groups=frame[d['group']] if d.get('group') else None,metadata={'dataset':d['name'],'mode':c['mode']})
            frozen.append((d,task,model))
    write(marker,{'status':'started','variant':variant_name,'plans_frozen':True,'system_version':__version__})
    results=[]
    for d,task,model in frozen:
        test=joblib.load(data/d['name']/'sealed_test.joblib')
        results.append({'dataset':d['name'],'task':task['name'],'variant':variant_name,
            **metrics(labels(test,task),model.predict(test[d['features']]),task)})
    write(out/'results.json',results);write(marker,{'status':'completed','variant':variant_name,'plans_frozen':True,'results':results})
    return results


def apply(model_path,csv,output,row_id=None):
    model=load_pipeline(model_path);frame=pd.read_csv(csv)
    if row_id:frame=frame.set_index(row_id)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    repaired=frame.copy()
    for column,values in model.transform(frame).items():repaired[column]=values
    repaired.to_csv(out/'repaired.csv',index_label=row_id or 'row_id')
    report=model.report(frame)
    write(out/'candidates.json',report);pd.DataFrame(report,columns=CANDIDATE_COLUMNS).to_csv(out/'candidates.csv',index=False)
    pd.DataFrame({'row_id':frame.index,'prediction':model.predict(frame)}).to_csv(out/'predictions.csv',index=False)
    return {'output':str(out),'candidate_cells':len(report)}
