"""Development-only runner and cached-reference reporting for the parameter fix."""
import argparse
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from .e2_parameterized import registry,ExpandedScorer,run_unit,plan_label
from .e2 import apply_final
from .e2_pilot import Engine,RAW,key
from .e2_search import fit_selected
from .scoring import predict,encode
from .e1 import metrics
from .io import read,write


def settings(path):
    path=Path(path).resolve();s=read(path);root=(path.parent/s['project_root']).resolve();s['project_root']=str(root)
    for field in ['reference_output','output_dir']:s[field]=str(root/s[field])
    reference=read(Path(s['reference_output'])/'config.json');c=reference['runtime']
    assert s['datasets']==reference['pilot']['datasets'] and s['seeds']==[42,123,456]
    assert s['calls_per_unit']==s['code_candidates']==5 and s['maximum_new_calls']==120
    c={**c,'output_dir':s['output_dir'],'unit_physical_limit':5,'lm_budget':5,'budgets':[5]}
    return s,c


def unit_path(s,name,seed):return Path(s['output_dir'])/'units'/name/str(seed)/'search'


def prepare(s,c):
    out=Path(s['output_dir']);records=[]
    for name in s['datasets']:
        for seed in s['seeds']:
            unit=unit_path(s,name,seed);old=Path(s['reference_output'])/'units'/name/str(seed)/'search'
            if (unit/'prepared.json').exists():records.append(read(unit/'prepared.json'));continue
            unit.mkdir(parents=True,exist_ok=True)
            for filename in ['source.joblib','split.json','inner_splits.json']:shutil.copy2(old/filename,unit/filename)
            source=joblib.load(unit/'source.joblib');previous=read(old/'menu.json');full=registry(source['x'],source.get('date_columns'))
            assert full[:len(previous)]==previous,'Historical operation semantics changed'
            write(unit/'menu.json',full)
            all_fields=set(source['x']);reached={op[k] for op in full for k in ['column','left','right','group'] if k in op}
            assert all_fields<=reached
            inherited=[]
            for score in (old/'scores').glob('*.json'):
                result=read(score)
                if result['spec']['kind']!='plan':continue
                token=score.stem;(unit/'scores').mkdir(exist_ok=True);shutil.copy2(score,unit/'scores'/score.name)
                for part in ['inner_models','final_models']:
                    if (old/part/token).exists():shutil.copytree(old/part/token,unit/part/token)
            if (old/'repair_bundles').exists():shutil.copytree(old/'repair_bundles',unit/'repair_bundles')
            inherited=[str(p.relative_to(unit)) for pattern in ['inner_models/*/*/fit.json','final_models/*/fit.json','repair_bundles/*.json'] for p in unit.glob(pattern)]
            write(unit/'reused_artifacts.json',inherited)
            part=read(unit/'split.json');assert source['x'].index.tolist()==part['search_row_ids']
            assert not set(part['search_row_ids'])&set(part['confirmation_row_ids'])
            row={'dataset':name,'seed':seed,'passed':True,'old_registry_size':len(previous),'full_registry_size':len(full),
                'unreachable_raw_fields':[],'historical_prefix_unchanged':True,'original_test_loads':0}
            write(unit/'prepared.json',row);records.append(row)
    assert len(records)==24;write(out/'preparation.json',{'passed':True,'records':records,'development_already_opened':True})


def fit(s,c,name,seed):
    unit=unit_path(s,name,seed)
    if (unit/'fitted.json').exists():return
    sc=ExpandedScorer({**c,'seed':seed},unit);selection=read(unit/'selection.json')
    for method in ['expanded_code','expanded_system']:
        spec=selection['plans'][method]
        try:
            fit_selected({**c,'seed':seed},sc,spec)
            from .e2_semantic import check_steps
            checks=check_steps(sc.x,sc.x.iloc[:2],joblib.load(unit/'final_models'/key(spec)/'transform_state.joblib')['steps'])
            write(unit/'final_models'/key(spec)/'expanded_final_check.json',{'passed':True,'checks':checks})
        except ValueError as exc:
            write(unit/(method+'_final_fallback.json'),{'error':repr(exc),'attempted':spec,'fallback':RAW,'before_recheck':True})
            selection['plans'][method]=RAW
    write(unit/'selection_final.json',selection);write(unit/'fitted.json',{'passed':True,'before_recheck':True})


def freeze(s,c):
    out=Path(s['output_dir']);calls=[]
    for name in s['datasets']:
        for seed in s['seeds']:
            unit=unit_path(s,name,seed);assert read(unit/'fitted.json')['passed']
            part=read(unit/'split.json');source=joblib.load(unit/'source.joblib');assert source['x'].index.tolist()==part['search_row_ids']
            for fold in read(unit/'inner_splits.json'):
                assert not set(fold['train_row_ids']+fold['validation_row_ids'])&set(part['confirmation_row_ids'])
            loaded=read(unit/'planner_load.json');assert loaded['file_to_loaded_tensor_check'] and loaded['loaded_adapter_tensors']==144
            history=read(unit/'history.json');ids=[tuple(v['spec']['ids']) for v in history if 'spec' in v];assert len(ids)==len(set(ids))
            assert sum(v['origin']=='language_model' for v in history)<=5
            for spec in read(unit/'selection_final.json')['plans'].values():
                folder=unit/'final_models'/key(spec);f=read(folder/'fit.json');assert set(f['training_row_ids'])<=set(part['search_row_ids'])
                transformed=apply_final(c,folder,source['x'].iloc[:4]);assert transformed.index.equals(source['x'].iloc[:4].index)
                assert np.isfinite(predict(joblib.load(folder/'bundle.joblib')['primary'],transformed)).all()
            calls.extend(read(p) for p in (unit/'physical_calls').glob('*.json'))
    assert len(calls)<=120 and all(v['status']=='completed' for v in calls)
    write(out/'all_selections_frozen.json',{'passed':True,'units':24,'new_methods':2,'actual_calls':len(calls),
        'before_current_recheck':True,'development_previously_opened':True,'original_test_loads':0})


def confirm(s,c):
    out=Path(s['output_dir']);reference=Path(s['reference_output']);assert read(out/'all_selections_frozen.json')['passed']
    for name in s['datasets']:
        task=next(d for d in c['datasets'] if d['name']==name)
        for seed in s['seeds']:
            unit=unit_path(s,name,seed);dest=out/'evaluation'/name/str(seed)
            if (dest/'complete.json').exists():continue
            assert not (dest/'access.json').exists(),'Inspect partial evaluation before recovery'
            write(dest/'access.json',{'all_new_models_frozen':True,'development_previously_opened':True,'original_test_opened':False})
            frame=joblib.load(reference/'development_confirmation'/name/str(seed)/'confirmation.joblib');cache={}
            for method in ['expanded_code','expanded_system']:
                spec=read(unit/'selection_final.json')['plans'][method];token=key(spec);folder=unit/'final_models'/token
                if token not in cache:
                    bundle=joblib.load(folder/'bundle.joblib');x=apply_final(c,folder,frame['x']);assert x.index.equals(frame['x'].index)
                    p=predict(bundle['primary'],x);matrix=encode(x,bundle['encoder'])
                    q=bundle['linear'].predict_proba(matrix)[:,1] if task['type']=='classification' else bundle['linear'].predict(matrix)
                    cache[token]=(p,q)
                record={'dataset':name,'seed':seed,'method':method,'type':task['type'],'spec':spec,
                    'target_dtype':str(np.asarray(frame['y']).dtype),'prediction_dtypes':{},'adaptive_development':True}
                for model,p in zip(['xgboost','linear'],cache[token]):
                    assert np.isfinite(p).all();record[model]=metrics(frame['y'],p,task);record['prediction_dtypes'][model]=str(p.dtype)
                    folder_result=dest/method;folder_result.mkdir(parents=True,exist_ok=True)
                    pd.DataFrame({'row_id':frame['x'].index,'y_true':frame['y'],'prediction':p}).to_csv(folder_result/(model+'_predictions.csv'),index=False)
                write(dest/method/'result.json',record)
            write(dest/'complete.json',{'passed':True,'new_methods':2,'original_test_loads':0})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['prepare','pilot','plan','fit','freeze','confirm','report'],required=True);p.add_argument('--lane',type=int,default=0)
    a=p.parse_args();s,c=settings(a.config);matrix=[(n,z) for n in s['datasets'] for z in s['seeds']]
    if a.stage=='prepare':prepare(s,c)
    elif a.stage in ['pilot','plan']:
        engine=Engine(c);jobs=matrix[a.lane::8]
        if a.stage=='pilot':jobs=jobs[:1]
        for name,seed in jobs:run_unit(s,{**c,'seed':seed},unit_path(s,name,seed),engine,pilot=a.stage=='pilot')
    elif a.stage=='fit':
        for name,seed in matrix[a.lane::8]:fit(s,c,name,seed)
    elif a.stage=='freeze':freeze(s,c)
    elif a.stage=='confirm':confirm(s,c)
    else:
        from .e2_parameterized_report import report
        report(s,c)


if __name__=='__main__':main()
