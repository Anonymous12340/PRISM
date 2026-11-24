"""E1: fixed pipelines, archived CV replay and one frozen final test batch.

This CPU experiment does not invoke the planning/training runtime. Definitions
match the originally registered eight E1 arms, with saved train-only state.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import importlib.metadata
import json
import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import f1_score,roc_auc_score,root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from xgboost import XGBClassifier,XGBRegressor
from .io import read,write,resolved
from . import __version__

PIPELINES=['none','impute_only','standard','iqr_then_impute','aggressive',
           'impute_then_clip','clip_then_impute','missing_flags']


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def metrics(y,prediction,task):
    # Preserve the registered sklearn dtype/accumulation convention exactly.
    if task['type']=='classification':return {'auc':float(roc_auc_score(y,prediction)),
        'f1':float(f1_score(y,np.asarray(prediction)>=.5,zero_division=0))}
    return {'rmse':float(root_mean_squared_error(y,prediction))}


def labels(frame,task):
    raw=frame[task['target']]
    if task['type']=='regression':
        y=pd.to_numeric(raw,errors='raise')
        if not np.isfinite(y).all():raise ValueError('Nonfinite outcome')
        return y
    if pd.api.types.is_numeric_dtype(raw):
        values=raw;positive=set(map(float,task['positive_labels']));negative=set(map(float,task['negative_labels']))
    else:
        values=raw.astype(str);positive=set(map(str,task['positive_labels']));negative=set(map(str,task['negative_labels']))
    if not positive or not negative or positive&negative or not values.isin(positive|negative).all():
        raise ValueError('Unspecified or overlapping binary outcomes')
    return values.isin(positive).astype(int)


def clean(frame,numeric):
    frame=frame.copy()
    for c in numeric:frame[c]=frame[c].astype(float).replace([np.inf,-np.inf],np.nan)
    return frame


def apply_steps(frame,numeric,steps,training=False):
    frame=clean(frame,numeric)
    for step in steps:
        action=step['action']
        if action=='drop_columns':frame=frame.drop(columns=step['columns'])
        elif action=='impute':
            for c,value in step['values'].items():
                if value is not None:frame[c]=frame[c].fillna(value)
        elif action=='clip':
            for c,bounds in step['bounds'].items():
                if bounds is not None:frame[c]=frame[c].clip(*bounds)
        elif action=='iqr_drop':
            if training:frame=frame.drop(index=step['removed_training_rows'])
        elif action=='missing_flags':
            for c,name in step['columns'].items():frame[name]=frame[c].isna().astype(float)
        else:raise ValueError('Unknown fixed preprocessing step')
    return frame


@dataclass
class Preprocessor:
    input_numeric: list
    numeric: list
    categorical: list
    steps: list
    encoder: object
    selected: object
    names: list

    def encode(self,frame):
        blocks=[frame[self.numeric].to_numpy(dtype=float)]
        if self.categorical:
            values=('value:'+frame[self.categorical].astype('string')).fillna('missing:')
            blocks.append(self.encoder.transform(values))
        return np.hstack(blocks)[:,self.selected]

    def transform(self,frame):return self.encode(apply_steps(frame,self.input_numeric,self.steps))


def fit_preprocessor(frame,method,settings):
    if method not in PIPELINES:raise ValueError('Unknown E1 pipeline')
    original=frame;numeric=[c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
    input_numeric=numeric.copy();a=clean(frame,numeric);steps=[]
    if method=='aggressive':
        dropped=a.columns[a.isna().mean()>settings['high_missing_rate']].tolist()
        steps.append({'action':'drop_columns','columns':dropped});a=a.drop(columns=dropped)
        numeric=[c for c in numeric if c not in dropped]
    actions={'none':[],'impute_only':['impute'],'standard':['impute','iqr_drop'],
        'iqr_then_impute':['iqr_drop','impute'],'aggressive':['impute','iqr_drop'],
        'impute_then_clip':['impute','clip'],'clip_then_impute':['clip','impute'],'missing_flags':['missing_flags']}[method]
    for action in actions:
        step={'action':action,'training_rows_before':len(a)}
        if action=='impute':
            step['values']={c:float(a[c].median()) if a[c].notna().any() else None for c in numeric}
            for c,value in step['values'].items():
                if value is not None:a[c]=a[c].fillna(value)
        elif action=='missing_flags':
            step['columns']={c:'__missing_'+c for c in numeric}
            for c,name in step['columns'].items():
                if name in a:raise ValueError('Missing indicator name collision')
                a[name]=a[c].isna().astype(float)
        else:
            step['bounds']={};keep=pd.Series(True,index=a.index)
            for c in numeric:
                if not a[c].notna().any():step['bounds'][c]=None;continue
                q1,q3=a[c].quantile([.25,.75]);lower=float(q1-settings['iqr_multiplier']*(q3-q1));upper=float(q3+settings['iqr_multiplier']*(q3-q1))
                step['bounds'][c]=[lower,upper]
                if action=='iqr_drop':keep&=~((a[c]<lower)|(a[c]>upper))
                else:a[c]=a[c].clip(lower,upper)
            if action=='iqr_drop':
                step['removed_training_rows']=a.index[~keep].tolist();a=a.loc[keep].copy()
        step['training_rows_after']=len(a);steps.append(step)
    if len(a)<2 or not len(a.columns):raise ValueError('Fixed pipeline removed training sample/features')
    # The original extra E1 arms call the same none encoder after their steps.
    # That encoder recognizes generated numerical flags as numeric features.
    numeric=[c for c in a if pd.api.types.is_numeric_dtype(a[c])]
    categorical=[c for c in a if c not in numeric];encoder=None;blocks=[a[numeric].to_numpy(float)]
    names=['numeric:'+c for c in numeric]
    if categorical:
        values=('value:'+a[categorical].astype('string')).fillna('missing:')
        encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        blocks.append(encoder.fit_transform(values));names+=['category:'+c for c in encoder.get_feature_names_out(categorical)]
    matrix=np.hstack(blocks);selected=np.ones(matrix.shape[1],dtype=bool)
    if method=='aggressive':
        values=pd.DataFrame(matrix);scale=values.max()-values.min()
        variance=((values-values.min())/scale.replace(0,1)).var(ddof=0).fillna(0)
        selected=(variance>settings['low_variance']).to_numpy()
        steps.append({'action':'selection_log','scaled_variance':dict(zip(names,variance.tolist()))})
    if not selected.any():raise ValueError('All encoded features removed')
    # Selection is applied to encoded features, not raw-frame operations.
    executable=[s for s in steps if s['action']!='selection_log']
    state=Preprocessor(input_numeric,numeric,categorical,executable,encoder,selected,[n for n,k in zip(names,selected) if k])
    keep=pd.Series(original.index.isin(a.index),index=original.index)
    log={'method':method,'steps':steps,'train_rows_before':len(original),'train_rows_after':len(a),
        'removed_training_rows':original.index[~keep].tolist(),'output_columns':state.names,
        'training_row_ids':original.index.tolist(),'categorical_levels':
            {c:levels.tolist() for c,levels in zip(categorical,encoder.categories_)} if encoder is not None else {}}
    return state,matrix[:,selected],keep,log


def load_config(path):
    path=Path(path).resolve();c=read(path)
    if c.get('schema_version')!=1 or c.get('experiment')!='E1':raise ValueError('Expected E1 configuration schema 1')
    root=resolved(path.parent,c.get('project_root','..'))
    for k in ['output_dir','prepared_dir','archive_dir']:c[k]=str(resolved(root,c[k]))
    c['project_root']=str(root)
    if len(c['pairs'])<10 or len(c['seeds'])!=3 or len(set(c['seeds']))!=3 or c['n_splits']!=5:
        raise ValueError('E1 needs at least ten pairs, three seeds and five folds')
    if c['pipelines']!=PIPELINES or c['downstream_models']!=['xgboost','linear']:
        raise ValueError('This E1 driver implements the eight registered arms and two models')
    for pair in c['pairs']:
        if len(pair['tasks'])!=2 or {t['target'] for t in pair['tasks']}&set(pair['features']):raise ValueError('Two distinct targets must be excluded')
        if pair.get('group_column') in pair['features']:raise ValueError('Group identity cannot be a predictor')
    return c


def folder(c,pair,seed):return Path(c['prepared_dir'])/pair['name']/str(seed)


def development(c,pair,seed):
    base=folder(c,pair,seed);meta=read(base/'prepared.json')
    if meta['pair']!=pair or meta['seed']!=seed or meta['n_splits']!=5:raise ValueError('Prepared metadata differs')
    frame=pd.read_pickle(base/'development.pkl').set_index('__row_id');folds=read(base/'folds.json')
    index=read(base/'indices.json')
    if not frame.index.is_unique or frame.index.tolist()!=index['development']:raise ValueError('Development identities differ')
    if set(index['development'])&set(index['test']):raise ValueError('Development/test overlap')
    return frame,folds


def freeze(c):
    out=Path(c['output_dir']);path=out/'config.json'
    if path.exists():
        if read(path)!=c:raise ValueError('Frozen E1 config differs')
        return
    archive=read(Path(c['archive_dir'])/'config.json')
    for key in ['pairs','seeds','n_splits','model','preprocessing','pipelines','downstream_models','criteria']:
        if archive[key]!=c[key]:raise ValueError('Archive setting differs: '+key)
    out.mkdir(parents=True,exist_ok=True);write(path,c)
    code=out/'code';code.mkdir(exist_ok=True)
    for name in ['e1.py','io.py','scoring.py']:(code/name).write_bytes((Path(__file__).parent/name).read_bytes())
    write(out/'environment.json',{'at':now(),'system_version':__version__,
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost-cpu','joblib']}})
    write(out/'state.json',{'at':now(),'status':'frozen','test_opened':False,'lm_calls':0})


def replay_split(c,pair,seed):
    checkpoint=Path(c['output_dir'])/'replay_units'/pair['name']/(str(seed)+'.json')
    if checkpoint.exists():return read(checkpoint)
    frame,folds=development(c,pair,seed);coverage=[];rows=[]
    for fold in folds:
        a,b=fold['train'],fold['validation'];coverage+=b
        if set(a)&set(b) or set(a)|set(b)!=set(range(len(frame))):raise ValueError('Invalid development fold')
        if pair.get('group_column') and set(frame.iloc[a][pair['group_column']])&set(frame.iloc[b][pair['group_column']]):raise ValueError('Groups cross development folds')
        for method in c['pipelines']:
            base=Path(c['archive_dir'])/'jobs'/pair['name']/str(seed)/str(fold['fold'])/method
            log=read(base/'preprocessing.json')
            for task in pair['tasks']:
                for model in c['downstream_models']:
                    unit=base/task['name']/model;score=read(unit/'score.json')
                    if score['status']!='passed' or score['test_table_read']:raise ValueError('Ineligible development score')
                    expected={'pair':pair['name'],'seed':seed,'fold':fold['fold'],'pipeline':method,'task':task['name'],'downstream':model}
                    if any(score[k]!=v for k,v in expected.items()):raise ValueError('Development score identity mismatch')
                    pred=pd.read_csv(unit/'predictions.csv',float_precision='round_trip')
                    if pred.row_id.tolist()!=frame.index[b].tolist():raise ValueError('Prediction row order mismatch')
                    y=labels(frame.iloc[b],task).to_numpy()
                    np.testing.assert_array_equal(pred.y_true.to_numpy(dtype=score['target_dtype']),y)
                    actual=metrics(y,pred.prediction.to_numpy(dtype=score['prediction_dtype']),task)
                    if any(abs(value-score[key])>1e-9 for key,value in actual.items()):raise ValueError('Archived prediction score mismatch: '+str(unit))
                    if score['train_rows']!=log['train_rows_after'] or score['encoded_features']!=len(log['output_columns'] if 'output_columns' in log else log['common_encoder']['output_columns']):
                        raise ValueError('Development preprocessing dimensions mismatch')
                    rows.append({**expected,'type':task['type'],**actual,'reused':True})
    if sorted(coverage)!=list(range(len(frame))):raise ValueError('Validation folds do not partition development')
    split={'pair':pair['name'],'seed':seed,'development_rows':len(frame),
        'sealed_rows':len(read(folder(c,pair,seed)/'indices.json')['test']),'fivefold_ids_and_groups_checked':True}
    result={'rows':rows,'split':split};write(checkpoint,result);return result


def replay_archive(c):
    out=Path(c['output_dir']);marker=out/'development_replay.json'
    if marker.exists():
        if not read(marker)['passed']:raise ValueError('Previous development replay failed')
        return
    rows=[];splits=[];selected=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=c.get('workers',4)) as pool:
        futures=[pool.submit(replay_split,c,p,s) for p in c['pairs'] for s in c['seeds']]
        for future in as_completed(futures):
            result=future.result();rows+=result['rows'];splits.append(result['split'])
            print('CV_REPLAY',result['split']['pair'],result['split']['seed'],len(rows),flush=True)
    rows.sort(key=lambda r:(r['pair'],r['seed'],r['fold'],r['pipeline'],r['task'],r['downstream']))
    for pair in c['pairs']:
        for seed in c['seeds']:
            for task in pair['tasks']:
                metric='auc' if task['type']=='classification' else 'rmse'
                for model in c['downstream_models']:
                    means={m:float(np.mean([r[metric] for r in rows if r['pair']==pair['name'] and r['seed']==seed and r['task']==task['name'] and r['downstream']==model and r['pipeline']==m])) for m in c['pipelines']}
                    chosen=(max if metric=='auc' else min)(means,key=means.get)
                    selected.append({'pair':pair['name'],'seed':seed,'task':task['name'],'downstream':model,'pipeline':chosen,'metric':metric,'cv_scores':means})
    expected=len(c['pairs'])*2*3*5*8*2
    if len(rows)!=expected:raise ValueError('Development grid incomplete')
    write(out/'development_scores.json',rows);pd.DataFrame(rows).to_csv(out/'development_scores.csv',index=False)
    write(out/'selection_from_cv.json',selected);write(out/'split_checks.json',splits)
    write(marker,{'passed':True,'replayed_predictions':len(rows),'new_predictor_fits':0,'test_opened':False,
        'seconds':time.monotonic()-started,'source':c['archive_dir'],'note':'Historical development replay, not new independent evidence.'})


def model_factory(task,name,c):
    if name=='xgboost':
        return XGBClassifier(**c['model'],objective='binary:logistic',eval_metric='logloss') if task['type']=='classification' else XGBRegressor(**c['model'],objective='reg:squarederror',eval_metric='rmse')
    estimator=LogisticRegression(C=1.,max_iter=2000,random_state=0) if task['type']=='classification' else LinearRegression()
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),estimator)


def final_folder(c,pair,seed,method):return Path(c['output_dir'])/'final_models'/pair['name']/str(seed)/method


def fit_unit(c,pair,seed,method):
    out=final_folder(c,pair,seed,method);complete=out/'complete.json'
    if complete.exists():return read(complete)
    started=time.monotonic();frame,_=development(c,pair,seed)
    state,matrix,keep,log=fit_preprocessor(frame[pair['features']],method,c['preprocessing'])
    out.mkdir(parents=True,exist_ok=True);joblib.dump(state,out/'preprocessor.joblib');write(out/'preprocessing.json',log)
    records=[]
    for task in pair['tasks']:
        y=labels(frame,task).loc[keep].to_numpy()
        for name in c['downstream_models']:
            target=out/task['name']/name;target.mkdir(parents=True,exist_ok=True)
            if (target/'fit.json').exists() and (target/'model.joblib').exists():records.append(read(target/'fit.json'));continue
            start=time.monotonic();model=model_factory(task,name,c)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(matrix,y)
            joblib.dump(model,target/'model.joblib')
            check=model.predict(matrix[:3]);np.testing.assert_array_equal(check,joblib.load(target/'model.joblib').predict(matrix[:3]))
            record={'pair':pair['name'],'seed':seed,'pipeline':method,'task':task['name'],'downstream':name,
                'train_rows':len(y),'encoded_features':matrix.shape[1],'seconds':time.monotonic()-start,
                'warnings':[str(w.message) for w in caught],'test_opened':False,'save_replay_passed':True}
            write(target/'fit.json',record);records.append(record)
    result={'pair':pair['name'],'seed':seed,'pipeline':method,'fits':len(records),'seconds':time.monotonic()-started,'test_opened':False}
    write(complete,result);return result


def fit_all(c):
    out=Path(c['output_dir'])
    if not read(out/'development_replay.json')['passed']:raise ValueError('Development replay required')
    if (out/'all_plans_frozen.json').exists():return
    jobs=[(p,s,m) for p in c['pairs'] for s in c['seeds'] for m in c['pipelines']]
    expected=len(jobs)*4
    if expected>c['max_new_predictor_fits']:raise ValueError('Declared final fit budget exceeded')
    started=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=c.get('workers',4)) as pool:
        futures=[pool.submit(fit_unit,c,*j) for j in jobs]
        for future in as_completed(futures):
            result=future.result();done+=result['fits']
            write(out/'progress.json',{'at':now(),'status':'fitting_before_test','completed_models':done,'expected_models':expected,'test_opened':False,'seconds':time.monotonic()-started})
            print('FINAL_FIT',done,'/',expected,result['pair'],result['seed'],result['pipeline'],flush=True)
    for pair,seed,method in jobs:
        for task in pair['tasks']:
            for model in c['downstream_models']:
                if not (final_folder(c,pair,seed,method)/task['name']/model/'model.joblib').exists():raise FileNotFoundError('Missing final model')
    write(out/'all_plans_frozen.json',{'at':now(),'all_methods_fitted':True,'models':done,'test_opened':False,
        'development_selection':'selection_from_cv.json','test_dependent_selection':False,'seconds':time.monotonic()-started})


def evaluate(c):
    out=Path(c['output_dir'])
    if (out/'final_scores.json').exists():return
    freeze=read(out/'all_plans_frozen.json')
    if not freeze['all_methods_fitted']:raise ValueError('All arms must be frozen first')
    marker=Path(c['prepared_dir'])/c['test_use_marker']
    if marker.exists():raise ValueError('E1 sealed tests already entered evaluation; inspect existing results before continuing')
    use={'at':now(),'status':'started','output_dir':str(out),'all_plans_frozen':True,'models':freeze['models'],'scope':'E1 only; all eight prescribed arms, no adaptation after tests'}
    with marker.open('x') as f:json.dump(use,f,ensure_ascii=False,indent=2)
    write(out/'test_use.json',use);rows=[];checks=[];started=time.monotonic()
    for pair in c['pairs']:
        for seed in c['seeds']:
            source=folder(c,pair,seed);meta=read(source/'prepared.json')
            path=Path(meta['test_kept_at']) if meta.get('test_kept_at') else source/'test.pkl'
            # This is the only place in the driver that opens sealed data.
            test=pd.read_pickle(path).set_index('__row_id');dev,_=development(c,pair,seed)
            ids=read(source/'indices.json')
            if test.index.tolist()!=ids['test'] or set(test.index)&set(dev.index):raise ValueError('Final test row identity mismatch')
            if pair.get('group_column') and set(test[pair['group_column']])&set(dev[pair['group_column']]):raise ValueError('Final test group overlap')
            checks.append({'pair':pair['name'],'seed':seed,'test_rows':len(test),'rows_and_groups_disjoint':True,'file':str(path)})
            for method in c['pipelines']:
                base=final_folder(c,pair,seed,method);state=joblib.load(base/'preprocessor.joblib')
                matrix=state.transform(test[pair['features']]);assert len(matrix)==len(test)
                for task in pair['tasks']:
                    y=labels(test,task).to_numpy()
                    for name in c['downstream_models']:
                        model=joblib.load(base/task['name']/name/'model.joblib')
                        prediction=model.predict_proba(matrix)[:,1] if task['type']=='classification' else model.predict(matrix)
                        values=metrics(y,prediction,task);target=out/'test_predictions'/pair['name']/str(seed)/method/task['name']/name
                        target.mkdir(parents=True,exist_ok=True)
                        pd.DataFrame({'row_id':test.index,'y_true':y,'prediction':prediction}).to_csv(target/'predictions.csv',index=False)
                        saved=pd.read_csv(target/'predictions.csv',float_precision='round_trip')
                        actual=metrics(saved.y_true.to_numpy(dtype=y.dtype),saved.prediction.to_numpy(dtype=prediction.dtype),task)
                        if any(abs(actual[k]-v)>1e-9 for k,v in values.items()):raise ValueError('Final prediction replay mismatch')
                        row={'pair':pair['name'],'task':task['name'],'seed':seed,'pipeline':method,'downstream':name,
                            'type':task['type'],**values,'test_rows':len(y),'prediction_dtype':str(prediction.dtype),
                            'target_dtype':str(y.dtype),'prediction_replay_passed':True}
                        write(target/'score.json',row);rows.append(row)
            print('FINAL_TEST',pair['name'],seed,len(rows),flush=True)
    if len(rows)!=freeze['models']:raise ValueError('Final grid incomplete')
    write(out/'final_scores.json',rows);pd.DataFrame(rows).to_csv(out/'final_scores.csv',index=False)
    write(out/'final_test_checks.json',checks)
    use.update(status='completed',completed_at=now(),results=len(rows),seconds=time.monotonic()-started)
    write(marker,use);write(out/'test_use.json',use)


def benefit(before,after,metric):
    return after-before if metric in ('auc','f1') else (before-after)/max(abs(before),1e-12)


def summarize_rows(c,rows,stage):
    frame=pd.DataFrame(rows);task_rows=[];pair_rows=[];orders=[];seed_rows=[];details={}
    for model in c['downstream_models']:
        for pair in c['pairs']:
            tasks=[]
            for task in pair['tasks']:
                metric=c['primary_metric'][task['type']]
                rr=frame[(frame.pair==pair['name'])&(frame.task==task['name'])&(frame.downstream==model)]
                per_seed={}
                for method in c['pipelines']:
                    samples=rr[rr.pipeline==method]
                    if len(samples)!=(15 if stage=='development' else 3):raise ValueError('Incomplete summary cell')
                    for measured in (['auc','f1'] if task['type']=='classification' else ['rmse']):
                        values=samples.groupby('seed')[measured].mean().reindex(c['seeds'])
                        if values.isna().any():raise ValueError('Missing split metric')
                        task_rows.append({'stage':stage,'pair':pair['name'],'task':task['name'],'downstream':model,
                            'pipeline':method,'metric':measured,'mean':float(values.mean()),'std':float(values.std(ddof=1)),'data_splits':3})
                        for seed,value in values.items():seed_rows.append({'stage':stage,'pair':pair['name'],'task':task['name'],
                            'downstream':model,'pipeline':method,'metric':measured,'seed':int(seed),'value':float(value)})
                        if measured==metric:per_seed[method]=values.tolist()
                means={m:float(np.mean(per_seed[m])) for m in c['pipelines']}
                best=(max if metric=='auc' else min)(means,key=means.get)
                cutoff=c['criteria']['auc_difference' if metric=='auc' else 'rmse_relative_difference']
                tie=c['criteria']['auc_near_tie' if metric=='auc' else 'rmse_relative_near_tie']
                near=[m for m in c['pipelines'] if benefit(means[m],means[best],metric)<=tie]
                harm=[-benefit(a,b,metric) for a,b in zip(per_seed['none'],per_seed['standard'])]
                detail={'pair':pair['name'],'task':task['name'],'downstream':model,'metric':metric,'means':means,
                    'per_seed':per_seed,'best_descriptive':best,'near_best_descriptive':near,'cutoff':cutoff,
                    'standard_harm_mean':float(np.mean(harm)),'standard_harm':all(v>0 for v in harm) and float(np.mean(harm))>=cutoff}
                tasks.append(detail);details[(pair['name'],task['name'],model)]=detail
                for first,second in c['order_pairs']:
                    gains=[benefit(a,b,metric) for a,b in zip(per_seed[first],per_seed[second])]
                    orders.append({'stage':stage,'pair':pair['name'],'task':task['name'],'downstream':model,'metric':metric,
                        'first':first,'second':second,'gain_mean':float(np.mean(gains)),'gain_std':float(np.std(gains,ddof=1)),
                        'gains_by_seed':gains,'nonzero':any(abs(v)>1e-12 for v in gains),
                        'consistent_direction':all(v>1e-12 for v in gains) or all(v< -1e-12 for v in gains),
                        'material_consistent_effect':(all(v>0 for v in gains) or all(v<0 for v in gains)) and abs(float(np.mean(gains)))>=cutoff})
            a,b=tasks;regrets=[]
            for own,other in [(a,b),(b,a)]:
                values=[benefit(x,y,own['metric']) for x,y in zip(own['per_seed'][other['best_descriptive']],own['per_seed'][own['best_descriptive']])]
                regrets.append({'task':own['task'],'other_task_pipeline':other['best_descriptive'],
                    'regret_by_seed':values,'mean':float(np.mean(values)),'material_consistent':all(v>0 for v in values) and float(np.mean(values))>=own['cutoff']})
            different=not(set(a['near_best_descriptive'])&set(b['near_best_descriptive'])) and any(r['material_consistent'] for r in regrets)
            harmful=any(t['standard_harm'] for t in tasks)
            pair_rows.append({'stage':stage,'pair':pair['name'],'cohort':pair['cohort'],'downstream':model,
                'different_material_winners':different,'standard_harm':harmful,
                'either_registered_effect':bool(different or harmful),'tasks':tasks,'cross_task_regrets':regrets})
    return task_rows,seed_rows,pair_rows,orders,details


def report(c):
    out=Path(c['output_dir']);rows=read(out/'final_scores.json');dev=read(out/'development_scores.json')
    task_rows=[];seed_rows=[];evidence=[];orders=[];all_details={}
    for stage,source in [('development',dev),('final_test',rows)]:
        a,b,d,e,details=summarize_rows(c,source,stage)
        task_rows+=a;seed_rows+=b;evidence+=d;orders+=e;all_details[stage]=details
    pd.DataFrame(task_rows).to_csv(out/'pipeline_summary.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'seed_scores.csv',index=False)
    write(out/'pair_evidence.json',evidence);write(out/'order_effects.json',orders)
    pd.DataFrame([{**r,'gains_by_seed':json.dumps(r['gains_by_seed'])} for r in orders]).to_csv(out/'order_effects.csv',index=False)
    # A deployed choice is made by archived CV, never by the final test scores.
    selections=read(out/'selection_from_cv.json');selected_rows=[]
    for choice in selections:
        detail=all_details['final_test'][(choice['pair'],choice['task'],choice['downstream'])]
        i=c['seeds'].index(choice['seed']);score=detail['per_seed'][choice['pipeline']][i];raw=detail['per_seed']['none'][i]
        selected_rows.append({**{k:choice[k] for k in ['pair','task','downstream','seed','pipeline','metric']},
            'test_score':score,'none_test_score':raw,'gain_vs_none':benefit(raw,score,choice['metric']),
            'selected_before_test':True})
    pd.DataFrame(selected_rows).to_csv(out/'cv_selected_test.csv',index=False);write(out/'cv_selected_test.json',selected_rows)
    rankings=[]
    summary_by_model={}
    for model in c['downstream_models']:
        groups=[p for p in evidence if p['stage']=='final_test' and p['downstream']==model]
        order=[p for p in orders if p['stage']=='final_test' and p['downstream']==model]
        summary_by_model[model]={'pairs':len(groups),'standard_harm_pairs':sum(p['standard_harm'] for p in groups),
            'different_material_winner_pairs':sum(p['different_material_winners'] for p in groups),
            'either_registered_effect_pairs':sum(p['either_registered_effect'] for p in groups),
            'order_comparisons':len(order),'nonzero_order_comparisons':sum(p['nonzero'] for p in order),
            'material_consistent_order_comparisons':sum(p['material_consistent_effect'] for p in order)}
        for method in c['pipelines']:
            ranks=[];gains=[];wins=ties=losses=0
            for pair in c['pairs']:
                for task in pair['tasks']:
                    d=all_details['final_test'][(pair['name'],task['name'],model)]
                    rank=pd.Series(d['means']).rank(ascending=d['metric']!='auc',method='average')[method];ranks.append(float(rank))
                    gain=benefit(d['means']['none'],d['means'][method],d['metric']);gains.append(gain)
                    if gain>1e-10:wins+=1
                    elif gain< -1e-10:losses+=1
                    else:ties+=1
            rankings.append({'downstream':model,'pipeline':method,'average_task_rank':float(np.mean(ranks)),
                'wins_vs_none':wins,'ties_vs_none':ties,'losses_vs_none':losses,'win_rate':wins/len(ranks),
                'note':'Ranks across task means. Tiny positive effects count in wins; material effects are reported separately.'})
    pd.DataFrame(rankings).to_csv(out/'ranking_summary.csv',index=False)
    fits=[]
    for pair in c['pairs']:
        for seed in c['seeds']:
            for method in c['pipelines']:
                for task in pair['tasks']:
                    for model in c['downstream_models']:fits.append(read(final_folder(c,pair,seed,method)/task['name']/model/'fit.json'))
    costs={'historical_development_predictor_fits_reused':len(dev),'new_development_predictor_fits':0,
        'new_final_predictor_fits':len(fits),'new_preprocessors':len(c['pairs'])*3*8,
        'final_fit_seconds_sum':sum(r['seconds'] for r in fits),
        'final_fit_phase_seconds':read(out/'all_plans_frozen.json')['seconds'],
        'test_phase_seconds':read(out/'test_use.json')['seconds'],
        'development_replay_seconds':read(out/'development_replay.json')['seconds'],
        'lm_calls':0,'input_tokens':0,'output_tokens':0,'gpu_bytes':0,'sft_updates':0,'grpo_updates':0}
    write(out/'costs.json',costs)
    summary={'at':now(),'status':'completed','experiment':'E1','pairs':len(c['pairs']),'tasks':len(c['pairs'])*2,
        'pipelines':len(c['pipelines']),'data_split_seeds':c['seeds'],'historical_cv_predictions_replayed':len(dev),
        'new_final_results':len(rows),'test_evaluation_complete':True,'all_arms_frozen_before_test':True,
        'prediction_score_replay_passed':all(r['prediction_replay_passed'] for r in rows),
        'by_model':summary_by_model,'lm_or_training_effect_evaluated':False,'next_stage_started':False}
    write(out/'summary.json',summary);write(out/'state.json',summary)
    names={'credit':'信贷','medical':'住院','credit_card':'信用卡','bike':'自行车','student':'学生','parkinsons':'帕金森',
        'appliances':'家电能耗','air_quality':'空气质量','residential':'住宅建筑','sml2010':'SML2010'}
    x=summary_by_model['xgboost'];linear=summary_by_model['linear']
    lines=['# E1正式固定流水线实验结果','',f"完成时间：{summary['at']}。十对二十目标、八流水线、三套独立数据划分、XGBoost及逻辑/线性回归。",'',
        f"主XGBoost：{x['standard_harm_pairs']}/10对标准清洗稳定有害；{x['different_material_winner_pairs']}/10对达到预设两任务不同最优及实际损失条件。",
        f"线性复核：对应为{linear['standard_harm_pairs']}/10和{linear['different_material_winner_pairs']}/10。",
        f"顺序比较：XGBoost {x['nonzero_order_comparisons']}/{x['order_comparisons']}非零，线性{linear['nonzero_order_comparisons']}/{linear['order_comparisons']}非零；实际且三种子同向分别{x['material_consistent_order_comparisons']}、{linear['material_consistent_order_comparisons']}项。",'',
        '## 封存测试的分项结论','',
        '|任务对|XGB标准有害|XGB不同最优|线性标准有害|线性不同最优|','|---|---|---|---|---|']
    for pair in c['pairs']:
        ds=[next(p for p in evidence if p['stage']=='final_test' and p['pair']==pair['name'] and p['downstream']==m) for m in c['downstream_models']]
        vals=[names.get(pair['name'],pair['name'])]+['是' if d[k] else '否' for d in ds for k in ['standard_harm','different_material_winners']]
        lines.append('|'+ '|'.join(vals)+'|')
    lines+=['','“标准有害”和“两任务最优不同”分别统计；前者即使两个任务都受害，也不证明任务条件化的必要性。近优/效应判据沿用旧E1，不是任务书另给的通关线或显著性检验。',
        '','## XGBoost完整二十目标','',
        '|表/目标|指标|不处理均值±SD|标准均值±SD|测试描述性最优|开发CV选择后测试均值±SD|','|---|---|---|---|---|---|']
    def fmt(v):return f'{v:.6g}'
    for pair in c['pairs']:
        for task in pair['tasks']:
            d=all_details['final_test'][(pair['name'],task['name'],'xgboost')]
            vals=[r['test_score'] for r in selected_rows if r['pair']==pair['name'] and r['task']==task['name'] and r['downstream']=='xgboost']
            score=lambda m:fmt(d['means'][m])+' ± '+fmt(float(np.std(d['per_seed'][m],ddof=1)))
            lines.append('|'+ '|'.join([names.get(pair['name'],pair['name'])+'/'+task['name'],d['metric'],score('none'),score('standard'),d['best_descriptive'],fmt(float(np.mean(vals)))+' ± '+fmt(float(np.std(vals,ddof=1)))])+'|')
    lines+=['','完整分类F1、线性复核及全部八条流水线见pipeline_summary.csv；每种子及开发结果单独保存。不同单位RMSE不直接平均。测试最佳是预登记描述，不是用测试选择部署方案。',
        '','## 计算与复核','',
        f"复用并逐项复算{len(dev)}条历史开发预测，新增开发拟合0；新增最终拟合{len(fits)}，全部方法与CV选择先冻结，再读取各测试文件一次完成统一评价。",
        '实际种子为原E1的20261006/20261007/20261008，未重新命名为42/123/456；R15单开发池结果未导入。三种子均值±样本标准差，非置信区间。独立划分之间可能共享行，所以不是三份互不重叠新数据。',
        f"最终拟合阶段墙钟{costs['final_fit_phase_seconds']:.2f}秒；测试阶段{costs['test_phase_seconds']:.2f}秒；语言模型调用、token、GPU使用、SFT/GRPO更新均0。",
        '全部960份最终预测保存并重算分数通过，预处理状态/模型保存并重载检查；原历史分数不覆盖。开发复算是复用而非新增独立确认。',
        '','## 边界及后续','',
        '本实验检验固定处理对任务的影响，不检验语言模型、SFT或GRPO增益；不能将任何现象支持率写成模型胜率。',
        '原先已观察过开发结果的十表不是全新盲选样本，最终模型和操作参数在测试前冻结；保留全部负结果，不按测试改变任务或阈值。',
        '这些测试已用于E1最终评价；后续不能再标作未触达，也不能据此调方法后沿用同一测试宣称独立确认。新的完全未见表泛化评价须预留未开发数据。',
        '本次不自动进入E2–E7、不重启训练、不改系统规划方法或晋升模型。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run(path,stage='all'):
    c=load_config(path);freeze(c)
    if stage in ('all','verify'):replay_archive(c)
    if stage in ('all','fit'):fit_all(c)
    if stage in ('all','test'):evaluate(c)
    if stage in ('all','report'):return report(c)
    return {'stage':stage,'output_dir':c['output_dir']}
