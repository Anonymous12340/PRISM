"""Prepare prespecified public E2 tables and independent group-safe splits."""
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold,GroupShuffleSplit
from .io import read,write
from .e2_operations import normalize
from .scoring import splits


def config(path):
    path=Path(path).resolve();c=read(path);root=(path.parent/c['project_root']).resolve();c['project_root']=str(root)
    for name in ['registry','data_dir','output_dir','classical_python','upstream']:c[name]=str(root/c[name])
    registry=read(c['registry']);c['datasets']=registry['datasets']
    if c['seeds']!=[42,123,456] or c['folds']!=5 or c['inner_folds']!=3 or c['budgets']!=[5,10,20]:raise ValueError('Frozen E2 split/budget settings changed')
    if c.get('planning_scope')=='complete_development_only':
        from .e2_pilot import METHODS
        if c.get('data_source')!='OpenML' or registry.get('source')!='OpenML':raise ValueError('Revised main E2 requires OpenML sources')
        if len(c['datasets'])!=20 or len({d['name'] for d in c['datasets']})!=20:raise ValueError('Main E2 requires twenty distinct registered tables')
        if len({d['openml_id'] for d in c['datasets']})!=20 or sum(d['type']=='classification' for d in c['datasets'])!=10:
            raise ValueError('Main E2 requires twenty OpenML identities and ten classification tasks')
        if c['methods']!=METHODS or c['lm_budget']!=20 or c['unit_physical_limit']!=100 or c['generation']['temperature']!=0:
            raise ValueError('Registered main E2 methods/budgets/deterministic generation changed')
        if c['resources']['max_calls_formal']!=20*3*5*20:raise ValueError('Main E2 request limit must be 6000')
        if c['data_dir']==str(root/'data/e2_formal_20261105') or c['output_dir']==str(root/'results/e2_formal_20261105'):
            raise ValueError('Revised main E2 must use independent data and result directories')
    return c


def labels(values,item):
    if item['type']=='regression':return pd.to_numeric(values,errors='raise').astype(float)
    if item.get('label_rule')=='greater_than_zero':return (pd.to_numeric(values,errors='raise')>0).astype(int)
    positive=item['positive']
    if isinstance(positive,(int,float)):result=pd.to_numeric(values,errors='raise').eq(positive)
    else:result=values.astype(str).str.casefold().eq(str(positive).casefold())
    if result.nunique()!=2 or values.nunique()!=2:raise ValueError('Expected exactly the registered two source classes')
    return result.astype(int)


def source_table(c,item):
    root=Path(c['data_dir'])/'raw'
    if c.get('data_source')!='OpenML':
        return pd.read_csv(root/(item['name']+'.csv'),na_values=['?']),read(root/(item['name']+'_metadata.json'))['data']
    import arff
    folder=root/str(item['openml_id']);desc=read(folder/'description.json')['data_set_description']
    if int(desc['id'])!=item['openml_id'] or int(desc['version'])!=item['openml_version']:
        raise ValueError('OpenML source id/version differs from registered table')
    with (folder/'data.arff').open() as f:opened=arff.load(f)
    raw=pd.DataFrame(opened['data'],columns=[v[0] for v in opened['attributes']])
    raw=pd.read_csv(StringIO(raw.to_csv(index=False)),na_values=['?'])
    mapping=item.get('openml_column_mapping',{});raw=raw.rename(columns=mapping)
    native=read(folder/'features.json')['data_features']['feature'];variables=[]
    for v in native:
        name=mapping.get(v['name'],v['name'])
        role='Target' if name==item['target'] or v.get('is_target')=='true' else ('ID' if v.get('is_row_identifier')=='true' else 'Feature')
        variables.append({'name':name,'role':role,'type':'Categorical' if v['data_type'] in ['nominal','string'] else 'Continuous'})
    if item.get('group_metadata_file'):
        groups=pd.read_csv(root/item['group_metadata_file'])
        if len(groups)!=len(raw) or groups['row_id'].tolist()!=list(range(len(raw))):
            raise ValueError('Auxiliary group metadata row mapping differs')
        raw[item['group_column']]=groups[item['group_column']].to_numpy()
    meta={'repository_url':'https://www.openml.org/d/'+str(item['openml_id']),
        'dataset_doi':None,'num_instances':len(raw),'variables':variables,
        'abstract':desc.get('description','')[:c.get('background_char_limit',4000)],'openml_id':item['openml_id'],
        'openml_version':item['openml_version'],'license':desc.get('licence')}
    return raw,meta


def cohort(c,item):
    raw,meta=source_table(c,item);raw.index=pd.Index(range(len(raw)),name='__row_id')
    original=len(raw);header=raw[item['target']].astype(str).eq(item['target'])
    missing=raw[item['target']].isna();excluded=raw.index[header|missing].tolist();raw=raw.loc[~(header|missing)].copy()
    registry=read(c['registry']);limit=registry['research_row_limit']
    if len(raw)>limit:
        chosen=np.sort(np.random.default_rng(registry['sampling_seed']).choice(len(raw),limit,replace=False));raw=raw.iloc[chosen]
    y=labels(raw[item['target']],item)
    declared_targets={v['name'] for v in meta.get('variables',[]) if v['role']=='Target'}|{item['target']}
    ids={v['name'] for v in meta.get('variables',[]) if v['role']=='ID'}
    excluded_columns=declared_targets|ids|set(item['exclude'])|{item.get('group_column')}
    features=[v for v in raw if v not in excluded_columns]
    x=raw[features].copy();dates=item.get('date_columns',{});cats=set(item.get('categorical',[]))
    for variable in meta.get('variables',[]):
        if variable.get('type') in ['Categorical','Binary'] and variable['name'] in x:cats.add(variable['name'])
    if item.get('categorical_all'):cats=set(x)
    for v in x:
        if v in dates:x[v]=pd.to_datetime(x[v],format=dates[v],errors='raise').astype('int64').astype(float)/1.e9
        elif v in cats:x[v]=x[v].astype('string').astype(object).where(x[v].notna(),np.nan)
        else:x[v]=pd.to_numeric(x[v],errors='raise').astype(float)
    x=normalize(x)
    # Exact predictor duplicates always share a split. Explicit day/object groups
    # are merged with duplicates, so either relation suffices to keep rows together.
    parent=list(range(len(x)))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    relations=[pd.util.hash_pandas_object(x,index=False).to_numpy()]
    if item.get('group_column'):relations.append(raw[item['group_column']].astype(str).str.strip().str.casefold().to_numpy())
    if item.get('group_day'):
        col=item['group_day'];parsed=pd.to_datetime(raw[col],format=dates[col],errors='raise')
        relations.append(parsed.dt.strftime('%Y-%m-%d').to_numpy())
    for relation in relations:
        first={}
        for i,value in enumerate(relation):
            if value in first:union(first[value],i)
            else:first[value]=i
    groups=pd.Series([find(i) for i in range(len(x))],index=x.index,name='__group')
    task={'name':item['name'],'target':item['target'],'type':item['type'],'description':item['task_description']}
    receipt={'name':item['name'],'uci_id':item['uci_id'],'source':meta['repository_url'],'doi':meta.get('dataset_doi'),
        'source_rows':original,'excluded_target_missing_or_repeated_header_row_ids':excluded,'research_rows':len(x),
        'feature_count':len(x.columns),'features':list(x),'excluded_columns':sorted(v for v in excluded_columns if v),
        'categorical_columns':[v for v in x if not pd.api.types.is_numeric_dtype(x[v])],
        'date_columns':{v:'unix_seconds' for v in dates},'common_source_date_formats':dates,
        'groups':int(groups.nunique()),'group_rule':item.get('group_day') or item.get('group_column') or 'exact_predictor_duplicates',
        'task':task,'background':meta.get('abstract',''),'source_metadata_rows':meta.get('num_instances'),
        'row_count_discrepancy_retained':original!=meta.get('num_instances'),'model_scores_used_for_selection':False}
    if c.get('data_source')=='OpenML':
        receipt.update(data_source='OpenML',openml_id=item['openml_id'],openml_version=item['openml_version'],
            source_license=meta['license'],group_metadata_source=item.get('group_metadata_source'))
    return x,y,groups,receipt


def _check(x,y,groups,a,b,task):
    if set(x.index[a])&set(x.index[b]) or set(groups.iloc[a])&set(groups.iloc[b]):raise ValueError('Row/group overlap')
    if len(a)+len(b)!=len(x):raise ValueError('Partition incomplete')
    if task['type']=='classification' and (y.iloc[a].nunique()!=2 or y.iloc[b].nunique()!=2):raise ValueError('A classification partition lacks a class')


def prepare(c):
    out=Path(c['data_dir'])/'prepared';manifest=[];checks=0
    for item in c['datasets']:
        directory=out/item['name'];directory.mkdir(parents=True,exist_ok=True);done=directory/'complete.json'
        if done.exists():
            previous=read(done)
            if previous['dataset_settings']!=item:raise ValueError('Prepared dataset settings differ')
            manifest.append(read(directory/'manifest.json'));checks+=previous['partition_checks'];continue
        x,y,groups,receipt=cohort(c,item);local_checks=0
        previous_tests=set()
        for seed in c['seeds']:
            folder=directory/str(seed);folder.mkdir(exist_ok=True)
            cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed) if item['type']=='classification' else GroupKFold(5,shuffle=True,random_state=seed)
            candidate=0
            if item.get('holdout')=='group_shuffle_unique':
                for candidate,(development,test) in enumerate(GroupShuffleSplit(100,test_size=.2,random_state=seed).split(x,y,groups)):
                    signature=tuple(sorted(x.index[test]))
                    if signature in previous_tests or not .1<=len(test)/len(x)<=.3:continue
                    if item['type']=='classification' and (y.iloc[development].nunique()!=2 or y.iloc[test].nunique()!=2):continue
                    break
                else:raise ValueError('No structurally valid independent group holdout')
            else:development,test=next(cv.split(x,y,groups))
            signature=tuple(sorted(x.index[test]))
            if signature in previous_tests:raise ValueError('Two data seeds produced the same sealed partition')
            previous_tests.add(signature)
            _check(x,y,groups,development,test,receipt['task']);local_checks+=1
            source={'x':x.iloc[development],'y':y.iloc[development],'groups':groups.iloc[development],
                'task':receipt['task'],'pair':{'name':item['name'],'background':receipt['background'],'tasks':[receipt['task']],
                'features':list(x)},'seed':seed,'date_columns':receipt['date_columns']}
            joblib.dump(source,folder/'development.joblib')
            joblib.dump({'x':x.iloc[test],'y':y.iloc[test],'groups':groups.iloc[test]},folder/'sealed_test.joblib')
            main=c.get('planning_scope')=='complete_development_only';records=[]
            if main:
                for a,b in splits(source['x'],source['y'],source['task'],c['inner_folds'],seed,source['groups']):
                    _check(source['x'],source['y'],source['groups'],a,b,source['task']);local_checks+=1
                    records.append({'train':a.tolist(),'validation':b.tolist()})
                write(folder/'main_inner_folds.json',records)
            else:
                folds=splits(source['x'],source['y'],source['task'],5,seed,source['groups'])
                for a,b in folds:
                    _check(source['x'],source['y'],source['groups'],a,b,source['task']);local_checks+=1
                    xx,yy,gg=source['x'].iloc[a],source['y'].iloc[a],source['groups'].iloc[a]
                    for ia,ib in splits(xx,yy,source['task'],3,seed,gg):_check(xx,yy,gg,ia,ib,source['task']);local_checks+=1
                    records.append({'train':a.tolist(),'validation':b.tolist()})
                write(folder/'folds.json',records)
            write(folder/'split.json',{'seed':seed,'development_row_ids':source['x'].index.tolist(),'test_row_ids':x.index[test].tolist(),
                'development_rows':len(development),'test_rows':len(test),'test_fraction_actual':len(test)/len(x),
                'holdout_rule':item.get('holdout','first_group_cv_fold'),'holdout_candidate_index':candidate,
                'group_split':True,'test_evaluated':False,'outer_folds':0 if main else 5,'inner_folds':3})
        write(directory/'manifest.json',receipt);write(done,{'dataset_settings':item,'partition_checks':local_checks,'test_evaluated':False})
        manifest.append(receipt);checks+=local_checks
    write(out/'manifest.json',manifest)
    result={'datasets':len(manifest),'classification':sum(v['task']['type']=='classification' for v in manifest),
        'regression':sum(v['task']['type']=='regression' for v in manifest),'split_seeds':c['seeds'],'partition_checks':checks,
        'test_evaluated':False,'model_fits':0,'lm_calls':0,'scores_used_for_dataset_selection':False}
    write(out/'preparation.json',result);return result
