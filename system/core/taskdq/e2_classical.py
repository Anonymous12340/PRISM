"""Isolated native OpenFE/AutoFeat fit/apply worker for the E2 engineering batch."""
import argparse
import os
from pathlib import Path
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from .io import read,write


def openfe_aliases(frame):
    """Keep upstream formula parsing independent of semantic source column names."""
    return {column:f'F{i:04d}' for i,column in enumerate(frame.columns)}


def native_transform(train,valid,features):
    from openfe.FeatureGenerator import FNode
    from openfe.utils import tree_to_formula
    a,b=pd.DataFrame(index=train.index),pd.DataFrame(index=valid.index);states=[]
    for i,node in enumerate(features):
        name='__native_'+str(i);kind=node.name;record={'name':name,'kind':kind,'formula':tree_to_formula(node)}
        if kind=='freq':
            column=node.children[0].name;mapping=train[column].value_counts()
            left,right=train[column].map(mapping),valid[column].map(mapping);record.update(column=column,mapping=mapping)
        elif kind.startswith('GroupByThen'):
            column,group=[child.name for child in node.children]
            if kind=='GroupByThenRank':
                groups={key:np.sort(s.dropna().to_numpy()) for key,s in train.groupby(group,observed=True)[column]}
                def rank(frame):
                    values=[]
                    for value,key in zip(frame[column],frame[group]):
                        ref=groups.get(key,np.array([]))
                        values.append(np.nan if pd.isna(value) or not len(ref) else min(1.,sum(np.searchsorted(ref,value,side=s) for s in ['left','right'])/(2*len(ref))+1/(2*len(ref))))
                    return np.asarray(values)
                left,right=rank(train),rank(valid);record['groups']=groups
            else:
                mapping=train.groupby(group,observed=True)[column].agg(kind.removeprefix('GroupByThen').lower())
                left,right=train[group].map(mapping),valid[group].map(mapping);record['mapping']=mapping
            record.update(column=column,group=group)
        elif kind in ['Combine','CombineThenFreq']:
            ca,cb=[child.name for child in node.children]
            def combine(x):return (x[ca].astype(str)+'_'+x[cb].astype(str)).where(x[ca].notna()&x[cb].notna())
            left,right=combine(train),combine(valid)
            if kind=='CombineThenFreq':
                mapping=left.value_counts();left,right=left.map(mapping),right.map(mapping);record['mapping']=mapping
            else:
                levels=left.dropna().unique();left=pd.Categorical(left,categories=levels);right=pd.Categorical(right,categories=levels)
                record['levels']=levels
        else:
            if not all(isinstance(child,FNode) for child in node.children):raise ValueError('Pilot permits native order-one candidates only')
            left,right=node.calculate(train),node.calculate(valid)
        if kind=='Combine':a[name]=pd.Series(left,index=train.index).astype(object);b[name]=pd.Series(right,index=valid.index).astype(object)
        else:a[name]=np.asarray(left,dtype=float);b[name]=np.asarray(right,dtype=float)
        states.append(record)
    return a,b,states


def run(folder,method):
    base=Path(folder);meta=read(base/'input.json');out=base/method;out.mkdir(exist_ok=True)
    if (out/'generation.json').exists() and read(out/'generation.json')['status']=='passed':return
    started=time.monotonic();x=pd.read_csv(base/'train.csv',float_precision='round_trip').set_index('__row_id')
    valid=pd.read_csv(base/'valid.csv',float_precision='round_trip').set_index('__row_id');y=np.load(base/'labels.npy')
    for c in meta['categorical']:
        x[c]=x[c].astype('string').astype(object);valid[c]=valid[c].astype('string').astype(object)
    np.random.seed(42)
    try:
        if method=='openfe':
            from openfe import OpenFE
            from openfe.openfe import get_candidate_features
            # Upstream serializes generated features as formulas and parses them
            # again in selection. A hyphen in a source name becomes subtraction.
            # Only the native worker sees aliases; LM prompts keep source names.
            aliases=openfe_aliases(x);x=x.rename(columns=aliases);valid=valid.rename(columns=aliases)
            cats=[c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
            for c in cats:
                levels=x[c].dropna().astype(str).unique();x[c]=pd.Categorical(x[c].astype('string'),categories=levels)
                valid[c]=pd.Categorical(valid[c].astype('string'),categories=levels)
            positions=np.arange(len(x));groups=np.load(base/'groups.npy',allow_pickle=True) if (base/'groups.npy').exists() else None
            if groups is not None:train_ids,val_ids=next(GroupShuffleSplit(1,test_size=.25,random_state=42).split(x,groups=groups))
            else:train_ids,val_ids=train_test_split(positions,test_size=.25,random_state=42,stratify=y if meta['task']['type']=='classification' else None)
            # pandas.Index is required by upstream stage2_select(). list breaks .to_list().
            generator=OpenFE();features=generator.fit(data=x.reset_index(drop=True),label=pd.DataFrame(y),task=meta['task']['type'],
                train_index=pd.Index(train_ids),val_index=pd.Index(val_ids),categorical_features=cats,
                n_data_blocks=4,min_candidate_features=200,n_jobs=2,seed=0,feature_boosting=True,
                tmp_save_path=str(out/'native_training.feather'),verbose=False)
            selected=features[:20];a,b,states=native_transform(x,valid,selected)
            joblib.dump({'nodes':selected,'states':states,'original_train':x,'column_aliases':aliases},out/'generator.joblib')
            loaded=joblib.load(out/'generator.joblib');aa,bb,_=native_transform(loaded['original_train'],valid,loaded['nodes'])
            pd.testing.assert_frame_equal(a,aa);pd.testing.assert_frame_equal(b,bb)
            native={'ranked_features':len(features),'selected_features':len(selected),'train_index_type':'pandas.Index',
                'selection_training_row_ids':x.index[train_ids].tolist(),'selection_validation_row_ids':x.index[val_ids].tolist(),
                'formulas':[s['formula'] for s in states],'column_aliases':aliases,'feature_boosting':True}
        else:
            from autofeat import AutoFeatClassifier,AutoFeatRegressor
            from joblib import parallel_backend
            from .e2_autofeat_compat import install_constant_guard,constant_guard_backend
            install_constant_guard()
            numeric=[c for c in x if pd.api.types.is_numeric_dtype(x[c])]
            if not numeric:
                a=pd.DataFrame(index=x.index);b=pd.DataFrame(index=valid.index)
                joblib.dump({'numeric_empty':True},out/'generator.joblib')
                a.to_csv(out/'new_train.csv',index_label='__row_id');b.to_csv(out/'new_valid.csv',index_label='__row_id')
                write(out/'generation.json',{'status':'passed','method':method,'seconds':time.monotonic()-started,
                    'new_features':[],'categorical':[],'input_training_row_ids':x.index.tolist(),'application_row_ids':valid.index.tolist(),
                    'save_apply_replay_passed':True,'native':{'numeric_input_columns':[],'selected_features':[],
                        'note':'Native numeric-only AutoFeat has no applicable inputs; original categories retained.'},'test_opened':False})
                return
            median=x[numeric].median().fillna(0.);a0=x[numeric].fillna(median).replace([np.inf,-np.inf],0.)
            b0=valid[numeric].fillna(median).replace([np.inf,-np.inf],0.)
            names={c:f'x{i}' for i,c in enumerate(numeric)};a0=a0.rename(columns=names);b0=b0.rename(columns=names)
            cls=AutoFeatClassifier if meta['task']['type']=='classification' else AutoFeatRegressor
            jobs=int(os.environ.get('TASKDQ_AUTOFEAT_JOBS','2'))
            generator=cls(feateng_steps=2,featsel_runs=5,max_gb=1,n_jobs=jobs,verbose=0)
            with parallel_backend(constant_guard_backend()):
                raw=generator.fit_transform(a0,y)
            raw_valid=generator.transform(b0)
            selected=[c for c in raw if c not in a0][:32]
            a=raw[selected].copy();b=raw_valid[selected].copy()
            # AutoFeat returns positional RangeIndex; recover the unchanged input
            # row identities only after comparing a saved-state transformation.
            check=generator.transform(a0)[selected]
            np.testing.assert_allclose(a.to_numpy(float),check.to_numpy(float),rtol=0,atol=1e-9,equal_nan=True)
            a.index=x.index;b.index=valid.index
            joblib.dump({'model':generator,'median':median,'names':names,'selected':selected},out/'generator.joblib')
            loaded=joblib.load(out/'generator.joblib');pd.testing.assert_frame_equal(raw_valid,loaded['model'].transform(b0))
            native={'feateng_steps':2,'featsel_runs':5,'max_gb':1,'n_jobs':jobs,'numeric_input_columns':numeric,
                'zero_variance_scaling':'Upstream scaler retained; zero training standard deviation maps to zero after centering.',
                'constant_input_columns':[col for col in numeric if a0[names[col]].nunique(dropna=False)==1],
                'selected_features':selected,'formulas':{str(k):str(v) for k,v in generator.feature_formulas_.items() if k in selected},
                'original_categorical_features_retained_by_common_scorer':True}
        a.columns=['__native_'+str(i) for i in range(a.shape[1])];b.columns=list(a.columns)
        a.to_csv(out/'new_train.csv',index_label='__row_id');b.to_csv(out/'new_valid.csv',index_label='__row_id')
        write(out/'generation.json',{'status':'passed','method':method,'seconds':time.monotonic()-started,
            'new_features':list(a),'categorical':[c for c in a if not pd.api.types.is_numeric_dtype(a[c])],
            'input_training_row_ids':x.index.tolist(),'application_row_ids':valid.index.tolist(),
            'save_apply_replay_passed':True,'native':native,'test_opened':False})
    except Exception:
        write(out/'generation.json',{'status':'failed','error':traceback.format_exc(),'seconds':time.monotonic()-started})
        raise


def repair_autofeat_indices(folder):
    base=Path(folder);out=base/'autofeat';meta=read(base/'input.json');record=read(out/'generation.json')
    saved=joblib.load(out/'generator.joblib');x=pd.read_csv(base/'train.csv',float_precision='round_trip').set_index('__row_id')
    valid=pd.read_csv(base/'valid.csv',float_precision='round_trip').set_index('__row_id')
    columns=list(saved['names']);a0=x[columns].fillna(saved['median']).replace([np.inf,-np.inf],0.).rename(columns=saved['names'])
    b0=valid[columns].fillna(saved['median']).replace([np.inf,-np.inf],0.).rename(columns=saved['names'])
    a=saved['model'].transform(a0)[saved['selected']];b=saved['model'].transform(b0)[saved['selected']]
    old_a=pd.read_csv(out/'new_train.csv',float_precision='round_trip').set_index('__row_id')
    old_b=pd.read_csv(out/'new_valid.csv',float_precision='round_trip').set_index('__row_id')
    np.testing.assert_allclose(a.to_numpy(float),old_a.to_numpy(float),rtol=0,atol=1e-9,equal_nan=True)
    np.testing.assert_allclose(b.to_numpy(float),old_b.to_numpy(float),rtol=0,atol=1e-9,equal_nan=True)
    if old_a.index.tolist()==x.index.tolist() and old_b.index.tolist()==valid.index.tolist():return
    old_a.to_csv(out/'new_train_before_index_alignment.csv',index_label='__row_id')
    old_b.to_csv(out/'new_valid_before_index_alignment.csv',index_label='__row_id')
    a.index=x.index;b.index=valid.index;a.columns=list(old_a);b.columns=list(old_b)
    a.to_csv(out/'new_train.csv',index_label='__row_id');b.to_csv(out/'new_valid.csv',index_label='__row_id')
    write(out/'index_alignment.json',{'passed':True,'training_rows':len(a),'application_rows':len(b),
        'saved_model_values_match_old_output':True,'new_model_fits':0,'new_lm_calls':0,'only_row_ids_corrected':True})


def apply_saved(folder,method,input_csv,output_csv):
    base=Path(folder);record=read(base/method/'generation.json');saved=joblib.load(base/method/'generator.joblib')
    frame=pd.read_csv(input_csv,float_precision='round_trip').set_index('__row_id')
    if method=='openfe':
        train=saved['original_train']
        frame=frame.rename(columns=saved.get('column_aliases',{}))
        for c in train.select_dtypes(include='category'):
            frame[c]=pd.Categorical(frame[c].astype('string'),categories=train[c].cat.categories)
        _,result,_=native_transform(train,frame,saved['nodes'])
    else:
        if saved.get('numeric_empty'):result=pd.DataFrame(index=frame.index)
        else:
            values=frame[list(saved['names'])].fillna(saved['median']).replace([np.inf,-np.inf],0.).rename(columns=saved['names'])
            result=saved['model'].transform(values)[saved['selected']].copy();result.index=frame.index
            result.columns=['__native_'+str(i) for i in range(result.shape[1])]
    result.to_csv(output_csv,index_label='__row_id')
    write(Path(output_csv).with_suffix('.json'),{'training_fits':0,'application_rows':len(result),'method':method,'saved_state_only':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True)
    parser.add_argument('--method',choices=['openfe','autofeat'],required=True);parser.add_argument('--repair-indices',action='store_true')
    parser.add_argument('--apply-csv');parser.add_argument('--output-csv')
    args=parser.parse_args()
    if args.apply_csv:apply_saved(args.folder,args.method,args.apply_csv,args.output_csv)
    elif args.repair_indices:repair_autofeat_indices(args.folder)
    else:run(args.folder,args.method)
