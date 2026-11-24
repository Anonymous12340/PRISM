"""Training-only policy evaluation, with shared inner-fold fits and caches."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,f1_score
from xgboost import XGBClassifier,XGBRegressor
from .operations import fit_bundle,apply_plan,POLICIES
from .io import read,write


def splits(x,y,task,n,seed,groups=None):
    if groups is not None:
        cv=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed) if task['type']=='classification' else GroupKFold(n_splits=n,shuffle=True,random_state=seed)
        return list(cv.split(x,y,groups))
    cv=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed) if task['type']=='classification' else KFold(n_splits=n,shuffle=True,random_state=seed)
    return list(cv.split(x,y))


def fit_encoder(x):
    ns=[c for c in x if pd.api.types.is_numeric_dtype(x[c])]
    return {'numeric':ns,'categorical':{c:sorted(x[c].dropna().astype(str).unique()) for c in x if c not in ns}}


def encode(x,state):
    blocks=[pd.to_numeric(x[c],errors='raise').replace([np.inf,-np.inf],np.nan).to_numpy(float).reshape(-1,1) for c in state['numeric']]
    for c,levels in state['categorical'].items():
        s=x[c].astype('string')
        blocks += [(s==level).fillna(False).to_numpy(float).reshape(-1,1) for level in levels]
        blocks.append(s.isna().to_numpy(float).reshape(-1,1))
    return np.concatenate(blocks,axis=1).astype(np.float32)


def fit_predictor(train,y,task,parameters):
    encoder=fit_encoder(train);cls=XGBClassifier if task['type']=='classification' else XGBRegressor
    model=cls(**parameters);model.fit(encode(train,encoder),np.asarray(y))
    return {'model':model,'encoder':encoder,'task':task}


def predict(predictor,x):
    matrix=encode(x,predictor['encoder']);model=predictor['model']
    return model.predict_proba(matrix)[:,1] if predictor['task']['type']=='classification' else model.predict(matrix)


def metrics(y,prediction,task):
    if task['type']=='classification':return {'auc':float(roc_auc_score(y,prediction)),
        'f1':float(f1_score(y,np.asarray(prediction)>=.5,zero_division=0))}
    return {'rmse':float(np.sqrt(np.mean((np.asarray(y)-prediction)**2)))}


def utility(score,task):return score['auc'] if task['type']=='classification' else -score['rmse']


class SourceScorer:
    """Only outer-training rows enter this object; never outer/test labels."""
    def __init__(self,directory,source=None):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.new_fits=0
        path=self.directory/'source.joblib'
        if not path.exists():
            if source is None:raise FileNotFoundError(path)
            joblib.dump(source,path)
        self.source=joblib.load(path);s=self.source
        if source is not None and (not source['x'].equals(s['x']) or not source['y'].equals(s['y'])):
            raise ValueError('Scorer cache belongs to different training data')
        if source is not None:
            keys=['model','imputer','inner_folds','repair_columns']
            if any(source['config'][k]!=s['config'][k] for k in keys) or source['task']!=s['task'] or source['seed']!=s['seed']:
                raise ValueError('Scorer cache belongs to different evaluation settings')
            if not source.get('reference',source['x']).equals(s.get('reference',s['x'])):
                raise ValueError('Scoring reference changed')
            a,b=source.get('groups'),s.get('groups')
            if (a is None)!=(b is None) or (a is not None and not a.equals(b)):
                raise ValueError('Group split changed')
        self.x,self.y,self.task=s['x'],s['y'],s['task']
        self.reference=s.get('reference',self.x);self.config=s['config']
        if self.task['target'] in self.x:raise ValueError('Target leaked into scorer features')
        self.fold_splits=splits(self.x,self.y,self.task,self.config['inner_folds'],s['seed'],s.get('groups'))
        write(self.directory/'splits.json',[{'train_ids':self.x.index[a].tolist(),'validation_ids':self.x.index[b].tolist()} for a,b in self.fold_splits])

    def bundles(self,i):
        path=self.directory/'bundles'/f'{i}.joblib'
        if not path.exists():
            a,_=self.fold_splits[i];path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump({c:fit_bundle(self.x.iloc[a],c,self.config['imputer']) for c in self.config['repair_columns']},path)
            saved=joblib.load(path);write(path.with_suffix('.json'),{'fitted_imputers':sum(b['model'] is not None for b in saved.values())})
        return joblib.load(path)

    def evaluate(self,plan):
        if set(plan)-set(self.config['repair_columns']) or any(a not in POLICIES for a in plan.values()):
            raise ValueError('Unknown plan column or action')
        ordered={c:plan.get(c,'keep') for c in self.config['repair_columns']}
        key='__'.join(ordered.values());path=self.directory/'scores'/f'{key}.json'
        if path.exists():return read(path)
        values=[]
        for i,(a,b) in enumerate(self.fold_splits):
            bundles=self.bundles(i)
            train,_=apply_plan(self.x.iloc[a],ordered,bundles)
            valid,_=apply_plan(self.reference.iloc[b],ordered,bundles)
            folder=self.directory/'models'/key;folder.mkdir(parents=True,exist_ok=True)
            model_path=folder/f'{i}.joblib'
            if model_path.exists():predictor=joblib.load(model_path)
            else:
                predictor=fit_predictor(train,self.y.iloc[a],self.task,self.config['model'])
                joblib.dump(predictor,model_path);self.new_fits+=1
            values.append(metrics(self.y.iloc[b],predict(predictor,valid),self.task))
        average={k:float(np.mean([v[k] for v in values])) for k in values[0]}
        result={'plan':ordered,'scores':average,'fold_scores':values,'utility':utility(average,self.task)}
        write(path,result);return result

    def needed_fits(self,plan):
        key='__'.join(plan.get(c,'keep') for c in self.config['repair_columns'])
        if (self.directory/'scores'/f'{key}.json').exists():return 0
        return sum(not (self.directory/'models'/key/f'{i}.joblib').exists() for i in range(len(self.fold_splits)))
