"""The shared seven-policy library. Fit functions accept features only."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .config import IMPUTER
from .io import plain

POLICIES=('keep','flag_only','median_clip','median_repair','median_repair_with_flag','model_repair','model_repair_with_flag')


def numeric(s):return pd.to_numeric(s,errors='raise').replace([np.inf,-np.inf],np.nan)


def candidate_mask(frame,state):
    x=numeric(frame[state['column']]);mask=x.isna()
    if state['lower'] is not None:mask=mask|(x<state['lower'])|(x>state['upper'])
    return mask


def predictors(frame,column):
    x=frame.drop(columns=[column]).copy()
    for c in x:
        x[c]=numeric(x[c]) if pd.api.types.is_numeric_dtype(x[c]) else x[c].astype(object).where(x[c].notna(),np.nan)
    return x


def fit_bundle(train,column,model_params=None):
    values=numeric(train[column]);q1,q3=values.quantile([.25,.75])
    state={'column':column,'lower':float(q1-1.5*(q3-q1)) if np.isfinite(q1) else None,
        'upper':float(q3+1.5*(q3-q1)) if np.isfinite(q3) else None,'training_row_ids':train.index.tolist()}
    mask=candidate_mask(train,state);trusted=~mask;x=predictors(train,column)
    observed=values.dropna();good=values.loc[trusted]
    state.update(median_observed=float(observed.median()) if len(observed) else None,
        median_trusted=float(good.median()) if len(good) else None,
        trusted_min=float(good.min()) if len(good) else None,trusted_max=float(good.max()) if len(good) else None,
        trusted_row_ids=train.index[trusted].tolist(),excluded_row_ids=train.index[mask].tolist(),feature_columns=list(x))
    model=None
    if trusted.sum()>=30 and len(x.columns):
        ns=[c for c in x if pd.api.types.is_numeric_dtype(x[c])];cs=[c for c in x if c not in ns];parts=[]
        if ns:parts.append(('numeric',SimpleImputer(strategy='median',keep_empty_features=True),ns))
        if cs:parts.append(('categorical',Pipeline([('fill',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
            ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cs))
        model=Pipeline([('features',ColumnTransformer(parts)),('regressor',HistGradientBoostingRegressor(**(model_params or IMPUTER)))])
        model.fit(x.loc[trusted],values.loc[trusted])
    return {'state':state,'model':model}


def available(bundle):
    actions=list(POLICIES)
    if bundle['model'] is None:actions=[a for a in actions if not a.startswith('model_repair')]
    if bundle['state']['median_trusted'] is None:actions=[a for a in actions if not a.startswith('median_repair')]
    if bundle['state']['median_observed'] is None:actions.remove('median_clip')
    return actions


def apply_policy(frame,policy,bundle):
    if policy not in available(bundle):raise ValueError(f'Unavailable repair policy: {policy}')
    s=bundle['state'];c=s['column'];result=frame.copy();mask=candidate_mask(frame,s);values=numeric(frame[c])
    if policy=='median_clip':result[c]=values.fillna(s['median_observed']).clip(s['lower'],s['upper'])
    elif policy.startswith('median_repair'):
        result[c]=values.astype(float);result.loc[mask,c]=s['median_trusted']
    elif policy.startswith('model_repair') and mask.any():
        result[c]=values.astype(float)
        result.loc[mask,c]=np.clip(bundle['model'].predict(predictors(frame.loc[mask],c)),s['trusted_min'],s['trusted_max'])
    flag=policy=='flag_only' or policy.endswith('_with_flag')
    if flag:
        name='__repair_candidate_flag__'+c
        if name in frame:raise ValueError('Input column collides with generated candidate flag: '+name)
        result[name]=mask.astype(float)
    events=[]
    for row in frame.index[mask]:
        old,new=frame.loc[row,c],result.loc[row,c];changed=not np.isclose(old,new,rtol=0,atol=1e-9,equal_nan=True)
        if changed or flag:events.append(plain({'row_id':row,'column':c,'policy':policy,'old_value':old,'new_value':new,
            'effect':'modified' if changed else 'flagged'}))
    pd.testing.assert_frame_equal(frame.loc[~mask],result.loc[~mask,frame.columns],check_dtype=False)
    return result,mask,events


def apply_plan(frame,plan,bundles):
    result=frame.copy();events=[]
    for column,bundle in bundles.items():
        result,_,found=apply_policy(result,plan.get(column,'keep'),bundle);events+=found
    return result,events
