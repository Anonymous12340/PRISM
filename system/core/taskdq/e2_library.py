"""Finite E2 operator library; train-fitted state, evaluation rows preserved."""
from copy import deepcopy
import difflib
import numpy as np
import pandas as pd
from .e2_operations import normalize
from .operations import fit_bundle,apply_policy

FAMILIES={'fill':'missing','model_repair':'missing','flag':'outlier','clip':'outlier',
    'drop_missing':'missing','drop_outlier':'outlier','canonicalize':'format','unit_scale':'format',
    'deduplicate':'format','ordinal':'encoding','onehot':'encoding','signed_log':'feature',
    'square':'feature','bin5':'feature','ratio':'feature','difference':'feature','sum':'feature',
    'product':'feature','frequency':'feature','group_mean':'feature','date_parts':'feature',
    'drop':'selection','drop_constant':'selection','drop_redundant':'selection',
    'numeric_parse':'format','affine':'feature'}


def menu(frame,date_columns=None,limit=160):
    dates=date_columns or {};numeric=[c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
    cats=[c for c in frame if c not in numeric and c not in dates];groups={k:[] for k in FAMILIES}
    def add(kind,**args):groups[kind].append({'kind':kind,'family':FAMILIES[kind],**args})
    # Candidate priority uses X only, never outcomes or held-out data.
    priority=sorted(frame,key=lambda c:(-float(frame[c].isna().mean()),str(c)))
    ns=[c for c in priority if c in numeric];cs=[c for c in priority if c in cats]
    for c in priority:
        if c in dates:continue
        for method in (['median','mean'] if c in numeric else ['mode']):add('fill',column=c,method=method)
        add('drop_missing',column=c);add('drop',column=c)
    for c in ns:
        add('model_repair',column=c);add('flag',column=c);add('clip',column=c);add('drop_outlier',column=c)
        for kind in ['signed_log','square','bin5']:add(kind,column=c)
        # Explicit physical-unit rescaling, not inferred mixed-unit recovery.
        for factor in [.001,.01,1000.]:add('unit_scale',column=c,factor=factor)
    for c in cs:
        for kind in ['canonicalize','ordinal','onehot','frequency']:add(kind,column=c)
    for c,fmt in dates.items():add('date_parts',column=c,format=fmt)
    for i,a in enumerate(ns):
        for b in ns[i+1:]:
            for kind in ['ratio','difference','sum','product']:add(kind,left=a,right=b)
    for c in cs:
        for n in ns:add('group_mean',group=c,column=n)
    add('deduplicate');add('drop_constant');add('drop_redundant',threshold=.99)
    # Round robin prevents a wide numeric table from excluding entire families.
    # None keeps every parameterization; historical calls keep the same 160
    # default and the exact same prefix/IDs.
    if limit is None: limit=sum(len(values) for values in groups.values())
    result=[];position=0
    while len(result)<limit:
        found=False
        for values in groups.values():
            if position<len(values):
                found=True;op=deepcopy(values[position]);op['id']=f'O{len(result):03}';result.append(op)
                if len(result)==limit:break
        if not found:break
        position+=1
    return result


def _bounds(x):
    a,b=x.quantile([.25,.75]);return (float(a-1.5*(b-a)),float(b+1.5*(b-a)))


def _canonical(series):return series.astype('string').str.strip().str.casefold()


def parse_numeric(series):
    """Strict type repair; never silently turn unknown observed tokens into NaN."""
    values=pd.to_numeric(series,errors='raise').to_numpy(dtype=float,na_value=np.nan)
    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError('Numeric parsing produced infinite observed values')
    return pd.Series(values,index=series.index,name=series.name)


def fit_plan(frame,operations,bundles=None):
    x=normalize(frame);steps=[];initial=x.index
    for proposal in operations:
        op=deepcopy(proposal);kind=op['kind'];c=op.get('column');s=x[c] if c else None
        op['output']='__e2_'+op['id']
        if kind=='fill':
            value=s.median() if op['method']=='median' else s.mean() if op['method']=='mean' else (s.mode().iloc[0] if len(s.mode()) else '__missing__')
            op['value']=float(value) if op['method']!='mode' else value
        elif kind in ['clip','flag','drop_outlier']:op['bounds']=_bounds(s)
        elif kind=='model_repair':op['bundle']=(bundles or {}).get(c) or fit_bundle(x,c)
        elif kind=='bin5':op['edges']=[-np.inf,*sorted(s.quantile([.2,.4,.6,.8]).dropna().unique().tolist()),np.inf]
        elif kind=='frequency':op['mapping']=s.value_counts(normalize=True,dropna=True).to_dict()
        elif kind in ['ordinal','onehot']:op['levels']=sorted(s.dropna().astype(str).unique().tolist())
        elif kind=='canonicalize':
            values=_canonical(s);counts=values.value_counts();common=counts[counts>=5].index.tolist();mapping={}
            for v in counts[counts<5].index:
                matches=difflib.get_close_matches(v,common,n=1,cutoff=.9)
                if matches:mapping[v]=matches[0]
            op['mapping']=mapping
        elif kind=='group_mean':
            op['mapping']=x.groupby(op['group'],dropna=True,observed=True)[c].mean().to_dict();op['default']=float(s.mean())
        elif kind=='drop_constant':op['columns']=[v for v in x if x[v].nunique(dropna=False)<=1]
        elif kind=='drop_redundant':
            numeric=x.select_dtypes(include=np.number);corr=numeric.corr().abs();op['columns']=[]
            for i,v in enumerate(corr):
                if any(corr.loc[w,v]>=op['threshold'] for w in corr.columns[:i] if w not in op['columns']):op['columns'].append(v)
        x=apply_step(x,op,training=True);steps.append(op)
        if not len(x) or not len(x.columns):raise ValueError('Operation removed all training rows or features')
        if len(x.columns)>len(frame.columns)+32:raise ValueError('More than 32 additional features')
    return steps,x,initial.isin(x.index)


def apply_step(frame,op,training=False):
    x=frame.copy();kind=op['kind'];c=op.get('column');name=op['output']
    if kind=='fill':x[c]=x[c].fillna(op['value'])
    elif kind=='model_repair':x=apply_policy(x,'model_repair',op['bundle'])[0]
    elif kind in ['flag','clip','drop_outlier']:
        lo,hi=op['bounds'];bad=(x[c]<lo)|(x[c]>hi)
        if kind=='flag':x[name]=(bad|x[c].isna()).astype(float)
        elif kind=='clip':x[c]=x[c].clip(lo,hi)
        elif training:x=x.loc[~bad]
    elif kind=='drop_missing':
        if training:x=x.loc[x[c].notna()]
    elif kind=='deduplicate':
        if training:x=x.loc[~x.duplicated()]
    elif kind=='drop':x=x.drop(columns=[c])
    elif kind in ['drop_constant','drop_redundant']:x=x.drop(columns=op['columns'])
    elif kind=='canonicalize':
        s=_canonical(x[c]);x[c]=s.replace(op['mapping']).astype(object).where(s.notna(),np.nan)
    elif kind=='numeric_parse':x[c]=parse_numeric(x[c])
    elif kind=='unit_scale':x[c]=x[c]*op['factor']
    elif kind=='ordinal':x[c]=x[c].astype('string').map({v:i for i,v in enumerate(op['levels'])}).fillna(-1.).astype(float)
    elif kind=='onehot':
        s=x[c].astype('string')
        for i,v in enumerate(op['levels']):x[name+'_'+str(i)]=(s==v).fillna(False).astype(float)
        x[name+'_missing']=s.isna().astype(float);x=x.drop(columns=[c])
    elif kind=='frequency':x[name]=x[c].map(op['mapping']).fillna(0.).astype(float)
    elif kind=='group_mean':x[name]=x[op['group']].map(op['mapping']).fillna(op['default'])
    elif kind=='date_parts':
        dates=(pd.to_datetime(x[c],unit='s',errors='coerce') if op['format']=='unix_seconds'
            else pd.to_datetime(x[c],format=op['format'],errors='coerce'))
        for part in ['year','month','dayofweek','hour']:x[name+'_'+part]=getattr(dates.dt,part).astype(float)
        x=x.drop(columns=[c])
    elif kind=='signed_log':x[name]=np.sign(x[c])*np.log1p(abs(x[c]))
    elif kind=='square':x[name]=x[c]**2
    elif kind=='affine':x[name]=op['scale']*x[c]+op['offset']
    elif kind=='bin5':x[name]=pd.cut(x[c],op['edges'],labels=False,include_lowest=True)
    elif kind=='ratio':x[name]=x[op['left']]/x[op['right']].where(x[op['right']].abs()>1e-9)
    elif kind=='difference':x[name]=x[op['left']]-x[op['right']]
    elif kind=='sum':x[name]=x[op['left']]+x[op['right']]
    elif kind=='product':x[name]=x[op['left']]*x[op['right']]
    else:raise ValueError('Unsupported operation '+kind)
    return normalize(x)


def apply_plan(frame,state):
    x=normalize(frame)
    for op in state:x=apply_step(x,op)
    if not x.index.equals(frame.index):raise ValueError('Evaluation row identities changed')
    return x


def cleaning_only(operation):
    # Paper-derived cleaning baseline cannot add features, including missing flags.
    return operation['kind'] in {'fill','model_repair','clip','drop_missing','drop_outlier',
        'canonicalize','unit_scale','numeric_parse','deduplicate','drop','drop_constant','drop_redundant'}
