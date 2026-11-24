"""Bounded E2 pilot operations and the train-fit/apply contract for baseline code."""
import ast
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from .operations import fit_bundle,apply_policy
from .config import IMPUTER

IDENTITY='def modify_features(df_input, state=None, fit=False):\n    return df_input.copy(), {} if state is None else state\n'
CONTRACT=('Return JSON {"code": "..."} containing one complete standalone function '
    'def modify_features(df_input, state=None, fit=False). Return (df_output, state). '
    'np and pd are available. Start df_output = df_input.copy(). Preserve original row index and order. '
    'The target is not an input. Learn medians, quantiles, group/category/frequency maps and bin edges '
    'ONLY inside an explicit if fit: block. fit=False must only apply stored state, never recompute batch statistics. '
    'Use row-wise arithmetic outside that block, so it also runs on application rows. '
    'Store only scalar statistics or category/column-indexed aggregate maps in state, including aggregate Series. '
    'Do not store training rows, row-indexed Series or per-row values. No imports or file/network access. '
    'Keep all original columns unless a redundant feature is explicitly dropped. At most 32 new columns. '
    'Do not call example-version functions; return one self-contained function. '
    'Categorical inputs are object strings. Do not one-hot encode them; the common scorer does that. '
    'Handle missing values and zero denominators. Keep code short, under the 1024-token output limit.')


def compile_code(code):
    tree=ast.parse(code)
    functions=[]
    for node in tree.body:
        if isinstance(node,ast.Import) and all({'numpy':'np','pandas':'pd'}.get(a.name)==a.asname for a in node.names):continue
        functions.append(node)
    if len(functions)!=1 or not isinstance(functions[0],ast.FunctionDef):raise ValueError('Expected one standalone function')
    fn=functions[0]
    if fn.name!='modify_features' or [a.arg for a in fn.args.args]!=['df_input','state','fit']:
        raise ValueError('Expected modify_features(df_input,state=None,fit=False)')
    class FitScope(ast.NodeVisitor):
        fitting=False
        def visit_If(self,node):
            self.visit(node.test);old=self.fitting
            self.fitting=old or isinstance(node.test,ast.Name) and node.test.id=='fit'
            for child in node.body:self.visit(child)
            self.fitting=old
            for child in node.orelse:self.visit(child)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and not self.fitting:
                name=node.func.attr
                if name in {'groupby','value_counts','qcut','quantile','rank','rolling','fit','fit_transform','factorize'}:
                    raise ValueError('Learn statistics only in if fit: '+name)
                if name in {'mean','median','std','var','sum','min','max','mode','nunique'}:
                    rowwise=any(k.arg=='axis' and isinstance(k.value,ast.Constant) and k.value.value in (1,'columns') for k in node.keywords)
                    if not rowwise:raise ValueError('Batch reduction outside if fit: '+name)
                if name=='cut':
                    explicit=(len(node.args)>1 and not isinstance(node.args[1],ast.Constant)) or any(k.arg=='bins' and not isinstance(k.value,ast.Constant) for k in node.keywords)
                    if not explicit:raise ValueError('pd.cut needs frozen bin edges')
            self.generic_visit(node)
    # Binding the already-provided np/pd imports changes no generated computation.
    FitScope().visit(fn);ns={'np':np,'pd':pd};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<local_baseline>','exec'),ns)
    return ns['modify_features']


def normalize(x):
    x=x.copy()
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]):x[c]=pd.to_numeric(x[c]).astype(float).replace([np.inf,-np.inf],np.nan)
        else:x[c]=x[c].astype(object).where(x[c].notna(),np.nan)
    return x


def _state_only(value,columns):
    if isinstance(value,pd.DataFrame):raise ValueError('State may not store training rows')
    if isinstance(value,pd.Series):
        aggregate_index=value.index.name in columns or value.index.isin(columns).all()
        if not value.index.is_unique or not aggregate_index:raise ValueError('State may not store row-indexed vectors')
    if isinstance(value,dict):
        for v in value.values():_state_only(v,columns)
    elif isinstance(value,(list,tuple)):
        for v in value:_state_only(v,columns)


def apply_code(frame,code,state):
    result,new_state=compile_code(code)(normalize(frame),deepcopy(state),False)
    if not isinstance(result,pd.DataFrame) or not result.index.equals(frame.index) or not result.columns.is_unique:
        raise ValueError('Generated transform changed row identities or has duplicate columns')
    if not result.shape[1] or len(set(result)-set(frame))>32:raise ValueError('Empty table or feature count exceeded')
    _state_only(new_state,frame.columns)
    return normalize(result)


def fit_code(frame,code):
    fn=compile_code(code);_,state=fn(normalize(frame),None,True);_state_only(state,frame.columns)
    return state,apply_code(frame,code,state)


def menu(frame,repair_column):
    numeric=[c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
    categorical=[c for c in frame if c not in numeric]
    ops=[]
    def add(kind,**kw):ops.append({'id':f'O{len(ops):03}','kind':kind,**kw})
    for policy in ['keep','flag_only','median_clip','median_repair','median_repair_with_flag','model_repair','model_repair_with_flag']:
        add('repair',column=repair_column,policy=policy)
    for c in numeric:
        for kind in ['signed_log','square','bin5']:add(kind,column=c)
    for i,a in enumerate(numeric):
        for b in numeric[i+1:]:
            for kind in ['ratio','difference']:add(kind,left=a,right=b)
    for c in categorical:add('frequency',column=c)
    for c in frame:add('drop',column=c)
    return ops[:100]


def fit_plan(frame,operations,bundle=None):
    a=normalize(frame);steps=[]
    for item in operations:
        op=deepcopy(item);kind=op['kind'];name='__e2_'+op['id']
        if kind=='repair':
            b=bundle or fit_bundle(a,op['column'],IMPUTER)
            a,_,_=apply_policy(a,op['policy'],b);op['bundle']=b
        elif kind=='drop':
            if len(a.columns)<=1:raise ValueError('Cannot drop last feature')
            a=a.drop(columns=[op['column']])
        else:
            op['output']=name
            if name in a:raise ValueError('Duplicate generated feature')
            if kind=='bin5':
                edges=a[op['column']].quantile([.2,.4,.6,.8]).dropna().unique()
                op['edges']=[-np.inf,*sorted(edges.tolist()),np.inf]
            elif kind=='frequency':op['mapping']=a[op['column']].value_counts(dropna=True,normalize=True).to_dict()
            a=apply_step(a,op)
        steps.append(op)
    return steps,a


def apply_step(frame,op):
    a=frame.copy();kind=op['kind']
    if kind=='repair':return apply_policy(a,op['policy'],op['bundle'])[0]
    if kind=='drop':return a.drop(columns=[op['column']])
    name=op['output']
    if kind=='signed_log':
        v=a[op['column']];a[name]=np.sign(v)*np.log1p(abs(v))
    elif kind=='square':a[name]=a[op['column']]**2
    elif kind=='bin5':a[name]=pd.cut(a[op['column']],bins=op['edges'],labels=False,include_lowest=True)
    elif kind=='frequency':a[name]=a[op['column']].map(op['mapping']).fillna(0.).astype(float)
    elif kind=='ratio':a[name]=a[op['left']]/a[op['right']].where(a[op['right']].abs()>1e-9)
    elif kind=='difference':a[name]=a[op['left']]-a[op['right']]
    else:raise ValueError('Unsupported operator '+kind)
    return normalize(a)


def apply_plan(frame,steps):
    result=normalize(frame)
    for step in steps:result=apply_step(result,step)
    return result


def fit_classical_clean(frame):
    x=normalize(frame);numeric=[c for c in x if pd.api.types.is_numeric_dtype(x[c])]
    values=x[numeric].to_numpy(float);median=np.nanmedian(values,axis=0);median=np.where(np.isfinite(median),median,0.)
    filled=np.where(np.isfinite(values),values,median)
    forest=IsolationForest(n_estimators=100,contamination=.05,random_state=0,n_jobs=2)
    keep=forest.fit_predict(filled)==1
    imputer=KNNImputer(n_neighbors=5,keep_empty_features=True).fit(values[keep])
    x=x.loc[keep].copy();x[numeric]=imputer.transform(values[keep])
    return {'numeric':numeric,'forest':forest,'imputer':imputer,'training_keep':keep},x


def apply_classical_clean(frame,state):
    x=normalize(frame);x[state['numeric']]=state['imputer'].transform(x[state['numeric']].to_numpy(float));return x
