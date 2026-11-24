"""Isolated Full adapter and typed policy environment for the registered AC pilot."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import os
import fcntl
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1];NAME='full_novelty_return_pilot_20261117'
OUT=ROOT/'results'/NAME;CFG=ROOT/'configs'/(NAME+'.json');RUNTIME=OUT/'runtime'
RAW=dict(kind='plan',ids=[]);NATIVE=dict(kind='native_joint',ids=['__native_openfe_component__'])

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=str));q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def key(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]
def unit(table):return OUT/'units'/table
def setup():
    importlib.invalidate_caches();sys.path.insert(0,str(RUNTIME))
    os.environ['PYTHONPATH']=str(RUNTIME)+os.pathsep+os.environ.get('PYTHONPATH','')
    for p in ['tmp/semantic_unit_deps_20261111','tmp/e6_inference_deps_20261110','tmp/tauprism_learning_deps_20261102']:
        sys.path.append(str(ROOT/p))
class BudgetExhausted(RuntimeError):pass

@contextmanager
def ledger_lock(name):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/(name+'.lock')).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def ledger(name,row,cap):
    with ledger_lock(name):
        p=OUT/name;v=read(p) if p.exists() else []
        if name=='LM_ledger.json':
            signature=lambda r:(r.get('table'),r.get('method'),r.get('number',r.get('turn')))
            used={signature(r) for r in v}
            if signature(row) not in used and len(used)>=cap:raise BudgetExhausted('Unique logical call budget exhausted: '+name)
        else:
            if len(v)>=cap:raise BudgetExhausted('Budget exhausted: '+name)
        v.append(row);write(p,v);return len(v)-1
def ledger_update(name,index,**fields):
    with ledger_lock(name):
        p=OUT/name;v=read(p);v[index].update(fields);write(p,v)
_TASKS={}
def task(table):
    import joblib
    if table not in _TASKS:_TASKS[table]=joblib.load(unit(table)/'source.joblib')['task']
    return _TASKS[table]

def progress(stage,**kw):write(OUT/'progress.json',dict(stage=stage,time=__import__('time').time(),pid=os.getpid(),**kw))

def fit(table,phase,train,y,valid,yv,folder,slot):
    import joblib
    import numpy as np
    from taskdq.scoring import fit_predictor,predict
    from taskdq.e1 import metrics
    if (folder/'result.json').exists():
        data=joblib.load(folder/'data.joblib');import pandas as pd
        pd.testing.assert_frame_equal(train,data['train']);pd.testing.assert_frame_equal(valid,data['valid'])
        np.testing.assert_array_equal(np.asarray(y),data['y_train']);np.testing.assert_array_equal(np.asarray(yv),data['y_valid'])
        return read(folder/'result.json')
    if (folder/'failure.json').exists():raise ValueError('Previously recorded predictor failure; no refit')
    assert not (folder/'model.joblib').exists(),'Uncommitted fit must not be repeated'
    folder.mkdir(parents=True,exist_ok=True)
    i=ledger('primary_ledger.json',dict(table=table,phase=phase,slot=slot,status='started',folder=str(folder.relative_to(OUT)),pid=os.getpid()),read(CFG)['limits']['primary_predictor_fits'])
    assert not set(train.index)&set(valid.index)
    try:
        model=fit_predictor(train,y,task(table),read(OUT/'run_config.json')['model'])
        pred=predict(model,valid);joblib.dump(model,folder/'model.joblib');np.save(folder/'predictions.npy',pred)
        joblib.dump(dict(train=train,valid=valid,y_train=np.asarray(y),y_valid=np.asarray(yv)),folder/'data.joblib')
        result=dict(scores=metrics(yv,pred,task(table)),folder=str(folder.relative_to(OUT)))
    except Exception as ex:
        write(folder/'failure.json',dict(exception_class=type(ex).__name__,time=__import__('time').time(),no_hidden_retry=True))
        ledger_update('primary_ledger.json',i,status='failed',exception_class=type(ex).__name__)
        raise
    ledger_update('primary_ledger.json',i,status='completed',scores=result['scores'],model_sha256=sha(folder/'model.joblib'),prediction_sha256=sha(folder/'predictions.npy'))
    write(folder/'result.json',result);return result

_PATCHED=False
def install_adapter():
    global _PATCHED
    if _PATCHED:return
    setup();import numpy as np
    from taskdq import e2_library as lib,e2,e2_unified,e2_parameterized
    orig_fit=lib.fit_plan;orig_apply=lib.apply_step;orig_bundle=e2.fit_bundle
    lib.FAMILIES['where_quantile']='feature';e2_parameterized.FEATURES.add('where_quantile')
    for k in ['conditional_unit','alias_map','date_difference']:lib.FAMILIES[k]='feature' if k=='date_difference' else 'format'
    e2_parameterized.FEATURES.add('date_difference')
    def apply(frame,op,training=False):
        if op['kind']=='conditional_unit':
            x=frame.copy();mask=x[op['unit_column']].astype('string')==str(op['unit_value']);x[op['column']]=x[op['column']].where(~mask.fillna(False),x[op['column']]*op['factor']);return lib.normalize(x)
        if op['kind']=='alias_map':
            x=frame.copy();x[op['column']]=x[op['column']].replace({p['source']:p['destination'] for p in op['aliases']});return lib.normalize(x)
        if op['kind']=='date_difference':
            import pandas as pd
            x=frame.copy();a=pd.to_datetime(x[op['left']],format=op['left_format'],errors='coerce');b=pd.to_datetime(x[op['right']],format=op['right_format'],errors='coerce');x[op['output']]=(a-b).dt.total_seconds()/86400.;return lib.normalize(x)
        if op['kind']!='where_quantile':return orig_apply(frame,op,training=training)
        x=frame.copy();mask=x[op['guard']]>op['threshold']
        x[op['output']]=x[op['column']].where(mask,0.)
        return lib.normalize(x)
    def plan(frame,operations,bundles=None):
        x=lib.normalize(frame);initial=x.index;steps=[]
        for proposal in operations:
            required={proposal[k] for k in ['column','left','right','group','guard','unit_column'] if k in proposal}
            if not required<=set(x):raise ValueError('Plan reads unavailable columns: '+','.join(sorted(required-set(x))))
            if proposal['kind'] in ['conditional_unit','alias_map','date_difference']:
                op=deepcopy(proposal);op['output']='__e2_'+op['id'];x=apply(x,op,training=True);steps.append(op)
            elif proposal['kind']=='where_quantile':
                op=deepcopy(proposal);op['output']='__e2_'+op['id']
                assert op['column'] in x and op['guard'] in x
                op['threshold']=float(x[op['guard']].quantile(op['q']))
                x=apply(x,op,training=True);steps.append(op)
            else:
                st,x,_=orig_fit(x,[proposal],bundles);steps.extend(st)
            assert len(x) and len(x.columns) and len(x.columns)<=len(frame.columns)+32
        return steps,x,initial.isin(x.index)
    def bundle(*args,**kwargs):
        ledger('auxiliary_ledger.json',dict(pid=os.getpid(),status='called'),read(CFG)['limits']['auxiliary_repair_calls'])
        return orig_bundle(*args,**kwargs)
    lib.apply_step=apply;lib.fit_plan=plan;e2_unified.fit_plan=plan;e2.fit_bundle=bundle;lib.fit_bundle=bundle
    _PATCHED=True

def scorer(table):
    install_adapter();import joblib
    import numpy as np
    from taskdq.e2_unified import UnifiedScorer
    from taskdq.e2 import FormalScorer
    from taskdq.e2_pilot import key as fullkey
    from taskdq.e2_parameterized import dependencies
    class PilotScorer(UnifiedScorer):
        def native(self,train,valid,y,groups,method,where):
            p=where/'native'/method/'generation.json'
            if not p.exists():ledger('native_ledger.json',dict(table=table,path=str(p.relative_to(OUT)),pid=os.getpid()),read(CFG)['limits']['native_generations'])
            return super().native(train,valid,y,groups,method,where)
        def transform(self,spec,train,valid,y,groups,slot,where):
            ids=[i for i in spec['ids'] if not i.startswith('__native')]
            assert len(ids)<=read(CFG)['max_primitives'] and len(ids)==len(set(ids))
            # Sequential fit_plan validates actual generated fields, including date/onehot suffixes.
            if spec['kind']=='plan':return FormalScorer.transform(self,spec,train,valid,y,groups,slot,where)
            return UnifiedScorer.transform(self,spec,train,valid,y,groups,slot,where)
        def inner(self,spec):
            token=fullkey(spec);p=self.unit/'scores'/f'{token}.json'
            if p.exists():return read(p)
            values=[]
            for i,f in enumerate(self.folds):
                folder=self.unit/'inner_models'/token/str(i);a,b=f['train'],f['validation']
                train,valid,yy=self.transform(spec,self.x.iloc[a],self.x.iloc[b],self.y.iloc[a],self.source['groups'].iloc[a],str(i),folder)
                result=fit(table,'CV',train,yy,valid,self.y.iloc[b],folder/'fit',i)
                np.save(folder/'predictions.npy',np.load(OUT/result['folder']/'predictions.npy'))
                values.append(result['scores'])
                write(folder/'fit.json',dict(training_row_ids=train.index.tolist(),validation_row_ids=valid.index.tolist()))
            scores={k:float(np.mean([v[k] for v in values])) for k in values[0]};utility=scores['auc'] if self.task['type']=='classification' else -scores['rmse'];record=dict(spec=spec,scores=scores,fold_scores=values,utility=utility)
            write(p,record);return record
    return PilotScorer(read(OUT/'run_config.json'),unit(table))

_SELECTION_CACHE={}
def choose(sc,pool,receipt=None):
    from taskdq.e2_unified import calibration
    token=(str(sc.unit),tuple(sorted(pool)),sc.c.get('selection_protocol'))
    if token not in _SELECTION_CACHE:_SELECTION_CACHE[token]=calibration(sc,{'system':pool})
    result,choices=_SELECTION_CACHE[token]
    if receipt is not None:write(receipt,dict(pool=list(pool.values()),chosen=choices['system']['spec'],calibration=result))
    return choices['system']
