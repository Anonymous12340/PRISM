"""Versioned, deliberately small configuration contract."""
from copy import deepcopy
from .io import read, resolved

MODEL = dict(n_estimators=200,max_depth=6,learning_rate=.1,subsample=1.,colsample_bytree=1.,
             tree_method='hist',n_jobs=2,random_state=0)
IMPUTER = dict(learning_rate=.1,max_iter=100,max_leaf_nodes=15,min_samples_leaf=20,
               l2_regularization=1.,early_stopping=False,random_state=0)


def load_config(path, output=None):
    path=resolved('.',path);c=deepcopy(read(path))
    if c.get('schema_version')!=1: raise ValueError('schema_version must be 1')
    c.setdefault('mode','research');c.setdefault('seeds',[42,123,456]);c.setdefault('folds',5)
    c.setdefault('inner_folds',3);c.setdefault('rounds',2);c.setdefault('test_size',.2);c.setdefault('split_seed',42)
    if c['mode'] not in ('demo','research'): raise ValueError('mode must be demo or research')
    if c['mode']=='research' and (len(c['seeds'])!=3 or len(set(c['seeds']))!=3 or c['folds']!=5):
        raise ValueError('Research runs require three data seeds and five folds')
    if c['rounds']!=2 or c['folds']<2 or c['inner_folds']<2 or not 0<c['test_size']<1:
        raise ValueError('v1 fixes two planning rounds; folds and test fraction must be valid')
    c['model']={**MODEL,**c.get('model',{})};c['imputer']={**IMPUTER,**c.get('imputer',{})}
    c.setdefault('conditions',[{'name':'native','type':None,'rate':0.}])
    for e in c['conditions']:
        if e['type'] not in (None,'missing','outlier') or not 0<=e.get('rate',0)<=1:
            raise ValueError('v1 supports numeric missing/outlier conditions')
    for key in ['datasets','variants','conditions']:
        names=[item['name'] for item in c[key]]
        if not names or len(names)!=len(set(names)) or any('/' in n or '\\' in n for n in names):
            raise ValueError(f'{key} need unique nonempty identifiers without path separators')
    for d in c['datasets']:
        d['csv']=str(resolved(path.parent,d['csv']));targets={t['target'] for t in d['tasks']}
        if targets & set(d['features']) or not set(d['repair_columns'])<=set(d['features']):
            raise ValueError('Exclude every target from features; repair columns must be features')
        if set(d['features']) & {d.get('row_id'),d.get('group')}:
            raise ValueError('Row and group identity cannot be predictive features')
        if not d['repair_columns'] or len(d['features'])!=len(set(d['features'])):
            raise ValueError('Declare nonempty repair_columns and distinct features')
        for t in d['tasks']:
            if t['type'] not in ('regression','classification'): raise ValueError('Unknown task type')
        task_names=[t['name'] for t in d['tasks']]
        if len(task_names)!=len(set(task_names)) or any('/' in n or '\\' in n for n in task_names):
            raise ValueError('Task names must be distinct path identifiers')
    for v in c['variants']:
        v.setdefault('include_task',True);p=v['planner'];p.setdefault('backend','fixed')
        if p['backend'] not in ('fixed','replay','local'): raise ValueError('Unknown planner backend')
        for key in ['model_path','adapter_path']:
            if p.get(key) and (key=='adapter_path' or p[key].startswith(('.','/','~')) or (path.parent/p[key]).exists()):
                p[key]=str(resolved(path.parent,p[key]))
    c['output_dir']=str(resolved('.' if output else path.parent,output or c['output_dir']))
    if c.get('prepared_dir'):c['prepared_dir']=str(resolved(path.parent,c['prepared_dir']))
    return c
