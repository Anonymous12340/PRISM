"""A self-contained synthetic engineering example; no research data or weights."""
from pathlib import Path
import numpy as np
import pandas as pd
from .io import write


def create(directory):
    out=Path(directory).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Demo output must be a new or empty directory')
    out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(42);n=240
    x=rng.uniform(-1,1,n);z=rng.uniform(0,1,n);sensor=2*x+.4*z+rng.normal(0,.025,n)
    frame=pd.DataFrame({'row_id':np.arange(n),'x':x,'z':z,'sensor':sensor,
        'response':3*sensor+.5*x+rng.normal(0,.05,n),'class_label':(sensor+rng.normal(0,.1,n)>.2).astype(int)})
    (out/'data').mkdir();frame.to_csv(out/'data/synthetic.csv',index=False)
    config={'schema_version':1,'mode':'demo','output_dir':'.','seeds':[42],'folds':2,'inner_folds':2,'rounds':2,
        'test_size':.2,'split_seed':42,'model':{'n_estimators':8,'max_depth':2,'n_jobs':1},
        'imputer':{'max_iter':8,'max_leaf_nodes':7,'min_samples_leaf':5},
        'datasets':[{'name':'synthetic','csv':'data/synthetic.csv','row_id':'row_id','background':'Synthetic engineering data; no empirical research claim.',
            'features':['x','z','sensor'],'repair_columns':['sensor'],'tasks':[
                {'name':'regression','target':'response','type':'regression','description':'Synthetic response estimation'},
                {'name':'classification','target':'class_label','type':'classification','description':'Synthetic binary classification'}]}],
        'conditions':[{'name':'missing10','type':'missing','rate':.1}],
        'variants':[{'name':'fixed','include_task':True,'planner':{'backend':'fixed'}},
            {'name':'offline_replay','include_task':True,'planner':{'backend':'replay','responses':['keep','model_repair']}}]}
    write(out/'config.json',config);return out/'config.json'
