import argparse, copy, json, os, shutil, subprocess, sys, time, traceback, fcntl
from pathlib import Path
import joblib, numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'results/e5_current_full_rerun_20261119/runtime'))
NAME='e5_current_full_rerun_20261119'; OUT=ROOT/'results'/NAME; CFG=ROOT/'configs'/(NAME+'.json')
FROZEN=ROOT/'results/export_shift_schema_stability_20261118'
BASE=__import__('e2_frozen_system_20261118'); b=BASE.b; expert=BASE.expert
BASE.NAME=NAME; BASE.OUT=OUT; BASE.CFG=CFG; BASE.FROZEN=FROZEN
BASE.DATASETS=('seoul_bike','garment_productivity','telco_churn','retinopathy')

SRC={
 'seoul_bike':lambda s:ROOT/f'results/e5_extension_search_20261110/units/seoul_bike/{s}/search/source.joblib',
 'garment_productivity':lambda s:ROOT/f'results/e5_extension_search_round9_20261111/units/garment_productivity/{s}/search/source.joblib',
 'retinopathy':lambda s:ROOT/f'results/e5_extension_search_20261110/units/retinopathy/{s}/search/source.joblib',
 'telco_churn':lambda s:ROOT/f'results/e3_classification_telco_20261108/units/telco_churn/{s}/duplicate_05/source.joblib'}
OUTER={
 'seoul_bike':lambda s:ROOT/'data/e5_extension_search_20261110/prepared/seoul_bike/final_test.joblib',
 'garment_productivity':lambda s:ROOT/'data/e5_extension_search_round9_20261111/prepared/garment_productivity/final_test.joblib',
 'retinopathy':lambda s:ROOT/'data/e5_extension_search_20261110/prepared/retinopathy/final_test.joblib',
 'telco_churn':lambda s:ROOT/f'data/e3_classification_telco_20261108/evaluation_reference/telco_churn/{s}/sealed_test.joblib'}
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str));q.replace(p)
def sha(p):return __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()
def cases():return read(OUT/'cases.json')
def configure():
 BASE.NAME=NAME;BASE.OUT=OUT;BASE.CFG=CFG;BASE.FROZEN=FROZEN;BASE.DATASETS=('seoul_bike','garment_productivity','telco_churn','retinopathy');BASE.cases=cases
 b.NAME=NAME;b.OUT=OUT;b.CFG=CFG;b.__file__=__file__;b.EXPERT=dict(kind='neural_expert',ids=['__frozen_task_expert__']);b.BLEND=dict(kind='fixed_blend',ids=['__frozen_task_expert__','__fixed_half_blend__']);expert.CFG=CFG;BASE.configure()
def prepare():
 assert not (OUT/'prepared.json').exists();OUT.mkdir(parents=True,exist_ok=True);ver=read(FROZEN/'CONFIRMED_VERSION.json');assert ver['formal_version_fixed']
 shutil.copytree(FROZEN/'runtime',OUT/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
 cfg=read(ROOT/'results/e2_frozen_system_20261118/run_config.json');expert_cfg=read(ROOT/'configs/e2_frozen_system_20261118.json');cfg.update({k:expert_cfg[k] for k in ('id','LM_max_new_tokens','max_primitives','neural_estimators','checkpoints','gate','allowed_gpus','tabicl','limits') if k in expert_cfg});cfg.update(output_dir=str(OUT),project_root=str(ROOT),seed=42,datasets=[],field_metadata={});write(CFG,cfg);configure();rt,_=b.bind();items=[]
 for code in BASE.DATASETS:
  for seed in (42,123,456):
   sp=SRC[code](seed);op=OUTER[code](seed);src=copy.deepcopy(joblib.load(sp));outer=joblib.load(op);tag=f'{code}__{seed}';src['pair']=dict(src.get('pair',{}),name=tag);u=rt.unit(tag);u.mkdir(parents=True,exist_ok=True)
   joblib.dump(src,u/'source.joblib');og=outer.get('groups',np.asarray(['__sealed_outer__'+str(i) for i in range(len(outer['x']))],dtype=object));joblib.dump(dict(x=outer['x'],groups=og),u/'sealed_outer_X.joblib');np.save(u/'sealed_outer_y.npy',np.asarray(outer['y']))
   _,support=b.folds(src,4,seed+11)[0];support=np.sort(support);rem=np.setdiff1d(np.arange(len(src['x'])),support);sub=copy.copy(src)
   for k in ('x','y','groups'):sub[k]=src[k].iloc[rem]
   fs=[]
   for a,z in b.folds(sub,3,seed+17):
    a=np.sort(np.r_[support,rem[a]]);z=np.sort(rem[z]);assert not set(support)&set(z);assert not set(src['groups'].iloc[a])&set(src['groups'].iloc[z]);fs.append(dict(train=a.tolist(),validation=z.tolist()))
   write(u/'inner_splits.json',fs);write(u/'partition.json',dict(support=support.tolist(),remaining=rem.tolist(),support_never_validation=True,current_system_inner_adapter=True));from taskdq.e2_parameterized import registry;write(u/'menu.json',registry(src['x'],src.get('date_columns',{})));write(u/'derived_ops.json',{});write(u/'normalization_receipt.json',dict(operation='identity',public_export_contracts=[]))
   items.append(dict(tag=tag,code=code,seed=seed,task=src['task']['type'],train_rows=len(src['x']),test_rows=len(outer['x']),source_path=str(sp.relative_to(ROOT)),outer_path=str(op.relative_to(ROOT)),source_sha256=sha(sp),outer_sha256=sha(op)));cfg['datasets'].append(dict(name=code,type=src['task']['type'],target=src['y'].name,exclude=[]))
 write(CFG,cfg);write(OUT/'run_config.json',cfg);write(OUT/'cases.json',items);write(OUT/'source_manifest.json',dict(entries=items,source_reads=0,test_y_reads=0));write(OUT/'current_system_manifest.json',dict(release='export_shift_schema_stability_20261118',confirmed_sha256=sha(FROZEN/'CONFIRMED_VERSION.json'),formal_version_fixed=True,implementation=ver['implementation']));ph={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'runtime').rglob('*.py')};ph[str(CFG.relative_to(ROOT))]=sha(CFG);ph[str(Path(__file__).relative_to(ROOT))]=sha(__file__);write(OUT/'protected_hashes.json',ph);write(OUT/'implementation_frozen.json',ph);b.verify();write(OUT/'prepared.json',dict(units=12,datasets=4,new_fits=0,new_lm_calls=0,test_reads=0,time=time.time()))
def preflight():
 configure();b.bind();rec=[]
 for r in cases():
  u=OUT/'units'/r['tag'];s=joblib.load(u/'source.joblib');f=read(u/'inner_splits.json')[0];assert s['y'].name not in s['x'].columns;assert not set(s['groups'].iloc[f['train']])&set(s['groups'].iloc[f['validation']]);rec.append(r['tag'])
 write(OUT/'schema_preflight.json',dict(passed=True,units=rec,new_fits=0,new_lm_calls=0,test_reads=0));write(OUT/'preflight_complete.json',dict(passed=True,units=12,new_fits=0,new_lm_calls=0,test_reads=0))
def phase(stage,gpus,deadline):
 ps=[]
 for i,g in enumerate(gpus):
  log=(OUT/f'{stage}_{i}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');p=subprocess.Popen([sys.executable,str(Path(__file__)),stage,'--index',str(i),'--workers',str(len(gpus))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ps.append((g,p,log))
 while any(p.poll() is None for _,p,_ in ps):assert time.time()<deadline;time.sleep(5)
 ex=[dict(stage=stage,gpu=g,pid=p.pid,exit=p.wait()) for g,p,l in ps];write(OUT/f'{stage}_exits.json',ex);[l.close() for _,_,l in ps];assert all(x['exit']==0 for x in ex),ex
def report():
 import pandas as pd
 from taskdq.e1 import metrics
 b.verify();rows=[]
 for r in cases():
  u=OUT/'units'/r['tag'];y=np.load(u/'sealed_outer_y.npy');p=np.load(u/'final_prediction.npy');rows.append(dict(dataset=r['code'],seed=r['seed'],task=r['task'],method='current_Full',status='passed',**metrics(y,p,{'type':r['task']})));assert sha(u/'final_prediction.npy')==read(u/'frozen_endpoint.json')['prediction_sha256']
 d=pd.DataFrame(rows);d.to_csv(OUT/'deliverables.csv',index=False,float_format='%.16g');out=[]
 for task in ('classification','regression'):
  z=d[d.task==task];m='auc' if task=='classification' else 'rmse';out.append(dict(task=task,metric=m,mean=float(z[m].mean()),sample_sd=float(z[m].std(ddof=1)),n=len(z)))
 pd.DataFrame(out).to_csv(OUT/'summary.csv',index=False);write(OUT/'outer_scoring_receipt.json',dict(time=time.time(),units=len(rows),test_y_reads=len(rows)));write(OUT/'report_complete.json',dict(units=12,test_scores_verified=True,old_baselines_reused=False));write(OUT/'controller_complete.json',dict(exit=0,seconds=time.time()-read(OUT/'dispatch.json')['time']))
def controller():
 with (OUT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert (OUT/'preflight_complete.json').exists() and not (OUT/'dispatch.json').exists();inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);rows=[[v.strip() for v in x.split(',')] for x in inv.splitlines()];g=[int(x[0]) for x in rows if int(x[0])==0 and int(x[1])<2000 and int(x[2])<5];assert g,inv;start=time.time();write(OUT/'dispatch.json',dict(time=start,deadline=start+21600,pid=os.getpid(),gpus=g))
  for stage in ('prefix_worker','neural_worker','final_worker'):phase(stage,g,start+21600)
  b.bind();report()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage');a.add_argument('--index',type=int,default=0);a.add_argument('--workers',type=int,default=1);x=a.parse_args();configure()
 if x.stage=='prepare':prepare()
 elif x.stage=='preflight':preflight()
 elif x.stage=='controller':controller()
 elif x.stage=='prefix_worker':BASE.prefix_worker(x.index,x.workers)
 elif x.stage=='neural_worker':b.neural_worker(x.index,x.workers)
 elif x.stage=='final_worker':BASE.final_worker(x.index,x.workers)
 elif x.stage=='report':report()
