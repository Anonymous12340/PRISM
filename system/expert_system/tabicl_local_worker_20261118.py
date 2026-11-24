"""Offline official TabICLv2; receives numeric arrays and training targets only."""
import pathlib,json,time,os,sys,hashlib,traceback
import numpy as np
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,v):pathlib.Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def main():
 import pandas as pd,torch,tabicl,importlib.metadata
 from tabicl import TabICLClassifier,TabICLRegressor
 req=pathlib.Path(sys.argv[1]);r=json.loads(req.read_text());out=req.parent;assert not (out/'started.json').exists();write(out/'started.json',dict(time=time.time(),pid=os.getpid(),request_sha256=sha(req)))
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.is_available();assert sha(r['checkpoint'])==r['checkpoint_sha256']
 d=np.load(out/'input.npz',allow_pickle=False);assert set(d.files)=={'train_x','train_y','predict_x','categorical_indices'}
 def frame(a):
  x=pd.DataFrame(a.copy())
  for i in d['categorical_indices']:
   vals=a[:,i];x[int(i)]=pd.Series(['category_'+str(int(v)) if np.isfinite(v) else np.nan for v in vals],dtype=object)
  return x
 tx=frame(d['train_x']);vx=frame(d['predict_x']);cls=TabICLClassifier if r['task']=='classification' else TabICLRegressor
 m=cls(n_estimators=r['n_estimators'],model_path=r['checkpoint'],checkpoint_version=r['checkpoint_version'],allow_auto_download=False,device='cuda:0',random_state=0,n_jobs=2,use_amp=False,use_fa3=False)
 start=time.time();m.fit(tx,d['train_y']);pred=lambda: m.predict_proba(vx)[:,1] if r['task']=='classification' else m.predict(vx)
 v=np.asarray(pred(),float);repeat=np.asarray(pred(),float);error=float(np.max(np.abs(v-repeat)));assert v.shape==(len(vx),) and np.isfinite(v).all();assert error<=1e-5,'Replay mismatch'
 np.save(out/'predictions.npy',v);write(out/'complete.json',dict(time=time.time(),seconds=time.time()-start,fit_count=1,replay_maxdiff=error,input_sha256=sha(out/'input.npz'),checkpoint_sha256=sha(r['checkpoint']),predictions_sha256=sha(out/'predictions.npy'),version=importlib.metadata.version('tabicl'),train_rows=len(tx),prediction_rows=len(vx),no_gradient_updates=True))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  p=pathlib.Path(sys.argv[1]).parent;write(p/'failure.json',dict(time=time.time(),error_class=type(e).__name__,traceback=traceback.format_exc(),no_hidden_retry=True));raise
