"""Isolated fixed public V2 neural worker. Never receives validation targets."""
import argparse, hashlib, json, os, pathlib, time, traceback
import numpy as np

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p, v):
    p=pathlib.Path(p); q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
    req=pathlib.Path(a.request);r=json.loads(req.read_text());out=req.parent
    assert not (out/'started.json').exists(), 'No silent repeated neural fit'
    write(out/'started.json',dict(time=time.time(),pid=os.getpid(),request_sha256=sha(req)))
    import torch,tabpfn,sklearn,pandas
    from tabpfn import TabPFNClassifier,TabPFNRegressor
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    assert sha(r['checkpoint'])==r['checkpoint_sha256'];assert torch.cuda.is_available()
    d=np.load(out/'input.npz',allow_pickle=False)
    assert set(d.files)=={'train_x','train_y','predict_x','categorical_indices'}
    cls=TabPFNClassifier if r['task']=='classification' else TabPFNRegressor
    m=cls(n_estimators=r['n_estimators'],categorical_features_indices=d['categorical_indices'].tolist(),model_path=r['checkpoint'],device='cuda:0',random_state=0,n_jobs=2,fit_mode='fit_preprocessors',memory_saving_mode=True,ignore_pretraining_limits=True)
    t=time.time();m.fit(d['train_x'],d['train_y'])
    pred=lambda: m.predict_proba(d['predict_x'])[:,1] if r['task']=='classification' else m.predict(d['predict_x'])
    v=np.asarray(pred(),float); vv=np.asarray(pred(),float);err=float(np.max(np.abs(v-vv)))
    assert v.shape==(len(d['predict_x']),) and np.isfinite(v).all()
    assert err<=1e-5, 'Neural replay mismatch'
    np.save(out/'predictions.npy',v)
    write(out/'complete.json',dict(time=time.time(),seconds=time.time()-t,fit_count=1,replay_maxdiff=err,checkpoint_sha256=sha(r['checkpoint']),input_sha256=sha(out/'input.npz'),predictions_sha256=sha(out/'predictions.npy'),versions=dict(torch=torch.__version__,tabpfn=tabpfn.__version__,sklearn=sklearn.__version__,pandas=pandas.__version__),train_rows=len(d['train_y']),prediction_rows=len(v),no_gradient_updates=True))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        try:
            import sys
            req=pathlib.Path(sys.argv[sys.argv.index('--request')+1]);write(req.parent/'failure.json',dict(time=time.time(),error_class=type(e).__name__,traceback=traceback.format_exc(),no_retry=True))
        finally:raise
