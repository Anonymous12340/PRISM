import argparse, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import joblib, numpy as np, pandas as pd
import ast
for _legacy_ast_name in ('Str','Bytes','Num','NameConstant','Ellipsis'):
    if not hasattr(ast,_legacy_ast_name): setattr(ast,_legacy_ast_name,ast.Constant)

ROOT=Path('<PROJECT_ROOT>')
import sys
sys.path.insert(0,str(ROOT/'e4_current_src_20261119'))
SOURCE=ROOT/'results/e2_frozen_system_20261118/units'
OUT=ROOT/'results/e7_system_quality_cost_20261120_retry3'
DATASETS=['qsar-biodeg','cpu_act','houses']
SEEDS=[42,123,456]
METHODS=['standard','llmfe','clean_then_openfe']
GPU_IDS=[0,1,5,6,7]

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)); q.replace(p)
def unit(code,seed): return OUT/'units'/code/str(seed)
def srcunit(code,seed): return SOURCE/f'{code}__{seed}'

def base_config():
    c=read(ROOT/'configs/e2_unified_main_20261107.json')
    c.update(output_dir=str(OUT), model_path='/root/fshare/models/Qwen/Qwen3-4B',
        adapter_path='<PROJECT_ROOT>/results/task_c_sft_v2_20261104/training/adapter',
        adapter_dependency_dir=str(ROOT/'tmp/tauprism_learning_deps_20261102'),
        inference_device='cuda:0', inference_dtype='bfloat16', inference_max_new_tokens=1024,
        inference_threads=2, lm_budget=5, budgets=[5], unit_physical_limit=5,
        candidate_timeout=90, native_timeout=900, strict_replay_existing_calls=False,
        code_prompt_adapter='fit_apply_v4_compact', code_output_contract='optional_statistics_v1',
        cleaning_prompt_adapter='operator_table_v1', fail_on_method_errors=False,
        fail_on_final_fit_errors=False, datasets=[{'name':x} for x in DATASETS],
        seeds=SEEDS, methods=METHODS)
    return c

def prepare_unit(code,seed):
    su=srcunit(code,seed); u=unit(code,seed)
    if (u/'prepared.json').exists(): return u
    u.mkdir(parents=True,exist_ok=True)
    for name in ['source.joblib','inner_splits.json','menu.json','partition.json','derived_ops.json','normalization_receipt.json']:
        p=su/name
        if p.exists(): shutil.copy2(p,u/name)
    write(u/'prepared.json',{'dataset':code,'seed':seed,'source':str(su),'methods':METHODS,'sealed_outer_not_loaded':True})
    return u

def one(code,seed,gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    from taskdq.e2 import FormalScorer
    from taskdq.e2_pilot import Engine,run_method
    from taskdq.scoring import fit_predictor,predict
    c=base_config(); c['seed']=seed; u=prepare_unit(code,seed); scorer=FormalScorer(c,u); records=[]
    for method in METHODS:
        target=u/'methods'/method; done=target/'cost_complete.json'
        if done.exists():
            old=read(done)
            if old.get('status')=='passed': records.append(old); continue
            shutil.copy2(done,done.with_name(f'cost_complete.failed.{int(time.time())}.json'))
        before_fits={str(p) for p in (u/'inner_models').rglob('fit.json')}
        before_native={str(p) for p in u.rglob('generation.json')}
        before_calls={str(p) for p in (u/'physical_calls').glob('*.json')}
        started=time.monotonic()
        try:
            result=run_method(c,scorer,Engine(c),method); selection=read(target/'selection.json'); spec=selection['spec']
            outer=joblib.load(srcunit(code,seed)/'sealed_outer_X.joblib'); folder=u/'final_models'/method
            members=spec['members'] if spec.get('kind')=='ensemble' else [spec]
            tfit=time.monotonic(); member_predictions=[]
            for index,member in enumerate(members):
                member_folder=folder/f'member_{index}'
                train,valid,y=scorer.transform(member,scorer.x,outer['x'],scorer.y,scorer.source['groups'],'final',member_folder)
                model=fit_predictor(train,y,scorer.task,c['model']); current=predict(model,valid)
                joblib.dump(model,member_folder/'model.joblib'); np.save(member_folder/'predictions.npy',current)
                member_predictions.append(current)
            pred=np.mean(member_predictions,axis=0); fit_seconds=time.monotonic()-tfit
            np.save(folder/'predictions.npy',pred); calls=[p for p in (u/'physical_calls').glob('*.json') if read(p).get('method')==method]
            row={'dataset':code,'seed':seed,'method':method,'gpu':gpu,'status':'passed',
                'selection_seconds':float(result.get('seconds',0)),'logical_lm_calls':int(result.get('logical_calls',0)),
                'physical_lm_calls':len(calls),'input_tokens':sum((read(p).get('input_tokens') or 0) for p in calls),
                'output_tokens':sum((read(p).get('output_tokens') or 0) for p in calls),
                'candidate_validations':int(result.get('valid_proposals',0)),
                'inner_predictor_fit_receipts':len({str(p) for p in (u/'inner_models').rglob('fit.json')}-before_fits),
                'native_generation_jobs':len({str(p) for p in u.rglob('generation.json')}-before_native),
                'final_predictor_fits':len(members),'final_fit_seconds':float(fit_seconds),
                'wall_seconds':float(time.monotonic()-started)}
        except BaseException:
            row={'dataset':code,'seed':seed,'method':method,'gpu':gpu,'status':'failed',
                 'error':traceback.format_exc(),'wall_seconds':float(time.monotonic()-started)}
        write(done,row); records.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
    return records

def controller():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(d,s,GPU_IDS[i%len(GPU_IDS)]) for i,(d,s) in enumerate((d,s) for d in DATASETS for s in SEEDS)]
    exits=[]
    for start in range(0,len(jobs),len(GPU_IDS)):
        batch=jobs[start:start+len(GPU_IDS)]
        procs=[]
        for d,s,g in batch:
            env=dict(os.environ); env['PYTHONPATH']=str(ROOT/'e4_current_src_20261119')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH',''); procs.append(subprocess.Popen([sys.executable,__file__,'--one',d,str(s),str(g)],cwd=ROOT,env=env,start_new_session=True))
        exits.extend(p.wait() for p in procs)
    rows=[read(p) for p in OUT.glob('units/*/*/methods/*/cost_complete.json')]
    pd.DataFrame(rows).to_csv(OUT/'baseline_cost_by_seed.csv',index=False)
    write(OUT/'baseline_controller_complete.json',{'jobs':len(jobs),'exits':exits,'rows':len(rows),'time':time.time()})
    print(json.dumps(read(OUT/'baseline_controller_complete.json')),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--controller',action='store_true'); a.add_argument('--one',nargs=3); x=a.parse_args()
    if x.controller: controller()
    elif x.one: one(x.one[0],int(x.one[1]),int(x.one[2]))
