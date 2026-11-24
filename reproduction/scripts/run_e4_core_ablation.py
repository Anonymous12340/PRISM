"""Two ordinary A100 workers for exactly six E4 data units; no agents."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SOURCE=Path(__file__).absolute().parent.parent
sys.path.insert(0,str(SOURCE))
from taskdq.e4_data import settings,runtime,matrix,unit_dir,LM_VARIANTS
from taskdq.e4 import E4Engine,run_unit
from taskdq.io import read,write


def worker(path,gpu,slot):
    s=settings(path);root=Path(s['output_dir']);c=runtime(s);engine=E4Engine(c)
    tasks=[(dataset_index,seed) for dataset_index in range(len(s['datasets'])) for seed in s['seeds']]
    for dataset_index,seed in tasks[slot::2]:
        name=s['datasets'][dataset_index]['name']
        if (root/'execution_hold.json').exists():return
        try:run_unit(path,name,seed,engine)
        except Exception:
            write(root/f'worker_{slot}_failure.json',{'gpu':gpu,'slot':slot,'dataset':name,'seed':seed,'error':traceback.format_exc(),'time':time.time()})
            write(root/'execution_hold.json',{'gpu':gpu,'slot':slot,'dataset':name,'seed':seed,'reason':'Engineering failure; preserve in-flight work','time':time.time()})
            raise


def controller(path):
    s=settings(path);root=Path(s['output_dir']);lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'preflight.json')['passed']
    assert read(root/'preparation_complete.json')['sealed_before_planning']
    env={**os.environ,'PYTHONPATH':str(SOURCE),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    started=time.time();workers={};logs={}
    gpus=[0,1]
    for slot,gpu in enumerate(gpus):
        rows=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=name,memory.used','--format=csv,noheader,nounits'],text=True)
        assert int(rows.strip().split(',')[-1])<20000, 'Authorized GPU not idle before start'
        logs[gpu]=(root/f'gpu{gpu}.log').open('a')
        workers[slot]=subprocess.Popen([sys.executable,str(Path(__file__).absolute()),'--config',str(Path(path).absolute()),'--worker',str(gpu),'--slot',str(slot)],
            cwd=SOURCE,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=logs[gpu],stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'dispatch.json',{'pid':os.getpid(),'started':started,'server':'rtx4090','gpus':gpus,
          'worker_pids':{str(g):v.pid for g,v in workers.items()},'source':str(SOURCE),'python':sys.executable,
          'data_units':12,'variant_choices':84,'maximum_scheduled_lm_calls':180})
    while True:
        complete=[e for e in matrix(s) if (unit_dir(s,*e)/'complete.json').exists()]
        sentinels=[]
        if False and all(e in complete for e in sentinels) and not (root/'sentinels_passed.json').exists():
            write(root/'sentinels_passed.json',{'passed':True,'units':sentinels,'test_opened':False,
                  'gate':'completed execution and exact Full-request replay; no outcome score threshold'})
        physical=0;logical=0;trajectories=0;choices=0
        for entry in matrix(s):
            unit=unit_dir(s,*entry)
            physical+=sum(len(list((unit/'arms'/v/'physical_calls').glob('*.json'))) for v in LM_VARIANTS)
            logical+=sum(len(list((unit/'arms'/v/'methods'/v/'calls').glob('*.json'))) for v in LM_VARIANTS)
            trajectories+=sum((unit/'arms'/v/'selection.json').exists() for v in LM_VARIANTS)
            choices+=sum((unit/'arms'/v/'final.json').exists() for v in s['variants'])
        alive=sum(p.poll() is None for p in workers.values())
        write(root/'progress.json',{'completed_data_units':len(complete),'total_data_units':12,'total_variant_choices':84,
              'physical_calls':physical,'logical_calls':logical,'active_workers':alive,'gpus':gpus,
              'completed_lm_trajectories':trajectories,'total_lm_trajectories':60,'frozen_variant_choices':choices,
              'stage':'engineering_hold' if (root/'execution_hold.json').exists() else 'running','time':time.time(),'test_opened':False})
        assert physical<=320 and logical<=320
        if not alive:break
        time.sleep(10)
    codes={str(g):p.wait() for g,p in workers.items()}
    for log in logs.values():log.close()
    write(root/'worker_exit.json',codes)
    if any(codes.values()) or len(complete)!=6 or (root/'execution_hold.json').exists():return
    write(root/'all_units_complete.json',{'units':12,'choices':84,'test_opened':False,'time':time.time()})
    with (root/'final_evaluation.log').open('a') as log:
        code=subprocess.call([sys.executable,'-m','taskdq.e4','--config',str(Path(path).absolute()),'--stage','test'],
              cwd=SOURCE,env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT)
    write(root/'controller_exit.json',{'returncode':code,'stage':'numeric_complete' if code==0 else 'evaluation_failure',
          'time':time.time(),'new_stage_started':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--worker',type=int);p.add_argument('--slot',type=int);a=p.parse_args()
    if a.worker is None:controller(a.config)
    else:worker(a.config,a.worker,a.slot)
