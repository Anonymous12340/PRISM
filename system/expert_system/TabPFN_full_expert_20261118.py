"""Fixed five-query original Full vs prediction experts. No gradient training."""
import argparse,copy,gc,hashlib,importlib.util,json,os,pathlib,shutil,signal,subprocess,sys,time,traceback
from crossfit_neural_feature_20261118 import read,write,sha,key,matrix,idle,folds
ROOT=pathlib.Path(__file__).resolve().parents[1];NAME='TabPFN_full_expert_pilot_20261118';OUT=ROOT/'results'/NAME;CFG=ROOT/'configs'/(NAME+'.json')
PUBLIC=ROOT/'results/crossfit_neural_feature_pilot_20261118';PARENT=ROOT/'results/full_novelty_return_pilot_20261117';OLD=ROOT/'results/rule_backed_budget_pilot_20261117'
PYTHON=ROOT/'tmp/tauprism_baseline_llm_20261029/bin/python';NEURAL=ROOT/'tmp/neural_feature_v2_20261118/bin/python'
RAW=dict(kind='plan',ids=[]);EXPERT=dict(kind='neural_expert',ids=['__TabPFN_expert__']);BLEND=dict(kind='fixed_blend',ids=['__TabPFN_expert__','__fixed_half_blend__'])

def bind():
    import full_novelty_return_runtime_20261117 as rt
    rt.NAME=NAME;rt.OUT=OUT;rt.CFG=CFG;rt.RUNTIME=OUT/'runtime';rt.setup();rt.install_adapter()
    s=importlib.util.spec_from_file_location('_TabPFN_Full_helper',ROOT/'scripts/structured_actor_critic_pilot_20261116.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
    for n,v in [('ROOT',ROOT),('OUT',OUT),('CFG',CFG),('RUNTIME',OUT/'runtime'),('RAW',RAW),('read',read),('write',write),('sha',sha),('key',key),('unit',rt.unit),('rt',rt)]:setattr(h,n,v)
    return rt,h
def cases():return read(OUT/'cases.json')
def verify():
    for manifest in ['protected_hashes.json','implementation_frozen.json']:
        for p,h in read(OUT/manifest).items():assert sha(ROOT/p)==h,'Frozen file changed: '+p

def prepare():
    import numpy as np,pandas as pd,joblib
    import qualify_grounded_transfer_20261117 as qualifier
    assert not (OUT/'prepared.json').exists() and not (OUT/'dispatch.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    m=read(OLD/'qualification/manifest.json');entries=m['development'];assert len(entries)==4 and sum(r['task']=='classification' for r in entries)==2
    assert not (OLD/'outer_open_receipt.json').exists(),'Reserved sources already evaluated'
    previous=read(ROOT/'results/grounded_transfer_mini_20261117/qualification/manifest.json');used={r['family'] for r in previous['training']+previous['development']};assert not used&{r['family'] for r in entries}
    shutil.copytree(PARENT/'runtime',OUT/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
    base=read(PARENT/'run_config.json');assert base['model']==read(ROOT/'configs/e5_formal_20261110.json')['model'];assert base['selection_protocol']==read(CFG)['selection_protocol'];base.update(output_dir=str(OUT));write(OUT/'run_config.json',base)
    rt,h=bind();from taskdq.e2_parameterized import registry
    protected={str((OLD/'qualification/manifest.json').relative_to(ROOT)):sha(OLD/'qualification/manifest.json')};items=[]
    for row in entries:
        tag=row['code']+'__42';assert not (OLD/'units'/tag/'inner_models').exists() and not (OLD/'units'/tag/'deploy').exists(),'Reserved source already scientifically used'
        p=ROOT/row['path'];assert sha(p)==row['source_sha256'];protected[str(p.relative_to(ROOT))]=sha(p);source=joblib.load(p);task=row['task'];checked,cat,numeric=qualifier.validate(source,task)
        assert checked['x'].equals(source['x']) and checked['y'].equals(source['y']);assert cat==row['category_column'] and numeric==row['unit_columns']
        x=source['x'].copy();dirty=x.copy();ordered=sorted(range(len(x)),key=lambda i:key('column_bundle_export_v1:'+str(x.index[i])));regimes=np.full(len(x),-1,int)
        for k,i in enumerate(ordered[:int(.9*len(x))]):regimes[i]=k%4
        dirty['export_system']=['canonical' if z<0 else 'export_'+str(z) for z in regimes];contracts=[]
        for j,c in enumerate(numeric):
            mult=1000. if j==0 else 100.;mask=np.array([z>=0 and z in ([0,2,3] if j==0 else [1,2,3]) for z in regimes]);dirty[c]=pd.to_numeric(dirty[c]).astype(float);dirty.loc[mask,c]*=mult;marker=str(c)+'__export_unit';dirty[marker]=np.where(mask,'scaled_'+str(int(mult)),'canonical');contracts.append(dict(column=c,unit_column=marker,canonical='original public benchmark value',alternate='scaled_'+str(int(mult)),definition='alternate numeric value equals canonical value multiplied by '+str(mult),physical_unit_not_invented=True))
        original=x[cat].astype('string');v=original.copy()
        for i in range(4):
            mask=regimes==i;v.loc[mask]=original.loc[mask].str.upper().str.replace(' ','_',regex=False) if i%2==0 else ' '+original.loc[mask].str.lower().str.replace('_',' ',regex=False)+' '
        dirty[cat]=v.astype(object).where(v.notna(),np.nan);ss=copy.deepcopy(source);ss.update(x=dirty,seed=42);ss['pair']['name']=tag;ss['pair']['background']=source['pair'].get('background','')[:1200]+'\nPublic synthetic export definitions (identical for both methods): '+json.dumps(contracts)+'. Text category values may differ only by documented case, whitespace and underscore styles; canonical category meanings are unchanged.'
        a,b=folds(ss,5,42)[0];a=np.sort(a);b=np.sort(b);assert not set(ss['groups'].iloc[a])&set(ss['groups'].iloc[b]);folder=rt.unit(tag);folder.mkdir(parents=True)
        joblib.dump(dict(x=ss['x'].iloc[b],groups=ss['groups'].iloc[b]),folder/'sealed_outer_X.joblib');np.save(folder/'sealed_outer_y.npy',np.asarray(ss['y'].iloc[b]))
        for k in ['x','y','groups']:ss[k]=ss[k].iloc[a]
        _,support=folds(ss,4,53)[0];support=np.sort(support);rest=np.setdiff1d(np.arange(len(ss['x'])),support);sub=copy.copy(ss)
        for k in ['x','y','groups']:sub[k]=ss[k].iloc[rest]
        partitions=[]
        for a,b in folds(sub,3,59):
            a=np.sort(np.r_[support,rest[a]]);b=np.sort(rest[b]);assert not set(ss['groups'].iloc[a])&set(ss['groups'].iloc[b]);assert not set(support)&set(b);partitions.append(dict(train=a.tolist(),validation=b.tolist()))
        joblib.dump(ss,folder/'source.joblib');write(folder/'partition.json',dict(support=support.tolist(),remaining=rest.tolist(),support_never_validation=True));write(folder/'inner_splits.json',partitions);write(folder/'menu.json',registry(ss['x'],ss.get('date_columns',{})));write(folder/'derived_ops.json',{})
        for name in ['source.joblib','partition.json','inner_splits.json','menu.json','sealed_outer_X.joblib','sealed_outer_y.npy']:protected[str((folder/name).relative_to(ROOT))]=sha(folder/name)
        items.append(dict(tag=tag,code=row['code'],task=task,seed=42,role='development',source_sha256=sha(folder/'source.joblib'),outer_y_sha256=sha(folder/'sealed_outer_y.npy')))
    for f,h in read(CFG)['checkpoints'].values():assert sha(PUBLIC/f)==h;protected[str((PUBLIC/f).relative_to(ROOT))]=h
    for p in (OUT/'runtime').rglob('*.py'):protected[str(p.relative_to(ROOT))]=sha(p)
    protected[str((PARENT/'run_config.json').relative_to(ROOT))]=sha(PARENT/'run_config.json');protected[str((PUBLIC/'controller_complete.json').relative_to(ROOT))]=sha(PUBLIC/'controller_complete.json')
    adapter=pathlib.Path(read(CFG)['adapter_path'])/'adapter_model.safetensors';protected[str(adapter.relative_to(ROOT))]=sha(adapter)
    write(OUT/'cases.json',items);write(OUT/'protected_hashes.json',protected)
    files=[CFG,ROOT/'designs'/NAME/'PLAN.md',ROOT/'scripts/TabPFN_full_expert_20261118.py',ROOT/'scripts/neural_feature_worker_20261118.py',ROOT/'scripts/crossfit_neural_feature_20261118.py',ROOT/'scripts/full_novelty_return_runtime_20261117.py',ROOT/'scripts/structured_actor_critic_pilot_20261116.py']
    write(OUT/'implementation_frozen.json',{str(p.relative_to(ROOT)):sha(p) for p in files});write(OUT/'prepared.json',dict(time=time.time(),sources=4,scientific_fits=0,source_performance_selection=False,all_previous_inputs_unchanged=True))

def synthetic_check():
    import numpy as np,pandas as pd,types
    rt,h=bind();verify();from taskdq.e2_unified import calibration
    assert read(PUBLIC/'preflight_complete.json')['checkpoint_load_and_replay_ok']
    records=[]
    for task in ['classification','regression']:
        folder=OUT/'preflight'/task;folder.mkdir(parents=True);y=np.tile([0.,1.],12) if task=='classification' else np.arange(24,dtype=float);sc=types.SimpleNamespace(c=read(OUT/'run_config.json'),unit=folder,task=dict(type=task),y=pd.Series(y),folds=[dict(validation=list(range(i*8,(i+1)*8))) for i in range(3)],source=dict(groups=pd.Series(np.arange(24)),seed=42),ops={})
        sc.ops.update({v:dict(kind='neural_expert') for v in BLEND['ids']});pool={}
        for spec in [RAW,EXPERT,BLEND]:
            pred=(.15+.7*y) if task=='classification' else y+(1. if spec==RAW else .2)
            from taskdq.e1 import metrics
            vals=[]
            for i in range(3):
                p=folder/'inner_models'/key(spec)/str(i);p.mkdir(parents=True);v=pred[i*8:(i+1)*8];np.save(p/'predictions.npy',v);vals.append(metrics(y[i*8:(i+1)*8],v,dict(type=task)))
            scores={k:float(np.mean([v[k] for v in vals])) for k in vals[0]};pool[key(spec)]=dict(spec=spec,scores=scores,fold_scores=vals,utility=scores['auc'] if task=='classification' else -scores['rmse'])
        result,choice=calibration(sc,dict(system=pool));assert choice['system']['spec'] in [RAW,EXPERT,BLEND];records.append(dict(task=task,selection_executed=True))
    write(OUT/'preflight_complete.json',dict(time=time.time(),real_fits=0,neural_prefit_reused=True,synthetic_selector_checks=records))

def full_worker(index,workers):
    rt,h=bind();verify();from taskdq.e2_unified import metadata
    from taskdq.e2_transfer import generic_candidates
    from taskdq.e2_structured import compile_answer
    engine=h.infer_engine();engine.c['unit_physical_limit']=6
    for row in [r for i,r in enumerate(cases()) if i%workers==index]:
        sc=rt.scorer(row['tag']);assert not (sc.unit/'Full_complete.json').exists();pool={key(RAW):sc.inner(RAW)};errors=[]
        for cand in generic_candidates(sc.x,metadata(sc),limit=12,allow_short=True):
            try:
                spec=compile_answer(sc,dict(diagnosis='feature_opportunity',reason='Shared code candidate.',operations=cand['steps']),metadata(sc)['numeric_type_hints']);pool[key(spec)]=sc.inner(spec)
            except (AssertionError,ValueError,KeyError,TypeError) as e:errors.append(dict(candidate=cand,error_class=type(e).__name__))
        pool=h.extend_native(sc,pool);write(sc.unit/'public_pool.json',dict(pool=list(pool.values()),errors=errors));publickeys=set(pool);paid=dict(pool);owned=set(pool);history=[];count=[0];original_inner=sc.inner
        class BudgetReached(RuntimeError):pass
        def snapshot(n):
            write(sc.unit/('Full_N'+str(n)+'.json'),dict(queries=n,pool=list(paid.values()),selected=rt.choose(sc,paid),history=copy.deepcopy(history),source='original Full paid prefix'))
        def limited(spec):
            token=key(spec)
            if token in owned:return original_inner(spec)
            if count[0]>=5:raise BudgetReached()
            count[0]+=1;rt.ledger('query_ledger.json',dict(table=row['tag'],method='original_Full',slot=count[0],spec_key=token),read(CFG)['limits']['logical_queries'])
            try:r=original_inner(spec)
            except BaseException:
                if count[0]==3:snapshot(3)
                raise
            paid[token]=r;owned.add(token)
            if count[0]==3:snapshot(3)
            return r
        sc.inner=limited
        for turn in range(1,6):
            if count[0]>=5:break
            before=count[0]
            try:
                pool=h.LM_query(engine,sc,pool,history,turn,'original_Full');paid.update(pool)
                if count[0]==before:
                    count[0]+=1;rt.ledger('query_ledger.json',dict(table=row['tag'],method='original_Full',slot=count[0],invalid_or_duplicate=True),read(CFG)['limits']['logical_queries'])
                    if count[0]==3:snapshot(3)
                pool=h.extend_native(sc,pool);paid.update(pool)
            except BudgetReached:break
        assert count[0]==5;snapshot(5);assert (sc.unit/'Full_N3.json').exists();write(sc.unit/'derived_ops.json',sc.derived);write(sc.unit/'Full_complete.json',dict(time=time.time(),queries=5,public_keys=sorted(publickeys),LLM_calls=len(history)))
    verify()

def context(folder,task,train,y,valid):
    import numpy as np
    assert not folder.exists();folder.mkdir(parents=True);cfg=read(CFG);f,h=cfg['checkpoints'][task];tx,vx,cats=matrix(train,valid)
    np.savez(folder/'input.npz',train_x=tx,train_y=np.asarray(y),predict_x=vx,categorical_indices=np.asarray(cats,dtype=int));write(folder/'request.json',dict(task=task,checkpoint=str(PUBLIC/f),checkpoint_sha256=h,n_estimators=cfg['neural_estimators'][task]))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (folder/'worker.log').open('w') as log:r=subprocess.run([str(NEURAL),str(ROOT/'scripts/neural_feature_worker_20261118.py'),'--request',str(folder/'request.json')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=360)
    assert r.returncode==0 and (folder/'complete.json').exists(),'Neural context failure; no hidden retry';return np.load(folder/'predictions.npy')

def neural_worker(index,workers):
    import numpy as np
    rt,h=bind();verify();from taskdq.e1 import metrics
    for row in [r for i,r in enumerate(cases()) if i%workers==index]:
        sc=rt.scorer(row['tag']);prefix=read(sc.unit/'Full_N3.json');pool={key(v['spec']):v for v in prefix['pool']};base=prefix['selected']['spec'];assert not (sc.unit/'expert_selection.json').exists()
        values={key(EXPERT):[],key(BLEND):[]}
        for slot in [1,2,3]:rt.ledger('query_ledger.json',dict(table=row['tag'],method='Full_neural_experts',slot=slot,shared_identical_Full_prefix=True),read(CFG)['limits']['logical_queries'])
        for i,fold in enumerate(sc.folds):
            a,b=fold['train'],fold['validation'];assert not set(sc.source['groups'].iloc[a])&set(sc.source['groups'].iloc[b]);v=context(sc.unit/'neural_CV'/str(i),row['task'],sc.x.iloc[a],sc.y.iloc[a],sc.x.iloc[b]);bp=np.load(sc.unit/'inner_models'/key(base)/str(i)/'predictions.npy');assert len(v)==len(bp)==len(b)
            for spec,pred in [(EXPERT,v),(BLEND,.5*v+.5*bp)]:
                folder=sc.unit/'inner_models'/key(spec)/str(i);folder.mkdir(parents=True);np.save(folder/'predictions.npy',pred);values[key(spec)].append(metrics(sc.y.iloc[b],pred,sc.task))
        for spec,slot in [(EXPERT,4),(BLEND,5)]:
            vals=values[key(spec)];scores={k:float(np.mean([v[k] for v in vals])) for k in vals[0]};record=dict(spec=spec,scores=scores,fold_scores=vals,utility=scores['auc'] if row['task']=='classification' else -scores['rmse']);pool[key(spec)]=record;write(sc.unit/'scores'/(key(spec)+'.json'),record);rt.ledger('query_ledger.json',dict(table=row['tag'],method='Full_neural_experts',slot=slot,spec_key=key(spec)),read(CFG)['limits']['logical_queries'])
        for v in BLEND['ids']:sc.ops[v]=dict(id=v,kind='neural_expert',family='feature')
        selected=rt.choose(sc,pool);write(sc.unit/'expert_selection.json',dict(time=time.time(),selected=selected,queries=5,pool=list(pool.values()),neural_source='public pretrained TabPFN-v2, no gradient updates',prefix_base=base));write(sc.unit/'neural_complete.json',dict(time=time.time(),context_fits=3,validation_targets_given_to_worker=False))
    verify()

def final_worker(index,workers):
    import numpy as np,joblib
    rt,h=bind();verify();from taskdq.scoring import fit_predictor,predict
    for row in [r for i,r in enumerate(cases()) if i%workers==index]:
        sc=rt.scorer(row['tag']);outer=joblib.load(sc.unit/'sealed_outer_X.joblib');assert not set(outer['groups'])&set(sc.source['groups']);full=read(sc.unit/'Full_N5.json')['selected']['spec'];prefix=read(sc.unit/'Full_N3.json')['selected']['spec'];chosen=read(sc.unit/'expert_selection.json')['selected']['spec'];preds={};paths={}
        for spec in [full,prefix,chosen] if chosen['kind'] not in ['neural_expert','fixed_blend'] else [full,prefix]:
            token=key(spec)
            if token in preds:continue
            folder=sc.unit/'final_XGB'/token;assert not folder.exists();folder.mkdir(parents=True);train,valid,y=sc.transform(spec,sc.x,outer['x'],sc.y,sc.source['groups'],'final',folder)
            n=rt.ledger('primary_ledger.json',dict(table=row['tag'],phase='final',status='started',folder=str(folder.relative_to(OUT))),read(CFG)['limits']['primary_predictor_fits']);m=fit_predictor(train,y,sc.task,read(OUT/'run_config.json')['model']);pred=predict(m,valid);joblib.dump(m,folder/'model.joblib');np.save(folder/'predictions.npy',pred);joblib.dump(dict(train=train,valid=valid,y_train=np.asarray(y)),folder/'data.joblib');assert float(np.max(abs(predict(joblib.load(folder/'model.joblib'),valid)-pred)))==0;rt.ledger_update('primary_ledger.json',n,status='completed',model_sha256=sha(folder/'model.joblib'),prediction_sha256=sha(folder/'predictions.npy'));preds[token]=pred;paths[token]=str(folder.relative_to(OUT))
        vp=context(sc.unit/'neural_final',row['task'],sc.x,sc.y,outer['x']);preds[key(EXPERT)]=vp;preds[key(BLEND)]=.5*vp+.5*preds[key(prefix)]
        out=sc.unit/'final_outputs';out.mkdir();inventory=[]
        for arm,spec in [('original_Full',full),('Full_neural_experts',chosen),('direct_neural_reference',EXPERT),('fixed_blend_reference',BLEND)]:
            p=out/(arm+'.npy');np.save(p,preds[key(spec)]);inventory.append(dict(tag=row['tag'],source=row['code'],task=row['task'],seed=42,arm=arm,spec=spec,path=str(p.relative_to(OUT)),prediction_sha256=sha(p),outer_y_sha256=row['outer_y_sha256']))
        write(sc.unit/'frozen_endpoints.json',dict(time=time.time(),inventory=inventory,outer_labels_opened=False));verify()

def freeze():
    verify();files={};inventory=[]
    for row in cases():
        folder=OUT/'units'/row['tag'];inventory.extend(read(folder/'frozen_endpoints.json')['inventory'])
        for p in folder.rglob('*'):
            if p.is_file() and not p.name.endswith('.log'):files[str(p.relative_to(OUT))]=sha(p)
    assert len(inventory)==16;write(OUT/'frozen_all_endpoints.json',dict(time=time.time(),inventory=inventory,files=files,outer_metrics_opened=False))

def report():
    import numpy as np,pandas as pd,joblib
    rt,h=bind();verify();from taskdq.e1 import metrics
    from taskdq.scoring import predict
    assert not (OUT/'outer_open_receipt.json').exists();f=read(OUT/'frozen_all_endpoints.json')
    for p,v in f['files'].items():assert sha(OUT/p)==v
    write(OUT/'outer_open_receipt.json',dict(time=time.time(),frozen_sha256=sha(OUT/'frozen_all_endpoints.json'),all16_endpoints_frozen=True));ys={};rows=[]
    for r in f['inventory']:
        if r['tag'] not in ys:ys[r['tag']]=np.load(OUT/'units'/r['tag']/'sealed_outer_y.npy')
        score=metrics(ys[r['tag']],np.load(OUT/r['path']),dict(type=r['task']));rows.append(dict(source=r['source'],task=r['task'],seed=42,method=r['arm'],N=5,**score))
    d=pd.DataFrame(rows);effects=[]
    for task in ['classification','regression']:
        z=d[d.task==task];metric='auc' if task=='classification' else 'rmse';table=z.pivot(index='source',columns='method',values=metric)
        for arm in ['Full_neural_experts','direct_neural_reference','fixed_blend_reference']:
            g=100*(table[arm]-table['original_Full']) if task=='classification' else 100*(1-table[arm]/table['original_Full']);e=dict(task=task,method=arm,gain=float(g.mean()),source_gains={k:float(v) for k,v in g.items()},positive_sources=int((g>1e-12).sum()),nondecreasing_sources=int((g>=-1e-12).sum()))
            if task=='classification':
                v=z.pivot(index='source',columns='method',values='f1');e['F1_gain_pp']=float(100*(v[arm]-v['original_Full']).mean());e['source_AUC_ceiling_pp']={k:float(100*(1-v)) for k,v in table['original_Full'].items()}
            effects.append(e)
    cls=next(e for e in effects if e['task']=='classification' and e['method']=='Full_neural_experts');reg=next(e for e in effects if e['task']=='regression' and e['method']=='Full_neural_experts');c=read(CFG)['gate'];gate=cls['gain']>=c['auc_pp'] and cls['F1_gain_pp']>=c['f1_pp'] and cls['nondecreasing_sources']>=2 and reg['gain']>=c['regression_rmse_percent'] and reg['positive_sources']>=2
    xgb=read(OUT/'primary_ledger.json');replay=[];failed=0
    for r in xgb:
        p=OUT/r['folder']
        if r['status']=='failed':assert (p/'failure.json').exists();failed+=1;continue
        assert r['status']=='completed';m=joblib.load(p/'model.joblib');data=joblib.load(p/'data.joblib');err=float(np.max(abs(predict(m,data['valid'])-np.load(p/'predictions.npy'))));assert err==0;replay.append(err)
    neural=list((OUT/'units').glob('*/neural_CV/*/complete.json'))+list((OUT/'units').glob('*/neural_final/complete.json'));assert len(neural)==16 and len(read(OUT/'query_ledger.json'))==40
    summary=dict(time=time.time(),effects=effects,pilot_gate=bool(gate),formal_version_fixed=False,user_target_met=bool(cls['gain']>=5 and cls['F1_gain_pp']>=0 and reg['gain']>=5),primary_fit_attempts=len(xgb),primary_failed_attempts=failed,primary_replay_max=max(replay),neural_context_fits=len(neural),neural_replay_max=max(read(p)['replay_maxdiff'] for p in neural),LM_calls=len(read(OUT/'LM_ledger.json')),logical_queries=40,independent_sources=4,no_new_neural_gradient_training=True)
    write(OUT/'summary.json',summary);out=OUT/'deliverables';out.mkdir();d.to_csv(out/'SCORE_REPORT.csv',index=False,float_format='%.16g')
    lines=['# Fixed-budget Full and prediction experts','', 'Built with PriorLabs-TabPFN. Independent four-source pilot, seed42. AUC/F1 gains in percentage points; RMSE gains are relative reductions. All signs retained.','', '|Source|Task|Method|AUC|F1|RMSE|','|---|---|---|---:|---:|---:|']
    for r in rows:lines.append('|'+r['source']+'|'+r['task']+'|'+r['method']+'|'+ '|'.join(f'{r[m]:.8f}' if m in r else '' for m in ['auc','f1','rmse'])+'|')
    lines+=['','Pilot continuation gate: '+str(bool(gate)),'','```json',json.dumps(summary,indent=2),'```','','Same N5 and original selection rules. New arm spends three identical Full prefix attempts and two neural/blend attempts. It does not reuse Full final output for free. Public initial pool identical; compute costs for neural contexts are additional and explicit. Different prediction families are part of this intervention. No SFT/GRPO, RL or novel pretrained-NN attribution. Two sources/task and one seed cannot establish broad generalization/statistical significance. No final version fixed or E1–E7 expansion.']
    (out/'SCORE_REPORT.md').write_text('\n'.join(lines));write(OUT/'report_hashes.json',{p.name:sha(p) for p in out.iterdir()});verify()

def phase(name,gpus,deadline):
    ps=[]
    for i,gpu in enumerate(gpus):
        log=(OUT/(name+'_'+str(i)+'.log')).open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');p=subprocess.Popen([str(PYTHON),str(pathlib.Path(__file__)),name,'--index',str(i),'--workers',str(len(gpus))],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ps.append((gpu,p,log))
    registry=read(OUT/'process_registry.json') if (OUT/'process_registry.json').exists() else [];registry.extend(dict(phase=name,gpu=g,pid=p.pid) for g,p,l in ps);write(OUT/'process_registry.json',registry)
    while any(p.poll() is None for g,p,l in ps):
        if time.time()>=deadline:
            for g,p,l in ps:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            time.sleep(2)
            for g,p,l in ps:
                if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
            raise TimeoutError('Original hard science limit, no extension')
        write(OUT/'controller_status.json',dict(time=time.time(),phase=name,deadline=deadline,workers=[dict(pid=p.pid,gpu=g,exit=p.poll()) for g,p,l in ps]));time.sleep(2)
    exits=[dict(phase=name,gpu=g,pid=p.pid,exit=p.wait()) for g,p,l in ps];old=read(OUT/'worker_exits.json') if (OUT/'worker_exits.json').exists() else [];write(OUT/'worker_exits.json',old+exits)
    for g,p,l in ps:l.close()
    assert all(r['exit']==0 for r in exits),'Worker failure retained, no automatic retry'

def controller():
    import fcntl
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'dispatch.json').exists();verify();assert (OUT/'preflight_complete.json').exists();gpus=idle()[:3];assert gpus
    start=time.time();deadline=start+read(CFG)['limits']['science_seconds'];write(OUT/'dispatch.json',dict(time=start,pid=os.getpid(),deadline=deadline,gpus=gpus,science_seconds=2700,single_launch=True))
    try:
        for name in ['full_worker','neural_worker','final_worker']:phase(name,gpus,deadline)
        assert time.time()<deadline;freeze();report();assert time.time()<deadline
        write(OUT/'controller_complete.json',dict(time=time.time(),seconds=time.time()-start,exit=0,pilot_gate=read(OUT/'summary.json')['pilot_gate'],formal_version_fixed=False))
    except BaseException as e:write(OUT/'controller_failure.json',dict(time=time.time(),seconds=time.time()-start,error_class=type(e).__name__,traceback=traceback.format_exc(),no_automatic_retry=True));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','synthetic_check','full_worker','neural_worker','final_worker','controller']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.stage.endswith('_worker'):globals()[a.stage](a.index,a.workers)
    else:globals()[a.stage]()
if __name__=='__main__':main()
