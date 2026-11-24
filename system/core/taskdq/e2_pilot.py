"""Four-task E2 engineering check. Development rows only; hard limit 100 LM calls."""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .io import read,write,append,plain
from .e1 import fit_preprocessor,labels,metrics
from .scoring import splits,fit_predictor,predict,fit_encoder,encode
from .operations import fit_bundle
from .config import IMPUTER
from .profile import profile_table
from . import e2_operations as ops

METHODS=['raw','standard','openfe','autofeat','caafe','llmfe','classical_clean',
    'lm_cleaning_simplified','clean_then_openfe','one_shot','without_task','random_order',
    'fixed_probe','no_lm_search','ours']
RAW={'kind':'plan','ids':[]}
CODE_SCHEMA={'type':'object','properties':{'code':{'type':'string','minLength':1}},'required':['code'],'additionalProperties':False}


def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]


def load_config(path):
    path=Path(path).resolve();c=read(path);root=(path.parent/c['project_root']).resolve();c['project_root']=str(root)
    for name in ['output_dir','prepared_dir','baseline_config','classical_python','upstream']:
        c[name]=str(root/c[name])
    if c['lm_budget']!=5 or len(c['tasks'])!=4 or c['physical_call_limit']!=100:raise ValueError('This entry is only the four-task N5 pilot')
    return c


def prepare(c):
    out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        if read(out/'config.json')!=c:raise ValueError('Existing pilot settings differ')
        return
    base=read(c['baseline_config']);write(out/'config.json',c);pairs={p['name']:p for p in base['pairs']};manifest=[]
    for item in c['tasks']:
        pair=pairs[item['dataset']];task=next(t for t in pair['tasks'] if t['name']==item['task'])
        folder=Path(c['prepared_dir'])/pair['name']/str(c['seed']);frame=pd.read_pickle(folder/'development.pkl').set_index('__row_id')
        fold=read(folder/'folds.json')[c['fold']-1];rng=np.random.default_rng(c['seed'])
        positions={}
        for part,limit in [('train',c['train_limit']),('validation',c['validation_limit'])]:
            ids=np.asarray(fold[part]);positions[part]=np.sort(rng.choice(ids,limit,replace=False)) if len(ids)>limit else ids
        train=frame.iloc[positions['train']];valid=frame.iloc[positions['validation']]
        if set(train.index)&set(valid.index):raise ValueError('Outer rows overlap')
        group=pair.get('group_column')
        if group and set(train[group])&set(valid[group]):raise ValueError('Outer groups overlap')
        x=ops.normalize(train[pair['features']]);y=labels(train,task)
        if {t['target'] for t in pair['tasks']}&set(x):raise ValueError('Target column in predictors')
        unit=out/'units'/pair['name'];unit.mkdir(parents=True,exist_ok=True)
        source={'x':x,'y':y,'groups':train[group] if group else None,'task':task,'pair':pair,
            'repair_column':item['repair_column'],'seed':c['seed'],'row_ids':x.index.tolist()}
        joblib.dump(source,unit/'source.joblib');joblib.dump({'x':ops.normalize(valid[pair['features']]),'y':labels(valid,task)},unit/'outer_validation.joblib')
        candidates=ops.menu(x,item['repair_column']);write(unit/'menu.json',candidates)
        folds=splits(x,y,task,c['inner_folds'],c['seed'],source['groups'])
        write(unit/'inner_splits.json',[{'train':a.tolist(),'validation':b.tolist(),
            'train_row_ids':x.index[a].tolist(),'validation_row_ids':x.index[b].tolist()} for a,b in folds])
        manifest.append({'dataset':pair['name'],'task':task['name'],'type':task['type'],'train_rows':len(x),'validation_rows':len(valid),
            'feature_count':len(x.columns),'candidate_operations':len(candidates),'group_column':group,
            'train_ids':train.index.tolist(),'validation_ids':valid.index.tolist(),'sealed_test_opened':False})
    write(out/'manifest.json',manifest)
    write(out/'environment.json',{'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','xgboost-cpu','torch','transformers']},
        'method_version':'E2 pilot adapter v1; original taskdq-v1.0.0 unchanged','training_updates':0})


class Engine:
    def __init__(self,c):self.c=c;self.base=None
    def request(self,messages,schema):
        request={'messages':messages,'schema':schema,'model_path':self.c['model_path'],
            'generation':{'temperature':0,'top_p':1.,'top_k':0,'max_new_tokens':self.c.get('inference_max_new_tokens',1024),'thinking':False,'dtype':self.c.get('inference_dtype','bfloat16')}}
        if 'inference_device' in self.c:
            request['execution']={'device':self.c['inference_device'],'threads':self.c.get('inference_threads',2)}
        if self.c.get('adapter_path'):
            request['planner_weights']={'adapter_path':self.c['adapter_path'],
                'variant':self.c['weight_variant'],'revision':self.c['weight_revision']}
        if self.c.get('inference_sampling'):
            request['generation'].update(self.c['inference_sampling'])
        return request

    def generate(self,messages,schema,unit,method,round_number):
        request=self.request(messages,schema)
        token=key(request);cache=unit/'lm_cache'/(token+'.json');target=unit/'methods'/method/'calls'/f'{round_number}.json'
        if self.c.get('strict_replay_existing_calls') and target.exists():
            old=read(target)
            if old.get('request') is None or key(old['request'])!=token:
                raise ValueError('Existing round has a different request; refuse resampling or relabelling its output')
        if cache.exists():
            record=read(cache);record['physical_call']=False;write(target,record);return record
        previous=[read(p) for p in (unit/'physical_calls').glob('*.json')]
        if any(r['method']==method and r['round']==round_number and r['status']=='started' for r in previous):
            raise ValueError('Prior request has unresolved started status; inspect interruption, never silently resample')
        interrupted=[r for r in previous if r['method']==method and r['round']==round_number and r['status']=='interrupted']
        if interrupted:
            # No output exists to reuse. Discard this planned request and consume
            # the original round even if a seeded buffer would now draw another prompt.
            record={'request':None,'original_request_key':interrupted[0]['request_key'],
                'discarded_planned_request_key':token,'format_valid':False,'physical_call':False,
                'error':'Original generation interrupted; request budget consumed, no retry',
                'interrupted':True,'raw_output':None,'input_tokens':None,'output_tokens':None}
            write(target,record);return record
        physical=len(list((unit/'physical_calls').glob('*.json')))
        if physical>=self.c.get('unit_physical_limit',25):raise ValueError('Per-unit physical budget exhausted')
        if self.base is None:
            from .planner import Planner
            if self.c.get('adapter_path') and self.c.get('adapter_dependency_dir'):
                # Append optional existing PEFT dependencies without shadowing
                # the frozen torch/transformers/scoring environment.
                if self.c['adapter_dependency_dir'] not in sys.path:sys.path.append(self.c['adapter_dependency_dir'])
            planner={'backend':'local','model_path':self.c['model_path'],'dtype':self.c.get('inference_dtype','bfloat16'),
                'device':self.c.get('inference_device','cuda:0'),'threads':self.c.get('inference_threads',2)}
            if self.c.get('adapter_path'):planner['adapter_path']=self.c['adapter_path']
            self.base=Planner(planner)
            self.base.load()
            if self.c.get('adapter_path'):
                from peft import get_peft_model_state_dict
                from peft.utils.save_and_load import load_peft_weights
                saved=load_peft_weights(self.c['adapter_path'],device='cpu')
                actual=get_peft_model_state_dict(self.base.model)
                assert set(saved)==set(actual) and saved
                for name,value in actual.items():
                    assert self.base.torch.equal(value.detach().cpu(),saved[name].to(dtype=value.dtype)),name
                self.base.weight_receipt={'variant':self.c['weight_variant'],'adapter_path':self.c['adapter_path'],
                    'loaded_adapter_tensors':len(actual),'file_to_loaded_tensor_check':True,
                    'nonzero_tensors':sum(bool(v.count_nonzero()) for v in actual.values()),
                    'active_adapters':list(self.base.model.active_adapters),'base_model_path':self.c['model_path']}
        if self.c.get('adapter_path') and not (unit/'planner_load.json').exists():
            write(unit/'planner_load.json',self.base.weight_receipt)
        if self.c.get('adapter_path') and 'inference_device' in self.c:
            write(unit/('planner_load_'+method+'.json'),{**self.base.weight_receipt,
                'execution_device':str(self.base.device),'loaded_dtype':str(self.base.dtype)})
        b=self.base;t=b.torch
        prompt=b.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=b.tokenizer(prompt,return_tensors='pt').to(b.device)
        if self.c.get('enforce_context_limit'):
            capacity=b.model.config.max_position_embeddings
            if ids.input_ids.shape[1]+self.c.get('inference_max_new_tokens',1024)>capacity:
                write(unit/'methods'/method/'context_failure.json',{'input_tokens':int(ids.input_ids.shape[1]),
                    'reserved_output_tokens':self.c.get('inference_max_new_tokens',1024),'model_max_positions':capacity,
                    'physical_generation_started':False})
                raise ValueError('Input plus reserved output exceeds model context capacity')
        sampling=self.c.get('inference_sampling')
        if sampling:t.manual_seed(sampling['sampling_seed'])
        actual=b.GenerationConfig(do_sample=bool(sampling),temperature=sampling['temperature'] if sampling else 1.,
            top_p=sampling.get('top_p',1.) if sampling else 1.,top_k=sampling.get('top_k',0) if sampling else 0,
            max_new_tokens=self.c.get('inference_max_new_tokens',1024),
            pad_token_id=b.tokenizer.eos_token_id,eos_token_id=b.model.generation_config.eos_token_id,
            bos_token_id=b.model.generation_config.bos_token_id,use_cache=True)
        started=time.monotonic();physical_path=unit/'physical_calls'/f'{physical+1:03}.json'
        write(physical_path,{'status':'started','method':method,'round':round_number,'request_key':token})
        if str(b.device).startswith('cuda'):t.cuda.reset_peak_memory_stats()
        from transformers import LogitsProcessorList
        mask=CachedSchemaMask(b.json_prefix(b.tokens,schema),t)
        with t.inference_mode():generated=b.model.generate(**ids,generation_config=actual,logits_processor=LogitsProcessorList([mask]))
        comp=generated[0,ids.input_ids.shape[1]:].tolist();raw=b.tokenizer.decode(comp,skip_special_tokens=True)
        record={'request':request,'raw_output':raw,'prompt_token_ids':ids.input_ids[0].tolist(),'completion_token_ids':comp,
            'input_tokens':ids.input_ids.shape[1],'output_tokens':len(comp),'seconds':time.monotonic()-started,
            'peak_allocated_bytes':t.cuda.max_memory_allocated() if str(b.device).startswith('cuda') else 0,
            'execution_device':str(b.device),'effective_generation':actual.to_dict(),'physical_call':True,
            'eos_completed':bool(comp and comp[-1] in ([actual.eos_token_id] if isinstance(actual.eos_token_id,int) else actual.eos_token_id))}
        try:record.update(answer=json.loads(raw),format_valid=True)
        except json.JSONDecodeError as exc:record.update(format_valid=False,error=str(exc))
        write(physical_path,{'status':'completed','method':method,'round':round_number,'request_key':token,
            'input_tokens':record['input_tokens'],'output_tokens':record['output_tokens'],'seconds':record['seconds']})
        write(cache,record);write(target,record);return record


class CachedSchemaMask:
    """Same allowed token sets; cache device masks to avoid repeated Python-list GPU indexing."""
    def __init__(self,allowed,torch):self.allowed=allowed;self.torch=torch;self.cache={}
    def __call__(self,input_ids,scores):
        if len(input_ids)!=1:raise ValueError('Pilot decoder supports batch size one')
        tokens=self.allowed(0,input_ids[0]);signature=tuple(tokens)
        if signature not in self.cache:
            if len(self.cache)>=32:self.cache.pop(next(iter(self.cache)))
            mask=self.torch.zeros(scores.shape[-1],dtype=self.torch.bool,device=scores.device)
            mask[self.torch.tensor(tokens,dtype=self.torch.long,device=scores.device)]=True
            self.cache[signature]=mask
        return scores.masked_fill(~self.cache[signature].unsqueeze(0),-self.torch.inf)


def run_native_command(command,env,log,timeout):
    """A native search and its joblib children share a disposable process group."""
    proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:return proc.wait(timeout=timeout)
    except BaseException:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        # The root may exit before a joblib grandchild that ignores SIGTERM.
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        raise


def timeout_handler(signum,frame):raise TimeoutError('Pilot candidate execution exceeded frozen timeout')


def native_time_limit(c,method):
    """Per-method compute allowance; it does not alter native search parameters."""
    allowance=c.get('native_timeouts',{}).get(method,c['native_timeout'])
    if c.get('output_dir'):
        revision=Path(c['output_dir'])/'native_runtime_revision.json'
        if revision.exists():
            allowance=read(revision).get('native_timeouts',{}).get(method,allowance)
    return allowance


class Scorer:
    def __init__(self,c,unit):
        self.c=c;self.unit=unit;self.source=joblib.load(unit/'source.joblib');self.x=self.source['x'];self.y=self.source['y']
        self.task=self.source['task'];self.menu=read(unit/'menu.json');self.ops={v['id']:v for v in self.menu}
        self.folds=read(unit/'inner_splits.json');self.auxiliary_fits=0;self.outer_prepared={}

    def bundle(self,train,slot):
        path=self.unit/'repair_bundles'/(slot+'.joblib')
        if not path.exists():
            result=fit_bundle(train,self.source['repair_column'],IMPUTER);path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(result,path);self.auxiliary_fits+=int(result['model'] is not None)
            write(path.with_suffix('.json'),{'fitted_imputers':int(result['model'] is not None),'training_row_ids':train.index.tolist()})
        return joblib.load(path)

    def native(self,train,valid,y,groups,method,where):
        folder=where/'native';folder.mkdir(parents=True,exist_ok=True);meta={'task':self.task,
            'categorical':[c for c in train if not pd.api.types.is_numeric_dtype(train[c])]}
        if not (folder/'input.json').exists():
            train.to_csv(folder/'train.csv',index_label='__row_id');valid.to_csv(folder/'valid.csv',index_label='__row_id')
            np.save(folder/'labels.npy',np.asarray(y));write(folder/'input.json',meta)
            if groups is not None:np.save(folder/'groups.npy',np.asarray(groups,dtype=object))
        artifact=folder/method/'generation.json'
        if not artifact.exists():
            env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
                'TASKDQ_AUTOFEAT_JOBS':str(self.c.get('native_jobs',{}).get('autofeat',2))}
            started=time.monotonic()
            with (folder/'worker.log').open('w') as log:
                try:
                    code=run_native_command([self.c['classical_python'],'-m','taskdq.e2_classical','--folder',str(folder),'--method',method],
                        env,log,native_time_limit(self.c,method))
                except subprocess.TimeoutExpired:
                    write(artifact,{'status':'timeout','method':method,'seconds':time.monotonic()-started,
                        'time_limit':native_time_limit(self.c,method),'children_stopped':True,'error':'Native search exceeded the frozen wall-time limit; no automatic retry'})
                    raise
            if code:raise ValueError('Native '+method+' failed: '+str(folder/'worker.log'))
        result=read(artifact)
        if result['status']!='passed':raise ValueError('Native feature generation failed: '+result.get('error',''))
        a=pd.read_csv(folder/method/'new_train.csv',float_precision='round_trip').set_index('__row_id')
        b=pd.read_csv(folder/method/'new_valid.csv',float_precision='round_trip').set_index('__row_id')
        if a.index.tolist()!=train.index.tolist() or b.index.tolist()!=valid.index.tolist():raise ValueError('Native generated row order differs')
        for col in result['categorical']:a[col]=a[col].astype('string').astype(object);b[col]=b[col].astype('string').astype(object)
        return pd.concat([train,a],axis=1),pd.concat([valid,b],axis=1),{'artifact':str(artifact),'state_replay_passed':True}

    def transform(self,spec,train,valid,y,groups,slot,where):
        kind=spec['kind'];keep=np.ones(len(train),dtype=bool)
        if kind=='plan':
            operations=[self.ops[i] for i in spec['ids']]
            bundle=self.bundle(train,slot) if any(o['kind']=='repair' for o in operations) else None
            state,a=ops.fit_plan(train,operations,bundle);b=ops.apply_plan(valid,state)
        elif kind=='standard':
            state,matrix,mask,log=fit_preprocessor(train,'standard',self.c['preprocessing']);keep=mask.to_numpy()
            a=pd.DataFrame(matrix,index=train.index[keep]);b=pd.DataFrame(state.transform(valid),index=valid.index)
            a.columns=[f'e{i}' for i in range(a.shape[1])];b.columns=list(a)
        elif kind in ['classical_clean','clean_then_openfe']:
            state,a=ops.fit_classical_clean(train);keep=np.asarray(state['training_keep']);b=ops.apply_classical_clean(valid,state)
            if kind=='clean_then_openfe':
                a,b,native=self.native(a,b,np.asarray(y)[keep],np.asarray(groups)[keep] if groups is not None else None,'openfe',where)
                state={'cleaning':state,'native':native}
        elif kind in ['openfe','autofeat']:a,b,state=self.native(train,valid,y,groups,kind,where)
        elif kind=='code':
            state=[];a,b=train.copy(),valid.copy()
            for code in spec['codes']:
                learned,a=ops.fit_code(a,code);b=ops.apply_code(b,code,learned);state.append(learned)
        else:raise ValueError('Unknown specification '+kind)
        if len(b)!=len(valid) or not b.index.equals(valid.index):raise ValueError('Validation rows changed')
        if any(c not in a for c in b):raise ValueError('Application invented unseen feature columns')
        if kind!='standard' and len(a.columns)-len(train.columns)>32:raise ValueError('More than 32 generated columns')
        where.mkdir(parents=True,exist_ok=True);joblib.dump(state,where/'transform_state.joblib')
        # Frozen transformation replay, including evaluation in different batches.
        if kind=='plan':
            loaded=joblib.load(where/'transform_state.joblib');pd.testing.assert_frame_equal(b,ops.apply_plan(valid,loaded))
            if len(b)>1:pd.testing.assert_frame_equal(b.iloc[:2],ops.apply_plan(valid.iloc[:2],loaded))
        elif kind=='code':
            loaded=joblib.load(where/'transform_state.joblib');replayed=valid.copy()
            for code,s in zip(spec['codes'],loaded):replayed=ops.apply_code(replayed,code,s)
            pd.testing.assert_frame_equal(b,replayed)
            if len(b)>1:
                replayed=valid.iloc[:2].copy()
                for code,s in zip(spec['codes'],loaded):replayed=ops.apply_code(replayed,code,s)
                pd.testing.assert_frame_equal(b.iloc[:2],replayed)
        write(where/'transform_check.json',{'passed':True,'training_rows_before':len(train),'training_rows_after':len(a),
            'training_row_ids':a.index.tolist(),'validation_row_ids':b.index.tolist(),'features':list(a),'state_saved':True,
            'batch_invariance_checked':kind in ['plan','code'],'test_opened':False})
        return a,b,np.asarray(y)[keep]

    def inner(self,spec):
        token=key(spec);path=self.unit/'scores'/(token+'.json')
        if path.exists():return read(path)
        started=time.monotonic();values=[]
        if spec['kind']=='ensemble':
            for member in spec['members']:self.inner(member)
            for i,fold in enumerate(self.folds):
                predictions=[np.load(self.unit/'inner_models'/key(member)/str(i)/'predictions.npy') for member in spec['members']]
                prediction=np.mean(predictions,axis=0);values.append(metrics(self.y.iloc[fold['validation']],prediction,self.task))
        else:
            for i,fold in enumerate(self.folds):
                folder=self.unit/'inner_models'/token/str(i);model_file=folder/'model.joblib'
                a,b=fold['train'],fold['validation'];groups=self.source['groups'].iloc[a] if self.source['groups'] is not None else None
                if not model_file.exists():
                    train,valid,y=self.transform(spec,self.x.iloc[a],self.x.iloc[b],self.y.iloc[a],groups,str(i),folder)
                    start=time.monotonic();model=fit_predictor(train,y,self.task,self.c['model']);joblib.dump(model,model_file)
                    prediction=predict(model,valid);np.save(folder/'predictions.npy',prediction)
                    write(folder/'fit.json',{'training_row_ids':train.index.tolist(),'validation_row_ids':valid.index.tolist(),
                        'seconds':time.monotonic()-start,'predictor_fits':1,'features':len(train.columns)})
                prediction=np.load(folder/'predictions.npy');values.append(metrics(self.y.iloc[b],prediction,self.task))
        average={k:float(np.mean([v[k] for v in values])) for k in values[0]}
        value=average['auc'] if self.task['type']=='classification' else -average['rmse']
        result={'spec':spec,'scores':average,'utility':value,'fold_scores':values,'seconds':time.monotonic()-started}
        write(path,result);return result

    def outer_member(self,spec,validation):
        token=key(spec);folder=self.unit/'outer_models'/token;prediction_file=folder/'prediction.npy'
        if prediction_file.exists():return np.load(prediction_file),np.load(folder/'linear_prediction.npy')
        a,b,y=self.transform(spec,self.x,validation['x'],self.y,self.source['groups'],'outer',folder)
        start=time.monotonic();model=fit_predictor(a,y,self.task,self.c['model']);joblib.dump(model,folder/'model.joblib')
        prediction=predict(model,b);np.testing.assert_array_equal(prediction,predict(joblib.load(folder/'model.joblib'),b))
        encoder=fit_encoder(a);x,xv=encode(a,encoder),encode(b,encoder)
        estimator=LogisticRegression(C=1.,max_iter=2000,random_state=0) if self.task['type']=='classification' else LinearRegression()
        linear=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),estimator)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');linear.fit(x,y)
        linear_pred=linear.predict_proba(xv)[:,1] if self.task['type']=='classification' else linear.predict(xv)
        joblib.dump({'encoder':encoder,'model':linear},folder/'linear.joblib')
        np.save(prediction_file,prediction);np.save(folder/'linear_prediction.npy',linear_pred)
        write(folder/'fit.json',{'predictor_fits':2,'seconds':time.monotonic()-start,'features':len(a.columns),
            'warnings':[str(w.message) for w in caught],'saved_model_replay':True})
        return prediction,linear_pred


def evidence(scorer,include_task=True):
    if hasattr(scorer,'formal_evidence'):return scorer.formal_evidence(include_task)
    p=profile_table(scorer.x);p.pop('sample_targets',None)
    cols=[]
    for item in p['column_profiles']:
        row={k:item[k] for k in ['name','kind','missing_count','unique_count']}
        if 'quantiles' in item:row['quantiles']=item['quantiles'];row['iqr_outliers']=item['iqr_outliers']
        else:row['top_values']=item['top_values'][:3]
        cols.append(row)
    result={'background':scorer.source['pair']['background'],'rows':len(scorer.x),'columns':cols,
        'samples':p['sample_rows'][:3],'task_type':scorer.task['type']}
    if include_task:result['task']=scorer.task
    return result


def limited_schema(scorer,clean_only=False):
    if scorer.c.get('operator_version'):
        from .e2_library import cleaning_only
        ids=[o['id'] for o in scorer.menu if not clean_only or cleaning_only(o)]
    else:ids=[o['id'] for o in scorer.menu if not clean_only or o['kind']=='repair']
    return {'type':'object','properties':{'reason':{'type':'string','maxLength':160},
        'operations':{'type':'array','items':{'type':'string','enum':ids},'maxItems':4,'uniqueItems':True}},
        'required':['reason','operations'],'additionalProperties':False}


def upstream_functions(path,names):
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns


def register_buffer(c,scorer):
    upstream=Path(c['upstream'])/'LLMFE';sys.path.insert(0,str(upstream))
    from llmfe import buffer,config,code_manipulation
    if c.get('operator_version'):
        native=getattr(buffer,'_taskdq_native_is_categorical',buffer.is_categorical)
        buffer._taskdq_native_is_categorical=native
        categorical={col for col in scorer.x if not pd.api.types.is_numeric_dtype(scorer.x[col])}
        def training_type(series):
            # A random ten-row example may contain only nulls from a genuine
            # categorical predictor. Prompt display must respect its train dtype.
            return True if series.name in categorical else native(series)
        buffer.is_categorical=training_type
    template=code_manipulation.text_to_program('"""[PREFIX] Use task semantics and feature names. [FEATURES] [EXAMPLES] [SUFFIX]"""\n'+ops.IDENTITY)
    db=buffer.ExperienceBuffer(config.ExperienceBufferConfig(),template,'modify_features',{})
    def register(code,utility,island):
        fn=code_manipulation.text_to_function(code);fn.data_input=scorer.x.reset_index(drop=True);fn.data_output=scorer.y.to_numpy()
        db.register_program(fn,island,{'inner_utility':utility},fn.data_input,fn.data_output)
    register(ops.IDENTITY,scorer.inner(RAW)['utility'],None)
    return db,register


def run_method(c,scorer,engine,method):
    folder=scorer.unit/'methods'/method;path=folder/'selection.json'
    if path.exists():return read(path)
    stop=Path(c['output_dir'])/'code_baselines_engineering_stop.json'
    if method in ['caafe','llmfe'] and stop.exists():
        raise ValueError('Code-baseline engineering gate stopped further generation; see '+str(stop))
    raw=scorer.inner(RAW);best=raw;logical=0;valid=0;proposals=[];started=time.monotonic();db=None;probe_rejection=None
    probe=getattr(scorer,'probe',{'kind':'plan','ids':['O006']})
    if method=='raw':pass
    elif method in ['standard','openfe','autofeat','classical_clean','clean_then_openfe']:
        best=scorer.inner({'kind':method})
    elif method in ['one_shot','random_order']:
        ours=read(scorer.unit/'methods/ours/selection.json')
        if method=='one_shot':best=ours['first_choice'];logical=1;valid=int(ours['proposals'][0].get('executable',False))
        else:
            spec=deepcopy(ours['spec']);random.Random(c['seed']).shuffle(spec['ids']);best=scorer.inner(spec)
    else:
        if method in ['ours','without_task','fixed_probe','no_lm_search']:
            try:
                checked=scorer.inner(probe)
            except ValueError as exc:
                if method!='fixed_probe' or not c.get('fixed_probe_invalid_fallback') or not str(exc).startswith('Candidate creates consistency violations:'):
                    raise
                probe_rejection={'spec':probe,'executable':False,'error':str(exc),'retained_spec':RAW,
                                 'reason':'Predetermined control proposal violates an existing consistency rule; no alternative sampled.'}
                write(folder/'probe_rejection.json',probe_rejection)
            else:
                if checked['utility']>best['utility']+1e-12:best=checked
        first_choice=deepcopy(best)
        if method=='fixed_probe':pass
        elif method=='no_lm_search':
            rng=random.Random(c['seed']);choices=[]
            # Same complete menu as the LM, not a repair-only weak comparator.
            feature_ids=[o['id'] for o in scorer.menu if o['kind'] not in ['repair','drop']]
            choices.append(['O002']);choices.append([feature_ids[0]])
            choices.append(['O006',feature_ids[len(feature_ids)//2]])
            for _ in range(2):choices.append(rng.sample([o['id'] for o in scorer.menu],2))
            if c.get('operator_version'):
                features=[o['id'] for o in scorer.menu if o['family']=='feature']
                fills=[o['id'] for o in scorer.menu if o['kind']=='fill']
                choices=[fills[:1],features[:1],list(dict.fromkeys(probe['ids']+features[:1]))]
                choices += [rng.sample([o['id'] for o in scorer.menu],min(2,len(scorer.menu))) for _ in range(c['lm_budget']-3)]
            for ids in choices:
                try:
                    candidate=scorer.inner({'kind':'plan','ids':ids})
                    proposals.append({'spec':candidate['spec'],'utility':candidate['utility'],'executable':True})
                    if candidate['utility']>best['utility']+1e-12:best=candidate
                except Exception as exc:
                    proposals.append({'spec':{'kind':'plan','ids':ids},'executable':False,'error':repr(exc)})
                save_checkpoint(c,scorer,method,len(proposals),best,proposals,0)
        else:
            if method=='llmfe':
                np.random.seed(c['seed']);random.seed(c['seed']);db,register=register_buffer(c,scorer)
            feedback={'raw_scores':raw['scores'],'best_scores':best['scores'],'best_spec':best['spec'],
                'fixed_probe':scorer.inner(probe) if method in ['ours','without_task'] else None}
            for iteration in range(c['lm_budget']):
                if method in ['ours','without_task','lm_cleaning_simplified']:
                    cleaning=method=='lm_cleaning_simplified';schema=limited_schema(scorer,cleaning)
                    if c.get('operator_version'):
                        from .e2_library import cleaning_only
                        visible=[dict(o) for o in scorer.menu if not cleaning or cleaning_only(o)]
                    else:visible=[{k:v for k,v in o.items()} for o in scorer.menu if not cleaning or o['kind']=='repair']
                    messages=[{'role':'system','content':('Select a task-relevant data processing plan from the fixed library. '
                        'The code, not you, edits cells. Propose up to four operation IDs in execution order, or []. '
                        'Use only the training facts and previous inner-CV feedback. No hidden scores. Return JSON reason/operations. '
                        +('Cleaning-only simplified agent: choose repair actions only.' if cleaning else 'Cleaning and feature generation are available.'))},
                        {'role':'user','content':json.dumps(plain({'input':evidence(scorer,method!='without_task'),
                            'operation_library':visible,'previous_training_feedback':feedback}),ensure_ascii=False)}]
                    if cleaning and c.get('operator_version'):
                        messages[0]['content']=('Cleaning-only local adaptation of Bendinelli et al. (2025). '
                            'Identify inconsistent or contextually implausible values, repair existing data and optionally drop training rows/columns. '
                            'You cannot add features, add rows or modify the fixed predictor. Choose up to four IDs from this cleaning-only library. '
                            'Use training facts and prior inner-CV feedback; the code performs edits, preserves all evaluation rows, fits statistics on training only. '
                            'Return JSON reason/operations; [] is allowed. Do not invent an error when the provided facts do not support it.')
                elif method=='caafe':
                    funcs=upstream_functions(Path(c['upstream'])/'CAAFE/caafe/caafe.py',{'get_prompt','build_prompt_from_df'})
                    displayed=scorer.x
                    if best['spec']['kind']=='code':
                        for code in best['spec']['codes']:_,displayed=ops.fit_code(displayed,code)
                    descriptor=[None,None,None,None,list(displayed.columns)+[scorer.task['target']],scorer.source['pair']['background']]
                    prompt=funcs['build_prompt_from_df'](descriptor,displayed,iterative=5)
                    if scorer.task['type']=='regression':prompt=prompt.replace('classification','regression').replace('classifier','regressor').replace('accuracy','RMSE (lower is better)')
                    schema=CODE_SCHEMA;messages=[{'role':'system','content':ops.CONTRACT},
                        {'role':'user','content':prompt+'\nTRAINING FEEDBACK:\n'+json.dumps(plain(feedback),ensure_ascii=False)+
                            '\nReturn a complete function that adds the next feature to the CURRENT columns listed above.'}]
                    if c.get('code_prompt_adapter') in ['fit_apply_v2','fit_apply_v3','fit_apply_v4_compact']:
                        from .e2_code_prompt import messages as code_messages
                        original=funcs['build_prompt_from_df'](descriptor,displayed,iterative=5)
                        messages=code_messages(method,original,scorer.task,plain(feedback),compact=c.get('code_prompt_adapter')=='fit_apply_v4_compact',
                            optional_statistics=c.get('code_output_contract')=='optional_statistics_v1')
                else:
                    previous_cwd=os.getcwd()
                    try:
                        os.chdir(Path(c['upstream'])/'LLMFE');selected=db.get_prompt()
                    finally:os.chdir(previous_cwd)
                    context=selected.code.rsplit('\ndef modify_features_v',1)[0]
                    schema=CODE_SCHEMA;messages=[{'role':'system','content':ops.CONTRACT},
                        {'role':'user','content':json.dumps(plain(evidence(scorer)),ensure_ascii=False)+'\nOFFICIAL EVOLUTION BUFFER:\n'+context}]
                    if c.get('code_prompt_adapter') in ['fit_apply_v2','fit_apply_v3','fit_apply_v4_compact']:
                        from .e2_code_prompt import messages as code_messages
                        original=json.dumps(plain(evidence(scorer)),ensure_ascii=False)+'\nOFFICIAL EVOLUTION BUFFER:\n'+selected.code
                        messages=code_messages(method,original,scorer.task,plain(feedback),compact=c.get('code_prompt_adapter')=='fit_apply_v4_compact',
                            optional_statistics=c.get('code_output_contract')=='optional_statistics_v1')
                if method=='lm_cleaning_simplified' and c.get('cleaning_prompt_adapter')=='operator_table_v1':
                    from .e2_cleaning_prompt import compact_messages
                    messages=compact_messages(messages,visible,list(scorer.x))
                record=engine.generate(messages,schema,scorer.unit,method,iteration+1);logical+=1
                row={'round':iteration+1,'format_valid':record['format_valid'],'physical_call':record['physical_call']}
                try:
                    if not record['format_valid']:raise ValueError('Invalid JSON output')
                    answer=record['answer']
                    if method in ['ours','without_task','lm_cleaning_simplified']:
                        ids=answer['operations']
                        if len(ids)>4 or len(set(ids))!=len(ids) or any(i not in {v['id'] for v in visible} for i in ids):raise ValueError('Invalid bounded plan')
                        spec={'kind':'plan','ids':ids}
                    else:
                        ops.compile_code(answer['code'])
                        prior=best['spec']['codes'] if method=='caafe' and best['spec']['kind']=='code' else []
                        spec={'kind':'code','codes':prior+[answer['code']]}
                    old=signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(c['candidate_timeout'])
                    try:result=scorer.inner(spec)
                    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
                    valid+=1;accepted=result['utility']>best['utility']+1e-12
                    if accepted:best=result
                    if method=='llmfe':register(answer['code'],result['utility'],int(selected.island_id))
                    row.update(executable=True,spec=spec,scores=result['scores'],accepted=accepted)
                    feedback={'previous_scores':result['scores'],'accepted':accepted,'best_scores':best['scores'],'best_spec':best['spec']}
                except Exception as exc:
                    row.update(executable=False,error=repr(exc));feedback={'error':repr(exc),'best_scores':best['scores'],'best_spec':best['spec']}
                proposals.append(row);write(folder/'proposals.json',proposals)
                if iteration==0:first_choice=deepcopy(best)
                save_checkpoint(c,scorer,method,iteration+1,best,proposals,logical,db)
                print('LM_PROPOSAL',scorer.source['pair']['name'],method,iteration+1,row.get('executable'),flush=True)
            if method=='llmfe':
                codes=[str(fn) for fn in db._best_program_per_island]
                members=[RAW if code.strip()==ops.IDENTITY.strip() else {'kind':'code','codes':[code]} for code in codes]
                best=scorer.inner({'kind':'ensemble','members':members})
                write(folder/'native_island_selection.json',{'islands':len(members),'selection':'official per-island winners; mean predictions','programs':codes})
        write(folder/'first_choice.json',first_choice)
    result={'method':method,'spec':best['spec'],'inner_scores':best['scores'],'logical_calls':logical,
        'valid_proposals':valid,'proposals':proposals,'selected_before_outer_labels':True,'seconds':time.monotonic()-started,
        'first_choice':read(folder/'first_choice.json') if (folder/'first_choice.json').exists() else best,
        'status':'fallback' if (logical and valid==0) or probe_rejection else 'passed'}
    if probe_rejection:result['probe_rejection']=probe_rejection
    write(path,result);return result


def save_checkpoint(c,scorer,method,iteration,best,proposals,logical,db=None):
    if iteration not in c.get('budgets',[]):return
    chosen=best
    if db is not None:
        codes=[str(fn) for fn in db._best_program_per_island]
        members=[RAW if code.strip()==ops.IDENTITY.strip() else {'kind':'code','codes':[code]} for code in codes]
        chosen=scorer.inner({'kind':'ensemble','members':members})
    valid=sum(bool(p.get('executable',method=='no_lm_search')) for p in proposals)
    write(scorer.unit/'methods'/method/'checkpoints'/f'{iteration:03}.json',
        {'method':method,'budget':iteration,'spec':chosen['spec'],'inner_scores':chosen['scores'],
            'logical_calls':logical,'valid_proposals':valid,'proposals':deepcopy(proposals),
            'status':'fallback' if logical and valid==0 else 'passed','selected_before_outer_labels':True})


def run_unit(c,name):
    unit=Path(c['output_dir'])/'units'/name;scorer=Scorer(c,unit);engine=Engine(c)
    order=[m for m in METHODS if m not in ['one_shot','random_order']]+['one_shot','random_order']
    for method in order:
        try:run_method(c,scorer,engine,method)
        except Exception:
            write(unit/'methods'/method/'failure.json',{'error':traceback.format_exc(),'stage':'before_outer_evaluation'})
            print('METHOD_BLOCKED',name,method,traceback.format_exc(),flush=True)
    # All proposals and choices for this task are frozen before outer labels enter.
    write(unit/'all_selections_frozen.json',{'methods':order,'sealed_test_opened':False,
        'failures':[m for m in order if not (unit/'methods'/m/'selection.json').exists()]})
    validation=joblib.load(unit/'outer_validation.joblib')
    for method in METHODS:
        target=unit/'methods'/method
        if (target/'result.json').exists() and read(target/'result.json')['status'] in ['passed','fallback']:continue
        if not (target/'selection.json').exists():
            write(target/'result.json',{'dataset':name,'task':scorer.task['name'],'method':method,'status':'blocked','sealed_test_opened':False});continue
        selection=read(target/'selection.json');spec=selection['spec']
        try:
            members=spec['members'] if spec['kind']=='ensemble' else [spec]
            pred=[scorer.outer_member(member,validation) for member in members]
            primary=np.mean([p[0] for p in pred],axis=0);linear=np.mean([p[1] for p in pred],axis=0)
            for model,p in [('xgboost',primary),('linear',linear)]:
                pd.DataFrame({'row_id':validation['x'].index,'y_true':validation['y'],'prediction':p}).to_csv(target/(model+'_predictions.csv'),index=False)
                loaded=pd.read_csv(target/(model+'_predictions.csv'),float_precision='round_trip')
                values=metrics(validation['y'],p,scorer.task);actual=metrics(loaded.y_true.to_numpy(dtype=validation['y'].dtype),loaded.prediction.to_numpy(dtype=p.dtype),scorer.task)
                if any(abs(values[k]-actual[k])>1e-9 for k in values):raise ValueError('Prediction replay mismatch')
            result={'dataset':name,'task':scorer.task['name'],'type':scorer.task['type'],'method':method,'status':selection['status'],
                'xgboost':metrics(validation['y'],primary,scorer.task),'linear':metrics(validation['y'],linear,scorer.task),
                'logical_calls':selection['logical_calls'],'valid_proposals':selection['valid_proposals'],
                'selected_spec':spec,'prediction_replay_passed':True,'sealed_test_opened':False}
            write(target/'result.json',result);print('OUTER_RESULT',name,method,result['xgboost'],flush=True)
        except Exception:
            write(target/'failure.json',{'error':traceback.format_exc(),'stage':'outer_transform_or_score'})
            write(target/'result.json',{'dataset':name,'task':scorer.task['name'],'method':method,'status':'blocked','sealed_test_opened':False})


def report(c):
    import csv
    out=Path(c['output_dir']);rows=[];flat=[];errors=[];calls=[];fits=[];aux=[];native=[];inner_requests=0
    for task in c['tasks']:
        name=task['dataset'];unit=out/'units'/name
        for method in METHODS:
            path=unit/'methods'/method/'result.json'
            row=read(path) if path.exists() else {'dataset':name,'method':method,'status':'pending'}
            rows.append(row)
            if row['status'] in ('passed','fallback'):
                for model in ['xgboost','linear']:
                    flat.append({k:row[k] for k in ['dataset','task','type','method','status','logical_calls','valid_proposals']}|
                        {'downstream':model,**row[model]})
            failure=unit/'methods'/method/'failure.json'
            if failure.exists():errors.append({'dataset':name,'method':method,**read(failure),
                'current_status':row['status'],'resolved':row['status'] in ('passed','fallback')})
        calls += [read(p) for p in (unit/'physical_calls').glob('*.json')]
        fits += [read(p) for p in (unit/'inner_models').glob('*/*/fit.json')]
        fits += [read(p) for p in (unit/'outer_models').glob('*/fit.json')]
        aux += [read(p) for p in (unit/'repair_bundles').glob('*.json')]
        native += [read(p) for tree in ['inner_models','outer_models'] for p in (unit/tree).glob('**/generation.json')]
        inner_requests+=len(list((unit/'scores').glob('*.json')))
    physical=len(calls);logical=0;logical_by_method=[]
    for task in c['tasks']:
        unit=out/'units'/task['dataset'];physical_records=[read(p) for p in (unit/'physical_calls').glob('*.json')]
        for method in ['caafe','llmfe','lm_cleaning_simplified','without_task','ours']:
            # Include unfinished trajectories and interruptions, not just saved selections.
            rounds={int(p.stem) for p in (unit/'methods'/method/'calls').glob('*.json')}
            rounds.update(r['round'] for r in physical_records if r['method']==method)
            logical+=len(rounds)
            logical_by_method.append({'dataset':task['dataset'],'method':method,'attempted_rounds':sorted(rounds)})
    write(out/'logical_calls.json',logical_by_method)
    costs={'logical_requests_iterative_methods':logical,'physical_lm_calls':physical,'physical_budget':100,
        'input_tokens':sum(r.get('input_tokens',0) for r in calls),'output_tokens':sum(r.get('output_tokens',0) for r in calls),
        'generation_seconds_sum':sum(r.get('seconds',0) for r in calls),
        'inner_score_cache_entries':inner_requests,'primary_and_auxiliary_linear_predictor_fits':sum(r['predictor_fits'] for r in fits),
        'predictor_fit_seconds_sum':sum(r['seconds'] for r in fits),'conditional_imputer_fits':sum(r['fitted_imputers'] for r in aux),
        'native_generation_jobs':len(native),'native_generation_seconds_sum':sum(r.get('seconds',0) for r in native),
        'native_generation_failures':sum(r['status']!='passed' for r in native),
        'interrupted_generations':sum(r['status']=='interrupted' for r in calls),
        'token_counts_complete':all(r['status']=='completed' for r in calls),
        'token_count_note':'Measured completed generations only; interrupted output token counts are unknown, not zero.',
        'sft_updates':0,'grpo_updates':0,'sealed_test_opened':False}
    write(out/'results.json',rows);write(out/'costs.json',costs);write(out/'failures.json',errors)
    pd.DataFrame(flat).to_csv(out/'results.csv',index=False)
    completed=sum(r['status'] in ('passed','fallback') for r in rows)
    summary={'stage':'E2 four-task engineering pilot, not formal E2','task_results':len(rows),'expected_results':60,
        'scored_results':completed,'blocked_results':sum(r['status']=='blocked' for r in rows),
        'fallback_results':sum(r['status']=='fallback' for r in rows),'physical_budget_passed':physical<=100,
        'full_e2_ready':False,'readiness_note':'Code baselines stopped on output-contract/truncation failures; external cleaning-agent provenance and full operator/method freeze remain before formal E2.',
        'new_training':False,'sealed_test_opened':False,'automatic_full_e2':False}
    write(out/'summary.json',summary)
    labels_by_method={'raw':'不处理','standard':'标准清洗','openfe':'OpenFE','autofeat':'AutoFeat','caafe':'CAAFE-local',
        'llmfe':'LLM-FE-local','classical_clean':'IF+KNN','lm_cleaning_simplified':'清洗LM（简化适配）',
        'clean_then_openfe':'清洗→OpenFE','one_shot':'一次规划','without_task':'无任务语义','random_order':'随机顺序',
        'fixed_probe':'固定代码探索','no_lm_search':'同库无LM搜索','ours':'任务条件规划'}
    names={'credit':'信贷分类 AUC↑','medical':'住院分类 AUC↑','bike':'自行车回归 RMSE↓','student':'学生回归 RMSE↓'}
    table=[]
    for method in METHODS:
        row=[labels_by_method[method]]
        for task in c['tasks']:
            result=next(r for r in rows if r['dataset']==task['dataset'] and r['method']==method)
            if result['status'] not in ('passed','fallback'):row.append('未通过');continue
            metric='auc' if result['type']=='classification' else 'rmse'
            row.append(f"{result['xgboost'][metric]:.6f}"+('（回退）' if result['status']=='fallback' else ''))
        table.append(row)
    headers=['方法']+[names[t['dataset']] for t in c['tasks']]
    with (out/'E2_小批结果表.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(table)
    lines=['# E2四任务小批结果','',
        '旧开发数据、seed42/外折1、N=5；每任务训练最多2000行、验证最多1000行。不是正式20表主结果，没有三种子SD或测试集成绩。','',
        '|'+ '|'.join(headers)+'|','|---|---|---|---|---|']
    lines += ['|'+ '|'.join(row)+'|' for row in table]
    lines += ['',f'已评分{completed}/60，blocked {summary["blocked_results"]}，全部提案不可执行回退{summary["fallback_results"]}项。',
        '“回退”不是该原生方法成功，不用于宣称优于失败基线；失败细节见failures.json，逐调用原文和执行记录见各methods目录。',
        '代码基线因执行契约冲突和长输出截断提前停止；中断请求占用原预算、不重采。该停止是工程问题处置，不是按下游成绩筛选。CAAFE已完成单元的执行器修正回放仅作独立诊断，不替换原在线结果。',
        '', '## 成本','',json.dumps(costs,ensure_ascii=False,indent=2),
        '', '## 判读与尚待完成','',
        '只能据此验收接口、预算、训练内拟合和状态重放，不能由单种子首折推断方法泛化、显著收益或主实验胜率。',
        '所有任务选择在外验证标签评分前冻结；没有打开旧E1或新表封存测试，也没有训练模型权重。',
        '清洗LM是明确标记的任务书清洗限定简化适配，外部论文基线出处及实现仍须确定。当前有限库未覆盖全部日期/单位/去重功能，正式E2方法和数据尚待冻结。',
        'CAAFE保留官方提示/逐步生成采用，LLM-FE使用官方三岛经验库及预测平均；两者采用本地Qwen和显式训练fit/apply函数、AUC/RMSE协议，非原论文闭源模型结果。',
        '成本中的predictor_fits只计共同评分器及辅助线性模型；原生OpenFE/AutoFeat内部估计器拟合次数未单独计量，另报生成任务数与时长。',
        '完整辅助线性指标、分类F1见results.csv。本轮到此交付，不自动扩表、增加调用或重训。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['prepare','run','report'],required=True);parser.add_argument('--datasets')
    args=parser.parse_args();c=load_config(args.config)
    if args.stage=='prepare':prepare(c)
    elif args.stage=='run':
        for name in args.datasets.split(','):run_unit(c,name)
    else:report(c)


if __name__=='__main__':main()
