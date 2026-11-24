"""Source-only trajectory export, SFT and GRPO through the same runtime contracts."""
from pathlib import Path
import json
import math
import random
import time
from .io import read,write,append,records,resolved


def export(run,output,datasets,kind='sft'):
    run=Path(run).resolve();output=Path(output).resolve();chosen={};skipped=0
    pattern='contexts.jsonl' if kind=='contexts' else 'trajectories.jsonl'
    for path in run.glob('units/*/*/*/*/*/variants/*/'+pattern):
        for r in records(path):
            m=r['metadata']
            if m['dataset'] not in datasets:continue
            if kind=='contexts':
                r['scorer_dir']=str(resolved(path.parent,r['scorer_dir']))
                r['scorer_dir']=__import__('os').path.relpath(r['scorer_dir'],output.parent)
                key=json.dumps([r['payload'],r['feedback'],r['scorer_dir'],r['best_plan']],sort_keys=True)
            else:
                eligible=m['model_generated'] and m['format_valid'] and m['facts_match'] and m['utility']>m['raw_utility']+1e-12 and m['utility']>=m['known_best_utility']-1e-12
                if not eligible:skipped+=1;continue
                key=json.dumps(r['messages'][:-1],sort_keys=True,ensure_ascii=False)
                if key in chosen and chosen[key]['metadata']['utility']>=m['utility']:continue
            chosen[key]=r
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen.values()),encoding='utf-8')
    summary={'kind':kind,'records':len(chosen),'source_datasets':datasets,'skipped_ineligible':skipped,
        'model_calls':0,'truth_used_for_selection':False}
    write(output.with_suffix('.summary.json'),summary);return summary


def training_config(path):
    path=resolved('.',path);c=read(path)
    if c.get('schema_version')!=1:raise ValueError('Training schema_version must be 1')
    for key in ['data','output_dir','adapter_path']:
        if c.get(key):c[key]=str(resolved(path.parent,c[key]))
    if c.get('model_path') and (c['model_path'].startswith(('.','/','~')) or (path.parent/c['model_path']).exists()):
        c['model_path']=str(resolved(path.parent,c['model_path']))
    if not c.get('source_datasets') or set(c['source_datasets'])&set(c.get('heldout_datasets',[])):
        raise ValueError('Declare disjoint source_datasets and heldout_datasets')
    return c


def validate_sources(c,kind):
    rows=records(c['data'])
    if not rows:raise ValueError('No training records')
    for r in rows:
        m=r['metadata']
        if m['dataset'] not in c['source_datasets'] or m['dataset'] in c.get('heldout_datasets',[]):
            raise ValueError('Training record is outside declared source tables')
        if m.get('mode')=='demo' and not c.get('allow_demo_sources',False):raise ValueError('Synthetic demo training requires explicit allow_demo_sources')
        if kind=='sft':
            if not m['model_generated'] or not m['facts_match'] or not m['format_valid'] or m['utility']<=m['raw_utility']+1e-12 or m['utility']<m['known_best_utility']-1e-12:
                raise ValueError('SFT record lacks eligible real positive trajectory provenance')
            if r['messages'][-1]['role']!='assistant':raise ValueError('SFT assistant response missing')
        else:
            source=resolved(Path(c['data']).parent,r['scorer_dir'])/'source.joblib'
            if not source.is_file():raise FileNotFoundError(source)
    return rows


def train_sft(c,validate_only=False):
    rows=validate_sources(c,'sft')
    if len(rows)<c.get('min_examples',1):raise ValueError('Not enough eligible SFT examples')
    if validate_only:return {'validated':True,'records':len(rows),'updates':0,'weights_loaded':False}
    out=Path(c['output_dir'])
    if out.exists():raise ValueError('Training output exists; use a new directory')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,set_seed
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    out.mkdir(parents=True);write(out/'config.json',c);set_seed(c.get('seed',42))
    torch.set_num_threads(c.get('threads',2))
    tokenizer=AutoTokenizer.from_pretrained(c['model_path'],local_files_only=c.get('local_files_only',True))
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    examples=[];token_checks=[]
    for r in rows:
        answer=r['messages'][-1]['content']
        rendered=tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=False,enable_thinking=False)
        prefix,found,suffix=rendered.rpartition(answer)
        if not found:raise ValueError('Assistant response not found in rendered teacher')
        encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True);ids=encoded['input_ids']
        if len(ids)>c.get('max_tokens',3072):raise ValueError('Teacher exceeds max_tokens; never silently truncate')
        labels=[token if end>len(prefix) else -100 for token,(_,end) in zip(ids,encoded['offset_mapping'])]
        first=next(i for i,v in enumerate(labels) if v!=-100)
        assert json.loads(tokenizer.decode(ids[first:],skip_special_tokens=True))==json.loads(answer)
        examples.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels})
        token_checks.append({'total_tokens':len(ids),'masked_prompt_tokens':first,'supervised_tokens':len(ids)-first})
    write(out/'token_checks.json',{'assistant_only':True,'no_truncation':True,'records':token_checks})
    def collate(batch):
        width=math.ceil(max(len(r['input_ids']) for r in batch)/8)*8
        return {k:torch.tensor([r[k]+[pad]*(width-len(r[k])) for r in batch])
            for k,pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)]}
    dtype=getattr(torch,c.get('dtype','bfloat16'))
    model=AutoModelForCausalLM.from_pretrained(c['model_path'],local_files_only=c.get('local_files_only',True),
        torch_dtype=dtype,attn_implementation='sdpa')
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=c.get('rank',8),lora_alpha=c.get('alpha',16),
        lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'));model.config.use_cache=False
    model.enable_input_require_grads();accum=c.get('gradient_accumulation',4);epochs=c.get('epochs',3)
    arguments=TrainingArguments(output_dir=str(out/'trainer'),num_train_epochs=epochs,per_device_train_batch_size=1,
        gradient_accumulation_steps=accum,learning_rate=c.get('learning_rate',5e-5),warmup_steps=math.ceil(.1*math.ceil(len(rows)/accum)*epochs),
        bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},save_strategy='no',eval_strategy='no',
        report_to=[],logging_steps=1,remove_unused_columns=False,seed=c.get('seed',42))
    trainer=Trainer(model=model,args=arguments,train_dataset=examples,data_collator=collate,processing_class=tokenizer)
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();result=trainer.train();trainer.save_model(out/'adapter');tokenizer.save_pretrained(out/'adapter')
    saved=load_peft_weights(str(out/'adapter'),device='cpu');current=get_peft_model_state_dict(trainer.model)
    assert set(saved)==set(current) and all(torch.equal(saved[k],current[k].detach().cpu()) for k in saved)
    summary={'status':'trained_pending_evaluation','kind':'sft','records':len(rows),'optimizer_updates':trainer.state.global_step,
        'training_loss':result.training_loss,'seconds':time.monotonic()-started,'adapter_path':'adapter',
        'tensor_save_check':True,'source_datasets':c['source_datasets'],'heldout_datasets':c.get('heldout_datasets',[]),'promoted':False,
        'peak_gpu_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        'trainer_log_history':trainer.state.log_history,
        'actual_source_counts':{name:sum(r['metadata']['dataset']==name for r in rows) for name in c['source_datasets']}}
    write(out/'summary.json',summary);return summary


def advantages(rewards,tolerance=1e-8):
    from statistics import mean,pstdev
    sd=pstdev(rewards);mu=mean(rewards)
    return [(r-mu)/sd if sd>tolerance else 0. for r in rewards]


def grpo_objective(current,old,reference,advantage,epsilon=.2,beta=.02):
    import torch
    ratio=(current-old).exp();delta=reference-current;kl=delta.exp()-delta-1
    surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)
    return (-surrogate+beta*kl).mean(),ratio,kl


def train_grpo(c,validate_only=False):
    contexts=validate_sources(c,'contexts')
    if validate_only:return {'validated':True,'contexts':len(contexts),'updates':0,'weights_loaded':False}
    from statistics import mean
    import torch
    from peft import get_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from .planner import Planner
    from .scoring import SourceScorer
    out=Path(c['output_dir'])
    if out.exists():raise ValueError('Training output exists; use a new directory')
    out.mkdir(parents=True);write(out/'config.json',c)
    random.Random(c.get('seed',42)).shuffle(contexts)
    pconfig={k:c[k] for k in ['model_path','adapter_path']};pconfig.update(backend='local',dtype='float32',
        device=c.get('device','cuda:0'),local_files_only=c.get('local_files_only',True),threads=c.get('threads',2))
    planner=Planner(pconfig,trainable=True);planner.load()
    params=[p for p in planner.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=c.get('learning_rate',1e-5),weight_decay=0.)
    batch_groups=c.get('groups_per_update',4);draws=c.get('samples_per_group',4)
    if draws<2:raise ValueError('GRPO requires at least two genuine draws per group')
    updates=[];calls=0;started=time.monotonic();scorers={};new_fits=0
    try:
        for step,start in enumerate(range(0,len(contexts),batch_groups)):
            if step>=c.get('max_updates',6):break
            batch=[]
            for number,context in enumerate(contexts[start:start+batch_groups]):
                directory=resolved(Path(c['data']).parent,context['scorer_dir'])
                if str(directory) not in scorers:scorers[str(directory)]=SourceScorer(directory)
                scorer=scorers[str(directory)]
                if scorer.source['metadata']['dataset']!=context['metadata']['dataset']:raise ValueError('Context/scorer source table mismatch')
                group=[]
                for draw in range(draws):
                    target=out/'rollouts'/str(step)/str(number)/str(draw)
                    seed=c.get('seed',42)+step*10000+number*draws+draw
                    r=planner.generate(context,target,{'temperature':1.2,'top_p':1.,'top_k':0,'max_new_tokens':320,'sampling_seed':seed},
                        revision=f'grpo-step-{step}',behavior=True);calls+=1
                    old=planner.logprobs(r,'policy').detach().cpu();ref=planner.logprobs(r,'reference').detach().cpu()
                    diff=float((old-torch.tensor(r['behavior_logprobs'])).abs().max())
                    if diff>c.get('probability_tolerance',.1):raise ValueError(f'Behavior/full-prefix logprob mismatch: {diff}')
                    if step==0 and not torch.equal(old,ref):raise ValueError('Initial policy/reference differ')
                    r.update(old_logprobs=old.tolist(),reference_logprobs=ref.tolist(),probability_difference=diff,
                        context_metadata=context['metadata'])
                    if r['format_valid']:
                        proposal={**context['best_plan'],context['column']:r['policy']};before=scorer.new_fits
                        if new_fits+scorer.needed_fits(proposal)>c.get('max_new_fits',100000):raise ValueError('Declared fit budget would be exceeded')
                        value=scorer.evaluate(proposal);new_fits+=scorer.new_fits-before
                        if new_fits>c.get('max_new_fits',100000):raise ValueError('Declared fit budget exceeded')
                        gain=(value['utility']-context['known_best_utility'])/max(abs(context['raw_utility']),1e-12)
                        r.update(task_gain=gain,reward=gain+.01-.001)
                    else:r.update(task_gain=None,reward=-1.)
                    r['_path']=str(target/'rollout.json');group.append(r)
                for r,adv in zip(group,advantages([r['reward'] for r in group])):
                    r['advantage']=adv;write(Path(r['_path']),{k:v for k,v in r.items() if k!='_path'})
                batch+=group
            optimizer.zero_grad(set_to_none=True);losses=[];kls=[];clips=[]
            for r in batch:
                current=planner.logprobs(r,'policy',grad=True)
                loss,ratio,kl=grpo_objective(current,torch.tensor(r['old_logprobs'],device=planner.device),
                    torch.tensor(r['reference_logprobs'],device=planner.device),r['advantage'],c.get('clip_epsilon',.2),c.get('kl_beta',.02))
                if not torch.isfinite(loss) or not torch.isfinite(kl).all():raise ValueError('Nonfinite GRPO objective')
                (loss/len(batch)).backward();losses.append(float(loss.detach()));kls.append(float(kl.detach().mean()))
                clips.append(float(((ratio.detach()-1).abs()>c.get('clip_epsilon',.2)).float().mean()))
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm) or mean(kls)>.1 or mean(clips)>.5:raise ValueError('GRPO numerical stop condition')
            optimizer.step();planner.select('policy',False)
            planner.model.save_pretrained(out/'checkpoints'/str(step+1),selected_adapters=['policy'],safe_serialization=True)
            row={'step':step+1,'loss':mean(losses),'reference_kl':mean(kls),'clip_fraction':mean(clips),'gradient_norm':float(norm)}
            updates.append(row);write(out/'updates.json',updates);print(json.dumps(row),flush=True)
        adapter=out/'checkpoints'/str(len(updates))/'policy';planner.tokenizer.save_pretrained(adapter)
        saved=load_peft_weights(str(adapter),device='cpu');current=get_peft_model_state_dict(planner.model,adapter_name='policy')
        assert set(saved)==set(current) and all(torch.equal(saved[k],current[k].detach().cpu()) for k in saved)
        summary={'status':'trained_pending_evaluation','kind':'grpo','optimizer_updates':len(updates),'model_calls':calls,
            'new_predictor_fits':new_fits,'seconds':time.monotonic()-started,'tensor_save_check':True,
            'adapter_path':str(adapter.relative_to(out)),'source_datasets':c['source_datasets'],'promoted':False}
        write(out/'summary.json',summary);return summary
    except Exception as exc:
        write(out/'failure.json',{'error':str(exc),'completed_updates':len(updates),'model_calls':calls,'new_fits':new_fits,'automatic_retry':False})
        raise
