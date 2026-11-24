"""One planner contract for offline fixtures, base models and trained adapters."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from .io import read,write
from .operations import POLICIES
from .prompts import INSTRUCTION,PROMPT_VERSION


def schema(actions=POLICIES):
    return {'type':'object','properties':{'diagnosis':{'type':'string','enum':['missing','outlier_candidate','none','uncertain']},
        'reason':{'type':'string','minLength':1,'maxLength':100},
        'operations':{'type':'array','items':{'type':'string','enum':list(actions)},'maxItems':1,'uniqueItems':True}},
        'required':['diagnosis','reason','operations'],'additionalProperties':False}


def messages(context):
    return [{'role':'system','content':INSTRUCTION+json.dumps(context['schema'],ensure_ascii=False)},
        {'role':'user','content':json.dumps({'input':context['payload'],'own_training_feedback':context['feedback']},ensure_ascii=False)}]


def validate(answer,actions):
    if not isinstance(answer,dict) or set(answer)!={'diagnosis','reason','operations'}:raise ValueError('Invalid response envelope')
    if answer['diagnosis'] not in ['missing','outlier_candidate','none','uncertain']:raise ValueError('Invalid diagnosis')
    if not isinstance(answer['reason'],str) or not 1<=len(answer['reason'])<=100:raise ValueError('Invalid reason')
    if not isinstance(answer['operations'],list) or len(answer['operations'])>1 or any(a not in actions for a in answer['operations']):
        raise ValueError('Invalid policy')
    return (answer['operations'] or ['keep'])[0]


def facts_match(answer,payload):
    import re
    e=payload['evidence'];pairs=re.findall(r'缺失=(\d+)[;；]IQR=(\d+)[;；]',answer['reason'])
    expected='missing' if e['missing_count'] else 'outlier_candidate' if e['iqr_nonmissing_candidates'] else 'none'
    return bool(pairs) and all(int(a)==e['missing_count'] and int(b)==e['iqr_nonmissing_candidates'] for a,b in pairs) and answer['diagnosis']==expected


class Planner:
    def __init__(self,config,cache_dir=None,trainable=False):
        self.config=deepcopy(config);self.backend=config.get('backend','fixed');self.cache_dir=Path(cache_dir) if cache_dir else None
        self.trainable=trainable;self.loaded=False

    def load(self):
        if self.loaded:return
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
        from .json_decode import tokenizer_data,json_prefix
        self.torch=torch;self.GenerationConfig=GenerationConfig;self.json_prefix=json_prefix
        self.device=self.config.get('device','cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype=getattr(torch,self.config.get('dtype','float32' if self.trainable else 'bfloat16'))
        torch.set_num_threads(self.config.get('threads',2))
        if self.trainable:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        offline=self.config.get('local_files_only',True)
        self.tokenizer=AutoTokenizer.from_pretrained(self.config['model_path'],local_files_only=offline)
        self.model=AutoModelForCausalLM.from_pretrained(self.config['model_path'],local_files_only=offline,
            torch_dtype=self.dtype,attn_implementation='sdpa').to(self.device).eval()
        if self.config.get('adapter_path'):
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,self.config['adapter_path'],
                adapter_name='policy' if self.trainable else 'default',is_trainable=self.trainable).eval()
            if self.trainable:self.model.load_adapter(self.config['adapter_path'],adapter_name='reference',is_trainable=False)
        elif self.trainable:raise ValueError('GRPO requires an explicit SFT adapter reference')
        if self.trainable:
            for module in self.model.modules():
                if isinstance(module,torch.nn.Dropout):module.p=0.
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.model.enable_input_require_grads();self.select('policy',False)
        self.tokens=tokenizer_data(self.tokenizer);self.loaded=True

    def select(self,adapter,grad):
        self.model.set_adapter(adapter)
        for name,p in self.model.named_parameters():p.requires_grad_('.policy.' in name)
        self.model.train(grad)

    def generate(self,context,directory,generation=None,revision='inference',behavior=False):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        msg=messages(context)
        gen={'temperature':0.,'top_p':1.,'top_k':0,'max_new_tokens':320,'sampling_seed':42,
             **self.config.get('generation',{}),**(generation or {})}
        request={'messages':msg,'schema':context['schema'],'planner':self.config,'generation':gen,'revision':revision,'prompt_version':PROMPT_VERSION}
        key=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        cached=self.cache_dir/(key+'.json') if self.cache_dir else None
        if self.backend!='replay' and cached and cached.exists():
            r=read(cached);r['physical_call']=False;write(directory/'response.json',r);return r
        started=time.monotonic()
        r={'messages':msg,'schema':context['schema'],'payload':context['payload'],'feedback':context['feedback'],
            'generation':gen,'revision':revision,'prompt_version':PROMPT_VERSION,
            'backend':self.backend,'model_generated':self.backend=='local','physical_call':self.backend=='local'}
        if self.backend in ('fixed','replay'):
            e=context['payload']['evidence']
            policy=context['best_plan'][context['column']]
            if self.backend=='replay':
                choices=self.config.get('responses',['keep']);policy=choices[context.get('round',0)%len(choices)]
            diagnosis='missing' if e['missing_count'] else 'outlier_candidate' if e['iqr_nonmissing_candidates'] else 'none'
            r['raw_output']=json.dumps({'diagnosis':diagnosis,'reason':e['fact_summary']+'离线流程测试响应。','operations':[policy]},ensure_ascii=False)
            r.update(input_tokens=0,output_tokens=0)
        else:
            self.load();t=self.torch
            if self.trainable:self.select('policy',False)
            t.manual_seed(gen['sampling_seed']);prompt=self.tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            encoded=self.tokenizer(prompt,return_tensors='pt').to(self.device)
            actual=self.GenerationConfig(do_sample=gen['temperature']>0,temperature=gen['temperature'] if gen['temperature']>0 else 1.,
                top_p=gen['top_p'],top_k=gen['top_k'],max_new_tokens=gen['max_new_tokens'],num_beams=1,repetition_penalty=1.,
                bos_token_id=self.model.generation_config.bos_token_id,eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,use_cache=True,return_dict_in_generate=True,output_scores=behavior)
            with t.inference_mode():
                result=self.model.generate(**encoded,generation_config=actual,prefix_allowed_tokens_fn=self.json_prefix(self.tokens,context['schema']))
            completion=result.sequences[0,encoded.input_ids.shape[1]:].tolist()
            r.update(raw_output=self.tokenizer.decode(completion,skip_special_tokens=True),
                prompt_token_ids=encoded.input_ids[0].tolist(),completion_token_ids=completion,
                effective_generation=actual.to_dict(),input_tokens=int(encoded.input_ids.shape[1]),output_tokens=len(completion))
            if behavior:r['behavior_logprobs']=[float(score[0].float().log_softmax(-1)[token]) for score,token in zip(result.scores,completion)]
        r['seconds']=time.monotonic()-started
        # Preserve the real output even if parsing or schema validation fails.
        try:
            r['answer']=json.loads(r['raw_output']);r['policy']=validate(r['answer'],context['schema']['properties']['operations']['items']['enum'])
            r['format_valid']=True;r['facts_match']=facts_match(r['answer'],context['payload'])
        except (json.JSONDecodeError,ValueError) as exc:r.update(format_valid=False,facts_match=False,error=str(exc))
        write(directory/'response.json',r)
        if self.backend!='replay' and cached:write(cached,r)
        return r

    def logprobs(self,record,adapter,grad=False):
        self.load();self.select(adapter,grad);t=self.torch
        prompt=record['prompt_token_ids'];comp=record['completion_token_ids'];ids=t.tensor([prompt+comp],device=self.device)
        allowed=self.json_prefix(self.tokens,record['schema'])
        with t.set_grad_enabled(grad):
            logits=self.model(input_ids=ids,attention_mask=t.ones_like(ids),logits_to_keep=len(comp)+1,use_cache=False).logits[0,:-1].float()
            mask=t.ones_like(logits,dtype=t.bool)
            for i,token in enumerate(comp):
                legal=allowed(0,t.tensor(prompt+comp[:i]));assert token in legal;mask[i,legal]=False
            logits=logits.masked_fill(mask,-t.inf)/record['generation']['temperature']
            return logits.log_softmax(-1).gather(1,t.tensor(comp,device=self.device)[:,None]).squeeze(1)
