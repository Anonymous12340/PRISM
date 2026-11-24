"""Complete field coverage and composed fixed-library plans, with unchanged transforms."""
from pathlib import Path
import itertools
import random
import signal
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from . import e2_library as lib
from .e2 import FormalScorer
from .e2_pilot import Engine, Scorer, RAW, key, timeout_handler
from .e2_semantic import usable, check_steps, label
from .e2_search import paired_stats, select_pool, fit_selected
from .io import read, write, plain

FEATURES={'flag','signed_log','square','bin5','ratio','difference','sum','product','frequency','group_mean','affine'}
ARITHMETIC={'ratio','difference','sum','product'}


def registry(frame, dates=None):
    values=lib.menu(frame,dates,limit=None)
    original=list(values)
    for operation in original:
        if operation['kind'] not in ['ratio','difference']:continue
        reverse={k:v for k,v in operation.items() if k!='id'}
        reverse['left'],reverse['right']=reverse['right'],reverse['left']
        values.append({**reverse,'id':'R'+key(reverse)})
    assert len({v['id'] for v in values})==len(values)
    return values


def dependencies(columns, operations):
    available=set(columns)
    for operation in operations:
        required={operation[k] for k in ['column','left','right','group'] if k in operation}
        absent=required-available
        if absent:raise ValueError('Plan reads unavailable columns: '+','.join(sorted(absent)))
        if operation['kind'] in ['onehot','drop','date_parts']:available.discard(operation['column'])
        if operation['kind'] in FEATURES:available.add('__e2_'+operation['id'])


def plan_label(spec, operations):
    aliases={};parts=[]
    for index,identifier in enumerate(spec['ids'],1):
        op=operations[identifier];display=dict(op)
        for field in ['column','left','right','group']:
            if field in display:display[field]=aliases.get(display[field],display[field])
        parts.append(label(display))
        if op['kind'] in FEATURES:aliases['__e2_'+identifier]='@'+str(index)
        elif op['kind']=='ordinal':aliases[op['column']]='@'+str(index)
    return ' -> '.join(parts) or 'keep'


class ExpandedScorer(FormalScorer):
    def __init__(self,c,unit):
        super().__init__(c,unit)
        path=unit/'derived_ops.json';self.derived=read(path) if path.exists() else {}
        self.ops.update(self.derived)

    def register(self,kind,**arguments):
        op={'kind':kind,'family':lib.FAMILIES[kind],**arguments};identifier='D'+key(op)
        op={**op,'id':identifier}
        if identifier in self.ops:assert self.ops[identifier]==op
        self.ops[identifier]=op;self.derived[identifier]=op
        return identifier

    def transform(self,spec,train,valid,y,groups,slot,where):
        if spec['kind']=='plan':dependencies(train.columns,[self.ops[i] for i in spec['ids']])
        result=FormalScorer.transform(self,spec,train,valid,y,groups,slot,where)
        if spec['kind']=='plan':
            checks=check_steps(train,valid,joblib.load(where/'transform_state.joblib')['steps'])
            write(where/'expanded_check.json',{'passed':True,'checks':checks,'training_rules_only':True})
        return result

    def inner(self,spec):
        if spec['kind']=='plan':
            dependencies(self.x.columns,[self.ops[i] for i in spec['ids']])
            if (self.unit/'scores'/(key(spec)+'.json')).exists():
                for index,fold in enumerate(self.folds):
                    folder=self.unit/'inner_models'/key(spec)/str(index);marker=folder/'expanded_check.json'
                    if not marker.exists():
                        checks=check_steps(self.x.iloc[fold['train']],self.x.iloc[fold['validation']],joblib.load(folder/'transform_state.joblib')['steps'])
                        write(marker,{'passed':True,'checks':checks,'training_rules_only':True})
        return Scorer.inner(self,spec)


def choices(sc,best,known,focus,correlations):
    result={};prefixes=[[],best['spec']['ids']] if best['spec']['ids'] else [[]]
    def add(ids):
        if not ids or len(ids)>4 or len(set(ids))!=len(ids):return
        spec={'kind':'plan','ids':ids}
        if key(spec) in known:return
        try:dependencies(sc.x.columns,[sc.ops[i] for i in ids])
        except ValueError:return
        name=plan_label(spec,sc.ops)
        assert name not in result or result[name]==spec
        result[name]=spec
    # All applicable primitive actions on every field are reachable every round.
    for op in sc.menu:
        if not usable(op,sc.x):continue
        for prefix in prefixes:
            if op['id'] not in prefix:add(prefix+[op['id']])
    numeric=list(sc.x.select_dtypes(include=np.number))
    order=sorted(numeric,key=lambda col:(-abs(correlations.get(col,0.)),col))
    selected=list(dict.fromkeys(([focus] if focus in numeric else [])+order))[:5]
    first_steps=[op for op in sc.menu if op['kind'] in ARITHMETIC and op['left'] in selected and op['right'] in selected]
    for first in first_steps:
        intermediate='__e2_'+first['id']
        for third in selected:
            if third in [first['left'],first['right']]:continue
            for kind in ['ratio','product','difference','sum']:
                operands=[(intermediate,third)]
                if kind in ['ratio','difference']:operands.append((third,intermediate))
                for left,right in operands:
                    second=sc.register(kind,left=left,right=right)
                    add([first['id'],second])
                    prefix=best['spec']['ids']
                    if first['id'] in prefix:add(prefix+[second])
                    elif prefix:add(prefix+[first['id'],second])
    prefix=best['spec']['ids']
    if prefix and len(prefix)<4:
        last=sc.ops[prefix[-1]]
        operand=('__e2_'+last['id'] if last['kind'] in FEATURES else last.get('column') if last['kind']=='ordinal' else None)
        if operand:
            for column in numeric:
                for kind in ['ratio','product','difference','sum']:
                    pairs=[(operand,column)]+([(column,operand)] if kind in ['ratio','difference'] else [])
                    for left,right in pairs:add(prefix+[sc.register(kind,left=left,right=right)])
    write(sc.unit/'derived_ops.json',sc.derived)
    return result


def context(sc,raw,best,history,number,s):
    evidence=sc.formal_evidence(True);corr=evidence.get('training_only_target_spearman',{})
    columns=evidence['columns'];ordered=sorted(columns,key=lambda v:(-v['missing_count']/len(sc.x),-abs(corr.get(v['name'],0.)),v['name']))
    focus=ordered[(number-1)%len(ordered)]['name'];detail=[focus]+[v['name'] for v in ordered if v['name']!=focus][:3]
    # Templates replace a huge flat menu in the prompt. The decoding enum still
    # includes every applicable raw-field action, including formerly omitted fields.
    templates={kind:label(next(o for o in sc.menu if o['kind']==kind)) for kind in dict.fromkeys(o['kind'] for o in sc.menu)}
    for kind in templates:
        example=next(o for o in sc.menu if o['kind']==kind)
        generic=dict(example)
        for field in ['column','left','right','group']:
            if field in generic:generic[field]=field
        templates[kind]=label(generic)
    payload={'input':{'background':sc.source['pair']['background'][:900],'task':sc.task,'focus_column':focus,
        'profiles':[v for v in columns if v['name'] in detail],
        'all_fields':[{'column':v['name'],'type':v['kind'],'missing':v['missing_count'],'target_rho':round(corr.get(v['name'],0.),4)} for v in columns],
        'operator_templates':templates,'derived_reference':'@1 means the numeric result of step 1; @2 means step 2, etc.'},
        'own_training_feedback':history,'raw_scores':raw['scores'],'best_plan':plan_label(best['spec'],sc.ops),'best_scores':best['scores'],
        'remaining_local_model_calls':s['calls_per_unit']-number+1}
    return payload,focus,corr


INSTRUCTION='''依据任务和字段含义，用固定六类操作规划数据处理，输出JSON diagnosis/reason/operations。
diagnosis为missing/outlier_candidate/none/uncertain。reason用中文1–2句说明任务机制、字段关系和风险，通常不超过80字。
operations最多一个字符串，字符串内为完整的1至4步计划，操作间使用“ -> ”连接。调用格式按operator_templates，列名必须精确使用all_fields中的名字。允许用@1、@2引用前面产生的数值特征，例如先做两个字段的乘积，再把它除以第三个字段；所有引用必须指向前面的输出。编码之后才能对新数值做运算。每次只能选约束解码允许的序列。
现在所有原始字段都有可用操作，不能因为前一个短目录没有列出就认为不能处理。结合任务考虑有实际意义的比率、交互、分组关系或标记，保持原值和有效信号。已有标记没有增益时，应考虑不同的任务机制，而不是把同一种无效动机换列重复。
IQR极端不代表真错；无缺失且没有验证的语义错误时，不自动模型替换。单位缩放须有明确的单位不一致依据，不能把缩放本身当错误证据。格式、计数表示及训练中成立的字段关系应保留。不能臆造反馈或确认集分数。
输入包含明确标注的代码探索反馈与自己已尝试的计划，可在真实有效的已知计划上修订，也可提出新假设；已评完整非空序列不能重复。keep或空数组仅表示当前字段保持，随后代码会继续其他字段，所有模型调用最多预设次数。'''


def evaluate(sc,spec,timeout):
    old=signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(timeout)
    try:return sc.inner(spec)
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


def run_unit(s,c,unit,engine,pilot=False):
    if (unit/'complete.json').exists():return read(unit/'complete.json')
    sc=ExpandedScorer(c,unit);raw=sc.inner(RAW);pool={key(RAW):raw};known={key(RAW)};history=[];best=raw
    rng=random.Random(c['seed'])
    for number in range(1,s['code_candidates']+1):
        payload,focus,corr=context(sc,raw,best,history,number,s)
        available=choices(sc,best,known,focus,corr)
        if not available:break
        chosen=rng.choice(sorted(available));spec=available[chosen];known.add(key(spec))
        entry={'origin':'code_exploration','round':number,'plan':chosen,'spec':spec,'executable':False}
        try:
            result=evaluate(sc,spec,c['candidate_timeout']);pool[key(spec)]=result
            entry.update(executable=True,scores=result['scores'],paired_vs_raw=paired_stats(result,raw))
            if result['utility']>best['utility']+1e-12:best=result
        except Exception as exc:entry['error']=repr(exc)
        history.append(entry)
    write(unit/'code_history.json',history)
    _,code_selected=select_pool(pool,raw,sc.ops,s)
    code_pool=dict(pool);code_history=list(history)
    bound=1 if pilot else s['calls_per_unit']
    for number in range(1,bound+1):
        payload,focus,corr=context(sc,raw,best,history,number,s);available=choices(sc,best,known,focus,corr)
        if not available:break
        schema={'type':'object','properties':{'diagnosis':{'type':'string','enum':['missing','outlier_candidate','none','uncertain']},
            'reason':{'type':'string','minLength':1,'maxLength':240},
            'operations':{'type':'array','items':{'type':'string','enum':['keep',*available]},'maxItems':1}},
            'required':['diagnosis','reason','operations'],'additionalProperties':False}
        record=engine.generate([{'role':'system','content':INSTRUCTION},{'role':'user','content':json.dumps(plain(payload),ensure_ascii=False)}],schema,unit,'parameterized',number)
        entry={'origin':'language_model','round':number,'executable':False,'input_tokens':record.get('input_tokens')}
        try:
            if not record['format_valid']:raise ValueError('Invalid JSON output')
            chosen=(record['answer']['operations'] or ['keep'])[0];entry['plan']=chosen
            if chosen=='keep':entry.update(executable=True,local_keep=True)
            else:
                spec=available[chosen];assert key(spec) not in known;known.add(key(spec));entry['spec']=spec
                result=evaluate(sc,spec,c['candidate_timeout']);pool[key(spec)]=result
                entry.update(executable=True,scores=result['scores'],paired_vs_raw=paired_stats(result,raw))
                if result['utility']>best['utility']+1e-12:best=result
        except Exception as exc:entry['error']=repr(exc)
        history.append(entry);write(unit/'history.json',history)
        print({'unit':str(unit),'round':number,'entry':entry},flush=True)
    _,chosen=select_pool(pool,raw,sc.ops,s)
    selection={'plans':{'raw':RAW,'expanded_code':code_selected['spec'],'expanded_system':chosen['spec']},
        'code_candidates':[v for v in code_pool.values()],'system_candidates':list(pool.values()),'before_development_recheck':True}
    write(unit/('pilot_selection.json' if pilot else 'selection.json'),selection)
    if not pilot:
        write(unit/'complete.json',{'passed':True,'model_calls':sum(v['origin']=='language_model' for v in history),
            'code_candidates':len(code_history),'new_nonempty_proposals':sum('spec' in v for v in history),
            'failed':sum(not v['executable'] for v in history)})
    return selection
