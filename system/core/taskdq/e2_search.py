"""Bounded history-search development pilot, with an untouched planning-confirmation split."""
import argparse
from copy import deepcopy
from pathlib import Path
import signal
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .e2 import FormalScorer,apply_final
from .e2_pilot import Engine,key,RAW,limited_schema,timeout_handler
from .e2_library import menu
from .e1 import metrics
from .scoring import splits,fit_predictor,predict,fit_encoder,encode
from .io import read,write,plain


def settings(path):
    p=Path(path).resolve();s=read(p);root=(p.parent/s['project_root']).resolve();s['project_root']=str(root)
    for name in ['source_config','source_output','historical_output','output_dir','adapter_path','adapter_dependency_dir']:
        s[name]=str(root/s[name])
    c=read(s['source_config']);items=c['datasets']
    chosen=[]
    for typ in ['classification','regression']:chosen += [d['name'] for d in sorted(items,key=lambda v:v['name']) if d['type']==typ][:2]
    assert s['datasets']==chosen and s['seeds']==[42,123,456]
    assert s['maximum_new_calls']==120 and s['calls_per_unit']==10
    c={**c,'output_dir':s['output_dir'],'adapter_path':s['adapter_path'],'adapter_dependency_dir':s['adapter_dependency_dir'],
        'weight_variant':'sft_grpo','weight_revision':'frozen_SFT_then_GRPO_final_step6','strict_replay_existing_calls':True,
        'unit_physical_limit':10,'lm_budget':10,'budgets':[10]}
    return s,c


def unit_path(s,name,seed):return Path(s['output_dir'])/'units'/name/str(seed)/'search'


def prepare(s,c):
    out=Path(s['output_dir']);records=[]
    for name in s['datasets']:
        for seed in s['seeds']:
            unit=unit_path(s,name,seed)
            if (unit/'split.json').exists():records.append(read(unit/'split.json'));continue
            old=Path(s['source_output'])/'units'/name/str(seed)/'final';source=joblib.load(old/'source.joblib')
            x,y,g=source['x'],source['y'],source['groups'];task=source['task']
            cls=StratifiedGroupKFold if task['type']=='classification' else GroupKFold
            aa,bb=next(cls(n_splits=5,shuffle=True,random_state=seed).split(x,y,g))
            assert not set(g.iloc[aa])&set(g.iloc[bb])
            if task['type']=='classification':assert y.iloc[aa].nunique()==y.iloc[bb].nunique()==2
            search=deepcopy(source)
            for field in ['x','y','groups']:search[field]=source[field].iloc[aa]
            unit.mkdir(parents=True,exist_ok=True);joblib.dump(search,unit/'source.joblib')
            confirmation={field:source[field].iloc[bb] for field in ['x','y','groups']}
            dest=out/'development_confirmation'/name/str(seed);dest.mkdir(parents=True,exist_ok=True)
            joblib.dump(confirmation,dest/'confirmation.joblib')
            inner=splits(search['x'],search['y'],task,3,seed,search['groups'])
            write(unit/'inner_splits.json',[{'train':a.tolist(),'validation':b.tolist(),
                'train_row_ids':search['x'].index[a].tolist(),'validation_row_ids':search['x'].index[b].tolist()} for a,b in inner])
            write(unit/'menu.json',menu(search['x'],search.get('date_columns'),c['menu_limit']))
            original_split=read(Path(c['data_dir'])/'prepared'/name/str(seed)/'split.json')
            assert set(x.index)==set(original_split['development_row_ids'])
            assert not set(x.index)&set(original_split['test_row_ids'])
            record={'dataset':name,'seed':seed,'search_rows':len(aa),'confirmation_rows':len(bb),
                'search_row_ids':search['x'].index.tolist(),'confirmation_row_ids':confirmation['x'].index.tolist(),
                'original_test_rows_used':0,'groups_disjoint':True,'new_inner_folds':3,'historical_fits_not_reused':True}
            write(unit/'split.json',record);records.append(record)
    assert len(records)==12
    write(out/'preparation.json',{'passed':True,'units':12,'datasets':s['datasets'],'seeds':s['seeds'],
        'search_confirmation_disjoint':True,'original_test_loads':0,'model_calls':0})


def paired_stats(result,raw):
    metric='auc' if 'auc' in raw['scores'] else 'rmse';sign=1 if metric=='auc' else -1
    values=np.asarray([sign*(a[metric]-b[metric]) for a,b in zip(result['fold_scores'],raw['fold_scores'])],dtype=float)
    return {'mean_gain':float(values.mean()),'se':float(values.std(ddof=1)/np.sqrt(len(values))),
        'positive_folds':int((values>1e-12).sum()),'fold_gains':values.tolist()}


def select_pool(pool,raw,ops,s):
    values=list(pool.values());mean=raw
    for r in values:
        if r['utility']>mean['utility']+1e-12:mean=r
    eligible=[]
    for r in values:
        stats=paired_stats(r,raw)
        if stats['positive_folds']>=s['minimum_positive_folds'] and stats['mean_gain']>s['robust_se_coefficient']*stats['se']+1e-12:
            eligible.append(r)
    if not eligible:return mean,raw
    best=max(eligible,key=lambda r:r['utility']);width=s['robust_se_coefficient']*paired_stats(best,raw)['se']
    nearby=[r for r in eligible if r['utility']>=best['utility']-width-1e-12]
    harmful={'model_repair','clip','drop_missing','drop_outlier','drop','drop_constant','drop_redundant'}
    def complexity(r):
        ids=r['spec']['ids'];return (sum(ops[v]['kind'] in harmful for v in ids),len(ids),-r['utility'])
    return mean,min(nearby,key=complexity)


def annotate(sc):
    counts={}
    for col in sc.x:
        series=sc.x[col];entry={'missing':int(series.isna().sum()),'numeric':bool(pd.api.types.is_numeric_dtype(series))}
        if entry['numeric']:
            q1,q3=series.quantile([.25,.75]);entry['iqr_candidates']=int(((series<q1-1.5*(q3-q1))|(series>q3+1.5*(q3-q1))).sum())
        counts[col]=entry
    result=[]
    for original in sc.menu:
        op=dict(original);fact=counts.get(op.get('column'),{});kind=op['kind'];note='Evaluate task usefulness; membership in this library does not prove an error.'
        if kind in ['fill','drop_missing'] and fact.get('missing')==0:note='No missing cells: this action alone is a no-op.'
        elif kind in ['clip','drop_outlier'] and fact.get('iqr_candidates')==0:note='No observed IQR candidates: this action alone is a no-op.'
        elif kind=='flag' and fact.get('iqr_candidates',0)+fact.get('missing',0)==0:note='No missing or IQR candidates: the new flag would be constant.'
        elif kind=='model_repair' and fact.get('iqr_candidates',0)+fact.get('missing',0)==0:note='No cells are currently eligible for missing/outlier repair.'
        elif kind=='unit_scale':note='Use only with semantic evidence of unit inconsistency; scale alone is not evidence of wrong values.'
        elif kind in ['clip','flag','drop_outlier','model_repair']:note='IQR marks suspected values, not confirmed errors. Extreme values may be useful task signals.'
        op['applicability_note']=note;result.append(op)
    return result


def fit_selected(c,sc,spec):
    folder=sc.unit/'final_models'/key(spec)
    if (folder/'fit.json').exists():return
    a,b,y=sc.transform(spec,sc.x,sc.x.iloc[:2],sc.y,sc.source['groups'],'outer',folder)
    model=fit_predictor(a,y,sc.task,c['model']);encoder=fit_encoder(a)
    estimator=LogisticRegression(C=1.,max_iter=2000,random_state=0) if sc.task['type']=='classification' else LinearRegression()
    linear=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),estimator).fit(encode(a,encoder),y)
    joblib.dump({'primary':model,'linear':linear,'encoder':encoder,'spec':spec},folder/'bundle.joblib')
    replay=apply_final(c,folder,sc.x.iloc[:2]);pd.testing.assert_frame_equal(b,replay)
    write(folder/'fit.json',{'predictor_fits':2,'training_row_ids':a.index.tolist(),'confirmation_labels_opened':False})


def run_unit(s,c,name,seed,engine):
    unit=unit_path(s,name,seed);cc={**c,'seed':seed};sc=FormalScorer(cc,unit)
    if (unit/'complete.json').exists():return read(unit/'complete.json')
    raw=sc.inner(RAW);probe=sc.inner(sc.probe);pool={key(RAW):raw,key(probe['spec']):probe};known={k:{'result':v} for k,v in pool.items()}
    best=raw if raw['utility']>=probe['utility']-1e-12 else probe
    history=[];stall=0;visible=annotate(sc);input_facts=sc.formal_evidence(True)
    instruction=('Plan task-relevant table processing using the complete fixed operation library. Return JSON reason/operations, up to four operation IDs in order; [] is valid. '
        'Usually change only one or two key actions. Preserve useful signal. IQR candidates are not confirmed errors. '
        'Read the FULL attempt history: do not propose an already evaluated sequence, including the initial raw/probe plans. '
        'Explain a new, applicable hypothesis using task semantics and training facts. Try a different justified operation family or column when prior attempts fail. '
        'Do not fill without missing values or rescale without a semantic unit reason. Keep raw if no justified change remains. '
        'Only search-training CV is available. Never invent feedback or hidden scores. The code applies the plan.')
    for round_number in range(1,s['calls_per_unit']+1):
        state={'input':input_facts,'operation_library':visible,'initial_evaluations':[
            {'spec':r['spec'],'scores':r['scores'],'paired_vs_raw':paired_stats(r,raw)} for r in [raw,probe]],
            'full_attempt_history':[{k:v for k,v in attempt.items() if k!='physical_call'} for attempt in history],
            'best_by_original_mean_rule':{'spec':best['spec'],'scores':best['scores']},
            'instruction':'Propose a new applicable sequence not present in the evaluated initial plans or attempt history.'}
        messages=[{'role':'system','content':instruction},{'role':'user','content':__import__('json').dumps(plain(state),ensure_ascii=False)}]
        record=engine.generate(messages,limited_schema(sc),unit,'history_search',round_number)
        row={'round':round_number,'format_valid':record['format_valid'],'executable':False,'duplicate':False,
            'physical_call':record['physical_call']};new=False
        try:
            if not record['format_valid']:raise ValueError('Invalid JSON output')
            ids=record['answer']['operations']
            if len(ids)>4 or len(set(ids))!=len(ids) or any(i not in sc.ops for i in ids):raise ValueError('Invalid bounded plan')
            spec={'kind':'plan','ids':ids};token=key(spec);row['spec']=spec
            row['families']=[sc.ops[i]['family'] for i in ids]
            if token in known:
                row['duplicate']=True
                if 'error' in known[token]:raise ValueError('Previously failed plan: '+known[token]['error'])
                result=known[token]['result']
            else:
                old=signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(c['candidate_timeout'])
                try:result=sc.inner(spec)
                except Exception as exc:known[token]={'error':repr(exc)};raise
                finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
                known[token]={'result':result};pool[token]=result;new=True
            row.update(executable=True,scores=result['scores'],paired_vs_raw=paired_stats(result,raw))
            accepted=result['utility']>best['utility']+1e-12
            if accepted:best=result
            row['accepted_by_original_mean']=accepted
        except Exception as exc:row['error']=repr(exc)
        history.append(row);write(unit/'history.json',history)
        stall=0 if new else stall+1
        print({'dataset':name,'seed':seed,'round':round_number,'new_executable':new,'duplicate':row['duplicate']},flush=True)
        if stall>=s['consecutive_stall_stop']:break
    mean,robust=select_pool(pool,raw,sc.ops,s);selected={'raw':raw['spec'],'history_mean':mean['spec'],'history_robust':robust['spec']}
    write(unit/'selection.json',{'plans':selected,'candidate_results':list(pool.values()),'logical_calls':len(history),
        'budget_ceiling':10,'early_stopped':len(history)<10,'confirmation_opened':False})
    for token,spec in {key(spec):spec for spec in selected.values()}.items():fit_selected(cc,sc,spec)
    distinct={key(r['spec']) for r in history if r['executable']}
    result={'dataset':name,'seed':seed,'passed':True,'logical_calls':len(history),'physical_calls':len(list((unit/'physical_calls').glob('*.json'))),
        'different_executable_emitted_plans':len(distinct),'new_executable_candidates':len(pool)-len({key(RAW),key(probe['spec'])}),
        'executable_proposals':sum(r['executable'] for r in history),'duplicates':sum(r['duplicate'] for r in history),
        'mean_and_robust_differ':mean['spec']!=robust['spec'],'confirmation_opened':False}
    write(unit/'complete.json',result);return result


def freeze(s,c):
    records=[]
    for name in s['datasets']:
        for seed in s['seeds']:
            unit=unit_path(s,name,seed);r=read(unit/'complete.json');assert r['passed'] and r['physical_calls']<=10
            source=joblib.load(unit/'source.joblib');part=read(unit/'split.json')
            assert source['x'].index.tolist()==part['search_row_ids']
            assert not set(part['search_row_ids'])&set(part['confirmation_row_ids'])
            loaded=read(unit/'planner_load.json');assert loaded['file_to_loaded_tensor_check'] and loaded['loaded_adapter_tensors']==144
            for spec in read(unit/'selection.json')['plans'].values():
                folder=unit/'final_models'/key(spec);fit=read(folder/'fit.json')
                assert set(fit['training_row_ids'])<=set(part['search_row_ids'])
                x=source['x'].iloc[:16];a=apply_final(c,folder,x);assert a.index.equals(x.index)
                assert np.isfinite(predict(joblib.load(folder/'bundle.joblib')['primary'],a)).all()
            records.append(r)
    assert len(records)==12 and sum(r['physical_calls'] for r in records)<=120
    write(Path(s['output_dir'])/'all_selections_frozen.json',{'passed':True,'units':12,'confirmation_loaded':False,'details':records})


def confirm(s,c):
    out=Path(s['output_dir']);assert read(out/'all_selections_frozen.json')['passed']
    for name in s['datasets']:
        item=next(d for d in c['datasets'] if d['name']==name)
        for seed in s['seeds']:
            dest=out/'confirmation_results'/name/str(seed)
            if (dest/'complete.json').exists():continue
            if (dest/'access.json').exists():raise ValueError('Partial confirmation access; inspect failure before recovery')
            write(dest/'access.json',{'all_selections_and_models_frozen':True,'original_sealed_test_loaded':False})
            frame=joblib.load(out/'development_confirmation'/name/str(seed)/'confirmation.joblib');unit=unit_path(s,name,seed)
            cache={};rows=[]
            for method,spec in read(unit/'selection.json')['plans'].items():
                token=key(spec);folder=unit/'final_models'/token
                if token not in cache:
                    bundle=joblib.load(folder/'bundle.joblib');x=apply_final(c,folder,frame['x']);assert x.index.equals(frame['x'].index)
                    p=predict(bundle['primary'],x);matrix=encode(x,bundle['encoder'])
                    q=bundle['linear'].predict_proba(matrix)[:,1] if item['type']=='classification' else bundle['linear'].predict(matrix)
                    cache[token]=(p,q)
                record={'dataset':name,'seed':seed,'method':method,'type':item['type'],'spec':spec,
                    'target_dtype':str(np.asarray(frame['y']).dtype),'prediction_dtypes':{},'evaluation':'development_confirmation'}
                for model,p in zip(['xgboost','linear'],cache[token]):
                    assert np.isfinite(p).all();record[model]=metrics(frame['y'],p,item);record['prediction_dtypes'][model]=str(p.dtype)
                    target=dest/method;target.mkdir(parents=True,exist_ok=True)
                    pd.DataFrame({'row_id':frame['x'].index,'y_true':frame['y'],'prediction':p}).to_csv(target/(model+'_predictions.csv'),index=False)
                write(dest/method/'result.json',record);rows.append(record)
            write(dest/'complete.json',{'methods':3,'development_confirmation_loads':1,'original_test_loads':0,'new_fits':0})


def report(s,c):
    from sklearn.metrics import roc_auc_score,f1_score,root_mean_squared_error
    out=Path(s['output_dir']);rows=[];predictions=0
    for p in (out/'confirmation_results').glob('*/*/*/result.json'):
        r=read(p);part=read(unit_path(s,r['dataset'],r['seed'])/'split.json')
        for model in ['xgboost','linear']:
            f=pd.read_csv(p.parent/(model+'_predictions.csv'),float_precision='round_trip');assert f.row_id.tolist()==part['confirmation_row_ids']
            y=f.y_true.to_numpy(dtype=r['target_dtype']);a=f.prediction.to_numpy(dtype=r['prediction_dtypes'][model])
            scores={'auc':float(roc_auc_score(y,a)),'f1':float(f1_score(y,a>=.5,zero_division=0))} if r['type']=='classification' else {'rmse':float(root_mean_squared_error(y,a))}
            for metric,value in scores.items():
                assert abs(value-r[model][metric])<=1e-10
                rows.append({'dataset':r['dataset'],'type':r['type'],'seed':r['seed'],'method':r['method'],'downstream':model,'metric':metric,'value':value})
            predictions+=1
    assert predictions==72;df=pd.DataFrame(rows);df.to_csv(out/'confirmation_seed_metrics.csv',index=False)
    summary=df.groupby(['dataset','type','method','downstream','metric'],as_index=False).agg(mean=('value','mean'),sd=('value','std'),seeds=('seed','nunique'))
    assert len(summary)==36 and (summary.seeds==3).all();summary.to_csv(out/'confirmation_summary.csv',index=False)
    records=read(out/'all_selections_frozen.json')['details'];historical=[]
    for r in records:
        p=Path(s['historical_output'])/'units/sft_grpo'/r['dataset']/str(r['seed'])/'final/methods/ours/selection.json'
        old=read(p)['proposals'][:10];historical.append({'dataset':r['dataset'],'seed':r['seed'],
            'distinct_old_first10':len({key(v['spec']) for v in old if v.get('executable')}),
            'distinct_new_at_budget10':r['different_executable_emitted_plans'],'new_actual_calls':r['logical_calls'],
            'new_candidate_count_excluding_initial':r['new_executable_candidates'],'input_partitions_differ':True})
    pd.DataFrame(historical).to_csv(out/'exploration_comparison.csv',index=False)
    median=float(np.median([r['different_executable_emitted_plans'] for r in records]));calls=sum(r['logical_calls'] for r in records)
    coverage=sum(r['executable_proposals'] for r in records)/calls;gate=median>=s['minimum_median_unique_plans'] and coverage>=s['minimum_executable_fraction']
    changes=sum(r['mean_and_robust_differ'] for r in records)
    comparisons=[]
    for model in ['xgboost','linear']:
        primary=summary[(summary.downstream==model)&(((summary.type=='classification')&(summary.metric=='auc'))|((summary.type=='regression')&(summary.metric=='rmse')))].copy()
        primary['utility']=np.where(primary.type=='classification',primary['mean'],-primary['mean'])
        pivot=primary.pivot(index='dataset',columns='method',values='utility')
        for a,b in [('history_mean','raw'),('history_robust','raw'),('history_robust','history_mean')]:
            d=pivot[a]-pivot[b];comparisons.append({'downstream':model,'method':a,'reference':b,'datasets':4,
                'wins':int((d>1e-9).sum()),'ties':int((abs(d)<=1e-9).sum()),'losses':int((d< -1e-9).sum())})
    pd.DataFrame(comparisons).to_csv(out/'confirmation_comparisons.csv',index=False)
    physical=[read(p) for p in (out/'units').glob('*/*/search/physical_calls/*.json')];assert all(r['status']=='completed' for r in physical)
    fits=[read(p) for p in (out/'units').glob('*/*/search/inner_models/*/*/fit.json')]+[read(p) for p in (out/'units').glob('*/*/search/final_models/*/fit.json')]
    imputers=[read(p) for p in (out/'units').glob('*/*/search/repair_bundles/*.json')]
    cost={'physical_calls':len(physical),'logical_calls':calls,'ceiling':120,'predictor_and_linear_fits':sum(r['predictor_fits'] for r in fits),
        'conditional_imputer_fits':sum(r.get('fitted_imputers',0) for r in imputers),'original_test_loads':0,'new_weight_updates':0,
        'input_tokens':sum(r.get('input_tokens',0) for r in physical),'output_tokens':sum(r.get('output_tokens',0) for r in physical),
        'wall_seconds':time.time()-read(out/'launch.json')['started_unix']}
    write(out/'costs.json',cost)
    write(out/'verification.json',{'passed':True,'saved_prediction_files_recomputed':72,'confirmation_units':12,'methods':3,
        'all_selections_frozen_before_confirmation':True,'original_test_loads':0,'new_fits_for_verification':0})
    findings={'complete':True,'exploration_gate_passed':gate,'median_distinct_executable_plans':median,'executable_fraction':coverage,
        'selection_changes_mean_to_robust':changes,'confirmation_comparisons':comparisons,'costs':cost,'historical_input_partitions_differ':True,
        'not_independent_new_dataset_generalization':True,'training_automatically_started':False}
    write(out/'FINAL_SUMMARY.json',findings)
    lines=['# 历史反馈与保守选择：开发小批结果','',
        '四张表按任务类型/登记名固定抽取：bank_marketing、blood_transfusion、abalone、airfoil；三种子，固定SFT后GRPO权重，温度0、完整E2库。每单元最多10次新调用，连续3次无新可执行计划可早停。',
        '仅原development内部的search/确认划分；确认不参与画像、拟合、搜索或选择。12单元全部模型冻结后才统一确认。未读取原封存测试。该确认不等于全新数据集独立盲测。',
        '',f'新不同可执行提案中位数：{median:.2f}；可执行比例：{coverage:.2%}；预设搜索机制门：'+('通过' if gate else '未通过')+'。',
        f'原均值门与保守门在{changes}/12单元选择不同；保守门仅同池回放，没有新增LM。','',
        '|模型|比较|胜/平/负（4表均值）|','|---|---|---|']
    for r in comparisons:lines.append(f'|{r["downstream"]}|{r["method"]} 对 {r["reference"]}|{r["wins"]}/{r["ties"]}/{r["losses"]}|')
    for model in ['xgboost','linear']:
        table=[]
        for name in s['datasets']:
            typ=next(d['type'] for d in c['datasets'] if d['name']==name);metric='auc' if typ=='classification' else 'rmse';row=[name,metric.upper()]
            for method in ['raw','history_mean','history_robust']:
                r=summary[(summary.dataset==name)&(summary.method==method)&(summary.downstream==model)&(summary.metric==metric)].iloc[0]
                row.append(f'{r["mean"]:.6g} ± {r["sd"]:.4g}')
            table.append(row)
        frame=pd.DataFrame(table,columns=['dataset','metric','不处理','新候选＋原均值门','新候选＋保守门']);frame.to_csv(out/f'小批结果_{model}.csv',index=False,encoding='utf-8-sig')
        markdown=['|'+'|'.join(frame.columns)+'|','|'+'|'.join(['---']*len(frame.columns))+'|']+['|'+'|'.join(row)+'|' for row in table]
        lines+=['',f'## {model}开发确认均值±三种子样本SD','','\n'.join(markdown)]
    lines+=['','## 边界与成本','',
        '历史轨迹只作为探索重复率参考；新search行发生变化，不把其下游分数或候选差异当成纯历史提示的因果收益。原均值门与保守门使用同一真实候选池，因此其选择比较有匹配输入。',
        f'新增真实生成{cost["physical_calls"]}次/实际逻辑{calls}轮，上限120；主/线性预测器拟合{cost["predictor_and_linear_fits"]}，辅助修复{cost["conditional_imputer_fits"]}，权重更新0、原测试访问0。',
        '72份保存预测独立复算通过。搜索机制门不代替下游效果门；本4表结果不足以证明跨表稳定收益，不自动扩训练或正式实验。完整每种子指标、F1、候选和失败全部保留。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage',choices=['prepare','worker','freeze','confirm','report'],required=True)
    p.add_argument('--lane',type=int);a=p.parse_args();s,c=settings(a.config)
    if a.stage=='prepare':prepare(s,c)
    elif a.stage=='worker':
        engine=Engine(c)
        for name,seed in [(n,z) for n in s['datasets'] for z in s['seeds']][a.lane::8]:print(run_unit(s,c,name,seed,engine),flush=True)
    elif a.stage=='freeze':freeze(s,c)
    elif a.stage=='confirm':confirm(s,c)
    else:report(s,c)


if __name__=='__main__':main()
