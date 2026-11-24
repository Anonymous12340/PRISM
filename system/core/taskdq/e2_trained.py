"""Fixed SFT and SFT→GRPO weights in the full E2 runtime; post-hoc supplement."""
import argparse
import datetime
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from .e2 import FormalScorer,fit_final,apply_final
from .e2_pilot import Engine,run_method,key,RAW
from .e1 import metrics
from .scoring import predict,encode
from .io import read,write


def settings(path):
    p=Path(path).resolve();s=read(p);root=(p.parent/s['project_root']).resolve()
    s['project_root']=str(root)
    for name in ['source_config','source_output','output_dir']:s[name]=str(root/s[name])
    if s.get('adapter_dependency_dir'):s['adapter_dependency_dir']=str(root/s['adapter_dependency_dir'])
    c=read(s['source_config']);assert c['seeds']==[42,123,456] and c['budgets']==[5,10,20]
    assert len(c['datasets'])==20 and s['maximum_new_calls']==2400
    for v in s['variants']:
        v['adapter_path']=str(root/v['adapter_path']);assert Path(v['adapter_path'],'adapter_model.safetensors').exists()
    return s,c


def old_unit(s,name,seed):return Path(s['source_output'])/'units'/name/str(seed)/'final'
def unit_path(s,variant,name,seed):return Path(s['output_dir'])/'units'/variant/name/str(seed)/'final'


def prepare(s,c,variant,name,seed):
    old=old_unit(s,name,seed);unit=unit_path(s,variant['name'],name,seed);unit.mkdir(parents=True,exist_ok=True)
    for file in ['source.joblib','inner_splits.json','menu.json']:
        dest=unit/file
        if not dest.exists():dest.symlink_to(old/file)
    repair=unit/'repair_bundles';repair.mkdir(exist_ok=True)
    for path in (old/'repair_bundles').glob('*'):
        if path.is_file() and not (repair/path.name).exists():(repair/path.name).symlink_to(path)
    if not (unit/'final_registry.json').exists():write(unit/'final_registry.json',read(old/'final_registry.json'))
    cc={**c,'output_dir':s['output_dir'],'seed':seed,'adapter_path':variant['adapter_path'],
        'weight_variant':variant['name'],'weight_revision':variant['revision'],'strict_replay_existing_calls':True,
        'adapter_dependency_dir':s.get('adapter_dependency_dir')}
    return cc,unit


class SharedScorer(FormalScorer):
    def __init__(self,c,unit,source):
        super().__init__(c,unit);self.source_unit=source;self.reused_scores=set()
    def inner(self,spec):
        token=key(spec);path=self.source_unit/'scores'/(token+'.json')
        if path.exists():
            result=read(path);assert result['spec']==spec;self.reused_scores.add(token);return result
        return super().inner(spec)


def reuse_final(s,unit,source,spec):
    assert spec['kind']=='plan'
    token=key(spec);dest=unit/'final_models'/token
    if dest.exists():return
    # Plan-only application reads saved state without modifying the old folder.
    sources=[source]+[unit_path(s,v['name'],unit.parents[1].name,int(unit.parent.name)) for v in s['variants']]
    for origin in sources:
        candidate=origin/'final_models'/token
        if candidate!=dest and (candidate/'fit.json').exists() and (candidate/'bundle.joblib').exists():
            dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(candidate.resolve(),target_is_directory=True);return


class PilotStop(Exception):pass
class FirstRoundEngine(Engine):
    def generate(self,messages,schema,unit,method,round_number):
        if round_number>1:raise PilotStop()
        return super().generate(messages,schema,unit,method,round_number)


def pilot(s,c,variant,name,seed):
    cc,unit=prepare(s,c,variant,name,seed);source=old_unit(s,name,seed);scorer=SharedScorer(cc,unit,source)
    if (unit/'pilot_receipt.json').exists():return read(unit/'pilot_receipt.json')
    try:run_method(cc,scorer,FirstRoundEngine(cc),'ours')
    except PilotStop:pass
    record=read(unit/'methods/ours/calls/1.json');previous=read(source/'methods/ours/calls/1.json')
    assert record['request']['messages']==previous['request']['messages']
    assert record['request']['schema']==previous['request']['schema']
    assert record['prompt_token_ids']==previous['prompt_token_ids']
    assert record['request']['planner_weights']['variant']==variant['name']
    loaded=read(unit/'planner_load.json');assert loaded['file_to_loaded_tensor_check'] and loaded['loaded_adapter_tensors']==144
    proposals=read(unit/'methods/ours/proposals.json');assert len(proposals)==1
    write(unit/'pilot_call.json',record)
    result={'variant':variant['name'],'dataset':name,'seed':seed,'passed':True,
        'format_valid':record['format_valid'],'proposal_executable':proposals[0]['executable'],
        'input_schema_and_prompt_ids_match_frozen_base':True,'loaded_adapter_tensors':144,
        'loaded_file_tensor_match':True,'full_operator_menu':True,'test_loads':0,
        'physical_calls':len(list((unit/'physical_calls').glob('*.json')))}
    write(unit/'pilot_receipt.json',result);return result


def run_unit(s,c,variant,name,seed,engine):
    cc,unit=prepare(s,c,variant,name,seed);source=old_unit(s,name,seed)
    if (unit/'complete.json').exists():return read(unit/'complete.json')
    started=time.time();scorer=SharedScorer(cc,unit,source)
    selected=run_method(cc,scorer,engine,'ours');assert selected['logical_calls']==20
    specs=[RAW]+[read(unit/'methods/ours/checkpoints'/f'{b:03}.json')['spec'] for b in c['budgets']]
    for spec in specs:reuse_final(s,unit,source,spec)
    fit_final(cc,scorer,only_method='ours')
    assert not list((unit/'methods/ours/final_fit_failures').glob('*.json')),'Inspect final-fit engineering failure'
    registry=[v for v in read(unit/'final_registry.json') if v['method']=='ours'];assert len(registry)==3
    for row in registry:
        folder=unit/'final_models'/key(row['spec']);bundle=joblib.load(folder/'bundle.joblib')
        frame=scorer.x.iloc[np.unique(np.linspace(0,len(scorer.x)-1,min(16,len(scorer.x))).astype(int))]
        transformed=apply_final(cc,folder,frame);assert transformed.index.equals(frame.index)
        assert np.isfinite(predict(bundle['primary'],transformed)).all()
        assert np.isfinite(bundle['linear'].predict(encode(transformed,bundle['encoder']))).all()
    if (unit/'pilot_call.json').exists():
        old,new=read(unit/'pilot_call.json'),read(unit/'methods/ours/calls/1.json')
        assert all(old[k]==new[k] for k in ['request','raw_output','prompt_token_ids','completion_token_ids'])
    write(unit/'ours_registry.json',registry)
    result={'variant':variant['name'],'dataset':name,'seed':seed,'passed':True,'test_loads':0,
        'physical_calls':len(list((unit/'physical_calls').glob('*.json'))),'logical_calls':20,
        'unique_old_inner_scores_reused':len(scorer.reused_scores),'wall_seconds':time.time()-started,
        'loaded_weights_verified':read(unit/'planner_load.json')['file_to_loaded_tensor_check']}
    write(unit/'complete.json',result);return result


def worker(s,c,lane,gpu,pilot_only=False):
    if pilot_only:
        tasks=[(v,name,42) for v in s['variants'] for name in s['pilot_datasets']]
        variant,name,seed=tasks[lane];return pilot(s,c,variant,name,seed)
    tasks=[(v,d['name'],seed) for v in s['variants'] for d in c['datasets'] for seed in c['seeds']]
    engine=None;active=None;results=[]
    for variant,name,seed in tasks[lane::8]:
        cc,unit=prepare(s,c,variant,name,seed)
        if active!=variant['name']:
            if engine is not None:
                del engine;gc.collect()
                import torch
                torch.cuda.empty_cache()
            engine=Engine(cc);active=variant['name']
        result=run_unit(s,c,variant,name,seed,engine);results.append(result);print(result,flush=True)
    return results


def verify_all(s,c):
    out=Path(s['output_dir']);assert not (out/'all_choices_frozen.json').exists()
    results=[]
    for variant in s['variants']:
        for item in c['datasets']:
            for seed in c['seeds']:
                unit=unit_path(s,variant['name'],item['name'],seed);r=read(unit/'complete.json');assert r['passed'] and r['loaded_weights_verified']
                source=old_unit(s,item['name'],seed)
                for file in ['source.joblib','inner_splits.json','menu.json']:assert (unit/file).resolve()==(source/file).resolve()
                loaded=read(unit/'planner_load.json');assert loaded['variant']==variant['name'] and loaded['adapter_path']==variant['adapter_path']
                assert loaded['file_to_loaded_tensor_check'] and loaded['loaded_adapter_tensors']==144
                assert not list((unit/'methods/ours/final_fit_failures').glob('*.json'))
                assert all(read(p)['status']=='completed' for p in (unit/'physical_calls').glob('*.json'))
                reg=read(unit/'ours_registry.json');assert len(reg)==3 and {v['budget'] for v in reg}=={5,10,20}
                selection=read(unit/'methods/ours/selection.json');assert selection['logical_calls']==20
                for budget in c['budgets']:
                    cp=read(unit/'methods/ours/checkpoints'/f'{budget:03}.json')
                    assert cp['logical_calls']==budget and cp['proposals']==selection['proposals'][:budget]
                results.append(r)
    assert len(results)==120 and sum(r['physical_calls'] for r in results)<=2400
    write(out/'all_choices_frozen.json',{'units':120,'choices':360,'test_loaded_by_supplement':False,
        'physical_calls':sum(r['physical_calls'] for r in results),'logical_calls':2400,'details':results})


def evaluate(s,c):
    out=Path(s['output_dir']);assert read(out/'all_choices_frozen.json')['units']==120
    for item in c['datasets']:
        for seed in c['seeds']:
            target=out/'test_units'/item['name']/str(seed)
            if (target/'complete.json').exists():continue
            if (target/'access.json').exists():raise ValueError('Partial supplementary test access requires inspection')
            write(target/'access.json',{'post_hoc_supplement':True,'original_E2_test_already_used':True,'all_choices_fixed':True})
            test=joblib.load(Path(c['data_dir'])/'prepared'/item['name']/str(seed)/'sealed_test.joblib')
            cache={}
            for variant in s['variants']:
                unit=unit_path(s,variant['name'],item['name'],seed)
                for row in read(unit/'ours_registry.json'):
                    token=key(row['spec']);folder=unit/'final_models'/token
                    if token not in cache:
                        bundle=joblib.load(folder/'bundle.joblib');x=apply_final(c,folder,test['x'])
                        p=predict(bundle['primary'],x);matrix=encode(x,bundle['encoder'])
                        q=bundle['linear'].predict_proba(matrix)[:,1] if item['type']=='classification' else bundle['linear'].predict(matrix)
                        cache[token]=(p,q)
                    r={'dataset':item['name'],'type':item['type'],'seed':seed,'variant':variant['name'],
                        'budget':row['budget'],'status':row['status'],'target_dtype':str(np.asarray(test['y']).dtype),'prediction_dtypes':{},
                        'post_hoc_supplement':True,'new_model_fits':0}
                    dest=target/variant['name']/str(row['budget']);dest.mkdir(parents=True,exist_ok=True)
                    for model,p in zip(['xgboost','linear'],cache[token]):
                        r[model]=metrics(test['y'],p,item);r['prediction_dtypes'][model]=str(p.dtype)
                        pd.DataFrame({'row_id':test['x'].index,'y_true':test['y'],'prediction':p}).to_csv(dest/(model+'_predictions.csv'),index=False)
                    write(dest/'result.json',r)
            write(target/'complete.json',{'test_source_loads_this_supplement':1,'trained_variant_budget_results':6,
                'original_E2_test_already_used':True,'new_model_fits':0,'new_lm_calls':0})
            print({'supplement_test_complete':item['name'],'seed':seed},flush=True)


def report(s,c):
    from sklearn.metrics import roc_auc_score,f1_score,root_mean_squared_error
    out=Path(s['output_dir']);rows=[];files=0;maximum_delta=0.
    for item in c['datasets']:
        for seed in c['seeds']:
            directory=out/'test_units'/item['name']/str(seed);assert read(directory/'complete.json')['test_source_loads_this_supplement']==1
            expected=read(Path(c['data_dir'])/'prepared'/item['name']/str(seed)/'split.json')['test_row_ids']
            for v in s['variants']:
                for b in c['budgets']:
                    folder=directory/v['name']/str(b);r=read(folder/'result.json')
                    for model in ['xgboost','linear']:
                        saved=pd.read_csv(folder/(model+'_predictions.csv'),float_precision='round_trip');assert saved.row_id.tolist()==expected
                        y=saved.y_true.to_numpy(dtype=r['target_dtype']);p=saved.prediction.to_numpy(dtype=r['prediction_dtypes'][model])
                        values={'auc':float(roc_auc_score(y,p)),'f1':float(f1_score(y,p>=.5,zero_division=0))} if item['type']=='classification' else {'rmse':float(root_mean_squared_error(y,p))}
                        for metric,value in values.items():
                            delta=abs(value-r[model][metric]);maximum_delta=max(maximum_delta,delta);assert delta<=1e-10
                            rows.append({'dataset':item['name'],'type':item['type'],'seed':seed,'variant':v['name'],'budget':b,
                                'status':r['status'],'downstream':model,'metric':metric,'value':value})
                        files+=1
    new=pd.DataFrame(rows);old=pd.read_csv(Path(s['source_output'])/'test_results.csv',float_precision='round_trip')
    old_ours=old[old.method=='ours'].copy();old_ours['variant']='base'
    all_models=pd.concat([old_ours[list(new.columns)],new],ignore_index=True)
    all_models.to_csv(out/'three_model_seed_metrics.csv',index=False)
    summary=all_models.groupby(['dataset','type','variant','budget','downstream','metric'],as_index=False).agg(
        mean=('value','mean'),sd=('value','std'),seeds=('seed','nunique'),fallback_seeds=('status',lambda x:sum(v=='fallback' for v in x)))
    assert len(summary)==540 and (summary.seeds==3).all()
    summary.to_csv(out/'three_model_summary.csv',index=False)
    old_summary=pd.read_csv(Path(s['source_output'])/'test_summary.csv',float_precision='round_trip')
    baseline=old_summary.copy();baseline['variant']=baseline['method'];columns=list(summary.columns)
    merged=pd.concat([baseline[columns],summary[summary.variant!='base']],ignore_index=True)
    comparisons=[];rank_rows=[];full_tables=[]
    for model in ['xgboost','linear']:
        for budget in c['budgets']:
            primary=merged[(merged.downstream==model)&(merged.budget==budget)&(((merged.type=='classification')&(merged.metric=='auc'))|((merged.type=='regression')&(merged.metric=='rmse')))].copy()
            primary['utility']=np.where(primary.type=='classification',primary['mean'],-primary['mean'])
            primary['rank']=primary.groupby('dataset').utility.rank(ascending=False,method='average')
            for variant,g in primary.groupby('variant'):
                rank_rows.append({'variant':'base' if variant=='ours' else variant,'downstream':model,
                    'budget':budget,'datasets':20,'mean_rank':float(g['rank'].mean()),'coverage':'all_registered_marked_fallback'})
            for variant in ['sft','sft_grpo']:
                ours=primary[primary.variant==variant].set_index('dataset')
                for reference in list(c['methods'])+['sft','sft_grpo']:
                    if reference==variant:continue
                    ref=primary[primary.variant==reference].set_index('dataset');delta=ours.utility-ref.utility
                    comparisons.append({'variant':variant,'reference':'base' if reference=='ours' else reference,
                        'downstream':model,'budget':budget,'datasets':20,'wins':int((delta>1e-9).sum()),
                        'ties':int((abs(delta)<=1e-9).sum()),'losses':int((delta< -1e-9).sum())})
            table=[]
            for item in c['datasets']:
                metric='auc' if item['type']=='classification' else 'rmse';row=[item['name'],metric.upper()]
                for v in ['base','sft','sft_grpo']:
                    r=summary[(summary.dataset==item['name'])&(summary.variant==v)&(summary.budget==budget)&(summary.downstream==model)&(summary.metric==metric)].iloc[0]
                    row.append(f'{r["mean"]:.6g} ± {r["sd"]:.4g}'+(f' [回退{r.fallback_seeds}/3]' if r.fallback_seeds else ''))
                table.append(row)
            pd.DataFrame(table,columns=['dataset','metric','提示版','SFT','SFT→GRPO']).to_csv(out/f'E2_三模型_{model}_N{budget}.csv',index=False,encoding='utf-8-sig')
            for section,metric_filter in [('主指标',None),('分类F1','f1')]:
                table=[];variants=list(c['methods'])+['sft','sft_grpo']
                for item in c['datasets']:
                    if metric_filter and item['type']!='classification':continue
                    metric=metric_filter or ('auc' if item['type']=='classification' else 'rmse');row=[item['name'],metric.upper()]
                    for variant in variants:
                        r=merged[(merged.dataset==item['name'])&(merged.variant==variant)&(merged.budget==budget)&(merged.downstream==model)&(merged.metric==metric)].iloc[0]
                        row.append(f'{r["mean"]:.6g} ± {r["sd"]:.4g}'+(f' [回退{r.fallback_seeds}/3]' if r.fallback_seeds else ''))
                    table.append(row)
                labels=['提示版' if v=='ours' else 'SFT→GRPO' if v=='sft_grpo' else v for v in variants]
                frame=pd.DataFrame(table,columns=['dataset','metric']+labels)
                frame.to_csv(out/f'E2_全基线三版本_{section}_{model}_N{budget}.csv',index=False,encoding='utf-8-sig')
                full_tables.append((model,budget,section,frame))
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False)
    pd.DataFrame(rank_rows).to_csv(out/'average_ranks.csv',index=False)
    calls=[read(p) for p in (out/'units').glob('*/*/*/final/physical_calls/*.json')]
    fits=[read(p) for p in (out/'units').glob('*/*/*/final/inner_models/*/*/fit.json')]
    fits += [read(p) for p in (out/'units').glob('*/*/*/final/final_models/*/fit.json') if not p.parent.is_symlink()]
    cost={'physical_lm_calls':len(calls),'logical_calls':2400,'input_tokens':sum(r.get('input_tokens',0) for r in calls),
        'output_tokens':sum(r.get('output_tokens',0) for r in calls),'new_downstream_predictor_fits':sum(r.get('predictor_fits',0) for r in fits),
        'peak_gpu_allocated_bytes':max([r.get('peak_allocated_bytes',0) for p in (out/'units').glob('*/*/*/final/lm_cache/*.json') for r in [read(p)]] or [0]),
        'new_sft_grpo_updates':0,'previous_base_generations_reused':1918,'additional_test_source_loads':60,
        'post_hoc_supplement':True,'wall_seconds':time.time()-read(out/'launch.json')['started_unix']}
    cost['new_conditional_imputer_fits']=sum(read(p).get('fitted_imputers',0) for p in (out/'units').glob('*/*/*/final/repair_bundles/*.json') if not p.is_symlink())
    write(out/'costs.json',cost)
    write(out/'independent_verification.json',{'passed':True,'new_prediction_files':files,'metric_values':len(new),'maximum_metric_delta':maximum_delta,
        'training_variants':2,'units':120,'budgets':[5,10,20],'model_fits_for_verification':0,'test_rows_preserved':True})
    lines=['# E2完整框架：提示、SFT与SFT→GRPO补充结果','',
        '同一完整E2操作库/提示/评分/执行器，实际加载固定SFT或从该SFT继续GRPO的最终权重，仅权重不同。旧教师只训练七种修复策略，本轮评价其向完整全表操作库的迁移，未新增训练。',
        '原20表测试已用于提示版E2，本轮明确为事后补充，不称独立新盲测；所有新增版本的选择和模型先冻结，再统一补充评价。',
        '', '## 主N=10：完整GRPO版的胜平负','', '|比较对象|XGBoost胜/平/负|辅助线性胜/平/负|','|---|---|---|']
    for ref in ['base','sft','raw','standard','openfe','autofeat','no_lm_search']:
        cells=[]
        for model in ['xgboost','linear']:
            r=next(v for v in comparisons if v['variant']=='sft_grpo' and v['reference']==ref and v['downstream']==model and v['budget']==10)
            cells.append(f'{r["wins"]}/{r["ties"]}/{r["losses"]}')
        lines.append('|'+ref+'|'+'|'.join(cells)+'|')
    for model in ['xgboost','linear']:
        for budget in [10,5,20]:
            table=pd.read_csv(out/f'E2_三模型_{model}_N{budget}.csv')
            markdown=['|'+'|'.join(table.columns)+'|','|'+'|'.join(['---']*len(table.columns))+'|']
            markdown+=['|'+'|'.join(map(str,row))+'|' for row in table.itertuples(index=False,name=None)]
            lines+=['',f'## {model}，N={budget}','','\n'.join(markdown)]
    for model,budget,section,table in sorted(full_tables,key=lambda x:({'xgboost':0,'linear':1}[x[0]],{10:0,5:1,20:2}[x[1]],x[2])):
        markdown=['|'+'|'.join(table.columns)+'|','|'+'|'.join(['---']*len(table.columns))+'|']
        markdown+=['|'+'|'.join(map(str,row))+'|' for row in table.itertuples(index=False,name=None)]
        lines+=['',f'## 全基线与三权重：{model} N={budget} {section}','','\n'.join(markdown)]
    lines+=['','## 运行证据','',f'120个新增权重单元，所有单位有144张量磁盘与实际加载核验；{files}份新增预测独立复算通过，最大指标差{maximum_delta}。',
        f'新增真实生成{len(calls)}次，逻辑2400轮；SFT/GRPO更新0次。完整基座及传统方法结果来自同一数据/划分/预测器的已冻结主实验，不重跑原生基线。',
        'SFT→GRPO与同框架base/SFT的比较回答权重贡献；与仍用基础权重的代码基线比较不能单独归因于架构。完整权重谱系、加载记录、真实提案、反馈及操作日志保留服务器。']
    (out/'E2_完整版结果表.md').write_text('\n'.join(lines)+'\n')
    write(out/'FINAL_SUMMARY.json',{'complete':True,'post_hoc_supplement':True,'comparisons':comparisons,'costs':cost})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--lane',type=int);p.add_argument('--gpu',type=int)
    p.add_argument('--stage',choices=['pilot','worker','verify','test','report'],required=True);a=p.parse_args();s,c=settings(a.config)
    if a.stage in ['pilot','worker']:print(worker(s,c,a.lane,a.gpu,a.stage=='pilot'),flush=True)
    elif a.stage=='verify':verify_all(s,c)
    elif a.stage=='test':evaluate(s,c)
    else:report(s,c)


if __name__=='__main__':main()
