"""Numerical verification and complete cached-reference tables for the N5 pilot."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,f1_score,root_mean_squared_error
from .io import read,write
from .e2_parameterized_run import unit_path


def report(s,c):
    out=Path(s['output_dir']);reference=Path(s['reference_output']);rows=[];files=0
    for p in sorted(out.glob('evaluation/*/*/*/result.json')):
        r=read(p);part=read(unit_path(s,r['dataset'],r['seed'])/'split.json')
        for model in ['xgboost','linear']:
            f=pd.read_csv(p.parent/(model+'_predictions.csv'),float_precision='round_trip');assert f.row_id.tolist()==part['confirmation_row_ids']
            y=f.y_true.to_numpy(dtype=r['target_dtype']);prediction=f.prediction.to_numpy(dtype=r['prediction_dtypes'][model])
            values=({'auc':float(roc_auc_score(y,prediction)),'f1':float(f1_score(y,prediction>=.5,zero_division=0))}
                if r['type']=='classification' else {'rmse':float(root_mean_squared_error(y,prediction))})
            for metric,value in values.items():
                assert abs(value-r[model][metric])<=1e-10
                rows.append({'dataset':r['dataset'],'type':r['type'],'seed':r['seed'],'method':r['method'],
                    'downstream':model,'metric':metric,'value':value,'cohort':'adaptive_development'})
            files+=1
    assert files==96
    old=pd.read_csv(reference/'seed_metrics.csv',float_precision='round_trip');old['cohort']='adaptive_development'
    frame=pd.concat([old,pd.DataFrame(rows)],ignore_index=True);frame.to_csv(out/'seed_metrics.csv',index=False)
    summary=frame.groupby(['dataset','type','method','downstream','metric'],as_index=False).agg(mean=('value','mean'),sd=('value','std'),seeds=('seed','nunique'))
    methods=frame.method.unique().tolist();assert len(summary)==24*len(methods) and (summary.seeds==3).all()
    summary.to_csv(out/'summary.csv',index=False);comparisons=[];rankings=[]
    for model in ['xgboost','linear']:
        primary=summary[(summary.downstream==model)&(((summary.type=='classification')&(summary.metric=='auc'))|((summary.type=='regression')&(summary.metric=='rmse')))].copy()
        primary['utility']=np.where(primary.type=='classification',primary['mean'],-primary['mean'])
        primary['rank']=primary.groupby('dataset').utility.rank(ascending=False,method='average')
        for method,g in primary.groupby('method'):rankings.append({'downstream':model,'method':method,'average_rank':float(g['rank'].mean())})
        pivot=primary.pivot(index='dataset',columns='method',values='utility')
        for target in ['expanded_code','expanded_system']:
            for ref in [m for m in methods if m!=target]:
                delta=pivot[target]-pivot[ref];stable=0
                for name in delta[delta>1e-9].index:
                    typ=primary[primary.dataset==name].iloc[0]['type'];metric='auc' if typ=='classification' else 'rmse'
                    seed=frame[(frame.dataset==name)&(frame.downstream==model)&(frame.metric==metric)].pivot(index='seed',columns='method',values='value')
                    sign=1 if metric=='auc' else -1;stable+=int((sign*(seed[target]-seed[ref])>1e-9).sum()>=2)
                comparisons.append({'downstream':model,'method':target,'reference':ref,'wins':int((delta>1e-9).sum()),
                    'ties':int((abs(delta)<=1e-9).sum()),'losses':int((delta< -1e-9).sum()),'winning_tables_with_2_of_3_seeds':stable})
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False);pd.DataFrame(rankings).to_csv(out/'average_ranks.csv',index=False)
    inherited=set()
    for name in s['datasets']:
        for seed in s['seeds']:
            unit=unit_path(s,name,seed);inherited.update(str(unit/v) for v in read(unit/'reused_artifacts.json'))
    fits=[read(p) for pattern in ['units/*/*/search/inner_models/*/*/fit.json','units/*/*/search/final_models/*/fit.json'] for p in out.glob(pattern) if str(p) not in inherited]
    repairs=[read(p) for p in out.glob('units/*/*/search/repair_bundles/*.json') if str(p) not in inherited]
    calls=[read(p) for p in out.glob('units/*/*/search/physical_calls/*.json')];assert len(calls)<=120 and all(v['status']=='completed' for v in calls)
    outputs=[read(p) for p in out.glob('units/*/*/search/methods/parameterized/calls/*.json')]
    histories=[read(p) for p in out.glob('units/*/*/search/history.json')];assert len(histories)==24
    costs={'new_lm_calls':len(calls),'ceiling':120,'new_predictor_and_linear_fits':sum(r['predictor_fits'] for r in fits),
        'new_imputer_fits':sum(r.get('fitted_imputers',0) for r in repairs),'new_native_baseline_fits':0,'new_weight_updates':0,'original_test_loads':0,
        'input_tokens':sum(v.get('input_tokens',0) for v in calls),'output_tokens':sum(v.get('output_tokens',0) for v in calls),
        'wall_seconds':time.time()-read(out/'launch.json')['started_unix']}
    assert costs['new_predictor_and_linear_fits']<=s['maximum_new_predictor_fits'] and costs['new_imputer_fits']<=96
    stats={'mean_input_tokens':float(np.mean([r['input_tokens'] for r in outputs])),
        'reasons_at_limit':sum(len(r['answer']['reason'])==240 for r in outputs),
        'failed_proposals':sum(sum(not e['executable'] for e in h) for h in histories),
        'repeated_nonempty_plans':sum(len([e for e in h if 'spec' in e])-len({tuple(e['spec']['ids']) for e in h if 'spec' in e}) for h in histories),
        'composed_proposals':sum(sum(' -> ' in e.get('plan','') for e in h if e['origin']=='language_model') for h in histories),
        'all_raw_fields_reachable':all(not r['unreachable_raw_fields'] for r in read(out/'preparation.json')['records'])}
    gate_refs=['raw','semantic'];gate=[r for r in comparisons if r['downstream']=='xgboost' and r['method']=='expanded_system' and r['reference'] in gate_refs]
    passed=all(r['wins']>=s['minimum_table_wins'] and r['losses']<=s['maximum_table_losses'] and r['winning_tables_with_2_of_3_seeds']==r['wins'] for r in gate)
    write(out/'costs.json',costs);write(out/'mechanism.json',stats)
    write(out/'verification.json',{'passed':True,'prediction_files_recomputed':files,'new_methods':2,'units':24,
        'all_selected_models_frozen_before_recheck':True,'original_test_loads':0,'adaptive_development_only':True})
    write(out/'FINAL_SUMMARY.json',{'complete':True,'development_signal_gate_passed':passed,'comparisons':comparisons,'costs':costs,
        'mechanism':stats,'adaptive_development_only':True,'new_independent_confirmation_claim':False,'new_training_claim':False})
    display=['raw','legacy_robust','semantic','expanded_code','expanded_system','no_lm_search','openfe','autofeat']
    lines=['# 字段参数覆盖与复合特征修复：N5开发结果','',
        '全部8表开发确认已经在之前查看；本轮是adaptive development，无新盲测/正式E2/训练有效性结论。旧报告和分数保留。',
        '固定原Qwen3-4B + SFT→GRPO权重、下游模型/参数、划分和六类函数。新系统先5代码探索候选再最多5本地LM调用；配套expanded_code只使用相同5代码候选。原NoLM10及OpenFE/AutoFeat等旧结果完全同划分复用，原N10旧系统对新N5只作开发参照。',
        '所有新24单元方案/最终模型冻结后统一重评开发确认，原E2测试源0读取。代码探索成本包含在系统成本，不能把其贡献归给SFT/GRPO训练。',
        f'开发信号门：{"通过" if passed else "未通过"}。','',
        '|模型|方法|对照|胜/平/负|获胜表≥2/3种子|','|---|---|---|---|---:|']
    lines += [f'|{r["downstream"]}|{r["method"]}|{r["reference"]}|{r["wins"]}/{r["ties"]}/{r["losses"]}|{r["winning_tables_with_2_of_3_seeds"]}|' for r in comparisons]
    for model in ['xgboost','linear']:
        table=[]
        for name in s['datasets']:
            typ=next(d['type'] for d in c['datasets'] if d['name']==name);metric='auc' if typ=='classification' else 'rmse';row=[name,metric.upper()]
            for method in display:
                r=summary[(summary.dataset==name)&(summary.downstream==model)&(summary.method==method)&(summary.metric==metric)].iloc[0]
                row.append(f'{r["mean"]:.7f} ± {r["sd"]:.5f}')
            table.append(row)
        columns=['dataset','metric',*display];pd.DataFrame(table,columns=columns).to_csv(out/f'结果表_{model}.csv',index=False,encoding='utf-8-sig')
        lines+=['',f'## {model}：三种子均值±样本SD','','|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']
        lines+=['|'+'|'.join(row)+'|' for row in table]
    lines+=['','## 机制与成本','',str(stats),'',str(costs),'',
        '完整实例修复保证所有原始字段有操作可达；高阶候选是有界构造，不等于所有表达式都被穷举。整数/比例检查为训练推断的保守规则，不声称全为真实业务约束。',
        '全部失败、局部保留与最终回退保留。主/辅助模型和分类/回归完整报告，不混合RMSE量纲，不宣称统计显著。该组合改进不是纯单因素消融。',
        '本轮不据确认挑选更好版本或重训；新的独立泛化结论仍需另行冻结未用于开发的数据。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
