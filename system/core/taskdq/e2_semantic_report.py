"""Independent saved-prediction aggregation for the frozen semantic pilot."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, root_mean_squared_error
from .io import read, write
from .e2_semantic import unit_path


def report(s, c):
    out = Path(s['output_dir']); rows = []; count = 0
    methods = ['raw', 'legacy_mean', 'legacy_robust', *(['semantic_v1'] if s.get('focus_coverage_corrected') else []), 'semantic', *s['baselines']]
    for p in sorted(out.glob('confirmation_results/*/*/*/result.json')):
        r = read(p); part = read(unit_path(s, r['dataset'], r['seed']) / 'split.json')
        for model in ['xgboost', 'linear']:
            frame = pd.read_csv(p.parent / (model + '_predictions.csv'), float_precision='round_trip')
            assert frame.row_id.tolist() == part['confirmation_row_ids']
            y = frame.y_true.to_numpy(dtype=r['target_dtype'])
            prediction = frame.prediction.to_numpy(dtype=r['prediction_dtypes'][model])
            scores = ({'auc': float(roc_auc_score(y, prediction)), 'f1': float(f1_score(y, prediction >= .5, zero_division=0))}
                      if r['type'] == 'classification' else {'rmse': float(root_mean_squared_error(y, prediction))})
            for metric, value in scores.items():
                assert abs(value - r[model][metric]) <= 1e-10
                rows.append({'dataset': r['dataset'], 'type': r['type'], 'seed': r['seed'], 'method': r['method'],
                    'downstream': model, 'metric': metric, 'value': value,
                    'cohort': 'validation' if r['dataset'] in s['validation_datasets'] else 'diagnostic'})
            count += 1
    assert count == 24 * len(methods) * 2
    frame = pd.DataFrame(rows); frame.to_csv(out / 'seed_metrics.csv', index=False)
    summary = frame.groupby(['cohort', 'dataset', 'type', 'method', 'downstream', 'metric'], as_index=False).agg(
        mean=('value', 'mean'), sd=('value', 'std'), seeds=('seed', 'nunique'))
    assert len(summary) == 24 * len(methods) and (summary.seeds == 3).all()
    summary.to_csv(out / 'summary.csv', index=False)
    comparisons = []; ranks = []
    for cohort in ['diagnostic', 'validation']:
        for model in ['xgboost', 'linear']:
            primary = summary[(summary.cohort == cohort) & (summary.downstream == model) &
                (((summary.type == 'classification') & (summary.metric == 'auc')) |
                 ((summary.type == 'regression') & (summary.metric == 'rmse')))].copy()
            primary['utility'] = np.where(primary.type == 'classification', primary['mean'], -primary['mean'])
            primary['rank'] = primary.groupby('dataset').utility.rank(ascending=False, method='average')
            for method, group in primary.groupby('method'):
                ranks.append({'cohort': cohort, 'downstream': model, 'method': method, 'average_rank': float(group['rank'].mean())})
            pivot = primary.pivot(index='dataset', columns='method', values='utility')
            for reference in ['raw', 'legacy_mean', 'legacy_robust', *(['semantic_v1'] if s.get('focus_coverage_corrected') else []), *s['baselines']]:
                diff = pivot['semantic'] - pivot[reference]; stable = []
                for dataset in diff[diff > 1e-9].index:
                    task_type = primary[primary.dataset == dataset].iloc[0]['type']; metric = 'auc' if task_type == 'classification' else 'rmse'
                    seeds = frame[(frame.dataset == dataset) & (frame.downstream == model) & (frame.metric == metric)].pivot(index='seed', columns='method', values='value')
                    sign = 1 if metric == 'auc' else -1
                    stable.append(int((sign * (seeds.semantic - seeds[reference]) > 1e-9).sum()))
                comparisons.append({'cohort': cohort, 'downstream': model, 'method': 'semantic', 'reference': reference,
                    'wins': int((diff > 1e-9).sum()), 'ties': int((abs(diff) <= 1e-9).sum()), 'losses': int((diff < -1e-9).sum()),
                    'winning_tables_with_2_of_3_seeds': sum(v >= 2 for v in stable)})
    pd.DataFrame(comparisons).to_csv(out / 'comparisons.csv', index=False)
    pd.DataFrame(ranks).to_csv(out / 'average_ranks.csv', index=False)
    physical = [read(p) for p in out.glob('units/*/*/search/physical_calls/*.json')]
    assert len(physical) <= s['maximum_new_calls'] and all(v['status'] == 'completed' for v in physical)
    reused = set()
    for name in s['datasets']:
        for seed in s['seeds']:
            unit = unit_path(s, name, seed)
            reused.update(str(unit / v) for v in read(unit / 'reused_artifacts.json'))
    fits = [read(p) for pattern in ['units/*/*/search/inner_models/*/*/fit.json', 'units/*/*/search/final_models/*/fit.json'] for p in out.glob(pattern) if str(p) not in reused]
    repairs = [read(p) for p in out.glob('units/*/*/search/repair_bundles/*.json') if str(p) not in reused]
    native = [read(p) for pattern in ['units/*/*/search/inner_models/*/*/native/*/generation.json', 'units/*/*/search/final_models/*/native/*/generation.json'] for p in out.glob(pattern)]
    history_name = 'semantic_focus_history.json' if s.get('focus_coverage_corrected') else 'semantic_history.json'
    histories = [read(p) for p in out.glob('units/*/*/search/' + history_name)]
    semantic_calls = [read(p) for name in ['semantic', *(['semantic_focus'] if s.get('focus_coverage_corrected') else [])] for p in out.glob('units/*/*/search/methods/' + name + '/calls/*.json')]
    final_fallbacks = [p for pattern in ['units/*/*/search/semantic_final_fallback.json', 'units/*/*/search/focus_final_fallback.json'] for p in out.glob(pattern)]
    assert len(histories) == 24
    costs = {'new_lm_calls': len(physical), 'ceiling': s['maximum_new_calls'],
        'new_predictor_and_linear_fits': sum(v['predictor_fits'] for v in fits),
        'new_conditional_imputer_fits': sum(v.get('fitted_imputers', 0) for v in repairs),
        'native_generation_slots': len(native), 'native_failed': sum(v['status'] != 'passed' for v in native),
        'input_tokens': sum(v.get('input_tokens', 0) for v in physical), 'output_tokens': sum(v.get('output_tokens', 0) for v in physical),
        'new_weight_updates': 0, 'original_test_loads': 0,
        'wall_seconds': time.time() - read(out / 'launch.json')['started_unix']}
    assert costs['native_failed'] == 0
    assert costs['new_predictor_and_linear_fits'] <= 2400 and costs['new_conditional_imputer_fits'] <= 384
    write(out / 'costs.json', costs)
    mechanism = {'actual_semantic_calls': len(semantic_calls), 'evaluated_plans': sum(sum('spec' in v for v in h) for h in histories),
        'executable_nonstop_plans': sum(sum(v['executable'] and 'spec' in v for v in h) for h in histories),
        'repeated_nonempty_plans': sum(len([v for v in h if 'spec' in v]) - len({tuple(v['spec']['ids']) for v in h if 'spec' in v}) for h in histories),
        'median_distinct_executable_plans': float(np.median([len({tuple(v['spec']['ids']) for v in h if 'spec' in v and v['executable']}) for h in histories])),
        'local_keeps': sum(sum(bool(v.get('local_keep')) for v in h) for h in histories),
        'failed_proposals': sum(sum(not v['executable'] for v in h) for h in histories),
        'final_consistency_fallbacks': len(final_fallbacks),
        'mean_input_tokens': float(np.mean([v['input_tokens'] for v in semantic_calls])),
        'reasons_at_limit': sum(len(v['answer']['reason']) == s['reason_max_characters'] for v in semantic_calls),
        'all_within_per_unit_semantic_ceiling': all(len(h) <= 10 for h in histories),
        'focus_coverage_corrected_before_confirmation': bool(s.get('focus_coverage_corrected'))}
    write(out / 'mechanism.json', mechanism)
    gate_comparisons = [r for r in comparisons if r['cohort'] == 'validation' and r['downstream'] == 'xgboost' and r['reference'] in ['raw', 'legacy_robust']]
    passed = all(r['wins'] >= s['minimum_table_wins_for_development'] and r['losses'] <= s['maximum_table_losses_for_development'] and
                 r['winning_tables_with_2_of_3_seeds'] == r['wins'] for r in gate_comparisons)
    findings = {'complete': True, 'development_gate_passed': passed, 'comparisons': comparisons, 'costs': costs,
        'mechanism': mechanism, 'all_selections_frozen_before_confirmation': True,
        'historical_development_domains': True, 'new_independent_test_claim': False, 'new_training_started': False}
    write(out / 'FINAL_SUMMARY.json', findings)
    write(out / 'verification.json', {'passed': True, 'prediction_files_recomputed': count, 'methods': len(methods), 'units': 24,
        'original_test_loads': 0, 'all_selections_frozen_before_confirmation': True, 'new_fits_for_verification': 0})
    lines = ['# 语义规划修复：开发小批结果', '', '固定完整Qwen3-4B + SFT→GRPO第6步权重，原操作库、主/辅助预测器与超参数不变；本轮没有参数训练。',
        '旧4表为诊断复核，新4表（breast_cancer/haberman/auto_mpg/concrete）为预先固定的开发验证；都不是新领域盲测。原E2测试源本轮0读取。',
        f'开发改善门：{"通过" if passed else "未通过"}。强基线的完整比较如下，不以修复组件或诊断表的提升替代开发验证。', '',
        '|组|模型|新方案对照|胜/平/负|获胜表中≥2/3种子改善|', '|---|---|---|---|---:|']
    for r in comparisons:
        lines.append(f'|{r["cohort"]}|{r["downstream"]}|{r["reference"]}|{r["wins"]}/{r["ties"]}/{r["losses"]}|{r["winning_tables_with_2_of_3_seeds"]}|')
    for model in ['xgboost', 'linear']:
        table = []
        for name in s['datasets']:
            typ = next(d['type'] for d in c['datasets'] if d['name'] == name); metric = 'auc' if typ == 'classification' else 'rmse'
            row = [name, '开发验证' if name in s['validation_datasets'] else '诊断复核', metric.upper()]
            for method in methods:
                value = summary[(summary.dataset == name) & (summary.downstream == model) & (summary.method == method) & (summary.metric == metric)].iloc[0]
                row.append(f'{value["mean"]:.7f} ± {value["sd"]:.5f}')
            table.append(row)
        columns = ['dataset', '组', 'metric', *methods]
        pd.DataFrame(table, columns=columns).to_csv(out / f'结果表_{model}.csv', index=False, encoding='utf-8-sig')
        lines += ['', f'## {model}：三种子均值±样本SD', '', '|' + '|'.join(columns) + '|', '|' + '|'.join(['---'] * len(columns)) + '|']
        lines += ['|' + '|'.join(r) + '|' for r in table]
    lines += ['', '## 机制和成本', '', str(mechanism), '', str(costs), '',
        '确认前依据规划覆盖记录纠正局部keep导致全表提前停止；只补原每单元10调用的剩余额度。semantic_v1保留为初始工程检查点，不据确认选择版本。两版本及全部基线均先冻结后评价。',
        '全部失败、停止、保护拒绝和最终回退保留；单表AUC/RMSE差异不作跨量纲平均，未宣称统计显著。该组合修复不是单因素消融。',
        '保护规则来自训练折，推断的整数/精确比例是保守经验规则，不能等同所有真实业务约束。候选数量和理由完整度不代替下游效果。',
        '确认开封后不改本批方法再报同一确认为独立证据；本批无论正负均不自动扩训练/模型/数据。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
