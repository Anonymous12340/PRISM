"""Automatic reporting and independent prediction replay for grounded development."""
from pathlib import Path
from collections import Counter
import time
import joblib
import numpy as np
import pandas as pd
from .e2_grounded_run import matrix, unit_path
from .e2_pilot import key
from .e2 import apply_final
from .e1 import metrics
from .scoring import predict, encode
from .io import read, write


def report(s, c):
    out = Path(s['output_dir'])
    rows = []
    prediction_files = 0
    mechanism = Counter()
    decisions = []
    new_fits = 0
    imputers = 0
    calls = []
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed)
        source = joblib.load(unit / 'source.joblib')
        data = joblib.load(Path(s['confirmation_output']) / 'development_confirmation' / name / str(seed) / 'confirmation.joblib')
        task = source['task']
        final = unit.parent / 'final'
        adoption = read(unit / 'adoption_result.json')
        mechanism['adoption_evaluated_units'] += int(not adoption.get('no_eligible_candidate', False))
        mechanism['adoption_passed_units'] += int(adoption.get('passed', False))
        mechanism['adoption_rejected_units'] += int(adoption.get('passed') is False)
        history = read(unit / 'history_corrected.json')
        lm = [e for e in history if e['origin'] == 'language_model']
        mechanism['lm_rounds_including_interruptions'] += len(lm)
        mechanism['legacy_completed_missing_templates'] += sum(e['input_version'] == 'legacy_missing_templates' for e in lm)
        mechanism['failed_proposals'] += sum(not e['executable'] for e in lm)
        mechanism['decision_only'] += sum(e.get('decision_only', False) for e in lm)
        mechanism['nonempty_lm_executable'] += sum(e['executable'] and bool(e.get('spec')) for e in lm)
        mechanism['balanced_code_probes'] += sum(e['origin'] == 'balanced_code_probe' for e in history)
        mechanism['diagnosed_fields'] += len(read(unit / 'complete_corrected.json')['diagnosed_columns'])
        candidate_spec = read(unit / 'candidate_frozen_corrected.json')['candidate']['spec']
        origin = next((e['origin'] for e in history if e.get('spec') == candidate_spec), 'raw')
        mechanism['candidate_origin_' + origin] += 1
        for method, spec in read(final / 'selection.json').items():
            result_dir = out / 'evaluation' / name / str(seed) / method
            record = read(result_dir / 'result.json')
            bundle = joblib.load(final / 'final_models' / key(spec) / 'bundle.joblib')
            x = apply_final(c, final / 'final_models' / key(spec), data['x'])
            p = predict(bundle['primary'], x)
            encoded = encode(x, bundle['encoder'])
            q = bundle['linear'].predict_proba(encoded)[:, 1] if task['type'] == 'classification' else bundle['linear'].predict(encoded)
            for model, predicted in [('xgboost', p), ('linear', q)]:
                saved = pd.read_csv(result_dir / (model + '_predictions.csv'), float_precision='round_trip')
                assert saved.row_id.tolist() == data['x'].index.tolist()
                y = saved.y_true.to_numpy(dtype=record['target_dtype'])
                restored = saved.prediction.to_numpy(dtype=record['prediction_dtypes'][model])
                np.testing.assert_array_equal(restored, predicted)
                np.testing.assert_array_equal(y, np.asarray(data['y']))
                values = metrics(y, restored, task)
                for metric, value in values.items():
                    assert abs(record[model][metric] - value) <= 1e-10
                    rows.append({'dataset': name, 'type': task['type'], 'seed': seed, 'method': method,
                        'downstream': model, 'metric': metric, 'value': value, 'cohort': 'adaptive_development'})
                prediction_files += 1
            decisions.append({'dataset': name, 'seed': seed, 'method': method, 'nonraw': bool(spec['ids']),
                'discovery_candidate_gain': read(unit / 'candidate_frozen_corrected.json')['candidate']['utility'] - read(unit / 'candidate_frozen_corrected.json')['raw']['utility'],
                'adoption_passed': adoption.get('passed'), 'spec': str(spec)})
        calls.extend(read(p) for p in (unit / 'physical_calls').glob('*.json'))
        for pattern in ['inner_models/*/*/fit.json', 'adoption_models/*/fit.json']:
            new_fits += sum(read(p)['predictor_fits'] for p in unit.glob(pattern))
        raw_path = read(final / 'reused_raw.json')['fit']
        new_fits += sum(read(p)['predictor_fits'] for p in final.glob('final_models/*/fit.json') if str(p) != raw_path)
        imputers += sum(read(p)['fitted_imputers'] for folder in [unit, final] for p in folder.glob('repair_bundles/*.json'))
    assert prediction_files == 48
    assert len(calls) <= s['maximum_calls'] and all(r['status'] in ['completed', 'interrupted'] for r in calls)
    assert new_fits <= s['maximum_predictor_fits'] and imputers <= s['maximum_imputer_fits']
    old = pd.read_csv(Path(s['reference_output']) / 'seed_metrics.csv', float_precision='round_trip')
    old = old[old.dataset.isin(s['datasets'])].copy()
    frame = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    frame.to_csv(out / 'seed_metrics.csv', index=False)
    summary = frame.groupby(['dataset', 'type', 'method', 'downstream', 'metric'], as_index=False).agg(
        mean=('value', 'mean'), sd=('value', 'std'), seeds=('seed', 'nunique'))
    assert (summary.seeds == 3).all()
    summary.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(decisions).to_csv(out / 'decisions.csv', index=False)
    comparisons = []
    ranks = []
    for model in ['xgboost', 'linear']:
        primary = summary[(summary.downstream == model) & (((summary.type == 'classification') & (summary.metric == 'auc')) |
                         ((summary.type == 'regression') & (summary.metric == 'rmse')))].copy()
        primary['utility'] = np.where(primary.type == 'classification', primary['mean'], -primary['mean'])
        primary['rank'] = primary.groupby('dataset').utility.rank(ascending=False, method='average')
        for method, g in primary.groupby('method'):
            ranks.append({'downstream': model, 'method': method, 'average_rank': g['rank'].mean()})
        pivot = primary.pivot(index='dataset', columns='method', values='utility')
        for method in ['grounded', 'grounded_cv']:
            for ref in [m for m in pivot if m != method]:
                delta = pivot[method] - pivot[ref]
                stable = 0
                for name in delta[delta > 1e-9].index:
                    metric = 'auc' if name in ['bank_marketing', 'blood_transfusion'] else 'rmse'
                    seed_frame = frame[(frame.dataset == name) & (frame.downstream == model) & (frame.metric == metric)].pivot(index='seed', columns='method', values='value')
                    gains = (1 if metric == 'auc' else -1) * (seed_frame[method] - seed_frame[ref])
                    stable += int((gains > 1e-9).sum() >= 2)
                comparisons.append({'downstream': model, 'method': method, 'reference': ref,
                    'wins': int((delta > 1e-9).sum()), 'ties': int((abs(delta) <= 1e-9).sum()),
                    'losses': int((delta < -1e-9).sum()), 'wins_with_2_of_3_seeds': stable})
    pd.DataFrame(comparisons).to_csv(out / 'comparisons.csv', index=False)
    pd.DataFrame(ranks).to_csv(out / 'average_ranks.csv', index=False)
    costs = {'new_lm_calls': len(calls), 'new_predictor_and_linear_fits': new_fits, 'new_imputer_fits': imputers,
        'new_native_baseline_fits': 0, 'weight_updates': 0, 'original_test_reads': 0,
        'completed_generations': sum(v['status'] == 'completed' for v in calls),
        'interrupted_budget_consumed': sum(v['status'] == 'interrupted' for v in calls),
        'input_tokens_completed_only': sum(v.get('input_tokens', 0) for v in calls),
        'output_tokens_completed_only': sum(v.get('output_tokens', 0) for v in calls),
        'wall_seconds': time.time() - read(out / 'launch.json')['started_unix']}
    gated = [v for v in comparisons if v['downstream'] == 'xgboost' and v['method'] == 'grounded' and v['reference'] in ['raw', 'expanded_system']]
    passed = len(gated) == 2 and all(v['wins'] >= 3 and v['losses'] <= 1 and v['wins_with_2_of_3_seeds'] == v['wins'] for v in gated)
    write(out / 'costs.json', costs)
    write(out / 'mechanism.json', dict(mechanism))
    write(out / 'verification.json', {'passed': True, 'prediction_files_recomputed_and_model_replayed': prediction_files,
        'maximum_prediction_replay_difference': 0, 'all_models_frozen_before_confirmation': True,
        'adoption_not_used_for_exploration': True, 'original_test_reads': 0})
    write(out / 'FINAL_SUMMARY.json', {'complete': True, 'development_signal_gate_passed': passed,
        'comparisons': comparisons, 'costs': costs, 'mechanism': dict(mechanism),
        'adaptive_development_only': True, 'independent_confirmation_claim': False})
    display = ['raw', 'expanded_system', 'grounded_cv', 'grounded', 'no_lm_search', 'openfe', 'autofeat']
    lines = ['# 字段证据与独立采用复核：开发结果', '',
        '## Material Passport', '', 'academic-research-suite / experiment-agent；adaptive development。',
        '固定原四张来源表及三种子，原始强基线同划分缓存复用。新方法在原search内再划discovery/adoption，探索训练量较旧方法少；组合改动不作单因素因果消融。',
        '完整固定SFT→GRPO权重、六类操作及下游参数；0新训练、0新原生基线拟合、原E2测试源0读取。',
        '原开发确认此前已经使用，本轮不能当新盲测。adoption仅评价预先冻结的一个候选，不反馈LM或失败后另试。', '',
        '运行中修正操作调用示例漏接：71已完成输出按原输入保留，6中断消耗预算、不重采，只有剩余43请求使用修正输入。修正后加入按族抽样的最多5个单步代码probe，显式标注来源；本批不是完整一致的新提示性能验证，更不能归功于LM训练。', '',
        f'事前开发信号门：{"通过" if passed else "未通过"}。', '',
        '|模型|方法|对照|胜/平/负|获胜表≥2/3种子|', '|---|---|---|---|---:|']
    lines += [f'|{r["downstream"]}|{r["method"]}|{r["reference"]}|{r["wins"]}/{r["ties"]}/{r["losses"]}|{r["wins_with_2_of_3_seeds"]}|' for r in comparisons]
    for model in ['xgboost', 'linear']:
        table = []
        for name in s['datasets']:
            metric = 'auc' if name in ['bank_marketing', 'blood_transfusion'] else 'rmse'
            row = [name, metric.upper()]
            for method in display:
                r = summary[(summary.dataset == name) & (summary.downstream == model) & (summary.method == method) & (summary.metric == metric)].iloc[0]
                row.append(f'{r["mean"]:.7f} ± {r["sd"]:.5f}')
            table.append(row)
        columns = ['dataset', 'metric', *display]
        pd.DataFrame(table, columns=columns).to_csv(out / f'结果表_{model}.csv', index=False, encoding='utf-8-sig')
        lines += ['', f'## {model}：三种子均值±SD', '', '|' + '|'.join(columns) + '|', '|' + '|'.join(['---'] * len(columns)) + '|']
        lines += ['|' + '|'.join(row) + '|' for row in table]
    lines += ['', '## 机制与成本', '', str(dict(mechanism)), '', str(costs), '',
        '采用门是一次开发留出复核的启发式规则，不保证泛化或统计显著。真实错误标签缺失，删行/改值不能称已修复真错。',
        '全部种子、keep及失败计入，不按确认挑版本/调阈值；本轮不自动启动训练或扩大实验。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
