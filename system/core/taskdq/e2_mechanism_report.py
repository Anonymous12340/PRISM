"""Complete developer-pilot tables with saved-model and prediction replay."""
from pathlib import Path
from collections import Counter
import time
import joblib
import numpy as np
import pandas as pd
from .e2_mechanism import matrix, unit_path
from .e2 import apply_final
from .e2_pilot import key
from .scoring import predict, encode
from .e1 import metrics
from .io import read, write


def report(s, c):
    out = Path(s['output_dir']); rows = []; replayed = 0; mechanism = Counter(); decisions = []
    new_fits = 0; imputer_fits = 0; calls = []
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed); source = joblib.load(unit / 'source.joblib'); task = source['task']
        data = joblib.load(Path(s['confirmation_output']) / 'development_confirmation' / name / str(seed) / 'confirmation.joblib')
        selection = read(unit / 'selection_final.json')
        mechanism['same_system_as_code'] += int(selection['plans']['mechanism_code'] == selection['plans']['mechanism_system'])
        mechanism['nonraw_system'] += int(bool(selection['plans']['mechanism_system']['ids']))
        if (unit / 'lm_history.json').exists():
            history = read(unit / 'lm_history.json')
            mechanism['lm_rounds'] += len(history)
            mechanism['lm_failed'] += sum(not v['executable'] for v in history)
            mechanism['lm_new_nonempty_executable'] += sum(v['executable'] and 'spec' in v for v in history)
        mechanism['code_failed'] += read(unit / 'code_complete.json')['failed']
        for method, spec in selection['plans'].items():
            folder = unit / 'final_models' / key(spec); bundle = joblib.load(folder / 'bundle.joblib')
            x = apply_final(c, folder, data['x']); p = predict(bundle['primary'], x); enc = encode(x, bundle['encoder'])
            q = bundle['linear'].predict_proba(enc)[:, 1] if task['type'] == 'classification' else bundle['linear'].predict(enc)
            result_dir = out / 'evaluation' / name / str(seed) / method; record = read(result_dir / 'result.json')
            for model, predicted in [('xgboost', p), ('linear', q)]:
                f = pd.read_csv(result_dir / (model + '_predictions.csv'), float_precision='round_trip')
                assert f.row_id.tolist() == data['x'].index.tolist()
                y = f.y_true.to_numpy(dtype=record['target_dtype']); restored = f.prediction.to_numpy(dtype=record['prediction_dtypes'][model])
                np.testing.assert_array_equal(y, np.asarray(data['y'])); np.testing.assert_array_equal(predicted, restored)
                scores = metrics(y, restored, task)
                for metric, value in scores.items():
                    assert abs(value - record[model][metric]) <= 1e-10
                    rows.append({'dataset': name, 'seed': seed, 'type': task['type'], 'method': method, 'downstream': model,
                        'metric': metric, 'value': value, 'cohort': 'adaptive_development'})
                replayed += 1
            from .e2_parameterized import plan_label
            from .e2_parameterized import ExpandedScorer
            sc = ExpandedScorer({**c, 'seed': seed}, unit)
            decisions.append({'dataset': name, 'seed': seed, 'method': method, 'plan': plan_label(spec, sc.ops),
                'nonraw': bool(spec['ids']), 'lm_enabled': selection['lm_enabled']})
        inherited = set(read(unit / 'reused_artifacts.json')) if (unit / 'reused_artifacts.json').exists() else set()
        for pattern in ['inner_models/*/*/fit.json', 'final_models/*/fit.json']:
            new_fits += sum(read(p)['predictor_fits'] for p in unit.glob(pattern) if str(p.relative_to(unit)) not in inherited)
        imputer_fits += sum(read(p)['fitted_imputers'] for p in unit.glob('repair_bundles/*.json'))
        calls += [read(p) for p in (unit / 'physical_calls').glob('*.json')]
    assert replayed == 48 and len(calls) <= 24 and all(v['status'] == 'completed' for v in calls)
    assert new_fits <= s['maximum_predictor_fits'] and imputer_fits <= s['maximum_imputer_fits']
    old = pd.read_csv(Path(s['reference_output']) / 'seed_metrics.csv', float_precision='round_trip')
    old = old[old.dataset.isin(s['datasets'])].copy()
    frame = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    frame.to_csv(out / 'seed_metrics.csv', index=False)
    summary = frame.groupby(['dataset', 'type', 'method', 'downstream', 'metric'], as_index=False).agg(
        mean=('value', 'mean'), sd=('value', 'std'), seeds=('seed', 'nunique'))
    assert (summary.seeds == 3).all(); summary.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(decisions).to_csv(out / 'decisions.csv', index=False)
    comparisons = []; ranks = []
    for model in ['xgboost', 'linear']:
        primary = summary[(summary.downstream == model) & (((summary.type == 'classification') & (summary.metric == 'auc')) |
                          ((summary.type == 'regression') & (summary.metric == 'rmse')))].copy()
        primary['utility'] = np.where(primary.type == 'classification', primary['mean'], -primary['mean'])
        primary['rank'] = primary.groupby('dataset').utility.rank(ascending=False, method='average')
        for method, g in primary.groupby('method'): ranks.append({'downstream': model, 'method': method, 'average_rank': g['rank'].mean()})
        pivot = primary.pivot(index='dataset', columns='method', values='utility')
        for method in ['mechanism_code', 'mechanism_system']:
            for ref in [m for m in pivot if m != method]:
                delta = pivot[method] - pivot[ref]; stable = 0
                for name in delta[delta > 1e-9].index:
                    metric = 'auc' if name in s['datasets'][:2] else 'rmse'
                    ff = frame[(frame.dataset == name) & (frame.downstream == model) & (frame.metric == metric)].pivot(index='seed', columns='method', values='value')
                    gain = (1 if metric == 'auc' else -1) * (ff[method] - ff[ref])
                    stable += int((gain > 1e-9).sum() >= 2)
                comparisons.append({'downstream': model, 'method': method, 'reference': ref, 'wins': int((delta > 1e-9).sum()),
                    'ties': int((abs(delta) <= 1e-9).sum()), 'losses': int((delta < -1e-9).sum()), 'winning_tables_with_2_of_3_seeds': stable})
    pd.DataFrame(comparisons).to_csv(out / 'comparisons.csv', index=False); pd.DataFrame(ranks).to_csv(out / 'average_ranks.csv', index=False)
    costs = {'new_lm_calls': len(calls), 'new_predictor_and_linear_fits': new_fits, 'new_imputer_fits': imputer_fits,
        'new_native_baseline_fits': 0, 'weight_updates': 0, 'original_test_reads': 0,
        'input_tokens': sum(v['input_tokens'] for v in calls), 'output_tokens': sum(v['output_tokens'] for v in calls),
        'wall_seconds': time.time() - read(out / 'launch.json')['started_unix']}
    gate = [r for r in comparisons if r['downstream'] == 'xgboost' and r['method'] == 'mechanism_system' and r['reference'] in ['raw', 'expanded_system']]
    passed = len(gate) == 2 and all(r['wins'] >= 3 and r['losses'] <= 1 and r['winning_tables_with_2_of_3_seeds'] == r['wins'] for r in gate)
    write(out / 'costs.json', costs); write(out / 'mechanism.json', dict(mechanism))
    write(out / 'verification.json', {'passed': True, 'prediction_files_recomputed_and_model_replayed': 48,
        'maximum_prediction_replay_difference': 0, 'all_selections_frozen_before_confirmation': True, 'original_test_reads': 0})
    write(out / 'FINAL_SUMMARY.json', {'complete': True, 'development_signal_gate_passed': passed,
        'cross_seed_signal': read(out / 'cross_seed_signal.json'), 'comparisons': comparisons, 'costs': costs,
        'mechanism': dict(mechanism), 'adaptive_development_only': True, 'new_training_claim': False})
    display = ['raw', 'expanded_system', 'mechanism_code', 'mechanism_system', 'no_lm_search', 'openfe', 'autofeat']
    lines = ['# 机制候选与类型修复：开发小批结果', '', '## Material Passport', '',
        'academic-research-suite / experiment-agent；四张原来源表×三种子，全部是adaptive development，原确认早已使用。',
        '固定原下游模型/超参数、完整SFT→GRPO权重和原六类库；新增明确语义支持的numeric_parse类型修复，原始raw及旧基线输入不改。',
        '每表固定12个公开字段机制候选；三种子均评价，不按确认挑公式。代码池探索和LM贡献分开；旧NoLM10/强基线缓存复用，新12代码+最多2LM预算不同，不作严格同预算因果结论。',
        '该批是候选引导规划组件验证，不替代正式完整逐列N5/10/20 E2，也不证明SFT/GRPO有效。', '',
        f'事前开发效果门：{"通过" if passed else "未通过"}；LM采样前置门：{read(out / "cross_seed_signal.json")["passed_for_bounded_lm"]}。', '',
        '|模型|方法|对照|胜/平/负|获胜表≥2/3种子|', '|---|---|---|---|---:|']
    lines += [f'|{r["downstream"]}|{r["method"]}|{r["reference"]}|{r["wins"]}/{r["ties"]}/{r["losses"]}|{r["winning_tables_with_2_of_3_seeds"]}|' for r in comparisons]
    for model in ['xgboost', 'linear']:
        table = []
        for name in s['datasets']:
            metric = 'auc' if name in s['datasets'][:2] else 'rmse'; row = [name, metric.upper()]
            for method in display:
                r = summary[(summary.dataset == name) & (summary.downstream == model) & (summary.metric == metric) & (summary.method == method)].iloc[0]
                row.append(f'{r["mean"]:.7f} ± {r["sd"]:.5f}')
            table.append(row)
        columns = ['dataset', 'metric', *display]
        pd.DataFrame(table, columns=columns).to_csv(out / f'结果表_{model}.csv', index=False, encoding='utf-8-sig')
        lines += ['', f'## {model}：三种子均值±SD', '', '|' + '|'.join(columns) + '|', '|' + '|'.join(['---'] * len(columns)) + '|']
        lines += ['|' + '|'.join(row) + '|' for row in table]
    lines += ['', '## 成本和机制', '', str(costs), '', str(dict(mechanism)), '',
        '全表/各种子、失败、回退均保留。三种子重复分数并非独立样本显著性检验，观察到正向也只能称开发证据。原E2测试源0读取。',
        '本轮不自动训练或扩大矩阵；若LM和代码相同，不将代码收益归给模型。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
