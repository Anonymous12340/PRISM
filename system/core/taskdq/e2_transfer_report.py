"""Complete same-split tables and model replay for held-out-table development."""
from pathlib import Path
from collections import Counter
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, root_mean_squared_error
from .e2_transfer import matrix, unit_path
from .e2_pilot import key
from .e2_parameterized import ExpandedScorer, plan_label
from .e2 import apply_final
from .scoring import predict, encode
from .io import read, write


def report(s, c):
    out = Path(s['output_dir']); rows = []; decisions = []; counts = Counter(); calls = []; fit_count = 0; imputer_count = 0
    candidate_counts = Counter(); signatures = {}
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed); src = joblib.load(unit / 'source.joblib')
        data = joblib.load(out / 'development_confirmation' / name / str(seed) / 'confirmation.joblib')
        part = read(unit / 'split.json'); sc = ExpandedScorer({**c, 'seed': seed}, unit)
        code = read(unit / 'code_selection.json'); candidate_counts['attempted'] += len(code['records'])
        candidate_counts['failed'] += sum(not v['executable'] for v in code['records'])
        steps = [v['steps'] for v in read(unit / 'candidates.json')]
        if name in signatures: assert steps == signatures[name]
        signatures[name] = steps
        history = read(unit / 'lm_history.json')
        counts['lm_outputs'] += len(history); counts['lm_failed'] += sum(not v['executable'] for v in history)
        counts['lm_new_nonempty_executable'] += sum(v['executable'] and 'spec' in v for v in history)
        registry = read(unit / 'evaluation_registry.json')
        counts['same_system_as_code'] += int(registry['transfer_code'] == registry['transfer_system'])
        replay_cache = {}
        for method, spec in registry.items():
            token = key(spec); folder = unit / 'final_models' / token
            if token not in replay_cache:
                bundle = joblib.load(folder / 'bundle.joblib'); x = apply_final(c, folder, data['x'])
                assert x.index.equals(data['x'].index)
                params = bundle['primary']['model'].get_params()
                assert all(params[k] == v for k, v in c['model'].items())
                assert set(read(folder / 'fit.json')['training_row_ids']) <= set(part['search_row_ids'])
                p = predict(bundle['primary'], x); encoded = encode(x, bundle['encoder'])
                q = bundle['linear'].predict_proba(encoded)[:, 1] if src['task']['type'] == 'classification' else bundle['linear'].predict(encoded)
                replay_cache[token] = (p, q)
            dest = out / 'evaluation' / name / str(seed) / method; result = read(dest / 'result.json')
            for model, predicted in zip(['xgboost', 'linear'], replay_cache[token]):
                saved = pd.read_csv(dest / (model + '_predictions.csv'), float_precision='round_trip')
                assert saved.row_id.tolist() == part['confirmation_row_ids']
                y = saved.y_true.to_numpy(dtype=result['target_dtype']); p = saved.prediction.to_numpy(dtype=result['prediction_dtypes'][model])
                np.testing.assert_array_equal(y, np.asarray(data['y'])); np.testing.assert_array_equal(p, predicted)
                score = ({'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, p >= .5, zero_division=0))}
                         if src['task']['type'] == 'classification' else {'rmse': float(root_mean_squared_error(y, p))})
                for metric, value in score.items():
                    assert abs(value - result[model][metric]) <= 1e-10
                    rows.append({'dataset': name, 'seed': seed, 'type': src['task']['type'], 'method': method,
                        'downstream': model, 'metric': metric, 'value': value, 'cohort': 'held_out_table_development'})
                counts['prediction_files_recomputed_and_model_replayed'] += 1
            decisions.append({'dataset': name, 'seed': seed, 'method': method,
                'plan': plan_label(spec, sc.ops) if spec['kind'] == 'plan' else spec['kind'],
                'nonraw': spec != {'kind': 'plan', 'ids': []}})
        calls += [read(p) for p in (unit / 'physical_calls').glob('*.json')]
        fit_count += sum(read(p)['predictor_fits'] for pattern in ['inner_models/*/*/fit.json', 'final_models/*/fit.json'] for p in unit.glob(pattern))
        imputer_count += sum(read(p)['fitted_imputers'] for p in unit.glob('repair_bundles/*.json'))
        for native in ['openfe', 'autofeat']:
            generation = read(unit / 'final_models' / key({'kind': native}) / 'native' / native / 'generation.json')
            assert generation['status'] == 'passed'; counts['native_fits'] += 1
    assert counts['prediction_files_recomputed_and_model_replayed'] == 120
    assert len(calls) == 24 and all(v['status'] == 'completed' for v in calls)
    assert fit_count <= s['maximum_predictor_fits'] and imputer_count <= s['maximum_imputer_fits']
    assert counts['native_fits'] == s['maximum_native_fits']
    frame = pd.DataFrame(rows); frame.to_csv(out / 'seed_metrics.csv', index=False)
    summary = frame.groupby(['dataset', 'type', 'method', 'downstream', 'metric'], as_index=False).agg(
        mean=('value', 'mean'), sd=('value', 'std'), seeds=('seed', 'nunique'))
    assert (summary.seeds == 3).all(); summary.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(decisions).to_csv(out / 'decisions.csv', index=False)
    comparisons = []; ranks = []
    for model in ['xgboost', 'linear']:
        part = summary[(summary.downstream == model) & (((summary.type == 'classification') & (summary.metric == 'auc')) |
                        ((summary.type == 'regression') & (summary.metric == 'rmse')))].copy()
        part['utility'] = np.where(part.type == 'classification', part['mean'], -part['mean'])
        part['rank'] = part.groupby('dataset').utility.rank(ascending=False, method='average')
        for method, group in part.groupby('method'): ranks.append({'downstream': model, 'method': method, 'average_rank': group['rank'].mean()})
        pivot = part.pivot(index='dataset', columns='method', values='utility')
        for method in ['transfer_code', 'transfer_system']:
            for ref in [m for m in pivot if m != method]:
                delta = pivot[method] - pivot[ref]; stable = 0
                for name in delta[delta > 1e-9].index:
                    typ = part[part.dataset == name].iloc[0]['type']; metric = 'auc' if typ == 'classification' else 'rmse'
                    f = frame[(frame.dataset == name) & (frame.downstream == model) & (frame.metric == metric)].pivot(index='seed', columns='method', values='value')
                    gain = (1 if metric == 'auc' else -1) * (f[method] - f[ref]); stable += int((gain > 1e-9).sum() >= 2)
                comparisons.append({'downstream': model, 'method': method, 'reference': ref,
                    'wins': int((delta > 1e-9).sum()), 'ties': int((abs(delta) <= 1e-9).sum()),
                    'losses': int((delta < -1e-9).sum()), 'winning_tables_with_2_of_3_seeds': stable})
    pd.DataFrame(comparisons).to_csv(out / 'comparisons.csv', index=False); pd.DataFrame(ranks).to_csv(out / 'average_ranks.csv', index=False)
    component = next(v for v in comparisons if v['downstream'] == 'xgboost' and v['method'] == 'transfer_system' and v['reference'] == 'raw')
    extra = next(v for v in comparisons if v['downstream'] == 'xgboost' and v['method'] == 'transfer_system' and v['reference'] == 'transfer_code')
    component_passed = component['wins'] >= 3 and component['losses'] <= 1 and component['winning_tables_with_2_of_3_seeds'] == component['wins']
    extra_passed = extra['wins'] >= 2 and extra['losses'] == 0 and extra['winning_tables_with_2_of_3_seeds'] == extra['wins']
    failed_native_attempts = len(list((out / 'failed_native_attempts').glob('*/native/*/generation.json')))
    assert counts['native_fits'] + failed_native_attempts <= s.get('maximum_native_attempts', s['maximum_native_fits'])
    costs = {'new_lm_calls': len(calls), 'new_predictor_and_linear_fits': fit_count, 'new_imputer_fits': imputer_count,
        'new_native_baseline_fits': counts['native_fits'], 'weight_updates': 0, 'original_test_reads': 0,
        'native_fit_attempts_including_timeouts': counts['native_fits'] + failed_native_attempts,
        'failed_native_attempts_preserved': failed_native_attempts,
        'input_tokens': sum(v['input_tokens'] for v in calls), 'output_tokens': sum(v['output_tokens'] for v in calls),
        'wall_seconds': time.time() - read(out / 'launch.json')['started_unix']}
    write(out / 'costs.json', costs)
    write(out / 'mechanism.json', {**dict(counts), 'candidate_counts': dict(candidate_counts), 'same_templates_across_seeds': True})
    write(out / 'verification.json', {'passed': True, 'prediction_files_recomputed_and_model_replayed': 120,
        'maximum_prediction_replay_difference': 0, 'all_native_fits_passed': True, 'fixed_model_parameters_checked': True,
        'all_selections_frozen_before_confirmation': True, 'original_test_reads': 0})
    write(out / 'FINAL_SUMMARY.json', {'complete': True, 'component_signal_gate_passed': component_passed,
        'lm_extra_signal_gate_passed': extra_passed, 'comparisons': comparisons, 'costs': costs,
        'held_out_table_development_only': True, 'new_blind_test_claim': False, 'new_training_claim': False})
    methods = ['raw', 'transfer_code', 'transfer_system', 'openfe', 'autofeat']
    lines = ['# 通用候选的留出表开发迁移结果', '', '## Material Passport', '',
        'academic-research-suite / experiment-agent；固定Heart Disease/HTRU2/Electrical Grid/Energy Efficiency、三种子、统一模板构造器。',
        '四表未参与最近方法修改，但原20表E2曾评价，因此不是新盲测。当前确认从原development中预先留出，全部模型冻结后才统一评价；原测试源0读取。',
        '原四表手工角色公式没有复制进新表；本轮将原有模板族固定成通用构造器，只补公开字段定义及同量纲组，不含新逐表公式。故不是逐字相同旧12公式的迁移。',
        'HTRU2/456 AutoFeat首次900秒超时，无可恢复生成器，失败目录完整保留；只对该槽允许一次1800秒执行恢复。算法、数据、种子和超参数不变；24成功原生拟合之外另计1超时尝试，不重采LM或重跑其他基线。',
        '固定下游模型/参数和现有完整SFT→GRPO权重；每表12代码候选+每单元2次LM。代码/系统同池对照，OpenFE/AutoFeat在同search重新拟合各一次，不混旧分数。', '',
        f'通用候选正向门：{component_passed}；LM额外收益门：{extra_passed}。', '',
        '|模型|方法|对照|胜/平/负|获胜表≥2/3种子|', '|---|---|---|---|---:|']
    lines += [f'|{v["downstream"]}|{v["method"]}|{v["reference"]}|{v["wins"]}/{v["ties"]}/{v["losses"]}|{v["winning_tables_with_2_of_3_seeds"]}|' for v in comparisons]
    for model in ['xgboost', 'linear']:
        table = []
        for name in s['datasets']:
            typ = next(d['type'] for d in c['datasets'] if d['name'] == name); metric = 'auc' if typ == 'classification' else 'rmse'
            row = [name, metric.upper()]
            for method in methods:
                rr = summary[(summary.dataset == name) & (summary.downstream == model) & (summary.metric == metric) & (summary.method == method)].iloc[0]
                row.append(f'{rr["mean"]:.7f} ± {rr["sd"]:.5f}')
            table.append(row)
        columns = ['dataset', 'metric', *methods]; pd.DataFrame(table, columns=columns).to_csv(out / f'结果表_{model}.csv', index=False, encoding='utf-8-sig')
        lines += ['', f'## {model}：三种子均值±SD', '', '|' + '|'.join(columns) + '|', '|' + '|'.join(['---'] * len(columns)) + '|']
        lines += ['|' + '|'.join(row) + '|' for row in table]
    lines += ['', '## 成本与边界', '', str(costs), '',
        '全表/种子/失败均保留，目标关联不作因果，均值与SD不作显著性检验；多候选探索和历史开发限制已披露，不按确认再改规则。LM与代码相同则不归功于训练。',
        '主/辅模型与分类F1均见完整CSV；不同量纲RMSE不直接平均。本批不自动训练或加表。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
