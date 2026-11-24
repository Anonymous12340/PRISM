"""Bounded development repair: named plans, focused evidence and fold-local checks."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import signal
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .e2 import FormalScorer, apply_final
from .e2_pilot import Engine, RAW, key, run_method, timeout_handler
from .e2_search import paired_stats, select_pool, fit_selected, run_unit as legacy_run
from .e2_library import menu, apply_step
from .e2_operations import normalize
from .io import read, write, plain
from .scoring import splits, predict, encode
from .e1 import metrics


def settings(path):
    path = Path(path).resolve(); s = read(path)
    root = (path.parent / s['project_root']).resolve(); s['project_root'] = str(root)
    for name in ['source_config', 'source_output', 'previous_output', 'output_dir', 'adapter_path', 'adapter_dependency_dir']:
        s[name] = str(root / s[name])
    c = read(s['source_config'])
    expected = [n for typ in ['classification', 'regression'] for n in [d['name'] for d in sorted(c['datasets'], key=lambda d: d['name']) if d['type'] == typ][:4]]
    assert s['datasets'] == expected and s['seeds'] == [42, 123, 456]
    assert s['maximum_new_calls'] == 360 and s['calls_per_unit'] == 10
    assert s['consecutive_stall_stop'] == 3
    c = {**c, 'output_dir': s['output_dir'], 'adapter_path': s['adapter_path'],
         'adapter_dependency_dir': s['adapter_dependency_dir'], 'weight_variant': 'sft_grpo',
         'weight_revision': 'frozen_SFT_then_GRPO_final_step6', 'strict_replay_existing_calls': True,
         'unit_physical_limit': 20, 'lm_budget': 10, 'budgets': [10],
         'native_timeout': s['native_timeout_seconds'],
         'native_timeouts': {'openfe': s['native_timeout_seconds'], 'autofeat': s['native_timeout_seconds']},
         'native_jobs': {'autofeat': 5}}
    return s, c


def unit_path(s, name, seed):
    return Path(s['output_dir']) / 'units' / name / str(seed) / 'search'


def prepare(s, c):
    out = Path(s['output_dir']); records = []
    for name in s['datasets']:
        for seed in s['seeds']:
            unit = unit_path(s, name, seed)
            if (unit / 'split.json').exists():
                records.append(read(unit / 'split.json')); continue
            previous = Path(s['previous_output']) / 'units' / name / str(seed) / 'search'
            unit.mkdir(parents=True, exist_ok=True)
            if name in s['diagnostic_datasets']:
                for filename in ['source.joblib', 'split.json', 'inner_splits.json', 'menu.json']:
                    shutil.copy2(previous / filename, unit / filename)
                for folder in ['scores', 'inner_models', 'repair_bundles', 'final_models']:
                    if (previous / folder).exists(): shutil.copytree(previous / folder, unit / folder)
                shutil.copy2(previous / 'selection.json', unit / 'legacy_selection.json')
                destination = out / 'development_confirmation' / name / str(seed)
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(s['previous_output']) / 'development_confirmation' / name / str(seed) / 'confirmation.joblib', destination / 'confirmation.joblib')
            else:
                source = joblib.load(Path(s['source_output']) / 'units' / name / str(seed) / 'final/source.joblib')
                x, y, g = source['x'], source['y'], source['groups']; task = source['task']
                cls = StratifiedGroupKFold if task['type'] == 'classification' else GroupKFold
                a, b = next(cls(n_splits=5, shuffle=True, random_state=seed).split(x, y, g))
                assert not set(g.iloc[a]) & set(g.iloc[b])
                search = deepcopy(source)
                for field in ['x', 'y', 'groups']: search[field] = source[field].iloc[a]
                joblib.dump(search, unit / 'source.joblib')
                destination = out / 'development_confirmation' / name / str(seed)
                destination.mkdir(parents=True, exist_ok=True)
                joblib.dump({field: source[field].iloc[b] for field in ['x', 'y', 'groups']}, destination / 'confirmation.joblib')
                inner = splits(search['x'], search['y'], task, 3, seed, search['groups'])
                write(unit / 'inner_splits.json', [{'train': aa.tolist(), 'validation': bb.tolist(),
                    'train_row_ids': search['x'].index[aa].tolist(), 'validation_row_ids': search['x'].index[bb].tolist()} for aa, bb in inner])
                write(unit / 'menu.json', menu(search['x'], search.get('date_columns'), c['menu_limit']))
                original = read(Path(c['data_dir']) / 'prepared' / name / str(seed) / 'split.json')
                assert set(x.index) == set(original['development_row_ids'])
                assert not set(x.index) & set(original['test_row_ids'])
                write(unit / 'split.json', {'dataset': name, 'seed': seed, 'search_rows': len(a), 'confirmation_rows': len(b),
                    'search_row_ids': search['x'].index.tolist(), 'confirmation_row_ids': source['x'].index[b].tolist(),
                    'groups_disjoint': True, 'original_test_rows_used': 0})
            inherited = sorted(str(p.relative_to(unit)) for pattern in ['scores/*.json', 'inner_models/*/*/fit.json', 'final_models/*/fit.json', 'repair_bundles/*.json'] for p in unit.glob(pattern))
            write(unit / 'reused_artifacts.json', inherited)
            part = read(unit / 'split.json'); source = joblib.load(unit / 'source.joblib')
            assert source['x'].index.tolist() == part['search_row_ids']
            assert not set(part['search_row_ids']) & set(part['confirmation_row_ids'])
            records.append(part)
    assert len(records) == 24
    write(out / 'preparation.json', {'passed': True, 'units': 24, 'original_test_loads': 0, 'confirmation_not_used_in_search': True})


def label(operation):
    kind = operation['kind']
    fields = [str(operation[k]) for k in ['column', 'left', 'right'] if k in operation]
    fields += [k + '=' + str(operation[k]) for k in ['group', 'method', 'factor', 'format', 'threshold', 'scale', 'offset'] if k in operation]
    return kind + '(' + ','.join(fields) + ')'


def dependencies(columns, operations):
    available = set(columns)
    for operation in operations:
        required = {operation[k] for k in ['column', 'left', 'right', 'group'] if k in operation}
        missing = required - available
        if missing: raise ValueError('Plan reads removed/absent columns: ' + ','.join(sorted(missing)))
        if operation['kind'] in ['onehot', 'drop', 'date_parts']: available.discard(operation['column'])


def integer_columns(frame):
    result = []
    for column in frame.select_dtypes(include=np.number):
        values = frame[column].dropna().to_numpy(float)
        if len(values) >= 30 and np.unique(values).size >= 3 and np.all(abs(values - np.round(values)) <= 1e-9): result.append(column)
    return result


def exact_relations(frame):
    columns = list(frame.select_dtypes(include=np.number)); result = []
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            a, b = frame[left], frame[right]; valid = a.notna() & b.notna()
            ratio_rows = valid & (abs(b) > 1e-9)
            if ratio_rows.sum() < 30 or a.loc[ratio_rows].nunique() < 3: continue
            factor = float((a.loc[ratio_rows] / b.loc[ratio_rows]).median())
            if not np.isfinite(factor) or abs(factor) <= 1e-9: continue
            if np.isclose(a.loc[valid], factor * b.loc[valid], rtol=1e-9, atol=1e-9).all():
                result.append({'left': left, 'right': right, 'factor': factor, 'training_support': int(valid.sum())})
    return result


def consistent_change(train_before, train_after, valid_before, valid_after, changed_column):
    # No labels; all inferred representation/relationship rules come from train.
    rules = exact_relations(train_before); integer = changed_column in integer_columns(train_before)
    summary = {'column': changed_column, 'integer_like_in_training': integer, 'relations': rules, 'new_violations': []}
    for role, before, after in [('train', train_before, train_after), ('validation', valid_before, valid_after)]:
        common = before.index.intersection(after.index)
        if integer:
            old_values = before.loc[common, changed_column].to_numpy(float)
            values = after.loc[common, changed_column].to_numpy(float)
            previously_ok = np.isnan(old_values) | (abs(old_values - np.round(old_values)) <= 1e-9)
            count = int((previously_ok & np.isfinite(values) & (abs(values - np.round(values)) > 1e-9)).sum())
            if count: summary['new_violations'].append({'part': role, 'rule': 'integer_representation', 'count': count})
        for rule in rules:
            left, right, factor = rule['left'], rule['right'], rule['factor']
            if changed_column not in [left, right] or left not in after or right not in after: continue
            old_ok = np.isclose(before.loc[common, left], factor * before.loc[common, right], rtol=1e-9, atol=1e-9)
            new_ok = np.isclose(after.loc[common, left], factor * after.loc[common, right], rtol=1e-9, atol=1e-9)
            count = int((old_ok & ~new_ok).sum())
            if count: summary['new_violations'].append({'part': role, 'rule': rule, 'count': count})
    if summary['new_violations']: raise ValueError('Candidate creates consistency violations: ' + str(plain(summary)))
    return summary


def check_steps(train, valid, steps):
    a, b = normalize(train), normalize(valid); checks = []
    for operation in steps:
        after_a, after_b = apply_step(a, operation, training=True), apply_step(b, operation)
        if operation['kind'] in ['model_repair', 'clip']:
            checks.append(consistent_change(a, after_a, b, after_b, operation['column']))
        a, b = after_a, after_b
    return checks


class CheckedScorer(FormalScorer):
    def transform(self, spec, train, valid, y, groups, slot, where):
        if spec['kind'] == 'plan': dependencies(train.columns, [self.ops[i] for i in spec['ids']])
        result = super().transform(spec, train, valid, y, groups, slot, where)
        if spec['kind'] == 'plan':
            checks = check_steps(train, valid, joblib.load(where / 'transform_state.joblib')['steps'])
            write(where / 'semantic_check.json', {'passed': True, 'checks': checks, 'rules_from_training_only': True})
        return result

    def inner(self, spec):
        if spec['kind'] == 'plan':
            dependencies(self.x.columns, [self.ops[i] for i in spec['ids']])
            # A copied score may predate the checks; check saved transforms first.
            if (self.unit / 'scores' / (key(spec) + '.json')).exists():
                for i, fold in enumerate(self.folds):
                    folder = self.unit / 'inner_models' / key(spec) / str(i)
                    marker = folder / 'semantic_check.json'
                    if not marker.exists():
                        state = joblib.load(folder / 'transform_state.joblib')['steps']
                        checks = check_steps(self.x.iloc[fold['train']], self.x.iloc[fold['validation']], state)
                        write(marker, {'passed': True, 'checks': checks, 'rules_from_training_only': True})
        return super().inner(spec)


def usable(operation, frame):
    kind, column = operation['kind'], operation.get('column')
    if kind in ['fill', 'drop_missing', 'model_repair'] and not frame[column].isna().any(): return False
    if kind in ['clip', 'flag', 'drop_outlier']:
        values = frame[column]; q1, q3 = values.quantile([.25, .75])
        outlier = ((values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))).any()
        if not outlier and (kind != 'flag' or not values.isna().any()): return False
    return True


def options(sc, best, known):
    result = {}; prefixes = [[], best['spec']['ids']] if best['spec']['ids'] else [[]]
    for prefix in prefixes:
        if len(prefix) >= 4: continue
        for operation in sc.menu:
            if operation['id'] in prefix or not usable(operation, sc.x): continue
            spec = {'kind': 'plan', 'ids': prefix + [operation['id']]}
            if key(spec) in known: continue
            try: dependencies(sc.x.columns, [sc.ops[i] for i in spec['ids']])
            except ValueError: continue
            name = ' -> '.join(label(sc.ops[i]) for i in spec['ids'])
            assert name not in result or result[name] == spec
            result[name] = spec
    return result


def compact_context(sc, best, raw, history, round_number, s):
    full = sc.formal_evidence(True); columns = full['columns']; correlations = full.get('training_only_target_spearman', {})
    ordered = sorted(columns, key=lambda v: (-v['missing_count'] / len(sc.x), -abs(correlations.get(v['name'], 0.)), str(v['name'])))
    focus = ordered[(round_number - 1) % len(ordered)]['name']
    related = []
    if pd.api.types.is_numeric_dtype(sc.x[focus]):
        for c in sc.x.select_dtypes(include=np.number):
            if c == focus: continue
            r = sc.x[focus].corr(sc.x[c], method='spearman')
            if np.isfinite(r): related.append((abs(r), c))
    detailed = [focus] + [v[1] for v in sorted(related, reverse=True)[:2]]
    summary = [{'column': v['name'], 'type': v['kind'], 'missing': v['missing_count'],
                'target_rho': round(correlations.get(v['name'], 0.), 4)} for v in columns]
    catalog = [label(v) for v in sc.menu if usable(v, sc.x)]
    return {'input': {'background': sc.source['pair']['background'][:s['background_characters']], 'task': sc.task,
            'focus_column': focus, 'focus_and_related_profiles': [v for v in columns if v['name'] in detailed],
            'other_columns_minimal_summary': summary, 'training_exact_proportions': exact_relations(sc.x),
            'training_integer_like_columns': integer_columns(sc.x), 'available_named_operations': catalog},
            'own_training_feedback': history,
            'raw_inner_scores': raw['scores'], 'best_plan': ' -> '.join(label(sc.ops[i]) for i in best['spec']['ids']) or 'keep',
            'best_inner_scores': best['scores'], 'remaining_calls': 11 - round_number}


INSTRUCTION = '''你依据任务、字段含义及训练画像规划表格处理，代码执行。输出JSON diagnosis/reason/operations。
diagnosis只能为missing/outlier_candidate/none/uncertain；理由用中文1–2个完整短句，说明本次处理的字段、假设及风险，通常不超过80字。
operations最多一个字符串：一个具名操作，或“已知best计划 -> 一个新操作”的完整序列，总操作最多4个。只能使用目录列出的精确具名操作；keep或空数组表示停止搜索并保留已知最佳候选。不要输出O编号。
焦点列提供细节，但可选择目录中的其他有任务依据的字段或跨字段操作；不要求每轮都改焦点列。原值的尾部可能是重要信号，IQR不代表真错。无缺失且没有验证的语义错误时不自动模型替换；优先考虑保留原值的有据标记、比值、交互和分组特征，避免无意义的单位缩放。模型能处理缺失和常规编码，不必重复做。
参考完整反馈，提出新的可执行假设；失败和已评价完整序列不能再选，但它们的组成操作可用于其他新序列。先分组再独热，避免后面读取已删除的列。修复不得破坏已成立的计数表示和数值关系。若无合理新假设则keep，不为凑数乱改。所有反馈仅是search三折；不可编造确认分数。'''


def semantic_run(s, c, name, seed, engine):
    unit = unit_path(s, name, seed); cc = {**c, 'seed': seed}; sc = CheckedScorer(cc, unit)
    if (unit / 'semantic_complete.json').exists(): return read(unit / 'semantic_complete.json')
    raw = sc.inner(RAW); best = raw; pool = {key(RAW): raw}; known = {key(RAW)}; history = []
    for number in range(1, s['calls_per_unit'] + 1):
        choices = options(sc, best, known)
        if not choices: break
        context = compact_context(sc, best, raw, history, number, s)
        schema = {'type': 'object', 'properties': {'diagnosis': {'type': 'string', 'enum': ['missing', 'outlier_candidate', 'none', 'uncertain']},
                  'reason': {'type': 'string', 'minLength': 1, 'maxLength': s['reason_max_characters']},
                  'operations': {'type': 'array', 'items': {'type': 'string', 'enum': ['keep', *choices]}, 'maxItems': 1}},
                  'required': ['diagnosis', 'reason', 'operations'], 'additionalProperties': False}
        import json
        messages = [{'role': 'system', 'content': INSTRUCTION}, {'role': 'user', 'content': json.dumps(plain(context), ensure_ascii=False)}]
        record = engine.generate(messages, schema, unit, 'semantic', number)
        entry = {'round': number, 'format_valid': record['format_valid'], 'executable': False,
                 'input_tokens': record.get('input_tokens'), 'diagnosis': record.get('answer', {}).get('diagnosis')}
        try:
            if not record['format_valid']: raise ValueError('Invalid JSON output')
            chosen = (record['answer']['operations'] or ['keep'])[0]; entry['plan'] = chosen
            if chosen == 'keep':
                entry.update(executable=True, stop=True); history.append(entry); break
            spec = choices[chosen]; token = key(spec); assert token not in known; known.add(token); entry['spec'] = spec
            old = signal.signal(signal.SIGALRM, timeout_handler); signal.alarm(c['candidate_timeout'])
            try: result = sc.inner(spec)
            finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)
            pool[token] = result; entry.update(executable=True, paired_vs_raw=paired_stats(result, raw), scores=result['scores'])
            if result['utility'] > best['utility'] + 1e-12: best = result
        except Exception as exc: entry['error'] = repr(exc)
        history.append(entry); write(unit / 'semantic_history.json', history)
        print({'dataset': name, 'seed': seed, 'round': number, 'semantic': entry}, flush=True)
    write(unit / 'semantic_history.json', history)
    mean, robust = select_pool(pool, raw, sc.ops, s)
    write(unit / 'semantic_selection.json', {'spec': robust['spec'], 'mean_spec_descriptive': mean['spec'],
          'candidate_results': list(pool.values()), 'selection_frozen_before_confirmation': True})
    result = {'dataset': name, 'seed': seed, 'passed': True, 'calls': len(history),
              'distinct_executable_plans': len(pool) - 1, 'failed': sum(not v['executable'] for v in history),
              'stopped': bool(history and history[-1].get('stop')), 'confirmation_opened': False}
    write(unit / 'semantic_complete.json', result); return result


def planning_unit(s, c, name, seed, engine, pilot=False):
    unit = unit_path(s, name, seed)
    if not (unit / 'legacy_selection.json').exists():
        legacy_run(s, c, name, seed, engine)
        (unit / 'selection.json').rename(unit / 'legacy_selection.json')
    if pilot:
        # Pilot is the same first request, not a disposable sampling experiment.
        limited = {**s, 'calls_per_unit': 1}
        semantic_run(limited, c, name, seed, engine)
        # Resume with the frozen ten-round budget by replaying this exact request.
        (unit / 'semantic_complete.json').rename(unit / 'semantic_pilot.json')
        (unit / 'semantic_selection.json').rename(unit / 'semantic_pilot_selection.json')
    else:
        semantic_run(s, c, name, seed, engine)


def fit_unit(s, c, name, seed):
    unit = unit_path(s, name, seed); cc = {**c, 'seed': seed}; sc = FormalScorer(cc, unit)
    if (unit / 'fitted.json').exists(): return read(unit / 'fitted.json')
    legacy = read(unit / 'legacy_selection.json')['plans']
    plans = {'raw': RAW, 'legacy_mean': legacy['history_mean'], 'legacy_robust': legacy['history_robust'],
             'semantic': read(unit / 'semantic_selection.json')['spec']}
    semantic = plans['semantic']
    try:
        checked = CheckedScorer(cc, unit); fit_selected(cc, checked, semantic)
        folder = unit / 'final_models' / key(semantic)
        checks = check_steps(sc.x, sc.x.iloc[:2], joblib.load(folder / 'transform_state.joblib')['steps'])
        write(folder / 'semantic_final_check.json', {'passed': True, 'checks': checks})
    except ValueError as exc:
        write(unit / 'semantic_final_fallback.json', {'error': repr(exc), 'attempted': semantic,
              'fallback': RAW, 'before_confirmation': True})
        plans['semantic'] = RAW
    for method in s['baselines']:
        selected = run_method(cc, sc, None, method); plans[method] = selected['spec']
    for spec in {key(v): v for v in plans.values()}.values():
        fit_selected(cc, sc, spec)
    write(unit / 'selection.json', {'plans': plans, 'confirmation_opened': False})
    write(unit / 'fitted.json', {'passed': True, 'methods': len(plans), 'confirmation_opened': False})
    return read(unit / 'fitted.json')


def freeze(s, c):
    records = []
    for name in s['datasets']:
        for seed in s['seeds']:
            unit = unit_path(s, name, seed); assert read(unit / 'fitted.json')['passed']
            part = read(unit / 'split.json'); source = joblib.load(unit / 'source.joblib')
            assert set(source['x'].index) == set(part['search_row_ids'])
            assert not set(part['search_row_ids']) & set(part['confirmation_row_ids'])
            for f in read(unit / 'inner_splits.json'):
                assert not set(f['train_row_ids'] + f['validation_row_ids']) & set(part['confirmation_row_ids'])
            receipt = read(unit / 'planner_load.json'); assert receipt['file_to_loaded_tensor_check'] and receipt['loaded_adapter_tensors'] == 144
            for spec in read(unit / 'selection.json')['plans'].values():
                folder = unit / 'final_models' / key(spec); fit = read(folder / 'fit.json')
                assert set(fit['training_row_ids']) <= set(part['search_row_ids'])
                x = source['x'].iloc[:4]; transformed = apply_final(c, folder, x)
                assert x.index.equals(transformed.index)
                assert np.isfinite(predict(joblib.load(folder / 'bundle.joblib')['primary'], transformed)).all()
            records.append(read(unit / 'semantic_complete.json'))
    calls = [read(p) for p in Path(s['output_dir']).glob('units/*/*/search/physical_calls/*.json')]
    assert len(records) == 24 and len(calls) <= s['maximum_new_calls'] and all(r['status'] == 'completed' for r in calls)
    write(Path(s['output_dir']) / 'all_selections_frozen.json', {'passed': True, 'units': 24, 'methods': 8,
          'confirmation_loaded': False, 'records': records, 'actual_calls': len(calls)})


def confirm(s, c):
    out = Path(s['output_dir']); assert read(out / 'all_selections_frozen.json')['passed']
    for name in s['datasets']:
        task = next(d for d in c['datasets'] if d['name'] == name)
        for seed in s['seeds']:
            unit = unit_path(s, name, seed); dest = out / 'confirmation_results' / name / str(seed)
            if (dest / 'complete.json').exists(): continue
            if (dest / 'access.json').exists(): raise ValueError('Partial confirmation access; inspect before recovery')
            write(dest / 'access.json', {'all_methods_frozen': True, 'original_test_opened': False})
            frame = joblib.load(out / 'development_confirmation' / name / str(seed) / 'confirmation.joblib'); cache = {}
            for method, spec in read(unit / 'selection.json')['plans'].items():
                token = key(spec); folder = unit / 'final_models' / token
                if token not in cache:
                    bundle = joblib.load(folder / 'bundle.joblib'); x = apply_final(c, folder, frame['x'])
                    assert x.index.equals(frame['x'].index)
                    primary = predict(bundle['primary'], x); matrix = encode(x, bundle['encoder'])
                    auxiliary = bundle['linear'].predict_proba(matrix)[:, 1] if task['type'] == 'classification' else bundle['linear'].predict(matrix)
                    cache[token] = (primary, auxiliary)
                result = {'dataset': name, 'seed': seed, 'method': method, 'type': task['type'], 'spec': spec,
                          'target_dtype': str(np.asarray(frame['y']).dtype), 'prediction_dtypes': {}}
                for model, p in zip(['xgboost', 'linear'], cache[token]):
                    assert np.isfinite(p).all(); result[model] = metrics(frame['y'], p, task); result['prediction_dtypes'][model] = str(p.dtype)
                    folder_result = dest / method; folder_result.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({'row_id': frame['x'].index, 'y_true': frame['y'], 'prediction': p}).to_csv(folder_result / (model + '_predictions.csv'), index=False)
                write(dest / method / 'result.json', result)
            write(dest / 'complete.json', {'passed': True, 'methods': len(read(unit / 'selection.json')['plans']), 'original_test_loads': 0, 'new_fits': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True, choices=['prepare', 'pilot', 'plan', 'fit', 'freeze', 'confirm', 'report'])
    parser.add_argument('--lane', type=int, default=0); args = parser.parse_args(); s, c = settings(args.config)
    matrix = [(n, z) for n in s['datasets'] for z in s['seeds']]
    if args.stage == 'prepare': prepare(s, c)
    elif args.stage in ['pilot', 'plan']:
        engine = Engine(c)
        selected = matrix[args.lane::8]
        if args.stage == 'pilot': selected = selected[:1]
        for name, seed in selected: planning_unit(s, c, name, seed, engine, pilot=args.stage == 'pilot')
    elif args.stage == 'fit':
        for name, seed in matrix[args.lane::8]: print(fit_unit(s, c, name, seed), flush=True)
    elif args.stage == 'freeze': freeze(s, c)
    elif args.stage == 'confirm': confirm(s, c)
    else:
        from .e2_semantic_report import report
        report(s, c)


if __name__ == '__main__': main()
