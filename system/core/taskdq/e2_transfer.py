"""Frozen template-family candidate construction and held-out-table development."""
import argparse
from collections import deque
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .e2_parameterized import registry, ExpandedScorer, choices, plan_label, evaluate
from .e2_search import paired_stats, select_pool, fit_selected
from .e2_pilot import Engine, RAW, key
from .e2_semantic import label
from .e2_mechanism import INSTRUCTION, request_schema
from .e2 import apply_final
from .e1 import metrics
from .profile import profile_table
from .scoring import splits, predict, encode
from .io import read, write, plain


FAMILY_ORDER = ['numeric_parse', 'same_quantity_ratio', 'group_mean', 'frequency',
                'signed_log', 'square', 'product', 'log_ratio']


def interleave(groups):
    queues = [deque(values) for values in groups if values]
    while queues:
        next_round = []
        for queue in queues:
            yield queue.popleft()
            if queue: next_round.append(queue)
        queues = next_round


def generic_candidates(frame, metadata, limit=12, allow_short=False):
    """No dataset identity, target, scores, or source-specific formulas are inputs."""
    numeric = sorted(frame.select_dtypes(include=np.number).columns)
    categorical = sorted(set(frame) - set(numeric))
    assert set(frame) == set(metadata['columns'])
    hints = metadata['numeric_type_hints']
    assert set(hints) <= set(categorical)
    groups = []
    for group in metadata['same_quantity_groups']:
        assert set(group) <= set(numeric)
        names = sorted(group)
        groups.append([{'kind': 'ratio', 'left': left, 'right': right}
                       for left in names for right in names if left != right])
    ratios = list(interleave(groups))
    unary = lambda kind, col: {'kind': kind, 'column': col}
    families = {
        'numeric_parse': [[unary('numeric_parse', col)] for col in sorted(hints)],
        'same_quantity_ratio': [[op] for op in ratios],
        'group_mean': [[{'kind': 'group_mean', 'group': group, 'column': col}]
                       for group in categorical for col in numeric],
        'frequency': [[unary('frequency', col)] for col in categorical],
        'signed_log': [[unary('signed_log', col)] for col in numeric],
        'square': [[unary('square', col)] for col in numeric],
        'product': [[{'kind': 'product', 'left': left, 'right': right}] for left, right in combinations(numeric, 2)],
        'log_ratio': [[op, unary('signed_log', '@1')] for op in ratios]}
    selected = []; known = set()
    for position in range(max([len(v) for v in families.values()] or [0])):
        for family in FAMILY_ORDER:
            if position >= len(families[family]): continue
            steps = families[family][position]; token = key(steps)
            if token in known: continue
            known.add(token)
            selected.append({'name': family + '_' + str(position + 1), 'family': family,
                'mechanism': 'Frozen template family ' + family + '; quantity and category definitions from public metadata.',
                'steps': steps})
            if len(selected) == limit: return selected
    if allow_short:return selected
    raise ValueError('Fewer than the fixed candidate count are applicable')


def compile_candidates(sc, metadata):
    candidates = generic_candidates(sc.x, metadata)
    lookup = {key({k: v for k, v in op.items() if k not in ['id', 'family']}): op['id'] for op in sc.menu}
    result = []
    for candidate in candidates:
        ids = []
        for step in candidate['steps']:
            op = dict(step)
            for field in ['column', 'left', 'right', 'group']:
                if field in op and str(op[field]).startswith('@'):
                    position = int(op[field][1:]) - 1
                    assert 0 <= position < len(ids)
                    op[field] = '__e2_' + ids[position]
            identifier = lookup.get(key(op))
            if identifier is None:
                identifier = sc.register(op['kind'], **{k: v for k, v in op.items() if k != 'kind'})
                if op['kind'] == 'numeric_parse' and identifier not in {o['id'] for o in sc.menu}:
                    sc.menu.append(sc.ops[identifier])
            ids.append(identifier)
        spec = {'kind': 'plan', 'ids': ids}
        result.append({**candidate, 'spec': spec, 'plan': plan_label(spec, sc.ops)})
    write(sc.unit / 'derived_ops.json', sc.derived)
    return result


def settings(path):
    path = Path(path).resolve(); s = read(path); root = (path.parent / s['project_root']).resolve()
    for field in ['source_output', 'runtime_reference', 'output_dir']: s[field] = str(root / s[field])
    assert s['candidate_family_order'] == FAMILY_ORDER
    assert s['seeds'] == [42, 123, 456] and s['maximum_calls'] == 24
    registry_data = read(Path(s['source_output']) / 'effective_final_config.json')
    expected = []
    for kind in ['classification', 'regression']:
        remaining = sorted(d['name'] for d in registry_data['datasets']
                           if d['type'] == kind and d['name'] not in s['excluded_development_tables'])
        expected += remaining[:2]
    assert s['datasets'] == expected
    c = read(Path(s['runtime_reference']) / 'config.json')['runtime']
    c = {**c, 'output_dir': s['output_dir'], 'unit_physical_limit': 2, 'lm_budget': 2, 'budgets': [2],
        'native_timeout': s['native_timeout_seconds'],
        'native_timeouts': {'openfe': s['native_timeout_seconds'], 'autofeat': s['native_timeout_seconds']},
        'native_jobs': {'autofeat': 5}}
    return s, c


def matrix(s): return [(name, seed) for seed in s['seeds'] for name in s['datasets']]
def unit_path(s, name, seed): return Path(s['output_dir']) / 'units' / name / str(seed) / 'search'


def prepare(s, c):
    out = Path(s['output_dir']); records = []
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed)
        if not (unit / 'prepared.json').exists():
            source = joblib.load(Path(s['source_output']) / 'units' / name / str(seed) / 'final/source.joblib')
            x, y, groups, task = source['x'], source['y'], source['groups'], source['task']
            a, b = splits(x, y, task, 5, seed, groups)[0]
            assert not set(groups.iloc[a]) & set(groups.iloc[b])
            search = deepcopy(source)
            for field in ['x', 'y', 'groups']: search[field] = source[field].iloc[a]
            unit.mkdir(parents=True, exist_ok=True); joblib.dump(search, unit / 'source.joblib')
            destination = out / 'development_confirmation' / name / str(seed); destination.mkdir(parents=True, exist_ok=True)
            joblib.dump({field: source[field].iloc[b] for field in ['x', 'y', 'groups']}, destination / 'confirmation.joblib')
            write(unit / 'menu.json', registry(search['x'], search.get('date_columns')))
            inner = splits(search['x'], search['y'], task, 3, seed, search['groups'])
            write(unit / 'inner_splits.json', [{'train': aa.tolist(), 'validation': bb.tolist(),
                'train_row_ids': search['x'].index[aa].tolist(), 'validation_row_ids': search['x'].index[bb].tolist()} for aa, bb in inner])
            assert task['target'] not in search['x']
            write(unit / 'split.json', {'dataset': name, 'seed': seed, 'search_row_ids': search['x'].index.tolist(),
                'confirmation_row_ids': x.index[b].tolist(), 'development_source_only': True, 'group_disjoint': True})
            sc = ExpandedScorer({**c, 'seed': seed}, unit)
            candidates = compile_candidates(sc, s['field_metadata'][name])
            write(unit / 'candidates.json', candidates)
            write(unit / 'prepared.json', {'passed': True, 'candidate_count': len(candidates), 'search_rows': len(a),
                'confirmation_rows': len(b), 'original_test_reads': 0})
        records.append(read(unit / 'prepared.json'))
    write(out / 'preparation.json', {'passed': True, 'units': 12, 'records': records, 'candidate_rules_frozen': FAMILY_ORDER})


def score_unit(s, c, name, seed):
    unit = unit_path(s, name, seed)
    if (unit / 'code_complete.json').exists(): return
    sc = ExpandedScorer(c, unit); candidates = compile_candidates(sc, s['field_metadata'][name])
    assert candidates == read(unit / 'candidates.json')
    raw = sc.inner(RAW); pool = {key(RAW): raw}; rows = []
    for candidate in candidates:
        row = {**candidate, 'executable': False}
        try:
            value = evaluate(sc, candidate['spec'], c['candidate_timeout']); pool[key(candidate['spec'])] = value
            row.update(executable=True, scores=value['scores'], paired_vs_raw=paired_stats(value, raw))
        except Exception as exc: row['error'] = repr(exc)
        rows.append(row)
    _, chosen = select_pool(pool, raw, sc.ops, s)
    write(unit / 'code_selection.json', {'spec': chosen['spec'], 'raw': raw, 'pool': list(pool.values()), 'records': rows})
    write(unit / 'code_complete.json', {'passed': True, 'failed': sum(not r['executable'] for r in rows)})


def lm_unit(s, c, name, seed, engine, pilot=False):
    unit = unit_path(s, name, seed)
    if (unit / 'lm_complete.json').exists(): return
    sc = ExpandedScorer(c, unit); compile_candidates(sc, s['field_metadata'][name])
    code = read(unit / 'code_selection.json'); raw = code['raw']; pool = {key(v['spec']): v for v in code['pool']}
    known = set(pool); history = []; profile = profile_table(sc.x, sc.y, sc.task['type'], top_k=3)
    correlations = sc.formal_evidence(True)['training_only_target_spearman']
    for number in range(1, (1 if pilot else 2) + 1):
        _, best = select_pool(pool, raw, sc.ops, s)
        available = choices(sc, best, known, list(sc.x)[number - 1], correlations)
        templates = {}
        for op in sc.menu:
            if op['kind'] in templates: continue
            generic = dict(op)
            for field in ['column', 'left', 'right', 'group']:
                if field in generic: generic[field] = field
            templates[op['kind']] = label(generic)
        payload = {'task': sc.task, 'public_field_meanings': s['field_metadata'][name],
            'profiles': profile['column_profiles'], 'sample_rows': profile['sample_rows'], 'operator_templates': templates,
            'code_candidate_feedback': [{k: v for k, v in r.items() if k not in ['steps', 'spec']} for r in code['records']],
            'raw_scores': raw['scores'], 'best_plan': plan_label(best['spec'], sc.ops), 'best_scores': best['scores'],
            'own_feedback': history, 'remaining_calls': 3 - number}
        generated = engine.generate([{'role': 'system', 'content': INSTRUCTION},
            {'role': 'user', 'content': json.dumps(plain(payload), ensure_ascii=False)}],
            request_schema(available), unit, 'transfer', number)
        row = {'round': number, 'origin': 'language_model', 'executable': False}
        try:
            if not generated['format_valid'] or not generated['eos_completed']: raise ValueError('Incomplete JSON output')
            answer = generated['answer']; row['answer'] = answer; plan = (answer['operations'] or ['keep'])[0]; row['plan'] = plan
            if plan in ['keep', 'retain_best']: row['executable'] = True
            else:
                spec = available[plan]; assert key(spec) not in known; known.add(key(spec)); row['spec'] = spec
                columns = {sc.ops[i]['column'] for i in spec['ids'] if sc.ops[i]['kind'] == 'model_repair'}
                current = len(list((unit / 'repair_bundles').glob('*.json')))
                needed = sum(not (unit / 'repair_bundles' / (str(i) + '_' + key(col) + '.joblib')).exists()
                             for col in columns for i in range(3))
                # At most two repair columns per unit: six fold bundles and
                # two final bundles. Preserve rejected proposals, never exceed 96.
                if current + needed > 6:
                    raise ValueError('Fixed per-unit imputer budget exceeded; proposal retained without fitting')
                value = evaluate(sc, spec, c['candidate_timeout']); pool[key(spec)] = value
                row.update(executable=True, scores=value['scores'], paired_vs_raw=paired_stats(value, raw))
        except Exception as exc: row['error'] = repr(exc)
        history.append(row); write(unit / 'lm_history.json', history)
    if not pilot:
        _, chosen = select_pool(pool, raw, sc.ops, s)
        write(unit / 'system_selection.json', {'spec': chosen['spec'], 'pool': list(pool.values())})
        write(unit / 'lm_complete.json', {'passed': True, 'calls': len(history)})


def fit_unit(s, c, name, seed):
    unit = unit_path(s, name, seed)
    if (unit / 'fitted.json').exists(): return
    sc = ExpandedScorer(c, unit)
    plans = {'raw': RAW, 'transfer_code': read(unit / 'code_selection.json')['spec'],
             'transfer_system': read(unit / 'system_selection.json')['spec']}
    for method, spec in list(plans.items()):
        try: fit_selected(c, sc, spec)
        except ValueError as exc:
            if method == 'raw': raise
            write(unit / (method + '_fallback.json'), {'error': repr(exc), 'attempted': spec, 'fallback': RAW})
            plans[method] = RAW
    write(unit / 'selection_final.json', {'plans': plans, 'before_confirmation': True})
    write(unit / 'fitted.json', {'passed': True})


def fit_native(s, c, name, seed, method):
    unit = unit_path(s, name, seed); marker = unit / ('native_' + method + '.json')
    if marker.exists():
        assert read(marker)['passed']; return
    override = s.get('native_execution_overrides', {}).get(name + '/' + str(seed) + '/' + method)
    if override:
        c = {**c, 'native_timeouts': {**c['native_timeouts'], method: override}}
    sc = ExpandedScorer(c, unit); spec = {'kind': method}
    fit_selected(c, sc, spec)
    generation = read(unit / 'final_models' / key(spec) / 'native' / method / 'generation.json')
    assert generation['status'] == 'passed'
    assert generation['input_training_row_ids'] == sc.x.index.tolist()
    write(marker, {'passed': True, 'spec': spec, 'before_confirmation': True,
        'execution_timeout_seconds': c['native_timeouts'][method]})


def freeze(s, c):
    calls = []
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed); assert read(unit / 'fitted.json')['passed']
        source = joblib.load(unit / 'source.joblib'); part = read(unit / 'split.json')
        assert source['x'].index.tolist() == part['search_row_ids']
        for fold in read(unit / 'inner_splits.json'):
            assert not set(fold['train_row_ids'] + fold['validation_row_ids']) & set(part['confirmation_row_ids'])
        plans = read(unit / 'selection_final.json')['plans']
        for method in ['openfe', 'autofeat']:
            native = read(unit / ('native_' + method + '.json')); assert native['passed']; plans[method] = native['spec']
        for spec in plans.values():
            fit = read(unit / 'final_models' / key(spec) / 'fit.json')
            assert set(fit['training_row_ids']) <= set(part['search_row_ids'])
        weights = read(unit / 'planner_load.json'); assert weights['file_to_loaded_tensor_check'] and weights['loaded_adapter_tensors'] == 144
        physical = [read(p) for p in (unit / 'physical_calls').glob('*.json')]
        assert len(physical) == 2 and all(v['status'] == 'completed' for v in physical)
        calls += physical
        write(unit / 'evaluation_registry.json', plans)
    assert len(calls) == 24
    write(Path(s['output_dir']) / 'all_selections_frozen.json', {'passed': True, 'units': 12,
        'methods': 5, 'all_native_methods_passed': True, 'before_confirmation': True, 'original_test_reads': 0})


def confirm(s, c):
    out = Path(s['output_dir']); assert read(out / 'all_selections_frozen.json')['passed']
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed); dest = out / 'evaluation' / name / str(seed)
        if (dest / 'complete.json').exists(): continue
        assert not (dest / 'access.json').exists(), 'Inspect partial evaluation before recovery'
        write(dest / 'access.json', {'all_models_frozen': True, 'held_out_table_development': True})
        data = joblib.load(out / 'development_confirmation' / name / str(seed) / 'confirmation.joblib')
        task = joblib.load(unit / 'source.joblib')['task']; cache = {}
        for method, spec in read(unit / 'evaluation_registry.json').items():
            folder = unit / 'final_models' / key(spec)
            if key(spec) not in cache:
                bundle = joblib.load(folder / 'bundle.joblib'); x = apply_final(c, folder, data['x'])
                assert x.index.equals(data['x'].index)
                p = predict(bundle['primary'], x); encoded = encode(x, bundle['encoder'])
                q = bundle['linear'].predict_proba(encoded)[:, 1] if task['type'] == 'classification' else bundle['linear'].predict(encoded)
                cache[key(spec)] = (p, q)
            r = {'dataset': name, 'seed': seed, 'type': task['type'], 'method': method, 'spec': spec,
                 'target_dtype': str(np.asarray(data['y']).dtype), 'prediction_dtypes': {}, 'held_out_table_development': True}
            target = dest / method; target.mkdir(parents=True, exist_ok=True)
            for model, p in zip(['xgboost', 'linear'], cache[key(spec)]):
                assert np.isfinite(p).all(); r[model] = metrics(data['y'], p, task); r['prediction_dtypes'][model] = str(p.dtype)
                pd.DataFrame({'row_id': data['x'].index, 'y_true': data['y'], 'prediction': p}).to_csv(target / (model + '_predictions.csv'), index=False)
            write(target / 'result.json', r)
        write(dest / 'complete.json', {'passed': True, 'methods': 5})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True)
    p.add_argument('--stage', required=True, choices=['prepare', 'score', 'pilot', 'plan', 'fit', 'native', 'freeze', 'confirm', 'report'])
    p.add_argument('--lane', type=int, default=0); p.add_argument('--dataset'); p.add_argument('--seed', type=int)
    p.add_argument('--method', choices=['openfe', 'autofeat']); a = p.parse_args(); s, c = settings(a.config)
    if a.stage == 'prepare': prepare(s, c)
    elif a.stage in ['score', 'fit']:
        for name, seed in matrix(s)[a.lane::8]:
            (score_unit if a.stage == 'score' else fit_unit)(s, {**c, 'seed': seed}, name, seed)
    elif a.stage in ['pilot', 'plan']:
        engine = Engine(c); jobs = matrix(s)[a.lane::8]
        for name, seed in (jobs[:1] if a.stage == 'pilot' else jobs):
            lm_unit(s, {**c, 'seed': seed}, name, seed, engine, pilot=a.stage == 'pilot')
    elif a.stage == 'native': fit_native(s, {**c, 'seed': a.seed}, a.dataset, a.seed, a.method)
    elif a.stage == 'freeze': freeze(s, c)
    elif a.stage == 'confirm': confirm(s, c)
    else:
        from .e2_transfer_report import report
        report(s, c)


if __name__ == '__main__': main()
