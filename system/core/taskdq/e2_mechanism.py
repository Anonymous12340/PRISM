"""Fixed, interpretable candidate supply; source development pilot only."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from .e2_parameterized import ExpandedScorer, choices, plan_label, evaluate
from .e2_search import paired_stats, select_pool, fit_selected
from .e2_pilot import Engine, RAW, key
from .e2_semantic import label
from .e2 import apply_final
from .e1 import metrics
from .profile import profile_table
from .scoring import predict, encode
from .io import read, write, plain


def settings(path):
    path = Path(path).resolve(); s = read(path)
    root = (path.parent / s['project_root']).resolve()
    for field in ['reference_output', 'confirmation_output', 'output_dir']:
        s[field] = str(root / s[field])
    s['semantic_config'] = str(path.parent.parent / s['semantic_config'])
    assert Path(s['semantic_config']).is_file(), 'Bundled semantic definitions missing'
    c = read(Path(s['reference_output']) / 'config.json')['runtime']
    assert s['datasets'] == ['bank_marketing', 'blood_transfusion', 'abalone', 'airfoil']
    assert s['seeds'] == [42, 123, 456] and s['maximum_calls'] == 24
    c = {**c, 'output_dir': s['output_dir'], 'unit_physical_limit': 2, 'lm_budget': 2, 'budgets': [2]}
    return s, c


def matrix(s):
    return [(name, seed) for seed in s['seeds'] for name in s['datasets']]


def unit_path(s, name, seed):
    return Path(s['output_dir']) / 'units' / name / str(seed) / 'search'


def formulas(name, roles):
    """Twelve predetermined plans per source, using public quantity meanings."""
    def unary(kind, column): return {'kind': kind, 'column': column}
    def ratio(left, right): return {'kind': 'ratio', 'left': left, 'right': right}
    def product(left, right): return {'kind': 'product', 'left': left, 'right': right}
    def group(column, by): return {'kind': 'group_mean', 'column': column, 'group': by}
    r = roles
    if name == 'bank_marketing':
        day, amount, age, occupation, outcome = [r[k] for k in ['day', 'amount', 'age', 'group', 'outcome']]
        parse = unary('numeric_parse', day); aggregate = group(amount, occupation)
        frequency = unary('frequency', outcome); log = unary('signed_log', amount); lifecycle = ratio(amount, age)
        values = [
            ('day_numeric', 'Public source defines day of month as a number, not 31 unrelated categories.', [parse]),
            ('occupation_balance', 'Within-occupation balance summarises financial context without target encoding.', [aggregate]),
            ('previous_outcome_frequency', 'Previous-campaign outcome frequency describes source prevalence.', [frequency]),
            ('occupation_context', 'Occupation mean balance and prevalence describe client-group context.', [aggregate, unary('frequency', occupation)]),
            ('balance_per_age', 'Balance relative to client age is a lifecycle proxy, not a physical identity.', [lifecycle]),
            ('day_occupation', 'Day ordering and occupation financial context may be complementary.', [parse, aggregate]),
            ('day_lifecycle', 'Calendar timing and lifecycle balance describe different mechanisms.', [parse, lifecycle]),
            ('day_previous_outcome', 'Calendar timing and previous-campaign context.', [parse, frequency]),
            ('day_outcome_balance', 'Calendar timing and previous-outcome group balance, fitted without y.', [parse, group(amount, outcome)]),
            ('day_log_lifecycle', 'Calendar order and signed monetary compression preserve observed negative balances.', [parse, log, lifecycle]),
            ('day_occupation_outcome', 'Calendar order, occupation finance, previous-campaign prevalence.', [parse, aggregate, frequency]),
            ('day_finance_context', 'Combine ordered calendar with signed balance and two prior client contexts.', [parse, log, aggregate, frequency])]
    elif name == 'blood_transfusion':
        count, history, recency = [r[k] for k in ['count', 'history', 'recency']]
        rate, fraction = ratio(count, history), ratio(recency, history)
        interval = ratio(history, count)
        values = [
            ('donation_rate', 'Donations per month of observed history.', [rate]),
            ('relative_recency', 'Time since last donation relative to observed history.', [fraction]),
            ('rate_and_recency', 'Activity rate and relative recent inactivity.', [rate, fraction]),
            ('log_count', 'Compress a skewed cumulative count while retaining originals.', [unary('signed_log', count)]),
            ('log_recency', 'Compress skewed time since last donation.', [unary('signed_log', recency)]),
            ('log_history', 'Compress skewed donor history duration.', [unary('signed_log', history)]),
            ('count_and_recency', 'Cumulative experience and relative recent inactivity.', [unary('signed_log', count), fraction]),
            ('log_rate', 'Activity rate and its compressed representation.', [rate, unary('signed_log', '@1')]),
            ('log_relative_recency', 'Relative recent inactivity and compressed representation.', [fraction, unary('signed_log', '@1')]),
            ('mean_interval', 'Observed-history months per donation.', [interval]),
            ('recency_interval', 'Relative inactivity and average inter-donation interval.', [fraction, interval, ratio('@1', '@2')]),
            ('rate_recency_interaction', 'Interaction of activity rate with relative inactivity.', [rate, fraction, product('@1', '@2')])]
    elif name == 'abalone':
        length, diameter, height, whole, meat, viscera, shell = [r[k] for k in ['length', 'diameter', 'height', 'whole', 'meat', 'viscera', 'shell']]
        meat_share, shell_share, viscera_share = ratio(meat, whole), ratio(shell, whole), ratio(viscera, whole)
        shape1, shape2 = ratio(diameter, length), ratio(height, length)
        volume = [product(length, diameter), product('@1', height)]
        values = [
            ('meat_share', 'Meat fraction of whole weight.', [meat_share]),
            ('shell_share', 'Dry shell fraction of whole weight.', [shell_share]),
            ('viscera_share', 'Viscera fraction of whole weight.', [viscera_share]),
            ('diameter_ratio', 'Dimensionless shell shape ratio.', [shape1]),
            ('height_ratio', 'Dimensionless shell height relative to length.', [shape2]),
            ('meat_shell_shares', 'Two tissue composition fractions.', [meat_share, shell_share]),
            ('composition', 'Three component shares, retaining all original weights.', [meat_share, shell_share, viscera_share]),
            ('whole_density_proxy', 'Whole weight relative to a shell-volume proxy; not an exact body-density identity.', volume + [ratio(whole, '@2')]),
            ('component_density_proxies', 'Meat and shell weight relative to the same geometric volume proxy.', volume + [ratio(meat, '@2'), ratio(shell, '@2')]),
            ('shell_shape', 'Two dimensionless geometric aspect ratios.', [shape1, shape2]),
            ('log_density_proxy', 'Volume-scaled whole weight with signed-log compression.', volume + [ratio(whole, '@2'), unary('signed_log', '@3')]),
            ('shape_and_composition', 'Physical shape plus tissue composition.', [meat_share, shell_share, shape1, shape2])]
    elif name == 'airfoil':
        frequency, length, thickness, velocity = [r[k] for k in ['frequency', 'length', 'thickness', 'velocity']]
        chord_time = [product(frequency, length), ratio('@1', velocity)]
        boundary_time = [product(frequency, thickness), ratio('@1', velocity)]
        aspect = ratio(thickness, length)
        values = [
            ('chord_strouhal', 'Dimensionless frequency times chord over flow velocity.', chord_time),
            ('boundary_strouhal', 'Dimensionless frequency times boundary thickness over velocity.', boundary_time),
            ('boundary_aspect', 'Dimensionless displacement thickness relative to chord.', [aspect]),
            ('log_chord_strouhal', 'Chord-based frequency scale with logarithmic compression.', chord_time + [unary('signed_log', '@2')]),
            ('log_boundary_strouhal', 'Boundary-based frequency scale with logarithmic compression.', boundary_time + [unary('signed_log', '@2')]),
            ('chord_scale_and_shape', 'Dimensionless frequency and boundary-layer geometry.', chord_time + [aspect]),
            ('boundary_scale_and_shape', 'Boundary-based frequency scale and geometric ratio.', boundary_time + [aspect]),
            ('log_frequency', 'Log frequency helps represent a broad spectral range.', [unary('signed_log', frequency)]),
            ('boundary_log_coordinates', 'Compressed frequency, thickness and flow-speed coordinates.', [unary('signed_log', frequency), unary('signed_log', thickness), unary('signed_log', velocity)]),
            ('chord_log_coordinates', 'Compressed frequency, chord and flow-speed coordinates.', [unary('signed_log', frequency), unary('signed_log', length), unary('signed_log', velocity)]),
            ('coupled_dimensionless_scales', 'Chord-based frequency scale combined with boundary aspect ratio.', chord_time + [aspect, product('@2', '@3')]),
            ('frequency_and_log_shape', 'Chord-based frequency scale and compressed boundary aspect.', chord_time + [aspect, unary('signed_log', '@3')])]
    else:
        raise ValueError('Unregistered source')
    assert len(values) == 12
    return [{'name': name, 'mechanism': why, 'steps': steps} for name, why, steps in values]


def compile_plans(sc, name, s):
    lookup = {key({k: v for k, v in op.items() if k not in ['id', 'family']}): op['id'] for op in sc.menu}
    result = []
    for candidate in formulas(name, s['roles'][name]):
        ids = []
        for step in candidate['steps']:
            operation = dict(step)
            for field in ['column', 'left', 'right', 'group']:
                if field in operation and str(operation[field]).startswith('@'):
                    position = int(operation[field][1:]) - 1
                    if not 0 <= position < len(ids): raise ValueError('Forward reference')
                    operation[field] = '__e2_' + ids[position]
            identifier = lookup.get(key(operation))
            if identifier is None:
                arguments = {k: v for k, v in operation.items() if k != 'kind'}
                identifier = sc.register(operation['kind'], **arguments)
                if operation['kind'] == 'numeric_parse' and identifier not in {o['id'] for o in sc.menu}:
                    assert operation['column'] in s['numeric_type_hints'].get(name, [])
                    sc.menu.append(sc.ops[identifier])
            ids.append(identifier)
        spec = {'kind': 'plan', 'ids': ids}
        result.append({**candidate, 'spec': spec, 'plan': plan_label(spec, sc.ops)})
    write(sc.unit / 'derived_ops.json', sc.derived)
    assert len({key(v['spec']) for v in result}) == 12
    return result


def prepare(s, c):
    out = Path(s['output_dir']); records = []
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed); old = Path(s['reference_output']) / 'units' / name / str(seed) / 'search'
        if not (unit / 'prepared.json').exists():
            unit.mkdir(parents=True, exist_ok=True)
            for filename in ['source.joblib', 'menu.json', 'inner_splits.json', 'split.json']:
                shutil.copy2(old / filename, unit / filename)
            sc = ExpandedScorer({**c, 'seed': seed}, unit)
            candidates = compile_plans(sc, name, s)
            write(unit / 'candidates.json', candidates)
            write(unit / 'prepared.json', {'passed': True, 'fixed_candidates': 12, 'source': str(old),
                'original_test_reads': 0, 'before_confirmation': True})
        records.append(read(unit / 'prepared.json'))
    write(out / 'preparation.json', {'passed': True, 'units': 12, 'records': records})


def reuse(sc, spec, s, name, seed):
    """Copy only exact same-spec/source/fold caches; the original is immutable."""
    old = Path(s['reference_output']) / 'units' / name / str(seed) / 'search'
    token = key(spec); score = old / 'scores' / (token + '.json')
    if not score.exists() or (sc.unit / 'scores' / score.name).exists(): return
    old_ops = {v['id']: v for v in read(old / 'menu.json')}
    old_ops.update(read(old / 'derived_ops.json'))
    if any(i not in old_ops or old_ops[i] != sc.ops[i] for i in spec['ids']): return
    assert sc.source['x'].equals(joblib.load(old / 'source.joblib')['x'])
    assert sc.folds == read(old / 'inner_splits.json')
    inherited = read(sc.unit / 'reused_artifacts.json') if (sc.unit / 'reused_artifacts.json').exists() else []
    for kind in ['inner_models', 'final_models']:
        origin = old / kind / token; target = sc.unit / kind / token
        if origin.exists() and not target.exists():
            shutil.copytree(origin, target)
            inherited.extend(str(p.relative_to(sc.unit)) for p in target.glob('**/fit.json'))
    (sc.unit / 'scores').mkdir(exist_ok=True); shutil.copy2(score, sc.unit / 'scores' / score.name)
    write(sc.unit / 'reused_artifacts.json', sorted(set(inherited)))


def score_unit(s, c, name, seed):
    unit = unit_path(s, name, seed)
    if (unit / 'code_complete.json').exists(): return
    sc = ExpandedScorer(c, unit); candidates = compile_plans(sc, name, s)
    reuse(sc, RAW, s, name, seed); raw = sc.inner(RAW); pool = {key(RAW): raw}; records = []
    for candidate in candidates:
        spec = candidate['spec']; record = {**candidate, 'executable': False}
        try:
            reuse(sc, spec, s, name, seed)
            result = evaluate(sc, spec, c['candidate_timeout'])
            pool[key(spec)] = result
            record.update(executable=True, scores=result['scores'], paired_vs_raw=paired_stats(result, raw))
        except Exception as exc:
            record['error'] = repr(exc)
        records.append(record)
    _, selected = select_pool(pool, raw, sc.ops, s)
    write(unit / 'code_selection.json', {'spec': selected['spec'], 'raw': raw, 'records': records, 'pool': list(pool.values()),
        'selected_before_confirmation': True})
    write(unit / 'code_complete.json', {'passed': True, 'attempts': 12, 'failed': sum(not r['executable'] for r in records)})


def signal(s):
    records = []; tables = {}
    for name in s['datasets']:
        by_plan = defaultdict(list)
        for seed in s['seeds']:
            for r in read(unit_path(s, name, seed) / 'code_selection.json')['records']:
                if r['executable']:
                    stats = r['paired_vs_raw']
                    eligible = stats['positive_folds'] >= 2 and stats['mean_gain'] > stats['se'] + 1e-12
                    by_plan[r['name']].append({'seed': seed, 'eligible': eligible, **stats})
        stable = []
        for plan, rows in by_plan.items():
            assert len(rows) == 3, (name, plan, len(rows))
            row = {'dataset': name, 'candidate': plan, 'eligible_seeds': sum(v['eligible'] for v in rows), 'seeds': rows}
            records.append(row)
            if row['eligible_seeds'] >= 2: stable.append(plan)
        tables[name] = stable
    passed = any(tables[n] for n in s['datasets'][:2]) and any(tables[n] for n in s['datasets'][2:])
    result = {'passed_for_bounded_lm': passed, 'stable_candidates': tables, 'records': records,
        'before_confirmation': True, 'criterion': 'at least one classification and one regression source have a fixed plan eligible in >=2/3 seeds'}
    write(Path(s['output_dir']) / 'cross_seed_signal.json', result)
    return result


INSTRUCTION = """Review real training-CV candidates and propose a task-relevant plan in JSON.
This is a candidate-guided planning pilot. The code proposals are explicitly marked and are not your outputs.
Return diagnosis, reason, operations. Reason: 1-2 concise sentences explaining the mechanism and risk.
You may freely keep raw ([]/keep), retain the validated best (retain_best), or propose one allowed full plan string.
The full six-family library remains available; @1 refers to the numeric feature produced by step 1.
IQR extremes are not known errors. Do not replace/delete observed values without evidence; never invent a score.
Numeric parsing requires a documented numerical meaning and strictly parseable values; numerical category codes alone are not sufficient.
Unit scaling requires evidence of inconsistent units. No missing cells means no imputation is needed.
Even a clean table can benefit from physically or task-interpretable features. Avoid arbitrary sums of unlike quantities.
All displayed scores use search-training folds only; no confirmation data is available.
Choosing an existing candidate is valid and does not retrain it; modify only with a new supported hypothesis.
"""


def request_schema(available):
    return {'type': 'object', 'properties': {
        'diagnosis': {'type': 'string', 'enum': ['none', 'missing', 'outlier_candidate', 'format', 'feature_opportunity', 'uncertain']},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 240},
        'operations': {'type': 'array', 'items': {'type': 'string', 'enum': ['keep', 'retain_best', *available]}, 'maxItems': 1}},
        'required': ['diagnosis', 'reason', 'operations'], 'additionalProperties': False}


def lm_unit(s, c, name, seed, engine, pilot=False):
    unit = unit_path(s, name, seed)
    if (unit / 'lm_complete.json').exists(): return
    sc = ExpandedScorer(c, unit); compile_plans(sc, name, s)
    code = read(unit / 'code_selection.json'); raw = code['raw']
    pool = {key(v['spec']): v for v in code['pool']}; known = set(pool); history = []
    prof = profile_table(sc.x, sc.y, sc.task['type'], top_k=3)
    semantic = read(s['semantic_config'])['field_meanings'][name]
    corr = sc.formal_evidence(True)['training_only_target_spearman']
    for number in range(1, (1 if pilot else 2) + 1):
        _, best = select_pool(pool, raw, sc.ops, s)
        focus = list(sc.x)[(number - 1) % len(sc.x.columns)]
        available = choices(sc, best, known, focus, corr)
        templates = {}
        for op in sc.menu:
            if op['kind'] in templates: continue
            generic = dict(op)
            for field in ['column', 'left', 'right', 'group']:
                if field in generic: generic[field] = field
            templates[op['kind']] = label(generic)
        payload = {'task': sc.task, 'public_field_meanings': semantic, 'operator_templates': templates,
            'profiles': prof['column_profiles'], 'sample_rows': prof['sample_rows'],
            'code_candidate_feedback': [{k: v for k, v in r.items() if k not in ['steps', 'spec']} for r in code['records']],
            'best_plan': plan_label(best['spec'], sc.ops), 'best_scores': best['scores'],
            'raw_scores': raw['scores'], 'own_feedback': history, 'remaining_calls': 3 - number}
        schema = request_schema(available)
        record = engine.generate([{'role': 'system', 'content': INSTRUCTION},
            {'role': 'user', 'content': json.dumps(plain(payload), ensure_ascii=False)}], schema, unit, 'mechanism', number)
        entry = {'round': number, 'executable': False, 'origin': 'language_model'}
        try:
            if not record['format_valid'] or not record['eos_completed']: raise ValueError('Incomplete JSON output')
            entry['answer'] = record['answer']
            chosen = (record['answer']['operations'] or ['keep'])[0]; entry['plan'] = chosen
            if chosen in ['keep', 'retain_best']:
                entry['executable'] = True
            else:
                spec = available[chosen]; assert key(spec) not in known; known.add(key(spec)); entry['spec'] = spec
                result = evaluate(sc, spec, c['candidate_timeout']); pool[key(spec)] = result
                entry.update(executable=True, scores=result['scores'], paired_vs_raw=paired_stats(result, raw))
        except Exception as exc:
            entry['error'] = repr(exc)
        history.append(entry); write(unit / 'lm_history.json', history)
    if not pilot:
        _, best = select_pool(pool, raw, sc.ops, s)
        write(unit / 'system_selection.json', {'spec': best['spec'], 'pool': list(pool.values()), 'before_confirmation': True})
        write(unit / 'lm_complete.json', {'passed': True, 'calls': len(history)})


def fit(s, c, name, seed):
    unit = unit_path(s, name, seed)
    if (unit / 'fitted.json').exists(): return
    sc = ExpandedScorer(c, unit)
    code = read(unit / 'code_selection.json')
    enabled = read(Path(s['output_dir']) / 'cross_seed_signal.json')['passed_for_bounded_lm']
    system = read(unit / 'system_selection.json')['spec'] if enabled else code['spec']
    plans = {'mechanism_code': code['spec'], 'mechanism_system': system}
    reuse(sc, RAW, s, name, seed)
    fit_selected(c, sc, RAW)
    for method, spec in list(plans.items()):
        try: fit_selected(c, sc, spec)
        except ValueError as exc:
            write(unit / (method + '_fallback.json'), {'attempted': spec, 'error': repr(exc), 'fallback': RAW})
            plans[method] = RAW
    write(unit / 'selection_final.json', {'plans': plans, 'lm_enabled': enabled, 'before_confirmation': True})
    write(unit / 'fitted.json', {'passed': True})


def freeze(s, c):
    calls = []
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed); assert read(unit / 'fitted.json')['passed']
        source = joblib.load(unit / 'source.joblib'); part = read(unit / 'split.json')
        assert source['x'].index.tolist() == part['search_row_ids']
        for fold in read(unit / 'inner_splits.json'):
            assert not set(fold['train_row_ids'] + fold['validation_row_ids']) & set(part['confirmation_row_ids'])
        selected = read(unit / 'selection_final.json')
        if selected['lm_enabled']:
            weights = read(unit / 'planner_load.json')
            assert weights['file_to_loaded_tensor_check'] and weights['loaded_adapter_tensors'] == 144
        for spec in selected['plans'].values():
            assert set(read(unit / 'final_models' / key(spec) / 'fit.json')['training_row_ids']) <= set(part['search_row_ids'])
        calls += [read(p) for p in (unit / 'physical_calls').glob('*.json')]
    assert len(calls) <= 24 and all(v['status'] == 'completed' for v in calls)
    write(Path(s['output_dir']) / 'all_selections_frozen.json', {'passed': True, 'units': 12, 'calls': len(calls),
        'before_confirmation': True, 'original_test_reads': 0})


def confirm(s, c):
    out = Path(s['output_dir']); assert read(out / 'all_selections_frozen.json')['passed']
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed); dest = out / 'evaluation' / name / str(seed)
        if (dest / 'complete.json').exists(): continue
        assert not (dest / 'access.json').exists(), 'Inspect partial evaluation before recovery'
        write(dest / 'access.json', {'all_models_frozen': True, 'adaptive_development': True})
        data = joblib.load(Path(s['confirmation_output']) / 'development_confirmation' / name / str(seed) / 'confirmation.joblib')
        task = joblib.load(unit / 'source.joblib')['task']; cache = {}
        for method, spec in read(unit / 'selection_final.json')['plans'].items():
            folder = unit / 'final_models' / key(spec)
            if key(spec) not in cache:
                bundle = joblib.load(folder / 'bundle.joblib'); x = apply_final(c, folder, data['x'])
                p = predict(bundle['primary'], x); encoded = encode(x, bundle['encoder'])
                q = bundle['linear'].predict_proba(encoded)[:, 1] if task['type'] == 'classification' else bundle['linear'].predict(encoded)
                cache[key(spec)] = (p, q)
            record = {'dataset': name, 'seed': seed, 'method': method, 'type': task['type'], 'spec': spec,
                'target_dtype': str(np.asarray(data['y']).dtype), 'prediction_dtypes': {}, 'adaptive_development': True}
            folder_out = dest / method; folder_out.mkdir(parents=True, exist_ok=True)
            for model, p in zip(['xgboost', 'linear'], cache[key(spec)]):
                assert np.isfinite(p).all()
                record[model] = metrics(data['y'], p, task); record['prediction_dtypes'][model] = str(p.dtype)
                pd.DataFrame({'row_id': data['x'].index, 'y_true': data['y'], 'prediction': p}).to_csv(folder_out / (model + '_predictions.csv'), index=False)
            write(folder_out / 'result.json', record)
        write(dest / 'complete.json', {'passed': True, 'methods': 2})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True)
    p.add_argument('--stage', required=True, choices=['prepare', 'score', 'signal', 'pilot', 'plan', 'fit', 'freeze', 'confirm', 'report'])
    p.add_argument('--lane', type=int, default=0); a = p.parse_args(); s, c = settings(a.config)
    if a.stage == 'prepare': prepare(s, c)
    elif a.stage == 'signal': signal(s)
    elif a.stage in ['score', 'fit']:
        for name, seed in matrix(s)[a.lane::8]:
            (score_unit if a.stage == 'score' else fit)(s, {**c, 'seed': seed}, name, seed)
    elif a.stage in ['pilot', 'plan']:
        engine = Engine(c); jobs = matrix(s)[a.lane::8]
        for name, seed in (jobs[:1] if a.stage == 'pilot' else jobs):
            lm_unit(s, {**c, 'seed': seed}, name, seed, engine, pilot=a.stage == 'pilot')
    elif a.stage == 'freeze': freeze(s, c)
    elif a.stage == 'confirm': confirm(s, c)
    else:
        from .e2_mechanism_report import report
        report(s, c)


if __name__ == '__main__': main()
