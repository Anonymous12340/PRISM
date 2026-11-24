"""Fixed residual-guided feature proposals; only training rows enter ranking."""
from collections import Counter
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from .e2_parameterized import ExpandedScorer, dependencies, plan_label, evaluate
from .e2_library import fit_plan
from .e2_pilot import RAW, key
from .e2_search import paired_stats, select_pool, fit_selected
from .scoring import predict
from .io import read, write

KINDS = {'ratio', 'difference', 'sum', 'product', 'signed_log', 'square',
         'bin5', 'frequency', 'group_mean', 'numeric_parse'}


def residual_correlation(values, residual):
    a = np.asarray(values, dtype=float); b = np.asarray(residual, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 30: return 0.
    a, b = a[keep], b[keep]
    a = np.clip(a, *np.quantile(a, [.01, .99]))
    if a.std() <= 1e-15 or b.std() <= 1e-15: return 0.
    return float(np.corrcoef(a, b)[0, 1])


def diverse_plans(ranked, count=9):
    selected = []; families = Counter(); fields_seen = set()
    for row in sorted(ranked, key=lambda r: (-r['screen_score'], r['plan'])):
        fields = tuple(sorted(row['raw_fields']))
        if row['screen_score'] <= 0 or families[row['kind']] >= 4 or fields in fields_seen: continue
        selected.append(row); families[row['kind']] += 1; fields_seen.add(fields)
        if len(selected) == count: break
    return selected


def reuse(sc, previous, spec):
    """Exact same-source/folds/operations only; old artifacts remain immutable."""
    token = key(spec); source = previous / 'scores' / (token + '.json')
    if not source.exists(): return False
    old_ops = {v['id']: v for v in read(previous / 'menu.json')}
    if (previous / 'derived_ops.json').exists(): old_ops.update(read(previous / 'derived_ops.json'))
    if any(old_ops.get(i) != sc.ops[i] for i in spec['ids']): return False
    assert sc.folds == read(previous / 'inner_splits.json')
    old = joblib.load(previous / 'source.joblib')
    for k in ['x', 'y', 'groups']: assert sc.source[k].equals(old[k])
    reused = read(sc.unit / 'reused_files.json') if (sc.unit / 'reused_files.json').exists() else []
    for kind in ['inner_models', 'final_models']:
        origin = previous / kind / token; dest = sc.unit / kind / token
        if origin.exists() and not dest.exists():
            shutil.copytree(origin, dest)
            reused += [str(p.relative_to(sc.unit)) for p in dest.glob('**/fit.json')]
    (sc.unit / 'scores').mkdir(exist_ok=True)
    shutil.copy2(source, sc.unit / 'scores' / source.name)
    write(sc.unit / 'reused_files.json', sorted(set(reused)))
    return True


def rank_candidates(sc, hints):
    operators = [op for op in sc.menu if op['kind'] in KINDS and op['kind'] != 'numeric_parse']
    for col in hints:
        assert col in sc.x and not pd.api.types.is_numeric_dtype(sc.x[col])
        identifier = sc.register('numeric_parse', column=col)
        operators.append(sc.ops[identifier])
    write(sc.unit / 'derived_ops.json', sc.derived)
    fold_data = []
    for i, fold in enumerate(sc.folds):
        folder = sc.unit / 'inner_models' / key(RAW) / str(i)
        model = joblib.load(folder / 'model.joblib'); fit = read(folder / 'fit.json')
        train = sc.x.iloc[fold['train']]
        assert train.index.tolist() == fit['training_row_ids']
        assert not set(train.index) & set(sc.x.index[fold['validation']])
        residual = np.asarray(sc.y.iloc[fold['train']], dtype=float) - predict(model, train)
        fold_data.append((train, residual))
    ranked = []
    for op in operators:
        correlations = []; errors = []
        for train, residual in fold_data:
            try:
                state, x, mask = fit_plan(train, [op])
                assert mask.all() and x.index.equals(train.index)
                col = op['column'] if op['kind'] == 'numeric_parse' else state[-1]['output']
                correlations.append(residual_correlation(x[col], residual))
            except (ValueError, TypeError, KeyError, FloatingPointError) as exc:
                correlations.append(0.); errors.append(repr(exc))
        values = np.asarray(correlations)
        agreement = max(int((values > 0).sum()), int((values < 0).sum()))
        score = float(np.median(abs(values)) * max(0., 2 * agreement / 3 - 1))
        spec = {'kind': 'plan', 'ids': [op['id']]}
        ranked.append({'spec': spec, 'plan': plan_label(spec, sc.ops), 'kind': op['kind'],
            'raw_fields': sorted({op[k] for k in ['column', 'left', 'right', 'group'] if k in op}),
            'fold_training_residual_correlations': correlations, 'screen_score': score, 'errors': errors})
    write(sc.unit / 'residual_screen.json', {'ranking_scope': 'inner_fold_training_rows_only',
        'fold_training_row_ids': [v.index.tolist() for v, _ in fold_data], 'records': ranked})
    return ranked


def construct_plans(sc, ranked):
    singles = diverse_plans(ranked)
    plans = [{'spec': v['spec'], 'origin': 'residual_single', 'screen_score': v['screen_score']} for v in singles]
    for count in [2, 3, 4]:
        ids = [v['spec']['ids'][0] for v in singles[:count]]
        if len(ids) != count: continue
        try: dependencies(sc.x.columns, [sc.ops[i] for i in ids])
        except ValueError: continue
        plans.append({'spec': {'kind': 'plan', 'ids': ids}, 'origin': 'residual_block', 'screen_score': None})
    # A rare invalid block is replaced deterministically without reading CV.
    known = {key(v['spec']) for v in plans}
    for v in sorted(ranked, key=lambda r: (-r['screen_score'], r['plan'])):
        if len(plans) >= 12: break
        if key(v['spec']) in known: continue
        plans.append({'spec': v['spec'], 'origin': 'fixed_fallback_single', 'screen_score': v['screen_score']})
        known.add(key(v['spec']))
    assert len(plans) == 12 and len({key(v['spec']) for v in plans}) == 12
    return [{**v, 'plan': plan_label(v['spec'], sc.ops)} for v in plans]


def run_unit(s, c, previous, unit, name, seed):
    if (unit / 'complete.json').exists(): return
    unit.mkdir(parents=True, exist_ok=True)
    for filename in ['source.joblib', 'menu.json', 'inner_splits.json', 'split.json']:
        if not (unit / filename).exists(): shutil.copy2(previous / filename, unit / filename)
    if (previous / 'derived_ops.json').exists() and not (unit / 'derived_ops.json').exists():
        shutil.copy2(previous / 'derived_ops.json', unit / 'derived_ops.json')
    sc = ExpandedScorer({**c, 'seed': seed}, unit)
    assert reuse(sc, previous, RAW), 'Raw must be reused, not refitted'
    raw = sc.inner(RAW)
    ranked = rank_candidates(sc, s['numeric_type_hints'].get(name, []))
    plans = construct_plans(sc, ranked); write(unit / 'candidates_frozen.json', plans)
    pool = {key(RAW): raw}; records = []
    for row in plans:
        entry = {**row, 'executable': False}
        try:
            reused = reuse(sc, previous, row['spec'])
            value = evaluate(sc, row['spec'], c['candidate_timeout']); pool[key(row['spec'])] = value
            entry.update(executable=True, reused=reused, scores=value['scores'], paired_vs_raw=paired_stats(value, raw))
        except Exception as exc: entry['error'] = repr(exc)
        records.append(entry)
    _, chosen = select_pool(pool, raw, sc.ops, s)
    write(unit / 'selection.json', {'spec': chosen['spec'], 'raw': raw, 'records': records,
        'before_confirmation': True, 'pool': list(pool.values())})
    reuse(sc, previous, chosen['spec']); fit_selected(c, sc, chosen['spec'])
    write(unit / 'complete.json', {'passed': True, 'failed_candidates': sum(not v['executable'] for v in records),
        'plan': plan_label(chosen['spec'], sc.ops), 'dataset': name, 'seed': seed})
