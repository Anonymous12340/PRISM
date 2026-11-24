"""Execution stages for the bounded grounded development pilot."""
import argparse
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .e2_grounded import prepare_unit, run_unit, adopt
from .e2_parameterized import ExpandedScorer
from .e2_pilot import Engine, RAW, key
from .e2_search import fit_selected
from .e2 import apply_final
from .e1 import metrics
from .scoring import predict, encode
from .io import read, write


def settings(path):
    path = Path(path).resolve()
    s = read(path)
    root = (path.parent / s['project_root']).resolve()
    for field in ['reference_output', 'confirmation_output', 'output_dir']:
        s[field] = str(root / s[field])
    c = read(Path(s['reference_output']) / 'config.json')['runtime']
    assert s['datasets'] == ['bank_marketing', 'blood_transfusion', 'abalone', 'airfoil']
    assert s['seeds'] == [42, 123, 456] and s['calls_per_unit'] == 10 and s['maximum_calls'] == 120
    c = {**c, 'output_dir': s['output_dir'], 'unit_physical_limit': 10, 'lm_budget': 10, 'budgets': [10]}
    return s, c


def matrix(s):
    return [(name, seed) for seed in s['seeds'] for name in s['datasets']]


def unit_path(s, name, seed):
    return Path(s['output_dir']) / 'units' / name / str(seed) / 'discovery'


def prepare(s, c):
    for name, seed in matrix(s):
        prepare_unit(s, {**c, 'seed': seed}, name, seed, unit_path(s, name, seed))
    write(Path(s['output_dir']) / 'preparation.json', {'passed': True, 'units': 12,
        'original_test_reads': 0, 'new_confirmation_predictions': 0})


def final_fit(s, c, name, seed):
    unit = unit_path(s, name, seed)
    if (unit / 'fitted.json').exists():
        return
    adopt(s, c, unit)
    selection = read(unit / 'adoption_result.json')['plans']
    final = unit.parent / 'final'
    final.mkdir(exist_ok=True)
    shutil.copy2(unit / 'full_search.joblib', final / 'source.joblib')
    for name_file in ['menu.json', 'derived_ops.json']:
        shutil.copy2(unit / name_file, final / name_file)
    write(final / 'inner_splits.json', [])
    # RAW was already fit on exactly the full search rows with the same settings.
    old = Path(s['reference_output']) / 'units' / name / str(seed) / 'search' / 'final_models' / key(RAW)
    target = final / 'final_models' / key(RAW)
    if not target.exists():
        shutil.copytree(old, target)
    write(final / 'reused_raw.json', {'fit': str(target / 'fit.json'), 'source': str(old)})
    sc = ExpandedScorer(c, final)
    assert read(target / 'fit.json')['training_row_ids'] == sc.x.index.tolist()
    for method, spec in list(selection.items()):
        try:
            fit_selected(c, sc, spec)
        except ValueError as exc:
            write(final / (method + '_fallback.json'), {'attempted': spec, 'error': repr(exc), 'fallback': RAW})
            selection[method] = RAW
    write(final / 'selection.json', selection)
    write(unit / 'fitted.json', {'passed': True, 'before_confirmation': True})


def freeze(s, c):
    calls = []
    records = []
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed)
        assert read(unit / 'fitted.json')['passed']
        part = read(unit / 'split.json')
        discovery = joblib.load(unit / 'source.joblib')
        adoption = joblib.load(unit / 'adoption.joblib')
        assert set(discovery['x'].index) | set(adoption['x'].index) == set(part['search_row_ids'])
        assert not set(discovery['groups']) & set(adoption['groups'])
        for fold in read(unit / 'inner_splits.json'):
            assert not set(fold['train_row_ids'] + fold['validation_row_ids']) & set(part['adoption_row_ids'] + part['confirmation_row_ids'])
        complete = read(unit / 'complete_corrected.json')
        assert complete['diagnosed_columns'] == sorted(discovery['x'].columns), (name, seed, complete)
        weights = read(unit / 'planner_load.json')
        assert weights['file_to_loaded_tensor_check'] and weights['loaded_adapter_tensors'] == 144
        final = unit.parent / 'final'
        for spec in read(final / 'selection.json').values():
            folder = final / 'final_models' / key(spec)
            assert set(read(folder / 'fit.json')['training_row_ids']) <= set(part['search_row_ids'])
            assert not set(read(folder / 'fit.json')['training_row_ids']) & set(part['confirmation_row_ids'])
        physical = [read(p) for p in (unit / 'physical_calls').glob('*.json')]
        assert len(physical) <= 10 and all(v['status'] in ['completed', 'interrupted'] for v in physical)
        calls.extend(physical)
        records.append({'dataset': name, 'seed': seed, 'diagnosed_fields': len(complete['diagnosed_columns'])})
    assert len(calls) <= 120
    write(Path(s['output_dir']) / 'all_selections_frozen.json', {'passed': True, 'units': 12,
        'physical_calls': len(calls), 'records': records, 'before_confirmation': True, 'original_test_reads': 0})


def confirm(s, c):
    out = Path(s['output_dir'])
    assert read(out / 'all_selections_frozen.json')['passed']
    for name, seed in matrix(s):
        unit = unit_path(s, name, seed)
        final = unit.parent / 'final'
        dest = out / 'evaluation' / name / str(seed)
        if (dest / 'complete.json').exists():
            continue
        assert not (dest / 'access.json').exists(), 'Inspect partial evaluation before recovery'
        write(dest / 'access.json', {'all_models_frozen': True, 'adaptive_development': True})
        data = joblib.load(Path(s['confirmation_output']) / 'development_confirmation' / name / str(seed) / 'confirmation.joblib')
        task = joblib.load(unit / 'source.joblib')['task']
        selection = read(final / 'selection.json')
        cache = {}
        for method, spec in selection.items():
            folder = final / 'final_models' / key(spec)
            if key(spec) not in cache:
                bundle = joblib.load(folder / 'bundle.joblib')
                x = apply_final(c, folder, data['x'])
                p = predict(bundle['primary'], x)
                encoded = encode(x, bundle['encoder'])
                q = bundle['linear'].predict_proba(encoded)[:, 1] if task['type'] == 'classification' else bundle['linear'].predict(encoded)
                cache[key(spec)] = (p, q)
            result = {'dataset': name, 'seed': seed, 'method': method, 'spec': spec, 'type': task['type'],
                'target_dtype': str(np.asarray(data['y']).dtype), 'prediction_dtypes': {}, 'adaptive_development': True}
            folder_out = dest / method
            folder_out.mkdir(parents=True, exist_ok=True)
            for model, p in zip(['xgboost', 'linear'], cache[key(spec)]):
                assert np.isfinite(p).all()
                result[model] = metrics(data['y'], p, task)
                result['prediction_dtypes'][model] = str(p.dtype)
                pd.DataFrame({'row_id': data['x'].index, 'y_true': data['y'], 'prediction': p}).to_csv(folder_out / (model + '_predictions.csv'), index=False)
            write(folder_out / 'result.json', result)
        write(dest / 'complete.json', {'passed': True, 'methods': 2})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['prepare', 'pilot', 'plan', 'fit', 'freeze', 'confirm', 'report'], required=True)
    p.add_argument('--lane', type=int, default=0)
    a = p.parse_args()
    s, c = settings(a.config)
    if a.stage == 'prepare':
        prepare(s, c)
    elif a.stage in ['pilot', 'plan']:
        engine = Engine(c)
        jobs = matrix(s)[a.lane::8]
        if a.stage == 'pilot':
            jobs = jobs[:1]
        for name, seed in jobs:
            run_unit(s, {**c, 'seed': seed}, name, seed, unit_path(s, name, seed), engine, pilot=a.stage == 'pilot')
    elif a.stage == 'fit':
        for name, seed in matrix(s)[a.lane::8]:
            final_fit(s, {**c, 'seed': seed}, name, seed)
    elif a.stage == 'freeze':
        freeze(s, c)
    elif a.stage == 'confirm':
        confirm(s, c)
    else:
        from .e2_grounded_report import report
        report(s, c)


if __name__ == '__main__':
    main()
