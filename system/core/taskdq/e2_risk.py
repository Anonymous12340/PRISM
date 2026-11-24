"""Training-fold uncertainty filter for frozen candidate predictions.

This conditional bootstrap is a development selection aid, not a guarantee
against degradation on new data. Candidate discovery and overlapping model
training introduce uncertainty beyond resampling saved validation predictions.
"""
import numpy as np


def weighted_engine(y, predictions, classification):
    y, predictions = np.asarray(y), np.asarray(predictions, dtype=float)
    if not classification:
        squared = (y.astype(float)[:, None] - predictions) ** 2
        return lambda weights: np.sqrt((weights @ squared) / weights.sum(axis=1)[:, None])
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('AUC requires both binary target classes in each fold')
    rankings = []
    for prediction in predictions.T:
        order = np.argsort(prediction, kind='stable')
        starts = np.r_[0, np.flatnonzero(np.diff(prediction[order])) + 1]
        rankings.append((order, starts, y[order] == 1))

    def auc(weights):
        columns = []
        for order, starts, positive in rankings:
            weight = weights[:, order]
            pos = np.add.reduceat(weight * positive, starts, axis=1)
            neg = np.add.reduceat(weight * ~positive, starts, axis=1)
            denominator = pos.sum(axis=1) * neg.sum(axis=1)
            if not (denominator > 0).all():
                raise ValueError('Bootstrap draw lost one class; no automatic redraw')
            numerator = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(axis=1)
            columns.append(numerator / denominator)
        return np.column_stack(columns)
    return auc


def calibrate(folds, raw_index, classification, seed, draws=1000, batch_size=20, alpha=.05):
    """folds: (target, prediction matrix, zero-based group codes, group count)."""
    candidates = folds[0][1].shape[1]
    observed = np.zeros(candidates)
    scores = np.zeros((draws, candidates))
    rng = np.random.default_rng(seed + 4000)
    for y, predictions, codes, groups in folds:
        engine = weighted_engine(y, predictions, classification)
        observed += engine(np.ones((1, len(y))))[0] / len(folds)
        for start in range(0, draws, batch_size):
            size = min(batch_size, draws - start)
            counts = rng.multinomial(groups, np.full(groups, 1. / groups), size=size)
            scores[start:start+size] += engine(counts[:, codes].astype(float)) / len(folds)
    sign = 1 if classification else -1
    gain = sign * (observed - observed[raw_index])
    bootstrap_gain = sign * (scores - scores[:, raw_index:raw_index+1])
    spread = bootstrap_gain.std(axis=0, ddof=1)
    active = spread > 1e-15
    maximum = (np.max((bootstrap_gain[:, active] - gain[active]) / spread[active], axis=1)
               if active.any() else np.zeros(draws))
    critical = max(0., float(np.quantile(maximum, 1-alpha, method='higher')))
    lower = gain - critical * spread
    return {'observed_scores': observed, 'bootstrap_scores': scores, 'gains': gain,
            'bootstrap_sd': spread, 'critical': critical, 'lower': lower,
            'passes': lower > 1e-12}


def select_cached(unit, pools_file, output, seed):
    """Select from an explicit existing development unit, without fitting models."""
    from pathlib import Path
    import joblib
    import pandas as pd
    from .io import read, write
    from .e2_pilot import RAW, key
    from .e2_search import select_pool
    from .e1 import metrics
    unit, pools_file, output = Path(unit), Path(pools_file), Path(output)
    if output.exists():
        raise ValueError('Selection output must be a new directory')
    source, folds, branches = joblib.load(unit / 'source.joblib'), read(unit / 'inner_splits.json'), read(pools_file)
    operations = {op['id']: op for op in read(unit / 'menu.json')}
    if (unit / 'derived_ops.json').exists():
        operations.update(read(unit / 'derived_ops.json'))
    operations['__native_openfe_component__'] = {'kind': 'openfe_component'}
    pool = {key(v['spec']): v for branch in branches.values() for v in branch['pool']}
    tokens = sorted(pool); raw_index = tokens.index(key(RAW)); data = []
    for i, fold in enumerate(folds):
        validation, training = fold['validation'], fold['train']
        if set(source['groups'].iloc[validation]) & set(source['groups'].iloc[training]):
            raise ValueError('Training/validation groups overlap')
        predictions = []
        for token in tokens:
            folder = unit / 'inner_models' / token / str(i)
            if read(folder / 'fit.json')['validation_row_ids'] != source['x'].index[validation].tolist():
                raise ValueError('Candidate predictions belong to different validation rows')
            prediction = np.load(folder / 'predictions.npy')
            actual = metrics(source['y'].iloc[validation], prediction, source['task'])
            if any(abs(actual[m] - pool[token]['fold_scores'][i][m]) > 1e-10 for m in actual):
                raise ValueError('Candidate score differs from its saved predictions')
            predictions.append(prediction)
        codes, groups = pd.factorize(source['groups'].iloc[validation], sort=True)
        data.append((np.asarray(source['y'].iloc[validation]), np.column_stack(predictions), codes, len(groups)))
    result = calibrate(data, raw_index, source['task']['type'] == 'classification', seed)
    allowed = {tokens[i] for i, passing in enumerate(result['passes']) if passing}
    settings = {'minimum_positive_folds': 2, 'robust_se_coefficient': 1.}
    choices = {}
    for name, branch in branches.items():
        subset = {key(v['spec']): v for v in branch['pool'] if key(v['spec']) in allowed or v['spec'] == RAW}
        _, chosen = select_pool(subset, pool[key(RAW)], operations, settings)
        choices[name] = chosen['spec']
    output.mkdir(parents=True)
    np.save(output / 'bootstrap_fold_average_scores.npy', result['bootstrap_scores'])
    write(output / 'calibration.json', {'critical': result['critical'], 'draws': 1000, 'alpha': .05,
        'candidates': [{'spec': pool[token]['spec'], 'gain': float(result['gains'][i]),
                        'lower': float(result['lower'][i]), 'passes': bool(result['passes'][i])}
                       for i, token in enumerate(tokens)]})
    summary = {'choices': choices, 'source_unit': str(unit.resolve()), 'pool_file': str(pools_file.resolve()),
        'seed': seed, 'new_fits_calls_training': 0, 'test_loaded': False,
        'conditional_fixed_prediction_check': True, 'not_a_non_degradation_guarantee': True}
    write(output / 'selection.json', summary)
    return summary
