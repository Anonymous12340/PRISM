"""Development component: grouped cross-fitted, smoothed target encoding."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import TargetEncoder


def selected_columns(frame, mode):
    assert mode in ['categorical', 'numeric_bins', 'all']
    return [c for c in frame if mode == 'all' or
            (mode == 'numeric_bins') == pd.api.types.is_numeric_dtype(frame[c])]


def keys(frame, columns, edges):
    output = {}
    for col in columns:
        values = pd.cut(frame[col], edges[col], labels=False, include_lowest=True) if col in edges else frame[col]
        output[col] = values.astype('string').fillna('__missing__').to_numpy(dtype=object)
    return pd.DataFrame(output, index=frame.index)


def fit_mapping(frame, target, columns):
    edges = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            internal = sorted(frame[col].replace([np.inf, -np.inf], np.nan).quantile([.2, .4, .6, .8]).dropna().unique())
            edges[col] = [-np.inf, *internal, np.inf]
    learned = TargetEncoder(smooth=10., target_type='continuous').fit(keys(frame, columns, edges), np.asarray(target))
    return {'columns': columns, 'edges': edges, 'encoder': learned, 'smoothing': 10.}


def apply_encoding(frame, state):
    if state is None: return frame.copy()
    output = frame.copy(); values = state['encoder'].transform(keys(frame, state['columns'], state['edges']))
    for i, col in enumerate(state['columns']): output['__target_mean_' + col] = values[:, i]
    assert output.index.equals(frame.index)
    return output


def fit_cross_encoding(frame, target, groups, mode):
    """Each training row uses an encoder fitted without its entire source group."""
    columns = selected_columns(frame, mode)
    if not columns: return None, frame.copy(), []
    assert len(frame) == len(target) == len(groups)
    folds = min(5, pd.Series(np.asarray(groups)).nunique()); assert folds >= 2
    split = GroupKFold(n_splits=folds, shuffle=True, random_state=0)
    output = np.full((len(frame), len(columns)), np.nan); receipts = []
    for a, b in split.split(frame, groups=groups):
        assert not set(np.asarray(groups)[a]) & set(np.asarray(groups)[b])
        part = fit_mapping(frame.iloc[a], np.asarray(target)[a], columns)
        output[b] = part['encoder'].transform(keys(frame.iloc[b], columns, part['edges']))
        receipts.append({'fit_row_ids': frame.index[a].tolist(), 'apply_row_ids': frame.index[b].tolist()})
    assert np.isfinite(output).all()
    state = fit_mapping(frame, target, columns); train = frame.copy()
    for i, col in enumerate(columns): train['__target_mean_' + col] = output[:, i]
    assert len(train.columns) - len(frame.columns) <= 32
    return state, train, receipts
