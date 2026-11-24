"""Explicit CV selection, separate from uncertainty reporting and outer evaluation."""
import json
import math

LEGACY_SELECTION = 'bootstrap_robust_v1'
MEAN_SELECTION = 'mean_cv_v1'
PRESERVING_FOLD_SELECTION = 'legacy_preserving_fold_v1'
HARMFUL = {'model_repair', 'clip', 'drop_missing', 'drop_outlier', 'drop',
           'drop_constant', 'drop_redundant'}


def selection_protocol(config):
    mode = config.get('selection_protocol', LEGACY_SELECTION)
    if mode not in {LEGACY_SELECTION, MEAN_SELECTION, PRESERVING_FOLD_SELECTION}:
        raise ValueError('Unknown selection protocol: ' + str(mode))
    return mode


def selection_metadata(config):
    mode = selection_protocol(config)
    if mode == LEGACY_SELECTION:
        return {}
    if mode == PRESERVING_FOLD_SELECTION:
        return {'selection_protocol': mode,
                'uncertainty_role': 'bootstrap_defines_conservative_reference_not_global_veto',
                'fold_reference': 'two_positive_folds_and_mean_gain_above_one_SE',
                'simplicity_role': 'one_SE_for_fold_reference_then_numerical_tie_break',
                'conservative_reference_CV_cannot_worsen': True,
                'outer_evaluation_used_for_selection': False}
    return {'selection_protocol': MEAN_SELECTION,
            'uncertainty_role': 'report_only_not_an_adoption_gate',
            'simplicity_role': 'numerical_tie_break_only',
            'outer_evaluation_used_for_selection': False}


def choose_cv_mean(pool, raw, operations, tie_tolerance=1e-12):
    """Choose the best finite CV metric; complexity breaks numerical ties only.

    Inputs must already have passed execution/data-boundary checks. This function
    makes no claim that a validation winner will improve an outer evaluation.
    """
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError('Tie tolerance must be finite and nonnegative')
    metric = 'auc' if 'auc' in raw['scores'] else 'rmse'
    sign = 1 if metric == 'auc' else -1

    def utility(value):
        score = value.get('scores', {}).get(metric)
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            return None
        return sign * float(score)

    if utility(raw) is None:
        raise ValueError('Raw must have a finite validation metric')
    # Derive ordering from the metric rather than trusting stale utility metadata.
    values = [raw] + [v for v in pool.values() if utility(v) is not None]
    best = max(utility(v) for v in values)
    tied = [v for v in values if best - utility(v) <= tie_tolerance]

    def complexity(value):
        ids = value['spec'].get('ids', [])
        return (sum(operations[i]['kind'] in HARMFUL for i in ids), len(ids),
                -utility(value), json.dumps(value['spec'], sort_keys=True))

    return min(tied, key=complexity)


def choose_preserving_fold(conservative, fold_champion, raw, operations):
    """Adopt a fold-stable challenger only if its CV is no worse than the reference.

    The caller computes the conservative legacy selection and the original
    fold/one-SE selection from the same candidate pool. Neither uses outer data.
    """
    return choose_cv_mean({'conservative': conservative, 'fold': fold_champion}, raw, operations)
