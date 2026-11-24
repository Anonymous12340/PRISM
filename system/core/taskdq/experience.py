"""A small JSON case list; retrieval never accepts labels or evaluation scores."""
from copy import deepcopy


def retrieve(cases, dataset, task_type, numeric_columns, categorical_columns, limit=2):
    candidates = [v for v in cases if v['source_dataset'] != dataset]
    total = max(1, numeric_columns + categorical_columns)
    share = categorical_columns / total
    def order(case):
        source_share = case['categorical_columns'] / max(1, case['numeric_columns'] + case['categorical_columns'])
        return (case['task_type'] != task_type, abs(source_share - share), -case['eligible_seeds'],
                -case['mean_relative_inner_gain'], case['case_id'])
    selected = []; used_tables = set()
    for case in sorted(candidates, key=order):
        if case['source_dataset'] in used_tables: continue
        selected.append(deepcopy(case)); used_tables.add(case['source_dataset'])
        if len(selected) == limit: break
    return selected
