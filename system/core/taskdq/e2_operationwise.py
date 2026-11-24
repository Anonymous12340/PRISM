"""Explicit prefix validation of already proposed fixed-library operations."""
from .e2_pilot import key


def prefixes(plans, known):
    result = {}
    for full in plans:
        assert full['kind'] == 'plan'
        for length in range(1, len(full['ids'])):
            candidate = {'kind': 'plan', 'ids': full['ids'][:length]}
            token = key(candidate)
            if token in known: continue
            if token not in result:
                result[token] = {'spec': candidate, 'origin': 'code_prefix_validation_of_saved_plan', 'parents': []}
            result[token]['parents'].append({'spec': full, 'prefix_length': length})
    return list(result.values())
