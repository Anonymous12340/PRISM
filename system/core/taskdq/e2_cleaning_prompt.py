"""Compact presentation of the complete cleaning action library, retaining IDs."""
from copy import deepcopy
import json
import math
from .io import plain

VERSION = 'operator_table_v1'
COLUMN_ARGUMENTS = {'column', 'left', 'right', 'group'}


def pack_operations(operations, columns):
    positions = {name: i for i, name in enumerate(columns)}
    groups = {}
    for operation in operations:
        kind = operation['kind']
        fields = [k for k in operation if k not in {'kind', 'family'}]
        group = groups.setdefault(kind, {'family': operation['family'], 'fields': fields, 'rows': []})
        assert group['fields'] == fields and group['family'] == operation['family']
        group['rows'].append([positions[operation[k]] if k in COLUMN_ARGUMENTS else operation[k] for k in fields])
    return {'columns': list(columns), 'groups': groups,
            'interpretation': 'Each row follows fields. Integer column/left/right/group arguments index columns (zero based). '
                              'Select the original operation IDs; every permitted operation is present.'}


def unpack_operations(packed):
    result = []
    for kind, group in packed['groups'].items():
        for row in group['rows']:
            op = {'kind': kind, 'family': group['family']}
            for field, value in zip(group['fields'], row):
                op[field] = packed['columns'][value] if field in COLUMN_ARGUMENTS else value
            result.append(op)
    return result


def display_precision(value):
    if isinstance(value, dict):
        return {k: display_precision(v) for k, v in value.items()}
    if isinstance(value, list):
        return [display_precision(v) for v in value]
    if isinstance(value, float) and math.isfinite(value):
        return float(f'{value:.6g}')
    return value


def compact_messages(messages, visible, columns):
    result = deepcopy(messages)
    payload = json.loads(result[1]['content'])
    assert payload['operation_library'] == plain(visible)
    packed = pack_operations(visible, columns)
    # Exact round trip of every permitted operation, independent of display rounding.
    assert {v['id']: v for v in unpack_operations(packed)} == {v['id']: v for v in visible}
    payload.pop('operation_library')
    payload = display_precision(payload)
    payload.update(operation_library=packed, display_significant_digits=6,
                   display_note='Only displayed numeric summaries/examples are rounded; stored rows, fitted statistics and scores are unchanged.',
                   prompt_adapter=VERSION)
    result[0]['content'] += ' The operation library is a lossless table: read fields and rows, resolve column indices, and output original IDs.'
    result[1]['content'] = json.dumps(plain(payload), ensure_ascii=False, separators=(',', ':'))
    return result
