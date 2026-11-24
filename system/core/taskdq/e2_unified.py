"""Uniform candidate construction and fixed native-risk adoption for new E2 runs."""
from copy import deepcopy
from pathlib import Path
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from .e2_parameterized import ExpandedScorer
from .e2_transfer import generic_candidates
from .e2_structured import compile_answer
from .e2_aligned_sft import context as aligned_context
from .e2_library import fit_plan, apply_plan
from .e2_pilot import RAW, key, run_native_command, Engine, METHODS
from .e2_search import select_pool, paired_stats
from .e2_risk import calibrate
from .e2_selection import (MEAN_SELECTION, LEGACY_SELECTION, PRESERVING_FOLD_SELECTION,
                           choose_cv_mean, choose_preserving_fold, selection_protocol,
                           selection_metadata)
from .io import read, write, plain

REVISION = 'native-risk-unified-v1'
MARKER = '__native_openfe_component__'
NATIVE = {'kind': 'native_joint', 'ids': [MARKER]}
SETTINGS = {'minimum_positive_folds': 2, 'robust_se_coefficient': 1.}
IMPLEMENTATION_PATCH = 'readiness-wiring-v5'
BASE_LM_METHODS = {'caafe', 'llmfe', 'lm_cleaning_simplified'}
FULL_LM_METHODS = {'ours', 'without_task', 'one_shot', 'random_order'}


def method_weight_manifest(config):
    return {method: {'base_model_path': config['model_path'],
                     'adapter_path': config.get('adapter_path') if method in FULL_LM_METHODS else None,
                     'model_role': 'shared_base' if method in BASE_LM_METHODS else 'full_system' if method in FULL_LM_METHODS else 'no_lm',
                     'implementation_patch': IMPLEMENTATION_PATCH}
            for method in METHODS}


class MethodEngine:
    """Our learned planner and independent LM baselines have explicit weights."""
    def __init__(self, config):
        self.full = Engine(config)
        base = {k: v for k, v in config.items() if k not in {'adapter_path', 'adapter_dependency_dir', 'weight_variant', 'weight_revision'}}
        self.base = Engine(base)

    def generate(self, messages, schema, unit, method, number):
        if method in {'ours', 'without_task'}:
            return self.full.generate(messages, schema, unit, method, number)
        if method not in BASE_LM_METHODS:
            raise ValueError('Unexpected LM method: ' + method)
        result = self.base.generate(messages, schema, unit, method, number)
        assert 'planner_weights' not in result['request']
        if self.base.base is not None:
            model = self.base.base.model
            assert not any('lora_' in name for name, _ in model.named_parameters())
            write(unit / ('baseline_planner_load_' + method + '.json'),
                  {'base_model_path': self.base.c['model_path'], 'adapter_path': None, 'loaded_adapter_tensors': 0,
                   'runtime_no_lora_parameters_checked': True, 'method': method})
        return result


class CombinationNotExecutable(ValueError):
    pass


class UnifiedScorer(ExpandedScorer):
    def __init__(self, config, unit):
        super().__init__(config, unit)
        self.ops[MARKER] = {'id': MARKER, 'kind': 'openfe_component', 'family': 'feature'}

    def native(self, train, valid, y, groups, method, where):
        if self.c.get('forbid_new_native'):
            assert (where / 'native' / method / 'generation.json').exists(), 'Engineering run forbids new native generation'
        return super().native(train, valid, y, groups, method, where)

    def transform(self, spec, train, valid, y, groups, slot, where):
        if spec['kind'] != 'native_joint':
            return super().transform(spec, train, valid, y, groups, slot, where)
        assert spec['ids'][0] == MARKER
        cache = self.unit / 'native_cache' / str(slot)
        a, b, _ = self.native(train, valid, y, groups, 'openfe', cache)
        operations = [self.ops[i] for i in spec['ids'][1:]]
        assert len(operations) <= 8 and all(v['kind'] != 'model_repair' for v in operations)
        state, a, keep = fit_plan(a, operations); b = apply_plan(b, state)
        if len(a.columns) > len(train.columns) + 32:
            raise CombinationNotExecutable('Native combination exceeds the fixed 32 additional feature limit')
        assert b.index.equals(valid.index)
        where.mkdir(parents=True, exist_ok=True)
        joblib.dump({'kind': 'native_joint', 'steps': state, 'native_cache': str(cache.resolve())}, where / 'transform_state.joblib')
        write(where / 'transform_check.json', {'passed': True, 'training_row_ids': a.index.tolist(),
            'validation_row_ids': b.index.tolist(), 'training_rows_before': len(train), 'training_rows_after': len(a), 'features': list(a)})
        return a, b, np.asarray(y)[keep]

    def inner(self, spec):
        # Identical native component for our candidate and the pure OpenFE arm.
        if spec['kind'] == 'openfe':
            return super().inner(NATIVE)
        return super().inner(spec)


def native_apply(config, state, frame, tag):
    cache = Path(state['native_cache']) / 'native'
    inp, out = cache / (tag + '_input.csv'), cache / (tag + '_features.csv')
    frame.to_csv(inp, index_label='__row_id')
    with (cache / (tag + '_apply.log')).open('w') as log:
        command = [config['classical_python'], '-m', 'taskdq.e2_classical', '--folder', str(cache),
                   '--method', 'openfe', '--apply-csv', str(inp), '--output-csv', str(out)]
        code = run_native_command(command, {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}, log,
                                  config.get('native_timeouts', {}).get('openfe', config['native_timeout']))
        if code:
            raise RuntimeError('Saved native feature application failed')
    added = pd.read_csv(out, float_precision='round_trip').set_index('__row_id')
    assert added.index.equals(frame.index)
    for column in read(cache / 'openfe/generation.json')['categorical']:
        added[column] = added[column].astype('string').astype(object)
    return apply_plan(pd.concat([frame, added], axis=1), state['steps'])


def metadata(sc):
    supplied = sc.c.get('field_metadata', {}).get(sc.source.get('pair', {}).get('name', ''), {})
    return {'columns': supplied.get('columns', {c: 'Original source field; no additional semantic claim.' for c in sc.x}),
            'numeric_type_hints': supplied.get('numeric_type_hints', sc.source.get('numeric_type_hints', [])),
            'same_quantity_groups': supplied.get('same_quantity_groups', [])}


def code_candidates(sc):
    info = metadata(sc); result = []
    for value in generic_candidates(sc.x, info, limit=12, allow_short=True):
        spec = compile_answer(sc, {'diagnosis': 'feature_opportunity', 'reason': 'Uniform code candidate, not a model discovery.',
                                  'operations': value['steps']}, info['numeric_type_hints'])
        result.append({**value, 'spec': spec})
    write(sc.unit / 'derived_ops.json', sc.derived)
    return result


def context(sc, pool, history, number, include_task=True):
    # Cache accounting belongs to logs; it must not change later model inputs.
    visible_history = [{k: v for k, v in row.items() if k != 'physical_call'} for row in history]
    messages, schema, _, _ = aligned_context(sc, pool, visible_history, number)
    payload = json.loads(messages[1]['content'])
    # Every column receives a diagnosis within the first N5 prefix.
    names = sorted(sc.x)
    batch = names[(number - 1) % 5::5]
    if not batch:
        batch = [names[(number - 1) % len(names)]]
    primary = batch[0]; old_diagnostic = next(iter(schema['properties']['diagnostics']['properties'].values()))
    definitions = {}
    for column in batch:
        definition = deepcopy(old_diagnostic)
        definition['properties']['reason']['maxLength'] = 60
        if sc.x[column].isna().any():
            if 'missing' not in definition['properties']['problem']['enum']:
                definition['properties']['problem']['enum'].append('missing')
            if 'fill' not in definition['properties']['recommendation']['enum']:
                definition['properties']['recommendation']['enum'].append('fill')
        else:
            definition['properties']['problem']['enum'] = [v for v in definition['properties']['problem']['enum'] if v != 'missing']
            definition['properties']['recommendation']['enum'] = [v for v in definition['properties']['recommendation']['enum'] if v != 'fill']
        definitions[column] = definition
    schema['properties']['diagnostics'] = {'type': 'object', 'properties': definitions, 'required': batch, 'additionalProperties': False}
    from .profile import profile_table
    profiles = profile_table(sc.x, sc.y if include_task else None, sc.task['type'], top_k=2)
    payload['focus_column'], payload['focus_columns'] = primary, batch
    payload['profiles'] = [{k: v for k, v in profile.items() if k != 'example_values'}
                           for profile in profiles['column_profiles'] if profile['name'] in batch]
    payload['five_sample_rows'] = [{c: row[c] for c in batch} for row in profiles['sample_rows'][:5]]
    payload['documented_numeric_type_hints'] = metadata(sc)['numeric_type_hints']
    payload['planning_scope']['diagnostics'] = '逐列判断focus_columns全部字段，每条理由用10至20字；operations仍是全表完整方案。'
    if not include_task:
        payload.pop('task', None); payload.pop('public_background', None)
        payload['task_type'] = sc.task['type']
        payload['all_field_columns'] = payload['all_field_columns'][:3]
        payload['all_fields'] = [row[:3] for row in payload['all_fields']]
        for profile in payload['profiles']:
            profile.pop('target_association', None)
    messages[0]['content'] += '\n本版按focus_columns逐列诊断，每列理由10至20字；不复述画像。'
    messages[1]['content'] = json.dumps(plain(payload), ensure_ascii=False, separators=(',', ':'))
    return messages, schema, batch


def calibration(sc, pools):
    mode = selection_protocol(sc.c)
    combined = {token: value for pool in pools.values() for token, value in pool.items()}
    tokens = sorted(combined); fold_data = []
    for i, fold in enumerate(sc.folds):
        predictions = np.column_stack([np.load(sc.unit / 'inner_models' / token / str(i) / 'predictions.npy') for token in tokens])
        codes, groups = pd.factorize(sc.source['groups'].iloc[fold['validation']], sort=True)
        fold_data.append((np.asarray(sc.y.iloc[fold['validation']]), predictions, codes, len(groups)))
    try:
        value = calibrate(fold_data, tokens.index(key(RAW)), sc.task['type'] == 'classification', sc.source['seed'])
    except ValueError as exc:
        # A failed diagnostic bootstrap is not an execution failure of finite-CV
        # candidates. Preserve the old hard failure for the legacy protocol.
        if mode == LEGACY_SELECTION or not str(exc).startswith('Bootstrap draw lost one class'):
            raise
        result = {'critical': None, 'draws': 1000, 'alpha': .05,
                  'uncertainty_available': False, 'uncertainty_error': str(exc),
                  'records': [{'spec': combined[token]['spec'], 'gain': None,
                               'lower': None, 'passes': None} for token in tokens]}
        result.update(selection_metadata(sc.c))
        if mode == MEAN_SELECTION:
            choices = {name: choose_cv_mean(pool, combined[key(RAW)], sc.ops)
                       for name, pool in pools.items()}
        else:
            choices = {}
            for name, pool in pools.items():
                _, stable = select_pool(pool, combined[key(RAW)], sc.ops, SETTINGS)
                choices[name] = choose_preserving_fold(combined[key(RAW)], stable,
                                                      combined[key(RAW)], sc.ops)
        return result, choices
    allowed = {tokens[i] for i, passing in enumerate(value['passes']) if passing}
    result = {'critical': value['critical'], 'draws': 1000, 'alpha': .05,
              'records': [{'spec': combined[token]['spec'], 'gain': float(value['gains'][i]),
                           'lower': float(value['lower'][i]), 'passes': bool(value['passes'][i])} for i, token in enumerate(tokens)]}
    choices = {}
    for name, pool in pools.items():
        if mode == MEAN_SELECTION:
            chosen = choose_cv_mean(pool, combined[key(RAW)], sc.ops)
        else:
            subset = {t: v for t, v in pool.items() if t in allowed or v['spec'] == RAW}
            _, chosen = select_pool(subset, combined[key(RAW)], sc.ops, SETTINGS)
            if mode == PRESERVING_FOLD_SELECTION:
                _, stable = select_pool(pool, combined[key(RAW)], sc.ops, SETTINGS)
                chosen = choose_preserving_fold(chosen, stable, combined[key(RAW)], sc.ops)
        choices[name] = chosen
    result.update(selection_metadata(sc.c))
    return result, choices


def add_component(sc, pool, raw):
    expanded = dict(pool); expanded[key(NATIVE)] = sc.inner(NATIVE)
    _, best = select_pool(pool, raw, sc.ops, SETTINGS)
    if not any(sc.ops[i]['kind'] == 'model_repair' for i in best['spec']['ids']):
        spec = {'kind': 'native_joint', 'ids': [MARKER, *best['spec']['ids']]}
        try:
            expanded[key(spec)] = sc.inner(spec)
        except CombinationNotExecutable as exc:
            write(sc.unit / 'component_rejections' / (key(spec) + '.json'),
                  {'spec': spec, 'reason': str(exc), 'fixed_feature_limit': 32, 'uncombined_candidates_preserved': True})
    return expanded


def freeze_code_control(config, sc, code_pool):
    """The zero-LM arm is frozen before seeing any model proposals."""
    evidence, choices = calibration(sc, {'code': code_pool})
    folder = sc.unit / 'methods/no_lm_search'
    write(folder / 'risk_calibration.json', evidence)
    for number in config['budgets']:
        selected = {'method': 'no_lm_search', 'spec': choices['code']['spec'], 'inner_scores': choices['code']['scores'],
                    'logical_calls': 0, 'valid_proposals': len(code_pool)-1, 'status': 'passed', 'budget': number,
                    'selected_before_outer_labels': True, 'frozen_before_any_lm_proposal': True,
                    'matched_code_and_native_constructor': True, 'method_revision': REVISION,
                    'implementation_patch': IMPLEMENTATION_PATCH}
        selected.update(selection_metadata(config))
        write(folder / 'checkpoints' / f'{number:03}.json', selected)
    write(folder / 'selection.json', selected)


def run_planner(config, sc, engine, method='ours'):
    folder = sc.unit / 'methods' / method
    if (folder / 'selection.json').exists():
        return read(folder / 'selection.json')
    raw = sc.inner(RAW); initial = {key(RAW): raw}
    candidates = code_candidates(sc); write(sc.unit / 'unified_code_candidates.json', candidates)
    for candidate in candidates:
        value = sc.inner(candidate['spec']); initial[key(value['spec'])] = value
    code_pool = add_component(sc, initial, raw); pool = dict(initial)
    if method == 'ours':
        freeze_code_control(config, sc, code_pool)
    history = []; first = None; started = time.monotonic(); valid = 0
    include_task = method != 'without_task'
    for number in range(1, config['lm_budget'] + 1):
        messages, schema, focus = context(sc, pool, history, number, include_task)
        call = engine.generate(messages, schema, sc.unit, method, number)
        row = {'round': number, 'executable': False, 'focus_columns': focus, 'physical_call': call['physical_call']}
        try:
            if not call['format_valid'] or not call['eos_completed']:
                raise ValueError('Incomplete structured model output')
            answer = call['answer']; assert set(answer['diagnostics']) == set(focus)
            flags = []
            for column in focus:
                diagnosis = answer['diagnostics'][column]
                if not sc.x[column].isna().any() and (diagnosis['problem'] == 'missing' or diagnosis['recommendation'] == 'fill'):
                    flags.append(column + ': missing claim contradicts observed facts')
            assert not flags, flags
            spec = compile_answer(sc, {k: answer[k] for k in ['diagnosis', 'reason', 'operations']}, metadata(sc)['numeric_type_hints'])
            repair_columns = {sc.ops[i]['column'] for i in spec['ids'] if sc.ops[i]['kind'] == 'model_repair'}
            existing = {read(p)['column'] for p in (sc.unit / 'repair_bundles').glob('*.json') if 'column' in read(p)}
            if len(existing | repair_columns) > 2:
                raise ValueError('Fixed engineering repair-column budget exceeded')
            value = sc.inner(spec); pool[key(spec)] = value; valid += 1
            row.update(executable=True, spec=spec, scores=value['scores'], paired_vs_raw=paired_stats(value, raw), reason=answer['reason'])
        except Exception as exc:
            row['error'] = repr(exc)
        history.append(row); write(folder / 'proposals.json', history)
        if number == 1 or number in config['budgets']:
            system_pool = add_component(sc, pool, raw)
            evidence, choices = calibration(sc, {'code': code_pool, 'system': system_pool})
            write(folder / 'risk_calibration' / (str(number) + '.json'), evidence)
            write(folder / 'candidate_pools' / (str(number) + '.json'), {'code': {'pool': list(code_pool.values())}, 'system': {'pool': list(system_pool.values())}})
            if number == 1:
                first = choices['system']
            if number in config['budgets']:
                selected = {'method': method, 'spec': choices['system']['spec'], 'inner_scores': choices['system']['scores'],
                            'logical_calls': number, 'valid_proposals': valid, 'status': 'passed', 'budget': number,
                            'selected_before_outer_labels': True, 'method_revision': REVISION}
                selected.update(selection_metadata(config))
                write(folder / 'checkpoints' / f'{number:03}.json', selected)
    final = read(folder / 'checkpoints' / f'{config["budgets"][-1]:03}.json')
    write(folder / 'selection.json', {**final, 'proposals': history, 'first_choice': first, 'seconds': time.monotonic()-started})
    write(sc.unit / 'derived_ops.json', sc.derived)
    return read(folder / 'selection.json')
