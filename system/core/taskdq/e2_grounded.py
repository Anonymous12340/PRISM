"""Small development repair: field evidence, free LM plans, held-out adoption."""
from copy import deepcopy
from pathlib import Path
import json
import random
import joblib
import numpy as np
from .e2_parameterized import ExpandedScorer, choices, registry, plan_label, evaluate
from .e2_pilot import RAW, key
from .e2_search import paired_stats
from .e2_semantic import exact_relations
from .e2_semantic import usable, label
from .profile import profile_table
from .scoring import splits, fit_predictor, predict
from .e1 import metrics
from .io import read, write, plain


INSTRUCTION = '''目标是提高固定下游模型的验证分数。按字段含义和任务需求诊断并规划，输出固定JSON。
原表可能没有真实错误；仍可自由提出有任务机制依据的特征生成/编码方案。诊断none不等于禁止有依据的特征生成，但不能为凑操作而编造错误。
对focus_columns中的每个字段各给一项diagnostics：column、problem、recommendation、reason；不要重复字段。IQR只表示候选极端，不是真错；不要删除正常高风险人群或把合法类别unknown当拼写错。reason用简短中文解释依据。
读取field_meanings中的定义/单位/特殊取值，以及profiles、5行样例、已执行的真实反馈。数值关联只表示关联，不是因果。不同量纲直接相加/相减需要明确理由。优先考虑对预测机制有意义的关系；字段无缺失且无错证据时应保留原值。任务机会可以是新特征，不能伪称检测到了错误。
operations是最多一个完整计划字符串（字符串内最多4步，使用 -> 连接）；只能从约束解码允许的固定六类库计划中自由选择。@1/@2表示对应步骤产生的数值列。可用任何原始字段，必要证据不足时保留并说明。所有字段动作都可达。
[]或keep表示保留原始输入；retain_best表示保留已验证的best候选。它们不代表整张表已完成诊断，代码仍会处理后续字段。选择已知best不会重新拟合。不要为了新颖而给负收益方案换列，不要臆造分数。reason说明完整计划与本任务的机制及风险，1-2句即可。
只可看到discovery三折反馈，adoption和最终确认都不可见。已有候选的收益和执行后实际列/行变化会显示；被回退的诊断记录仍保留。'''


def batches(columns):
    """Five stable batches, repeated once: every field gets a real diagnosis."""
    columns = list(columns)
    return [columns[i::5] or [columns[i % len(columns)]] for i in range(5)]


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return float(f'{value:.6g}')
    return plain(value)


def prepare_unit(s, c, name, seed, dest):
    if (dest / 'prepared.json').exists():
        return
    old = Path(s['reference_output']) / 'units' / name / str(seed) / 'search'
    full = joblib.load(old / 'source.joblib')
    assert set(full['x']) == set(s['field_meanings'][name]['columns'])
    a, b = splits(full['x'], full['y'], full['task'], 5, seed, full['groups'])[0]
    source = deepcopy(full)
    for field in ['x', 'y', 'groups']:
        source[field] = full[field].iloc[a]
    adoption = {field: full[field].iloc[b] for field in ['x', 'y', 'groups']}
    outer = read(old / 'split.json')
    assert not set(source['groups']) & set(adoption['groups'])
    assert not set(source['x'].index) & set(outer['confirmation_row_ids'])
    assert not set(adoption['x'].index) & set(outer['confirmation_row_ids'])
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(source, dest / 'source.joblib')
    joblib.dump(full, dest / 'full_search.joblib')
    joblib.dump(adoption, dest / 'adoption.joblib')
    write(dest / 'menu.json', registry(source['x'], source.get('date_columns')))
    folds = splits(source['x'], source['y'], source['task'], 3, seed, source['groups'])
    write(dest / 'inner_splits.json', [{'train': aa.tolist(), 'validation': bb.tolist(),
        'train_row_ids': source['x'].index[aa].tolist(), 'validation_row_ids': source['x'].index[bb].tolist()}
        for aa, bb in folds])
    write(dest / 'split.json', {**outer, 'discovery_row_ids': source['x'].index.tolist(),
        'adoption_row_ids': adoption['x'].index.tolist(), 'adoption_not_visible_to_planner': True})
    write(dest / 'prepared.json', {'passed': True, 'discovery_rows': len(a), 'adoption_rows': len(b),
        'field_batches': batches(source['x'].columns), 'original_test_reads': 0})


def best_eligible(pool, raw, s):
    candidates = []
    for result in pool.values():
        stats = paired_stats(result, raw)
        if stats['positive_folds'] >= s['minimum_positive_folds'] and stats['mean_gain'] > s['robust_se_coefficient'] * stats['se'] + 1e-12:
            candidates.append(result)
    return max(candidates, key=lambda v: v['utility']) if candidates else raw


def effects(sc, spec):
    values = []
    for i in range(3):
        folder = sc.unit / 'inner_models' / key(spec) / str(i)
        log = read(folder / 'operation_log.json')
        check = read(folder / 'transform_check.json')
        values.append({'fold': i, 'training_rows_before': check['training_rows_before'],
            'training_rows_after': check['training_rows_after'],
            'added_columns': log['new_features'],
            'removed_columns': sorted(set(sc.x) - set(check['features'])),
            'evaluation_rows_deleted': log['evaluation_rows_deleted']})
    return values


def context(sc, s, name, profiles, associations, raw, best, history, number):
    group = batches(sc.x.columns)[(number - 1) % 5]
    ordered = sorted(sc.x.columns, key=lambda col: (-abs(associations[col].get('value') or 0.), col))
    needed = list(dict.fromkeys(group + ordered[:2]))
    meanings = s['field_meanings'][name]
    templates = {}
    for op in sc.menu:
        if op['kind'] in templates:
            continue
        generic = dict(op)
        for field in ['column', 'left', 'right', 'group']:
            if field in generic:
                generic[field] = field
        templates[op['kind']] = label(generic)
    payload = {'input': {'task': sc.task, 'background': meanings['background'],
        'semantic_source': meanings['source'], 'field_meanings': meanings['columns'],
        'operator_templates': templates,
        'all_fields': [{'column': p['name'], 'type': p['kind'], 'missing': p['missing_count'],
                        'target_association': associations[p['name']]} for p in profiles['column_profiles']],
        'focus_columns': group,
        'profiles': [p for p in profiles['column_profiles'] if p['name'] in needed],
        'sample_rows': [{k: row[k] for k in needed} for row in profiles['sample_rows']],
        'training_exact_relations': exact_relations(sc.x),
        'scope': 'discovery only; all transformations fit inside respective training fold'},
        'best_plan': plan_label(best['spec'], sc.ops), 'best_scores': best['scores'],
        'raw_scores': raw['scores'], 'own_training_feedback': [
            {**{k: entry[k] for k in ['origin', 'input_version', 'round', 'plan', 'scores', 'paired_vs_raw', 'error'] if k in entry},
             **({'actual_execution': {
                 'training_rows_removed_per_fold': [v['training_rows_before'] - v['training_rows_after'] for v in entry['actual_execution']],
                 'removed_columns': sorted({col for v in entry['actual_execution'] for col in v['removed_columns']}),
                 'added_columns': sorted({col for v in entry['actual_execution'] for col in v['added_columns']}),
                 'evaluation_rows_deleted': 0}} if 'actual_execution' in entry else {})}
            for entry in history],
        'remaining_calls': s['calls_per_unit'] - number + 1}
    return compact(payload), group


def answer_schema(available, group):
    item = {'type': 'object', 'properties': {
        'column': {'type': 'string', 'enum': group or ['__feedback_only__']},
        'problem': {'type': 'string', 'enum': ['missing', 'outlier_candidate', 'format', 'feature_opportunity', 'none', 'uncertain']},
        'recommendation': {'type': 'string', 'enum': ['keep', 'consider_change']},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 80}},
        'required': ['column', 'problem', 'recommendation', 'reason'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {
        'diagnostics': {'type': 'array', 'items': item, 'minItems': len(group), 'maxItems': len(group)},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 120},
        'operations': {'type': 'array', 'items': {'type': 'string', 'enum': ['keep', 'retain_best', *available]}, 'maxItems': 1}},
        'required': ['diagnostics', 'reason', 'operations'], 'additionalProperties': False}


def run_unit(s, c, name, seed, unit, engine, pilot=False):
    if (unit / 'complete_corrected.json').exists():
        return
    sc = ExpandedScorer(c, unit)
    raw = sc.inner(RAW)
    best = raw
    pool = {key(RAW): raw}
    known = {key(RAW)}
    history = []
    prof = profile_table(sc.x, sc.y, sc.task['type'], top_k=3)
    associations = {v['name']: v['target_association'] for v in prof['column_profiles']}
    numeric_rho = {col: float(sc.x[col].corr(sc.y, method='spearman')) for col in sc.x.select_dtypes(include=np.number)}
    numeric_rho = {k: v for k, v in numeric_rho.items() if np.isfinite(v)}
    # Five code probes sample an operation family first, then a single primitive.
    # They are marked as code, never LM demonstrations or evidence of LM benefit.
    rng = random.Random(seed)
    for family in ['missing', 'outlier', 'encoding', 'feature', 'selection']:
        eligible = [op for op in sc.menu if op['family'] == family and usable(op, sc.x)]
        if not eligible:
            continue
        op = rng.choice(sorted(eligible, key=lambda v: label(v)))
        spec = {'kind': 'plan', 'ids': [op['id']]}
        if key(spec) in known:
            continue
        known.add(key(spec))
        entry = {'origin': 'balanced_code_probe', 'round': 0, 'plan': label(op), 'spec': spec, 'executable': False}
        try:
            result = evaluate(sc, spec, c['candidate_timeout'])
            pool[key(spec)] = result
            entry.update(executable=True, scores=result['scores'], paired_vs_raw=paired_stats(result, raw), actual_execution=effects(sc, spec))
            best = best_eligible(pool, raw, s)
        except Exception as exc:
            entry['error'] = repr(exc)
        history.append(entry)
    for number in range(1, (1 if pilot else s['calls_per_unit']) + 1):
        payload, group = context(sc, s, name, prof, associations, raw, best, history, number)
        focus = group[0] if group else list(sc.x)[(number - 1) % len(sc.x.columns)]
        available = choices(sc, best, known, focus, numeric_rho)
        old_call = unit / 'methods/grounded/calls' / (str(number) + '.json')
        saved = read(old_call) if old_call.exists() else None
        legacy = bool(saved and saved.get('request') and 'operator_templates' not in json.loads(saved['request']['messages'][-1]['content'])['input'])
        if legacy:
            # Preserve the actual old input/output, not a fictional replay under
            # corrected evidence. All such completed proposals were keep/best.
            record = engine.generate(saved['request']['messages'], saved['request']['schema'], unit, 'grounded', number)
            group = json.loads(saved['request']['messages'][-1]['content'])['input']['focus_columns']
        else:
            record = engine.generate([{'role': 'system', 'content': INSTRUCTION},
                {'role': 'user', 'content': json.dumps(plain(payload), ensure_ascii=False)}],
                answer_schema(available, group), unit, 'grounded', number)
        entry = {'origin': 'language_model', 'round': number, 'focus_columns': group, 'executable': False,
                 'input_version': 'legacy_missing_templates' if legacy else 'corrected_templates'}
        try:
            if not record['format_valid'] or not record.get('eos_completed', False):
                raise ValueError('Incomplete JSON output')
            answer = record['answer']
            columns = [v['column'] for v in answer['diagnostics']]
            if sorted(columns) != sorted(group):
                raise ValueError('Duplicate or incomplete field diagnoses')
            entry['diagnostics'] = answer['diagnostics']
            text = (answer['operations'] or ['keep'])[0]
            entry['plan'] = text
            if text in ['keep', 'retain_best']:
                entry.update(executable=True, decision_only=True)
            else:
                if legacy:
                    raise ValueError('Unexpected nonempty legacy output; inspect before reinterpretation')
                spec = available[text]
                assert key(spec) not in known
                known.add(key(spec))
                entry['spec'] = spec
                # Reserve eight adoption/outer imputer slots; reject a proposal before
                # it can silently exceed the bounded component-fitting budget.
                repair_columns = {sc.ops[i]['column'] for i in spec['ids'] if sc.ops[i]['kind'] == 'model_repair'}
                current = len(list((unit / 'repair_bundles').glob('*.json')))
                needed = sum(not (unit / 'repair_bundles' / (str(i) + '_' + key(col) + '.joblib')).exists()
                             for col in repair_columns for i in range(3))
                if current + needed > 8:
                    raise ValueError('Per-unit exploration imputer limit; proposal preserved without fitting')
                result = evaluate(sc, spec, c['candidate_timeout'])
                pool[key(spec)] = result
                entry.update(executable=True, scores=result['scores'], paired_vs_raw=paired_stats(result, raw),
                             actual_execution=effects(sc, spec))
                best = best_eligible(pool, raw, s)
        except Exception as exc:
            entry['error'] = repr(exc)
        history.append(entry)
        write(unit / 'history_corrected.json', history)
        print({'dataset': name, 'seed': seed, 'round': number, 'valid': entry['executable'], 'plan': entry.get('plan')}, flush=True)
    if not pilot:
        candidate = best_eligible(pool, raw, s)
        write(unit / 'candidate_frozen_corrected.json', {'candidate': candidate, 'raw': raw,
            'candidate_results': list(pool.values()), 'before_adoption_labels': True})
        write(unit / 'complete_corrected.json', {'passed': True, 'calls': sum(e['origin'] == 'language_model' for e in history),
            'diagnosed_columns': sorted({v['column'] for e in history for v in e.get('diagnostics', [])}),
            'failed': sum(not e['executable'] for e in history)})


def adoption_pass(candidate_score, raw_score, task):
    field = 'auc' if task['type'] == 'classification' else 'rmse'
    sign = 1 if field == 'auc' else -1
    return sign * (candidate_score[field] - raw_score[field]) > 1e-12


def adopt(s, c, unit):
    if (unit / 'adoption_result.json').exists():
        return
    frozen = read(unit / 'candidate_frozen_corrected.json')
    candidate = frozen['candidate']['spec']
    if not candidate['ids']:
        write(unit / 'adoption_result.json', {'plans': {'grounded_cv': RAW, 'grounded': RAW},
            'no_eligible_candidate': True, 'adoption_loaded': False})
        return
    sc = ExpandedScorer(c, unit)
    validation = joblib.load(unit / 'adoption.joblib')
    scores = {}
    for spec in [RAW, candidate]:
        folder = unit / 'adoption_models' / key(spec)
        receipt = folder / 'score.json'
        if receipt.exists():
            scores[key(spec)] = read(receipt)['score']
            continue
        a, b, y = sc.transform(spec, sc.x, validation['x'], sc.y, sc.source['groups'], 'adoption', folder)
        model = fit_predictor(a, y, sc.task, c['model'])
        joblib.dump(model, folder / 'model.joblib')
        prediction = predict(model, b)
        np.testing.assert_array_equal(prediction, predict(joblib.load(folder / 'model.joblib'), b))
        np.save(folder / 'predictions.npy', prediction)
        score = metrics(validation['y'], prediction, sc.task)
        write(folder / 'fit.json', {'predictor_fits': 1, 'training_row_ids': a.index.tolist(),
            'validation_row_ids': b.index.tolist(), 'candidate_frozen_before_labels': True})
        write(receipt, {'spec': spec, 'score': score})
        scores[key(spec)] = score
    passed = adoption_pass(scores[key(candidate)], scores[key(RAW)], sc.task)
    write(unit / 'adoption_result.json', {'plans': {'grounded_cv': candidate, 'grounded': candidate if passed else RAW},
        'passed': passed, 'scores': scores, 'single_candidate_only': True, 'not_fed_back_to_planner': True})
