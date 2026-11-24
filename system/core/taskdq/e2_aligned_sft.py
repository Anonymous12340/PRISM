"""Compact, identical train/deploy contexts for the bounded E2 SFT trial."""
import json
import numpy as np
from .e2_parameterized import FEATURES
from .e2_structured import schema, OPERATOR_HELP
from .e2_search import paired_stats
from .e2_pilot import key
from .profile import profile_table
from .io import plain


INSTRUCTION = '''依据公开任务、字段含义、训练画像和真实验证反馈，诊断focus_column并提出完整处理计划。只输出给定JSON，不写代码。
诊断区分已观察事实、IQR候选与特征机会；无缺失不声称缺失，IQR不是真错。所有字段均可选，最多4步，允许空计划。只能用操作库的真实语义，numeric_parse才转数值，canonicalize仍是类别；@1/@2/@3只引用前面实际生成的新数值列。
代码探索不是你的发现。可以采用输入中已验证有效的方案，也可以提出有任务依据的新组合；不要把弱候选失效推断为不存在更好方案。已知方案没有依据时可以保留，不保证改善。
新增表示须解释任务机制，相关性不等于因果；单位缩放需要单位混用证据。原列默认保留，类型/删列动作须说明依据。看清每折增益和自己真实的失败，反馈后不应遗忘已知最佳。
reason和diagnostics理由只用简短中文，各1至2句，不复述全表或虚构分数；输出完整operations计划，代码执行及原采用门决定最终部署。'''

BRIEF_HELP = {
    'fill': 'fill(column,method)：训练折均值/中位数/众数填缺失。',
    'model_repair': 'model_repair(column)：其他字段估计缺失及IQR候选值，可能误改有效极端。',
    'flag': 'flag(column)：新增缺失/IQR候选标记，保留原值。',
    'clip': 'clip(column)：按训练IQR界截断。',
    'drop_outlier': 'drop_outlier(column)：仅训练中删IQR候选行。',
    'drop_missing': 'drop_missing(column)：仅训练中删缺失行。',
    'canonicalize': 'canonicalize(column)：归并类别拼写，输出仍是类别。',
    'numeric_parse': 'numeric_parse(column)：公开数值提示字段在原列严格转数值。',
    'unit_scale': 'unit_scale(column,factor)：整列乘倍率，仅有单位不一致依据时用。',
    'ordinal': 'ordinal(column)：训练词表按字符串排序转序号，改原列。',
    'onehot': 'onehot(column)：训练词表独热；raw预测器已自动独热。',
    'signed_log': 'signed_log(column)：新增sign(x)*log1p(abs(x))。',
    'square': 'square(column)：新增平方。',
    'bin5': 'bin5(column)：训练分位数划5箱，新增编号。',
    'ratio': 'ratio(left,right)：新增比值，近零分母记缺失。',
    'product': 'product(left,right)：新增乘积。',
    'sum': 'sum(left,right)：新增和，应具有合理量纲。',
    'difference': 'difference(left,right)：新增差，应具有合理量纲。',
    'frequency': 'frequency(column)：新增训练类别频率，未知为0。',
    'group_mean': 'group_mean(group,column)：新增训练类别组数值均值，未知组回退全局均值。',
    'date_parts': 'date_parts(column,format)：解析日期生成分量，删除原日期列。',
    'drop': 'drop(column)：删预测列，须有任务依据。',
    'drop_constant': 'drop_constant()：删训练常量列。',
    'drop_redundant': 'drop_redundant(threshold)：按0.99数值相关阈值删冗余列。',
    'deduplicate': 'deduplicate()：仅训练中删完全重复预测行。'}


def compact(value):
    if isinstance(value, dict): return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list): return [compact(v) for v in value]
    if isinstance(value, float) and np.isfinite(value): return float(f'{value:.6g}')
    return value


def operation_objects(spec, ops):
    aliases = {}; result = []
    for pos, identifier in enumerate(spec['ids'], 1):
        value = {k: v for k, v in ops[identifier].items() if k not in ['id', 'family']}
        for field in ['column', 'left', 'right', 'group']:
            if field in value: value[field] = aliases.get(value[field], value[field])
        result.append(value)
        if ops[identifier]['kind'] in FEATURES: aliases['__e2_' + identifier] = '@' + str(pos)
    return result


def context(sc, pool, history, round_number):
    profile = profile_table(sc.x, sc.y, sc.task['type'], top_k=2)
    columns = profile['column_profiles']
    association = lambda v: abs(v.get('target_association', {}).get('value') or 0.)
    ordered = sorted(columns, key=lambda v: (-association(v), v['name']))
    focus = ordered[(round_number - 1) % len(ordered)]['name']
    detailed = [focus] + [v['name'] for v in ordered if v['name'] != focus][:2]
    raw = next(v for v in pool.values() if not v['spec']['ids'])
    best = max(pool.values(), key=lambda v: v['utility'])
    shown = sorted(pool.values(), key=lambda v: -v['utility'])[:3]
    def feedback(value): return compact({'operations': operation_objects(value['spec'], sc.ops), 'scores': value['scores'], 'paired_vs_raw': paired_stats(value, raw)})
    hints = sc.source.get('numeric_type_hints', [])
    detail = []
    for v in columns:
        if v['name'] not in detailed: continue
        item = {k: v[k] for k in ['name', 'kind', 'missing_count', 'unique_count', 'target_association']}
        for field in ['quantiles', 'iqr_outliers']:
            if field in v: item[field] = v[field]
        item['top_values'] = [{k: z[k] for k in ['value', 'count']} for z in v['top_values']]
        detail.append(compact(item))
    payload = {'task': sc.task, 'public_background': sc.source['pair']['background'][:1800], 'focus_column': focus,
        'documented_numeric_type_hints': hints,
        'source_rows': len(sc.x), 'aggregate_display_significant_digits': 6, 'profiles': detail,
        'five_sample_rows': [{c: v[c] for c in detailed} for v in profile['sample_rows']],
        'all_field_columns': ['name', 'kind', 'missing', 'association_metric', 'association_value'],
        'all_fields': [[v['name'], v['kind'], v['missing_count'], v['target_association']['metric'], compact(v['target_association']['value'])] for v in columns],
        'operator_definitions': {k: BRIEF_HELP[k] for k in sorted({v['kind'] for v in sc.menu} | ({'numeric_parse'} if hints else set()))},
        'actual_downstream': 'Fixed XGBoost; original categories automatically one-hot encoded; all transforms fitted within inner training folds.',
        'raw_scores': compact(raw['scores']), 'known_mean_best': feedback(best),
        'known_top_candidates': [feedback(v) for v in shown],
        'candidate_origin': 'Code exploration and explicitly listed own proposals; code candidates are not language-model discoveries.',
        'own_actual_feedback': compact(history), 'round': round_number,
        'planning_scope': {
            'diagnostics': '只判断focus_column，正常列可建议keep。',
            'operations': '这是整张表的完整替代计划，不是只对focus_column的动作。',
            'empty_operations': '明确重置为原始全表；不表示只保留当前列。',
            'retaining_known_plan': '当前列无需改动时，其他列已验证有效的处理仍可保留；如选择known_mean_best，请完整输出其operations。也可自由提出新计划，禁止虚构收益。'}}
    messages = [{'role': 'system', 'content': INSTRUCTION}, {'role': 'user', 'content': json.dumps(plain(payload), ensure_ascii=False, separators=(',', ':'))}]
    allowed = schema(sc, hints)
    allowed['properties']['reason']['maxLength'] = 600
    diagnostic = {'type': 'object', 'properties': {
        'problem': {'type': 'string', 'enum': ['none', 'missing', 'outlier_candidate', 'format', 'feature_opportunity', 'uncertain']},
        'recommendation': {'type': 'string', 'enum': ['keep', 'fill', 'handle_outliers', 'repair_type', 'encode', 'generate_feature', 'review', 'drop']},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 240}},
        'required': ['problem', 'recommendation', 'reason'], 'additionalProperties': False}
    allowed['properties']['diagnostics'] = {'type': 'object', 'properties': {focus: diagnostic}, 'required': [focus], 'additionalProperties': False}
    allowed['required'].append('diagnostics')
    if not sc.x[focus].isna().any():
        diagnostic['properties']['problem']['enum'].remove('missing')
        diagnostic['properties']['recommendation']['enum'].remove('fill')
    if not sc.x.isna().any().any(): allowed['properties']['diagnosis']['enum'].remove('missing')
    return messages, allowed, focus, best['utility']


def audit(answer, sc, focus):
    flags = []
    if set(answer.get('diagnostics', {})) != {focus}: flags.append('focus column missing')
    diagnostic = answer.get('diagnostics', {}).get(focus, {})
    if diagnostic.get('problem') == 'missing' and not sc.x[focus].isna().any(): flags.append('claimed missing with zero observed missing cells')
    if diagnostic.get('recommendation') == 'fill' and not sc.x[focus].isna().any(): flags.append('recommended filling a fully observed focus column')
    if answer.get('diagnosis') == 'missing' and not sc.x.isna().any().any(): flags.append('global missing diagnosis with no observed missing cells')
    if len(answer.get('reason', '')) >= 600: flags.append('global reason reached length limit')
    if len(diagnostic.get('reason', '')) >= 240: flags.append('diagnostic reason reached length limit')
    return flags


def probe_specs(sc):
    columns = profile_table(sc.x, sc.y, sc.task['type'], top_k=2)['column_profiles']
    assoc = {v['name']: abs(v.get('target_association', {}).get('value') or 0.) for v in columns}
    numeric = sorted(sc.x.select_dtypes(include=np.number), key=lambda c: (-assoc[c], c))
    cats = sorted(set(sc.x) - set(numeric), key=lambda c: (-assoc[c], c))
    weak = min(sc.x, key=lambda c: (assoc[c], c))
    if len(numeric) < 2:
        assert len(cats) >= 2
        proposals = [{'kind': 'frequency', 'column': cats[0]}, {'kind': 'ordinal', 'column': cats[0]},
            {'kind': 'frequency', 'column': cats[1]}, {'kind': 'ordinal', 'column': cats[1]},
            {'kind': 'numeric_parse', 'column': sc.source['numeric_type_hints'][0]} if sc.source.get('numeric_type_hints') else {'kind': 'canonicalize', 'column': cats[0]},
            {'kind': 'drop', 'column': weak}]
    else:
        proposals = [{'kind': 'flag', 'column': numeric[0]}, {'kind': 'signed_log', 'column': numeric[0]},
            {'kind': 'ratio', 'left': numeric[0], 'right': numeric[1]},
            {'kind': 'product', 'left': numeric[0], 'right': numeric[1]},
            {'kind': 'frequency', 'column': cats[0]} if cats else {'kind': 'square', 'column': numeric[1]},
            {'kind': 'drop', 'column': weak}]
    lookup = {key({k: v for k, v in op.items() if k not in ['id', 'family']}): op['id'] for op in sc.ops.values()}
    result = []
    for proposal in proposals:
        identifier = lookup.get(key(proposal))
        if identifier is None:
            identifier = sc.register(proposal['kind'], **{k: v for k, v in proposal.items() if k != 'kind'})
        result.append({'kind': 'plan', 'ids': [identifier]})
    return result
