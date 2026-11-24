"""Explicit typed operation parameters, without a flat full-plan string enum."""
from .e2_parameterized import dependencies, FEATURES
from .e2_pilot import key


OPERATOR_HELP = {
    'numeric_parse': '把公开定义为数值的字符串/类别严格解析为数值，在原列上转换；后续使用原列名。仅限documented_numeric_type_hints。',
    'canonicalize': '对类别字符串去首尾空格、统一大小写并合并少见近似拼写；输出仍为类别类型。',
    'ordinal': '按训练词表的字符串排序将类别映射到0,1,...；这是类别编码，多位数字字符串的词序未必是数值大小顺序。',
    'onehot': '将类别列展开为训练词表二元列；raw下游已经自动独热，单独再选通常不增加表示信息。',
    'fill': '用训练折的均值/中位数填数值缺失，或众数填类别缺失；保留已有非缺失观察。',
    'model_repair': '以其他字段估计缺失及训练IQR候选值，模型只拟合训练折；候选极端可能是有效信号，替代需谨慎。',
    'flag': '新增缺失或训练IQR候选指示特征，原观察值保留。',
    'clip': '用训练IQR边界截断数值；边界只在训练折拟合，合法极端信号也可能受影响。',
    'drop_outlier': '训练时删除IQR候选极端行，验证/测试样本保留；可能丢失有用信号。',
    'drop_missing': '只在训练中删该列缺失行，验证/测试行保留。',
    'unit_scale': '整列乘固定系数换单位；需要明确单位不一致证据。统一倍率仅改尺度，树模型通常无需按尺度标准化。',
    'signed_log': '新增sign(x)*log(1+abs(x))，保留原列。',
    'square': '新增数值列的平方，保留原列。',
    'affine': '新增scale*x+offset，原列保留；固定常数须说明领域依据。与比值等复合可表达含常数基准的相对指标，单独线性换尺度通常不给树模型新增信息。',
    'bin5': '用训练分位数划5箱，新增箱编号，保留原数值列。',
    'ratio': '新增left/right，近零分母记缺失；两个操作数都须为数值。',
    'product': '新增left*right，两个操作数都须为数值。',
    'sum': '新增left+right，两个操作数须为数值且相加需有合理意义。',
    'difference': '新增left-right，两个操作数须为数值且相减需有合理意义。',
    'frequency': '按训练词表频率为类别值生成新数值列，未知类别频率为0。',
    'group_mean': '按类别group计算数值column的训练均值并生成新列，未知组回退训练均值；统计不读取目标标签。',
    'drop': '删除指定预测字段，需依据预测时点、泄漏或冗余等任务证据。',
    'drop_constant': '删除训练中只有一种取值的列。',
    'drop_redundant': '按训练数值相关阈值0.99依次删除高相关列；高相关不自动证明对任务无用。',
    'deduplicate': '只在训练中删除完全重复的预测行。',
    'date_parts': '按给定格式解析日期，生成年/月/星期/小时并删除原日期列。'
}


INSTRUCTION = '''你负责依据任务和字段证据提出处理假设，由代码执行和验证；只输出规定JSON，不写代码。
operations是0至4个操作对象，每个对象明确填写kind和字段参数，按顺序执行。@1/@2/@3只引用前面步骤产生的数值列。字段名必须精确使用输入。reason用简短中文说明具体机制及风险。可以保留原始数据（operations=[]）。
探索提案与最终部署分开：新提案只有验证通过才被采用，失败会保留已有方案，不必为保住当前方案而重复它。先根据任务提出自己的假设，再依据真实执行反馈修订。现有少量候选没有改善，不能说明所有未探索方案都无效或raw已经最优。不要臆造分数或保证改善。
无缺失不填补。IQR极端只是候选，不可断言真错；没有语义错误证据不自动改值/删行。类型修复只用于公开定义明确的数值字段。单位缩放需要明确的不一致证据。合法计数、负值和类别应保留其含义。
表没有真实错误时，也可提出有任务意义的比率、交互、分组特征，明确是特征机会而不是错误；不得编造因果关系。原列默认保留。列间推导可用不同字段或前一步输出；避免复述输入和反复提交相同完整计划。'''


def schema(sc, numeric_hints, allow_affine=False):
    numeric = [c for c in sc.x if sc.x[c].dtype.kind in 'iuf']
    cats = [c for c in sc.x if c not in numeric]
    refs = ['@1', '@2', '@3']; numeric_operands = numeric + list(numeric_hints)
    # Parameters are constrained independently; execution checks typed dependencies.
    sample = {}
    for op in sc.menu: sample.setdefault(op['kind'], op)
    if numeric_hints: sample['numeric_parse'] = {'kind': 'numeric_parse', 'column': numeric_hints[0]}
    if allow_affine and numeric: sample['affine'] = {'kind': 'affine', 'column': numeric[0], 'scale': 1., 'offset': 0.}
    branches = []
    for kind, op in sample.items():
        properties = {'kind': {'type': 'string', 'enum': [kind]}}
        for field, value in op.items():
            if field in ['id', 'family', 'kind']: continue
            if field in ['left', 'right']:
                properties[field] = {'type': 'string', 'enum': numeric_operands + refs}
            elif field == 'group': properties[field] = {'type': 'string', 'enum': cats}
            elif field == 'column':
                values = (numeric_hints if kind == 'numeric_parse' else cats if kind in ['frequency', 'ordinal', 'onehot', 'canonicalize']
                          else numeric_operands + refs if kind in ['signed_log', 'square', 'bin5', 'group_mean', 'affine']
                          else numeric if kind in ['clip', 'flag', 'model_repair', 'unit_scale', 'drop_outlier']
                          else list(sc.x))
                properties[field] = {'type': 'string', 'enum': values}
            elif field == 'method': properties[field] = {'type': 'string', 'enum': ['median', 'mean', 'mode']}
            elif field == 'factor': properties[field] = {'type': 'number', 'enum': [.001, .01, 1000.]}
            elif field == 'threshold': properties[field] = {'type': 'number', 'enum': [.99]}
            elif field == 'scale': properties[field] = {'type': 'number', 'minimum': -10000., 'maximum': 10000.}
            elif field == 'offset': properties[field] = {'type': 'number', 'minimum': -1000000., 'maximum': 1000000.}
            else: properties[field] = {'type': 'string', 'enum': [value]}
        if any(v.get('enum') == [] for v in properties.values()): continue
        if kind == 'fill':
            for columns, methods in [(numeric, ['median', 'mean']), (cats, ['mode'])]:
                if not columns: continue
                values = {**properties, 'column': {'type': 'string', 'enum': columns}, 'method': {'type': 'string', 'enum': methods}}
                branches.append({'type': 'object', 'properties': values, 'required': list(values), 'additionalProperties': False})
        else:
            branches.append({'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False})
    return {'type': 'object', 'properties': {
        'diagnosis': {'type': 'string', 'enum': ['none', 'missing', 'outlier_candidate', 'format', 'feature_opportunity', 'uncertain']},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 140},
        'operations': {'type': 'array', 'items': {'anyOf': branches}, 'maxItems': 4}},
        'required': ['diagnosis', 'reason', 'operations'], 'additionalProperties': False}


def compile_answer(sc, answer, numeric_hints, allow_affine=False):
    assert set(answer) == {'diagnosis', 'reason', 'operations'}
    assert len(answer['operations']) <= 4
    lookup = {key({k: v for k, v in op.items() if k not in ['id', 'family']}): op['id'] for op in sc.ops.values()}
    ids = []; numeric = {c for c in sc.x if sc.x[c].dtype.kind in 'iuf'}
    for proposal in answer['operations']:
        op = dict(proposal)
        if op['kind'] == 'affine':
            assert allow_affine, 'Affine feature requires the explicit experimental option'
            assert -10000. <= op['scale'] <= 10000. and -1000000. <= op['offset'] <= 1000000.
        if op['kind'] == 'numeric_parse': assert op['column'] in numeric_hints
        for field in ['column', 'left', 'right', 'group']:
            if field in op and op[field].startswith('@'):
                pos = int(op[field][1:]) - 1
                assert 0 <= pos < len(ids), 'Forward/missing derived reference'
                assert sc.ops[ids[pos]]['kind'] in FEATURES, 'Reference must name a generated numeric feature'
                op[field] = '__e2_' + ids[pos]
        fields = ([op['left'], op['right']] if op['kind'] in ['ratio', 'difference', 'sum', 'product'] else
                  [op['column']] if op['kind'] in ['signed_log', 'square', 'bin5', 'group_mean', 'flag', 'clip', 'model_repair', 'drop_outlier', 'unit_scale', 'affine'] else [])
        assert set(fields) <= numeric, 'Numerical operation reads a nonnumerical field'
        identifier = lookup.get(key(op))
        if identifier is None:
            assert op['kind'] in ['ratio', 'difference', 'sum', 'product', 'signed_log', 'square', 'bin5', 'numeric_parse', 'group_mean'] + (['affine'] if allow_affine else []), 'Unsupported dynamic operation'
            identifier = sc.register(op['kind'], **{k: v for k, v in op.items() if k != 'kind'})
        ids.append(identifier)
        if op['kind'] in FEATURES: numeric.add('__e2_' + identifier)
        if op['kind'] in ['numeric_parse', 'ordinal']: numeric.add(op['column'])
    assert len(set(ids)) == len(ids), 'Duplicate operation'
    dependencies(sc.x.columns, [sc.ops[i] for i in ids])
    return {'kind': 'plan', 'ids': ids}


def complete_numeric_types(frame, answer, numeric_hints):
    """Optional lowering only; every inserted cast is returned as code provenance."""
    from copy import deepcopy
    from .e2_library import parse_numeric
    completed = deepcopy(answer); operations = []; origins = []; inserted = []
    numeric = {c for c in frame if frame[c].dtype.kind in 'iuf'}
    available = set(frame); positions = {}
    for original_position, original in enumerate(answer['operations'], 1):
        op = deepcopy(original)
        for field in ['column', 'left', 'right', 'group']:
            if field in op and op[field].startswith('@'):
                old = int(op[field][1:]); assert old in positions, 'Forward reference remains invalid'
                op[field] = '@' + str(positions[old])
        needed = ([op['left'], op['right']] if op['kind'] in ['ratio', 'difference', 'sum', 'product'] else
                  [op['column']] if op['kind'] in ['signed_log', 'square', 'bin5', 'group_mean', 'flag', 'clip', 'model_repair', 'drop_outlier', 'unit_scale'] else [])
        for col in needed:
            if col.startswith('@') or col in numeric or col not in available or col not in numeric_hints: continue
            parse_numeric(frame[col])  # No coercion of invalid observed tokens.
            cast = {'kind': 'numeric_parse', 'column': col}
            operations.append(cast); origins.append('code_documented_lossless_type_completion'); numeric.add(col)
            inserted.append({'operation': cast, 'before_original_step': original_position,
                'source': 'public numeric definition plus actual input dtype; no labels or scores'})
        operations.append(op); origins.append('language_model_original_operation')
        positions[original_position] = len(operations)
        if op['kind'] in ['numeric_parse', 'ordinal']: numeric.add(op['column'])
        if op['kind'] in ['onehot', 'drop', 'date_parts']:
            available.discard(op['column']); numeric.discard(op['column'])
        assert len(operations) <= 4, 'Type completion would exceed the fixed four-step budget'
    completed['operations'] = operations
    return completed, {'inserted': inserted, 'operation_origins': origins,
        'original_model_operations_preserved_separately': deepcopy(answer['operations']),
        'not_an_independently_generated_model_plan_or_teacher': True}
