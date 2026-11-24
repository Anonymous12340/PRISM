"""使用lm-format-enforcer公开核心接口，适配本项目的transformers 5。

tokenizer适配依据官方0.11.3的integrations/transformers.py：
https://github.com/noamgat/lm-format-enforcer/blob/v0.11.3/lmformatenforcer/integrations/transformers.py
该版本的旧transformers导入路径不兼容5.x；这里不修改已安装依赖。
"""


def response_schema(column):
    return {'type': 'object', 'properties': {
        'column': {'type': 'string', 'enum': [column]},
        'outliers': {'type': 'string', 'enum': ['keep', 'handle', 'uncertain']},
        'leakage': {'anyOf': [{'type': 'boolean'}, {'type': 'string', 'enum': ['uncertain']}]},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 160}},
        'required': ['column', 'outliers', 'leakage', 'reason'], 'additionalProperties': False}


def tokenizer_data(tokenizer):
    from lmformatenforcer import TokenEnforcerTokenizerData
    from lmformatenforcer.tokenizerprefixtree import JsonFreetextTokenCache
    # 复用官方的前缀解码口径，保留词首空格；tokenizer词表只预处理一次。
    zero = tokenizer.encode('0', add_special_tokens=False)[-1]
    special = set(tokenizer.all_special_ids)
    tokens = []
    for index in range(len(tokenizer)):
        if index not in special:
            after_zero = tokenizer.decode([zero, index], clean_up_tokenization_spaces=False)[1:]
            alone = tokenizer.decode([index], clean_up_tokenization_spaces=False)
            tokens.append((index, after_zero, len(after_zero) > len(alone)))

    def decode(ids):
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False).rstrip('�')

    data = TokenEnforcerTokenizerData(tokens, decode, tokenizer.eos_token_id, False, len(tokenizer))
    # 0.11.3的自由文本快捷缓存漏检原始制表符等控制字符。
    # 仅修复字符串内的缓存；JSON字符串外的空白仍由原解析树处理。
    cache = JsonFreetextTokenCache(False, len(tokenizer))
    for index, value, _ in tokens:
        if not any(ord(char) < 32 for char in value):
            cache.add_token(value, index)
    cache.freeze()
    data.tokenizer_tree.json_freetext_tokens = cache
    return data


def json_prefix(data, schema):
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer, CharacterLevelParserConfig
    # 初始化字符串状态前传入真实词表，避免minLength阶段丢失中文。
    parser = JsonSchemaParser(schema, config=CharacterLevelParserConfig(alphabet=data.tokenizer_alphabet))
    enforcer = TokenEnforcer(data, parser)

    def allowed(batch_id, ids):
        return enforcer.get_allowed_tokens(ids.tolist()).allowed_tokens

    return allowed
