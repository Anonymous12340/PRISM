"""Shared, score-blind constraints for the original Full operation-object schema.

No completion rewriting or retry: the next operation's grammar is restricted by
already emitted operations. Keep the original JSON grammar as an intersection.
"""
from copy import deepcopy
import json

PROTOCOL = 'full_explicit_dependency_v1'
FEATURES = {'flag','signed_log','square','bin5','ratio','difference','sum','product','frequency','group_mean','affine'}
FIELDS = {'column','left','right','group'}
DELETES = {'drop','onehot','date_parts'}


def remaining_branches(schema, completed):
    """Intersect original parameter enums with availability, prior references and uniqueness."""
    original = schema['properties']['operations']['items']['anyOf']
    available = {v for b in original for k,p in b['properties'].items() if k in FIELDS
                 for v in p.get('enum',[]) if isinstance(v,str) and not v.startswith('@')}
    references = set()
    for i,op in enumerate(completed,1):
        if op['kind'] in DELETES: available.discard(op['column'])
        if op['kind'] in FEATURES: references.add('@'+str(i))
    result=[]
    for template in original:
        branch=deepcopy(template)
        for k,p in branch['properties'].items():
            if k in FIELDS:
                p['enum']=[v for v in p['enum'] if v in available or v in references]
        if any(p.get('enum')==[] for p in branch['properties'].values()): continue
        variants=[branch]
        for banned in completed:
            next_variants=[]
            for variant in variants:
                props=variant['properties']
                if set(props)!=set(banned) or any(v not in props[k].get('enum',[]) for k,v in banned.items()):
                    next_variants.append(variant); continue
                # Disjoint first-different-field partitions remove one exact object
                # without requiring unsupported JSON Schema `not` or `uniqueItems`.
                fixed=deepcopy(variant)
                for field in props:
                    values=fixed['properties'][field]['enum']
                    alternatives=[v for v in values if v!=banned[field]]
                    if alternatives:
                        piece=deepcopy(fixed);piece['properties'][field]['enum']=alternatives
                        next_variants.append(piece)
                    fixed['properties'][field]['enum']=[banned[field]]
            variants=next_variants
        result.extend(variants)
    return result


def dependency_parser(data, schema):
    from lmformatenforcer import JsonSchemaParser, CharacterLevelParser, CharacterLevelParserConfig
    config=CharacterLevelParserConfig(alphabet=data.tokenizer_alphabet)
    branch_cache={}
    def branches(completed):
        token=json.dumps(completed,sort_keys=True,ensure_ascii=False)
        if token not in branch_cache:branch_cache[token]=remaining_branches(schema,completed)
        return branch_cache[token]

    class DependencyParser(CharacterLevelParser):
        def __init__(self, base, depth=0, in_string=False, escape=False, last='', root_key=None,
                     key_text=None, in_operations=False, completed=(), op_parser=None, op_text=''):
            super().__init__(config)
            self.base=base;self.depth=depth;self.in_string=in_string;self.escape=escape;self.last=last
            self.root_key=root_key;self.key_text=key_text;self.in_operations=in_operations
            self.completed=completed;self.op_parser=op_parser;self.op_text=op_text

        def get_allowed_characters(self):
            allowed=self.base.get_allowed_characters()
            if self.op_parser is not None:
                permitted=set(self.op_parser.get_allowed_characters())
                return ''.join(c for c in allowed if c in permitted)
            if self.in_operations and not branches(self.completed):
                return ''.join(c for c in allowed if c not in '{,')
            return allowed

        def add_character(self, c):
            # No mutations of parser states shared by the token-prefix trie.
            depth=self.depth;inside=self.in_string;escape=self.escape;last=self.last
            root_key=self.root_key;key_text=self.key_text;in_ops=self.in_operations
            completed=self.completed;op=self.op_parser;text=self.op_text
            if op is not None:
                op=op.add_character(c);text+=c
            elif in_ops and not inside and depth==2 and c=='{':
                choices=branches(completed)
                assert choices, 'No legal next operation'
                op=JsonSchemaParser({'anyOf':choices},config=config).add_character(c);text=c
            base=self.base.add_character(c)
            if inside:
                if key_text is not None:key_text+=c
                if escape:escape=False
                elif c=='\\':escape=True
                elif c=='"':
                    inside=False
                    if key_text is not None:root_key=json.loads(key_text);key_text=None
            else:
                if c=='"':
                    inside=True
                    if depth==1 and last in '{,':key_text='"'
                elif c in '[{':
                    if c=='[' and depth==1 and root_key=='operations':in_ops=True
                    depth+=1
                elif c in ']}':
                    if c==']' and depth==2 and in_ops:in_ops=False
                    depth-=1
                if c.strip():last=c
            if op is not None and not inside and c=='}' and depth==2:
                assert op.can_end()
                completed=(*completed,json.loads(text));op=None;text=''
            return DependencyParser(base,depth,inside,escape,last,root_key,key_text,in_ops,completed,op,text)

        def can_end(self):return self.base.can_end()
        def shortcut_key(self):
            return self.base.shortcut_key() if not self.in_operations else None

    return DependencyParser(JsonSchemaParser(schema,config=config))


def dependency_prefix(data, schema):
    from lmformatenforcer import TokenEnforcer
    enforcer=TokenEnforcer(data,dependency_parser(data,schema))
    def allowed(batch_id,ids):return enforcer.get_allowed_tokens(ids.tolist()).allowed_tokens
    return allowed


def install():
    """Wrap the normal frozen-weight loader; all callers use the same constraints."""
    from taskdq.planner import Planner
    original=Planner.load
    def load(self):
        original(self)
        self.json_prefix=dependency_prefix
        self.dependency_decoder_protocol=PROTOCOL
    Planner.load=load


def check_operations(operations, columns):
    """Independent post-generation invariant; does not fix, insert or delete actions."""
    available=set(columns);seen=[];refs=set()
    for pos,op in enumerate(operations,1):
        assert op not in seen, 'Repeated operation'
        for field in FIELDS & set(op):
            assert op[field] in available or op[field] in refs, 'Unavailable operation input: '+op[field]
        if op['kind'] in DELETES:available.discard(op['column'])
        if op['kind'] in FEATURES:refs.add('@'+str(pos))
        seen.append(op)
    return True
