"""Complete local field decisions within the original ten-call semantic allowance.

Pre-confirmation correction: a local keep does not end whole-table planning.
The initial semantic checkpoint, requests and comparisons are retained.
"""
import argparse
import json
from pathlib import Path
import signal
import joblib
from . import e2_semantic as base
from .e2_pilot import Engine, RAW, key, timeout_handler
from .e2_search import paired_stats, select_pool, fit_selected
from .io import read, write, plain


INSTRUCTION = base.INSTRUCTION + '''
本轮修正逐列覆盖：keep或空数组仅表示当前焦点字段保留原样，不表示整张表停止。代码会转到其他字段，所有调用合计仍最多10次。
反馈中的早期stop是旧接线记录，当前只解释为当时焦点字段的保留。没有确定错误时仍可提出有任务依据的特征假设；不要把“无错误”直接等同“没有可用的任务特征”。不能为了生成新方案随意改原值。'''


def continue_unit(s, c, name, seed, engine):
    unit = base.unit_path(s, name, seed)
    if (unit / 'focus_complete.json').exists(): return read(unit / 'focus_complete.json')
    sc = base.CheckedScorer({**c, 'seed': seed}, unit)
    initial = read(unit / 'semantic_selection.json'); history = read(unit / 'semantic_history.json')
    pool = {key(v['spec']): v for v in initial['candidate_results']}; raw = pool[key(RAW)]
    best = raw
    for value in pool.values():
        if value['utility'] > best['utility'] + 1e-12: best = value
    known = {key(RAW), *(key(v['spec']) for v in history if 'spec' in v)}
    original_count = len(history)
    assert original_count <= s['calls_per_unit']
    for number in range(original_count + 1, s['calls_per_unit'] + 1):
        choices = base.options(sc, best, known)
        if not choices: break
        context = base.compact_context(sc, best, raw, history, number, s)
        context['keep_scope'] = 'current_focus_field_only'
        schema = {'type': 'object', 'properties': {'diagnosis': {'type': 'string', 'enum': ['missing', 'outlier_candidate', 'none', 'uncertain']},
                  'reason': {'type': 'string', 'minLength': 1, 'maxLength': s['reason_max_characters']},
                  'operations': {'type': 'array', 'items': {'type': 'string', 'enum': ['keep', *choices]}, 'maxItems': 1}},
                  'required': ['diagnosis', 'reason', 'operations'], 'additionalProperties': False}
        record = engine.generate([{'role': 'system', 'content': INSTRUCTION},
            {'role': 'user', 'content': json.dumps(plain(context), ensure_ascii=False)}], schema, unit, 'semantic_focus', number)
        entry = {'round': number, 'format_valid': record['format_valid'], 'executable': False,
                 'focus_column': context['input']['focus_column'], 'input_tokens': record.get('input_tokens'),
                 'diagnosis': record.get('answer', {}).get('diagnosis')}
        try:
            if not record['format_valid']: raise ValueError('Invalid JSON output')
            chosen = (record['answer']['operations'] or ['keep'])[0]; entry['plan'] = chosen
            if chosen == 'keep': entry.update(executable=True, local_keep=True)
            else:
                spec = choices[chosen]; token = key(spec); assert token not in known; known.add(token); entry['spec'] = spec
                old = signal.signal(signal.SIGALRM, timeout_handler); signal.alarm(c['candidate_timeout'])
                try: result = sc.inner(spec)
                finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)
                pool[token] = result
                entry.update(executable=True, paired_vs_raw=paired_stats(result, raw), scores=result['scores'])
                if result['utility'] > best['utility'] + 1e-12: best = result
        except Exception as exc: entry['error'] = repr(exc)
        history.append(entry); write(unit / 'semantic_focus_history.json', history)
        print({'dataset': name, 'seed': seed, 'focus_completion': entry}, flush=True)
    write(unit / 'semantic_focus_history.json', history)
    mean, robust = select_pool(pool, raw, sc.ops, s)
    write(unit / 'focus_selection.json', {'spec': robust['spec'], 'mean_spec_descriptive': mean['spec'],
          'candidate_results': list(pool.values()), 'selection_frozen_before_confirmation': True})
    result = {'dataset': name, 'seed': seed, 'passed': True, 'calls_combined': len(history),
              'initial_checkpoint_calls': original_count, 'added_calls': len(history) - original_count,
              'distinct_executable_plans': len(pool) - 1, 'confirmation_opened': False}
    assert len(history) <= 10
    write(unit / 'focus_complete.json', result); return result


def refresh_unit(s, c, name, seed):
    unit = base.unit_path(s, name, seed)
    if (unit / 'focus_refit.json').exists(): return
    assert read(unit / 'fitted.json')['passed'] and read(unit / 'focus_complete.json')['passed']
    sc = base.CheckedScorer({**c, 'seed': seed}, unit)
    plans = read(unit / 'selection.json')['plans']; assert 'semantic_v1' not in plans
    write(unit / 'selection_pre_focus.json', read(unit / 'selection.json'))
    plans['semantic_v1'] = plans['semantic']; proposed = read(unit / 'focus_selection.json')['spec']
    plans['semantic'] = proposed
    try:
        fit_selected({**c, 'seed': seed}, sc, proposed)
        folder = unit / 'final_models' / key(proposed)
        checks = base.check_steps(sc.x, sc.x.iloc[:2], joblib.load(folder / 'transform_state.joblib')['steps'])
        write(folder / 'focus_final_check.json', {'passed': True, 'checks': checks})
    except ValueError as exc:
        if 'consistency violations' not in str(exc): raise
        write(unit / 'focus_final_fallback.json', {'attempted': proposed, 'fallback': RAW, 'error': repr(exc), 'before_confirmation': True})
        plans['semantic'] = RAW
    write(unit / 'selection.json', {'plans': plans, 'confirmation_opened': False, 'focus_coverage_completed': True})
    write(unit / 'focus_refit.json', {'passed': True, 'methods': len(plans), 'new_baseline_fits': 0, 'confirmation_opened': False})


def freeze(s, c):
    out = Path(s['output_dir'])
    for name in s['datasets']:
        for seed in s['seeds']:
            unit = base.unit_path(s, name, seed); assert read(unit / 'focus_refit.json')['passed']
            h = read(unit / 'semantic_focus_history.json'); assert len(h) <= 10
            specs = [key(v['spec']) for v in h if 'spec' in v]; assert len(specs) == len(set(specs))
    hold = out / 'hold_before_confirmation.json'
    if hold.exists(): hold.rename(out / 'hold_resolved_before_confirmation.json')
    base.freeze(s, c)
    record = read(out / 'all_selections_frozen.json'); record.update(methods=9, focus_coverage_corrected=True)
    record['initial_checkpoint_records'] = record['records']
    record['records'] = [read(base.unit_path(s, name, seed) / 'focus_complete.json') for name in s['datasets'] for seed in s['seeds']]
    write(out / 'all_selections_frozen.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True, choices=['continue', 'refit', 'freeze', 'confirm', 'report'])
    parser.add_argument('--lane', type=int, default=0); args = parser.parse_args(); s, c = base.settings(args.config)
    matrix = [(n, z) for n in s['datasets'] for z in s['seeds']]
    if args.stage == 'continue':
        engine = Engine(c)
        for name, seed in matrix[args.lane::8]: print(continue_unit(s, c, name, seed, engine), flush=True)
    elif args.stage == 'refit':
        for name, seed in matrix[args.lane::8]: refresh_unit(s, c, name, seed)
    elif args.stage == 'freeze': freeze(s, c)
    elif args.stage == 'confirm': base.confirm(s, c)
    else:
        from .e2_semantic_report import report
        report({**s, 'focus_coverage_corrected': True}, c)


if __name__ == '__main__': main()
