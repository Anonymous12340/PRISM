"""Authorized E6 inference-only adapter around the immutable final TaskDQ release."""
import argparse
from copy import deepcopy
import fcntl
import importlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/e6_core_current_20261120'
OLD = ROOT / 'results/e2_unified_independent_20261107'
RELEASE = ROOT / 'e4_current_src_20261119'
SOURCE = OUT / 'source'
NAMES = ['banknote-authentication', 'vote', 'sick', 'spambase',
         'kin8nm', 'cps_85_wages', 'diamonds', 'house_sales']
SEEDS = [42, 123, 456]
REFS = ['raw', 'standard', 'openfe', 'autofeat', 'no_lm_search']
ALLOWED_GPUS = [0,1,5,6,7]


def read(p): return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    q.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def copied(old, new, receipt=None):
    old, new = Path(old), Path(new)
    new.parent.mkdir(parents=True, exist_ok=True)
    if not new.exists(): shutil.copy2(old, new)
    h = sha(old); assert sha(new) == h, str(new)
    if receipt is not None:
        receipt.append(dict(source=str(old), target=str(new), sha256=h, size=old.stat().st_size))


def setup():
    sys.path.insert(0, str(SOURCE.resolve()))
    importlib.invalidate_caches()
    # Optional inference-only packages never shadow the frozen scoring environment.
    for p in ['tmp/e6_inference_deps_20261110', 'tmp/tauprism_learning_deps_20261102']:
        if str(ROOT / p) not in sys.path: sys.path.append(str(ROOT / p))


def check_core():
    record = read(OUT / 'frozen_source_identity.json')
    for r in record['files']: assert sha(SOURCE / r['path']) == r['sha256'], r['path']
    return len(record['files'])


def unit(condition, name, seed): return OUT / 'units' / condition / name / str(seed)
def old_unit(name, seed): return OLD / 'units' / name / str(seed) / 'final'


def condition_config(condition, seed):
    c = read(OUT / 'run_config.json')
    cond = next(v for v in c['conditions'] if v['id'] == condition)
    c.update(seed=seed, model_path=cond['model_path'], inference_dtype='float16',
             inference_device='cuda:0', unit_physical_limit=10, e6_condition=cond)
    for k in ['adapter_path', 'weight_variant', 'weight_revision']:
        c.pop(k, None)
    if cond['trained']:
        c.update(adapter_path=c['frozen_adapter_path'], weight_variant='current_frozen_4B',
                 weight_revision='e1_current_frozen_full_20261118')
    return c


def prepare():
    if (OUT / 'preparation_complete.json').exists():
        check_core(); print('Existing E6 preparation preserved'); return
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted((RELEASE / 'taskdq').rglob('*.py')):
        if p.name.startswith('._'): continue
        target = SOURCE / p.relative_to(RELEASE)
        copied(p, target)
        files.append(dict(path=str(target.relative_to(SOURCE)), sha256=sha(p)))
    write(OUT / 'frozen_source_identity.json', dict(files=files, passed=True, modified_core=False))
    setup()
    import joblib
    import numpy as np
    from taskdq.e2_unified import UnifiedScorer, code_candidates, NATIVE
    from taskdq.e2_pilot import key, RAW
    c = read(ROOT / 'results/e1_current_frozen_full_20261118/run_config.json')
    original_config = read(ROOT / 'results/e1_current_frozen_full_20261118/run_config.json')
    for k in ['model', 'preprocessing', 'operator_version', 'method_revision', 'field_metadata', 'inner_folds']:
        assert c[k] == original_config[k], k
    registry = read(ROOT / 'configs/e2_unified_datasets_20261107.json')['datasets']
    c.update(output_dir=str(OUT), budgets=[10], main_budget=10, lm_budget=10,
             inference_dtype='float16', strict_replay_existing_calls=True, unit_physical_limit=10,
             seeds=SEEDS, datasets=[next(d for d in registry if d['name'] == n) for n in NAMES],
             frozen_adapter_path=str(ROOT / 'results/task_c_sft_v2_20261104/training/adapter'), e6_version='e6_scale_quant_v1',
             planning_scope='complete_development_only', experiment='E6',
             resources=dict(gpu_workers=4, cpu_threads_per_worker=2, max_calls_formal=2640), authorized_gpus=ALLOWED_GPUS)
    c['generation']['dtype'] = 'float16'
    c['native_timeouts']['autofeat'] = 21600
    conditions = []
    for size in ['0.6', '1.7', '4', '8']:
        for precision in ['fp16', 'nf4']:
            conditions.append(dict(id='qwen' + size.replace('.', 'p') + '_' + precision,
                                   model_path='/root/fshare/models/Qwen/Qwen3-' + size + 'B',
                                   nominal_billions=float(size), precision=precision, trained=False, panel='scale'))
    for precision in ['fp16', 'nf4']:
        conditions.append(dict(id='qwen4_trained_' + precision,
                               model_path='/root/fshare/models/Qwen/Qwen3-4B', nominal_billions=4.,
                               precision=precision, trained=True, panel='deployment'))
    conditions.append(dict(id='phi3p5_fp16', model_path='/root/fshare/models/microsoft/Phi-3.5-mini-instruct',
                           nominal_billions=3.8, precision='fp16', trained=False, panel='family'))
    identities = {}
    for cond in conditions:
        p = Path(cond['model_path'])
        if str(p) not in identities:
            meta = []
            for q in sorted(p.iterdir()):
                if q.is_file() and (q.suffix in ['.json', '.safetensors', '.model', '.txt'] or q.name in ['LICENSE', 'README.md']):
                    meta.append(dict(name=q.name, bytes=q.stat().st_size, sha256=sha(q)))
            assert any(x['name'].endswith('.safetensors') for x in meta)
            cfg = read(p / 'config.json')
            identities[str(p)] = dict(files=meta, identity=hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(),
                                       model_type=cfg['model_type'], max_position_embeddings=cfg['max_position_embeddings'],
                                       upstream_revision='not present in local config; exact local file hashes define this run')
        cond['model_identity'] = identities[str(p)]['identity']
        if cond['trained']:
            cond['adapter_identity'] = sha(Path(c['frozen_adapter_path']) / 'adapter_model.safetensors')
        cond['quantization'] = (dict(load_in_4bit=True, quant_type='nf4', double_quant=True, compute_dtype='float16')
                                if cond['precision'] == 'nf4' else None)
    c['conditions'] = conditions
    write(OUT / 'model_identity.json', identities)
    write(OUT / 'run_config.json', c)
    
    failure_revision = RELEASE / 'run_records/final_fit_failure_revision.json'
    if failure_revision.exists(): copied(failure_revision, OUT / 'final_fit_failure_revision.json')
    all_reuse = []; cohorts = []
    for name in NAMES:
        for seed in SEEDS:
            old = old_unit(name, seed); common = OUT / 'common' / name / str(seed)
            assert old.is_dir()
            for filename in ['source.joblib', 'inner_splits.json', 'menu.json']:
                copied(old / filename, common / filename, all_reuse)
            sc = UnifiedScorer({**c, 'seed': seed}, common)
            candidates = code_candidates(sc)
            old_pool = read(old / 'methods/ours/candidate_pools/1.json')['code']['pool']
            allowed = {key(RAW), key(NATIVE), *(key(v['spec']) for v in candidates)}
            for value in old_pool:
                spec = value['spec']
                assert key(spec) in allowed or (spec['kind'] == 'native_joint' and spec['ids'][0] == NATIVE['ids'][0])
            tokens = {key(v['spec']) for v in old_pool}
            assert allowed <= tokens
            assert not set(sc.x.index) & set(read(ROOT / 'data/e2_unified_independent_20261107/prepared' / name / str(seed) / 'split.json')['test_row_ids'])
            for fold in sc.folds:
                assert not set(sc.source['groups'].iloc[fold['train']]) & set(sc.source['groups'].iloc[fold['validation']])
            for token in tokens:
                copied(old / 'scores' / (token + '.json'), common / 'scores' / (token + '.json'), all_reuse)
                for fold in range(3):
                    oldfold = old / 'inner_models' / token / str(fold)
                    for filename in ['predictions.npy', 'fit.json', 'expanded_check.json', 'transform_state.joblib']:
                        q = oldfold / filename
                        if q.exists(): copied(q, common / 'inner_models' / token / str(fold) / filename, all_reuse)
            source = joblib.load(common / 'source.joblib')
            assert source['seed'] == seed and len(sc.folds) == 3
            cohorts.append(dict(dataset=name, seed=seed, rows=len(sc.x), fields=len(sc.x.columns),
                                code_tokens=sorted(tokens), source_sha256=sha(common / 'source.joblib'),
                                inner_splits_sha256=sha(common / 'inner_splits.json')))
            for cond in conditions:
                target = unit(cond['id'], name, seed); target.mkdir(parents=True, exist_ok=True)
                reused = []
                for q in common.rglob('*'):
                    if q.is_file(): copied(q, target / q.relative_to(common), reused)
                write(target / 'reuse_manifest.json', dict(files=reused, language_model_outputs_copied=0,
                       shared_initial_code_only=True, original_unit=str(old), original_config_sha256=sha(OLD / 'run_config.json')))
                # Existing raw final fit is identical and has no relocatable native state.
                for q in (old / 'final_models' / key(RAW)).rglob('*'):
                    if q.is_file(): copied(q, target / 'final_models' / key(RAW) / q.relative_to(old / 'final_models' / key(RAW)), reused)
                write(target / 'reuse_manifest.json', dict(files=reused, language_model_outputs_copied=0,
                       shared_initial_code_only=True, original_unit=str(old), original_config_sha256=sha(OLD / 'run_config.json')))
                write(target / 'final_registry.json', [])
    write(OUT / 'source_and_cache_audit.json', dict(cohorts=cohorts, copied_files=all_reuse,
           no_new_fits=True, no_lm_calls=True, test_source_reads=0, reused_test_benchmark=True))
    jobs = [dict(condition=d['id'], dataset=n, seed=s) for d in conditions for n in NAMES for s in SEEDS]
    timing = [dict(condition=d['id'], dataset=n, seed=42) for d in conditions for n in ['banknote-authentication', 'diamonds']]
    remaining = [j for j in jobs if j not in timing]
    assert len(jobs) == 264 and len(timing) == 22 and len(remaining) == 242
    write(OUT / 'jobs.json', dict(controlled_timing=timing, parallel=remaining, total=264, logical_limit=2640))
    write(OUT / 'preparation_complete.json', dict(time=time.time(), model_conditions=11, datasets=8,
           data_units=24, model_units=264, logical_limit=2640, frozen_python_files=len(files), fits=0, calls=0,
           test_source_reads=0, planned_reference_selections=120))
    print(json.dumps(read(OUT / 'preparation_complete.json')), flush=True)


def install_loader(c):
    import torch
    from taskdq.planner import Planner
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
    from taskdq.json_decode import tokenizer_data, json_prefix
    cond = c['e6_condition']

    def load(self):
        if self.loaded: return
        started = time.monotonic()
        self.torch = torch; self.GenerationConfig = GenerationConfig; self.json_prefix = json_prefix
        self.device = 'cuda:0'; self.dtype = torch.float16; torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_path'], local_files_only=True)
        kwargs = dict(local_files_only=True, torch_dtype=torch.float16, attn_implementation='sdpa')
        if cond['precision'] == 'nf4':
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16), device_map={'': self.device})
        self.model = AutoModelForCausalLM.from_pretrained(self.config['model_path'], **kwargs)
        if cond['precision'] == 'fp16': self.model = self.model.to(self.device)
        if self.config.get('adapter_path'):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.config['adapter_path'],
                                                  is_trainable=False, autocast_adapter_dtype=False)
            # A 4bit base has integer storage, so PEFT does not inherit its
            # floating dtype. Enforce the preregistered FP16 adapter explicitly.
            for name, parameter in self.model.named_parameters():
                if 'lora_' in name and parameter.dtype != torch.float16:
                    parameter.data = parameter.data.to(dtype=torch.float16)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.tokens = tokenizer_data(self.tokenizer); self.loaded = True
        torch.cuda.synchronize()
        # PEFT's adapter wrapper is also named Linear4bit, but it is not the
        # bitsandbytes quantized base layer and has no compute_dtype attribute.
        if cond['precision'] == 'nf4':
            from bitsandbytes.nn import Linear4bit as QuantizedLinear4bit
            four = [m for m in self.model.modules() if isinstance(m, QuantizedLinear4bit)]
        else:
            four = []
        assert bool(four) == (cond['precision'] == 'nf4')
        if four: assert all(m.compute_dtype == torch.float16 and m.weight.quant_state is not None for m in four)
        lora = [p for n, p in self.model.named_parameters() if 'lora_' in n]
        assert (len(lora) == 144) if cond['trained'] else not lora
        assert all(p.dtype == torch.float16 for p in lora)
        self.e6_load_receipt = dict(condition=cond, load_seconds=time.monotonic()-started,
             linear4bit_modules=len(four), adapter_tensors=len(lora), adapter_dtype='float16' if lora else None,
             parameters_trainable=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
             non_quantized_parameter_dtypes=sorted({str(p.dtype) for p in self.model.parameters() if p.dtype.is_floating_point}),
             peak_load_allocated_bytes=torch.cuda.max_memory_allocated(),
             peak_load_reserved_bytes=torch.cuda.max_memory_reserved(), inference_only=True,
             device_name=torch.cuda.get_device_name(), transformers_attention='sdpa')
    Planner.load = load


def engine(c):
    from taskdq.e2_pilot import Engine
    class E6Engine(Engine):
        def request(self, messages, schema):
            r = super().request(messages, schema)
            r['e6_inference_identity'] = self.c['e6_condition']
            return r

        def generate(self, messages, schema, target, method, number):
            existing=target/'methods'/method/'calls'/f'{number}.json'
            if existing.exists():
                previous=read(existing)
                if previous.get('request') is None and previous.get('interrupted'):
                    from taskdq.e2_pilot import key
                    assert previous['original_request_key']==key(self.request(messages,schema))
                    return previous
            result = super().generate(messages, schema, target, method, number)
            if self.base is not None:
                write(target / 'e6_model_load.json', self.base.e6_load_receipt)
                result['e6_peak_reserved_bytes'] = self.base.torch.cuda.max_memory_reserved()
                if result['physical_call']:
                    from taskdq.e2_pilot import key
                    write(target / 'lm_cache' / (key(result['request']) + '.json'), result)
                    write(target / 'methods' / method / 'calls' / f'{number}.json', result)
            return result
    return E6Engine(c)


def worker(condition, name, seed, phase):
    if os.environ.get('E6_CPU_AFFINITY'):
        os.sched_setaffinity(0, {int(x) for x in os.environ['E6_CPU_AFFINITY'].split(',')})
    setup(); check_core()
    import joblib
    import numpy as np
    import pandas as pd
    from taskdq.e2_unified import UnifiedScorer, run_planner
    from taskdq.e2 import fit_final
    c = condition_config(condition, seed); target = unit(condition, name, seed)
    with (target / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (target / 'complete.json').exists(): return
        if (target / 'runner_failure.json').exists(): raise RuntimeError('Unresolved previous attempt; inspect before resume')
        started = time.time(); owner = dict(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
             condition=condition, dataset=name, seed=seed, phase=phase, started=started)
        write(target / 'execution_ownership.json', owner)
        original = old_unit(name, seed)
        class E6Scorer(UnifiedScorer):
            def native(self, train, valid, y, groups, method, where):
                # Only the immutable, identical raw-data OpenFE component is reusable.
                rel = where.relative_to(self.unit)
                oldwhere = original / rel
                slot = rel.parts[-1]
                if method == 'openfe' and rel.parts[0] == 'native_cache' and not (where / 'native').exists():
                    a = np.arange(len(self.x)) if slot == 'outer' else np.asarray(self.folds[int(slot)]['train'])
                    b = np.arange(2) if slot == 'outer' else np.asarray(self.folds[int(slot)]['validation'])
                    pd.testing.assert_frame_equal(train, self.x.iloc[a]); pd.testing.assert_frame_equal(valid, self.x.iloc[b])
                    assert np.array_equal(np.asarray(y), np.asarray(self.y.iloc[a]))
                    assert (oldwhere / 'native/openfe/generation.json').exists()
                    generation = read(oldwhere / 'native/openfe/generation.json'); assert generation['status'] == 'passed'
                    reused = []
                    for q in (oldwhere / 'native').rglob('*'):
                        # Saved applications are disposable and may contain an old
                        # evaluation batch; only fitted state and train caches move.
                        if q.is_file() and 'application' not in q.name:
                            copied(q, where / 'native' / q.relative_to(oldwhere / 'native'), reused)
                    write(self.unit / 'native_reuse' / (slot + '.json'), dict(files=reused, identical_training_fold=True))
                return super().native(train, valid, y, groups, method, where)
        scorer = E6Scorer(c, target); install_loader(c); lm = engine(c)
        # The synthetic format check is performed once per condition, outside formal N10.
        check = OUT / 'model_checks' / condition
        check.mkdir(parents=True,exist_ok=True)
        with (check/'check.lock').open('a') as check_lock:
            fcntl.flock(check_lock,fcntl.LOCK_EX)
            if not (check / 'complete.json').exists():
                cc = {**c, 'inference_max_new_tokens': 32, 'unit_physical_limit': 1}
                probe = engine(cc)
                schema = dict(type='object', properties=dict(ok=dict(type='boolean')), required=['ok'], additionalProperties=False)
                result = probe.generate([{'role':'user','content':'Return a JSON object with ok equal to true.'}], schema, check, 'synthetic', 1)
                assert result['format_valid'] and result['eos_completed']
                lm.base = probe.base
                write(check / 'complete.json', dict(passed=True, format_only=True, no_fits=True, test_reads=0,
                                                   physical_calls=len(list((check / 'physical_calls').glob('*')))))
        run_planner(c, scorer, lm, 'ours')
        fit_final(c, scorer, only_method='ours')
        reg = read(target / 'final_registry.json'); assert len(reg) == 1
        assert len(list((target / 'physical_calls').glob('*.json'))) <= 10
        previous_cost=read(target/'resume_costs.json') if (target/'resume_costs.json').exists() else dict(attempts=[])
        interruptions=[read(p) for p in (target/'physical_calls').glob('*.json') if read(p)['status']=='interrupted']
        write(target / 'complete.json', dict(**owner, completed=time.time(), wall_seconds=time.time()-started+sum(v['active_seconds'] for v in previous_cost['attempts']),
             interrupted_requests=len(interruptions),resumed_attempts=len(previous_cost['attempts']),current_attempt_wall_seconds=time.time()-started,
             logical_calls=read(target / 'methods/ours/selection.json')['logical_calls'],
             physical_calls=len(list((target / 'physical_calls').glob('*.json'))), status=reg[0]['status'],
             selected_spec=reg[0]['spec'], test_read=False, training_updates=0))
        print(json.dumps(read(target / 'complete.json')), flush=True)


def available_gpus():
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    busy = {x.split(',')[0].strip() for x in processes.splitlines() if x.strip()}
    result = []
    for line in inventory.splitlines():
        i, uuid, memory, utilization = [v.strip() for v in line.split(',')]
        if uuid not in busy and int(memory) < 256 and int(utilization) == 0: result.append(int(i))
    return result


def progress(stage, active=()):
    done = list((OUT / 'units').glob('*/*/*/complete.json'))
    physical = list((OUT / 'units').glob('*/*/*/physical_calls/*.json'))
    failures = list((OUT / 'units').glob('*/*/*/runner_failure.json'))
    write(OUT / 'progress.json', dict(stage=stage, completed_units=len(done), total_units=264,
          physical_calls=len(physical), logical_limit=2640, failures=[str(p.relative_to(OUT)) for p in failures],
          active=list(active), test_evaluated=False, time=time.time(), training_updates=0))


def launch_job(job, gpu, phase):
    target = unit(job['condition'], job['dataset'], job['seed'])
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(SOURCE),
           'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'TOKENIZERS_PARALLELISM':'false'}
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 16
    env['E6_CPU_AFFINITY'] = ','.join(str(x) for x in cpus[2*gpu:2*gpu+2])
    logpath = OUT / 'logs' / job['condition'] / job['dataset'] / f"{job['seed']}.log"
    logpath.parent.mkdir(parents=True, exist_ok=True); log = logpath.open('a')
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', 'worker', '--condition', job['condition'],
               '--dataset', job['dataset'], '--seed', str(job['seed']), '--phase', phase]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return dict(**job, gpu=gpu, pid=process.pid, process=process, log=log, started=time.time())


class AttachedProcess:
    def __init__(self,pid,target): self.pid=pid;self.target=target
    def poll(self):
        try:
            state=(Path('/proc')/str(self.pid)/'stat').read_text().split(') ',1)[1].split()[0]
            if state!='Z': return None
        except FileNotFoundError: pass
        return 0 if (self.target/'complete.json').exists() else 1


def dispatch(jobs, gpus, phase, serial=False):
    queue = [j for j in jobs if not (unit(j['condition'], j['dataset'], j['seed']) / 'complete.json').exists()]
    active = {}; failed = []
    while queue or active:
        busy = {r['gpu'] for r in active.values()}
        free = [g for g in available_gpus() if g in gpus and g not in busy]
        if serial and active: free = []
        if serial: free = free[:1]
        for gpu in free:
            if not queue: break
            j = queue.pop(0)
            if (unit(j['condition'], j['dataset'], j['seed']) / 'runner_failure.json').exists():
                raise RuntimeError('A prior failed unit needs engineering inspection')
            r = launch_job(j, gpu, phase); active[r['pid']] = r
        for pid, r in list(active.items()):
            code = r['process'].poll()
            if code is None: continue
            r['log'].close(); del active[pid]
            if code or not (unit(r['condition'], r['dataset'], r['seed']) / 'complete.json').exists():
                failed.append({k:v for k,v in r.items() if k not in ['process','log']})
        progress(phase, [{k:v for k,v in r.items() if k not in ['process','log']} for r in active.values()])
        if failed and not active:
            write(OUT / 'engineering_hold.json', dict(failures=failed, remaining=queue, phase=phase, time=time.time()))
            raise RuntimeError('Engineering inspection required; no failed request is automatically repeated')
        if failed: queue = []
        if queue or active: time.sleep(10)


def controller():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(OUT / 'controller_ownership.json', dict(pid=os.getpid(), time=time.time(),stage='waiting_for_preparation'))
        while not (OUT/'preparation_complete.json').exists():
            if (OUT/'prepare_failure.json').exists():raise RuntimeError('Preparation failed; inspect before generating')
            progress('preparing_validated_inputs')
            time.sleep(10)
        check_core()
        if (OUT / 'engineering_hold.json').exists(): raise RuntimeError('Inspect existing engineering hold')
        if not (OUT / 'launch.json').exists():
            write(OUT / 'launch.json', dict(controller_pid=os.getpid(), started=time.time(), total_units=264,
                 logical_limit=2640, authorized_gpus=ALLOWED_GPUS, no_training=True))
        write(OUT / 'controller_ownership.json', dict(pid=os.getpid(), time=time.time()))
        jobs = read(OUT / 'jobs.json')
        # Latest user instruction: only 0/3/4/5; other users now have 1/2 and 6/7.
        # The same 22 timing units retain a fixed GPU/CPU lane while three lanes run.
        timing = [j for j in jobs['controlled_timing'] if not (unit(j['condition'],j['dataset'],j['seed'])/'complete.json').exists()]
        parallel = [j for j in jobs['parallel'] if not (unit(j['condition'],j['dataset'],j['seed'])/'complete.json').exists()]
        active = {}; failures = []
        for j in timing+parallel:
            target=unit(j['condition'],j['dataset'],j['seed']);ownership=target/'execution_ownership.json'
            if not ownership.exists():continue
            owner=read(ownership);pid=owner['pid']
            try:command=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().replace('\x00',' ')
            except FileNotFoundError:continue
            process=AttachedProcess(pid,target)
            if process.poll() is not None:continue
            assert 'run_e6_20261110.py' in command and '--stage worker' in command
            assert int(owner['gpu']) in ALLOWED_GPUS
            active[pid]=dict(**j,gpu=int(owner['gpu']),pid=pid,started=owner['started'],
                            process=process,log=open(os.devnull,'w'),adopted=True)
        attached={(r['condition'],r['dataset'],r['seed']) for r in active.values()}
        timing=[j for j in timing if (j['condition'],j['dataset'],j['seed']) not in attached]
        parallel=[j for j in parallel if (j['condition'],j['dataset'],j['seed']) not in attached]
        while timing or parallel or active:
            busy = {r['gpu'] for r in active.values()}
            for gpu in available_gpus():
                if gpu in busy or gpu not in ALLOWED_GPUS: continue
                if gpu == 0 and timing: job = timing.pop(0); phase = 'fixed_gpu_timing_under_parallel_load'
                elif parallel: job = parallel.pop(0); phase = 'parallel_evaluation'
                else: continue
                if (unit(job['condition'],job['dataset'],job['seed'])/'runner_failure.json').exists():
                    raise RuntimeError('Prior worker failure requires engineering inspection')
                r = launch_job(job,gpu,phase); active[r['pid']] = r
            for pid,r in list(active.items()):
                code = r['process'].poll()
                if code is None: continue
                r['log'].close();del active[pid]
                if code or not (unit(r['condition'],r['dataset'],r['seed'])/'complete.json').exists():
                    failures.append({k:v for k,v in r.items() if k not in ['process','log']})
            progress('parallel_with_fixed_timing_lane', [{k:v for k,v in r.items() if k not in ['process','log']} for r in active.values()])
            if failures:
                write(OUT/'engineering_hold.json',dict(failures=failures,pending_timing=timing,pending_parallel=parallel,
                      active=[{k:v for k,v in r.items() if k not in ['process','log']} for r in active.values()],time=time.time()))
                # Healthy workers keep their original PIDs; recovery must adopt them.
                raise RuntimeError('Worker failure recorded; healthy in-flight workers were not interrupted')
            if timing or parallel or active: time.sleep(10)
        progress('all_models_frozen')
        command = [sys.executable, '-u', str(ROOT / 'scripts/review_e6_20261110.py')]
        subprocess.run(command, cwd=ROOT, env={**os.environ,'PYTHONPATH':str(SOURCE)}, check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['prepare','controller','worker'], required=True)
    p.add_argument('--condition'); p.add_argument('--dataset'); p.add_argument('--seed', type=int); p.add_argument('--phase')
    args = p.parse_args()
    try:
        if args.stage == 'prepare': prepare()
        elif args.stage == 'controller': controller()
        else: worker(args.condition, args.dataset, args.seed, args.phase)
    except BaseException:
        record = dict(stage=args.stage, error=traceback.format_exc(), time=time.time())
        destination = unit(args.condition,args.dataset,args.seed) / 'runner_failure.json' if args.stage == 'worker' else OUT / (args.stage + '_failure.json')
        write(destination, record); raise
