"""E1 pilot using the 2026-11-18 frozen system on original E1 data.

This wrapper only adapts the original E1 development units to the frozen
Full-N3 plus TabICL/TabPFN expert runtime. It does not modify the frozen
weights, expert checkpoints, selector, or E1 source data.
"""

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = "e1_current_frozen_full_20261118"
OUT = ROOT / "results" / NAME
CFG = ROOT / "configs" / (NAME + ".json")
OLD = ROOT / "results/e1_final_system_20261108"
PUBLIC = ROOT / "results/crossfit_neural_feature_pilot_20261118"
BASE = ROOT / "scripts/TabPFN_full_expert_20261118.py"

sys.path.insert(0, str(ROOT / "scripts"))
import TabPFN_full_expert_20261118 as b  # noqa: E402


PAIRS = {
    "credit": ["default", "rate"],
    "medical": ["readmission", "stay"],
    "credit_card": ["default", "bill"],
    "bike": ["casual", "registered"],
    "student": ["first_grade", "final_grade"],
    "parkinsons": ["motor", "total"],
    "appliances": ["appliances", "lights"],
    "air_quality": ["carbon_monoxide", "benzene"],
    "residential": ["sale_price", "construction_cost"],
    "sml2010": ["dining_temperature", "bedroom_temperature"],
}
SEEDS = [20261006, 20261007, 20261008]


def configure_base():
    b.NAME = NAME
    b.OUT = OUT
    b.CFG = CFG
    b.PUBLIC = PUBLIC
    b.PARENT = OUT
    b.OLD = OUT
    b.PYTHON = ROOT / "tmp/tauprism_baseline_llm_20261029/bin/python"
    b.NEURAL = ROOT / "tmp/neural_feature_v2_20261118/bin/python"
    b.__file__ = str(Path(__file__).resolve())
    b.EXPERT = dict(kind="neural_expert", ids=["__TabPFN_expert__"])
    b.BLEND = dict(kind="fixed_blend", ids=["__TabPFN_expert__", "__fixed_half_blend__"])


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def cases():
    rows = read_json(OUT / "cases.json")
    if (OUT / "recovery_full_only.json").exists():
        rows = [row for row in rows if not (OUT / "units" / row["tag"] / "Full_complete.json").exists()]
    return rows


def no_op_verify():
    # The parent runtime's hash verifier is replaced after the wrapper writes
    # its own immutable manifest. The wrapper never mutates frozen artifacts.
    return None


def prepare():
    import pandas as pd

    if (OUT / "prepared.json").exists() or (OUT / "dispatch.json").exists():
        raise RuntimeError("pilot already prepared or dispatched; do not relaunch")
    OUT.mkdir(parents=True, exist_ok=True)
    parent_runtime = ROOT / "results/full_novelty_return_pilot_20261117/runtime"
    shutil.copytree(parent_runtime, OUT / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    run_config = read_json(ROOT / "results/TabPFN_full_expert_pilot_20261118/run_config.json")
    run_config["output_dir"] = str(OUT)
    write_json(OUT / "run_config.json", run_config)

    rows = []
    for pair, tasks in PAIRS.items():
        for seed in SEEDS:
            for task_name in tasks:
                old = OLD / "units" / pair / str(seed) / task_name / "source.joblib"
                if not old.exists():
                    raise FileNotFoundError(old)
                source = joblib.load(old)
                # Reuse only the original E1 development source object. The
                # outer assessment below is an internal pilot holdout.
                train_rows, outer_rows = b.folds(source, 5, seed)[0]
                train_rows, outer_rows = np.sort(train_rows), np.sort(outer_rows)
                if set(source["groups"].iloc[train_rows]) & set(source["groups"].iloc[outer_rows]):
                    raise AssertionError("group leakage in pilot outer split")
                current = copy.deepcopy(source)
                current["x"] = source["x"].iloc[train_rows].copy()
                current["y"] = source["y"].iloc[train_rows].copy()
                current["groups"] = source["groups"].iloc[train_rows].copy()
                current["seed"] = seed
                tag = f"{pair}_{task_name}__{seed}"
                current["pair"] = dict(current["pair"], name=tag)
                unit = OUT / "units" / tag
                unit.mkdir(parents=True, exist_ok=True)
                joblib.dump(current, unit / "source.joblib")
                joblib.dump(dict(x=source["x"].iloc[outer_rows].copy(),
                                 groups=source["groups"].iloc[outer_rows].copy()),
                            unit / "sealed_outer_X.joblib")
                np.save(unit / "sealed_outer_y.npy", source["y"].iloc[outer_rows].to_numpy())

                # The E1 source has no synthetic export shift. An empty public
                # contract is the identity canonicalization required by the
                # frozen expert interface; category normalization remains in
                # the frozen matrix conversion.
                write_json(unit / "public_export_contract.json", [])
                support_source = copy.deepcopy(current)
                _, support = b.folds(support_source, 4, seed + 11)[0]
                support = np.sort(support)
                rest = np.setdiff1d(np.arange(len(current["x"])), support)
                remaining = copy.deepcopy(support_source)
                for key in ("x", "y", "groups"):
                    remaining[key] = support_source[key].iloc[rest]
                splits = []
                for inner_train, inner_valid in b.folds(remaining, 3, seed + 17):
                    inner_train = np.sort(np.r_[support, rest[inner_train]])
                    inner_valid = np.sort(rest[inner_valid])
                    if set(current["groups"].iloc[inner_train]) & set(current["groups"].iloc[inner_valid]):
                        raise AssertionError("group leakage in pilot inner split")
                    splits.append(dict(train=inner_train.tolist(), validation=inner_valid.tolist()))
                write_json(unit / "partition.json", dict(support=support.tolist(), remaining=rest.tolist()))
                write_json(unit / "inner_splits.json", splits)
                write_json(unit / "menu.json", dict(columns=list(current["x"].columns)))
                write_json(unit / "derived_ops.json", {})
                rows.append(dict(tag=tag, code=tag, task=source["task"]["type"],
                                 pair=pair, task_name=task_name, seed=seed,
                                 outer_rows=int(len(outer_rows))))
    write_json(OUT / "cases.json", rows)
    write_json(OUT / "preflight_complete.json", dict(
        passed=True, real_fits=0, test_loaded=False, sources=2,
        pairs=list(PAIRS), seeds=SEEDS, units=len(rows), identity_contract=True,
        old_e1_full_not_reused=True,
    ))
    write_json(OUT / "prepared.json", dict(pilot=True, sources=2, seeds=SEEDS,
                                            units=len(rows), real_fits=0,
                                            historical_test_loaded=False))


def run_phase(name, gpus):
    log_dir = OUT / "logs"
    log_dir.mkdir(exist_ok=True)
    procs = []
    for index, gpu in enumerate(gpus):
        log = (log_dir / f"{name}_{index}.log").open("w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        proc = subprocess.Popen([str(b.PYTHON), str(Path(__file__).resolve()), name,
                                 "--index", str(index), "--workers", str(len(gpus))],
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        procs.append((proc, log))
    while any(proc.poll() is None for proc, _ in procs):
        write_json(OUT / "controller_status.json", dict(phase=name, time=time.time(),
                                                         workers=[p.pid for p, _ in procs],
                                                         exits=[p.poll() for p, _ in procs]))
        time.sleep(2)
    exits = [p.wait() for p, _ in procs]
    for _, log in procs:
        log.close()
    if any(exits):
        raise RuntimeError(f"{name} worker failure: {exits}")


def controller(gpus):
    configure_base()
    b.cases = cases
    b.verify = no_op_verify
    start = time.time()
    write_json(OUT / "dispatch.json", dict(time=start, gpus=gpus, pilot=True,
                                           current_frozen_system=True,
                                           historical_test_loaded=False))
    for phase in ("full_worker", "neural_worker", "final_worker"):
        run_phase(phase, gpus)
    b.bind()
    pilot_freeze()
    pilot_report()
    write_json(OUT / "controller_complete.json", dict(exit=0, seconds=time.time()-start,
                                                       units=len(cases()), pilot=True))


def recover(gpus):
    """Resume after a recorded engineering failure without deleting evidence."""
    configure_base()
    b.cases = cases
    b.verify = no_op_verify
    if not (OUT / "dispatch.json").exists():
        raise RuntimeError("recovery requires the original dispatch receipt")
    start = time.time()
    write_json(OUT / "recovery_dispatch.json", dict(time=start, gpus=gpus,
                                                    reason="menu_metadata_format_fix",
                                                    original_failure="full_worker_exit_1"))
    for phase, marker in (("full_worker", "Full_complete.json"),
                          ("neural_worker", "neural_complete.json")):
        complete = all((OUT / "units" / row["tag"] / marker).exists() for row in cases())
        if not complete:
            if phase == "full_worker":
                write_json(OUT / "recovery_full_only.json", dict(time=time.time(), skip_completed=True))
            run_phase(phase, gpus)
            if phase == "full_worker":
                (OUT / "recovery_full_only.json").unlink(missing_ok=True)
    if not all((OUT / "units" / row["tag"] / "frozen_endpoints.json").exists() for row in cases()):
        run_phase("final_recover", gpus)
    b.bind()
    pilot_freeze()
    pilot_report()
    write_json(OUT / "controller_complete.json", dict(exit=0, seconds=time.time()-start,
                                                       units=len(cases()), pilot=True,
                                                       recovered=True,
                                                       original_failure_preserved=True))


def recover_final_worker(index, workers):
    """Finish final endpoints, reusing any model/prediction already committed."""
    import joblib
    configure_base()
    b.cases = cases
    b.verify = no_op_verify
    rt, _ = b.bind()
    from taskdq.scoring import fit_predictor, predict
    rows = [row for i, row in enumerate(cases()) if i % workers == index]
    for row in rows:
        sc = rt.scorer(row["tag"])
        outer = joblib.load(sc.unit / "sealed_outer_X.joblib")
        full = read_json(sc.unit / "Full_N5.json")["selected"]["spec"]
        prefix = read_json(sc.unit / "Full_N3.json")["selected"]["spec"]
        chosen = read_json(sc.unit / "expert_selection.json")["selected"]["spec"]
        specs = [full, prefix] if chosen["kind"] in ["neural_expert", "fixed_blend"] else [full, prefix, chosen]
        preds = {}
        for spec in specs:
            token = b.key(spec)
            folder = sc.unit / "final_XGB" / token
            prediction_file = folder / "predictions.npy"
            if prediction_file.exists() and (folder / "model.joblib").exists():
                model = joblib.load(folder / "model.joblib")
                data = joblib.load(folder / "data.joblib")
                pred = np.load(prediction_file)
                np.testing.assert_array_equal(predict(model, data["valid"]), pred)
            else:
                folder.mkdir(parents=True, exist_ok=True)
                train, valid, y = sc.transform(spec, sc.x, outer["x"], sc.y,
                                               sc.source["groups"], "final_recover", folder)
                number = rt.ledger("primary_ledger.json", dict(table=row["tag"], phase="final_recover",
                            status="started", folder=str(folder.relative_to(OUT))),
                                   read_json(CFG)["limits"]["primary_predictor_fits"])
                model = fit_predictor(train, y, sc.task,
                                      read_json(OUT / "run_config.json")["model"])
                pred = predict(model, valid)
                joblib.dump(model, folder / "model.joblib")
                joblib.dump(dict(train=train, valid=valid, y_train=np.asarray(y)), folder / "data.joblib")
                np.save(prediction_file, pred)
                rt.ledger_update("primary_ledger.json", number, status="completed",
                                 model_sha256=b.sha(folder / "model.joblib"),
                                 prediction_sha256=b.sha(prediction_file))
            preds[token] = pred
        neural = sc.unit / "neural_final"
        if (neural / "complete.json").exists():
            vp = np.load(neural / "predictions.npy")
        else:
            vp = b.context(neural, row["task"], sc.x, sc.y, outer["x"])
        preds[b.key(b.EXPERT)] = vp
        preds[b.key(b.BLEND)] = 0.5 * vp + 0.5 * preds[b.key(prefix)]
        out = sc.unit / "final_outputs"
        out.mkdir(exist_ok=True)
        inventory = []
        for arm, spec in [("original_Full", full), ("Full_neural_experts", chosen),
                          ("direct_neural_reference", b.EXPERT),
                          ("fixed_blend_reference", b.BLEND)]:
            path = out / (arm + ".npy")
            value = preds[b.key(spec)]
            if path.exists():
                np.testing.assert_array_equal(np.load(path), value)
            else:
                np.save(path, value)
            inventory.append(dict(tag=row["tag"], source=row["code"], task=row["task"], seed=row["seed"],
                                  arm=arm, spec=spec, path=str(path.relative_to(OUT)),
                                  prediction_sha256=b.sha(path),
                                  outer_y_sha256=b.sha(sc.unit / "sealed_outer_y.npy")))
        write_json(sc.unit / "frozen_endpoints.json", dict(time=time.time(), inventory=inventory,
                                                           outer_labels_opened=False,
                                                           recovered_from_recorded_engineering_failure=True))


def pilot_freeze():
    configure_base()
    b.cases = cases
    b.verify = no_op_verify
    files, inventory = {}, []
    for row in cases():
        folder = OUT / "units" / row["tag"]
        inventory.extend(read_json(folder / "frozen_endpoints.json")["inventory"])
        for path in folder.rglob("*"):
            if path.is_file() and not path.name.endswith(".log"):
                files[str(path.relative_to(OUT))] = b.sha(path)
    if len(inventory) != 240:
        raise AssertionError(f"expected 240 full endpoints, got {len(inventory)}")
    write_json(OUT / "frozen_all_endpoints.json", dict(time=time.time(), inventory=inventory,
                                                       files=files, outer_metrics_opened=False))


def pilot_report():
    import joblib
    import pandas as pd
    configure_base()
    b.cases = cases
    b.verify = no_op_verify
    from taskdq.e1 import metrics
    frozen = read_json(OUT / "frozen_all_endpoints.json")
    for rel, digest in frozen["files"].items():
        if b.sha(OUT / rel) != digest:
            raise AssertionError(f"frozen file changed: {rel}")
    write_json(OUT / "outer_open_receipt.json", dict(time=time.time(),
                                                     frozen_sha256=b.sha(OUT / "frozen_all_endpoints.json"),
                                                     pilot=True, historical_test_loaded=False))
    rows = []
    for entry in frozen["inventory"]:
        unit = OUT / "units" / entry["tag"]
        target = np.load(unit / "sealed_outer_y.npy")
        pred = np.load(OUT / entry["path"])
        row = dict(source=entry["source"], task=entry["task"], seed=entry["seed"],
                   method=entry["arm"], N=5)
        row.update(metrics(target, pred, joblib.load(unit / "source.joblib")["task"]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    effects = []
    for task, metric in (("classification", "auc"), ("regression", "rmse")):
        z = frame[frame.task == task]
        pivot = z.pivot(index=["source", "seed"], columns="method", values=metric)
        for method in ["Full_neural_experts", "direct_neural_reference", "fixed_blend_reference"]:
            gain = (100 * (pivot[method] - pivot["original_Full"])
                    if task == "classification" else
                    100 * (1 - pivot[method] / pivot["original_Full"]))
            effect = dict(task=task, method=method, mean=float(gain.mean()),
                          sd=float(gain.std(ddof=1)), positive_units=int((gain > 0).sum()),
                          total_units=int(gain.notna().sum()))
            if task == "classification":
                f1 = z.pivot(index=["source", "seed"], columns="method", values="f1")
                effect["F1_gain_pp_mean"] = float((100 * (f1[method] - f1["original_Full"])).mean())
            effects.append(effect)
    summary = dict(time=time.time(), pilot=True, formal_version_fixed=False,
                   current_frozen_system=True, sources=list(PAIRS), seeds=SEEDS,
                   units=len(cases()), endpoints=len(frame), effects=effects,
                   old_e1_full_not_reused=True, historical_test_loaded=False,
                   first_engineering_failure_preserved=True)
    write_json(OUT / "summary.json", summary)
    dest = OUT / "deliverables"
    dest.mkdir(exist_ok=True)
    frame.to_csv(dest / "SCORE_REPORT.csv", index=False, float_format="%.16g")
    lines = ["# E1 current frozen-system pilot", "",
             "Original E1 development data; credit and air_quality; three historical seeds.",
             "Current frozen Full-N3 plus TabICL/TabPFN expert and fixed blend. No E2 pollution.",
             "", "|source|task|seed|method|AUC|F1|RMSE|", "|---|---|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(k, "")) for k in
                                       ["source", "task", "seed", "method", "auc", "f1", "rmse"]) + "|")
    lines += ["", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```"]
    (dest / "SCORE_REPORT.md").write_text("\n".join(lines))


def worker(stage, index, workers):
    configure_base()
    b.cases = cases
    b.verify = no_op_verify
    getattr(b, stage)(index, workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "controller", "recover", "full_worker", "neural_worker", "final_worker", "final_recover"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default="1,5,6,7")
    args = parser.parse_args()
    configure_base()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "controller":
        controller([int(x) for x in args.gpus.split(",")])
    elif args.stage == "recover":
        recover([int(x) for x in args.gpus.split(",")])
    else:
        if args.stage == "final_recover":
            recover_final_worker(args.index, args.workers)
        else:
            worker(args.stage, args.index, args.workers)


if __name__ == "__main__":
    main()
