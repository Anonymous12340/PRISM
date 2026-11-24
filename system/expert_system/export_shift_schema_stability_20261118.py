"""Confirm a frozen export-shift method on the same sources at two new seeds."""

import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import time
import traceback

import export_shift_full_pilot_20261118 as method
import qualify_export_schema_recovery_20261118 as qualification


ROOT = method.ROOT
NAME = "export_shift_schema_stability_20261118"
OUT = ROOT / "results" / NAME
CFG = ROOT / "configs" / (NAME + ".json")
PARENT = ROOT / "results/export_shift_schema_recovery_full_20261118"
read, write, sha = method.read, method.write, method.sha


def configure():
    method.NAME, method.OUT, method.CFG, method.__file__ = NAME, OUT, CFG, __file__
    method.configure()


def prepare():
    import joblib
    import numpy as np
    import pandas as pd

    assert read(PARENT / "summary.json")["pilot_gate"]
    assert not (OUT / "prepared.json").exists() and not (OUT / "dispatch.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PARENT / "runtime", OUT / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    config = read(PARENT / "run_config.json")
    config["output_dir"] = str(OUT)
    write(OUT / "run_config.json", config)
    rt, _ = method.bind()
    from taskdq.e2_parameterized import registry

    protected = dict(read(PARENT / "protected_hashes.json"))
    for path in PARENT.rglob("*"):
        if path.is_file() and (path.name == "model.joblib" or path.suffix == ".npy"
                               or path.name in ("SCORE_REPORT.csv", "summary.json", "frozen_all_endpoints.json")):
            protected[str(path.relative_to(ROOT))] = sha(path)
    cases = []
    for entry in read(PARENT / "qualification/manifest.json")["development"]:
        path = ROOT / entry["path"]
        assert sha(path) == entry["source_sha256"]
        source = joblib.load(path)
        checked, _, numeric = qualification.validate(source, entry["task"])
        assert checked["x"].equals(source["x"]) and checked["y"].equals(source["y"])
        assert numeric == entry["unit_columns"]
        clean = source["x"].copy()
        for column in clean.select_dtypes(exclude="number"):
            values = clean[column].astype("string").str.strip().str.casefold()
            clean[column] = values.astype(object).where(values.notna(), np.nan)
        contract = read(PARENT / "units" / ("R_" + entry["code"] + "__42") / "public_export_contract.json")
        background = joblib.load(PARENT / "units" / ("R_" + entry["code"] + "__42")
                                 / "source.joblib")["pair"]["background"]
        for seed in (123, 456):
            a, z = method.b.folds(source, 5, seed)[0]
            a, z = np.sort(a), np.sort(z)
            assert not set(source["groups"].iloc[a]) & set(source["groups"].iloc[z])
            train = method.mechanism.export(clean.iloc[a], contract, False)
            outer = method.mechanism.export(clean.iloc[z], contract, True)
            canonical_train = method.mechanism.restore(train, contract)
            canonical_outer = method.mechanism.restore(outer, contract)
            pd.testing.assert_frame_equal(canonical_train, clean.iloc[a],
                                          check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
            pd.testing.assert_frame_equal(canonical_outer, clean.iloc[z],
                                          check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
            current = copy.copy(source)
            for name in ("x", "y", "groups"):
                current[name] = source[name].iloc[a]
            _, support = method.b.folds(current, 4, seed + 11)[0]
            support = np.sort(support)
            rest = np.setdiff1d(np.arange(len(a)), support)
            remaining = copy.copy(current)
            for name in ("x", "y", "groups"):
                remaining[name] = current[name].iloc[rest]
            splits = []
            for tr, va in method.b.folds(remaining, 3, seed + 17):
                tr, va = np.sort(np.r_[support, rest[tr]]), np.sort(rest[va])
                assert not set(current["groups"].iloc[tr]) & set(current["groups"].iloc[va])
                splits.append(dict(train=tr.tolist(), validation=va.tolist()))
            for prefix, x, valid in (("R", train, outer), ("N", canonical_train, canonical_outer)):
                tag = prefix + "_" + entry["code"] + "__" + str(seed)
                unit = rt.unit(tag)
                assert not (unit / "source.joblib").exists()
                unit.mkdir(parents=True, exist_ok=True)
                data = copy.deepcopy(current)
                data.update(x=x, seed=seed)
                data["pair"] = dict(data["pair"], name=tag, background=background)
                joblib.dump(data, unit / "source.joblib")
                joblib.dump(dict(x=valid, groups=source["groups"].iloc[z]), unit / "sealed_outer_X.joblib")
                np.save(unit / "sealed_outer_y.npy", source["y"].iloc[z].to_numpy())
                write(unit / "inner_splits.json", splits)
                write(unit / "partition.json", dict(support=support.tolist(), remaining=rest.tolist()))
                write(unit / "public_export_contract.json", contract)
                write(unit / "menu.json", registry(x, data.get("date_columns", {})))
                write(unit / "derived_ops.json", {})
                for file in unit.iterdir():
                    if file.name != "derived_ops.json":
                        protected[str(file.relative_to(ROOT))] = sha(file)
                cases.append(dict(tag=tag, code=entry["code"], task=entry["task"], seed=seed,
                                  outer_y_sha256=sha(unit / "sealed_outer_y.npy")))
    assert len(cases) == 16
    write(OUT / "cases.json", cases)
    for file in (OUT / "runtime").rglob("*.py"):
        protected[str(file.relative_to(ROOT))] = sha(file)
    for path in (OUT / "cases.json", OUT / "run_config.json"):
        protected[str(path.relative_to(ROOT))] = sha(path)
    files = [Path(__file__).resolve(), CFG, ROOT / "designs" / NAME / "PLAN.md",
             ROOT / "scripts/export_shift_full_pilot_20261118.py",
             ROOT / "scripts/qualify_export_schema_recovery_20261118.py"]
    files.extend(ROOT / path for path in read(PARENT / "implementation_frozen.json"))
    write(OUT / "protected_hashes.json", protected)
    write(OUT / "implementation_frozen.json", {str(path.relative_to(ROOT)): sha(path) for path in files})
    method.b.verify()
    method.b.synthetic_check()
    write(OUT / "prepared.json", dict(sources=4, new_seeds=[123, 456], cases=16, real_fits=0))


def report(deadline):
    import joblib
    import numpy as np
    import pandas as pd
    from taskdq.e1 import metrics
    from taskdq.scoring import predict

    method.b.verify()
    assert time.time() < deadline
    cases = method.records()
    inventory = []
    for row in cases:
        endpoints = read(OUT / "units" / row["tag"] / "frozen_endpoints.json")["inventory"]
        inventory.extend(dict(endpoint, seed=row["seed"]) for endpoint in endpoints)
    assert len(inventory) == 40
    for entry in read(OUT / "primary_ledger.json"):
        assert entry["status"] == "completed"
        folder = OUT / entry["folder"]
        data = joblib.load(folder / "data.joblib")
        np.testing.assert_array_equal(predict(joblib.load(folder / "model.joblib"), data["valid"]),
                                      np.load(folder / "predictions.npy"))
    contexts = [path for row in method.normalized_records()
                for pattern in ("neural_CV/*/complete.json", "neural_final/complete.json")
                for path in (OUT / "units" / row["tag"]).glob(pattern)]
    assert len(contexts) == 32
    assert max(read(path)["replay_maxdiff"] for path in contexts) <= 1e-5
    queries = pd.DataFrame(read(OUT / "query_ledger.json"))
    assert len(queries) == 120 and queries.groupby(["table", "method"]).size().eq(5).all()
    assert time.time() < deadline
    assert not (OUT / "outer_open_receipt.json").exists()
    for entry in inventory:
        assert sha(OUT / entry["path"]) == entry["prediction_sha256"]
    write(OUT / "frozen_all_endpoints.json", dict(inventory=inventory, time=time.time()))
    write(OUT / "outer_open_receipt.json", dict(time=time.time(), endpoints=40))
    rows = []
    for entry in inventory:
        target = np.load(OUT / "units" / entry["tag"] / "sealed_outer_y.npy")
        scores = metrics(target, np.load(OUT / entry["path"]), dict(type=entry["task"]))
        method_name = entry["arm"]
        if entry["tag"].startswith("N_"):
            method_name = {"original_Full": "canonical_Full", "Full_neural_experts": "canonical_expert_system",
                           "direct_neural_reference": "canonical_direct_reference",
                           "fixed_blend_reference": "canonical_blend_reference"}[method_name]
        rows.append(dict(source=entry["source"], task=entry["task"], seed=entry["seed"], N=5,
                         method=method_name, **scores))
    frame = pd.concat([pd.read_csv(PARENT / "deliverables/SCORE_REPORT.csv"),
                       pd.DataFrame(rows)], ignore_index=True)
    assert len(frame) == 60
    effects = {}
    for task, metric in (("classification", "auc"), ("regression", "rmse")):
        subset = frame[frame.task == task]
        pivot = subset.pivot(index=["source", "seed"], columns="method", values=metric)
        gains = (100 * (pivot.canonical_expert_system - pivot.original_Full) if task == "classification"
                 else 100 * (1 - pivot.canonical_expert_system / pivot.original_Full))
        result = dict(mean=float(gains.groupby(level="source").mean().mean()),
                      sources=gains.groupby(level="source").mean().to_dict(),
                      seeds={str(k):float(v) for k,v in gains.groupby(level="seed").mean().items()})
        if task == "classification":
            f1 = subset.pivot(index=["source", "seed"], columns="method", values="f1")
            result["f1_pp"] = float((100 * (f1.canonical_expert_system - f1.original_Full))
                                    .groupby(level="source").mean().mean())
        effects[task] = result
    c, r = effects["classification"], effects["regression"]
    gate = (c["mean"] >= 5 and c["f1_pp"] >= 0 and r["mean"] >= 5
            and min(c["sources"].values()) > 0 and min(r["sources"].values()) > 0
            and min(c["seeds"].values()) > 0 and min(r["seeds"].values()) > 0)
    summary = dict(effects=effects, confirmed=bool(gate), sources=4, seeds=[42,123,456],
                   scope="training-difficulty-screened cross-export subgroup", independent_sources=4,
                   primary_fits=len(read(OUT / "primary_ledger.json")), neural_contexts=32)
    write(OUT / "summary.json", summary)
    dest = OUT / "deliverables"
    dest.mkdir()
    frame.to_csv(dest / "SCORE_REPORT.csv", index=False, float_format="%.16g")
    assert time.time() < deadline
    if gate:
        version = dict(method="mandatory public export normalization, Full N3, fixed task expert and half-blend",
                       N=5, effects=effects, scope=summary["scope"], sources=4, seeds=[42,123,456],
                       implementation=read(OUT / "implementation_frozen.json"),
                       config_sha256=sha(CFG), formal_version_fixed=True)
        write(OUT / "CONFIRMED_VERSION.json", version)
        write(dest / "VERSION_MANIFEST.json", version)


def controller():
    with (OUT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT / "dispatch.json").exists()
        method.b.verify()
        import subprocess
        lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True).splitlines()
        gpus = [int(v[0]) for line in lines if (v:=line.replace(" ","").split(","))
                and int(v[0]) in [1,5,6,7] and int(v[1]) < 10000 and int(v[2]) < 5]
        assert gpus
        start = time.time()
        deadline = start + 2700
        write(OUT / "dispatch.json", dict(time=start, deadline=deadline, pid=os.getpid(), gpus=gpus))
        try:
            for phase in ("full_worker","neural_worker","raw_final_worker","final_worker"):
                method.b.phase(phase,gpus,deadline)
            method.bind()
            report(deadline)
            write(OUT / "controller_complete.json", dict(exit=0, seconds=time.time()-start,
                                                         confirmed=read(OUT/"summary.json")["confirmed"]))
        except BaseException as error:
            write(OUT / "controller_failure.json",
                  dict(error_class=type(error).__name__, traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    configure()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "controller":
        controller()
    elif args.stage in ("neural_worker","final_worker"):
        method.b.cases=method.normalized_records
        getattr(method.b,args.stage)(args.index,args.workers)
        if args.stage == "final_worker":
            for i, row in enumerate(method.normalized_records()):
                if i % args.workers != args.index:
                    continue
                path = OUT / "units" / row["tag"] / "frozen_endpoints.json"
                frozen = read(path)
                for entry in frozen["inventory"]:
                    entry["seed"] = row["seed"]
                write(path, frozen)
    elif args.stage == "full_worker":
        method.b.full_worker(args.index,args.workers)
    elif args.stage == "raw_final_worker":
        method.bind()
        method.raw_final_worker(args.index,args.workers)
