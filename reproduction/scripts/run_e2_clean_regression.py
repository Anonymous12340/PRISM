"""Re-evaluate the frozen system on the eight original E2 summary datasets."""

import argparse
import copy
import csv
import fcntl
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import TabICL_expert_comparison_20261118 as expert


b = expert.full
ROOT = b.ROOT
NAME = "e2_frozen_system_20261118"
OUT = ROOT / "results" / NAME
CFG = ROOT / "configs" / (NAME + ".json")
OLD = ROOT / "results/e2_unified_independent_20261107"
DATA = ROOT / "data/e2_unified_independent_20261107/prepared"
FROZEN = ROOT / "results/export_shift_schema_stability_20261118"
read, write, sha, key = b.read, b.write, b.sha, b.key
DATASETS = ("kr-vs-kp", "tic-tac-toe", "vote", "qsar-biodeg",
            "cpu_act", "Ailerons", "houses", "diamonds")
BASELINES = ("raw", "standard", "autofeat", "caafe", "llmfe", "classical_clean",
             "lm_cleaning_simplified", "clean_then_openfe")


def cases():
    return read(OUT / "cases.json")


def configure():
    b.NAME, b.OUT, b.CFG, b.__file__ = NAME, OUT, CFG, __file__
    b.cases, b.context = cases, context
    b.EXPERT = dict(kind="neural_expert", ids=["__frozen_task_expert__"])
    b.BLEND = dict(kind="fixed_blend", ids=["__frozen_task_expert__", "__fixed_half_blend__"])
    expert.CFG = CFG


def context(folder, task, train, y, valid):
    import numpy as np

    tx, vx, cats = b.matrix(train, valid)
    if (folder / "complete.json").exists():
        with np.load(folder / "input.npz", allow_pickle=False) as cached:
            for name, value in dict(train_x=tx, train_y=np.asarray(y), predict_x=vx,
                                    categorical_indices=cats).items():
                np.testing.assert_array_equal(cached[name], value)
        assert sha(folder / "predictions.npy") == read(folder / "complete.json")["predictions_sha256"]
        return np.load(folder / "predictions.npy", allow_pickle=False)
    if task == "classification":
        return expert.tabicl_context(folder, task, train, y, valid)
    folder.mkdir(parents=True)
    np.savez(folder / "input.npz", train_x=tx, train_y=np.asarray(y), predict_x=vx,
             categorical_indices=np.asarray(cats, dtype=int))
    filename, digest = read(CFG)["checkpoints"]["regression"]
    write(folder / "request.json", dict(checkpoint=str(b.PUBLIC / filename),
                                        checkpoint_sha256=digest, n_estimators=8))
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    with (folder / "worker.log").open("x") as log:
        subprocess.run([str(b.NEURAL), str(ROOT / "scripts/e2_frozen_neural_worker_20261118.py"),
                        str(folder / "request.json")], stdout=log, stderr=subprocess.STDOUT,
                       env=env, timeout=3600, check=True)
    assert sha(folder / "predictions.npy") == read(folder / "complete.json")["predictions_sha256"]
    return np.load(folder / "predictions.npy", allow_pickle=False)


def prepare():
    import joblib
    import numpy as np
    import pandas as pd

    assert not (OUT / "prepared.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    version = read(FROZEN / "CONFIRMED_VERSION.json")
    assert version["formal_version_fixed"]
    for path, digest in version["implementation"].items():
        assert sha(ROOT / path) == digest
    shutil.copytree(FROZEN / "runtime", OUT / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    old = read(OLD / "run_config.json")
    current = read(FROZEN / "run_config.json")
    assert old["model"] == current["model"]
    current["output_dir"] = str(OUT)
    current["field_metadata"] = {}
    write(OUT / "run_config.json", current)
    rt, _ = b.bind()
    from taskdq.e2_parameterized import registry

    protected = {str((FROZEN / "CONFIRMED_VERSION.json").relative_to(ROOT)):
                 sha(FROZEN / "CONFIRMED_VERSION.json")}
    entries = []
    for dataset in DATASETS:
        for seed in (42, 123, 456):
            old_unit = OLD / "units" / dataset / str(seed) / "final"
            original_path = old_unit / "source.joblib"
            original = joblib.load(original_path)
            development_path = DATA / dataset / str(seed) / "development.joblib"
            development = joblib.load(development_path)
            for name in ("x", "y", "groups"):
                assert original[name].equals(development[name])
            outer_path = DATA / dataset / str(seed) / "sealed_test.joblib"
            outer = joblib.load(outer_path)
            assert not set(original["groups"]) & set(outer["groups"])
            tag = dataset + "__" + str(seed)
            unit = rt.unit(tag)
            unit.mkdir(parents=True)
            source = copy.deepcopy(original)
            source["pair"] = dict(source["pair"], name=tag)
            joblib.dump(source, unit / "source.joblib")
            joblib.dump(dict(x=outer["x"], groups=outer["groups"]), unit / "sealed_outer_X.joblib")
            np.save(unit / "sealed_outer_y.npy", outer["y"].to_numpy())
            folds_path = old_unit / "inner_splits.json"
            write(unit / "legacy_inner_splits.json", read(folds_path))
            _, support = b.folds(source, 4, seed + 11)[0]
            support = np.sort(support)
            remaining = np.setdiff1d(np.arange(len(source["x"])), support)
            subset = copy.copy(source)
            for field in ("x", "y", "groups"):
                subset[field] = source[field].iloc[remaining]
            folds = []
            for a, z in b.folds(subset, 3, seed + 17):
                a = np.sort(np.r_[support, remaining[a]])
                z = np.sort(remaining[z])
                assert not set(support) & set(z)
                folds.append(dict(train=a.tolist(), validation=z.tolist()))
            for fold in folds:
                assert not set(source["groups"].iloc[fold["train"]]) & set(source["groups"].iloc[fold["validation"]])
            write(unit / "inner_splits.json", folds)
            write(unit / "partition.json", dict(support=support.tolist(),
                                                remaining=remaining.tolist(),
                                                support_never_validation=True,
                                                user_approved_inner_adapter=True))
            write(unit / "menu.json", registry(source["x"], source.get("date_columns", {})))
            write(unit / "derived_ops.json", {})
            write(unit / "normalization_receipt.json", dict(
                public_export_contracts=[], operation="identity",
                reason="Original E2 inputs contain no registered synthetic export contract.",
                preserved_input_values=True))
            for path in (original_path, development_path, folds_path, outer_path):
                protected[str(path.relative_to(ROOT))] = sha(path)
            for path in unit.iterdir():
                if path.name != "derived_ops.json":
                    protected[str(path.relative_to(ROOT))] = sha(path)
            for method in (*BASELINES, "ours"):
                for budget in (5, 10, 20):
                    directory = old_unit / "test_results" / method / str(budget)
                    prediction = directory / "xgboost_predictions.csv"
                    assert prediction.exists() and (directory / "result.json").exists()
                    ids = pd.read_csv(prediction, usecols=["row_id"])["row_id"]
                    np.testing.assert_array_equal(ids, outer["x"].index)
                    for path in (prediction, directory / "result.json"):
                        protected[str(path.relative_to(ROOT))] = sha(path)
            current["field_metadata"][tag] = old.get("field_metadata", {}).get(dataset, {})
            entries.append(dict(tag=tag, code=dataset, seed=seed, task=source["task"]["type"],
                                train_rows=len(source["x"]), test_rows=len(outer["x"])))
    write(OUT / "run_config.json", current)
    write(OUT / "cases.json", entries)
    for path in (OUT / "run_config.json", OUT / "cases.json"):
        protected[str(path.relative_to(ROOT))] = sha(path)
    for path in (OUT / "runtime").rglob("*.py"):
        protected[str(path.relative_to(ROOT))] = sha(path)
    # Preserve all pretrained checkpoints and adapter hashes from the frozen run.
    for path, digest in read(FROZEN / "protected_hashes.json").items():
        if path.endswith((".ckpt", ".safetensors")):
            protected[path] = digest
    paths = [Path(__file__).resolve(), CFG, ROOT / "scripts/e2_frozen_neural_worker_20261118.py",
             ROOT / "designs/e2_frozen_system_20261118/PLAN.md"]
    frozen_code = dict(version["implementation"])
    frozen_code.update({str(path.relative_to(ROOT)):sha(path) for path in paths})
    write(OUT / "protected_hashes.json", protected)
    write(OUT / "implementation_frozen.json", frozen_code)
    b.verify()
    write(OUT / "prepared.json", dict(units=24, datasets=8, new_fits=0,
                                      old_baseline_files_verified=True, input_values_preserved=True))


def prefix_worker(index, workers):
    # Only the stop boundary changes. The already-saved third prefix is retained.
    source = inspect.getsource(b.full_worker)
    assert source.count("if count[0]>=5") == 2
    source = source.replace("if count[0]>=5", "if count[0]>=3")
    source = source.replace("for turn in range(1,6):", "for turn in range(1,4):")
    source = source.replace("assert count[0]==5;snapshot(5);", "assert count[0]==3;")
    source = source.replace("queries=5,public_keys", "queries=3,public_keys")
    exec(compile(source, "<frozen-three-attempt-prefix>", "exec"), b.__dict__)
    b.full_worker(index, workers)


def preflight(code):
    import joblib

    row = next(row for row in cases() if row["code"] == code and row["seed"] == 42)
    unit = OUT / "units" / row["tag"]
    source = joblib.load(unit / "source.joblib")
    fold = read(unit / "inner_splits.json")[0]
    context(unit / "neural_CV" / "0", row["task"],
            source["x"].iloc[fold["train"]], source["y"].iloc[fold["train"]],
            source["x"].iloc[fold["validation"]])
    write(OUT / (code + "_entry_checked.json"), dict(training_only=True, fits_reused_in_formal=True))


def final_worker(index, workers):
    import joblib
    import numpy as np
    from taskdq.scoring import fit_predictor, predict

    rt, _ = b.bind()
    for row in [r for i,r in enumerate(cases()) if i % workers == index]:
        sc = rt.scorer(row["tag"])
        outer = joblib.load(sc.unit / "sealed_outer_X.joblib")
        prefix = read(sc.unit / "Full_N3.json")["selected"]["spec"]
        chosen = read(sc.unit / "expert_selection.json")["selected"]["spec"]
        predictions = {}
        for spec in (prefix, chosen):
            token = key(spec)
            if token in predictions or spec["kind"] in ("neural_expert", "fixed_blend"):
                continue
            folder = sc.unit / "final_XGB" / token
            folder.mkdir(parents=True)
            train, valid, y = sc.transform(spec, sc.x, outer["x"], sc.y,
                                           sc.source["groups"], "final", folder)
            number = rt.ledger("primary_ledger.json", dict(table=row["tag"], phase="final",
                               folder=str(folder.relative_to(OUT)), status="started"),
                               read(CFG)["limits"]["primary_predictor_fits"])
            model = fit_predictor(train, y, sc.task, read(OUT / "run_config.json")["model"])
            pred = predict(model, valid)
            joblib.dump(model, folder / "model.joblib")
            joblib.dump(dict(train=train, valid=valid, y_train=np.asarray(y)), folder / "data.joblib")
            np.save(folder / "predictions.npy", pred)
            np.testing.assert_array_equal(predict(joblib.load(folder / "model.joblib"), valid), pred)
            rt.ledger_update("primary_ledger.json", number, status="completed")
            predictions[token] = pred
        direct = context(sc.unit / "neural_final", row["task"], sc.x, sc.y, outer["x"])
        predictions[key(b.EXPERT)] = direct
        predictions[key(b.BLEND)] = .5 * direct + .5 * predictions[key(prefix)]
        np.save(sc.unit / "final_prediction.npy", predictions[key(chosen)])
        logical_calls = len(list((sc.unit / "methods/original_Full/calls").glob("*.json")))
        physical_calls = len(list((sc.unit / "physical_calls").glob("*.json")))
        assert logical_calls <= 5 and physical_calls <= 5
        write(sc.unit / "frozen_endpoint.json", dict(
            prediction_sha256=sha(sc.unit / "final_prediction.npy"),
            chosen=chosen, seed=row["seed"], candidate_attempts=5,
            actual_lm_calls=logical_calls, physical_lm_calls=physical_calls))
    b.verify()


def report():
    import joblib
    import numpy as np
    import pandas as pd
    from taskdq.e1 import metrics
    from taskdq.scoring import predict

    b.verify()
    assert not (OUT / "outer_scoring_receipt.json").exists()
    endpoints = {row["tag"]:read(OUT / "units" / row["tag"] / "frozen_endpoint.json") for row in cases()}
    for tag, endpoint in endpoints.items():
        assert sha(OUT / "units" / tag / "final_prediction.npy") == endpoint["prediction_sha256"]
    for entry in read(OUT / "primary_ledger.json"):
        assert entry["status"] == "completed"
        folder = OUT / entry["folder"]
        saved = joblib.load(folder / "data.joblib")
        np.testing.assert_array_equal(predict(joblib.load(folder / "model.joblib"), saved["valid"]),
                                      np.load(folder / "predictions.npy"))
    write(OUT / "all_endpoints_frozen.json", endpoints)
    write(OUT / "outer_scoring_receipt.json", dict(time=time.time(), endpoints=24))
    rows, costs = [], []
    for row in cases():
        unit = OUT / "units" / row["tag"]
        target = np.load(unit / "sealed_outer_y.npy")
        prediction = np.load(unit / "final_prediction.npy")
        score = metrics(target, prediction, dict(type=row["task"]))
        for budget in (5,10,20):
            for name,value in score.items():
                rows.append(dict(dataset=row["code"],type=row["task"],seed=row["seed"],
                                 method="ours",budget=budget,status="passed",metric=name,value=value))
            for method in (*BASELINES,"ours"):
                directory = OLD / "units" / row["code"] / str(row["seed"]) / "final/test_results" / method / str(budget)
                cached = pd.read_csv(directory / "xgboost_predictions.csv", float_precision="round_trip")
                np.testing.assert_array_equal(cached.y_true, target)
                reference = read(directory / "result.json")
                values = metrics(target, cached.prediction.to_numpy(), dict(type=row["task"]))
                for name,value in values.items():
                    assert np.isclose(value,reference["xgboost"][name],rtol=1e-10,atol=1e-10)
                    rows.append(dict(dataset=row["code"],type=row["task"],seed=row["seed"],
                                     method="old_Full" if method=="ours" else method,
                                     budget=budget,status=reference["status"],metric=name,value=value))
        costs.append(dict(dataset=row["code"],seed=row["seed"],**endpoints[row["tag"]]))
    dest = OUT / "deliverables"
    dest.mkdir()
    pd.DataFrame(rows).to_csv(dest / "all_seeds.csv",index=False,float_format="%.16g")
    pd.DataFrame(costs).to_csv(dest / "costs.csv",index=False)
    write(OUT / "report_complete.json",dict(units=24,all_baselines_reused=True,all_test_scores_verified=True))


def controller():
    with (OUT / "controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT / "dispatch.json").exists()
        assert (OUT / "kr-vs-kp_entry_checked.json").exists()
        assert (OUT / "cpu_act_entry_checked.json").exists()
        b.verify()
        lines=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"],text=True).splitlines()
        gpus=[int(v[0]) for line in lines if (v:=line.replace(" ","").split(","))
              and int(v[0]) in [1,5,6,7] and int(v[1])<10000 and int(v[2])<5]
        assert gpus
        start=time.time()
        deadline=start+21600
        write(OUT / "dispatch.json",dict(time=start,deadline=deadline,pid=os.getpid(),gpus=gpus))
        try:
            for phase in ("prefix_worker","neural_worker","final_worker"):
                b.phase(phase,gpus,deadline)
            b.bind()
            report()
            write(OUT / "controller_complete.json",dict(exit=0,seconds=time.time()-start))
        except BaseException as error:
            write(OUT / "controller_failure.json",dict(error_class=type(error).__name__,
                                                        traceback=traceback.format_exc()))
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--code")
    args=parser.parse_args()
    configure()
    if args.stage=="neural_worker":
        b.neural_worker(args.index,args.workers)
    elif args.stage=="prefix_worker":
        prefix_worker(args.index,args.workers)
    elif args.stage=="final_worker":
        b.bind()
        final_worker(args.index,args.workers)
    elif args.stage=="preflight":
        preflight(args.code)
    else:
        globals()[args.stage]()
