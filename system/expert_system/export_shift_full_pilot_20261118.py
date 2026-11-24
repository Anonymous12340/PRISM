"""Same-N5 Full and normalized Full experts under a fixed export-format shift."""

import argparse
import copy
import csv
import fcntl
import os
from pathlib import Path
import shutil
import time
import traceback

import export_shift_mechanism_20261118 as mechanism
import TabICL_expert_comparison_20261118 as expert


b = expert.full
ROOT = b.ROOT
NAME = "export_shift_full_pilot_20261118"
OUT = ROOT / "results" / NAME
CFG = ROOT / "configs" / (NAME + ".json")
read, write, sha, key = b.read, b.write, b.sha, b.key
ORIGINAL_BIND = b.bind
TABPFN_CONTEXT = expert.tabpfn_context


def records():
    return read(OUT / "cases.json")


def normalized_records():
    return [row for row in records() if row["tag"].startswith("N_")]


def bind():
    rt, helper = ORIGINAL_BIND()
    if not getattr(rt, "_export_shift_installed", False):
        original_scorer = rt.scorer

        def scorer(tag):
            sc = original_scorer(tag)
            if tag.startswith("R_"):
                original_transform = sc.transform
                contract = read(sc.unit / "public_export_contract.json")

                def transform(spec, train, valid, y, groups, slot, where):
                    canonical = mechanism.restore(valid, contract)
                    shifted = mechanism.export(canonical, contract, True)
                    return original_transform(spec, train, shifted, y, groups, slot, where)

                sc.transform = transform
            return sc

        rt.scorer = scorer
        rt._export_shift_installed = True
    return rt, helper


def context(folder, task, train, y, valid):
    if task == "classification":
        return expert.tabicl_context(folder, task, train, y, valid)
    return TABPFN_CONTEXT(folder, task, train, y, valid)


def configure():
    b.NAME, b.OUT, b.CFG, b.__file__ = NAME, OUT, CFG, __file__
    b.bind, b.context, b.cases = bind, context, records
    b.EXPERT = dict(kind="neural_expert", ids=["__frozen_task_expert__"])
    b.BLEND = dict(kind="fixed_blend", ids=["__frozen_task_expert__", "__fixed_half_blend__"])
    expert.CFG = CFG


def prepare():
    import joblib
    import numpy as np
    import pandas as pd
    import qualify_grounded_transfer_20261117 as qualifier

    assert read(OUT / "qualification_supervisor_complete.json")["qualified_complete"]
    assert not (OUT / "prepared.json").exists() and not (OUT / "dispatch.json").exists()
    manifest = OUT / "qualification/manifest.json"
    entries = read(manifest)["development"]
    assert len(entries) == 4
    shutil.copytree(b.PARENT / "runtime", OUT / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    run_config = read(b.PARENT / "run_config.json")
    assert run_config["model"] == read(ROOT / "configs/e5_formal_20261110.json")["model"]
    run_config["output_dir"] = str(OUT)
    write(OUT / "run_config.json", run_config)
    rt, _ = bind()
    from taskdq.e2_parameterized import registry

    protected = {str(manifest.relative_to(ROOT)): sha(manifest)}
    cases = []
    for entry in entries:
        path = ROOT / entry["path"]
        assert sha(path) == entry["source_sha256"]
        protected[str(path.relative_to(ROOT))] = sha(path)
        source = joblib.load(path)
        validated, _, numeric = qualifier.validate(source, entry["task"])
        assert validated["x"].equals(source["x"]) and validated["y"].equals(source["y"])
        assert numeric == entry["unit_columns"]
        clean = source["x"].copy()
        for column in clean.select_dtypes(exclude="number"):
            original = clean[column].astype("string")
            normalized = original.str.strip().str.casefold()
            if normalized.dropna().nunique() != original.dropna().nunique():
                raise ValueError("category_normalization_collision")
            clean[column] = normalized.astype(object).where(normalized.notna(), np.nan)
        contract = [{"column": column, "unit_column": str(column) + "__export_unit"}
                    for column in numeric]
        train_rows, outer_rows = b.folds(source, 5, 42)[0]
        train_rows, outer_rows = np.sort(train_rows), np.sort(outer_rows)
        assert not set(source["groups"].iloc[train_rows]) & set(source["groups"].iloc[outer_rows])
        raw_train = mechanism.export(clean.iloc[train_rows], contract, False)
        raw_outer = mechanism.export(clean.iloc[outer_rows], contract, True)
        canonical_train = mechanism.restore(raw_train, contract)
        canonical_outer = mechanism.restore(raw_outer, contract)
        pd.testing.assert_frame_equal(canonical_train, clean.iloc[train_rows],
                                      check_exact=False, check_dtype=False, rtol=1e-12, atol=1e-12)
        pd.testing.assert_frame_equal(canonical_outer, clean.iloc[outer_rows],
                                      check_exact=False, check_dtype=False, rtol=1e-12, atol=1e-12)
        support_source = copy.copy(source)
        for name in ("x", "y", "groups"):
            support_source[name] = source[name].iloc[train_rows]
        _, support = b.folds(support_source, 4, 53)[0]
        support = np.sort(support)
        rest = np.setdiff1d(np.arange(len(train_rows)), support)
        remaining_source = copy.copy(support_source)
        for name in ("x", "y", "groups"):
            remaining_source[name] = support_source[name].iloc[rest]
        splits = []
        for a, z in b.folds(remaining_source, 3, 59):
            a, z = np.sort(np.r_[support, rest[a]]), np.sort(rest[z])
            assert not set(support_source["groups"].iloc[a]) & set(support_source["groups"].iloc[z])
            splits.append(dict(train=a.tolist(), validation=z.tolist()))
        background = (
            source["pair"].get("background", "")[:1200]
            + "\nPublic export rules available to every method: "
            + str(contract)
            + ". Each observed numeric value equals its canonical value multiplied by the "
              "number after times_ in the corresponding unit column. Training uses times_1 "
              "and times_1000. Validation and deployment use times_100. Text category meaning "
              "is invariant to letter case and surrounding whitespace. export_system and unit "
              "columns describe transport format. All evaluation uses original targets and entities."
        )
        for prefix, x, outer in (("R", raw_train, raw_outer), ("N", canonical_train, canonical_outer)):
            tag = prefix + "_" + entry["code"] + "__42"
            folder = rt.unit(tag)
            assert not (folder / "source.joblib").exists()
            folder.mkdir(parents=True, exist_ok=True)
            current = copy.deepcopy(support_source)
            current.update(x=x, seed=42)
            current["pair"] = dict(current["pair"], name=tag, background=background)
            joblib.dump(current, folder / "source.joblib")
            joblib.dump(dict(x=outer, groups=source["groups"].iloc[outer_rows]),
                        folder / "sealed_outer_X.joblib")
            np.save(folder / "sealed_outer_y.npy", source["y"].iloc[outer_rows].to_numpy())
            write(folder / "partition.json", dict(support=support.tolist(), remaining=rest.tolist()))
            write(folder / "inner_splits.json", splits)
            write(folder / "public_export_contract.json", contract)
            write(folder / "menu.json", registry(x, current.get("date_columns", {})))
            write(folder / "derived_ops.json", {})
            for file in folder.iterdir():
                if file.name != "derived_ops.json":
                    protected[str(file.relative_to(ROOT))] = sha(file)
            cases.append(dict(tag=tag, code=entry["code"], task=entry["task"], seed=42,
                              outer_y_sha256=sha(folder / "sealed_outer_y.npy")))
    for file in (OUT / "runtime").rglob("*.py"):
        protected[str(file.relative_to(ROOT))] = sha(file)
    adapter = Path(read(CFG)["adapter_path"]) / "adapter_model.safetensors"
    protected[str(adapter.relative_to(ROOT))] = sha(adapter)
    for task, (filename, digest) in read(CFG)["checkpoints"].items():
        checkpoint = b.PUBLIC / filename
        assert sha(checkpoint) == digest
        protected[str(checkpoint.relative_to(ROOT))] = digest
    for filename, digest, _ in read(CFG)["tabicl"].values():
        checkpoint = expert.ASSETS / filename
        assert sha(checkpoint) == digest
        protected[str(checkpoint.relative_to(ROOT))] = digest
    files = [CFG, Path(__file__).resolve(), ROOT / "designs" / NAME / "PLAN.md",
             ROOT / "scripts/export_shift_mechanism_20261118.py",
             ROOT / "scripts/TabPFN_full_expert_20261118.py",
             ROOT / "scripts/TabICL_expert_comparison_20261118.py",
             ROOT / "scripts/tabicl_local_worker_20261118.py",
             ROOT / "scripts/full_novelty_return_runtime_20261117.py"]
    write(OUT / "cases.json", cases)
    protected[str((OUT / "cases.json").relative_to(ROOT))] = sha(OUT / "cases.json")
    protected[str((OUT / "run_config.json").relative_to(ROOT))] = sha(OUT / "run_config.json")
    write(OUT / "protected_hashes.json", protected)
    write(OUT / "implementation_frozen.json", {str(p.relative_to(ROOT)): sha(p) for p in files})
    b.verify()
    b.synthetic_check()
    write(OUT / "prepared.json", dict(cases=8, sources=4, real_fits=0))


def raw_final_worker(index, workers):
    import joblib
    import numpy as np
    from taskdq.scoring import fit_predictor, predict

    rt, _ = bind()
    for row in [r for i, r in enumerate([v for v in records() if v["tag"].startswith("R_")])
                if i % workers == index]:
        sc = rt.scorer(row["tag"])
        outer = joblib.load(sc.unit / "sealed_outer_X.joblib")
        spec = read(sc.unit / "Full_N5.json")["selected"]["spec"]
        folder = sc.unit / "final_XGB" / key(spec)
        folder.mkdir(parents=True)
        train, valid, y = sc.transform(spec, sc.x, outer["x"], sc.y, sc.source["groups"], "final", folder)
        number = rt.ledger("primary_ledger.json", dict(table=row["tag"], phase="final",
                            status="started", folder=str(folder.relative_to(OUT))),
                           read(CFG)["limits"]["primary_predictor_fits"])
        model = fit_predictor(train, y, sc.task, read(OUT / "run_config.json")["model"])
        prediction = predict(model, valid)
        joblib.dump(model, folder / "model.joblib")
        joblib.dump(dict(train=train, valid=valid, y_train=np.asarray(y)), folder / "data.joblib")
        np.save(folder / "predictions.npy", prediction)
        np.testing.assert_array_equal(predict(joblib.load(folder / "model.joblib"), valid), prediction)
        rt.ledger_update("primary_ledger.json", number, status="completed")
        write(sc.unit / "frozen_endpoints.json", dict(inventory=[dict(
            tag=row["tag"], source=row["code"], task=row["task"], arm="original_Full",
            path=str((folder / "predictions.npy").relative_to(OUT)),
            prediction_sha256=sha(folder / "predictions.npy"))]))
    b.verify()


def report():
    import joblib
    import numpy as np
    import pandas as pd
    from taskdq.e1 import metrics
    from taskdq.scoring import predict

    b.verify()
    inventory = [endpoint for row in records()
                 for endpoint in read(OUT / "units" / row["tag"] / "frozen_endpoints.json")["inventory"]]
    assert len(inventory) == 20
    for entry in read(OUT / "primary_ledger.json"):
        assert entry["status"] == "completed"
        folder = OUT / entry["folder"]
        data = joblib.load(folder / "data.joblib")
        np.testing.assert_array_equal(predict(joblib.load(folder / "model.joblib"), data["valid"]),
                                      np.load(folder / "predictions.npy"))
    contexts = [path for row in normalized_records()
                for pattern in ("neural_CV/*/complete.json", "neural_final/complete.json")
                for path in (OUT / "units" / row["tag"]).glob(pattern)]
    assert len(contexts) == 16
    assert max(read(path)["replay_maxdiff"] for path in contexts) <= 1e-5
    for code in {row["code"] for row in records()}:
        np.testing.assert_array_equal(
            np.load(OUT / "units" / ("R_" + code + "__42") / "sealed_outer_y.npy"),
            np.load(OUT / "units" / ("N_" + code + "__42") / "sealed_outer_y.npy"))
    queries = pd.DataFrame(read(OUT / "query_ledger.json"))
    assert len(queries) == 60
    assert queries.groupby(["table", "method"]).size().eq(5).all()
    assert not (OUT / "outer_open_receipt.json").exists()
    write(OUT / "frozen_all_endpoints.json", dict(inventory=inventory, time=time.time()))
    write(OUT / "outer_open_receipt.json", dict(time=time.time(), endpoints=20))
    rows = []
    for entry in inventory:
        path = OUT / entry["path"]
        assert sha(path) == entry["prediction_sha256"]
        target = np.load(OUT / "units" / entry["tag"] / "sealed_outer_y.npy")
        scores = metrics(target, np.load(path), dict(type=entry["task"]))
        method = entry["arm"]
        if entry["tag"].startswith("N_"):
            method = {"original_Full": "canonical_Full", "Full_neural_experts": "canonical_expert_system",
                      "direct_neural_reference": "canonical_direct_reference",
                      "fixed_blend_reference": "canonical_blend_reference"}[method]
        rows.append(dict(source=entry["source"], task=entry["task"], seed=42, N=5,
                         method=method, **scores))
    frame = pd.DataFrame(rows)
    effects = {}
    for task, metric in (("classification", "auc"), ("regression", "rmse")):
        subset = frame[frame.task == task]
        pivot = subset.pivot(index="source", columns="method", values=metric)
        for method in ("canonical_Full", "canonical_expert_system"):
            gains = (100 * (pivot[method] - pivot.original_Full) if task == "classification"
                     else 100 * (1 - pivot[method] / pivot.original_Full))
            effect = dict(mean=float(gains.mean()), sources=gains.to_dict(),
                          positive_sources=int(gains.gt(0).sum()))
            if task == "classification":
                f1 = subset.pivot(index="source", columns="method", values="f1")
                effect["f1_pp"] = float(100 * (f1[method] - f1.original_Full).mean())
            effects[task + "_" + method] = effect
    c, r = effects["classification_canonical_expert_system"], effects["regression_canonical_expert_system"]
    gate = c["mean"] >= 5 and c["f1_pp"] >= 0 and r["mean"] >= 5 and c["positive_sources"] == r["positive_sources"] == 2
    write(OUT / "summary.json", dict(effects=effects, pilot_gate=bool(gate), formal_version_fixed=False))
    dest = OUT / "deliverables"
    dest.mkdir()
    frame.to_csv(dest / "SCORE_REPORT.csv", index=False, float_format="%.16g")


def controller():
    with (OUT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT / "dispatch.json").exists()
        b.verify()
        import subprocess
        lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True).splitlines()
        gpus = [int(values[0]) for line in lines if (values := line.replace(" ", "").split(","))
                and int(values[0]) in [1, 5, 6, 7] and int(values[1]) < 10000 and int(values[2]) < 5]
        assert gpus
        start, deadline = time.time(), time.time() + 2700
        write(OUT / "dispatch.json", dict(time=start, deadline=deadline, gpus=gpus, pid=os.getpid()))
        try:
            for phase in ("full_worker", "neural_worker", "raw_final_worker", "final_worker"):
                b.phase(phase, gpus, deadline)
            bind()
            report()
            write(OUT / "controller_complete.json", dict(exit=0, seconds=time.time()-start,
                  pilot_gate=read(OUT / "summary.json")["pilot_gate"]))
        except BaseException as error:
            write(OUT / "controller_failure.json", dict(error_class=type(error).__name__,
                                                        traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    configure()
    if args.stage in ("neural_worker", "final_worker"):
        b.cases = normalized_records
        getattr(b, args.stage)(args.index, args.workers)
    elif args.stage == "full_worker":
        b.full_worker(args.index, args.workers)
    elif args.stage == "raw_final_worker":
        bind()
        raw_final_worker(args.index, args.workers)
    else:
        globals()[args.stage]()
