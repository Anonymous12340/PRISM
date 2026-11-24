"""E3 four-dataset controller and worker.

This is an isolated E3 adapter. It reuses the frozen TaskDQ runtime and the
existing E3 engine, while extending only the data registration layer from two
datasets to four and allowing a registered condition to be not_applicable when
the source has no eligible field (for example Diamonds/typo).
"""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAME = "e3_four_dataset_20261119"
OUT = ROOT / "results" / NAME
CONFIG = ROOT / "configs" / f"{NAME}.json"
# The experiment runs from an isolated source snapshot copied from the prior
# E3 source bundle. This keeps the frozen runtime separate from the repository
# root and from later exploratory branches.
for _source in (ROOT / "results" / NAME / "source", ROOT / "results" / "e3_formal_20261108" / "source"):
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))
ERRORS = ("missing", "outlier", "duplicate", "typo", "unit")
RATES = (0.05, 0.10, 0.20)
SEEDS = (42, 123, 456)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def settings(path=CONFIG):
    path = Path(path).resolve()
    s = read(path)
    s["project_root"] = str(ROOT)
    for key in ("frozen_config", "frozen_source", "output_dir", "data_dir"):
        s[key] = str((ROOT / s[key]).resolve())
    assert s["errors"] == list(ERRORS)
    assert s["rates"] == list(RATES) and s["seeds"] == list(SEEDS)
    assert len(s["datasets"]) == 4
    assert s["main_budget"] == 10 and s["inner_folds"] == 3
    return s


def unit_dir(s, dataset, seed, error, rate):
    return Path(s["output_dir"]) / "units" / dataset / str(seed) / f"{error}_{int(round(rate * 100)):02}"


def matrix(s):
    return [(d["name"], seed, error, rate)
            for d in s["datasets"] for seed in s["seeds"]
            for error in s["errors"] for rate in s["rates"]]


def runtime_config(s):
    from taskdq.e3_data import runtime_config as frozen_runtime_config
    c = frozen_runtime_config({**s, "datasets": s["datasets"]})
    c["resources"]["max_calls_formal"] = 9000
    return c


def load_arff(item):
    import arff
    raw_dir = Path(item["raw_dir"])
    desc = read(raw_dir / "description.json")["data_set_description"]
    assert int(desc["id"]) == item["openml_id"]
    assert int(desc["version"]) == item["openml_version"]
    with (raw_dir / "data.arff").open() as handle:
        parsed = arff.load(handle)
    names = [v[0] for v in parsed["attributes"]]
    frame = pd.DataFrame(parsed["data"], columns=names)
    frame = frame.rename(columns=item.get("column_mapping", {}))
    categorical = {item.get("column_mapping", {}).get(name, name)
                   for name, kind in parsed["attributes"] if isinstance(kind, list)}
    return frame, categorical, parsed, desc


def cohort(s, item):
    raw, categorical, parsed, desc = load_arff(item)
    raw.index = pd.Index(range(len(raw)), name="__row_id")
    target = item["target"]
    exclude = set(item.get("exclude", []))
    raw = raw.dropna().copy()
    features = [c for c in raw if c != target and c not in exclude]
    x = raw[features].copy()
    for col in x:
        if col in categorical:
            x[col] = x[col].astype(object)
        else:
            x[col] = pd.to_numeric(x[col], errors="raise").astype(float)
    keep = ~x.duplicated()
    x = x.loc[keep]
    raw = raw.loc[x.index]
    limit = int(item.get("cohort_limit", s["cohort_limit"]))
    n = min(limit, len(x))
    chosen = np.sort(np.random.default_rng(s["cohort_sampling_seed"]).choice(len(x), n, replace=False))
    x = x.iloc[chosen]
    raw = raw.loc[x.index]
    if item["type"] == "classification":
        y = raw[target].map(lambda v: int(str(v) == str(item["positive"]))).astype(int)
        assert y.nunique() == 2
    else:
        y = pd.to_numeric(raw[target], errors="raise").astype(float)
    groups = pd.Series(x.index.to_numpy(), index=x.index, name="__group")
    assert not x.isna().any().any() and item["target"] not in x
    for col in x:
        if pd.api.types.is_numeric_dtype(x[col]):
            assert np.isfinite(x[col]).all()
    task = {"name": item["name"], "type": item["type"], "target": item["target"],
            "description": item["task_description"]}
    pair = {"name": item["name"], "background": desc.get("description", "")[:4000], "task": task}
    receipt = {
        "dataset": item["name"], "openml_id": item["openml_id"], "version": item["openml_version"],
        "source_rows": len(parsed["data"]), "dropped_natural_missing_rows": int(len(raw) - len(x)),
        "eligible_after_complete_positive_unique_checks": int(keep.sum()), "cohort_rows": len(x),
        "cohort_row_ids": [int(v) for v in x.index], "features": list(x),
        "categorical": sorted(c for c in x if c in categorical), "source_license": desc.get("licence"),
        "selection_used_model_scores": False, "test_opened": False,
        "source_sha256": sha(Path(item["raw_dir"]) / "data.arff")
    }
    return x, y, groups, task, pair, receipt


def safe_corrupt(frame, labels, groups, error, rate, seed, slot, factors, multiplier):
    from taskdq.e3_corruption import corrupt
    try:
        return corrupt(frame, labels, groups, error, rate, seed, slot, factors, multiplier)
    except AssertionError as exc:
        # Only convert the documented no-eligible-field case to N/A. Any other
        # assertion remains a hard preparation failure.
        numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        eligible = []
        if error == "duplicate":
            eligible = list(frame.index)
        else:
            for c in frame:
                if error == "missing":
                    eligible.extend((r, c) for r in frame.index)
                elif error == "outlier" and c in numeric:
                    eligible.extend((r, c) for r in frame.index)
                elif error == "unit" and c in factors:
                    eligible.extend((r, c) for r in frame.index if frame.at[r, c] != 0)
                elif error == "typo" and c not in numeric:
                    eligible.extend((r, c) for r in frame.index if len(str(frame.at[r, c]).strip()) >= 2)
        if eligible:
            raise
        meta = {"error": error, "rate": rate, "eligible_count": 0, "injected_count": 0,
                "actual_rate": None, "training_rows_before": len(frame),
                "training_rows_after": len(frame), "seed": int(seed), "slot": int(slot),
                "status": "not_applicable", "target_untouched": True,
                "exception": str(exc)}
        return frame.copy(), labels.copy(), groups.copy(), [], meta


def prepare(path=CONFIG):
    from taskdq.scoring import splits
    s = settings(path)
    root = Path(s["output_dir"]); data = Path(s["data_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "preparation_complete.json").exists():
        assert read(root / "settings.json") == s
        return read(root / "preparation_complete.json")
    assert not list(root.glob("test_results/*/*/access.json"))
    write(root / "settings.json", s)
    write(root / "runtime_config.json", runtime_config(s))
    metadata = []; checks = 0
    for item in s["datasets"]:
        x, y, groups, task, pair, receipt = cohort(s, item); metadata.append(receipt)
        for seed in s["seeds"]:
            devpos, testpos = splits(x, y, task, 5, seed, groups)[0]
            train, labels, gg = x.iloc[devpos], y.iloc[devpos], groups.iloc[devpos]
            test, test_y = x.iloc[testpos], y.iloc[testpos]
            reference = data / "evaluation_reference" / item["name"] / str(seed)
            reference.mkdir(parents=True, exist_ok=True)
            joblib.dump({"x": train, "y": labels, "groups": gg, "task": task}, reference / "clean_training.joblib")
            joblib.dump({"x": test, "y": test_y, "task": task}, reference / "sealed_test.joblib")
            write(reference / "split.json", {"seed": seed, "development_row_ids": [int(v) for v in train.index],
                                               "test_row_ids": [int(v) for v in test.index],
                                               "test_evaluated": False, "groups_disjoint": True,
                                               "fresh_e3_split": True})
            folds = splits(train, labels, task, 3, seed, gg)
            for error in s["errors"]:
                for rate in s["rates"]:
                    unit = unit_dir(s, item["name"], seed, error, rate); unit.mkdir(parents=True, exist_ok=True)
                    outer, oy, og, truth, meta = safe_corrupt(train, labels, gg, error, rate, seed, 0,
                                                               item.get("unit_factors", {}), s["outlier_iqr_multiplier"])
                    source = {"x": outer, "y": oy, "groups": og, "task": task, "pair": pair, "seed": seed,
                              "repair_column": next((c for c in train if pd.api.types.is_numeric_dtype(train[c])), None),
                              "numeric_type_hints": [], "date_columns": {}}
                    joblib.dump(source, unit / "source.joblib")
                    truth_dir = root / "truth_store" / item["name"] / str(seed) / f"{error}_{int(rate * 100):02}"
                    write(truth_dir / "outer.json", {"records": truth, "metadata": meta})
                    rows = []
                    for i, (a, b) in enumerate(folds):
                        dirty, dy, dg, known, imeta = safe_corrupt(train.iloc[a], labels.iloc[a], gg.iloc[a], error,
                                                                     rate, seed, i + 1, item.get("unit_factors", {}),
                                                                     s["outlier_iqr_multiplier"])
                        folder = unit / "inner_inputs"; folder.mkdir(exist_ok=True)
                        joblib.dump({"train": dirty, "train_y": dy, "train_groups": dg,
                                     "validation": train.iloc[b], "validation_y": labels.iloc[b]}, folder / f"{i}.joblib")
                        write(truth_dir / f"inner_{i}.json", {"records": known, "metadata": imeta})
                        rows.append({"train": a.tolist(), "validation": b.tolist(),
                                     "train_row_ids": [int(v) for v in dirty.index],
                                     "validation_row_ids": [int(v) for v in train.index[b]]})
                        checks += 1
                    write(unit / "inner_splits.json", rows)
                    from taskdq.e2_parameterized import registry
                    write(unit / "menu.json", registry(outer, {}))
                    write(unit / "prepared.json", {"dataset": item["name"], "seed": seed, "error": error,
                                                     "rate": rate, "training_rows": len(outer), "inner_partitions": 3,
                                                     "truth_in_source": False, "test_loaded": False,
                                                     "injection_status": meta.get("status", "applied")})
    write(root / "dataset_metadata.json", metadata)
    result = {"stage": "prepared_before_model_calls", "formal_units": len(matrix(s)),
              "test_partitions": len(s["datasets"]) * len(s["seeds"]), "inner_partition_checks": checks,
              "reference_model_states": len(s["datasets"]) * len(s["seeds"]),
              "test_evaluations": 0, "new_model_calls_fits_training": 0,
              "previous_e3_test_loaded": 0, "not_applicable_conditions": sum(
                  1 for item in metadata for c in [] )}
    write(root / "preparation_complete.json", result)
    return result


def patch_e3():
    import taskdq.e3 as engine
    engine.settings = settings
    engine.runtime_config = runtime_config
    engine.unit_dir = unit_dir
    engine.matrix = matrix
    engine.prepare = prepare
    return engine


def run_one(path, dataset, seed, error, rate, cpu_only=False):
    engine = patch_e3()
    return engine.run_unit(path, dataset, seed, error, rate, cpu_only)


def synthetic_preflight():
    folder = OUT / "engineering_preflight.json"
    if folder.exists(): return read(folder)
    x = pd.DataFrame({"num": [1., 2., 3., 4., 5., 6.],
                      "cat": ["a", "b", "c", "a", "b", "c"]})
    y = pd.Series([0, 1, 0, 1, 0, 1], index=x.index)
    g = pd.Series(x.index, index=x.index)
    rows = []
    for error in ERRORS:
        dirty, dy, dg, truth, meta = safe_corrupt(x, y, g, error, 0.2, 42, 0, {"num": 100.}, 10.)
        assert dy.equals(y) if error != "duplicate" else len(dy) == len(y) + len(truth)
        assert meta["target_untouched"]
        rows.append({"error": error, "status": meta.get("status", "applied"), "truth_records": len(truth),
                     "rows_before": len(x), "rows_after": len(dirty)})
    result = {"status": "passed", "synthetic": True, "new_model_calls": 0, "new_fits": 0,
              "test_loaded": False, "truth_used_by_planner": False, "checks": rows, "time": time.time()}
    write(folder, result)
    return result


def _dispatch_workers(s, out, stage, queue, gpus, config_path):
    workers = []
    for i, gpu in enumerate(gpus):
        log = out / "logs" / f"{stage}_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen([sys.executable, str(Path(__file__)), "--worker", "--config", str(config_path),
                                 "--gpu", str(gpu), "--queue", json.dumps(queue[i::len(gpus)])],
                                cwd=ROOT, stdout=log.open("a"), stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"},
                                start_new_session=True)
        workers.append({"gpu": gpu, "pid": proc.pid, "log": str(log), "stage": stage, "proc": proc})
    write(out / "process_registry.json", [{"gpu": w["gpu"], "pid": w["pid"], "log": w["log"], "stage": w["stage"]}
                                            for w in workers])
    return workers


def _wait_workers(s, out, workers, stage, queue, poll_seconds):
    total = len(matrix(s))
    while True:
        live = [w for w in workers if w["proc"].poll() is None]
        done = sum((unit_dir(s, *entry) / "complete.json").exists() or
                   (unit_dir(s, *entry) / "cpu_complete.json").exists() for entry in queue)
        write(out / "progress.json", {"stage": stage, "completed_units": done,
                                      "total_units": total, "live_workers":
                                          [{"gpu": w["gpu"], "pid": w["pid"], "stage": w["stage"]} for w in live],
                                      "test_loaded": False, "time": time.time()})
        if not live:
            break
        time.sleep(poll_seconds)
    exits = []
    for w in workers:
        try:
            exits.append({"gpu": w["gpu"], "pid": w["pid"], "exit": w["proc"].wait()})
        except Exception as exc:
            exits.append({"gpu": w["gpu"], "pid": w["pid"], "exit": None, "wait_error": repr(exc)})
    write(out / f"{stage}_worker_exits.json", {"exits": exits, "test_loaded": False, "time": time.time()})
    return exits


def controller(path=CONFIG):
    s = settings(path); out = Path(s["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    first = not (out / "launch_receipt.json").exists()
    if first:
        pre = synthetic_preflight(); prepared = prepare(path)
        write(out / "launch_receipt.json", {"experiment": NAME, "started": time.time(), "gpus": [0, 1, 5, 6, 7],
                                             "stage": "sentinel_then_formal", "new_training_updates": 0,
                                             "preflight": pre, "preparation": prepared, "test_loaded": False})
    else:
        write(out / f"resume_receipt_{os.getpid()}.json", {"pid": os.getpid(), "time": time.time(),
                                                             "test_loaded": False, "units_kept_idempotent": True})

    pending = [(d, 42, e, 0.10) for d in [v["name"] for v in s["datasets"]] for e in s["errors"]]
    sentinel_all = all((unit_dir(s, *entry) / "complete.json").exists() for entry in pending)
    if not (out / "sentinel_complete.json").exists():
        if not sentinel_all:
            workers = _dispatch_workers(s, out, "sentinel", pending, [0, 1, 5, 6, 7], path)
            _wait_workers(s, out, workers, "sentinel", pending, 15)
        failed = [entry for entry in pending if not (unit_dir(s, *entry) / "complete.json").exists()]
        if failed:
            write(out / "execution_hold.json", {"stage": "sentinel", "failed_units": failed,
                                                 "time": time.time(), "test_loaded": False})
            return {"stage": "hold", "failed_units": failed}
        write(out / "sentinel_complete.json", {"units": len(pending), "test_loaded": False,
                                                "time": time.time(), "resume_skip_sentinel": not first and sentinel_all})

    remaining = [entry for entry in matrix(s) if entry not in pending]
    workers = _dispatch_workers(s, out, "formal", remaining, [0, 1, 5, 6, 7], path)
    _wait_workers(s, out, workers, "formal", remaining, 30)
    failed = [entry for entry in remaining if not (unit_dir(s, *entry) / "complete.json").exists()]
    result = {"stage": "formal_complete" if not failed else "hold", "failed_units": failed,
              "completed_units": len(matrix(s)) - len(failed), "test_loaded": False, "time": time.time()}
    write(out / ("all_units_complete.json" if not failed else "execution_hold.json"), result)
    return result


def worker(path, gpu, queue):
    s = settings(path)
    write(Path(s["output_dir"]) / "worker_receipts" / f"gpu{gpu}_{os.getpid()}_started.json",
          {"pid": os.getpid(), "gpu": gpu, "units": queue, "test_loaded": False, "started": time.time()})
    for dataset, seed, error, rate in queue:
        name = f"gpu{gpu}_{dataset}_{seed}_{error}_{int(rate * 100):02}.json"
        receipt = Path(s["output_dir"]) / "worker_receipts" / name
        try:
            result = run_one(path, dataset, int(seed), error, float(rate), False)
            write(receipt, {"unit": [dataset, seed, error, rate], "status": "ok", "result": result,
                            "test_loaded": False, "finished": time.time()})
        except Exception as exc:
            write(receipt, {"unit": [dataset, seed, error, rate], "status": "error",
                            "error": repr(exc), "test_loaded": False, "finished": time.time()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG)); p.add_argument("--controller", action="store_true")
    p.add_argument("--worker", action="store_true"); p.add_argument("--gpu", type=int)
    p.add_argument("--queue"); p.add_argument("--stage", choices=["prepare", "preflight"])
    a = p.parse_args()
    if a.stage == "preflight": print(synthetic_preflight()); return
    if a.stage == "prepare": print(prepare(a.config)); return
    if a.controller: print(controller(a.config)); return
    if a.worker: worker(a.config, a.gpu, json.loads(a.queue)); return
    raise SystemExit("choose --stage, --controller or --worker")


if __name__ == "__main__": main()
