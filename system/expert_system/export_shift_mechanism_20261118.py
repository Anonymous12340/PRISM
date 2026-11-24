"""Measure reversible export-format shift on fixed training-source folds."""

import argparse
import csv
import hashlib
import sys

import joblib
import numpy as np
import pandas as pd

import catboost_expert_probe_20261118 as base


ROOT = base.ROOT
OUT = ROOT / "results/export_shift_mechanism_20261118"
read, write, sha = base.read, base.write, base.sha


def export(frame, contracts, shifted):
    output = frame.copy()
    regime = np.array([int(hashlib.sha256(str(i).encode()).hexdigest()[:8], 16) % 2
                       for i in output.index])
    for contract in contracts:
        column = contract["column"]
        factor = np.full(len(output), 100.0) if shifted else np.where(regime == 0, 1.0, 1000.0)
        output[column] = pd.to_numeric(output[column]).astype(float) * factor
        output[contract["unit_column"]] = ["times_" + str(int(f)) for f in factor]
    for column in frame.select_dtypes(exclude="number"):
        values = frame[column].astype("string")
        style = " " + values + " " if shifted else values.where(regime == 0, values.str.upper())
        output[column] = style.astype(object).where(values.notna(), np.nan)
    output["export_system"] = "source_C" if shifted else np.where(regime == 0, "source_A", "source_B")
    return output


def restore(frame, contracts):
    output = frame.copy()
    for contract in contracts:
        column, marker = contract["column"], contract["unit_column"]
        factors = output[marker].astype("string").str.replace("times_", "", regex=False).astype(float)
        assert factors.isin([1.0, 100.0, 1000.0]).all()
        output[column] = pd.to_numeric(output[column]).astype(float) / factors
    for column in output.select_dtypes(exclude="number"):
        values = output[column].astype("string").str.strip().str.casefold()
        output[column] = values.astype(object).where(values.notna(), np.nan)
    return output.drop(columns=["export_system", *[c["unit_column"] for c in contracts]])


def worker(code):
    sys.path.insert(0, str(base.PARENT / "runtime"))
    from taskdq.scoring import fit_predictor, predict

    task = base.candidates()[code]
    unit = base.PARENT / "units" / (code + "__42")
    protected = read(base.OUT / code / "frozen.json")["inputs"]
    assert all(sha(unit / name) == digest for name, digest in protected.items())
    source = joblib.load(unit / "source.joblib")
    contracts = read(unit / "public_export_contract.json")
    clean = base.comparison.contract.canonical(source["x"], contracts)
    clean = clean.drop(columns=["export_system", *[c["unit_column"] for c in contracts]])
    for column in clean.select_dtypes(exclude="number"):
        values = clean[column].astype("string").str.strip()
        clean[column] = values.astype(object).where(values.notna(), np.nan)
    folder = OUT / code
    folder.mkdir()
    frozen = {}
    model_params = read(base.PARENT / "run_config.json")["model"]
    inference = base.comparison.tabicl_context if task == "classification" else base.comparison.tabpfn_context
    for i, fold in enumerate(read(unit / "inner_splits.json")):
        a, b = fold["train"], fold["validation"]
        assert not set(source["groups"].iloc[a]) & set(source["groups"].iloc[b])
        train = export(clean.iloc[a], contracts, False)
        conditions = {name: export(clean.iloc[b], contracts, shifted)
                      for name, shifted in [("shared_format", False), ("new_format", True)]}
        restored_train = restore(train, contracts)
        restored = {name: restore(frame, contracts) for name, frame in conditions.items()}
        pd.testing.assert_frame_equal(restored_train, clean.iloc[a], check_exact=False, rtol=1e-12, atol=1e-12)
        for frame in restored.values():
            pd.testing.assert_frame_equal(frame, clean.iloc[b], check_exact=False, rtol=1e-12, atol=1e-12)
        raw_model = fit_predictor(train, source["y"].iloc[a], source["task"], model_params)
        clean_model = fit_predictor(restored_train, source["y"].iloc[a], source["task"], model_params)
        expert_cache = {}
        for condition, valid in conditions.items():
            restored_valid = restored[condition]
            reference = restored["shared_format"]
            cache_key = "shared_format" if restored_valid.equals(reference) else condition
            if cache_key not in expert_cache:
                expert_cache[cache_key] = inference(folder / f"expert{i}_{cache_key}", task,
                                                    restored_train, source["y"].iloc[a], restored_valid)
            values = dict(raw_XGB=predict(raw_model, valid),
                          canonical_XGB=predict(clean_model, restored_valid),
                          canonical_expert=expert_cache[cache_key])
            for method, value in values.items():
                assert value.shape == (len(b),) and np.isfinite(value).all()
                path = folder / f"{i}_{condition}_{method}.npy"
                np.save(path, value)
                frozen[path.name] = sha(path)
    assert all(sha(unit / name) == digest for name, digest in protected.items())
    write(folder / "frozen.json", dict(predictions=frozen, inputs=protected,
                                        source_performance_selection=False))


def report():
    sys.path.insert(0, str(base.PARENT / "runtime"))
    from taskdq.e1 import metrics

    rows = []
    for code, task in base.candidates().items():
        unit = base.PARENT / "units" / (code + "__42")
        folder = OUT / code
        frozen = read(folder / "frozen.json")
        assert all(sha(unit / name) == digest for name, digest in frozen["inputs"].items())
        source = joblib.load(unit / "source.joblib")
        for i, fold in enumerate(read(unit / "inner_splits.json")):
            for condition in ("shared_format", "new_format"):
                for method in ("raw_XGB", "canonical_XGB", "canonical_expert"):
                    path = folder / f"{i}_{condition}_{method}.npy"
                    assert sha(path) == frozen["predictions"][path.name]
                    score = metrics(source["y"].iloc[fold["validation"]],
                                    np.load(path, allow_pickle=False), source["task"])
                    names = ("auc", "f1") if task == "classification" else ("rmse",)
                    rows.append(dict(source=code, task=task, fold=i, condition=condition, method=method,
                                     **{name: score[name] for name in names}))
    assert len(rows) == 108
    with (OUT / "SCORE_REPORT.csv").open("x") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "task", "fold", "condition", "method", "auc", "f1", "rmse"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["worker", "report"])
    parser.add_argument("--code")
    args = parser.parse_args()
    if args.stage == "worker":
        worker(args.code)
    else:
        report()
