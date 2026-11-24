"""Compare two frozen neural experts on identical inputs and validation slots."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import TabPFN_contract_expert_20261118 as contract

full = contract.b
ROOT = full.ROOT
NAME = "TabICL_expert_comparison_20261118"
OUT = ROOT / "results" / NAME
CFG = ROOT / "configs" / (NAME + ".json")
ASSETS = ROOT / "results/TabICL_public_preflight_20261118"
WORKER = ROOT / "scripts/tabicl_local_worker_20261118.py"
TABICL_PY = ROOT / "tmp/tabicl_v2_20261118/bin/python"
read, write, sha = full.read, full.write, full.sha
tabpfn_context = full.context


def configure():
    for module in [contract, full]:
        module.NAME, module.OUT, module.CFG = NAME, OUT, CFG
    full.OLD = OUT
    full.__file__ = __file__
    # Same selection complexity for both expert pipelines.
    full.EXPERT = dict(kind="neural_expert", ids=["__public_units__", "__text_style__", "__TabPFN__"])
    full.BLEND = dict(kind="fixed_blend", ids=full.EXPERT["ids"] + ["__fixed_half_blend__"])
    contract.CLEAN = dict(kind="contract_neural_expert", ids=["__public_units__", "__text_style__", "__TabICL__"])
    contract.CBLEND = dict(kind="contract_fixed_blend", ids=contract.CLEAN["ids"] + ["__fixed_half_blend__"])


def tabicl_context(folder, task, train, y, valid):
    import numpy as np

    filename, digest, version = read(CFG)["tabicl"][task]
    checkpoint = ASSETS / filename
    folder.mkdir(parents=True)
    tx, vx, cats = full.matrix(train, valid)
    np.savez(folder / "input.npz", train_x=tx, train_y=np.asarray(y), predict_x=vx,
             categorical_indices=np.asarray(cats, dtype=np.int64))
    request = folder / "request.json"
    write(request, dict(task=task, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                        checkpoint_version=version, n_estimators=8))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with (folder / "worker.log").open("w") as log:
        subprocess.run([str(TABICL_PY), str(WORKER), str(request)], stdout=log,
                       stderr=subprocess.STDOUT, env=env, timeout=360, check=True)
    receipt = read(folder / "complete.json")
    assert receipt["version"] == "2.2.0"
    assert receipt["predictions_sha256"] == sha(folder / "predictions.npy")
    return np.load(folder / "predictions.npy", allow_pickle=False)


def normalized_tabpfn_context(folder, task, train, y, valid):
    unit = folder.parent if folder.name == "neural_final" else folder.parent.parent
    definitions = read(unit / "public_export_contract.json")
    return tabpfn_context(folder, task, contract.canonical(train, definitions), y,
                          contract.canonical(valid, definitions))


def synthetic():
    import numpy as np
    import pandas as pd

    base = ASSETS / "repaired_preflight"
    base.mkdir()
    rng = np.random.default_rng(11)
    x = pd.DataFrame(rng.normal(size=(72, 4)))
    x["category"] = [str(i % 4) for i in range(len(x))]
    x.loc[70, "category"] = "unseen"
    x.loc[71, "category"] = None
    train, valid = x.iloc[:60], x.iloc[60:]
    for task in ["classification", "regression"]:
        y = (train[0] > 0).astype(int) if task == "classification" else 3 * train[0] + train[1]
        prediction = tabicl_context(base / task, task, train, y, valid)
        assert len(prediction) == len(valid)
    write(ASSETS / "synthetic_complete.json", dict(synthetic_contexts=2, real_source_fits=0, offline=True))


def prepare():
    assert read(ASSETS / "synthetic_complete.json")["offline"]
    contract.prepare()
    protected = read(OUT / "protected_hashes.json")
    for filename, digest, _ in read(CFG)["tabicl"].values():
        path = ASSETS / filename
        assert sha(path) == digest
        protected[str(path.relative_to(ROOT))] = digest
    previous = ROOT / "results/TabPFN_contract_expert_pilot_20261118"
    for name in ["controller_complete.json", "report_hashes.json"]:
        path = previous / name
        protected[str(path.relative_to(ROOT))] = sha(path)
    write(OUT / "protected_hashes.json", protected)
    implementation = read(OUT / "implementation_frozen.json")
    for path in [Path(__file__).resolve(), WORKER, ROOT / "scripts/qualify_tabicl_expert_20261118.py"]:
        implementation[str(path.relative_to(ROOT))] = sha(path)
    write(OUT / "implementation_frozen.json", implementation)
    full.verify()
    contract.preflight()


ARM_NAMES = {
    "original_Full": "original_Full",
    "Full_neural_experts": "TabPFN_system",
    "direct_neural_reference": "TabPFN_direct",
    "fixed_blend_reference": "TabPFN_blend",
    "Full_contract_experts": "TabICL_system",
    "direct_clean_reference": "TabICL_direct",
    "blend_clean_reference": "TabICL_blend",
}


def report():
    import joblib
    import numpy as np
    import pandas as pd

    full.bind()
    full.verify()
    from taskdq.e1 import metrics
    from taskdq.scoring import predict

    frozen = read(OUT / "frozen_all_endpoints.json")
    assert len(frozen["inventory"]) == 28
    for path, digest in frozen["files"].items():
        assert sha(OUT / path) == digest

    queries = pd.DataFrame(read(OUT / "query_ledger.json"))
    counts = queries.groupby(["table", "method"]).size()
    assert len(counts) == 12 and counts.eq(5).all()
    fits = read(OUT / "primary_ledger.json")
    failed = 0
    for event in fits:
        folder = OUT / event["folder"]
        if event["status"] == "failed":
            assert (folder / "failure.json").exists()
            failed += 1
            continue
        assert event["status"] == "completed"
        data = joblib.load(folder / "data.joblib")
        replay = predict(joblib.load(folder / "model.joblib"), data["valid"])
        np.testing.assert_array_equal(replay, np.load(folder / "predictions.npy"))
    contexts = [path for unit in (OUT / "units").iterdir()
                for pattern in ["neural_CV/*/complete.json", "clean_neural_CV/*/complete.json",
                                "neural_final/complete.json", "clean_neural_final/complete.json"]
                for path in unit.glob(pattern)]
    assert len(contexts) == 32
    max_replay = max(read(path)["replay_maxdiff"] for path in contexts)
    assert max_replay <= 1e-5

    assert not (OUT / "outer_open_receipt.json").exists()
    write(OUT / "outer_open_receipt.json", dict(time=time.time(), endpoints=28,
          frozen_sha256=sha(OUT / "frozen_all_endpoints.json")))
    labels, rows = {}, []
    for entry in frozen["inventory"]:
        tag, task = entry["tag"], entry["task"]
        if tag not in labels:
            labels[tag] = np.load(OUT / "units" / tag / "sealed_outer_y.npy")
        scores = metrics(labels[tag], np.load(OUT / entry["path"]), dict(type=task))
        rows.append(dict(source=entry["source"], task=task, seed=42,
                         method=ARM_NAMES[entry["arm"]], N=5, **scores))
    frame = pd.DataFrame(rows)
    # This dispatch rule was chosen on historical training folds before new-table fitting.
    routed = frame[
        ((frame.task == "classification") & (frame.method == "TabICL_system"))
        | ((frame.task == "regression") & (frame.method == "TabPFN_system"))
    ].copy()
    routed["method"] = "task_routed_system"
    frame = pd.concat([frame, routed], ignore_index=True)
    results = {}
    for task, metric in [("classification", "auc"), ("regression", "rmse")]:
        subset = frame[frame.task == task]
        table = subset.pivot(index="source", columns="method", values=metric)
        for baseline in ["original_Full", "TabPFN_system"]:
            current = table.TabICL_system
            differences = 100 * (current - table[baseline]) if task == "classification" else 100 * (1 - current / table[baseline])
            result = dict(mean=float(differences.mean()), by_source=differences.to_dict(),
                          positive_sources=int((differences > 0).sum()))
            if task == "classification":
                f1 = subset.pivot(index="source", columns="method", values="f1")
                result["f1_pp"] = float(100 * (f1.TabICL_system - f1[baseline]).mean())
                result["baseline_auc"] = table[baseline].to_dict()
            results[f"{task}_vs_{baseline}"] = result

    classification = results["classification_vs_original_Full"]
    regression = results["regression_vs_original_Full"]
    controls = read(CFG)["gate"]
    passed = (classification["mean"] >= controls["auc_pp"] and classification["f1_pp"] >= 0
              and regression["mean"] >= controls["rmse_percent"]
              and classification["positive_sources"] == regression["positive_sources"] == 2
              and results["classification_vs_TabPFN_system"]["mean"] >= controls["vs_V1_auc_pp"]
              and results["regression_vs_TabPFN_system"]["mean"] >= -1e-12)
    routed_regression = frame[frame.task == "regression"].pivot(
        index="source", columns="method", values="rmse"
    )
    routed_deltas = 100 * (1 - routed_regression.task_routed_system / routed_regression.original_Full)
    routed_passed = (
        classification["mean"] >= controls["auc_pp"] and classification["f1_pp"] >= 0
        and classification["positive_sources"] == 2
        and float(routed_deltas.mean()) >= controls["rmse_percent"]
        and bool(routed_deltas.gt(0).all())
        and results["classification_vs_TabPFN_system"]["mean"] >= controls["vs_V1_auc_pp"]
    )
    routed_result = dict(
        classification_vs_original_Full=classification,
        regression_vs_original_Full=dict(mean=float(routed_deltas.mean()),
                                         by_source=routed_deltas.to_dict()),
        selection_rule="TabICL for classification, TabPFN for regression",
        derived_from_frozen_predictions=True,
        pilot_gate=bool(routed_passed),
    )
    summary = dict(effects=results, pilot_gate=bool(passed), task_routed=routed_result,
                   routed_pilot_gate=bool(routed_passed), formal_version_fixed=False,
                   primary_fits=len(fits), primary_failures=failed, primary_replay_max=0,
                   neural_contexts=len(contexts), neural_replay_max=max_replay,
                   logical_queries=len(queries), local_lm_calls=len(read(OUT / "LM_ledger.json")))
    write(OUT / "summary.json", summary)
    output = OUT / "deliverables"
    output.mkdir()
    frame.to_csv(output / "SCORE_REPORT.csv", index=False, float_format="%.16g")
    lines = [
        "# TabICLv2 专家小试", "",
        "四个独立新表，seed42，三组同 N5。TabICL 与 TabPFN 使用相同的公开格式预处理。",
        "AUC 和 F1 以百分点计，RMSE 以相对降幅计。原 Full 和两种专家的全部参考结果见 CSV。",
        "", "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```", "",
        "冻结公开模型用于预测，本批没有梯度训练。单种子结果用于小试，达门后仍需独立确认。",
    ]
    (output / "SCORE_REPORT.md").write_text("\n".join(lines) + "\n")
    write(OUT / "report_hashes.json", {path.name: sha(path) for path in output.iterdir()})
    full.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    configure()
    if args.stage == "synthetic":
        synthetic()
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "controller":
        contract.report = report
        contract.controller()
    elif args.stage in ["neural_worker", "final_worker"]:
        full.context = normalized_tabpfn_context
        getattr(full, args.stage)(args.index, args.workers)
    elif args.stage in ["clean_worker", "clean_final_worker"]:
        full.context = tabicl_context
        getattr(contract, args.stage)(args.index, args.workers)
    elif args.stage == "full_worker":
        full.full_worker(args.index, args.workers)
    else:
        parser.error("Unknown stage")
