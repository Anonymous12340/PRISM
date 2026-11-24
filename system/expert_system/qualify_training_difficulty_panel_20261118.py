"""Screen a preregistered harder subgroup using training-only reference CV."""

import os
import subprocess
import sys

import qualify_contract_expert_20261118 as qualifier


ROOT = qualifier.ROOT
OUT = ROOT / "results/export_shift_difficulty_full_pilot_20261118"
read, write = qualifier.read, qualifier.write


def qualify():
    import copy
    import joblib
    import numpy as np

    import crossfit_neural_feature_20261118 as split_helper

    qualifier.OUT = OUT
    qualifier.Q = OUT / "qualification"
    sys.path.insert(0, str(ROOT / "results/full_novelty_return_pilot_20261117/runtime"))
    from taskdq.scoring import fit_predictor, predict
    from taskdq.e1 import metrics

    original_validate = qualifier.q.validate
    original_key = qualifier.key
    original_write = qualifier.write
    settings = read(ROOT / "results/full_novelty_return_pilot_20261117/run_config.json")["model"]
    log = []
    fits = 0
    qualifier.key = lambda value: original_key(value.replace(
        "contract_expert_sources_v1:", "training_difficulty_panel_v1:", 1))

    def annotated_write(path, value):
        if path.name == "ranking.json":
            value = dict(value, no_model_scores=False, training_only_difficulty_screen=True)
        if path.name == "manifest.json":
            value = dict(value, model_scores_used=True, outer_scores_used=False,
                         training_only_difficulty_screen=True, qualification_reference_fits=fits)
        original_write(path, value)

    def validate(source, task):
        nonlocal fits
        source, category, numeric = original_validate(source, task)
        if task != "classification":
            return source, category, numeric
        assert fits + 3 <= 60, "qualification_reference_fit_cap"
        folder = OUT / "training_difficulty" / str(len(log))
        folder.mkdir(parents=True)
        outer_train, outer_test = split_helper.folds(source, 5, 42)[0]
        outer_train = np.sort(outer_train)
        assert not set(source["groups"].iloc[outer_train]) & set(source["groups"].iloc[outer_test])
        inner = copy.copy(source)
        for name in ("x", "y", "groups"):
            inner[name] = source[name].iloc[outer_train]
        fold_scores = []
        for i, (a, z) in enumerate(split_helper.folds(inner, 3, 59)):
            assert not set(inner["groups"].iloc[a]) & set(inner["groups"].iloc[z])
            assert not set(inner["x"].iloc[a].index) & set(source["x"].iloc[outer_test].index)
            fits += 1
            model = fit_predictor(inner["x"].iloc[a], inner["y"].iloc[a], inner["task"], settings)
            prediction = predict(model, inner["x"].iloc[z])
            score = metrics(inner["y"].iloc[z], prediction, inner["task"])["auc"]
            joblib.dump(model, folder / f"model{i}.joblib")
            np.save(folder / f"prediction{i}.npy", prediction)
            fold_scores.append(float(score))
        average = float(np.mean(fold_scores))
        accepted = .60 <= average <= .90
        log.append(dict(anonymous_source=len(log), fold_auc=fold_scores, mean_auc=average,
                        accepted=accepted, reference_fits=fits))
        original_write(OUT / "training_difficulty/registry.json", log)
        if not accepted:
            raise ValueError("training_reference_auc_outside_registered_range")
        return source, category, numeric

    qualifier.write = annotated_write
    qualifier.q.validate = validate
    qualifier.main()


if __name__ == "__main__":
    if sys.argv[1:] == ["qualify"]:
        qualify()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "qualification_launch_receipt.json").open("x") as receipt:
            qualifier.json.dump(dict(time=qualifier.time.time(), pid=os.getpid()), receipt)
        with (OUT / "qualification.log").open("x") as log:
            process = subprocess.run(["timeout", "--kill-after=3s", "1200s", sys.executable,
                                      __file__, "qualify"], stdout=log, stderr=subprocess.STDOUT)
        manifest = OUT / "qualification/manifest.json"
        complete = process.returncode == 0 and manifest.exists() and read(manifest)["complete"]
        write(OUT / "qualification_supervisor_complete.json",
              dict(qualified_complete=complete, exit=process.returncode, science_started=False))
