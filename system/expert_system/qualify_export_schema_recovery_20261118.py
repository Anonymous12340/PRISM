"""Recover only unscored schema exclusions under an explicit type-aware protocol."""

import copy
import json
import os
from pathlib import Path
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd

import qualify_grounded_transfer_20261117 as q
from crossfit_neural_feature_20261118 import folds


ROOT = q.ROOT
PARENT = ROOT / "results/export_shift_difficulty_full_pilot_20261118"
OUT = ROOT / "results/export_shift_schema_recovery_full_20261118"
read, write, sha = q.read, q.write, q.sha


def validate(source, task):
    x, y = source["x"].copy(), source["y"].copy()
    assert 3 <= len(x.columns) <= 90, "schema_width"
    assert sum(not re.fullmatch(r"(v|x|att|attr|attribute|feature|col)?[0-9]+", str(c), re.I)
               for c in x) >= 3, "anonymous_schema"
    assert y.notna().all(), "missing_target"
    assert not any(str(c) == "export_system" or str(c).endswith("__export_unit") for c in x)
    numeric = list(x.select_dtypes(include=np.number))
    for column in x:
        assert not x[column].astype("string").equals(y.astype("string")), "literal_target_copy"
        if column in numeric:
            a, b = pd.to_numeric(x[column], errors="coerce"), pd.to_numeric(y, errors="coerce")
            if a.notna().all() and b.notna().all():
                assert not np.array_equal(np.asarray(a, float), np.asarray(b, float)), "numeric_target_copy"
        else:
            values = x[column].astype("string")
            assert values.dropna().nunique() == values.str.strip().str.casefold().dropna().nunique(), "category_collision"
    groups = pd.util.hash_pandas_object(x, index=False).astype(str)
    sizes = groups.value_counts()
    chosen, total = [], 0
    for group in sorted(groups.unique(), key=lambda value: q.key("column_bundle_rows_v1:" + str(value))):
        if total >= 1800:
            break
        if total + int(sizes[group]) <= 2400:
            chosen.append(group)
            total += int(sizes[group])
    mask = groups.isin(chosen)
    x, y, groups = x.loc[mask].copy(), y.loc[mask].copy(), groups.loc[mask]
    assert len(x) >= 500 and groups.nunique() >= 50, "insufficient_group_support"
    if task == "classification":
        assert y.nunique() == 2 and y.value_counts().min() >= 20
        assert y.value_counts(normalize=True).min() >= .03
        y = y.astype(int)
    else:
        y = y.astype(float)
        assert np.isfinite(y).all() and y.nunique() >= 20
    result = dict(source, x=x, y=y, groups=groups, task=dict(source["task"], type=task))
    return result, None, sorted(numeric, key=lambda value: q.key("column_bundle_numeric_v1:" + str(value)))[:2]


def main():
    sys.path.insert(0, str(ROOT / "results/full_novelty_return_pilot_20261117/runtime"))
    from taskdq.scoring import fit_predictor, predict
    from taskdq.e1 import metrics

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "qualification_launch_receipt.json").open("x") as receipt:
        json.dump(dict(time=time.time(), pid=os.getpid()), receipt)
    start = time.time()
    assert not read(PARENT / "conditional_complete.json")["science_started"]
    events = read(PARENT / "qualification/qualification_log.json")
    eligible = {str(row["id"]) for row in events
                if row.get("task") == "classification" and row.get("reason") == "unit_and_category_eligibility"}
    ranking = read(PARENT / "qualification/ranking.json")["candidates"]
    candidates = [row for row in ranking if row["id"] in eligible]
    request_paths = {}
    for path in (PARENT / "fetch_requests").glob("*.json"):
        request_paths[str(read(path)["data_id"])] = path.with_suffix(".joblib")
    carried = read(PARENT / "qualification/manifest.json")["development"]
    assert len(carried) == 2 and all(row["task"] == "regression" for row in carried)
    rows, log = list(carried), []
    families = {q.fam(row["family"]) for row in carried}
    schemas = {q.sig(joblib.load(ROOT / row["path"])["x"]) for row in carried}
    settings = read(ROOT / "results/full_novelty_return_pilot_20261117/run_config.json")["model"]
    fits = 0
    for item in candidates:
        if time.time() - start >= 600 or len(rows) == 4:
            break
        event = dict(id=item["id"], status="checking")
        try:
            assert q.fam(item["family"]) not in families, "used_family"
            ds = joblib.load(request_paths[item["id"]])
            metadata = read(PARENT / "qualification/metadata" / (item["id"] + ".json"))
            y = ds.target.astype("string")
            ok = y.notna()
            levels = sorted(y.loc[ok].unique())
            assert len(levels) == 2
            x = ds.data.loc[ok].copy()
            identifiers = [field["name"] for field in metadata["features"]
                           if str(field.get("is_row_identifier", "false")).lower() == "true"]
            x = x.drop(columns=[column for column in identifiers if column in x])
            source = dict(x=x, y=y.loc[ok].map({v:i for i,v in enumerate(levels)}).astype(int),
                          task=dict(type="classification", target=metadata["description"]["default_target_attribute"]),
                          pair=dict(name=item["family"], background=metadata["description"].get("description", "")[:4000]),
                          date_columns={}, numeric_type_hints=[])
            source, _, numeric = validate(source, "classification")
            assert q.sig(source["x"]) not in schemas, "used_schema"
            a, sealed = folds(source, 5, 42)[0]
            inner = copy.copy(source)
            for name in ("x", "y", "groups"):
                inner[name] = source[name].iloc[np.sort(a)]
            scores = []
            for tr, va in folds(inner, 3, 59):
                assert not set(inner["groups"].iloc[tr]) & set(inner["groups"].iloc[va])
                assert not set(inner["x"].iloc[tr].index) & set(source["x"].iloc[sealed].index)
                assert fits < 36
                model = fit_predictor(inner["x"].iloc[tr], inner["y"].iloc[tr], inner["task"], settings)
                scores.append(float(metrics(inner["y"].iloc[va], predict(model, inner["x"].iloc[va]),
                                            inner["task"])["auc"]))
                fits += 1
            event["reference_auc"] = scores
            assert .60 <= float(np.mean(scores)) <= .90, "training_reference_outside_registered_range"
            code = "VC" + str(sum(row["task"] == "classification" for row in rows) + 1)
            path = OUT / "qualification/sources" / code / "source.joblib"
            path.parent.mkdir(parents=True)
            source["pair"]["name"] = code
            joblib.dump(source, path)
            rows.append(dict(code=code, role="development", task="classification", path=str(path.relative_to(ROOT)),
                             source_sha256=sha(path), family=item["family"], openml_id=item["id"],
                             category_column=None, unit_columns=numeric, historical=False))
            families.add(q.fam(item["family"]))
            schemas.add(q.sig(source["x"]))
            event.update(status="qualified", code=code)
        except Exception as error:
            event.update(status="excluded", reason=str(error), error_class=type(error).__name__)
        log.append(event)
        write(OUT / "qualification/qualification_log.json", log)
    complete = len(rows) == 4
    write(OUT / "qualification/manifest.json", dict(complete=complete, training=[], development=rows,
          model_scores_used=True, outer_scores_used=False, reference_fits=fits, schema_only_recovery=True))
    write(OUT / "qualification_supervisor_complete.json",
          dict(qualified_complete=complete, science_started=False, reference_fits=fits,
               checked=len(log), seconds=time.time()-start))


if __name__ == "__main__":
    main()
