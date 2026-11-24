"""Full-row TabPFN adapter with chunked queries and unchanged frozen weights."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    import torch
    import tabpfn
    from tabpfn import TabPFNRegressor
    from tabpfn_power_finite_repair_20261118 import install

    repair_events = install()

    request = Path(sys.argv[1])
    folder = request.parent
    config = json.loads(request.read_text())
    assert not (folder / "started.json").exists()
    (folder / "started.json").write_text(json.dumps(dict(pid=__import__("os").getpid(), time=time.time())))
    assert digest(config["checkpoint"]) == config["checkpoint_sha256"]
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = np.load(folder / "input.npz", allow_pickle=False)
    bypass = len(data["train_x"]) > 10000
    model = TabPFNRegressor(
        n_estimators=config["n_estimators"],
        categorical_features_indices=data["categorical_indices"].tolist(),
        model_path=config["checkpoint"], device="cuda:0", random_state=0, n_jobs=2,
        fit_mode="fit_preprocessors", memory_saving_mode=True,
        ignore_pretraining_limits=bypass,
    )
    started = time.time()
    model.fit(data["train_x"], data["train_y"])
    query = data["predict_x"]
    pieces = []
    for offset in range(0, len(query), 256):
        pieces.append(np.asarray(model.predict(query[offset:offset + 256]), dtype=float))
    prediction = np.concatenate(pieces)
    count = min(256, len(query))
    replay = np.asarray(model.predict(query[:count]), dtype=float)
    np.testing.assert_allclose(prediction[:count], replay, atol=1e-5, rtol=1e-5)
    assert prediction.shape == (len(query),) and np.isfinite(prediction).all()
    np.save(folder / "predictions.npy", prediction)
    (folder / "complete.json").write_text(json.dumps(dict(
        fit_count=1, train_rows=len(data["train_x"]), prediction_rows=len(query),
        all_training_rows_used=True, pretraining_limit_guard_bypassed=bypass,
        query_chunk=256, replay_maxdiff=float(np.max(np.abs(prediction[:count] - replay))),
        replay_relative_tolerance=1e-5, checkpoint_sha256=config["checkpoint_sha256"],
        input_sha256=digest(folder / "input.npz"),
        predictions_sha256=digest(folder / "predictions.npy"),
        seconds=time.time() - started, tabpfn_version=tabpfn.__version__,
        power_overflow_repairs=repair_events,
    ), indent=2))


if __name__ == "__main__":
    main()
