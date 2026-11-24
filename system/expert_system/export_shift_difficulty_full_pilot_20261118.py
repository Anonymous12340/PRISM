"""Reuse the frozen method on the preregistered training-difficulty panel."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import export_shift_full_pilot_20261118 as method


ROOT = method.ROOT
NAME = os.environ.get("EXPORT_SHIFT_RUN_NAME", "export_shift_difficulty_full_pilot_20261118")
assert NAME in ("export_shift_difficulty_full_pilot_20261118", "export_shift_schema_recovery_full_20261118")
OUT = ROOT / "results" / NAME
CFG = ROOT / "configs" / (NAME + ".json")


def configure():
    method.NAME, method.OUT, method.CFG = NAME, OUT, CFG
    method.__file__ = __file__
    method.configure()


def prepare():
    if NAME == "export_shift_schema_recovery_full_20261118":
        import qualify_export_schema_recovery_20261118 as qualification
        import qualify_grounded_transfer_20261117 as original
        with patch.object(original, "validate", qualification.validate):
            method.prepare()
    else:
        method.prepare()
    frozen = method.read(OUT / "implementation_frozen.json")
    paths = [ROOT / "scripts/export_shift_full_pilot_20261118.py",
             ROOT / "scripts/qualify_training_difficulty_panel_20261118.py",
             ROOT / "designs/training_difficulty_panel_20261118/PLAN.md"]
    if NAME == "export_shift_schema_recovery_full_20261118":
        paths.append(ROOT / "scripts/qualify_export_schema_recovery_20261118.py")
    for path in paths:
        frozen[str(path.relative_to(ROOT))] = method.sha(path)
    method.write(OUT / "implementation_frozen.json", frozen)
    method.b.verify()


def supervise():
    with (OUT / "conditional_receipt.json").open("x") as receipt:
        json.dump(dict(pid=os.getpid(), time=time.time()), receipt)
    deadline = method.read(OUT / "qualification_launch_receipt.json")["time"] + 1230
    outcome = OUT / "qualification_supervisor_complete.json"
    while not outcome.exists() and time.time() < deadline:
        time.sleep(4)
    if not outcome.exists() or not method.read(outcome)["qualified_complete"]:
        method.write(OUT / "conditional_complete.json", dict(science_started=False,
                                                            reason="qualification_incomplete"))
        return
    assert not (OUT / "dispatch.json").exists()
    with (OUT / "prepare.log").open("x") as log:
        subprocess.run(["timeout", "--kill-after=3s", "900s", sys.executable, __file__, "prepare"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    with (OUT / "controller.log").open("x") as log:
        process = subprocess.Popen([sys.executable, __file__, "controller"],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        method.write(OUT / "launch_receipt.json", dict(pid=process.pid, time=time.time(), hard_seconds=2700))
        result = process.wait()
    method.write(OUT / "conditional_complete.json", dict(science_started=True, exit=result))
    if result:
        raise SystemExit(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    configure()
    if args.stage == "supervise":
        try:
            supervise()
        except BaseException as error:
            method.write(OUT / "conditional_failure.json",
                         dict(error_class=type(error).__name__, traceback=traceback.format_exc()))
            raise
    elif args.stage == "prepare":
        prepare()
    elif args.stage in ("neural_worker", "final_worker"):
        method.b.cases = method.normalized_records
        getattr(method.b, args.stage)(args.index, args.workers)
    elif args.stage == "full_worker":
        method.b.full_worker(args.index, args.workers)
    elif args.stage == "raw_final_worker":
        method.bind()
        method.raw_final_worker(args.index, args.workers)
    else:
        getattr(method, args.stage)()
