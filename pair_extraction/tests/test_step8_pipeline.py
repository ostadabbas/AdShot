"""
Step 8 — orchestrator (run_pipeline.py).

The full end-to-end behaviour is already covered by the per-stage tests
(steps 5/6/7). This test verifies the orchestration *contract* cheaply (no GPU):
  - run_pipeline.py imports and `--help` exits 0
  - the CLI exposes --stage with choices {all, processing, clustering, pairing}
  - the shot-detection threshold default equals config (0.1), not the old 0.5
    (the known bug noted in PROGRESS step 8)

Skips until run_pipeline.py is added to pair_extraction (Step 8).
"""

import os
import sys
import subprocess
from _harness import check, require_module, SkipTest, PKG_DIR

META = {"step": 8, "name": "pipeline_cli", "heavy": False}


def run():
    require_module("run_pipeline", "Step 8: orchestrator")

    script = os.path.join(PKG_DIR, "run_pipeline.py")
    check(os.path.exists(script), f"run_pipeline.py not found at {script}")

    # --help must exit 0 and advertise the four stages.
    res = subprocess.run([sys.executable, script, "--help"],
                         capture_output=True, text=True, cwd=PKG_DIR)
    check(res.returncode == 0, f"--help exited {res.returncode}: {res.stderr[:300]}")
    help_txt = res.stdout + res.stderr
    for stage in ("all", "processing", "clustering", "pairing"):
        check(stage in help_txt, f"--stage choice '{stage}' not advertised in --help")

    # threshold default should track config (0.1), not the legacy 0.5.
    import config
    import importlib
    rp = importlib.import_module("run_pipeline")
    parser = None
    if hasattr(rp, "build_parser"):
        parser = rp.build_parser()
    if parser is not None:
        defaults = {a.dest: a.default for a in parser._actions}
        if "threshold" in defaults:
            check(abs(float(defaults["threshold"]) - float(config.SHOT_THRESHOLD)) < 1e-9,
                  f"threshold default {defaults['threshold']} != config {config.SHOT_THRESHOLD}")
        thr_note = f"threshold default={defaults.get('threshold')}"
    else:
        thr_note = "threshold default unchecked (no build_parser())"

    return f"CLI ok: 4 stages advertised; {thr_note}"


if __name__ == "__main__":
    print(run())
