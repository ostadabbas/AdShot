#!/usr/bin/env python
"""
Test runner for the pair_extraction pipeline.

Discovers test_step*.py modules in this directory, each exposing:
  META = {"step": int, "name": str, "heavy": bool}
  def run(): ...   # returns None=pass, raises SkipTest=skip, else=fail

Usage:
  python tests/run_tests.py                 # run everything (heavy tests included)
  python tests/run_tests.py --quick         # skip heavy tests (GPU/model/video)
  python tests/run_tests.py --steps 0,2,7   # only these step numbers
  python tests/run_tests.py --list          # list discovered tests

Run with the `shot` env:
  PY=/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python
  TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch $PY tests/run_tests.py
"""

import argparse
import glob
import importlib.util
import os
import sys
import time
import traceback

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.dirname(TESTS_DIR))  # package dir

from _harness import SkipTest  # noqa: E402

PASS, SKIP, FAIL = "PASS", "SKIP", "FAIL"
_ICON = {PASS: "✅", SKIP: "⏭️ ", FAIL: "❌"}


def discover():
    """Load each test_step*.py as a module; return list sorted by step."""
    mods = []
    for path in glob.glob(os.path.join(TESTS_DIR, "test_step*.py")):
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        meta = getattr(mod, "META", {"step": 999, "name": name, "heavy": False})
        mods.append((meta, mod))
    mods.sort(key=lambda m: m[0].get("step", 999))
    return mods


def run_one(meta, mod):
    """Execute a single test; return (status, detail, seconds)."""
    t0 = time.time()
    try:
        detail = mod.run()
        return PASS, (detail or ""), time.time() - t0
    except SkipTest as e:
        return SKIP, str(e), time.time() - t0
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        return FAIL, f"{type(e).__name__}: {e}  [{tb}]", time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="pair_extraction test runner")
    ap.add_argument("--quick", action="store_true", help="skip heavy (GPU/model/video) tests")
    ap.add_argument("--steps", type=str, default=None, help="comma-separated step numbers to run")
    ap.add_argument("--list", action="store_true", help="list discovered tests and exit")
    args = ap.parse_args()

    mods = discover()

    if args.list:
        print(f"{'step':>4}  {'heavy':>5}  name")
        for meta, _ in mods:
            print(f"{meta.get('step'):>4}  {str(meta.get('heavy', False)):>5}  {meta.get('name')}")
        return 0

    want = None
    if args.steps:
        want = {int(s) for s in args.steps.split(",") if s.strip()}

    results = []
    for meta, mod in mods:
        step = meta.get("step")
        if want is not None and step not in want:
            continue
        if args.quick and meta.get("heavy"):
            results.append((meta, SKIP, "skipped (heavy; --quick)", 0.0))
            continue
        status, detail, secs = run_one(meta, mod)
        results.append((meta, status, detail, secs))
        print(f"{_ICON[status]} step {step:>2} {meta.get('name'):<26} {status}  ({secs:.1f}s)")
        if detail:
            print(f"        {detail}")

    n_pass = sum(1 for _, s, _, _ in results if s == PASS)
    n_skip = sum(1 for _, s, _, _ in results if s == SKIP)
    n_fail = sum(1 for _, s, _, _ in results if s == FAIL)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_pass} passed, {n_skip} skipped, {n_fail} failed")
    print("=" * 60)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
