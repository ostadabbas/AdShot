# pair_extraction tests

Persistent, re-runnable test gates for each pipeline stage. Each stage is checked
against the **reference artifacts** produced by the old pipeline
(`/projects/adsum-mllm-om/data/0.1/` + `data/fullmatch.csv`).

Wherever possible a stage is fed the *reference inputs* of the prior stage, so
float drift from one stage never compounds into the next stage's comparison. The
exception is the Stage-1 driver test (step 5), an end-to-end integration test.

## Running

```bash
PY=/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python
export TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch   # Swin3D weights

$PY tests/run_tests.py            # everything (heavy tests included)
$PY tests/run_tests.py --quick    # skip heavy (GPU / model / video / full-dataset) tests
$PY tests/run_tests.py --steps 0,6,7   # only these steps
$PY tests/run_tests.py --list     # list discovered tests
```

A single test can also be run directly: `$PY tests/test_step6_clustering.py`.

## Status semantics
- **PASS** — output matched the reference within the stage's tolerance.
- **SKIP** — stage module not built yet, or a required input is missing
  (e.g. the test video, or no GPU). Not a failure.
- **FAIL** — output diverged from the reference.

## Comparison strategy (exact + tolerant mix)
| Step | Stage | Comparison | Kind |
|------|-------|-----------|------|
| 0 | brand_id fix | regenerated `_fix.csv` byte-identical (CRLF-normalized) | exact |
| 2 | shot detection | shot boundary list for test ad | exact |
| 3 | keyframes | filename set + sampled md5 (test ad) | exact |
| 4 | embeddings | cosine ≥ 0.9999 vs reference (test ad) | tolerant |
| 5 | Stage-1 driver | shots+keyframes exact, embeddings cosine ≥ 0.9999 | mixed (e2e) |
| 6 | clustering | n_shots/n_clusters exact, membership ARI ≥ 0.95 | mixed |
| 7 | pairing | `fullmatch.csv` columns + row count + focal set | exact |
| 8 | orchestrator | CLI contract (stages, threshold default) | contract |

`heavy=True` tests need a GPU/model/video or run over the full dataset; `--quick`
skips them. The canonical test brand is `975 Aqua Living Factory Outlets`; only ad
`XEEvoOwsqEY` has a local video, so heavy Stage-1 tests target that ad.

## Contract for not-yet-built stages
- **Step 7** `s3_pairing.run_pairing(shotsplits_dir=None, clusters_dir=None,
  data_csvs=None, out_path=None)` — each defaults to config when `None`.
- **Step 8** `run_pipeline.py` — exposes `--stage {all,processing,clustering,pairing}`;
  optional `build_parser()` lets the test assert the threshold default tracks config.
