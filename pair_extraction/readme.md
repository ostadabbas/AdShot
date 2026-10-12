# pair_extraction

Clean, config-driven rebuild of the `pair_extraction/` pipeline. Mirrors the root
video-download pipeline: a central `config.yaml` + a thin orchestrator
(`run_pipeline.py`) + parameterized paths. The underlying algorithms are unchanged
from the original; this is a reorganization with targeted fixes.

## Pipeline flow

```
batchN_videos.csv
   │  s0_fix_brand_id.py        (b1306 -> 1306)
   ▼
batchN_videos_fix.csv
   │
   ▼  STAGE 1  s1_1_process_brands_main.py  (per brand)
   ├─ s1_shot_detection.py   AutoShot      -> shotsplits/{id}_{name}_shotsplits.json
   ├─ s1_keyframes.py        ffmpeg        -> keyframes/{id}/{ad}_shot_NN_pXX.jpg
   └─ s1_embedding.py        Swin3D-B      -> embeddings/{id}_{name}_embeddings.h5
   │
   ▼  STAGE 2  s2_clustering.py            DBSCAN per brand
                                           -> clusters/{id}_{name}_clusters.json
   │
   ▼  STAGE 3  s3_pairing.py               shot-overlap matching
                                           -> fullmatch.csv
```

## Quick start

```bash
PY=/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python
export TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch   # Swin3D weights

# one brand, all stages
$PY run_pipeline.py --stage all --brand_id 975

# individual stages
$PY run_pipeline.py --stage processing --brand_id 975
$PY run_pipeline.py --stage clustering --brand_id 975
$PY run_pipeline.py --stage pairing

# on the cluster (GPU)
sbatch runs/run_pipeline.sh                          # stage=all, default batch
STAGE=clustering BRAND_ID=975 sbatch runs/run_pipeline.sh
```

## Configuration (`config.yaml`)

All paths/params live in `config.yaml`, loaded by `config.py` with
`{data_root}`/`{version}`/`{output_dir}` templating. Key knobs:

| Section | Knob | Meaning |
|---------|------|---------|
| paths | `data_root` | output root (`/projects/adsum-mllm-om/data2`) |
| paths | `version` | pipeline version subdir (`0.1`) |
| paths | `video_dir` | source `videomedia/<brand>/<ad>.mp4` |
| paths | `default_batch` / `batch_csvs` | which CSV feeds the run |
| model | `autoshot_ckpt` | AutoShot checkpoint (vendored under `models/`) |
| params | `shot_threshold` | shot-boundary confidence (0.1) |
| params | `keyframe_percentiles` | `[0.1,0.25,0.5,0.75,0.9]` |
| params | `embedding_dim` | 1024 (Swin3D-B) |

## Stages

| Script | Stage | Output |
|--------|-------|--------|
| `s0_fix_brand_id.py` | brand_id normalization | `batchN_videos_fix.csv` |
| `s1_1_process_brands_main.py` | Stage 1 driver | shotsplits / keyframes / embeddings |
| `s1_shot_detection.py` | shot detection (AutoShot) | — |
| `s1_keyframes.py` | keyframe extraction | — |
| `s1_embedding.py` | Swin3D embeddings | — |
| `s2_clustering.py` | DBSCAN clustering | `clusters/*.json` |
| `s3_pairing.py` | shot-overlap pairing | `fullmatch.csv` |
| `run_pipeline.py` | orchestrator | runs `--stage {all,processing,clustering,pairing}` |

## Running each script individually

All commands assume this preamble (run from the package dir so `import config` resolves):

```bash
PY=/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python
export TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch
cd /projects/adsum-mllm-om/mllm/pair_extraction
```

Test brand used in the examples: `--brand_id 975` (`Aqua Living Factory Outlets`,
in `batch7_videos_fix.csv`). Each stage has a one-shot **smoke test** under `tests/`
that runs the stage and compares to reference — the fastest way to see if a script
works or where it breaks.

### config.py — central config loader (no CLI)
Sanity-check that paths/checkpoint resolve:
```bash
$PY -c "import config; print(config.OUTPUT_BASE_DIR); print(config.AUTOSHOT_MODEL_PATH); print(config.SHOT_THRESHOLD)"
```
Breaks if: `config.yaml` missing a key, or `models/ckpt_0_200_0.pth` absent.

### s0_fix_brand_id.py — brand_id normalization (CLI)
```bash
$PY s0_fix_brand_id.py --input /projects/adsum-mllm-om/data2/batch7_videos.csv
# -> writes batch7_videos_fix.csv next to the input
$PY s0_fix_brand_id.py --input in.csv --output out.csv     # explicit output
```
Smoke: `$PY tests/test_step0_fix_brand_id.py`

### s1_shot_detection.py — AutoShot (importable class, no CLI)
```bash
$PY -c "
import config; from s1_shot_detection import ShotDetector
d = ShotDetector(model_path=config.AUTOSHOT_MODEL_PATH, threshold=config.SHOT_THRESHOLD)
splits, preds = d.detect_shots('/projects/adsum-mllm-om/videomedia/Aqua Living Factory Outlets/XEEvoOwsqEY.mp4')
print('boundaries:', splits)"
```
Smoke: `$PY tests/test_step2_shot_detection.py`  (needs GPU/model + a local video)
Breaks if: checkpoint path wrong, video missing, or CUDA OOM.

### s1_keyframes.py — keyframe extraction (importable class, no CLI)
```bash
$PY -c "
from s1_keyframes import KeyframeExtractor
KeyframeExtractor().extract_keyframes(
    '/projects/adsum-mllm-om/videomedia/Aqua Living Factory Outlets/XEEvoOwsqEY.mp4',
    [0, 30, 60], 'XEEvoOwsqEY', '/tmp/kf_out', set())
print('done')"
```
Smoke: `$PY tests/test_step3_keyframes.py`  (needs `ffmpeg` on PATH; config sets it)
Breaks if: ffmpeg not found, or `shot_splits` not a list of ints.

### s1_embedding.py — Swin3D-B embeddings (importable class, no CLI)
```bash
$PY -c "
from s1_keyframes import KeyframeExtractor
from s1_embedding import EmbeddingGenerator
shots = KeyframeExtractor().find_valid_shots('/projects/adsum-mllm-om/data/0.1/keyframes/975')
EmbeddingGenerator().process_brand(shots[:2], 975, '/tmp/emb.h5', resume=False)
print('wrote /tmp/emb.h5')"
```
Smoke: `$PY tests/test_step4_embedding.py`  (downloads/loads Swin3D weights via TORCH_HOME)
Breaks if: TORCH_HOME unset (re-downloads weights), keyframes dir has != 5 frames/shot.

### s1_1_process_brands_main.py — Stage 1 driver (CLI)
Chains shots → keyframes → embeddings for a brand. Output defaults to `config.OUTPUT_BASE_DIR`.
```bash
$PY s1_1_process_brands_main.py \
    --data_csv /projects/adsum-mllm-om/data2/batch7_videos_fix.csv \
    --video_dir /projects/adsum-mllm-om/videomedia \
    --brand_id 975
# other flags: --brand "Name"  --output_dir DIR  --skip_shots  --skip_keyframes  --num_workers N
```
Smoke (end-to-end vs reference): `$PY tests/test_step5_driver.py`
Breaks if: brand has no video folder under `--video_dir`; status file resume skips work
(delete `<output_dir>/processing_status.json` to force a full re-run).

### s2_clustering.py — DBSCAN clustering (CLI)
Reads `embeddings/` + `keyframes/`, writes `clusters/{id}_{name}_clusters.json`. Defaults from config.
```bash
$PY s2_clustering.py --brand_id 975
# override dirs: --embeddings_dir DIR --keyframes_dir DIR --output_dir DIR
# all brands: omit --brand_id
```
Smoke: `$PY tests/test_step6_clustering.py`  (no GPU needed)
Breaks if: embeddings `.h5` missing for the brand, or keyframes dir name != `{brand_id}`.

### s3_pairing.py — shot-overlap pairing (CLI)
Reads `shotsplits/` + `clusters/` + ad-metadata CSV(s), writes `fullmatch.csv`. Defaults from config.
```bash
$PY s3_pairing.py                                   # all config defaults
$PY s3_pairing.py \
    --shotsplits_dir /projects/adsum-mllm-om/data/0.1/shotsplits \
    --clusters_dir   /projects/adsum-mllm-om/data/0.1/clusters \
    --data_csv /projects/adsum-mllm-om/data2/batch1_videos_wen.csv \
    --data_csv /projects/adsum-mllm-om/data2/batch2_videos_wen.csv \
    --out_path /tmp/fullmatch.csv
```
Smoke (vs committed reference): `$PY tests/test_step7_pairing.py`  (no GPU; ~2 min over full dataset)
Breaks if: a CSV lacks `relative_path`/`ad_length`; clusters dir contains non-JSON entries
(e.g. `.ipynb_checkpoints` — skipped with a warning).

### run_pipeline.py — orchestrator (CLI)
```bash
$PY run_pipeline.py --stage all        --brand_id 975   # full pipeline, one brand
$PY run_pipeline.py --stage processing --brand_id 975   # Stage 1 only
$PY run_pipeline.py --stage clustering --brand_id 975   # Stage 2 only
$PY run_pipeline.py --stage pairing                     # Stage 3 only
$PY run_pipeline.py --help                              # all flags
```
Smoke (CLI contract): `$PY tests/test_step8_pipeline.py`

### runs/run_pipeline.sh — sbatch wrapper (GPU)
```bash
sbatch runs/run_pipeline.sh                                # stage=all, default batch
STAGE=clustering BRAND_ID=975 sbatch runs/run_pipeline.sh
DATA_CSV=/path/batchN_videos_fix.csv sbatch runs/run_pipeline.sh
```

### Run-all smoke
Exercise every stage at once (per-stage PASS/SKIP/FAIL):
```bash
$PY tests/run_tests.py            # all (heavy stages included)
$PY tests/run_tests.py --quick    # skip heavy (GPU/video/full-dataset)
$PY tests/run_tests.py --list     # list stages
```

## Tests

`tests/` holds a re-runnable gate per stage, comparing against the reference
artifacts produced by the old pipeline. See `tests/README.md`.

```bash
$PY tests/run_tests.py --quick      # fast deterministic gates
$PY tests/run_tests.py              # everything (heavy stages included)
```

## Environment notes
- Use the `shot` conda env: `/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python`
  (the `mllm2` env hangs on import).
- Set `TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch` for the Swin3D weights.
