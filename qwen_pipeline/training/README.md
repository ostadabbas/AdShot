# Qwen3-VL-8B LoRA SFT — shot-selection task

Fine-tune `Qwen/Qwen3-VL-8B-Instruct` to pick the subset of a 30s ad's shots that appear in the
15s short ad. This folder is **self-contained and additive**: it reuses the eval data path and the
eval entry point unchanged, and modifies nothing outside `training/`.

> Interpreter is always the explicit path — **no `$PY` shortcut**:
> `/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python`

> **Prerequisite (once):** training uses trl's `SFTTrainer`. Install it *no-deps* so the
> pinned transformers 4.57.5 + flash-attn2 stack is left untouched (trl only sets a
> `transformers>=4.56.1` lower bound):
> ```
> /projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python -m pip install --no-deps trl==0.24.0
> ```

---

## 1. Where this sits in the pipeline

```
 pair_extraction ─▶ data_prep.py ─▶ data/{train,test}.{csv,jsonl}      (labels: answer = {selected_shots,total_length_sec})
                          │
 keyframes/kf_build.py ─▶ keyframes/keyframes_k5.h5                     (frames: keyed by focal_ad_id, eval-identical pixels)
                          │
                          ▼
         ┌──────────────────────── training/ (this folder) ───────────────────────┐
         │  sft_dataset.py   frames(h5)+labels(jsonl) ─▶ masked SFT batches         │
         │  train_lora.py    base model + LoRA  ───────▶ adapter dir                │
         │  merge_adapter.py base + adapter   ─────────▶ standalone merged model dir│
         └──────────────────────────────────────────────────────────────────────┬─┘
                                                                                  ▼
 keyframes/kf_batch_infer.py --model <merged dir> --csv data/test.csv  ─▶  results/sft_test.{json,csv}
 (UNCHANGED eval entry point)                                              accuracy / precision / recall / f1 / iou
```

Design principle — **reuse over reinvention**: `sft_dataset.py` subclasses the eval dataset
(`keyframes/kf_dataset.py::H5ShotSelectionDataset`), so training frames/prompt are byte-identical to
eval; `merge_adapter.py` emits a plain model dir so the existing `kf_batch_infer.py` evaluates the
fine-tuned model with zero changes.

---

## 2. Scripts — each has its own runnable endpoint

Design principle — **single responsibility + independently debuggable**: every script does one thing
and is runnable on its own with `--help`. Validate cheap, CPU-only stages before paying for a GPU.

### `sft_dataset.py` — data + collate (and a CPU self-check)
| | |
|---|---|
| **Role** | `SFTShotSelectionDataset` appends the assistant target turn; `make_collate_fn` tokenizes + masks the prompt so loss is on the JSON answer only. |
| **Inputs** | `--csv` `--jsonl` (row-aligned) `--h5` `--fps` `--limit` |
| **Output** | Prints masking diagnostics; exit 0 = `CHECK PASSED`. No files, no GPU, no model weights. |
| **Endpoint (debug)** | validates the load-bearing label masking |

```
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/sft_dataset.py \
  --csv data/train.csv --jsonl data/train.jsonl --h5 keyframes/keyframes_k5.h5 --limit 3
```
Pass criteria: alignment asserts pass; per row `pad_at_tail=True`, `target_in_decode=True`, and the
decoded unmasked span equals the JSON answer ending in `<|im_end|>`.

### `train_lora.py` — LoRA SFT trainer
| | |
|---|---|
| **Role** | Loads base bf16 + flash-attn2, freezes vision tower, applies LoRA (r=16, α=32) to the LLM linears, trains with **trl `SFTTrainer`** (`dataset_kwargs={"skip_prepare_dataset": True}` + our `make_collate_fn` masking; already-PeftModel passed with `peft_config=None`). |
| **Inputs** | `--model --csv --jsonl --h5 --output_dir` (+ batch/accum/lr/epochs; `--limit`/`--max_steps` for smoke). |
| **Output** | `--output_dir`: LoRA **adapter** + processor (not full weights). |
| **Endpoint (smoke)** | tiny run that gates the GPU path |

GPU smoke test (1 step on 8 samples) — prints the same masking diagnostics on the first batch:
```
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/train_lora.py \
  --csv data/train.csv --jsonl data/train.jsonl --h5 keyframes/keyframes_k5.h5 \
  --output_dir /tmp/sft_smoke --limit 8 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps 1 --logging_steps 1 --save_strategy no \
  --report_to none
```
Full run is the SLURM job below (do not run on the shared interactive node — it OOMs).

**Metrics → W&B.** Training defaults to `--report_to wandb` (project `qwen3vl8b-sft-lora`, run name =
`--output_dir` basename). `WANDB_API_KEY` is read from the repo-root **`.env`** (gitignored) — `train_lora.py`
calls `load_dotenv(find_dotenv(usecwd=True))`, which walks up from CWD, so it works from any dir and on the
SLURM node (shared NFS). Pass `--report_to none` to disable (smoke runs should, to avoid throwaway W&B runs).
Local run logs land in `qwen_pipeline/wandb/` (gitignored); metrics live on wandb.ai.

**In-loop eval loss.** Pass `--eval_csv data/test.csv --eval_jsonl data/test.jsonl` to evaluate on the
held-out 124-row test set each epoch (`eval_strategy="epoch"`, `per_device_eval_batch_size=1`) — wandb then
shows `eval/loss` alongside `train/loss` to catch overfitting. The SLURM launcher sets this by default. This
is teacher-forced eval *loss* only; the task metrics (acc/f1/iou) are still the separate post-merge eval below.

### `merge_adapter.py` — adapter → standalone model
| | |
|---|---|
| **Role** | `merge_and_unload()` folds the adapter into base weights and saves a full model dir + processor. |
| **Inputs** | `--base --adapter <train output_dir> --out` |
| **Output** | `--out`: a model dir loadable by `evaluations/qwen3vl_infer_video_input.load_qwen3vl()`. |
| **Endpoint** | run on a GPU node (base weights loaded in memory) |
```
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/merge_adapter.py \
  --base Qwen/Qwen3-VL-8B-Instruct \
  --adapter training/runs/qwen3vl8b_lora_v2 \
  --out     training/runs/qwen3vl8b_lora_v2_merged
```

### `runs/run_sft_lora_h200.sh` — SLURM launcher
Dedicated H200, 16 cpus, 96GB, 8h; sets `HF_HOME` / `PYTORCH_CUDA_ALLOC_CONF`; calls `train_lora.py`
with the production hyper-params. Submit: `sbatch training/runs/run_sft_lora_h200.sh` (from
`qwen_pipeline/`). Watch: `squeue -u $USER` and `logs/sft_lora_<jobid>.out`.

---

## 3. End-to-end run order (debug-first)

```
# 1. CPU collate check (seconds, no GPU)
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/sft_dataset.py \
  --csv data/train.csv --jsonl data/train.jsonl --h5 keyframes/keyframes_k5.h5 --limit 3

# 2. GPU smoke (1 step) — see train_lora.py block above

# 3. Full training on SLURM
sbatch training/runs/run_sft_lora_h200.sh        # -> training/runs/qwen3vl8b_lora_v2/

# 4. Merge for eval
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/merge_adapter.py \
  --base Qwen/Qwen3-VL-8B-Instruct --adapter training/runs/qwen3vl8b_lora_v2 \
  --out training/runs/qwen3vl8b_lora_v2_merged

# 5. Eval the fine-tuned model on the held-out 124 (UNCHANGED entry point)
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python keyframes/kf_batch_infer.py \
  --model training/runs/qwen3vl8b_lora_v2_merged \
  --csv data/test.csv --h5 keyframes/keyframes_k5.h5 \
  --output_json results/sft_test.json --output_csv results/sft_test.csv \
  --fps 5 --attn_impl flash_attention_2 --batch_size 8

# 6. Base run on the SAME 124 for comparison
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python keyframes/kf_batch_infer.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --csv data/test.csv --h5 keyframes/keyframes_k5.h5 \
  --output_json results/base_test.json --output_csv results/base_test.csv \
  --fps 5 --attn_impl flash_attention_2 --batch_size 8
```

---

## 4. Design decisions & contracts (so changes don't silently break)

- **Target = 2-field JSON** (`selected_shots`, `total_length_sec`) from `data/*.jsonl::answer`. No
  `explanations` (the system prompt mentions it, but GT has none). This is exactly what the eval parser
  scores, so SFT optimizes the metric. GT `total_length_sec` is often <14s and does *not* satisfy the
  eval's 14.5–15.5s "valid" gate — we train on true GT regardless; the gate only affects the eval `ok` flag.
- **Positional join** between jsonl and csv (line i ↔ row i), guarded by asserts in `__init__`
  (length + `focal_ad_id in id`). String-parsing the id is fragile (some focal ids contain `__`).
- **Label masking** uses the prompt-prefix length: `add_generation_prompt=True` text is an exact prefix
  of the full text, so the first `prompt_len` tokens (after identical image expansion) are the prompt.
  Verified in the `--check` endpoint by decoding the unmasked span.
- **Right padding for training** (eval flips to left for generation). The collate asserts
  `padding_side == "right"`; left padding would break "first prompt_len tokens".
- **`remove_unused_columns=False`** is mandatory — else `Trainer` strips `pixel_values`/`image_grid_thw`.
- **Vision tower frozen**; LoRA targets only `q,k,v,o,gate,up,down_proj`. `enable_input_require_grads()`
  + `use_reentrant=False` make gradient checkpointing work with the frozen base.
- **bs=1**: a single ad can carry ~13 shots × 5 frames; sequences reach ~10k tokens. Effective batch via
  `gradient_accumulation_steps=8`.

---

## 5. Eval/test splits, per-epoch eval, and the r-sweep (2026-07-02)

**Split** — the original 124-row `data/test.{csv,jsonl}` is split (seeded, csv↔jsonl aligned) into:
- `data/eval.{csv,jsonl}` (62 rows) — **in-loop validation, run every epoch**.
- `data/test_final.{csv,jsonl}` (62 rows) — **final held-out test, evaluated ONCE at end of training**.
Regenerate: `python data/split_test_eval.py --frac_eval 0.5 --seed 0`. Originals are left untouched.

`train_lora.py` args:
- `--eval_csv/--eval_jsonl` → `eval_strategy="epoch"` → wandb `eval/loss`, `eval/token_accuracy`, `eval/seq_exact_match` each epoch.
- `--test_csv/--test_jsonl` → after `trainer.train()`, one `trainer.evaluate(..., metric_key_prefix="test")` → `test/*` on wandb + `test_metrics.json`/`test_results.json` in the output dir.
- `--lora_r` (default 16), `--lora_alpha` (default `2*r`).

**LoRA rank sweep** — three self-contained launchers (gpu/h200, 8h, auto-resume, resubmit to continue):
```
sbatch training/runs/run_sft_lora_r16.sh   # -> training/runs/qwen3vl8b_lora_r16
sbatch training/runs/run_sft_lora_r32.sh   # -> training/runs/qwen3vl8b_lora_r32
sbatch training/runs/run_sft_lora_r64.sh   # -> training/runs/qwen3vl8b_lora_r64
```
Quick end-to-end validation before a real run (1h `sharing` h200, 2 rows, 1 epoch):
```
sbatch training/runs/dry_run_sft.sh
```
The HF token is **not** hardcoded in these files — it is read at runtime from `run_sft_lora_h200.sh`.
