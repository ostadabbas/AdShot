# Qwen3-VL-8B LoRA SFT — Eval Summary (2026-07-01)

Evaluation of the fine-tuned model (`qwen3vl8b_lora_v1_merged`) vs the base
`Qwen/Qwen3-VL-8B-Instruct`, both on the 124 held-out rows (`data/test.csv`),
fps=5, flash-attn2, bs=8, 0 errors each.

Result files:
- `sft_test.{json,csv}` — fine-tuned model
- `base_test.{json,csv}` — base model

## Headline: `ok` looks catastrophic, but the real metrics improved

`ok` is **not** a quality score. It is a hard constraint gate defined in
`evaluations/qwen3vl_infer_video_input.py:57`:

```
ok = (5 <= #shots <= 12) and (14.5 <= total_length <= 15.5) and (#shots < n)
```

Both models emit clean, parseable JSON — nothing failed to parse. `ok=False`
only means the selection fell outside the 5–12 shot / 15s length window.

| view                         | base      | SFT v1    |
|------------------------------|-----------|-----------|
| ok (15s length gate)         | 12 / 124  | **0 / 124** |

Looking at `ok` alone makes SFT look like a total failure. It isn't.

## Real selection metrics (averaged over all 124 rows)

| metric    | base   | SFT v1 | Δ        |
|-----------|--------|--------|----------|
| accuracy  | 0.5492 | 0.6115 | +0.062   |
| precision | 0.6166 | 0.6280 | +0.011   |
| recall    | 0.6346 | 0.8634 | **+0.229** |
| f1_score  | 0.5984 | 0.7093 | **+0.111** |
| iou       | 0.4443 | 0.5648 | **+0.121** |
| auc       | 0.5373 | 0.5673 | +0.030   |
| avg #shots selected | 10.2 | 13.5 | +3.3 |
| avg total length (s)| 18.7 | 24.7 | +6.0 |

SFT improves across the board. The win is driven by **recall (+22.9pt)** at
roughly flat precision — i.e. the model over-includes shots, catching more of
the ground-truth set but selecting more (and longer) than needed.

## Root cause: the training labels are long, the eval gate wants 15s

The model produces long selections because **that is what it was trained on**.
Target-length distribution of the training labels
(`data/train.jsonl::answer.total_length_sec`, n=1114):

- mean **18.1s**, median 17.6s, max 30s
- only **10%** fall inside the 14.5–15.5s window; **72% are >15.5s**
- #shots: mean 8.8, 79% already inside 5–12 (only the *length* is off)

Test split is the same shape: mean 18.0s, only 9% inside the 15s window.

Consequence: a model that perfectly reproduced the ground truth would still
score only ~10% `ok`. So `ok=0/124` for SFT is a **metric/label mismatch**, not
a broken model. The fine-tune correctly learned the ~18s label distribution and
runs slightly longer at inference (~24.7s).

## Open decision (determines the fix)

- **If the target is a fixed ~15s cut** (i.e. the `ok` gate is the real spec):
  the training data is wrong for that goal — labels must be re-derived/filtered
  to ~15s targets, then retrain. #shots (5–12) is already ~79% fine; only length
  is off. Training on 18s labels will never yield 15s output.

- **If the target is the matched-shot set** (~18s is correct): the `ok` gate is
  the wrong metric — drop it or widen the window, and report f1/iou, where SFT
  already wins clearly.

## Notes

- Harmless known bug: `evaluate2` prints `Overall accuracy (total): 0/3066.xx =
  0.0000` — a broken length-aggregate counter, not a real metric.
- Training run: job 8001840, 420 steps / 3 epochs, train_loss 0.157,
  eval_loss@ep1 0.259. Adapter: `training/runs/qwen3vl8b_lora_v1/`;
  merged: `training/runs/qwen3vl8b_lora_v1_merged/`.
