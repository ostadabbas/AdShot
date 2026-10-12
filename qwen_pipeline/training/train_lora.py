#!/usr/bin/env python3
"""
train_lora.py — LoRA SFT of Qwen3-VL-8B-Instruct on the shot-selection task.

Reuses SFTShotSelectionDataset (frames from keyframes_k5.h5, eval-identical messages)
and make_collate_fn (prompt-length label masking). Trains with trl's SFTTrainer:
Qwen3VLForConditionalGeneration.forward(labels=...) returns .loss and accepts the
multimodal kwargs the collate emits — no compute_loss override needed.

Why SFTTrainer works with our custom (already-tokenized, multimodal) pipeline:
  - dataset is a plain torch Dataset yielding {messages, full_messages, target_text}
    (no prompt/completion/image keys), so SFTTrainer sets completion_only_loss=False
    and _is_vision_dataset=False — it never tries to auto-tokenize or auto-collate;
  - `dataset_kwargs={"skip_prepare_dataset": True}` skips SFTTrainer._prepare_dataset
    (its .map/.with_transform/truncate steps are datasets.Dataset-only and would choke
    on a torch Dataset); our make_collate_fn stays the sole source of masking;
  - we pass the already-PeftModel with peft_config=None, so SFTTrainer uses it as-is
    (no re-wrap) — the vision-tower freeze + gradient-checkpointing setup is preserved.

Single-GPU LoRA (vision tower frozen). Run via runs/run_sft_lora_h200.sh on SLURM, or
locally for the --limit smoke test. Requires trl (installed no-deps: `pip install
--no-deps trl==0.24.0`; only a >=4.56.1 transformers lower-bound, so the pinned 4.57.5
+ flash-attn2 stack is untouched).

  /projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/train_lora.py \
      --csv data/train.csv --jsonl data/train.jsonl --h5 keyframes/keyframes_k5.h5 \
      --output_dir training/runs/qwen3vl8b_lora_v1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# transformers.Trainer pulls in the TF/Keras integration path on import; this env
# (mllm2) has Keras 3, which transformers rejects. We never need TF for torch LoRA,
# so disable the TF backend before transformers is imported. Must precede any
# transformers import below.
os.environ.setdefault("USE_TF", "0")

import torch
from torch.utils.data import Subset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

# transformers>=4.5x refuses to torch.load() checkpoint optimizer/scheduler state unless
# torch>=2.6 (CVE-2025-32434). This env is torch 2.5.1 + flash-attn2 (upgrading torch would
# break the flash-attn ABI), and the only files we torch.load on --resume are OUR OWN
# checkpoint state — trusted, not the untrusted pickles the CVE concerns. trainer.py binds
# the guard into its own namespace at import time, so patch it there to a no-op. SFTTrainer
# inherits _load_optimizer_and_scheduler from transformers.trainer, so this patch still lands.
# (User-approved 2026-06-30: required for the 8h-wall resubmit/resume strategy.)
import transformers.trainer as _hf_trainer  # noqa: E402
_hf_trainer.check_torch_load_is_safe = lambda *a, **k: None

from trl import SFTConfig, SFTTrainer  # noqa: E402  (subclass transformers Trainer/TrainingArguments)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from sft_dataset import SFTShotSelectionDataset, make_collate_fn  # noqa: E402

LLM_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def preprocess_logits_for_metrics(logits, labels):
    """Argmax on-device before the Trainer accumulates eval predictions.

    Without this, the Trainer gathers full [N, T, vocab] logits across the entire
    eval set into host memory -> OOM. Returning the per-position argmax (token ids)
    keeps it to [N, T] ints.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """Teacher-forced accuracy on the (label-masked) response tokens.

    `preds` are already per-position argmax token ids (see preprocess_logits_for_metrics).
    Causal LM shifts by one: logits at position i predict token i+1, so compare
    preds[:, :-1] against labels[:, 1:] over the unmasked (!= -100) response span.

    Reports:
      token_accuracy  - fraction of response tokens predicted correctly
      seq_exact_match - fraction of samples whose entire response span is correct
    Note: this is teacher-forced (each step conditions on the gold prefix), so it is an
    optimistic upper bound on free-running generation accuracy, not the pairing metric.
    """
    import numpy as np
    preds, labels = eval_pred
    preds = np.asarray(preds)[:, :-1]
    labels = np.asarray(labels)[:, 1:]
    mask = labels != -100
    correct = (preds == labels) & mask

    n_tok = int(mask.sum())
    token_accuracy = float(correct.sum() / n_tok) if n_tok else 0.0

    per_total = mask.sum(axis=1)
    per_correct = correct.sum(axis=1)
    valid = per_total > 0
    seq_exact_match = (
        float((per_correct[valid] == per_total[valid]).mean()) if valid.any() else 0.0
    )
    return {"token_accuracy": token_accuracy, "seq_exact_match": seq_exact_match}


def build_model(model_name: str, attn_impl: str, lora_r: int = 16, lora_alpha: int = 32):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()        # frozen base + GC needs a grad path at the embeddings
    for p in model.model.visual.parameters():  # freeze vision tower (LoRA targets LLM linears only)
        p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=LLM_TARGETS))
    model.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--attn_impl", default="flash_attention_2",
                    choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank r.")
    ap.add_argument("--lora_alpha", type=int, default=None,
                    help="LoRA alpha (default: 2*r, matching the r=16/alpha=32 baseline).")
    ap.add_argument("--eval_csv", default=None,
                    help="In-loop EVAL split CSV, validated every epoch (e.g. data/eval.csv). None = no eval.")
    ap.add_argument("--eval_jsonl", default=None, help="Row-aligned jsonl for --eval_csv.")
    ap.add_argument("--eval_limit", type=int, default=None, help="Use first N eval samples (smoke).")
    ap.add_argument("--test_csv", default=None,
                    help="Final TEST split CSV, evaluated ONCE at the end of training (e.g. data/test_final.csv).")
    ap.add_argument("--test_jsonl", default=None, help="Row-aligned jsonl for --test_csv.")
    ap.add_argument("--test_limit", type=int, default=None, help="Use first N test samples (smoke).")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_strategy", default="steps", choices=["epoch", "steps", "no"])
    ap.add_argument("--save_every_epochs", type=float, default=0.25,
                    help="When save_strategy=steps: checkpoint every this many epochs.")
    ap.add_argument("--eval_every_epochs", type=float, default=0.5,
                    help="Run in-loop eval every this many epochs (steps-based).")
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Use first N samples (smoke test).")
    ap.add_argument("--max_steps", type=int, default=-1, help="Cap training steps (smoke test).")
    ap.add_argument("--report_to", default="wandb", choices=["wandb", "none"],
                    help="Metric logging backend. 'none' for smoke runs.")
    ap.add_argument("--wandb_project", default="qwen3vl8b-sft-lora")
    ap.add_argument("--run_name", default=None,
                    help="W&B run name (default: output_dir basename).")
    ap.add_argument("--resume", default="auto", choices=["auto", "off"],
                    help="auto: resume from latest checkpoint in output_dir if present "
                         "(for 8h-cap resubmits); off: always start fresh.")
    args = ap.parse_args()

    if args.report_to == "wandb":
        # WANDB_API_KEY is loaded from the repo-root .env (gitignored); wandb reads it
        # from the env. .env is searched upward from CWD, so this works from anywhere.
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(usecwd=True))
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
    print(f"[lora] r={args.lora_r} alpha={lora_alpha}")
    model = build_model(args.model, args.attn_impl, lora_r=args.lora_r, lora_alpha=lora_alpha)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"  # training: pads at the tail

    train_ds = SFTShotSelectionDataset(args.csv, args.jsonl, args.h5, fps=args.fps)
    if args.limit is not None:
        train_ds = Subset(train_ds, list(range(min(args.limit, len(train_ds)))))

    eval_ds = None
    if args.eval_csv:
        assert args.eval_jsonl, "--eval_csv requires --eval_jsonl"
        eval_ds = SFTShotSelectionDataset(args.eval_csv, args.eval_jsonl, args.h5, fps=args.fps)
        if args.eval_limit is not None:
            eval_ds = Subset(eval_ds, list(range(min(args.eval_limit, len(eval_ds)))))
    do_eval = eval_ds is not None

    # Final held-out TEST split — evaluated ONCE after training completes (not in the loop).
    test_ds = None
    if args.test_csv:
        assert args.test_jsonl, "--test_csv requires --test_jsonl"
        test_ds = SFTShotSelectionDataset(args.test_csv, args.test_jsonl, args.h5, fps=args.fps)
        if args.test_limit is not None:
            test_ds = Subset(test_ds, list(range(min(args.test_limit, len(test_ds)))))

    # Convert epoch-fractions -> optimizer steps. One optimizer step consumes
    # bs * grad_accum * world_size samples, so steps_per_epoch = ceil(N / that).
    import math
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_batch))
    save_steps = max(1, round(steps_per_epoch * args.save_every_epochs))
    print(f"[sched] steps/epoch={steps_per_epoch} (N={len(train_ds)}, eff_batch={eff_batch}) "
          f"-> save_steps={save_steps} (every {args.save_every_epochs} ep), "
          f"eval=every epoch")

    targs = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch" if do_eval else "no",  # validate on the eval split every epoch
        bf16=True, tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=save_steps,
        save_total_limit=2,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,   # keep pixel_values / image_grid_thw
        label_names=["labels"],
        report_to=["wandb"] if args.report_to == "wandb" else [],
        run_name=args.run_name or Path(args.output_dir).name,
        # --- SFT-specific: keep our custom pipeline; disable all of TRL's own data handling ---
        dataset_kwargs={"skip_prepare_dataset": True},  # skip TRL's datasets.Dataset preprocessing
        max_length=None,       # no truncation (our sequences are pre-built by the collate)
        packing=False,
        assistant_only_loss=False,  # masking is done in make_collate_fn, not by TRL
    )

    # Metrics are needed for the per-epoch eval AND the final test eval.
    want_metrics = do_eval or (test_ds is not None)
    trainer = SFTTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=make_collate_fn(processor, verbose_first=True),
        processing_class=processor,
        compute_metrics=compute_metrics if want_metrics else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if want_metrics else None,
    )

    # The gpu partition caps at 8h but a full run is ~16h, so the run spans multiple
    # resubmits. Auto-detect the latest checkpoint in output_dir and resume from it;
    # restores global_step/optimizer/scheduler so a resubmit continues, not restarts.
    resume = None
    if args.resume != "off" and os.path.isdir(args.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(args.output_dir)
    print(f"[resume] {'from ' + resume if resume else 'no checkpoint found -> fresh start'}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)   # LoRA adapter only
    processor.save_pretrained(args.output_dir)
    print(f"Saved adapter + processor to {args.output_dir}")

    # Final held-out TEST evaluation — run ONCE, after training completes. Teacher-forced
    # loss + token/seq metrics (same as the per-epoch eval); the pairing task metrics
    # (acc/f1/iou) are still computed post-merge via the eval pipeline. Logged to wandb
    # under the `test/` prefix and written to test_metrics.json in output_dir.
    if test_ds is not None:
        print(f"[test] evaluating final model on {len(test_ds)} held-out test rows ...")
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)  # -> output_dir/test_results.json
        import json as _json
        with open(Path(args.output_dir) / "test_metrics.json", "w") as f:
            _json.dump(test_metrics, f, indent=2)
        print(f"[test] final test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
