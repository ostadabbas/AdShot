#!/usr/bin/env python3
"""
sft_dataset.py — SFT dataset + collate for LoRA fine-tuning Qwen3-VL-8B on the
shot-selection task.

Reuses the eval data path verbatim: H5ShotSelectionDataset (keyframes/kf_dataset.py)
already assembles the exact eval [system, user] messages (PIL frames from the HDF5
container). Training only appends an assistant target turn and, in the collate,
masks the prompt tokens so loss is computed on the JSON answer alone.

The supervised target is the 2-field JSON stored in data/*.jsonl `answer`
({"selected_shots": [...], "total_length_sec": ...}). This matches what the eval
parser scores (parse_selected_indices extracts the shot integers; validate_selection
uses the durations) — no `explanations` field, by design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

QP = Path("/projects/adsum-mllm-om/mllm/qwen_pipeline")
KF = QP / "keyframes"
if str(KF) not in sys.path:
    sys.path.insert(0, str(KF))
from kf_dataset import H5ShotSelectionDataset  # noqa: E402  (reused verbatim)

from qwen_vl_utils import process_vision_info  # noqa: E402


def load_answers_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Answers in file order. Line i aligns positionally with csv row i."""
    out = []
    with open(jsonl_path) as f:
        for line in f:
            out.append(json.loads(line)["answer"])  # {"selected_shots":[...], "total_length_sec": float}
    return out


def _jsonl_ids(jsonl_path: str) -> List[str]:
    with open(jsonl_path) as f:
        return [json.loads(line)["id"] for line in f]


class SFTShotSelectionDataset(H5ShotSelectionDataset):
    """H5ShotSelectionDataset + an assistant target turn.

    Positional join: self.answers[idx] <-> csv row idx (verified 1114/1114). __init__
    guards alignment (length + focal_ad_id-in-id). __getitem__ adds:
      - target_text  : json.dumps(answer)
      - full_messages: messages + [{"role":"assistant", ...target_text...}]
    Rows whose frame decode failed (messages is None) are passed through unchanged
    and dropped in the collate.
    """

    def __init__(self, csv_path, jsonl_path, h5_path, fps: int = 5):
        super().__init__(csv_path, h5_path, fps=fps)
        self.answers = load_answers_jsonl(jsonl_path)
        assert len(self.answers) == len(self.df), (
            f"jsonl ({len(self.answers)}) vs csv ({len(self.df)}) length mismatch")
        ids = _jsonl_ids(jsonl_path)
        for i in range(len(self.df)):
            fa = str(self.df.loc[i, "focal_ad_id"])
            assert fa in ids[i], f"row {i}: focal_ad_id {fa} not in jsonl id {ids[i]}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        if item["messages"] is None:
            return item  # decode failed -> collate drops it
        target = json.dumps(self.answers[idx], ensure_ascii=False)
        item["target_text"] = target
        item["full_messages"] = item["messages"] + [
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        ]
        return item


def _image_token_id(processor) -> int:
    tid = getattr(processor, "image_token_id", None)
    if tid is None:
        tid = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return int(tid)


def _video_token_id(processor) -> int:
    tid = getattr(processor, "video_token_id", None)
    if tid is None:
        tid = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    return int(tid)


def make_collate_fn(processor, verbose_first: bool = False):
    """Build the SFT collate. Prompt-length masking: the generation-prompt text is an
    exact string prefix of the full conversation text, so re-tokenizing the prompt
    prefix (with the same images) gives the index where the assistant answer begins.
    Training uses RIGHT padding so real tokens are [0, real_len).
    """
    tok = processor.tokenizer
    assert tok.padding_side == "right", (
        f"training collate needs right padding, got {tok.padding_side!r}")
    IMAGE_TOKEN_ID = _image_token_id(processor)
    VIDEO_TOKEN_ID = _video_token_id(processor)
    state = {"printed": False}

    def collate(batch):
        batch = [b for b in batch if b.get("full_messages") is not None]
        if not batch:
            return None

        full_msgs = [b["full_messages"] for b in batch]
        prompt_msgs = [b["messages"] for b in batch]  # [system, user]

        full_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                      for m in full_msgs]
        prompt_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                        for m in prompt_msgs]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            full_msgs, return_video_kwargs=True)

        inputs = processor(
            text=full_texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt", **(video_kwargs or {}),
        )

        # Per-row prompt length AFTER image-token expansion: tokenize the prompt
        # prefix through the same processor (images live in the user turn only).
        prompt_lens = []
        for ptxt, pmsg in zip(prompt_texts, prompt_msgs):
            p_img, p_vid, p_vk = process_vision_info(pmsg, return_video_kwargs=True)
            p = processor(text=[ptxt], images=p_img, videos=p_vid,
                          return_tensors="pt", **(p_vk or {}))
            prompt_lens.append(int(p["input_ids"].shape[1]))

        input_ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        labels[attn == 0] = -100
        labels[input_ids == IMAGE_TOKEN_ID] = -100
        labels[input_ids == VIDEO_TOKEN_ID] = -100
        inputs["labels"] = labels

        if verbose_first and not state["printed"]:
            state["printed"] = True
            n_unmasked = int((labels != -100).sum())
            print(f"[smoke] rows={input_ids.shape[0]} seq_len={input_ids.shape[1]} "
                  f"unmasked_label_tokens={n_unmasked} (expect ~30-40 per row)")
            print(f"[smoke] padding_side={tok.padding_side} "
                  f"image_token_id={IMAGE_TOKEN_ID} n_image_tok={int((input_ids==IMAGE_TOKEN_ID).sum())}")
            row0 = input_ids[0][labels[0] != -100]
            print(f"[smoke] row0 prompt_len={prompt_lens[0]} target_decode={tok.decode(row0)!r}")
            assert full_texts[0].startswith(prompt_texts[0]), "prompt is not a prefix of full text!"

        return dict(inputs)

    return collate


# ----------------------------------------------------------------------------
# Debug endpoint: validate the dataset join + collate label masking on CPU only
# (no model load, no GPU). Run this before any training to catch masking/padding
# bugs cheaply.
#
#   /projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/sft_dataset.py \
#       --csv data/train.csv --jsonl data/train.jsonl --h5 keyframes/keyframes_k5.h5 \
#       --limit 3
# ----------------------------------------------------------------------------
def _check(args):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "right"
    tok = proc.tokenizer

    ds = SFTShotSelectionDataset(args.csv, args.jsonl, args.h5, fps=args.fps)
    print(f"dataset len: {len(ds)} (alignment asserts passed)")

    collate = make_collate_fn(proc, verbose_first=True)
    items = [ds[i] for i in range(min(args.limit, len(ds)))]
    batch = collate(items)
    input_ids, labels, attn = batch["input_ids"], batch["labels"], batch["attention_mask"]
    print(f"batch shapes: input_ids {tuple(input_ids.shape)} labels {tuple(labels.shape)}")

    all_ok = True
    for i in range(input_ids.shape[0]):
        real = int(attn[i].sum())
        pad_at_tail = bool((attn[i][:real] == 1).all() and (attn[i][real:] == 0).all())
        unmasked = labels[i][labels[i] != -100]
        dec = tok.decode(unmasked)
        ok = items[i]["target_text"] in dec
        all_ok &= ok and pad_at_tail
        print(f"row{i}: real_len={real} pad_at_tail={pad_at_tail} "
              f"n_unmasked={len(unmasked)} target_in_decode={ok}")
    print("CHECK PASSED" if all_ok else "CHECK FAILED")
    return 0 if all_ok else 1


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Validate SFT dataset + collate masking (CPU only).")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=3, help="Number of rows to collate-check.")
    raise SystemExit(_check(ap.parse_args()))


if __name__ == "__main__":
    main()
