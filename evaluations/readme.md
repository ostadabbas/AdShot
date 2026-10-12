## Setup

```bash
export HF_TOKEN="<your token>"   # or set it in the repo-root .env
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p /projects/adsum-mllm-om/.hf_cache
export HF_HOME=/projects/adsum-mllm-om/.hf_cache
export HF_HUB_CACHE=/projects/adsum-mllm-om/.hf_cache/hub
export HF_DATASETS_CACHE=/projects/adsum-mllm-om/.hf_cache/datasets
export TRANSFORMERS_CACHE=/projects/adsum-mllm-om/.hf_cache/transformers
export HF_HUB_DISABLE_XET=1
cd /projects/adsum-mllm-om/mllm/evaluations
```

## Evaluating Qwen2.5_32B

```bash
python evaluate2.py --csv data/fullmatch.csv --model Qwen/Qwen2.5-VL-32B-Instruct --output_json results_qwen25vl_32b_instruct.json --output_csv results_qwen25vl_32b_instruct.csv
```

## Evaluating Qwen3_32B

```bash
python evaluate2.py --csv data/fullmatch.csv --model Qwen/Qwen3-VL-32B-Instruct --output_json results_qwen3vl_32b_instruct.json --output_csv results_qwen3vl_32b_instruct.csv
```

## Evaluating LlavaOneVision 72B

```bash
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python evaluate2.py --csv data/fullmatch.csv --model llava-hf/llava-onevision-qwen2-72b-ov-hf --output_json results_llava-onevision-qwen2-72b.json --output_csv results_llava-onevision-qwen2-72b.csv --attn_impl flash_attention_2
```

## Evaluating LlavaOneVision 7B

```bash
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python evaluate2.py --csv data/fullmatch.csv --model llava-hf/llava-onevision-qwen2-7b-ov-hf --output_json results_llava-onevision-qwen2-7b.json --output_csv results_llava-onevision-qwen2-7b.csv --attn_impl flash_attention_2
```

## Evaluating InternVL2.5 8B

```bash
python evaluate2_wen.py \
  --csv /projects/adsum-mllm-om/data/fullmatch.csv \
  --model "OpenGVLab/InternVL2_5-8B" \
  --output_json results_internvl25_8b_fps3.json \
  --output_csv results_internvl25_8b_fps3.csv \
  --fps 3
```

## Evaluating Molmo2 8B

```bash
python evaluate4.py --model allenai/Molmo2-8B --csv data/fullmatch823.csv --attn_impl flash_attention_2 --fps 5
```

## Build Message

```bash
projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python build_message2.py --csv data/fullmatch.csv
```
