#!/usr/bin/env python3
"""
evaluate_internvl.py

InternVL-only evaluation script for AdShot benchmark.
Uses transformers==4.45.0 (compatible with InternVL generate method).

Supported models:
- OpenGVLab/InternVL2_5-8B
- OpenGVLab/InternVL2_5-26B
- OpenGVLab/InternVL2_5-4B
- OpenGVLab/InternVL2_5-38B

Usage:
    python evaluate_internvl.py \
        --csv /path/to/fullmatch.csv \
        --model OpenGVLab/InternVL2_5-8B \
        --output_json results_internvl.json \
        --output_csv results_internvl.csv
"""

import pandas as pd
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Import utilities
from utils import extract_shots_from_mapping, create_one_hot, calculate_metrics

# Import InternVL-specific functions
from internvl25_infer import load_internvl25, inference_internvl25


def load_existing_results(output_json):
    """Load existing results from JSON file if it exists."""
    if output_json.exists():
        with open(output_json, 'r') as f:
            results = json.load(f)
        # Convert list to dict keyed by row index for faster lookup
        return {r['row']: r for r in results}
    return {}


def save_results(results_dict, output_json):
    """Save results dict to JSON file."""
    results_list = list(results_dict.values())
    with open(output_json, 'w') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)


def extract_metrics_to_csv(output_json, output_csv):
    """Read JSON results and extract metrics to CSV."""
    with open(output_json, 'r') as f:
        results = json.load(f)
    
    metrics_list = []
    for result in results:
        row_data = {
            'row': result['row'],
            'video': result['video'],
        }
        
        if 'error' in result:
            row_data['error'] = result['error']
            row_data['accuracy'] = None
            row_data['precision'] = None
            row_data['recall'] = None
            row_data['f1_score'] = None
            row_data['iou'] = None
            row_data['auc'] = None
            row_data['ok'] = None
            row_data['total'] = None
        else:
            row_data['ok'] = result['ok']
            row_data['total'] = result['total']
            row_data['accuracy'] = result.get('accuracy')
            row_data['precision'] = result.get('precision')
            row_data['recall'] = result.get('recall')
            row_data['f1_score'] = result.get('f1_score')
            row_data['iou'] = result.get('iou')
            row_data['auc'] = result.get('auc')
            row_data['selected_shots'] = ','.join(map(str, result['selected']))
            row_data['gt_shots'] = ','.join(map(str, result['gt']))
        
        metrics_list.append(row_data)
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(output_csv, index=False)
    
    # Calculate and print mean metrics
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'iou', 'auc']
        mean_metrics = {}
        
        for metric in metric_names:
            values = [r.get(metric) for r in valid_results if r.get(metric) is not None]
            if values:
                mean_metrics[metric] = sum(values) / len(values)
            else:
                mean_metrics[metric] = None
        
        total_ok = sum(r['ok'] for r in valid_results if r.get('ok') is not None)
        total_count = sum(r['total'] for r in valid_results if r.get('total') is not None)
        
        print(f"\n{'='*60}")
        print(f"METRICS SUMMARY")
        print(f"{'='*60}")
        print(f"Total videos processed: {len(results)}")
        print(f"Valid results: {len(valid_results)}")
        print(f"Errors: {len(results) - len(valid_results)}")
        print(f"Overall accuracy (total): {total_ok}/{total_count} = {total_ok/total_count:.4f}" if total_count > 0 else "N/A")
        print(f"-" * 60)
        print("Mean metrics across all videos:")
        for metric, value in mean_metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: N/A")
        print(f"{'='*60}\n")
    else:
        print("\nNo valid results to calculate metrics.")
    
    return df_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project_root",
        default="/projects/adsum-mllm-om",
        help="Absolute path to repo root that should be added to sys.path",
    )
    ap.add_argument("--model", required=True, help="InternVL model name (e.g., OpenGVLab/InternVL2_5-8B)")
    ap.add_argument("--csv", required=True, help="Path to eval.csv")
    ap.add_argument("--output_json", default="results.json", help="Path to output JSON file")
    ap.add_argument("--output_csv", default="metrics.csv", help="Path to output metrics CSV file")
    ap.add_argument("--fps", type=int, default=5, help="Frames per shot (will be overridden to 3 for InternVL)")
    ap.add_argument("--max_pixels", type=int, default=360 * 420, help="Max pixels per image")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = deterministic)")
    ap.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling (only used if temperature > 0)")
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"--project_root does not exist: {project_root}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    
    # Load existing results
    results_dict = load_existing_results(output_json)
    print(f"Loaded {len(results_dict)} existing results from {output_json}")

    df = pd.read_csv((project_root / args.csv).resolve())

    # Load InternVL model
    print(f"Loading InternVL model: {args.model}")
    #print(f"Note: Will use 3 frames per shot (10%, 50%, 90%) to fit 16K context limit")
    
    model, tokenizer = load_internvl25(args.model)
    
    print(f"\nModel loaded successfully!")
    print(f"Starting evaluation on {len(df)} videos...")
    print(f"Results will be saved incrementally to: {output_json}\n")

    for row_idx in tqdm(range(len(df)), desc="Processing videos"):
        # Skip if already processed
        if row_idx in results_dict:
            tqdm.write(f"Row {row_idx} already processed, skipping...")
            continue
            
        #check by focal ad id, target ad id
        
        video_path = (project_root / Path(df.loc[row_idx, "focal_relative_path"])).resolve()
        if not video_path.exists():
            results_dict[row_idx] = {
                "row": row_idx,
                "video": str(video_path),
                "error": "video file not found",
            }
            save_results(results_dict, output_json)
            continue

        video_info = df.loc[row_idx, :].to_dict()

        try:
            # Extract ground truth from video_info
            selected_shots_gt, all_shots = extract_shots_from_mapping(video_info['mapping'])
            one_hot_gt = create_one_hot(all_shots, selected_shots_gt)
            
            # Run InternVL inference
            out, ok, total, selected_shots_pred = inference_internvl25(
                model=model,
                processor=tokenizer,
                video_path=video_path,
                video_info=video_info,
                fps=args.fps,  
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            # Calculate metrics
            one_hot_pred = create_one_hot(all_shots, selected_shots_pred)
            metrics = calculate_metrics(one_hot_gt, one_hot_pred)
            
            results_dict[row_idx] = {
                "row": row_idx,
                "video": str(video_path.resolve()),
                "answers": out,
                "ok": ok,
                "total": total,
                "selected": selected_shots_pred,
                "gt": selected_shots_gt,
                "accuracy": metrics['accuracy'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1_score": metrics['f1_score'],
                "iou": metrics['iou'],
                "auc": metrics['auc']
            }
            
            tqdm.write(f"Row {row_idx}: ✓ Processed (accuracy={metrics['accuracy']:.3f})")
            
        except Exception as e:
            tqdm.write(f"Row {row_idx}: ✗ Error - {str(e)}")
            results_dict[row_idx] = {
                "row": row_idx,
                "video": str(video_path),
                "error": str(e),
            }
        
        # Save after each iteration
        save_results(results_dict, output_json)

    print(f"\nAll results saved to {output_json}")
    
    # Extract metrics to CSV and calculate means
    print(f"\nExtracting metrics to {output_csv}...")
    df_metrics = extract_metrics_to_csv(output_json, output_csv)
    print(f"Metrics saved to {output_csv}")


if __name__ == "__main__":
    main()