import pandas as pd
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from utils import FFMPEG_PATH, extract_shots_from_mapping, create_one_hot, calculate_metrics
from aria_infer_video_input import load_aria, inference_aria
from minicpm_infer_video_input import load_minicpm, inference_minicpm
from molmo2_infer_video_input import load_molmo2, inference_molmo2

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

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
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

    ap.add_argument("--model", default="rhymes-ai/Aria")
    ap.add_argument("--csv", required=False, default="eval.csv", help="Path to eval.csv")
    ap.add_argument("--output_json", default="results.json", help="Path to output JSON file")
    ap.add_argument("--output_csv", default="metrics.csv", help="Path to output metrics CSV file")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--ffmpeg", default=FFMPEG_PATH, help="Path to ffmpeg binary")
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
    
    if "Aria" in args.model:
        print(f"Loading Aria model: {args.model}")
        model, processor = load_aria(args.model, attn_impl=args.attn_impl)
        inference_fn = inference_aria
    elif "MiniCPM" in args.model:
        print(f"Loading MiniCPM model: {args.model}")
        model, processor = load_minicpm(args.model, attn_impl=args.attn_impl)
        inference_fn = inference_minicpm
    elif "Molmo2" in args.model:
        print(f"Loading Molmo2 model: {args.model}")
        model, processor = load_molmo2(args.model, attn_impl=args.attn_impl)
        inference_fn = inference_molmo2


    for row_idx in tqdm(range(len(df)), desc="Processing rows"):
        # Skip if already processed
        if row_idx in results_dict:
            tqdm.write(f"Row {row_idx} already processed, skipping...")
            continue
        
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
            selected_shots_gt, all_shots = extract_shots_from_mapping(video_info['mapping'])
            one_hot_gt = create_one_hot(all_shots, selected_shots_gt)
            out, ok, total, selected_shots_pred = inference_fn(
                model=model,
                processor=processor,
                video_path=video_path,
                video_info=video_info,
                fps=args.fps,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

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
        except Exception as e:
            tqdm.write(f"Error processing row {row_idx}: {str(e)}")
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