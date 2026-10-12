"""
Orchestrator for the pair-extraction pipeline.

Runs the stages end-to-end or individually:
  processing  -> Stage 1 (shots -> keyframes -> embeddings)  [s1_1_process_brands_main]
  clustering  -> Stage 2 (DBSCAN shot clustering)            [s2_clustering]
  pairing     -> Stage 3 (shot-overlap pairing -> fullmatch) [s3_pairing]
  all         -> processing -> clustering -> pairing

Defaults come from config.yaml (via config.py). Based on the old
`pair_extraction/s0_full_pipeline.py`, extended with the pairing stage and with
the threshold default fixed to track config (was hardcoded 0.5).

Examples:
  python run_pipeline.py --stage all --brand_id 975
  python run_pipeline.py --stage clustering --brand_id 975
  python run_pipeline.py --stage pairing
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("run_pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_stage1_processing(args):
    """Stage 1: shot detection, keyframe extraction, embeddings."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Video Processing (Shots → Keyframes → Embeddings)")
    logger.info("=" * 60)

    import s1_1_process_brands_main as stage1

    data_df = pd.read_csv(args.data_csv)
    logger.info(f"Loaded {len(data_df)} videos from {data_df['brand_name'].nunique()} brands")

    num_workers = args.num_workers
    if num_workers is None:
        import torch
        from multiprocessing import cpu_count
        if torch.cuda.is_available():
            num_workers = 1
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}; using 1 worker")
        else:
            num_workers = max(1, cpu_count() // 4)
            logger.info(f"⚠️  No GPU; using {num_workers} CPU workers")

    processor = stage1.BrandVideoProcessor(
        data_df=data_df,
        video_base_dir=args.video_dir,
        output_base_dir=args.output_dir,
        num_workers=num_workers,
    )

    brand_to_process = None
    if args.brand_id is not None:
        brand_row = data_df[data_df["brand_id"] == args.brand_id]
        if brand_row.empty:
            logger.error(f"Brand ID {args.brand_id} not found in data")
            return False
        brand_to_process = brand_row.iloc[0]["brand_name"]
        logger.info(f"Processing brand ID {args.brand_id}: {brand_to_process}")
    elif args.brand is not None:
        brand_to_process = args.brand

    if brand_to_process:
        return processor.process_single_brand(
            brand_to_process, skip_shots=args.skip_shots, skip_keyframes=args.skip_keyframes)
    processor.process_all_brands(skip_shots=args.skip_shots, skip_keyframes=args.skip_keyframes)
    return True


def run_stage2_clustering(args):
    """Stage 2: DBSCAN shot clustering."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Shot Clustering")
    logger.info("=" * 60)

    import s2_clustering as stage2

    processor = stage2.BrandClusteringProcessor(
        embeddings_dir=Path(args.output_dir) / "embeddings",
        keyframes_dir=Path(args.output_dir) / "keyframes",
        output_dir=Path(args.output_dir) / "clusters",
    )

    if args.brand_id is not None or args.brand is not None:
        available = processor.get_available_brands()
        match = None
        for brand_id, brand_name in available:
            if (args.brand_id is not None and brand_id == args.brand_id) or \
               (args.brand is not None and brand_name == args.brand):
                match = (brand_id, brand_name)
                break
        if not match:
            logger.error(f"Brand not found in embeddings: {args.brand or args.brand_id}")
            return False
        return processor.process_brand_clustering(match[0], match[1]) is not None
    processor.process_all_brands()
    return True


def run_stage3_pairing(args):
    """Stage 3: shot-overlap pairing -> fullmatch.csv."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Pairing")
    logger.info("=" * 60)

    import s3_pairing

    s3_pairing.run_pairing(
        shotsplits_dir=str(Path(args.output_dir) / "shotsplits"),
        clusters_dir=str(Path(args.output_dir) / "clusters"),
        data_csvs=[args.data_csv] if args.data_csv else None,
        out_path=args.fullmatch_out,
    )
    return True


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pair-extraction pipeline: shots -> keyframes -> embeddings -> clustering -> pairing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_csv", type=str, default=config.DATA_CSV,
                       help="Ad metadata CSV (default: config default batch)")
    parser.add_argument("--video_dir", type=str, default=config.VIDEO_BASE_DIR)
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_BASE_DIR)
    parser.add_argument("--fullmatch_out", type=str, default=config.FULLMATCH_OUT)

    parser.add_argument("--brand", type=str, default=None,
                       help='Process specific brand by name (quote spaces)')
    parser.add_argument("--brand_id", type=int, default=None,
                       help="Process specific brand by ID")

    parser.add_argument("--stage", type=str, default="all",
                       choices=["all", "processing", "clustering", "pairing"],
                       help="Which stage to run (default: all)")

    parser.add_argument("--num_workers", type=int, default=None,
                       help="CPU workers for Stage 1 (default: auto-detect)")
    parser.add_argument("--skip_shots", action="store_true")
    parser.add_argument("--skip_keyframes", action="store_true")
    parser.add_argument("--threshold", type=float, default=config.SHOT_THRESHOLD,
                       help=f"Shot detection threshold (default: config = {config.SHOT_THRESHOLD})")
    return parser


def main():
    args = build_parser().parse_args()

    logger.info("=" * 60)
    logger.info("PAIR-EXTRACTION PIPELINE")
    logger.info(f"stage={args.stage}  output_dir={args.output_dir}")
    logger.info("=" * 60)

    success = True
    if args.stage in ("all", "processing"):
        success = run_stage1_processing(args)
        if not success:
            logger.error("Stage 1 processing failed")
            return

    if args.stage in ("all", "clustering"):
        success = run_stage2_clustering(args)
        if not success:
            logger.error("Stage 2 clustering failed")
            return

    if args.stage in ("all", "pairing"):
        success = run_stage3_pairing(args)
        if not success:
            logger.error("Stage 3 pairing failed")
            return

    logger.info("\n" + "=" * 60)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
