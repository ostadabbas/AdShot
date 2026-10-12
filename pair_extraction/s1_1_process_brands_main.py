"""
Main script to process video brands: shot detection -> keyframe extraction -> embeddings
Processes each brand independently and saves brand-specific embedding files.
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

# Import modules
from s1_shot_detection import ShotDetector
from s1_keyframes import KeyframeExtractor
from s1_embedding import EmbeddingGenerator
from s1_processing_status import ProcessingStatusTracker
import utils
import config  # central config.yaml loader (was: import s1_config as config)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brand_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Module-level worker function (must be picklable for multiprocessing)
def extract_keyframes_worker_func(work_item):
    """
    Static worker function for parallel keyframe extraction
    Must be at module level to be picklable
    """
    try:
        from s1_keyframes import KeyframeExtractor
        extractor = KeyframeExtractor()
        
        extractor.extract_keyframes(
            work_item['video_path'],
            work_item['shot_splits'],
            work_item['ad_id'],
            work_item['output_dir'],
            set()
        )
        return (True, work_item['ad_id'])
    
    except Exception as e:
        print(f"Error in worker for {work_item['ad_id']}: {str(e)}")
        return (False, work_item['ad_id'])


class BrandVideoProcessor:
    """Process videos for each brand: shots -> keyframes -> embeddings"""
    
    def __init__(self, data_df, video_base_dir, output_base_dir, num_workers=1):
        self.data_df = data_df
        self.video_base_dir = Path(video_base_dir)
        self.output_base_dir = Path(output_base_dir)
        self.num_workers = num_workers
        
        # Create output directories
        self.shots_dir = self.output_base_dir / 'shotsplits'
        self.keyframes_dir = self.output_base_dir / 'keyframes'
        self.embeddings_dir = self.output_base_dir / 'embeddings'
        
        for d in [self.shots_dir, self.keyframes_dir, self.embeddings_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        #self.shot_detector = ShotDetector()
        # Initialize with config values
        self.shot_detector = ShotDetector(
            model_path=config.AUTOSHOT_MODEL_PATH,
            threshold=config.SHOT_THRESHOLD
        )
        self.keyframe_extractor = KeyframeExtractor()
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize status tracker
        self.status_tracker = ProcessingStatusTracker(
            status_file=self.output_base_dir / 'processing_status.json'
        )
        
        logger.info(f"Initialized BrandVideoProcessor with {num_workers} workers")
    
    def get_available_brands(self):
        """Get list of brands that have video folders"""
        available_brands = []
        for brand_name in self.data_df['brand_name'].unique():
            brand_folder = self.video_base_dir / brand_name
            if brand_folder.exists() and brand_folder.is_dir():
                video_files = list(brand_folder.glob('*.mp4'))
                if video_files:
                    available_brands.append(brand_name)
                    logger.info(f"Found brand: {brand_name} with {len(video_files)} videos")
        
        return available_brands
    
    def process_brand_shots(self, brand_name, brand_id, num_workers=1):
        """Step 1: Detect shots for all videos of a brand"""
        logger.info(f"Processing shots for brand: {brand_name} (ID: {brand_id})")
        
        brand_df = self.data_df[self.data_df['brand_name'] == brand_name]
        all_ad_ids = brand_df['ad_id'].tolist()
        
        # Check status tracker instead of file
        pending_ad_ids = self.status_tracker.get_pending_shots(brand_id, all_ad_ids)
        
        if not pending_ad_ids:
            logger.info(f"All {len(all_ad_ids)} videos already processed for {brand_name}")
            return self.shots_dir / f"{brand_id}_{brand_name}_shotsplits.json"
        
        logger.info(f"Found {len(all_ad_ids) - len(pending_ad_ids)} completed, processing {len(pending_ad_ids)} videos")
        
        # Build videos to process list
        videos_to_process = []
        for _, row in brand_df.iterrows():
            if row['ad_id'] in pending_ad_ids:
                video_path = self.video_base_dir / brand_name / f"{row['ad_id']}.mp4"
                if video_path.exists():
                    videos_to_process.append({
                        'ad_id': row['ad_id'],
                        'ad_length': row['ad_length'],
                        'brand_id': brand_id,
                        'brand_name': brand_name,
                        'video_path': str(video_path)
                    })
        
        if not videos_to_process:
            logger.info(f"No new videos to process for {brand_name}")
            return self.shots_dir / f"{brand_id}_{brand_name}_shotsplits.json"
        
        output_file = self.shots_dir / f"{brand_id}_{brand_name}_shotsplits.json"
        fail_file = self.shots_dir / f"{brand_id}_{brand_name}_shotsplits_fail.json"
        
        # Process videos
        success_count = 0
        completed_ad_ids = []
        
        for video_info in tqdm(videos_to_process, desc=f"Shot detection - {brand_name}"):
            success = self._process_single_video_shots(video_info, output_file, fail_file)
            if success:
                success_count += 1
                completed_ad_ids.append(video_info['ad_id'])
        
        # Update status tracker
        self.status_tracker.mark_shots_complete(brand_id, completed_ad_ids)
        
        logger.info(f"Brand {brand_name}: {success_count} successful, {len(videos_to_process) - success_count} failed")
        return output_file
    
    def _process_single_video_shots(self, video_info, output_file, fail_file):
        """Process a single video for shot detection"""
        results = {
            'ad_id': video_info['ad_id'],
            'ad_length': video_info['ad_length'],
            'brand_id': video_info['brand_id'],
            'brand_name': video_info['brand_name']
        }
        
        try:
            shot_splits, predictions = self.shot_detector.detect_shots(video_info['video_path'])
            results['shotsplit'] = str(shot_splits)
            results['predictions'] = str(predictions)
            utils.write_json_row(str(output_file), results)
            return True
        
        except Exception as e:
            results.update({
                'error': str(e),
                'error_type': type(e).__name__
            })
            utils.write_json_row(str(fail_file), results)
            logger.error(f"Error processing {video_info['ad_id']}: {str(e)}")
            return False
    
    def process_brand_keyframes(self, brand_name, brand_id, shots_file):
        """Step 2: Extract keyframes for all shots of a brand"""
        logger.info(f"Extracting keyframes for brand: {brand_name}")
        
        shots_df = pd.read_json(shots_file, lines=True)
        shots_df['shotsplit'] = shots_df['shotsplit'].apply(utils.parse_shot_splits)
        shots_df = shots_df.drop_duplicates(subset='ad_id', keep='first').reset_index(drop=True)
        
        all_ad_ids = shots_df['ad_id'].tolist()
        
        # Check status tracker
        pending_ad_ids = self.status_tracker.get_pending_keyframes(brand_id, all_ad_ids)
        
        if not pending_ad_ids:
            logger.info(f"All {len(all_ad_ids)} videos already processed for {brand_name}")
            return self.keyframes_dir / str(brand_id)
        
        logger.info(f"Found {len(all_ad_ids) - len(pending_ad_ids)} completed, processing {len(pending_ad_ids)} videos")
        
        output_dir = self.keyframes_dir / str(brand_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter to pending videos
        videos_to_process = shots_df[shots_df['ad_id'].isin(pending_ad_ids)]
        
        # Determine workers for keyframe extraction
        from multiprocessing import cpu_count
        import torch
        
        if torch.cuda.is_available():
            # If GPU available, use more CPUs for keyframes (GPU not needed for this step)
            keyframe_workers = min(8, cpu_count() // 2)
        else:
            # If no GPU, use 25% of CPUs for keyframes
            keyframe_workers = max(1, cpu_count() // 4)
        
        logger.info(f"Using {keyframe_workers} workers for keyframe extraction")
        
        # Parallel or sequential processing
        if keyframe_workers > 1 and len(videos_to_process) > 1:
            success_count, completed_ad_ids = self._process_keyframes_parallel(
                videos_to_process, brand_name, output_dir, keyframe_workers
            )
        else:
            success_count, completed_ad_ids = self._process_keyframes_sequential(
                videos_to_process, brand_name, output_dir
            )
        
        # Update status tracker
        self.status_tracker.mark_keyframes_complete(brand_id, completed_ad_ids)
        
        logger.info(f"Extracted keyframes for {success_count} videos in {brand_name}")
        return output_dir
    
    def _process_keyframes_sequential(self, videos_df, brand_name, output_dir):
        """Process keyframes sequentially (single worker)"""
        success_count = 0
        completed_ad_ids = []
        
        for _, row in tqdm(videos_df.iterrows(), total=len(videos_df),
                          desc=f"Keyframe extraction - {brand_name}"):
            ad_id = row['ad_id']
            video_path = self.video_base_dir / brand_name / f"{ad_id}.mp4"
            
            if not video_path.exists():
                continue
            
            try:
                self.keyframe_extractor.extract_keyframes(
                    str(video_path),
                    row['shotsplit'],
                    ad_id,
                    str(output_dir),
                    set()
                )
                success_count += 1
                completed_ad_ids.append(ad_id)
            
            except Exception as e:
                logger.error(f"Error extracting keyframes for {ad_id}: {str(e)}")
        
        return success_count, completed_ad_ids
    
    def _process_keyframes_parallel(self, videos_df, brand_name, output_dir, num_workers):
        """Process keyframes in parallel (multiple workers)"""
        from multiprocessing import Pool
        
        # Prepare work items
        work_items = []
        for _, row in videos_df.iterrows():
            ad_id = row['ad_id']
            video_path = self.video_base_dir / brand_name / f"{ad_id}.mp4"
            
            if video_path.exists():
                work_items.append({
                    'video_path': str(video_path),
                    'shot_splits': row['shotsplit'],
                    'ad_id': ad_id,
                    'output_dir': str(output_dir)
                })
        
        # Process in parallel using module-level function
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(extract_keyframes_worker_func, work_items),
                total=len(work_items),
                desc=f"Keyframe extraction - {brand_name}"
            ))
        
        # Collect results
        success_count = sum(1 for success, _ in results if success)
        completed_ad_ids = [ad_id for success, ad_id in results if success]
        
        return success_count, completed_ad_ids
    
    def process_brand_embeddings(self, brand_name, brand_id, keyframes_dir):
        """Step 3: Generate embeddings for all shots of a brand"""
        logger.info(f"Generating embeddings for brand: {brand_name}")
        
        # Check if already complete
        if self.status_tracker.is_embeddings_complete(brand_id):
            logger.info(f"Embeddings already complete for {brand_name}")
            return self.embeddings_dir / f"{brand_id}_{brand_name}_embeddings.h5"
        
        output_file = self.embeddings_dir / f"{brand_id}_{brand_name}_embeddings.h5"
        
        shot_info = self.keyframe_extractor.find_valid_shots(str(keyframes_dir))
        
        if not shot_info:
            logger.warning(f"No valid shots found for {brand_name}")
            return None
        
        logger.info(f"Found {len(shot_info)} valid shots for {brand_name}")
        
        self.embedding_generator.process_brand(
            shot_info,
            brand_id,
            str(output_file),
            resume=True
        )
        
        # Mark as complete
        self.status_tracker.mark_embeddings_complete(brand_id)
        
        logger.info(f"Saved embeddings to {output_file}")
        return output_file
    
    def process_single_brand(self, brand_name, skip_shots=False, skip_keyframes=False):
        """Process a single brand through all steps"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing brand: {brand_name}")
        logger.info(f"{'='*60}")
        
        brand_df = self.data_df[self.data_df['brand_name'] == brand_name]
        if brand_df.empty:
            logger.error(f"No data found for brand: {brand_name}")
            return False
        
        brand_id = brand_df.iloc[0]['brand_id']
        
        # Step 1: Shot detection
        if not skip_shots:
            shots_file = self.process_brand_shots(brand_name, brand_id, self.num_workers)
        else:
            shots_file = self.shots_dir / f"{brand_id}_{brand_name}_shotsplits.json"
            if not shots_file.exists():
                logger.error(f"Shot file not found: {shots_file}")
                return False
        
        # Step 2: Keyframe extraction
        if not skip_keyframes:
            keyframes_dir = self.process_brand_keyframes(brand_name, brand_id, shots_file)
        else:
            keyframes_dir = self.keyframes_dir / str(brand_id)
            if not keyframes_dir.exists():
                logger.error(f"Keyframes directory not found: {keyframes_dir}")
                return False
        
        # Step 3: Embedding generation
        embedding_file = self.process_brand_embeddings(brand_name, brand_id, keyframes_dir)
        
        if embedding_file:
            logger.info(f"✓ Successfully processed brand: {brand_name}")
            return True
        else:
            logger.error(f"✗ Failed to process brand: {brand_name}")
            return False
    
    def process_all_brands(self, skip_shots=False, skip_keyframes=False):
        """Process all available brands"""
        available_brands = self.get_available_brands()
        logger.info(f"Found {len(available_brands)} brands to process")
        
        results = {}
        for brand_name in available_brands:
            success = self.process_single_brand(brand_name, skip_shots, skip_keyframes)
            results[brand_name] = success
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Processing Summary:")
        logger.info(f"{'='*60}")
        successful = sum(results.values())
        logger.info(f"Successful: {successful}/{len(results)}")
        logger.info(f"Failed: {len(results) - successful}/{len(results)}")
        
        if successful < len(results):
            logger.info("\nFailed brands:")
            for brand, success in results.items():
                if not success:
                    logger.info(f"  - {brand}")


def main():
    parser = argparse.ArgumentParser(description='Process brand videos: shots -> keyframes -> embeddings')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to CSV file with video metadata')
    parser.add_argument('--video_dir', type=str, default='videomedia',
                       help='Base directory containing brand video folders')
    # parser.add_argument('--output_dir', type=str, default='data',
    #                    help='Base output directory')
    parser.add_argument('--brand', type=str, default=None,
                       help='Process specific brand by name (use quotes for spaces: "Discount Tire")')
    parser.add_argument('--brand_id', type=int, default=None,
                       help='Process specific brand by ID (alternative to --brand)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of CPU workers for parallel processing (default: auto-detect)')
    parser.add_argument('--skip_shots', action='store_true',
                       help='Skip shot detection (use existing)')
    parser.add_argument('--skip_keyframes', action='store_true',
                       help='Skip keyframe extraction (use existing)')
    # parser.add_argument('--threshold', type=float, default=0.5,
    #                help='Shot detection threshold (default: 0.5)')
    
    # Use config values
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_BASE_DIR,
                       help='Base output directory')
    parser.add_argument('--threshold', type=float, default=config.SHOT_THRESHOLD,
                       help='Shot detection threshold')

    
    args = parser.parse_args()
    
    # Auto-detect optimal number of workers
    if args.num_workers is None:
        import torch
        from multiprocessing import cpu_count
        
        # Check if GPU is available
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            # GPU available: use 1 worker (GPU processing is already fast)
            args.num_workers = 1
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info("Using 1 worker (GPU acceleration enabled)")
        else:
            # No GPU: use multiple CPU workers
            total_cpus = cpu_count()
            # Use 25% of available CPUs (good balance of speed vs memory)
            args.num_workers = max(1, total_cpus // 4)
            logger.info(f"⚠️  No GPU detected, using CPU-only mode")
            logger.info(f"Using {args.num_workers} workers out of {total_cpus} available CPUs")
            logger.info(f"💡 Tip: Install PyTorch with CUDA for 10-50x speedup!")
    else:
        logger.info(f"Using manually specified {args.num_workers} workers")
    
    # Load data
    logger.info(f"Loading data from {args.data_csv}")
    data_df = pd.read_csv(args.data_csv)
    logger.info(f"Loaded {len(data_df)} videos from {data_df['brand_name'].nunique()} brands")
    
    # Initialize processor
    processor = BrandVideoProcessor(
        data_df=data_df,
        video_base_dir=args.video_dir,
        output_base_dir=args.output_dir,
        num_workers=args.num_workers
    )
    
    # Determine which brand to process
    brand_to_process = None
    if args.brand_id is not None:
        # Find brand name from brand_id
        brand_row = data_df[data_df['brand_id'] == args.brand_id]
        if brand_row.empty:
            logger.error(f"Brand ID {args.brand_id} not found in data")
            return
        brand_to_process = brand_row.iloc[0]['brand_name']
        logger.info(f"Processing brand ID {args.brand_id}: {brand_to_process}")
    elif args.brand is not None:
        brand_to_process = args.brand
    
    # Process
    if brand_to_process:
        processor.process_single_brand(
            brand_to_process,
            skip_shots=args.skip_shots,
            skip_keyframes=args.skip_keyframes
        )
    else:
        processor.process_all_brands(
            skip_shots=args.skip_shots,
            skip_keyframes=args.skip_keyframes
        )


if __name__ == "__main__":
    main()