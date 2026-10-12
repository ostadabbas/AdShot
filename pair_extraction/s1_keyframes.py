"""
Keyframe extraction module
"""

import os
import numpy as np
from PIL import Image
import ffmpeg
from collections import defaultdict


class KeyframeExtractor:
    """Extract keyframes from video shots"""
    
    def __init__(self, width=320, height=180, percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
        self.width = width
        self.height = height
        self.percentiles = percentiles
    
    def get_frames(self, video_path):
        """Load all frames from video as RGB numpy array"""
        video_stream, err = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24',
                   s=f'{self.width}x{self.height}')
            # -nostdin: keep ffmpeg off the terminal (no echo-disable). Pixel output
            # is unchanged, so this is still byte-identical to the reference decode.
            .global_args('-nostdin')
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(video_stream, np.uint8).reshape(
            [-1, self.height, self.width, 3]
        )
        return video

    def _decode_indices(self, video_path, indices):
        """Return the requested frame indices, byte-identical to the reference.

        Decodes the WHOLE video exactly as get_frames() -- the mechanism that
        produced the reference keyframes (data/0.1) -- and indexes into it. The
        previous `select`+`vsync=0` shortcut changed the frame->index mapping on
        variable-frame-rate videos (passthrough vs the default-vsync decode the
        shotsplits were computed against), so its pixels did NOT match the
        reference. Indexing into the full decode is the only thing guaranteed to
        reproduce frames[target_idx] bit-for-bit.

        Out-of-range indices (idx >= decoded frame count) are dropped, matching
        extract_keyframes' original `if target_idx >= total_frames: continue`.

        Returns: dict {frame_index: np.ndarray(H, W, 3)}.
        """
        uniq = sorted({int(i) for i in indices})
        if not uniq:
            return {}
        frames = self.get_frames(video_path)
        n = frames.shape[0]
        return {idx: frames[idx] for idx in uniq if idx < n}

    def extract_keyframes(self, video_path, shot_splits, video_name, output_dir,
                         existing_shots=None):
        """
        Extract keyframes from video shots

        Decodes only the frames it actually saves (via _decode_indices) instead
        of decoding the whole video, while producing byte-identical JPEGs.

        Args:
            video_path: Path to video file
            shot_splits: List of frame indices marking shot boundaries
            video_name: Name/ID for the video
            output_dir: Directory to save keyframes
            existing_shots: Set of shot_ids to skip (for resuming)
        """
        if existing_shots is None:
            existing_shots = set()

        os.makedirs(output_dir, exist_ok=True)

        # Ensure shot_splits starts with 0
        if shot_splits[0] != 0:
            shot_splits = [0] + shot_splits

        # total_frames is the count shot detection recorded as the final split
        # (detect_shots sets shot_splits[-1] = video_frames.shape[0] from the
        # same ffmpeg decode the old get_frames() used). Using it reproduces the
        # old clamp exactly without decoding the whole video.
        total_frames = shot_splits[-1]

        # Plan every (frame_index, output_path) up front -- no decoding yet.
        plan = []  # list of (frame_index, output_path)
        for i in range(len(shot_splits) - 1):
            start = shot_splits[i]
            end = min(shot_splits[i + 1], total_frames)
            shot_len = end - start

            # Generate shot_id
            shot_id = f"{video_name}_shot_{i+1:02d}"

            # Skip if already exists
            if shot_id in existing_shots:
                continue

            # Short shot: Save all frames (< 5 frames)
            if shot_len < 5:
                for idx in range(start, end):
                    plan.append((idx, f"{output_dir}/{shot_id}_frame_{idx}.jpg"))

            # Long shot: Sample at percentiles
            else:
                for p in self.percentiles:
                    target_idx = start + int(p * shot_len)
                    if target_idx >= total_frames:
                        continue
                    plan.append(
                        (target_idx, f"{output_dir}/{shot_id}_p{int(p*100)}.jpg")
                    )

        if not plan:
            return

        # Decode only the needed frames, then save.
        frames_by_idx = self._decode_indices(video_path, (idx for idx, _ in plan))
        for idx, out_path in plan:
            rgb_frame = frames_by_idx.get(idx)
            if rgb_frame is None:
                continue
            Image.fromarray(rgb_frame).save(out_path, quality=95, subsampling=0)

    def find_valid_shots(self, keyframes_dir):
        """
        Find all valid shots (with exactly 5 keyframes) in a directory
        
        Returns:
            List of tuples: [(shot_id, [frame_paths])]
        """
        shot_files = defaultdict(list)
        
        for fname in os.listdir(keyframes_dir):
            if not fname.lower().endswith('.jpg'):
                continue
            
            try:
                # Parse filename: ad_id_shot_XX_pYY.jpg
                parts = fname.replace('.jpg', '').split('_')
                
                # Find shot number (after 'shot')
                shot_idx = None
                for i, part in enumerate(parts):
                    if part == 'shot' and i + 1 < len(parts):
                        shot_idx = i + 1
                        break
                
                if shot_idx is None:
                    continue
                
                # Reconstruct shot_id (everything except last pXX part)
                shot_id = '_'.join(parts[:-1])
                
                # Get percentile (pXX)
                percentile_part = parts[-1]
                if not percentile_part.startswith('p'):
                    continue
                
                frame_num = int(percentile_part[1:])
                full_path = os.path.join(keyframes_dir, fname)
                shot_files[shot_id].append((frame_num, full_path))
            
            except Exception as e:
                print(f"Error parsing filename {fname}: {e}")
                continue
        
        # Filter for valid shots (exactly 5 frames at correct percentiles)
        valid_shots = []
        for shot_id, frames in shot_files.items():
            frame_nums = {fn for fn, _ in frames}
            if frame_nums == {10, 25, 50, 75, 90}:
                # Sort frames by percentile
                sorted_frames = [fp for _, fp in sorted(frames, key=lambda x: x[0])]
                valid_shots.append((shot_id, sorted_frames))

        return valid_shots


# ---------------------------------------------------------------------------
# Batch-level keyframe processor
# ---------------------------------------------------------------------------
# Unit of work is a BATCH (one videos CSV). We flatten the batch into a flat
# list of per-video items, look up each video's shot boundaries from the
# shotsplits stage output, and extract keyframes. Brand identity matters only
# at SAVE time: keyframes for a video go to `{keyframes_dir}/{brand_id}/`, the
# layout the embedding stage expects.

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
import utils

logger = logging.getLogger(__name__)


def _keyframe_worker(item):
    """Module-level (picklable) worker: extract keyframes for one video."""
    extractor = KeyframeExtractor()
    try:
        extractor.extract_keyframes(
            item['video_path'], item['shot_splits'], item['ad_id'],
            item['output_dir'], set(),
        )
        return (True, item['ad_id'])
    except Exception as e:
        return (False, f"{item['ad_id']}: {e}")


class KeyframeProcessor:
    """Extract keyframes over a whole batch (CSV), keyed to the shotsplits output."""

    def __init__(self, data_df, shotsplits_dir=None, video_base_dir=None,
                 output_dir=None, num_workers=None):
        self.data_df = data_df
        self.shotsplits_dir = Path(shotsplits_dir or config.SHOTSPLITS_DIR)
        self.video_base_dir = Path(video_base_dir or config.VIDEO_BASE_DIR)
        self.keyframes_dir = Path(output_dir or config.KEYFRAMES_DIR)
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)

        if num_workers is None:
            from multiprocessing import cpu_count
            num_workers = min(8, max(1, cpu_count() // 2))
        self.num_workers = num_workers

    def _load_shotsplits(self):
        """Map ad_id -> parsed shot boundaries, from per-brand shotsplits JSONL."""
        mapping = {}
        for f in self.shotsplits_dir.glob('*_shotsplits.json'):
            try:
                df = pd.read_json(f, lines=True)
            except ValueError:
                continue
            if 'shotsplit' not in df.columns:
                continue
            df = df.drop_duplicates(subset='ad_id', keep='first')
            for _, row in df.iterrows():
                mapping[str(row['ad_id'])] = utils.parse_shot_splits(row['shotsplit'])
        return mapping

    def _resolve_video_path(self, row):
        """Authoritative `relative_path`, falling back to brand_name convention."""
        rel = row.get('relative_path')
        if isinstance(rel, str) and rel:
            return self.video_base_dir.parent / rel
        return self.video_base_dir / row['brand_name'] / f"{row['ad_id']}.mp4"

    @staticmethod
    def _already_done(brand_dir, ad_id):
        return brand_dir.exists() and any(brand_dir.glob(f"{ad_id}_shot_*"))

    def _build_items(self, skip_existing=True):
        shot_map = self._load_shotsplits()
        items = []
        missing_shots = missing_video = done = 0
        for _, row in self.data_df.iterrows():
            ad_id = str(row['ad_id'])
            shots = shot_map.get(ad_id)
            if shots is None:
                missing_shots += 1
                continue
            video_path = self._resolve_video_path(row)
            if not video_path.exists():
                missing_video += 1
                logger.warning(f"Missing video file: {video_path}")
                continue
            brand_dir = self.keyframes_dir / str(row['brand_id'])
            if skip_existing and self._already_done(brand_dir, ad_id):
                done += 1
                continue
            items.append({
                'ad_id': ad_id,
                'shot_splits': shots,
                'video_path': str(video_path),
                'output_dir': str(brand_dir),
            })

        logger.info(
            f"Batch: {len(self.data_df)} videos | {done} already done | "
            f"{missing_shots} without shotsplits | {missing_video} missing files | "
            f"{len(items)} to process"
        )
        return items

    def run(self, skip_existing=True):
        items = self._build_items(skip_existing=skip_existing)
        for it in items:
            Path(it['output_dir']).mkdir(parents=True, exist_ok=True)
        if not items:
            logger.info("Nothing to process.")
            return 0, 0

        success, failures = 0, []
        if self.num_workers > 1 and len(items) > 1:
            from multiprocessing import Pool
            logger.info(f"Extracting keyframes with {self.num_workers} workers")
            with Pool(self.num_workers) as pool:
                for ok, info in tqdm(pool.imap_unordered(_keyframe_worker, items),
                                     total=len(items), desc="Keyframes (batch)"):
                    if ok:
                        success += 1
                    else:
                        failures.append(info)
        else:
            for it in tqdm(items, desc="Keyframes (batch)"):
                ok, info = _keyframe_worker(it)
                if ok:
                    success += 1
                else:
                    failures.append(info)

        for info in failures:
            logger.error(f"Keyframe fail: {info}")
        logger.info(f"Keyframe extraction done: {success} ok, {len(failures)} failed "
                    f"-> {self.keyframes_dir}")
        return success, len(failures)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Keyframe extraction processor (batch CSV is the unit of work)")
    parser.add_argument('--data_csv', required=True,
                        help="Path to the batch videos CSV")
    parser.add_argument('--shotsplits_dir', default=config.SHOTSPLITS_DIR,
                        help="Dir with per-brand *_shotsplits.json from the shot stage")
    parser.add_argument('--video_dir', default=config.VIDEO_BASE_DIR)
    parser.add_argument('--output_dir', default=config.KEYFRAMES_DIR,
                        help="Keyframes root; per-brand subdirs {brand_id}/ are created")
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--no_skip_existing', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_df = pd.read_csv(args.data_csv)
    logger.info(f"Loaded {len(data_df)} videos from "
                f"{data_df['brand_name'].nunique()} brands ({args.data_csv})")

    processor = KeyframeProcessor(
        data_df=data_df,
        shotsplits_dir=args.shotsplits_dir,
        video_base_dir=args.video_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
    processor.run(skip_existing=not args.no_skip_existing)


if __name__ == '__main__':
    main()