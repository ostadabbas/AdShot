"""
Shot detection module using AutoShot model
"""

import os
import numpy as np
import torch
from utils_autoshot import get_frames, get_batches
import utils


class ShotDetector:
    """Detect shot boundaries in videos using AutoShot"""
    
    def __init__(self, model_path='./ckpt_0_200_0.pth', device=None, threshold=0.5,
                 window_batch_size=64):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.threshold = threshold  # ← ADD THIS LINE (you're missing it!)
        # How many 100-frame windows to push through the GPU per forward pass.
        self.window_batch_size = max(1, int(window_batch_size))
        print(f"ShotDetector initialized on {self.device}")
    
    def _load_model(self, model_path):
        """Load AutoShot model"""
        from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet
        
        model = TransNetV2Supernet().eval()
        
        if os.path.exists(model_path):
            print(f'Loading pretrained model from {model_path}')
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict['net'].items() if k in model_dict}
            print(f"Current model has {len(model_dict)} params, updating {len(pretrained_dict)} params")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            raise Exception(f"Error: Cannot find pretrained model at {model_path}")
        
        if self.device == "cuda":
            model = model.cuda(0)
        
        model.eval()
        return model
    
    def _predict_batch(self, batch):
        """Predict shot boundaries for a single 100-frame window (batch-1 reference path)."""
        batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
        batch = batch.to(self.device)

        with torch.no_grad():
            one_hot = self.model(batch)
            if isinstance(one_hot, tuple):
                one_hot = one_hot[0]
            return torch.sigmoid(one_hot[0])

    def _predict_windows(self, windows):
        """Predict for a stack of windows at once.

        Args:
            windows: np.ndarray (B, 100, 27, 48, 3) -- B temporal windows.
        Returns:
            np.ndarray (B, 100) of per-frame boundary probabilities.
        """
        # (B, 100, 27, 48, 3) -> (B, 3, 100, 27, 48)
        batch = torch.from_numpy(windows.transpose((0, 4, 1, 2, 3))).float()
        batch = batch.to(self.device)

        with torch.no_grad():
            one_hot = self.model(batch)
            if isinstance(one_hot, tuple):
                one_hot = one_hot[0]
            one_hot = torch.sigmoid(one_hot)
        return one_hot.detach().cpu().numpy()

    def _predictions_from_frames(self, frames):
        """Frame-level boundary predictions from already-decoded frames (GPU)."""
        windows = list(get_batches(frames))  # each (100, 27, 48, 3)

        predictions = []
        for i in range(0, len(windows), self.window_batch_size):
            chunk = np.stack(windows[i:i + self.window_batch_size], axis=0)  # (b,100,27,48,3)
            out = self._predict_windows(chunk)                              # (b, 100)
            for w in out:
                predictions.append(w[25:75])  # keep the middle 50 of each window

        predictions = np.concatenate(predictions, 0)[:len(frames)].reshape(-1)
        return predictions

    def _get_predictions(self, video_path):
        """Decode a video and get frame-level shot boundary predictions."""
        frames = get_frames(video_path)
        return frames, self._predictions_from_frames(frames)

    def _boundaries(self, predictions, n_frames, threshold=None, min_gap=6):
        """Turn per-frame predictions into shot-boundary frame indices."""
        if threshold is None:
            threshold = self.threshold  # Use instance default

        predictions_filtered = np.where(predictions < threshold, 0, predictions).round(3)

        if np.all(predictions_filtered == 0):
            # No shots detected, use entire video as one shot
            return [n_frames]

        shot_splits = [
            i[0] for i in utils.find_max_in_sequences(predictions_filtered, min_gap=min_gap)
            if i[0] != 0
        ]
        if not shot_splits:
            return [n_frames]
        # Add end frame if not already included
        if shot_splits[-1] != n_frames:
            shot_splits.append(n_frames)
        return shot_splits

    def detect_shots_from_frames(self, frames, threshold=None, min_gap=6):
        """Detect shots from pre-decoded frames (GPU-only half of detect_shots)."""
        predictions = self._predictions_from_frames(frames)
        shot_splits = self._boundaries(predictions, frames.shape[0], threshold, min_gap)
        return shot_splits, predictions

    def detect_shots(self, video_path, threshold=None, min_gap=6):
        """
        Detect shot boundaries in a video

        Args:
            video_path: Path to video file
            threshold: Confidence threshold for shot detection
            min_gap: Minimum frames between shot boundaries

        Returns:
            List of frame indices marking shot boundaries
        """
        video_frames, predictions = self._get_predictions(video_path)
        shot_splits = self._boundaries(predictions, video_frames.shape[0], threshold, min_gap)
        return shot_splits, predictions


# ---------------------------------------------------------------------------
# Batch-level shot-detection processor
# ---------------------------------------------------------------------------
# The unit of work is a BATCH (one videos CSV), not a brand. We iterate over
# every video row in the batch as a flat list of items and run shot detection
# on each one. Brand identity matters only at SAVE time: each result is appended
# to a per-brand JSONL file `{brand_id}_{brand_name}_shotsplits.json`, so the
# downstream stages keep the existing brand-keyed layout.

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


class ShotDetectionProcessor:
    """Run shot detection over a whole batch (CSV) of videos.

    Processing loop is over batch items (videos); outputs are grouped by brand.
    """

    def __init__(self, data_df, video_base_dir=None, output_dir=None,
                 model_path=None, threshold=None, window_batch_size=64):
        self.data_df = data_df
        self.video_base_dir = Path(video_base_dir or config.VIDEO_BASE_DIR)
        self.shots_dir = Path(output_dir or config.SHOTSPLITS_DIR)
        self.shots_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = model_path or config.AUTOSHOT_MODEL_PATH
        self.threshold = threshold if threshold is not None else config.SHOT_THRESHOLD
        # Per-forward GPU window batch; caps peak VRAM per inference call. With
        # several GPU workers sharing one card, lower this to fit.
        self.window_batch_size = window_batch_size
        # Loaded lazily: the parallel pipeline loads the model inside GPU workers,
        # so the parent process must not hold a (wasted) GPU model.
        self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            self._detector = ShotDetector(model_path=self.model_path, threshold=self.threshold,
                                          window_batch_size=self.window_batch_size)
        return self._detector

    def _resolve_video_path(self, row):
        """Locate a video file. The CSV `relative_path` column is authoritative
        (it records the actual on-disk folder, which can differ from the curated
        `brand_name`); fall back to the brand_name convention if it's absent."""
        rel = row.get('relative_path')
        if isinstance(rel, str) and rel:
            # relative_path is like "videomedia/<folder>/<ad_id>.mp4", rooted at
            # the parent of video_base_dir (.../videomedia).
            return self.video_base_dir.parent / rel
        return self.video_base_dir / row['brand_name'] / f"{row['ad_id']}.mp4"

    def _build_items(self):
        """Flatten the batch into a list of per-video work items."""
        items = []
        for _, row in self.data_df.iterrows():
            video_path = self._resolve_video_path(row)
            items.append({
                'ad_id': row['ad_id'],
                'ad_length': row.get('ad_length'),
                'brand_id': row['brand_id'],
                'brand_name': row['brand_name'],
                'video_path': video_path,
            })
        return items

    def _completed_ad_ids(self):
        """ad_ids already present in any existing shotsplits JSONL (for resume)."""
        done = set()
        for f in self.shots_dir.glob('*_shotsplits.json'):
            try:
                df = pd.read_json(f, lines=True)
                if 'ad_id' in df.columns:
                    done.update(df['ad_id'].astype(str).tolist())
            except (ValueError, KeyError):
                pass
        return done

    def _out_files(self, item):
        stem = f"{item['brand_id']}_{item['brand_name']}_shotsplits"
        return (self.shots_dir / f"{stem}.json", self.shots_dir / f"{stem}_fail.json")

    def _process_one(self, item):
        out_file, fail_file = self._out_files(item)
        record = {
            'ad_id': item['ad_id'],
            'ad_length': item['ad_length'],
            'brand_id': item['brand_id'],
            'brand_name': item['brand_name'],
        }
        try:
            shot_splits, _ = self.detector.detect_shots(str(item['video_path']))
            record['shotsplit'] = str(shot_splits)
            utils.write_json_row(str(out_file), record)
            return True
        except Exception as e:
            record.update({'error': str(e), 'error_type': type(e).__name__})
            utils.write_json_row(str(fail_file), record)
            logger.error(f"Error on {item['ad_id']}: {e}")
            return False

    def _partition(self, skip_existing=True):
        """Resolve the batch into the list of videos still to process."""
        items = self._build_items()
        done = self._completed_ad_ids() if skip_existing else set()

        missing = [it for it in items if not it['video_path'].exists()]
        pending = [it for it in items
                   if it['video_path'].exists() and str(it['ad_id']) not in done]

        logger.info(
            f"Batch: {len(items)} videos | {len(done)} already done | "
            f"{len(missing)} missing files | {len(pending)} to process"
        )
        for it in missing:
            logger.warning(f"Missing video file: {it['video_path']}")
        return pending

    def _record(self, item, extra):
        rec = {
            'ad_id': item['ad_id'],
            'ad_length': item['ad_length'],
            'brand_id': item['brand_id'],
            'brand_name': item['brand_name'],
        }
        rec.update(extra)
        return rec

    def run(self, skip_existing=True):
        """Single-process: decode + infer one video at a time (reference path)."""
        pending = self._partition(skip_existing)
        success = 0
        for it in tqdm(pending, desc="Shot detection (batch)"):
            if self._process_one(it):
                success += 1
        failed = len(pending) - success
        logger.info(f"Shot detection done: {success} ok, {failed} failed "
                    f"-> {self.shots_dir}")
        return success, failed

    def run_parallel(self, skip_existing=True, num_decode_workers=8,
                     num_gpu_workers=2, queue_size=None):
        """Two-stage pipeline: N CPU decode workers feed M GPU inference workers
        through a BOUNDED queue (backpressure caps RAM). All file writes happen
        here in the parent, so there is no cross-process write race."""
        import multiprocessing as mp
        import threading

        pending = self._partition(skip_existing)
        if not pending:
            logger.info("Nothing to process.")
            return 0, 0
        if num_decode_workers < 1 or num_gpu_workers < 1:
            raise ValueError("num_decode_workers and num_gpu_workers must be >= 1")

        ctx = mp.get_context('spawn')
        queue_size = queue_size or max(2, num_decode_workers * 2)
        task_q = ctx.Queue()
        decoded_q = ctx.Queue(maxsize=queue_size)   # bounded -> backpressure
        results_q = ctx.Queue()

        decode_procs = [ctx.Process(target=_decode_worker, args=(task_q, decoded_q),
                                    daemon=True)
                        for _ in range(num_decode_workers)]
        gpu_procs = [ctx.Process(target=_gpu_worker,
                                 args=(decoded_q, results_q, self.model_path,
                                       self.threshold, self.window_batch_size),
                                 daemon=True)
                     for _ in range(num_gpu_workers)]
        for p in decode_procs + gpu_procs:
            p.start()

        # Feed work, then one STOP per decode worker.
        for it in pending:
            task_q.put({
                'ad_id': it['ad_id'], 'ad_length': it['ad_length'],
                'brand_id': it['brand_id'], 'brand_name': it['brand_name'],
                'video_path': str(it['video_path']),
            })
        for _ in range(num_decode_workers):
            task_q.put(None)

        # Once all decoders are done, send one STOP per GPU worker.
        def _closer():
            for p in decode_procs:
                p.join()
            for _ in range(num_gpu_workers):
                decoded_q.put(None)
        closer = threading.Thread(target=_closer, daemon=True)
        closer.start()

        logger.info(f"Pipeline: {num_decode_workers} decode workers -> "
                    f"queue(maxsize={queue_size}) -> {num_gpu_workers} GPU workers")

        success = failed = finished_gpus = 0
        pbar = tqdm(total=len(pending), desc="Shot detection (pipeline)")
        while finished_gpus < num_gpu_workers:
            res = results_q.get()
            if res.get('gpu_done'):
                finished_gpus += 1
                continue
            item = res['meta']
            out_file, fail_file = self._out_files(item)
            if 'error' in res:
                utils.write_json_row(str(fail_file), self._record(
                    item, {'error': res['error'], 'error_type': res.get('error_type')}))
                logger.error(f"Error on {item['ad_id']}: {res['error']}")
                failed += 1
            else:
                utils.write_json_row(str(out_file), self._record(
                    item, {'shotsplit': res['shotsplit']}))
                success += 1
            pbar.update(1)
        pbar.close()
        closer.join()
        for p in gpu_procs:
            p.join()

        logger.info(f"Shot detection done: {success} ok, {failed} failed "
                    f"-> {self.shots_dir}")
        return success, failed


def _decode_worker(task_q, decoded_q):
    """CPU stage: decode a video's frames and push them onto the bounded queue.
    Blocks on a full queue (backpressure), so RAM stays bounded."""
    from utils_autoshot import get_frames
    while True:
        item = task_q.get()
        if item is None:
            break
        try:
            # threads=1: one core per ffmpeg, so N workers scale linearly instead
            # of N ffmpegs each grabbing all cores and thrashing.
            frames = get_frames(item['video_path'], threads=1)
            decoded_q.put({'meta': item, 'frames': frames})
        except Exception as e:
            decoded_q.put({'meta': item, 'error': str(e), 'error_type': type(e).__name__})


def _gpu_worker(decoded_q, results_q, model_path, threshold, window_batch_size=64):
    """GPU stage: load the model ONCE, then infer on pre-decoded frames."""
    # Reduce fragmentation when several GPU workers share one card.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    detector = ShotDetector(model_path=model_path, threshold=threshold,
                            window_batch_size=window_batch_size)
    while True:
        item = decoded_q.get()
        if item is None:                      # STOP
            results_q.put({'gpu_done': True})
            break
        meta = item['meta']
        if 'error' in item:                   # decode failed upstream
            results_q.put({'meta': meta, 'error': item['error'],
                           'error_type': item.get('error_type')})
            continue
        try:
            shot_splits, _ = detector.detect_shots_from_frames(item['frames'])
            results_q.put({'meta': meta, 'shotsplit': str(shot_splits)})
        except Exception as e:
            results_q.put({'meta': meta, 'error': str(e), 'error_type': type(e).__name__})


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Shot detection processor (batch CSV is the unit of work)")
    parser.add_argument('--data_csv', required=True,
                        help="Path to the batch videos CSV")
    parser.add_argument('--video_dir', default=config.VIDEO_BASE_DIR)
    parser.add_argument('--output_dir', default=config.SHOTSPLITS_DIR,
                        help="Where to write per-brand shotsplits JSONL files")
    parser.add_argument('--model_path', default=config.AUTOSHOT_MODEL_PATH)
    parser.add_argument('--threshold', type=float, default=config.SHOT_THRESHOLD)
    parser.add_argument('--no_skip_existing', action='store_true',
                        help="Reprocess videos even if already in output")
    parser.add_argument('--num_decode_workers', type=int, default=24,
                        help="CPU processes decoding video frames (producers, threads=1 each)")
    parser.add_argument('--num_gpu_workers', type=int, default=1,
                        help="GPU processes running inference (consumers); 1 is plenty")
    parser.add_argument('--queue_size', type=int, default=None,
                        help="Bounded decoded-frame queue size (default: 2x decode workers)")
    parser.add_argument('--window_batch_size', type=int, default=64,
                        help="GPU windows per forward; lower to cut peak VRAM per worker")
    parser.add_argument('--sequential', action='store_true',
                        help="Single-process reference path (no CPU/GPU pipeline)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_df = pd.read_csv(args.data_csv)
    logger.info(f"Loaded {len(data_df)} videos from "
                f"{data_df['brand_name'].nunique()} brands ({args.data_csv})")

    processor = ShotDetectionProcessor(
        data_df=data_df,
        video_base_dir=args.video_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        window_batch_size=args.window_batch_size,
    )

    skip_existing = not args.no_skip_existing
    if args.sequential:
        processor.run(skip_existing=skip_existing)
    else:
        processor.run_parallel(
            skip_existing=skip_existing,
            num_decode_workers=args.num_decode_workers,
            num_gpu_workers=args.num_gpu_workers,
            queue_size=args.queue_size,
        )


if __name__ == '__main__':
    main()