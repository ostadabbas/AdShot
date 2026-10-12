"""
Embedding generation module using Swin3D
"""

import os
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
import shutil


class Swin3DEmbedder:
    """Extract Swin3D embeddings from video clips"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        model = swin3d_b(weights=weights)
        
        # Remove classification head, keep features
        self.features = torch.nn.Sequential(*list(model.children())[:-2])
        self.avgpool = torch.nn.AdaptiveAvgPool3d(1)
        self.features.to(self.device)
        self.features.eval()
        
        print(f"Swin3DEmbedder initialized on {self.device}")
    
    def custom_transform(self, video_clip, resize_size=256, crop_size=224,
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Transform video clip for Swin3D input"""
        if isinstance(video_clip, np.ndarray):
            video_clip = torch.from_numpy(video_clip).float()
        
        assert video_clip.ndim == 4, "Input must be (T, C, H, W)"
        
        NT, C, H, W = video_clip.shape
        transformed_frames = []
        
        # Define transforms
        resize = T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR)
        crop = T.CenterCrop(crop_size)
        to_tensor = T.ToTensor()
        normalize = T.Normalize(mean=mean, std=std)
        
        # Process each frame
        for t in range(NT):
            frame = video_clip[t].permute(1, 2, 0).numpy()  # (H, W, C)
            frame = Image.fromarray(frame.astype(np.uint8))
            frame = resize(frame)
            frame = crop(frame)
            frame = to_tensor(frame)  # (C, H, W)
            frame = normalize(frame)
            transformed_frames.append(frame)
        
        # Stack and add batch dimension
        transformed_clip = torch.stack(transformed_frames)  # (T, C, H, W)
        transformed_clip = transformed_clip.permute(1, 0, 2, 3).unsqueeze(0)
        
        return transformed_clip
    
    def run(self, video_clip):
        """Forward pass through model"""
        with torch.no_grad():
            if video_clip.ndimension() != 5:
                raise ValueError("Input must be (batch, channels, temporal, height, width)")
            
            feat = self.features(video_clip)
            feat = feat.permute(0, 4, 1, 2, 3)  # B, C, _T, _H, _W
            feat = self.avgpool(feat)
            feat = feat.squeeze()
        
        return feat
    
    def get_embedding(self, frame_paths):
        """
        Extract embedding from 5 keyframes

        Args:
            frame_paths: List of 5 image file paths

        Returns:
            1024-dimensional embedding (numpy array)
        """
        # Load frames
        frames = [np.array(Image.open(fp)) for fp in frame_paths]

        # Convert to tensor (5, H, W, 3)
        clip_tensor = torch.tensor(np.array(frames))
        clip_tensor = clip_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)

        # Transform and move to device
        clip_tensor = self.custom_transform(clip_tensor).to(self.device)

        # Extract features
        features = self.run(clip_tensor)

        return features.squeeze().cpu().numpy()

    def _transform_clip(self, frame_paths):
        """Load + transform one shot's keyframes -> (C, T, H, W) tensor on CPU.

        Uses the exact same ops as custom_transform/get_embedding so a batched
        forward produces embeddings identical (up to float noise) to batch-1.
        """
        frames = [np.array(Image.open(fp)) for fp in frame_paths]
        clip = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)  # (T, C, H, W)
        # custom_transform returns (1, C, T, H, W); drop the leading batch dim.
        return self.custom_transform(clip).squeeze(0)

    def _forward_batch(self, batch_clip):
        """Forward a stacked batch (B, C, T, H, W) -> (B, 1024) on device."""
        with torch.no_grad():
            feat = self.features(batch_clip)
            feat = feat.permute(0, 4, 1, 2, 3)   # B, C, _T, _H, _W
            feat = self.avgpool(feat)
            feat = feat.flatten(1)               # (B, C) -- robust for any B
        return feat

    def get_embeddings_batch(self, list_of_frame_paths):
        """
        Extract embeddings for several shots in a single GPU forward pass.

        Args:
            list_of_frame_paths: list of per-shot keyframe path lists.

        Returns:
            numpy array (B, 1024) aligned with the input order.
        """
        clips = [self._transform_clip(fp) for fp in list_of_frame_paths]  # each (C, T, H, W)
        batch = torch.stack(clips).to(self.device)                        # (B, C, T, H, W)
        feats = self._forward_batch(batch)
        return feats.cpu().numpy()


class EmbeddingGenerator:
    """Generate and save embeddings for brand shots"""
    
    def __init__(self, batch_size=32):
        self.embedder = Swin3DEmbedder()
        self.batch_size = batch_size
    
    def process_brand(self, shot_info, brand_id, output_path, resume=True):
        """
        Process all shots for a brand and save embeddings to H5
        
        Args:
            shot_info: List of (shot_id, frame_paths) tuples
            brand_id: Brand ID
            output_path: Output H5 file path
            resume: Whether to resume from existing file
        """
        temp_path = output_path + '.temp'
        lock_path = output_path + '.lock'
        
        # Check for previous crash
        if os.path.exists(lock_path):
            print("Warning: Found lock file from previous run. Attempting recovery...")
            if os.path.exists(temp_path):
                os.remove(lock_path)
        
        # Create lock file
        open(lock_path, 'w').close()
        
        try:
            processed_shots = set()
            initial_size = 0
            
            # Resume from existing file
            if resume and os.path.exists(output_path):
                try:
                    with h5py.File(output_path, 'r') as hf:
                        if 'shot_ids' in hf:
                            processed_shots.update(
                                sid.decode('ascii') for sid in hf['shot_ids'][:]
                            )
                            initial_size = len(hf['shot_ids'])
                    print(f"Resuming from {initial_size} processed shots")
                    shutil.copyfile(output_path, temp_path)
                except Exception as e:
                    print(f"Error reading existing file: {e}. Starting fresh.")
                    resume = False
            
            # Create new file if needed
            if not resume or not os.path.exists(temp_path):
                with h5py.File(temp_path, 'w') as hf:
                    hf.create_dataset(
                        "embeddings", (0, 1024), maxshape=(None, 1024),
                        dtype='float32', chunks=(self.batch_size, 1024),
                        compression="gzip"
                    )
                    hf.create_dataset(
                        "brand_ids", (0,), maxshape=(None,), dtype='int32',
                        chunks=(self.batch_size,), compression="gzip"
                    )
                    hf.create_dataset(
                        "shot_ids", (0,), maxshape=(None,), dtype='S100',
                        chunks=(self.batch_size,), compression="gzip"
                    )
            
            # Process shots
            with h5py.File(temp_path, 'a') as hf:
                self._process_shots(
                    shot_info, brand_id, hf, processed_shots, initial_size
                )
            
            # Move temp to final location
            shutil.move(temp_path, output_path)
            print(f"Successfully saved embeddings for {len(processed_shots)} shots")
        
        except Exception as e:
            print(f"Error during processing: {e}")
            print("Temporary files preserved for recovery")
            raise
        
        finally:
            # Clean up lock
            if os.path.exists(lock_path):
                os.remove(lock_path)
    
    def _process_shots(self, shot_info, brand_id, hf, processed_shots, initial_size):
        """Process shots and save to H5; GPU forward pass is batched (batch_size)."""
        current_size = initial_size
        pending = []  # list of (shot_id, frame_paths) awaiting a GPU forward

        def flush():
            """Run one batched forward over `pending`, then write to H5."""
            nonlocal current_size, pending
            if not pending:
                return

            # Batched GPU forward; on failure, fall back to per-shot so a single
            # bad clip cannot drop the whole batch (preserves old robustness).
            try:
                embs = self.embedder.get_embeddings_batch([fp for _, fp in pending])
                results = list(zip(pending, embs))
            except Exception as e:
                print(f"Batched forward failed ({e}); falling back to per-shot")
                results = []
                for sid, fps in pending:
                    try:
                        results.append(((sid, fps), self.embedder.get_embedding(fps)))
                    except Exception as e2:
                        print(f"Error processing shot {sid}: {e2}")

            b_emb, b_bid, b_sid = [], [], []
            for (sid, _fps), emb in results:
                b_emb.append(emb)
                b_bid.append(brand_id)
                b_sid.append(sid.encode('ascii'))
                processed_shots.add(sid)

            pending = []
            if b_emb:
                self._save_batch(hf, b_emb, b_bid, b_sid, current_size)
                current_size += len(b_emb)

        for shot_id, frame_paths in tqdm(shot_info, desc="Generating embeddings"):
            # Skip if already processed
            if shot_id in processed_shots:
                continue

            # Load images and verify
            images_ok = True
            for fp in frame_paths:
                try:
                    Image.open(fp).verify()
                except Exception:
                    print(f"Invalid image in shot {shot_id}: {fp}")
                    images_ok = False
                    break
            if not images_ok:
                continue

            pending.append((shot_id, frame_paths))
            if len(pending) >= self.batch_size:
                flush()

        # Flush remaining
        flush()
    
    def _save_batch(self, hf, embeddings, brand_ids, shot_ids, current_size):
        """Save a batch of embeddings to H5"""
        try:
            new_size = current_size + len(embeddings)
            
            # Resize datasets
            hf["embeddings"].resize(new_size, axis=0)
            hf["brand_ids"].resize(new_size, axis=0)
            hf["shot_ids"].resize(new_size, axis=0)
            
            # Add data
            hf["embeddings"][current_size:new_size] = np.array(embeddings)
            hf["brand_ids"][current_size:new_size] = np.array(brand_ids)
            hf["shot_ids"][current_size:new_size] = np.array(shot_ids)
            
            # Flush
            hf.flush()
        
        except Exception as e:
            print(f"Error saving batch: {e}")
            raise


def load_embeddings(h5_path):
    """
    Load embeddings from H5 file

    Returns:
        embeddings: numpy array (N, 1024)
        brand_ids: numpy array (N,)
        shot_ids: list of strings
    """
    try:
        with h5py.File(h5_path, 'r') as hf:
            embeddings = hf["embeddings"][:]
            brand_ids = hf["brand_ids"][:]
            shot_ids = [sid.decode('ascii') for sid in hf["shot_ids"][:]]
        return embeddings, brand_ids, shot_ids
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise


# ---------------------------------------------------------------------------
# Batch-level embedding processor
# ---------------------------------------------------------------------------
# Unit of work is a BATCH (one videos CSV). The Swin3D-B model is loaded ONCE
# for the whole batch; all valid 5-keyframe shots across every brand in the
# batch are collected into one flat list and pushed through the GPU in batches
# that may span brands (the embedding does not depend on brand). Brand identity
# matters only at SAVE time: each embedding is routed to its brand's
# `{brand_id}_{brand_name}_embeddings.h5`.

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config
from s1_keyframes import KeyframeExtractor

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Generate Swin3D embeddings for a whole batch (CSV) of brands' keyframes."""

    def __init__(self, data_df, keyframes_dir=None, output_dir=None, batch_size=32):
        self.data_df = data_df
        self.keyframes_dir = Path(keyframes_dir or config.KEYFRAMES_DIR)
        self.embeddings_dir = Path(output_dir or config.EMBEDDINGS_DIR)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        self.embedder = Swin3DEmbedder()        # loaded ONCE for the batch
        self.extractor = KeyframeExtractor()    # only for find_valid_shots

    def _brands(self):
        """Ordered dict brand_id -> brand_name for the brands in this batch."""
        brands = {}
        for _, row in self.data_df.iterrows():
            brands.setdefault(row['brand_id'], row['brand_name'])
        return brands

    def _h5_path(self, brand_id, brand_name):
        return self.embeddings_dir / f"{brand_id}_{brand_name}_embeddings.h5"

    @staticmethod
    def _existing_shots(h5_path):
        if not h5_path.exists():
            return set()
        try:
            with h5py.File(h5_path, 'r') as hf:
                if 'shot_ids' in hf:
                    return {s.decode('ascii') for s in hf['shot_ids'][:]}
        except Exception:
            pass
        return set()

    def _collect_items(self, skip_existing=True):
        """Flat list of (shot_id, frame_paths, brand_id, brand_name) across batch."""
        items = []
        n_total = n_done = 0
        for brand_id, brand_name in self._brands().items():
            brand_dir = self.keyframes_dir / str(brand_id)
            if not brand_dir.exists():
                logger.warning(f"No keyframes dir for brand {brand_id} ({brand_name})")
                continue
            done = self._existing_shots(self._h5_path(brand_id, brand_name)) if skip_existing else set()
            shots = self.extractor.find_valid_shots(str(brand_dir))
            n_total += len(shots)
            for shot_id, frame_paths in shots:
                if shot_id in done:
                    n_done += 1
                    continue
                items.append((shot_id, frame_paths, brand_id, brand_name))
        logger.info(
            f"Batch: {len(self._brands())} brands | {n_total} valid shots | "
            f"{n_done} already embedded | {len(items)} to process"
        )
        return items

    def _write_brand_h5(self, brand_id, brand_name, shot_ids, embeddings):
        """Append (or create) a brand's H5 with the standard schema."""
        h5_path = self._h5_path(brand_id, brand_name)
        embeddings = np.asarray(embeddings, dtype='float32')
        mode = 'a' if h5_path.exists() else 'w'
        with h5py.File(h5_path, mode) as hf:
            if 'embeddings' not in hf:
                hf.create_dataset("embeddings", (0, 1024), maxshape=(None, 1024),
                                  dtype='float32', chunks=(self.batch_size, 1024),
                                  compression="gzip")
                hf.create_dataset("brand_ids", (0,), maxshape=(None,), dtype='int32',
                                  chunks=(self.batch_size,), compression="gzip")
                hf.create_dataset("shot_ids", (0,), maxshape=(None,), dtype='S100',
                                  chunks=(self.batch_size,), compression="gzip")
            cur = hf['embeddings'].shape[0]
            new = cur + len(shot_ids)
            hf['embeddings'].resize(new, axis=0)
            hf['brand_ids'].resize(new, axis=0)
            hf['shot_ids'].resize(new, axis=0)
            hf['embeddings'][cur:new] = embeddings
            hf['brand_ids'][cur:new] = np.full(len(shot_ids), brand_id, dtype='int32')
            hf['shot_ids'][cur:new] = np.array([s.encode('ascii') for s in shot_ids])
            hf.flush()

    def run(self, skip_existing=True):
        items = self._collect_items(skip_existing=skip_existing)
        if not items:
            logger.info("Nothing to process.")
            return 0

        # GPU forward in batches that may span brands.
        results = defaultdict(lambda: ([], []))  # brand_id -> (shot_ids, embeddings)
        for i in tqdm(range(0, len(items), self.batch_size), desc="Embeddings (batch)"):
            chunk = items[i:i + self.batch_size]
            frame_path_lists = [fp for _, fp, _, _ in chunk]
            try:
                embs = self.embedder.get_embeddings_batch(frame_path_lists)
            except Exception as e:
                logger.warning(f"Batched forward failed ({e}); falling back to per-shot")
                embs = np.stack([self.embedder.get_embedding(fp) for fp in frame_path_lists])
            for (shot_id, _fp, brand_id, _bn), emb in zip(chunk, embs):
                sids, es = results[brand_id]
                sids.append(shot_id)
                es.append(emb)

        # Route results to per-brand H5 files.
        total = 0
        for brand_id, brand_name in self._brands().items():
            if brand_id not in results:
                continue
            sids, es = results[brand_id]
            self._write_brand_h5(brand_id, brand_name, sids, es)
            total += len(sids)
            logger.info(f"  brand {brand_id} ({brand_name}): {len(sids)} embeddings -> "
                        f"{self._h5_path(brand_id, brand_name).name}")

        logger.info(f"Embedding generation done: {total} embeddings -> {self.embeddings_dir}")
        return total


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Embedding processor (batch CSV is the unit of work)")
    parser.add_argument('--data_csv', required=True,
                        help="Path to the batch videos CSV")
    parser.add_argument('--keyframes_dir', default=config.KEYFRAMES_DIR,
                        help="Keyframes root with per-brand {brand_id}/ subdirs")
    parser.add_argument('--output_dir', default=config.EMBEDDINGS_DIR,
                        help="Where to write per-brand *_embeddings.h5")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_skip_existing', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_df = pd.read_csv(args.data_csv)
    logger.info(f"Loaded {len(data_df)} videos from "
                f"{data_df['brand_name'].nunique()} brands ({args.data_csv})")

    processor = EmbeddingProcessor(
        data_df=data_df,
        keyframes_dir=args.keyframes_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    processor.run(skip_existing=not args.no_skip_existing)


if __name__ == '__main__':
    main()