"""
Clustering script for brand shot embeddings
Processes each brand independently and saves clustering results
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

import utils
import config  # central config.yaml loader (CLI defaults)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brand_clustering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _shot_sort_key(shot_id):
    """Sort key matching the legacy keyframe enumeration: (ad_id, shot_num)."""
    parts = shot_id.rsplit('_shot_', 1)
    if len(parts) == 2:
        return (parts[0], int(parts[1]))
    return (shot_id, 0)


class BrandClusteringProcessor:
    """Process shot clustering for each brand"""
    
    def __init__(self, embeddings_dir, keyframes_dir, output_dir):
        self.embeddings_dir = Path(embeddings_dir)
        self.keyframes_dir = Path(keyframes_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized BrandClusteringProcessor")
    
    def extract_unique_shots(self, brand_id):
        """
        Extract unique shot identifiers from percentile-based filenames
        
        Args:
            brand_id: The brand ID (e.g., 19)
            
        Returns:
            List of unique shot identifiers sorted by ad_id and shot number
        """
        brand_path = self.keyframes_dir / str(brand_id)
        if not brand_path.exists():
            raise FileNotFoundError(f"Directory not found: {brand_path}")
        
        # Pattern to match percentile-based filenames
        pattern = re.compile(r'(.+_shot_\d+)_p\d+\.jpg$')
        
        unique_shots = set()
        
        for filename in os.listdir(brand_path):
            match = pattern.match(filename)
            if match:
                unique_shots.add(match.group(1))
        
        # Sort by ad_id and then shot number
        def sort_key(shot_id):
            parts = shot_id.rsplit('_shot_', 1)
            if len(parts) == 2:
                ad_id = parts[0]
                shot_num = int(parts[1])
                return (ad_id, shot_num)
            return (shot_id, 0)
        
        return sorted(unique_shots, key=sort_key)
    
    def get_embeddings_from_memory(self, all_embeddings, all_shot_ids, target_shots):
        """
        Ultra-fast embedding retrieval from pre-loaded arrays
        
        Args:
            all_embeddings: numpy array of shape (N, embedding_dim)
            all_shot_ids: list of N shot IDs (strings)
            target_shots: list of shot IDs to find
            
        Returns:
            dict: {shot_id: embedding} for found shots
            list: missing shot IDs
        """
        all_shot_ids = np.array(all_shot_ids)
        
        embeddings_dict = {}
        missing_shots = []
        
        for shot in target_shots:
            mask = (all_shot_ids == shot)
            if np.any(mask):
                idx = np.where(mask)[0][0]
                embeddings_dict[shot] = all_embeddings[idx]
            else:
                missing_shots.append(shot)
        
        return embeddings_dict, missing_shots
    
    def prepare_embeddings(self, embeddings_dict):
        """Convert and normalize embeddings for consistent processing"""
        emb_matrix = np.array(list(embeddings_dict.values()))
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Found zero-norm embeddings. Check your input vectors.")
        return emb_matrix / norms

#     def find_best_eps_and_cluster(self, embeddings, eps_range=None, min_samples=1):
#         """
#         Scan multiple eps values for DBSCAN and return best clustering

#         Args:
#             embeddings: Dict[shot_id, embedding]
#             eps_range: Range of eps values to test
#             min_samples: Minimum samples for DBSCAN

#         Returns:
#             Dictionary with best_eps, best_score, clusters, all_results
#         """
#         if eps_range is None:
#             eps_range = np.linspace(0.05, 0.9, 20)

#         shot_ids = list(embeddings.keys())
#         emb_matrix = np.array(list(embeddings.values()))

#         # Normalize embeddings
#         norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
#         if np.any(norms == 0):
#             raise ValueError("Zero-norm embeddings found.")
#         emb_matrix = emb_matrix / norms

#         best_result = {'eps': None, 'silhouette': -1, 'labels': None}
#         all_results = []

#         for eps in eps_range:
#             try:
#                 db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
#                 labels = db.fit_predict(emb_matrix)
#                 unique_labels = set(labels) - {-1}
#                 n_clusters = len(unique_labels)

#                 if 2 <= n_clusters < len(emb_matrix):
#                     score = silhouette_score(emb_matrix, labels)
#                 else:
#                     score = -1
#             except:
#                 score = -1
#                 n_clusters = -1
#                 labels = None

#             all_results.append({
#                 'eps': float(eps),
#                 'silhouette': float(score),
#                 'n_clusters': int(n_clusters)
#             })

#             if score > best_result['silhouette']:
#                 best_result = {
#                     'eps': eps,
#                     'silhouette': score,
#                     'labels': labels
#                 }

#         # Rebuild cluster assignment using best labels
#         clusters = {}
#         if best_result['labels'] is not None:
#             for idx, label in enumerate(best_result['labels']):
#                 label_key = int(label)
#                 if label_key not in clusters:
#                     clusters[label_key] = []
#                 clusters[label_key].append(shot_ids[idx])

#         return {
#             'best_eps': float(best_result['eps']) if best_result['eps'] is not None else None,
#             'best_score': float(best_result['silhouette']) if best_result['silhouette'] != -1 else None,
#             'clusters': clusters,
#             'all_results': all_results
#         }

    def find_best_eps_and_cluster(self, embeddings, eps_range=None, min_samples=1):
        """
        Smart clustering with automatic mode detection
        Detects local maxima in silhouette scores and picks the first valid mode

        Args:
            embeddings: Dict[shot_id, embedding]
            eps_range: Range of eps values to test
            min_samples: Minimum samples for DBSCAN

        Returns:
            Dictionary with best_eps, best_score, clusters, all_results
        """
        if eps_range is None:
            eps_range = np.linspace(0.05, 0.9, 35)  # Fine-grained for mode detection

        shot_ids = list(embeddings.keys())
        emb_matrix = np.array(list(embeddings.values()))

        # Normalize embeddings
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero-norm embeddings found.")
        emb_matrix = emb_matrix / norms

        total_shots = len(emb_matrix)
        all_results = []

        # Scan all eps values
        for eps in eps_range:
            try:
                db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                labels = db.fit_predict(emb_matrix)
                n_clusters = len(set(labels) - {-1})

                if 2 <= n_clusters < total_shots:
                    sil_score = silhouette_score(emb_matrix, labels)
                    cluster_sizes = np.bincount(labels[labels >= 0])
                    largest_cluster = int(cluster_sizes.max())
                    giant_ratio = largest_cluster / total_shots
                else:
                    sil_score = -1
                    largest_cluster = 0
                    giant_ratio = 0.0

                all_results.append({
                    'eps': float(eps),
                    'silhouette': float(sil_score),
                    'n_clusters': int(n_clusters),
                    'largest_cluster': largest_cluster,
                    'giant_ratio': float(giant_ratio),
                    'labels': labels if sil_score > 0 else None
                })

            except Exception as e:
                all_results.append({
                    'eps': float(eps),
                    'silhouette': -1,
                    'n_clusters': -1,
                    'largest_cluster': 0,
                    'giant_ratio': 0.0,
                    'labels': None
                })

        # Convert to array for mode detection
        silhouettes = np.array([r['silhouette'] for r in all_results])

        # DETECT MODES: Find local maxima in silhouette curve
        from scipy.signal import find_peaks

        # Find peaks with minimum prominence to avoid noise
        peaks, properties = find_peaks(
            silhouettes, 
            prominence=0.01,  # Minimum prominence to be considered a peak
            distance=2         # Minimum distance between peaks
        )

        # Filter peaks: only keep those with valid silhouette scores
        valid_peaks = [p for p in peaks if silhouettes[p] > 0]

        if valid_peaks:
            # SELECT FIRST MODE (lowest eps peak)
            first_peak_idx = valid_peaks[0]
            best_result = all_results[first_peak_idx]

            print(f"Detected {len(valid_peaks)} modes at eps: {[eps_range[p] for p in valid_peaks]}")
            print(f"Selecting first mode at eps={eps_range[first_peak_idx]:.3f}")
        else:
            # No clear modes detected, fall back to simple max silhouette
            # But filter out giant cluster cases first
            valid_results = [r for r in all_results if r['silhouette'] > 0 and r['giant_ratio'] < 0.5]

            if valid_results:
                best_result = max(valid_results, key=lambda x: x['silhouette'])
                print("No clear modes detected, using best silhouette with giant_ratio < 0.5")
            else:
                # Last resort: just use max silhouette
                best_result = max(all_results, key=lambda x: x['silhouette'])
                print("Warning: No good clustering found, using best available")

        # Build clusters from best result
        clusters = {}
        if best_result['labels'] is not None:
            for idx, label in enumerate(best_result['labels']):
                label_key = int(label)
                if label_key not in clusters:
                    clusters[label_key] = []
                clusters[label_key].append(shot_ids[idx])

        return {
            'best_eps': best_result['eps'],
            'best_score': best_result['silhouette'],
            'n_clusters': best_result['n_clusters'],
            'clusters': clusters,
            'all_results': [
                {k: v for k, v in r.items() if k != 'labels'}  # Remove labels from saved results
                for r in all_results
            ]
        }
    

    def load_brand_embeddings(self, brand_id, brand_name):
        """Load embeddings for a specific brand"""
        embedding_file = self.embeddings_dir / f"{brand_id}_{brand_name}_embeddings.h5"
        
        if not embedding_file.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
        
        try:
            with h5py.File(embedding_file, 'r') as hf:
                embeddings = hf["embeddings"][:]
                brand_ids = hf["brand_ids"][:]
                shot_ids = [sid.decode('ascii') for sid in hf["shot_ids"][:]]
            
            logger.info(f"Loaded {len(shot_ids)} embeddings for brand {brand_name}")
            return embeddings, shot_ids
        
        except Exception as e:
            logger.error(f"Error loading embeddings for {brand_name}: {e}")
            raise
    
    def get_brand_shot_embeddings(self, brand_id, embeddings, shot_ids):
        """
        Get all shot embeddings for a given brand

        Args:
            brand_id: The brand ID
            embeddings: Array of embeddings from h5 file
            shot_ids: List of shot_ids from h5 file

        Returns:
            Dictionary mapping shot_id to embedding
        """
        # The shots to cluster come straight from the embeddings H5 (shot_ids),
        # which s1bc/s1_embedding populated with exactly the valid shots. This
        # used to be re-derived by globbing keyframe JPGs (extract_unique_shots);
        # the H5 already carries the identical set, so clustering no longer needs
        # the keyframe directory. Re-sort by (ad_id, shot_num) so the order --
        # and therefore the clustering output -- is unchanged.
        target_shots = sorted(set(shot_ids), key=_shot_sort_key)

        # Get embeddings from memory
        shot_embeddings, missing = self.get_embeddings_from_memory(
            embeddings,
            shot_ids,
            target_shots
        )
        
        if missing:
            logger.warning(f"{len(missing)} shots missing embeddings for brand {brand_id}")
        
        return shot_embeddings
    
    def process_brand_clustering(self, brand_id, brand_name):
        """Process clustering for a single brand"""
        logger.info(f"Processing clustering for brand: {brand_name} (ID: {brand_id})")
        
        output_file = self.output_dir / f"{brand_id}_{brand_name}_clusters.json"
        
        # Check if already processed
        if output_file.exists():
            logger.info(f"Clustering already exists for {brand_name}, skipping")
            return output_file
        
        try:
            # Load embeddings
            embeddings, shot_ids = self.load_brand_embeddings(brand_id, brand_name)
            
            # Get shot embeddings for this brand
            shot_embeddings = self.get_brand_shot_embeddings(brand_id, embeddings, shot_ids)
            
            if not shot_embeddings:
                logger.warning(f"No shot embeddings found for brand {brand_name}")
                return None
            
            logger.info(f"Found {len(shot_embeddings)} shots for clustering")
            
            # Find best clustering
            result = self.find_best_eps_and_cluster(shot_embeddings)
            
            # Prepare results
            results = {
                'brand_id': int(brand_id),
                'brand_name': brand_name,
                'n_shots': len(shot_embeddings),
                'best_eps': result['best_eps'],
                'best_score': result['best_score'],
                'n_clusters': len(result['clusters']) if result['clusters'] else 0,
                'clusters': result['clusters']
            }
            
            # Save to JSON
            utils.write_json_row(str(output_file), results)
            
            logger.info(f"Saved clustering results: {result['best_eps']:.3f} eps, "
                       f"{len(result['clusters']) if result['clusters'] else 0} clusters, "
                       f"{result['best_score']:.3f} silhouette")
            
            return output_file
        
        except Exception as e:
            logger.error(f"Error processing brand {brand_name}: {str(e)}")
            return None
    
    def get_available_brands(self):
        """Get list of brands with embedding files"""
        available_brands = []
        
        for embedding_file in self.embeddings_dir.glob('*_embeddings.h5'):
            # Parse filename: {brand_id}_{brand_name}_embeddings.h5
            filename = embedding_file.stem.replace('_embeddings', '')
            
            # Split by first underscore to get brand_id and brand_name
            parts = filename.split('_', 1)
            if len(parts) == 2:
                brand_id = int(parts[0])
                brand_name = parts[1]
                available_brands.append((brand_id, brand_name))
        
        available_brands.sort(key=lambda x: x[0])  # Sort by brand_id
        return available_brands
    
    def process_all_brands(self):
        """Process clustering for all available brands"""
        available_brands = self.get_available_brands()
        logger.info(f"Found {len(available_brands)} brands with embeddings")
        
        results = {}
        for brand_id, brand_name in tqdm(available_brands, desc="Processing brands"):
            success = self.process_brand_clustering(brand_id, brand_name)
            results[brand_name] = success is not None
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Clustering Summary:")
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
    parser = argparse.ArgumentParser(description='Cluster brand shot embeddings using DBSCAN')
    parser.add_argument('--embeddings_dir', type=str, default=config.EMBEDDINGS_DIR,
                       help='Directory containing brand embedding H5 files')
    parser.add_argument('--keyframes_dir', type=str, default=config.KEYFRAMES_DIR,
                       help='Directory containing brand keyframe folders')
    parser.add_argument('--output_dir', type=str, default=config.CLUSTERS_DIR,
                       help='Output directory for clustering results')
    parser.add_argument('--brand', type=str, default=None,
                       help='Process specific brand by name (optional)')
    parser.add_argument('--brand_id', type=int, default=None,
                       help='Process specific brand by ID (optional)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BrandClusteringProcessor(
        embeddings_dir=args.embeddings_dir,
        keyframes_dir=args.keyframes_dir,
        output_dir=args.output_dir
    )
    
    # Process specific brand or all brands
    if args.brand_id is not None or args.brand is not None:
        # Get available brands to find the match
        available_brands = processor.get_available_brands()
        
        brand_to_process = None
        for brand_id, brand_name in available_brands:
            if args.brand_id is not None and brand_id == args.brand_id:
                brand_to_process = (brand_id, brand_name)
                break
            elif args.brand is not None and brand_name == args.brand:
                brand_to_process = (brand_id, brand_name)
                break
        
        if brand_to_process:
            processor.process_brand_clustering(brand_to_process[0], brand_to_process[1])
        else:
            logger.error(f"Brand not found: {args.brand or args.brand_id}")
    else:
        processor.process_all_brands()


if __name__ == "__main__":
    main()