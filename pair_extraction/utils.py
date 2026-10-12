import os
import requests
import ast
import json
import numpy as np
import pickle
import shutil
import pandas as pd
import json
import cv2 
import matplotlib.pyplot as plt

# change_settings({"FFMPEG_BINARY": "/work/tvision/bin/bin/ffmpeg"})
# change_settings({"FFMPEG_BINARY": "/projects/adsum-mllm-om/miniconda3/envs/shot/bin/ffmpeg"})
os.environ["FFMPEG_BINARY"] = "/projects/adsum-mllm-om/miniconda3/envs/shot/bin/ffmpeg"

import moviepy as mp

from utils_autoshot import evaluate_scenes, predictions_to_scenes
from utils_autoshot import get_frames, get_batches, scenes2zero_one_representation, visualize_predictions
import ffmpeg
import torch

import math
from PIL import Image
from typing import Union, List
import re
from typing import Dict
import h5py


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import json
from tqdm import tqdm
import concurrent.futures

import os, shutil

os.environ["PATH"] = "/projects/adsum-mllm-om/miniconda3/envs/shot/bin:" + os.environ["PATH"]
print("ffmpeg after patch:", shutil.which("ffmpeg"))


def find_max_in_sequences(data, min_gap=3):
    """
    Find the maximum value in each sequence of non-zero values, where a sequence is defined as a series of non-zero values separated by less than 3 consecutive zeros.
    
    Args:
        data (list): The input list of values.
        min_gap (int): The minimum number of consecutive zeros required to consider a new sequence.
        
    Returns:
        list: A list of tuples, where each tuple contains the index and value of the maximum in a sequence.
    """
    max_in_sequences = []
    in_segment = False
    prev_value = 0
    max_value = 0
    max_index = 0
    zero_count = 0
    
    for i, value in enumerate(data):
        if value > 0:
            if not in_segment or value > prev_value:
                in_segment = True
                prev_value = value
                max_value = value
                max_index = i
            else:
                if value > max_value:
                    max_value = value
                    max_index = i
            zero_count = 0
        else:
            zero_count += 1
            if zero_count >= min_gap:
                if in_segment:
                    max_in_sequences.append((max_index, max_value))
                in_segment = False
                prev_value = 0
                max_value = 0
                max_index = 0
    
    if in_segment:
        max_in_sequences.append((max_index, max_value))
    
    return max_in_sequences


def write_json_row(path, row):
    """
    Writes a single JSON row to a file.
    
    Args:
        path (str): The path to the output JSON file.
        row (dict): The dictionary containing the data for a single row.
    """
    
    for key, value in row.items():
        # Handle numpy types
        if isinstance(value, np.float32):
            row[key] = float(value)
        elif isinstance(value, (np.int64, np.int32)):
            row[key] = int(value)
        elif isinstance(value, np.ndarray):
            row[key] = value.tolist()
        # Handle pandas types
        elif hasattr(value, 'item'):  # pandas scalar types
            row[key] = value.item()
            
    with open(path, 'a') as jsonfile:
        json.dump(row, jsonfile)
        jsonfile.write('\n')

        
def get_frames_by_indices(video_path, frame_indices):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    frames_rgb = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure the indices are valid
    valid_indices = [i for i in frame_indices if 0 <= i < total_frames]

    # Read and store frames at the given indices
    for idx in valid_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the video to the specific frame
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
        else:
            print(f"Error reading frame at index {idx}")
    
    cap.release()
    return frames_rgb


def show_frames_in_row(frames, frame_indices):
    # Set the figure size based on the number of frames
    plt.figure(figsize=(15, 5))
    
    # Plot each frame in a subplot
    for i, frame in enumerate(frames):
        plt.subplot(1, len(frames), i + 1)
        plt.imshow(frame)  # Assuming frames are already in RGB format
        plt.title(f"Frame {frame_indices[i]}")
        plt.axis('off')  # Hide axis

    plt.tight_layout()
    plt.show()
    
    
def predict(batch, model):
    if torch.cuda.is_available() is True:
        device = "cuda"
    else:
        device = "cpu"
    
    batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
    batch = batch.to(device)
    one_hot = model(batch)
    if isinstance(one_hot, tuple):
        one_hot = one_hot[0]
    return torch.sigmoid(one_hot[0])


def get_autoshot_pred(video_path, model):
    predictions = []
    frames = get_frames(video_path)
    
    for batch in get_batches(frames):
        one_hot = predict(batch, model)
        one_hot = one_hot.detach().cpu().numpy()
        
        predictions.append(one_hot[25:75])
    
    predictions = np.concatenate(predictions, 0)[:len(frames)].reshape(-1)
    return frames, predictions


def show_images(
    images: Union[List[str], np.ndarray, List[np.ndarray]], 
    rows: int = 1,
    figsize: tuple = (15, 8),
    titles: Union[List[str], None] = None,
    channel_order: str = 'rgb'
):
    """
    Enhanced image display for (N,180,320,3) arrays and file paths.
    
    Args:
        images: Input can be:
                - (N,180,320,3) numpy array
                - List of image paths
                - List of (180,320,3) arrays
        rows: Number of display rows
        figsize: Figure dimensions in inches
        titles: Optional custom titles
        channel_order: 'rgb' or 'bgr' for numpy arrays
    """
    # Convert single array to list of arrays if needed
    if isinstance(images, np.ndarray):
        if images.ndim == 4:
            images = [images[i] for i in range(images.shape[0])]
        elif images.ndim == 3:
            images = [images]
        else:
            raise ValueError(f"Expected 3D or 4D array, got {images.ndim}D")

    num_images = len(images)
    cols = math.ceil(num_images / rows)
    
    plt.figure(figsize=figsize)
    
    for i, img in enumerate(images, 1):
        plt.subplot(rows, cols, i)
        
        try:
            # Handle path inputs
            if isinstance(img, str):
                img_display = Image.open(img)
                title = img.split('/')[-1].split('\\')[-1]  # Cross-platform filename
                
            # Handle array inputs
            elif isinstance(img, np.ndarray):
                if img.ndim not in [2, 3]:
                    raise ValueError(f"Array must be 2D or 3D, got {img.ndim}D")
                
                # Channel ordering
                if channel_order.lower() == 'bgr' and img.ndim == 3:
                    img = img[..., ::-1]  # BGR to RGB
                
                # Normalize if needed
                if img.dtype == np.float32:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                
                img_display = img
                title = f"Frame {i}"
                
            else:
                raise TypeError(f"Unsupported type: {type(img)}")
            
            plt.imshow(img_display, vmin=0, vmax=255)
            plt.title(titles[i-1] if titles else title, fontsize=8)
            
        except Exception as e:
            print(f"Error displaying image {i}: {str(e)}")
            plt.text(0.5, 0.5, f"Error\n{type(img).__name__}", 
                    ha='center', va='center', fontsize=8)
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    


def visualize_all_clusters(clusters: Dict[int, List[str]], pair_id: int, max_shots_per_cluster=20,
                           max_cols=20, percentile=25, save_path=None):
    """
    Visualize all clusters as a grid with each cluster in one row, and shots in columns with individual titles.
    """
    base_dir = f"frames/shot_frames_campaign/{pair_id}"
    cluster_ids = [cid for cid in clusters if clusters[cid]]
    num_clusters = len(cluster_ids)
    
    # Dynamically determine figure size
    fig_width = max_cols * 1.5
    fig_height = num_clusters * 1.5
    fig, axes = plt.subplots(num_clusters, max_cols, figsize=(fig_width, fig_height))

    if num_clusters == 1:
        axes = np.expand_dims(axes, 0)
    if max_cols == 1:
        axes = np.expand_dims(axes, 1)

    for i, cluster_id in enumerate(cluster_ids):
        shot_ids = clusters[cluster_id][:max_shots_per_cluster]
        for j in range(max_cols):
            ax = axes[i, j]
            ax.axis('off')
            if j < len(shot_ids):
                shot_id = shot_ids[j]
                frame_path = os.path.join(base_dir, f"{shot_id}_p{percentile}.jpg")
                try:
                    img = Image.open(frame_path)
                    ax.imshow(img)
                    ax.set_title(shot_id.split('-')[-1], fontsize=8)
                except Exception as e:
                    ax.set_title("Missing", fontsize=8)
                    print(f"Error loading {shot_id}: {e}")
        
        # Label the row (cluster)
        axes[i, 0].text(-0.5, 0.5, f"Cluster {cluster_id}", fontsize=10,
                        ha='right', va='center', rotation=90, transform=axes[i, 0].transAxes)
    
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"pair_{pair_id}_clusters_rowwise.png")
        plt.savefig(full_path, bbox_inches='tight', dpi=400)  # <- High-res save
        print(f"Saved: {full_path}")
    
    plt.show()

    
def extract_unique_shots(pair_id, frames_path='frames/shot_frames_campaign'):
    """
    Extract unique shot identifiers from percentile-based filenames in format:
    'AE16546-2020-52-00730_shot_01_p10.jpg' → 'AE16546-2020-52-00730_shot_01'
    
    Args:
        pair_id: The campaign pair ID (e.g., 16508)
        frames_path: Base path containing pair_id folders
        
    Returns:
        List of unique shot identifiers sorted by shot number
    """
    campaign_path = os.path.join(frames_path, str(pair_id))
    if not os.path.exists(campaign_path):
        raise FileNotFoundError(f"Directory not found: {campaign_path}")

    # Pattern to strictly match percentile-based filenames
    pattern = re.compile(r'(.+_shot_\d+)_p\d+\.jpg$')
    
    unique_shots = set()
    
    for filename in os.listdir(campaign_path):
        match = pattern.match(filename)
        if match:
            unique_shots.add(match.group(1))  # Extract the shot base name
    
    # Sort by shot number (01, 02, etc.)
    return sorted(unique_shots, key=lambda x: int(x.split('_')[-1]))


def get_embeddings_from_memory(all_embeddings, all_shot_ids, target_shots):
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
    # Convert to numpy array for fast operations
    all_shot_ids = np.array(all_shot_ids)
    
    # Create dictionary for results
    embeddings_dict = {}
    missing_shots = []
    
    # Find indices of target shots using numpy's vectorized operations
    for shot in target_shots:
        mask = (all_shot_ids == shot)
        if np.any(mask):
            idx = np.where(mask)[0][0]
            embeddings_dict[shot] = all_embeddings[idx]
        else:
            missing_shots.append(shot)
    
    return embeddings_dict, missing_shots


def load_embeddings(h5_path):
    """Safely load embeddings with error handling"""
    try:
        with h5py.File(h5_path, 'r') as hf:
            embeddings = hf["embeddings"][:]
            pair_ids = hf["pair_ids"][:]
            shot_ids = [sid.decode('ascii') for sid in hf["shot_ids"][:]]
        return embeddings, pair_ids, shot_ids
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise
        
        
def plot_silhouette_scan(embeddings_dict):
    emb_matrix = prepare_embeddings(embeddings_dict)
    eps_range = np.linspace(0.05, 0.9, 30)
    scores = []
    
    for eps in eps_range:
        labels = DBSCAN(eps=eps, min_samples=1, metric='cosine').fit_predict(emb_matrix)
        if len(set(labels)) > 1:
            scores.append(silhouette_score(emb_matrix, labels))
        else:
            scores.append(-1)
    
    plt.plot(eps_range, scores, 'bo-')
    plt.xlabel('eps')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for DBSCAN')
    plt.show()
    
    

def prepare_embeddings(embeddings_dict):
    """Convert and normalize embeddings for consistent processing"""
    emb_matrix = np.array(list(embeddings_dict.values()))
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Found zero-norm embeddings. Check your input vectors.")
    return emb_matrix / norms

def eps_silhouette_scan(embeddings_dict, eps_range=np.linspace(0.05, 0.9, 20)):  # Expanded range
    """
    Enhanced version that:
    - Tests full theoretical range (0.05-1.5 for cosine distance)
    - Returns ALL scores for analysis
    - Handles edge cases robustly
    """
    emb_matrix = prepare_embeddings(embeddings_dict)
    results = []
    
    for eps in eps_range:
        try:
            labels = DBSCAN(eps=eps, min_samples=1, metric='cosine').fit_predict(emb_matrix)
            unique_labels = set(labels) - {-1}  # Exclude noise
            n_clusters = len(unique_labels)
            
            if 2 <= n_clusters < len(emb_matrix):
                score = silhouette_score(emb_matrix, labels)
            else:
                score = -1  # Invalid for silhouette
        except:
            score = -1
        
        results.append({
            'eps': float(eps),
            'silhouette': float(score),
            'n_clusters': int(n_clusters) if 'n_clusters' in locals() else -1
        })
    
    # Find best valid eps
    valid_results = [r for r in results if r['silhouette'] != -1]
    best_result = max(valid_results, key=lambda x: x['silhouette']) if valid_results else None
    
    return {
        'best_eps': best_result['eps'] if best_result else None,
        'best_score': best_result['silhouette'] if best_result else None,
        'all_results': results
    }


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def count_files_in_dir(directory, extension='.mp4'):
    """Count files with specific extension in directory"""
    if not Path(directory).exists():
        return 0
    return len(list(Path(directory).glob(f'*{extension}')))


def parse_shot_splits(shot_str):
    """
    Parse shot split string to list
    
    Args:
        shot_str: String representation of list (e.g., "[10, 50, 100]")
    
    Returns:
        List of integers
    """
    if isinstance(shot_str, str):
        return ast.literal_eval(shot_str)
    return shot_str

