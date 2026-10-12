"""
Track processing status for each brand and video
"""

import json
import os
from pathlib import Path
from typing import Dict, Set, List
import pandas as pd
import numpy as np


class ProcessingStatusTracker:
    """Track which videos have completed each processing stage"""
    
    def __init__(self, status_file='data/processing_status.json'):
        self.status_file = Path(status_file)
        self.status = self._load_status()
    
    def _load_status(self) -> Dict:
        """Load existing status or create new"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_status(self):
        """Save status to file with proper type conversion"""
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy/pandas types to native Python types
        clean_status = self._convert_to_json_serializable(self.status)
        
        with open(self.status_file, 'w') as f:
            json.dump(clean_status, f, indent=2)
    
    def _convert_to_json_serializable(self, obj):
        """Recursively convert numpy/pandas types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # pandas scalar types
            return obj.item()
        else:
            return obj
    
    def get_brand_status(self, brand_id: int) -> Dict:
        """Get status for a brand"""
        brand_key = str(brand_id)
        if brand_key not in self.status:
            self.status[brand_key] = {
                'brand_id': int(brand_id),  # Ensure Python int
                'shots_completed': [],
                'keyframes_completed': [],
                'embeddings_completed': False
            }
        return self.status[brand_key]
    
    def mark_shots_complete(self, brand_id: int, ad_ids: List[str]):
        """Mark videos as having completed shot detection"""
        brand_status = self.get_brand_status(brand_id)
        for ad_id in ad_ids:
            # Convert to string to ensure JSON compatibility
            ad_id_str = str(ad_id)
            if ad_id_str not in brand_status['shots_completed']:
                brand_status['shots_completed'].append(ad_id_str)
        self._save_status()
    
    def mark_keyframes_complete(self, brand_id: int, ad_ids: List[str]):
        """Mark videos as having completed keyframe extraction"""
        brand_status = self.get_brand_status(brand_id)
        for ad_id in ad_ids:
            # Convert to string to ensure JSON compatibility
            ad_id_str = str(ad_id)
            if ad_id_str not in brand_status['keyframes_completed']:
                brand_status['keyframes_completed'].append(ad_id_str)
        self._save_status()
    
    def mark_embeddings_complete(self, brand_id: int):
        """Mark brand as having completed embeddings"""
        brand_status = self.get_brand_status(brand_id)
        brand_status['embeddings_completed'] = True
        self._save_status()
    
    def get_pending_shots(self, brand_id: int, all_ad_ids: List[str]) -> List[str]:
        """Get ad_ids that need shot detection"""
        brand_status = self.get_brand_status(brand_id)
        completed = set(str(ad_id) for ad_id in brand_status['shots_completed'])
        return [str(ad_id) for ad_id in all_ad_ids if str(ad_id) not in completed]
    
    def get_pending_keyframes(self, brand_id: int, all_ad_ids: List[str]) -> List[str]:
        """Get ad_ids that need keyframe extraction"""
        brand_status = self.get_brand_status(brand_id)
        completed = set(str(ad_id) for ad_id in brand_status['keyframes_completed'])
        return [str(ad_id) for ad_id in all_ad_ids if str(ad_id) not in completed]
    
    def is_embeddings_complete(self, brand_id: int) -> bool:
        """Check if embeddings are complete for brand"""
        brand_status = self.get_brand_status(brand_id)
        return brand_status.get('embeddings_completed', False)
    
    def get_summary(self, brand_id: int) -> Dict:
        """Get summary of brand processing status"""
        brand_status = self.get_brand_status(brand_id)
        return {
            'shots_completed': len(brand_status['shots_completed']),
            'keyframes_completed': len(brand_status['keyframes_completed']),
            'embeddings_completed': brand_status.get('embeddings_completed', False)
        }