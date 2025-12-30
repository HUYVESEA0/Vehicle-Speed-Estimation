"""
Speed Estimator using View Transformer
Calculates vehicle speed based on transformed coordinates
"""

import numpy as np
from typing import Dict, Any, Optional, Deque
from collections import deque, defaultdict
import yaml
from pathlib import Path
import logging
from ..utils.view_transformer import ViewTransformer

logger = logging.getLogger(__name__)

class SpeedEstimator:
    def __init__(self, config: Dict[str, Any]):
        speed_config = config.get('speed', {})
        self.fps = 30
        
        # Coordinates history: track_id -> deque of (x, y, timestamp)
        self.coordinates = defaultdict(lambda: deque(maxlen=30))
        
        # Calibration
        self.calibration_file = speed_config.get('calibration_file', 'config/calibration.yaml')
        self.view_transformer = None
        self.calibration_points = None
        
        # Correction
        self.correction_factor = speed_config.get('correction_factor', 2.0)
        
        self._load_calibration()
        logger.info("✓ SpeedEstimator (Supervision Style) initialized")

    def set_fps(self, fps: float):
        self.fps = fps

    def _load_calibration(self):
        calib_path = Path(self.calibration_file)
        if calib_path.exists():
            try:
                with open(calib_path, 'r') as f:
                    calib = yaml.safe_load(f)
                
                points = np.array(calib['points'])
                self.calibration_points = points
                
                # Setup ViewTransformer
                # Target: Rectangle based on real world dimensions
                width = calib['width_meters']
                height = calib['height_meters']
                
                # Create target points corresponding to source points
                # Assuming order: TL, TR, BR, BL
                target = np.array([
                    [0, 0],
                    [width, 0],
                    [width, height],
                    [0, height]
                ])
                
                self.calibrated_resolution = (calib.get('frame_width'), calib.get('frame_height'))
                self.view_transformer = ViewTransformer(points, target)
                logger.info(f"✓ ViewTransformer loaded: {width}m x {height}m")
                
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")
                self.view_transformer = None

    def update(self, detections: Any): # sv.Detections
        # This method is optional if we process track by track in VideoProcessor
        pass

    def estimate_speed(self, track_id: int, anchor_point: np.ndarray) -> Optional[float]:
        """
        Estimate speed for a single track
        Args:
            track_id: ID of the track
            anchor_point: Point on screen [x, y] (usually bottom-center)
        """
        if self.view_transformer is None:
            return None
            
        # 1. Transform point to real-world coordinates
        transformed_point = self.view_transformer.transform_points(np.array([anchor_point]))[0]
        
        # 2. Store point
        self.coordinates[track_id].append(transformed_point)
        
        # 3. Calculate speed if enough history
        if len(self.coordinates[track_id]) < 5:
            return None
            
        # Get start and end points in window
        # We calculate speed over the last 0.5 seconds (approx 15 frames)
        history = self.coordinates[track_id]
        distance = np.linalg.norm(history[-1] - history[0])
        time_seconds = len(history) / self.fps
        
        if time_seconds == 0: return None
        
        speed_mps = distance / time_seconds
        speed_kmh = speed_mps * 3.6 * self.correction_factor
        
        if speed_kmh > 300: # Filter noise
            return None
            
        return speed_kmh
