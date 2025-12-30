"""
Vehicle Tracker using Roboflow Supervision
Wraps sv.ByteTrack for robust multi-object tracking
"""

import supervision as sv
from typing import Dict, Any, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VehicleTracker:
    """
    Wrapper for supervision.ByteTrack
    """
    
    def __init__(self, config: Dict[str, Any]):
        tracking_config = config.get('tracking', {})
        
        # Initialize ByteTrack from supervision
        # frame_rate is important for internal buffer
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,  # Keep lost tracks for 30 frames
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        
        self.total_tracks = 0
        logger.info("✓ Supervision ByteTrack initialized")

    def update(self, detections: Dict[str, Any]) -> sv.Detections:
        """
        Update tracker with detections
        Args:
            detections: Dictionary with 'xyxy', 'confidence', 'class_id'
            
        Returns:
            sv.Detections object with tracker_id
        """
        # Convert our dict to sv.Detections
        if detections['count'] == 0:
            return sv.Detections.empty()

        # Parse boxes (ensure xyxy format)
        boxes = detections['boxes']
        scores = detections['scores']
        class_ids = detections['class_ids']
        
        # Create sv.Detections
        sv_detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids
        )
        
        # Update tracker
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        # Update stats
        current_ids = tracked_detections.tracker_id
        if len(current_ids) > 0:
            max_id = np.max(current_ids)
            if max_id > self.total_tracks:
                self.total_tracks = max_id
                
        return tracked_detections

    def reset(self):
        self.tracker.reset()
        self.total_tracks = 0
