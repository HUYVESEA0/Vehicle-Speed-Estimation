"""
Core package initialization
"""

from .gpu_manager import GPUManager
from .detector import VehicleDetector
from .tracker import VehicleTracker
from .speed_estimator import SpeedEstimator
from .video_processor import VideoProcessor

__all__ = [
    'GPUManager',
    'VehicleDetector',
    'VehicleTracker',
    'SpeedEstimator',
    'VideoProcessor'
]
