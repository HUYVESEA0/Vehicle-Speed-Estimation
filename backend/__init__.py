"""
Backend package initialization
"""

from .core.gpu_manager import GPUManager
from .core.detector import VehicleDetector
from .core.tracker import VehicleTracker
from .core.speed_estimator import SpeedEstimator
from .core.video_processor import VideoProcessor
from .utils.config_loader import load_config
from .utils.logger import setup_logger

__all__ = [
    'GPUManager',
    'VehicleDetector',
    'VehicleTracker', 
    'SpeedEstimator',
    'VideoProcessor',
    'load_config',
    'setup_logger'
]

__version__ = '1.0.0'
