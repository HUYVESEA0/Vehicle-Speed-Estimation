import cv2
import numpy as np
from typing import List, Tuple

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Initialize view transformer
        
        Args:
            source: Source points (polygon on screen)
            target: Target points (rectangle in real world)
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from source to target coordinate system
        
        Args:
            points: Array of points [N, 2]
            
        Returns:
            Transformed points [N, 2]
        """
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
