"""
Vehicle Detector - ONNX Runtime with DirectML Support
Optimized for AMD GPU acceleration via DirectML
"""

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    YOLOv8 vehicle detector optimized for AMD GPU using ONNX Runtime DirectML
    """
    
    # COCO vehicle classes mapping
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, config: Dict[str, Any], gpu_manager):
        """
        Initialize vehicle detector with ONNX Runtime
        """
        self.config = config
        self.gpu_manager = gpu_manager
        
        # Model settings
        model_config = config.get('model', {})
        self.model_name = model_config.get('name', 'yolov8n')
        self.confidence = model_config.get('confidence', 0.4)
        self.iou_threshold = model_config.get('iou_threshold', 0.5)
        self.image_size = model_config.get('image_size', 640)
        
        # Filter classes
        self.filter_classes = model_config.get('classes', list(self.VEHICLE_CLASSES.keys()))
        
        # Initialize ONNX Session
        self.session = None
        self.input_name = None
        self.output_names = None
        self._init_session()
        
        logger.info(f"✓ VehicleDetector (ONNX+DirectML) initialized")
    
    def _init_session(self):
        """Initialize ONNX Runtime session with DirectML"""
        try:
            # Check/Create ONNX model
            onnx_path = Path(f"models/{self.model_name}.onnx")
            if not onnx_path.exists():
                logger.info(f"Exporting {self.model_name} to ONNX format...")
                # Use Ultralytics to export
                model = YOLO(f"{self.model_name}.pt")
                model.export(format='onnx', imgsz=self.image_size, opset=12, dynamic=False)
                # Move exported file to models dir
                exported_path = Path(f"{self.model_name}.onnx")
                if exported_path.exists():
                    exported_path.rename(onnx_path)
            
            # Configure providers
            providers = []
            if self.gpu_manager.is_gpu():
                logger.info("⚡ Enabling DirectML Execution Provider")
                providers.append('DmlExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            # Create Session
            logger.info(f"Loading ONNX model: {onnx_path}")
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            # Get input/output details
            model_inputs = self.session.get_inputs()
            self.input_name = model_inputs[0].name
            self.input_shape = model_inputs[0].shape  # [1, 3, 640, 640]
            
            model_outputs = self.session.get_outputs()
            self.output_names = [output.name for output in model_outputs]
            
            current_providers = self.session.get_providers()
            logger.info(f"Active Providers: {current_providers}")
            if 'DmlExecutionProvider' in current_providers:
                logger.info("✅ DirectML acceleration is ACTIVE")
            else:
                logger.warning("⚠️  DirectML not active, running on CPU")
                
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            raise
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess frame for YOLO inference (Letterbox)
        Returns: (processed_image, ratio, (pad_w, pad_h))
        """
        shape = frame.shape[:2]  # current shape [height, width]
        new_shape = (self.image_size, self.image_size)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add border
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Transforms: HWC to CHW, BGR to RGB
        frame = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        frame = np.ascontiguousarray(frame)
        
        frame = frame.astype(np.float32)
        frame /= 255.0  # 0 - 255 to 0.0 - 1.0
        frame = frame[None]  # Expand dimensions to [1, 3, 640, 640]
        
        return frame, r, (dw, dh)

    def detect(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        return_results: bool = False
    ) -> Dict[str, Any]:
        """
        Detect vehicles in frame
        """
        conf_thres = confidence if confidence is not None else self.confidence
        
        try:
            # 1. Preprocess
            img, ratio, (pad_w, pad_h) = self._preprocess(frame)
            
            # 2. Inference
            outputs = self.session.run(self.output_names, {self.input_name: img})
            
            # 3. Postprocess
            # Output shape: [1, 84, 8400] -> 4 coordinates + 80 classes
            predictions = np.squeeze(outputs[0]).T  # [8400, 84]
            
            # Filter by confidence
            scores = np.max(predictions[:, 4:], axis=1)
            mask = scores > conf_thres
            predictions = predictions[mask]
            
            if len(predictions) == 0:
                return self._empty_detections()
            
            scores = scores[mask]
            class_ids = np.argmax(predictions[:, 4:], axis=1)
            
            # Extract boxes
            boxes = predictions[:, :4]
            input_shape = np.array([self.image_size, self.image_size, self.image_size, self.image_size])
            
            # Convert xywh to xyxy
            # x, y, w, h -> x1, y1, x2, y2
            xyxy = np.copy(boxes)
            xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
            xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
            xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
            xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
            
            # Rescale boxes to original image
            xyxy[:, 0] -= pad_w
            xyxy[:, 1] -= pad_h
            xyxy[:, 2] -= pad_w
            xyxy[:, 3] -= pad_h
            xyxy[:, :4] /= ratio
            
            # Clip boxes to image bounds
            h, w = frame.shape[:2]
            xyxy[:, 0] = np.clip(xyxy[:, 0], 0, w)
            xyxy[:, 1] = np.clip(xyxy[:, 1], 0, h)
            xyxy[:, 2] = np.clip(xyxy[:, 2], 0, w)
            xyxy[:, 3] = np.clip(xyxy[:, 3], 0, h)
            
            # NMS
            indices = cv2.dnn.NMSBoxes(
                bboxes=xyxy.tolist(),
                scores=scores.tolist(),
                score_threshold=conf_thres,
                nms_threshold=self.iou_threshold
            )
            
            if len(indices) == 0:
                return self._empty_detections()
            
            indices = indices.flatten()
            
            # Filter classes
            final_boxes = []
            final_scores = []
            final_class_ids = []
            final_class_names = []
            
            for i in indices:
                cid = class_ids[i]
                if cid in self.filter_classes:
                    final_boxes.append(xyxy[i])
                    final_scores.append(scores[i])
                    final_class_ids.append(cid)
                    final_class_names.append(self.get_class_name(cid))
            
            if not final_boxes:
                return self._empty_detections()
                
            return {
                'boxes': np.array(final_boxes),
                'scores': np.array(final_scores),
                'class_ids': np.array(final_class_ids),
                'class_names': final_class_names,
                'count': len(final_boxes)
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return self._empty_detections()
    
    def _empty_detections(self) -> Dict[str, Any]:
        """Return empty detections"""
        return {
            'boxes': np.array([]),
            'scores': np.array([]),
            'class_ids': np.array([]),
            'class_names': [],
            'count': 0
        }
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID"""
        return self.VEHICLE_CLASSES.get(class_id, 'unknown')
    
    def warmup(self, frame_size: tuple = (640, 640, 3)):
        """Warm up model"""
        logger.info("🔥 Warming up ONNX detector...")
        try:
            dummy_frame = np.zeros(frame_size, dtype=np.uint8)
            for _ in range(3):
                self.detect(dummy_frame)
            logger.info("✓ Detector warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def detect_batch(self, frames, confidence=None):
        # Fallback to single frame detection loop for simplicity in ONNX implementation
        # Can be optimized later if needed
        results = []
        for frame in frames:
            results.append(self.detect(frame, confidence))
        return results

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': self.model_name,
            'backend': 'ONNX Runtime',
            'providers': self.session.get_providers() if self.session else [],
            'device': 'DirectML' if self.gpu_manager.is_gpu() else 'CPU'
        }
