import cv2
import numpy as np
import supervision as sv
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm
import time
from pathlib import Path
import queue
import threading
import csv
from collections import defaultdict

from .detector import VehicleDetector
from .tracker import VehicleTracker
from .speed_estimator import SpeedEstimator
from .gpu_manager import GPUManager
from ..utils.stream_loader import StreamLoader

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, config: Dict[str, Any], gpu_manager: GPUManager):
        self.config = config
        
        # Components
        self.detector = VehicleDetector(config, gpu_manager)
        self.tracker = VehicleTracker(config) # Wraps sv.ByteTrack
        self.speed_estimator = SpeedEstimator(config)
        
        # Settings
        self.skip_frames = config.get('video', {}).get('skip_frames', 0)
        self.show_zone = True
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=10,
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=50
        )
        
        # Polygon Zone (for visualization)
        self.polygon_zone = None
        if self.speed_estimator.calibration_points is not None:
            self.polygon_zone = sv.PolygonZone(
                polygon=self.speed_estimator.calibration_points,
            )
        if self.speed_estimator.calibration_points is not None:
            self.polygon_zone = sv.PolygonZone(
                polygon=self.speed_estimator.calibration_points,
            )
            self.zone_annotator = sv.PolygonZoneAnnotator(
                zone=self.polygon_zone,
                color=sv.Color.WHITE,
                thickness=2,
            )

        # State
        self.tracker_state = {} # track_id -> speed
        self.total_detections_count = 0
        self.counted_ids = set()
        self.counts = defaultdict(int)
        self.violation_ids = set()
        
        # Threaded Snapshot Saver
        self.snapshot_queue = queue.Queue()
        self.snapshot_thread = threading.Thread(target=self._snapshot_worker, daemon=True)
        self.snapshot_thread.start()

    def _snapshot_worker(self):
        """Background thread to save snapshots"""
        while True:
            task = self.snapshot_queue.get()
            if task is None: break
            
            try:
                frame, filename = task
                cv2.imwrite(filename, frame)
                logger.info(f"📸 SAVED: {filename}")
            except Exception as e:
                logger.error(f"Snapshot save failed: {e}")
            finally:
                self.snapshot_queue.task_done()

    def _save_violation_snapshot(self, frame, bbox, track_id, speed, class_name):
        """Queue snapshot for saving and log to CSV"""
        try:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            date_str = time.strftime("%Y-%m-%d")
            timestamp_file = time.strftime("%Y%m%d_%H%M%S")
            
            output_dir = Path(f"output/violations/{date_str}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Crop vehicle
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            pad = 20
            x1 = max(0, x1-pad); y1 = max(0, y1-pad)
            x2 = min(w, x2+pad); y2 = min(h, y2+pad)
            
            crop = frame[y1:y2, x1:x2]
            
            filename = str(output_dir / f"{timestamp_file}_ID{track_id}_{class_name}_{int(speed)}kmh.jpg")
            
            # Put to queue for image saving (slow IO)
            self.snapshot_queue.put((crop.copy(), filename))
            
            # Log to CSV (fast IO)
            log_file = Path("output/violations_log.csv")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_exists = log_file.exists()
            
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Date', 'Track ID', 'Class', 'Speed (km/h)', 'Image Path'])
                writer.writerow([timestamp_str, date_str, track_id, class_name, f"{speed:.1f}", filename])
            
        except Exception as e:
            logger.error(f"Failed to process violation: {e}")

    def process_video(self, input_path: str, output_path: Optional[str] = None, show_realtime: bool = True):
        # Use StreamLoader for YouTube/URL/Webcam support
        cap = StreamLoader(input_path)
        
        fps = cap.fps
        width = cap.width
        height = cap.height
        start_time = time.time()
        
        self.speed_estimator.set_fps(fps)
        if self.polygon_zone:
            self.polygon_zone.triggering_position = sv.Position.BOTTOM_CENTER
        
        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        self.detector.warmup((height, width, 3))
        
        self.detector.warmup((height, width, 3))
        
        frame_idx = 0
        pbar = tqdm(total=cap.total_frames) if cap.total_frames > 0 else None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Resize for performance (Limit height to 480px - SD Quality)
                target_h = 480
                if frame.shape[0] > target_h:
                    scale = target_h / frame.shape[0]
                    w = int(frame.shape[1] * scale)
                    frame = cv2.resize(frame, (w, target_h))
                
                curr_h, curr_w = frame.shape[:2]

                # Update Zone Scale based on Calibrated Resolution
                calib_w, calib_h = self.speed_estimator.calibrated_resolution
                global_scale_x = 1.0
                global_scale_y = 1.0

                if calib_w and calib_h:
                    global_scale_x = curr_w / calib_w
                    global_scale_y = curr_h / calib_h
                    
                    # Update Polygon Zone (vectorized scale)
                    if self.polygon_zone is not None:
                        original_points = self.speed_estimator.calibration_points
                        if original_points is not None:
                            scaled_points = original_points.astype(float) * [global_scale_x, global_scale_y]
                            self.polygon_zone.polygon = scaled_points.astype(int)
                
                # --- PROCESSING ---
                # 1. Detect
                results = self.detector.detect(frame) # Dict
                
                # 2. Track (using sv.ByteTrack wrapper)
                detections = self.tracker.update(results) # sv.Detections
                self.total_detections_count += len(detections)
                
                # Filter interactions with zone
                if self.polygon_zone is not None:
                    is_in_zone = self.polygon_zone.trigger(detections)
                    # Filter detections in zone for speed calculation
                    # Note: We still track everyone, but only calculate speed for those in zone
                    detections_in_zone = detections[is_in_zone]
                else:
                    detections_in_zone = detections

                # 3. Estimate Speed
                labels = []
                if detections.tracker_id is not None:
                    # We iterate over ALL tracks for labels, but check zone logic
                    for i, (tracker_id, bbox, class_id) in enumerate(zip(detections.tracker_id, detections.xyxy, detections.class_id)):
                        
                        # Check if this specific track is in zone
                        in_zone = True
                        if self.polygon_zone is not None:
                            in_zone = is_in_zone[i]
                        
                        # --- COUNTING ---
                        # Only count if in zone (crossed the line effectively)
                        if in_zone and tracker_id not in self.counted_ids:
                            self.counted_ids.add(tracker_id)
                            class_name = self.detector.get_class_name(int(class_id))
                            self.counts[class_name] += 1
                        else:
                            class_name = self.detector.get_class_name(int(class_id))

                        # --- SPEED ---
                        speed_str = ""
                        # Calculate anchor (bottom-center)
                        anchor = np.array([
                            (bbox[0] + bbox[2]) / 2,
                            bbox[3]
                        ])
                        
                        # Scale anchor back to original resolution for accurate speed estimation
                        anchor_original = anchor / [global_scale_x, global_scale_y]
                        
                        # Always estimate to keep history smooth
                        speed = self.speed_estimator.estimate_speed(tracker_id, anchor_original)
                        
                        # Store speed
                        if speed is not None:
                            self.tracker_state[tracker_id] = speed
                        
                        current_speed = self.tracker_state.get(tracker_id, 0)
                        
                        # ONLY show/check speed if in zone
                        if in_zone and current_speed > 5:
                            
                            # Check Violation
                            speed_limit = self.config.get('speed', {}).get('speed_limit', 100)
                            if current_speed > speed_limit:
                                speed_str = f" {int(current_speed)} km/h 🚨"
                                if tracker_id not in self.violation_ids:
                                    self._save_violation_snapshot(frame, bbox, tracker_id, current_speed, class_name)
                                    self.violation_ids.add(tracker_id)
                            else:
                                speed_str = f" {int(current_speed)} km/h"
                        
                        labels.append(f"#{tracker_id}{speed_str}")
                
                # --- VISUALIZATION ---
                annotated_frame = frame.copy()
                
                # Draw Counts
                y_off = 30
                for cls, count in self.counts.items():
                    cv2.putText(annotated_frame, f"{cls}: {count}", (10, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_off += 30
                
                # Draw Zone
                if self.show_zone and self.polygon_zone:
                    annotated_frame = self.zone_annotator.annotate(scene=annotated_frame)
                
                # Draw Traces
                if detections.tracker_id is not None:
                    annotated_frame = self.trace_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections
                    )
                    
                    annotated_frame = self.box_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections
                    )
                    
                    annotated_frame = self.label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections,
                        labels=labels
                    )
                
                # --- OUTPUT ---
                if writer: writer.write(annotated_frame)
                
                if show_realtime:
                    cv2.imshow("Speed Estimation (Supervision)", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): break
                    if key == ord('z'): self.show_zone = not self.show_zone
                
                if pbar: pbar.update(1)
                frame_idx += 1
                
        finally:
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()
            if pbar: pbar.close()
            
        elapsed_time = time.time() - start_time
        avg_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate speed stats
        speeds = list(self.tracker_state.values())
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        
        return {
            'frames_processed': frame_idx,
            'processing_time': elapsed_time,
            'fps': avg_fps,
            'total_detections': getattr(self, 'total_detections_count', 0),
            'tracking': {
                'active_tracks': len(detections.tracker_id) if detections.tracker_id is not None else 0,
                'total_tracks': self.tracker.total_tracks
            },
            'speed': {
                'count': len(speeds),
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'violations': len([s for s in speeds if s > self.config.get('speed', {}).get('speed_limit', 100)])
            }
        }

    def process_stream_generator(self, input_source):
        """Generator for Streamlit App - Yields (frame, stats)"""
        cap = StreamLoader(input_source)
        fps = cap.fps
        width = cap.width
        height = cap.height
        
        self.speed_estimator.set_fps(fps)
        self.detector.warmup((height, width, 3))
        
        frame_idx = 0
        start_time = time.time()
        self.last_resolution_wh = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Resize (480p)
                target_h = 480
                if frame.shape[0] > target_h:
                    scale = target_h / frame.shape[0]
                    w = int(frame.shape[1] * scale)
                    frame = cv2.resize(frame, (w, target_h))
                
                # Zone Scale Logic
                curr_h, curr_w = frame.shape[:2]
                calib_w, calib_h = self.speed_estimator.calibrated_resolution
                global_scale_x = 1.0; global_scale_y = 1.0
                
                if (curr_w, curr_h) != self.last_resolution_wh:
                    self.last_resolution_wh = (curr_w, curr_h)
                    
                    if calib_w and calib_h and self.polygon_zone:
                        orig_pts = self.speed_estimator.calibration_points
                        if orig_pts is not None:
                            # Recalculate polygon based on new scale
                            global_scale_x = curr_w / calib_w
                            global_scale_y = curr_h / calib_h
                            scaled_polygon = (orig_pts.astype(float) * [global_scale_x, global_scale_y]).astype(int)
                            
                            # Re-initialize PolygonZone
                            # VERSION COMPATIBILITY FIX: supervision 0.27.0 does not accept resolution in init
                            # We must manually set it to valid full-frame resolution to avoid IndexError
                            self.polygon_zone = sv.PolygonZone(
                                polygon=scaled_polygon, 
                                triggering_anchors=[sv.Position.BOTTOM_CENTER]
                            )
                            
                            # FORCE FULL FRAME MASK (Fixes IndexError: index out of bounds)
                            self.polygon_zone.frame_resolution_wh = (curr_w, curr_h)
                            mask = np.zeros((curr_h, curr_w), dtype=np.uint8)
                            cv2.fillPoly(mask, [scaled_polygon], 1)
                            self.polygon_zone.mask = mask.astype(bool)

                            # Re-bind annotator
                            self.zone_annotator = sv.PolygonZoneAnnotator(
                                zone=self.polygon_zone, 
                                color=sv.Color.RED,
                                thickness=2
                            )
                        
                # Processing
                results = self.detector.detect(frame)
                detections = self.tracker.update(results)
                
                # Zone Trigger with Safety Catch
                is_in_zone = [True] * len(detections)
                if self.polygon_zone:
                    try:
                        is_in_zone = self.polygon_zone.trigger(detections)
                    except IndexError:
                        # Fallback if zone mask mismatch occurs
                        is_in_zone = [False] * len(detections)
                
                labels = []
                speed_limit = self.config.get('speed', {}).get('speed_limit', 100)
                
                if detections.tracker_id is not None:
                    for i, (tracker_id, bbox, class_id) in enumerate(zip(detections.tracker_id, detections.xyxy, detections.class_id)):
                        in_zone = is_in_zone[i] if self.polygon_zone else True
                        
                        # Count
                        if in_zone and tracker_id not in self.counted_ids:
                            self.counted_ids.add(tracker_id)
                            cls = self.detector.get_class_name(int(class_id))
                            self.counts[cls] += 1
                        
                        # Speed
                        anchor = np.array([(bbox[0]+bbox[2])/2, bbox[3]])
                        anchor_orig = anchor / [global_scale_x, global_scale_y]
                        speed = self.speed_estimator.estimate_speed(tracker_id, anchor_orig)
                        
                        if speed: self.tracker_state[tracker_id] = speed
                        curr_speed = self.tracker_state.get(tracker_id, 0)
                        
                        speed_str = ""
                        if in_zone and curr_speed > 5:
                            if curr_speed > speed_limit:
                                speed_str = f" {int(curr_speed)}km/h" # No alarm text to keep clean
                                if tracker_id not in self.violation_ids:
                                    cls = self.detector.get_class_name(int(class_id))
                                    self._save_violation_snapshot(frame, bbox, tracker_id, curr_speed, cls)
                                    self.violation_ids.add(tracker_id)
                            else:
                                speed_str = f" {int(curr_speed)}km/h"
                        labels.append(f"#{tracker_id}{speed_str}")

                # Visualize directly
                annotated_frame = frame.copy()
                if self.show_zone and self.polygon_zone:
                    annotated_frame = self.zone_annotator.annotate(scene=annotated_frame)
                
                if detections.tracker_id is not None:
                    annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                
                frame_idx += 1
                yield annotated_frame, {
                    'fps': frame_idx / (time.time() - start_time),
                    'counts': self.counts,
                    'violations': len(self.violation_ids)
                }
                
        finally:
            cap.release()

    def get_statistics(self):
        return {}
