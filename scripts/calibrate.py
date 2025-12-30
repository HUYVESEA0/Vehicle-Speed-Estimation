"""
Camera Calibration Script
Calibrate camera for accurate speed measurement
"""

import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
import yaml

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.logger import setup_logger
from backend.utils.stream_loader import StreamLoader

# Global variables
points = []
frame = None
window_name = "Camera Calibration"


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for selecting points"""
    global points, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            
            # Draw point (on a copy, handled in main loop)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Camera Calibration for Speed Estimation'
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Input video path (URL, YouTube, Webcam ID, or File)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='config/calibration.yaml',
        help='Output calibration file'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    global points, frame
    
    args = parse_args()
    logger = setup_logger(log_level='INFO')
    
    logger.info("=" * 70)
    logger.info("  CAMERA CALIBRATION")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Instructions:")
    logger.info("  1. Click 4 corners of the measurement area (rectangle)")
    logger.info("  2. Click in order: top-left, top-right, bottom-right, bottom-left")
    logger.info("  3. Press 's' to save calibration")
    logger.info("  4. Press 'r' to reset points")
    logger.info("  5. Press 'q' to quit without saving")
    logger.info("")
    
    # Open video using StreamLoader
    try:
        cap = StreamLoader(args.video)
    except Exception as e:
        logger.error(f"Could not open source: {args.video} - {e}")
        return 1
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        logger.error("Could not read frame from source")
        return 1
        
    # Keep original frame for reset
    original_frame = frame.copy()
    
    # Display frame
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    logger.info("Click on the video to select 4 corners...")
    
    while True:
        # We redraw on original frame every loop to support interactive drawing
        display_frame = frame.copy()
        
        for i, pt in enumerate(points):
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            if i > 0:
                cv2.line(display_frame, points[i-1], pt, (0, 255, 0), 2)
        
        if len(points) == 4:
            cv2.line(display_frame, points[3], points[0], (0, 255, 0), 2)
            
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(20) & 0xFF
        
        # Reset
        if key == ord('r'):
            points = []
            frame = original_frame.copy()
            logger.info("Reset points")
        
        # Save
        elif key == ord('s'):
            if len(points) != 4:
                logger.warning(f"Need 4 points, only have {len(points)}")
                continue
            
            # Get real-world dimensions
            logger.info("\nEnter real-world dimensions:")
            try:
                width_m = float(input("  Width (meters): "))
                height_m = float(input("  Height (meters): "))
            except ValueError:
                logger.error("Invalid input")
                continue
            
            # Calculate perspective transform
            src_points = np.float32(points)
            dst_points = np.float32([
                [0, 0],
                [width_m * 100, 0],
                [width_m * 100, height_m * 100],
                [0, height_m * 100]
            ])
            
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            pixels_per_meter = 100  # We scale to 100 pixels per meter
            
            # Save calibration
            calibration = {
                'points': [[float(p[0]), float(p[1])] for p in points],
                'width_meters': float(width_m),
                'height_meters': float(height_m),
                'transform_matrix': matrix.tolist(),
                'pixels_per_meter': pixels_per_meter,
                'frame_width': frame.shape[1],
                'frame_height': frame.shape[0]
            }
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(calibration, f, default_flow_style=False)
            
            logger.info(f"\n✓ Calibration saved to: {output_path}")
            logger.info(f"  Area: {width_m:.2f}m x {height_m:.2f}m")
            logger.info(f"  Pixels per meter: {pixels_per_meter}")
            
            break
        
        # Quit
        elif key == ord('q'):
            logger.info("Calibration cancelled")
            break
    
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
