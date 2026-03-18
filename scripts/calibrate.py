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
        else:
            print("Already have 4 points. Press 'r' to reset.")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to remove last point
        if points:
            removed = points.pop()
            print(f"Removed point: {removed}")
        else:
            print("No points to remove.")


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
    logger.info("  1. Click 4 corners of the measurement area to form a rectangle")
    logger.info("  2. Click in order: top-left, top-right, bottom-right, bottom-left")
    logger.info("  3. Left click: Add point")
    logger.info("  4. Right click: Remove last point")
    logger.info("  5. Press 's' to save calibration (will ask for real dimensions)")
    logger.info("  6. Press 'r' to reset all points")
    logger.info("  7. Press 'q' to quit without saving")
    logger.info("")
    logger.info("Tips:")
    logger.info("  • Select a rectangular area where vehicles will be measured")
    logger.info("  • Choose points that form a clear perspective view")
    logger.info("  • Avoid points too close to image edges")
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

                # Validate dimensions
                if width_m <= 0 or height_m <= 0:
                    logger.error("Dimensions must be positive")
                    continue

                # Check if points form a valid quadrilateral
                points_array = np.array(points)
                if len(np.unique(points_array, axis=0)) != 4:
                    logger.error("Points must be unique")
                    continue

            except ValueError:
                logger.error("Invalid input - please enter numbers only")
                continue
            except KeyboardInterrupt:
                logger.info("Calibration cancelled by user")
                break

            try:
                # Calculate perspective transform
                src_points = np.float32(points)
                dst_points = np.float32([
                    [0, 0],
                    [width_m * 100, 0],
                    [width_m * 100, height_m * 100],
                    [0, height_m * 100]
                ])

                matrix = cv2.getPerspectiveTransform(src_points, dst_points)

                # Validate matrix
                if matrix is None:
                    logger.error("Cannot compute perspective transform - check point selection")
                    continue

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

            except Exception as e:
                logger.error(f"Error saving calibration: {e}")
                logger.info("Please try again or check point selection")
        
        # Quit
        elif key == ord('q'):
            logger.info("Calibration cancelled")
            break
    
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
