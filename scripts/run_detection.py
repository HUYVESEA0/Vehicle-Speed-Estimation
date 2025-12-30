"""
Run Detection - Main script to process videos
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.gpu_manager import GPUManager
from backend.core.video_processor import VideoProcessor
from backend.utils.config_loader import load_config
from backend.utils.logger import setup_logger, log_section, log_device_info

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Vehicle Speed Estimation - AMD GPU Optimized'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input video path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output video path (default: output/videos/output.mp4)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Config file path (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show video in realtime while processing'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(
        name='vehicle_speed',
        log_level=args.log_level,
        log_file=None,
        log_dir='logs'
    )
    
    log_section(logger, "AMD GPU VEHICLE SPEED ESTIMATION")
    
    try:
        # Load config
        logger.info("Loading configuration...")
        config = load_config(args.config)
        logger.info(f"✓ Config loaded from: {args.config}")
        
        # Check input video
        input_str = args.input
        is_stream = str(input_str).isdigit() or str(input_str).startswith(('http', 'rtsp', 'rtmp'))
        
        if not is_stream:
            input_path = Path(input_str)
            if not input_path.exists():
                logger.error(f"Input video not found: {input_path}")
                return 1
        else:
            logger.info(f"Input is detected as stream/URL: {input_str}")
        
        # Set output path
        if args.output is None:
            output_dir = Path('output/videos')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if is_stream:
                output_path = output_dir / "output_stream.mp4"
            else:
                output_path = output_dir / f"output_{input_path.stem}.mp4"
        else:
            output_path = Path(args.output)
        
        logger.info(f"Input: {input_str}")
        logger.info(f"Output: {output_path}")
        
        # Initialize GPU Manager
        logger.info("\nInitializing GPU...")
        gpu_manager = GPUManager(config)
        log_device_info(logger, gpu_manager)
        
        # Initialize Video Processor
        logger.info("Initializing video processor...")
        processor = VideoProcessor(config, gpu_manager)
        
        # Process video
        log_section(logger, "PROCESSING VIDEO")
        
        stats = processor.process_video(
            input_path=str(input_str),
            output_path=str(output_path),
            show_realtime=args.show
        )
        
        # Print statistics
        log_section(logger, "FINAL STATISTICS")
        
        logger.info(f"Processing:")
        logger.info(f"  Frames processed: {stats['frames_processed']}")
        logger.info(f"  Processing time: {stats['processing_time']:.2f}s")
        logger.info(f"  Average FPS: {stats['fps']:.2f}")
        
        logger.info(f"\nDetection:")
        logger.info(f"  Total detections: {stats['total_detections']}")
        logger.info(f"  Active tracks: {stats['tracking']['active_tracks']}")
        logger.info(f"  Total tracks: {stats['tracking']['total_tracks']}")
        
        if stats['speed']['count'] > 0:
            logger.info(f"\nSpeed Statistics:")
            logger.info(f"  Vehicles tracked: {stats['speed']['count']}")
            logger.info(f"  Average speed: {stats['speed']['avg_speed']:.1f} km/h")
            logger.info(f"  Max speed: {stats['speed']['max_speed']:.1f} km/h")
            logger.info(f"  Violations: {stats['speed']['violations']}")
        
        log_section(logger, "PROCESSING COMPLETE")
        logger.info(f"✓ Output saved to: {output_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\nProcessing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
