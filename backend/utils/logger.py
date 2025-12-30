"""
Logger Setup with Color Support
Configures logging with console and file output
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from colorama import init, Fore, Style

# Initialize colorama for Windows color support
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support"""
    
    # Color mapping for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            record.levelname = colored_levelname
        
        return super().format(record)


def setup_logger(
    name: str = 'vehicle_speed',
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_dir: str = 'logs',
    colored: bool = True
) -> logging.Logger:
    """
    Setup logger with console and optional file output
    
    Args:
        name: Logger name
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Log file name (None for auto-generated)
        log_dir: Directory to store logs
        colored: Use colored output in console
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if colored:
        console_format = ColoredFormatter(
            fmt='%(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            fmt='%(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file or log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(
            log_path / log_file,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        logger.info(f"📝 Logging to file: {log_path / log_file}")
    
    return logger


def get_logger(name: str = 'vehicle_speed') -> logging.Logger:
    """Get existing logger or create new one with default settings"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def log_separator(logger: logging.Logger, char: str = '=', length: int = 80):
    """Log a separator line"""
    logger.info(char * length)


def log_section(logger: logging.Logger, title: str, char: str = '=', length: int = 80):
    """Log a section header"""
    logger.info(char * length)
    logger.info(f"  {title}")
    logger.info(char * length)


def log_config(logger: logging.Logger, config: dict, title: str = "Configuration"):
    """Log configuration in a readable format"""
    log_section(logger, title)
    
    def log_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{'  ' * indent}{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info(f"{'  ' * indent}{key}: {value}")
    
    log_dict(config)
    log_separator(logger)


def log_device_info(logger: logging.Logger, gpu_manager):
    """Log device information"""
    info = gpu_manager.get_device_info()
    
    log_section(logger, "Device Information")
    logger.info(f"  Device: {info['device']}")
    logger.info(f"  Name: {info['device_name']}")
    logger.info(f"  Type: {info.get('device_type', 'N/A')}")
    logger.info(f"  GPU: {'✓ Enabled' if info['is_gpu'] else '✗ Disabled'}")
    logger.info(f"  Platform: {info['platform']}")
    logger.info(f"  CPU: {info['cpu_count']} cores / {info['cpu_threads']} threads")
    logger.info(f"  RAM: {info['ram_total_gb']} GB")
    logger.info(f"  PyTorch: {info['pytorch_version']}")
    
    if 'directml_version' in info:
        logger.info(f"  DirectML: {info['directml_version']}")
    
    log_separator(logger)


def log_progress(logger: logging.Logger, current: int, total: int, message: str = "Progress"):
    """Log progress bar"""
    percent = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    logger.info(f"{message}: [{bar}] {current}/{total} ({percent:.1f}%)")
