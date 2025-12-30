"""
Configuration Loader
Load and validate YAML configuration files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Config loaded from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML format: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✓ Config saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['device', 'model', 'video']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate device config
    device_config = config['device']
    if 'type' not in device_config:
        raise ValueError("Missing 'type' in device config")
    
    valid_device_types = ['dml', 'cuda', 'cpu', 'auto']
    if device_config['type'] not in valid_device_types:
        raise ValueError(f"Invalid device type. Must be one of: {valid_device_types}")
    
    # Validate model config
    model_config = config['model']
    if 'name' not in model_config:
        raise ValueError("Missing 'name' in model config")
    
    logger.info("✓ Config validation passed")
    return True


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'device.batch_size')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Example:
        >>> get_config_value(config, 'device.batch_size', 4)
        8
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override taking precedence
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
