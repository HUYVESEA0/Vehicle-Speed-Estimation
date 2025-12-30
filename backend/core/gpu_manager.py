"""
GPU Manager - AMD DirectML Optimization
Manages GPU device selection and optimization for AMD GPUs
"""

import torch
import platform
import psutil
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Manages GPU device (AMD/NVIDIA/CPU) with DirectML optimization
    Automatically selects best available device
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('device', {})
        self.device = None
        self.device_name = "CPU"
        self.is_gpu_available = False
        self.device_type = self.config.get('type', 'dml')  # 'dml', 'cuda', 'cpu'
        
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize the best available device"""
        try:
            device_type = self.device_type.lower()
            
            # DirectML (AMD GPU)
            if device_type == 'dml':
                self._initialize_directml()
            
            # CUDA (NVIDIA GPU)
            elif device_type == 'cuda':
                self._initialize_cuda()
            
            # CPU
            elif device_type == 'cpu':
                self._fallback_to_cpu()
            
            # Auto-detect
            else:
                logger.info("🔍 Auto-detecting best device...")
                if not self._initialize_directml():
                    if not self._initialize_cuda():
                        self._fallback_to_cpu()
                        
        except Exception as e:
            logger.error(f"❌ Device initialization failed: {e}")
            self._fallback_to_cpu()
    
    def _initialize_directml(self) -> bool:
        """Initialize DirectML for AMD GPU"""
        try:
            import torch_directml
            
            if torch_directml.is_available():
                self.device = torch_directml.device()
                self.device_name = f"AMD GPU (DirectML) - {self.get_gpu_name()}"
                self.is_gpu_available = True
                logger.info(f"✅ {self.device_name} initialized!")
                logger.info(f"   Device: {self.device}")
                return True
            else:
                logger.warning("⚠️  DirectML not available")
                return False
                
        except ImportError:
            logger.warning("⚠️  torch-directml not installed")
            logger.info("   Install with: pip install torch-directml")
            return False
        except Exception as e:
            logger.warning(f"⚠️  DirectML initialization failed: {e}")
            return False
    
    def _initialize_cuda(self) -> bool:
        """Initialize CUDA for NVIDIA GPU"""
        try:
            if torch.cuda.is_available():
                device_id = self.config.get('device_id', 0)
                self.device = torch.device(f'cuda:{device_id}')
                self.device_name = f"NVIDIA GPU - {torch.cuda.get_device_name(device_id)}"
                self.is_gpu_available = True
                logger.info(f"✅ {self.device_name} initialized!")
                return True
            else:
                logger.warning("⚠️  CUDA not available")
                return False
        except Exception as e:
            logger.warning(f"⚠️  CUDA initialization failed: {e}")
            return False
    
    def _fallback_to_cpu(self):
        """Fallback to CPU if GPU not available"""
        self.device = torch.device('cpu')
        self.device_name = f"CPU - {platform.processor()}"
        self.is_gpu_available = False
        logger.info(f"💻 Using {self.device_name}")
    
    def get_gpu_name(self) -> str:
        """Get GPU name (for AMD, returns generic name)"""
        try:
            # Try to get AMD GPU name from system
            if platform.system() == 'Windows':
                import subprocess
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        line = line.strip()
                        if line and 'AMD' in line.upper():
                            return line
        except:
            pass
        
        return "AMD Radeon"
    
    def get_device(self) -> torch.device:
        """Get the current device"""
        return self.device
    
    def get_device_name(self) -> str:
        """Get the device name"""
        return self.device_name
    
    def is_gpu(self) -> bool:
        """Check if GPU is being used"""
        return self.is_gpu_available
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        info = {
            'device': str(self.device),
            'device_name': self.device_name,
            'device_type': self.device_type,
            'is_gpu': self.is_gpu_available,
            'platform': platform.system(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'pytorch_version': torch.__version__,
        }
        
        # Add DirectML version if available
        if self.is_gpu_available and self.device_type == 'dml':
            try:
                import torch_directml
                try:
                    info['directml_version'] = torch_directml.__version__
                except AttributeError:
                    info['directml_version'] = 'N/A (installed)'
            except:
                pass
        
        return info
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage"""
        memory_info = {
            'ram_used_gb': round(psutil.virtual_memory().used / (1024**3), 2),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        }
        
        if self.is_gpu_available:
            if self.device_type == 'cuda':
                try:
                    memory_info['gpu_used_gb'] = round(torch.cuda.memory_allocated() / (1024**3), 2)
                    memory_info['gpu_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                except:
                    pass
            elif self.device_type == 'dml':
                memory_info['gpu_status'] = 'active'
                memory_info['gpu_backend'] = 'DirectML'
        
        return memory_info
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on device"""
        if self.is_gpu_available:
            return self.config.get('batch_size', 8)
        else:
            return 1  # CPU processes 1 frame at a time
    
    def warmup(self, model, input_shape=(640, 640, 3)):
        """Warm up GPU with dummy inference"""
        if not self.config.get('warmup', True):
            return
        
        logger.info("🔥 Warming up GPU...")
        try:
            import numpy as np
            dummy_frame = np.zeros(input_shape, dtype=np.uint8)
            
            for _ in range(3):
                _ = model(dummy_frame, verbose=False)
            
            logger.info("✓ Warmup complete")
        except Exception as e:
            logger.warning(f"⚠️  Warmup failed: {e}")
    
    def clear_cache(self):
        """Clear GPU cache to free memory"""
        try:
            if self.is_gpu_available:
                if self.device_type == 'cuda':
                    torch.cuda.empty_cache()
                else:
                    # DirectML: trigger garbage collection
                    import gc
                    gc.collect()
                logger.debug("🧹 Cache cleared")
        except Exception as e:
            logger.warning(f"⚠️  Cache clearing warning: {e}")
    
    def __repr__(self):
        return f"GPUManager(device={self.device}, type={self.device_type}, gpu={self.is_gpu_available})"


# Singleton instance
_gpu_manager_instance: Optional[GPUManager] = None


def get_gpu_manager(config: Optional[Dict[str, Any]] = None) -> GPUManager:
    """Get or create GPU manager singleton"""
    global _gpu_manager_instance
    
    if _gpu_manager_instance is None:
        if config is None:
            raise ValueError("Config required for first initialization")
        _gpu_manager_instance = GPUManager(config)
    
    return _gpu_manager_instance


def reset_gpu_manager():
    """Reset GPU manager (for testing)"""
    global _gpu_manager_instance
    _gpu_manager_instance = None
