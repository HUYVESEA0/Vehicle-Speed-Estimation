"""
Installation Verification Script
Test all dependencies and GPU functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_check(name, status, details=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {name:<30} {'OK' if status else 'FAILED'}")
    if details:
        print(f"   {details}")

def main():
    print_header("AMD GPU VEHICLE SPEED ESTIMATION - INSTALLATION CHECK")
    
    all_ok = True
    
    # 1. Python Version
    print_header("1. Python Version")
    import sys
    python_ok = sys.version_info >= (3, 10)
    print_check("Python 3.10+", python_ok, f"Version: {sys.version.split()[0]}")
    all_ok &= python_ok
    
    # 2. Core Dependencies
    print_header("2. Core Dependencies")
    
    deps = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'ultralytics': 'Ultralytics',
        'supervision': 'Supervision',
        'yaml': 'PyYAML',
        'colorama': 'Colorama',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
    }
    
    for module, name in deps.items():
        try:
            __import__(module)
            version = ""
            try:
                imported = sys.modules[module]
                if hasattr(imported, '__version__'):
                    version = f"v{imported.__version__}"
            except:
                pass
            print_check(name, True, version)
        except ImportError:
            print_check(name, False, "Not installed")
            all_ok = False
    
    # 3. GPU Support
    print_header("3. GPU Support")
    
    import torch
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print_check("CUDA (NVIDIA)", cuda_available)
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Check DirectML
    directml_available = False
    try:
        import torch_directml
        directml_available = torch_directml.is_available()
        print_check("DirectML (AMD)", directml_available)
        if directml_available:
            device = torch_directml.device()
            print(f"   Device: {device}")
            
            # Try to get GPU name
            try:
                from backend.core.gpu_manager import GPUManager
                dummy_config = {'device': {'type': 'dml'}}
                gpu_mgr = GPUManager(dummy_config)
                print(f"   GPU: {gpu_mgr.get_gpu_name()}")
            except:
                pass
    except ImportError:
        print_check("DirectML (AMD)", False, "torch-directml not installed")
    
    # 4. Project Structure
    print_header("4. Project Structure")
    
    required_dirs = [
        'backend/core',
        'backend/utils',
        'config',
        'scripts',
        'data',
        'output',
        'logs',
        'models'
    ]
    
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        print_check(dir_path, exists)
        all_ok &= exists
    
    # 5. Configuration
    print_header("5. Configuration")
    
    config_exists = Path('config/config.yaml').exists()
    print_check("config.yaml", config_exists)
    
    if config_exists:
        try:
            from backend.utils.config_loader import load_config
            config = load_config('config/config.yaml')
            print_check("Config valid", True)
        except Exception as e:
            print_check("Config valid", False, str(e))
            all_ok = False
    else:
        all_ok = False
    
    # 6. Backend Modules
    print_header("6. Backend Modules")
    
    modules = {
        'backend.core.gpu_manager': 'GPU Manager',
        'backend.utils.config_loader': 'Config Loader',
        'backend.utils.logger': 'Logger',
    }
    
    for module, name in modules.items():
        try:
            __import__(module)
            print_check(name, True)
        except Exception as e:
            print_check(name, False, str(e))
            all_ok = False
    
    # 7. GPU Benchmark (if available)
    if directml_available or cuda_available:
        print_header("7. GPU Benchmark")
        
        try:
            print("Running quick GPU test...")
            
            # Simple matrix multiplication test
            import time
            size = 2000
            
            if directml_available:
                import torch_directml
                device = torch_directml.device()
            else:
                device = torch.device('cuda:0')
            
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warmup
            _ = torch.matmul(a, b)
            
            # Benchmark
            start = time.time()
            c = torch.matmul(a, b)
            _ = c.sum().item()  # Force synchronization
            elapsed = time.time() - start
            
            gflops = (size * size * size * 2) / elapsed / 1e9
            
            print_check("GPU Compute", True, f"{elapsed:.3f}s | {gflops:.2f} GFLOPS")
            
        except Exception as e:
            print_check("GPU Benchmark", False, str(e))
    
    # Final Result
    print_header("Installation Status")
    
    if all_ok:
        print("✅ ✅ ✅  ALL CHECKS PASSED!  ✅ ✅ ✅")
        print("\nYou're ready to use the system!")
        print("\nNext steps:")
        print("  1. Place your video in data/ folder")
        print("  2. Run: python scripts/run_detection.py --input data/your_video.mp4")
    else:
        print("❌ ❌ ❌  SOME CHECKS FAILED  ❌ ❌ ❌")
        print("\nPlease fix the issues above before proceeding.")
        print("\nFor GPU support:")
        print("  pip install torch-directml --no-cache-dir")
    
    print("=" * 70)
    print()
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
