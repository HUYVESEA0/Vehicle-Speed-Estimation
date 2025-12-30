# 🚗 AMD GPU Vehicle Speed Estimation

High-performance vehicle speed estimation system optimized for **AMD GPUs** using **DirectML**.

## ✨ Features

- 🚀 **AMD GPU Acceleration** - DirectML support for 3-5x faster inference
- 🎯 **YOLOv8/v11 Detection** - State-of-the-art vehicle detection
- 📊 **ByteTrack Tracking** - Accurate multi-object tracking
- 📐 **Perspective Transform** - Calibrated speed estimation
- 💾 **Data Export** - CSV, JSON, Statistics
- 🎨 **Rich Visualization** - Real-time annotated output

## 📋 Requirements

- **Python**: 3.10+
- **GPU**: AMD Radeon (DirectML compatible)
- **OS**: Windows 10/11
- **RAM**: 8GB+ recommended

## 🚀 Quick Start

### 1. Setup (5-10 minutes)

```bash
# Clone or download this project
cd AMD_GPU

# Run automated setup
setup.bat

# Activate environment
venv\Scripts\activate
```

### 2. Verify Installation

```bash
python test_installation.py
```

Expected output:
```
✅ Python 3.10+
✅ All dependencies installed
✅ AMD GPU detected (DirectML)
✅ GPU benchmark passed
```

### 3. Run Demo

```bash
# Test GPU
python test_GPU.py

# Test DirectML features
python test_direct.py
```

## 📦 Project Structure

```
AMD_GPU/
├── backend/              # Core system
│   ├── core/            # Main modules
│   │   ├── gpu_manager.py
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── speed_estimator.py
│   │   └── video_processor.py
│   └── utils/           # Utilities
│       ├── config_loader.py
│       └── logger.py
├── config/              # Configuration
│   └── config.yaml
├── scripts/             # Execution scripts
│   ├── run_detection.py
│   └── calibrate.py
├── data/                # Input videos
├── output/              # Results
├── models/              # AI models
└── logs/                # Log files
```

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
device:
  type: 'dml'            # 'dml' for AMD GPU, 'cpu' for CPU
  batch_size: 8          # Increase for better GPU
  half_precision: true   # FP16 for 2x speed

model:
  name: 'yolov8n'        # or 'yolov11n'
  confidence: 0.4

tracking:
  max_age: 30
  min_hits: 3
```

## 🎯 Usage

### Calibrate Camera

```bash
python scripts/calibrate.py --video data/your_video.mp4
```

### Run Detection

```bash
python scripts/run_detection.py --input data/video.mp4 --show
```

### Export Results

Results saved to:
- `output/videos/` - Annotated video
- `output/data/` - CSV, JSON data
- `logs/` - Execution logs

## 📊 Performance

| GPU Model        | YOLOv8n FPS | YOLOv11n FPS |
|------------------|-------------|--------------|
| RX 6800 XT       | ~60 FPS     | ~55 FPS      |
| RX 7900 XTX      | ~80 FPS     | ~75 FPS      |
| RX 6600          | ~40 FPS     | ~35 FPS      |
| CPU (Ryzen 7)    | ~8 FPS      | ~6 FPS       |

*With batch processing + FP16 + DirectML optimization

## 🔧 Troubleshooting

### DirectML not working

```bash
pip uninstall torch-directml -y
pip install torch-directml --no-cache-dir
```

### NumPy version conflict

```bash
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

### Low FPS

- Increase `batch_size` in config
- Enable `half_precision`
- Use smaller model (yolov8n)

## 📝 License

MIT License - Free to use and modify

## 🙏 Acknowledgments

- Ultralytics YOLOv8/v11
- ByteTrack
- Microsoft DirectML
- Roboflow Supervision

---

**Made with ❤️ for AMD GPU users**
