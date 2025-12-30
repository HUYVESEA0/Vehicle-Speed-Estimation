# 🔧 Hướng Dẫn Cài Đặt Chi Tiết - AMD GPU Support

## ❓ Tại sao torch-directml phải cài riêng?

### 🔴 Vấn đề Dependencies Hell

#### 1. **Xung đột phiên bản PyTorch**
```
torch-directml 0.2.x → yêu cầu torch==2.0.0 hoặc 2.1.0 (cụ thể)
ultralytics >=8.0   → yêu cầu torch>=2.0.0 (bất kỳ)
opencv-python       → yêu cầu numpy<2.0
```

Khi pip install tất cả cùng lúc:
- Pip resolver có thể chọn torch 2.4.1 (mới nhất)
- torch-directml không tương thích → FAILED
- Hoặc pip downgrade torch → ultralytics không hoạt động

#### 2. **NumPy Breaking Changes**
```
NumPy 2.0 có breaking changes (Jun 2024)
torch-directml → chưa support NumPy 2.0
opencv-python mới → có thể pull NumPy 2.0
```

Kết quả: Runtime errors khó debug!

#### 3. **Build Dependencies**
torch-directml cần compile native code:
- C++ compiler
- DirectX 12
- Windows SDK

Nếu cài trước PyTorch → build fails với lỗi "torch not found"

## ✅ Giải pháp

### Option 1: Cài từng bước (RECOMMENDED)

```batch
# 1. NumPy (phải <2.0)
pip install "numpy>=1.24.0,<2.0.0"

# 2. PyTorch CPU (cố định phiên bản)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 3. OpenCV (phải <4.10)
pip install "opencv-python>=4.8.0,<4.10.0"

# 4. Core packages
pip install ultralytics supervision pyyaml pandas

# 5. DirectML (CUỐI CÙNG)
pip install torch-directml --no-cache-dir
```

### Option 2: Dùng setup.bat (EASIEST)

setup.bat đã handle đúng thứ tự:

```batch
setup.bat
```

### Option 3: Requirements constraints (ADVANCED)

Tạo file `constraints.txt`:
```
numpy>=1.24.0,<2.0.0
torch==2.1.0
torchvision==0.16.0
opencv-python>=4.8.0,<4.10.0
```

Sau đó:
```batch
pip install -c constraints.txt -r requirements.txt
pip install torch-directml --no-cache-dir
```

## 📊 So sánh các cách

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| **setup.bat** | ✅ Tự động<br>✅ Đúng thứ tự<br>✅ Dễ dùng | ❌ Windows only |
| **Từng bước** | ✅ Kiểm soát hoàn toàn<br>✅ Dễ debug | ❌ Mất thời gian<br>❌ Dễ nhầm |
| **Constraints** | ✅ Professional<br>✅ Reproducible | ❌ Phức tạp setup |
| **All-in-one** | ✅ Nhanh | ❌ Dễ fail<br>❌ Khó debug |

## 🐛 Xử lý lỗi thường gặp

### Lỗi: "torch-directml requires torch==2.0.0"

**Nguyên nhân:** Đã cài torch 2.4+ trước

**Giải pháp:**
```batch
pip uninstall torch torchvision torch-directml -y
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml --no-cache-dir
```

### Lỗi: "numpy 2.0 is not supported"

**Nguyên nhân:** opencv-python kéo theo numpy 2.0

**Giải pháp:**
```batch
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
pip install "opencv-python>=4.8.0,<4.10.0" --force-reinstall
```

### Lỗi: "Could not build wheels for torch-directml"

**Nguyên nhân:** Thiếu build tools

**Giải pháp:**
1. Cài Visual Studio Build Tools
2. Hoặc dùng pre-built wheel:
```batch
pip install torch-directml --no-cache-dir --only-binary :all:
```

## 💡 Best Practices

### ✅ DO:
- Luôn cài NumPy trước
- Cố định phiên bản PyTorch
- Dùng virtual environment
- Test sau mỗi bước
- Đọc error messages

### ❌ DON'T:
- Cài tất cả cùng lúc từ requirements.txt
- Dùng pip upgrade --all
- Mix conda và pip
- Ignore version warnings
- Skip testing

## 🔬 Kiểm tra sau khi cài

```python
import torch
import torch_directml

# Check DirectML
print(f"DirectML available: {torch_directml.is_available()}")  # Should be True
print(f"Device: {torch_directml.device()}")  # Should show privateuseone:0

# Check NumPy
import numpy as np
print(f"NumPy version: {np.__version__}")  # Should be <2.0

# Test compute
device = torch_directml.device()
x = torch.randn(100, 100, device=device)
y = torch.matmul(x, x)
print(f"GPU compute test: PASSED")
```

## 📚 Tham khảo

- [torch-directml GitHub](https://github.com/microsoft/DirectML)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [NumPy 2.0 Migration](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)

---

**Tóm lại:** Cài riêng torch-directml để tránh dependency hell! 🚀
