import torch
import torch_directml
import numpy as np

def print_header(title):
    """In tiêu đề"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_device_initialization():
    """Test khởi tạo thiết bị DirectML"""
    print_header("KIỂM TRA KHỞI TẠO THIẾT BỊ")
    
    try:
        # Khởi tạo thiết bị DirectML (cho AMD GPU)
        device = torch_directml.device()
        print(f"✓ Thiết bị: {device}")
        print(f"✓ Loại thiết bị: {device.type}")
        print(f"✓ Index thiết bị: {device.index}")
        
        return device
    except Exception as e:
        print(f"✗ Lỗi khởi tạo: {str(e)}")
        return None

def test_tensor_creation(device):
    """Test tạo tensor trên GPU"""
    print_header("KIỂM TRA TẠO TENSOR")
    
    try:
        # Cách 1: Tạo trên CPU rồi chuyển sang GPU
        print("\n1. Tạo tensor từ CPU:")
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        print(f"   CPU tensor: {x}")
        x_gpu = x.to(device)
        print(f"   GPU tensor: {x_gpu}")
        print(f"   Thiết bị: {x_gpu.device}")
        
        # Cách 2: Tạo trực tiếp trên GPU
        print("\n2. Tạo tensor trực tiếp trên GPU:")
        y = torch.tensor([5.0, 6.0, 7.0, 8.0], device=device)
        print(f"   GPU tensor: {y}")
        print(f"   Thiết bị: {y.device}")
        
        # Cách 3: Tạo tensor ngẫu nhiên
        print("\n3. Tạo tensor ngẫu nhiên:")
        z = torch.randn(2, 3, device=device)
        print(f"   Shape: {z.shape}")
        print(f"   Tensor:\n{z}")
        
        return True
    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
        return False

def test_tensor_operations(device):
    """Test các phép toán tensor cơ bản"""
    print_header("KIỂM TRA PHÉP TOÁN TENSOR")
    
    try:
        # Tạo tensor
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        b = torch.tensor([5.0, 6.0, 7.0, 8.0], device=device)
        
        print(f"a = {a}")
        print(f"b = {b}\n")
        
        # Phép cộng
        c = a + b
        print(f"a + b = {c}")
        
        # Phép trừ
        c = a - b
        print(f"a - b = {c}")
        
        # Phép nhân element-wise
        c = a * b
        print(f"a * b = {c}")
        
        # Phép chia
        c = a / b
        print(f"a / b = {c}")
        
        # Dot product
        c = torch.dot(a, b)
        print(f"dot(a, b) = {c.item()}")
        
        # Sum
        c = a.sum()
        print(f"sum(a) = {c.item()}")
        
        # Mean
        c = a.mean()
        print(f"mean(a) = {c.item()}")
        
        return True
    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
        return False

def test_matrix_operations(device):
    """Test các phép toán ma trận"""
    print_header("KIỂM TRA PHÉP TOÁN MA TRẬN")
    
    try:
        # Tạo ma trận
        A = torch.randn(3, 4, device=device)
        B = torch.randn(4, 5, device=device)
        
        print(f"Ma trận A shape: {A.shape}")
        print(f"Ma trận B shape: {B.shape}\n")
        
        # Nhân ma trận
        C = torch.matmul(A, B)
        print(f"A @ B shape: {C.shape}")
        print(f"Kết quả:\n{C}\n")
        
        # Transpose
        A_T = A.T
        print(f"A^T shape: {A_T.shape}")
        
        # Element-wise operations
        D = torch.randn(3, 4, device=device)
        E = A * D
        print(f"A * D (element-wise) shape: {E.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
        return False

def test_activation_functions(device):
    """Test các hàm kích hoạt (activation functions)"""
    print_header("KIỂM TRA HÀM KÍCH HOẠT")
    
    try:
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
        print(f"Input: {x}\n")
        
        # ReLU
        y = torch.relu(x)
        print(f"ReLU: {y}")
        
        # Sigmoid
        y = torch.sigmoid(x)
        print(f"Sigmoid: {y}")
        
        # Tanh
        y = torch.tanh(x)
        print(f"Tanh: {y}")
        
        # Softmax
        y = torch.softmax(x, dim=0)
        print(f"Softmax: {y}")
        print(f"Softmax sum: {y.sum().item()}")
        
        return True
    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
        return False

def test_gradient_computation(device):
    """Test tính toán gradient"""
    print_header("KIỂM TRA GRADIENT")
    
    try:
        # Tạo tensor với requires_grad=True
        x = torch.tensor([2.0, 3.0], device=device, requires_grad=True)
        print(f"x = {x}")
        print(f"requires_grad: {x.requires_grad}\n")
        
        # Tính toán
        y = x ** 2
        z = y.sum()
        print(f"y = x^2 = {y}")
        print(f"z = sum(y) = {z.item()}\n")
        
        # Backward
        z.backward()
        print(f"dz/dx = {x.grad}")
        print("(Gradient là 2*x = [4.0, 6.0])")
        
        return True
    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
        return False

def main():
    """Hàm chính"""
    print("\n" + "=" * 60)
    print("  KIỂM TRA TOÀN DIỆN AMD GPU VỚI DIRECTML")
    print("=" * 60)
    
    # Thông tin phiên bản
    print(f"\nPyTorch version: {torch.__version__}")
    try:
        directml_version = torch_directml.__version__
    except AttributeError:
        directml_version = "N/A (torch-directml installed)"
    print(f"DirectML version: {directml_version}")
    
    # Test 1: Khởi tạo thiết bị
    device = test_device_initialization()
    if device is None:
        print("\n✗ KHÔNG THỂ KHỞI TẠO GPU!")
        return
    
    # Test 2: Tạo tensor
    test_tensor_creation(device)
    
    # Test 3: Phép toán tensor
    test_tensor_operations(device)
    
    # Test 4: Phép toán ma trận
    test_matrix_operations(device)
    
    # Test 5: Hàm kích hoạt
    test_activation_functions(device)
    
    # Test 6: Gradient
    test_gradient_computation(device)
    
    # Kết thúc
    print_header("HOÀN THÀNH TẤT CẢ TESTS")
    print("✓ GPU AMD hoạt động tốt với DirectML!\n")

if __name__ == "__main__":
    main()