import torch
import torch_directml
import time
import sys

def print_separator():
    """In dòng phân cách"""
    print("=" * 80)

def get_gpu_info():
    """Lấy thông tin về GPU"""
    print_separator()
    print("THÔNG TIN THIẾT BỊ")
    print_separator()
    
    # Khởi tạo thiết bị DirectML
    device = torch_directml.device()
    print(f"Thiết bị DirectML: {device}")
    
    # Thông tin PyTorch
    print(f"Phiên bản PyTorch: {torch.__version__}")
    try:
        directml_version = torch_directml.__version__
    except AttributeError:
        directml_version = "N/A (torch-directml installed)"
    print(f"Phiên bản torch_directml: {directml_version}")
    
    # Kiểm tra CUDA (thường không có trên AMD với DirectML)
    print(f"CUDA có sẵn: {torch.cuda.is_available()}")
    
    return device

def test_matrix_multiplication(device, size=10000):
    """Test nhân ma trận"""
    print_separator()
    print(f"TEST 1: NHÂN MA TRẬN {size}x{size}")
    print_separator()
    
    try:
        # Tạo 2 ma trận ngẫu nhiên
        print(f"Đang tạo ma trận {size}x{size}...")
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm-up (chạy thử 1 lần để GPU "khởi động")
        print("Warm-up GPU...")
        _ = torch.matmul(a, b)
        
        # Đo thời gian thực tế
        print("Bắt đầu tính toán...")
        start_time = time.time()
        
        c = torch.matmul(a, b)
        
        # DirectML tự động đồng bộ khi truy cập kết quả
        # Không cần gọi synchronize() như CUDA
        result_sum = c.sum().item()  # Force synchronization
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✓ Hoàn thành!")
        print(f"  Thời gian: {elapsed:.4f} giây")
        print(f"  Tốc độ: {(size * size * size * 2) / elapsed / 1e9:.2f} GFLOPS")
        print(f"  Tổng kết quả: {result_sum:.2e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
        return False

def test_vector_operations(device, size=100000000):
    """Test các phép toán vector"""
    print_separator()
    print(f"TEST 2: PHÉP TOÁN VECTOR ({size} phần tử)")
    print_separator()
    
    try:
        # Tạo vector
        print(f"Đang tạo vector {size} phần tử...")
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        
        # Test cộng
        start_time = time.time()
        c = a + b
        _ = c.sum().item()
        elapsed_add = time.time() - start_time
        print(f"✓ Cộng vector: {elapsed_add:.4f} giây")
        
        # Test nhân element-wise
        start_time = time.time()
        c = a * b
        _ = c.sum().item()
        elapsed_mul = time.time() - start_time
        print(f"✓ Nhân element-wise: {elapsed_mul:.4f} giây")
        
        # Test hàm kích hoạt (ReLU)
        start_time = time.time()
        c = torch.relu(a)
        _ = c.sum().item()
        elapsed_relu = time.time() - start_time
        print(f"✓ ReLU: {elapsed_relu:.4f} giây")
        
        return True
        
    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
        return False

def test_memory_transfer(device):
    """Test tốc độ chuyển dữ liệu CPU <-> GPU"""
    print_separator()
    print("TEST 3: TỐC ĐỘ CHUYỂN DỮ LIỆU")
    print_separator()
    
    try:
        size = 1000
        
        # CPU -> GPU
        cpu_tensor = torch.randn(size, size)
        start_time = time.time()
        gpu_tensor = cpu_tensor.to(device)
        _ = gpu_tensor.sum().item()  # Force transfer
        elapsed_to_gpu = time.time() - start_time
        
        data_size_mb = (size * size * 4) / (1024 * 1024)  # float32 = 4 bytes
        print(f"✓ CPU -> GPU: {elapsed_to_gpu:.4f} giây ({data_size_mb:.2f} MB)")
        
        # GPU -> CPU
        start_time = time.time()
        cpu_tensor = gpu_tensor.cpu()
        elapsed_to_cpu = time.time() - start_time
        print(f"✓ GPU -> CPU: {elapsed_to_cpu:.4f} giây ({data_size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Lỗi: {str(e)}")
        return False

def main():
    """Hàm chính"""
    print("\n")
    print_separator()
    print("CHƯƠNG TRÌNH KIỂM TRA AMD GPU VỚI DIRECTML")
    print_separator()
    print("\n")
    
    try:
        # Lấy thông tin GPU
        device = get_gpu_info()
        print("\n")
        
        # Chạy các test
        results = []
        
        # Test 1: Nhân ma trận
        results.append(test_matrix_multiplication(device, size=5000))
        print("\n")
        
        # Test 2: Phép toán vector
        results.append(test_vector_operations(device, size=50000000))
        print("\n")
        
        # Test 3: Chuyển dữ liệu
        results.append(test_memory_transfer(device))
        print("\n")
        
        # Kết quả tổng hợp
        print_separator()
        print("KẾT QUẢ TỔNG HỢP")
        print_separator()
        passed = sum(results)
        total = len(results)
        print(f"Đã vượt qua: {passed}/{total} tests")
        
        if passed == total:
            print("✓ TẤT CẢ TESTS THÀNH CÔNG!")
        else:
            print("✗ Một số tests thất bại")
        
        print_separator()
        
        return 0 if passed == total else 1
        
    except Exception as e:
        print(f"\n✗ LỖI NGHIÊM TRỌNG: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())