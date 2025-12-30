import streamlit as st
import cv2
import numpy as np
import time
import queue
import threading
from pathlib import Path
import yaml
import pandas as pd
import altair as alt
from backend.utils.config_loader import load_config
from backend.core.gpu_manager import GPUManager
from backend.core.video_processor import VideoProcessor
from backend.utils.stream_loader import StreamLoader
# --- MONKEY PATCH FOR STREAMLIT >= 1.39 ---
# Fixes compatibility issue between streamlit-drawable-canvas and newer Streamlit versions
import streamlit.elements.image as st_image
try:
    from streamlit.elements.lib.image_utils import image_to_url as real_image_to_url
    
    class MockWidthConfig:
        def __init__(self, width):
            self.width = width

    def shim_image_to_url(image, width, clamp, channels, output_format, image_id, allow_emoji=False):
        """
        Shim function to adapt streamlit-drawable-canvas's old call signature.
        Streamlit 1.52 expects 'width_config' object as 2nd arg, not int.
        """
        return real_image_to_url(image, MockWidthConfig(width), clamp, channels, output_format, image_id)
        
    st_image.image_to_url = shim_image_to_url
except ImportError:
    pass # Fallback or different version
# ------------------------------------------

from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Hệ thống giám sát giao thông AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
@st.cache_resource
def load_system_config():
    config = load_config('config/config.yaml')
    gpu_manager = GPUManager(config)
    return config, gpu_manager

config, gpu_manager = load_system_config()

# Sidebar
st.sidebar.title("🚦 Điều khiển hệ thống")
page = st.sidebar.radio("Chế độ", ["Giám sát", "Phân tích", "Hiệu chỉnh"])

# 1. Source Selection
st.sidebar.subheader("Nguồn đầu vào")
# ... (Source selection logic remains same)
source_type = st.sidebar.selectbox("Loại", ["Video có sẵn", "YouTube URL", "Webcam"])
input_source = None
if source_type == "Video có sẵn":
    input_source = st.sidebar.text_input("Đường dẫn", "data/M6 Motorway Traffic.mp4")
elif source_type == "YouTube URL":
    input_source = st.sidebar.text_input("URL", "https://youtu.be/sMrorDb5T0E?si=_D6vzFwltpf2obst")
else:
    input_source = st.sidebar.select_slider("ID Webcam", options=[0, 1, 2], value=0)

# 2. Model Configuration (ONNX for DirectML)
st.sidebar.subheader("🧠 Mô hình AI (ONNX)")
model_options = ['yolo11n.onnx', 'yolo11s.onnx', 'yolo11m.onnx', 'yolov8n.onnx', 'Mô hình tùy chỉnh...']
selected_option = st.sidebar.selectbox("Chọn mô hình", model_options, index=0)

target_onnx_path = None

if selected_option == 'Mô hình tùy chỉnh...':
    custom_path_str = st.sidebar.text_input("Đường dẫn đến mô hình (.pt hoặc .onnx)", "models/my_model.pt")
    custom_path = Path(custom_path_str)
    
    if custom_path.suffix == '.onnx':
        target_onnx_path = custom_path
    elif custom_path.suffix == '.pt':
        # Suggest ONNX path
        suggested_onnx = custom_path.with_suffix('.onnx')
        target_onnx_path = suggested_onnx
        
        if not suggested_onnx.exists():
            st.sidebar.warning(f"⚠️ Không tìm thấy phiên bản ONNX cho {custom_path.name}")
            if st.sidebar.button(f"⚡ Xuất mô hình tùy chỉnh"):
                if not custom_path.exists():
                    st.sidebar.error("Không tìm thấy file nguồn .pt!")
                else:
                    try:
                        with st.spinner("Đang xuất mô hình tùy chỉnh..."):
                            from ultralytics import YOLO
                            model = YOLO(str(custom_path))
                            path = model.export(format='onnx', dynamic=True)
                            st.sidebar.success(f"✅ Đã xuất tới {path}")
                            st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Xuất thất bại: {e}")
    else:
        st.sidebar.error("Vui lòng chọn file .pt hoặc .onnx.")

else:
    # Standard models
    target_onnx_path = Path(f"models/{selected_option}")
    if not target_onnx_path.exists():
        st.sidebar.warning(f"⚠️ Không tìm thấy {selected_option}.")
        if st.sidebar.button(f"⚡ Xuất {selected_option}"):
            try:
                with st.spinner(f"Đang xuất {selected_option}..."):
                    from ultralytics import YOLO
                    pt_name = selected_option.replace('.onnx', '.pt')
                    model = YOLO(pt_name)
                    path = model.export(format='onnx', dynamic=True) 
                    
                    # Move to models/ dir
                    exported_path = Path(path)
                    target_path = Path(f"models/{selected_option}")
                    target_path.parent.mkdir(exist_ok=True)
                    
                    if exported_path.resolve() != target_path.resolve():
                        exported_path.replace(target_path)
                    st.sidebar.success(f"✅ Sẵn sàng!")
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Xuất thất bại: {e}")

# Update config
if target_onnx_path:
    config['model']['path'] = str(target_onnx_path)

# --- ANALYTICS PAGE ---
if page == "Phân tích":
    st.title("📊 Phân tích giao thông")
    csv_file = Path("output/violations_log.csv")
    
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Hour'] = df['Timestamp'].dt.hour
            
            # KPI Cards
            k1, k2, k3 = st.columns(3)
            k1.metric("Tổng số vi phạm", len(df))
            k2.metric("Tốc độ vi phạm trung bình", f"{df['Speed (km/h)'].mean():.1f} km/h")
            k3.metric("Phổ biến nhất", df['Class'].mode()[0] if not df.empty else "N/A")
            
            st.markdown("---")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Loại phương tiện")
                chart_type = alt.Chart(df).mark_arc(innerRadius=50).encode(
                    theta='count()',
                    color='Class',
                    tooltip=['Class', 'count()']
                ).properties(height=300)
                st.altair_chart(chart_type, theme="streamlit")
                
            with c2:
                st.subheader("Phân bố tốc độ")
                chart_speed = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Speed (km/h)', bin=True),
                    y='count()',
                    color=alt.value('orange'),
                    tooltip=['count()']
                ).properties(height=300)
                st.altair_chart(chart_speed, theme="streamlit")
                
            st.subheader("Vi phạm theo thời gian (Giờ trong ngày)")
            chart_time = alt.Chart(df).mark_line(point=True).encode(
                x='Hour:O',
                y='count()',
                tooltip=['Hour', 'count()']
            ).properties(height=300)
            st.altair_chart(chart_time, theme="streamlit")
            
            st.markdown("### 📝 Nhật ký dữ liệu thô")
            st.dataframe(df.sort_values('Timestamp', ascending=False), use_container_width=True)
            
        except Exception as e:
            st.error(f"Lỗi tải dữ liệu: {e}")
    else:
        st.info("Không có dữ liệu. Hãy chạy chế độ 'Giám sát' để ghi nhận vi phạm trước.")

# --- CALIBRATION PAGE ---
elif page == "Hiệu chỉnh":
    st.title("📏 Hiệu chỉnh Camera")
    st.markdown("Vẽ 4 điểm để xác định vùng phối cảnh. Thứ tự: **Trên-Trái -> Trên-Phải -> Dưới-Phải -> Dưới-Trái**")
    
    col1, col2 = st.columns([3, 1])
    
    if "calib_frame" not in st.session_state:
        st.session_state.calib_frame = None
        
    # Define inputs FIRST so they are available
    with col2:
        if st.button("📸 Chụp khung hình"):
            try:
                with st.spinner("Đang chụp khung hình..."):
                    cap = StreamLoader(input_source)
                    # Read a few frames to settle (reduced to 5)
                    for _ in range(5): 
                        ret, frame = cap.read()
                        if not ret: break
                    
                    if ret and frame is not None:
                        # Resize to 480p consistent with processing
                        if frame.shape[0] > 480:
                            scale = 480 / frame.shape[0]
                            w = int(frame.shape[1] * scale)
                            frame = cv2.resize(frame, (w, 480))
                        
                        st.session_state.calib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.rerun()
                    else:
                        st.error("Không thể chụp khung hình. Kiểm tra URL/Nguồn.")
                        
                    cap.release()
            except Exception as e:
                st.error(f"Lỗi chụp khung hình: {e}")

        real_w = st.number_input("Chiều rộng thực (m)", value=10.0)
        real_h = st.number_input("Chiều cao thực (m)", value=30.0)
        
    # Canvas Area
    with col1:
        display_frame = st.session_state.calib_frame
        
        # Default placeholder if no frame captured
        if display_frame is None:
            st.info("👈 Nhấn 'Chụp khung hình' để tải ảnh từ nguồn video.")
            # Create a black placeholder frame
            display_frame = np.zeros((480, 854, 3), dtype=np.uint8)
        
        # Create a canvas component
        from PIL import Image
        bg_image = Image.fromarray(display_frame)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=bg_image,
            update_streamlit=True,
            height=display_frame.shape[0],
            width=display_frame.shape[1],
            drawing_mode="point",
            point_display_radius=5,
            key="canvas",
        )
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            points = [[obj["left"], obj["top"]] for obj in objects]
            
            st.write(f"Điểm đã chọn: {len(points)}")
            
            if len(points) == 4:
                if st.button("💾 Lưu hiệu chỉnh"):
                    # Calculate Matrix
                    src_points = np.float32(points)
                    dst_points = np.float32([
                        [0, 0],
                        [real_w * 100, 0],
                        [real_w * 100, real_h * 100],
                        [0, real_h * 100]
                    ])
                    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    
                    calibration = {
                        'points': points,
                        'width_meters': float(real_w),
                        'height_meters': float(real_h),
                        'transform_matrix': matrix.tolist(),
                        'pixels_per_meter': 100,
                        'frame_width': display_frame.shape[1],
                        'frame_height': display_frame.shape[0]
                    }
                    
                    with open("config/calibration.yaml", 'w') as f:
                        yaml.dump(calibration, f)
                    st.success("Đã lưu hiệu chỉnh! Vui lòng chuyển sang chế độ Giám sát.")


# --- MONITOR PAGE ---
elif page == "Giám sát":
    # 2. Parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Tham số")
    speed_limit = st.sidebar.slider("Giới hạn tốc độ (km/h)", 20, 150, 100)
    confidence = st.sidebar.slider("Ngưỡng tin cậy", 0.1, 1.0, 0.4)

    # Apply runtime config changes
    config['speed']['speed_limit'] = speed_limit
    config['model']['confidence'] = confidence

    # Export Data Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Xuất dữ liệu")
    csv_path = Path("output/violations_log.csv")
    if csv_path.exists():
        with open(csv_path, "rb") as f:
            st.sidebar.download_button(
                label="Download CSV Report",
                data=f,
                file_name="violations_log.csv",
                mime="text/csv"
            )
    else:
        st.sidebar.caption("Chưa có dữ liệu vi phạm nào được ghi nhận.")

    # Main Interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📹 Nguồn trực tiếp")
        video_placeholder = st.empty()
        
        c1, c2 = st.columns(2)
        start_button = c1.button("▶️ Bắt đầu giám sát", type="primary")
        stop_button = c2.button("⏹️ Dừng")

    with col2:
        st.markdown("### 📊 Thống kê thời gian thực")
        
        # Metrics
        m1, m2 = st.columns(2)
        kpi_fps = m1.metric("FPS", "0.0")
        kpi_vehicles = m2.metric("Số lượng xe", "0")
        
        m3, m4 = st.columns(2)
        kpi_avg_speed = m3.metric("Tốc độ TB (km/h)", "0.0")
        kpi_violations = m4.metric("Vi phạm", "0", delta_color="inverse")
        
        # Charts
        st.markdown("#### Số lượng xe")
        chart_placeholder = st.empty()

        st.markdown("#### Bộ sưu tập vi phạm")
        gallery_placeholder = st.empty()

    # Processing Logic
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

    if start_button:
        st.session_state.is_running = True

    if stop_button:
        st.session_state.is_running = False

    def run_processing():
        # Reload config to pick up new calibration
        config, gpu_manager = load_system_config()
        processor = VideoProcessor(config, gpu_manager)
        
        # Helper Charts & Gallery (Keep existing)
        def altair_chart(df):
            return alt.Chart(df).mark_bar().encode(
                x='Vehicle',
                y='Count',
                color='Vehicle'
            )

        def update_gallery(placeholder):
            import glob
            import os
            latest_dir = Path(f"output/violations/{time.strftime('%Y-%m-%d')}")
            if latest_dir.exists():
                files = sorted(glob.glob(str(latest_dir / "*.jpg")), key=os.path.getmtime, reverse=True)[:4]
                if files:
                    with placeholder.container():
                        cols = st.columns(2)
                        for i, f in enumerate(files):
                            try:
                                img = cv2.imread(f)
                                if img is not None:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    cols[i%2].image(img, caption=Path(f).name)
                            except: pass
        
        frame_count = 0
        
        # Use the multi-threaded generator pipeline
        # processor.process_stream_generator yields (frame, stats)
        for annotated_frame, stats in processor.process_stream_generator(input_source):
            if not st.session_state.is_running:
                break
                
            # Update Streamlit UI
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")
            
            # Update Stats every 10 frames to reduce UI overhead
            if frame_count % 10 == 0:
                kpi_fps.metric("FPS", f"{stats['fps']:.1f}")
                
                # stats['counts'] is a defaultdict, sum values for total
                total_k = sum(stats['counts'].values()) if stats['counts'] else 0
                kpi_vehicles.metric("Xe", f"{total_k}")
                kpi_violations.metric("Vi phạm", f"{stats['violations']}")
                
                if stats['counts']:
                    df = pd.DataFrame(list(stats['counts'].items()), columns=['Vehicle', 'Count'])
                    chart_placeholder.altair_chart(altair_chart(df), use_container_width=True)
                
                update_gallery(gallery_placeholder)
            
            frame_count += 1

    if st.session_state.is_running:
        run_processing()
