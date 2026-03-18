# YouTube Streaming Examples

## 🎥 Supported YouTube Sources

### Traffic & Highway Videos
```
# Highway Traffic (Multiple Lanes)
https://youtu.be/sMrorDb5T0E

# City Traffic (Urban Roads)
https://youtu.be/c1grCVfIDxU

# Motorway Traffic (High Speed)
https://youtu.be/MNn9qKG2UFI

# Intersection Traffic
https://youtu.be/Y_iXfbxfwDA
```

### Live Traffic Streams
```
# Live Traffic Cameras (if available)
https://youtu.be/LIVE_TRAFFIC_ID

# Traffic CCTV Feeds
https://youtu.be/CCTV_FEED_ID
```

## 🚀 How to Use in App

### 1. Web Interface
1. Mở Streamlit app: `http://localhost:8501`
2. Chọn tab "Hiệu chỉnh"
3. Chọn "YouTube" làm nguồn
4. Paste YouTube URL vào field "URL"
5. Nhấn "📸 Chụp khung hình"
6. Thực hiện hiệu chỉnh với 4 điểm
7. Chuyển sang tab "Giám sát" để bắt đầu phát hiện

### 2. Command Line
```bash
# Run detection on YouTube video
uv run python scripts/run_detection.py --input "https://youtu.be/VIDEO_ID"

# Calibrate using YouTube video
uv run python scripts/calibrate.py --video "https://youtu.be/VIDEO_ID"
```

## ⚡ Performance Tips

### Resolution Selection
- StreamLoader tự động chọn độ phân giải tối ưu: 720p → 480p → 1080p → 360p
- Độ phân giải thấp hơn = xử lý nhanh hơn
- Chất lượng phát hiện vẫn tốt ở 480p-720p

### Network Optimization
- Kết nối internet ổn định (>5 Mbps khuyến nghị)
- Sử dụng WiFi thay vì mobile data nếu có thể
- Tránh các video có nhiều ads/interruptions

## 🛠️ Troubleshooting

### Common Issues

**❌ "YouTube streaming not supported"**
```bash
# Cài đặt dependencies:
uv pip install yt-dlp cap-from-youtube
```

**❌ "Could not connect to YouTube stream"**
- Kiểm tra URL có hợp lệ không
- Thử URL khác (một số video bị hạn chế vùng)
- Kiểm tra kết nối internet

**⚠️ Warning: "No supported JavaScript runtime"**
- Không ảnh hưởng đến chức năng chính
- Chỉ ảnh hưởng một số format đặc biệt

**⚠️ Warning: "ffmpeg not found"**
- Không cần thiết cho streaming trực tiếp
- Chỉ cần nếu muốn download video

### Performance Issues
- Giảm buffer_size trong StreamLoader (mặc định: 2)
- Sử dụng resolution thấp hơn
- Kiểm tra CPU/GPU usage

## 🎯 Best Practices

### Video Selection
- ✅ Chọn video có góc camera cố định
- ✅ Traffic flows rõ ràng, không bị che khuất
- ✅ Lighting conditions tốt (ban ngày)
- ❌ Tránh video có camera di chuyển
- ❌ Tránh video quá tối hoặc mưa to

### Calibration Tips
- Chọn khu vực rectangular có nhiều xe đi qua
- 4 điểm tạo thành perspective view tốt
- Đo kích thước thật của khu vực (sử dụng Google Maps)