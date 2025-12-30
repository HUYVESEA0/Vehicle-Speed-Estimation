import cv2
import time
import threading
from queue import Queue
import logging
from cap_from_youtube import cap_from_youtube
import yt_dlp

logger = logging.getLogger(__name__)

class StreamLoader:
    def __init__(self, source, buffer_size=2):
        self.source = source
        self.is_youtube = "youtube.com" in str(source) or "youtu.be" in str(source)
        self.stopped = False
        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        
        # Initialize Capture
        self.cap = self._get_capture()
        
        # FPS and dimensions
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.fps == 0: self.fps = 30 # Default if unknown
        
        # Start reading thread if it's a stream (not a local file)
        # We classify something as a stream if it's a URL or int (webcam)
        self.is_stream = isinstance(source, int) or str(source).startswith(('http', 'rtsp', 'rtmp'))
        
        if self.is_stream:
            logger.info("⚡ Live Stream detected - Starting separate reading thread")
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _get_capture(self):
        if self.is_youtube:
            logger.info(f"Connecting to YouTube stream: {self.source}")
            # Try resolutions in order of preference to find an available one
            resolutions = ['720p', '480p', '1080p', '360p', 'best']
            last_err = None
            
            for res in resolutions:
                try:
                    logger.info(f"Attempting YouTube resolution: {res}")
                    cap = cap_from_youtube(self.source, resolution=res)
                    if cap and cap.isOpened():
                        logger.info(f"Connected to YouTube with resolution: {res}")
                        return cap
                except Exception as e:
                    logger.warning(f"Resolution {res} failed or not available: {e}")
                    last_err = e
            
            logger.error(f"Failed to connect to YouTube with any resolution. Last error: {last_err}")
            if last_err:
                logger.warning(f"cap_from_youtube failed: {last_err}. Trying direct yt_dlp fallback...")
            
            # Fallback: Use yt_dlp directly with different client (android/ios) to bypass SABR
            try:
                ydl_opts = {
                    'format': 'best',
                    'quiet': True,
                    # Bypass "no url" issue by using mobile clients
                    'extractor_args': {'youtube': {'player_client': ['android', 'ios']}},
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source, download=False)
                    url = info.get('url')
                    if url:
                        logger.info(f"Got direct URL via yt_dlp fallback: {url}")
                        return cv2.VideoCapture(url)
            except Exception as e:
                 logger.error(f"Direct yt_dlp fallback failed: {e}")
            
            if last_err:
                raise last_err
            raise ValueError("Could not connect to YouTube stream (no valid resolution found)")
        else:
            # Local file or Webcam/RTSP
            str_source = str(self.source)
            # Check if it's an integer (webcam index)
            if str_source.isdigit():
                return cv2.VideoCapture(int(str_source))
            return cv2.VideoCapture(str_source)

    def _update(self):
        """Thread worker to keep reading frames"""
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
                
            ret, frame = self.cap.read()
            if not ret:
                # For streams, we might want to retry reconnection logic here
                # For now, just stop
                self.stopped = True
                break
            
            # Keep queue size small to always have latest frame
            if not self.queue.empty():
                try:
                    self.queue.get_nowait() # Discard old frame
                except:
                    pass
            
            self.queue.put(frame)

    def read(self):
        """Return the next frame"""
        if self.is_stream:
            # For stream, get from queue
            if self.stopped and self.queue.empty():
                return False, None
            
            try:
                frame = self.queue.get(timeout=1.0)
                return True, frame
            except:
                return False, None
        else:
            # For local file, standard read
            return self.cap.read()

    def release(self):
        self.stopped = True
        if self.is_stream and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
