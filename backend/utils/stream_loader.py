import cv2
import time
import threading
from queue import Queue
import logging

logger = logging.getLogger(__name__)

# Optional YouTube support
try:
    from cap_from_youtube import cap_from_youtube
    YOUTUBE_SUPPORT = True
except ImportError:
    logger.warning("cap_from_youtube not available - YouTube streaming disabled")
    cap_from_youtube = None
    YOUTUBE_SUPPORT = False

try:
    import yt_dlp
    YT_DLP_SUPPORT = True
except ImportError:
    logger.warning("yt_dlp not available - YouTube fallback disabled")
    yt_dlp = None
    YT_DLP_SUPPORT = False

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
            if not YOUTUBE_SUPPORT and not YT_DLP_SUPPORT:
                logger.error("YouTube streaming not supported - missing cap_from_youtube and yt_dlp packages")
                raise ImportError("Install cap-from-youtube and yt-dlp to support YouTube streaming")

            logger.info(f"Connecting to YouTube stream: {self.source}")

            # Try cap_from_youtube first if available
            if YOUTUBE_SUPPORT:
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

                logger.error(f"Failed to connect to YouTube with cap_from_youtube. Last error: {last_err}")

            # Fallback: Use yt_dlp directly if available
            if YT_DLP_SUPPORT:
                try:
                    logger.info("Trying yt_dlp fallback...")
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

            raise ValueError("Could not connect to YouTube stream (no valid method available)")
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
