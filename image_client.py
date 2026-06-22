import cv2
import zmq
import numpy as np
import time
import struct
import pickle
import threading
from collections import deque
from multiprocessing import shared_memory
import logging_mp
logger_mp = logging_mp.getLogger(__name__)
 
class ImageClient:
    def __init__(self, tv_img_shape = None, tv_img_shm_name = None, wrist_img_shape = None, wrist_img_shm_name = None,
                       image_show = False, server_address = "192.168.123.164", port = 5556, Unit_Test = False,
                       frame_counter = None):
        """
        tv_img_shape: User's expected head camera resolution shape (H, W, C). It should match the output of the image service terminal.

        tv_img_shm_name: Shared memory is used to easily transfer images across processes to the Vuer.

        wrist_img_shape: User's expected wrist camera resolution shape (H, W, C). It should maintain the same shape as tv_img_shape.

        wrist_img_shm_name: Shared memory is used to easily transfer images.
        
        image_show: Whether to display received images in real time.

        server_address: The ip address to execute the image server script.

        port: The port number to bind to. It should be the same as the image server.

        Unit_Test: When both server and client are True, it can be used to test the image transfer latency, \
                   network jitter, frame loss rate and other information.

        frame_counter: Optional multiprocessing.Value('Q', lock=True). When provided, it is incremented after every
                       successful write to the head/wrist shared-memory arrays. Cross-process consumers (e.g. the Vuer
                       subprocess) can read this counter to dedup and avoid re-encoding/re-sending unchanged frames.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port
 
        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape
 
        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = self.tv_image_shm.buf)
            self.tv_enable_shm = True
        
        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = self.wrist_image_shm.buf)
            self.wrist_enable_shm = True
 
        # Depth data storage (updated on each frame if depth is available)
        self.depth_raw = None   # numpy array (H, W) uint16 - raw depth values in mm
        self.head_depth = None  # numpy array (H, W) uint16 - depth in mm (legacy)
        self.wrist_depth = None  # numpy array (H, W) uint16 - depth in mm (legacy)
        self.has_depth = False  # True if server is sending depth data
 
        # Guards writes to the shared-memory image arrays (tv_img_array,
        # wrist_img_array) so consumers can take a torn-free snapshot via
        # `with img_client.frame_lock: img = arr.copy()`.
        self.frame_lock = threading.Lock()

        # Optional cross-process counter, bumped on every successful write so
        # other processes (e.g. the Vuer subprocess) can dedup unchanged frames.
        self.frame_counter = frame_counter

        # Performance evaluation parameters
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

        self._dim_mismatch_log_counter = 0
 
    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID
 
        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window
 
        # Data transmission quality metrics
        self._latencies = deque()  # Latencies of frames within the time window
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs
 
    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # Update latency
        latency = receive_time - timestamp
        self._latencies.append(latency)
 
        # Remove latencies outside the time window
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()
 
        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()
 
        # Update frame counts for lost frame calculation
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                logger_mp.info(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                logger_mp.warning(f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}")
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1
 
        self._frame_count += 1
 
    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # Calculate real-time FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0
 
            # Calculate latency metrics
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0
 
            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0
 
            logger_mp.info(f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, \
                  Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%")
    
    def _bump_frame_counter(self):
        if self.frame_counter is None:
            return
        try:
            with self.frame_counter.get_lock():
                self.frame_counter.value += 1
        except Exception:
            # Frame counter is best-effort; never let it kill the receive loop.
            pass

    def _close(self):
        self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        logger_mp.info("Image client has been closed.")
 
    
    def receive_process(self):
        # Set up ZeroMQ context and socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        # Drop stale frames: keep only the most recent message and discard the
        # rest. CONFLATE must be set BEFORE connect/subscribe to take effect.
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
 
        logger_mp.info("Image client has started, waiting to receive data...")
        depth_log_counter = 0
        try:
            while self.running:
                # Poll with timeout so we can check self.running and exit cleanly
                if self._socket.poll(timeout=100) == 0:
                    continue
                message = self._socket.recv()
                receive_time = time.time()
 
                # Detect message format: raw jpg (starts with FFD8) or pickle with depth
                jpg_bytes = None
                head_jpg_bytes = None
                wrist_jpg_bytes = None
                timestamp = None
                frame_id = None
 
                # JPG always starts with FF D8
                is_jpg = len(message) > 2 and message[0:2] == b'\xff\xd8'
 
                if is_jpg:
                    # Raw JPG format (no depth)
                    jpg_bytes = message
                    self.depth_raw = None
                else:
                    # Try pickle format (image server with depth support)
                    try:
                        data = pickle.loads(message)
                        if isinstance(data, dict):
                            head_jpg_bytes = data.get('head_image')
                            wrist_jpg_bytes = data.get('wrist_image')
                            jpg_bytes = data.get('image') or data.get('rgb')
                            self.depth_raw = data.get('depth_raw')  # 16-bit uint16 or None
                            self.head_depth = data.get('head_depth')
                            self.wrist_depth = data.get('wrist_depth')
                            frame_id = data.get('frame_id')
                            timestamp = data.get('timestamp')
                            self.has_depth = (self.depth_raw is not None or self.head_depth is not None or self.wrist_depth is not None)
                            if jpg_bytes is None and head_jpg_bytes is None:
                                jpg_bytes = message  # fallback
                        else:
                            jpg_bytes = message
                            self.depth_raw = None
                    except Exception:
                        jpg_bytes = message
                        self.depth_raw = None
 
                # Preferred path: separate head/wrist streams at native resolution.
                if head_jpg_bytes is not None:
                    head_img = cv2.imdecode(np.frombuffer(head_jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    wrist_img = None
                    if wrist_jpg_bytes is not None:
                        wrist_img = cv2.imdecode(np.frombuffer(wrist_jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if head_img is None:
                        logger_mp.warning("[Image Client] Failed to decode head image.")
                        continue
                    if self.tv_enable_shm and head_img.shape != self.tv_img_array.shape:
                        head_img = cv2.resize(head_img, (self.tv_img_shape[1], self.tv_img_shape[0]))
                    if self.wrist_enable_shm and wrist_img is not None and wrist_img.shape != self.wrist_img_array.shape:
                        wrist_img = cv2.resize(wrist_img, (self.wrist_img_shape[1], self.wrist_img_shape[0]))
                    with self.frame_lock:
                        if self.tv_enable_shm:
                            np.copyto(self.tv_img_array, head_img)
                        if self.wrist_enable_shm and wrist_img is not None:
                            np.copyto(self.wrist_img_array, wrist_img)
                    self._bump_frame_counter()
                    if self._image_show:
                        if wrist_img is not None:
                            if wrist_img.shape[0] != head_img.shape[0]:
                                scale = head_img.shape[0] / wrist_img.shape[0]
                                target_w = int(wrist_img.shape[1] * scale)
                                wrist_show = cv2.resize(wrist_img, (target_w, head_img.shape[0]))
                            else:
                                wrist_show = wrist_img
                            show = cv2.hconcat([head_img, wrist_show])
                        else:
                            show = head_img
                        h, w = show.shape[:2]
                        cv2.imshow('Image Client Stream', cv2.resize(show, (w // 2, h // 2)))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.running = False
                    if self._enable_performance_eval:
                        self._update_performance_metrics(timestamp, frame_id, receive_time)
                        self._print_performance_metrics(receive_time)
                    continue
                
                if jpg_bytes is None:
                    continue
                
                # Unit_Test mode: server may send header + jpg instead of pickle
                if self._enable_performance_eval and jpg_bytes == message and not is_jpg:
                    header_size = struct.calcsize('dI')
                    if len(message) > header_size:
                        try:
                            header = message[:header_size]
                            jpg_bytes = message[header_size:]
                            timestamp, frame_id = struct.unpack('dI', header)
                        except struct.error:
                            pass  # Keep jpg_bytes as message
                
                # Decode image
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if current_image is None:
                    logger_mp.warning("[Image Client] Failed to decode image.")
                    continue
 
                # Validate received image dimensions
                recv_height, recv_width = current_image.shape[:2]
                
                # Calculate required dimensions
                required_width = 0
                if self.tv_enable_shm:
                    required_width += self.tv_img_shape[1]
                    expected_height = self.tv_img_shape[0]
                if self.wrist_enable_shm:
                    required_width += self.wrist_img_shape[1]
                    if not self.tv_enable_shm:
                        expected_height = self.wrist_img_shape[0]
                
                # Validate height (should be same for both cameras if both enabled)
                if self.tv_enable_shm or self.wrist_enable_shm:
                    if recv_height != expected_height:
                        self._dim_mismatch_log_counter += 1
                        if self._dim_mismatch_log_counter <= 5 or self._dim_mismatch_log_counter % 120 == 0:
                            logger_mp.warning(
                                f"[Image Client] Dropping frame: height {recv_height} != expected {expected_height}. "
                                "Match head_image_height / wrist_image_height in eval config to the image server."
                            )
                        continue
 
                    if recv_width < required_width:
                        self._dim_mismatch_log_counter += 1
                        if self._dim_mismatch_log_counter <= 5 or self._dim_mismatch_log_counter % 120 == 0:
                            logger_mp.warning(
                                f"[Image Client] Dropping frame: width {recv_width} < required {required_width} "
                                f"(head shm width {self.tv_img_shape[1] if self.tv_enable_shm else 0}, "
                                f"wrist shm width {self.wrist_img_shape[1] if self.wrist_enable_shm else 0}). "
                                "Set head_image_width + wrist layout in EvalRealConfig to match the stitched server output "
                                f"(e.g. RealSense 848 + one wrist 640 => {848 + 640} px wide)."
                            )
                        continue
                
                # Extract camera images from concatenated image
                with self.frame_lock:
                    if self.tv_enable_shm:
                        tv_width = self.tv_img_shape[1]
                        np.copyto(self.tv_img_array, np.array(current_image[:, :tv_width]))
                    if self.wrist_enable_shm:
                        wrist_width = self.wrist_img_shape[1]
                        np.copyto(self.wrist_img_array, np.array(current_image[:, -wrist_width:]))
                self._bump_frame_counter()
                
                if self._image_show:
                    height, width = current_image.shape[:2]
                    resized_image = cv2.resize(current_image, (width // 2, height // 2))
                    cv2.imshow('Image Client Stream', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
 
                if self._enable_performance_eval:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                    self._print_performance_metrics(receive_time)
 
        except KeyboardInterrupt:
            logger_mp.info("Image client interrupted by user.")
        except Exception as e:
            logger_mp.warning(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()
 
if __name__ == "__main__":
    # example1
    # tv_img_shape = (480, 1280, 3)
    # img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    # img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    # img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    # img_client.receive_process()
 
    # example2
    # Initialize the client
    # For local test (server on same machine):
    # client = ImageClient(image_show=True, server_address='192.168.0.144', Unit_Test=False)
    # For robot (server on robot):
    client = ImageClient(image_show=True, server_address='192.168.123.164', Unit_Test=False)
    client.receive_process()