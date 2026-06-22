import cv2
import zmq
import time
import pickle
from collections import deque
import numpy as np
import pyrealsense2 as rs
import logging

try:
    import logging_mp
    logger_mp = logging_mp.getLogger(__name__)
except (ImportError, AttributeError):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger_mp = logging.getLogger(__name__)

try:
    from teleop.image_server.depth_visualization_3ddp import depth_to_visualization
except ImportError:
    from depth_visualization_3ddp import depth_to_visualization


class RealSenseCamera:
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False):
        """
        img_shape: [height, width]
        serial_number: serial number
        """
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth
        self.align = rs.align(rs.stream.color)
        self._init_pipeline()

    def _detect_usb_fps(self):
        """Reduce FPS for USB 2.x devices (limited bandwidth)."""
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.get_info(rs.camera_info.serial_number) == self.serial_number:
                    usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
                    if usb_type.startswith("2"):
                        logger_mp.warning(f'[RealSense] USB {usb_type} detected, reducing to 15fps')
                        return 15
                    break
        except Exception:
            pass
        return self.fps

    def _init_depth_filters(self):
        """Set up depth post-processing: spatial -> temporal -> hole_fill."""
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        try:
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
            try:
                self.hole_filling_filter.set_option(rs.option.holes_fill, 2)
            except Exception:
                pass
            logger_mp.info('[RealSense] Depth filters: spatial + temporal + hole_fill')
        except Exception as e:
            logger_mp.warning(f'[RealSense] Filter setup: {e}')

    def _init_pipeline(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number is not None:
            config.enable_device(self.serial_number)

        rs_fps = self._detect_usb_fps()
        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, rs_fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, rs_fps)

        logger_mp.info(f'[RealSense] Starting pipeline: {self.img_shape[1]}x{self.img_shape[0]} @ {rs_fps}fps')
        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self._device is None:
            logger_mp.error('[RealSense] pipe_profile.get_device() is None.')

        if self.enable_depth:
            assert self._device is not None
            self.g_depth_scale = self._device.first_depth_sensor().get_depth_scale()
            self._init_depth_filters()
        else:
            self.spatial_filter = None
            self.temporal_filter = None
            self.hole_filling_filter = None

        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # Warm-up: discard first frames to stabilize
        logger_mp.info(f'[RealSense {self.serial_number}] Warming up...')
        for i in range(10):
            try:
                self.pipeline.wait_for_frames(timeout_ms=2000)
            except Exception as e:
                logger_mp.warning(f'[RealSense] Warm-up frame {i} failed: {e}')
        logger_mp.info(f'[RealSense {self.serial_number}] Ready')

    def get_frame(self):
        """Returns (color_image, depth_image). depth_image is None if depth disabled."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except Exception as e:
            logger_mp.warning(f'[RealSense] Frame timeout: {e}')
            return None, None

        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return None, None

        depth_image = None
        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()
            for filt in (self.spatial_filter, self.temporal_filter, self.hole_filling_filter):
                if depth_frame and filt:
                    depth_frame = filt.process(depth_frame)
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())

        return np.asanyarray(color_frame.get_data()), depth_image

    def release(self):
        self.pipeline.stop()

    def log_info(self, label):
        logger_mp.info(f"[Image Server] {label} camera {self.serial_number} resolution: {self.img_shape[0]}x{self.img_shape[1]}")


class OpenCVCamera:
    def __init__(self, device_id, img_shape, fps):
        """
        device_id: /dev/video* or *
        img_shape: [height, width]
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._can_read_frame():
            logger_mp.error(f"[Image Server] Camera {self.id} Error: Failed to initialize or read frames.")
            self.release()

    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def release(self):
        self.cap.release()

    def get_frame(self):
        """Returns (color_image, None) to match RealSenseCamera interface."""
        ret, color_image = self.cap.read()
        if not ret:
            return None, None
        return color_image, None

    def log_info(self, label):
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        logger_mp.info(f"[Image Server] {label} camera {self.id} resolution: {h}x{w}")


class ImageServer:
    def __init__(self, config, port=5556, Unit_Test=False):
        """
        config example:
        {
            'fps': 30,
            'head_camera_type': 'realsense',          # opencv or realsense
            'head_camera_image_shape': [480, 640],     # [height, width]
            'head_camera_id_numbers': ["serial"],      # serial (realsense) or device id (opencv)
            'wrist_camera_type': 'opencv',             # optional
            'wrist_camera_image_shape': [480, 640],    # optional
            'wrist_camera_id_numbers': [0, 1],         # optional
            'enable_depth': True,
            'depth_near_mm': 250,
            'depth_far_mm': 4000,
            'depth_style': '3ddp',                     # 3ddp, turbo, jet
        }
        """
        logger_mp.info(config)
        self.fps = config.get('fps', 30)
        self.enable_depth = config.get('enable_depth', False)
        self.depth_near_mm = config.get('depth_near_mm', 250)
        self.depth_far_mm = config.get('depth_far_mm', 4000)
        self.depth_style = config.get('depth_style', '3ddp')
        self.Unit_Test = Unit_Test

        self.head_cameras = self._create_cameras(
            config.get('head_camera_type', 'opencv'),
            config.get('head_camera_image_shape', [480, 640]),
            config.get('head_camera_id_numbers', [0]),
            enable_depth=self.enable_depth,
        )
        self.wrist_cameras = self._create_cameras(
            config.get('wrist_camera_type'),
            config.get('wrist_camera_image_shape', [480, 640]),
            config.get('wrist_camera_id_numbers'),
        )

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        for cam in self.head_cameras:
            cam.log_info("Head")
        for cam in self.wrist_cameras:
            cam.log_info("Wrist")

        logger_mp.info("[Image Server] Image server has started, waiting for client connections...")

    def _create_cameras(self, camera_type, img_shape, id_numbers, enable_depth=False):
        """Create a list of cameras from config. Returns [] if type/ids are None."""
        if not camera_type or not id_numbers:
            return []

        cameras = []
        if camera_type == 'opencv':
            for device_id in id_numbers:
                cameras.append(OpenCVCamera(device_id=device_id, img_shape=img_shape, fps=self.fps))
        elif camera_type == 'realsense':
            for serial in id_numbers:
                cameras.append(RealSenseCamera(img_shape=img_shape, fps=self.fps, serial_number=serial, enable_depth=enable_depth))
        else:
            logger_mp.warning(f"[Image Server] Unsupported camera_type: {camera_type}")
        return cameras

    def _capture_frames(self, cameras, label):
        """Capture color frames from a list of cameras. Returns (frames, raw_depth) or (None, None) on failure."""
        frames = []
        raw_depth = None
        for cam in cameras:
            color, depth = cam.get_frame()
            if color is None:
                logger_mp.error(f"[Image Server] {label} camera frame read failed.")
                return None, None
            frames.append(color)
            if depth is not None:
                raw_depth = depth.copy()
                depth_colormap = depth_to_visualization(
                    depth,
                    style=self.depth_style,
                    near_mm=self.depth_near_mm,
                    far_mm=self.depth_far_mm,
                )
                frames.append(depth_colormap)
        return frames, raw_depth

    def _init_performance_metrics(self):
        self.frame_count = 0
        self.time_window = 1.0
        self.frame_times = deque()
        self.start_time = time.time()

    def _update_performance_metrics(self, current_time):
        self.frame_times.append(current_time)
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed = current_time - self.start_time
            fps = len(self.frame_times) / self.time_window
            logger_mp.info(f"[Image Server] FPS: {fps:.2f}, Frames: {self.frame_count}, Elapsed: {elapsed:.2f}s")

    def _close(self):
        for cam in self.head_cameras + self.wrist_cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        logger_mp.info("[Image Server] The server has been closed.")

    def send_process(self):
        frame_count = 0
        log_interval = 30  # Log every 30 frames (~1 sec at 30fps)
        try:
            while True:
                t_capture_start = time.time()

                head_frames, raw_depth = self._capture_frames(self.head_cameras, "Head")
                if not head_frames:
                    break
                combined = cv2.hconcat(head_frames)

                if self.wrist_cameras:
                    wrist_frames, _ = self._capture_frames(self.wrist_cameras, "Wrist")
                    if wrist_frames:
                        wrist_combined = cv2.hconcat(wrist_frames)
                        if wrist_combined.shape[0] != combined.shape[0]:
                            target_h = combined.shape[0]
                            scale = target_h / wrist_combined.shape[0]
                            target_w = int(wrist_combined.shape[1] * scale)
                            wrist_combined = cv2.resize(wrist_combined, (target_w, target_h))
                        combined = cv2.hconcat([combined, wrist_combined])

                t_encode_start = time.time()
                ret, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ret:
                    logger_mp.error("[Image Server] Frame imencode failed.")
                    continue
                t_encode_end = time.time()

                jpg_bytes = buffer.tobytes()
                # Pull intrinsics from the first head camera (they're
                # identical for all cameras at the same resolution).
                _intr = None
                if self.head_cameras:
                    _i = self.head_cameras[0].intrinsics
                    _intr = {
                        'fx': float(_i.fx), 'fy': float(_i.fy),
                        'cx': float(_i.ppx), 'cy': float(_i.ppy),
                        'width': int(_i.width), 'height': int(_i.height),
                    }
                    # publish the real device depth scale (metres per raw unit) so clients
                    # deproject with the true scale instead of assuming the D435 default 0.001.
                    _ds = getattr(self.head_cameras[0], 'g_depth_scale', None)
                    if _ds is not None:
                        _intr['depth_scale'] = float(_ds)
                message = pickle.dumps({
                    'image': jpg_bytes,
                    'depth_raw': raw_depth,
                    'intrinsics': _intr,
                    'timestamp': time.time(),
                    'frame_id': frame_count,
                })

                t_send_start = time.time()
                self.socket.send(message)
                t_send_end = time.time()

                frame_count += 1
                if frame_count % log_interval == 0:
                    capture_ms = (t_encode_start - t_capture_start) * 1000
                    encode_ms = (t_encode_end - t_encode_start) * 1000
                    send_ms = (t_send_end - t_send_start) * 1000
                    total_ms = (t_send_end - t_capture_start) * 1000
                    size_kb = len(jpg_bytes) / 1024
                    logger_mp.info(
                        f"[Server Latency] capture={capture_ms:.1f}ms  encode={encode_ms:.1f}ms  "
                        f"send={send_ms:.1f}ms  total={total_ms:.1f}ms  size={size_kb:.0f}KB  "
                        f"frame={frame_count}  res={combined.shape[1]}x{combined.shape[0]}"
                    )

                if self.Unit_Test:
                    t = time.time()
                    self._update_performance_metrics(t)
                    self._print_performance_metrics(t)

        except KeyboardInterrupt:
            logger_mp.warning("[Image Server] Interrupted by user.")
        finally:
            self._close()


def _parse_resolution(value, default):
    """Parse 'WxH' string into [height, width]. Returns default on failure."""
    try:
        w, h = map(int, value.lower().split('x'))
        return [h, w]
    except Exception:
        logger_mp.warning(f'[CONFIG] Invalid resolution "{value}", using {default[1]}x{default[0]}')
        return default


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrist', type=int, nargs='+', default=[], help='Wrist camera ports (empty or -1 to disable)')
    parser.add_argument('--no-depth', action='store_true', help='Disable depth')
    parser.add_argument('--no-wrist', action='store_true', help='Disable wrist cameras')
    parser.add_argument('--depth-near', type=int, default=250, help='Depth near plane (mm)')
    parser.add_argument('--depth-far', type=int, default=4000, help='Depth far plane (mm)')
    parser.add_argument('--depth-style', type=str, default='turbo', choices=['3ddp', 'turbo', 'jet'],
                        help='3ddp=3D-Diffusion-Policy style (u,v,depth as RGB), turbo/jet=colormap')
    parser.add_argument('--resolution', type=str, default='1280x720', help='Head camera: WxH')
    parser.add_argument('--wrist-resolution', type=str, default='640x480', help='Wrist camera: WxH')
    args = parser.parse_args()

    # Auto-detect RealSense
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("ERROR: No RealSense camera found!")
        exit(1)
    serial = devices[0].get_info(rs.camera_info.serial_number)
    print(f"[AUTO] RealSense: {serial}")

    wrist_ports = [] if args.no_wrist or args.wrist == [-1] else args.wrist
    head_shape = _parse_resolution(args.resolution, [720, 1280])
    wrist_shape = _parse_resolution(args.wrist_resolution, [480, 640])

    config = {
        'fps': 30,
        'head_camera_type': 'realsense',
        'head_camera_image_shape': head_shape,
        'head_camera_id_numbers': [serial],
        'enable_depth': not args.no_depth,
        'depth_near_mm': args.depth_near,
        'depth_far_mm': args.depth_far,
        'depth_style': args.depth_style,
    }
    if wrist_ports:
        config['wrist_camera_type'] = 'opencv'
        config['wrist_camera_image_shape'] = wrist_shape
        config['wrist_camera_id_numbers'] = wrist_ports

    print(f"[CONFIG] Head resolution: {head_shape[1]}x{head_shape[0]}")
    print(f"[CONFIG] Head: RealSense RGB + {'Depth' if config['enable_depth'] else 'No Depth'}")
    if config['enable_depth']:
        print(f"[CONFIG] Depth: {config['depth_near_mm']}-{config['depth_far_mm']}mm, style={config['depth_style']}")
    if wrist_ports:
        print(f"[CONFIG] Wrist resolution: {wrist_shape[1]}x{wrist_shape[0]}")
    print(f"[CONFIG] Wrist: {wrist_ports if wrist_ports else 'DISABLED'}")

    server = ImageServer(config, Unit_Test=False)
    server.send_process()

