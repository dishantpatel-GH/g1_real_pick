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

    def reset(self):
        """Tear down and reinitialise the streaming pipeline.

        Used by ImageServer to recover from transient frame timeouts (USB
        hiccups, autosuspend wake races, etc.) without restarting the process.
        """
        try:
            self.pipeline.stop()
        except Exception as e:
            logger_mp.warning(f'[RealSense {self.serial_number}] pipeline.stop during reset failed: {e}')
        time.sleep(0.5)
        self._init_pipeline()

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

    def get_intrinsics_dict(self):
        """Color-stream intrinsics (depth is aligned to color, so these apply to depth_raw too)
        + depth scale (raw_uint16 * depth_scale = metres). Published so a client can deproject."""
        i = self.intrinsics
        d = {'fx': float(i.fx), 'fy': float(i.fy), 'ppx': float(i.ppx), 'ppy': float(i.ppy),
             'width': int(i.width), 'height': int(i.height),
             'model': str(i.model), 'coeffs': [float(c) for c in i.coeffs]}
        if self.enable_depth:
            d['depth_scale'] = float(self.g_depth_scale)
        return d

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

    def reset(self):
        """Reopen the V4L2 capture. Mirrors RealSenseCamera.reset()."""
        try:
            self.cap.release()
        except Exception as e:
            logger_mp.warning(f'[OpenCVCamera {self.id}] release during reset failed: {e}')
        time.sleep(0.5)
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

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

        # Head-camera intrinsics + depth scale, published once per frame so a detection client
        # (object_detection.py) can deproject depth_raw to 3D without owning the camera.
        self.head_intrinsics = None
        if self.head_cameras and isinstance(self.head_cameras[0], RealSenseCamera):
            self.head_intrinsics = self.head_cameras[0].get_intrinsics_dict()
            logger_mp.info(f"[Image Server] Head intrinsics: {self.head_intrinsics}")

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # Cap outbound buffer to 1 frame so we never publish stale frames.
        # Without this the kernel/ZMQ buffer can hold hundreds of frames if
        # any client briefly stalls, and the client then plays the backlog.
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
        # Auto-recovery: tolerate transient frame-read failures by stopping and
        # restarting each camera pipeline rather than crashing the process.
        # After MAX_CONSECUTIVE_FAILURES bad cycles we give up and exit so a
        # supervisor (systemd, screen, tmux) can decide what to do.
        MAX_CONSECUTIVE_FAILURES = 30
        consecutive_failures = 0
        try:
            while True:
                head_frames, raw_depth = self._capture_frames(self.head_cameras, "Head")
                if not head_frames:
                    consecutive_failures += 1
                    logger_mp.warning(
                        f"[Image Server] Frame read failed "
                        f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}); "
                        f"attempting camera recovery."
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger_mp.error(
                            f"[Image Server] Giving up after "
                            f"{MAX_CONSECUTIVE_FAILURES} consecutive frame failures."
                        )
                        break
                    for cam in self.head_cameras + self.wrist_cameras:
                        try:
                            cam.reset()
                        except Exception as e:
                            logger_mp.error(f"[Image Server] Camera reset failed: {e}")
                    time.sleep(0.5)
                    continue
                consecutive_failures = 0
                head_combined = cv2.hconcat(head_frames)

                wrist_combined = None
                if self.wrist_cameras:
                    wrist_frames, _ = self._capture_frames(self.wrist_cameras, "Wrist")
                    if wrist_frames:
                        wrist_combined = cv2.hconcat(wrist_frames)

                jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                ret_h, head_buf = cv2.imencode('.jpg', head_combined, jpg_params)
                if not ret_h:
                    logger_mp.error("[Image Server] Head imencode failed.")
                    continue

                payload = {
                    'image': head_buf.tobytes(),
                    'depth_raw': raw_depth,
                    'intrinsics': self.head_intrinsics,
                }
                if wrist_combined is not None:
                    ret_w, wrist_buf = cv2.imencode('.jpg', wrist_combined, jpg_params)
                    if ret_w:
                        payload['wrist_image'] = wrist_buf.tobytes()

                self.socket.send(pickle.dumps(payload), flags=zmq.NOBLOCK if False else 0)

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
    parser.add_argument('--no-realsense', action='store_true', help='Disable RealSense head camera (wrist-only mode)')
    parser.add_argument('--depth-near', type=int, default=250, help='Depth near plane (mm)')
    parser.add_argument('--depth-far', type=int, default=4000, help='Depth far plane (mm)')
    parser.add_argument('--depth-style', type=str, default='turbo', choices=['3ddp', 'turbo', 'jet'],
                        help='3ddp=3D-Diffusion-Policy style (u,v,depth as RGB), turbo/jet=colormap')
    parser.add_argument('--resolution', type=str, default='640x480', help='Head camera: WxH')
    parser.add_argument('--wrist-resolution', type=str, default='640x480', help='Wrist camera: WxH')
    args = parser.parse_args()

    wrist_ports = [] if args.no_wrist or args.wrist == [-1] else args.wrist
    head_shape = _parse_resolution(args.resolution, [480, 640])
    wrist_shape = _parse_resolution(args.wrist_resolution, [480, 640])

    if args.no_realsense:
        if not wrist_ports:
            print("ERROR: --no-realsense requires at least one --wrist port (no head camera + no wrist = nothing to send).")
            exit(1)
        config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': wrist_shape,
            'head_camera_id_numbers': [wrist_ports[0]],
            'enable_depth': False,
        }
        extra_wrist_ports = wrist_ports[1:]
        if extra_wrist_ports:
            config['wrist_camera_type'] = 'opencv'
            config['wrist_camera_image_shape'] = wrist_shape
            config['wrist_camera_id_numbers'] = extra_wrist_ports

        print("[CONFIG] RealSense: DISABLED (--no-realsense)")
        print(f"[CONFIG] Head (OpenCV) port: {wrist_ports[0]} @ {wrist_shape[1]}x{wrist_shape[0]}")
        print(f"[CONFIG] Wrist: {extra_wrist_ports if extra_wrist_ports else 'DISABLED'}")
    else:
        # Auto-detect RealSense
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("ERROR: No RealSense camera found!")
            exit(1)
        serial = devices[0].get_info(rs.camera_info.serial_number)
        print(f"[AUTO] RealSense: {serial}")

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
