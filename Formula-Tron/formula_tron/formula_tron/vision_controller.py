#!/usr/bin/env python3
"""
Formula-Tron Vision Controller

Main node that processes camera images and controls the car.
Uses track detection, safety checks, and sends commands to VESC.
Supports POLY_LOOKAHEAD (PD with Polynomial Lookahead) and LEGACY (Classic Histogram PD) control modes.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool, Float64, Float64MultiArray, Int32, String
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import cv2  # Must import before cv_bridge on Jetson (cv_bridge_boost init order)
from cv_bridge import CvBridge
import numpy as np
import time
import math

from . import config
from .utils.track_detection import TrackDetector, TrackDetectionResult
from .utils.lap_timer import AprilTagLapTimer
from .utils.safety import (
    SafetyValidator,
    WatchdogTimer,
    ExponentialMovingAverage,
    ConnectionMonitor,
    safe_normalize,
    ObstacleDetector
)
from .utils.telemetry import TelemetryRecord, create_vision_mode_data
from .utils.base_controller import ControlState, ControlOutput
from .utils.pd_controller import PDController
from .utils.controller_registry import ControllerRegistry

try:
    from .utils.apex_pursuit_controller import ApexPursuitController
    _APEX_PURSUIT_AVAILABLE = True
except ImportError:
    _APEX_PURSUIT_AVAILABLE = False

try:
    from .utils.mpc_controller import MPCController
    _MPC_AVAILABLE = True
except ImportError:
    _MPC_AVAILABLE = False




class VisionController(Node):
    def __init__(self):
        super().__init__('formula_tron_vision')

        # Topics (from config)
        self.camera_topic = self._detect_camera_topic()
        self.motor_topic = config.MOTOR_TOPIC
        self.servo_topic = config.SERVO_TOPIC
        
        # VESC Parameters (from config)
        self.speed_to_erpm_gain = config.VESC_SPEED_TO_ERPM_GAIN
        self.speed_to_erpm_offset = config.VESC_SPEED_TO_ERPM_OFFSET
        self.steering_to_servo_gain = config.VESC_STEERING_TO_SERVO_GAIN
        self.steering_to_servo_offset = config.VESC_STEERING_TO_SERVO_OFFSET
        self.max_steering_angle = config.MAX_STEERING_ANGLE_EFFECTIVE
        self.steering_min_angle = -self.max_steering_angle
        self.steering_max_angle = self.max_steering_angle
        
        # Tuning parameters (can be changed from GUI)
        self.declare_parameter('kp', config.KP_DEFAULT)
        self.declare_parameter('kd', config.KD_DEFAULT)
        self.declare_parameter('base_speed', config.BASE_SPEED)
        self.declare_parameter('turn_slowdown', config.TURN_SLOWDOWN)
        self.declare_parameter('track_width', config.VISUAL_TRACK_WIDTH)
        self.declare_parameter('steering_bias', config.STEERING_BIAS_DEFAULT)
        
        # HSV Defaults (from config)
        self.declare_parameter('hsv_h_min', config.HSV_H_MIN)
        self.declare_parameter('hsv_h_max', config.HSV_H_MAX)
        self.declare_parameter('hsv_s_min', config.HSV_S_MIN)
        self.declare_parameter('hsv_v_min', config.HSV_V_MIN)
        
        # State variables
        self.autonomous_enabled = False
        self.autonomous_running = False
        self.first_run = True  # Flag to handle initial controller state
        self.control_mode = "POLY_LOOKAHEAD"  # "POLY_LOOKAHEAD" or "LEGACY"
        
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.base_speed = self.get_parameter('base_speed').value
        self.turn_slowdown = self.get_parameter('turn_slowdown').value
        self.expected_width = self.get_parameter('track_width').value
        self.steering_bias = self.get_parameter('steering_bias').value
        
        # HSV
        h_min = self.get_parameter('hsv_h_min').value
        h_max = self.get_parameter('hsv_h_max').value
        s_min = self.get_parameter('hsv_s_min').value
        v_min = self.get_parameter('hsv_v_min').value
        self.hsv_lower = np.array([h_min, s_min, v_min])
        self.hsv_upper = np.array([h_max, 255, 255])
        
        # Control vars
        self.prev_error = 0.0
        self.last_time = time.time()
        self.last_steering = 0.0  # For rate limiting
        self.current_speed = 0.0
        self.current_steering = 0.0
        self._lla_last_healthy_time = time.time()
        self._lla_runtime_state = "OK"
        self._lla_runtime_reason = ""
        self._runtime_source = "controller"
        
        # Initialize Utility Modules
        # Track Detector (Option 2 naming)
        self.track_detector = TrackDetector(
            hsv_lower=np.array([config.HSV_H_MIN, config.HSV_S_MIN, config.HSV_V_MIN]),
            hsv_upper=np.array([config.HSV_H_MAX, 255, 255]),
            track_width=config.VISUAL_TRACK_WIDTH,
            roi_ratio=config.ROI_HEIGHT_RATIO,
            bev_top_width=config.BEV_TOP_WIDTH,
            bev_padding=config.BEV_PADDING,
            lookahead_ratio=config.LOOKAHEAD_RATIO,
            physical_track_width=config.PHYSICAL_TRACK_WIDTH,
        )
        self.target_smoother = ExponentialMovingAverage(
            alpha=config.TARGET_SMOOTHING_ALPHA,
            max_jump=150
        )
        
        # Controller Registry (replaces if/elif chain for control modes)
        self.registry = ControllerRegistry()
        
        # Rev0 Controllers
        poly_pd = PDController("POLY_LOOKAHEAD", kp=self.kp, kd=self.kd, steering_bias=self.steering_bias)
        legacy_pd = PDController("LEGACY", detection_mode="LEGACY", kp=self.kp, kd=self.kd, steering_bias=self.steering_bias)
        self.registry.register(poly_pd)
        self.registry.register(legacy_pd)
        
        # Rev1 Controllers — Apex Pursuit
        if _APEX_PURSUIT_AVAILABLE:
            self.registry.register(ApexPursuitController(
                wheelbase=config.WHEELBASE_METERS,
                min_lookahead=0.5,
                lookahead_gain=0.6,
                friction_coeff=getattr(config, 'LLA_DEFAULT_FRICTION', 0.8)
            ))
            self.get_logger().info('Registered: Apex Pursuit')
        
        # Rev1 Controllers — MPC-Standard (fixed friction model)
        if _MPC_AVAILABLE:
            self.registry.register(MPCController(
                friction=getattr(config, 'LLA_DEFAULT_FRICTION', 0.8),
                horizon=getattr(config, 'LLA_HORIZON', 5),
                dt=getattr(config, 'LLA_DT', 0.05),
                max_speed=config.BASE_SPEED,
            ))
            self.get_logger().info('Registered: MPC-Standard')
        

        

        
        # Broadcast initial parameters to all controllers
        self.registry.update_all_params({
            'kp': self.kp,
            'kd': self.kd,
            'steering_bias': self.steering_bias,
            'base_speed': self.base_speed
        })
        
        self.safety_validator = SafetyValidator()
        self.watchdog = WatchdogTimer(
            timeout=config.VISION_TIMEOUT_SEC,
            max_consecutive_errors=config.MAX_CONSECUTIVE_ERRORS,
        )
        self.camera_monitor = ConnectionMonitor(
            timeout=1.0,
            frame_drop_threshold=config.FRAME_DROP_WARNING_INTERVAL,
        )
        
        # Lap Counting (disappearance-based detection to prevent false positives at low speeds)
        self.lap_timer = AprilTagLapTimer(
            tag_id=config.LAP_TAG_ID,
            min_lap_time=config.MIN_LAP_TIME,
            tag_size_m=config.LAP_TAG_SIZE_M,
            max_dist=config.LAP_MAX_DIST,
            min_frames_without_tag=config.LAP_MIN_FRAMES_WITHOUT_TAG,
            min_frames_with_tag=config.LAP_MIN_FRAMES_WITH_TAG,
            min_tag_pixel_width=config.LAP_MIN_TAG_PIXEL_WIDTH,
        )
        self.frame_count = 0  # For frame skipping
        
        # Depth Safety (Obstacle Avoidance) - State-of-the-art detector
        self.min_safe_distance = config.MIN_SAFE_DISTANCE_DEFAULT
        self.obstacle_detector = ObstacleDetector(
            min_safe_distance=config.MIN_SAFE_DISTANCE_DEFAULT,
            min_valid_depth_mm=200,  # 20cm minimum
            max_valid_depth_mm=4000,  # 4m maximum
            consecutive_frames_required=3
        )
        self.obstacle_detected = False
        self.last_obstacle_distance = float('inf')
        self.obstacle_confidence = 0.0
        
        # Debug image control
        self.debug_enabled = config.DEBUG_ENABLED_DEFAULT
        self.debug_frame_skip = config.DEBUG_FRAME_SKIP
        self.debug_frame_counter = 0
        
        # AprilTag visualization control
        self.april_tag_view_enabled = False
        self.april_tag_frame_skip = config.DEBUG_FRAME_SKIP  # Same skip rate as debug
        self.april_tag_frame_counter = 0

        # ROS Setup
        self.bridge = CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, qos)
        self.auto_sub = self.create_subscription(Bool, '/autonomous_enabled', self.auto_callback, 10)
        self.auto_start_sub = self.create_subscription(Bool, '/autonomous_start', self.auto_start_callback, 10)
        
        # Tuning Subs
        self.pd_sub = self.create_subscription(Float64MultiArray, '/tuning/pd', self.pd_callback, 10)
        self.hsv_sub = self.create_subscription(Float64MultiArray, '/tuning/hsv', self.hsv_callback, 10)
        self.speed_sub = self.create_subscription(Float64, '/tuning/auto_speed', self.speed_callback, 10)
        self.width_sub = self.create_subscription(Float64, '/tuning/track_width', self.width_callback, 10)
        self.bias_sub = self.create_subscription(Float64, '/tuning/steering_bias', self.bias_callback, 10)
        self.slowdown_sub = self.create_subscription(Float64, '/tuning/turn_slowdown', self.slowdown_callback, 10)
        self.smoothing_sub = self.create_subscription(Float64, '/tuning/smoothing_alpha', self.smoothing_callback, 10)
        self.mode_sub = self.create_subscription(String, '/tuning/control_mode', self.mode_callback, 10)
        self.debug_sub = self.create_subscription(Bool, '/debug_enabled', self.debug_callback, 10)
        self.april_tag_view_sub = self.create_subscription(Bool, '/april_tag_view_enabled', self.april_tag_view_callback, 10)
        
        # POLY_LOOKAHEAD Mode Tuning Subs
        self.lookahead_sub = self.create_subscription(Float64, '/tuning/lookahead_ratio', self.lookahead_callback, 10)
        self.bev_top_width_sub = self.create_subscription(Float64, '/tuning/bev_top_width', self.bev_top_width_callback, 10)
        self.bev_padding_sub = self.create_subscription(Float64, '/tuning/bev_padding', self.bev_padding_callback, 10)
        self.auto_calibrate_sub = self.create_subscription(Bool, '/tuning/auto_calibrate_hsv', self.auto_calibrate_callback, 10)
        
        # MPC Tuning Subs
        self.mpc_horizon_sub = self.create_subscription(Int32, '/tuning/mpc_horizon', self.mpc_horizon_callback, 10)
        self.mpc_tracking_sub = self.create_subscription(Float64, '/tuning/mpc_tracking', self.mpc_tracking_callback, 10)
        self.mpc_smoothness_sub = self.create_subscription(Float64, '/tuning/mpc_smoothness', self.mpc_smoothness_callback, 10)
        
        # Supervised MPC Tuning Sub
        self.smpc_tuning_sub = self.create_subscription(String, '/tuning/smpc_param', self.smpc_param_callback, 10)
        self.smpc_steer_assist = 0.0
        self.smpc_steer_mode = "blend"  # "blend" or "override"
        self.smpc_speed_override = None  # None=MPC auto, float=locked preset
        
        # Depth Safety (Obstacle Avoidance) - State-of-the-art detector
        self.depth_sub = self.create_subscription(
            Image, self._detect_depth_topic(), self.depth_callback, qos)
        self.depth_distance_sub = self.create_subscription(
            Float64, '/tuning/min_safe_distance', self.depth_distance_callback, 10)
        self.obstacle_enable_sub = self.create_subscription(
            Bool, '/obstacle_detection_enabled', self.obstacle_enable_callback, 10)
        
        # Joystick Override (Deadman Switch)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.deadman_button = 4  # LB - deadman switch (hold to drive manually)
        self.last_joy_override_time = 0.0

        # Odometry subscriber (for telemetry)
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_heading = 0.0
        
        # Subscribe to common F1TENTH odom topics with compatible QoS
        from rclpy.qos import qos_profile_sensor_data
        self.odom_subs = []
        for topic in ['/odom', '/vesc/odom', '/ego_racecar/odom']:
            self.odom_subs.append(
                self.create_subscription(Odometry, topic, self.odom_callback, qos_profile_sensor_data)
            )

        # Publishers
        self.motor_pub = self.create_publisher(Float64, self.motor_topic, 10)
        self.servo_pub = self.create_publisher(Float64, self.servo_topic, 10)
        debug_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.debug_pub = self.create_publisher(Image, '/vision_debug', debug_qos)
        self.april_tag_pub = self.create_publisher(Image, '/vision_april_tag', debug_qos)
        self.lap_pub = self.create_publisher(Int32, '/lap_count', 10)
        self.lap_time_pub = self.create_publisher(Float64, '/lap_time', 10)
        
        # Status feedback publisher (for GUI sync)
        self.status_pub = self.create_publisher(Bool, '/autonomous_status', 10)
        
        # Telemetry publisher (JSON string for flexibility)
        self.telemetry_pub = self.create_publisher(String, '/telemetry', 10)
        self.telemetry_frame_count = 0
        self.frame_start_time = 0.0
        
        # Control Timer (100 Hz for snappier response, match joystick feel)
        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info(f'Vision Controller Started (Mode: {self.control_mode})')
        self.get_logger().info(f'Using camera topic: {self.camera_topic}')
        self.get_logger().info(f'Max steering (effective): +/-{self.max_steering_angle:.3f} rad')
        self.get_logger().info(f'Depth safety enabled: stop distance = {self.min_safe_distance:.2f}m')
        self.get_logger().info('Obstacle detection: State-of-the-art (trapezoidal ROI, percentile filtering, temporal smoothing)')
    
    def _detect_camera_topic(self):
        """Auto-detect which camera topic exists (handles both single and double namespace)."""
        import subprocess
        try:
            result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            topics = result.stdout.split('\n')
            
            # Check for double namespace first (most common)
            if '/camera/camera/color/image_raw' in topics:
                return '/camera/camera/color/image_raw'
            # Fallback to single namespace
            elif '/camera/color/image_raw' in topics:
                return '/camera/color/image_raw'
            # Default to config (will fail gracefully if not found)
            else:
                self.get_logger().warn(f'Camera topic not found, using config default: {config.CAMERA_TOPIC}')
                return config.CAMERA_TOPIC
        except Exception as e:
            self.get_logger().warn(f'Could not detect camera topic, using config default: {config.CAMERA_TOPIC} ({e})')
            return config.CAMERA_TOPIC
    
    def _detect_depth_topic(self):
        """Auto-detect which depth topic exists (handles both single and double namespace)."""
        import subprocess
        try:
            result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            topics = result.stdout.split('\n')
            
            # Check for double namespace first (most common)
            if '/camera/camera/depth/image_rect_raw' in topics:
                return '/camera/camera/depth/image_rect_raw'
            # Fallback to single namespace
            elif '/camera/depth/image_rect_raw' in topics:
                return '/camera/depth/image_rect_raw'
            # Default to config
            else:
                self.get_logger().warn(f'Depth topic not found, using config default: {config.DEPTH_TOPIC}')
                return config.DEPTH_TOPIC
        except Exception as e:
            self.get_logger().warn(f'Could not detect depth topic, using config default: {config.DEPTH_TOPIC} ({e})')
            return config.DEPTH_TOPIC

    def _safe_get_controller(self):
        """Get the active controller, falling back to POLY_LOOKAHEAD if unavailable."""
        # SUPERVISED_MPC uses the MPC controller internally
        lookup_mode = self.control_mode
        if lookup_mode == "SUPERVISED_MPC":
            lookup_mode = "MPC"
        if self.registry.has(lookup_mode):
            return self.registry.get(lookup_mode)
        # Mode not registered — fall back
        self.get_logger().warn(
            f"Mode '{self.control_mode}' not registered, "
            f"falling back to POLY_LOOKAHEAD",
            throttle_duration_sec=10.0,
        )
        return self.registry.get("POLY_LOOKAHEAD")

    def process_track_lines(self, frame):
        """Find track lines using current mode."""
        try:
            # Use the active controller's declared detection mode
            controller = self._safe_get_controller()
            detect_mode = controller.detection_mode
            return self.track_detector.detect(frame, mode=detect_mode)
        except Exception as e:
            # Return safe default object
            h, w = frame.shape[:2] if frame is not None and frame.size > 0 else (480, 640)
            safe_mask = np.zeros((h//4, w, 1), dtype=np.uint8)
            safe_hist = np.zeros(w, dtype=np.float32)
            return TrackDetectionResult(
                target_x=None, left_peak=None, right_peak=None,
                mask=safe_mask, histogram=safe_hist,
                all_peaks=np.array([]), used_peaks=np.array([]),
                status=f"ERROR: {str(e)}"
            )
    
    def image_callback(self, msg):
        """Process image and calculate controls."""
        self.frame_start_time = time.time()  # For telemetry processing time measurement
        self.frame_count += 1
        if self.watchdog.should_stop() and self.autonomous_running:
            self.get_logger().error('Watchdog timeout - stopping', throttle_duration_sec=2.0)
            self.autonomous_running = False
            self.autonomous_enabled = False
            self.publish_drive(0.0, 0.0)
            return
        
        # Check connection
        self.camera_monitor.on_frame()
        
        try:
            # Decode
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.watchdog.on_error()
                return
            
            if not self.safety_validator.validate_frame(frame):
                self.watchdog.on_error()
                return

            # Check for Lap (every 3rd frame to save CPU)
            # Only check when actually driving autonomously
            if self.autonomous_running and (self.frame_count % 3 == 0):
                is_lap, lap_time, count = self.lap_timer.check_lap(frame)
                # CRITICAL: Only publish when a new lap is completed AND lap_time > 0
                # This prevents publishing current lap duration or invalid times
                if is_lap and lap_time > 0.0:
                    self.lap_pub.publish(Int32(data=count))
                    self.lap_time_pub.publish(Float64(data=lap_time))
                    self.get_logger().info(f"🏁 LAP {count} COMPLETED! Time: {lap_time:.3f}s")
                    

                    # Forward lap count to MPC for per-lap speed/smoothness ramp
                    if self.control_mode in ("MPC", "SUPERVISED_MPC") and self.registry.has("MPC"):
                        ctrl = self.registry.get("MPC")
                        ctrl.on_lap_completed(count)
            
            # AprilTag Visualization (when enabled)
            if self.april_tag_view_enabled:
                self.april_tag_frame_counter += 1
                if self.april_tag_frame_counter >= self.april_tag_frame_skip:
                    self.april_tag_frame_counter = 0
                    try:
                        # Detect tags for visualization (doesn't count laps)
                        corners, ids, rejected = self.lap_timer.detect_tags_for_visualization(frame)
                        # Draw visualization
                        vis_frame = self.lap_timer.draw_april_tag_visualization(frame.copy(), corners, ids, rejected)
                        # Publish visualization
                        self.april_tag_pub.publish(self.bridge.cv2_to_imgmsg(vis_frame, "bgr8"))
                    except Exception as e:
                        self.get_logger().warn(f'AprilTag visualization error: {e}', throttle_duration_sec=5.0)
            
            h, w = frame.shape[:2]
            center_x = w // 2
        
            # Detect
            try:
                result = self.process_track_lines(frame)
            except Exception as e:
                self.watchdog.on_error()
                self.get_logger().error(f'Detection failed: {e}')
                return # Skip frame
            
            target_x = result.target_x
            
            # Auto-calibration processing
            if getattr(self, '_calibrate_requested', False):
                self._calibrate_requested = False
                self.get_logger().info('Running Auto-Calibration on current frame...')
                calib_result = self.track_detector.auto_calibrate(frame)
                if calib_result:
                    h_min = calib_result["hsv_h_min"]
                    h_max = calib_result["hsv_h_max"]
                    s_min = calib_result["hsv_s_min"]
                    v_min = calib_result["hsv_v_min"]
                    self.get_logger().info(f'Calibration successful! Bounds: '
                                           f'H[{h_min}-{h_max}] '
                                           f'S[{s_min}-255] '
                                           f'V[{v_min}-255]')
                    # Update local state (track_detector already updated internally by auto_calibrate)
                    self.hsv_lower = np.array([h_min, s_min, v_min])
                    self.hsv_upper = np.array([h_max, 255, 255])
                else:
                    self.get_logger().warn('Calibration failed: Unable to find valid bounds.')
            
            # Smoothing
            if target_x is not None:
                target_x = self.target_smoother.update(target_x)
            else:
                smoothed = self.target_smoother.get()
                target_x = smoothed if smoothed is not None else center_x
            
            # Control Logic
            steering = 0.0
            speed = 0.0
            
            # Reset controller state on first run to prevent derivative kick and rate limiting issues
            if self.first_run:
                init_error = safe_normalize(target_x, center_x, w / 2.0)
                self.prev_error = init_error
                self.last_steering = 0.0  # Start from neutral steering
                self.last_time = time.time()  # Reset time reference
                
                # Prime the controller with initial error to prevent derivative kick
                try:
                    self._safe_get_controller().reset(initial_error=init_error)
                except Exception:
                    pass
                    
                self.first_run = False
                self.get_logger().info(f'First run init: error={init_error:.3f}, target_x={target_x:.1f}')
            
            try:
                now = time.time()
                dt = max(now - self.last_time, 0.001)
                dt = min(dt, 1.0)
                
                # Calculate visual track offset (-0.5 to 0.5)
                # target_x is the track center. center_x is the camera center (w/2)
                # Negative = track is to the left (car is right of center)
                # Positive = track is to the right (car is left of center)
                track_offset_px = target_x - center_x
                track_offset_norm = track_offset_px / float(w)
                

                # --- CONTROLLER DISPATCH (via registry) ---
                controller = self._safe_get_controller()
                state = ControlState(
                    target_x=target_x,
                    center_x=center_x,
                    frame_width=float(w),
                    current_speed=self.current_speed,
                    detection=result,
                    current_steering=self.current_steering,
                    autonomous_running=self.autonomous_running,
                    raw_frame=frame,
                    beta_path=self._build_beta_path(controller, result),
                    base_speed=float(self.base_speed),
                    odom_x=self.odom_x,
                    odom_y=self.odom_y,
                    odom_heading=self.odom_heading,
                    track_offset=track_offset_norm,
                    auto_steer_recommendation=auto_steer_rec,
                )
                output = controller.compute(state, dt)

                is_lla_controller = (self.control_mode == "LLA_MPC" and getattr(controller, "name", "") == "LLA_MPC")
                if is_lla_controller and hasattr(controller, "get_status"):
                    try:
                        lla_status = controller.get_status()
                    except Exception:
                        lla_status = {}
                    output = self._apply_lla_fallback(output, state, dt, now, lla_status)
                    if self._lla_runtime_state != "OK":
                        self.get_logger().warn(
                            f"LLA degraded ({self._lla_runtime_state}): {self._lla_runtime_reason}",
                            throttle_duration_sec=1.0,
                        )
                else:
                    self._runtime_source = getattr(controller, "name", self.control_mode)
                    self._lla_runtime_state = "OK"
                    self._lla_runtime_reason = ""

                steering_target = output.steering

                # Global limits and rate limiting (applies to ALL controllers)
                if np.isnan(steering_target) or np.isinf(steering_target):
                    steering_target = 0.0
                steering_target = max(self.steering_min_angle, min(self.steering_max_angle, steering_target))

                # Rate limiting
                max_change = config.MAX_STEERING_RATE * dt
                delta = steering_target - self.last_steering
                if abs(delta) > max_change:
                    steering_target = self.last_steering + max_change * (1.0 if delta > 0 else -1.0)
                steering = steering_target
                self.last_steering = steering
                self.last_time = now

                # --- Supervised MPC steering assist ---
                if self.control_mode == "SUPERVISED_MPC" and self.smpc_steer_assist != 0.0:
                    if self.smpc_steer_mode == "override":
                        # Full replacement: human steering replaces MPC
                        steering = self.smpc_steer_assist
                    else:
                        # Blend: add human delta to MPC's output
                        steering = steering + self.smpc_steer_assist
                    # Re-clamp after assist
                    steering = max(self.steering_min_angle, min(self.steering_max_angle, steering))
                    self.last_steering = steering

                # Speed: use controller's speed if it manages its own, otherwise generic formula.
                # IMPORTANT: Use post-limited steering to avoid throttle/brake pulsing from transient
                # controller spikes that are never actually sent to the servo.
                if is_lla_controller:
                    speed = output.speed
                elif controller.manages_speed:
                    speed = output.speed
                else:
                    max_steer = self.max_steering_angle
                    turn_factor = abs(steering) / max_steer if max_steer > 0 else 0.0
                    speed = self.base_speed * (1.0 - self.turn_slowdown * turn_factor)
                    speed = max(0.5, min(5.0, speed))

                if np.isnan(speed) or np.isinf(speed):
                    speed = self.current_speed

                # --- COLD-START SPEED FLOOR for beta controllers ---
                # When a beta controller has a valid path but outputs near-zero
                # speed (common at startup when current_speed ≈ 0), enforce a
                # minimum of 0.5 m/s to break the cold-start death spiral.
                # Standard F1TENTH practice: start at a safe constant speed.
                BETA_MIN_SPEED = 0.5
                ctrl_name = getattr(controller, 'name', '')
                if (controller.manages_speed
                        and (ctrl_name.startswith('BETA_') or ctrl_name == 'APEX_PURSUIT')
                        and self.autonomous_running
                        and speed < BETA_MIN_SPEED):
                    speed = BETA_MIN_SPEED

                speed = self._limit_speed_rate(float(speed), dt)

                # --- Supervised MPC speed override ---
                if self.control_mode == "SUPERVISED_MPC" and self.smpc_speed_override is not None:
                    speed = float(self.smpc_speed_override)

            except Exception as e:
                self.watchdog.on_error()
                self.get_logger().error(f'Control logic failed: {e}')
                steering = 0.0
                speed = 0.0
            
            if self.autonomous_running:
                self.current_speed = speed
                self.current_steering = steering
            
            self.watchdog.on_success()
            
            # Emit Telemetry
            self._emit_telemetry(result, target_x, center_x, speed, steering, w)
            
            # Debug Image
            if self.debug_enabled:
                self.debug_frame_counter += 1
                if self.debug_frame_counter >= self.debug_frame_skip:
                    self.debug_frame_counter = 0
                    self.publish_debug_image(frame, result, target_x, center_x)
                
        except Exception as e:
            self.watchdog.on_error()
            self.get_logger().error(f'Image callback critical: {e}')

    def publish_debug_image(self, frame, result, target_x, center_x):
        try:
            self._publish_legacy_debug(frame, result, target_x, center_x)
        except Exception:
            pass

    def _publish_legacy_debug(self, frame, result, target_x, center_x):
        """Original debug 2×2 for non-MPC modes."""
        h, w = frame.shape[:2]
        half_h, half_w = h // 2, w // 2

        # Top-Left: Raw
        tl = cv2.resize(frame.copy(), (half_w, half_h))
        cv2.putText(tl, "RAW", (10, half_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # Top-Right: Mask or BEV
        if self.control_mode == "POLY_LOOKAHEAD" and result.bev_mask is not None:
            bev_h, bev_w = result.bev_mask.shape[:2]
            mask_display = cv2.cvtColor(result.bev_mask, cv2.COLOR_GRAY2BGR)
            tr = cv2.resize(mask_display, (half_w, half_h))
            cv2.putText(tr, "BIRD'S EYE VIEW", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            if result.poly_coeffs is not None:
                plot_y = np.linspace(0, bev_h-1, num=50)
                plot_x = result.poly_coeffs[0]*plot_y**2 + result.poly_coeffs[1]*plot_y + result.poly_coeffs[2]
                scale_x = half_w / bev_w
                scale_y = half_h / bev_h
                points = []
                for i in range(len(plot_y)):
                    px = int(plot_x[i] * scale_x)
                    py = int(plot_y[i] * scale_y)
                    if 0 <= px < half_w and 0 <= py < half_h:
                        points.append((px, py))
                if len(points) > 1:
                    cv2.polylines(tr, [np.array(points)], False, (0, 0, 255), 2)
                if result.target_x_bev is not None:
                    look_y = int((bev_h * (1.0 - self.track_detector.lookahead_ratio)) * scale_y)
                    look_x = int(result.target_x_bev * scale_x)
                    cv2.circle(tr, (look_x, look_y), 5, (0, 255, 0), -1)
        else:
            mask_display = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
            tr = cv2.resize(mask_display, (half_w, half_h))
            cv2.putText(tr, "MASK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # Bottom-Left: Histogram
        bl = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        if len(result.histogram) > 0:
            try:
                hist_2d = result.histogram.reshape(1, -1).astype(np.float32)
                hist_resized = cv2.resize(hist_2d, (half_w, 1), interpolation=cv2.INTER_LINEAR).flatten()
                hist_max = np.max(hist_resized)
                if hist_max > 0:
                    hist_norm = (hist_resized / hist_max * (half_h - 20)).astype(int)
                    for i in range(0, min(half_w, len(hist_norm)), 2):
                        val = min(hist_norm[i], half_h)
                        cv2.line(bl, (i, half_h), (i, half_h-val), (255, 0, 255), 1)
            except Exception:
                pass
        scale_x = half_w / w
        if len(result.all_peaks) > 0:
            for p in result.all_peaks:
                px = int(p * scale_x)
                if 0 <= px < half_w:
                    cv2.line(bl, (px, 0), (px, half_h), (0, 0, 255), 2)
        cv2.putText(bl, f"HIST ({len(result.all_peaks)} peaks)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Bottom-Right: Final
        br = cv2.resize(frame.copy(), (half_w, half_h))
        if target_x is not None:
            scale_x = half_w / w
            tx = int(target_x * scale_x)
            cv2.line(br, (tx, 0), (tx, half_h), (0, 255, 0), 3)
            cv2.circle(br, (tx, half_h//2), 8, (0, 255, 0), -1)
        cv2.putText(br, "CONTROL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Combine
        top = np.hstack((tl, tr))
        bottom = np.hstack((bl, br))
        grid = np.vstack((top, bottom))

        # Status Bar
        cv2.rectangle(grid, (5, 5), (350, 60), (0,0,0), -1)
        cv2.putText(grid, f"MODE: {self.control_mode} | {result.status}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(grid, f"STR: {self.current_steering:.2f} SPD: {self.current_speed:.1f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(grid, "bgr8"))

    def _publish_mpc_debug(self, frame, result, target_x, center_x):
        """MPC-specific debug 2×2: Raw | BEV+Trajectory | Cost Heatmap | Horizon Chart."""
        ctrl = self.registry.get("MPC")
        dbg = ctrl.get_debug_data() if hasattr(ctrl, 'get_debug_data') else {}
        h, w = frame.shape[:2]
        half_h, half_w = h // 2, w // 2

        # ═══ TL: Raw Camera ═══
        tl = cv2.resize(frame.copy(), (half_w, half_h))
        # Overlay current steering/speed HUD
        cv2.rectangle(tl, (0, 0), (half_w, 32), (0, 0, 0), -1)
        cost_str = f"J={dbg['solver_cost']:.1f}" if dbg.get('solver_cost') is not None else "J=--"
        cv2.putText(tl, f"MPC | STR:{self.current_steering:.2f} SPD:{self.current_speed:.1f} {cost_str}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ═══ TR: BEV + MPC Trajectory ═══
        tr = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        # Draw BEV mask as background if available
        if result.bev_mask is not None:
            bev_bg = cv2.cvtColor(result.bev_mask, cv2.COLOR_GRAY2BGR)
            bev_bg = cv2.resize(bev_bg, (half_w, half_h))
            tr = (bev_bg * 0.3).astype(np.uint8)  # dim background

        pred_xy = dbg.get('predicted_xy')
        ref_xy = dbg.get('ref_xy')
        pred_v = dbg.get('predicted_v')
        v_refs = dbg.get('v_refs')
        max_spd = dbg.get('max_speed', 3.0)

        if ref_xy is not None and pred_xy is not None:
            # Auto-scale: fit both ref and pred into the view
            all_pts = np.vstack([ref_xy, pred_xy])
            x_min, x_max = all_pts[:, 0].min() - 0.1, all_pts[:, 0].max() + 0.1
            y_min, y_max = all_pts[:, 1].min() - 0.1, all_pts[:, 1].max() + 0.1
            x_range = max(x_max - x_min, 0.01)
            y_range = max(y_max - y_min, 0.01)
            scale = min((half_w - 20) / x_range, (half_h - 40) / y_range)

            def to_px(xy):
                px = int((xy[0] - x_min) * scale + 10)
                py = int((half_h - 20) - (xy[1] - y_min) * scale)
                return (px, py)

            # Draw reference path (dashed blue)
            ref_pts = [to_px(ref_xy[i]) for i in range(len(ref_xy))]
            for i in range(0, len(ref_pts) - 1, 2):
                cv2.line(tr, ref_pts[i], ref_pts[min(i+1, len(ref_pts)-1)], (180, 120, 40), 1)

            # Draw predicted path as color-gradient (green=fast, red=slow)
            for i in range(len(pred_xy) - 1):
                p1 = to_px(pred_xy[i])
                p2 = to_px(pred_xy[i + 1])
                speed_frac = min(pred_v[i] / max(max_spd, 0.01), 1.0) if pred_v is not None else 0.5
                # Green (fast) → Red (slow)
                r = int(255 * (1.0 - speed_frac))
                g = int(255 * speed_frac)
                cv2.line(tr, p1, p2, (0, g, r), 3)

            # Ghost dots at each horizon step
            for i in range(len(pred_xy)):
                pt = to_px(pred_xy[i])
                speed_frac = min(pred_v[i] / max(max_spd, 0.01), 1.0) if pred_v is not None else 0.5
                radius = max(2, int(4 * speed_frac + 2))
                r = int(255 * (1.0 - speed_frac))
                g = int(255 * speed_frac)
                cv2.circle(tr, pt, radius, (0, g, r), -1)
                cv2.circle(tr, pt, radius, (255, 255, 255), 1)

        cv2.putText(tr, "BEV + TRAJECTORY", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # Legend
        cv2.putText(tr, "-- ref", (half_w - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 120, 40), 1)
        cv2.circle(tr, (half_w - 90, 35), 4, (0, 255, 0), -1)
        cv2.putText(tr, "fast", (half_w - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.circle(tr, (half_w - 90, 55), 4, (0, 0, 255), -1)
        cv2.putText(tr, "slow", (half_w - 80, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # ═══ BL: Cost Landscape Heatmap ═══
        bl = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        controls = dbg.get('controls')
        max_steer = dbg.get('max_steering', 0.4)

        if controls is not None and len(controls) > 0 and pred_v is not None:
            # Create a heatmap: X=steering, Y=speed
            # Sample a grid and compute approximate cost = w_steer*delta^2 + w_vel*(v-v_ref)^2
            grid_size = 64
            steer_range = np.linspace(-max_steer, max_steer, grid_size)
            speed_range = np.linspace(0.0, max_spd, grid_size)
            cost_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

            v_ref_0 = v_refs[0] if v_refs is not None and len(v_refs) > 0 else 1.0
            for si, s in enumerate(steer_range):
                for vi, v in enumerate(speed_range):
                    # Approximate single-step cost
                    cost = 20.0 * s**2 + 20.0 * (v - v_ref_0)**2 + 50.0 * (v * np.tan(s) / 0.33)**2
                    cost_grid[grid_size - 1 - vi, si] = cost

            # Normalize and colormap
            c_min, c_max = cost_grid.min(), cost_grid.max()
            if c_max > c_min:
                cost_norm = ((cost_grid - c_min) / (c_max - c_min) * 255).astype(np.uint8)
            else:
                cost_norm = np.zeros_like(cost_grid, dtype=np.uint8)
            heatmap = cv2.applyColorMap(cost_norm, cv2.COLORMAP_INFERNO)
            bl = cv2.resize(heatmap, (half_w, half_h))

            # Mark optimal point (first control action)
            opt_steer = controls[0, 1]
            opt_speed = pred_v[0]
            opt_px = int((opt_steer - (-max_steer)) / (2 * max_steer) * half_w)
            opt_py = int((1.0 - opt_speed / max(max_spd, 0.01)) * half_h)
            opt_px = max(6, min(half_w - 6, opt_px))
            opt_py = max(6, min(half_h - 6, opt_py))
            cv2.circle(bl, (opt_px, opt_py), 8, (255, 255, 255), 2)
            cv2.circle(bl, (opt_px, opt_py), 4, (0, 255, 255), -1)

            # Axis labels
            cv2.putText(bl, f"steer: [{-max_steer:.1f}, {max_steer:.1f}]",
                        (8, half_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            cv2.putText(bl, f"speed: [0, {max_spd:.1f}]",
                        (8, half_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        cv2.putText(bl, "COST LANDSCAPE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ═══ BR: Horizon Strip Chart ═══
        br = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        # Draw dark grid background
        for gy in range(0, half_h, half_h // 4):
            cv2.line(br, (0, gy), (half_w, gy), (30, 30, 30), 1)

        if controls is not None and len(controls) > 0 and pred_v is not None:
            N = len(controls)
            margin_l, margin_r = 40, 10
            margin_t, margin_b = 35, 20
            plot_w = half_w - margin_l - margin_r
            plot_h = half_h - margin_t - margin_b

            # Steering profile (cyan)
            steer_vals = controls[:, 1]
            steer_norm = steer_vals / max(max_steer, 0.01) * 0.5 + 0.5  # normalize to [0, 1]
            for i in range(N - 1):
                x1 = margin_l + int(i / max(N - 1, 1) * plot_w)
                x2 = margin_l + int((i + 1) / max(N - 1, 1) * plot_w)
                y1 = margin_t + int((1.0 - np.clip(steer_norm[i], 0, 1)) * plot_h)
                y2 = margin_t + int((1.0 - np.clip(steer_norm[i + 1], 0, 1)) * plot_h)
                cv2.line(br, (x1, y1), (x2, y2), (255, 200, 0), 2)  # cyan

            # Speed profile (magenta)
            speed_norm = pred_v[:N] / max(max_spd, 0.01)
            for i in range(N - 1):
                x1 = margin_l + int(i / max(N - 1, 1) * plot_w)
                x2 = margin_l + int((i + 1) / max(N - 1, 1) * plot_w)
                y1 = margin_t + int((1.0 - np.clip(speed_norm[i], 0, 1)) * plot_h)
                y2 = margin_t + int((1.0 - np.clip(speed_norm[i + 1], 0, 1)) * plot_h)
                cv2.line(br, (x1, y1), (x2, y2), (200, 0, 255), 2)  # magenta

            # Reference speed (dashed green)
            if v_refs is not None:
                vr_norm = v_refs[:N] / max(max_spd, 0.01)
                for i in range(0, N - 1, 2):
                    x1 = margin_l + int(i / max(N - 1, 1) * plot_w)
                    x2 = margin_l + int((i + 1) / max(N - 1, 1) * plot_w)
                    y1 = margin_t + int((1.0 - np.clip(vr_norm[i], 0, 1)) * plot_h)
                    y2 = margin_t + int((1.0 - np.clip(vr_norm[i + 1], 0, 1)) * plot_h)
                    cv2.line(br, (x1, y1), (x2, y2), (0, 100, 0), 1)

            # Zero line for steering
            zero_y = margin_t + plot_h // 2
            cv2.line(br, (margin_l, zero_y), (margin_l + plot_w, zero_y), (60, 60, 60), 1)

            # Y-axis labels
            cv2.putText(br, f"+{max_steer:.1f}", (2, margin_t + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 200, 0), 1)
            cv2.putText(br, "0", (2, zero_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1)
            cv2.putText(br, f"-{max_steer:.1f}", (2, margin_t + plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 200, 0), 1)
            cv2.putText(br, f"{max_spd:.0f}", (half_w - 28, margin_t + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 0, 255), 1)
            cv2.putText(br, "0", (half_w - 16, margin_t + plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 0, 255), 1)

        cv2.putText(br, "HORIZON PLAN", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        # Legend
        cv2.line(br, (half_w - 110, 12), (half_w - 90, 12), (255, 200, 0), 2)
        cv2.putText(br, "steer", (half_w - 85, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)
        cv2.line(br, (half_w - 55, 12), (half_w - 35, 12), (200, 0, 255), 2)
        cv2.putText(br, "speed", (half_w - 30, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 255), 1)

        # ═══ Combine ═══
        top = np.hstack((tl, tr))
        bottom = np.hstack((bl, br))
        grid = np.vstack((top, bottom))

        # Status Bar overlay
        cv2.rectangle(grid, (5, 5), (420, 30), (0, 0, 0), -1)
        hg_str = ""
        heading_gate = dbg.get('heading_gate')
        if heading_gate is not None and len(heading_gate) > 0:
            hg_str = f" | TURN:{heading_gate[0]:.2f}"
        cv2.putText(grid, f"MPC | {result.status}{hg_str}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(grid, "bgr8"))

    def control_loop(self):
        if self.autonomous_running and self.watchdog.should_stop():
            self.get_logger().error('Watchdog timeout - stopping', throttle_duration_sec=2.0)
            self.autonomous_running = False
            self.autonomous_enabled = False
            self.publish_drive(0.0, 0.0)
            return
        
        # Depth Safety: Stop if obstacle detected
        # Note: is_obstacle already requires 3 consecutive detections, so confidence is implicitly high
        if self.autonomous_running and self.obstacle_detected:
            self.get_logger().warn(
                f'🛑 OBSTACLE STOP! Distance: {self.last_obstacle_distance:.2f}m '
                f'< {self.min_safe_distance:.2f}m (confidence: {self.obstacle_confidence:.1%})',
                throttle_duration_sec=0.5
            )
            self.publish_drive(0.0, 0.0)
            return

        if self.autonomous_running:
            self.publish_drive(self.current_speed, self.current_steering)

    def publish_drive(self, speed, steering):
        try:
            steering = max(self.steering_min_angle, min(self.steering_max_angle, steering))
            erpm = self.speed_to_erpm_gain * speed + self.speed_to_erpm_offset
            servo = self.steering_to_servo_gain * steering + self.steering_to_servo_offset
            servo = max(config.SERVO_CMD_MIN, min(config.SERVO_CMD_MAX, servo))
            
            self.motor_pub.publish(Float64(data=float(erpm)))
            self.servo_pub.publish(Float64(data=float(servo)))
        except Exception:
            pass

    def _emit_telemetry(self, result: TrackDetectionResult, target_x: float, 
                        center_x: int, speed: float, steering: float, img_width: int):
        """Emit telemetry record for the current frame."""
        import json
        
        self.telemetry_frame_count += 1
        
        # Calculate processing time
        processing_time = (time.time() - self.frame_start_time) * 1000 if self.frame_start_time > 0 else 0.0
        
        # Calculate track error in pixels
        track_error = target_x - center_x if target_x is not None else 0.0
        
        # Determine safety state
        if self.obstacle_detected:
            safety_state = "STOPPED"
        elif self.control_mode == "LLA_MPC" and self._lla_runtime_state != "OK":
            safety_state = "WARNING"
        elif not self.watchdog.is_healthy():
            safety_state = "WARNING"
        else:
            safety_state = "OK"
        
        # Get detection mode from result
        detection_mode = result.detection_mode if hasattr(result, 'detection_mode') else "UNKNOWN"
        
        # Create mode-specific data for Vision mode
        mode_data = create_vision_mode_data(
            detection_mode=detection_mode,
            track_error_px=track_error,
            target_x=target_x if target_x else 0.0,
            left_poly=result.left_poly.tolist() if result.left_poly is not None else None,
            right_poly=result.right_poly.tolist() if result.right_poly is not None else None,
            center_poly=result.center_poly.tolist() if result.center_poly is not None else None,
        )
        mode_data['runtime_source'] = self._runtime_source
        mode_data['lla_runtime_state'] = self._lla_runtime_state
        if self._lla_runtime_reason:
            mode_data['lla_runtime_reason'] = self._lla_runtime_reason
        
        # Enrich with controller-specific status (RL training, LLA friction, etc.)
        mode_data['active_mode'] = self.control_mode
        mode_data['odom_x'] = self.odom_x
        mode_data['odom_y'] = self.odom_y
        mode_data['odom_heading'] = self.odom_heading
        
        # Calculate track offset for telemetry (same as passed to ControlState)
        _t_x = target_x if target_x else 0.0
        mode_data['track_offset'] = (_t_x - float(center_x)) / float(img_width) if img_width > 0 else 0.0
        if self.registry.has(self.control_mode):
            try:
                ctrl = self.registry.get(self.control_mode)
                if hasattr(ctrl, 'get_status'):
                    status = ctrl.get_status()
                    mode_data.update(status)

            except Exception:
                pass
        
        # Create telemetry record
        record = TelemetryRecord(
            timestamp=time.time(),
            frame_number=self.telemetry_frame_count,
            control_mode=self.control_mode,
            speed_cmd=speed,
            steering_cmd=steering,
            speed_actual=getattr(self, 'speed_actual_odom', 0.0),  # From /odom
            imu_accel_x=0.0,   # TODO: Subscribe to IMU for accel data
            imu_accel_y=0.0,
            imu_yaw_rate=0.0,
            lap_number=self.lap_timer.lap_count,
            lap_time_current=self.lap_timer.current_lap_time,
            lap_time_last=self.lap_timer.last_completed_lap_time,
            lap_time_best=self.lap_timer.best_lap_time,
            safety_state=safety_state,
            mode_data=mode_data,
            processing_time_ms=processing_time,
        )
        
        # Publish as JSON string
        try:
            record_dict = record.to_dict()
            # Handle infinity
            if record_dict.get('lap_time_best') == float('inf'):
                record_dict['lap_time_best'] = -1.0
                
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
                    
            json_str = json.dumps(record_dict, cls=NumpyEncoder)
            self.telemetry_pub.publish(String(data=json_str))
        except Exception as e:
            self.get_logger().warn(f'Telemetry publish error: {e}', throttle_duration_sec=5.0)

    def auto_callback(self, msg):
        self.autonomous_enabled = msg.data
        if not self.autonomous_enabled:
            # Exiting autonomous mode completely - clear state
            self.autonomous_running = False
            self.publish_drive(0.0, 0.0)
            self.target_smoother.reset()
            # Clear lap display (but don't reset timer - it will be reset on next start)
            self.lap_pub.publish(Int32(data=0))
            # Clear obstacle detector consecutive detection counter
            self.obstacle_detector.reset()
            self.get_logger().info('Autonomous mode disabled - state cleared')

    def odom_callback(self, msg):
        """Extract position and heading from /odom for telemetry."""
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_heading = math.atan2(siny_cosp, cosy_cosp)
        # Also store actual speed from odometry
        self.speed_actual_odom = msg.twist.twist.linear.x

    def auto_start_callback(self, msg):
        if self.autonomous_enabled:
            self.autonomous_running = msg.data
            if self.autonomous_running:
                self.watchdog.reset()
                self.lap_timer.reset()
                # FIX: Publish lap_count=0 to reset GUI's internal state (best_lap_time, etc.)
                # This ensures a fresh start when restarting driving within the same autonomous session
                self.lap_pub.publish(Int32(data=0))
                
                # FIX: Always reset smoother to clear stale data from when car was stopped
                # This prevents the "sharp turn on first start" bug where old garbage
                # target values (from floor/wall/ceiling while carrying the car) cause swerving
                self.target_smoother.reset()
                
                # FIX: Also clear track detector state to prevent "sticky" wrong strategies
                self.track_detector.last_strategy = "NONE"
                self.track_detector.strategy_lock_counter = 0
                
                # Reset obstacle detector state
                self.obstacle_detector.reset()
                

                
                self.first_run = True
                self._lla_last_healthy_time = time.time()
                self._lla_runtime_state = "OK"
                self._lla_runtime_reason = ""
                self._runtime_source = self.control_mode
                self.get_logger().info('Autonomous started - ALL state cleared for fresh start')
            else:
                self.publish_drive(0.0, 0.0)

    # Callbacks
    def pp_lookahead_callback(self, msg):
        self.registry.update_all_params({'pp_min_lookahead': msg.data})
        self.get_logger().info(f'Pure Pursuit Min Lookahead updated to: {msg.data:.2f}')

    def pp_gain_callback(self, msg):
        self.registry.update_all_params({'pp_velocity_gain': msg.data})
        self.get_logger().info(f'Pure Pursuit Velocity Gain updated to: {msg.data:.3f}')

    def stanley_k_callback(self, msg):
        self.registry.update_all_params({'stanley_k': msg.data})
        self.get_logger().info(f'Stanley Cross-Track Gain (k) updated to: {msg.data:.2f}')

    def stanley_ks_callback(self, msg):
        self.registry.update_all_params({'stanley_ks': msg.data})
        self.get_logger().info(f'Stanley Softening (ks) updated to: {msg.data:.2f}')

    def pd_callback(self, msg): 
        if len(msg.data) >= 2: 
            self.kp, self.kd = msg.data[0], msg.data[1]
            self.registry.update_all_params({'kp': self.kp, 'kd': self.kd})
    def hsv_callback(self, msg):
        """Update track detector HSV thresholds from GUI."""
        h_min, h_max, s_min, v_min = msg.data
        self.track_detector.update_hsv(h_min, h_max, s_min, v_min)
        # Sync local state for telemetry/testing
        self.hsv_lower = np.array([h_min, s_min, v_min])
        self.hsv_upper = np.array([h_max, 255, 255])
    def auto_calibrate_callback(self, msg):
        """Flag to run auto-calibration on the next received frame."""
        if msg.data:
            self._calibrate_requested = True

    def speed_callback(self, msg): 
        self.base_speed = msg.data
        self.registry.update_all_params({'base_speed': self.base_speed})
    def width_callback(self, msg): 
        self.expected_width = int(msg.data)
        self.track_detector.update_track_width(int(msg.data))
    def bias_callback(self, msg): 
        self.steering_bias = msg.data
        self.registry.update_all_params({'steering_bias': self.steering_bias})
    def slowdown_callback(self, msg): self.turn_slowdown = max(0.0, min(1.0, msg.data))
    def smoothing_callback(self, msg): 
        self.target_smoother.set_alpha(max(0.0, min(1.0, msg.data)))
    def mode_callback(self, msg):
        old_mode = self.control_mode
        new_mode = msg.data

        # Validate mode exists (or warn and keep using it via fallback)
        if not self.registry.has(new_mode):
            self.get_logger().error(
                f"Mode '{new_mode}' is not registered. "
                f"Car will fall back to POLY_LOOKAHEAD while mode is set to '{new_mode}'."
            )

        self.control_mode = new_mode

        # Clear ALL state when switching modes
        # This prevents stale data from one mode affecting the other
        if old_mode != self.control_mode:
            self.track_detector.reset_perspective_cache()
            self.target_smoother.reset()
            self.first_run = True
            self._lla_last_healthy_time = time.time()
            self._lla_runtime_state = "OK"
            self._lla_runtime_reason = ""
            self._runtime_source = self.control_mode
            try:
                self._safe_get_controller().reset()
            except Exception:
                pass
            legacy_note = " (Classic)" if self.control_mode == "LEGACY" else ""
            self.get_logger().info(f"Switched Control Mode to: {self.control_mode}{legacy_note} (ALL state cleared)")
    def debug_callback(self, msg): self.debug_enabled = msg.data
    
    def april_tag_view_callback(self, msg):
        """Enable/disable AprilTag visualization publishing."""
        self.april_tag_view_enabled = msg.data
        if msg.data:
            self.get_logger().info('AprilTag visualization enabled')
        else:
            self.april_tag_frame_counter = 0  # Reset counter when disabled
    def lookahead_callback(self, msg): 
        self.track_detector.update_lookahead_ratio(msg.data)
        self.get_logger().info(f"Lookahead ratio: {msg.data:.2f}")
    def bev_top_width_callback(self, msg): 
        self.track_detector.update_bev_top_width(msg.data)
        self.get_logger().info(f"BEV top width: {msg.data:.2f}")
    def bev_padding_callback(self, msg): 
        self.track_detector.update_bev_padding(msg.data)
        self.get_logger().info(f"BEV padding: {msg.data:.2f}")
    
    def depth_callback(self, msg):
        """Process depth image for obstacle detection using state-of-the-art methods."""
        try:
            # Convert depth image to numpy (16-bit depth in mm)
            depth_frame = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            
            # Use advanced obstacle detector with current speed for adaptive threshold
            is_obstacle, distance, confidence = self.obstacle_detector.detect(
                depth_frame, 
                current_speed=self.current_speed
            )
            
            self.obstacle_detected = is_obstacle
            self.last_obstacle_distance = distance
            self.obstacle_confidence = confidence
            
            # Log obstacle detection with confidence
            if is_obstacle:
                self.get_logger().warn(
                    f'🛑 OBSTACLE DETECTED! Distance: {distance:.2f}m '
                    f'(threshold: {self.min_safe_distance:.2f}m, confidence: {confidence:.1%})',
                    throttle_duration_sec=0.5
                )
            
        except Exception as e:
            # Don't crash on depth errors, just log and continue
            self.get_logger().warn(f'Depth callback error: {e}', throttle_duration_sec=5.0)
            self.obstacle_detected = False
            self.obstacle_confidence = 0.0
    
    def depth_distance_callback(self, msg):
        """Update minimum safe distance from GUI."""
        self.min_safe_distance = max(
            config.MIN_SAFE_DISTANCE_MIN,
            min(config.MIN_SAFE_DISTANCE_MAX, msg.data)
        )
        # Update detector threshold
        self.obstacle_detector.update_threshold(self.min_safe_distance)
        self.get_logger().info(f"Min safe distance: {self.min_safe_distance:.2f}m")


    def smpc_param_callback(self, msg):
        """Parse Supervised MPC override data from GUI."""
        import json
        try:
            params = json.loads(msg.data)
            self.smpc_steer_assist = float(params.get('smpc_steer_assist', 0.0))
            self.smpc_steer_mode = params.get('smpc_steer_mode', 'blend')
            raw_speed = params.get('smpc_speed_override', None)
            self.smpc_speed_override = float(raw_speed) if raw_speed is not None else None
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.get_logger().error(f"Invalid SMPC params: {e}", throttle_duration_sec=5.0)



    # MPC Tuning Callbacks
    def mpc_horizon_callback(self, msg):
        self.registry.update_all_params({'mpc_horizon': msg.data})
        self.get_logger().info(f"MPC Horizon updated: {msg.data}")

    def mpc_tracking_callback(self, msg):
        self.registry.update_all_params({'mpc_tracking': msg.data})
        self.get_logger().info(f"MPC Tracking weight updated: {msg.data:.1f}")

    def mpc_smoothness_callback(self, msg):
        self.registry.update_all_params({'mpc_smoothness': msg.data})
        self.get_logger().info(f"MPC Smoothness weight updated: {msg.data:.1f}")

    def cem_samples_callback(self, msg):
        self.registry.update_all_params({'cem_num_samples': msg.data})
        self.get_logger().info(f"CEM num_samples updated: {msg.data}")

    def cem_horizon_callback(self, msg):
        self.registry.update_all_params({'cem_horizon': msg.data})
        self.get_logger().info(f"CEM horizon updated: {msg.data}")

    def hmpcc_horizon_callback(self, msg):
        self.registry.update_all_params({'hmpcc_horizon': msg.data})
        self.get_logger().info(f"Hybrid MPCC Horizon updated: {msg.data}")

    def hmpcc_wv_callback(self, msg):
        self.registry.update_all_params({'hmpcc_wv': msg.data})
        self.get_logger().info(f"Hybrid MPCC W_v (Velocity Weight) updated: {msg.data}")

    def hmpcc_wc_callback(self, msg):
        self.registry.update_all_params({'hmpcc_wc': msg.data})
        self.get_logger().info(f"Hybrid MPCC W_c (Contouring Weight) updated: {msg.data}")

    def obstacle_enable_callback(self, msg):
        """Enable or disable obstacle detection from GUI."""
        self.obstacle_detector.set_enabled(msg.data)
        status = "ENABLED" if msg.data else "DISABLED"
        self.get_logger().info(f"Obstacle detection: {status}")


    
    def joy_callback(self, msg):
        """Joystick Deadman Switch - instant override of autonomous mode.
        
        LB (button 4) - Hold to drive manually, auto-disables autonomous
        """
        if len(msg.buttons) > self.deadman_button:
            lb_pressed = msg.buttons[self.deadman_button] == 1
            
            # If LB pressed while autonomous is running, disable it
            if lb_pressed and self.autonomous_running:
                # Instant override - disable autonomous and stop the car
                self.autonomous_enabled = False
                self.autonomous_running = False
                self.publish_drive(0.0, 0.0)
                
                # Notify GUI of status change
                self.status_pub.publish(Bool(data=False))
                
                # Throttle logging to once per second
                now = time.time()
                if now - self.last_joy_override_time > 1.0:
                    self.get_logger().warn('🎮 JOYSTICK OVERRIDE - Autonomous disabled by LB')
                    self.last_joy_override_time = now

def main(args=None):
    rclpy.init(args=args)
    node = VisionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_drive(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
