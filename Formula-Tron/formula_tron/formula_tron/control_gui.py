#!/usr/bin/env python3
"""
Formula-Tron Control GUI

ROS 2 node: control panel for F1TENTH car with RealSense camera.

The mode overlay lists entries from mode_config.MODES. Additional control_mode
values may be applied via /tuning/control_mode or presets (e.g.
SUPERVISED_MPC, EXPO).

Features:
- Live camera feed (Raw or OpenCV Debug view)
- Manual driving (WASD / Arrow keys)
- Autonomous mode toggle
- Speed limiter
- Emergency stop
"""

import sys
import time
import threading
import numpy as np
import json
import os
import re
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Joy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, Float64, Float64MultiArray, Int32, String
import cv2  # Must import before cv_bridge on Jetson (cv_bridge_boost init order)
from cv_bridge import CvBridge
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QGridLayout, QScrollArea, QMessageBox, QFileDialog, QSizePolicy, QComboBox, QLineEdit, QButtonGroup, QTabWidget, QFrame, QTextEdit, QProgressBar, QDoubleSpinBox, QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QColor, QDoubleValidator, QPainter, QPen, QLinearGradient, QPainterPath

# Telemetry imports
from . import config
from .utils.telemetry import TelemetryRecord, TelemetryCollector
from .utils.mode_config import MODES, SECTION_HEADERS, MODE_TOOLTIPS


class RewardSparkline(QWidget):
    """Compact sparkline chart showing RL reward trend over time."""

    def __init__(self, max_points: int = 120, parent=None):
        super().__init__(parent)
        self._data: list = []
        self._max = max_points
        self.setFixedHeight(52)
        self.setMinimumWidth(100)
        self._label = ""

    def append(self, value: float):
        self._data.append(value)
        if len(self._data) > self._max:
            self._data.pop(0)
        self.update()

    def clear_data(self):
        self._data.clear()
        self._label = ""
        self.update()

    def set_label(self, text: str):
        self._label = text
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(30, 30, 30))
        p.setPen(QPen(QColor(60, 60, 60)))
        p.drawRect(0, 0, w - 1, h - 1)

        n = len(self._data)
        if n < 2:
            p.setPen(QColor(100, 100, 100))
            p.drawText(4, h // 2 + 4, "Collecting data..." if n == 0 else "Collecting data...")
            p.end()
            return

        # Zero-line
        lo = min(self._data)
        hi = max(self._data)
        span = hi - lo if hi != lo else 1.0
        margin = 4

        def y_of(v):
            return margin + (h - 2 * margin) * (1.0 - (v - lo) / span)

        zero_y = y_of(0.0)
        if lo <= 0 <= hi:
            p.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
            p.drawLine(0, int(zero_y), w, int(zero_y))

        # Gradient fill under curve
        path = QPainterPath()
        xs = [margin + (w - 2 * margin) * i / (n - 1) for i in range(n)]
        ys = [y_of(v) for v in self._data]
        path.moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(x, y)

        # Fill
        fill_path = QPainterPath(path)
        fill_path.lineTo(xs[-1], h)
        fill_path.lineTo(xs[0], h)
        fill_path.closeSubpath()

        avg = sum(self._data[-20:]) / min(20, len(self._data))
        if avg >= 0:
            fill_color = QColor(0, 180, 80, 40)
            line_color = QColor(0, 220, 100)
        else:
            fill_color = QColor(220, 60, 60, 40)
            line_color = QColor(255, 80, 80)

        p.fillPath(fill_path, fill_color)
        p.setPen(QPen(line_color, 2))
        p.drawPath(path)

        # Latest value dot
        p.setBrush(line_color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(xs[-1]) - 3, int(ys[-1]) - 3, 6, 6)

        # Label
        if self._label:
            p.setPen(QColor(200, 200, 200))
            p.drawText(margin + 2, 12, self._label)

        p.end()


class RosSignals(QObject):
    image_received = pyqtSignal(np.ndarray)
    debug_image_received = pyqtSignal(np.ndarray)
    april_tag_image_received = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    lap_update = pyqtSignal(int, float)  # lap_count, lap_time
    joy_update = pyqtSignal(bool, bool, float, float)  # connected, deadman, speed, steering
    autonomous_override = pyqtSignal(bool)  # autonomous status changed externally
    telemetry_received = pyqtSignal(object)  # TelemetryRecord
    auto_calibrate_hsv = pyqtSignal()  # trigger auto calibration
    vesc_voltage_received = pyqtSignal(float)  # battery voltage from VESC


class CarControlNode(Node):
    def __init__(self, signals: RosSignals):
        super().__init__('formula_tron_control')
        self.signals = signals
        self.bridge = CvBridge()
        
        # Topics - direct VESC commands
        self.camera_topic = self._detect_camera_topic()
        self.motor_topic = '/commands/motor/speed'
        self.servo_topic = '/commands/servo/position'
        self.debug_topic = '/vision_debug'
        
        # VESC conversion parameters from vesc.yaml
        self.speed_to_erpm_gain = config.VESC_SPEED_TO_ERPM_GAIN
        self.speed_to_erpm_offset = config.VESC_SPEED_TO_ERPM_OFFSET
        self.steering_to_servo_gain = config.VESC_STEERING_TO_SERVO_GAIN
        self.steering_to_servo_offset = config.VESC_STEERING_TO_SERVO_OFFSET
        self.max_steering_angle = config.MAX_STEERING_ANGLE_EFFECTIVE
        self.steering_min_angle = -self.max_steering_angle
        self.steering_max_angle = self.max_steering_angle
        
        # State
        self.autonomous_enabled = False  # In autonomous mode (tuning available)
        self.autonomous_running = False   # Actually driving autonomously
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.target_speed = 0.0
        self.target_steering = 0.0
        self.connected = False
        self.debug_connected = False
        self.last_image_time = None
        self.cmd_count = 0
        
        # Smooth ramping parameters
        self.steer_step = 0.032  # 3.2 rad/s at 100 Hz - aligned with FilesFromCar max_servo_speed
        self.speed_step = 0.1   # How much to change speed per loop
        
        # Joystick state (Logitech F-710)
        self.joy_connected = False
        self.joy_deadman_pressed = False  # LB button (button 4)
        self.joy_speed = 0.0              # Left stick Y (axis 1)
        self.joy_steering = 0.0           # Right stick X (axis 3)
        self.joy_max_speed = 2.0          # Will be synced with GUI slider
        self.joy_last_msg_time = None
        
        # Joystick button mappings (Logitech F-710)
        self.JOY_BTN_LB = 4      # Deadman switch for manual control
        self.JOY_BTN_A = 0       # (reserved)
        self.JOY_BTN_B = 1       # (reserved)
        self.JOY_AXIS_LEFT_Y = 1   # Speed (forward/back)
        self.JOY_AXIS_RIGHT_X = 3  # Steering (left/right)
        
        # Publishers - direct VESC
        self.motor_pub = self.create_publisher(Float64, self.motor_topic, 10)
        self.servo_pub = self.create_publisher(Float64, self.servo_topic, 10)
        self.auto_enable_pub = self.create_publisher(Bool, '/autonomous_enabled', 10)
        self.auto_start_pub = self.create_publisher(Bool, '/autonomous_start', 10)
        
        # Tuning publishers
        self.pd_pub = self.create_publisher(Float64MultiArray, '/tuning/pd', 10)
        self.hsv_pub = self.create_publisher(Float64MultiArray, '/tuning/hsv', 10)
        self.auto_speed_pub = self.create_publisher(Float64, '/tuning/auto_speed', 10)
        self.track_width_pub = self.create_publisher(Float64, '/tuning/track_width', 10)
        self.steering_bias_pub = self.create_publisher(Float64, '/tuning/steering_bias', 10)
        self.turn_slowdown_pub = self.create_publisher(Float64, '/tuning/turn_slowdown', 10)
        self.smoothing_alpha_pub = self.create_publisher(Float64, '/tuning/smoothing_alpha', 10)
        self.control_mode_pub = self.create_publisher(String, '/tuning/control_mode', 10)
        self.debug_enabled_pub = self.create_publisher(Bool, '/debug_enabled', 10)
        self.april_tag_view_enabled_pub = self.create_publisher(Bool, '/april_tag_view_enabled', 10)
        self.auto_calibrate_pub = self.create_publisher(Bool, '/tuning/auto_calibrate_hsv', 10)

        
        # Supervised MPC mode pub
        self.smpc_tuning_pub = self.create_publisher(String, '/tuning/smpc_param', QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        
        # Poly LookAhead Mode tuning publishers
        self.lookahead_pub = self.create_publisher(Float64, '/tuning/lookahead_ratio', 10)
        self.bev_top_width_pub = self.create_publisher(Float64, '/tuning/bev_top_width', 10)
        self.bev_padding_pub = self.create_publisher(Float64, '/tuning/bev_padding', 10)
        
        # Depth Safety publishers
        self.min_safe_distance_pub = self.create_publisher(Float64, '/tuning/min_safe_distance', 10)
        self.obstacle_enable_pub = self.create_publisher(Bool, '/obstacle_detection_enabled', 10)
        
        # MPC Tuning publishers
        self.mpc_horizon_pub = self.create_publisher(Int32, '/tuning/mpc_horizon', 10)
        self.mpc_tracking_pub = self.create_publisher(Float64, '/tuning/mpc_tracking', 10)
        self.mpc_smoothness_pub = self.create_publisher(Float64, '/tuning/mpc_smoothness', 10)
        
        # CEM Tuning publishers
        self.cem_num_samples_pub = self.create_publisher(Int32, '/tuning/cem_num_samples', 10)
        self.cem_horizon_pub = self.create_publisher(Int32, '/tuning/cem_horizon', 10)
        
        # Hybrid MPCC Tuning publishers
        self.hmpcc_horizon_pub = self.create_publisher(Int32, '/tuning/hmpcc_horizon', 10)
        self.hmpcc_wv_pub = self.create_publisher(Int32, '/tuning/hmpcc_wv', 10)
        self.hmpcc_wc_pub = self.create_publisher(Int32, '/tuning/hmpcc_wc', 10)
        
        # Pure Pursuit Tuning Publishers
        self.pp_lookahead_pub = self.create_publisher(Float64, '/tuning/pp_min_lookahead', 10)
        self.pp_gain_pub = self.create_publisher(Float64, '/tuning/pp_velocity_gain', 10)
        
        # Stanley Tuning Publishers
        self.stanley_k_pub = self.create_publisher(Float64, '/tuning/stanley_k', 10)
        self.stanley_ks_pub = self.create_publisher(Float64, '/tuning/stanley_ks', 10)
        
        
        # Current tuning values
        self.kp = 0.85
        self.kd = 0.15
        self.hsv_h_min = 35
        self.hsv_h_max = 90
        self.hsv_s_min = 50
        self.hsv_v_min = 50
        self.auto_base_speed = 1.5
        self.steering_bias = 0.0
        self.turn_slowdown = 0.35
        self.smoothing_alpha = 0.55
        
        # Poly LookAhead Mode defaults
        self.lookahead_ratio = 0.3
        self.bev_top_width = 0.4
        self.bev_padding = 0.2
        
        # Depth Safety defaults
        self.min_safe_distance = 0.50
        self.obstacle_detection_enabled = True
        
        # MPC defaults
        self.mpc_horizon = 15
        self.mpc_tracking = 1.0  # Default weight multiplier
        self.mpc_smoothness = 1.0 # Default weight multiplier
        
        # Active control mode tracking
        self.control_mode = 'POLY_LOOKAHEAD'
        
        # Camera subscribers (buffer 5 frames for network drops)
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5  # Buffer frames for network drops
        )
        # Debug topic uses BEST_EFFORT so it won't freeze over wifi
        debug_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, camera_qos)
        self.debug_sub = self.create_subscription(Image, self.debug_topic, self.debug_image_callback, debug_qos)
        self.april_tag_sub = self.create_subscription(Image, '/vision_april_tag', self.april_tag_image_callback, debug_qos)
        self.last_debug_time = None
        
        # VESC core sensor subscriber (battery voltage)
        try:
            from vesc_msgs.msg import VescStateStamped
            self.vesc_core_sub = self.create_subscription(
                VescStateStamped, '/sensors/core', self.vesc_core_callback,
                QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1))
            self._vesc_msg_type = 'vesc_state'
        except ImportError:
            # Fallback: try Float64MultiArray or skip
            self._vesc_msg_type = None
            self.get_logger().info('VESC message type not available, battery display disabled')
        
        # Lap counter subscribers
        self.lap_count_sub = self.create_subscription(Int32, '/lap_count', self.lap_count_callback, 10)
        self.lap_time_sub = self.create_subscription(Float64, '/lap_time', self.lap_time_callback, 10)
        self.lap_count = 0
        self.last_lap_time = 0.0
        self.best_lap_time = float('inf')
        
        # Joystick subscriber
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # Autonomous status subscriber (feedback from vision controller)
        self.auto_status_sub = self.create_subscription(Bool, '/autonomous_status', self.auto_status_callback, 10)
        
        # Telemetry subscriber
        self.telemetry_sub = self.create_subscription(String, '/telemetry', self.telemetry_callback, 10)
        
        # Timers
        self.manual_timer = self.create_timer(0.01, self.publish_manual_drive)  # 100 Hz
        self.connection_timer = self.create_timer(1.0, self.check_connection)
        
        # Start with zero commands
        self.publish_drive(0.0, 0.0)

        self.get_logger().info('Formula-Tron Control Started')
        self.get_logger().info(f'  Camera: {self.camera_topic}')
        self.get_logger().info(f'  Motor:  {self.motor_topic} (DIRECT VESC)')
        self.get_logger().info(f'  Servo:  {self.servo_topic} (DIRECT VESC)')
        self.get_logger().info(f'  Debug:  {self.debug_topic}')

    
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
            # Default fallback
            else:
                self.get_logger().warn('Camera topic not found, defaulting to /camera/camera/color/image_raw')
                return '/camera/camera/color/image_raw'
        except Exception as e:
            self.get_logger().warn(f'Could not detect camera topic, defaulting to /camera/camera/color/image_raw ({e})')
            return '/camera/camera/color/image_raw'
        self.signals.status_update.emit('Waiting for camera...')
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn('Received empty image from camera')
                return
            self.signals.image_received.emit(cv_image)
            self.last_image_time = self.get_clock().now()
            if not self.connected:
                self.connected = True
                self.get_logger().info('Camera connected - receiving images')
                self.signals.status_update.emit('[OK] Camera connected!')
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}', throttle_duration_sec=2.0)
    
    def debug_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.signals.debug_image_received.emit(cv_image)
            self.last_debug_time = self.get_clock().now()
            if not self.debug_connected:
                self.debug_connected = True
                self.get_logger().info('Vision debug stream connected!')
        except Exception as e:
            self.get_logger().error(f'Debug image error: {e}')
    
    def april_tag_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.signals.april_tag_image_received.emit(cv_image)
        except Exception as e:
            self.get_logger().error(f'AprilTag image error: {e}')
    
    def lap_count_callback(self, msg):
        new_count = msg.data
        # If lap count is reset to 0, reset all lap data
        if new_count == 0:
            self.lap_count = 0
            self.last_lap_time = 0.0
            self.best_lap_time = float('inf')
            # Reset GUI display
            self.signals.lap_update.emit(0, 0.0)
        else:
            self.lap_count = new_count
            # Emit update to show new lap count (including lap 1)
            # For lap 1, time will be 0.0 (no completed lap yet)
            self.signals.lap_update.emit(new_count, self.last_lap_time)
    
    def lap_time_callback(self, msg):
        if self.lap_count < 1 or msg.data <= 0.0:
            return
        self.last_lap_time = msg.data
        if msg.data < self.best_lap_time:
            self.best_lap_time = msg.data
        self.signals.lap_update.emit(self.lap_count, self.last_lap_time)
    
    def joy_callback(self, msg):
        """Process joystick input from Logitech F-710 controller."""
        self.joy_last_msg_time = self.get_clock().now()
        
        if not self.joy_connected:
            self.joy_connected = True
            self.get_logger().info('Joystick connected!')
            self.signals.status_update.emit('[OK] Joystick connected!')
        
        # Check button states
        if len(msg.buttons) > self.JOY_BTN_LB:
            self.joy_deadman_pressed = msg.buttons[self.JOY_BTN_LB] == 1
        
        # Read axis values (only apply if deadman is held)
        if len(msg.axes) > max(self.JOY_AXIS_LEFT_Y, self.JOY_AXIS_RIGHT_X):
            if self.joy_deadman_pressed:
                # Left stick Y-axis for speed (forward positive, back negative)
                raw_speed = msg.axes[self.JOY_AXIS_LEFT_Y]
                self.joy_speed = raw_speed * self.joy_max_speed
                
                # Right stick X-axis for steering (left positive, right negative)
                raw_steer = msg.axes[self.JOY_AXIS_RIGHT_X]
                self.joy_steering = raw_steer * self.max_steering_angle
                
            # Manual control handled in GUI timer to avoid keyboard/joystick conflict
            else:
                # Deadman not pressed - clear joystick inputs
                self.joy_speed = 0.0
                self.joy_steering = 0.0
        
        # Emit signal for GUI update
        self.signals.joy_update.emit(
            self.joy_connected,
            self.joy_deadman_pressed,
            self.joy_speed,
            self.joy_steering
        )
    
    def telemetry_callback(self, msg):
        """Process telemetry data from vision controller."""
        import json
        try:
            data = json.loads(msg.data)
            # Convert dict to TelemetryRecord
            record = TelemetryRecord(
                timestamp=data.get('timestamp', 0.0),
                frame_number=data.get('frame_number', 0),
                control_mode=data.get('control_mode', 'UNKNOWN'),
                speed_cmd=data.get('speed_cmd', 0.0),
                steering_cmd=data.get('steering_cmd', 0.0),
                speed_actual=data.get('speed_actual', 0.0),
                imu_accel_x=data.get('imu_accel_x', 0.0),
                imu_accel_y=data.get('imu_accel_y', 0.0),
                imu_yaw_rate=data.get('imu_yaw_rate', 0.0),
                lap_number=data.get('lap_number', 0),
                lap_time_current=data.get('lap_time_current', 0.0),
                lap_time_last=data.get('lap_time_last', 0.0),
                lap_time_best=data.get('lap_time_best', float('inf')) if data.get('lap_time_best', -1) > 0 else float('inf'),
                safety_state=data.get('safety_state', 'OK'),
                mode_data=data.get('mode_data', {}),
                processing_time_ms=data.get('processing_time_ms', 0.0),
            )
            self.signals.telemetry_received.emit(record)
        except Exception as e:
            self.get_logger().warn(f'Telemetry parse error: {e}', throttle_duration_sec=5.0)
    
    def vesc_core_callback(self, msg):
        """Extract battery voltage from VESC core sensor data."""
        try:
            voltage = msg.state.voltage_input
            if voltage > 0:
                self.signals.vesc_voltage_received.emit(float(voltage))
        except Exception as e:
            self.get_logger().warn(f'VESC core parse error: {e}', throttle_duration_sec=10.0)
    
    def check_connection(self):
        now = self.get_clock().now()
        
        # Camera connection watchdog
        if self.last_image_time:
            elapsed = (now - self.last_image_time).nanoseconds / 1e9
            if elapsed > 2.0:
                if self.connected:
                    self.connected = False
                    self.get_logger().warn(f'Camera connection lost! No frame for {elapsed:.1f}s')
                    self.signals.status_update.emit('[!] Camera connection lost!')
        
        # Debug stream watchdog - warn if debug view is selected but no data
        if self.last_debug_time:
            debug_elapsed = (now - self.last_debug_time).nanoseconds / 1e9
            if debug_elapsed > 2.0 and self.debug_connected:
                self.debug_connected = False
                self.get_logger().warn('Vision debug stream lost!')
        
        # Joystick connection watchdog
        if self.joy_last_msg_time:
            joy_elapsed = (now - self.joy_last_msg_time).nanoseconds / 1e9
            if joy_elapsed > 1.0 and self.joy_connected:
                self.joy_connected = False
                self.joy_deadman_pressed = False
                self.joy_speed = 0.0
                self.joy_steering = 0.0
                self.get_logger().warn('Joystick connection lost!')
                self.signals.status_update.emit('[!] Joystick connection lost!')
                self.signals.joy_update.emit(False, False, 0.0, 0.0)
    
    def set_autonomous(self, enabled):
        self.autonomous_enabled = enabled
        
        # Send stop command when switching modes
        self.target_speed = 0.0
        self.target_steering = 0.0
        self.publish_drive(0.0, 0.0)
        
        msg = Bool()
        msg.data = enabled
        self.auto_enable_pub.publish(msg)
        
        if not enabled:
            # Exiting autonomous mode also stops driving
            self.autonomous_running = False
            self.set_autonomous_start(False)
    
    def set_autonomous_start(self, running):
        self.autonomous_running = running
        msg = Bool()
        msg.data = running
        self.auto_start_pub.publish(msg)
    
    def auto_status_callback(self, msg):
        """Receive autonomous status updates from vision controller (e.g., joystick override)."""
        new_status = msg.data
        
        # If vision controller disabled autonomous (e.g., joystick override)
        if not new_status and self.autonomous_enabled:
            self.autonomous_enabled = False
            self.autonomous_running = False
            self.get_logger().warn('Autonomous mode disabled by external trigger (joystick override)')
            self.signals.status_update.emit('Ã°Å¸Å½Â® Joystick Override - Switched to Manual')
            self.signals.autonomous_override.emit(False)
    
    def set_manual_control(self, speed, steering):
        self.target_speed = speed
        self.target_steering = steering
    
    def publish_manual_drive(self):
        if not self.autonomous_enabled:  # Only in manual mode
            # Smooth steering
            diff = self.target_steering - self.current_steering
            if abs(diff) < self.steer_step:
                self.current_steering = self.target_steering
            else:
                self.current_steering += self.steer_step * (1.0 if diff > 0 else -1.0)
                
            # Smooth speed changes
            diff_spd = self.target_speed - self.current_speed
            if abs(diff_spd) < self.speed_step:
                self.current_speed = self.target_speed
            else:
                self.current_speed += self.speed_step * (1.0 if diff_spd > 0 else -1.0)
                
            # Stop publishing if stopped and no manual input.
            if (abs(self.current_speed) < 0.01 and 
                abs(self.current_steering) < 0.01 and 
                self.target_speed == 0.0 and 
                self.target_steering == 0.0):
                return

            self.publish_drive(self.current_speed, self.current_steering)
    
    def publish_drive(self, speed, steering):
        # Convert speed (m/s) to ERPM
        erpm = self.speed_to_erpm_gain * speed + self.speed_to_erpm_offset
        motor_msg = Float64()
        motor_msg.data = float(erpm)
        
        # Convert steering (radians) to servo position (0.0 - 1.0)
        steering = max(self.steering_min_angle, min(self.steering_max_angle, steering))
        servo_pos = self.steering_to_servo_gain * steering + self.steering_to_servo_offset
        servo_pos = max(config.SERVO_CMD_MIN, min(config.SERVO_CMD_MAX, servo_pos))  # SAFE LIMITS
        servo_msg = Float64()
        servo_msg.data = float(servo_pos)
        
        # Publish to VESC
        self.motor_pub.publish(motor_msg)
        self.servo_pub.publish(servo_msg)
        
        self.current_speed = speed
        self.current_steering = steering
        self.cmd_count += 1
    
    def emergency_stop(self):
        # Stop the car immediately
        self.autonomous_enabled = False
        self.autonomous_running = False
        self.target_speed = 0.0
        self.target_steering = 0.0
        self.current_speed = 0.0
        self.current_steering = 0.0
        
        # Send multiple stop commands
        for _ in range(5):
            self.publish_drive(0.0, 0.0)
        
        # Publish state changes to vision controller
        msg = Bool()
        msg.data = False
        self.auto_enable_pub.publish(msg)
        self.auto_start_pub.publish(msg)
        
        self.get_logger().warn('EMERGENCY STOP!')
    
    def publish_pd(self, kp, kd):
        self.kp = kp
        self.kd = kd
        msg = Float64MultiArray()
        msg.data = [float(kp), float(kd)]
        self.pd_pub.publish(msg)
    
    def publish_hsv(self, h_min, h_max, s_min, v_min):
        self.hsv_h_min = h_min
        self.hsv_h_max = h_max
        self.hsv_s_min = s_min
        self.hsv_v_min = v_min
        msg = Float64MultiArray()
        msg.data = [float(h_min), float(h_max), float(s_min), float(v_min)]
        self.hsv_pub.publish(msg)
    
    def publish_auto_speed(self, speed):
        self.auto_base_speed = speed
        msg = Float64()
        msg.data = float(speed)
        self.auto_speed_pub.publish(msg)

    def publish_track_width(self, width):
        msg = Float64()
        msg.data = float(width)
        self.track_width_pub.publish(msg)

    def publish_steering_bias(self, bias):
        self.steering_bias = bias
        msg = Float64()
        msg.data = float(bias)
        self.steering_bias_pub.publish(msg)
    
    def publish_turn_slowdown(self, slowdown):
        self.turn_slowdown = slowdown
        msg = Float64()
        msg.data = float(slowdown)
        self.turn_slowdown_pub.publish(msg)
    
    def publish_smoothing_alpha(self, alpha):
        self.smoothing_alpha = alpha
        msg = Float64()
        msg.data = float(alpha)
        self.smoothing_alpha_pub.publish(msg)
    
    def publish_control_mode(self, mode):
        self.control_mode = mode
        msg = String()
        msg.data = mode
        self.control_mode_pub.publish(msg)
    
    def publish_lookahead_ratio(self, ratio):
        self.lookahead_ratio = ratio
        msg = Float64()
        msg.data = float(ratio)
        self.lookahead_pub.publish(msg)
    
    def publish_bev_top_width(self, width):
        self.bev_top_width = width
        msg = Float64()
        msg.data = float(width)
        self.bev_top_width_pub.publish(msg)
    
    def publish_bev_padding(self, padding):
        self.bev_padding = padding
        msg = Float64()
        msg.data = float(padding)
        self.bev_padding_pub.publish(msg)
    
    def publish_min_safe_distance(self, distance):
        self.min_safe_distance = distance
        msg = Float64()
        msg.data = float(distance)
        self.min_safe_distance_pub.publish(msg)
    
    def publish_obstacle_enable(self, enabled):
        self.obstacle_detection_enabled = enabled
        msg = Bool()
        msg.data = enabled
        self.obstacle_enable_pub.publish(msg)

    def publish_mpc_horizon(self, horizon):
        self.mpc_horizon = horizon
        msg = Int32()
        msg.data = int(horizon)
        self.mpc_horizon_pub.publish(msg)

    def publish_mpc_tracking(self, weight):
        self.mpc_tracking = weight
        msg = Float64()
        msg.data = float(weight)
        self.mpc_tracking_pub.publish(msg)

    def publish_mpc_smoothness(self, weight):
        self.mpc_smoothness = weight
        msg = Float64()
        msg.data = float(weight)
        self.mpc_smoothness_pub.publish(msg)

    def publish_cem_num_samples(self, n):
        msg = Int32()
        msg.data = int(n)
        self.cem_num_samples_pub.publish(msg)

    def publish_cem_horizon(self, h):
        msg = Int32()
        msg.data = int(h)
        self.cem_horizon_pub.publish(msg)

    def publish_hmpcc_horizon(self, horizon):
        msg = Int32()
        msg.data = int(horizon)
        self.hmpcc_horizon_pub.publish(msg)

    def publish_hmpcc_wv(self, wv):
        msg = Int32()
        msg.data = int(wv)
        self.hmpcc_wv_pub.publish(msg)

    def publish_hmpcc_wc(self, wc):
        msg = Int32()
        msg.data = int(wc)
        self.hmpcc_wc_pub.publish(msg)




    def publish_smpc_override(self, steer_assist: float, steer_mode: str, speed_override):
        import json
        msg = String()
        msg.data = json.dumps({
            'smpc_steer_assist': steer_assist,
            'smpc_steer_mode': steer_mode,
            'smpc_speed_override': speed_override,
        })
        self.smpc_tuning_pub.publish(msg)

    def publish_pp_lookahead(self, l_min):
        msg = Float64()
        msg.data = float(l_min)
        self.pp_lookahead_pub.publish(msg)

    def publish_pp_gain(self, k):
        msg = Float64()
        msg.data = float(k)
        self.pp_gain_pub.publish(msg)

    def publish_stanley_k(self, k):
        msg = Float64()
        msg.data = float(k)
        self.stanley_k_pub.publish(msg)

    def publish_stanley_ks(self, ks):
        msg = Float64()
        msg.data = float(ks)
        self.stanley_ks_pub.publish(msg)



class ControlGUI(QMainWindow):
    # Color constants for standardization
    COLOR_GREEN = "#00aa00"      # Safe/Active
    COLOR_YELLOW = "#ffaa00"     # Warning/Moderate
    COLOR_RED = "#ff4444"        # Danger/Fast/Error
    COLOR_BLUE = "#0066cc"       # Info/Neutral
    COLOR_ORANGE = "#ff8800"     # Beta/Warning
    
    # Resolution presets (name, width, height)
    # Resolution presets - matches Linux display options
    RESOLUTIONS = {
        "1600Ãƒâ€”900 (16:9)": (1600, 900),
        "1440Ãƒâ€”900 (16:10)": (1440, 900),
        "1400Ãƒâ€”900 (3:2)": (1400, 900),
        "1368Ãƒâ€”768 (16:9)": (1368, 768),
        "1360Ãƒâ€”768 (16:9)": (1360, 768),
        "1280Ãƒâ€”960 (4:3)": (1280, 960),
        "1280Ãƒâ€”800 (16:10)": (1280, 800),
        "1152Ãƒâ€”864 (4:3)": (1152, 864),
        "1024Ãƒâ€”768 (4:3)": (1024, 768),
        "800Ãƒâ€”600 (4:3)": (800, 600),
    }
    
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle('Formula-Tron Control')
        
        # Base window size for scaling (Medium)
        self.base_width = 1280
        self.base_height = 800
        self.current_scale = 1.0
        
        # Default resolution
        self.current_resolution = "1280Ãƒâ€”800 (16:10)"
        self.apply_resolution(self.current_resolution)
        
        self.keys_pressed = set()
        self.max_speed = 2.0
        self.max_reverse_speed = 1.0  # Dedicated reverse speed limit
        self.current_view = "raw"  # "raw", "debug", or "april_tag"
        self.last_raw_image = None
        self.last_debug_image = None
        self.last_april_tag_image = None
        self.raw_image_received_count = 0  # Diagnostic counter
        self.debug_image_received_count = 0  # Diagnostic counter
        self.april_tag_image_received_count = 0  # Diagnostic counter
        


        # Discrete Speed Presets
        self.discrete_speed = None  # None=off, float=active preset speed
        self.discrete_active_key = -1  # -1=none active, 0-4=key index
        self.DISCRETE_SPEED_PRESETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

        # Supervised MPC State
        self.smpc_speed = None  # None=MPC controls speed, float=locked preset
        self.smpc_active_key = -1  # -1=none active, 0-6=preset index
        self.SMPC_SPEED_PRESETS = [1.5, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5]
        self.smpc_steer_mode = "blend"  # "blend" or "override"

        # Expo Mode State
        self.expo_active = False
        self.expo_current_routine = "sweep"
        self.expo_timer = None
        self.expo_phase = 0.0
        self.expo_speed_amplitude = 1.0   # m/s max
        self.expo_steer_amplitude = 0.35  # rad max
        self.expo_frequency = 1.0         # Hz
        self.expo_battery_voltage = 0.0


        
        # FPS tracking
        self.frame_timestamps = deque(maxlen=30)  # Store last 30 frame timestamps
        self.current_fps = 0.0
        self.frame_count = 0
        
        # Preset directory
        self.preset_dir = Path.home() / '.formula_tron' / 'presets'
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_style()
        self.setup_ui()
        
        self.ros_node.signals.image_received.connect(self.on_raw_image)
        self.ros_node.signals.debug_image_received.connect(self.on_debug_image)
        self.ros_node.signals.april_tag_image_received.connect(self.on_april_tag_image)
        self.ros_node.signals.status_update.connect(self.update_status)
        self.ros_node.signals.lap_update.connect(self.update_lap)
        self.ros_node.signals.joy_update.connect(self.update_joystick)
        self.ros_node.signals.autonomous_override.connect(self.on_autonomous_override)
        self.ros_node.signals.telemetry_received.connect(self.on_telemetry)
        self.ros_node.signals.vesc_voltage_received.connect(self._on_vesc_voltage)
        
        # Telemetry system
        self.telemetry_collector = TelemetryCollector()
        self.last_active_mode = None
        self.last_runtime_source = None
        self.telemetry_window = None
        
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(50)
        
        self.key_timer = QTimer()
        self.key_timer.timeout.connect(self.process_keys)
        self.key_timer.start(10)  # 100 Hz, match manual_timer publish rate
        
        self.setFocusPolicy(Qt.StrongFocus)
    
    def apply_resolution(self, resolution_name):
        """Apply window size based on resolution selection."""
        width, height = self.RESOLUTIONS[resolution_name]
        self.resize(width, height)
        self.setMinimumSize(800, 500)  # Allow shrinking but set minimum
        self.current_resolution = resolution_name
        self.update_font_scaling()
    
    def on_resolution_change(self, index):
        """Handle resolution dropdown selection."""
        resolution_name = list(self.RESOLUTIONS.keys())[index]
        self.apply_resolution(resolution_name)
    
    def resizeEvent(self, event):
        """Handle window resize - scale fonts dynamically."""
        super().resizeEvent(event)
        self._sync_overlay_geometry()
        self.update_font_scaling()

    def _sync_overlay_geometry(self):
        """Keep the fullscreen mode overlay aligned to the central widget."""
        cw = self.centralWidget()
        if cw is not None:
            if hasattr(self, 'overlay') and self.overlay is not None:
                self.overlay.setGeometry(cw.rect())
                if self.overlay.isVisible():
                    self.overlay.raise_()
            w = getattr(self, '_expo_welcome_overlay', None)
            if w is not None and w.isVisible():
                w.setGeometry(cw.rect())
                w.raise_()
    
    def update_font_scaling(self):
        """Update all font sizes based on current window size."""
        current_width = self.width()
        current_height = self.height()
        
        # Calculate scale factor (use average of width and height scaling)
        width_scale = current_width / self.base_width
        height_scale = current_height / self.base_height
        self.current_scale = min(width_scale, height_scale)  # Use smaller to prevent overflow
        
        # Clamp scale to reasonable range (0.5x to 1.5x)
        self.current_scale = max(0.5, min(1.5, self.current_scale))
        
        # Base font sizes (from Medium 1280Ãƒâ€”800)
        base_font_size = int(13 * self.current_scale)
        base_title_font = int(13 * self.current_scale)
        base_small_font = int(11 * self.current_scale)
        base_tiny_font = int(10 * self.current_scale)
        
        # Update main stylesheet with scaled fonts
        scaled_stylesheet = f"""
            QMainWindow {{ background-color: #1a1a1a; }}
            QLabel {{ color: #eaeaea; font-size: {base_font_size}px; }}
            QToolTip {{
                background-color: #161616;
                color: #f3f3f3;
                border: 1px solid #ffb000;
                border-radius: 6px;
                padding: 6px 8px;
                font-size: {base_small_font}px;
            }}
            QGroupBox {{
                color: #ff4444; font-size: {base_title_font}px; font-weight: bold;
                border: 2px solid #ff4444; border-radius: 6px;
                margin-top: {int(6 * self.current_scale)}px; padding-top: {int(6 * self.current_scale)}px;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: {int(10 * self.current_scale)}px; padding: 0 {int(5 * self.current_scale)}px; }}
            QPushButton {{
                background-color: #2a2a2a; color: #eaeaea;
                border: 2px solid #444; border-radius: 6px;
                padding: {int(8 * self.current_scale)}px {int(12 * self.current_scale)}px; 
                font-size: {base_font_size}px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #3a3a3a; border-color: #ff4444; }}
            QPushButton:pressed {{ background-color: #ff4444; color: #000; }}
            QSlider::groove:horizontal {{
                border: 1px solid #444; height: {int(10 * self.current_scale)}px;
                background: #2a2a2a; border-radius: {int(5 * self.current_scale)}px;
            }}
            QSlider::handle:horizontal {{
                background: #ff4444; border: 2px solid #ff4444;
                width: {int(20 * self.current_scale)}px; margin: {int(-6 * self.current_scale)}px 0; 
                border-radius: {int(10 * self.current_scale)}px;
            }}
            QComboBox {{
                font-size: {base_font_size}px;
                padding: {int(8 * self.current_scale)}px;
            }}
        """
        self.setStyleSheet(scaled_stylesheet)
        
        # Update specific labels that have custom font sizes
        self._update_widget_fonts(base_font_size, base_small_font, base_tiny_font)
        
        # Update all collapsible toggle buttons
        self._update_collapsible_buttons(base_font_size)
    
    def _update_widget_fonts(self, base_font, small_font, tiny_font):
        """Update font sizes for ALL text elements - comprehensive coverage."""
        try:
            # Calculate scaled versions of common font sizes
            large_font = int(18 * self.current_scale)  # For emergency stop, speed labels
            medium_font = int(16 * self.current_scale)  # For speed/steering displays
            title_font = int(13 * self.current_scale)  # For section titles
            
            # Update ALL QLabel widgets
            for label in self.findChildren(QLabel):
                current_style = label.styleSheet() or ""
                
                # Special handling for specific labels with known font sizes
                if hasattr(self, 'mode_status') and label == self.mode_status:
                    current_style = f'color: #00ff00; font-size: {small_font}px; padding: {int(5 * self.current_scale)}px;'
                elif hasattr(self, 'auto_status') and label == self.auto_status:
                    current_style = f"color: #888; font-size: {small_font}px;"
                elif hasattr(self, 'speed_lbl') and label == self.speed_lbl:
                    # Speed label uses 16px base
                    current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {medium_font}px', current_style) if 'font-size' in current_style else f'font-size: {medium_font}px; font-weight: bold;'
                elif hasattr(self, 'steer_lbl') and label == self.steer_lbl:
                    # Steering label uses 16px base
                    current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {medium_font}px', current_style) if 'font-size' in current_style else f'font-size: {medium_font}px; font-weight: bold;'
                elif hasattr(self, 'lap_count_lbl') and label == self.lap_count_lbl:
                    # Lap count uses 24px base
                    large_font_size = int(24 * self.current_scale)
                    current_style = f'font-size: {large_font_size}px; font-weight: bold; color: {self.COLOR_GREEN};'
                elif hasattr(self, 'lap_time_lbl') and label == self.lap_time_lbl:
                    # Lap time uses 14px base
                    small_font_size = int(14 * self.current_scale)
                    current_style = f'font-size: {small_font_size}px;'
                elif hasattr(self, 'speed_limit_lbl') and label == self.speed_limit_lbl:
                    # Speed limit uses 18px base
                    color_match = re.search(r'color:\s*([^;]+)', current_style)
                    color = color_match.group(1) if color_match else self.COLOR_GREEN
                    current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {large_font}px', current_style)
                    if 'font-size' not in current_style:
                        current_style = f'font-size: {large_font}px; font-weight: bold; color: {color};'
                elif hasattr(self, 'reverse_limit_lbl') and label == self.reverse_limit_lbl:
                    # Reverse limit uses 16px base
                    color_match = re.search(r'color:\s*([^;]+)', current_style)
                    color = color_match.group(1) if color_match else self.COLOR_YELLOW
                    current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {medium_font}px', current_style)
                    if 'font-size' not in current_style:
                        current_style = f'font-size: {medium_font}px; font-weight: bold; color: {color};'
                elif hasattr(self, 'joy_status_lbl') and label == self.joy_status_lbl:
                    # Joystick status uses 12px base
                    current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {small_font}px', current_style) if 'font-size' in current_style else current_style
                else:
                    # Update font-size for all other labels
                    if 'font-size' in current_style:
                        # Replace ALL font-size occurrences (handles multiple selectors)
                        current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {base_font}px', current_style)
                    else:
                        if current_style:
                            current_style += f"; font-size: {base_font}px;"
                        else:
                            current_style = f"font-size: {base_font}px;"
                    
                    # Help text (grey colors) - use smaller font
                    if 'color: #666' in current_style or 'color: #888' in current_style or 'color: #aaa' in current_style:
                        current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {tiny_font}px', current_style)
                    
                    # Title labels (bold, red) - use title font
                    if 'font-weight: bold' in current_style and ('#ff5555' in current_style or '#ff4444' in current_style):
                        current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {title_font}px', current_style)
                
                label.setStyleSheet(current_style)
            
            # Update ALL QPushButton widgets (except collapsible ones handled separately)
            for btn in self.findChildren(QPushButton):
                # Skip collapsible toggle buttons (handled separately)
                if hasattr(btn, 'parent') and isinstance(btn.parent(), QGroupBox):
                    if hasattr(btn.parent(), 'toggle_btn') and btn == btn.parent().toggle_btn:
                        continue
                
                current_style = btn.styleSheet() or ""
                
                # Special handling for emergency stop (18px base)
                if hasattr(self, 'stop_btn') and btn == self.stop_btn:
                    current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {large_font}px', current_style)
                # Auto start button (16px base)
                elif hasattr(self, 'auto_start_btn') and btn == self.auto_start_btn:
                    current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {medium_font}px', current_style)
                # Auto button (14px base)
                elif hasattr(self, 'auto_btn') and btn == self.auto_btn:
                    current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {int(14 * self.current_scale)}px', current_style)
                else:
                    # Update font-size for all other buttons (replace ALL occurrences for multi-selector stylesheets)
                    if 'font-size' in current_style:
                        current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {base_font}px', current_style)
                    else:
                        if current_style:
                            current_style += f"; font-size: {base_font}px;"
                        else:
                            current_style = f"font-size: {base_font}px;"
                
                btn.setStyleSheet(current_style)
            
            # Update ALL QComboBox widgets
            for combo in self.findChildren(QComboBox):
                current_style = combo.styleSheet() or ""
                if 'font-size' in current_style:
                    # Replace ALL font-size occurrences
                    current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {base_font}px', current_style)
                else:
                    if current_style:
                        current_style += f"; font-size: {base_font}px;"
                    else:
                        current_style = f"font-size: {base_font}px;"
                combo.setStyleSheet(current_style)
            
            # Update status bar
            if hasattr(self, 'status_bar'):
                self.status_bar.setStyleSheet(f"""
                    QStatusBar {{
                        background-color: #2a2a2a;
                        color: #eaeaea;
                        border-top: 1px solid #444;
                        font-size: {small_font}px;
                    }}
                """)
        except Exception as e:
            # Silently fail - widgets might not be created yet
            pass
    
    def _update_collapsible_buttons(self, font_size):
        """Update font sizes for collapsible group toggle buttons."""
        # Find all collapsible groups and update their toggle buttons
        for widget in self.findChildren(QGroupBox):
            if hasattr(widget, 'toggle_btn'):
                new_style = f"""
                    QPushButton {{
                        background-color: transparent;
                        color: #ff4444;
                        font-size: {font_size}px;
                        font-weight: bold;
                        border: none;
                        text-align: left;
                        padding: {int(5 * self.current_scale)}px;
                    }}
                    QPushButton:hover {{
                        color: #ff6666;
                    }}
                    QPushButton:checked {{
                        color: #ff4444;
                    }}
                """
                widget.toggle_btn.setStyleSheet(new_style)
    
    def create_collapsible_group(self, title):
        """Create a collapsible QGroupBox with toggle button."""
        group = QGroupBox()
        group.setTitle("")  # No title, we'll add custom header
        
        # Header layout with toggle button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(5, 5, 5, 5)
        
        toggle_btn = QPushButton("Ã¢â€“Â¼ " + title)
        toggle_btn.setCheckable(True)
        toggle_btn.setChecked(True)  # Start expanded
        # Font size will be set dynamically by update_font_scaling
        toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ff4444;
                font-weight: bold;
                border: none;
                text-align: left;
            }
            QPushButton:hover {
                color: #ff6666;
            }
            QPushButton:checked {
                color: #ff4444;
            }
        """)
        
        def toggle_content(checked):
            if checked:
                toggle_btn.setText("Ã¢â€“Â¼ " + title)
                group.content_widget.setVisible(True)
            else:
                toggle_btn.setText("Ã¢â€“Â¶ " + title)
                group.content_widget.setVisible(False)
        
        toggle_btn.toggled.connect(toggle_content)
        header_layout.addWidget(toggle_btn)
        header_layout.addStretch()
        
        # Main layout
        main_layout = QVBoxLayout(group)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(header_layout)
        
        # Content widget (everything inside the group)
        content_widget = QWidget()
        group.content_widget = content_widget
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 0, 5, 5)
        main_layout.addWidget(content_widget)
        
        # Store toggle button for later access
        group.toggle_btn = toggle_btn
        
        return group, content_layout
    
    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1a; }
            QLabel { color: #eaeaea; font-size: 13px; }
            QToolTip {
                background-color: #161616;
                color: #f3f3f3;
                border: 1px solid #ffb000;
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QGroupBox {
                color: #ff4444; font-size: 13px; font-weight: bold;
                border: 2px solid #ff4444; border-radius: 6px;
                margin-top: 6px; padding-top: 6px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton {
                background-color: #2a2a2a; color: #eaeaea;
                border: 2px solid #444; border-radius: 6px;
                padding: 8px 12px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3a3a3a; border-color: #ff4444; }
            QPushButton:pressed { background-color: #ff4444; color: #000; }
            QSlider::groove:horizontal {
                border: 1px solid #444; height: 10px;
                background: #2a2a2a; border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #ff4444; border: 2px solid #ff4444;
                width: 20px; margin: -6px 0; border-radius: 10px;
            }
            QTabWidget::pane { border: 1px solid #444; top: -1px; background: #1a1a1a; }
            QTabBar::tab {
                background: #2a2a2a; color: #aaa; padding: 8px 12px;
                border: 1px solid #444; border-bottom-color: #444;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                min-width: 60px;
            }
            QTabBar::tab:selected { background: #333; color: #fff; border-bottom-color: #333; font-weight: bold; }
            QTabBar::tab:hover { background: #333; }
        """)
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setSpacing(10)
        main.setContentsMargins(10, 10, 10, 10)
        
        # Helper function to create clickable info icon
        def info_icon(info_text, title="Info"):
            btn = QPushButton('?')
            btn.setFixedSize(20, 20)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    color: #4a9eff;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 0;
                }
                QPushButton:hover {
                    color: #6bb6ff;
                }
            """)
            btn.setCursor(Qt.WhatsThisCursor)
            
            def show_info():
                msg = QMessageBox()
                msg.setWindowTitle(title)
                msg.setText(info_text)
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet("""
                    QMessageBox { background-color: #ffffff; }
                    QLabel { color: #000000; font-size: 13px; }
                    QPushButton { background-color: #e0e0e0; color: #000000; border: 1px solid #999; border-radius: 3px; padding: 5px 15px; font-size: 12px; }
                    QPushButton:hover { background-color: #d0d0d0; }
                """)
                msg.exec_()
            
            btn.clicked.connect(show_info)
            return btn
        
        # ========== LEFT COLUMN - Camera & Status ==========
        left = QVBoxLayout()
        cam_group = QGroupBox('Vision Pipeline')
        cam_layout = QVBoxLayout(cam_group)
        cam_layout.setContentsMargins(5, 15, 5, 5)
        
        # View selection segmented control
        view_label = QLabel('Select View:')
        view_label.setStyleSheet('font-weight: bold; font-size: 11px; color: #aaa;')
        cam_layout.addWidget(view_label)
        
        view_row = QHBoxLayout()
        view_row.setSpacing(6)
        self.view_button_group = QButtonGroup(self)
        self.view_button_group.setExclusive(True)
        segment_style = """
            QPushButton {
                color: #d0d0d0;
                background-color: #1f1f1f;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                border: 1px solid #4fa3ff;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #2b6cb0;
                border: 1px solid #6bb6ff;
                color: #ffffff;
            }
        """

        self.view_raw_btn = QPushButton('Raw Camera')
        self.view_raw_btn.setCheckable(True)
        self.view_raw_btn.setChecked(True)
        self.view_raw_btn.setCursor(Qt.PointingHandCursor)
        self.view_raw_btn.setToolTip('Show raw camera feed')
        self.view_raw_btn.setStyleSheet(segment_style)
        self.view_button_group.addButton(self.view_raw_btn, 0)
        view_row.addWidget(self.view_raw_btn)

        self.view_debug_btn = QPushButton('OpenCV Pipeline')
        self.view_debug_btn.setCheckable(True)
        self.view_debug_btn.setCursor(Qt.PointingHandCursor)
        self.view_debug_btn.setToolTip('Show OpenCV debug pipeline (track detection, masks, histograms)')
        self.view_debug_btn.setStyleSheet(segment_style)
        self.view_button_group.addButton(self.view_debug_btn, 1)
        view_row.addWidget(self.view_debug_btn)

        self.view_april_tag_btn = QPushButton('AprilTag Detection')
        self.view_april_tag_btn.setCheckable(True)
        self.view_april_tag_btn.setCursor(Qt.PointingHandCursor)
        self.view_april_tag_btn.setToolTip('Show AprilTag detection visualization')
        self.view_april_tag_btn.setStyleSheet(segment_style)
        self.view_button_group.addButton(self.view_april_tag_btn, 2)
        view_row.addWidget(self.view_april_tag_btn)

        self.view_button_group.buttonClicked[int].connect(self.on_view_changed)
        # Keep node-side stream flags in sync with the default selected view.
        self.on_view_changed(0)
        cam_layout.addLayout(view_row)
        
        self.camera_label = QLabel('Waiting for camera...')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setStyleSheet(f"background-color: #000; border: 3px solid {self.COLOR_RED}; border-radius: 8px; color: #666;")
        cam_layout.addWidget(self.camera_label, stretch=1)
        cam_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left.addWidget(cam_group, stretch=1)
        
        
        # Lap Counter
        lap_group = QGroupBox('Lap Counter')
        lap_layout = QVBoxLayout(lap_group)
        lap_layout.setContentsMargins(4, 4, 4, 4)
        lap_layout.setSpacing(2)
        self.lap_count_lbl = QLabel('Lap: 1')
        self.lap_count_lbl.setObjectName("lap_count_lbl")
        self.lap_count_lbl.setAlignment(Qt.AlignCenter)
        self.lap_count_lbl.setStyleSheet(f'font-size: 24px; font-weight: bold; color: {self.COLOR_GREEN};')
        lap_layout.addWidget(self.lap_count_lbl)
        
        self.lap_time_lbl = QLabel('Last: --  Best: --')
        self.lap_time_lbl.setObjectName("lap_time_lbl")
        self.lap_time_lbl.setAlignment(Qt.AlignCenter)
        self.lap_time_lbl.setStyleSheet('font-size: 14px;')
        lap_layout.addWidget(self.lap_time_lbl)
        lap_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        left.addWidget(lap_group, stretch=0)
        
        # Speed/Steering Status
        status_group = QGroupBox('Current Status')
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(4, 4, 4, 4)
        status_layout.setSpacing(2)
        self.speed_lbl = QLabel('Speed: 0.00 m/s')
        self.steer_lbl = QLabel('Steering: 0.00 rad')
        self.speed_lbl.setAlignment(Qt.AlignCenter)
        self.steer_lbl.setAlignment(Qt.AlignCenter)
        self.speed_lbl.setStyleSheet('font-size: 16px; font-weight: bold;')
        self.steer_lbl.setStyleSheet('font-size: 16px; font-weight: bold;')
        status_layout.addWidget(self.speed_lbl)
        status_layout.addWidget(self.steer_lbl)
        status_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        left.addWidget(status_group, stretch=0)
        
        # ----- FIXED DRIVING CONTROLS (Moved from right column) -----
        # Emergency Stop
        self.stop_btn = QPushButton('EMERGENCY STOP [SPACE]')
        self.stop_btn.setToolTip('Immediately stop all movement. Press SPACE or click.')
        self.stop_btn.setStyleSheet(f"QPushButton {{ background-color: #cc0000; color: white; font-size: 16px; padding: 15px; border: 3px solid {self.COLOR_RED}; }} QPushButton:hover {{ background-color: {self.COLOR_RED}; }}")
        self.stop_btn.clicked.connect(self.emergency_stop)
        left.addWidget(self.stop_btn)
        
        # Top row: Telemetry + Autonomous toggle
        top_row = QHBoxLayout()
        self.telemetry_btn = QPushButton('Ã°Å¸â€œÅ  TELEMETRY')
        self.telemetry_btn.setToolTip('Open telemetry dashboard')
        self.telemetry_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_BLUE}; color: white; font-size: 12px; padding: 8px; }} QPushButton:hover {{ background-color: #0088ff; }}")
        self.telemetry_btn.clicked.connect(self.open_telemetry_window)
        top_row.addWidget(self.telemetry_btn)
        
        self.auto_btn = QPushButton('AUTONOMOUS')
        self.auto_btn.setCheckable(True)
        self.auto_btn.setToolTip('Toggle autonomous mode')
        self.auto_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; } QPushButton:checked { background-color: #0066cc; border-color: #0088ff; }")
        self.auto_btn.clicked.connect(self.on_auto_toggle)
        top_row.addWidget(self.auto_btn)
        left.addLayout(top_row)
        
        # Start/Pause button (only visible in autonomous mode)
        self.auto_start_btn = QPushButton('Ã¢â€“Â¶ START DRIVING')
        self.auto_start_btn.setCheckable(True)
        self.auto_start_btn.setToolTip('Start/pause autonomous driving')
        self.auto_start_btn.setStyleSheet(f"QPushButton {{ font-size: 14px; padding: 12px; font-weight: bold; }} QPushButton:checked {{ background-color: {self.COLOR_GREEN}; border-color: #00ff00; }}")
        self.auto_start_btn.clicked.connect(self.on_auto_start_toggle)
        self.auto_start_btn.setVisible(False)
        left.addWidget(self.auto_start_btn)
        

        main.addLayout(left, stretch=1)
        
        # ========== RIGHT COLUMN - Controls & Tabbed Tuning ==========
        right = QVBoxLayout()
        right.setSpacing(8)
        
        # ----- TABBED SECTION -----
        self.control_tabs = QTabWidget()
        self.control_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; background: #1a1a1a; }
            QTabBar::tab { background: #2a2a2a; color: #aaa; padding: 8px 12px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #3a3a3a; color: #fff; border-bottom: 2px solid #ff4444; }
            QTabBar::tab:hover { background: #3a3a3a; }
        """)
        
        # ===== TAB 1: CONTROL =====
        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setSpacing(6)
        
        # Manual Controls
        manual_group = QGroupBox('Manual (WASD)')
        manual_layout = QVBoxLayout(manual_group)
        grid = QGridLayout()
        self.btn_w = QPushButton('W')
        self.btn_s = QPushButton('S')
        self.btn_a = QPushButton('A')
        self.btn_d = QPushButton('D')
        for b in [self.btn_w, self.btn_s, self.btn_a, self.btn_d]:
            b.setMinimumSize(50, 40)
        grid.addWidget(self.btn_w, 0, 1)
        grid.addWidget(self.btn_a, 1, 0)
        grid.addWidget(self.btn_d, 1, 2)
        grid.addWidget(self.btn_s, 1, 1)
        self.btn_w.pressed.connect(lambda: self.keys_pressed.add(Qt.Key_W))
        self.btn_w.released.connect(lambda: self.keys_pressed.discard(Qt.Key_W))
        self.btn_s.pressed.connect(lambda: self.keys_pressed.add(Qt.Key_S))
        self.btn_s.released.connect(lambda: self.keys_pressed.discard(Qt.Key_S))
        self.btn_a.pressed.connect(lambda: self.keys_pressed.add(Qt.Key_A))
        self.btn_a.released.connect(lambda: self.keys_pressed.discard(Qt.Key_A))
        self.btn_d.pressed.connect(lambda: self.keys_pressed.add(Qt.Key_D))
        self.btn_d.released.connect(lambda: self.keys_pressed.discard(Qt.Key_D))
        manual_layout.addLayout(grid)
        self.joy_status_lbl = QLabel('Joystick: Not connected')
        self.joy_status_lbl.setAlignment(Qt.AlignCenter)
        self.joy_status_lbl.setStyleSheet('color: #888; font-size: 10px;')
        manual_layout.addWidget(self.joy_status_lbl)
        control_layout.addWidget(manual_group)
        
        # Speed Limits
        speed_group = QGroupBox('Speed Limits')
        speed_layout = QVBoxLayout(speed_group)
        
        fwd_row = QHBoxLayout()
        fwd_row.addWidget(QLabel('Forward:'))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(5)
        self.speed_slider.setMaximum(80)
        self.speed_slider.setValue(20)
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        fwd_row.addWidget(self.speed_slider)
        self.speed_limit_lbl = QLabel('2.0 m/s')
        self.speed_limit_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_GREEN};')
        self.speed_limit_lbl.setMinimumWidth(60)
        fwd_row.addWidget(self.speed_limit_lbl)
        speed_layout.addLayout(fwd_row)
        
        rev_row = QHBoxLayout()
        rev_row.addWidget(QLabel('Reverse:'))
        self.reverse_slider = QSlider(Qt.Horizontal)
        self.reverse_slider.setMinimum(30)
        self.reverse_slider.setMaximum(500)
        self.reverse_slider.setValue(100)
        self.reverse_slider.valueChanged.connect(self.on_reverse_change)
        rev_row.addWidget(self.reverse_slider)
        self.reverse_limit_lbl = QLabel('1.0 m/s')
        self.reverse_limit_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_YELLOW};')
        self.reverse_limit_lbl.setMinimumWidth(60)
        rev_row.addWidget(self.reverse_limit_lbl)
        speed_layout.addLayout(rev_row)
        control_layout.addWidget(speed_group)
        
        control_layout.addStretch()
        help_lbl = QLabel('WASD=Drive  SPACE=Stop')
        help_lbl.setStyleSheet('color: #666; font-size: 10px;')
        help_lbl.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(help_lbl)
        
        self.control_tabs.addTab(control_tab, "Control")
        
        # ===== TAB 2: TUNING =====
        tuning_tab = QWidget()
        tuning_layout = QVBoxLayout(tuning_tab)
        tuning_layout.setContentsMargins(8, 8, 8, 8)
        tuning_layout.setSpacing(6)
        
        # Resolution selector (wrapped in widget for per-mode visibility control)
        self.window_group = QWidget()
        wg_layout = QHBoxLayout(self.window_group)
        wg_layout.setContentsMargins(0, 0, 0, 0)
        wg_layout.addWidget(QLabel('Window:'))
        self.resolution_combo = QComboBox()
        self.resolution_combo.setStyleSheet("QComboBox { font-size: 11px; padding: 4px; background-color: #2a2a2a; color: #fff; }")
        for res_name in self.RESOLUTIONS.keys():
            self.resolution_combo.addItem(res_name)
        self.resolution_combo.setCurrentText("1280Ãƒâ€”800 (16:10)")
        self.resolution_combo.currentIndexChanged.connect(self.on_resolution_change)
        wg_layout.addWidget(self.resolution_combo)
        tuning_layout.addWidget(self.window_group)
        
        # Reset Options
        self.reset_btn = QPushButton('Ã¢â€ Âº Reset to Defaults')
        self.reset_btn.setToolTip("Reset all tuning parameters to their default values")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
                margin-top: 4px;
                margin-bottom: 4px;
            }
            QPushButton:hover { background-color: #444; }
        """)
        self.reset_btn.clicked.connect(self.reset_tuning)
        tuning_layout.addWidget(self.reset_btn)

        # Control Mode
        mode_group = QGroupBox('Algorithm')
        mode_layout = QVBoxLayout(mode_group)
        
        # Mode Badge (Button)
        self.mode_badge = QPushButton("Poly LookAhead Ã¢â€“Â¼")
        self.mode_badge.setToolTip('Click to open Algorithm Selection Menu')
        self.mode_badge.setCursor(Qt.PointingHandCursor)
        self.mode_badge.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(0, 255, 255, 0.1);
                border: 2px solid {self.COLOR_BLUE};
                color: {self.COLOR_BLUE};
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: rgba(0, 255, 255, 0.2);
            }}
        """)
        self.mode_badge.clicked.connect(self.show_mode_overlay)
        mode_layout.addWidget(self.mode_badge)
        
        self.mode_status = QLabel('Active: Poly LookAhead')
        self.mode_status.setStyleSheet('color: #0f0; font-size: 10px;')
        self.mode_status.setAlignment(Qt.AlignCenter)
        mode_layout.addWidget(self.mode_status)

        self.mode_runtime_warning = QLabel('')
        self.mode_runtime_warning.setAlignment(Qt.AlignCenter)
        self.mode_runtime_warning.setWordWrap(True)
        self.mode_runtime_warning.setVisible(False)
        self.mode_runtime_warning.setStyleSheet(
            'color: #ff4444; font-size: 11px; font-weight: bold; '
            'background-color: rgba(255, 68, 68, 0.12); '
            'border: 1px solid #ff4444; border-radius: 4px; padding: 6px;'
        )
        mode_layout.addWidget(self.mode_runtime_warning)
        tuning_layout.addWidget(mode_group)
        
        
        # PD Tuning
        self.pd_group = QGroupBox('PD Control')
        pd_layout = QVBoxLayout(self.pd_group)
        kp_row = QHBoxLayout()
        kp_row.addWidget(QLabel('Kp:'))
        self.kp_slider = QSlider(Qt.Horizontal)
        self.kp_slider.setMinimum(10)
        self.kp_slider.setMaximum(300)
        self.kp_slider.setValue(85)
        self.kp_slider.valueChanged.connect(self.on_pd_change)
        kp_row.addWidget(self.kp_slider)
        self.kp_lbl = QLabel('0.85')
        self.kp_lbl.setMinimumWidth(35)
        kp_row.addWidget(self.kp_lbl)
        pd_layout.addLayout(kp_row)
        kd_row = QHBoxLayout()
        kd_row.addWidget(QLabel('Kd:'))
        self.kd_slider = QSlider(Qt.Horizontal)
        self.kd_slider.setMinimum(0)
        self.kd_slider.setMaximum(100)
        self.kd_slider.setValue(20)
        self.kd_slider.valueChanged.connect(self.on_pd_change)
        kd_row.addWidget(self.kd_slider)
        self.kd_lbl = QLabel('0.20')
        self.kd_lbl.setMinimumWidth(35)
        kd_row.addWidget(self.kd_lbl)
        pd_layout.addLayout(kd_row)
        tuning_layout.addWidget(self.pd_group)

        # MPC Tuning (Advanced)
        self.mpc_group = QGroupBox('MPC Tuning (Advanced)')
        mpc_layout = QVBoxLayout(self.mpc_group)
        
        # Horizon
        horizon_row = QHBoxLayout()
        horizon_row.addWidget(QLabel('Horizon (N):'))
        self.horizon_slider = QSlider(Qt.Horizontal)
        self.horizon_slider.setMinimum(5)
        self.horizon_slider.setMaximum(30)
        self.horizon_slider.setValue(15)
        self.horizon_slider.setToolTip('Prediction Horizon: How many steps into the future the MPC plans')
        self.horizon_slider.valueChanged.connect(lambda v: [self.horizon_lbl.setText(str(v)), self.ros_node.publish_mpc_horizon(v)])
        horizon_row.addWidget(self.horizon_slider)
        self.horizon_lbl = QLabel('15')
        self.horizon_lbl.setMinimumWidth(35)
        horizon_row.addWidget(self.horizon_lbl)
        mpc_layout.addLayout(horizon_row)

        # Tracking Weight
        tracking_row = QHBoxLayout()
        tracking_row.addWidget(QLabel('Tracking:'))
        self.tracking_slider = QSlider(Qt.Horizontal)
        self.tracking_slider.setMinimum(1)
        self.tracking_slider.setMaximum(50) # 0.1 to 5.0
        self.tracking_slider.setValue(10)
        self.tracking_slider.setToolTip('Tracking Strength: Higher = tighter line following (more aggressive)')
        self.tracking_slider.valueChanged.connect(lambda v: [self.tracking_lbl.setText(f"{v/10.0:.1f}"), self.ros_node.publish_mpc_tracking(v/10.0)])
        tracking_row.addWidget(self.tracking_slider)
        self.tracking_lbl = QLabel('1.0')
        self.tracking_lbl.setMinimumWidth(35)
        tracking_row.addWidget(self.tracking_lbl)
        mpc_layout.addLayout(tracking_row)

        # Smoothness Weight
        smoothness_row = QHBoxLayout()
        smoothness_row.addWidget(QLabel('Smoothness:'))
        self.smoothness_slider = QSlider(Qt.Horizontal)
        self.smoothness_slider.setMinimum(1)
        self.smoothness_slider.setMaximum(50) # 0.1 to 5.0
        self.smoothness_slider.setValue(10)
        self.smoothness_slider.setToolTip('Control Smoothness: Higher = gentler steering changes (less twitchy)')
        self.smoothness_slider.valueChanged.connect(lambda v: [self.smoothness_lbl.setText(f"{v/10.0:.1f}"), self.ros_node.publish_mpc_smoothness(v/10.0)])
        smoothness_row.addWidget(self.smoothness_slider)
        self.smoothness_lbl = QLabel('1.0')
        self.smoothness_lbl.setMinimumWidth(35)
        smoothness_row.addWidget(self.smoothness_lbl)
        mpc_layout.addLayout(smoothness_row)
        
        self.mpc_group.setVisible(False) # Hidden by default
        tuning_layout.addWidget(self.mpc_group)
        

        
        # Auto Speed & Slowdown (stored as self for per-mode visibility)
        self.speed_group = QGroupBox('Auto Speed')
        speed_layout = QVBoxLayout(self.speed_group)
        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel('Speed:'))
        self.auto_speed_slider = QSlider(Qt.Horizontal)
        self.auto_speed_slider.setMinimum(5)
        self.auto_speed_slider.setMaximum(80)
        self.auto_speed_slider.setValue(15)
        self.auto_speed_slider.setToolTip('Target speed for autonomous driving (0.5 - 8.0 m/s)')
        self.auto_speed_slider.valueChanged.connect(self.on_auto_speed_change)
        auto_row.addWidget(self.auto_speed_slider)
        self.auto_speed_lbl = QLabel('1.5 m/s')
        self.auto_speed_lbl.setMinimumWidth(50)
        auto_row.addWidget(self.auto_speed_lbl)
        speed_layout.addLayout(auto_row)



        
        # Extras Widget (Hide for MPC)
        self.speed_extras_widget = QWidget()
        extras_layout = QVBoxLayout(self.speed_extras_widget)
        extras_layout.setContentsMargins(0,0,0,0)
        
        slow_row = QHBoxLayout()
        slow_row.addWidget(QLabel('Turn Brake:'))
        self.slowdown_slider = QSlider(Qt.Horizontal)
        self.slowdown_slider.setMinimum(0)
        self.slowdown_slider.setMaximum(100)
        self.slowdown_slider.setValue(35)
        self.slowdown_slider.valueChanged.connect(self.on_slowdown_change)
        slow_row.addWidget(self.slowdown_slider)
        self.slowdown_lbl = QLabel('35%')
        self.slowdown_lbl.setMinimumWidth(35)
        slow_row.addWidget(self.slowdown_lbl)
        extras_layout.addLayout(slow_row)
        
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel('Steer Filter:'))
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setMinimum(30)
        self.smoothing_slider.setMaximum(90)
        self.smoothing_slider.setValue(55)
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_change)
        smooth_row.addWidget(self.smoothing_slider)
        self.smoothing_lbl = QLabel('0.55')
        self.smoothing_lbl.setMinimumWidth(35)
        smooth_row.addWidget(self.smoothing_lbl)
        extras_layout.addLayout(smooth_row)
        
        speed_layout.addWidget(self.speed_extras_widget)
        tuning_layout.addWidget(self.speed_group)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Discrete Speed Presets Group (visible only in DISCRETE mode) Ã¢â€â‚¬Ã¢â€â‚¬
        self.discrete_group = QGroupBox('Discrete Speed Presets')
        self.discrete_group.setStyleSheet(
            f"QGroupBox {{ color: #ffcc00; font-weight: bold; border: 2px solid #ffcc00; "
            f"border-radius: 6px; margin-top: 8px; padding-top: 14px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        disc_layout = QVBoxLayout(self.discrete_group)
        disc_layout.setSpacing(4)

        disc_header = QLabel('Press 1-7 to switch speed instantly   |   0 to stop')
        disc_header.setStyleSheet('color: #aaa; font-size: 10px;')
        disc_header.setAlignment(Qt.AlignCenter)
        disc_layout.addWidget(disc_header)

        self.discrete_spinboxes = []
        self.discrete_indicators = []
        disc_grid = QHBoxLayout()
        disc_grid.setSpacing(6)
        for i in range(7):
            col = QVBoxLayout()
            col.setSpacing(2)
            key_lbl = QLabel(f'{i+1}')
            key_lbl.setStyleSheet(
                'color: #ffcc00; font-size: 18px; font-weight: bold; '
                'background: #2a2200; border: 1px solid #665500; border-radius: 4px; '
                'padding: 4px;')
            key_lbl.setAlignment(Qt.AlignCenter)
            col.addWidget(key_lbl)

            spin = QDoubleSpinBox()
            spin.setRange(0.0, 5.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setValue(self.DISCRETE_SPEED_PRESETS[i])
            spin.setSuffix(' m/s')
            spin.setStyleSheet(
                'QDoubleSpinBox { background: #222; color: #eee; border: 1px solid #555; '
                'border-radius: 3px; padding: 3px; font-size: 12px; } '
                'QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 14px; }')
            idx = i
            spin.valueChanged.connect(lambda val, k=idx: self._on_discrete_preset_changed(k, val))
            col.addWidget(spin)

            indicator = QLabel('')
            indicator.setFixedHeight(6)
            indicator.setStyleSheet('background: #333; border-radius: 3px;')
            col.addWidget(indicator)
            self.discrete_indicators.append(indicator)

            disc_grid.addLayout(col)
            self.discrete_spinboxes.append(spin)

        disc_layout.addLayout(disc_grid)

        self.discrete_banner = QLabel('Speed: Manual (W/S keys)')
        self.discrete_banner.setAlignment(Qt.AlignCenter)
        self.discrete_banner.setStyleSheet(
            'color: #888; font-size: 11px; font-weight: bold; '
            'background: #1a1a1a; padding: 4px; border-radius: 4px;')
        disc_layout.addWidget(self.discrete_banner)

        self.discrete_group.setVisible(False)
        tuning_layout.addWidget(self.discrete_group)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Supervised MPC Group (visible only in SUPERVISED_MPC mode) Ã¢â€â‚¬Ã¢â€â‚¬
        self.supervised_mpc_group = QGroupBox('Supervised MPC')
        self.supervised_mpc_group.setStyleSheet(
            f"QGroupBox {{ color: #00ff88; font-weight: bold; border: 2px solid #00ff88; "
            f"border-radius: 6px; margin-top: 8px; padding-top: 14px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        smpc_layout = QVBoxLayout(self.supervised_mpc_group)
        smpc_layout.setSpacing(4)

        # Steering Assist toggle (Blend / Override)
        steer_row = QHBoxLayout()
        steer_row.setSpacing(6)
        steer_lbl = QLabel('Steering Assist:')
        steer_lbl.setStyleSheet('color: #ccc; font-size: 11px;')
        steer_row.addWidget(steer_lbl)

        self.smpc_blend_btn = QPushButton('Blend')
        self.smpc_override_btn = QPushButton('Override')
        for btn in (self.smpc_blend_btn, self.smpc_override_btn):
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.setStyleSheet(
                'QPushButton { background: #333; color: #aaa; border: 1px solid #555; '
                'border-radius: 4px; padding: 2px 12px; font-size: 11px; } '
                'QPushButton:checked { background: #00ff88; color: #000; font-weight: bold; border-color: #00ff88; }')
        self.smpc_blend_btn.setChecked(True)  # default to Blend
        self.smpc_blend_btn.clicked.connect(lambda: self._set_smpc_steer_mode("blend"))
        self.smpc_override_btn.clicked.connect(lambda: self._set_smpc_steer_mode("override"))
        steer_row.addWidget(self.smpc_blend_btn)
        steer_row.addWidget(self.smpc_override_btn)
        steer_row.addStretch()
        smpc_layout.addLayout(steer_row)

        # Speed presets header
        smpc_header = QLabel('Press 1-7 to lock speed   |   0 to release (MPC auto)')
        smpc_header.setStyleSheet('color: #aaa; font-size: 10px;')
        smpc_header.setAlignment(Qt.AlignCenter)
        smpc_layout.addWidget(smpc_header)

        # 7 speed preset columns
        self.smpc_spinboxes = []
        self.smpc_indicators = []
        smpc_grid = QHBoxLayout()
        smpc_grid.setSpacing(6)
        for i in range(7):
            col = QVBoxLayout()
            col.setSpacing(2)
            key_lbl = QLabel(f'{i+1}')
            key_lbl.setStyleSheet(
                'color: #00ff88; font-size: 18px; font-weight: bold; '
                'background: #002211; border: 1px solid #005533; border-radius: 4px; '
                'padding: 4px;')
            key_lbl.setAlignment(Qt.AlignCenter)
            col.addWidget(key_lbl)

            spin = QDoubleSpinBox()
            spin.setRange(0.0, 6.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setValue(self.SMPC_SPEED_PRESETS[i])
            spin.setSuffix(' m/s')
            spin.setStyleSheet(
                'QDoubleSpinBox { background: #222; color: #eee; border: 1px solid #555; '
                'border-radius: 3px; padding: 3px; font-size: 12px; } '
                'QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 14px; }')
            idx = i
            spin.valueChanged.connect(lambda val, k=idx: self._on_smpc_preset_changed(k, val))
            col.addWidget(spin)

            indicator = QLabel('')
            indicator.setFixedHeight(6)
            indicator.setStyleSheet('background: #333; border-radius: 3px;')
            col.addWidget(indicator)
            self.smpc_indicators.append(indicator)

            smpc_grid.addLayout(col)
            self.smpc_spinboxes.append(spin)

        smpc_layout.addLayout(smpc_grid)

        # Status banner
        self.smpc_banner = QLabel('Speed: MPC Auto   |   Steering: Blend (A/D)')
        self.smpc_banner.setAlignment(Qt.AlignCenter)
        self.smpc_banner.setStyleSheet(
            'color: #888; font-size: 11px; font-weight: bold; '
            'background: #1a1a1a; padding: 4px; border-radius: 4px;')
        smpc_layout.addWidget(self.smpc_banner)

        self.supervised_mpc_group.setVisible(False)
        tuning_layout.addWidget(self.supervised_mpc_group)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Expo Mode Group (visible only in EXPO mode) Ã¢â€â‚¬Ã¢â€â‚¬
        self.expo_group = QGroupBox('Expo Mode')
        self.expo_group.setStyleSheet(
            f"QGroupBox {{ color: #ffcc00; font-weight: bold; border: 2px solid #ffcc00; "
            f"border-radius: 6px; margin-top: 8px; padding-top: 14px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        expo_layout = QVBoxLayout(self.expo_group)
        expo_layout.setSpacing(6)

        # Description box for team members (2-column layout)
        expo_desc_frame = QFrame()
        expo_desc_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        expo_desc_frame.setStyleSheet(
            'QFrame { background: #1a1a00; border: 1px solid #665500; '
            'border-radius: 4px; padding: 8px; }')
        desc_grid = QGridLayout(expo_desc_frame)
        desc_grid.setSpacing(8)
        desc_grid.setContentsMargins(10, 8, 10, 8)

        # Left column: What is Expo Mode
        left_title = QLabel('OVERVIEW')
        left_title.setStyleSheet('color: #ffcc00; font-size: 13px; font-weight: bold; background: transparent; border: none;')
        desc_grid.addWidget(left_title, 0, 0)

        left_body = QLabel(
            'Car must be on a raised stand with\n'
            'wheels off the ground. This mode\n'
            'sends choreographed motor and servo\n'
            'commands to demonstrate the system.\n\n'
            'No camera or track detection is used.\n'
            'Press SPACE for emergency stop.')
        left_body.setWordWrap(True)
        left_body.setStyleSheet('color: #ccc; font-size: 12px; background: transparent; border: none;')
        desc_grid.addWidget(left_body, 1, 0)

        # Right column: Routines reference
        right_title = QLabel('ROUTINES')
        right_title.setStyleSheet('color: #ffcc00; font-size: 13px; font-weight: bold; background: transparent; border: none;')
        desc_grid.addWidget(right_title, 0, 1)

        right_body = QLabel(
            'Sweep\n'
            '  Constant speed, steering sweeps L/R\n\n'
            'Heartbeat\n'
            '  Short speed bursts + servo snaps\n\n'
            'Wave\n'
            '  Speed and steer, 90 deg offset\n\n'
            'Figure-8\n'
            '  Steer at 1x freq, speed at 2x freq\n\n'
            'All (Auto-Cycle)\n'
            '  Rotates through each every 10s')
        right_body.setWordWrap(True)
        right_body.setStyleSheet('color: #ccc; font-size: 12px; background: transparent; border: none;')
        desc_grid.addWidget(right_body, 1, 1)

        desc_grid.setColumnStretch(0, 1)
        desc_grid.setColumnStretch(1, 1)
        expo_layout.addWidget(expo_desc_frame)

        intro_row = QHBoxLayout()
        self.expo_show_intro_cb = QCheckBox('Show welcome animation when entering Expo Mode')
        self.expo_show_intro_cb.setChecked(True)
        self.expo_show_intro_cb.setStyleSheet('color: #bbb; font-size: 11px;')
        intro_row.addWidget(self.expo_show_intro_cb)
        intro_row.addStretch(1)
        self.expo_replay_intro_btn = QPushButton('Replay welcome animation')
        self.expo_replay_intro_btn.setCursor(Qt.PointingHandCursor)
        self.expo_replay_intro_btn.setStyleSheet(
            'QPushButton { color: #c9a227; background: #2a2200; border: 1px solid #c9a227; '
            'border-radius: 4px; padding: 4px 10px; font-size: 11px; }'
            'QPushButton:hover { background: #3a3000; }')
        self.expo_replay_intro_btn.clicked.connect(self._replay_expo_welcome_overlay)
        intro_row.addWidget(self.expo_replay_intro_btn)
        expo_layout.addLayout(intro_row)

        # Routine selector
        routine_row = QHBoxLayout()
        routine_row.setSpacing(6)
        routine_lbl = QLabel('Routine:')
        routine_lbl.setStyleSheet('color: #ccc; font-size: 11px;')
        routine_row.addWidget(routine_lbl)
        self.expo_routine_combo = QComboBox()
        self.expo_routine_combo.addItems(['Sweep', 'Heartbeat', 'Wave', 'Figure-8', 'All (Auto-Cycle)'])
        self.expo_routine_combo.setStyleSheet(
            'QComboBox { background: #222; color: #eee; border: 1px solid #ffcc00; '
            'border-radius: 3px; padding: 4px; font-size: 11px; } '
            'QComboBox QAbstractItemView { background: #222; color: #eee; '
            'selection-background-color: #ffcc00; selection-color: #000; '
            'border: 1px solid #ffcc00; outline: 0px; } '
            'QComboBox QAbstractItemView::item { min-height: 24px; padding-left: 6px; }')
        self.expo_routine_combo.currentIndexChanged.connect(self._on_expo_routine_changed)
        routine_row.addWidget(self.expo_routine_combo)
        routine_row.addStretch(1)
        expo_layout.addLayout(routine_row)

        # Speed amplitude slider
        speed_row = QHBoxLayout()
        speed_row.setSpacing(4)
        speed_row.addWidget(QLabel('Speed:'))
        self.expo_speed_slider = QSlider(Qt.Horizontal)
        self.expo_speed_slider.setMinimum(0)
        self.expo_speed_slider.setMaximum(30)  # 0.0 - 3.0 m/s
        self.expo_speed_slider.setValue(10)     # default 1.0 m/s
        self.expo_speed_slider.valueChanged.connect(self._on_expo_speed_changed)
        speed_row.addWidget(self.expo_speed_slider)
        self.expo_speed_lbl = QLabel('1.0 m/s')
        self.expo_speed_lbl.setMinimumWidth(50)
        self.expo_speed_lbl.setStyleSheet('color: #ffcc00; font-weight: bold;')
        speed_row.addWidget(self.expo_speed_lbl)
        expo_layout.addLayout(speed_row)

        # Steering amplitude slider
        steer_row = QHBoxLayout()
        steer_row.setSpacing(4)
        steer_row.addWidget(QLabel('Steer:'))
        self.expo_steer_slider = QSlider(Qt.Horizontal)
        self.expo_steer_slider.setMinimum(0)
        self.expo_steer_slider.setMaximum(45)  # 0.0 - 0.45 rad
        self.expo_steer_slider.setValue(35)      # default 0.35 rad
        self.expo_steer_slider.valueChanged.connect(self._on_expo_steer_changed)
        steer_row.addWidget(self.expo_steer_slider)
        self.expo_steer_lbl = QLabel('0.35 rad')
        self.expo_steer_lbl.setMinimumWidth(50)
        self.expo_steer_lbl.setStyleSheet('color: #ffcc00; font-weight: bold;')
        steer_row.addWidget(self.expo_steer_lbl)
        expo_layout.addLayout(steer_row)

        # Frequency slider
        freq_row = QHBoxLayout()
        freq_row.setSpacing(4)
        freq_row.addWidget(QLabel('Freq:'))
        self.expo_freq_slider = QSlider(Qt.Horizontal)
        self.expo_freq_slider.setMinimum(5)   # 0.5 Hz
        self.expo_freq_slider.setMaximum(40)  # 4.0 Hz
        self.expo_freq_slider.setValue(10)     # default 1.0 Hz
        self.expo_freq_slider.valueChanged.connect(self._on_expo_freq_changed)
        freq_row.addWidget(self.expo_freq_slider)
        self.expo_freq_lbl = QLabel('1.0 Hz')
        self.expo_freq_lbl.setMinimumWidth(50)
        self.expo_freq_lbl.setStyleSheet('color: #ffcc00; font-weight: bold;')
        freq_row.addWidget(self.expo_freq_lbl)
        expo_layout.addLayout(freq_row)

        # Battery voltage display
        batt_row = QHBoxLayout()
        batt_row.setSpacing(6)
        batt_label = QLabel('Battery Level:')
        batt_label.setStyleSheet('color: #ccc; font-size: 12px; font-weight: bold;')
        batt_row.addWidget(batt_label)
        self.expo_batt_bar = QProgressBar()
        self.expo_batt_bar.setMinimum(0)
        self.expo_batt_bar.setMaximum(100)
        self.expo_batt_bar.setValue(0)
        self.expo_batt_bar.setFixedHeight(28)
        self.expo_batt_bar.setTextVisible(True)
        self.expo_batt_bar.setFormat('N/A')
        self.expo_batt_bar.setStyleSheet(
            'QProgressBar { background: #222; border: 1px solid #555; border-radius: 4px; '
            'text-align: center; color: #fff; font-size: 12px; font-weight: bold; } '
            'QProgressBar::chunk { background: #555; border-radius: 3px; }')
        batt_row.addWidget(self.expo_batt_bar, 1)
        self.expo_batt_volts_lbl = QLabel('-- V')
        self.expo_batt_volts_lbl.setStyleSheet('color: #888; font-size: 13px; font-weight: bold;')
        self.expo_batt_volts_lbl.setMinimumWidth(55)
        batt_row.addWidget(self.expo_batt_volts_lbl)
        expo_layout.addLayout(batt_row)

        # Start / Stop button
        self.expo_start_btn = QPushButton('START EXPO')
        self.expo_start_btn.setCheckable(True)
        self.expo_start_btn.setStyleSheet(
            'QPushButton { background: #2a2200; color: #ffcc00; border: 2px solid #ffcc00; '
            'border-radius: 6px; padding: 8px; font-size: 14px; font-weight: bold; } '
            'QPushButton:checked { background: #ffcc00; color: #000; } '
            'QPushButton:hover { background: #3a3000; }')
        self.expo_start_btn.toggled.connect(self._on_expo_toggle)
        expo_layout.addWidget(self.expo_start_btn)

        # Status banner
        self.expo_banner = QLabel('Ready -- Select a routine and press START')
        self.expo_banner.setAlignment(Qt.AlignCenter)
        self.expo_banner.setStyleSheet(
            'color: #888; font-size: 11px; font-weight: bold; '
            'background: #1a1a1a; padding: 4px; border-radius: 4px;')
        expo_layout.addWidget(self.expo_banner)
        
        # Add stretch to absorb vertical space so the box doesn't expand
        expo_layout.addStretch(1)

        self.expo_group.setVisible(False)
        tuning_layout.addWidget(self.expo_group)

        tuning_layout.addStretch()
        self.control_tabs.addTab(tuning_tab, "Tuning")
        
        # ===== TAB 3: VISION =====
        vision_tab = QWidget()
        vision_layout = QVBoxLayout(vision_tab)
        vision_layout.setContentsMargins(8, 8, 8, 8)
        vision_layout.setSpacing(6)
        
        # Track Width (LEGACY only)
        self.width_group = QGroupBox('Track Width (LEGACY)')
        width_layout = QVBoxLayout(self.width_group)
        width_row = QHBoxLayout()
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setMinimum(200)
        self.width_slider.setMaximum(800)
        self.width_slider.setValue(550)
        self.width_slider.valueChanged.connect(self.on_width_change)
        width_row.addWidget(self.width_slider)
        self.width_lbl = QLabel('550 px')
        self.width_lbl.setMinimumWidth(50)
        width_row.addWidget(self.width_lbl)
        width_layout.addLayout(width_row)
        self.width_group.setVisible(False)  # Hidden by default (POLY mode)
        vision_layout.addWidget(self.width_group)
        
        # Steering Bias
        bias_group = QGroupBox('Steering Bias')
        bias_layout = QVBoxLayout(bias_group)
        bias_row = QHBoxLayout()
        self.bias_slider = QSlider(Qt.Horizontal)
        self.bias_slider.setMinimum(-20)
        self.bias_slider.setMaximum(20)
        self.bias_slider.setValue(0)
        self.bias_slider.valueChanged.connect(self.on_bias_change)
        bias_row.addWidget(self.bias_slider)
        self.bias_lbl = QLabel('0.00')
        self.bias_lbl.setMinimumWidth(40)
        bias_row.addWidget(self.bias_lbl)
        bias_layout.addLayout(bias_row)
        vision_layout.addWidget(bias_group)
        
        # Safety Settings
        safety_group = QGroupBox('Safety')
        safety_layout = QVBoxLayout(safety_group)
        self.obstacle_toggle = QPushButton('Obstacle: ON')
        self.obstacle_toggle.setCheckable(True)
        self.obstacle_toggle.setChecked(True)
        self.obstacle_toggle.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_GREEN}; color: white; padding: 6px; }} QPushButton:checked {{ background-color: {self.COLOR_GREEN}; }}")
        self.obstacle_toggle.clicked.connect(self.on_obstacle_toggle)
        safety_layout.addWidget(self.obstacle_toggle)
        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel('Stop Dist:'))
        self.distance_input = QLineEdit('0.50')
        self.distance_input.setStyleSheet('font-size: 12px; padding: 4px; background: #2a2a2a; color: white; border: 1px solid #555;')
        self.distance_input.setMaximumWidth(60)
        self.distance_input.setToolTip('Stop distance in meters (0 - 20)\n0.3m=Close, 0.5m=Safe, 1.0m+=Far')
        # Add validator to restrict input
        distance_validator = QDoubleValidator(0.0, 20.0, 2)
        distance_validator.setNotation(QDoubleValidator.StandardNotation)
        self.distance_input.setValidator(distance_validator)
        self.distance_input.returnPressed.connect(self.on_distance_apply)
        dist_row.addWidget(self.distance_input)
        self.distance_apply_btn = QPushButton('Set')
        self.distance_apply_btn.setStyleSheet(f'background: {self.COLOR_BLUE}; color: white; padding: 4px 8px;')
        self.distance_apply_btn.clicked.connect(self.on_distance_apply)
        dist_row.addWidget(self.distance_apply_btn)
        self.distance_status_lbl = QLabel('0.50m')
        self.distance_status_lbl.setStyleSheet(f'color: {self.COLOR_GREEN}; font-weight: bold;')
        dist_row.addWidget(self.distance_status_lbl)
        safety_layout.addLayout(dist_row)
        vision_layout.addWidget(safety_group)
        
        # Path Detection Settings (Poly LookAhead / MPC)
        self.advanced_group = QGroupBox('Path Detection')
        advanced_layout = QVBoxLayout(self.advanced_group)
        look_row = QHBoxLayout()
        look_row.addWidget(QLabel('Lookahead:'))
        self.lookahead_slider = QSlider(Qt.Horizontal)
        self.lookahead_slider.setMinimum(5)
        self.lookahead_slider.setMaximum(95)
        self.lookahead_slider.setValue(30)
        self.lookahead_slider.valueChanged.connect(self.on_lookahead_change)
        look_row.addWidget(self.lookahead_slider)
        self.lookahead_lbl = QLabel('0.30')
        self.lookahead_lbl.setMinimumWidth(35)
        look_row.addWidget(self.lookahead_lbl)
        advanced_layout.addLayout(look_row)
        bev_w_row = QHBoxLayout()
        bev_w_row.addWidget(QLabel('BEV Width:'))
        self.bev_width_slider = QSlider(Qt.Horizontal)
        self.bev_width_slider.setMinimum(20)
        self.bev_width_slider.setMaximum(60)
        self.bev_width_slider.setValue(40)
        self.bev_width_slider.valueChanged.connect(self.on_bev_width_change)
        bev_w_row.addWidget(self.bev_width_slider)
        self.bev_width_lbl = QLabel('0.40')
        self.bev_width_lbl.setMinimumWidth(35)
        bev_w_row.addWidget(self.bev_width_lbl)
        advanced_layout.addLayout(bev_w_row)
        bev_p_row = QHBoxLayout()
        bev_p_row.addWidget(QLabel('BEV Pad:'))
        self.bev_padding_slider = QSlider(Qt.Horizontal)
        self.bev_padding_slider.setMinimum(10)
        self.bev_padding_slider.setMaximum(40)
        self.bev_padding_slider.setValue(20)
        self.bev_padding_slider.valueChanged.connect(self.on_bev_padding_change)
        bev_p_row.addWidget(self.bev_padding_slider)
        self.bev_padding_lbl = QLabel('0.20')
        self.bev_padding_lbl.setMinimumWidth(35)
        bev_p_row.addWidget(self.bev_padding_lbl)
        advanced_layout.addLayout(bev_p_row)
        vision_layout.addWidget(self.advanced_group)
        
        vision_layout.addStretch()
        self.control_tabs.addTab(vision_tab, "Vision")


        
        # Add tabs to right column
        right.addWidget(self.control_tabs)
        
        # Store widgets for enabling/disabling
        self.tuning_widgets = [self.pd_group, self.mpc_group, self.advanced_group, self.reset_btn]
        self.manual_widgets = [manual_group, speed_group]
        self.set_mode_controls(False)  # Start in manual mode
        
        main.addLayout(right, stretch=1)
        
        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("QStatusBar { background-color: #2a2a2a; color: #eaeaea; border-top: 1px solid #444; } QStatusBar::item { border: none; }")
        self.status_bar.showMessage("Initializing...")
        self.update_status_bar()
        
        # Apply initial font scaling
        self.update_font_scaling()
        
        # Setup Overlay (must be last to cover everything)
        self.setup_overlay()
    
    def update_status_bar(self):
        """Update status bar with connection status and FPS."""
        camera_status = "Camera: Connected" if self.ros_node.connected else "Camera: Disconnected"
        joy_status = "Joystick: Connected" if self.ros_node.joy_connected else "Joystick: Disconnected"
        fps_text = f"FPS: {self.current_fps:.1f}" if self.current_fps > 0 else "FPS: --"
        mode_text = f"Mode: {self.ros_node.control_mode if hasattr(self.ros_node, 'control_mode') else 'POLY_LOOKAHEAD'}"
        
        # Add diagnostic info about image reception
        raw_info = f"Raw: {self.raw_image_received_count}" if self.raw_image_received_count > 0 else "Raw: 0"
        debug_info = f"Debug: {self.debug_image_received_count}" if self.debug_image_received_count > 0 else ""
        april_tag_info = f"AprilTag: {self.april_tag_image_received_count}" if self.april_tag_image_received_count > 0 else ""
        img_info = f" | {raw_info}" + (f" {debug_info}" if debug_info else "") + (f" {april_tag_info}" if april_tag_info else "")
        
        # Use plain text since QStatusBar.showMessage() doesn't support HTML
        status_text = f"{camera_status} | {joy_status} | {fps_text} | {mode_text}{img_info}"
        self.status_bar.showMessage(status_text)

    
    def on_view_changed(self, view_id):
        """Handle view selection change from segmented buttons."""
        if view_id == 0:
            self.current_view = "raw"
            # Disable debug and april_tag views
            debug_msg = Bool()
            debug_msg.data = False
            self.ros_node.debug_enabled_pub.publish(debug_msg)
            april_tag_msg = Bool()
            april_tag_msg.data = False
            self.ros_node.april_tag_view_enabled_pub.publish(april_tag_msg)
            # Display raw image if available
            if self.last_raw_image is not None:
                self.display_image(self.last_raw_image)
        elif view_id == 1:
            self.current_view = "debug"
            # Enable debug view, disable april_tag
            debug_msg = Bool()
            debug_msg.data = True
            self.ros_node.debug_enabled_pub.publish(debug_msg)
            april_tag_msg = Bool()
            april_tag_msg.data = False
            self.ros_node.april_tag_view_enabled_pub.publish(april_tag_msg)
            # Display debug image if available
            if self.last_debug_image is not None:
                self.display_image(self.last_debug_image)
        elif view_id == 2:
            self.current_view = "april_tag"
            # Enable april_tag view, disable debug
            debug_msg = Bool()
            debug_msg.data = False
            self.ros_node.debug_enabled_pub.publish(debug_msg)
            april_tag_msg = Bool()
            april_tag_msg.data = True
            self.ros_node.april_tag_view_enabled_pub.publish(april_tag_msg)
            # Display april tag image if available
            if self.last_april_tag_image is not None:
                self.display_image(self.last_april_tag_image)
            
    def on_raw_image(self, img):
        if img is None:
            return
        self.last_raw_image = img
        self.raw_image_received_count += 1

        if self.current_view == "raw":
            self.display_image(img)
        # Update status on first image received
        if self.raw_image_received_count == 1:
            self.status_bar.showMessage('Status: Raw camera images received!')
    
    def on_debug_image(self, img):
        if img is None:
            return
        self.last_debug_image = img
        self.debug_image_received_count += 1
        if self.current_view == "debug":
            self.display_image(img)
    
    def on_april_tag_image(self, img):
        if img is None:
            return
        self.last_april_tag_image = img
        self.april_tag_image_received_count += 1
        if self.current_view == "april_tag":
            self.display_image(img)
    
    def display_image(self, img):
        if img is None: return
        try:
            # Check image
            if img.size == 0:
                return
            
            # Update FPS counter (every 10 frames)
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                import time
                current_time = time.time()
                self.frame_timestamps.append(current_time)
                
                # Calculate FPS from timestamps in last second
                if len(self.frame_timestamps) > 1:
                    time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
                    if time_span > 0:
                        self.current_fps = (len(self.frame_timestamps) - 1) / time_span
                    else:
                        self.current_fps = 0.0
                else:
                    self.current_fps = 0.0
            
            # Handle grayscale images (like mask)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                return  # Bad format
                
            h, w, c = img.shape
            if h == 0 or w == 0:
                return
            
            # Add FPS overlay
            fps_text = f"FPS: {self.current_fps:.1f}" if self.current_fps > 0 else "FPS: --"
            cv2.putText(img, fps_text, (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            # Create QImage
            img_contiguous = np.ascontiguousarray(img)
            qimg = QImage(img_contiguous.data, w, h, c*w, QImage.Format_RGB888)
            
            # Scale and display
            pixmap = QPixmap.fromImage(qimg)
            # Ensure camera_label has a valid size (use minimum size if not yet laid out)
            label_size = self.camera_label.size()
            if label_size.width() <= 1 or label_size.height() <= 1:
                label_size = self.camera_label.minimumSize()
            scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled)
            # Clear any text that might be showing
            self.camera_label.setText("")
        except Exception as e:
            # Log error but don't spam - only log once per second
            import time
            if not hasattr(self, '_last_display_error_time'):
                self._last_display_error_time = 0.0
            current_time = time.time()
            if current_time - self._last_display_error_time > 1.0:
                print(f"[GUI] Display image error: {e}")  # Print to console for debugging
                self._last_display_error_time = current_time
    
    def keyPressEvent(self, e):
        if e.isAutoRepeat(): return
        k = e.key()
        
        # Map WASD/Arrows to a canonical key format
        mapped_key = k
        if k in [Qt.Key_W, Qt.Key_Up]: mapped_key = Qt.Key_W
        elif k in [Qt.Key_S, Qt.Key_Down]: mapped_key = Qt.Key_S
        elif k in [Qt.Key_A, Qt.Key_Left]: mapped_key = Qt.Key_A
        elif k in [Qt.Key_D, Qt.Key_Right]: mapped_key = Qt.Key_D
        
        # X11 Auto-Repeat Debouncer: cancel any pending release
        if hasattr(self, 'key_release_times') and mapped_key in self.key_release_times:
            del self.key_release_times[mapped_key]
            
        self.keys_pressed.add(mapped_key)
        
        if k == Qt.Key_Space: self.emergency_stop()

        # Discrete Mode: keys 1-5 set speed presets, 0 cancels
        elif self.ros_node.control_mode == 'DISCRETE':
            if Qt.Key_1 <= k <= Qt.Key_7:
                idx = k - Qt.Key_1
                self._set_discrete_speed(idx)
            elif k == Qt.Key_0:
                self._clear_discrete_speed()

        # Supervised MPC Mode: keys 1-7 set speed presets, 0 releases to MPC auto
        elif self.ros_node.control_mode == 'SUPERVISED_MPC':
            if Qt.Key_1 <= k <= Qt.Key_7:
                idx = k - Qt.Key_1
                self._set_smpc_speed(idx)
            elif k == Qt.Key_0:
                self._clear_smpc_speed()
    
    def keyReleaseEvent(self, e):
        if e.isAutoRepeat(): return
        k = e.key()
        
        mapped_key = k
        if k in [Qt.Key_W, Qt.Key_Up]: mapped_key = Qt.Key_W
        elif k in [Qt.Key_S, Qt.Key_Down]: mapped_key = Qt.Key_S
        elif k in [Qt.Key_A, Qt.Key_Left]: mapped_key = Qt.Key_A
        elif k in [Qt.Key_D, Qt.Key_Right]: mapped_key = Qt.Key_D
        
        # Debounce X11 auto-repeat mask: mark release time, don't discard blindly
        if not hasattr(self, 'key_release_times'): self.key_release_times = {}
        self.key_release_times[mapped_key] = time.time()
    
    def process_keys(self):
        # Clean up expired keystrokes (X11 debouncing)
        if hasattr(self, 'key_release_times'):
            now = time.time()
            expired = [k for k, t in self.key_release_times.items() if now - t > 0.05]
            for k in expired:
                self.keys_pressed.discard(k)
                del self.key_release_times[k]

        if self.ros_node.autonomous_enabled:
            # Special case: Supervised MPC steering assist via A/D keys
            if self.ros_node.control_mode == 'SUPERVISED_MPC':
                steer = 0.0
                if Qt.Key_A in self.keys_pressed:
                    steer = self.ros_node.max_steering_angle
                elif Qt.Key_D in self.keys_pressed:
                    steer = -self.ros_node.max_steering_angle
                
                # Publish steering assist + speed override to vision controller
                self.ros_node.publish_smpc_override(
                    steer_assist=steer,
                    steer_mode=self.smpc_steer_mode,
                    speed_override=self.smpc_speed,
                )
                self._update_smpc_banner()
                return
            else:
                return  # Disable manual keys in autonomous mode
            
        # Keyboard inputs
        key_speed, key_steer = 0.0, 0.0
        
        if self.discrete_speed is not None and self.ros_node.control_mode == 'DISCRETE':
            # Discrete mode: speed is locked to preset, W/S ignored
            key_speed = self.discrete_speed
        elif self.ros_node.control_mode == 'EXPO':
            # Expo mode: all manual input disabled, expo timer drives everything
            return
        else:
            if Qt.Key_W in self.keys_pressed: key_speed = self.max_speed
            elif Qt.Key_S in self.keys_pressed: key_speed = -self.max_reverse_speed
        
        if Qt.Key_A in self.keys_pressed: key_steer = self.ros_node.max_steering_angle
        elif Qt.Key_D in self.keys_pressed: key_steer = -self.ros_node.max_steering_angle
        
        # Joystick inputs (from ROS node)
        joy_speed = self.ros_node.joy_speed
        joy_steer = self.ros_node.joy_steering
        joy_active = self.ros_node.joy_deadman_pressed
        
        # Joystick takes priority over keyboard
        if joy_active:
            final_speed = joy_speed
            final_steer = joy_steer
        elif key_speed != 0 or key_steer != 0:
            final_speed = key_speed
            final_steer = key_steer
        else:
            final_speed = 0.0
            final_steer = 0.0
            
        self.ros_node.set_manual_control(final_speed, final_steer)
    
    def emergency_stop(self):
        # Clear any key presses
        self.keys_pressed.clear()
        
        
        # Stop Expo routine
        if hasattr(self, 'expo_active') and self.expo_active:
            self._stop_expo()

        
        # Stop the car via ROS node
        self.ros_node.emergency_stop()
        
        # Reset all GUI states to manual mode
        self.auto_btn.setChecked(False)
        self.auto_btn.setText('ENTER AUTONOMOUS MODE')
        self.auto_start_btn.setChecked(False)
        self.auto_start_btn.setText('Ã¢â€“Â¶ START DRIVING')
        self.auto_start_btn.setStyleSheet(f"QPushButton {{ font-size: 16px; padding: 15px; font-weight: bold; }} QPushButton:checked {{ background-color: {self.COLOR_GREEN}; border-color: #00ff00; }}")  # Reset style
        self.auto_start_btn.setVisible(False)
        # Reset lap counter display
        self.lap_count_lbl.setText('Lap: 1')
        self.lap_count_lbl.setStyleSheet(f'font-size: 24px; font-weight: bold; color: {self.COLOR_GREEN};')
        self.lap_time_lbl.setText('Last: --  Best: --')
        
        # Enable manual controls, disable tuning
        self.set_mode_controls(False)
        
        # Reset status message after 3 seconds
        QTimer.singleShot(3000, self.reset_status)
    
    def reset_status(self):
        pass
    
    def update_status(self, s):
        pass
    
    # Lap counter update (AprilTag Based)
    def update_lap(self, lap_count, lap_time):
        # lap_count is the number of COMPLETED laps.
        # Display: max(1, lap_count) Ã¢â‚¬â€ starts at 1, stays at 1 after first crossing
        display_lap = max(1, lap_count)
        
        # Check if we just crossed the line for the very first time
        if not hasattr(self, '_last_raw_lap'):
            self._last_raw_lap = 0
            
        if self._last_raw_lap == 0 and lap_count == 1:
            # First lap triggered!
            self.lap_count_lbl.setText('Ã¢Å“â€ LAP COUNTING TRIGGERED Ã¢Å“â€')
            self.lap_count_lbl.setStyleSheet(f'font-size: 16px; font-weight: bold; color: {self.COLOR_YELLOW};')
            
            # Reset back to normal view after 2.5 seconds (display stays at 1)
            QTimer.singleShot(2500, lambda: self._restore_lap_display(display_lap))
        elif lap_count != self._last_raw_lap:
            # Normal lap update
            self._restore_lap_display(display_lap)
            
        self._last_raw_lap = lap_count

        if lap_count == 0:
            self.lap_time_lbl.setText('Last: --  Best: --')
        elif lap_time > 0.0:
            best = self.ros_node.best_lap_time
            if best < float('inf'):
                self.lap_time_lbl.setText(f'Last: {lap_time:.2f}s  Best: {best:.2f}s')
            else:
                self.lap_time_lbl.setText(f'Last: {lap_time:.2f}s  Best: --')
        else:
            self.lap_time_lbl.setText('Last: --  Best: --')

    def _restore_lap_display(self, display_lap):
        self.lap_count_lbl.setText(f'Lap: {display_lap}')
        self.lap_count_lbl.setStyleSheet(f'font-size: 24px; font-weight: bold; color: {self.COLOR_GREEN};')
    
    def update_joystick(self, connected, deadman, speed, steering):
        if connected:
            if deadman:
                self.joy_status_lbl.setText(f'Joystick: ACTIVE (LB held)')
                self.joy_status_lbl.setStyleSheet(f'color: {self.COLOR_GREEN}; font-size: 12px; font-weight: bold; margin-top: 8px;')
            else:
                self.joy_status_lbl.setText(f'Joystick: Ready (hold LB to drive)')
                self.joy_status_lbl.setStyleSheet(f'color: {self.COLOR_YELLOW}; font-size: 12px; margin-top: 8px;')
        else:
            self.joy_status_lbl.setText('Joystick: Not connected')
            self.joy_status_lbl.setStyleSheet('color: #888; font-size: 12px; margin-top: 8px;')
    
    def update_ui(self):
        self.speed_lbl.setText(f'Speed: {self.ros_node.current_speed:.2f} m/s')
        self.steer_lbl.setText(f'Steering: {self.ros_node.current_steering:.2f} rad')
        self.update_status_bar()
        
        # Ensure raw image is displayed if we're in raw view mode and have an image
        # This catches cases where initial display was missed
        if self.current_view == "raw" and self.last_raw_image is not None:
            # Check if camera_label is still showing text (no pixmap set)
            if self.camera_label.pixmap() is None or self.camera_label.pixmap().isNull():
                # Try to display the raw image
                self.display_image(self.last_raw_image)
        
        # Update visual indicators for autonomous running state
        if self.ros_node.autonomous_running and self.auto_start_btn.isChecked():
            # Ensure pulsing green style is maintained
            if not self.auto_start_btn.styleSheet().startswith("QPushButton"):
                self.auto_start_btn.setStyleSheet(f"""
                    QPushButton {{
                        font-size: 16px;
                        padding: 15px;
                        font-weight: bold;
                        background-color: {self.COLOR_GREEN};
                        border: 3px solid #00ff00;
                        color: white;
                    }}
                    QPushButton:hover {{
                        background-color: #00cc00;
                    }}
                """)
    
    def on_telemetry(self, record):
        """Handle incoming telemetry data."""
        # Add to collector
        self.telemetry_collector.record(record)
        
        # Forward to telemetry window if open
        if self.telemetry_window is not None:
            self.telemetry_window.add_record(record)
            

        if hasattr(self, 'mode_runtime_warning'):
            if active_mode == 'IL' and runtime_source != 'IL':
                self.mode_runtime_warning.setText(
                    f'IL selected, but runtime is {runtime_source}. '
                    'This usually means IL failed to register or the model/runtime could not load.'
                )
                self.mode_runtime_warning.setVisible(True)
            else:
                self.mode_runtime_warning.clear()
                self.mode_runtime_warning.setVisible(False)
        

        
        # LLA friction status
        if hasattr(self, 'lla_group') and self.lla_group.isVisible() and 'selected_mu' in mode_data:
            mu = mode_data['selected_mu']
            runtime_state = mode_data.get('lla_runtime_state', 'OK')
            runtime_reason = mode_data.get('lla_runtime_reason', '')
            runtime_source = mode_data.get('runtime_source', 'LLA_MPC')
            if runtime_state == 'OK':
                self.lla_friction_lbl.setText(f"Friction (mu): {mu:.2f} | Runtime: OK")
                self.lla_friction_lbl.setStyleSheet(f'color: {self.COLOR_GREEN}; font-size: 11px;')
            else:
                reason_txt = f" ({runtime_reason})" if runtime_reason else ""
                self.lla_friction_lbl.setText(f"Friction (mu): {mu:.2f} | Runtime: {runtime_state}{reason_txt}")
                self.lla_friction_lbl.setStyleSheet(f'color: {self.COLOR_YELLOW}; font-size: 11px; font-weight: bold;')
            history = mode_data.get('history_size', 0)
            models = mode_data.get('model_count', 0)
            solver_fails = mode_data.get('consecutive_solver_failures', 0)
            self.lla_models_lbl.setText(
                f"Model bank: {models} models | History: {history} | Source: {runtime_source} | Solver fails: {solver_fails}"
            )
    
    def open_telemetry_window(self):
        """Open the telemetry dashboard window."""
        from .telemetry_window import TelemetryWindow
        
        if self.telemetry_window is None or not self.telemetry_window.isVisible():
            self.telemetry_window = TelemetryWindow(self.telemetry_collector, self)
            self.telemetry_window.show()
        else:
            # Bring existing window to front
            self.telemetry_window.raise_()
            self.telemetry_window.activateWindow()
    
    def on_auto_toggle(self, checked):
        self.ros_node.set_autonomous(checked)
        self.auto_btn.setText('EXIT AUTONOMOUS MODE' if checked else 'ENTER AUTONOMOUS MODE')
        self.auto_start_btn.setVisible(checked)  # Show/hide start button
        
        if checked:
            # Entering autonomous mode - reset button to clean "START" state
            self.keys_pressed.clear()
            self.auto_start_btn.setChecked(False)  # Ensure unchecked
            self.auto_start_btn.setText('Ã¢â€“Â¶ START DRIVING')  # Reset text
            self.auto_start_btn.setStyleSheet(f"QPushButton {{ font-size: 16px; padding: 15px; font-weight: bold; }} QPushButton:checked {{ background-color: {self.COLOR_GREEN}; border-color: #00ff00; }}")  # Reset style
            self.set_mode_controls(True)  # Enable tuning
            # Visual indicator: blue border when autonomous mode enabled
            self.auto_btn.setStyleSheet(f"QPushButton {{ font-size: 14px; padding: 12px; background-color: {self.COLOR_BLUE}; border: 2px solid #0088ff; color: white; }}")
        else:
            # Exiting autonomous mode - fully reset button state AND lap counter
            self.auto_start_btn.setChecked(False)
            self.auto_start_btn.setText('Ã¢â€“Â¶ START DRIVING')  # Reset text
            self.auto_start_btn.setStyleSheet(f"QPushButton {{ font-size: 16px; padding: 15px; font-weight: bold; }} QPushButton:checked {{ background-color: {self.COLOR_GREEN}; border-color: #00ff00; }}")  # Reset style
            self.set_mode_controls(False)  # Disable tuning, enable manual
            # Reset to default style
            self.auto_btn.setStyleSheet(f"QPushButton {{ font-size: 14px; padding: 12px; }} QPushButton:checked {{ background-color: {self.COLOR_BLUE}; border-color: #0088ff; }}")
            # Reset lap counter display
            self.lap_count_lbl.setText('Lap: 1')
            self.lap_count_lbl.setStyleSheet(f'font-size: 24px; font-weight: bold; color: {self.COLOR_GREEN};')
            self.lap_time_lbl.setText('Last: --  Best: --')
    
    def on_autonomous_override(self, enabled):
        """Handle external autonomous status changes (e.g., joystick override)."""
        if not enabled:
            # Vision controller disabled autonomous (joystick pressed LB)
            # Update GUI to reflect manual mode
            self.auto_btn.setChecked(False)
            self.auto_btn.setText('ENTER AUTONOMOUS MODE')
            self.auto_start_btn.setVisible(False)
            self.auto_start_btn.setChecked(False)
            self.auto_start_btn.setText('Ã¢â€“Â¶ START DRIVING')  # Reset text
            self.auto_start_btn.setStyleSheet(f"QPushButton {{ font-size: 16px; padding: 15px; font-weight: bold; }} QPushButton:checked {{ background-color: {self.COLOR_GREEN}; border-color: #00ff00; }}")  # Reset style
            self.set_mode_controls(False)  # Disable tuning
            # Reset button style
            self.auto_btn.setStyleSheet(f"QPushButton {{ font-size: 14px; padding: 12px; }} QPushButton:checked {{ background-color: {self.COLOR_BLUE}; border-color: #0088ff; }}")
            # Reset lap counter display
            self.lap_count_lbl.setText('Lap: 1')
            self.lap_count_lbl.setStyleSheet(f'font-size: 24px; font-weight: bold; color: {self.COLOR_GREEN};')
            self.lap_time_lbl.setText('Last: --  Best: --')

    def _il_runtime_is_confirmed(self):
        """Return True only when telemetry proves IL is the active runtime source."""
        return self.last_active_mode == 'IL' and self.last_runtime_source == 'IL'

    def _reject_il_start(self):
        """Reset UI and explain why IL start is blocked."""
        self.auto_start_btn.blockSignals(True)
        self.auto_start_btn.setChecked(False)
        self.auto_start_btn.blockSignals(False)
        self.auto_start_btn.setText('Ã¢â€“Â¶ START DRIVING')
        self.auto_start_btn.setStyleSheet(
            f"QPushButton {{ font-size: 16px; padding: 15px; font-weight: bold; }} "
            f"QPushButton:checked {{ background-color: {self.COLOR_GREEN}; border-color: #00ff00; }}"
        )
        self.auto_status.setText('IL blocked - runtime not confirmed')
        QMessageBox.warning(
            self,
            'IL Not Ready',
            'IL mode is selected, but the vision node has not confirmed that IL is the active runtime source.\n\n'
            'This usually means the IL controller failed to register, the model failed to load, or telemetry has not arrived yet.\n\n'
            'Wait for the warning to clear, or check preflight and vision-controller logs.'
        )
    
    def on_auto_start_toggle(self, checked):
        if checked and self.ros_node.control_mode == 'IL' and not self._il_runtime_is_confirmed():
            self._reject_il_start()
            return

        self.ros_node.set_autonomous_start(checked)
        if checked:
            self.auto_start_btn.setText('Ã¢ÂÂ¸ PAUSE DRIVING')
            if hasattr(self, 'auto_status'):
                self.auto_status.setText('Driving autonomously - adjust parameters live')
            self.auto_start_btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: 16px; padding: 15px; font-weight: bold;
                    background-color: {self.COLOR_GREEN}; border: 3px solid #00ff00; color: white;
                }}
                QPushButton:hover {{ background-color: #00cc00; }}
            """)

        else:
            self.auto_start_btn.setText('Ã¢â€“Â¶ START DRIVING')
            if hasattr(self, 'auto_status'):
                self.auto_status.setText('Paused - Tune parameters, then resume')
            self.auto_start_btn.setStyleSheet(
                f"QPushButton {{ font-size: 16px; padding: 15px; font-weight: bold; }} "
                f"QPushButton:checked {{ background-color: {self.COLOR_GREEN}; border-color: #00ff00; }}"
            )


    # Ã¢â€â‚¬Ã¢â€â‚¬ Discrete Speed Mode Helpers Ã¢â€â‚¬Ã¢â€â‚¬
    def _on_discrete_preset_changed(self, key_idx, value):
        """Update stored preset when spinbox value changes."""
        self.DISCRETE_SPEED_PRESETS[key_idx] = value
        if self.discrete_active_key == key_idx:
            self.discrete_speed = value
            self._update_discrete_banner()

    def _set_discrete_speed(self, idx):
        """Set discrete speed preset (keys 1-5)."""
        if 0 <= idx < len(self.DISCRETE_SPEED_PRESETS):
            self.discrete_active_key = idx
            self.discrete_speed = self.DISCRETE_SPEED_PRESETS[idx]
            self._update_discrete_indicators()
            self._update_discrete_banner()

    def _clear_discrete_speed(self):
        """Cancel discrete speed (key 0)."""
        self.discrete_speed = None
        self.discrete_active_key = -1
        self._update_discrete_indicators()
        self._update_discrete_banner()

    def _update_discrete_indicators(self):
        """Update the active indicator dots under each preset."""
        for i, ind in enumerate(self.discrete_indicators):
            if i == self.discrete_active_key:
                ind.setStyleSheet('background: #ffcc00; border-radius: 3px;')
            else:
                ind.setStyleSheet('background: #333; border-radius: 3px;')

    def _update_discrete_banner(self):
        """Update the discrete speed status banner."""
        if self.discrete_speed is not None:
            speed = self.discrete_speed
            key_num = self.discrete_active_key + 1
            self.discrete_banner.setText(f'Speed LOCKED: {speed:.1f} m/s  [Key {key_num}]')
            self.discrete_banner.setStyleSheet(
                'color: #000; font-size: 11px; font-weight: bold; '
                'background: #ffcc00; padding: 4px; border-radius: 4px;')
        else:
            self.discrete_banner.setText('Speed: Manual (W/S keys)')
            self.discrete_banner.setStyleSheet(
                'color: #888; font-size: 11px; font-weight: bold; '
                'background: #1a1a1a; padding: 4px; border-radius: 4px;')
    
    # Ã¢â€â‚¬Ã¢â€â‚¬ Supervised MPC Mode Helpers Ã¢â€â‚¬Ã¢â€â‚¬
    def _set_smpc_steer_mode(self, mode):
        """Toggle between blend and override steering assist."""
        self.smpc_steer_mode = mode
        self.smpc_blend_btn.setChecked(mode == "blend")
        self.smpc_override_btn.setChecked(mode == "override")
        self._update_smpc_banner()

    def _on_smpc_preset_changed(self, key_idx, value):
        """Update stored preset when spinbox value changes."""
        self.SMPC_SPEED_PRESETS[key_idx] = value
        if self.smpc_active_key == key_idx:
            self.smpc_speed = value
            self._update_smpc_banner()
            # Sync slider so MPC plans for the correct speed
            slider_val = int(value * 10)
            self.auto_speed_slider.setValue(slider_val)
            self.on_auto_speed_change(slider_val)

    def _set_smpc_speed(self, idx):
        """Set supervised MPC speed preset (keys 1-7)."""
        if 0 <= idx < len(self.SMPC_SPEED_PRESETS):
            # Save the current slider value before first lock
            if self.smpc_active_key == -1:
                self._smpc_saved_slider = self.auto_speed_slider.value()
            self.smpc_active_key = idx
            self.smpc_speed = self.SMPC_SPEED_PRESETS[idx]
            # Snap the Auto Speed slider to match
            slider_val = int(self.smpc_speed * 10)
            self.auto_speed_slider.setValue(slider_val)
            self.on_auto_speed_change(slider_val)
            self._update_smpc_indicators()
            self._update_smpc_banner()

    def _clear_smpc_speed(self):
        """Release speed lock Ã¢â‚¬â€ MPC controls speed again (key 0)."""
        self.smpc_speed = None
        self.smpc_active_key = -1
        # Restore the slider to its pre-lock value
        if hasattr(self, '_smpc_saved_slider'):
            self.auto_speed_slider.setValue(self._smpc_saved_slider)
            self.on_auto_speed_change(self._smpc_saved_slider)
        self._update_smpc_indicators()
        self._update_smpc_banner()

    def _update_smpc_indicators(self):
        """Update the active indicator dots under each preset."""
        for i, ind in enumerate(self.smpc_indicators):
            if i == self.smpc_active_key:
                ind.setStyleSheet('background: #00ff88; border-radius: 3px;')
            else:
                ind.setStyleSheet('background: #333; border-radius: 3px;')

    def _update_smpc_banner(self):
        """Update the supervised MPC status banner."""
        # Speed part
        if self.smpc_speed is not None:
            speed = self.smpc_speed
            key_num = self.smpc_active_key + 1
            speed_text = f'Speed LOCKED: {speed:.1f} m/s [Key {key_num}]'
            banner_bg = '#00ff88'
            banner_fg = '#000'
        else:
            speed_text = 'Speed: MPC Auto'
            banner_bg = '#1a1a1a'
            banner_fg = '#888'
        # Steer part
        steer_text = f'Steering: {self.smpc_steer_mode.capitalize()} (A/D)'
        self.smpc_banner.setText(f'{speed_text}   |   {steer_text}')
        self.smpc_banner.setStyleSheet(
            f'color: {banner_fg}; font-size: 11px; font-weight: bold; '
            f'background: {banner_bg}; padding: 4px; border-radius: 4px;')

    # Ã¢â€â‚¬Ã¢â€â‚¬ Expo Mode Helpers Ã¢â€â‚¬Ã¢â€â‚¬
    def _on_expo_routine_changed(self, idx):
        routines = ['sweep', 'heartbeat', 'wave', 'figure8', 'all']
        if 0 <= idx < len(routines):
            self.expo_current_routine = routines[idx]
            if not self.expo_active:
                self.expo_banner.setText(f'Ready Ã¢â‚¬â€ {self.expo_routine_combo.currentText()}')

    def _on_expo_speed_changed(self, val):
        self.expo_speed_amplitude = val / 10.0
        self.expo_speed_lbl.setText(f'{self.expo_speed_amplitude:.1f} m/s')

    def _on_expo_steer_changed(self, val):
        self.expo_steer_amplitude = val / 100.0
        self.expo_steer_lbl.setText(f'{self.expo_steer_amplitude:.2f} rad')

    def _on_expo_freq_changed(self, val):
        self.expo_frequency = val / 10.0
        self.expo_freq_lbl.setText(f'{self.expo_frequency:.1f} Hz')

    def _on_expo_toggle(self, checked):
        if checked:
            self._start_expo()
        else:
            self._stop_expo()

    def _start_expo(self):
        self.expo_active = True
        self.expo_phase = 0.0
        self.expo_start_btn.setText('STOP EXPO')
        self.expo_banner.setText(f'Running: {self.expo_routine_combo.currentText()}')
        self.expo_banner.setStyleSheet(
            'color: #000; font-size: 11px; font-weight: bold; '
            'background: #ffcc00; padding: 4px; border-radius: 4px;')
        # 20Hz timer for smooth movements
        self.expo_timer = QTimer()
        self.expo_timer.timeout.connect(self._expo_tick)
        self.expo_timer.start(50)  # 20 Hz
        self._expo_start_time = time.time()
        self._expo_all_routine_idx = 0
        self._expo_all_last_switch = time.time()

    def _stop_expo(self):
        self.expo_active = False
        if self.expo_timer:
            self.expo_timer.stop()
            self.expo_timer = None
        self.expo_phase = 0.0
        self.expo_start_btn.setText('START EXPO')
        self.expo_start_btn.setChecked(False)
        self.expo_banner.setText('Stopped')
        self.expo_banner.setStyleSheet(
            'color: #888; font-size: 11px; font-weight: bold; '
            'background: #1a1a1a; padding: 4px; border-radius: 4px;')
        # Zero all commands
        self.ros_node.publish_drive(0.0, 0.0)

    def _expo_tick(self):
        """20Hz movement loop Ã¢â‚¬â€ compute speed+steering from current routine."""
        import math
        dt = 0.05  # 20Hz
        self.expo_phase += dt
        t = self.expo_phase
        f = self.expo_frequency
        A_spd = self.expo_speed_amplitude
        A_str = self.expo_steer_amplitude

        routine = self.expo_current_routine

        # Auto-cycle mode: switch every 10 seconds
        if routine == 'all':
            now = time.time()
            if now - self._expo_all_last_switch > 10.0:
                self._expo_all_last_switch = now
                self._expo_all_routine_idx = (self._expo_all_routine_idx + 1) % 4
            routine = ['sweep', 'heartbeat', 'wave', 'figure8'][self._expo_all_routine_idx]
            routine_name = ['Sweep', 'Heartbeat', 'Wave', 'Figure-8'][self._expo_all_routine_idx]
            elapsed = int(time.time() - self._expo_start_time)
            self.expo_banner.setText(f'Auto-Cycle: {routine_name}  ({elapsed}s)')

        if routine == 'sweep':
            # Smooth steering oscillation, constant speed
            speed = A_spd
            steer = A_str * math.sin(2.0 * math.pi * f * t)
        elif routine == 'heartbeat':
            # Short motor bursts with servo snaps
            phase = (t * f) % 1.0
            if phase < 0.3:
                speed = A_spd * (phase / 0.3)
            elif phase < 0.5:
                speed = A_spd * (1.0 - (phase - 0.3) / 0.2)
            else:
                speed = 0.0
            steer = A_str if (int(t * f) % 2 == 0) else -A_str
        elif routine == 'wave':
            # Sine on both, 90Ã‚Â° out of phase
            speed = A_spd * (0.5 + 0.5 * math.sin(2.0 * math.pi * f * t))
            steer = A_str * math.sin(2.0 * math.pi * f * t + math.pi / 2.0)
        elif routine == 'figure8':
            # Steering full sine, speed at 2x frequency
            steer = A_str * math.sin(2.0 * math.pi * f * t)
            speed = A_spd * (0.5 + 0.5 * math.sin(4.0 * math.pi * f * t))
        else:
            speed = 0.0
            steer = 0.0

        # Safety clamp
        speed = max(0.0, min(3.0, speed))
        steer = max(-0.45, min(0.45, steer))

        self.ros_node.publish_drive(speed, steer)

        # Update banner with current values (except in auto-cycle which has its own)
        if self.expo_current_routine != 'all':
            elapsed = int(time.time() - self._expo_start_time)
            self.expo_banner.setText(
                f'{self.expo_routine_combo.currentText()}  |  '
                f'spd={speed:.2f}  steer={steer:.3f}  ({elapsed}s)')

    def _on_vesc_voltage(self, voltage):
        """Update battery gauge from VESC voltage reading."""
        self.expo_battery_voltage = voltage
        # 3S LiPo: 9.0V=empty, 12.6V=full
        pct = max(0, min(100, int((voltage - 9.0) / (12.6 - 9.0) * 100)))
        self.expo_batt_bar.setValue(pct)
        self.expo_batt_bar.setFormat(f'{pct}%')
        self.expo_batt_volts_lbl.setText(f'{voltage:.1f} V')
        # Color code
        if pct > 50:
            chunk_color = '#00cc44'
            volts_color = '#0f0'
        elif pct > 20:
            chunk_color = '#ffaa00'
            volts_color = '#ffaa00'
        else:
            chunk_color = '#ff3333'
            volts_color = '#ff3333'
        self.expo_batt_bar.setStyleSheet(
            f'QProgressBar {{ background: #222; border: 1px solid #555; border-radius: 4px; '
            f'text-align: center; color: #fff; font-size: 10px; }} '
            f'QProgressBar::chunk {{ background: {chunk_color}; border-radius: 3px; }}')
        self.expo_batt_volts_lbl.setStyleSheet(
            f'color: {volts_color}; font-size: 11px; font-weight: bold;')

    def _on_il_speed_preset(self, speed, idx):
        """Handle click on an IL discrete speed button."""
        # Set the auto speed slider to this value
        slider_val = int(speed * 10)
        self.auto_speed_slider.setValue(slider_val)
        self.on_auto_speed_change(slider_val)
        
        # Highlight the active button
        for i, btn in enumerate(self.il_speed_btns):
            if i == idx:
                btn.setStyleSheet(
                    'QPushButton { background-color: #660066; color: #fff; border: 2px solid #ff00ff; '
                    'border-radius: 4px; font-size: 12px; font-weight: bold; padding: 2px 6px; }'
                )
            else:
                btn.setStyleSheet(
                    'QPushButton { background-color: #2a2a2a; color: #ccc; border: 2px solid #555; '
                    'border-radius: 4px; font-size: 12px; font-weight: bold; padding: 2px 6px; }'
                    'QPushButton:hover { border-color: #ff00ff; color: #fff; }'
                )

    def on_speed_change(self, v):
        self.max_speed = v / 10.0
        # Sync joystick max speed with slider
        self.ros_node.joy_max_speed = self.max_speed
        
        if self.max_speed <= 2.0:
            self.speed_limit_lbl.setText(f'{self.max_speed:.1f} m/s (SAFE)')
            self.speed_limit_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_GREEN};')
        elif self.max_speed <= 4.0:
            self.speed_limit_lbl.setText(f'{self.max_speed:.1f} m/s (MODERATE)')
            self.speed_limit_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_YELLOW};')
        else:
            self.speed_limit_lbl.setText(f'{self.max_speed:.1f} m/s (FAST!)')
            self.speed_limit_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_RED};')
    
    def on_reverse_change(self, v):
        self.max_reverse_speed = v / 100.0
        if self.max_reverse_speed <= 0.5:
            self.reverse_limit_lbl.setText(f'{self.max_reverse_speed:.1f} m/s (SAFE)')
            self.reverse_limit_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_GREEN};')
        elif self.max_reverse_speed <= 1.0:
            self.reverse_limit_lbl.setText(f'{self.max_reverse_speed:.1f} m/s (MODERATE)')
            self.reverse_limit_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_YELLOW};')
        else:
            self.reverse_limit_lbl.setText(f'{self.max_reverse_speed:.1f} m/s (FAST!)')
            self.reverse_limit_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_RED};')
    


    def on_pd_change(self):
        kp = self.kp_slider.value() / 100.0
        kd = self.kd_slider.value() / 100.0
        self.kp_lbl.setText(f'{kp:.2f}')
        self.kd_lbl.setText(f'{kd:.2f}')
        self.ros_node.publish_pd(kp, kd)
    
    def on_auto_speed_change(self, v):
        speed = v / 10.0
        self.auto_speed_lbl.setText(f'{speed:.1f} m/s')
        self.ros_node.publish_auto_speed(speed)
    
    def on_width_change(self, v):
        self.width_lbl.setText(f'{v} px')
        self.ros_node.publish_track_width(v)
    
    def on_bias_change(self, v):
        bias = v / 100.0  # Convert to radians (-0.20 to +0.20)
        if bias < -0.01:
            self.bias_lbl.setText(f'{bias:.2f} (LEFT)')
            self.bias_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_YELLOW};')
        elif bias > 0.01:
            self.bias_lbl.setText(f'{bias:.2f} (RIGHT)')
            self.bias_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_YELLOW};')
        else:
            self.bias_lbl.setText('0.00 (center)')
            self.bias_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_GREEN};')
        self.ros_node.publish_steering_bias(bias)
    
    def on_slowdown_change(self, v):
        slowdown = v / 100.0  # Convert to 0.0-1.0
        if slowdown <= 0.2:
            self.slowdown_lbl.setText(f'{v}% (SPEED DEMON)')
            self.slowdown_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_RED};')
        elif slowdown <= 0.5:
            self.slowdown_lbl.setText(f'{v}% (Aggressive)')
            self.slowdown_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_YELLOW};')
        elif slowdown <= 0.7:
            self.slowdown_lbl.setText(f'{v}% (Normal)')
            self.slowdown_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_GREEN};')
        else:
            self.slowdown_lbl.setText(f'{v}% (Cautious)')
            self.slowdown_lbl.setStyleSheet('font-weight: bold; color: #00aaff;')
        self.ros_node.publish_turn_slowdown(slowdown)
    
    def on_smoothing_change(self, v):
        alpha = v / 100.0  # Convert to 0.30-0.90
        if alpha <= 0.45:
            self.smoothing_lbl.setText(f'{alpha:.2f} (Very Smooth)')
            self.smoothing_lbl.setStyleSheet('font-weight: bold; color: #00aaff;')
        elif alpha <= 0.60:
            self.smoothing_lbl.setText(f'{alpha:.2f} (Balanced)')
            self.smoothing_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_GREEN};')
        elif alpha <= 0.75:
            self.smoothing_lbl.setText(f'{alpha:.2f} (Responsive)')
            self.smoothing_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_YELLOW};')
        else:
            self.smoothing_lbl.setText(f'{alpha:.2f} (Very Responsive)')
            self.smoothing_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_RED};')
        self.ros_node.publish_smoothing_alpha(alpha)
    
    def on_lookahead_change(self, v):
        ratio = v / 100.0  # Convert 5-95 to 0.05-0.95
        self.lookahead_lbl.setText(f'{ratio:.2f}')
        self.ros_node.publish_lookahead_ratio(ratio)
    
    def on_bev_width_change(self, v):
        width = v / 100.0  # Convert 20-60 to 0.20-0.60
        self.bev_width_lbl.setText(f'{width:.2f}')
        self.ros_node.publish_bev_top_width(width)
    
    def on_bev_padding_change(self, v):
        padding = v / 100.0  # Convert 10-40 to 0.10-0.40
        self.bev_padding_lbl.setText(f'{padding:.2f}')
        self.ros_node.publish_bev_padding(padding)
    

    def on_obstacle_toggle(self):
        """Toggle obstacle detection on/off."""
        enabled = self.obstacle_toggle.isChecked()
        if enabled:
            self.obstacle_toggle.setText('Obstacle Detection: ON')
            self.obstacle_toggle.setStyleSheet(f"""
                QPushButton {{ 
                    font-size: 14px; 
                    padding: 10px; 
                    font-weight: bold;
                    background-color: {self.COLOR_GREEN};
                    color: white;
                    border-radius: 5px;
                }}
            """)
            self.distance_input.setEnabled(True)
            self.distance_apply_btn.setEnabled(True)
        else:
            self.obstacle_toggle.setText('Obstacle Detection: OFF')
            self.obstacle_toggle.setStyleSheet(f"""
                QPushButton {{ 
                    font-size: 14px; 
                    padding: 10px; 
                    font-weight: bold;
                    background-color: #666;
                    color: #aaa;
                    border-radius: 5px;
                }}
            """)
            self.distance_input.setEnabled(False)
            self.distance_apply_btn.setEnabled(False)
        self.ros_node.publish_obstacle_enable(enabled)
    
    def on_distance_apply(self):
        """Apply stop distance from input box."""
        try:
            text = self.distance_input.text().strip()
            distance = float(text)
            
            # Validate range
            if distance < 0.15:
                distance = 0.15
                self.distance_input.setText('0.15')
            elif distance > 3.0:
                distance = 3.0
                self.distance_input.setText('3.00')
            
            # Update status label with color coding
            if distance <= 0.30:
                self.distance_status_lbl.setText(f'Current: {distance:.2f}m (Close)')
                self.distance_status_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_RED};')
            elif distance <= 0.60:
                self.distance_status_lbl.setText(f'Current: {distance:.2f}m (Safe)')
                self.distance_status_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_GREEN};')
            elif distance <= 1.00:
                self.distance_status_lbl.setText(f'Current: {distance:.2f}m (Cautious)')
                self.distance_status_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_YELLOW};')
            else:
                self.distance_status_lbl.setText(f'Current: {distance:.2f}m (Far)')
                self.distance_status_lbl.setStyleSheet('font-weight: bold; color: #00aaff;')
            
            self.ros_node.publish_min_safe_distance(distance)
            
        except ValueError:
            # Invalid input - reset to current value
            self.distance_input.setText(f'{self.ros_node.min_safe_distance:.2f}')
            self.distance_status_lbl.setText('Invalid input!')
            self.distance_status_lbl.setStyleSheet(f'font-weight: bold; color: {self.COLOR_RED};')
    
    # HSV change handler - commented out with HSV sliders
    # def on_hsv_change(self):
    #     h_min = self.h_min_slider.value()
    #     h_max = self.h_max_slider.value()
    #     s_min = self.s_min_slider.value()
    #     v_min = self.v_min_slider.value()
    #     self.h_min_lbl.setText(str(h_min))
    #     self.h_max_lbl.setText(str(h_max))
    #     self.s_min_lbl.setText(str(s_min))
    #     self.v_min_lbl.setText(str(v_min))
    #     self.ros_node.publish_hsv(h_min, h_max, s_min, v_min)
    
    def on_auto_calibrate(self):
        """Request the vision controller to auto-calibrate HSV thresholds based on the current frame."""
        # Only allow calibration if we have a recent raw image
        if self.last_raw_image is None:
            self.status_label.setText('Status: Need raw camera feed to calibrate!')
            self.status_label.setStyleSheet("background-color: #2a2a2a; padding: 12px; border-radius: 6px; color: #ffaa00; font-weight: bold;")
            return
            
        self.status_label.setText('Status: Auto-Calibrating HSV...')
        self.status_label.setStyleSheet(f"background-color: #2a2a2a; padding: 12px; border-radius: 6px; color: {self.COLOR_BLUE}; font-weight: bold;")
        
        # Publish the boolean trigger
        msg = Bool()
        msg.data = True
        self.ros_node.auto_calibrate_pub.publish(msg)

        # Notify via signals so vision_controller can pick it up directly if it's running in same process (not here, here it's ROS)
        self.signals.auto_calibrate_hsv.emit()
    



    def save_preset(self):
        """Save current tuning parameters to a JSON preset file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                'Save Preset',
                str(self.preset_dir / 'preset.json'),
                'JSON Files (*.json);;All Files (*)'
            )
            
            if not filename:
                return
            
            # Collect all current parameter values
            current_mode = self.ros_node.control_mode
            preset_data = {
                'resolution': self.current_resolution,
                'control_mode': current_mode,
                'kp': self.kp_slider.value() / 100.0,
                'kd': self.kd_slider.value() / 100.0,
                'auto_speed': self.auto_speed_slider.value() / 10.0,
                'track_width': self.width_slider.value(),
                'steering_bias': self.bias_slider.value() / 100.0,
                'turn_slowdown': self.slowdown_slider.value() / 100.0,
                'smoothing_alpha': self.smoothing_slider.value() / 100.0,
                'lookahead_ratio': self.lookahead_slider.value() / 100.0,
                'bev_top_width': self.bev_width_slider.value() / 100.0,
                'bev_padding': self.bev_padding_slider.value() / 100.0,
                'mpc_horizon': self.horizon_slider.value(),
                'mpc_tracking': self.tracking_slider.value() / 10.0,
                'mpc_smoothness': self.smoothness_slider.value() / 10.0,
                'cem_num_samples': self.cem_samples_slider.value(),
                'cem_horizon': self.cem_horizon_slider.value(),
            }
            
            with open(filename, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            msg = QMessageBox()
            msg.setWindowTitle('Preset Saved')
            msg.setText(f'Preset saved successfully to:\n{filename}')
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #ffffff;
                }
                QLabel {
                    color: #000000;
                    font-size: 13px;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    color: #000000;
                    border: 1px solid #999;
                    border-radius: 3px;
                    padding: 5px 15px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)
            msg.exec_()
        except Exception as e:
            msg = QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText(f'Failed to save preset:\n{str(e)}')
            msg.setIcon(QMessageBox.Warning)
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #ffffff;
                }
                QLabel {
                    color: #000000;
                    font-size: 13px;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    color: #000000;
                    border: 1px solid #999;
                    border-radius: 3px;
                    padding: 5px 15px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)
            msg.exec_()
    
    def load_preset(self):
        """Load tuning parameters from a JSON preset file."""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                'Load Preset',
                str(self.preset_dir),
                'JSON Files (*.json);;All Files (*)'
            )
            
            if not filename:
                return
            
            with open(filename, 'r') as f:
                preset_data = json.load(f)
            
            # Load resolution first
            if 'resolution' in preset_data:
                res_name = preset_data['resolution']
                if res_name in self.RESOLUTIONS:
                    self.resolution_combo.setCurrentText(res_name)
                    self.apply_resolution(res_name)
            
            # Load parameters (with defaults if missing)
            if 'control_mode' in preset_data:
                self.set_control_mode(preset_data['control_mode'])
            
            if 'kp' in preset_data:
                self.kp_slider.setValue(int(preset_data['kp'] * 100))
            if 'kd' in preset_data:
                self.kd_slider.setValue(int(preset_data['kd'] * 100))
            if 'auto_speed' in preset_data:
                self.auto_speed_slider.setValue(int(preset_data['auto_speed'] * 10))
            if 'track_width' in preset_data:
                self.width_slider.setValue(int(preset_data['track_width']))
            if 'steering_bias' in preset_data:
                self.bias_slider.setValue(int(preset_data['steering_bias'] * 100))
            if 'turn_slowdown' in preset_data:
                self.slowdown_slider.setValue(int(preset_data['turn_slowdown'] * 100))
            if 'smoothing_alpha' in preset_data:
                self.smoothing_slider.setValue(int(preset_data['smoothing_alpha'] * 100))
            if 'lookahead_ratio' in preset_data:
                self.lookahead_slider.setValue(int(preset_data['lookahead_ratio'] * 100))
            if 'bev_top_width' in preset_data:
                self.bev_width_slider.setValue(int(preset_data['bev_top_width'] * 100))
            if 'bev_padding' in preset_data:
                self.bev_padding_slider.setValue(int(preset_data['bev_padding'] * 100))
            
            if 'mpc_horizon' in preset_data:
                self.horizon_slider.setValue(int(preset_data['mpc_horizon']))
            if 'mpc_tracking' in preset_data:
                self.tracking_slider.setValue(int(preset_data['mpc_tracking'] * 10))
            if 'mpc_smoothness' in preset_data:
                self.smoothness_slider.setValue(int(preset_data['mpc_smoothness'] * 10))
            if 'cem_num_samples' in preset_data:
                self.cem_samples_slider.setValue(int(preset_data['cem_num_samples']))
            if 'cem_horizon' in preset_data:
                self.cem_horizon_slider.setValue(int(preset_data['cem_horizon']))
            
            msg = QMessageBox()
            msg.setWindowTitle('Preset Loaded')
            msg.setText(f'Preset loaded successfully from:\n{filename}')
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #ffffff;
                }
                QLabel {
                    color: #000000;
                    font-size: 13px;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    color: #000000;
                    border: 1px solid #999;
                    border-radius: 3px;
                    padding: 5px 15px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)
            msg.exec_()
        except Exception as e:
            msg = QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText(f'Failed to load preset:\n{str(e)}')
            msg.setIcon(QMessageBox.Warning)
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #ffffff;
                }
                QLabel {
                    color: #000000;
                    font-size: 13px;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    color: #000000;
                    border: 1px solid #999;
                    border-radius: 3px;
                    padding: 5px 15px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)
            msg.exec_()
    
    def set_mode_controls(self, auto_mode):
        # Tuning controls (only in AUTO mode)
        tuning_style_disabled = '''
            QGroupBox { color: #555; border-color: #333; }
            QGroupBox::title { color: #555; }
            QLabel { color: #555; }
            QSlider::groove:horizontal { background: #222; border-color: #333; }
            QSlider::handle:horizontal { background: #555; border-color: #444; }
        '''
        for widget in self.tuning_widgets:
            widget.setEnabled(auto_mode)
            widget.setStyleSheet('' if auto_mode else tuning_style_disabled)
        
        # Manual controls (only in MANUAL mode)
        manual_style_disabled = '''
            QGroupBox { color: #555; border-color: #333; }
            QGroupBox::title { color: #555; }
            QLabel { color: #555; }
            QPushButton { color: #555; background-color: #1a1a1a; border-color: #333; }
            QSlider::groove:horizontal { background: #222; border-color: #333; }
            QSlider::handle:horizontal { background: #555; border-color: #444; }
        '''
        for widget in self.manual_widgets:
            widget.setEnabled(not auto_mode)
            widget.setStyleSheet('' if not auto_mode else manual_style_disabled)
    
    def reset_tuning(self):
        # Confirmation dialog
        msg = QMessageBox()
        msg.setWindowTitle('Reset Parameters')
        msg.setText('Reset all tuning parameters to defaults?')
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #ffffff;
            }
            QLabel {
                color: #000000;
                font-size: 13px;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #000000;
                border: 1px solid #999;
                border-radius: 3px;
                padding: 5px 15px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)
        reply = msg.exec_()
        
        if reply != QMessageBox.Yes:
            return
        
        # Reset all tuning to defaults
        self.set_control_mode('POLY_LOOKAHEAD')  # Reset to Poly LookAhead mode
        
        # MPC defaults
        self.horizon_slider.setValue(15)
        self.tracking_slider.setValue(10)
        self.smoothness_slider.setValue(10)
        # CEM defaults
        self.cem_samples_slider.setValue(500)
        self.cem_horizon_slider.setValue(12)
        self.kp_slider.setValue(85)
        self.kd_slider.setValue(20)  # Updated to 0.20
        self.auto_speed_slider.setValue(15)
        self.width_slider.setValue(450)
        self.bias_slider.setValue(0)  # Reset steering bias to center
        self.slowdown_slider.setValue(35)  # Reset turn slowdown to 35%
        self.smoothing_slider.setValue(55)  # Reset smoothing alpha to 0.55
        # Poly LookAhead settings sliders
        self.lookahead_slider.setValue(30)  # Reset to 0.30
        self.bev_width_slider.setValue(40)  # Reset to 0.40
        self.bev_padding_slider.setValue(20)  # Reset to 0.20
        # HSV sliders commented out
        # self.h_min_slider.setValue(35)
        # self.h_max_slider.setValue(90)
        # self.s_min_slider.setValue(50)
        # self.v_min_slider.setValue(50)


    # =========================================
    # OVERLAY MENU & MODE HANDLING
    # =========================================
    def setup_overlay(self):
        """Create the fullscreen mode selection overlay with categories."""
        self.overlay = QFrame(self.centralWidget())
        self.overlay.hide()
        self.overlay.setStyleSheet("background-color: rgba(10, 10, 10, 0.98);")
        
        main_layout = QVBoxLayout(self.overlay)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = QLabel("SELECT CONTROL MODE")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: white; font-size: 24px; font-weight: bold; margin-bottom: 20px; font-family: Arial;")
        main_layout.addWidget(title)
        
        # Columns Container
        cols_widget = QWidget()
        cols_layout = QHBoxLayout(cols_widget)
        cols_layout.setSpacing(40)
        cols_layout.setContentsMargins(0, 0, 0, 0)
        cols_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        
        sections = {}
        for (name, label, section, enabled, color) in MODES:
            if section not in sections:
                sections[section] = []
            sections[section].append((name, label, enabled, color))
            
        # CREATE TWO MAIN COLUMNS
        # Left Column: Rev0, Rev1, Replay (Stacked vertically)
        # Right Column: Beta
        
        left_col_frame = QFrame()
        left_col_frame.setMinimumWidth(300)
        left_col_layout = QVBoxLayout(left_col_frame)
        left_col_layout.setSpacing(15)
        left_col_layout.setContentsMargins(15, 15, 15, 15)
        
        right_col_frame = QFrame()
        right_col_frame.setMinimumWidth(520)
        right_col_layout = QVBoxLayout(right_col_frame)
        right_col_layout.setSpacing(15)
        right_col_layout.setContentsMargins(15, 15, 15, 15)
        
        # Helper to build a section block
        def build_section(section_key, target_layout, force_cols=2):
            if section_key not in sections or section_key not in SECTION_HEADERS: 
                return
            modes_list = sections[section_key]
            enabled_modes = [(name, label, color) for (name, label, enabled, color) in modes_list if enabled]
            if not enabled_modes:
                return
                
            header_text, header_color = SECTION_HEADERS.get(section_key, (section_key.upper(), "#fff"))
            
            # Container for this section (creates the box effect)
            sec_frame = QFrame()
            sec_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: rgba(255, 255, 255, 0.03);
                    border: 1px solid #333;
                    border-radius: 12px;
                }}
            """)
            sec_layout = QVBoxLayout(sec_frame)
            sec_layout.setSpacing(10)
            sec_layout.setContentsMargins(12, 12, 12, 12)
            
            # Header
            head_lbl = QLabel(header_text)
            head_lbl.setAlignment(Qt.AlignCenter)
            head_lbl.setStyleSheet(f"color: {header_color}; font-size: 16px; font-weight: bold; border: none; background: transparent; margin-bottom: 5px;")
            sec_layout.addWidget(head_lbl)

            # Buttons Grid
            buttons_widget = QWidget()
            buttons_layout = QGridLayout(buttons_widget)
            buttons_layout.setSpacing(8)
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            for c in range(force_cols):
                buttons_layout.setColumnStretch(c, 1)
            
            for idx, (name, label, color) in enumerate(enabled_modes):
                btn = QPushButton(label)
                btn.setEnabled(True)
                btn.setMinimumHeight(45)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip(self._format_mode_tooltip(name, label))
                btn.setToolTipDuration(10000)
                
                text_col = color if color else "#eaeaea"
                border_col = header_color
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: #1a1a1a;
                        color: {text_col};
                        border: 1px solid {border_col};
                        border-radius: 8px;
                        font-size: 14px; font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: {border_col};
                        color: #000;
                        border: 1px solid {border_col};
                    }}
                """)
                btn.clicked.connect(lambda ch, m=name: self.set_control_mode(m))

                row = idx // force_cols
                col = idx % force_cols
                buttons_layout.addWidget(btn, row, col)

            sec_layout.addWidget(buttons_widget)
            target_layout.addWidget(sec_frame)

        # Build Left Stack (single column inside each section)
        build_section("rev0", left_col_layout, force_cols=1)
        build_section("rev1", left_col_layout, force_cols=1)
        left_col_layout.addStretch()
        
        # Build Right Stack: Showcase (Expo) above Beta
        build_section("showcase", right_col_layout, force_cols=1)
        build_section("beta", right_col_layout, force_cols=2)
        right_col_layout.addStretch()
        
        cols_layout.addWidget(left_col_frame)
        cols_layout.addWidget(right_col_frame)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidget(cols_widget)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.06);
                width: 14px;
                margin: 4px 0 4px 0;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 176, 0, 0.85);
                min-height: 40px;
                border-radius: 7px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical,
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: transparent;
                border: none;
            }
            QScrollBar:horizontal {
                background: rgba(255, 255, 255, 0.06);
                height: 14px;
                margin: 0 4px 0 4px;
                border-radius: 7px;
            }
            QScrollBar::handle:horizontal {
                background: rgba(255, 176, 0, 0.85);
                min-width: 40px;
                border-radius: 7px;
            }
        """)
        main_layout.addWidget(scroll_area, stretch=1)
        
        # Cancel button
        cancel_btn = QPushButton("CANCEL")
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setStyleSheet(f"color: {self.COLOR_RED}; background: transparent; border: 1px solid {self.COLOR_RED}; border-radius: 5px; font-weight: bold;")
        cancel_btn.clicked.connect(self.overlay.hide)
        main_layout.addWidget(cancel_btn, alignment=Qt.AlignCenter)

    def show_mode_overlay(self):
        """Show the mode selection overlay."""
        self._sync_overlay_geometry()
        self.overlay.show()
        self.overlay.raise_()

    def _format_mode_tooltip(self, mode_name, fallback_label):
        """Return readable hover text for the mode menu and active badge."""
        full_name = MODE_TOOLTIPS.get(mode_name, fallback_label)
        if full_name == fallback_label:
            return full_name
        return f"{fallback_label}\n{full_name}"
        
    def set_control_mode(self, mode_name):
        """Set mode from overlay and close it."""
        self.overlay.hide()



        if (
            mode_name == "EXPO"
            and getattr(self, "expo_show_intro_cb", None) is not None
            and self.expo_show_intro_cb.isChecked()
        ):
            self._show_expo_welcome_then_finalize()
            return

        self._finalize_control_mode(mode_name)

    def _show_expo_welcome_then_finalize(self):
        """Full-screen welcome; ROS publish only after dismiss."""
        from .expo_welcome_overlay import ExpoWelcomeOverlay

        cw = self.centralWidget()
        if cw is None:
            self._finalize_control_mode("EXPO")
            return

        self._close_expo_welcome_overlay_if_any()
        ov = ExpoWelcomeOverlay(cw)
        self._expo_welcome_overlay = ov
        ov.setGeometry(cw.rect())
        ov.finished.connect(self._on_expo_welcome_finished)
        ov.show()
        ov.raise_()

    def _on_expo_welcome_finished(self):
        self._expo_welcome_overlay = None
        self._finalize_control_mode("EXPO")

    def _close_expo_welcome_overlay_if_any(self):
        w = getattr(self, "_expo_welcome_overlay", None)
        if w is not None:
            try:
                w.hide()
                w.deleteLater()
            except RuntimeError:
                pass
            self._expo_welcome_overlay = None

    def _replay_expo_welcome_overlay(self):
        """Replay intro only; does not republish ROS mode."""
        from .expo_welcome_overlay import ExpoWelcomeOverlay

        cw = self.centralWidget()
        if cw is None:
            return
        self._close_expo_welcome_overlay_if_any()
        ov = ExpoWelcomeOverlay(cw)
        self._expo_welcome_overlay = ov
        ov.setGeometry(cw.rect())
        ov.finished.connect(lambda: setattr(self, "_expo_welcome_overlay", None))
        ov.show()
        ov.raise_()

    def _finalize_control_mode(self, mode_name):
        """Apply mode: ROS publish, badge, tuning visibility."""
        self.ros_node.publish_control_mode(mode_name)

        mode_label = next((label for (name, label, _section, _enabled, _color) in MODES if name == mode_name), mode_name)
        mode_tooltip = self._format_mode_tooltip(mode_name, mode_label)

        self.mode_badge.setText(f"{mode_label} Ã¢â€“Â¼")
        self.mode_badge.setToolTip(mode_tooltip)
        self.mode_badge.setToolTipDuration(10000)

        if mode_name.startswith("BETA_"):
            col = self.COLOR_ORANGE
        elif "MPC" in mode_name:
            col = self.COLOR_GREEN
        elif "LEGACY" in mode_name:
            col = self.COLOR_YELLOW
        elif "CALIBRATION" in mode_name:
            col = self.COLOR_ORANGE
        else:
            col = self.COLOR_BLUE

        self.mode_badge.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba({int(QColor(col).red())}, {int(QColor(col).green())}, {int(QColor(col).blue())}, 40);
                border: 2px solid {col};
                color: {col};
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
            }}
            QPushButton:hover {{ background-color: rgba({int(QColor(col).red())}, {int(QColor(col).green())}, {int(QColor(col).blue())}, 80); }}
        """)

        self.update_mode_ui(mode_name)


    def update_mode_ui(self, mode_name):
        """Update visible settings based on mode."""
        from .utils.mode_config import MODE_VISIBILITY, DEFAULT_MODE, MODE_STATUS_STYLE
        
        # Get visibility settings or default
        vis = MODE_VISIBILITY.get(mode_name, MODE_VISIBILITY[DEFAULT_MODE])
        
        # 1. Update Mode Status Labels logic
        if mode_name in MODE_STATUS_STYLE:
            status_text, status_style = MODE_STATUS_STYLE[mode_name]
            self.mode_status.setText(status_text)
            self.mode_status.setStyleSheet(status_style)
        else:
            self.mode_status.setText(f"Active: {mode_name}")
            self.mode_status.setStyleSheet('color: #0f0; font-size: 10px;')

        if hasattr(self, 'mode_runtime_warning') and mode_name != 'IL':
            self.mode_runtime_warning.clear()
            self.mode_runtime_warning.setVisible(False)


        # 2. Toggle Groups
        self.pd_group.setVisible(vis.get("pd_group", True))
        self.mpc_group.setVisible(vis.get("mpc_group", False))

        # Update Discrete controls visibility
        is_discrete = (mode_name == "DISCRETE")
        if hasattr(self, 'discrete_group'):
            self.discrete_group.setVisible(is_discrete)
            if not is_discrete:
                # Reset discrete state when leaving mode
                self.discrete_speed = None
                self.discrete_active_key = -1
                if hasattr(self, 'discrete_indicators'):
                    self._update_discrete_indicators()
                    self._update_discrete_banner()

        # Update Supervised MPC controls visibility
        is_smpc = (mode_name == "SUPERVISED_MPC")
        if hasattr(self, 'supervised_mpc_group'):
            self.supervised_mpc_group.setVisible(is_smpc)
            if not is_smpc:
                # Reset SMPC state when leaving mode
                self.smpc_speed = None
                self.smpc_active_key = -1
                if hasattr(self, 'smpc_indicators'):
                    self._update_smpc_indicators()
                    self._update_smpc_banner()

        # Update Expo controls visibility
        is_expo = (mode_name == "EXPO")
        if hasattr(self, 'expo_group'):
            self.expo_group.setVisible(is_expo)
            if not is_expo:
                # Stop expo routine when leaving mode
                self._stop_expo()
                self._close_expo_welcome_overlay_if_any()

        # 3. Vision Group Title
        if hasattr(self, 'advanced_group'):
            self.advanced_group.setVisible(vis.get("advanced_group", True))
            
        # 4. Extras (Speed/Steer filters)
        if hasattr(self, 'speed_extras_widget'):
            self.speed_extras_widget.setVisible(vis.get("speed_extras", True))
        
        # 4b. IL Discrete Speed Presets (only in IL mode)
        if hasattr(self, 'il_speed_presets_widget'):
            self.il_speed_presets_widget.setVisible(mode_name == "IL")
        
        # 5. Legacy Track Width
        if hasattr(self, 'width_group'):
            self.width_group.setVisible(mode_name == "LEGACY")
        
        # 6. CEM Tuning group (auto-expand/collapse)
        cem_vis = vis.get("cem_group", False)
        if hasattr(self, 'cem_group'):
            self.cem_group.setVisible(cem_vis)
            if hasattr(self.cem_group, 'toggle_btn'):
                self.cem_group.toggle_btn.setChecked(cem_vis)
        

        
        # 8. LLA-MPC Status group (auto-expand/collapse)
        lla_vis = vis.get("lla_group", False)
        if hasattr(self, 'lla_group'):
            self.lla_group.setVisible(lla_vis)
            if hasattr(self.lla_group, 'toggle_btn'):
                self.lla_group.toggle_btn.setChecked(lla_vis)


        # 10. Auto Speed group
        if hasattr(self, 'speed_group'):
            self.speed_group.setVisible(vis.get("speed_group", True))

        # 11. Window resolution dropdown
        if hasattr(self, 'window_group'):
            self.window_group.setVisible(vis.get("window_group", True))

        # 12. Controller-specific tuning groups
        if hasattr(self, 'hmpcc_group'):
            hmpcc_vis = vis.get("hmpcc_group", False)
            self.hmpcc_group.setVisible(hmpcc_vis)
            if hasattr(self.hmpcc_group, 'toggle_btn'):
                self.hmpcc_group.toggle_btn.setChecked(hmpcc_vis)
        if hasattr(self, 'pure_pursuit_group'):
            pp_vis = vis.get("pure_pursuit_group", False)
            self.pure_pursuit_group.setVisible(pp_vis)
            if hasattr(self.pure_pursuit_group, 'toggle_btn'):
                self.pure_pursuit_group.toggle_btn.setChecked(pp_vis)
        if hasattr(self, 'stanley_group'):
            stanley_vis = vis.get("stanley_group", False)
            self.stanley_group.setVisible(stanley_vis)
            if hasattr(self.stanley_group, 'toggle_btn'):
                self.stanley_group.toggle_btn.setChecked(stanley_vis)



def main():
    rclpy.init()
    app = QApplication(sys.argv)
    signals = RosSignals()
    ros_node = CarControlNode(signals)
    window = ControlGUI(ros_node)
    window.show()
    
    ros_thread = threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True)
    ros_thread.start()
    
    code = app.exec_()
    ros_node.emergency_stop()
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
