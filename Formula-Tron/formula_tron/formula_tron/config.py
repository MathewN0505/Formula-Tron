#!/usr/bin/env python3
"""
Formula-Tron Settings
=====================
All tunable values in one place.
"""

# ROS Topics
CAMERA_TOPIC = '/camera/camera/color/image_raw'
MOTOR_TOPIC = '/commands/motor/speed'
SERVO_TOPIC = '/commands/servo/position'

# VESC Motor Controller (don't change unless recalibrating)
VESC_SPEED_TO_ERPM_GAIN = 4614.0
VESC_SPEED_TO_ERPM_OFFSET = 0.0
VESC_STEERING_TO_SERVO_GAIN = -1.2135
VESC_STEERING_TO_SERVO_OFFSET = 0.5304
SERVO_CMD_MIN = 0.15
SERVO_CMD_MAX = 0.85

# Steering Control
KP_DEFAULT = 0.85           # Steering strength (higher = sharper turns)
KD_DEFAULT = 0.20           # Smoothness (higher = less wobbling) - increased for buttery smooth turns
MAX_STEERING_ANGLE = 0.45   # Max steering (radians)
MAX_STEERING_RATE = 3.2     # Max steering change per second (rad/s) - aligned with FilesFromCar throttle_interpolator
STEERING_BIAS_DEFAULT = 0.0 # Offset for camera misalignment


def _compute_steering_limits_from_servo():
    """
    Convert servo command limits to steering-angle limits using calibration.
    Returns (steering_min, steering_max, symmetric_abs_limit).
    """
    gain = VESC_STEERING_TO_SERVO_GAIN
    offset = VESC_STEERING_TO_SERVO_OFFSET
    if abs(gain) < 1e-9:
        # Fallback if calibration is invalid
        return -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE, MAX_STEERING_ANGLE

    s_a = (SERVO_CMD_MIN - offset) / gain
    s_b = (SERVO_CMD_MAX - offset) / gain
    steer_min = min(s_a, s_b)
    steer_max = max(s_a, s_b)
    symmetric_limit = min(abs(steer_min), abs(steer_max))
    symmetric_limit = min(MAX_STEERING_ANGLE, symmetric_limit)
    return steer_min, steer_max, symmetric_limit


# Physical steering limits implied by current servo calibration + safe servo bounds.
STEERING_MIN_ANGLE, STEERING_MAX_ANGLE, MAX_STEERING_ANGLE_EFFECTIVE = _compute_steering_limits_from_servo()

# Speed
BASE_SPEED = 1.5            # Default speed (m/s)
TURN_SLOWDOWN = 0.35        # How much to slow in turns (0=none, 1=full stop)
MAX_SPEED_RATE = 2.0        # Max speed change per second (m/s^2) for command smoothing

# Lane Detection
PHYSICAL_TRACK_WIDTH = 0.85  # Physical distance between lane markers (meters)
WHEELBASE_METERS = 0.32      # Physical distance between axles (meters)
VISUAL_TRACK_WIDTH = 450    # Expected pixel width between lanes
ROI_HEIGHT_RATIO = 0.4      # Use bottom 40% of camera

# Color Settings (HSV for green tape)
HSV_H_MIN = 35              # Green hue minimum
HSV_H_MAX = 90              # Green hue maximum
HSV_S_MIN = 50              # Saturation minimum
HSV_V_MIN = 50              # Brightness minimum

# Smoothing (lower = smoother but slower response)
TARGET_SMOOTHING_ALPHA = 0.55  # 0.55 = 55% new, 45% old - super smooth target tracking

# POLY_LOOKAHEAD Mode Settings (Bird's Eye View & Polynomial Lookahead PD)
BEV_TOP_WIDTH = 0.4           # Top width of trapezoid (0.4 = 40% of image width)
BEV_PADDING = 0.2             # Side padding in BEV output (0.2 = 20% on each side)
LOOKAHEAD_RATIO = 0.3         # How far ahead to look (0.3 = 30% up from bottom of ROI)

# Safety
MAX_CONSECUTIVE_ERRORS = 10   # Stop after this many failures
VISION_TIMEOUT_SEC = 2.0      # Stop if no frames for 2 seconds
FRAME_DROP_WARNING_INTERVAL = 1.0

# Depth Safety (Obstacle Avoidance)
DEPTH_TOPIC = '/camera/camera/depth/image_rect_raw'  # Must match camera namespace
MIN_SAFE_DISTANCE_DEFAULT = 0.50  # meters (50cm) - stop if obstacle closer than this
MIN_SAFE_DISTANCE_MIN = 0.15      # 15cm (absolute minimum)
MIN_SAFE_DISTANCE_MAX = 3.00      # 3m (maximum)

# Debug View
DEBUG_ENABLED_DEFAULT = False # Off by default (saves CPU)
DEBUG_FRAME_SKIP = 1          # Publish every frame when debug view is enabled

# Lap Counting (AprilTag)
# These defaults are intentionally conservative to prevent false positives.
MIN_LAP_TIME = 12.0      # Minimum seconds between laps
LAP_TAG_ID = 0          # AprilTag ID to trigger lap count (Standard 36h11)
LAP_TAG_SIZE_M = 0.16   # Physical size of the tag (160mm based on 200mm total size)
LAP_MAX_DIST = 2.7      # Only trigger lap if tag is closer than this (meters)
LAP_MIN_FRAMES_WITHOUT_TAG = 10  # Tag must be invisible for this many frames before counting again
LAP_MIN_FRAMES_WITH_TAG = 1      # Single valid detection triggers lap (distance/pixel/time gates already prevent false positives)
LAP_MIN_TAG_PIXEL_WIDTH = 24.0   # Reject tiny/far detections that are likely false positives

# Joystick Deadman Switch
# LB (button 4) - Hold to drive manually, auto-disables autonomous if running
# When autonomous is running and you press/hold LB, it instantly disables
# autonomous mode and lets you take manual control.
# This follows the F1TENTH standard deadman switch convention.


