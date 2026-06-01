#!/usr/bin/env python3
"""
Safety Module - Keeps the car from crashing due to software errors.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Any
import time


class SafetyValidator:
    """Checks inputs to make sure they're safe."""

    @staticmethod
    def validate_frame(frame: Any) -> bool:
        """Check if the camera frame is valid."""
        if frame is None:
            return False
        if not hasattr(frame, 'size') or frame.size == 0:
            return False
        if not hasattr(frame, 'shape') or len(frame.shape) < 2:
            return False
        h, w = frame.shape[:2]
        if h < 10 or w < 10 or h > 10000 or w > 10000:
            return False
        return True

    @staticmethod
    def validate_number(value: Any, min_val: float = None, max_val: float = None) -> bool:
        """Check if number is valid (not NaN/inf, in range)."""
        if not isinstance(value, (int, float, np.number)):
            return False
        if np.isnan(value) or np.isinf(value):
            return False
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True


class WatchdogTimer:
    """Stops the car if vision stops working."""

    def __init__(self, timeout: float = 2.0, max_consecutive_errors: int = 10):
        self.timeout = timeout
        self.max_consecutive_errors = max_consecutive_errors
        self.last_success_time = time.time()
        self.consecutive_errors = 0
        self.stopped = False

    def on_success(self):
        """Call when a frame works."""
        self.last_success_time = time.time()
        self.consecutive_errors = 0

    def on_error(self):
        """Call when a frame fails."""
        self.consecutive_errors += 1

    def should_stop(self) -> bool:
        """Returns True if car should stop."""
        if self.stopped:
            return True
        if (time.time() - self.last_success_time) > self.timeout:
            self.stopped = True
            return True
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.stopped = True
            return True
        return False

    def is_healthy(self) -> bool:
        """
        Returns True when watchdog is currently healthy.
        Non-mutating counterpart to should_stop() for telemetry/status checks.
        """
        if self.stopped:
            return False
        if (time.time() - self.last_success_time) > self.timeout:
            return False
        if self.consecutive_errors >= self.max_consecutive_errors:
            return False
        return True

    def reset(self):
        """Reset when autonomous mode restarts."""
        self.last_success_time = time.time()
        self.consecutive_errors = 0
        self.stopped = False


class ExponentialMovingAverage:
    """Smooths out noisy values."""

    def __init__(self, alpha: float = 0.7, max_jump: Optional[float] = None):
        self.alpha = alpha
        self.max_jump = max_jump
        self.value = None

    def update(self, new_value: float) -> float:
        """Add new value and return smoothed result."""
        if self.value is None:
            self.value = new_value
            return new_value
        if self.max_jump is not None and abs(new_value - self.value) > self.max_jump:
            # Clamp to max_jump in the correct direction instead of
            # silently ignoring the new reading.  Ignoring caused the
            # smoothed value to freeze on the old (wrong) position,
            # making the car stick to an outer line indefinitely.
            direction = 1.0 if new_value > self.value else -1.0
            new_value = self.value + direction * self.max_jump
        self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        """Reset filter."""
        self.value = None

    def get(self) -> Optional[float]:
        """Get current value without updating."""
        return self.value
    
    def set_alpha(self, alpha: float):
        """Update smoothing factor (0.0-1.0)."""
        self.alpha = max(0.0, min(1.0, alpha))


class ConnectionMonitor:
    """Checks if camera is connected."""

    def __init__(self, timeout: float = 1.0, frame_drop_threshold: float = 0.2):
        self.timeout = timeout
        self.frame_drop_threshold = frame_drop_threshold
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.connected = False

    def on_frame(self) -> Optional[float]:
        """Call when frame arrives. Returns time gap if frame was dropped."""
        now = time.time()
        gap = None
        if self.frame_count > 0:
            gap = now - self.last_frame_time
        self.last_frame_time = now
        self.frame_count += 1
        self.connected = True
        return gap

    def is_connected(self) -> bool:
        """Check if camera is connected."""
        if not self.connected:
            return False
        return (time.time() - self.last_frame_time) < self.timeout


def safe_normalize(value: float, center: float, half_range: float) -> float:
    """Normalize value to [-1, 1] range."""
    try:
        if half_range < 1.0:
            half_range = 1.0
        normalized = (value - center) / half_range
        if np.isnan(normalized) or np.isinf(normalized):
            return 0.0
        return max(-1.0, min(1.0, normalized))
    except Exception:
        return 0.0


class ObstacleDetector:
    """
    Simple, robust obstacle detection for depth cameras.
    
    Key design principles:
    1. GROUND REMOVAL: Only look at a horizontal strip in the MIDDLE of the image
       (ignore bottom 40% where ground is visible, ignore top 30% ceiling/sky)
    2. CENTER FOCUS: Only check center 50% of width (where path obstacles would be)
    3. MEDIAN FILTERING: Use median for robustness against noise
    4. HYSTERESIS: Require multiple consecutive detections to trigger
    5. SIMPLE THRESHOLDS: No complex adaptive thresholds that can fail
    """
    
    def __init__(self, 
                 min_safe_distance: float = 0.5,
                 min_valid_depth_mm: int = 200,
                 max_valid_depth_mm: int = 4000,
                 consecutive_frames_required: int = 3):
        """
        Args:
            min_safe_distance: Minimum safe distance in meters
            min_valid_depth_mm: Minimum valid depth in mm (20cm - closer is noise)
            max_valid_depth_mm: Maximum valid depth in mm (4m - RealSense reliable range)
            consecutive_frames_required: How many consecutive detections before triggering
        """
        self.min_safe_distance = min_safe_distance
        self.min_valid_depth_mm = min_valid_depth_mm
        self.max_valid_depth_mm = max_valid_depth_mm
        self.consecutive_frames_required = consecutive_frames_required
        
        # State
        self.consecutive_detections = 0
        self.last_distance = float('inf')
        self.enabled = True  # Can be disabled from GUI
        
    def detect(self, depth_frame: np.ndarray, current_speed: float = 0.0) -> Tuple[bool, float, float]:
        """
        Detect obstacles in depth frame using simple, robust logic.
        
        Args:
            depth_frame: numpy array of depth values (uint16, in mm)
            current_speed: Current vehicle speed (unused, kept for API compatibility)
            
        Returns:
            tuple (is_obstacle, distance_meters, confidence)
        """
        # If disabled, always return no obstacle
        if not self.enabled:
            return False, float('inf'), 0.0
        
        try:
            if depth_frame is None or depth_frame.size == 0:
                self.consecutive_detections = 0
                return False, float('inf'), 0.0
            
            h, w = depth_frame.shape[:2]
            if h < 20 or w < 20:
                self.consecutive_detections = 0
                return False, float('inf'), 0.0
            
            # === GROUND-AWARE ROI ===
            # Key insight: On a forward-facing camera angled down to see the track:
            # - Top 30% (0-30%): Far horizon/sky - ignore
            # - Middle 40% (30-70%): Obstacles at driving height - CHECK THIS
            # - Bottom 30% (70-100%): Ground plane very close - ignore
            
            # ROI: Horizontal strip from 30% to 70% of image height
            # This captures obstacles from medium to close range while avoiding ground
            roi_top = int(h * 0.30)     # Start at 30% from top
            roi_bottom = int(h * 0.70)  # End at 70% from top (was 60%, increased to catch closer obstacles)
            
            # Width: Center 60% of image (obstacles in driving path + some margin)
            roi_left = int(w * 0.20)
            roi_right = int(w * 0.80)
            
            # Extract ROI
            roi = depth_frame[roi_top:roi_bottom, roi_left:roi_right]
            
            # === FILTER VALID DEPTHS ===
            # RealSense D435i: reliable range is ~0.2m to ~4m
            valid_mask = (roi >= self.min_valid_depth_mm) & (roi <= self.max_valid_depth_mm)
            valid_depths = roi[valid_mask]
            
            # Need minimum pixels for reliable detection
            min_pixels = 100
            if len(valid_depths) < min_pixels:
                # Not enough valid depth data - don't trigger
                self.consecutive_detections = 0
                return False, self.last_distance, 0.0
            
            # === SIMPLE MEDIAN-BASED DETECTION ===
            # Use median (robust to outliers) instead of min (sensitive to noise)
            # Also check the 10th percentile as a "near obstacle" indicator
            median_dist_mm = np.median(valid_depths)
            near_dist_mm = np.percentile(valid_depths, 10)
            
            # Convert to meters
            median_dist_m = median_dist_mm / 1000.0
            near_dist_m = near_dist_mm / 1000.0
            
            # Use the nearer of the two (but median provides stability)
            # Weight towards median for stability, but consider near for safety
            measured_dist = 0.7 * median_dist_m + 0.3 * near_dist_m
            self.last_distance = measured_dist
            
            # === HYSTERESIS: Require consecutive detections ===
            # This prevents single-frame noise from triggering stops
            obstacle_in_frame = measured_dist < self.min_safe_distance
            
            if obstacle_in_frame:
                self.consecutive_detections += 1
            else:
                # Decay slower than we build up (prevents flickering)
                self.consecutive_detections = max(0, self.consecutive_detections - 1)
            
            # Only trigger if we have enough consecutive detections
            is_obstacle = self.consecutive_detections >= self.consecutive_frames_required
            
            # Confidence based on how many consecutive detections
            confidence = min(1.0, self.consecutive_detections / self.consecutive_frames_required)
            
            return is_obstacle, measured_dist, confidence
            
        except Exception as e:
            # On error, don't trigger false positive
            self.consecutive_detections = 0
            return False, self.last_distance, 0.0
    
    def reset(self):
        """Reset state (call when restarting autonomous mode)."""
        self.consecutive_detections = 0
        self.last_distance = float('inf')
    
    def update_threshold(self, min_safe_distance: float):
        """Update minimum safe distance threshold."""
        self.min_safe_distance = max(0.15, min(3.0, min_safe_distance))
    
    def set_enabled(self, enabled: bool):
        """Enable or disable obstacle detection."""
        self.enabled = enabled
        if not enabled:
            self.consecutive_detections = 0


# Legacy function for backward compatibility
def check_obstacle(depth_frame: np.ndarray, min_safe_distance: float) -> Tuple[bool, float]:
    """Legacy obstacle detection function."""
    detector = ObstacleDetector(min_safe_distance=min_safe_distance)
    is_obstacle, distance, _ = detector.detect(depth_frame)
    return is_obstacle, distance
