"""Unit tests for safety module."""

import pytest
import numpy as np
import time
from formula_tron.utils.safety import (
    SafetyValidator,
    WatchdogTimer,
    ExponentialMovingAverage,
    ConnectionMonitor,
    safe_normalize,
    check_obstacle,
)


@pytest.mark.unit
class TestSafetyValidator:
    """Tests for SafetyValidator class."""

    def test_validate_frame_valid(self, valid_frame):
        """Test that valid frames pass validation."""
        assert SafetyValidator.validate_frame(valid_frame) is True

    def test_validate_frame_none(self, invalid_frame_none):
        """Test that None frames are rejected."""
        assert SafetyValidator.validate_frame(invalid_frame_none) is False

    def test_validate_frame_too_small(self, invalid_frame_too_small):
        """Test that frames that are too small are rejected."""
        assert SafetyValidator.validate_frame(invalid_frame_too_small) is False

    def test_validate_frame_no_shape(self, invalid_frame_no_shape):
        """Test that frames without shape attribute are rejected."""
        assert SafetyValidator.validate_frame(invalid_frame_no_shape) is False

    def test_validate_frame_too_large(self):
        """Test that frames that are too large are rejected."""
        huge_frame = np.zeros((10001, 10001, 3), dtype=np.uint8)
        assert SafetyValidator.validate_frame(huge_frame) is False

    def test_validate_frame_valid_dimensions(self):
        """Test various valid frame dimensions."""
        # Minimum valid size
        assert SafetyValidator.validate_frame(np.zeros((10, 10, 3), dtype=np.uint8)) is True
        # Maximum valid size
        assert SafetyValidator.validate_frame(np.zeros((10000, 10000, 3), dtype=np.uint8)) is True
        # Grayscale (2D)
        assert SafetyValidator.validate_frame(np.zeros((480, 640), dtype=np.uint8)) is True

    def test_validate_number_valid(self):
        """Test that valid numbers pass validation."""
        assert SafetyValidator.validate_number(5.0) is True
        assert SafetyValidator.validate_number(0) is True
        assert SafetyValidator.validate_number(-10.5) is True

    def test_validate_number_nan(self):
        """Test that NaN values are rejected."""
        assert SafetyValidator.validate_number(np.nan) is False
        assert SafetyValidator.validate_number(float('nan')) is False

    def test_validate_number_inf(self):
        """Test that infinity values are rejected."""
        assert SafetyValidator.validate_number(np.inf) is False
        assert SafetyValidator.validate_number(float('inf')) is False
        assert SafetyValidator.validate_number(float('-inf')) is False

    def test_validate_number_range(self):
        """Test that range validation works."""
        assert SafetyValidator.validate_number(5.0, min_val=0.0, max_val=10.0) is True
        assert SafetyValidator.validate_number(15.0, min_val=0.0, max_val=10.0) is False
        assert SafetyValidator.validate_number(-5.0, min_val=0.0, max_val=10.0) is False

    def test_validate_number_wrong_type(self):
        """Test that non-numeric types are rejected."""
        assert SafetyValidator.validate_number("5.0") is False
        assert SafetyValidator.validate_number([5.0]) is False
        assert SafetyValidator.validate_number(None) is False


@pytest.mark.unit
class TestWatchdogTimer:
    """Tests for WatchdogTimer class."""

    def test_watchdog_initialization(self):
        """Test watchdog initializes correctly."""
        watchdog = WatchdogTimer(timeout=2.0, max_consecutive_errors=10)
        assert watchdog.timeout == 2.0
        assert watchdog.max_consecutive_errors == 10
        assert watchdog.consecutive_errors == 0
        assert watchdog.stopped is False

    def test_watchdog_success_resets_errors(self, watchdog_short_timeout):
        """Test that on_success resets consecutive errors."""
        watchdog = watchdog_short_timeout
        watchdog.on_error()
        watchdog.on_error()
        assert watchdog.consecutive_errors == 2
        watchdog.on_success()
        assert watchdog.consecutive_errors == 0

    def test_watchdog_timeout(self, watchdog_short_timeout):
        """Test that watchdog triggers after timeout."""
        watchdog = watchdog_short_timeout
        assert watchdog.should_stop() is False
        time.sleep(0.15)  # Wait longer than timeout
        assert watchdog.should_stop() is True
        assert watchdog.stopped is True

    def test_watchdog_consecutive_errors(self, watchdog_short_timeout):
        """Test that watchdog triggers after max consecutive errors."""
        watchdog = watchdog_short_timeout
        assert watchdog.should_stop() is False
        for _ in range(3):  # Trigger max_consecutive_errors
            watchdog.on_error()
        assert watchdog.should_stop() is True
        assert watchdog.stopped is True

    def test_watchdog_reset(self, watchdog_short_timeout):
        """Test that reset clears consecutive errors."""
        watchdog = watchdog_short_timeout
        # Add errors
        watchdog.on_error()
        watchdog.on_error()
        assert watchdog.consecutive_errors == 2
        # Reset
        watchdog.reset()
        assert watchdog.consecutive_errors == 0
        assert watchdog.stopped is False

    def test_watchdog_stays_stopped(self, watchdog_short_timeout):
        """Test that once stopped, watchdog stays stopped."""
        watchdog = watchdog_short_timeout
        watchdog.on_error()
        watchdog.on_error()
        watchdog.on_error()
        assert watchdog.should_stop() is True
        # Even after success, should stay stopped
        watchdog.on_success()
        assert watchdog.should_stop() is True


@pytest.mark.unit
class TestExponentialMovingAverage:
    """Tests for ExponentialMovingAverage class."""

    def test_ema_initialization(self):
        """Test EMA initializes correctly."""
        ema = ExponentialMovingAverage(alpha=0.5, max_jump=100)
        assert ema.alpha == 0.5
        assert ema.max_jump == 100
        assert ema.value is None

    def test_ema_first_value(self, exponential_moving_average):
        """Test that first value is returned as-is."""
        ema = exponential_moving_average
        result = ema.update(10.0)
        assert result == 10.0
        assert ema.value == 10.0

    def test_ema_smoothing(self):
        """Test that EMA smooths values correctly."""
        ema = ExponentialMovingAverage(alpha=0.5, max_jump=None)
        result1 = ema.update(10.0)
        assert result1 == 10.0
        result2 = ema.update(20.0)
        # With alpha=0.5: 0.5 * 20 + 0.5 * 10 = 15.0
        assert abs(result2 - 15.0) < 0.001

    def test_ema_glitch_rejection(self):
        """Test that large jumps are clamped (not silently ignored)."""
        ema = ExponentialMovingAverage(alpha=0.5, max_jump=50.0)
        ema.update(100.0)
        # Try to update with huge jump (1000 is 900 away, way over max_jump=50)
        result = ema.update(1000.0)
        # Should clamp to max_jump in the correct direction and then blend:
        # clamped_value = 100 + 50 = 150, result = 0.5*150 + 0.5*100 = 125
        assert abs(result - 125.0) < 0.001
        # Value moves towards the target, not frozen
        assert ema.value > 100.0

    def test_ema_no_glitch_rejection(self):
        """Test that small changes are not rejected."""
        ema = ExponentialMovingAverage(alpha=0.5, max_jump=50.0)
        ema.update(100.0)
        result = ema.update(120.0)  # Jump of 20, less than max_jump
        assert result != 100.0  # Should update

    def test_ema_reset(self, exponential_moving_average):
        """Test that reset clears the value."""
        ema = exponential_moving_average
        ema.update(10.0)
        assert ema.value is not None
        ema.reset()
        assert ema.value is None

    def test_ema_get(self, exponential_moving_average):
        """Test that get() returns value without updating."""
        ema = exponential_moving_average
        ema.update(10.0)
        value1 = ema.get()
        value2 = ema.get()
        assert value1 == value2 == 10.0

    def test_ema_set_alpha(self, exponential_moving_average):
        """Test that alpha can be updated."""
        ema = exponential_moving_average
        assert ema.alpha == 0.55
        ema.set_alpha(0.8)
        assert ema.alpha == 0.8
        # Test clamping
        ema.set_alpha(1.5)  # Should clamp to 1.0
        assert ema.alpha == 1.0
        ema.set_alpha(-0.5)  # Should clamp to 0.0
        assert ema.alpha == 0.0


@pytest.mark.unit
class TestConnectionMonitor:
    """Tests for ConnectionMonitor class."""

    def test_connection_monitor_initialization(self):
        """Test ConnectionMonitor initializes correctly."""
        monitor = ConnectionMonitor(timeout=1.0, frame_drop_threshold=0.2)
        assert monitor.timeout == 1.0
        assert monitor.frame_drop_threshold == 0.2
        assert monitor.frame_count == 0
        assert monitor.connected is False

    def test_connection_monitor_first_frame(self):
        """Test that first frame doesn't return gap."""
        monitor = ConnectionMonitor()
        gap = monitor.on_frame()
        assert gap is None
        assert monitor.frame_count == 1
        assert monitor.connected is True

    def test_connection_monitor_gap_detection(self):
        """Test that frame gaps are detected."""
        monitor = ConnectionMonitor()
        monitor.on_frame()
        time.sleep(0.1)
        gap = monitor.on_frame()
        assert gap is not None
        assert gap >= 0.1

    def test_connection_monitor_is_connected(self):
        """Test that is_connected works correctly."""
        monitor = ConnectionMonitor(timeout=0.1)
        assert monitor.is_connected() is False
        monitor.on_frame()
        assert monitor.is_connected() is True
        time.sleep(0.15)
        assert monitor.is_connected() is False


@pytest.mark.unit
class TestSafeNormalize:
    """Tests for safe_normalize function."""

    def test_normalize_center(self, normalization_test_case):
        """Test normalization with parametrized test cases."""
        if normalization_test_case['description'] == 'center':
            result = safe_normalize(
                normalization_test_case['target_x'],
                normalization_test_case['center_x'],
                normalization_test_case['half_range'],
            )
            assert abs(result - normalization_test_case['expected']) < 0.01

    def test_normalize_left_edge(self):
        """Test normalization at left edge."""
        result = safe_normalize(0, 320, 320)
        assert abs(result - (-1.0)) < 0.01

    def test_normalize_right_edge(self):
        """Test normalization at right edge."""
        result = safe_normalize(640, 320, 320)
        assert abs(result - 1.0) < 0.01

    def test_normalize_clamping(self):
        """Test that values are clamped to [-1, 1]."""
        # Way beyond range
        result = safe_normalize(2000, 320, 320)
        assert abs(result - 1.0) < 0.01
        result = safe_normalize(-1000, 320, 320)
        assert abs(result - (-1.0)) < 0.01

    def test_normalize_small_half_range(self):
        """Test that small half_range is handled."""
        result = safe_normalize(10, 5, 0.5)  # half_range < 1.0
        # Should use minimum 1.0 and clamp result
        assert abs(result) <= 1.0  # Result should be clamped to valid range

    def test_normalize_nan_handling(self):
        """Test that NaN inputs return 0."""
        result = safe_normalize(float('nan'), 320, 320)
        assert result == 0.0

    def test_normalize_inf_handling(self):
        """Test that infinity inputs return 0."""
        result = safe_normalize(float('inf'), 320, 320)
        assert result == 0.0

    def test_normalize_exception_handling(self):
        """Test that exceptions return 0."""
        # Pass invalid types that would cause exception
        result = safe_normalize("invalid", 320, 320)
        assert result == 0.0


@pytest.mark.unit
class TestDepthSafety:
    """Tests for ObstacleDetector class and check_obstacle function."""

    def test_obstacle_detected_with_hysteresis(self):
        """Test that obstacle is detected after multiple consecutive frames (hysteresis)."""
        from formula_tron.utils.safety import ObstacleDetector
        
        # Create detector with 3 consecutive frames required
        detector = ObstacleDetector(min_safe_distance=0.5, consecutive_frames_required=3)
        
        # Create depth image with obstacle in ROI (30-70% height, 20-80% width)
        # ROI for 480x640: height 144-336, width 128-512
        depth_frame = np.full((480, 640), 2000, dtype=np.uint16)  # 2m background
        depth_frame[150:330, 150:500] = 400  # 0.4m obstacle in ROI
        
        # First detection - should NOT trigger yet (need 3 consecutive)
        is_obstacle, distance, confidence = detector.detect(depth_frame)
        assert is_obstacle is False
        assert detector.consecutive_detections == 1
        
        # Second detection
        is_obstacle, distance, confidence = detector.detect(depth_frame)
        assert is_obstacle is False
        assert detector.consecutive_detections == 2
        
        # Third detection - NOW should trigger
        is_obstacle, distance, confidence = detector.detect(depth_frame)
        assert is_obstacle is True
        assert detector.consecutive_detections >= 3
        assert distance < 0.5  # Should be around 0.4m

    def test_obstacle_not_detected_far(self):
        """Test that far obstacles are ignored."""
        from formula_tron.utils.safety import ObstacleDetector
        
        detector = ObstacleDetector(min_safe_distance=0.5, consecutive_frames_required=1)
        
        # Create depth image with 1.0m depth (1000mm) - beyond threshold
        depth_frame = np.full((480, 640), 1000, dtype=np.uint16)
        
        is_obstacle, distance, confidence = detector.detect(depth_frame)
        
        assert is_obstacle is False
        assert abs(distance - 1.0) < 0.15  # Tolerance for median calculation

    def test_obstacle_empty_frame(self):
        """Test handles empty/None frames correctly."""
        assert check_obstacle(None, 0.5) == (False, float('inf'))
        assert check_obstacle(np.array([]), 0.5) == (False, float('inf'))

    def test_obstacle_nan_handling(self):
        """Test graceful handling of garbage data."""
        assert check_obstacle("not an array", 0.5) == (False, float('inf'))

    def test_region_of_interest(self):
        """Test that obstacles outside the ROI are ignored."""
        from formula_tron.utils.safety import ObstacleDetector
        
        detector = ObstacleDetector(min_safe_distance=0.5, consecutive_frames_required=1)
        
        # Create far background
        depth_frame = np.full((480, 640), 2000, dtype=np.uint16)
        
        # Place obstacle in top-left corner (outside ROI which is 30-70% height, 20-80% width)
        depth_frame[0:100, 0:100] = 300  # 0.3m - but outside ROI
        
        is_obstacle, distance, confidence = detector.detect(depth_frame)
        
        # Should NOT detect the top-left obstacle (it's outside ROI)
        assert is_obstacle is False
        assert distance > 1.5  # Should reflect background distance

    def test_enable_disable(self):
        """Test enable/disable functionality."""
        from formula_tron.utils.safety import ObstacleDetector
        
        detector = ObstacleDetector(min_safe_distance=0.5, consecutive_frames_required=1)
        
        # Create depth image with obstacle
        depth_frame = np.full((480, 640), 300, dtype=np.uint16)  # 0.3m - very close
        
        # Enabled - should detect
        detector.set_enabled(True)
        is_obstacle, _, _ = detector.detect(depth_frame)
        assert is_obstacle is True
        
        # Disabled - should NOT detect
        detector.set_enabled(False)
        is_obstacle, _, _ = detector.detect(depth_frame)
        assert is_obstacle is False

    def test_reset(self):
        """Test reset clears consecutive detection count."""
        from formula_tron.utils.safety import ObstacleDetector
        
        detector = ObstacleDetector(min_safe_distance=0.5, consecutive_frames_required=3)
        
        # Build up some detections
        depth_frame = np.full((480, 640), 300, dtype=np.uint16)
        detector.detect(depth_frame)
        detector.detect(depth_frame)
        assert detector.consecutive_detections == 2
        
        # Reset
        detector.reset()
        assert detector.consecutive_detections == 0
        
        # Need to build up again
        detector.detect(depth_frame)
        assert detector.consecutive_detections == 1

