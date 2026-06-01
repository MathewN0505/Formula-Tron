"""Shared pytest fixtures for Formula-Tron tests."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add package to path for imports
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))


@pytest.fixture
def valid_frame():
    """Create a valid 640x480 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_with_green_line():
    """Create a frame with a green vertical line in the center."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw green line (BGR format) in center
    frame[400:480, 310:330] = [0, 255, 0]  # Center vertical line
    return frame


@pytest.fixture
def frame_with_two_green_lines():
    """Create a frame with two green lines (left and right tracks)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Left track line
    frame[400:480, 150:170] = [0, 255, 0]
    # Right track line
    frame[400:480, 470:490] = [0, 255, 0]
    return frame


@pytest.fixture
def invalid_frame_none():
    """Invalid frame: None."""
    return None


@pytest.fixture
def invalid_frame_too_small():
    """Invalid frame: Too small."""
    return np.zeros((5, 5, 3), dtype=np.uint8)


@pytest.fixture
def invalid_frame_no_shape():
    """Invalid frame: No shape attribute."""
    class FakeFrame:
        size = 100
    return FakeFrame()


@pytest.fixture
def track_detector():
    """Create a default TrackDetector instance."""
    from formula_tron.utils.track_detection import TrackDetector
    return TrackDetector(
        hsv_lower=np.array([35, 50, 50]),
        hsv_upper=np.array([90, 255, 255]),
        track_width=450,
        roi_ratio=0.4,
    )


@pytest.fixture
def watchdog_short_timeout():
    """Create a WatchdogTimer with short timeout for testing."""
    from formula_tron.utils.safety import WatchdogTimer
    return WatchdogTimer(timeout=0.1, max_consecutive_errors=3)


@pytest.fixture
def exponential_moving_average():
    """Create an ExponentialMovingAverage instance."""
    from formula_tron.utils.safety import ExponentialMovingAverage
    return ExponentialMovingAverage(alpha=0.55, max_jump=150)


@pytest.fixture
def sample_histogram():
    """Create a sample histogram with two peaks."""
    hist = np.zeros(640, dtype=np.float32)
    # Left peak at x=150
    hist[140:160] = 500.0
    # Right peak at x=480
    hist[470:490] = 600.0
    # Center peak at x=320
    hist[310:330] = 400.0
    return hist


@pytest.fixture
def sample_histogram_single_peak():
    """Create a histogram with single center peak."""
    hist = np.zeros(640, dtype=np.float32)
    hist[310:330] = 500.0
    return hist


@pytest.fixture
def sample_histogram_no_peaks():
    """Create a histogram with no significant peaks."""
    return np.zeros(640, dtype=np.float32)


@pytest.fixture(params=[
    (np.zeros((480, 640, 3), dtype=np.uint8), True, "valid frame"),
    (None, False, "None frame"),
    (np.zeros((5, 5, 3), dtype=np.uint8), False, "too small"),
])
def frame_test_case(request):
    """Parametrized fixture for frame validation tests."""
    return request.param


@pytest.fixture(params=[
    (320, 320, 320, 0.0, "center"),
    (0, 320, 320, -1.0, "left edge"),
    (640, 320, 320, 1.0, "right edge"),
    (160, 320, 320, -0.5, "left half"),
    (480, 320, 320, 0.5, "right half"),
])
def normalization_test_case(request):
    """Parametrized fixture for error normalization tests."""
    target_x, center_x, half_range, expected, description = request.param
    return {
        'target_x': target_x,
        'center_x': center_x,
        'half_range': half_range,
        'expected': expected,
        'description': description,
    }


# ===========================================================================
# Rev1 Controller Fixtures
# ===========================================================================

@pytest.fixture
def mock_detection_with_poly():
    """Detection result with polynomial data (for MPC/RL controllers).
    Waypoints are in car-frame metres: x = forward, y = lateral (left +)."""
    from formula_tron.utils.track_detection import TrackDetectionResult
    return TrackDetectionResult(
        target_x=320.0,
        left_peak=None,
        right_peak=None,
        mask=np.zeros((192, 640), dtype=np.uint8),
        histogram=np.zeros(640, dtype=np.float32),
        all_peaks=np.array([], dtype=np.int32),
        used_peaks=np.array([], dtype=np.int32),
        status="POLY: CENTER",
        bev_mask=np.zeros((192, 640), dtype=np.uint8),
        poly_coeffs=np.array([0.0005, -0.1, 320.0]),
        target_x_bev=320.0,
        left_poly=np.array([0.0005, -0.1, 200.0]),
        right_poly=np.array([0.0005, -0.1, 440.0]),
        center_poly=np.array([0.0005, -0.1, 320.0]),
        detection_mode="CENTER",
        waypoints=np.array([[x, 0.05] for x in np.linspace(0.3, 2.0, 30)]),
    )


@pytest.fixture
def straight_detection():
    """Detection result for a straight track (zero curvature).
    Waypoints are in car-frame metres: x = forward, y = lateral (left +)."""
    from formula_tron.utils.track_detection import TrackDetectionResult
    return TrackDetectionResult(
        target_x=320.0,
        left_peak=None,
        right_peak=None,
        mask=np.zeros((192, 640), dtype=np.uint8),
        histogram=np.zeros(640, dtype=np.float32),
        all_peaks=np.array([], dtype=np.int32),
        used_peaks=np.array([], dtype=np.int32),
        status="POLY: CENTER",
        bev_mask=np.zeros((192, 640), dtype=np.uint8),
        poly_coeffs=np.array([0.0, 0.0, 320.0]),
        target_x_bev=320.0,
        left_poly=np.array([0.0, 0.0, 200.0]),
        right_poly=np.array([0.0, 0.0, 440.0]),
        center_poly=np.array([0.0, 0.0, 320.0]),
        detection_mode="L+R",
        waypoints=np.array([[x, 0.0] for x in np.linspace(0.3, 2.0, 30)]),
    )


@pytest.fixture
def curved_detection():
    """Detection result for a right curve.
    Waypoints are in car-frame metres: x = forward, y = lateral (left +).
    Negative y = curving to the right."""
    from formula_tron.utils.track_detection import TrackDetectionResult
    return TrackDetectionResult(
        target_x=280.0,
        left_peak=None,
        right_peak=None,
        mask=np.zeros((192, 640), dtype=np.uint8),
        histogram=np.zeros(640, dtype=np.float32),
        all_peaks=np.array([], dtype=np.int32),
        used_peaks=np.array([], dtype=np.int32),
        status="POLY: CENTER",
        bev_mask=np.zeros((192, 640), dtype=np.uint8),
        poly_coeffs=np.array([0.01, -0.5, 350.0]),
        target_x_bev=280.0,
        left_poly=np.array([0.01, -0.5, 230.0]),
        right_poly=np.array([0.01, -0.5, 470.0]),
        center_poly=np.array([0.01, -0.5, 350.0]),
        detection_mode="CENTER",
        waypoints=np.array([[x, -0.15 * x] for x in np.linspace(0.3, 2.0, 30)]),
    )


@pytest.fixture
def no_detection():
    """Detection result with no polynomial data (lost track)."""
    from formula_tron.utils.track_detection import TrackDetectionResult
    return TrackDetectionResult(
        target_x=None,
        left_peak=None,
        right_peak=None,
        mask=np.zeros((192, 640), dtype=np.uint8),
        histogram=np.zeros(640, dtype=np.float32),
        all_peaks=np.array([], dtype=np.int32),
        used_peaks=np.array([], dtype=np.int32),
        status="NO TRACK",
        bev_mask=None,
        poly_coeffs=None,
        detection_mode="LOST",
        waypoints=None,
    )


@pytest.fixture
def vehicle_model():
    """Shared bicycle model instance."""
    from formula_tron.utils.vehicle_model import BicycleModel
    return BicycleModel(wheelbase=0.33, max_steering=0.45, max_speed=5.0)


@pytest.fixture
def poly_lookahead_base():
    """POLY_LOOKAHEAD controller for use as RL baseline."""
    from formula_tron.utils.pd_controller import PDController
    return PDController("POLY_LOOKAHEAD", kp=0.85, kd=0.20)
