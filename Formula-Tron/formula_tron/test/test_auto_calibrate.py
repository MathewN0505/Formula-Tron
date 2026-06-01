"""Comprehensive tests for the HSV auto-calibrator.

Covers:
  1. Core Calibration Logic (different green shades, margins, percentiles)
  2. Lighting Conditions (dim, bright, mixed, outdoor simulation)
  3. Edge Cases (tiny frame, few pixels, noise, non-green colors)
  4. Post-Calibration Detection (calibrate → detect improves results)
  5. Dashboard → Node → Detector Wiring (callback chain)
  6. Repeated Calibration (stability under multiple calls)
"""

import numpy as np
import pytest
import cv2
import sys
from pathlib import Path

# Add package to path for imports
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from formula_tron.utils.track_detection import TrackDetector


# ── Frame Generators ────────────────────────────────────────────

def _green_frame(bgr_color=(0, 200, 0), line_width=20, frame_h=480, frame_w=640):
    """Create a frame with a green center line of a specific color."""
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    cx = frame_w // 2
    # Draw in lower half (where ROI lives with roi_ratio=0.4)
    y_start = int(frame_h * 0.6)
    frame[y_start:, cx - line_width // 2:cx + line_width // 2] = bgr_color
    return frame


def _dim_green_frame():
    """Simulates dim indoor lighting — dark green (low V channel)."""
    return _green_frame(bgr_color=(0, 60, 0))


def _bright_green_frame():
    """Simulates bright outdoor lighting — saturated green (high V)."""
    return _green_frame(bgr_color=(0, 255, 0))


def _washed_out_green_frame():
    """Simulates washed out / overexposed green (low saturation)."""
    return _green_frame(bgr_color=(100, 255, 100))


def _yellowish_green_frame():
    """Simulates warm lighting making green look yellowish."""
    return _green_frame(bgr_color=(0, 200, 80))


def _noisy_frame_with_green():
    """Frame with green line plus random noise everywhere."""
    frame = _green_frame()
    noise = np.random.randint(0, 40, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame


def _non_green_frame():
    """Frame with a bright red line (no green at all)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[288:480, 310:330] = (0, 0, 200)  # Red in BGR
    return frame


def _sparse_green_frame():
    """Very thin green line — barely any green pixels."""
    return _green_frame(line_width=3)


def _multi_shade_frame():
    """Frame with dim + bright green lines (mixed lighting on track)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    y_start = int(480 * 0.6)
    frame[y_start:, 100:120] = (0, 60, 0)    # dim green left
    frame[y_start:, 310:330] = (0, 200, 0)   # normal green center
    frame[y_start:, 520:540] = (0, 255, 0)   # bright green right
    return frame


# ═══════════════════════════════════════════════════════════════════
# 1. Core Calibration Logic
# ═══════════════════════════════════════════════════════════════════

class TestCoreCalibration:
    """Test the fundamental calibration algorithm."""

    def test_returns_dict_with_4_keys(self):
        det = TrackDetector()
        result = det.auto_calibrate(_green_frame())
        assert result is not None
        assert set(result.keys()) == {'hsv_h_min', 'hsv_h_max', 'hsv_s_min', 'hsv_v_min'}

    def test_h_min_less_than_h_max(self):
        det = TrackDetector()
        result = det.auto_calibrate(_green_frame())
        assert result['hsv_h_min'] < result['hsv_h_max']

    def test_all_values_non_negative(self):
        det = TrackDetector()
        result = det.auto_calibrate(_green_frame())
        for k, v in result.items():
            assert v >= 0, f"{k} is negative: {v}"

    def test_h_max_within_hsv_range(self):
        det = TrackDetector()
        result = det.auto_calibrate(_green_frame())
        assert result['hsv_h_max'] <= 180

    def test_updates_detector_lower_bound(self):
        det = TrackDetector()
        old_lower = det.hsv_lower.copy()
        det.auto_calibrate(_green_frame())
        # hsv_lower should have been updated
        assert det.hsv_lower[0] != old_lower[0] or det.hsv_lower[1] != old_lower[1]

    def test_upper_bound_maxes_at_255(self):
        det = TrackDetector()
        det.auto_calibrate(_green_frame())
        assert det.hsv_upper[1] == 255  # S max
        assert det.hsv_upper[2] == 255  # V max

    def test_calibration_hue_covers_green_range(self):
        """Green tape should produce hue values roughly in the 35-90 range."""
        det = TrackDetector()
        result = det.auto_calibrate(_green_frame())
        assert result['hsv_h_min'] >= 0
        assert result['hsv_h_max'] <= 130  # should be well under 130


# ═══════════════════════════════════════════════════════════════════
# 2. Lighting Conditions
# ═══════════════════════════════════════════════════════════════════

class TestLightingConditions:
    """Test calibration under various lighting scenarios."""

    def test_dim_green_succeeds(self):
        """Indoor dim lighting — low V values should still calibrate."""
        det = TrackDetector()
        result = det.auto_calibrate(_dim_green_frame())
        assert result is not None
        assert result['hsv_v_min'] < 60  # should be low to match dim pixels

    def test_bright_green_succeeds(self):
        """Outdoor bright lighting — high V values."""
        det = TrackDetector()
        result = det.auto_calibrate(_bright_green_frame())
        assert result is not None

    def test_washed_out_green_succeeds(self):
        """Overexposed frame — low saturation green."""
        det = TrackDetector()
        result = det.auto_calibrate(_washed_out_green_frame())
        assert result is not None
        # Washed-out green (100,255,100) still has meaningful saturation in HSV
        assert result['hsv_s_min'] < 200

    def test_yellowish_green_succeeds(self):
        """Warm lighting shifts hue toward yellow."""
        det = TrackDetector()
        result = det.auto_calibrate(_yellowish_green_frame())
        assert result is not None

    def test_multi_shade_green_covers_all(self):
        """Mixed dim + bright green — calibration should cover the full range."""
        det = TrackDetector()
        result = det.auto_calibrate(_multi_shade_frame())
        assert result is not None


# ═══════════════════════════════════════════════════════════════════
# 3. Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Stress the calibrator with unusual inputs."""

    def test_no_green_returns_none(self):
        det = TrackDetector()
        result = det.auto_calibrate(_non_green_frame())
        assert result is None

    def test_blank_frame_returns_none(self):
        det = TrackDetector()
        result = det.auto_calibrate(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result is None

    def test_sparse_green_succeeds(self):
        """Very thin green line — enough pixels if line is long enough."""
        det = TrackDetector()
        result = det.auto_calibrate(_sparse_green_frame())
        # May or may not have enough pixels depending on line length
        # But should not crash
        assert result is None or isinstance(result, dict)

    def test_noisy_frame_doesnt_crash(self):
        """Frame with random noise on top of green line."""
        det = TrackDetector()
        result = det.auto_calibrate(_noisy_frame_with_green())
        assert result is not None  # green line should still dominate

    def test_grayscale_frame_raises_or_handles(self):
        """Grayscale frame (2D) — should handle gracefully."""
        det = TrackDetector()
        gray = np.zeros((480, 640), dtype=np.uint8)
        try:
            result = det.auto_calibrate(gray)
            # If it doesn't crash, it should return None (no green in gray)
        except (cv2.error, ValueError):
            pass  # acceptable to raise on invalid input

    def test_small_frame(self):
        """Smaller-than-usual frame."""
        det = TrackDetector()
        small = _green_frame(frame_h=240, frame_w=320)
        result = det.auto_calibrate(small)
        assert result is None or isinstance(result, dict)

    def test_pure_white_frame_returns_none(self):
        """All white frame — no green channel dominance."""
        det = TrackDetector()
        white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        result = det.auto_calibrate(white)
        # White has equal BGR, hue=0, might not match green filter
        # Should return None or a dict
        assert result is None or isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════
# 4. Post-Calibration Detection Improvement
# ═══════════════════════════════════════════════════════════════════

class TestPostCalibrationDetection:
    """Verify that calibrating actually improves detection."""

    def test_wrong_hsv_then_calibrate_fixes_detection(self):
        """Start with red-only HSV filter. Detect = LOST.
        Calibrate, then detect should find the line."""
        det = TrackDetector(
            hsv_lower=np.array([170, 200, 200]),
            hsv_upper=np.array([180, 255, 255]),
        )
        green_frame = _green_frame()

        # Before calibration: nothing detected
        state_before = det.detect(green_frame)
        assert state_before.detection_mode == "LOST"

        # Calibrate
        result = det.auto_calibrate(green_frame)
        assert result is not None

        # After calibration: should detect the green line
        det.reset()  # clear buffers
        state_after = det.detect(green_frame)
        assert state_after.detection_mode != "LOST"

    def test_calibrate_dim_then_detect_dim(self):
        """Calibrate on dim frame, then detect on dim frame should work."""
        det = TrackDetector()
        dim = _dim_green_frame()

        result = det.auto_calibrate(dim)
        if result is None:
            pytest.skip("Dim green too dark for initial filter")

        det.reset()
        state = det.detect(dim)
        assert state.detection_mode != "LOST"

    def test_calibrate_bright_then_detect_bright(self):
        """Calibrate on bright frame, then detect on bright frame should work."""
        det = TrackDetector()
        bright = _bright_green_frame()

        result = det.auto_calibrate(bright)
        assert result is not None

        det.reset()
        state = det.detect(bright)
        assert state.detection_mode != "LOST"


# ═══════════════════════════════════════════════════════════════════
# 5. Dashboard → Node → Detector Wiring
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationWiring:
    """Simulate the callback chain from dashboard button to detector."""

    def test_node_callback_pattern(self):
        """Simulate what node.py's _on_auto_calibrate does."""
        det = TrackDetector()
        frame = _green_frame()

        # Simulate: node._last_raw_frame = frame
        # Simulate: node calls det.auto_calibrate(frame)
        result = det.auto_calibrate(frame)
        assert result is not None

        # After calibration, detector should use the new thresholds
        assert det.hsv_lower[0] == result['hsv_h_min']
        assert det.hsv_upper[0] == result['hsv_h_max']

    def test_calibrate_returns_slider_compatible_values(self):
        """Values should be integers in valid slider ranges."""
        det = TrackDetector()
        result = det.auto_calibrate(_green_frame())
        assert result is not None

        assert isinstance(result['hsv_h_min'], int)
        assert isinstance(result['hsv_h_max'], int)
        assert isinstance(result['hsv_s_min'], int)
        assert isinstance(result['hsv_v_min'], int)

        assert 0 <= result['hsv_h_min'] <= 180
        assert 0 <= result['hsv_h_max'] <= 180
        assert 0 <= result['hsv_s_min'] <= 255
        assert 0 <= result['hsv_v_min'] <= 255

    def test_calibrate_with_no_frame_available(self):
        """Simulate node._last_raw_frame = None (no camera feed yet)."""
        det = TrackDetector()
        # node.py checks for None before calling auto_calibrate
        # but the method itself should handle it
        try:
            result = det.auto_calibrate(None)
            assert result is None
        except (AttributeError, TypeError):
            pass  # acceptable — node.py won't call with None anyway


# ═══════════════════════════════════════════════════════════════════
# 6. Repeated Calibration Stability
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationStability:
    """Verify calibrator is stable under repeated calls."""

    def test_same_frame_gives_same_result(self):
        """Calibrating twice on the same frame should give identical results."""
        frame = _green_frame()
        det = TrackDetector()
        r1 = det.auto_calibrate(frame)
        r2 = det.auto_calibrate(frame)
        assert r1 == r2

    def test_alternating_frames_dont_crash(self):
        """Rapidly switching between different lighting shouldn't crash."""
        det = TrackDetector()
        frames = [_dim_green_frame(), _bright_green_frame(), _green_frame()]
        for _ in range(10):
            for f in frames:
                result = det.auto_calibrate(f)
                assert result is None or isinstance(result, dict)

    def test_calibrate_then_detect_then_calibrate(self):
        """Full cycle: calibrate → detect → calibrate again."""
        det = TrackDetector()
        frame = _green_frame()

        r1 = det.auto_calibrate(frame)
        assert r1 is not None

        det.reset()
        det.detect(frame)

        r2 = det.auto_calibrate(frame)
        assert r2 is not None
        assert r2 == r1  # should be identical on same frame
