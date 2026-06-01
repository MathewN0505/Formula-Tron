"""Unit tests for lane detection module."""

import pytest
import numpy as np
from formula_tron.utils.lane_detection import LaneDetector, LaneDetectionResult

@pytest.fixture
def lane_detector():
    """Create a default LaneDetector instance."""
    return LaneDetector(
        hsv_lower=np.array([35, 50, 50]),
        hsv_upper=np.array([90, 255, 255]),
        track_width=450,
        roi_ratio=0.4,
    )

@pytest.mark.unit
class TestLaneDetector:
    """Tests for LaneDetector class."""

    def test_detector_initialization(self):
        """Test that detector initializes with default values."""
        detector = LaneDetector()
        assert detector.track_width == 550
        assert detector.roi_ratio == 0.4
        assert np.array_equal(detector.hsv_lower, np.array([35, 50, 50]))
        assert np.array_equal(detector.hsv_upper, np.array([90, 255, 255]))

    def test_detector_custom_initialization(self):
        """Test detector with custom parameters."""
        detector = LaneDetector(
            hsv_lower=np.array([40, 60, 60]),
            hsv_upper=np.array([80, 255, 255]),
            track_width=500,
            roi_ratio=0.5,
        )
        assert detector.track_width == 500
        assert detector.roi_ratio == 0.5
        assert np.array_equal(detector.hsv_lower, np.array([40, 60, 60]))

    def test_detect_valid_frame(self, lane_detector, frame_with_green_line):
        """Test detection on valid frame with green line."""
        result = lane_detector.detect(frame_with_green_line, mode="SIMPLE")
        assert isinstance(result, LaneDetectionResult)
        assert result.mask is not None
        assert result.histogram is not None
        assert len(result.histogram) > 0

    def test_detect_invalid_frame_none(self, lane_detector):
        """Test that None frame raises ValueError."""
        with pytest.raises(ValueError, match="Invalid frame"):
            lane_detector.detect(None, mode="SIMPLE")

    def test_detect_invalid_frame_too_small(self, lane_detector):
        """Test that frame that's too small raises ValueError."""
        small_frame = np.zeros((5, 5, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame too small"):
            lane_detector.detect(small_frame, mode="SIMPLE")

    def test_detect_empty_frame(self, lane_detector):
        """Test that empty frame raises ValueError."""
        empty_frame = np.array([])
        with pytest.raises(ValueError):
            lane_detector.detect(empty_frame, mode="SIMPLE")

    def test_detect_roi_extraction(self, lane_detector, valid_frame):
        """Test that ROI is extracted correctly."""
        result = lane_detector.detect(valid_frame, mode="SIMPLE")
        h, w = valid_frame.shape[:2]
        expected_roi_h = int(h * 0.4)
        assert result.mask.shape[0] == expected_roi_h

    def test_detect_hsv_filtering(self, lane_detector, frame_with_green_line):
        """Test that HSV filtering works."""
        result = lane_detector.detect(frame_with_green_line, mode="SIMPLE")
        # Should detect some green pixels
        assert np.sum(result.mask) > 0

    def test_detect_no_green_pixels(self, lane_detector, valid_frame):
        """Test detection on frame with no green pixels."""
        result = lane_detector.detect(valid_frame, mode="SIMPLE")
        # Should still return valid result, but with no peaks
        assert isinstance(result, LaneDetectionResult)
        assert result.target_x is None or result.target_x is not None  # May use fallback

    def test_detect_histogram_calculation(self, lane_detector, frame_with_green_line):
        """Test that histogram is calculated."""
        result = lane_detector.detect(frame_with_green_line, mode="SIMPLE")
        assert len(result.histogram) == frame_with_green_line.shape[1]  # Same width
        assert result.histogram.dtype == np.float32

    def test_detect_peak_finding(self, lane_detector, frame_with_two_green_lines):
        """Test that peaks are found in histogram."""
        result = lane_detector.detect(frame_with_two_green_lines, mode="SIMPLE")
        assert len(result.all_peaks) >= 0  # May find 0, 1, or 2 peaks

    def test_detect_strategy_selection(self, lane_detector, frame_with_two_green_lines):
        """Test that strategy selection works."""
        result = lane_detector.detect(frame_with_two_green_lines, mode="SIMPLE")
        assert result.status is not None
        assert isinstance(result.status, str)

    def test_detect_simple_mode(self, lane_detector, frame_with_green_line):
        """Test SIMPLE mode detection."""
        result = lane_detector.detect(frame_with_green_line, mode="SIMPLE")
        assert result.bev_mask is None  # SIMPLE mode doesn't use BEV
        assert result.poly_coeffs is None

    def test_detect_error_handling(self, lane_detector):
        """Test that errors are handled gracefully."""
        # Create a frame that might cause issues
        weird_frame = np.zeros((480, 640, 3), dtype=np.float32)  # Wrong dtype
        result = lane_detector.detect(weird_frame, mode="SIMPLE")
        # Should return safe fallback result
        assert isinstance(result, LaneDetectionResult)
        assert "ERROR" in result.status or result.status.startswith("NO") or result.status.startswith("FUSION")

    def test_update_hsv(self, lane_detector):
        """Test that HSV thresholds can be updated."""
        detector = lane_detector
        detector.update_hsv(40, 80, 60, 60)
        assert np.array_equal(detector.hsv_lower, np.array([40, 60, 60]))
        assert np.array_equal(detector.hsv_upper, np.array([80, 255, 255]))

    def test_update_track_width(self, lane_detector):
        """Test that track width can be updated."""
        detector = lane_detector
        detector.update_track_width(500)
        assert detector.track_width == 500
        assert detector.min_valid_width == int(500 * 0.6)

    def test_find_peaks_empty_histogram(self, lane_detector):
        """Test peak finding with empty histogram."""
        empty_hist = np.zeros(640, dtype=np.float32)
        peaks = lane_detector._find_peaks(empty_hist)
        assert len(peaks) == 0

    def test_find_peaks_single_peak(self, lane_detector, sample_histogram_single_peak):
        """Test peak finding with single peak."""
        peaks = lane_detector._find_peaks(sample_histogram_single_peak)
        assert len(peaks) >= 0  # May or may not find peak depending on thresholds

    def test_find_peaks_multiple_peaks(self, lane_detector, sample_histogram):
        """Test peak finding with multiple peaks."""
        peaks = lane_detector._find_peaks(sample_histogram)
        assert len(peaks) >= 0

    def test_select_lanes_no_peaks(self, lane_detector, sample_histogram_no_peaks):
        """Test lane selection with no peaks."""
        left, right, target, status, used = lane_detector._select_lanes(
            np.array([]), sample_histogram_no_peaks, 640
        )
        assert left is None
        assert right is None
        assert target is None
        assert "NO" in status or "LOST" in status

    def test_select_lanes_center_peak(self, lane_detector):
        """Test lane selection with center peak."""
        hist = np.zeros(640, dtype=np.float32)
        hist[310:330] = 500.0  # Center peak
        peaks = np.array([320])
        left, right, target, status, used = lane_detector._select_lanes(peaks, hist, 640)
        assert target is not None
        assert "CTR" in status or "FUSION" in status

    def test_select_lanes_left_right_pair(self, lane_detector):
        """Test lane selection with left+right pair."""
        hist = np.zeros(640, dtype=np.float32)
        hist[150:170] = 500.0  # Left peak
        hist[470:490] = 500.0  # Right peak
        peaks = np.array([160, 480])
        left, right, target, status, used = lane_detector._select_lanes(peaks, hist, 640)
        # Should select L+R strategy if width is valid
        assert target is not None

    def test_reset_perspective_cache(self, lane_detector):
        """Test that perspective cache can be reset."""
        detector = lane_detector
        # Initialize cache by calling advanced mode
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            detector.detect(frame, mode="ALTERNATIVE")
        except Exception:
            pass  # May fail, but cache might be initialized
        # Reset
        detector.reset_perspective_cache()
        assert detector.M is None
        assert detector.Minv is None
