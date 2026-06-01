"""Unit tests for AprilTag Lap Timer."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch


@pytest.mark.unit
class TestAprilTagLapTimer:
    """Tests for AprilTagLapTimer class."""

    @staticmethod
    def _mock_corners(tag_count=1, pixel_width=60.0):
        """Create realistic mock corners for AprilTag detection.
        
        Returns corners list matching aruco detectMarkers format.
        Default pixel_width=60 gives estimated distance ~1.4m (well within max_dist=4.0).
        """
        corners = []
        for _ in range(tag_count):
            c = np.array([[[100.0, 100.0], [100.0 + pixel_width, 100.0],
                           [100.0 + pixel_width, 100.0 + pixel_width], [100.0, 100.0 + pixel_width]]])
            corners.append(c)
        return corners

    def test_initialization(self):
        """Test timer initializes correctly."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        timer = AprilTagLapTimer(tag_id=5, min_lap_time=2.0, min_frames_without_tag=10)
        
        assert timer.tag_id == 5
        assert timer.min_lap_time == 2.0
        assert timer.min_frames_without_tag == 10
        assert timer.lap_count == 0
        assert timer.best_lap_time == float('inf')
        assert timer.ready is False  # Should not be ready until reset()
        assert timer.lap_start_time is None  # Should be None until reset()
        assert timer.can_trigger_lap is True  # Default before reset

    def test_reset(self):
        """Test reset clears state."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        timer = AprilTagLapTimer(tag_id=0, min_frames_without_tag=10)
        
        # Manually set some state
        timer.lap_count = 5
        timer.best_lap_time = 10.0
        timer.can_trigger_lap = False
        timer.frames_without_tag = 0
        
        timer.reset()
        
        assert timer.lap_count == 0
        assert timer.best_lap_time == float('inf')
        assert timer.ready is True
        assert timer.lap_start_time is not None
        assert timer.last_completed_lap_time == 0.0
        assert timer.can_trigger_lap is True  # Ready for first lap
        assert timer.frames_without_tag == 10  # Set to threshold for immediate first detection

    def test_no_detection_blank_frame(self):
        """Test blank frame returns no detection."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        timer = AprilTagLapTimer(tag_id=0)
        
        # Blank frame - no tags
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        is_lap, duration, count = timer.check_lap(frame)
        
        assert is_lap is False
        assert count == 0

    def test_none_frame_handled(self):
        """Test None frame is handled gracefully."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        timer = AprilTagLapTimer(tag_id=0)
        
        is_lap, duration, count = timer.check_lap(None)
        
        assert is_lap is False
        assert count == 0

    def test_grayscale_frame(self):
        """Test grayscale frame works."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        timer = AprilTagLapTimer(tag_id=0)
        
        # 2D grayscale frame
        frame = np.zeros((480, 640), dtype=np.uint8)
        is_lap, duration, count = timer.check_lap(frame)
        
        assert is_lap is False  # No tag in blank frame
        assert count == 0

    def test_debounce_with_mock(self):
        """Test debounce logic by mocking _detect_markers.
        
        Now uses disappearance-based detection: tag must be invisible for N frames.
        """
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=5.0, min_frames_without_tag=5, min_frames_with_tag=1)
            timer.reset()  # Must reset to enable counting
            
            mock_ids = np.array([[0]])
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # First detection - triggers lap 1 (can_trigger_lap=True after reset)
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            assert timer.lap_count == 1
            assert timer.can_trigger_lap is False  # Now locked until tag disappears
            
            # Continuous detection (tag still visible) - should NOT count
            mock_time.return_value = 1010.0  # Even after time debounce
            timer.check_lap(frame)
            assert timer.lap_count == 1  # Still 1, tag never disappeared
            
            # Simulate tag disappearing for N frames
            timer._detect_markers = Mock(return_value=([], None, []))
            for i in range(5):
                timer.check_lap(frame)
            assert timer.can_trigger_lap is True  # Now ready for next lap
            
            # Now detect tag again - should trigger lap 2
            mock_time.return_value = 1020.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            assert timer.lap_count == 2

    def test_best_lap_tracked(self):
        """Test best lap time is tracked across multiple laps."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=1.0, min_frames_without_tag=3, min_frames_with_tag=1)
            timer.reset()  # Must reset to enable counting
            
            mock_ids = np.array([[0]])
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Start line (lap 1) — at 15s after reset
            mock_time.return_value = 1015.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            assert timer.lap_count == 1
            assert timer.best_lap_time == 15.0  # First lap: 15s
            
            # Simulate tag disappearing
            timer._detect_markers = Mock(return_value=([], None, []))
            for _ in range(3):
                timer.check_lap(frame)
            
            # Complete lap 2 (duration 2.0s — faster)
            mock_time.return_value = 1017.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            
            # Best lap should now be 2.0s (lap 2 was faster)
            assert timer.best_lap_time == 2.0

    def test_wrong_tag_ignored(self):
        """Test wrong tag ID is ignored."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        timer = AprilTagLapTimer(tag_id=5)  # Looking for tag 5
        timer.reset()  # Must reset to enable counting
        
        # Mock detector to return tag 99
        mock_ids = np.array([[99]])
        timer._detect_markers = Mock(return_value=([], mock_ids, []))
        frame = np.zeros((100, 100), dtype=np.uint8)
        
        timer.check_lap(frame)
        assert timer.lap_count == 0  # Should not count

    def test_multiple_tags_detected(self):
        """Test correct tag found among multiple."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        timer = AprilTagLapTimer(tag_id=5, min_lap_time=0.1, min_frames_with_tag=1)
        timer.reset()  # Must reset to enable counting
        
        # Multiple tags detected, including our target
        mock_ids = np.array([[1], [5], [10]])
        timer._detect_markers = Mock(return_value=(self._mock_corners(tag_count=3), mock_ids, []))
        frame = np.zeros((100, 100), dtype=np.uint8)
        
        timer.check_lap(frame)
        assert timer.lap_count == 1  # Should count tag 5
    
    def test_lap_duration_reported_correctly(self):
        """Test that completed lap duration is returned (not 0.0)."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=1.0, min_frames_without_tag=3, min_frames_with_tag=1)
            timer.reset()  # Must reset to enable counting
            
            mock_ids = np.array([[0]])
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # First lap
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            is_lap, duration, count = timer.check_lap(frame)
            assert is_lap is True
            assert count == 1
            
            # Simulate tag disappearing
            timer._detect_markers = Mock(return_value=([], None, []))
            for _ in range(3):
                timer.check_lap(frame)
            
            # Second lap (duration 3.5s)
            mock_time.return_value = 1003.5
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            is_lap, duration, count = timer.check_lap(frame)
            assert is_lap is True
            assert count == 2
            assert duration == 3.5

    def test_pit_start_scenario(self):
        """Test 'Pit Start': delayed detection after reset."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=5.0, min_frames_with_tag=1)
            timer.reset()  # Must reset to enable counting
            
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # 1. Drive for 20 seconds in pits (no tags)
            mock_time.return_value = 1020.0
            timer._detect_markers = Mock(return_value=([], None, []))
            is_lap, duration, count = timer.check_lap(frame)
            assert is_lap is False
            assert count == 0
            assert duration == 0.0  # Should return 0.0, not current duration
            
            # 2. Finally hit the start line
            mock_time.return_value = 1025.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), np.array([[0]]), []))
            is_lap, duration, count = timer.check_lap(frame)
            
            assert is_lap is True
            assert count == 1
            assert duration == 25.0 # Time since reset

    def test_best_lap_tracks_all_laps(self):
        """Verify that ALL laps (including lap 1) are tracked for Best Lap."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=1.0, min_frames_without_tag=3, min_frames_with_tag=1)
            timer.reset()
            
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # 1. Cross start line (Lap 1 — 20s after reset)
            mock_time.return_value = 1020.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), np.array([[0]]), []))
            timer.check_lap(frame)
            assert timer.lap_count == 1
            assert timer.best_lap_time == 20.0  # First lap counted
            
            # Simulate tag disappearing
            timer._detect_markers = Mock(return_value=([], None, []))
            for _ in range(3):
                timer.check_lap(frame)
            
            # 2. Complete Lap 2 (takes 10s)
            mock_time.return_value = 1030.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), np.array([[0]]), []))
            timer.check_lap(frame)
            assert timer.lap_count == 2
            assert timer.best_lap_time == 10.0  # 10s < 20s
            
            # Simulate tag disappearing
            timer._detect_markers = Mock(return_value=([], None, []))
            for _ in range(3):
                timer.check_lap(frame)
            
            # 3. Complete Lap 3 (takes 5s — should become new best)
            mock_time.return_value = 1035.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), np.array([[0]]), []))
            timer.check_lap(frame)
            assert timer.lap_count == 3
            assert timer.best_lap_time == 5.0

    def test_no_counting_before_reset(self):
        """CRITICAL: Timer should NOT count laps until reset() is called."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=1.0, min_frames_with_tag=1)
            
            # Timer should not be ready until reset() is called
            assert timer.ready is False
            assert timer.lap_start_time is None
            
            # Try to detect tag - should NOT count
            mock_ids = np.array([[0]])
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            is_lap, duration, count = timer.check_lap(frame)
            assert is_lap is False
            assert count == 0
            assert duration == 0.0
            
            # Now reset and it should work
            timer.reset()
            assert timer.ready is True
            assert timer.lap_start_time is not None
            
            is_lap, duration, count = timer.check_lap(frame)
            assert is_lap is True
            assert count == 1

    def test_only_returns_completed_lap_time(self):
        """CRITICAL FIX: Only return completed lap time, not current lap duration."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=1.0, min_frames_without_tag=3, min_frames_with_tag=1)
            timer.reset()
            
            mock_ids = np.array([[0]])
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Complete first lap (immediately after reset, duration will be ~0.0)
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            is_lap, duration, count = timer.check_lap(frame)
            assert is_lap is True
            assert count == 1
            assert duration >= 0.0  # Should return completed lap time (may be ~0 if immediate)
            
            # Wait 2 seconds and simulate tag disappearing for enough frames
            mock_time.return_value = 1002.0
            timer._detect_markers = Mock(return_value=([], None, []))
            
            # Check lap without tag - should return 0.0 (not current duration)
            # Also simulates tag disappearing
            for i in range(3):
                is_lap, duration, count = timer.check_lap(frame)
                assert is_lap is False
                assert duration == 0.0  # CRITICAL: Should be 0.0, not current duration
                assert count == 1  # Count unchanged
            
            # Complete second lap (now we have a real duration)
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            is_lap, duration, count = timer.check_lap(frame)
            assert is_lap is True
            assert count == 2
            assert duration == 2.0  # Completed lap time (time since last detection)

    def test_reset_clears_all_state(self):
        """Test reset clears all lap-related state."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=1.0, min_frames_without_tag=3, min_frames_with_tag=1)
            timer.reset()
            
            mock_ids = np.array([[0]])
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Complete lap 1
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            
            # Simulate tag disappearing
            timer._detect_markers = Mock(return_value=([], None, []))
            for _ in range(3):
                timer.check_lap(frame)
            
            # Complete lap 2
            mock_time.return_value = 1010.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            
            assert timer.lap_count == 2
            assert timer.best_lap_time < float('inf')
            assert timer.can_trigger_lap is False  # Tag currently visible
            
            # Reset
            timer.reset()
            
            assert timer.lap_count == 0
            assert timer.best_lap_time == float('inf')
            assert timer.last_completed_lap_time == 0.0
            assert timer.ready is True
            assert timer.lap_start_time is not None
            assert timer.can_trigger_lap is True  # Reset enables first detection
            assert timer.frames_without_tag == 3  # Set to threshold

    def test_lap_count_zero_resets_times(self):
        """Test that when lap_count is 0, times should be reset."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        timer = AprilTagLapTimer(tag_id=0)
        timer.reset()
        
        # Verify initial state
        assert timer.lap_count == 0
        assert timer.last_completed_lap_time == 0.0
        assert timer.best_lap_time == float('inf')
        
        # Check that no lap time is returned when count is 0
        frame = np.zeros((100, 100), dtype=np.uint8)
        is_lap, duration, count = timer.check_lap(frame)
        
        # Even if tag detected, if count is 0, duration should be 0 until reset
        assert count == 0 or duration == 0.0

    def test_disappearance_based_detection(self):
        """CRITICAL: Test that tag must disappear for N frames before counting again.
        
        This prevents false positives at low speeds where the car lingers near the tag.
        """
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=1.0, min_frames_without_tag=5, min_frames_with_tag=1)
            timer.reset()
            
            mock_ids = np.array([[0]])
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Lap 1 - should count (can_trigger_lap is True after reset)
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            assert timer.lap_count == 1
            assert timer.can_trigger_lap is False
            
            # Continuously see tag for many frames - should NOT count lap 2
            mock_time.return_value = 1100.0  # 100 seconds later
            for i in range(50):
                timer.check_lap(frame)
            assert timer.lap_count == 1  # Still 1, because tag never disappeared
            
            # Tag disappears for only 3 frames (less than threshold of 5)
            timer._detect_markers = Mock(return_value=([], None, []))
            for i in range(3):
                timer.check_lap(frame)
            assert timer.can_trigger_lap is False  # Not enough frames
            
            # Tag reappears - should NOT count
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            assert timer.lap_count == 1  # Still 1
            
            # Tag disappears for 5 frames (meets threshold)
            timer._detect_markers = Mock(return_value=([], None, []))
            for i in range(5):
                timer.check_lap(frame)
            assert timer.can_trigger_lap is True  # Now ready
            
            # Tag reappears - should count lap 2
            mock_time.return_value = 1200.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            assert timer.lap_count == 2

    def test_single_frame_detection_triggers_lap(self):
        """With min_frames_with_tag=1, a single detection triggers a lap."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(
                tag_id=0, min_lap_time=1.0,
                min_frames_without_tag=3, min_frames_with_tag=1,
            )
            timer.reset()
            
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Lap 1 — single detection
            timer._detect_markers = Mock(return_value=(self._mock_corners(), np.array([[0]]), []))
            is_lap, _, count = timer.check_lap(frame)
            assert is_lap is True
            assert count == 1
            
            # Tag disappears for 3 frames
            timer._detect_markers = Mock(return_value=([], None, []))
            for _ in range(3):
                timer.check_lap(frame)
            
            # Lap 2 — single detection again
            mock_time.return_value = 1010.0
            timer._detect_markers = Mock(return_value=(self._mock_corners(), np.array([[0]]), []))
            is_lap, duration, count = timer.check_lap(frame)
            assert is_lap is True
            assert count == 2
            assert duration == 10.0

    def test_min_tag_pixel_width_rejects_tiny(self):
        """Reject detections where tag pixel width is below threshold."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(
                tag_id=0, min_lap_time=1.0,
                min_frames_without_tag=3, min_tag_pixel_width=50.0,
            )
            timer.reset()
            
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Mock a tiny tag (corners very close together → small pixel width)
            tiny_corners = [np.array([[10.0, 10.0], [12.0, 10.0], [12.0, 12.0], [10.0, 12.0]])]
            timer._detect_markers = Mock(return_value=(tiny_corners, np.array([[0]]), []))
            is_lap, _, count = timer.check_lap(frame)
            assert is_lap is False
            assert count == 0  # Rejected — pixel width ~2 < threshold 50

    def test_continuous_visibility_no_double_count(self):
        """Test that continuous tag visibility doesn't cause double counting.
        
        Even with time debounce expired, if tag was never invisible, no new lap.
        """
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(tag_id=0, min_lap_time=1.0, min_frames_without_tag=3, min_frames_with_tag=1)
            timer.reset()
            
            mock_ids = np.array([[0]])
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Lap 1
            timer._detect_markers = Mock(return_value=(self._mock_corners(), mock_ids, []))
            timer.check_lap(frame)
            assert timer.lap_count == 1
            
            # Simulate 1000 frames of continuous visibility over 100 seconds
            # This simulates a car stopped in front of the tag
            for i in range(1000):
                mock_time.return_value = 1000.0 + (i * 0.1)  # 0.1s per frame
                timer.check_lap(frame)
            
            # Still lap 1 - no false positive despite long time
            assert timer.lap_count == 1

    def test_direction_based_approach_allows_lap(self):
        """Test that an approach pattern (decreasing distance) allows a lap to count.
        
        Uses min_frames_with_tag=2 so the direction gate has time to gather
        distance data before the lap can trigger.
        """
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(
                tag_id=0, min_lap_time=1.0,
                min_frames_without_tag=3, min_frames_with_tag=2,
            )
            timer.reset()
            
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Lap 1 (first lap after reset — direction check skipped)
            # Need 2 frames to trigger since min_frames_with_tag=2
            timer._detect_markers = Mock(return_value=(self._mock_corners(), np.array([[0]]), []))
            timer.check_lap(frame)  # frame 1
            timer.check_lap(frame)  # frame 2 — triggers lap 1
            assert timer.lap_count == 1
            
            # Tag disappears for enough frames
            timer._detect_markers = Mock(return_value=([], None, []))
            for _ in range(3):
                timer.check_lap(frame)
            assert timer.can_trigger_lap is True
            
            # Now simulate approach pattern: tag appears at decreasing distances
            # (larger pixel width = closer distance)
            mock_time.return_value = 1020.0
            
            # Frame 1: tag at ~2.0m (pixel_width ~42)
            timer._detect_markers = Mock(return_value=(self._mock_corners(pixel_width=42.0), np.array([[0]]), []))
            timer.check_lap(frame)
            assert timer.lap_count == 1  # Only 1 frame, need 2
            
            # Frame 2: tag at ~1.5m (pixel_width ~56) — approaching!
            timer._detect_markers = Mock(return_value=(self._mock_corners(pixel_width=56.0), np.array([[0]]), []))
            is_lap, _, count = timer.check_lap(frame)
            # 2 frames + approach detected (distance decreased) → should trigger lap 2
            assert is_lap is True
            assert count == 2

    def test_direction_based_no_approach_blocks_lap(self):
        """Test that a receding pattern (car moving AWAY from tag) blocks lap counting.
        
        Uses min_frames_with_tag=2 to force 2 frames of distance data before
        the lap can trigger, giving the direction gate real data to work with.
        """
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(
                tag_id=0, min_lap_time=1.0,
                min_frames_without_tag=3, min_frames_with_tag=2,
            )
            timer.reset()
            
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Complete lap 1 (direction check is skipped for first lap)
            timer._detect_markers = Mock(return_value=(self._mock_corners(), np.array([[0]]), []))
            timer.check_lap(frame)  # frame 1
            timer.check_lap(frame)  # frame 2 — triggers
            assert timer.lap_count == 1
            
            # Tag disappears
            timer._detect_markers = Mock(return_value=([], None, []))
            for _ in range(3):
                timer.check_lap(frame)
            assert timer.can_trigger_lap is True
            
            # Tag reappears but car is RECEDING (pixel width decreasing = distance increasing)
            mock_time.return_value = 1020.0
            
            # Frame 1: tag at ~1.4m (pixel_width=60)
            timer._detect_markers = Mock(return_value=(self._mock_corners(pixel_width=60.0), np.array([[0]]), []))
            timer.check_lap(frame)
            assert timer.lap_count == 1  # Only 1 frame, need 2
            
            # Frame 2: tag at ~1.8m (pixel_width=47) — receding!
            timer._detect_markers = Mock(return_value=(self._mock_corners(pixel_width=47.0), np.array([[0]]), []))
            timer.check_lap(frame)
            assert timer.lap_count == 1  # Blocked — distance is increasing (car moving away)

    def test_direction_reset_on_disappearance(self):
        """Test that direction state resets when tag disappears."""
        from formula_tron.utils.lap_timer import AprilTagLapTimer
        
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 1000.0
            timer = AprilTagLapTimer(
                tag_id=0, min_lap_time=1.0,
                min_frames_without_tag=3, min_frames_with_tag=1,
            )
            timer.reset()
            
            frame = np.zeros((100, 100), dtype=np.uint8)
            
            # Lap 1
            timer._detect_markers = Mock(return_value=(self._mock_corners(), np.array([[0]]), []))
            timer.check_lap(frame)
            assert timer.lap_count == 1
            
            # After a lap, distance history should be reset
            assert timer._distance_history == []
            assert timer._saw_approach is False
