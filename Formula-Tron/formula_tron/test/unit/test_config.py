"""Unit tests for config module."""

import pytest
import math
import formula_tron.config as config


@pytest.mark.unit
class TestConfigValues:
    """Tests for configuration values."""

    def test_vesc_gain_values(self):
        """Test that VESC gain values are reasonable."""
        assert isinstance(config.VESC_SPEED_TO_ERPM_GAIN, (int, float))
        assert config.VESC_SPEED_TO_ERPM_GAIN > 0
        assert config.VESC_SPEED_TO_ERPM_GAIN < 10000

        assert isinstance(config.VESC_STEERING_TO_SERVO_GAIN, (int, float))
        assert abs(config.VESC_STEERING_TO_SERVO_GAIN) < 10.0

    def test_vesc_offset_values(self):
        """Test that VESC offset values are reasonable."""
        assert isinstance(config.VESC_SPEED_TO_ERPM_OFFSET, (int, float))
        assert isinstance(config.VESC_STEERING_TO_SERVO_OFFSET, (int, float))
        assert 0.0 <= config.VESC_STEERING_TO_SERVO_OFFSET <= 1.0

    def test_pd_defaults(self):
        """Test that PD defaults are reasonable."""
        assert isinstance(config.KP_DEFAULT, (int, float))
        assert 0.0 < config.KP_DEFAULT < 10.0

        assert isinstance(config.KD_DEFAULT, (int, float))
        assert 0.0 <= config.KD_DEFAULT < 5.0

    def test_steering_limits(self):
        """Test that steering limits are reasonable."""
        assert isinstance(config.MAX_STEERING_ANGLE, (int, float))
        assert 0.0 < config.MAX_STEERING_ANGLE < math.pi / 2  # Less than 90 degrees

        assert isinstance(config.MAX_STEERING_RATE, (int, float))
        assert config.MAX_STEERING_RATE > 0
        assert config.MAX_STEERING_RATE < 10.0  # Reasonable rate

    def test_speed_values(self):
        """Test that speed values are reasonable."""
        assert isinstance(config.BASE_SPEED, (int, float))
        assert 0.0 < config.BASE_SPEED < 10.0  # Reasonable speed

        assert isinstance(config.TURN_SLOWDOWN, (int, float))
        assert 0.0 <= config.TURN_SLOWDOWN <= 1.0  # Percentage

    def test_track_width(self):
        """Test that track width is reasonable."""
        assert isinstance(config.VISUAL_TRACK_WIDTH, int)
        assert 100 < config.VISUAL_TRACK_WIDTH < 2000  # Reasonable pixel width
        
        assert isinstance(config.PHYSICAL_TRACK_WIDTH, (int, float))
        assert 0.1 < config.PHYSICAL_TRACK_WIDTH < 2.0  # Reasonable meter width

    def test_roi_ratio(self):
        """Test that ROI ratio is valid."""
        assert isinstance(config.ROI_HEIGHT_RATIO, (int, float))
        assert 0.0 < config.ROI_HEIGHT_RATIO <= 1.0

    def test_hsv_values(self):
        """Test that HSV values are in valid ranges."""
        assert isinstance(config.HSV_H_MIN, int)
        assert 0 <= config.HSV_H_MIN <= 180

        assert isinstance(config.HSV_H_MAX, int)
        assert 0 <= config.HSV_H_MAX <= 180
        assert config.HSV_H_MAX >= config.HSV_H_MIN

        assert isinstance(config.HSV_S_MIN, int)
        assert 0 <= config.HSV_S_MIN <= 255

        assert isinstance(config.HSV_V_MIN, int)
        assert 0 <= config.HSV_V_MIN <= 255

    def test_safety_values(self):
        """Test that safety values are reasonable."""
        assert isinstance(config.MAX_CONSECUTIVE_ERRORS, int)
        assert config.MAX_CONSECUTIVE_ERRORS > 0

        assert isinstance(config.VISION_TIMEOUT_SEC, (int, float))
        assert config.VISION_TIMEOUT_SEC > 0

    def test_smoothing_alpha(self):
        """Test that smoothing alpha is valid."""
        assert isinstance(config.TARGET_SMOOTHING_ALPHA, (int, float))
        assert 0.0 < config.TARGET_SMOOTHING_ALPHA <= 1.0

    def test_topic_names(self):
        """Test that topic names are strings."""
        assert isinstance(config.CAMERA_TOPIC, str)
        assert config.CAMERA_TOPIC.startswith('/')

        assert isinstance(config.MOTOR_TOPIC, str)
        assert config.MOTOR_TOPIC.startswith('/')

        assert isinstance(config.SERVO_TOPIC, str)
        assert config.SERVO_TOPIC.startswith('/')

    def test_depth_safety_values(self):
        """Test that depth safety values are reasonable."""
        assert isinstance(config.DEPTH_TOPIC, str)
        assert config.DEPTH_TOPIC.startswith('/')
        
        assert isinstance(config.MIN_SAFE_DISTANCE_DEFAULT, (int, float))
        assert config.MIN_SAFE_DISTANCE_DEFAULT > 0
        
        assert isinstance(config.MIN_SAFE_DISTANCE_MIN, (int, float))
        assert config.MIN_SAFE_DISTANCE_MIN > 0
        
        assert isinstance(config.MIN_SAFE_DISTANCE_MAX, (int, float))
        assert config.MIN_SAFE_DISTANCE_MAX > config.MIN_SAFE_DISTANCE_MIN
        
        # Check default inside range
        assert config.MIN_SAFE_DISTANCE_MIN <= config.MIN_SAFE_DISTANCE_DEFAULT <= config.MIN_SAFE_DISTANCE_MAX
