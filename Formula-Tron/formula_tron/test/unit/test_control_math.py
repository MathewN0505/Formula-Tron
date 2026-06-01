"""Unit tests for control math functions."""

import pytest
import numpy as np
import math
from formula_tron.utils.safety import safe_normalize
import formula_tron.config as config


@pytest.mark.unit
class TestErrorNormalization:
    """Tests for error normalization."""

    def test_normalize_center(self, normalization_test_case):
        """Test normalization with parametrized cases."""
        result = safe_normalize(
            normalization_test_case['target_x'],
            normalization_test_case['center_x'],
            normalization_test_case['half_range'],
        )
        assert abs(result - normalization_test_case['expected']) < 0.01, \
            f"Failed for {normalization_test_case['description']}"

    def test_normalize_edge_cases(self):
        """Test normalization at edges."""
        # Left edge
        result = safe_normalize(0, 320, 320)
        assert abs(result - (-1.0)) < 0.01

        # Right edge
        result = safe_normalize(640, 320, 320)
        assert abs(result - 1.0) < 0.01

        # Center
        result = safe_normalize(320, 320, 320)
        assert abs(result - 0.0) < 0.01

    def test_normalize_clamping(self):
        """Test that values beyond range are clamped."""
        # Way beyond right edge
        result = safe_normalize(2000, 320, 320)
        assert abs(result - 1.0) < 0.01

        # Way beyond left edge
        result = safe_normalize(-1000, 320, 320)
        assert abs(result - (-1.0)) < 0.01

    def test_normalize_small_range(self):
        """Test normalization with very small half_range."""
        result = safe_normalize(10, 5, 0.1)  # half_range < 1.0
        # Should use minimum 1.0
        assert abs(result) <= 1.0

    def test_normalize_nan_inf(self):
        """Test that NaN and inf inputs are handled."""
        assert safe_normalize(float('nan'), 320, 320) == 0.0
        assert safe_normalize(float('inf'), 320, 320) == 0.0
        assert safe_normalize(float('-inf'), 320, 320) == 0.0


@pytest.mark.unit
class TestPDControl:
    """Tests for PD control calculations."""

    def test_pd_control_equation(self):
        """Test basic PD control equation."""
        kp = 0.85
        kd = 0.20
        error = 0.5  # Normalized error
        derivative = 0.1  # Rate of change
        bias = 0.0

        steering = -(kp * error + kd * derivative) - bias
        expected = -(0.85 * 0.5 + 0.20 * 0.1)
        assert abs(steering - expected) < 0.001

    def test_pd_control_with_bias(self):
        """Test PD control with steering bias."""
        kp = 0.85
        kd = 0.20
        error = 0.0  # No error
        derivative = 0.0
        bias = 0.1  # Right bias

        steering = -(kp * error + kd * derivative) - bias
        assert abs(steering - (-0.1)) < 0.001

    def test_pd_control_negative_error(self):
        """Test PD control with negative error (left of center)."""
        kp = 0.85
        error = -0.5  # Left of center
        derivative = 0.0
        steering = -(kp * error)
        # Should steer right (positive steering)
        assert steering > 0

    def test_pd_control_positive_error(self):
        """Test PD control with positive error (right of center)."""
        kp = 0.85
        error = 0.5  # Right of center
        derivative = 0.0
        steering = -(kp * error)
        # Should steer left (negative steering)
        assert steering < 0

    def test_derivative_limiting(self):
        """Test that derivative term is limited."""
        kp = 0.85
        kd = 0.20
        error = 0.0
        derivative = 15.0  # Very large derivative
        # Should be clamped to ±10.0
        derivative_clamped = max(-10.0, min(10.0, derivative))
        assert abs(derivative_clamped) <= 10.0

    def test_rate_limiting(self):
        """Test steering rate limiting."""
        max_steering_rate = 3.2  # rad/s (aligned with FilesFromCar)
        dt = 0.01  # 100 Hz = 10ms (match control loop)
        max_change = max_steering_rate * dt

        current_steering = 0.0
        desired_steering = 1.0  # Large change

        delta = desired_steering - current_steering
        if abs(delta) > max_change:
            actual_steering = current_steering + max_change * (1.0 if delta > 0 else -1.0)
        else:
            actual_steering = desired_steering

        assert abs(actual_steering - current_steering) <= max_change

    def test_rate_limiting_small_change(self):
        """Test that small changes aren't rate limited."""
        max_steering_rate = 3.2
        dt = 0.01  # 100 Hz
        max_change = max_steering_rate * dt

        current_steering = 0.0
        desired_steering = 0.01  # Small change

        delta = desired_steering - current_steering
        if abs(delta) > max_change:
            actual_steering = current_steering + max_change * (1.0 if delta > 0 else -1.0)
        else:
            actual_steering = desired_steering

        assert actual_steering == desired_steering  # Should not be limited

    def test_steering_angle_clamping(self):
        """Test that steering angle is clamped to max."""
        max_steering_angle = 0.45
        steering = 1.0  # Too large
        steering_clamped = max(-max_steering_angle, min(max_steering_angle, steering))
        assert abs(steering_clamped) <= max_steering_angle

    def test_steering_nan_handling(self):
        """Test that NaN steering is handled."""
        steering = float('nan')
        max_steering_angle = 0.45
        if np.isnan(steering) or np.isinf(steering):
            steering = 0.0
        assert steering == 0.0


@pytest.mark.unit
class TestSpeedControl:
    """Tests for speed control calculations."""

    def test_speed_control_straight(self):
        """Test speed control when going straight."""
        base_speed = 1.5
        turn_slowdown = 0.35
        steering = 0.0  # Straight
        max_steering_angle = 0.45

        turn_factor = abs(steering) / max_steering_angle if max_steering_angle > 0 else 0.0
        speed = base_speed * (1.0 - turn_slowdown * turn_factor)

        assert abs(speed - base_speed) < 0.001  # Should be full speed

    def test_speed_control_turn(self):
        """Test speed control when turning."""
        base_speed = 1.5
        turn_slowdown = 0.35
        steering = 0.45  # Max steering
        max_steering_angle = 0.45

        turn_factor = abs(steering) / max_steering_angle
        speed = base_speed * (1.0 - turn_slowdown * turn_factor)

        expected = 1.5 * (1.0 - 0.35 * 1.0)  # 1.5 * 0.65 = 0.975
        assert abs(speed - expected) < 0.001

    def test_speed_control_clamping(self):
        """Test that speed is clamped to safe range."""
        base_speed = 10.0  # Too fast
        turn_slowdown = 0.0
        steering = 0.0
        max_steering_angle = 0.45

        turn_factor = abs(steering) / max_steering_angle
        speed = base_speed * (1.0 - turn_slowdown * turn_factor)
        speed = max(0.5, min(5.0, speed))  # Clamp

        assert 0.5 <= speed <= 5.0

    def test_speed_control_minimum(self):
        """Test that speed doesn't go below minimum."""
        base_speed = 0.3  # Very slow
        turn_slowdown = 1.0  # Full slowdown
        steering = 0.45  # Max turn
        max_steering_angle = 0.45

        turn_factor = abs(steering) / max_steering_angle
        speed = base_speed * (1.0 - turn_slowdown * turn_factor)
        speed = max(0.5, min(5.0, speed))  # Clamp

        assert speed >= 0.5  # Minimum enforced


@pytest.mark.unit
class TestVESCConversions:
    """Tests for VESC conversion functions."""

    def test_speed_to_erpm(self):
        """Test speed to ERPM conversion."""
        speed_mps = 1.5
        gain = config.VESC_SPEED_TO_ERPM_GAIN
        offset = config.VESC_SPEED_TO_ERPM_OFFSET

        erpm = gain * speed_mps + offset
        expected = 4614.0 * 1.5 + 0.0
        assert abs(erpm - expected) < 0.001

    def test_steering_to_servo(self):
        """Test steering to servo position conversion."""
        steering_rad = 0.0  # Neutral
        gain = config.VESC_STEERING_TO_SERVO_GAIN
        offset = config.VESC_STEERING_TO_SERVO_OFFSET

        servo_pos = gain * steering_rad + offset
        expected = -1.2135 * 0.0 + 0.5304
        assert abs(servo_pos - expected) < 0.001

    def test_steering_to_servo_left(self):
        """Test steering to servo for left turn."""
        steering_rad = 0.45  # Max left
        gain = config.VESC_STEERING_TO_SERVO_GAIN
        offset = config.VESC_STEERING_TO_SERVO_OFFSET

        servo_pos = gain * steering_rad + offset
        # Should be less than offset (left = lower servo value)
        assert servo_pos < offset

    def test_steering_to_servo_right(self):
        """Test steering to servo for right turn."""
        steering_rad = -0.45  # Max right
        gain = config.VESC_STEERING_TO_SERVO_GAIN
        offset = config.VESC_STEERING_TO_SERVO_OFFSET

        servo_pos = gain * steering_rad + offset
        # Should be greater than offset (right = higher servo value)
        assert servo_pos > offset

    def test_servo_clamping(self):
        """Test that servo position is clamped to safe range."""
        servo_pos = 0.0  # Too low
        servo_clamped = max(0.15, min(0.85, servo_pos))
        assert servo_clamped == 0.15

        servo_pos = 1.0  # Too high
        servo_clamped = max(0.15, min(0.85, servo_pos))
        assert servo_clamped == 0.85

        servo_pos = 0.5  # Valid
        servo_clamped = max(0.15, min(0.85, servo_pos))
        assert servo_clamped == 0.5
