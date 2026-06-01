"""Unit tests for the kinematic bicycle model and curvature utilities."""

import pytest
import numpy as np
from formula_tron.utils.vehicle_model import (
    BicycleModel,
    compute_curvature,
    curvature_to_max_speed,
    generate_reference_from_poly,
)


@pytest.mark.unit
class TestBicycleModel:

    def test_straight_line(self, vehicle_model):
        """Zero steering -> car goes forward, x unchanged."""
        state = np.array([0.0, 0.0, 0.0, 1.0])
        control = np.array([0.0, 0.0])
        ns = vehicle_model.predict(state, control, dt=0.1)
        assert ns[0] == pytest.approx(0.1, abs=1e-6)  # x += v*cos(0)*dt
        assert ns[1] == pytest.approx(0.0, abs=1e-6)  # y stays 0
        assert ns[2] == pytest.approx(0.0, abs=1e-6)  # theta stays 0
        assert ns[3] == pytest.approx(1.0, abs=1e-6)  # speed unchanged

    def test_left_turn(self, vehicle_model):
        """Positive steering -> heading increases (turns left)."""
        state = np.array([0.0, 0.0, 0.0, 1.0])
        control = np.array([0.0, 0.3])
        ns = vehicle_model.predict(state, control, dt=0.1)
        assert ns[2] > 0.0  # theta increased

    def test_right_turn(self, vehicle_model):
        """Negative steering -> heading decreases (turns right)."""
        state = np.array([0.0, 0.0, 0.0, 1.0])
        control = np.array([0.0, -0.3])
        ns = vehicle_model.predict(state, control, dt=0.1)
        assert ns[2] < 0.0

    def test_zero_speed_no_movement(self, vehicle_model):
        """Zero speed -> no position or heading change."""
        state = np.array([5.0, 3.0, 1.0, 0.0])
        control = np.array([0.0, 0.3])
        ns = vehicle_model.predict(state, control, dt=0.1)
        assert ns[0] == pytest.approx(5.0, abs=1e-9)
        assert ns[1] == pytest.approx(3.0, abs=1e-9)
        assert ns[2] == pytest.approx(1.0, abs=1e-9)

    def test_acceleration(self, vehicle_model):
        """Positive accel increases speed."""
        state = np.array([0.0, 0.0, 0.0, 1.0])
        control = np.array([1.0, 0.0])
        ns = vehicle_model.predict(state, control, dt=0.1)
        assert ns[3] == pytest.approx(1.1, abs=1e-6)

    def test_speed_clamped_to_max(self, vehicle_model):
        """Speed cannot exceed max_speed."""
        state = np.array([0.0, 0.0, 0.0, 4.9])
        control = np.array([2.0, 0.0])
        ns = vehicle_model.predict(state, control, dt=1.0)
        assert ns[3] <= vehicle_model.max_speed

    def test_speed_clamped_to_zero(self, vehicle_model):
        """Speed cannot go below zero."""
        state = np.array([0.0, 0.0, 0.0, 0.1])
        control = np.array([-5.0, 0.0])
        ns = vehicle_model.predict(state, control, dt=1.0)
        assert ns[3] >= 0.0

    def test_steering_clamped(self, vehicle_model):
        """Steering is clamped to max_steering."""
        state = np.array([0.0, 0.0, 0.0, 1.0])
        control = np.array([0.0, 10.0])  # Way over limit
        ns = vehicle_model.predict(state, control, dt=0.1)
        # Should still produce valid output, not crash
        assert not np.any(np.isnan(ns))

    def test_rollout_shape(self, vehicle_model):
        """Rollout returns (N+1, 4) trajectory."""
        init = np.array([0.0, 0.0, 0.0, 1.0])
        controls = np.zeros((10, 2))
        traj = vehicle_model.rollout(init, controls, dt=0.05)
        assert traj.shape == (11, 4)
        assert np.array_equal(traj[0], init)

    def test_rollout_batch_shape(self, vehicle_model):
        """Batch rollout returns (B, H+1, 4)."""
        init = np.array([0.0, 0.0, 0.0, 1.0])
        controls = np.zeros((50, 8, 2))
        trajs = vehicle_model.rollout_batch(init, controls, dt=0.05)
        assert trajs.shape == (50, 9, 4)


@pytest.mark.unit
class TestCurvatureUtilities:

    def test_zero_curvature_straight(self):
        """Straight polynomial has zero curvature."""
        poly = np.array([0.0, 0.5, 320.0])
        k = compute_curvature(poly, y=0.0)
        assert k == pytest.approx(0.0, abs=1e-9)

    def test_positive_curvature(self):
        """Positive quadratic coefficient -> positive curvature."""
        poly = np.array([0.01, 0.0, 320.0])
        k = compute_curvature(poly, y=0.0)
        assert k > 0.0
        assert k == pytest.approx(0.02, abs=1e-6)

    def test_none_poly_returns_zero(self):
        assert compute_curvature(None) == 0.0

    def test_short_poly_returns_zero(self):
        assert compute_curvature(np.array([1.0])) == 0.0

    def test_max_speed_straight(self):
        """Zero curvature -> max speed."""
        v = curvature_to_max_speed(0.0, v_max=5.0)
        assert v == 5.0

    def test_max_speed_curved(self):
        """High curvature -> reduced speed."""
        v = curvature_to_max_speed(0.05, v_max=5.0)
        assert v < 5.0
        assert v >= 0.5

    def test_reference_from_poly_shape(self):
        """Reference generation returns correct shape."""
        poly = np.array([0.001, -0.1, 320.0])
        ref = generate_reference_from_poly(poly, bev_height=192, n_points=15)
        assert ref.shape == (15, 2)

    def test_reference_from_none_poly(self):
        """None polynomial returns zeros."""
        ref = generate_reference_from_poly(None, bev_height=192, n_points=10)
        assert ref.shape == (10, 2)
        assert np.allclose(ref, 0.0)
