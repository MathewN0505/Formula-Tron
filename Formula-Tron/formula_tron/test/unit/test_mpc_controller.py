"""Unit tests for the MPC controller."""

import pytest
import numpy as np
from formula_tron.utils.base_controller import BaseController, ControlState, ControlOutput

try:
    from formula_tron.utils import mpc_controller as mpc_module
    MPCController = mpc_module.MPCController
    CASADI_AVAILABLE = bool(getattr(mpc_module, "CASADI_AVAILABLE", False))
except ImportError:
    CASADI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CASADI_AVAILABLE, reason="casadi not installed")


@pytest.fixture
def mpc():
    return MPCController(
        friction=0.7,
        horizon=12,
        dt=0.05,
        max_speed=5.0,
    )


@pytest.mark.unit
class TestMPCCompliance:

    def test_is_base_controller(self, mpc):
        assert isinstance(mpc, BaseController)

    def test_name(self, mpc):
        assert mpc.name == "MPC"

    def test_detection_mode(self, mpc):
        assert mpc.detection_mode == "POLY_LOOKAHEAD"

    def test_manages_speed(self, mpc):
        assert mpc.manages_speed is True


@pytest.mark.unit
class TestMPCBehavior:

    def test_straight_track(self, mpc, straight_detection):
        state = ControlState(
            target_x=320.0, center_x=320, frame_width=640.0,
            current_speed=1.5, detection=straight_detection,
        )
        output = mpc.compute(state, dt=0.05)
        assert isinstance(output, ControlOutput)
        assert output.speed >= 0.0  # Solver determines speed
        assert abs(output.steering) < 0.05  # Negligible steering

    def test_no_detection_safe_stop(self, mpc, no_detection):
        mpc._last_valid_output = ControlOutput(steering=0.08, speed=1.2)
        state = ControlState(
            target_x=320.0, center_x=320, frame_width=640.0,
            current_speed=1.5, detection=no_detection,
        )
        output = mpc.compute(state, dt=0.05)
        assert output.steering == pytest.approx(0.08)
        assert output.speed == pytest.approx(1.2)
        status = mpc.get_status()
        assert status["fallback_state"] == "DEGRADED"

    def test_update_params(self, mpc):
        mpc.update_params({"base_speed": 4.0})
        assert mpc._max_speed == 4.0


@pytest.mark.unit
class TestMPCLapRamp:

    def test_on_lap_completed_speed_increase(self, mpc):
        initial_speed = mpc._max_speed
        mpc.on_lap_completed(1)
        assert mpc._max_speed == pytest.approx(initial_speed + 0.4)

    def test_on_lap_completed_cumulative(self, mpc):
        initial_speed = mpc._max_speed
        mpc.on_lap_completed(1)
        assert mpc._max_speed == pytest.approx(initial_speed + 0.4)
        mpc.on_lap_completed(2)
        assert mpc._max_speed == pytest.approx(initial_speed + 0.8)
        mpc.on_lap_completed(3)
        assert mpc._max_speed == pytest.approx(initial_speed + 1.2)

    def test_on_lap_completed_smoothness_increase(self, mpc):
        initial_dsteer = mpc._w_dsteer
        mpc.on_lap_completed(1)
        expected_dsteer = (mpc._initial_smoothness_raw + 0.2) * 500.0
        assert mpc._w_dsteer == pytest.approx(expected_dsteer)
        assert mpc._w_dsteer > initial_dsteer

    def test_reset_restores_initial_values(self, mpc):
        initial_speed = mpc._max_speed
        initial_dsteer = mpc._initial_smoothness_raw * 500.0
        mpc.on_lap_completed(2)
        assert mpc._max_speed != initial_speed
        mpc.reset()
        assert mpc._max_speed == pytest.approx(initial_speed)
        assert mpc._w_dsteer == pytest.approx(initial_dsteer)
