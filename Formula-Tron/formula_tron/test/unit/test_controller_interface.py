import pytest
import numpy as np
from formula_tron.utils.base_controller import BaseController, ControlState, ControlOutput
from formula_tron.utils.pd_controller import PDController
from formula_tron.utils.controller_registry import ControllerRegistry
from formula_tron.utils.track_detection import TrackDetectionResult

@pytest.fixture
def mock_detection():
    """Create a basic valid detection result."""
    return TrackDetectionResult(
        target_x=320,
        left_peak=200,
        right_peak=440,
        mask=np.zeros((100, 100), dtype=np.uint8),
        histogram=np.zeros(100),
        all_peaks=[],
        used_peaks=[],
        status="OK",
        target_x_bev=None,
        bev_mask=None,
        poly_coeffs=None
    )

@pytest.fixture
def registry():
    """Create a fresh registry."""
    return ControllerRegistry()

@pytest.fixture
def pd_controller():
    """Create a test PD controller."""
    return PDController("TEST_PD", detection_mode="TEST_MODE", kp=1.0, kd=0.0)

@pytest.mark.unit
def test_registry_operations(registry, pd_controller):
    """Verify core registry functionality: register, get, has, list."""
    # Register
    registry.register(pd_controller)
    
    # Retrieval
    retrieved = registry.get("TEST_PD")
    assert retrieved == pd_controller
    assert retrieved.name == "TEST_PD"
    
    # Existence check
    assert registry.has("TEST_PD")
    assert not registry.has("NON_EXISTENT")
    
    # Available modes list
    modes = registry.available_modes()
    assert "TEST_PD" in modes

@pytest.mark.unit
def test_duplicate_registration_error(registry, pd_controller):
    """Ensure registering a duplicate name raises ValueError."""
    registry.register(pd_controller)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(pd_controller)

@pytest.mark.unit
def test_get_unknown_controller_error(registry):
    """Ensure getting an unknown controller raises KeyError."""
    with pytest.raises(KeyError, match="Unknown control mode"):
        registry.get("UNKNOWN_MODE")

@pytest.mark.unit
def test_registry_has_method(registry, pd_controller):
    """Test has() before and after registration."""
    assert registry.has("TEST_PD") is False
    registry.register(pd_controller)
    assert registry.has("TEST_PD") is True
    assert registry.has("NONEXISTENT") is False

@pytest.mark.unit
def test_registry_fallback_pattern(registry, pd_controller):
    """Simulate the _safe_get_controller fallback logic from vision_controller."""
    registry.register(pd_controller)
    target_mode = "NONEXISTENT"  # Not registered

    # Fallback: if target not available, use fallback (TEST_PD here, POLY_LOOKAHEAD in prod)
    if registry.has(target_mode):
        ctrl = registry.get(target_mode)
    else:
        ctrl = registry.get("TEST_PD")

    assert ctrl.name == "TEST_PD"

@pytest.mark.unit
@pytest.mark.parametrize("target_x, expected_steering", [
    (320, 0.0),       # Center -> 0 steering
    (340, -0.0625),   # Right -> Negative steering (Left)
    (300, 0.0625),    # Left -> Positive steering (Right)
])
def test_pd_compute_logic(pd_controller, mock_detection, target_x, expected_steering):
    """Verify PD controller math is correct using parametrization."""
    state = ControlState(
        target_x=target_x,
        center_x=320,
        frame_width=640,
        current_speed=1.0,
        detection=mock_detection
    )
    
    output = pd_controller.compute(state, dt=0.1)
    
    assert isinstance(output, ControlOutput)
    assert output.steering == pytest.approx(expected_steering, abs=1e-4)
    assert output.speed == 0.0

@pytest.mark.unit
def test_parameter_broadcast(registry, pd_controller):
    """Verify update_all_params reaches the controller."""
    registry.register(pd_controller)
    assert pd_controller.kp == 1.0
    
    registry.update_all_params({'kp': 5.0, 'kd': 2.5})
    
    assert pd_controller.kp == 5.0
    assert pd_controller.kd == 2.5

@pytest.mark.unit
def test_derivative_kick_prevention(pd_controller, mock_detection):
    """
    Verify that reset(initial_error) correctly seeds the D-term.
    This ensures smooth transitions when engaging autonomous mode with non-zero error.
    """
    # Scenario: Large initial error (0.25)
    # target_x=400, center=320 -> error = 80/320 = 0.25
    state = ControlState(
        target_x=400, center_x=320, frame_width=640, current_speed=1.0, detection=mock_detection
    )
    current_error = 0.25
    
    # Set gains
    pd_controller.kp = 1.0
    pd_controller.kd = 1.0
    
    # With reset(init_error), expect smooth start
    # d_error = (0.25 - 0.25)/dt = 0
    # steering = -(kp*error + kd*0) = -0.25
    pd_controller.reset(initial_error=current_error)
    output = pd_controller.compute(state, dt=0.05)
    
    assert output.steering == pytest.approx(-0.25, abs=1e-4)

@pytest.mark.unit
def test_mpc_compliance():
    """Verify MPCController follows the BaseController contract."""
    try:
        from formula_tron.utils.mpc_controller import MPCController
        mpc = MPCController()
        
        # Check Interface inheritance
        assert isinstance(mpc, BaseController)
        
        # Check Properties
        assert mpc.name == "MPC"
        assert mpc.detection_mode == "POLY_LOOKAHEAD"
        assert mpc.manages_speed is True
        
        # Check Methods exist
        assert hasattr(mpc, 'compute')
        assert hasattr(mpc, 'reset')
        assert hasattr(mpc, 'update_params')
        
    except ImportError:
        pytest.skip("casadi not installed or MPC dependencies missing")



