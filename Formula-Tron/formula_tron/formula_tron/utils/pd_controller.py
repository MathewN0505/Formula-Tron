"""
PD Controller — implements BaseController for POLY_LOOKAHEAD and LEGACY modes.

Extracted from vision_controller.py's image_callback to enable the registry pattern.
Both modes use identical PD math; only the track detection pipeline differs.
"""

from .base_controller import BaseController, ControlState, ControlOutput
from .safety import safe_normalize


class PDController(BaseController):
    """
    Proportional-Derivative controller for track following.
    
    Used for both POLY_LOOKAHEAD and LEGACY modes — the only difference
    is which track detection pipeline feeds the target_x.
    
    Usage:
        poly_pd = PDController("POLY_LOOKAHEAD", detection_mode="POLY_LOOKAHEAD")
        legacy_pd = PDController("LEGACY", detection_mode="LEGACY")
    """

    def __init__(self, mode_name: str, detection_mode: str = None,
                 kp: float = 0.85, kd: float = 0.15, steering_bias: float = 0.0):
        self._name = mode_name
        self._detection_mode = detection_mode or mode_name
        
        # PD gains (tunable from GUI)
        self.kp = kp
        self.kd = kd
        self.steering_bias = steering_bias
        
        # Internal state
        self.prev_error = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def detection_mode(self) -> str:
        return self._detection_mode

    @property
    def manages_speed(self) -> bool:
        return False  # Uses generic turn-slowdown formula

    def compute(self, state: ControlState, dt: float) -> ControlOutput:
        """PD control: error → proportional + derivative → steering."""
        error = safe_normalize(state.target_x, state.center_x, state.frame_width / 2.0)
        
        d_error = (error - self.prev_error) / dt
        d_error = max(-10.0, min(10.0, d_error))  # Limit D term
        
        self.prev_error = error
        steering = -(self.kp * error + self.kd * d_error) - self.steering_bias
        
        return ControlOutput(steering=steering)

    def reset(self, initial_error: float = 0.0):
        """Clear PD state to prevent derivative kick on mode switch."""
        self.prev_error = initial_error

    def update_params(self, params):
        """Accept GUI parameter updates."""
        if 'kp' in params:
            self.kp = params['kp']
        if 'kd' in params:
            self.kd = params['kd']
        if 'steering_bias' in params:
            self.steering_bias = params['steering_bias']
