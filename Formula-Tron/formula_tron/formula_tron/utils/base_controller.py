"""
Base Controller Interface for Formula-Tron control modes.

All control modes (PD, MPC, etc.) implement this interface.
This enables the controller registry pattern where adding a new mode
is just: create a controller file + register it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class ControlState:
    """Input state passed to every controller's compute() method."""
    target_x: float           # Smoothed track target (pixel x)
    center_x: float           # Frame center x
    frame_width: float        # Full frame width (for normalization)
    current_speed: float      # Current car speed (m/s)
    detection: Any            # TrackDetectionResult (full detection output)
    current_steering: float = 0.0  # Current steering command (radians)
    autonomous_running: bool = False # True if the car is actively auto-driving
    raw_frame: Any = None            # Raw BGR camera frame (used by IL controller)
    beta_path: Any = None            # Shared path contract for isolated beta controllers
    base_speed: float = 0.0          # Current GUI-selected speed cap / target speed
    odom_x: float = 0.0              # Odometry X position (meters)
    odom_y: float = 0.0              # Odometry Y position (meters)
    odom_heading: float = 0.0        # Odometry heading (radians)
    track_offset: float = 0.0        # Vision-based lateral track offset (-0.5 to 0.5)
    auto_steer_recommendation: float = 0.0  # Background autonomous steering suggestion


@dataclass
class ControlOutput:
    """Output from a controller's compute() method."""
    steering: float = 0.0     # Steering angle (radians, negative = right)
    speed: float = 0.0        # Speed command (m/s), only used if manages_speed=True


class BaseController(ABC):
    """
    Abstract base class for all control modes.
    
    To create a new mode:
    1. Subclass BaseController
    2. Set name, detection_mode, manages_speed
    3. Implement compute() and optionally reset() / update_params()
    4. Register in vision_controller's __init__
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Mode string identifier, e.g. 'MPC', 'POLY_LOOKAHEAD'.
        Must match the string used in GUI MODE_CONFIG and ROS topic.
        """
        ...

    @property
    def detection_mode(self) -> str:
        """
        Which track detection pipeline to use.
        Override if your controller needs a different detection mode
        than its own name (e.g. MPC needs POLY_LOOKAHEAD detection).
        Default: same as self.name.
        """
        return self.name

    @property
    def manages_speed(self) -> bool:
        """
        Whether this controller computes its own speed command.
        If False, vision_controller applies the generic turn-slowdown formula.
        If True, the speed from ControlOutput is used directly.
        Default: False.
        """
        return False

    @abstractmethod
    def compute(self, state: ControlState, dt: float) -> ControlOutput:
        """
        Compute steering (and optionally speed) for this frame.
        
        Args:
            state: Current control state (target, frame info, detection result)
            dt: Time since last frame (seconds)
            
        Returns:
            ControlOutput with steering (and speed if manages_speed=True)
        """
        ...

    def reset(self, initial_error: float = 0.0):
        """
        Called when switching to this mode or on autonomous start.
        Override to clear internal state (e.g. prev_error, integrators).
        Args:
            initial_error: Current error value (to seed D-term)
        """
        pass

    def update_params(self, params: Dict[str, Any]):
        """
        Called when GUI tuning parameters change.
        Override to accept parameter updates (e.g. kp, kd, steering_bias).
        
        Args:
            params: Dict of parameter names to values.
                    Controllers should ignore keys they don't recognize.
        """
        pass

    # --- Graceful degradation support ---
    # Subclasses should set self._last_good_output = output on every
    # successful compute().  When the path becomes invalid, call
    # self._degraded_output(dt) instead of returning hard zeros.
    _last_good_output: Optional[ControlOutput] = None

    def _degraded_output(self, dt: float) -> ControlOutput:
        """Return a decaying version of the last good output.

        Instead of snapping to zero steering/speed when the path is
        momentarily lost, this holds the last command while exponentially
        decaying speed toward zero over ~0.5 s.  If no previous output
        exists (first frame), returns safe zeros.
        """
        if self._last_good_output is None:
            return ControlOutput(steering=0.0, speed=0.0)
        decay = max(0.0, 1.0 - 2.0 * max(dt, 0.01))  # ~0.5s to zero
        decayed = ControlOutput(
            steering=self._last_good_output.steering * decay,
            speed=self._last_good_output.speed * decay,
        )
        # Update the stored output so repeated calls keep decaying
        self._last_good_output = decayed
        return decayed
