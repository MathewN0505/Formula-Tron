"""
Kinematic Bicycle Model for Formula-Tron MPC controllers.

Shared by CiMPCC, MPC-CEM, and LLA-MPC. Provides single-step prediction,
multi-step rollout, and curvature/speed-profiling utilities.

Reference: Standard kinematic bicycle model, validated for F1TENTH
at speeds under 5 m/s where tire slip is negligible.
"""

import numpy as np
from typing import Tuple, Optional

from .. import config

# Gravity constant (m/s^2)
_GRAVITY = 9.81


class BicycleModel:
    """
    Kinematic bicycle model for a rear-axle-referenced vehicle.

    State:   [x, y, theta, v]   (position, heading, speed)
    Control: [accel, steering]  (longitudinal accel, front-wheel steering angle)

    Dynamics (Euler integration):
        x'     = x + v * cos(theta) * dt
        y'     = y + v * sin(theta) * dt
        theta' = theta + (v / L) * tan(delta) * dt
        v'     = clamp(v + a * dt, 0, v_max)
    """

    def __init__(
        self,
        wheelbase: float = config.WHEELBASE_METERS,
        max_steering: float = config.MAX_STEERING_ANGLE,
        max_speed: float = 5.0,
        max_accel: float = 2.0,
    ):
        self.wheelbase = wheelbase
        self.max_steering = max_steering
        self.max_speed = max_speed
        self.max_accel = max_accel

    def predict(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Single-step state prediction.

        Args:
            state:   [x, y, theta, v]  (4,)
            control: [accel, steering]  (2,)
            dt:      time step in seconds

        Returns:
            next_state: [x, y, theta, v]  (4,)
        """
        x, y, theta, v = state
        a, delta = control

        # Clamp inputs
        a = float(np.clip(a, -self.max_accel, self.max_accel))
        delta = float(np.clip(delta, -self.max_steering, self.max_steering))

        # Euler integration
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        theta_next = theta + (v / self.wheelbase) * np.tan(delta) * dt
        v_next = np.clip(v + a * dt, 0.0, self.max_speed)

        return np.array([x_next, y_next, theta_next, v_next], dtype=np.float64)

    def rollout(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Multi-step forward simulation.

        Args:
            initial_state: [x, y, theta, v]  (4,)
            controls:      [[a, delta], ...]  (N, 2)
            dt:            time step in seconds

        Returns:
            trajectory: [[x, y, theta, v], ...]  (N+1, 4)
                        First row is the initial state.
        """
        n_steps = len(controls)
        trajectory = np.zeros((n_steps + 1, 4), dtype=np.float64)
        trajectory[0] = initial_state

        for k in range(n_steps):
            trajectory[k + 1] = self.predict(trajectory[k], controls[k], dt)

        return trajectory

    def rollout_batch(
        self,
        initial_state: np.ndarray,
        controls_batch: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Vectorized multi-step rollout for many control sequences at once.

        Args:
            initial_state:  [x, y, theta, v]        (4,)
            controls_batch: [[[a, delta], ...], ...]  (B, H, 2)
            dt:             time step in seconds

        Returns:
            trajectories: (B, H+1, 4)
        """
        b, h, _ = controls_batch.shape
        trajs = np.zeros((b, h + 1, 4), dtype=np.float64)
        trajs[:, 0, :] = initial_state

        for k in range(h):
            x = trajs[:, k, 0]
            y = trajs[:, k, 1]
            theta = trajs[:, k, 2]
            v = trajs[:, k, 3]

            a_raw = controls_batch[:, k, 0]
            delta_raw = controls_batch[:, k, 1]

            a = np.clip(a_raw, -self.max_accel, self.max_accel)
            delta = np.clip(delta_raw, -self.max_steering, self.max_steering)

            trajs[:, k + 1, 0] = x + v * np.cos(theta) * dt
            trajs[:, k + 1, 1] = y + v * np.sin(theta) * dt
            trajs[:, k + 1, 2] = theta + (v / self.wheelbase) * np.tan(delta) * dt
            trajs[:, k + 1, 3] = np.clip(v + a * dt, 0.0, self.max_speed)

        return trajs


# ---------------------------------------------------------------------------
# Curvature and speed-profiling utilities
# ---------------------------------------------------------------------------

def compute_curvature(poly_coeffs: np.ndarray, y: float = 0.0) -> float:
    """
    Compute curvature of the polynomial x = a*y^2 + b*y + c.

    For a second-degree polynomial, curvature at point y is:
        kappa = 2a / (1 + (2a*y + b)^2)^(3/2)

    At y = 0 (car position): kappa ≈ 2a  (when slope b is small).

    Args:
        poly_coeffs: [a, b, c] polynomial coefficients
        y:           evaluation point (default 0 = car position)

    Returns:
        Signed curvature (1/m in BEV-pixel space).
    """
    if poly_coeffs is None or len(poly_coeffs) < 3:
        return 0.0

    a, b, _ = poly_coeffs[0], poly_coeffs[1], poly_coeffs[2]
    dx_dy = 2.0 * a * y + b
    d2x_dy2 = 2.0 * a
    denom = (1.0 + dx_dy ** 2) ** 1.5

    if abs(denom) < 1e-9:
        return 0.0

    return d2x_dy2 / denom


def curvature_to_max_speed(
    kappa: float,
    v_max: float = 5.0,
    mu: float = 0.7,
    kappa_scale: float = 1.0,
) -> float:
    """
    Map curvature to a maximum safe speed.

    Uses lateral-acceleration limit:  v_max_curve = sqrt(mu * g / |kappa|)
    but kappa is in BEV-pixel space, so kappa_scale converts to meters.

    A simpler linear mapping is also blended for robustness:
        v_ref = v_max * (1 - alpha * |kappa_scaled|)

    Args:
        kappa:       curvature value (BEV-pixel space)
        v_max:       maximum straight-line speed (m/s)
        mu:          friction coefficient
        kappa_scale: pixels-per-meter conversion (set from BEV track width)

    Returns:
        Reference speed (m/s), clamped to [0.5, v_max].
    """
    kappa_m = abs(kappa) * kappa_scale
    if kappa_m < 1e-6:
        return v_max

    # Physics-based limit
    v_phys = np.sqrt(mu * _GRAVITY / kappa_m)

    # Linear fallback (more conservative)
    alpha = 0.5
    v_linear = v_max * max(0.0, 1.0 - alpha * kappa_m)

    # Use the more conservative of the two
    v_ref = min(v_phys, v_linear, v_max)
    return float(np.clip(v_ref, 0.5, v_max))


def generate_reference_from_poly(
    poly_coeffs: np.ndarray,
    bev_height: int,
    n_points: int = 15,
) -> np.ndarray:
    """
    Generate reference waypoints in BEV space from polynomial coefficients.

    The polynomial x = a*y^2 + b*y + c maps BEV-y (rows, top=far)
    to BEV-x (columns). We sample n_points from bottom to top.

    Args:
        poly_coeffs: [a, b, c]
        bev_height:  height of BEV image
        n_points:    number of reference points

    Returns:
        ref_points: (n_points, 2) array of [x_bev, y_bev]
    """
    if poly_coeffs is None or len(poly_coeffs) < 3:
        return np.zeros((n_points, 2), dtype=np.float64)

    a, b, c = poly_coeffs
    y_vals = np.linspace(bev_height - 1, 0, n_points)
    x_vals = a * y_vals ** 2 + b * y_vals + c

    return np.column_stack((x_vals, y_vals))
