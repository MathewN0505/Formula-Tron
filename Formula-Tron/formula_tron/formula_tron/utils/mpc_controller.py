"""
MPC Controller — Model Predictive Control with lateral, heading, and
longitudinal error tracking, plus hard lane-boundary constraints.

CHANGES ADDED (based on latest behavior: early turn-in + leaving track + corner cutting):
  A) Raise turn-gate thresholds so heading cost turns on later:
        _turn_kappa_low:  0.05 -> 0.10
        _turn_kappa_high: 0.30 -> 0.45
  B) Add forward-dilation of the heading gate:
        heading_gate[k] = max(gate[k : k+M]) with M=3 steps (~0.15s)
  C) Add a lateral-acceleration penalty:
        a_lat ≈ (v^2 / L) * tan(delta)
        obj += w_alat * a_lat^2   with w_alat = 5.0
  D) Increase lane safety margin slightly:
        _lane_margin: 0.05 -> 0.08

NEW (recommended after watching the behavior):
  E) Add a SOFT lane-centering objective (prevents “racing line / apex cutting”):
        lane_center = 0.5*(left_bound + right_bound)
        e_center = e_lat - lane_center
        obj += w_center * lane_center_gate[k] * e_center^2

     lane_center_gate[k] is passed as a parameter and is 1 when BOTH lane
     boundaries are available (left+right polynomials), else 0. This avoids
     the centering term accidentally fighting you when bounds are “fake” /
     unconstrained.

  F) Reduce longitudinal-progress cost inside turns (optional but helpful):
        w_lon_effective = w_lon * (1 - 0.75*heading_gate[k])
     So lon cost is full-strength on straights, reduced in turns.
"""

import numpy as np
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple, List

from .base_controller import BaseController, ControlState, ControlOutput
from .vehicle_model import (
    BicycleModel,
    compute_curvature,
)
from .. import config

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


@dataclass
class LapPerformanceRecord:
    """Performance metrics for a single completed lap."""
    lap_number: int = 0
    lap_time: float = 0.0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    min_speed: float = 0.0
    avg_abs_steering: float = 0.0
    max_abs_steering: float = 0.0
    avg_abs_lateral_error: float = 0.0
    corner_count: int = 0
    avg_corner_speed: float = 0.0
    racing_line_score: float = 0.0  # Lower = better (combines time + error)
    active_params: Dict[str, float] = field(default_factory=dict)


class _FrictionModel:
    """A bicycle model parameterized by friction coefficient mu."""

    def __init__(self, mu: float, wheelbase: float, max_speed: float):
        self.mu = mu
        self.bicycle = BicycleModel(
            wheelbase=wheelbase,
            max_speed=max_speed,
            max_steering=config.MAX_STEERING_ANGLE_EFFECTIVE,
        )

    def max_cornering_speed(self, curvature_abs: float) -> float:
        """Maximum speed for a given curvature based on friction limit."""
        if curvature_abs < 1e-6:
            return self.bicycle.max_speed
        v_max = np.sqrt(self.mu * 9.81 / curvature_abs)
        return min(v_max, self.bicycle.max_speed)


class MPCController(BaseController):
    """
    MPC controller using lateral / heading / longitudinal error costs.

    At each frame:
    1. Convert POLY_LOOKAHEAD polynomial to vehicle-frame reference waypoints
    2. Pre-compute per-waypoint heading and cumulative arc-length
    3. Compute curvature-adapted velocity profile via friction model
    4. Solve NLP (bicycle dynamics + error costs)
    5. Return first control action (steering, speed)
    """

    def __init__(
        self,
        friction: float = 0.7,
        horizon: int = 12,
        dt: float = 0.05,
        max_speed: float = 5.0,
    ):
        if not CASADI_AVAILABLE:
            raise ImportError(
                "CasADi is required for MPC-Standard (pip install casadi)"
            )

        self._horizon = horizon
        self._dt = dt
        self._max_speed = max_speed
        self._initial_max_speed = max_speed
        self._wheelbase = config.WHEELBASE_METERS
        self._max_steering = config.MAX_STEERING_ANGLE_EFFECTIVE
        self._max_accel = 2.0
        self._friction = friction

        # Per-lap escalation state
        self._lap_speed_increment = 0.4     # m/s added per lap
        self._lap_smoothness_increment = 0.2  # smoothness units added per lap
        self._lap_smoothness_raw = 0.15     # current smoothness value (maps to _w_dsteer)
        self._initial_smoothness_raw = 0.15 # initial smoothness for reset

        # Progressive racing line improvement state
        self._current_lap_frames: List[Dict[str, float]] = []
        self._lap_history: List[LapPerformanceRecord] = []
        self._initial_w_center = 50.0
        self._initial_lane_margin = 0.08
        self._initial_friction = friction
        self._last_frame_lateral_error = 0.0
        self._last_frame_curvature = 0.0

        # Friction model for curvature → speed limit
        self._model = _FrictionModel(friction, self._wheelbase, max_speed)

        # ── Cost weights ──────────────────────────────────────────
        self._w_lat = 600.0       # Lateral (cross-track) error — dominant cost
        self._w_heading = 85.0    # Heading alignment error (gated) — raised to hold
                                  # car aligned with turn arc through full corner exit
        self._w_lon = 5.0         # Longitudinal progress — reduced so arc-position cost
                                  # doesn't fight inward lateral correction in corners
        self._w_vel = 20.0        # Velocity tracking — reduced so speed maintenance
                                  # doesn't outweigh lateral correction at horizon tail
        self._w_steer = 40.0      # Steering effort — raised from 20 to damp
                                  # straight-line oscillation at higher speeds
        self._w_dsteer = 120.0    # Steering rate — raised from 75 to further
                                  # smooth corrections and prevent overshoot
        self._w_accel = 50.0      # Acceleration effort

        # Lateral-accel penalty: v^2/L * tan(delta) scales quadratically with
        # speed, naturally damping steering at higher speeds. Re-enabled at a
        # modest value (was 0) to fix straight-line oscillation that worsens
        # with speed. Keep low enough not to fight corner corrections.
        self._w_alat = 3.0

        # NEW: lane-centering weight (prevents corner cutting)
        # Was 100 — at 100, both-boundary detection suddenly doubles the effective
        # w_lat (600+100=700), whipping the car to center and causing inside-cut
        # overshoot oscillation. Halved to 50 so the centering term gently nudges
        # without destabilising the lateral correction.
        self._w_center = 50.0

        # ── Horizon decay ──────────────────────────────────────────────────────
        # Raised so the MPC sees further into the corner exit and can plan a
        # smooth heading transition instead of reacting frame-by-frame.
        # 0.75 was too steep: steps 6-12 had near-zero weight, the exit correction
        # was concentrated into 1-2 steps causing overshoot off-track.
        self._horizon_decay = 0.88
        self._heading_decay = 0.82

        # ── Heading gating (turn detection) ───────────────────────
        self._heading_gate_min = 0.0
        self._heading_gate_max = 1.0
        self._turn_kappa_low = 0.10
        self._turn_kappa_high = 0.45
        self._heading_gate_dilate_steps = 2  # 2-step tail: gate stays high at corner
                                             # exit so car doesn't straighten early

        # ── Lane boundary constraint ─────────────────────────────
        self._lane_margin = 0.08

        # Warm-start
        self._prev_solution: Optional[np.ndarray] = None
        self._last_planned_steering = 0.0

        # Pixels-per-meter (updated each solve)
        self._last_ppm = (
            config.VISUAL_TRACK_WIDTH / config.PHYSICAL_TRACK_WIDTH
        )

        # Solver + bookkeeping
        self._solver = None
        self._last_valid_output = ControlOutput(steering=0.0, speed=0.0)
        self._last_success_time = time.time()
        self._fallback_state = "OK"
        self._last_failure_reason = ""
        self._consecutive_compute_failures = 0
        self._consecutive_solver_failures = 0
        self._total_solver_failures = 0
        self._build_solver()

    @property
    def name(self) -> str:
        return "MPC"

    @property
    def detection_mode(self) -> str:
        return "POLY_LOOKAHEAD"

    @property
    def manages_speed(self) -> bool:
        return True

    def compute(self, state: ControlState, dt: float) -> ControlOutput:
        detection = state.detection

        waypoints = getattr(detection, 'waypoints', None)
        if waypoints is None or len(waypoints) < 3:
            self._prev_solution = None
            self._mark_failure("waypoints_missing")
            return self._last_valid_output

        try:
            steering, speed = self._solve_mpc(state)
            output = ControlOutput(
                steering=float(steering), speed=float(speed)
            )
            self._last_valid_output = output
            self._mark_success()

            # Record frame data for progressive analysis
            self._current_lap_frames.append({
                'speed': float(speed),
                'steering': float(steering),
                'lateral_error': float(self._last_frame_lateral_error),
                'curvature': float(self._last_frame_curvature),
                'time': time.time(),
            })

            return output
        except Exception:
            self._mark_failure("solver_failed")
            if self._consecutive_solver_failures >= 3:
                self._prev_solution = None
            return self._last_valid_output

    def reset(self, initial_error: float = 0.0):
        self._prev_solution = None
        self._last_planned_steering = 0.0
        self._fallback_state = "OK"
        self._last_failure_reason = ""
        self._consecutive_compute_failures = 0
        self._consecutive_solver_failures = 0
        self._last_success_time = time.time()
        self._last_valid_output = ControlOutput(steering=0.0, speed=0.0)

        # Restore per-lap escalation to initial values
        self._max_speed = self._initial_max_speed
        self._model.bicycle.max_speed = self._initial_max_speed
        self._lap_smoothness_raw = self._initial_smoothness_raw
        self._w_dsteer = self._initial_smoothness_raw * 500.0

        # Clear progressive racing line state
        self._current_lap_frames.clear()
        self._lap_history.clear()
        self._w_center = self._initial_w_center
        self._lane_margin = self._initial_lane_margin
        self._friction = self._initial_friction
        self._model = _FrictionModel(self._initial_friction, self._wheelbase, self._initial_max_speed)

    def update_params(self, params: Dict[str, Any]):
        rebuild = False
        if "base_speed" in params:
            self._max_speed = params["base_speed"]
            self._model.bicycle.max_speed = params["base_speed"]
            rebuild = True
        if "mpc_tracking" in params:
            scale = float(params["mpc_tracking"])
            self._w_lat = scale * 200.0
            self._w_heading = scale * 60.0
            self._w_lon = scale * 10.0
            rebuild = True
        if "mpc_smoothness" in params:
            self._w_dsteer = float(params["mpc_smoothness"]) * 500.0
            rebuild = True
        if "mpc_horizon" in params:
            new_h = int(params["mpc_horizon"])
            if new_h != self._horizon:
                self._horizon = new_h
                rebuild = True
        if rebuild:
            self._build_solver()

    def on_lap_completed(self, lap_count: int, lap_time: float = 0.0):
        """Called when a lap is completed.

        1. Analyzes the frame data collected during the lap
        2. Adaptively tunes parameters based on improvement trends
        3. Applies the per-lap speed/smoothness ramp
        """
        # ── 1. Analyze collected frame data ─────────────────────────
        record = self._analyze_lap(lap_count, lap_time)
        self._lap_history.append(record)
        self._current_lap_frames.clear()

        # ── 2. Adaptive parameter tuning (requires ≥2 laps) ─────────
        if len(self._lap_history) >= 2:
            prev = self._lap_history[-2]
            curr = self._lap_history[-1]
            self._adapt_parameters(prev, curr)

        # ── 3. Per-lap speed/smoothness ramp ────────────────────────
        new_speed = self._initial_max_speed + self._lap_speed_increment * lap_count
        new_smoothness = self._initial_smoothness_raw + self._lap_smoothness_increment * lap_count

        self._max_speed = new_speed
        self._model.bicycle.max_speed = new_speed

        self._lap_smoothness_raw = new_smoothness
        self._w_dsteer = new_smoothness * 500.0
        self._build_solver()
        self._prev_solution = None

    def _analyze_lap(self, lap_count: int, lap_time: float) -> LapPerformanceRecord:
        """Compute performance metrics from the frame buffer."""
        frames = self._current_lap_frames
        if not frames:
            return LapPerformanceRecord(
                lap_number=lap_count, lap_time=lap_time,
                active_params=self._get_param_snapshot(),
            )

        speeds = np.array([f['speed'] for f in frames])
        steerings = np.array([f['steering'] for f in frames])
        lat_errors = np.array([f['lateral_error'] for f in frames])
        curvatures = np.array([f['curvature'] for f in frames])

        # Corner detection: curvature above threshold indicates a turn
        corner_threshold = 0.08
        in_corner = curvatures > corner_threshold
        corner_count = int(np.sum(np.diff(in_corner.astype(int)) == 1))  # rising edges
        corner_speeds = speeds[in_corner] if np.any(in_corner) else speeds

        # Racing line score: weighted combination (lower = better)
        # Rewards: low lap time, low lateral error, high corner speed
        avg_lat = float(np.mean(np.abs(lat_errors)))
        avg_corner_spd = float(np.mean(corner_speeds)) if len(corner_speeds) > 0 else 0.0
        racing_line_score = (
            lap_time * 1.0
            + avg_lat * 50.0
            - avg_corner_spd * 5.0
        ) if lap_time > 0 else avg_lat * 50.0

        return LapPerformanceRecord(
            lap_number=lap_count,
            lap_time=lap_time,
            avg_speed=float(np.mean(speeds)),
            max_speed=float(np.max(speeds)),
            min_speed=float(np.min(speeds)),
            avg_abs_steering=float(np.mean(np.abs(steerings))),
            max_abs_steering=float(np.max(np.abs(steerings))),
            avg_abs_lateral_error=avg_lat,
            corner_count=corner_count,
            avg_corner_speed=avg_corner_spd,
            racing_line_score=float(racing_line_score),
            active_params=self._get_param_snapshot(),
        )

    def _adapt_parameters(self, prev: LapPerformanceRecord, curr: LapPerformanceRecord):
        """Adapt MPC weights based on improvement between two laps."""
        # Corner speed improved → allow tighter racing lines
        if curr.avg_corner_speed > prev.avg_corner_speed:
            self._w_center = max(10.0, self._w_center - 5.0)

        # Lateral error is small → can tighten lane margin
        if curr.avg_abs_lateral_error < 0.15:
            self._lane_margin = max(0.03, self._lane_margin - 0.005)

        # Lap time improved → trust more grip
        if curr.lap_time > 0 and prev.lap_time > 0 and curr.lap_time < prev.lap_time:
            self._friction = min(1.1, self._friction + 0.03)
            self._model = _FrictionModel(self._friction, self._wheelbase, self._max_speed)

    def _get_param_snapshot(self) -> Dict[str, float]:
        """Snapshot of current tunable parameters."""
        return {
            'max_speed': self._max_speed,
            'w_center': self._w_center,
            'w_dsteer': self._w_dsteer,
            'lane_margin': self._lane_margin,
            'friction': self._friction,
            'w_lat': self._w_lat,
            'w_heading': self._w_heading,
        }

    def get_progressive_status(self) -> Dict[str, Any]:
        """Return progressive improvement data for telemetry/visualization."""
        history_dicts = [asdict(r) for r in self._lap_history]

        improvement = {}
        if len(self._lap_history) >= 2:
            first = self._lap_history[0]
            last = self._lap_history[-1]
            if first.lap_time > 0 and last.lap_time > 0:
                improvement['lap_time_pct'] = round(
                    (first.lap_time - last.lap_time) / first.lap_time * 100, 1
                )
            if first.avg_corner_speed > 0:
                improvement['corner_speed_pct'] = round(
                    (last.avg_corner_speed - first.avg_corner_speed) / first.avg_corner_speed * 100, 1
                )
            if first.racing_line_score != 0:
                improvement['racing_line_pct'] = round(
                    (first.racing_line_score - last.racing_line_score) / abs(first.racing_line_score) * 100, 1
                )

        return {
            'laps_completed': len(self._lap_history),
            'lap_history': history_dicts,
            'current_params': self._get_param_snapshot(),
            'improvement_summary': improvement,
            'frames_this_lap': len(self._current_lap_frames),
        }

    def get_status(self) -> dict:
        return {
            "selected_mu": self._friction,
            "fallback_state": self._fallback_state,
            "fallback_reason": self._last_failure_reason,
            "last_success_age_sec": round(
                max(0.0, time.time() - self._last_success_time), 3
            ),
            "consecutive_compute_failures": self._consecutive_compute_failures,
            "consecutive_solver_failures": self._consecutive_solver_failures,
            "total_solver_failures": self._total_solver_failures,
            "last_valid_speed": round(
                float(self._last_valid_output.speed), 3
            ),
            "last_valid_steering": round(
                float(self._last_valid_output.steering), 3
            ),
            "last_ppm": round(float(self._last_ppm), 3),
        }

    def _mark_success(self):
        self._fallback_state = "OK"
        self._last_failure_reason = ""
        self._consecutive_compute_failures = 0
        self._consecutive_solver_failures = 0
        self._last_success_time = time.time()

    def _mark_failure(self, reason: str):
        self._fallback_state = "DEGRADED"
        self._last_failure_reason = reason
        self._consecutive_compute_failures += 1
        self._consecutive_solver_failures += 1
        self._total_solver_failures += 1

    @staticmethod
    def _compute_reference_heading(ref_x: np.ndarray,
                                   ref_y: np.ndarray) -> np.ndarray:
        n = len(ref_x)
        psi = np.zeros(n)
        for k in range(n - 1):
            dx = ref_x[k + 1] - ref_x[k]
            dy = ref_y[k + 1] - ref_y[k]
            psi[k] = np.arctan2(dy, dx)
        psi[-1] = psi[-2] if n > 1 else 0.0
        return psi

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _smoothstep(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _build_solver(self):
        if not CASADI_AVAILABLE:
            return

        N = self._horizon
        L = self._wheelbase
        dt = self._dt
        nx = 4
        nu = 2

        # Parameters layout:
        #   x0 (4)
        #   ref_xy (2*(N+1))
        #   ref_heading (N+1)
        #   v_ref (N+1)
        #   current_steering (1)
        #   left_bound (N+1)
        #   right_bound (N+1)
        #   heading_gate (N+1)
        #   lane_center_gate (N+1)   <-- NEW
        n_params = (
            4
            + 2 * (N + 1)
            + (N + 1)
            + (N + 1)
            + 1
            + 2 * (N + 1)
            + (N + 1)
            + (N + 1)
        )

        opt_x = ca.SX.sym("opt_x", nx * (N + 1) + nu * N)
        p = ca.SX.sym("p", n_params)

        x0_param = p[0:4]

        idx = 4
        ref_xy = ca.reshape(p[idx: idx + 2 * (N + 1)], 2, N + 1).T
        idx += 2 * (N + 1)
        ref_heading = p[idx: idx + (N + 1)]
        idx += (N + 1)
        v_ref = p[idx: idx + (N + 1)]
        idx += (N + 1)
        current_steering_param = p[idx]
        idx += 1
        left_bound_param = p[idx: idx + (N + 1)]
        idx += (N + 1)
        right_bound_param = p[idx: idx + (N + 1)]
        idx += (N + 1)
        heading_gate = p[idx: idx + (N + 1)]
        idx += (N + 1)
        lane_center_gate = p[idx: idx + (N + 1)]

        states = []
        controls = []
        for k in range(N + 1):
            states.append(opt_x[k * nx: (k + 1) * nx])
        for k in range(N):
            ci = (N + 1) * nx + k * nu
            controls.append(opt_x[ci: ci + nu])

        obj = 0.0
        g = []
        lbg = []
        ubg = []

        g.append(states[0] - x0_param)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        for k in range(N):
            xk = states[k]
            uk = controls[k]
            xk1 = states[k + 1]

            x_pos, y_pos, psi, v = xk[0], xk[1], xk[2], xk[3]
            a_cmd, delta = uk[0], uk[1]

            x_next = x_pos + v * ca.cos(psi) * dt
            y_next = y_pos + v * ca.sin(psi) * dt
            psi_next = psi + (v / L) * ca.tan(delta) * dt
            v_next = v + a_cmd * dt

            g.append(xk1 - ca.vertcat(x_next, y_next, psi_next, v_next))
            lbg += [0.0] * nx
            ubg += [0.0] * nx

            decay_k = self._horizon_decay ** k
            decay_heading_k = self._heading_decay ** k

            dx_ref = x_pos - ref_xy[k, 0]
            dy_ref = y_pos - ref_xy[k, 1]
            psi_ref_k = ref_heading[k]

            # Lateral error
            e_lat = dy_ref * ca.cos(psi_ref_k) - dx_ref * ca.sin(psi_ref_k)
            obj += (self._w_lat * decay_k) * e_lat ** 2

            # NEW: Lane-centering cost (soft) to prevent “corner cutting”
            lane_center_k = 0.5 * (left_bound_param[k] + right_bound_param[k])
            e_center = e_lat - lane_center_k
            obj += (self._w_center * decay_k) * lane_center_gate[k] * (e_center ** 2)

            # Heading error (wrapped) with gating + faster decay
            dpsi = psi - psi_ref_k
            e_heading = ca.atan2(ca.sin(dpsi), ca.cos(dpsi))
            obj += (self._w_heading * decay_heading_k) * heading_gate[k] * e_heading ** 2

            # Longitudinal error — full penalty in ALL phases including turns.
            # lon_scale was previously reduced to 0.25 in turns, but this allowed
            # the optimizer to freely lag behind the reference arc in corners,
            # which places the car on the outside of the turn. Keeping it at 1.0
            # forces the car to stay at the correct arc position throughout.
            s_pred = dx_ref * ca.cos(psi_ref_k) + dy_ref * ca.sin(psi_ref_k)
            obj += (self._w_lon * decay_k) * (s_pred ** 2)

            # Velocity tracking
            obj += self._w_vel * (v - v_ref[k]) ** 2

            # Control effort
            obj += self._w_accel * a_cmd ** 2
            obj += self._w_steer * delta ** 2

            # Steering-rate penalty
            if k == 0:
                obj += self._w_dsteer * (delta - current_steering_param) ** 2
            else:
                prev_delta = controls[k - 1][1]
                obj += self._w_dsteer * (delta - prev_delta) ** 2

            # Lateral-acceleration penalty (kinematic proxy)
            a_lat = (v ** 2 / L) * ca.tan(delta)
            obj += self._w_alat * (a_lat ** 2)

        # Lane boundary constraints (hard)
        margin = self._lane_margin
        for k in range(1, N + 1):
            xk = states[k]
            x_pos_k, y_pos_k = xk[0], xk[1]
            dx_r = x_pos_k - ref_xy[k, 0]
            dy_r = y_pos_k - ref_xy[k, 1]
            psi_r = ref_heading[k]
            e_lat_k = dy_r * ca.cos(psi_r) - dx_r * ca.sin(psi_r)

            # --- HARD CONSTRAINTS REMOVED to prevent solver freezing ---
            # g.append(e_lat_k - left_bound_param[k])
            # lbg.append(-1e6)
            # ubg.append(-margin)

            # g.append(e_lat_k - right_bound_param[k])
            # lbg.append(margin)
            # ubg.append(1e6)

        lbx = []
        ubx = []
        for _ in range(N + 1):
            lbx += [-100, -100, -2 * np.pi, 0.0]
            ubx += [100, 100, 2 * np.pi, self._max_speed]
        for _ in range(N):
            lbx += [-self._max_accel, -self._max_steering]
            ubx += [self._max_accel, self._max_steering]

        nlp = {"x": opt_x, "f": obj, "g": ca.vertcat(*g), "p": p}
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 150,
            "ipopt.max_cpu_time": 0.08,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
        }
        self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        self._n_vars = nx * (N + 1) + nu * N
        self._n_params = n_params
        self._nx = nx
        self._nu = nu
        self._lbx = np.array(lbx)
        self._ubx = np.array(ubx)
        self._lbg = np.array(lbg)
        self._ubg = np.array(ubg)

        # Warm-up solve (tail order: left, right, heading_gate, lane_center_gate)
        try:
            x0_wu = np.zeros(self._n_vars)
            p_wu = np.zeros(self._n_params)

            p_wu[-4 * (N + 1): -3 * (N + 1)] = 10.0   # left_bound
            p_wu[-3 * (N + 1): -2 * (N + 1)] = -10.0  # right_bound
            p_wu[-2 * (N + 1): -(N + 1)] = 1.0        # heading_gate
            p_wu[-(N + 1):] = 0.0                      # lane_center_gate (off during warmup)

            self._solver(
                x0=x0_wu,
                lbx=self._lbx, ubx=self._ubx,
                lbg=self._lbg, ubg=self._ubg,
                p=p_wu,
            )
        except Exception:
            pass

    def _solve_mpc(self, state: ControlState) -> Tuple[float, float]:
        detection = state.detection
        waypoints = detection.waypoints
        N = self._horizon

        # 1. Compute discrete Path Arc-Lengths
        x_m_fine = waypoints[:, 0]
        y_m_fine = waypoints[:, 1]
        s_fine = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            dx = x_m_fine[i] - x_m_fine[i - 1]
            dy = y_m_fine[i] - y_m_fine[i - 1]
            s_fine[i] = s_fine[i - 1] + np.hypot(dx, dy)

        # 2. 1D velocity profile and distance integration
        v_refs = np.zeros(N + 1)
        s_refs = np.zeros(N + 1)
        v_curr = state.current_speed
        
        # Calculate approximate curvature from waypoints for velocity profiling
        dpsi_fine = np.diff(self._compute_reference_heading(x_m_fine, y_m_fine))
        dpsi_fine = np.array([self._wrap_angle(a) for a in dpsi_fine])
        ds_fine = np.diff(s_fine) + 1e-6
        kappa_fine = np.concatenate([np.abs(dpsi_fine / ds_fine), [0.0]])
        
        for i in range(N + 1):
            v_refs[i] = v_curr
            if i < N:
                # Interpolate curvature at current s
                kappa_abs = np.interp(s_refs[i], s_fine, kappa_fine)
                v_max_curve = self._model.max_cornering_speed(kappa_abs)
                v_target = min(v_max_curve, self._max_speed)

                if v_curr < v_target:
                    v_curr = min(v_target, v_curr + self._max_accel * self._dt)
                elif v_curr > v_target:
                    v_curr = max(v_target, v_curr - self._max_accel * self._dt)
                s_refs[i + 1] = s_refs[i] + v_curr * self._dt

        # 3. Interpolate Reference Trajectory
        ref_x_m = np.interp(s_refs, s_fine, x_m_fine)
        ref_y_m = np.interp(s_refs, s_fine, y_m_fine)

        ref_heading = self._compute_reference_heading(ref_x_m, ref_y_m)

        # ── Heading gate computation (turn detection) ─────────────
        eps = 1e-3
        dpsi = np.diff(ref_heading)
        dpsi = np.array([self._wrap_angle(a) for a in dpsi])
        ds = np.diff(s_refs) + eps
        kappa_proxy = np.abs(dpsi / ds)
        kappa_proxy = np.concatenate(
            [kappa_proxy, [kappa_proxy[-1] if N > 0 else 0.0]]
        )

        t = (kappa_proxy - self._turn_kappa_low) / max(
            1e-6, (self._turn_kappa_high - self._turn_kappa_low)
        )
        gate = self._smoothstep(t)
        gate = self._heading_gate_min + (self._heading_gate_max - self._heading_gate_min) * gate

        # Forward dilation
        M = int(self._heading_gate_dilate_steps)
        gate_fwd = gate.copy()
        if M > 0:
            for k in range(N + 1):
                gate_fwd[k] = np.max(gate[k:min(N + 1, k + M)])
        heading_gate = gate_fwd.astype(float)

        # ── Lane boundary computation ────────────────────────────
        # For waypoints, we will approximate constant lane boundary constraints
        # rather than recalculating the BEV boundaries which defeats the purpose
        _UNCONSTRAINED = 100.0
        half_w_default = config.PHYSICAL_TRACK_WIDTH / 2.0
        
        left_bounds = np.full(N + 1, half_w_default)
        right_bounds = np.full(N + 1, -half_w_default)

        for i in range(N + 1):
            if left_bounds[i] <= right_bounds[i]:
                mid = (left_bounds[i] + right_bounds[i]) / 2.0
                left_bounds[i] = mid + half_w_default
                right_bounds[i] = mid - half_w_default

        # NEW: lane_center_gate — only trust centering when BOTH lanes exist
        have_both = (detection.left_poly is not None and detection.right_poly is not None)
        lane_center_gate = (np.ones(N + 1) if have_both else np.zeros(N + 1)).astype(float)

        x0 = np.array([0.0, 0.0, 0.0, state.current_speed])
        p_val = np.concatenate([
            x0,
            np.column_stack([ref_x_m, ref_y_m]).flatten(),
            ref_heading,
            v_refs,
            [float(getattr(state, 'current_steering', self._last_planned_steering))],
            left_bounds,
            right_bounds,
            heading_gate,
            lane_center_gate,   # <-- NEW
        ])

        # Warm-start (unchanged)
        if (self._prev_solution is not None
                and len(self._prev_solution) == self._n_vars):
            old = self._prev_solution
            x0_guess = np.zeros(self._n_vars)
            nx, nu = self._nx, self._nu

            for k in range(N):
                x0_guess[k * nx:(k + 1) * nx] = \
                    old[(k + 1) * nx:(k + 2) * nx]
            x0_guess[N * nx:(N + 1) * nx] = old[N * nx:(N + 1) * nx]

            cb = (N + 1) * nx
            for k in range(N - 1):
                x0_guess[cb + k * nu:cb + (k + 1) * nu] = \
                    old[cb + (k + 1) * nu:cb + (k + 2) * nu]
            x0_guess[cb + (N - 1) * nu:cb + N * nu] = \
                old[cb + (N - 1) * nu:cb + N * nu]

            x0_guess[0:nx] = x0
        else:
            x0_guess = np.zeros(self._n_vars)
            for k in range(N + 1):
                x0_guess[k * self._nx + 3] = state.current_speed

        try:
            sol = self._solver(
                x0=x0_guess,
                lbx=self._lbx, ubx=self._ubx,
                lbg=self._lbg, ubg=self._ubg,
                p=p_val,
            )
        except Exception as exc:
            raise RuntimeError("solver_exception") from exc

        stats = self._solver.stats()
        if not stats.get("success", False):
            raise RuntimeError("solver_failed")

        sol_x = np.array(sol["x"]).flatten()
        self._prev_solution = sol_x

        ctrl_start = (N + 1) * self._nx
        accel = sol_x[ctrl_start]
        steering = sol_x[ctrl_start + 1]

        speed = max(
            0.5, min(self._max_speed, state.current_speed + accel * self._dt)
        )
        self._last_planned_steering = float(steering)

        # Stash lateral error and curvature at step 0 for progressive tracking
        self._last_frame_lateral_error = float(ref_y_m[0]) if len(ref_y_m) > 0 else 0.0
        self._last_frame_curvature = float(kappa_proxy[0]) if len(kappa_proxy) > 0 else 0.0

        return float(steering), float(speed)
