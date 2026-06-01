#!/usr/bin/env python3
"""
Telemetry Module - Core data structures and collector for the telemetry system.

This module provides mode-agnostic telemetry collection that works for:
- Vision-based control (current)
- MPC control (future: LLA-MPC, CiMPCC, MPC-CEM)


The extensible mode_data dictionary allows each control mode to add its own
specific metrics without changing the core data structure.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class TelemetryRecord:
    """
    Core telemetry record - mode-agnostic data structure.
    
    All control modes populate the core fields. Mode-specific data
    goes into the mode_data dictionary for extensibility.
    """
    # Timing
    timestamp: float = 0.0          # Unix time with ms precision
    frame_number: int = 0           # Sequential frame count
    
    # Control Mode
    control_mode: str = "UNKNOWN"   # "VISION", "MPC", "POLY_LOOKAHEAD", etc.
    
    # Commands (what we told the car)
    speed_cmd: float = 0.0          # m/s
    steering_cmd: float = 0.0       # radians
    
    # Feedback (what the car actually did)
    speed_actual: float = 0.0       # From /odom
    imu_accel_x: float = 0.0        # From IMU
    imu_accel_y: float = 0.0
    imu_yaw_rate: float = 0.0       # Angular velocity (rad/s)
    
    # Session Data
    lap_number: int = 0
    lap_time_current: float = 0.0   # Seconds into current lap
    lap_time_last: float = 0.0      # Last completed lap time
    lap_time_best: float = float('inf')
    safety_state: str = "OK"        # "OK", "WARNING", "STOPPED"
    
    # Mode-Specific (extensible dictionary)
    mode_data: Dict[str, Any] = field(default_factory=dict)
    
    # Processing
    processing_time_ms: float = 0.0  # Frame processing latency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_mode_value(self, key: str, default: Any = None) -> Any:
        """Safely get a value from mode_data."""
        return self.mode_data.get(key, default)


class TelemetryCollector:
    """
    Collects and manages telemetry data.
    
    Features:
    - Circular buffer for real-time plotting (last N seconds)
    - Thread-safe data access
    - Session statistics tracking
    - Lap time recording
    """
    
    def __init__(self, buffer_size: int = 1800):
        """
        Initialize the telemetry collector.
        
        Args:
            buffer_size: Number of records to keep in memory (default: 1800 = 60s at 30Hz)
        """
        self.buffer_size = buffer_size
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        
        # Session tracking
        self._session_active = False
        self._session_start_time: Optional[float] = None
        self._frame_count = 0
        
        # Lap tracking
        self._lap_times: List[Tuple[int, float]] = []  # (lap_number, lap_time)
        self._current_lap = 0
        
        # Mode distribution tracking
        self._mode_counts: Dict[str, int] = {}
        
        # Callbacks for real-time updates
        self._on_record_callbacks: List[callable] = []
        self._on_lap_callbacks: List[callable] = []
    
    def record(self, data: TelemetryRecord) -> None:
        """
        Record a telemetry sample.
        
        Args:
            data: TelemetryRecord to store
        """
        with self._lock:
            # Update frame number
            self._frame_count += 1
            data.frame_number = self._frame_count
            
            # Add timestamp if not set
            if data.timestamp == 0.0:
                data.timestamp = time.time()
            
            # Track mode distribution
            mode = data.control_mode
            if mode in self._mode_counts:
                self._mode_counts[mode] += 1
            else:
                self._mode_counts[mode] = 1
            
            # Check for new lap
            if data.lap_number > self._current_lap and data.lap_time_last > 0:
                self._lap_times.append((data.lap_number, data.lap_time_last))
                self._current_lap = data.lap_number
                # Notify lap callbacks
                for callback in self._on_lap_callbacks:
                    try:
                        callback(data.lap_number, data.lap_time_last)
                    except Exception:
                        pass
            
            # Store record
            self._buffer.append(data)
        
        # Notify record callbacks (outside lock for performance)
        for callback in self._on_record_callbacks:
            try:
                callback(data)
            except Exception:
                pass
    
    def get_recent(self, seconds: float = 60.0) -> List[TelemetryRecord]:
        """
        Get records from the last N seconds.
        
        Args:
            seconds: How far back to look
            
        Returns:
            List of TelemetryRecord objects
        """
        cutoff = time.time() - seconds
        with self._lock:
            return [r for r in self._buffer if r.timestamp >= cutoff]
    
    def get_all(self) -> List[TelemetryRecord]:
        """Get all records in buffer."""
        with self._lock:
            return list(self._buffer)
    
    def get_latest(self) -> Optional[TelemetryRecord]:
        """Get the most recent record."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None
    
    def get_lap_times(self) -> List[Tuple[int, float]]:
        """
        Get all recorded lap times.
        
        Returns:
            List of (lap_number, lap_time) tuples
        """
        with self._lock:
            return list(self._lap_times)
    
    def get_mode_distribution(self) -> Dict[str, float]:
        """
        Get percentage distribution of control modes.
        
        Returns:
            Dictionary mapping mode name to percentage (0-100)
        """
        with self._lock:
            total = sum(self._mode_counts.values())
            if total == 0:
                return {}
            return {mode: (count / total) * 100 
                    for mode, count in self._mode_counts.items()}
    
    def get_detection_mode_distribution(self) -> Dict[str, float]:
        """
        Get percentage distribution of detection modes (Vision-specific).
        
        Returns:
            Dictionary mapping detection mode to percentage (0-100)
        """
        detection_counts: Dict[str, int] = {}
        with self._lock:
            for record in self._buffer:
                det_mode = record.get_mode_value("detection_mode", "UNKNOWN")
                if det_mode in detection_counts:
                    detection_counts[det_mode] += 1
                else:
                    detection_counts[det_mode] = 1
        
        total = sum(detection_counts.values())
        if total == 0:
            return {}
        return {mode: (count / total) * 100 
                for mode, count in detection_counts.items()}
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics summary.
        
        Returns:
            Dictionary with session stats
        """
        with self._lock:
            if not self._session_start_time:
                duration = 0.0
            else:
                duration = time.time() - self._session_start_time
            
            lap_times_list = [t for _, t in self._lap_times]
            
            return {
                "session_active": self._session_active,
                "duration_sec": duration,
                "total_frames": self._frame_count,
                "laps_completed": len(self._lap_times),
                "best_lap_time": min(lap_times_list) if lap_times_list else None,
                "avg_lap_time": np.mean(lap_times_list) if lap_times_list else None,
                "mode_distribution": self.get_mode_distribution(),
            }
    
    def start_session(self) -> None:
        """Start a new telemetry session."""
        with self._lock:
            self._session_active = True
            self._session_start_time = time.time()
            self._frame_count = 0
            self._buffer.clear()
            self._lap_times.clear()
            self._mode_counts.clear()
            self._current_lap = 0
    
    def end_session(self) -> Dict[str, Any]:
        """
        End the current session.
        
        Returns:
            Session statistics summary
        """
        stats = self.get_session_stats()
        with self._lock:
            self._session_active = False
        return stats
    
    def is_recording(self) -> bool:
        """Check if a session is active."""
        with self._lock:
            return self._session_active
    
    def get_frame_count(self) -> int:
        """Get total frames recorded."""
        with self._lock:
            return self._frame_count
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        with self._lock:
            if self._session_start_time:
                return time.time() - self._session_start_time
            return 0.0
    
    # Callback registration
    def on_record(self, callback: callable) -> None:
        """Register a callback for new records."""
        self._on_record_callbacks.append(callback)
    
    def on_lap(self, callback: callable) -> None:
        """Register a callback for new laps."""
        self._on_lap_callbacks.append(callback)
    
    def remove_callbacks(self) -> None:
        """Remove all callbacks."""
        self._on_record_callbacks.clear()
        self._on_lap_callbacks.clear()


# Convenience function for creating Vision mode data
def create_vision_mode_data(
    detection_mode: str = "LOST",
    track_error_px: float = 0.0,
    target_x: float = 0.0,
    left_poly: Optional[List[float]] = None,
    right_poly: Optional[List[float]] = None,
    center_poly: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Helper to create the data dictionary for Vision modes.
    
    Args:
        detection_mode: Strategy used
        track_error_px: Track error in pixels
        target_x: Target steering point (pixel x)
        left_poly: Left track polynomial coefficients
        right_poly: Right track polynomial coefficients
        center_poly: Center track polynomial coefficients
        
    Returns:
        Dictionary for TelemetryRecord.mode_data
    """
    data = {
        "mode": "VISION",
        "detection_mode": detection_mode,
        "track_error_px": track_error_px,
        "target_x": target_x,
    }
    if left_poly is not None:
        data["left_poly"] = list(left_poly)
    if right_poly is not None:
        data["right_poly"] = list(right_poly)
    if center_poly is not None:
        data["center_poly"] = list(center_poly)
    return data


# Convenience function for creating MPC mode data
def create_mpc_mode_data(
    solver_time_ms: float = 0.0,
    cost_value: float = 0.0,
    horizon_length: int = 20,
    friction_estimate: Optional[float] = None,
    contouring_error: Optional[float] = None,
    lag_error: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create mode_data dictionary for MPC control modes.
    
    Args:
        solver_time_ms: Solver computation time
        cost_value: Objective function result
        horizon_length: Prediction horizon
        friction_estimate: For LLA-MPC
        contouring_error: For CiMPCC
        lag_error: For CiMPCC
        
    Returns:
        Dictionary for TelemetryRecord.mode_data
    """
    data = {
        "solver_time_ms": solver_time_ms,
        "cost_value": cost_value,
        "horizon_length": horizon_length,
    }
    if friction_estimate is not None:
        data["friction_estimate"] = friction_estimate
    if contouring_error is not None:
        data["contouring_error"] = contouring_error
    if lag_error is not None:
        data["lag_error"] = lag_error
    return data


