#!/usr/bin/env python3
"""
Telemetry Logger - CSV and JSON file I/O for telemetry data.

Handles:
- Session folder creation with timestamps
- CSV export (flat format for Excel/Sheets)
- JSON export (nested format for code processing)
- Lap times export
- Summary report generation
- Chart export (PNG)
"""

import os
import csv
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .telemetry import TelemetryRecord, TelemetryCollector


class TelemetryLogger:
    """
    Handles file I/O for telemetry data.
    
    Features:
    - Creates session folders with timestamp names
    - Writes CSV for Excel analysis
    - Writes JSON for code processing
    - Batched writes for performance
    - Thread-safe operations
    """
    
    def __init__(self, base_path: Optional[str] = None, batch_size: int = 100):
        """
        Initialize the telemetry logger.
        
        Args:
            base_path: Base directory for telemetry data. 
                       Defaults to telemetry_data/ in project root.
            batch_size: Number of records to buffer before writing
        """
        if base_path is None:
            # Default to project root / telemetry_data
            project_root = Path(__file__).parent.parent.parent.parent
            base_path = project_root / "telemetry_data"
        
        self.base_path = Path(base_path)
        self.sessions_path = self.base_path / "sessions"
        self.exports_path = self.base_path / "exports"
        self.batch_size = batch_size
        
        # Current session
        self._session_path: Optional[Path] = None
        self._session_name: Optional[str] = None
        self._csv_file = None
        self._csv_writer = None
        self._write_buffer: List[TelemetryRecord] = []
        self._lock = threading.Lock()
        self._is_recording = False
        
        # Session metadata
        self._session_start_time: Optional[datetime] = None
        self._all_records: List[TelemetryRecord] = []
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.sessions_path.mkdir(parents=True, exist_ok=True)
        self.exports_path.mkdir(parents=True, exist_ok=True)
    
    def start_session(self, session_name: Optional[str] = None) -> str:
        """
        Start a new recording session.
        
        Args:
            session_name: Optional custom name. Defaults to timestamp.
            
        Returns:
            Session folder name
        """
        with self._lock:
            if self._is_recording:
                self.end_session()
            
            # Generate session name
            self._session_start_time = datetime.now()
            if session_name is None:
                session_name = self._session_start_time.strftime("%Y-%m-%d_%H-%M-%S")
            
            self._session_name = session_name
            self._session_path = self.sessions_path / session_name
            self._session_path.mkdir(parents=True, exist_ok=True)
            
            # Open CSV file
            csv_path = self._session_path / "telemetry.csv"
            self._csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
            self._csv_writer = csv.writer(self._csv_file)
            
            # Write CSV header
            self._csv_writer.writerow([
                'timestamp', 'frame_number', 'control_mode',
                'speed_cmd', 'steering_cmd', 'speed_actual',
                'imu_accel_x', 'imu_accel_y', 'imu_yaw_rate',
                'lap_number', 'lap_time_current', 'lap_time_last', 'lap_time_best',
                'safety_state', 'processing_time_ms', 'mode_data'
            ])
            
            self._write_buffer.clear()
            self._all_records.clear()
            self._is_recording = True
            
            return session_name
    
    def write_record(self, record: TelemetryRecord) -> None:
        """
        Write a telemetry record.
        
        Records are buffered and written in batches for performance.
        
        Args:
            record: TelemetryRecord to write
        """
        if not self._is_recording:
            return
        
        with self._lock:
            self._write_buffer.append(record)
            self._all_records.append(record)
            
            # Batch write
            if len(self._write_buffer) >= self.batch_size:
                self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Write buffered records to CSV file."""
        if not self._csv_writer or not self._write_buffer:
            return
        
        for record in self._write_buffer:
            # Serialize mode_data to JSON string
            mode_data_str = json.dumps(record.mode_data) if record.mode_data else "{}"
            
            # Handle infinity values
            lap_time_best = record.lap_time_best
            if lap_time_best == float('inf'):
                lap_time_best = -1.0
            
            self._csv_writer.writerow([
                f"{record.timestamp:.6f}",
                record.frame_number,
                record.control_mode,
                f"{record.speed_cmd:.4f}",
                f"{record.steering_cmd:.4f}",
                f"{record.speed_actual:.4f}",
                f"{record.imu_accel_x:.4f}",
                f"{record.imu_accel_y:.4f}",
                f"{record.imu_yaw_rate:.4f}",
                record.lap_number,
                f"{record.lap_time_current:.3f}",
                f"{record.lap_time_last:.3f}",
                f"{lap_time_best:.3f}" if lap_time_best > 0 else "",
                record.safety_state,
                f"{record.processing_time_ms:.2f}",
                mode_data_str
            ])
        
        self._csv_file.flush()
        self._write_buffer.clear()
    
    def end_session(self) -> Optional[Dict[str, Any]]:
        """
        End the current recording session.
        
        Writes remaining buffered records, generates summary and JSON export.
        
        Returns:
            Session summary dictionary, or None if no session was active
        """
        with self._lock:
            if not self._is_recording:
                return None
            
            # Flush remaining records
            self._flush_buffer()
            
            # Close CSV
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
            
            # Generate summary
            summary = self._generate_summary()
            
            # Write JSON export
            self._write_json_export()
            
            # Write lap times CSV
            self._write_lap_times_csv()
            
            # Write summary JSON
            self._write_summary_json(summary)
            
            self._is_recording = False
            
            return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate session summary statistics."""
        if not self._all_records:
            return {"error": "No records"}
        
        # Calculate duration
        start_time = self._all_records[0].timestamp
        end_time = self._all_records[-1].timestamp
        duration = end_time - start_time
        
        # Collect lap times
        lap_times = []
        seen_laps = set()
        for record in self._all_records:
            if record.lap_number > 0 and record.lap_number not in seen_laps:
                if record.lap_time_last > 0:
                    lap_times.append((record.lap_number, record.lap_time_last))
                    seen_laps.add(record.lap_number)
        
        # Mode distribution
        mode_counts: Dict[str, int] = {}
        detection_mode_counts: Dict[str, int] = {}
        for record in self._all_records:
            # Control mode
            mode = record.control_mode
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            # Detection mode (Vision-specific)
            det_mode = record.mode_data.get("detection_mode", "UNKNOWN")
            detection_mode_counts[det_mode] = detection_mode_counts.get(det_mode, 0) + 1
        
        total_frames = len(self._all_records)
        mode_distribution = {k: (v / total_frames) * 100 for k, v in mode_counts.items()}
        detection_distribution = {k: (v / total_frames) * 100 for k, v in detection_mode_counts.items()}
        
        # Speed/steering stats
        speeds = [r.speed_cmd for r in self._all_records]
        steerings = [r.steering_cmd for r in self._all_records]
        
        summary = {
            "session_name": self._session_name,
            "start_time": self._session_start_time.isoformat() if self._session_start_time else None,
            "duration_sec": duration,
            "total_frames": total_frames,
            "avg_fps": total_frames / duration if duration > 0 else 0,
            "laps_completed": len(lap_times),
            "lap_times": [{"lap": lap, "time": t} for lap, t in lap_times],
            "best_lap_time": min([t for _, t in lap_times]) if lap_times else None,
            "avg_lap_time": sum([t for _, t in lap_times]) / len(lap_times) if lap_times else None,
            "mode_distribution": mode_distribution,
            "detection_mode_distribution": detection_distribution,
            "speed_stats": {
                "avg": sum(speeds) / len(speeds) if speeds else 0,
                "max": max(speeds) if speeds else 0,
                "min": min(speeds) if speeds else 0,
            },
            "steering_stats": {
                "avg": sum(steerings) / len(steerings) if steerings else 0,
                "max": max(steerings) if steerings else 0,
                "min": min(steerings) if steerings else 0,
            },
        }
        
        return summary
    
    def _write_json_export(self) -> None:
        """Write full session data to JSON file."""
        if not self._session_path or not self._all_records:
            return
        
        json_path = self._session_path / "telemetry.json"
        
        # Convert records to dictionaries
        records_data = []
        for record in self._all_records:
            record_dict = asdict(record)
            # Handle infinity
            if record_dict.get('lap_time_best') == float('inf'):
                record_dict['lap_time_best'] = None
            records_data.append(record_dict)
        
        data = {
            "session": {
                "name": self._session_name,
                "start_time": self._session_start_time.isoformat() if self._session_start_time else None,
                "total_frames": len(self._all_records),
            },
            "records": records_data
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _write_lap_times_csv(self) -> None:
        """Write lap times to separate CSV file."""
        if not self._session_path or not self._all_records:
            return
        
        # Collect lap times
        lap_times = []
        seen_laps = set()
        for record in self._all_records:
            if record.lap_number > 0 and record.lap_number not in seen_laps:
                if record.lap_time_last > 0:
                    lap_times.append((record.lap_number, record.lap_time_last))
                    seen_laps.add(record.lap_number)
        
        if not lap_times:
            return
        
        csv_path = self._session_path / "lap_times.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['lap_number', 'lap_time_sec'])
            for lap, time_sec in sorted(lap_times):
                writer.writerow([lap, f"{time_sec:.3f}"])
    
    def _write_summary_json(self, summary: Dict[str, Any]) -> None:
        """Write session summary to JSON file."""
        if not self._session_path:
            return
        
        json_path = self._session_path / "summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self._lock:
            return self._is_recording
    
    def get_session_path(self) -> Optional[Path]:
        """Get current session folder path."""
        return self._session_path
    
    def get_record_count(self) -> int:
        """Get number of records in current session."""
        with self._lock:
            return len(self._all_records)
    
    @staticmethod
    def export_charts(
        records: List[TelemetryRecord],
        output_path: Path,
        prefix: str = "chart"
    ) -> List[str]:
        """
        Export telemetry data as chart images.
        
        Note: Requires matplotlib. If not available, returns empty list.
        
        Args:
            records: List of telemetry records
            output_path: Directory to save charts
            prefix: Filename prefix for charts
            
        Returns:
            List of generated file paths
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            return []
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        if not records:
            return generated_files
        
        # Extract data
        timestamps = [r.timestamp for r in records]
        start_time = timestamps[0]
        times = [(t - start_time) for t in timestamps]
        speeds = [r.speed_cmd for r in records]
        steerings = [r.steering_cmd for r in records]
        errors = [r.mode_data.get("track_error_px", 0) for r in records]
        
        # 1. Speed Profile
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, speeds, 'b-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Speed Profile')
        ax.grid(True, alpha=0.3)
        path = output_path / f"{prefix}_speed.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        generated_files.append(str(path))
        
        # 2. Steering Profile
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, steerings, 'r-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Steering (rad)')
        ax.set_title('Steering Profile')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        path = output_path / f"{prefix}_steering.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        generated_files.append(str(path))
        
        # 3. Track Error
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, errors, 'g-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Track Error (px)')
        ax.set_title('Track Tracking Error')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        path = output_path / f"{prefix}_error.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        generated_files.append(str(path))
        
        # 4. Lap Times Bar Chart
        lap_times = []
        seen_laps = set()
        for record in records:
            if record.lap_number > 0 and record.lap_number not in seen_laps:
                if record.lap_time_last > 0:
                    lap_times.append((record.lap_number, record.lap_time_last))
                    seen_laps.add(record.lap_number)
        
        if lap_times:
            lap_times.sort()
            laps = [f"Lap {l}" for l, _ in lap_times]
            times_sec = [t for _, t in lap_times]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(laps, times_sec, color='steelblue')
            ax.set_xlabel('Lap')
            ax.set_ylabel('Time (s)')
            ax.set_title('Lap Times')
            
            # Add value labels on bars
            for bar, t in zip(bars, times_sec):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
            
            path = output_path / f"{prefix}_lap_times.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(str(path))
        
        # 5. Detection Mode Pie Chart
        detection_counts: Dict[str, int] = {}
        for record in records:
            det_mode = record.mode_data.get("detection_mode", "UNKNOWN")
            detection_counts[det_mode] = detection_counts.get(det_mode, 0) + 1
        
        if detection_counts and len(detection_counts) > 1:
            fig, ax = plt.subplots(figsize=(6, 6))
            labels = list(detection_counts.keys())
            sizes = list(detection_counts.values())
            colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
            
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                   colors=colors[:len(labels)], startangle=90)
            ax.set_title('Detection Mode Distribution')
            
            path = output_path / f"{prefix}_mode_distribution.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(str(path))
        
        return generated_files
