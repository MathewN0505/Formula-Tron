#!/usr/bin/env python3
"""
Telemetry Dashboard Window - Real-time visualization of telemetry data.

Features:
- Real-time plots using pyqtgraph (speed, steering, track error)
- Lap times bar chart
- Detection mode statistics
- Session info display
- Record button with visual indicator
- Export functionality (CSV, JSON, Charts)
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QComboBox, QFileDialog,
    QMessageBox, QFrame, QSizePolicy, QApplication
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor

import numpy as np

# Try to import pyqtgraph
try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("Warning: pyqtgraph not installed. Install with: pip install pyqtgraph")

from .utils.telemetry import TelemetryRecord, TelemetryCollector
from .utils.telemetry_logger import TelemetryLogger


class TelemetrySignals(QObject):
    """Signals for thread-safe UI updates."""
    record_received = pyqtSignal(object)  # TelemetryRecord
    lap_completed = pyqtSignal(int, float)  # lap_number, lap_time


class TelemetryWindow(QMainWindow):
    """
    Telemetry Dashboard Window.
    
    Displays real-time telemetry data with plots and statistics.
    """
    
    def __init__(self, collector: Optional[TelemetryCollector] = None, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Telemetry Dashboard")
        self.setMinimumSize(900, 700)
        
        # Data management
        self.collector = collector or TelemetryCollector()
        self.logger = TelemetryLogger()
        self.signals = TelemetrySignals()
        
        # Plot data buffers (last 60 seconds at ~30Hz = 1800 points)
        self.buffer_size = 1800
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.speed_buffer = deque(maxlen=self.buffer_size)
        self.steering_buffer = deque(maxlen=self.buffer_size)
        self.error_buffer = deque(maxlen=self.buffer_size)
        self.start_time = time.time()
        
        # Lap times
        self.lap_times: List[tuple] = []  # (lap_number, time)
        
        # Mode stats
        self.detection_mode_counts: Dict[str, int] = {}
        
        # UI state
        self._is_recording = False
        
        # Setup UI
        self._setup_ui()
        
        # Connect signals
        self.signals.record_received.connect(self._on_record)
        self.signals.lap_completed.connect(self._on_lap)
        
        # Register callbacks with collector
        self.collector.on_record(lambda r: self.signals.record_received.emit(r))
        self.collector.on_lap(lambda lap, t: self.signals.lap_completed.emit(lap, t))
        
        # Update timer for plots (10Hz is smooth enough)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plots)
        self.update_timer.start(100)  # 100ms = 10Hz
        
        # Session timer for duration display
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self._update_session_time)
        self.session_timer.start(1000)  # 1Hz
    
    def _setup_ui(self):
        """Setup the UI components."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        
        # Top bar: Title + Controls
        top_bar = self._create_top_bar()
        layout.addWidget(top_bar)
        
        # Main content
        if PYQTGRAPH_AVAILABLE:
            content = self._create_plots_layout()
        else:
            content = self._create_no_pyqtgraph_fallback()
        layout.addWidget(content)
        
        # Status bar
        status_bar = self._create_status_bar()
        layout.addWidget(status_bar)
    
    def _create_top_bar(self) -> QWidget:
        """Create the top bar with title and controls."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("TELEMETRY DASHBOARD")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Record button
        self.record_btn = QPushButton("● Record")
        self.record_btn.setCheckable(True)
        self.record_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #e74c3c;
                color: white;
            }
        """)
        self.record_btn.clicked.connect(self._toggle_recording)
        layout.addWidget(self.record_btn)
        
        # Export dropdown
        self.export_combo = QComboBox()
        self.export_combo.addItems(["Export...", "CSV", "JSON", "Charts (PNG)", "All"])
        self.export_combo.currentIndexChanged.connect(self._on_export_selected)
        self.export_combo.setMinimumWidth(120)
        layout.addWidget(self.export_combo)
        
        return widget
    
    def _create_plots_layout(self) -> QWidget:
        """Create the main plots layout using pyqtgraph."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(10)
        
        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True)
        
        # Row 0: Speed + Steering + Lap Info
        # Speed plot
        speed_group = QGroupBox("Speed (m/s)")
        speed_layout = QVBoxLayout(speed_group)
        self.speed_plot = pg.PlotWidget()
        self.speed_plot.setBackground('w')
        self.speed_plot.showGrid(x=True, y=True, alpha=0.3)
        self.speed_plot.setLabel('bottom', 'Time', 's')
        self.speed_curve = self.speed_plot.plot(pen=pg.mkPen('b', width=2))
        speed_layout.addWidget(self.speed_plot)
        layout.addWidget(speed_group, 0, 0)
        
        # Steering plot
        steering_group = QGroupBox("Steering (rad)")
        steering_layout = QVBoxLayout(steering_group)
        self.steering_plot = pg.PlotWidget()
        self.steering_plot.setBackground('w')
        self.steering_plot.showGrid(x=True, y=True, alpha=0.3)
        self.steering_plot.setLabel('bottom', 'Time', 's')
        self.steering_plot.addLine(y=0, pen=pg.mkPen('k', width=1))
        self.steering_curve = self.steering_plot.plot(pen=pg.mkPen('r', width=2))
        steering_layout.addWidget(self.steering_plot)
        layout.addWidget(steering_group, 0, 1)
        
        # Lap info panel
        lap_group = QGroupBox("Lap Info")
        lap_layout = QVBoxLayout(lap_group)
        
        self.lap_number_label = QLabel("Lap: 0")
        self.lap_number_label.setFont(QFont("Arial", 16, QFont.Bold))
        lap_layout.addWidget(self.lap_number_label)
        
        self.lap_time_label = QLabel("Time: --")
        self.lap_time_label.setFont(QFont("Arial", 12))
        lap_layout.addWidget(self.lap_time_label)
        
        self.best_lap_label = QLabel("Best: --")
        self.best_lap_label.setFont(QFont("Arial", 12))
        self.best_lap_label.setStyleSheet("color: #27ae60;")
        lap_layout.addWidget(self.best_lap_label)
        
        lap_layout.addStretch()
        layout.addWidget(lap_group, 0, 2)
        
        # Row 1: Error + Mode Stats
        # Error plot
        error_group = QGroupBox("Track Error (px)")
        error_layout = QVBoxLayout(error_group)
        self.error_plot = pg.PlotWidget()
        self.error_plot.setBackground('w')
        self.error_plot.showGrid(x=True, y=True, alpha=0.3)
        self.error_plot.setLabel('bottom', 'Time', 's')
        self.error_plot.addLine(y=0, pen=pg.mkPen('k', width=1))
        self.error_curve = self.error_plot.plot(pen=pg.mkPen('g', width=2))
        error_layout.addWidget(self.error_plot)
        layout.addWidget(error_group, 1, 0, 1, 2)
        
        # Mode stats panel
        mode_group = QGroupBox("Detection Mode Stats")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_labels = {}
        for mode in ["CENTER", "L+R", "L_ONLY", "R_ONLY", "LOST"]:
            label = QLabel(f"{mode}: 0%")
            label.setFont(QFont("Arial", 10))
            mode_layout.addWidget(label)
            self.mode_labels[mode] = label
        
        mode_layout.addStretch()
        layout.addWidget(mode_group, 1, 2)
        
        # Row 2: Lap Times Bar Chart
        lap_times_group = QGroupBox("Lap Times")
        lap_times_layout = QVBoxLayout(lap_times_group)
        self.lap_bar_widget = pg.PlotWidget()
        self.lap_bar_widget.setBackground('w')
        self.lap_bar_widget.showGrid(x=False, y=True, alpha=0.3)
        self.lap_bar_widget.setLabel('left', 'Time', 's')
        self.lap_bar_widget.setLabel('bottom', 'Lap')
        lap_times_layout.addWidget(self.lap_bar_widget)
        layout.addWidget(lap_times_group, 2, 0, 1, 3)
        
        # Set row/column stretch
        layout.setRowStretch(0, 2)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 1)
        
        return widget
    
    def _create_no_pyqtgraph_fallback(self) -> QWidget:
        """Create fallback UI when pyqtgraph is not available."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel(
            "pyqtgraph is not installed.\n\n"
            "Install it with:\n"
            "pip install pyqtgraph\n\n"
            "Recording and export still work."
        )
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 12))
        layout.addWidget(label)
        
        # Still show lap info
        lap_group = QGroupBox("Lap Info")
        lap_layout = QVBoxLayout(lap_group)
        self.lap_number_label = QLabel("Lap: 0")
        self.lap_time_label = QLabel("Time: --")
        self.best_lap_label = QLabel("Best: --")
        lap_layout.addWidget(self.lap_number_label)
        lap_layout.addWidget(self.lap_time_label)
        lap_layout.addWidget(self.best_lap_label)
        layout.addWidget(lap_group)
        
        # Mode labels (for compatibility)
        self.mode_labels = {}
        
        return widget
    
    def _create_status_bar(self) -> QWidget:
        """Create the status bar at the bottom."""
        widget = QFrame()
        widget.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        self.mode_status = QLabel("Mode: --")
        layout.addWidget(self.mode_status)
        
        layout.addWidget(self._create_separator())
        
        self.frame_status = QLabel("Frames: 0")
        layout.addWidget(self.frame_status)
        
        layout.addWidget(self._create_separator())
        
        self.session_status = QLabel("Session: 00:00:00")
        layout.addWidget(self.session_status)
        
        layout.addStretch()
        
        return widget
    
    def _create_separator(self) -> QFrame:
        """Create a vertical separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep
    
    def _toggle_recording(self):
        """Toggle recording state."""
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """Start recording telemetry."""
        self._is_recording = True
        self.record_btn.setText("■ Stop")
        self.record_btn.setChecked(True)
        
        # Start session
        self.collector.start_session()
        session_name = self.logger.start_session()
        
        # Reset buffers
        self.time_buffer.clear()
        self.speed_buffer.clear()
        self.steering_buffer.clear()
        self.error_buffer.clear()
        self.lap_times.clear()
        self.detection_mode_counts.clear()
        self.start_time = time.time()
        
        # Reset UI
        self._update_mode_stats()
        self._update_lap_bar_chart()
        
        self.setWindowTitle(f"Telemetry Dashboard - Recording: {session_name}")
    
    def _stop_recording(self):
        """Stop recording telemetry."""
        self._is_recording = False
        self.record_btn.setText("● Record")
        self.record_btn.setChecked(False)
        
        # End session
        summary = self.collector.end_session()
        self.logger.end_session()
        
        session_path = self.logger.get_session_path()
        self.setWindowTitle("Telemetry Dashboard")
        
        if session_path:
            QMessageBox.information(
                self,
                "Recording Saved",
                f"Session saved to:\n{session_path}"
            )
    
    def _on_export_selected(self, index: int):
        """Handle export dropdown selection."""
        if index == 0:
            return  # "Export..." placeholder
        
        self.export_combo.setCurrentIndex(0)  # Reset dropdown
        
        records = self.collector.get_all()
        if not records:
            QMessageBox.warning(self, "No Data", "No telemetry data to export.")
            return
        
        options = ["", "CSV", "JSON", "Charts (PNG)", "All"]
        selection = options[index]
        
        # Get save location
        if selection == "All":
            folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
            if not folder:
                return
            output_path = Path(folder)
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
            if not folder:
                return
            output_path = Path(folder)
        
        exported = []
        
        if selection in ["CSV", "All"]:
            csv_path = output_path / "telemetry_export.csv"
            self._export_csv(records, csv_path)
            exported.append(str(csv_path))
        
        if selection in ["JSON", "All"]:
            json_path = output_path / "telemetry_export.json"
            self._export_json(records, json_path)
            exported.append(str(json_path))
        
        if selection in ["Charts (PNG)", "All"]:
            charts_path = output_path / "charts"
            chart_files = TelemetryLogger.export_charts(records, charts_path)
            exported.extend(chart_files)
        
        if exported:
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(exported)} file(s) to:\n{output_path}"
            )
    
    def _export_csv(self, records: List[TelemetryRecord], path: Path):
        """Export records to CSV."""
        import csv
        import json
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame_number', 'control_mode',
                'speed_cmd', 'steering_cmd', 'speed_actual',
                'lap_number', 'lap_time_current', 'safety_state',
                'processing_time_ms', 'mode_data'
            ])
            for r in records:
                writer.writerow([
                    f"{r.timestamp:.6f}", r.frame_number, r.control_mode,
                    f"{r.speed_cmd:.4f}", f"{r.steering_cmd:.4f}", f"{r.speed_actual:.4f}",
                    r.lap_number, f"{r.lap_time_current:.3f}", r.safety_state,
                    f"{r.processing_time_ms:.2f}", json.dumps(r.mode_data)
                ])
    
    def _export_json(self, records: List[TelemetryRecord], path: Path):
        """Export records to JSON."""
        import json
        from dataclasses import asdict
        
        data = {
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "record_count": len(records),
            "records": [asdict(r) for r in records]
        }
        
        # Handle infinity values
        for rec in data["records"]:
            if rec.get("lap_time_best") == float('inf'):
                rec["lap_time_best"] = None
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _on_record(self, record: TelemetryRecord):
        """Handle new telemetry record."""
        # Update buffers
        elapsed = record.timestamp - self.start_time
        self.time_buffer.append(elapsed)
        self.speed_buffer.append(record.speed_cmd)
        self.steering_buffer.append(record.steering_cmd)
        
        # Get track error from mode_data
        error = record.mode_data.get("track_error_px", 0)
        self.error_buffer.append(error)
        
        # Track detection mode
        det_mode = record.mode_data.get("detection_mode", "UNKNOWN")
        self.detection_mode_counts[det_mode] = self.detection_mode_counts.get(det_mode, 0) + 1
        
        # Write to logger if recording
        if self._is_recording:
            self.logger.write_record(record)
        
        # Update status bar
        self.mode_status.setText(f"Mode: {record.control_mode} ({det_mode})")
        self.frame_status.setText(f"Frames: {record.frame_number}")
        
        # Update lap info
        self.lap_number_label.setText(f"Lap: {record.lap_number}")
        self.lap_time_label.setText(f"Time: {record.lap_time_current:.1f}s")
        if record.lap_time_best < float('inf'):
            self.best_lap_label.setText(f"Best: {record.lap_time_best:.1f}s")
    
    def _on_lap(self, lap_number: int, lap_time: float):
        """Handle lap completion."""
        self.lap_times.append((lap_number, lap_time))
        self._update_lap_bar_chart()
    
    def _update_plots(self):
        """Update all plots with current buffer data."""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        if len(self.time_buffer) < 2:
            return
        
        times = list(self.time_buffer)
        
        # Speed plot
        self.speed_curve.setData(times, list(self.speed_buffer))
        
        # Steering plot
        self.steering_curve.setData(times, list(self.steering_buffer))
        
        # Error plot
        self.error_curve.setData(times, list(self.error_buffer))
        
        # Update mode stats every 10th update (1Hz)
        if int(time.time() * 10) % 10 == 0:
            self._update_mode_stats()
    
    def _update_mode_stats(self):
        """Update detection mode statistics display."""
        total = sum(self.detection_mode_counts.values())
        if total == 0:
            return
        
        for mode, label in self.mode_labels.items():
            count = self.detection_mode_counts.get(mode, 0)
            pct = (count / total) * 100
            label.setText(f"{mode}: {pct:.1f}%")
    
    def _update_lap_bar_chart(self):
        """Update the lap times bar chart."""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        self.lap_bar_widget.clear()
        
        if not self.lap_times:
            return
        
        # Sort by lap number
        sorted_laps = sorted(self.lap_times)
        x = list(range(len(sorted_laps)))
        heights = [t for _, t in sorted_laps]
        
        # Create bar chart
        bar = pg.BarGraphItem(x=x, height=heights, width=0.6, brush='steelblue')
        self.lap_bar_widget.addItem(bar)
        
        # Set x-axis labels
        ticks = [(i, f"Lap {lap}") for i, (lap, _) in enumerate(sorted_laps)]
        ax = self.lap_bar_widget.getAxis('bottom')
        ax.setTicks([ticks])
    
    def _update_session_time(self):
        """Update session duration display."""
        if self._is_recording:
            duration = self.collector.get_session_duration()
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            self.session_status.setText(f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def add_record(self, record: TelemetryRecord):
        """
        Add a telemetry record from external source.
        
        This is the main entry point for feeding data to the dashboard.
        """
        self.collector.record(record)
    
    def closeEvent(self, event):
        """Handle window close."""
        if self._is_recording:
            reply = QMessageBox.question(
                self,
                "Recording Active",
                "Recording is still active. Stop and save?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self._stop_recording()
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
        
        # Cleanup
        self.update_timer.stop()
        self.session_timer.stop()
        self.collector.remove_callbacks()


# Standalone testing
if __name__ == "__main__":
    import random
    
    app = QApplication(sys.argv)
    
    window = TelemetryWindow()
    window.show()
    
    # Simulate data
    frame = 0
    def generate_fake_data():
        global frame
        frame += 1
        record = TelemetryRecord(
            timestamp=time.time(),
            frame_number=frame,
            control_mode="VISION",
            speed_cmd=1.5 + random.uniform(-0.3, 0.3),
            steering_cmd=random.uniform(-0.3, 0.3),
            lap_number=frame // 100,
            lap_time_current=(frame % 100) * 0.033,
            lap_time_last=30.0 + random.uniform(-5, 5) if frame > 100 else 0,
            mode_data={
                "detection_mode": random.choice(["CENTER", "L+R", "L_ONLY", "R_ONLY"]),
                "track_error_px": random.uniform(-50, 50),
            }
        )
        window.add_record(record)
    
    timer = QTimer()
    timer.timeout.connect(generate_fake_data)
    timer.start(33)  # ~30Hz
    
    sys.exit(app.exec_())
