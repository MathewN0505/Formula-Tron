"""Tests for module imports."""

import pytest
import sys
from pathlib import Path


@pytest.mark.launch
class TestImports:
    """Tests for module import validation."""

    def test_import_vision_controller(self):
        """Test that vision_controller can be imported."""
        try:
            from formula_tron.vision_controller import VisionController
            assert VisionController is not None
        except ImportError as e:
            pytest.fail(f"Failed to import VisionController: {e}")

    def test_import_control_gui(self):
        """Test that control_gui can be imported."""
        try:
            from formula_tron.control_gui import ControlGUI, CarControlNode
            assert ControlGUI is not None
            assert CarControlNode is not None
        except ImportError as e:
            pytest.fail(f"Failed to import control_gui: {e}")

    def test_import_track_detection(self):
        """Test that track_detection can be imported."""
        try:
            from formula_tron.utils.track_detection import TrackDetector, TrackDetectionResult
            assert TrackDetector is not None
            assert TrackDetectionResult is not None
        except ImportError as e:
            pytest.fail(f"Failed to import track_detection: {e}")

    def test_import_safety(self):
        """Test that safety module can be imported."""
        try:
            from formula_tron.utils.safety import (
                SafetyValidator,
                WatchdogTimer,
                ExponentialMovingAverage,
                ConnectionMonitor,
                safe_normalize,
            )
            assert SafetyValidator is not None
            assert WatchdogTimer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import safety: {e}")

    def test_import_config(self):
        """Test that config can be imported."""
        try:
            import formula_tron.config as config
            assert config.KP_DEFAULT is not None
            assert config.CAMERA_TOPIC is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config: {e}")

    def test_import_all_modules(self):
        """Test that all main modules can be imported together."""
        try:
            from formula_tron.vision_controller import VisionController
            from formula_tron.control_gui import ControlGUI
            from formula_tron.utils.track_detection import TrackDetector
            from formula_tron.utils.safety import WatchdogTimer
            import formula_tron.config as config
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import all modules: {e}")
