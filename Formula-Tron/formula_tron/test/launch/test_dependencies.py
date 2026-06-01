"""Tests for Python dependencies."""

import pytest
import sys


@pytest.mark.launch
class TestDependencies:
    """Tests for required Python packages."""

    def test_rclpy_import(self):
        """Test that rclpy can be imported."""
        try:
            import rclpy
            assert True
        except ImportError:
            pytest.skip("rclpy not installed")

    def test_cv_bridge_import(self):
        """Test that cv_bridge can be imported (cv2 first on Jetson)."""
        try:
            import cv2
            from cv_bridge import CvBridge
            assert True
        except ImportError:
            pytest.skip("cv_bridge not installed")

    def test_opencv_import(self):
        """Test that OpenCV can be imported."""
        try:
            import cv2
            assert cv2.__version__ is not None
        except ImportError:
            pytest.skip("OpenCV (cv2) not installed")

    def test_numpy_import(self):
        """Test that numpy can be imported."""
        try:
            import numpy as np
            assert np.__version__ is not None
        except ImportError:
            pytest.fail("numpy not installed (required)")

    def test_scipy_import(self):
        """Test that scipy can be imported."""
        try:
            import scipy
            import scipy.signal
            assert True
        except ImportError:
            pytest.skip("scipy not installed")

    def test_pyqt5_import(self):
        """Test that PyQt5 can be imported."""
        try:
            from PyQt5.QtWidgets import QApplication
            assert True
        except ImportError:
            pytest.skip("PyQt5 not installed (required for GUI)")

    def test_formula_tron_imports(self):
        """Test that all formula_tron modules can be imported."""
        try:
            from formula_tron.vision_controller import VisionController
            from formula_tron.control_gui import ControlGUI
            from formula_tron.utils.track_detection import TrackDetector
            from formula_tron.utils.safety import WatchdogTimer
            from formula_tron import config
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import formula_tron modules: {e}")
