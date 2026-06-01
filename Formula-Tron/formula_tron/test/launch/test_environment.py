"""Tests for ROS 2 environment validation."""

import pytest
import os
import subprocess
import sys


@pytest.mark.launch
@pytest.mark.ros
class TestEnvironment:
    """Tests for ROS 2 environment setup."""

    def test_ros_distro_set(self):
        """Test that ROS_DISTRO environment variable is set."""
        ros_distro = os.environ.get('ROS_DISTRO')
        if ros_distro is None:
            pytest.skip("ROS_DISTRO not set (not in ROS environment)")
        assert ros_distro == 'foxy', f"Expected ROS_DISTRO=foxy, got {ros_distro}"

    def test_ros2_command_exists(self):
        """Test that ros2 command is available."""
        try:
            result = subprocess.run(
                ['ros2', '--help'],
                capture_output=True,
                timeout=5.0,
            )
            assert result.returncode == 0, "ros2 command not found or not working"
        except FileNotFoundError:
            pytest.skip("ros2 command not found (not in ROS environment)")

    def test_colcon_command_exists(self):
        """Test that colcon command is available."""
        try:
            result = subprocess.run(
                ['colcon', '--help'],
                capture_output=True,
                timeout=5.0,
            )
            assert result.returncode == 0, "colcon command not found or not working"
        except FileNotFoundError:
            pytest.skip("colcon command not found (not in ROS environment)")

    def test_ros2_python_packages(self):
        """Test that required ROS 2 Python packages are importable."""
        try:
            import rclpy
            from std_msgs.msg import Bool, Float64
            from sensor_msgs.msg import Image
            assert True
        except ImportError as e:
            pytest.skip(f"ROS 2 Python packages not available: {e}")

    def test_workspace_paths(self):
        """Test that workspace paths exist."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        formula_tron_dir = os.path.dirname(os.path.dirname(script_dir))
        assert os.path.exists(formula_tron_dir), f"formula_tron directory not found: {formula_tron_dir}"

        setup_py = os.path.join(formula_tron_dir, 'setup.py')
        assert os.path.exists(setup_py), f"setup.py not found: {setup_py}"

        package_xml = os.path.join(formula_tron_dir, 'package.xml')
        assert os.path.exists(package_xml), f"package.xml not found: {package_xml}"
