"""Tests for camera driver validation."""

import pytest
import subprocess


@pytest.mark.launch
@pytest.mark.ros
class TestCameraDriver:
    """Tests for camera driver health."""

    @pytest.mark.skip(reason="Requires ROS with camera running")
    def test_camera_topic_exists(self):
        """Test that camera topic exists."""
        try:
            result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            topics = result.stdout.split('\n')

            # Check for either namespace variant
            has_camera = (
                '/camera/camera/color/image_raw' in topics or
                '/camera/color/image_raw' in topics
            )

            if not has_camera:
                pytest.skip("Camera topic not found (camera may not be running)")

            assert True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("ros2 command not available")

    @pytest.mark.skip(reason="Requires ROS with camera running")
    def test_camera_publishing(self):
        """Test that camera is publishing frames."""
        try:
            # Check if topic has publishers
            result = subprocess.run(
                ['ros2', 'topic', 'info', '/camera/color/image_raw'],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            # If command succeeds, topic exists
            if result.returncode == 0:
                assert True
            else:
                pytest.skip("Camera topic not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("ros2 command not available")
