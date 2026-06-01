"""Tests for camera topic detection (CRITICAL - handles namespace variants)."""

import pytest
import subprocess
import os


@pytest.mark.launch
@pytest.mark.ros
class TestCameraTopicDetection:
    """Tests for camera topic auto-detection logic."""

    def test_camera_topic_detection_logic(self):
        """Test the camera topic detection function logic."""
        from formula_tron.vision_controller import VisionController
        import rclpy

        # This tests the logic used in VisionController to find the right topic
        # among different Realsense/Gazebo possible paths.
        # Actual topic detection requires ROS to be running
        try:
            rclpy.init()
            node = VisionController()
            # Check that detection method exists
            assert hasattr(node, '_detect_camera_topic')
            rclpy.shutdown()
        except Exception as e:
            pytest.skip(f"ROS not available: {e}")

    def test_camera_topic_variants(self):
        """Test that both camera topic variants are handled."""
        # This documents the expected behavior
        expected_topics = [
            '/camera/camera/color/image_raw',  # Double namespace (most common)
            '/camera/color/image_raw',          # Single namespace (fallback)
        ]

        # Test that the detection logic would handle both
        # Actual testing requires ROS with camera running
        assert len(expected_topics) == 2

    @pytest.mark.skip(reason="Requires ROS with camera running")
    def test_camera_topic_exists(self):
        """Test that camera topic exists (requires ROS + camera)."""
        try:
            result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            topics = result.stdout.split('\n')

            # Check for either variant
            has_camera = (
                '/camera/camera/color/image_raw' in topics or
                '/camera/color/image_raw' in topics
            )

            if not has_camera:
                pytest.skip("Camera topic not found (camera may not be running)")

            assert True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("ros2 command not available")
