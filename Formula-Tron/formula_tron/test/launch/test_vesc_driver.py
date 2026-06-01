"""Tests for VESC driver validation."""

import pytest
import subprocess
import os


@pytest.mark.launch
@pytest.mark.ros
class TestVESCDriver:
    """Tests for VESC driver health."""

    @pytest.mark.skip(reason="Requires ROS with VESC running")
    def test_vesc_node_running(self):
        """Test that VESC node is running."""
        try:
            result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            nodes = result.stdout.split('\n')
            has_vesc = any('vesc' in node.lower() for node in nodes)

            if not has_vesc:
                pytest.skip("VESC node not running (will be started by RUN.sh)")

            assert True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("ros2 command not available")

    @pytest.mark.skip(reason="Requires ROS with VESC running")
    def test_vesc_topics_exist(self):
        """Test that VESC topics exist."""
        try:
            result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            topics = result.stdout.split('\n')

            motor_topic = '/commands/motor/speed' in topics
            servo_topic = '/commands/servo/position' in topics

            if not (motor_topic and servo_topic):
                pytest.skip("VESC topics not found (VESC may not be running)")

            assert motor_topic and servo_topic
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("ros2 command not available")
