"""Tests for vision controller node launch."""

import pytest
import launch_testing
from launch import LaunchDescription
from launch_ros.actions import Node


@pytest.mark.launch
@pytest.mark.ros
@pytest.mark.slow
class TestVisionControllerLaunch:
    """Tests for vision controller node launch."""

    def generate_test_description(self):
        """Generate test launch description."""
        vision_node = Node(
            package='formula_tron',
            executable='vision_controller',
            name='vision_controller',
            output='screen',
            parameters=[{
                'kp': 0.85,
                'kd': 0.20,
                'base_speed': 1.5,
            }]
        )
        return LaunchDescription([
            vision_node,
            launch_testing.actions.ReadyToTest(),
        ])

    @pytest.mark.skip(reason="Requires ROS environment and may conflict with running nodes")
    def test_vision_controller_starts(self, generate_test_description):
        """Test that vision controller node starts successfully."""
        # This would use launch_testing framework
        # Requires ROS 2 to be running
        pytest.skip("Full launch test requires ROS environment")
