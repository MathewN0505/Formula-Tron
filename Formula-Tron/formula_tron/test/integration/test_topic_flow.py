"""Integration tests for topic flow validation."""

import pytest
import rclpy
import numpy as np
import sys
import os
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64
import cv2  # Must import before cv_bridge on Jetson (cv_bridge_boost init order)
from cv_bridge import CvBridge

# Add test directory to path for fixture imports
_test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)


@pytest.mark.integration
@pytest.mark.ros
@pytest.mark.slow
class TestTopicFlow:
    """Tests for ROS topic data flow."""

    @pytest.fixture(autouse=True)
    def setup_ros(self):
        """Setup ROS for each test."""
        try:
            rclpy.init()
            yield
            rclpy.shutdown()
        except Exception:
            pytest.skip("ROS 2 not available")

    def test_camera_to_vision_flow(self):
        """Test that camera topic → vision controller flow works."""
        from formula_tron.vision_controller import VisionController
        from fixtures.mock_frames import create_frame_with_green_line

        node = VisionController()
        bridge = CvBridge()

        # Create test image
        frame = create_frame_with_green_line()
        msg = bridge.cv2_to_imgmsg(frame, "bgr8")

        # Enable autonomous
        node.autonomous_enabled = True
        node.autonomous_running = True

        # Process image
        try:
            node.image_callback(msg)
            # Should process without error
            assert True
        except Exception as e:
            pytest.fail(f"Camera to vision flow failed: {e}")
        finally:
            node.destroy_node()

    def test_vision_to_vesc_flow(self):
        """Test that vision controller → VESC command flow works."""
        from formula_tron.vision_controller import VisionController

        node = VisionController()
        node.autonomous_running = True

        # Set some steering and speed
        node.current_steering = 0.1
        node.current_speed = 1.0

        # Publish drive command
        try:
            node.publish_drive(node.current_speed, node.current_steering)
            # Should publish without error
            assert True
        except Exception as e:
            pytest.fail(f"Vision to VESC flow failed: {e}")
        finally:
            node.destroy_node()

    def test_gui_to_vision_tuning_flow(self):
        """Test that GUI → vision controller tuning flow works."""
        from formula_tron.vision_controller import VisionController
        from std_msgs.msg import Float64MultiArray

        node = VisionController()

        # Send tuning parameters from GUI
        pd_msg = Float64MultiArray()
        pd_msg.data = [1.0, 0.3]
        node.pd_callback(pd_msg)

        speed_msg = Float64()
        speed_msg.data = 2.0
        node.speed_callback(speed_msg)

        assert node.kp == 1.0
        assert node.kd == 0.3
        assert node.base_speed == 2.0
        node.destroy_node()

    def test_autonomous_toggle_flow(self):
        """Test autonomous enable/disable flow."""
        from formula_tron.vision_controller import VisionController
        from std_msgs.msg import Bool

        node = VisionController()

        # Enable
        enable_msg = Bool()
        enable_msg.data = True
        node.auto_callback(enable_msg)
        assert node.autonomous_enabled is True

        # Start
        start_msg = Bool()
        start_msg.data = True
        node.auto_start_callback(start_msg)
        assert node.autonomous_running is True

        # Disable
        enable_msg.data = False
        node.auto_callback(enable_msg)
        assert node.autonomous_enabled is False
        assert node.autonomous_running is False

        node.destroy_node()
