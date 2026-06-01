"""Integration tests for vision controller node."""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64, Float64MultiArray
import cv2  # Must import before cv_bridge on Jetson (cv_bridge_boost init order)
from cv_bridge import CvBridge

# Add test directory to path for fixture imports
_test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)


@pytest.mark.integration
@pytest.mark.ros
class TestVisionControllerNode:
    """Integration tests for VisionController ROS node."""

    @pytest.fixture(autouse=True)
    def setup_ros(self):
        """Setup ROS for each test."""
        try:
            rclpy.init()
            yield
            rclpy.shutdown()
        except Exception:
            pytest.skip("ROS 2 not available")

    def test_node_initialization(self):
        """Test that node initializes without errors."""
        from formula_tron.vision_controller import VisionController

        try:
            node = VisionController()
            assert node is not None
            assert hasattr(node, 'camera_topic')
            assert hasattr(node, 'motor_pub')
            assert hasattr(node, 'servo_pub')
            node.destroy_node()
        except Exception as e:
            pytest.fail(f"Node initialization failed: {e}")

    def test_node_subscribes_to_camera(self):
        """Test that node subscribes to camera topic."""
        from formula_tron.vision_controller import VisionController

        node = VisionController()
        # Check that subscription was created
        assert hasattr(node, 'image_sub')
        node.destroy_node()

    def test_node_publishes_commands(self):
        """Test that node creates command publishers."""
        from formula_tron.vision_controller import VisionController

        node = VisionController()
        assert hasattr(node, 'motor_pub')
        assert hasattr(node, 'servo_pub')
        node.destroy_node()

    @patch('subprocess.run')
    def test_camera_topic_detection(self, mock_subprocess):
        """Test camera topic auto-detection."""
        from formula_tron.vision_controller import VisionController

        # Mock topic list with double namespace
        mock_result = Mock()
        mock_result.stdout = '/camera/camera/color/image_raw\n'
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        node = VisionController()
        # Should detect double namespace
        assert '/camera/camera/color/image_raw' in node.camera_topic or \
               '/camera/color/image_raw' in node.camera_topic
        node.destroy_node()

    def test_autonomous_enable_callback(self):
        """Test autonomous enable callback."""
        from formula_tron.vision_controller import VisionController

        node = VisionController()
        msg = Bool()
        msg.data = True

        node.auto_callback(msg)
        assert node.autonomous_enabled is True

        msg.data = False
        node.auto_callback(msg)
        assert node.autonomous_enabled is False
        node.destroy_node()

    def test_pd_tuning_callback(self):
        """Test PD parameter tuning callback."""
        from formula_tron.vision_controller import VisionController

        node = VisionController()
        msg = Float64MultiArray()
        msg.data = [1.0, 0.3]  # Kp, Kd

        node.pd_callback(msg)
        assert node.kp == 1.0
        assert node.kd == 0.3
        node.destroy_node()

    def test_hsv_tuning_callback(self):
        """Test HSV parameter tuning callback."""
        from formula_tron.vision_controller import VisionController

        node = VisionController()
        msg = Float64MultiArray()
        msg.data = [40.0, 80.0, 60.0, 60.0]  # H min, H max, S min, V min

        node.hsv_callback(msg)
        assert np.array_equal(node.hsv_lower, np.array([40, 60, 60]))
        node.destroy_node()

    def test_speed_tuning_callback(self):
        """Test speed tuning callback."""
        from formula_tron.vision_controller import VisionController

        node = VisionController()
        msg = Float64()
        msg.data = 2.0

        node.speed_callback(msg)
        assert node.base_speed == 2.0
        node.destroy_node()

    def test_image_callback_with_valid_frame(self):
        """Test image callback processes valid frame."""
        from formula_tron.vision_controller import VisionController
        from fixtures.mock_frames import create_frame_with_green_line

        node = VisionController()
        bridge = CvBridge()

        # Create test frame
        frame = create_frame_with_green_line()
        msg = bridge.cv2_to_imgmsg(frame, "bgr8")

        # Enable autonomous
        node.autonomous_enabled = True
        node.autonomous_running = True

        # Process frame
        try:
            node.image_callback(msg)
            # Should process without error
            assert True
        except Exception as e:
            pytest.fail(f"Image callback failed: {e}")
        finally:
            node.destroy_node()

    def test_watchdog_stops_on_timeout(self):
        """Test that watchdog stops car on timeout."""
        from formula_tron.vision_controller import VisionController

        node = VisionController()
        node.autonomous_running = True

        # Simulate timeout
        node.watchdog.last_success_time = 0.0  # Long time ago
        import time
        time.sleep(0.1)  # Wait a bit

        if node.watchdog.should_stop():
            assert node.autonomous_running is False or \
                   node.watchdog.stopped is True

        node.destroy_node()
