"""Integration tests for control GUI node."""

import pytest
import rclpy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String


@pytest.mark.integration
@pytest.mark.ros
class TestControlGUINode:
    """Integration tests for ControlGUI ROS node."""

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
        """Test that GUI node initializes without errors."""
        from formula_tron.control_gui import CarControlNode, RosSignals

        try:
            signals = RosSignals()
            node = CarControlNode(signals)
            assert node is not None
            assert hasattr(node, 'motor_pub')
            assert hasattr(node, 'servo_pub')
            node.destroy_node()
        except Exception as e:
            pytest.fail(f"GUI node initialization failed: {e}")

    def test_node_publishes_tuning_parameters(self):
        """Test that node creates tuning publishers."""
        from formula_tron.control_gui import CarControlNode, RosSignals

        signals = RosSignals()
        node = CarControlNode(signals)
        assert hasattr(node, 'pd_pub')
        assert hasattr(node, 'hsv_pub')
        assert hasattr(node, 'auto_speed_pub')
        node.destroy_node()

    def test_autonomous_enable_publishing(self):
        """Test autonomous enable publishing."""
        from formula_tron.control_gui import CarControlNode, RosSignals

        signals = RosSignals()
        node = CarControlNode(signals)
        node.set_autonomous(True)
        assert node.autonomous_enabled is True
        node.destroy_node()

    def test_pd_publishing(self):
        """Test PD parameter publishing."""
        from formula_tron.control_gui import CarControlNode, RosSignals

        signals = RosSignals()
        node = CarControlNode(signals)
        node.publish_pd(1.0, 0.3)
        assert node.kp == 1.0
        assert node.kd == 0.3
        node.destroy_node()

    def test_hsv_publishing(self):
        """Test HSV parameter publishing."""
        from formula_tron.control_gui import CarControlNode, RosSignals

        signals = RosSignals()
        node = CarControlNode(signals)
        node.publish_hsv(40, 80, 60, 60)
        assert node.hsv_h_min == 40
        assert node.hsv_h_max == 80
        node.destroy_node()

    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        from formula_tron.control_gui import CarControlNode, RosSignals

        signals = RosSignals()
        node = CarControlNode(signals)
        node.autonomous_enabled = True
        node.autonomous_running = True

        node.emergency_stop()
        assert node.autonomous_enabled is False
        assert node.autonomous_running is False
        node.destroy_node()
