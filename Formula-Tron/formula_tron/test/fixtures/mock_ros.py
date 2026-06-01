"""Mock ROS utilities for testing."""

import numpy as np
from unittest.mock import Mock, MagicMock
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64, Float64MultiArray
import cv2  # Must import before cv_bridge on Jetson (cv_bridge_boost init order)
from cv_bridge import CvBridge


def create_mock_image_message(frame):
    """Create a mock ROS Image message from numpy frame."""
    bridge = CvBridge()
    try:
        return bridge.cv2_to_imgmsg(frame, "bgr8")
    except Exception:
        # Fallback mock if cv_bridge fails
        msg = Mock(spec=Image)
        msg.width = frame.shape[1] if len(frame.shape) > 1 else 640
        msg.height = frame.shape[0] if len(frame.shape) > 0 else 480
        msg.encoding = "bgr8"
        return msg


def create_mock_bool_message(value):
    """Create a mock Bool message."""
    msg = Mock(spec=Bool)
    msg.data = value
    return msg


def create_mock_float64_message(value):
    """Create a mock Float64 message."""
    msg = Mock(spec=Float64)
    msg.data = value
    return msg


def create_mock_float64_multiarray(values):
    """Create a mock Float64MultiArray message."""
    msg = Mock(spec=Float64MultiArray)
    msg.data = list(values)
    return msg


class MockROSNode:
    """Mock ROS 2 node for testing."""

    def __init__(self):
        self.publishers = {}
        self.subscribers = {}
        self.parameters = {}
        self.logger = Mock()

    def create_publisher(self, msg_type, topic, qos):
        """Mock publisher creation."""
        pub = Mock()
        pub.publish = Mock()
        self.publishers[topic] = pub
        return pub

    def create_subscription(self, msg_type, topic, callback, qos):
        """Mock subscription creation."""
        sub = Mock()
        self.subscribers[topic] = callback
        return sub

    def declare_parameter(self, name, default_value):
        """Mock parameter declaration."""
        self.parameters[name] = default_value

    def get_parameter(self, name):
        """Mock parameter retrieval."""
        param = Mock()
        param.value = self.parameters.get(name, None)
        return param
