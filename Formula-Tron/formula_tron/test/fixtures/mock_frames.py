"""Mock camera frame generators for testing."""

import numpy as np


def create_test_frame(width=640, height=480):
    """Create a basic test frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def create_frame_with_green_line(width=640, height=480, x_position=320):
    """Create a frame with a green vertical line at specified x position."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw green line (BGR format)
    frame[height//2:, x_position-10:x_position+10] = [0, 255, 0]
    return frame


def create_frame_with_two_tracks(width=640, height=480, track_width=320):
    """Create a frame with two green track lines."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    center_x = width // 2
    # Left track
    left_x = center_x - track_width // 2
    frame[height//2:, left_x-10:left_x+10] = [0, 255, 0]
    # Right track
    right_x = center_x + track_width // 2
    frame[height//2:, right_x-10:right_x+10] = [0, 255, 0]
    return frame


def create_frame_with_center_line(width=640, height=480):
    """Create a frame with a center green line."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    center_x = width // 2
    frame[height//2:, center_x-10:center_x+10] = [0, 255, 0]
    return frame


def create_noisy_frame(width=640, height=480):
    """Create a frame with noise (for testing robustness)."""
    frame = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    return frame
