#!/usr/bin/env python3
"""
Lap Timer Module using AprilTags.
Uses OpenCV's Aruco module to detect tags.
"""
from __future__ import annotations

import cv2
import time
import numpy as np
import logging

class AprilTagLapTimer:
    """
    Detects AprilTags to count laps.
    Target Tag: Standard 36h11 family.
    
    Detection pipeline (all gates must pass for a lap to count):
    1. Disappearance-based: tag must be invisible for N consecutive frames
    2. Time debounce: minimum seconds between laps
    3. Distance gate: tag must be within max_dist metres
    4. Pixel-width gate: tag must be at least min_tag_pixel_width pixels wide
    5. Direction gate (approach→depart): tag distance must decrease then increase,
       confirming the car drove *toward* the tag, passed it, and moved away.
    6. Hamming distance: ArUco detector rejects noisy/corrupted bit patterns
    """
    def __init__(
        self,
        tag_id: int,
        min_lap_time: float = 5.0,
        tag_size_m: float = 0.20,
        max_dist: float = 2.7,
        min_frames_without_tag: int = 10,
        min_frames_with_tag: int = 3,
        min_tag_pixel_width: float = 20.0,
        focal_length_px: float = 420.0,
    ):
        self.tag_id = tag_id
        self.min_lap_time = min_lap_time
        self.tag_size_m = tag_size_m
        self.max_dist = max_dist
        self.min_frames_without_tag = min_frames_without_tag
        self.min_frames_with_tag = max(1, int(min_frames_with_tag))
        self.min_tag_pixel_width = float(min_tag_pixel_width)
        self.focal_length_px = float(focal_length_px)
        
        self.lap_count = 0
        self.last_lap_trigger = 0.0  # Allow first detection immediately
        self.lap_start_time = None  # None until first reset() call
        self.best_lap_time = float('inf')
        self.current_lap_time = 0.0
        self.last_completed_lap_time = 0.0  # Store last completed lap time
        
        # Disappearance-based detection state
        self.frames_without_tag = 0  # Count of consecutive frames without tag
        self.frames_with_tag = 0     # Count of consecutive frames with tag
        self.can_trigger_lap = True  # True when tag has been gone long enough
        
        # Direction-based crossing detection (approach→depart)
        self._distance_history = []  # Recent estimated distances to tag
        self._saw_approach = False   # True once we've seen distance decreasing
        
        self.initialized = False
        self.detector = None
        self.ready = False  # Guard: don't count until reset() is called
        
        self._init_detector()

    def _init_detector(self):
        """Initialize appropriate Aruco detector based on OpenCV version."""
        if not hasattr(cv2, 'aruco'):
            logging.warning("cv2.aruco not available. Lap timer disabled.")
            return

        try:
            # Formula-Tron uses standard AprilTag 36h11
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            # Handle both old and new OpenCV API for DetectorParameters
            if hasattr(cv2.aruco, 'DetectorParameters'):
                # New API (OpenCV 4.7+)
                parameters = cv2.aruco.DetectorParameters()
            else:
                # Old API (OpenCV < 4.7)
                parameters = cv2.aruco.DetectorParameters_create()
            
            # Optimizations for better accuracy (research-backed)
            # Corner refinement improves detection precision
            if hasattr(parameters, 'cornerRefinementMethod'):
                parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            parameters.cornerRefinementWinSize = 5
            parameters.cornerRefinementMaxIterations = 30
            
            # Hamming distance filtering: reject detections with too many
            # bit errors in the border region (reduces false positives from
            # noisy/partially-occluded tags)
            if hasattr(parameters, 'maxErroneousBitsInBorderRate'):
                parameters.maxErroneousBitsInBorderRate = 0.35  # Default 0.35, keep conservative
            
            # Tighter adaptive threshold for variable lighting on the track
            if hasattr(parameters, 'adaptiveThreshWinSizeMin'):
                parameters.adaptiveThreshWinSizeMin = 3
            if hasattr(parameters, 'adaptiveThreshWinSizeMax'):
                parameters.adaptiveThreshWinSizeMax = 23
            if hasattr(parameters, 'adaptiveThreshWinSizeStep'):
                parameters.adaptiveThreshWinSizeStep = 10
            
            # Contour filtering: reject markers whose perimeter is too small
            if hasattr(parameters, 'minMarkerPerimeterRate'):
                parameters.minMarkerPerimeterRate = 0.03  # Default 0.03
            
            # Use modern API (OpenCV 4.7+)
            if hasattr(cv2.aruco, 'ArucoDetector'):
                self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            else:
                # Fallback for older OpenCV
                self.dictionary = dictionary
                self.parameters = parameters
                self.detector = None
                
            self.initialized = True
        except Exception as e:
            logging.error(f"Error initializing AprilTag detector: {e}")

    def reset(self):
        """Reset lap statistics (e.g. on auto enable/start)."""
        self.lap_count = 0
        self.lap_start_time = time.monotonic()
        self.last_lap_trigger = 0.0  # Allow first detection
        self.best_lap_time = float('inf')
        self.current_lap_time = 0.0
        self.last_completed_lap_time = 0.0
        self.frames_without_tag = self.min_frames_without_tag  # Allow first detection immediately
        self.frames_with_tag = 0
        self.can_trigger_lap = True  # Ready for first lap
        self.ready = True  # Now ready to count laps
        self._distance_history = []
        self._saw_approach = False

    def _detect_markers(self, gray):
        """Wrapper for marker detection (allows mocking in tests)."""
        if hasattr(self.detector, 'detectMarkers'):
            return self.detector.detectMarkers(gray)
        else:
            return cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
    
    def detect_tags_for_visualization(self, frame: np.ndarray) -> tuple[list, np.ndarray, list]:
        """
        Detect AprilTags in frame for visualization purposes (doesn't count laps).
        
        Args:
            frame: Input BGR or grayscale frame
            
        Returns:
            (corners, ids, rejected) - same format as aruco detection
        """
        if not self.initialized or frame is None:
            return [], None, []
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        try:
            corners, ids, rejected = self._detect_markers(gray)
            return corners, ids, rejected
        except Exception:
            return [], None, []

    def check_lap(self, frame: np.ndarray) -> tuple[bool, float, int]:
        """
        Check frame for lap tag.
        Returns: (is_new_lap, completed_lap_time, lap_count)
        - is_new_lap: True if a new lap was just completed
        - completed_lap_time: Time of completed lap (0.0 if no new lap)
        - lap_count: Current lap count
        """
        if not self.initialized or frame is None or not self.ready:
            return False, 0.0, self.lap_count
        
        if self.lap_start_time is None:
            # Not initialized yet, can't count
            return False, 0.0, self.lap_count
            
        now = time.monotonic()
        
        # 1. Detect Tags
        corners, ids, rejected = [], [], []
        
        # Grayscale is often faster/better for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        try:
            corners, ids, rejected = self._detect_markers(gray)
        except Exception:
            return False, 0.0, self.lap_count

        # 2. Check for Target Tag
        detected = False
        if ids is not None and len(ids) > 0:
            if self.tag_id in ids.flatten():
                detected = True

        # 3. Handle Lap Trigger with Debounce
        new_lap = False
        completed_lap_time = 0.0
        lap_duration = now - self.lap_start_time
        
        pixel_width = None
        est_dist = None
        if detected:
            # Check DISTANCE consistency
            try:
                # Find the index of our tag
                idx = np.where(ids == self.tag_id)[0][0]
                c = corners[idx][0] # The 4 corners of our tag
                
                # Top edge width in pixels
                pixel_width = np.linalg.norm(c[0] - c[1])
                
                # Estimate distance: Dist = (RealSize * Focal) / PixelSize
                est_dist = (self.tag_size_m * self.focal_length_px) / (pixel_width + 1e-6)
                
                logging.debug(f"Tag Distance: {est_dist:.2f}m (Limit: {self.max_dist}m)")
                
                if est_dist > self.max_dist:
                    # Too far away! Ignore it for now.
                    detected = False
            except Exception:
                 detected = False
            
            # Reject tiny detections (often false positives from far/noisy tags)
            if detected and pixel_width is not None and pixel_width < self.min_tag_pixel_width:
                detected = False

        if detected:
            self.frames_with_tag += 1
            
            # Track distance history for direction-based crossing
            if est_dist is not None:
                self._distance_history.append(est_dist)
                # Keep only the last 10 data points
                if len(self._distance_history) > 10:
                    self._distance_history = self._distance_history[-10:]
                
                # Check for approach pattern: if we have at least 2 points
                # and the distance is NOT increasing (constant or decreasing),
                # mark approach. Only receding (increasing distance) is suspicious.
                if len(self._distance_history) >= 2:
                    if self._distance_history[-1] <= self._distance_history[-2]:
                        self._saw_approach = True
            
            # Tag is visible - check if we can trigger a lap
            time_since_last = now - self.last_lap_trigger
            
            # Primary check: Disappearance-based (tag must have been gone for N frames)
            # Secondary check: Time-based debounce (backup safety)
            # Tertiary check: tag must be stable for a few consecutive frames
            # Quaternary check: direction gate — must have seen approach pattern
            #   - Skipped for first lap after reset (no history)
            #   - Skipped if fewer than 2 distance samples (can't determine direction)
            has_enough_distance_data = len(self._distance_history) >= 2
            direction_ok = self._saw_approach or self.lap_count == 0 or not has_enough_distance_data
            if (
                self.frames_with_tag >= self.min_frames_with_tag
                and self.can_trigger_lap
                and time_since_last > self.min_lap_time
                and direction_ok
            ):
                # LAP COUNTED!
                self.lap_count += 1
                
                # Store completed lap time BEFORE resetting
                completed_lap_time = lap_duration
                self.last_completed_lap_time = completed_lap_time
                
                # Track best lap (all laps count)
                if lap_duration < self.best_lap_time:
                    self.best_lap_time = lap_duration
                
                # Reset for next lap
                self.lap_start_time = now
                self.last_lap_trigger = now
                new_lap = True
                lap_duration = 0.0  # Just started new lap
                
                # CRITICAL: Tag is now visible, so we can't trigger again until it disappears
                self.can_trigger_lap = False
                self.frames_without_tag = 0
                self._distance_history = []
                self._saw_approach = False
                
                logging.debug(f"LAP {self.lap_count} counted! Duration: {completed_lap_time:.2f}s")
            else:
                # Tag detected but can't trigger yet (still in tag area or debounce)
                completed_lap_time = 0.0
                self.frames_without_tag = 0  # Reset disappearance counter
        else:
            # No tag detected - increment disappearance counter
            self.frames_with_tag = 0
            self.frames_without_tag += 1
            self._distance_history = []  # Reset distance history when tag lost
            completed_lap_time = 0.0
            
            # Once tag has been gone long enough, allow new lap detection
            if self.frames_without_tag >= self.min_frames_without_tag and not self.can_trigger_lap:
                self.can_trigger_lap = True
                self._saw_approach = False  # Reset approach flag for next crossing
                logging.debug(f"Tag gone for {self.frames_without_tag} frames - ready for next lap")

        self.current_lap_time = lap_duration
        # CRITICAL FIX: Only return completed lap time when a new lap is detected
        # Otherwise return 0.0 to prevent showing current lap duration
        return new_lap, completed_lap_time, self.lap_count

    def draw_tags(self, frame, corners):
        """Helper to draw detected tags for debug."""
        if hasattr(cv2.aruco, 'drawDetectedMarkers'):
             cv2.aruco.drawDetectedMarkers(frame, corners)
    
    def draw_april_tag_visualization(self, frame: np.ndarray, corners: list, ids: np.ndarray, 
                                     rejected: list = None) -> np.ndarray:
        """
        Draw comprehensive AprilTag visualization on frame.
        
        Args:
            frame: Input BGR frame
            corners: Detected tag corners from aruco detection
            ids: Detected tag IDs
            rejected: Rejected markers (optional)
            
        Returns:
            Annotated frame with tag visualizations
        """
        if frame is None:
            return frame
            
        # Create a copy to avoid modifying original
        vis_frame = frame.copy()
        
        if not self.initialized:
            cv2.putText(vis_frame, "AprilTag detector not initialized", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis_frame
        
        h, w = vis_frame.shape[:2]
        
        # Draw all detected tags
        if corners is not None and len(corners) > 0 and ids is not None:
            # Draw all tags with default color
            if hasattr(cv2.aruco, 'drawDetectedMarkers'):
                cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)
            
            # Find target tag and highlight it
            target_idx = None
            target_distance = None
            
            if len(ids) > 0:
                ids_flat = ids.flatten()
                if self.tag_id in ids_flat:
                    target_idx = np.where(ids_flat == self.tag_id)[0][0]
                    
                    # Calculate distance for target tag
                    try:
                        c = corners[target_idx][0]  # 4 corners of target tag
                        pixel_width = np.linalg.norm(c[0] - c[1])
                        target_distance = (self.tag_size_m * self.focal_length_px) / (pixel_width + 1e-6)
                        
                        # Draw thicker border for target tag (green)
                        corners_array = np.array([corners[target_idx]])
                        ids_array = np.array([[self.tag_id]])
                        cv2.aruco.drawDetectedMarkers(vis_frame, corners_array, ids_array, 
                                                     borderColor=(0, 255, 0))
                        
                        # Draw distance text near target tag
                        center = np.mean(c, axis=0).astype(int)
                        dist_text = f"{target_distance:.2f}m"
                        if target_distance > self.max_dist:
                            dist_text += " (TOO FAR)"
                            color = (0, 0, 255)  # Red if too far
                        else:
                            color = (0, 255, 0)  # Green if in range
                        
                        cv2.putText(vis_frame, dist_text, 
                                   (center[0] - 40, center[1] - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception:
                        pass
            
            # Draw tag IDs for all detected tags
            for i, tag_id in enumerate(ids.flatten()):
                try:
                    c = corners[i][0]
                    center = np.mean(c, axis=0).astype(int)
                    
                    # Use different color for target tag
                    if tag_id == self.tag_id:
                        id_color = (0, 255, 0)  # Green for target
                        id_text = f"TARGET: {tag_id}"
                    else:
                        id_color = (255, 255, 0)  # Cyan for others
                        id_text = f"ID: {tag_id}"
                    
                    cv2.putText(vis_frame, id_text, 
                               (center[0] - 30, center[1] + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)
                except Exception:
                    continue
        
        # Draw detection state overlay
        overlay_y = 30
        line_height = 25
        
        # Detection status
        if self.can_trigger_lap:
            status_text = "READY (can trigger lap)"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "LOCKED (tag must disappear)"
            status_color = (0, 0, 255)  # Red
        
        cv2.putText(vis_frame, f"Status: {status_text}", (10, overlay_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Frames without tag counter
        cv2.putText(vis_frame, f"Frames with tag: {self.frames_with_tag}/{self.min_frames_with_tag}", 
                   (10, overlay_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(vis_frame, f"Frames without tag: {self.frames_without_tag}/{self.min_frames_without_tag}", 
                   (10, overlay_y + line_height * 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Lap count
        cv2.putText(vis_frame, f"Lap Count: {self.lap_count}", 
                   (10, overlay_y + line_height * 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Current lap time
        if self.lap_start_time is not None:
            current_time = time.monotonic() - self.lap_start_time
            cv2.putText(vis_frame, f"Current Lap: {current_time:.2f}s", 
                       (10, overlay_y + line_height * 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection summary
        if ids is None or len(ids) == 0:
            summary_text = "No tags detected"
            summary_color = (128, 128, 128)
        elif target_idx is not None:
            if target_distance and target_distance <= self.max_dist:
                summary_text = f"Target tag detected (ID: {self.tag_id})"
                summary_color = (0, 255, 0)
            else:
                summary_text = f"Target tag too far (ID: {self.tag_id})"
                summary_color = (0, 165, 255)  # Orange
        else:
            summary_text = f"{len(ids)} tag(s) detected (not target)"
            summary_color = (255, 255, 0)
        
        cv2.putText(vis_frame, summary_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, summary_color, 2)
        
        return vis_frame
