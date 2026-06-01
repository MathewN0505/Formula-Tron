#!/usr/bin/env python3
"""
Lane Detection - Finds where the lane lines are in a camera image.
Supports two modes:
1. SIMPLE: Histogram peak finding (Classic)
2. ALTERNATIVE: Bird's Eye View + Sliding Windows + Polyfit (BETA - Experimental)
"""

import cv2
import numpy as np
import scipy.signal
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class LaneDetectionResult:
    """Result from lane detection."""
    target_x: Optional[float]   # X position to aim for (pixels, in original image space)
    left_peak: Optional[int]    # Left lane line position
    right_peak: Optional[int]   # Right lane line position
    mask: np.ndarray            # Binary mask of detected pixels
    histogram: np.ndarray       # Column sums
    all_peaks: np.ndarray       # ALL detected peaks (for visualization)
    used_peaks: np.ndarray      # Peaks used in strategy (for visualization)
    status: str                 # Status message
    
    # Alternative Mode Fields (BETA)
    bev_mask: Optional[np.ndarray] = None  # Bird's Eye View mask
    poly_coeffs: Optional[np.ndarray] = None # Polynomial coefficients (ax^2 + bx + c)
    curvature: Optional[float] = None        # Radius of curvature (m)
    target_x_bev: Optional[float] = None     # Target X in BEV space (for debug visualization)


class LaneDetector:
    """Detects lane lines using color filtering and histogram peaks."""

    def __init__(
        self,
        hsv_lower: np.ndarray = None,
        hsv_upper: np.ndarray = None,
        track_width: int = 550,
        roi_ratio: float = 0.4,
        bev_top_width: float = 0.4,
        bev_padding: float = 0.2,
        lookahead_ratio: float = 0.3,
    ):
        self.hsv_lower = hsv_lower if hsv_lower is not None else np.array([35, 50, 50])
        self.hsv_upper = hsv_upper if hsv_upper is not None else np.array([90, 255, 255])
        self.track_width = track_width
        self.roi_ratio = roi_ratio
        self.min_valid_width = int(track_width * 0.6)
        
        # BEV Parameters (configurable)
        self.bev_top_width = bev_top_width      # Top width of source trapezoid (0.0-1.0)
        self.bev_padding = bev_padding          # Side padding in destination (0.0-0.5)
        self.lookahead_ratio = lookahead_ratio  # How far ahead to look (0.0-1.0 from bottom)
        
        # State memory for hysteresis (prevents stuttering)
        self.last_strategy = "NONE"
        self.strategy_lock_counter = 0
        self.LOCK_FRAMES = 5  # Stick to a strategy for at least 5 frames
        
        # Alternative Mode: Perspective Transform Cache (BETA)
        self.M = None
        self.Minv = None
        self.src_points = None
        self.dst_points = None

    def detect(self, frame: np.ndarray, mode: str = "SIMPLE") -> LaneDetectionResult:
        """
        Detect lane lines in a camera frame.
        mode: "SIMPLE" or "ALTERNATIVE" (BETA)
        """
        if frame is None or frame.size == 0 or len(frame.shape) < 2:
            raise ValueError("Invalid frame")

        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            raise ValueError("Frame too small")

        try:
            # 1. Color Filtering (Common to both modes)
            # Extract ROI
            roi_h = max(int(h * self.roi_ratio), 10)
            roi = frame[h - roi_h:h, :]
            
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            
            # Morphological Cleanup
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 2. Branch based on mode
            if mode == "ALTERNATIVE":
                return self._detect_advanced(frame, mask, h, w, roi_h)
            else:
                return self._detect_simple(mask, h, w, roi_h)

        except Exception as e:
            # Safe fallback
            safe_mask = np.zeros((h // 4, w), dtype=np.uint8)
            safe_hist = np.zeros(w, dtype=np.float32)
            return LaneDetectionResult(
                None, None, None, safe_mask, safe_hist, 
                np.array([]), np.array([]), f"ERROR: {str(e)}"
            )

    def _detect_simple(self, mask: np.ndarray, h: int, w: int, roi_h: int) -> LaneDetectionResult:
        """Classic Histogram Peak Finding."""
        # Sum columns
        histogram = np.sum(mask, axis=0).astype(np.float32)
        if len(histogram) > 10:
            histogram = cv2.GaussianBlur(histogram.reshape(1, -1), (1, 15), 0).flatten()

        # Find peaks
        all_peaks = self._find_peaks(histogram)

        # Select best lane pair
        left, right, target_x, status, used = self._select_lanes(all_peaks, histogram, w)

        return LaneDetectionResult(
            target_x=target_x,
            left_peak=left,
            right_peak=right,
            mask=mask,
            histogram=histogram,
            all_peaks=all_peaks,
            used_peaks=used,
            status=status
        )

    def _detect_advanced(self, frame: np.ndarray, mask: np.ndarray, h: int, w: int, roi_h: int) -> LaneDetectionResult:
        """Bird's Eye View + Sliding Windows + Polynomial Lookahead Point (BETA - Experimental)."""
        
        # 1. Perspective Transform (Bird's Eye View)
        bev_mask, M, Minv = self._bird_eye_view(mask)
        bev_h, bev_w = bev_mask.shape[:2]
        
        # 2. Sliding Window Detection (returns target in BEV space)
        target_x_bev, poly_coeffs, status = self._sliding_window(bev_mask)
        
        # 3. Inverse transform target point from BEV back to original image space
        # This ensures the controller gets coordinates in the camera's native space
        target_x = None
        if target_x_bev is not None:
            # The lookahead point in BEV is at (target_x_bev, lookahead_y)
            # Use same formula as _sliding_window: y = h * (1.0 - lookahead_ratio)
            lookahead_y = bev_h * (1.0 - self.lookahead_ratio)
            
            # Transform point from BEV to original ROI space
            point_bev = np.array([[[target_x_bev, lookahead_y]]], dtype=np.float32)
            point_original = cv2.perspectiveTransform(point_bev, Minv)
            target_x = float(point_original[0][0][0])
            
            # Clamp to valid image bounds
            target_x = max(0.0, min(float(w), target_x))
        
        # Create empty histogram/peaks for compatibility with debug view
        histogram = np.sum(mask, axis=0).astype(np.float32)
        all_peaks = np.array([], dtype=np.int32)
        used_peaks = np.array([], dtype=np.int32)
        
        return LaneDetectionResult(
            target_x=target_x,
            left_peak=None,
            right_peak=None,
            mask=mask,
            histogram=histogram,
            all_peaks=all_peaks,
            used_peaks=used_peaks,
            status=f"ALT (BETA): {status}",
            bev_mask=bev_mask,
            poly_coeffs=poly_coeffs,
            target_x_bev=target_x_bev  # Store BEV coords for debug visualization
        )

    def _bird_eye_view(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Warp the mask to bird's eye view."""
        h, w = mask.shape[:2]
        
        if self.M is None or self.src_points is None:
            # Define trapezoid for source points
            # Bottom width: full image width
            # Top width: configurable (bev_top_width)
            # Height: full ROI height
            
            # Calculate top edge positions from bev_top_width
            # bev_top_width=0.4 means top is 40% of width, centered
            top_margin = (1.0 - self.bev_top_width) / 2.0  # e.g., 0.3 for 40% width
            
            src_bottom_left  = [0, h]
            src_bottom_right = [w, h]
            src_top_left     = [w * top_margin, 0]
            src_top_right    = [w * (1.0 - top_margin), 0]
            
            self.src_points = np.float32([src_bottom_left, src_bottom_right, src_top_left, src_top_right])
            
            # Destination points (Rectangle)
            # Add configurable padding on sides to see curved lanes
            padding = w * self.bev_padding
            dst_bottom_left  = [padding, h]
            dst_bottom_right = [w - padding, h]
            dst_top_left     = [padding, 0]
            dst_top_right    = [w - padding, 0]
            
            self.dst_points = np.float32([dst_bottom_left, dst_bottom_right, dst_top_left, dst_top_right])
            
            self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
            
        warped = cv2.warpPerspective(mask, self.M, (w, h), flags=cv2.INTER_LINEAR)
        return warped, self.M, self.Minv

    def _sliding_window(self, bev_mask: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray], str]:
        """Fit polynomial using sliding windows."""
        h, w = bev_mask.shape[:2]
        
        # 1. Find starting point using histogram of bottom half
        histogram = np.sum(bev_mask[h//2:, :], axis=0)
        
        if np.max(histogram) == 0:
            return None, None, "NO LINE"
            
        midpoint = int(histogram.shape[0] // 2)
        # Find strongest peak (either left, right, or center)
        # For simplicity, we just look for the strongest signal
        base_x = np.argmax(histogram)
        
        # 2. Sliding Windows
        nwindows = 9
        window_height = int(h // nwindows)
        nonzero = bev_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        current_x = base_x
        margin = 60       # Window width +/- margin
        minpix = 30       # Min pixels to recenter window
        
        lane_inds = []
        
        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            win_x_low = current_x - margin
            win_x_high = current_x + margin
            
            # Identify nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            
            lane_inds.append(good_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzerox[good_inds]))
                
        # Concatenate arrays of indices
        lane_inds = np.concatenate(lane_inds)
        
        # Extract pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        
        if len(x) < 100: # Not enough points to fit
            return None, None, "WEAK FIT"
            
        # 3. Fit 2nd order polynomial: x = ay^2 + by + c
        try:
            poly_coeffs = np.polyfit(y, x, 2)
            
            # Calculate target point for lookahead-based control
            # lookahead_ratio is distance from bottom (e.g., 0.3 = 30% up from bottom)
            # Convert to y coordinate: y = h * (1.0 - lookahead_ratio)
            lookahead_y = h * (1.0 - self.lookahead_ratio)
            target_x = poly_coeffs[0]*lookahead_y**2 + poly_coeffs[1]*lookahead_y + poly_coeffs[2]
            
            return target_x, poly_coeffs, "POLYFIT"
            
        except Exception:
            return None, None, "FIT FAIL"

    def _find_peaks(self, histogram: np.ndarray) -> np.ndarray:
        """Find significant peaks in histogram."""
        hist_max = float(np.max(histogram)) if len(histogram) > 0 else 0.0
        if hist_max == 0:
            return np.array([], dtype=np.int32)

        peaks, _ = scipy.signal.find_peaks(
            histogram,
            height=max(hist_max * 0.15, 200),
            distance=max(int(self.track_width * 0.3), 10),
            prominence=max(hist_max * 0.1, 50),
        )
        return peaks

    def _select_lanes(self, peaks, histogram, width):
        """Pick the best lanes using stability-weighted fusion."""
        center_x = width // 2

        if len(peaks) == 0:
            self.last_strategy = "NONE"
            return None, None, None, "NO LINES", np.array([])

        # Filter peaks (must be significant)
        heights = histogram[peaks]
        # Sort by height (strongest first)
        strong_idxs = np.argsort(heights)[::-1]
        strong = peaks[strong_idxs]
        
        # Categorize peaks by position
        left_candidates = []
        center_candidates = []
        right_candidates = []
        
        # Define regions (Left | Center | Right)
        left_zone = width * 0.35
        right_zone = width * 0.65
        
        for p in strong:
            if p < left_zone:
                left_candidates.append(p)
            elif p > right_zone:
                right_candidates.append(p)
            else:
                center_candidates.append(p)
        
        # Candidates
        left = left_candidates[0] if left_candidates else None
        center = center_candidates[0] if center_candidates else None
        right = right_candidates[0] if right_candidates else None
        
        # Strategies: (Target, UsedPeaks, StrategyName, Score)
        strategies = []
        
        # 1. Center Line (Best)
        if center is not None:
            score = 2.0
            # Boost score if we were using it recently (Sticky logic)
            if "CTR" in self.last_strategy: score += 0.5
            strategies.append((center, [center], "CTR", score))
            
        # 2. Left + Right (Good)
        if left is not None and right is not None:
            # Check width validity
            lane_width = right - left
            width_error = abs(lane_width - self.track_width)
            if width_error < self.track_width * 0.3: # Within 30% of expected
                score = 1.5
                if "L+R" in self.last_strategy: score += 0.5
                midpoint = (left + right) / 2.0
                strategies.append((midpoint, [left, right], "L+R", score))
            
        # 3. Left Only (Fallback)
        if left is not None:
            score = 1.0
            if "L_ONLY" in self.last_strategy: score += 0.5
            target = left + (self.track_width / 2.0)
            strategies.append((target, [left], "L_ONLY", score))
            
        # 4. Right Only (Fallback)
        if right is not None:
            score = 1.0
            if "R_ONLY" in self.last_strategy: score += 0.5
            target = right - (self.track_width / 2.0)
            strategies.append((target, [right], "R_ONLY", score))
            
        # Select best strategy
        if not strategies:
            return None, None, None, "LOST", np.array([])
            
        # Sort by score descending
        strategies.sort(key=lambda x: x[3], reverse=True)
        best_target, best_peaks, best_name, _ = strategies[0]
        
        # Update memory
        self.last_strategy = best_name
        
        # Determine which peaks to return based on strategy
        return_left = None
        return_right = None
        if best_name == "L+R":
            return_left = left
            return_right = right
        elif best_name == "L_ONLY":
            return_left = left
        elif best_name == "R_ONLY":
            return_right = right
        # CTR doesn't return left/right peaks
        
        return (
            return_left,
            return_right,
            best_target, 
            f"FUSION: {best_name}", 
            np.unique(np.array(best_peaks))
        )

    def update_hsv(self, h_min, h_max, s_min, v_min):
        """Update color thresholds."""
        self.hsv_lower = np.array([h_min, s_min, v_min])
        self.hsv_upper = np.array([h_max, 255, 255])

    def update_track_width(self, width):
        """Update expected track width."""
        self.track_width = width
        self.min_valid_width = int(width * 0.6)

    def update_lookahead_ratio(self, ratio):
        """Update lookahead ratio for polynomial lookahead point selection (0.05-0.95)."""
        self.lookahead_ratio = max(0.05, min(0.95, ratio))

    def update_bev_top_width(self, width):
        """Update BEV trapezoid top width (0.2-0.6).
        
        Changing this requires resetting the perspective cache.
        """
        self.bev_top_width = max(0.2, min(0.6, width))
        self.reset_perspective_cache()  # Force recalculation

    def update_bev_padding(self, padding):
        """Update BEV side padding (0.1-0.4).
        
        Changing this requires resetting the perspective cache.
        """
        self.bev_padding = max(0.1, min(0.4, padding))
        self.reset_perspective_cache()  # Force recalculation

    def reset_perspective_cache(self):
        """Clear cached perspective transform matrices.
        
        Call this when switching modes or if camera parameters change.
        Forces recalculation on next frame.
        """
        self.M = None
        self.Minv = None
        self.src_points = None
        self.dst_points = None
        self.last_strategy = "NONE"
        self.strategy_lock_counter = 0
