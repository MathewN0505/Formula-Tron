#!/usr/bin/env python3
"""
Track Detection - Finds where the track lines are in a camera image.
Supports two modes:
1. LEGACY: Histogram peak finding (Classic)
2. POLY_LOOKAHEAD: Bird's Eye View + Sliding Windows + Polyfit
"""

import cv2
import numpy as np
import scipy.signal
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class TrackDetectionResult:
    """Result from track detection."""
    target_x: Optional[float]   # X position to aim for (pixels, in original image space)
    left_peak: Optional[int]    # Left track line position
    right_peak: Optional[int]   # Right track line position
    mask: np.ndarray            # Binary mask of detected pixels
    histogram: np.ndarray       # Column sums
    all_peaks: np.ndarray       # ALL detected peaks (for visualization)
    used_peaks: np.ndarray      # Peaks used in strategy (for visualization)
    status: str                 # Status message
    
    # POLY_LOOKAHEAD Mode Fields
    bev_mask: Optional[np.ndarray] = None  # Bird's Eye View mask
    poly_coeffs: Optional[np.ndarray] = None # Polynomial coefficients (ax^2 + bx + c)
    curvature: Optional[float] = None        # Radius of curvature (m)
    target_x_bev: Optional[float] = None     # Target X in BEV space
    waypoints: Optional[np.ndarray] = None   # Smoothed physical trajectory points [[x,y],...]
    
    # Multi-track detection fields (for debug visualization)
    left_poly: Optional[np.ndarray] = None   # Left boundary polynomial coefficients
    right_poly: Optional[np.ndarray] = None  # Right boundary polynomial coefficients
    center_poly: Optional[np.ndarray] = None # Center line polynomial coefficients
    detection_mode: str = "UNKNOWN"          # Detection strategy used: CENTER, L+R, L_ONLY, R_ONLY, LOST


class TrackDetector:
    """Detects track lines using color filtering and histogram peaks."""

    def __init__(
        self,
        hsv_lower: np.ndarray = None,
        hsv_upper: np.ndarray = None,
        track_width: int = 550,
        roi_ratio: float = 0.4,
        bev_top_width: float = 0.4,
        bev_padding: float = 0.2,
        lookahead_ratio: float = 0.3,
        physical_track_width: float = 0.85,
    ):
        self.hsv_lower = hsv_lower if hsv_lower is not None else np.array([35, 50, 50])
        self.hsv_upper = hsv_upper if hsv_upper is not None else np.array([90, 255, 255])
        self.track_width = track_width
        self.physical_track_width = physical_track_width
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
        
        # POLY_LOOKAHEAD Mode: Perspective Transform Cache
        self.M = None
        self.Minv = None
        self.src_points = None
        self.dst_points = None
        
        # Temporal Smoothing Buffers for POLY_LOOKAHEAD mode
        # Stores polynomial coefficients from recent frames to reduce jitter
        self.poly_buffer_left: List[np.ndarray] = []
        self.poly_buffer_right: List[np.ndarray] = []
        self.poly_buffer_center: List[np.ndarray] = []
        self.SMOOTHING_BUFFER_SIZE = 2  # 2-frame buffer: fast corner response
                                        # without the snap-to-straight at turn exit
        
        # BEV mode: track width in BEV pixel space (will be calibrated on first detection)
        self.bev_track_width: Optional[float] = None

    def reset(self):
        """Clear temporal state (strategy lock, polynomial buffers, BEV cache)."""
        self.last_strategy = "NONE"
        self.strategy_lock_counter = 0
        self.poly_buffer_left.clear()
        self.poly_buffer_right.clear()
        self.poly_buffer_center.clear()
        self.bev_track_width = None

    def detect(self, frame: np.ndarray, mode: str = "POLY_LOOKAHEAD") -> TrackDetectionResult:
        """
        Detect track lines in a camera frame.
        mode: "LEGACY" or "POLY_LOOKAHEAD"
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
            if mode == "POLY_LOOKAHEAD":
                return self._detect_advanced(frame, mask, h, w, roi_h)
            else:
                return self._detect_simple(mask, h, w, roi_h)

        except Exception as e:
            # Safe fallback
            safe_mask = np.zeros((h // 4, w), dtype=np.uint8)
            safe_hist = np.zeros(w, dtype=np.float32)
            return TrackDetectionResult(
                None, None, None, safe_mask, safe_hist, 
                np.array([]), np.array([]), f"ERROR: {str(e)}"
            )

    def _detect_simple(self, mask: np.ndarray, h: int, w: int, roi_h: int) -> TrackDetectionResult:
        """Classic Histogram Peak Finding."""
        # Sum columns
        histogram = np.sum(mask, axis=0).astype(np.float32)
        if len(histogram) > 10:
            histogram = cv2.GaussianBlur(histogram.reshape(1, -1), (1, 15), 0).flatten()

        # Find peaks
        all_peaks = self._find_peaks(histogram)

        # Select best track pair
        left, right, target_x, status, used = self._select_tracks(all_peaks, histogram, w)

        return TrackDetectionResult(
            target_x=target_x,
            left_peak=left,
            right_peak=right,
            mask=mask,
            histogram=histogram,
            all_peaks=all_peaks,
            used_peaks=used,
            status=status
        )

    def _detect_advanced(self, frame: np.ndarray, mask: np.ndarray, h: int, w: int, roi_h: int) -> TrackDetectionResult:
        """Bird's Eye View + Multi-Track Detection + Polynomial Lookahead Point."""
        
        # 1. Perspective Transform (Bird's Eye View)
        bev_mask, M, Minv = self._bird_eye_view(mask)
        bev_h, bev_w = bev_mask.shape[:2]
        
        # 2. Multi-track Sliding Window Detection (returns target in BEV space)
        target_x_bev, poly_coeffs, status, left_poly, right_poly, center_poly, detection_mode, waypoints = self._sliding_window(bev_mask)
        
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
        
        return TrackDetectionResult(
            target_x=target_x,
            left_peak=None,
            right_peak=None,
            mask=mask,
            histogram=histogram,
            all_peaks=all_peaks,
            used_peaks=used_peaks,
            status=f"POLY: {status}",
            bev_mask=bev_mask,
            poly_coeffs=poly_coeffs,
            target_x_bev=target_x_bev,  # Store BEV coords for debug visualization
            left_poly=left_poly,
            right_poly=right_poly,
            center_poly=center_poly,
            detection_mode=detection_mode,
            waypoints=waypoints
        )

    def _pixels_to_meters(self, x_pix: float, y_pix: float, w: int, h: int) -> Tuple[float, float]:
        """Convert BEV pixels to physical meters relative to car origin.
        Car is at (w/2, h) in pixel space, representing (0, 0) in meters.
        Forward is +x, Left is +y.

        Scale is anchored to the known physical track width and measured BEV
        track width. Falls back to a geometry-based estimate when the BEV
        track width hasn't been calibrated yet."""
        if self.bev_track_width and self.bev_track_width > 10:
            meters_per_pixel = self.physical_track_width / self.bev_track_width
        else:
            estimated_bev_track = self.track_width * (1.0 - 2.0 * self.bev_padding)
            meters_per_pixel = self.physical_track_width / max(1.0, estimated_bev_track)
        x_meters = (h - y_pix) * meters_per_pixel
        y_meters = (w / 2.0 - x_pix) * meters_per_pixel
        return x_meters, y_meters

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

    def _find_zone_peaks(self, histogram: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Divide image into two halves to find left and right lanes.
        Returns (left_peak, None, right_peak). Center peak is disabled.
        """
        w = len(histogram)
        if w == 0 or np.max(histogram) == 0:
            return None, None, None
            
        half_w = int(w * 0.5)
        threshold = max(np.max(histogram) * 0.15, 100)
        
        left_peak = None
        right_peak = None
        
        # LEFT zone (0 to 50%)
        left_hist = histogram[:half_w]
        if len(left_hist) > 0 and np.max(left_hist) > threshold:
            left_peak = int(np.argmax(left_hist))
            
        # RIGHT zone (50% to 100%)
        right_hist = histogram[half_w:]
        if len(right_hist) > 0 and np.max(right_hist) > threshold:
            right_peak = int(np.argmax(right_hist)) + half_w
            
        return left_peak, None, right_peak

    def _track_line(self, bev_mask: np.ndarray, start_x: int, 
                    nwindows: int = 9, margin: int = 60, minpix: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track a single track line using sliding windows starting from start_x.
        Returns (x_points, y_points) arrays of detected track pixels.
        """
        h, w = bev_mask.shape[:2]
        window_height = int(h // nwindows)
        
        # Get all nonzero pixel positions
        nonzero = bev_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        current_x = start_x
        track_inds = []
        # Track cumulative drift — if the window centre has drifted too
        # far from where it started we are probably following an adjacent
        # line and should stop adding pixels.
        max_total_drift = margin * nwindows * 0.45  # ~2.5× single margin
        
        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            
            # Stop if we have drifted too far (following wrong line)
            if abs(current_x - start_x) > max_total_drift:
                break
            
            win_x_low = max(0, current_x - margin)
            win_x_high = min(w, current_x + margin)
            
            # Find pixels in this window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            
            track_inds.append(good_inds)
            
            # Recenter if enough pixels found
            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzerox[good_inds]))
        
        # Concatenate all indices
        if len(track_inds) == 0:
            return np.array([]), np.array([])
        
        track_inds = np.concatenate(track_inds)
        
        if len(track_inds) == 0:
            return np.array([]), np.array([])
        
        return nonzerox[track_inds], nonzeroy[track_inds]

    def _smooth_polynomial(self, new_poly: np.ndarray, buffer: List[np.ndarray]) -> np.ndarray:
        """
        Apply temporal smoothing to polynomial coefficients.
        Adds new_poly to buffer and returns the smoothed average.
        """
        buffer.append(new_poly.copy())
        if len(buffer) > self.SMOOTHING_BUFFER_SIZE:
            buffer.pop(0)
        return np.mean(buffer, axis=0)

    def _eval_poly(self, poly: np.ndarray, y: float) -> float:
        """Evaluate polynomial at y coordinate: x = a*y^2 + b*y + c"""
        return poly[0] * y**2 + poly[1] * y + poly[2]

    def _sliding_window(self, bev_mask: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray], str, 
                                                              Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str,
                                                              Optional[np.ndarray]]:
        """
        Multi-track detection with priority-based center tracking.
        
        Returns:
            target_x: X position for lookahead point in BEV space
            poly_coeffs: The polynomial used for the target (center, or computed center)
            status: Status message
            left_poly: Left boundary polynomial (if detected)
            right_poly: Right boundary polynomial (if detected)
            center_poly: Center line polynomial (computed or direct)
            detection_mode: Strategy used (CENTER, L+R, L_ONLY, R_ONLY, LOST)
            waypoints: Physical waypoints array [[x,y],...] in car-frame metres, or None
        """
        h, w = bev_mask.shape[:2]
        MIN_POINTS = 80  # Minimum points needed to fit a polynomial
        
        # 1. Create histogram from bottom half of BEV
        histogram = np.sum(bev_mask[h//2:, :], axis=0).astype(np.float32)
        
        # Apply smoothing to histogram
        if len(histogram) > 10:
            histogram = cv2.GaussianBlur(histogram.reshape(1, -1), (1, 21), 0).flatten()
        
        if np.max(histogram) == 0:
            return None, None, "NO TRACK", None, None, None, "LOST", None
        
        # 2. Find peaks in each zone (LEFT, CENTER, RIGHT)
        left_peak, center_peak, right_peak = self._find_zone_peaks(histogram)
        
        # 3. Track each detected track branch and fit polynomials
        left_poly = None
        right_poly = None
        center_poly_direct = None  # Direct center line detection
        
        lookahead_y = h * (1.0 - self.lookahead_ratio)
        
        # Track LEFT boundary
        if left_peak is not None:
            x_pts, y_pts = self._track_line(bev_mask, left_peak)
            if len(x_pts) >= MIN_POINTS:
                try:
                    left_poly = np.polyfit(y_pts, x_pts, 2)
                except Exception:
                    left_poly = None
        
        # Track CENTER line directly
        if center_peak is not None:
            x_pts, y_pts = self._track_line(bev_mask, center_peak)
            if len(x_pts) >= MIN_POINTS:
                try:
                    center_poly_direct = np.polyfit(y_pts, x_pts, 2)
                except Exception:
                    center_poly_direct = None
        
        # Track RIGHT boundary
        if right_peak is not None:
            x_pts, y_pts = self._track_line(bev_mask, right_peak)
            if len(x_pts) >= MIN_POINTS:
                try:
                    right_poly = np.polyfit(y_pts, x_pts, 2)
                except Exception:
                    right_poly = None
        
        # 4. Priority-based strategy selection
        target_x = None
        final_poly = None
        detection_mode = "LOST"
        
        # PRIORITY 1: Direct CENTER line detection (best case)
        if center_poly_direct is not None:
            smoothed = self._smooth_polynomial(center_poly_direct, self.poly_buffer_center)
            target_x = self._eval_poly(smoothed, lookahead_y)
            final_poly = smoothed
            detection_mode = "CENTER"
            # Clear other buffers since we're using center directly
            self.poly_buffer_left.clear()
            self.poly_buffer_right.clear()
        
        # PRIORITY 2: LEFT + RIGHT boundaries → compute center
        elif left_poly is not None and right_poly is not None:
            # Smooth all three coefficients [a, b, c] over the 2-frame buffer.
            # Averaging a and b is necessary: a single-frame quadratic fit to
            # near-straight pixels has high variance in `a`, and a*h^2 amplifies
            # any noise into a large spurious lateral offset at lookahead_y.
            # 2-frame window (was 3) gives faster corner response (~100ms at 20Hz)
            # without snap-to-straight at turn exit that 1-frame causes.
            left_smoothed = self._smooth_polynomial(left_poly, self.poly_buffer_left)
            right_smoothed = self._smooth_polynomial(right_poly, self.poly_buffer_right)

            # Compute center as average of left and right
            center_computed = (left_smoothed + right_smoothed) / 2.0

            # Update BEV track width for single-track fallback
            left_x = self._eval_poly(left_smoothed, lookahead_y)
            right_x = self._eval_poly(right_smoothed, lookahead_y)
            self.bev_track_width = abs(right_x - left_x)

            target_x = self._eval_poly(center_computed, lookahead_y)
            final_poly = center_computed
            left_poly = left_smoothed
            right_poly = right_smoothed
            detection_mode = "L+R"
            # Clear center buffer since we're computing from boundaries
            self.poly_buffer_center.clear()
        
        # PRIORITY 3: Only LEFT boundary → offset by half track width
        elif left_poly is not None:
            left_smoothed = self._smooth_polynomial(left_poly, self.poly_buffer_left)
            left_x = self._eval_poly(left_smoothed, lookahead_y)

            offset = self.bev_track_width / 2.0 if self.bev_track_width else w * 0.25
            target_x = left_x + offset

            # Create pseudo-center polynomial by offsetting left_poly
            final_poly = left_smoothed.copy()
            final_poly[2] += offset  # Offset the constant term
            left_poly = left_smoothed
            detection_mode = "L_ONLY"
            self.poly_buffer_right.clear()
            self.poly_buffer_center.clear()
        
        # PRIORITY 4: Only RIGHT boundary → offset by half track width
        elif right_poly is not None:
            right_smoothed = self._smooth_polynomial(right_poly, self.poly_buffer_right)
            right_x = self._eval_poly(right_smoothed, lookahead_y)

            offset = self.bev_track_width / 2.0 if self.bev_track_width else w * 0.25
            target_x = right_x - offset

            # Create pseudo-center polynomial by offsetting right_poly
            final_poly = right_smoothed.copy()
            final_poly[2] -= offset  # Offset the constant term
            right_poly = right_smoothed
            detection_mode = "R_ONLY"
            self.poly_buffer_left.clear()
            self.poly_buffer_center.clear()
            
        # 5. Generate Physical Waypoints from the detection
        waypoints = None
        if final_poly is not None:
            # Sample the full BEV height (y=h near car to y=0 far ahead) so that
            # experimental controllers have enough forward range for geometric
            # lookahead and MPC horizons. The polynomial is fit from sliding-window
            # data across the full BEV, so evaluation over [0, h] is valid.
            num_points = 30
            y_samples = np.linspace(h, 0, num_points)
            x_samples = self._eval_poly(final_poly, y_samples)
            
            # Convert these pixel coordinates to physical meters
            physical_points = []
            for px, py in zip(x_samples, y_samples):
                mx, my = self._pixels_to_meters(px, py, w, h)
                physical_points.append([mx, my])
            waypoints = np.array(physical_points)
        
        # PRIORITY 5: No lanes detected
        else:
            # Clear all buffers
            self.poly_buffer_left.clear()
            self.poly_buffer_right.clear()
            self.poly_buffer_center.clear()
            return None, None, "NO TRACK", None, None, None, "LOST", None
        
        # Clamp target_x to valid BEV bounds
        if target_x is not None:
            target_x = max(0.0, min(float(w), target_x))
        
        status = f"{detection_mode}"
        return target_x, final_poly, status, left_poly, right_poly, final_poly, detection_mode, waypoints

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

    def _select_tracks(self, peaks, histogram, width):
        """Pick the best tracks using stability-weighted fusion."""
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
            track_width = right - left
            width_error = abs(track_width - self.track_width)
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

    def auto_calibrate(self, frame: np.ndarray) -> Optional[Dict[str, int]]:
        """Automatically determine ideal HSV thresholds from the current frame.

        Assumes the car is on the track with at least one green line visible.
        Samples the ROI, finds green-ish pixels using a wide initial filter,
        then computes tight bounds using percentiles + safety margins.

        Returns a dict like {'hsv_h_min': 38, 'hsv_h_max': 82, ...} or None
        if not enough green pixels were found.
        """
        h = frame.shape[0]
        roi_h = max(int(h * self.roi_ratio), 10)
        roi = frame[h - roi_h:h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Wide initial green filter — catches any possible green under any lighting
        loose_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))

        # Focus on the center 60% of width for more reliable sampling
        h_roi, w_roi = loose_mask.shape[:2]
        x_start = int(w_roi * 0.2)
        x_end   = int(w_roi * 0.8)
        center_mask = loose_mask[:, x_start:x_end]
        center_hsv  = hsv[:, x_start:x_end]

        green_pixels = center_hsv[center_mask > 0]

        if len(green_pixels) < 50:
            # Not enough green — try full width as fallback
            green_pixels = hsv[loose_mask > 0]
            if len(green_pixels) < 50:
                return None

        # Compute tight bounds using percentiles (robust to outliers)
        h_vals = green_pixels[:, 0].astype(float)
        s_vals = green_pixels[:, 1].astype(float)
        v_vals = green_pixels[:, 2].astype(float)

        MARGIN_H = 8   # hue margin (degrees)
        MARGIN_SV = 15  # sat/val margin

        h_min = max(0,   int(np.percentile(h_vals, 5))  - MARGIN_H)
        h_max = min(180, int(np.percentile(h_vals, 95)) + MARGIN_H)
        s_min = max(0,   int(np.percentile(s_vals, 5))  - MARGIN_SV)
        v_min = max(0,   int(np.percentile(v_vals, 5))  - MARGIN_SV)

        # Apply to our own detector immediately
        self.hsv_lower = np.array([h_min, s_min, v_min])
        self.hsv_upper = np.array([h_max, 255, 255])

        return {
            'hsv_h_min': h_min,
            'hsv_h_max': h_max,
            'hsv_s_min': s_min,
            'hsv_v_min': v_min,
        }

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
        """Clear cached perspective transform matrices and smoothing buffers.
        
        Call this when switching modes or if camera parameters change.
        Forces recalculation on next frame.
        """
        self.M = None
        self.Minv = None
        self.src_points = None
        self.dst_points = None
        self.last_strategy = "NONE"
        self.strategy_lock_counter = 0
        
        # Clear temporal smoothing buffers
        self.poly_buffer_left.clear()
        self.poly_buffer_right.clear()
        self.poly_buffer_center.clear()
        self.bev_track_width = None