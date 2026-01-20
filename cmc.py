"""
Camera Motion Compensation (CMC) for tracking.

Implements hybrid ORB/ECC strategy for handling camera shake and pans.
Ported from McByte/yolox/tracker/gmc.py with enhancements.
"""

import copy
from typing import Optional, Tuple, List
import cv2
import numpy as np

from .config import CMCConfig


class CameraMotionCompensator:
    """
    Camera motion compensation using feature-based or dense alignment.

    Supports multiple methods with automatic fallback:
    - orb: Fast ORB feature matching
    - sift: More accurate SIFT features
    - ecc: Dense ECC alignment
    - sparseOptFlow: Lucas-Kanade optical flow
    - hybrid: ORB with ECC fallback
    """

    def __init__(self, config: CMCConfig):
        """
        Initialize the CMC module.

        Args:
            config: CMC configuration
        """
        self.config = config
        self.method = config.method
        self.downscale = max(1, config.downscale)

        # Initialize method-specific components
        self._init_orb()
        self._init_ecc()
        self._init_optical_flow()

        # State for temporal tracking
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[List] = None
        self.prev_descriptors: Optional[np.ndarray] = None
        self.initialized = False

    def _init_orb(self) -> None:
        """Initialize ORB detector and matcher."""
        self.orb_detector = cv2.FastFeatureDetector_create(20)
        self.orb_extractor = cv2.ORB_create(
            nfeatures=self.config.orb_n_features,
            scaleFactor=self.config.orb_scale_factor,
            nlevels=self.config.orb_n_levels
        )
        self.orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _init_ecc(self) -> None:
        """Initialize ECC alignment parameters."""
        self.ecc_warp_mode = cv2.MOTION_EUCLIDEAN
        self.ecc_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.config.ecc_iterations,
            self.config.ecc_epsilon
        )

    def _init_optical_flow(self) -> None:
        """Initialize sparse optical flow parameters."""
        self.flow_feature_params = dict(
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=1,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04
        )

    def reset(self) -> None:
        """Reset state for new video/sequence."""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.initialized = False

    def apply(self, frame: np.ndarray,
              detections: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute camera motion transformation between current and previous frame.

        Args:
            frame: Current BGR frame
            detections: Optional (N, 4) array of detections in xyxy format
                       to mask out moving objects

        Returns:
            2x3 affine transformation matrix (identity if first frame or failed)
        """
        if self.method == 'hybrid':
            return self._apply_hybrid(frame, detections)
        elif self.method == 'orb':
            return self._apply_orb(frame, detections)
        elif self.method == 'sift':
            return self._apply_sift(frame, detections)
        elif self.method == 'ecc':
            return self._apply_ecc(frame, detections)
        elif self.method == 'sparseOptFlow':
            return self._apply_optical_flow(frame, detections)
        else:
            return np.eye(2, 3, dtype=np.float32)

    def _apply_hybrid(self, frame: np.ndarray,
                      detections: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Hybrid ORB/ECC strategy: try ORB first, fall back to ECC.

        This provides fast computation with ORB while having ECC as
        a robust fallback for challenging scenes.
        """
        # Try ORB first
        H = self._apply_orb(frame, detections)

        # Check if ORB failed (returns identity or near-identity)
        if self._is_identity(H):
            # Fall back to ECC
            H_ecc = self._apply_ecc(frame, detections)
            if not self._is_identity(H_ecc):
                return H_ecc

        return H

    def _apply_orb(self, frame: np.ndarray,
                   detections: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply ORB feature matching for CMC."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale
        if self.downscale > 1:
            gray = cv2.resize(gray, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Create mask (exclude detection regions)
        mask = np.zeros_like(gray)
        mask[int(0.02 * height):int(0.98 * height),
             int(0.02 * width):int(0.98 * width)] = 255

        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        # Detect keypoints
        keypoints = self.orb_detector.detect(gray, mask)
        keypoints, descriptors = self.orb_extractor.compute(gray, keypoints)

        # Handle first frame
        if not self.initialized:
            self.prev_frame = gray.copy()
            self.prev_keypoints = copy.copy(keypoints)
            self.prev_descriptors = copy.copy(descriptors)
            self.initialized = True
            return H

        # Handle missing descriptors
        if descriptors is None or self.prev_descriptors is None:
            self._update_state_orb(gray, keypoints, descriptors)
            return H

        if len(descriptors) < 4 or len(self.prev_descriptors) < 4:
            self._update_state_orb(gray, keypoints, descriptors)
            return H

        # Match descriptors
        try:
            knn_matches = self.orb_matcher.knnMatch(
                self.prev_descriptors, descriptors, k=2
            )
        except cv2.error:
            self._update_state_orb(gray, keypoints, descriptors)
            return H

        if len(knn_matches) == 0:
            self._update_state_orb(gray, keypoints, descriptors)
            return H

        # Filter matches using Lowe's ratio test and spatial distance
        matches = []
        spatial_distances = []
        max_spatial_distance = 0.25 * np.array([width, height])

        for match in knn_matches:
            if len(match) < 2:
                continue
            m, n = match
            if m.distance < 0.9 * n.distance:
                prev_pt = self.prev_keypoints[m.queryIdx].pt
                curr_pt = keypoints[m.trainIdx].pt

                spatial_dist = (prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1])

                if (abs(spatial_dist[0]) < max_spatial_distance[0] and
                    abs(spatial_dist[1]) < max_spatial_distance[1]):
                    spatial_distances.append(spatial_dist)
                    matches.append(m)

        if len(matches) < 4:
            self._update_state_orb(gray, keypoints, descriptors)
            return H

        # Filter outliers using mean/std
        spatial_distances = np.array(spatial_distances)
        mean_dist = np.mean(spatial_distances, axis=0)
        std_dist = np.std(spatial_distances, axis=0)

        inliers = np.abs(spatial_distances - mean_dist) < 2.5 * std_dist

        prev_points = []
        curr_points = []
        for i, m in enumerate(matches):
            if inliers[i, 0] and inliers[i, 1]:
                prev_points.append(self.prev_keypoints[m.queryIdx].pt)
                curr_points.append(keypoints[m.trainIdx].pt)

        prev_points = np.array(prev_points, dtype=np.float32)
        curr_points = np.array(curr_points, dtype=np.float32)

        # Estimate affine transform
        if len(prev_points) >= 4:
            H, _ = cv2.estimateAffinePartial2D(
                prev_points, curr_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_reproj_threshold
            )
            if H is not None:
                # Handle downscale
                if self.downscale > 1:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
            else:
                H = np.eye(2, 3, dtype=np.float32)

        self._update_state_orb(gray, keypoints, descriptors)
        return H

    def _apply_sift(self, frame: np.ndarray,
                    detections: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply SIFT feature matching for CMC (more accurate, slower)."""
        # Similar to ORB but using SIFT
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        if self.downscale > 1:
            gray = cv2.resize(gray, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Use SIFT detector
        sift = cv2.SIFT_create(
            nOctaveLayers=3,
            contrastThreshold=0.02,
            edgeThreshold=20
        )

        mask = np.zeros_like(gray)
        mask[int(0.02 * height):int(0.98 * height),
             int(0.02 * width):int(0.98 * width)] = 255

        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        keypoints, descriptors = sift.detectAndCompute(gray, mask)

        if not self.initialized:
            self.prev_frame = gray.copy()
            self.prev_keypoints = copy.copy(keypoints)
            self.prev_descriptors = copy.copy(descriptors)
            self.initialized = True
            return H

        if descriptors is None or self.prev_descriptors is None:
            self._update_state_orb(gray, keypoints, descriptors)
            return H

        # Match with L2 norm
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        try:
            knn_matches = matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        except cv2.error:
            self._update_state_orb(gray, keypoints, descriptors)
            return H

        # Same filtering as ORB
        matches = []
        spatial_distances = []
        max_spatial_distance = 0.25 * np.array([width, height])

        for match in knn_matches:
            if len(match) < 2:
                continue
            m, n = match
            if m.distance < 0.75 * n.distance:  # Stricter for SIFT
                prev_pt = self.prev_keypoints[m.queryIdx].pt
                curr_pt = keypoints[m.trainIdx].pt
                spatial_dist = (prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1])

                if (abs(spatial_dist[0]) < max_spatial_distance[0] and
                    abs(spatial_dist[1]) < max_spatial_distance[1]):
                    spatial_distances.append(spatial_dist)
                    matches.append(m)

        if len(matches) >= 4:
            spatial_distances = np.array(spatial_distances)
            mean_dist = np.mean(spatial_distances, axis=0)
            std_dist = np.std(spatial_distances, axis=0)
            inliers = np.abs(spatial_distances - mean_dist) < 2.5 * std_dist

            prev_points = []
            curr_points = []
            for i, m in enumerate(matches):
                if inliers[i, 0] and inliers[i, 1]:
                    prev_points.append(self.prev_keypoints[m.queryIdx].pt)
                    curr_points.append(keypoints[m.trainIdx].pt)

            prev_points = np.array(prev_points, dtype=np.float32)
            curr_points = np.array(curr_points, dtype=np.float32)

            if len(prev_points) >= 4:
                H, _ = cv2.estimateAffinePartial2D(
                    prev_points, curr_points, method=cv2.RANSAC
                )
                if H is not None and self.downscale > 1:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale

        self._update_state_orb(gray, keypoints, descriptors)
        return H if H is not None else np.eye(2, 3, dtype=np.float32)

    def _apply_ecc(self, frame: np.ndarray,
                   detections: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply ECC (Enhanced Correlation Coefficient) alignment."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        if self.downscale > 1:
            gray = cv2.GaussianBlur(gray, (3, 3), 1.5)
            gray = cv2.resize(gray, (width // self.downscale, height // self.downscale))

        if not self.initialized:
            self.prev_frame = gray.copy()
            self.initialized = True
            return H

        try:
            _, H = cv2.findTransformECC(
                self.prev_frame, gray, H,
                self.ecc_warp_mode, self.ecc_criteria, None, 1
            )
            if self.downscale > 1:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        except cv2.error:
            H = np.eye(2, 3, dtype=np.float32)

        self.prev_frame = gray.copy()
        return H

    def _apply_optical_flow(self, frame: np.ndarray,
                            detections: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply sparse Lucas-Kanade optical flow."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        if self.downscale > 1:
            gray = cv2.resize(gray, (width // self.downscale, height // self.downscale))

        keypoints = cv2.goodFeaturesToTrack(gray, mask=None, **self.flow_feature_params)

        if not self.initialized:
            self.prev_frame = gray.copy()
            self.prev_keypoints = copy.copy(keypoints)
            self.initialized = True
            return H

        if keypoints is None or self.prev_keypoints is None:
            self.prev_frame = gray.copy()
            self.prev_keypoints = copy.copy(keypoints)
            return H

        # Compute optical flow
        matched_kps, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_keypoints, None
        )

        prev_points = []
        curr_points = []
        for i in range(len(status)):
            if status[i]:
                prev_points.append(self.prev_keypoints[i])
                curr_points.append(matched_kps[i])

        prev_points = np.array(prev_points, dtype=np.float32)
        curr_points = np.array(curr_points, dtype=np.float32)

        if len(prev_points) >= 4:
            H, _ = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC)
            if H is not None and self.downscale > 1:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale

        self.prev_frame = gray.copy()
        self.prev_keypoints = copy.copy(keypoints)
        return H if H is not None else np.eye(2, 3, dtype=np.float32)

    def _update_state_orb(self, gray: np.ndarray, keypoints, descriptors) -> None:
        """Update ORB state for next iteration."""
        self.prev_frame = gray.copy()
        self.prev_keypoints = copy.copy(keypoints)
        self.prev_descriptors = copy.copy(descriptors)

    def _is_identity(self, H: np.ndarray, threshold: float = 1e-3) -> bool:
        """Check if transformation is near-identity."""
        identity = np.eye(2, 3, dtype=np.float32)
        return np.allclose(H, identity, atol=threshold)


def apply_cmc_to_tracks(tracks: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply camera motion compensation to track positions.

    Used to update Kalman filter states when camera moves.

    Args:
        tracks: (N, 8) array of track states [x, y, w, h, dx, dy, dw, dh]
        H: 2x3 affine transformation matrix

    Returns:
        Updated track states
    """
    if tracks.size == 0:
        return tracks

    tracks = tracks.copy()

    # Extract rotation/scale and translation
    R = H[:2, :2]  # 2x2 rotation/scale
    t = H[:2, 2]   # 2x1 translation

    # Build 8x8 transformation for full state
    # State: [x, y, w, h, dx, dy, dw, dh]
    # Position (x, y) and velocity (dx, dy) are transformed
    # Size (w, h) and size velocity (dw, dh) are only scaled

    R8x8 = np.kron(np.eye(4, dtype=np.float32), R)

    # Apply to each track
    for i in range(len(tracks)):
        mean = tracks[i]
        mean = R8x8.dot(mean)
        mean[:2] += t
        tracks[i] = mean

    return tracks


def create_cmc(config: Optional[CMCConfig] = None) -> CameraMotionCompensator:
    """
    Factory function to create a CMC instance.

    Args:
        config: Optional CMC config. Uses defaults if None.

    Returns:
        Configured CameraMotionCompensator instance
    """
    if config is None:
        config = CMCConfig()
    return CameraMotionCompensator(config)
