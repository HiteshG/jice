"""
McByte Tracker with mask-conditioned association.

Multi-object tracking with 4-step association process, integrating mask cues
from SAM2/CUTIE for improved occlusion handling.

Ported from McByte/yolox/tracker/mcbyte_tracker.py.
"""

from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

from .config import TrackingConfig
from .data_types import Detection, Track, TrackState as DataTrackState


# Constants from McByte
MIN_MASK_AVG_CONF = 0.6
MIN_MM1 = 0.9  # Mask coverage ratio threshold
MIN_MM2 = 0.05  # Mask fill ratio threshold

MAX_COST_1ST_ASSOC_STEP = 0.9
MAX_COST_2ND_ASSOC_STEP = 0.5
MAX_COST_UNCONFIRMED_ASSOC_STEP = 0.7


class TrackState(IntEnum):
    """Internal track state for ByteTrack."""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class KalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space.

    8-dimensional state space: [x, y, w, h, vx, vy, vw, vh]
    where (x, y) is center, (w, h) is size, and v* are velocities.
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Motion matrix (constant velocity model)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.

        Args:
            measurement: Bounding box [x, y, w, h] in center format

        Returns:
            Tuple of (mean, covariance)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self._project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))

        return new_mean, new_covariance

    def _project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        ))
        return mean, covariance + innovation_cov

    def multi_predict(self, means: np.ndarray,
                      covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized prediction for multiple tracks."""
        std_pos = [
            self._std_weight_position * means[:, 2],
            self._std_weight_position * means[:, 3],
            self._std_weight_position * means[:, 2],
            self._std_weight_position * means[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * means[:, 2],
            self._std_weight_velocity * means[:, 3],
            self._std_weight_velocity * means[:, 2],
            self._std_weight_velocity * means[:, 3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = np.array([np.diag(sqr[i]) for i in range(len(means))])

        means = np.dot(means, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariances).transpose((1, 0, 2))
        covariances = np.dot(left, self._motion_mat.T) + motion_cov

        return means, covariances


class STrack:
    """
    Single track object with Kalman filter state.

    State: [x, y, w, h, vx, vy, vw, vh] in center format
    """

    _count = 0
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float, class_id: int = 0):
        """
        Initialize a track.

        Args:
            tlwh: Bounding box [top, left, width, height]
            score: Detection confidence
            class_id: Object class ID
        """
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

        self.score = score
        self.class_id = class_id
        self.tracklet_len = 0
        self.track_id = 0

        self.state = TrackState.New
        self.frame_id = 0
        self.start_frame = 0

        # Last detected bbox for mask creation
        self.last_det_tlwh = tlwh.copy()

    @staticmethod
    def next_id() -> int:
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id() -> None:
        STrack._count = 0

    def predict(self) -> None:
        """Predict next state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: List['STrack']) -> None:
        """Vectorized prediction for multiple tracks."""
        if len(stracks) == 0:
            return

        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0

        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: List['STrack'], H: np.ndarray = None) -> None:
        """
        Apply camera motion compensation to track states.

        Args:
            stracks: List of tracks to update
            H: 2x3 affine transformation matrix
        """
        if H is None:
            H = np.eye(2, 3)

        if len(stracks) == 0:
            return

        R = H[:2, :2]
        R8x8 = np.kron(np.eye(4, dtype=float), R)
        t = H[:2, 2]

        for st in stracks:
            mean = R8x8.dot(st.mean)
            mean[:2] += t
            cov = R8x8.dot(st.covariance).dot(R8x8.T)
            st.mean = mean
            st.covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False) -> None:
        """Re-activate a lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.last_det_tlwh = new_track.tlwh.copy()

    def update(self, new_track: 'STrack', frame_id: int) -> None:
        """Update a matched track."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.last_det_tlwh = new_track.tlwh.copy()

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @property
    def tlwh(self) -> np.ndarray:
        """Get current position in tlwh format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Get current position in tlbr format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get current position in xywh (center) format."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert tlwh to center format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        """Convert tlbr to tlwh format."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self) -> str:
        return f'STrack_{self.track_id}_({self.start_frame}-{self.end_frame})'


class McByteTracker:
    """
    McByte tracker with mask-conditioned association.

    Uses 4-step association process:
    1. High-conf detections vs tracked+lost (with mask cues)
    2. Low-conf detections vs remaining tracked
    3. Unconfirmed tracklets matching
    4. New tracklet creation
    """

    def __init__(self, config: TrackingConfig, frame_rate: int = 30):
        """
        Initialize the tracker.

        Args:
            config: Tracking configuration
            frame_rate: Video frame rate
        """
        self.config = config
        self.frame_rate = frame_rate

        # Track storage
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.det_thresh = config.track_thresh + 0.1

        # Buffer size for lost tracks (1 second default)
        buffer = config.track_buffer if config.track_buffer else frame_rate
        self.buffer_size = int(frame_rate / 30.0 * buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

        # Reset track IDs
        STrack.reset_id()

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_id()

    def update(
        self,
        detections: List[Detection],
        img_size: Tuple[int, int],
        prediction_mask: Optional[np.ndarray] = None,
        tracklet_mask_dict: Optional[Dict[int, int]] = None,
        mask_avg_prob_dict: Optional[Dict[int, float]] = None,
        warp_matrix: Optional[np.ndarray] = None
    ) -> Tuple[List[STrack], List[int], List[STrack]]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects
            img_size: Image size (height, width)
            prediction_mask: Mask prediction from CUTIE
            tracklet_mask_dict: Track ID to mask ID mapping
            mask_avg_prob_dict: Mask ID to average probability mapping
            warp_matrix: Camera motion compensation matrix

        Returns:
            Tuple of (output_tracks, removed_track_ids, new_tracks)
        """
        self.frame_id += 1
        img_h, img_w = img_size

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Convert detections to STracks
        if len(detections) > 0:
            # Split by confidence
            high_conf = [d for d in detections if d.score >= self.config.track_thresh]
            low_conf = [d for d in detections if 0.1 < d.score < self.config.track_thresh]

            det_stracks = [
                STrack(d.bbox_tlwh, d.score, d.class_id) for d in high_conf
            ]
            det_stracks_second = [
                STrack(d.bbox_tlwh, d.score, d.class_id) for d in low_conf
            ]
        else:
            det_stracks = []
            det_stracks_second = []

        # Step 1: Separate unconfirmed and tracked
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: First association with high score detections
        strack_pool = _joint_stracks(tracked_stracks, self.lost_stracks)

        # Kalman prediction
        STrack.multi_predict(strack_pool)

        # Camera motion compensation
        if warp_matrix is not None:
            STrack.multi_gmc(strack_pool, warp_matrix)
            STrack.multi_gmc(unconfirmed, warp_matrix)

        # Compute IoU distance
        dists = _iou_distance(strack_pool, det_stracks)

        # Fuse detection scores
        dists = _fuse_score(dists, det_stracks)

        # Mask-conditioned assignment
        matches, u_track, u_detection = self._conditioned_assignment(
            dists, MAX_COST_1ST_ASSOC_STEP,
            strack_pool, det_stracks,
            prediction_mask, tracklet_mask_dict, mask_avg_prob_dict,
            (img_h, img_w)
        )

        # Update matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = det_stracks[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association with low score detections
        r_tracked_stracks = [
            strack_pool[i] for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]

        dists = _iou_distance(r_tracked_stracks, det_stracks_second)

        matches, u_track, u_detection_second = self._conditioned_assignment(
            dists, MAX_COST_2ND_ASSOC_STEP,
            r_tracked_stracks, det_stracks_second,
            prediction_mask, tracklet_mask_dict, mask_avg_prob_dict,
            (img_h, img_w)
        )

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = det_stracks_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Mark unmatched as lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 4: Deal with unconfirmed tracks
        remaining_dets = [det_stracks[i] for i in u_detection]
        dists = _iou_distance(unconfirmed, remaining_dets)
        dists = _fuse_score(dists, remaining_dets)

        matches, u_unconfirmed, u_detection = self._conditioned_assignment(
            dists, MAX_COST_UNCONFIRMED_ASSOC_STEP,
            unconfirmed, remaining_dets,
            prediction_mask, tracklet_mask_dict, mask_avg_prob_dict,
            (img_h, img_w)
        )

        new_confirmed_tracks = []
        for itracked, idet in matches:
            unconfirmed[itracked].update(remaining_dets[idet], self.frame_id)
            new_confirmed_tracks.append(unconfirmed[itracked])
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 5: Initialize new tracks
        for inew in u_detection:
            track = remaining_dets[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 6: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = _sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = _sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Output activated tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        removed_track_ids = [track.track_id for track in removed_stracks]

        return output_stracks, removed_track_ids, new_confirmed_tracks

    def _conditioned_assignment(
        self,
        dists: np.ndarray,
        max_cost: float,
        strack_pool: List[STrack],
        detections: List[STrack],
        prediction_mask: Optional[np.ndarray],
        tracklet_mask_dict: Optional[Dict[int, int]],
        mask_avg_prob_dict: Optional[Dict[int, float]],
        img_info: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Mask-conditioned assignment using Hungarian algorithm.

        Modifies cost matrix based on mask cues when available.
        """
        if dists.size == 0:
            return [], list(range(len(strack_pool))), list(range(len(detections)))

        dists_cp = np.copy(dists)
        img_h, img_w = img_info

        # Apply mask conditioning if masks available
        if prediction_mask is not None and tracklet_mask_dict is not None:
            for i in range(dists_cp.shape[0]):
                for j in range(dists_cp.shape[1]):
                    if dists[i, j] <= max_cost:
                        # Check if clear match
                        row_matches = np.sum(dists[i, :] <= max_cost)
                        col_matches = np.sum(dists[:, j] <= max_cost)

                        if row_matches == 1 and col_matches == 1:
                            # Clear match - increase cost of others
                            dists_cp[i, :] += 10
                            dists_cp[:, j] += 10
                            dists_cp[i, j] = dists[i, j]
                        else:
                            # Ambiguous - use mask cue
                            strack = strack_pool[i]
                            det = detections[j]

                            strack_id = strack.track_id
                            if strack_id in tracklet_mask_dict:
                                mask_id = tracklet_mask_dict[strack_id]
                                unique_masks = list(np.unique(prediction_mask))[1:]

                                if mask_id in unique_masks:
                                    if mask_avg_prob_dict and mask_id in mask_avg_prob_dict:
                                        if mask_avg_prob_dict[mask_id] >= MIN_MASK_AVG_CONF:
                                            # Compute mask metrics
                                            x, y, w, h = det.tlwh
                                            x, y = max(0, int(x)), max(0, int(y))
                                            x2 = min(img_w, int(x + w))
                                            y2 = min(img_h, int(y + h))

                                            bbox_mask = prediction_mask[y:y2, x:x2]
                                            mask_in_bbox = np.sum(bbox_mask == mask_id)
                                            total_mask = np.sum(prediction_mask == mask_id)
                                            bbox_area = (y2 - y) * (x2 - x)

                                            if total_mask > 0 and bbox_area > 0:
                                                mm1 = mask_in_bbox / total_mask
                                                mm2 = mask_in_bbox / bbox_area

                                                if mm2 >= MIN_MM2 and mm1 >= MIN_MM1:
                                                    dists_cp[i, j] -= mm2

        # Hungarian algorithm
        matches, u_track, u_detection = _linear_assignment(dists_cp, max_cost)

        return matches, u_track, u_detection


# Helper functions

def _joint_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    """Join two track lists without duplicates."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def _sub_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    """Subtract tlistb from tlista."""
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        if t.track_id in stracks:
            del stracks[t.track_id]
    return list(stracks.values())


def _iou_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
    """Compute IoU distance matrix."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.empty((len(atracks), len(btracks)), dtype=np.float32)

    atlbrs = np.asarray([t.tlbr for t in atracks])
    btlbrs = np.asarray([t.tlbr for t in btracks])

    ious = _compute_iou_batch(atlbrs, btlbrs)
    return 1 - ious


def _compute_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes."""
    n, m = len(boxes1), len(boxes2)
    ious = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            b1 = boxes1[i]
            b2 = boxes2[j]

            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = area1 + area2 - inter

            ious[i, j] = inter / union if union > 0 else 0

    return ious


def _fuse_score(dists: np.ndarray, detections: List[STrack]) -> np.ndarray:
    """Fuse detection scores into distance matrix."""
    if dists.size == 0:
        return dists

    det_scores = np.array([d.score for d in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(dists.shape[0], axis=0)
    dists = dists * (1 - det_scores)
    return dists


def _linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Linear assignment with threshold.

    Returns:
        Tuple of (matches, unmatched_a, unmatched_b)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    # Use scipy's linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    return matches, unmatched_a, unmatched_b


def create_tracker(config: Optional[TrackingConfig] = None,
                   frame_rate: int = 30) -> McByteTracker:
    """
    Factory function to create a tracker.

    Args:
        config: Optional tracking config. Uses defaults if None.
        frame_rate: Video frame rate

    Returns:
        Configured McByteTracker instance
    """
    if config is None:
        config = TrackingConfig()
    return McByteTracker(config, frame_rate)
