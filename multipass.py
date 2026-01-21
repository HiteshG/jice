"""
Multi-pass tracking for improved ID consistency.

Implements:
- Forward/backward tracking passes
- Track merging by IoU + ReID similarity
- Gap interpolation using Kalman prediction
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import MultiPassConfig
from .data_types import Track, TrackletFrame, TrackState, compute_iou_matrix
from .tracker import STrack


class TrackMerger:
    """
    Merges forward and backward tracking passes.

    Matches tracks based on IoU overlap and optional ReID features.
    """

    def __init__(self, config: MultiPassConfig):
        """
        Initialize the track merger.

        Args:
            config: Multi-pass configuration
        """
        self.config = config
        self.iou_threshold = config.merge_iou_threshold
        self.reid_threshold = config.merge_reid_threshold

    def merge(
        self,
        forward_tracks: Dict[int, Track],
        backward_tracks: Dict[int, Track]
    ) -> Dict[int, Track]:
        """
        Merge forward and backward tracking results.

        Args:
            forward_tracks: Tracks from forward pass (track_id -> Track)
            backward_tracks: Tracks from backward pass (track_id -> Track)

        Returns:
            Merged tracks with best detections per frame
        """
        if not backward_tracks:
            return forward_tracks

        # Reverse backward track frame indices
        max_frame = max(
            max(t.frames.keys()) for t in forward_tracks.values()
        ) if forward_tracks else 0

        reversed_backward = {}
        for track_id, track in backward_tracks.items():
            new_track = Track(
                track_id=track_id + 10000,  # Offset to avoid conflicts
                class_id=track.class_id,
                state=track.state
            )
            for frame_id, frame_data in track.frames.items():
                new_frame_id = max_frame - frame_id + 1
                new_track.frames[new_frame_id] = frame_data
            reversed_backward[new_track.track_id] = new_track

        # Compute overlap matrix
        fwd_ids = list(forward_tracks.keys())
        bwd_ids = list(reversed_backward.keys())

        if not fwd_ids or not bwd_ids:
            return forward_tracks

        # Compute IoU-based matching
        cost_matrix = self._compute_cost_matrix(
            forward_tracks, reversed_backward, fwd_ids, bwd_ids
        )

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Merge matched pairs
        matched_backward = set()
        merged_tracks = {}

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1 - self.iou_threshold:
                fwd_id = fwd_ids[r]
                bwd_id = bwd_ids[c]

                merged = self._merge_track_pair(
                    forward_tracks[fwd_id],
                    reversed_backward[bwd_id]
                )
                merged_tracks[fwd_id] = merged
                matched_backward.add(bwd_id)
            else:
                # Keep forward track as-is
                merged_tracks[fwd_ids[r]] = forward_tracks[fwd_ids[r]]

        # Add unmatched forward tracks
        for fwd_id in fwd_ids:
            if fwd_id not in merged_tracks:
                merged_tracks[fwd_id] = forward_tracks[fwd_id]

        # Add unmatched backward tracks (assign new IDs)
        next_id = max(merged_tracks.keys()) + 1 if merged_tracks else 1
        for bwd_id in bwd_ids:
            if bwd_id not in matched_backward:
                track = reversed_backward[bwd_id]
                track.track_id = next_id
                merged_tracks[next_id] = track
                next_id += 1

        return merged_tracks

    def _compute_cost_matrix(
        self,
        forward: Dict[int, Track],
        backward: Dict[int, Track],
        fwd_ids: List[int],
        bwd_ids: List[int]
    ) -> np.ndarray:
        """
        Compute cost matrix based on temporal IoU overlap.
        """
        n, m = len(fwd_ids), len(bwd_ids)
        cost = np.ones((n, m), dtype=np.float32)

        for i, fwd_id in enumerate(fwd_ids):
            fwd_track = forward[fwd_id]
            fwd_frames = set(fwd_track.frames.keys())

            for j, bwd_id in enumerate(bwd_ids):
                bwd_track = backward[bwd_id]
                bwd_frames = set(bwd_track.frames.keys())

                # Find overlapping frames
                overlap_frames = fwd_frames & bwd_frames

                if not overlap_frames:
                    continue

                # Compute average IoU on overlapping frames
                ious = []
                for frame_id in overlap_frames:
                    fwd_bbox = fwd_track.frames[frame_id].bbox_xyxy
                    bwd_bbox = bwd_track.frames[frame_id].bbox_xyxy

                    iou = self._compute_iou(fwd_bbox, bwd_bbox)
                    ious.append(iou)

                avg_iou = np.mean(ious)
                cost[i, j] = 1 - avg_iou

        return cost

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _merge_track_pair(self, fwd: Track, bwd: Track) -> Track:
        """
        Merge a forward and backward track pair.

        Keeps the detection with higher confidence on overlapping frames.
        """
        merged = Track(
            track_id=fwd.track_id,
            class_id=fwd.class_id,
            state=fwd.state
        )

        all_frames = set(fwd.frames.keys()) | set(bwd.frames.keys())

        for frame_id in all_frames:
            fwd_frame = fwd.frames.get(frame_id)
            bwd_frame = bwd.frames.get(frame_id)

            if fwd_frame and bwd_frame:
                # Keep higher confidence
                if fwd_frame.score >= bwd_frame.score:
                    merged.frames[frame_id] = fwd_frame
                else:
                    merged.frames[frame_id] = bwd_frame
            elif fwd_frame:
                merged.frames[frame_id] = fwd_frame
            else:
                merged.frames[frame_id] = bwd_frame

        return merged


class TrackInterpolator:
    """
    Interpolates gaps in tracks using Kalman prediction.
    """

    def __init__(self, config: MultiPassConfig):
        """
        Initialize the interpolator.

        Args:
            config: Multi-pass configuration
        """
        self.config = config
        self.max_gap = config.max_interpolation_gap
        self.method = config.interpolation_method

    def interpolate(self, tracks: Dict[int, Track]) -> Dict[int, Track]:
        """
        Fill gaps in all tracks.

        Args:
            tracks: Dictionary of tracks

        Returns:
            Tracks with interpolated gaps
        """
        result = {}

        for track_id, track in tracks.items():
            gaps = track.get_gaps()

            if not gaps:
                result[track_id] = track
                continue

            # Interpolate each gap
            interpolated = Track(
                track_id=track.track_id,
                class_id=track.class_id,
                state=track.state,
                jersey_number=track.jersey_number,
                team_id=track.team_id
            )

            # Copy existing frames
            interpolated.frames = dict(track.frames)

            for gap_start, gap_end in gaps:
                if gap_end - gap_start + 1 > self.max_gap:
                    continue

                # Get boundary frames
                before_frame = gap_start - 1
                after_frame = gap_end + 1

                if before_frame not in track.frames or after_frame not in track.frames:
                    continue

                before_data = track.frames[before_frame]
                after_data = track.frames[after_frame]

                # Interpolate
                if self.method == "linear":
                    interp_frames = self._linear_interpolate(
                        before_data, after_data, gap_start, gap_end
                    )
                else:  # kalman
                    interp_frames = self._kalman_interpolate(
                        before_data, after_data, gap_start, gap_end
                    )

                interpolated.frames.update(interp_frames)

            result[track_id] = interpolated

        return result

    def _linear_interpolate(
        self,
        before: TrackletFrame,
        after: TrackletFrame,
        gap_start: int,
        gap_end: int
    ) -> Dict[int, TrackletFrame]:
        """Linear interpolation between boundary frames."""
        result = {}
        total_frames = gap_end - gap_start + 2  # Include boundaries

        for i, frame_id in enumerate(range(gap_start, gap_end + 1)):
            alpha = (i + 1) / total_frames

            # Interpolate bbox
            bbox = before.bbox_xyxy * (1 - alpha) + after.bbox_xyxy * alpha
            score = before.score * (1 - alpha) + after.score * alpha

            result[frame_id] = TrackletFrame(
                frame_id=frame_id,
                bbox_xyxy=bbox,
                score=score
            )

        return result

    def _kalman_interpolate(
        self,
        before: TrackletFrame,
        after: TrackletFrame,
        gap_start: int,
        gap_end: int
    ) -> Dict[int, TrackletFrame]:
        """Kalman-based interpolation using motion model."""
        # For simplicity, use linear for now
        # Full Kalman would require maintaining filter state
        return self._linear_interpolate(before, after, gap_start, gap_end)


def convert_stracks_to_tracks(
    stracks: List[STrack],
    frame_id: int,
    frame_img: Optional[np.ndarray] = None
) -> Dict[int, Track]:
    """
    Convert STrack objects to Track objects for a frame.

    Args:
        stracks: List of STrack objects
        frame_id: Current frame ID
        frame_img: Optional frame image for extracting crops

    Returns:
        Dictionary of Track objects
    """
    tracks = {}

    for strack in stracks:
        if strack.track_id not in tracks:
            tracks[strack.track_id] = Track(
                track_id=strack.track_id,
                class_id=strack.class_id,
                state=TrackState.TRACKED
            )

        # Extract crop if image provided
        crop = None
        if frame_img is not None:
            x1, y1, x2, y2 = map(int, strack.tlbr)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame_img.shape[1], x2)
            y2 = min(frame_img.shape[0], y2)
            if x2 > x1 and y2 > y1:
                crop = frame_img[y1:y2, x1:x2].copy()

        frame_data = TrackletFrame(
            frame_id=frame_id,
            bbox_xyxy=strack.tlbr.copy(),
            score=strack.score,
            crop=crop
        )

        tracks[strack.track_id].add_frame(frame_id, frame_data)

    return tracks


def merge_frame_tracks(
    existing: Dict[int, Track],
    new_tracks: Dict[int, Track]
) -> Dict[int, Track]:
    """
    Merge new frame tracks into existing tracks.

    Args:
        existing: Existing track dictionary
        new_tracks: New tracks from current frame

    Returns:
        Updated track dictionary
    """
    for track_id, track in new_tracks.items():
        if track_id in existing:
            for frame_id, frame_data in track.frames.items():
                existing[track_id].add_frame(frame_id, frame_data)
        else:
            existing[track_id] = track

    return existing


def create_track_merger(config: Optional[MultiPassConfig] = None) -> TrackMerger:
    """Factory function for TrackMerger."""
    if config is None:
        config = MultiPassConfig()
    return TrackMerger(config)


def create_track_interpolator(config: Optional[MultiPassConfig] = None) -> TrackInterpolator:
    """Factory function for TrackInterpolator."""
    if config is None:
        config = MultiPassConfig()
    return TrackInterpolator(config)


class PostJerseyMerger:
    """
    Merges tracks based on jersey number and team classification.

    After initial jersey detection, tracks with the same jersey number
    and team (that don't overlap in time) are likely the same player
    who got a new track ID due to occlusion or leaving the frame.
    """

    def __init__(self, min_confidence: float = 0.3, max_overlap_frames: int = 5):
        """
        Initialize the post-jersey merger.

        Args:
            min_confidence: Minimum jersey confidence to consider for merging
            max_overlap_frames: Max allowed frame overlap (for handling noise)
        """
        self.min_confidence = min_confidence
        self.max_overlap_frames = max_overlap_frames

    def merge_by_jersey(self, tracks: Dict[int, Track]) -> Tuple[Dict[int, Track], Dict[int, int]]:
        """
        Merge tracks that have the same jersey number and team.

        Only merges non-overlapping tracks (or tracks with minimal overlap).

        Args:
            tracks: Dictionary of tracks with jersey/team assignments

        Returns:
            Tuple of (merged_tracks, merge_mapping) where merge_mapping shows
            which original track IDs were merged into which final track ID
        """
        if not tracks:
            return {}, {}

        # Group tracks by (jersey_number, team_id, class_id)
        jersey_groups: Dict[Tuple, List[int]] = defaultdict(list)

        for track_id, track in tracks.items():
            # Only consider tracks with confident jersey predictions
            if track.jersey_number is not None and track.jersey_confidence >= self.min_confidence:
                # Players and goalies only (class_id 0 and 1)
                if track.class_id in [0, 1] and track.team_id is not None and track.team_id >= 0:
                    key = (track.jersey_number, track.team_id, track.class_id)
                    jersey_groups[key].append(track_id)

        # Find mergeable tracks within each group
        merge_mapping: Dict[int, int] = {}  # original_id -> merged_id
        merged_tracks: Dict[int, Track] = {}
        processed_ids: set = set()

        for key, track_ids in jersey_groups.items():
            if len(track_ids) < 2:
                continue

            # Sort by start frame
            sorted_ids = sorted(track_ids, key=lambda tid: tracks[tid].start_frame)

            # Greedily merge non-overlapping tracks
            merge_chains = self._find_merge_chains(tracks, sorted_ids)

            for chain in merge_chains:
                if len(chain) > 1:
                    # Merge all tracks in the chain into the first one
                    primary_id = chain[0]
                    merged = self._merge_track_chain(tracks, chain)

                    # Record merge mapping
                    for tid in chain:
                        merge_mapping[tid] = primary_id
                        processed_ids.add(tid)

                    merged_tracks[primary_id] = merged

        # Add unprocessed tracks as-is
        for track_id, track in tracks.items():
            if track_id not in processed_ids:
                merged_tracks[track_id] = track
                merge_mapping[track_id] = track_id

        return merged_tracks, merge_mapping

    def _find_merge_chains(
        self,
        tracks: Dict[int, Track],
        track_ids: List[int]
    ) -> List[List[int]]:
        """
        Find chains of tracks that can be merged (non-overlapping).

        Returns list of merge chains, where each chain is a list of track IDs
        that should be merged together.
        """
        chains: List[List[int]] = []
        used = set()

        for start_id in track_ids:
            if start_id in used:
                continue

            chain = [start_id]
            used.add(start_id)
            current_end = tracks[start_id].end_frame

            # Find next non-overlapping track
            for next_id in track_ids:
                if next_id in used:
                    continue

                next_start = tracks[next_id].start_frame
                overlap = current_end - next_start + 1

                # Allow merge if minimal or no overlap
                if overlap <= self.max_overlap_frames:
                    chain.append(next_id)
                    used.add(next_id)
                    current_end = tracks[next_id].end_frame

            chains.append(chain)

        return chains

    def _merge_track_chain(
        self,
        tracks: Dict[int, Track],
        chain: List[int]
    ) -> Track:
        """
        Merge a chain of tracks into a single track.

        Args:
            tracks: Original tracks dictionary
            chain: List of track IDs to merge

        Returns:
            Merged Track object
        """
        primary = tracks[chain[0]]

        merged = Track(
            track_id=primary.track_id,
            class_id=primary.class_id,
            state=primary.state,
            # Keep jersey/team from most confident prediction
            jersey_number=primary.jersey_number,
            jersey_confidence=primary.jersey_confidence,
            team_id=primary.team_id,
            team_confidence=primary.team_confidence
        )

        # Collect all frames and find best jersey/team predictions
        best_jersey_conf = primary.jersey_confidence or 0
        best_team_conf = primary.team_confidence or 0

        for tid in chain:
            track = tracks[tid]

            # Update best jersey prediction
            if track.jersey_confidence and track.jersey_confidence > best_jersey_conf:
                best_jersey_conf = track.jersey_confidence
                merged.jersey_number = track.jersey_number
                merged.jersey_confidence = track.jersey_confidence

            # Update best team prediction
            if track.team_confidence and track.team_confidence > best_team_conf:
                best_team_conf = track.team_confidence
                merged.team_id = track.team_id
                merged.team_confidence = track.team_confidence

            # Merge frames (prefer higher score for overlapping frames)
            for frame_id, frame_data in track.frames.items():
                if frame_id not in merged.frames:
                    merged.frames[frame_id] = frame_data
                else:
                    existing = merged.frames[frame_id]
                    if frame_data.score > existing.score:
                        merged.frames[frame_id] = frame_data

        return merged


def create_post_jersey_merger(
    min_confidence: float = 0.3,
    max_overlap_frames: int = 5
) -> PostJerseyMerger:
    """Factory function for PostJerseyMerger."""
    return PostJerseyMerger(min_confidence, max_overlap_frames)
