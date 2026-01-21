"""
Shared data structures for the unified pipeline.

Defines Detection, Track, TrackState, FrameResult and other core types.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class TrackState(Enum):
    """Track lifecycle states."""
    NEW = auto()        # Just created, not yet confirmed
    TRACKED = auto()    # Actively tracked with recent detections
    LOST = auto()       # Temporarily lost, may be recovered
    REMOVED = auto()    # Permanently removed from tracking


class ObjectClass(Enum):
    """Internal object class IDs."""
    PLAYER = 0
    GOALTENDER = 1
    REFEREE = 2
    PUCK = 3


@dataclass
class Detection:
    """Single detection from YOLO."""
    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: int  # Internal class ID (0=player, 1=goalie, 2=ref, 3=puck)

    @property
    def bbox_tlwh(self) -> np.ndarray:
        """Convert to [top, left, width, height] format."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1])

    @property
    def bbox_xywh(self) -> np.ndarray:
        """Convert to [center_x, center_y, width, height] format."""
        x1, y1, x2, y2 = self.bbox_xyxy
        w, h = x2 - x1, y2 - y1
        return np.array([x1 + w / 2, y1 + h / 2, w, h])

    @property
    def area(self) -> float:
        """Compute bounding box area."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    def iou(self, other: 'Detection') -> float:
        """Compute IoU with another detection."""
        return compute_iou(self.bbox_xyxy, other.bbox_xyxy)


@dataclass
class TrackletFrame:
    """Single frame data within a tracklet."""
    frame_id: int
    bbox_xyxy: np.ndarray
    score: float
    crop: Optional[np.ndarray] = None  # Player crop for jersey/team
    mask: Optional[np.ndarray] = None  # Segmentation mask
    features: Optional[np.ndarray] = None  # ReID features

    @property
    def bbox_tlwh(self) -> np.ndarray:
        """Convert to [top, left, width, height] format."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1])


@dataclass
class Track:
    """
    Complete tracklet across multiple frames.

    Stores all frame-level data and aggregated attributes (jersey, team).
    """
    track_id: int
    class_id: int  # Internal class ID
    state: TrackState = TrackState.NEW

    # Frame-indexed data
    frames: Dict[int, TrackletFrame] = field(default_factory=dict)

    # Aggregated attributes (computed after tracking)
    jersey_number: Optional[int] = None
    jersey_confidence: float = 0.0
    team_id: Optional[int] = None  # 0=away/white, 1=home/colored
    team_confidence: float = 0.0

    # Kalman filter state (for interpolation)
    mean: Optional[np.ndarray] = None  # [x, y, w, h, dx, dy, dw, dh]
    covariance: Optional[np.ndarray] = None

    # Tracking metadata
    start_frame: int = 0
    end_frame: int = 0
    hits: int = 0  # Number of frames with detection
    age: int = 0  # Frames since track started
    time_since_update: int = 0  # Frames since last detection

    def __post_init__(self):
        if self.frames:
            self.start_frame = min(self.frames.keys())
            self.end_frame = max(self.frames.keys())
            self.hits = len(self.frames)

    def add_frame(self, frame_id: int, frame_data: TrackletFrame) -> None:
        """Add a frame to the tracklet."""
        self.frames[frame_id] = frame_data
        self.start_frame = min(self.start_frame, frame_id) if self.frames else frame_id
        self.end_frame = max(self.end_frame, frame_id)
        self.hits = len(self.frames)
        self.time_since_update = 0

    def get_bbox_at_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get bbox at specific frame, or None if not present."""
        if frame_id in self.frames:
            return self.frames[frame_id].bbox_xyxy
        return None

    def get_crops(self) -> List[np.ndarray]:
        """Get all valid (non-None, non-empty) crops from the tracklet."""
        crops = []
        for f in self.frames.values():
            if f.crop is not None and f.crop.size > 0 and f.crop.shape[0] > 0 and f.crop.shape[1] > 0:
                crops.append(f.crop)
        return crops

    def get_features(self) -> Optional[np.ndarray]:
        """Get stacked features array from all frames."""
        features = [f.features for f in self.frames.values() if f.features is not None]
        if features:
            return np.stack(features)
        return None

    @property
    def frame_ids(self) -> List[int]:
        """Get sorted list of frame IDs."""
        return sorted(self.frames.keys())

    @property
    def duration(self) -> int:
        """Track duration in frames."""
        return self.end_frame - self.start_frame + 1 if self.frames else 0

    @property
    def coverage(self) -> float:
        """Fraction of frames with detections."""
        if self.duration == 0:
            return 0.0
        return self.hits / self.duration

    def has_gap(self, max_gap: int = 1) -> bool:
        """Check if track has gaps larger than max_gap frames."""
        if len(self.frames) < 2:
            return False
        frame_ids = self.frame_ids
        for i in range(len(frame_ids) - 1):
            if frame_ids[i + 1] - frame_ids[i] > max_gap:
                return True
        return False

    def get_gaps(self) -> List[Tuple[int, int]]:
        """Get list of (start, end) gaps in the track."""
        gaps = []
        if len(self.frames) < 2:
            return gaps
        frame_ids = self.frame_ids
        for i in range(len(frame_ids) - 1):
            if frame_ids[i + 1] - frame_ids[i] > 1:
                gaps.append((frame_ids[i] + 1, frame_ids[i + 1] - 1))
        return gaps


@dataclass
class FrameResult:
    """Results for a single frame."""
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    affine_matrix: Optional[np.ndarray] = None  # CMC transform
    masks: Optional[np.ndarray] = None  # All masks for frame
    mask_ids: Optional[List[int]] = None  # Track IDs corresponding to masks

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None


@dataclass
class VideoMetadata:
    """Video file metadata."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float

    @classmethod
    def from_video(cls, video_path: str) -> 'VideoMetadata':
        """Extract metadata from video file."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return cls(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration
        )


@dataclass
class MaskMetrics:
    """Metrics for mask-detection matching."""
    mm1: float  # Mask coverage ratio (mc in paper)
    mm2: float  # Mask fill ratio (mf in paper)
    avg_confidence: float

    def is_valid(self, min_mm1: float = 0.9, min_mm2: float = 0.05,
                 min_conf: float = 0.6) -> bool:
        """Check if mask metrics meet thresholds."""
        return (self.mm1 >= min_mm1 and
                self.mm2 >= min_mm2 and
                self.avg_confidence >= min_conf)


@dataclass
class JerseyPrediction:
    """Single jersey number prediction."""
    number: Optional[int]
    confidence: float
    char_confidences: Optional[List[float]] = None  # Per-character confidences
    frame_id: Optional[int] = None
    is_legible: bool = True


@dataclass
class TeamAssignment:
    """Team classification result."""
    team_id: int  # 0=away/white, 1=home/colored, -1=unknown
    confidence: float
    is_outlier: bool = False
    method: str = "unknown"  # "hybrid", "robust", "fallback"


# Utility functions

def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute IoU between two bboxes in xyxy format.

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bboxes.

    Args:
        bboxes1: (N, 4) array in xyxy format
        bboxes2: (M, 4) array in xyxy format

    Returns:
        (N, M) IoU matrix
    """
    n, m = len(bboxes1), len(bboxes2)
    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(bboxes1[i], bboxes2[j])

    return iou_matrix


def tlwh_to_xyxy(tlwh: np.ndarray) -> np.ndarray:
    """Convert [top, left, width, height] to [x1, y1, x2, y2]."""
    x1, y1, w, h = tlwh
    return np.array([x1, y1, x1 + w, y1 + h])


def xyxy_to_tlwh(xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [top, left, width, height]."""
    x1, y1, x2, y2 = xyxy
    return np.array([x1, y1, x2 - x1, y2 - y1])


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert [center_x, center_y, width, height] to [x1, y1, x2, y2]."""
    cx, cy, w, h = xywh
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [center_x, center_y, width, height]."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w/2, y1 + h/2, w, h])


def clip_bbox(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """Clip bbox to image boundaries."""
    bbox = bbox.copy()
    bbox[0] = max(0, min(bbox[0], img_width))
    bbox[1] = max(0, min(bbox[1], img_height))
    bbox[2] = max(0, min(bbox[2], img_width))
    bbox[3] = max(0, min(bbox[3], img_height))
    return bbox


def expand_bbox(bbox: np.ndarray, scale: float = 1.0,
                img_width: Optional[int] = None,
                img_height: Optional[int] = None) -> np.ndarray:
    """
    Expand bbox by a scale factor around its center.

    Args:
        bbox: [x1, y1, x2, y2]
        scale: Expansion factor (1.0 = no change, 1.5 = 50% larger)
        img_width: Optional image width for clipping
        img_height: Optional image height for clipping

    Returns:
        Expanded bbox
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    new_w, new_h = w * scale, h * scale

    new_bbox = np.array([
        cx - new_w / 2,
        cy - new_h / 2,
        cx + new_w / 2,
        cy + new_h / 2
    ])

    if img_width is not None and img_height is not None:
        new_bbox = clip_bbox(new_bbox, img_width, img_height)

    return new_bbox
