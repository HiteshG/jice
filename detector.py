"""
YOLO detection wrapper with class filtering for hockey tracking.

Loads hockey_yolo.pt and filters to trackable classes:
- Goaltender (3) -> internal 1
- Player (4) -> internal 0
- Puck (5) -> internal 3
- Referee (6) -> internal 2
"""

from typing import List, Tuple, Optional
import numpy as np
import torch

from .config import DetectionConfig, YOLO_TO_INTERNAL, TRACKABLE_CLASSES
from .data_types import Detection


class YOLODetector:
    """
    YOLO-based detector for hockey players, goalies, referees, and pucks.

    Handles class filtering, confidence thresholds, and ID remapping.
    """

    def __init__(self, config: DetectionConfig):
        """
        Initialize the YOLO detector.

        Args:
            config: Detection configuration
        """
        self.config = config
        self.device = config.device
        self.model = None

        # Class-specific confidence thresholds
        # Internal IDs: 0=player, 1=goalie, 2=referee, 3=puck
        self.conf_thresholds = {
            0: config.player_confidence,      # player
            1: config.goaltender_confidence,  # goaltender
            2: config.referee_confidence,     # referee
            3: config.puck_confidence,        # puck (lower threshold)
        }

        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.config.model_path)

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")

            # Set half precision if configured
            if self.config.half and self.device == "cuda":
                self.model.model.half()

        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.config.model_path}: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List of Detection objects with internal class IDs
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Run inference
        results = self.model(
            frame,
            imgsz=self.config.imgsz,
            verbose=False,
            device=self.device
        )

        detections = []

        # Process results
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            scores = result.boxes.conf.cpu().numpy()  # (N,)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for i in range(len(boxes)):
                yolo_class = class_ids[i]

                # Filter to trackable classes
                if yolo_class not in TRACKABLE_CLASSES:
                    continue

                # Remap to internal class ID
                internal_class = YOLO_TO_INTERNAL[yolo_class]
                score = float(scores[i])

                # Apply class-specific confidence threshold
                threshold = self.conf_thresholds.get(internal_class, 0.4)
                if score < threshold:
                    continue

                detections.append(Detection(
                    bbox_xyxy=boxes[i].copy(),
                    score=score,
                    class_id=internal_class
                ))

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Run detection on a batch of frames.

        Args:
            frames: List of BGR images

        Returns:
            List of detection lists, one per frame
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Run batch inference
        results = self.model(
            frames,
            imgsz=self.config.imgsz,
            verbose=False,
            device=self.device
        )

        all_detections = []

        for result in results:
            frame_detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    yolo_class = class_ids[i]

                    if yolo_class not in TRACKABLE_CLASSES:
                        continue

                    internal_class = YOLO_TO_INTERNAL[yolo_class]
                    score = float(scores[i])

                    threshold = self.conf_thresholds.get(internal_class, 0.4)
                    if score < threshold:
                        continue

                    frame_detections.append(Detection(
                        bbox_xyxy=boxes[i].copy(),
                        score=score,
                        class_id=internal_class
                    ))

            all_detections.append(frame_detections)

        return all_detections

    def detect_raw(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run detection and return raw arrays (for tracker integration).

        Args:
            frame: BGR image as numpy array

        Returns:
            Tuple of (bboxes_xyxy, scores, class_ids) as numpy arrays
            Returns empty arrays if no detections
        """
        detections = self.detect(frame)

        if not detections:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32)
            )

        bboxes = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        scores = np.array([d.score for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=np.int32)

        return bboxes, scores, class_ids

    def split_by_class(self, detections: List[Detection]) -> dict:
        """
        Split detections by class.

        Args:
            detections: List of Detection objects

        Returns:
            Dict mapping internal class ID to list of detections
        """
        by_class = {0: [], 1: [], 2: [], 3: []}  # player, goalie, ref, puck

        for det in detections:
            if det.class_id in by_class:
                by_class[det.class_id].append(det)

        return by_class

    def filter_overlapping(self, detections: List[Detection],
                          iou_threshold: float = 0.5) -> List[Detection]:
        """
        Apply NMS-like filtering to remove overlapping detections.

        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression

        Returns:
            Filtered list of detections
        """
        if len(detections) <= 1:
            return detections

        # Sort by score descending
        sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)

            # Remove overlapping detections
            sorted_dets = [
                d for d in sorted_dets
                if best.iou(d) < iou_threshold
            ]

        return keep

    @property
    def class_names(self) -> dict:
        """Get internal class ID to name mapping."""
        return {
            0: "player",
            1: "goaltender",
            2: "referee",
            3: "puck"
        }


def create_detector(config: Optional[DetectionConfig] = None) -> YOLODetector:
    """
    Factory function to create a detector.

    Args:
        config: Optional detection config. Uses defaults if None.

    Returns:
        Configured YOLODetector instance
    """
    if config is None:
        config = DetectionConfig()
    return YOLODetector(config)
