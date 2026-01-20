"""
Video annotation module.

Provides clean overlay annotations with:
- Bounding boxes colored by team
- Jersey numbers
- Optional track IDs and confidence scores
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from .config import AnnotationConfig
from .data_types import Track, TrackletFrame


class VideoAnnotator:
    """
    Annotates video frames with tracking results.

    Clean overlay: bbox with team color + jersey number only.
    """

    def __init__(self, config: AnnotationConfig):
        """
        Initialize the annotator.

        Args:
            config: Annotation configuration
        """
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Track],
        frame_id: int
    ) -> np.ndarray:
        """
        Annotate a frame with tracking results.

        Args:
            frame: BGR frame image
            tracks: Dictionary of tracks
            frame_id: Current frame ID

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for track_id, track in tracks.items():
            if frame_id not in track.frames:
                continue

            frame_data = track.frames[frame_id]
            bbox = frame_data.bbox_xyxy

            # Get color based on class and team
            color = self._get_color(track)

            # Draw bounding box
            self._draw_bbox(annotated, bbox, color)

            # Draw label (jersey number for players/goalies)
            label = self._get_label(track)
            if label:
                self._draw_label(annotated, bbox, label, color)

        return annotated

    def _get_color(self, track: Track) -> Tuple[int, int, int]:
        """Get color for track based on class and team."""
        # Referee
        if track.class_id == 2:
            return self.config.referee_color

        # Puck
        if track.class_id == 3:
            return self.config.puck_color

        # Player or Goaltender
        if track.team_id is not None and track.team_id in self.config.team_colors:
            return self.config.team_colors[track.team_id]

        # Unknown team
        return self.config.team_colors.get(-1, (128, 128, 128))

    def _get_label(self, track: Track) -> str:
        """Get label text for track."""
        parts = []

        # Jersey number (players and goalies only)
        if self.config.show_jersey_number and track.class_id in [0, 1]:
            if track.jersey_number is not None:
                parts.append(f"#{track.jersey_number}")

        # Track ID
        if self.config.show_track_id:
            parts.append(f"ID:{track.track_id}")

        # Confidence
        if self.config.show_confidence:
            parts.append(f"{track.jersey_confidence:.2f}")

        # Class label
        if self.config.show_class_label:
            class_names = {0: "P", 1: "G", 2: "R", 3: "Pk"}
            parts.append(class_names.get(track.class_id, "?"))

        return " ".join(parts)

    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw bounding box with transparency."""
        x1, y1, x2, y2 = map(int, bbox)
        thickness = self.config.bbox_thickness

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Optional: Draw filled corners for better visibility
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

    def _draw_label(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        label: str,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw label above bounding box."""
        if not label:
            return

        x1, y1, x2, y2 = map(int, bbox)

        # Get text size
        font_scale = self.config.font_scale
        thickness = self.config.font_thickness
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, font_scale, thickness
        )

        # Position above bbox
        text_x = x1
        text_y = y1 - 5

        # Ensure text stays in frame
        if text_y - text_height < 0:
            text_y = y1 + text_height + 5

        # Draw background rectangle
        bg_x1 = text_x - 2
        bg_y1 = text_y - text_height - 2
        bg_x2 = text_x + text_width + 2
        bg_y2 = text_y + 2

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # Determine text color (black or white for contrast)
        brightness = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        # Draw text
        cv2.putText(
            frame, label, (text_x, text_y),
            self.font, font_scale, text_color, thickness, cv2.LINE_AA
        )


class MOTWriter:
    """
    Writes tracking results in MOT format.

    Format: frame,id,x,y,w,h,conf,class,team,jersey
    """

    def __init__(self, output_path: str, format_str: str = None):
        """
        Initialize the MOT writer.

        Args:
            output_path: Path to output file
            format_str: Format string for columns
        """
        self.output_path = output_path
        self.format_str = format_str or "frame,id,x,y,w,h,conf,class,team,jersey"
        self.file = None

    def __enter__(self):
        self.file = open(self.output_path, 'w')
        self.file.write(f"# {self.format_str}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write_frame(self, tracks: Dict[int, Track], frame_id: int) -> None:
        """Write all track entries for a frame."""
        if self.file is None:
            raise RuntimeError("MOTWriter not opened")

        for track_id, track in tracks.items():
            if frame_id not in track.frames:
                continue

            frame_data = track.frames[frame_id]
            bbox = frame_data.bbox_xyxy

            # Convert to tlwh
            x, y = bbox[0], bbox[1]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            team = track.team_id if track.team_id is not None else -1
            jersey = track.jersey_number if track.jersey_number is not None else -1

            line = f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
            line += f"{frame_data.score:.3f},{track.class_id},{team},{jersey}\n"

            self.file.write(line)

    def write_all(self, tracks: Dict[int, Track]) -> None:
        """Write all tracks to file."""
        if self.file is None:
            raise RuntimeError("MOTWriter not opened")

        # Get all frame IDs
        all_frames = set()
        for track in tracks.values():
            all_frames.update(track.frames.keys())

        for frame_id in sorted(all_frames):
            self.write_frame(tracks, frame_id)


class VideoWriter:
    """
    Writes annotated video frames to file.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v"
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path to output video
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec
        """
        self.output_path = output_path
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path, self.fourcc, fps, (width, height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video."""
        self.writer.write(frame)

    def release(self) -> None:
        """Release the video writer."""
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def create_annotator(config: Optional[AnnotationConfig] = None) -> VideoAnnotator:
    """Factory function for VideoAnnotator."""
    if config is None:
        config = AnnotationConfig()
    return VideoAnnotator(config)
