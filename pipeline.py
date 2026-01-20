"""
Main pipeline orchestration.

Multi-pass offline pipeline:
1. Forward tracking pass
2. Backward tracking pass
3. Track merging + interpolation
4. Jersey + Team classification
5. Video annotation + MOT output
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import numpy as np
import cv2
from tqdm import tqdm

from .config import PipelineConfig, load_config
from .data_types import Track, VideoMetadata, TrackletFrame
from .detector import YOLODetector, create_detector
from .tracker import McByteTracker, create_tracker, STrack
from .cmc import CameraMotionCompensator, create_cmc
from .mask_sam2_cutie import SAM2CutieMaskManager, create_mask_manager
from .jersey import JerseyRecognizer, create_jersey_recognizer
from .team import TeamClassifier, create_team_classifier
from .multipass import (
    TrackMerger, TrackInterpolator,
    create_track_merger, create_track_interpolator,
    convert_stracks_to_tracks, merge_frame_tracks
)
from .annotator import VideoAnnotator, MOTWriter, VideoWriter, create_annotator


class UnifiedPipeline:
    """
    Unified ice hockey player tracking pipeline.

    Integrates detection, tracking, mask propagation, jersey recognition,
    and team classification with multi-pass processing.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.verbose = config.verbose

        # Initialize components
        self.detector = create_detector(config.detection)
        self.cmc = create_cmc(config.cmc)
        self.tracker = None  # Created per-pass
        self.mask_manager = None  # Lazy loaded if needed

        self.jersey_recognizer = create_jersey_recognizer(config.jersey)
        self.team_classifier = create_team_classifier(config.team)

        self.track_merger = create_track_merger(config.multipass)
        self.track_interpolator = create_track_interpolator(config.multipass)
        self.annotator = create_annotator(config.annotation)

        # Video metadata
        self.video_metadata: Optional[VideoMetadata] = None

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> Dict[int, Track]:
        """
        Process a video file through the full pipeline.

        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)

        Returns:
            Dictionary of final tracks
        """
        # Load video metadata
        self.video_metadata = VideoMetadata.from_video(video_path)
        fps = self.video_metadata.fps

        if self.verbose:
            print(f"Processing: {video_path}")
            print(f"  Resolution: {self.video_metadata.width}x{self.video_metadata.height}")
            print(f"  FPS: {fps:.2f}")
            print(f"  Frames: {self.video_metadata.total_frames}")

        # PASS 1: Forward tracking
        if self.verbose:
            print("\n=== PASS 1: Forward Tracking ===")
        forward_tracks = self._run_tracking_pass(video_path, reverse=False)

        # PASS 2: Backward tracking (optional)
        backward_tracks = {}
        if self.config.multipass.enable_backward_pass:
            if self.verbose:
                print("\n=== PASS 2: Backward Tracking ===")
            backward_tracks = self._run_tracking_pass(video_path, reverse=True)

        # PASS 3: Merge + Interpolate
        if self.verbose:
            print("\n=== PASS 3: Merge + Interpolate ===")
        merged_tracks = self.track_merger.merge(forward_tracks, backward_tracks)

        if self.config.multipass.enable_interpolation:
            merged_tracks = self.track_interpolator.interpolate(merged_tracks)

        if self.verbose:
            print(f"  Tracks after merge: {len(merged_tracks)}")

        # PASS 4: Jersey + Team classification
        if self.verbose:
            print("\n=== PASS 4: Jersey + Team Classification ===")
        self._classify_tracks(merged_tracks, video_path)

        # Generate outputs
        if self.verbose:
            print("\n=== Generating Outputs ===")

        if output_path is None:
            output_path = str(Path(video_path).with_suffix('')) + "_tracked.mp4"

        self._generate_outputs(video_path, output_path, merged_tracks)

        return merged_tracks

    def _run_tracking_pass(
        self,
        video_path: str,
        reverse: bool = False
    ) -> Dict[int, Track]:
        """
        Run a single tracking pass (forward or backward).

        Args:
            video_path: Path to video
            reverse: Whether to process frames in reverse

        Returns:
            Dictionary of tracks
        """
        # Create fresh tracker
        fps = self.video_metadata.fps if self.video_metadata else 30
        self.tracker = create_tracker(self.config.tracking, int(fps))
        self.cmc.reset()

        # Initialize mask manager if needed
        use_masks = self.config.mask.sam2_checkpoint or self.config.mask.sam2_model_id
        if use_masks:
            if self.mask_manager is None:
                try:
                    self.mask_manager = create_mask_manager(self.config.mask)
                except Exception as e:
                    print(f"Warning: Failed to load mask manager: {e}")
                    self.mask_manager = None
            else:
                self.mask_manager.reset()

        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get frames in order
        if reverse:
            frame_indices = list(range(total_frames - 1, -1, -1))
        else:
            frame_indices = list(range(total_frames))

        tracks: Dict[int, Track] = {}
        prev_frame = None

        desc = "Backward pass" if reverse else "Forward pass"
        for idx in tqdm(frame_indices, desc=desc, disable=not self.verbose):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect
            detections = self.detector.detect(frame)

            # Camera motion compensation
            det_boxes = np.array([d.bbox_xyxy for d in detections]) if detections else None
            warp = self.cmc.apply(frame, det_boxes)

            # Mask propagation (simplified - skip for speed)
            prediction_mask = None
            tracklet_mask_dict = {}
            mask_avg_prob_dict = {}

            # Track
            frame_id = idx + 1 if not reverse else total_frames - idx
            img_size = (frame.shape[0], frame.shape[1])

            output_stracks, removed_ids, new_tracks = self.tracker.update(
                detections, img_size,
                prediction_mask, tracklet_mask_dict, mask_avg_prob_dict,
                warp
            )

            # Convert to Track objects
            frame_tracks = convert_stracks_to_tracks(output_stracks, frame_id, frame)
            tracks = merge_frame_tracks(tracks, frame_tracks)

            prev_frame = frame

        cap.release()
        return tracks

    def _classify_tracks(
        self,
        tracks: Dict[int, Track],
        video_path: str
    ) -> None:
        """
        Run jersey and team classification on all tracks.

        Args:
            tracks: Dictionary of tracks to classify
            video_path: Path to video for extracting crops
        """
        # Reset classifiers
        self.jersey_recognizer.reset()
        self.team_classifier.reset()

        # Collect all crops for team clustering
        all_player_crops = []
        player_track_ids = []

        for track_id, track in tracks.items():
            # Only classify players and goalies
            if track.class_id not in [0, 1]:
                continue

            crops = track.get_crops()
            if crops:
                all_player_crops.extend(crops)
                player_track_ids.extend([track_id] * len(crops))

        # Fit team classifier
        if all_player_crops:
            self.team_classifier.fit(all_player_crops)

        # Process each track
        for track_id, track in tqdm(tracks.items(), desc="Classifying",
                                    disable=not self.verbose):
            if track.class_id not in [0, 1]:
                continue

            crops = track.get_crops()
            if not crops:
                continue

            # Jersey recognition
            jersey_number = self.jersey_recognizer.process_tracklet(crops, track_id)
            track.jersey_number = jersey_number

            pred = self.jersey_recognizer.get_prediction(track_id)
            track.jersey_confidence = pred.confidence

            # Team classification
            team_assignments = self.team_classifier.predict(crops, [track_id] * len(crops))
            if team_assignments:
                # Use most common assignment
                teams = [a.team_id for a in team_assignments]
                track.team_id = max(set(teams), key=teams.count)
                track.team_confidence = np.mean([a.confidence for a in team_assignments])

    def _generate_outputs(
        self,
        video_path: str,
        output_path: str,
        tracks: Dict[int, Track]
    ) -> None:
        """
        Generate annotated video and MOT output.

        Args:
            video_path: Input video path
            output_path: Output video path
            tracks: Final tracks
        """
        # MOT output
        if self.config.output.output_mot:
            mot_path = str(Path(output_path).with_suffix('.txt'))
            with MOTWriter(mot_path) as mot_writer:
                mot_writer.write_all(tracks)
            if self.verbose:
                print(f"  MOT output: {mot_path}")

        # Video output
        if self.config.output.output_video:
            cap = cv2.VideoCapture(video_path)
            fps = self.config.output.video_fps or self.video_metadata.fps
            width = self.video_metadata.width
            height = self.video_metadata.height

            with VideoWriter(output_path, fps, width, height,
                           self.config.output.video_codec) as writer:
                frame_id = 0
                pbar = tqdm(total=self.video_metadata.total_frames,
                           desc="Writing video", disable=not self.verbose)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_id += 1
                    annotated = self.annotator.annotate_frame(frame, tracks, frame_id)
                    writer.write(annotated)
                    pbar.update(1)

                pbar.close()

            cap.release()
            if self.verbose:
                print(f"  Video output: {output_path}")


def create_pipeline(config: Optional[PipelineConfig] = None) -> UnifiedPipeline:
    """
    Factory function to create a pipeline.

    Args:
        config: Optional pipeline config. Uses defaults if None.

    Returns:
        Configured UnifiedPipeline instance
    """
    if config is None:
        config = PipelineConfig()
    return UnifiedPipeline(config)


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> Dict[int, Track]:
    """
    Convenience function to process a video.

    Args:
        video_path: Path to input video
        output_path: Path to output video
        config_path: Path to config YAML file

    Returns:
        Dictionary of final tracks
    """
    config = load_config(config_path)
    pipeline = create_pipeline(config)
    return pipeline.process_video(video_path, output_path)
