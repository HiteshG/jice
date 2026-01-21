"""
Main pipeline orchestration.

Multi-pass offline pipeline:
1. Forward tracking pass
2. Backward tracking pass
3. Track merging + interpolation
4. Jersey + Team classification
5. Video annotation + MOT output
"""

from collections import defaultdict
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
    TrackMerger, TrackInterpolator, PostJerseyMerger,
    create_track_merger, create_track_interpolator, create_post_jersey_merger,
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
        self.post_jersey_merger = create_post_jersey_merger()
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

        # PASS 4: Initial Jersey + Team classification
        if self.verbose:
            print("\n=== PASS 4: Initial Jersey + Team Classification ===")
        self._classify_tracks(merged_tracks, video_path)

        # PASS 5: Post-Jersey Merge (merge tracks with same jersey + team)
        if self.verbose:
            print("\n=== PASS 5: Post-Jersey Track Consolidation ===")
        final_tracks, merge_mapping = self.post_jersey_merger.merge_by_jersey(merged_tracks)

        # Count how many merges occurred
        merges_count = len(set(merge_mapping.values()))
        original_count = len(merge_mapping)
        if self.verbose:
            merged_count = original_count - merges_count
            print(f"  Merged {merged_count} tracks based on jersey+team matching")
            print(f"  Final track count: {len(final_tracks)}")

        # PASS 6: Re-run jersey detection on merged tracks for better accuracy
        if merged_count > 0:
            if self.verbose:
                print("\n=== PASS 6: Re-classify Merged Tracks ===")
            self._reclassify_merged_tracks(final_tracks, merge_mapping, video_path)

        # Generate outputs
        if self.verbose:
            print("\n=== Generating Outputs ===")

        if output_path is None:
            output_path = str(Path(video_path).with_suffix('')) + "_tracked.mp4"

        self._generate_outputs(video_path, output_path, final_tracks)

        return final_tracks

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

        # Sampling parameters for performance
        MAX_CROPS_TEAM_FIT = 200  # Max total crops for team clustering fit
        MAX_CROPS_PER_TRACK_JERSEY = 30  # Max crops per track for jersey recognition
        MAX_CROPS_PER_TRACK_TEAM = 20  # Max crops per track for team classification
        SAMPLE_STRIDE = 5  # Sample every Nth crop

        # Collect sampled crops for team clustering
        all_player_crops = []
        player_track_ids = []

        for track_id, track in tracks.items():
            # Only classify players and goalies
            if track.class_id not in [0, 1]:
                continue

            crops = track.get_crops()
            if crops:
                # Sample crops: every Nth frame, capped
                sampled = crops[::SAMPLE_STRIDE][:MAX_CROPS_PER_TRACK_TEAM]
                all_player_crops.extend(sampled)
                player_track_ids.extend([track_id] * len(sampled))

        # Cap total crops for fitting
        if len(all_player_crops) > MAX_CROPS_TEAM_FIT:
            indices = np.linspace(0, len(all_player_crops) - 1, MAX_CROPS_TEAM_FIT, dtype=int)
            all_player_crops = [all_player_crops[i] for i in indices]
            player_track_ids = [player_track_ids[i] for i in indices]

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

            # Sample crops for jersey recognition (more samples needed for accuracy)
            jersey_crops = crops[::SAMPLE_STRIDE][:MAX_CROPS_PER_TRACK_JERSEY]

            # Jersey recognition
            jersey_number = self.jersey_recognizer.process_tracklet(jersey_crops, track_id)
            track.jersey_number = jersey_number

            pred = self.jersey_recognizer.get_prediction(track_id)
            track.jersey_confidence = pred.confidence

            # Sample crops for team classification
            team_crops = crops[::SAMPLE_STRIDE][:MAX_CROPS_PER_TRACK_TEAM]

            # Team classification
            team_assignments = self.team_classifier.predict(team_crops, [track_id] * len(team_crops))
            if team_assignments:
                # Filter out invalid assignments
                valid_teams = [a.team_id for a in team_assignments if a.team_id >= 0]
                if valid_teams:
                    track.team_id = max(set(valid_teams), key=valid_teams.count)
                    track.team_confidence = np.mean([a.confidence for a in team_assignments if a.team_id >= 0])

    def _reclassify_merged_tracks(
        self,
        tracks: Dict[int, Track],
        merge_mapping: Dict[int, int],
        video_path: str
    ) -> None:
        """
        Re-run jersey detection on tracks that were merged.

        Merged tracks have more frames/crops, which should give better
        jersey number predictions.

        Args:
            tracks: Dictionary of merged tracks
            merge_mapping: Mapping from original track IDs to merged track IDs
            video_path: Path to video (for logging)
        """
        # Find track IDs that received merges (have multiple source tracks)
        merge_counts = defaultdict(int)
        for orig_id, merged_id in merge_mapping.items():
            merge_counts[merged_id] += 1

        merged_track_ids = {tid for tid, count in merge_counts.items() if count > 1}

        if not merged_track_ids:
            return

        # Sampling parameters (use more samples for merged tracks)
        MAX_CROPS_MERGED = 50  # More crops for better accuracy
        SAMPLE_STRIDE = 3  # Sample more frequently

        # Re-classify only merged tracks
        for track_id in tqdm(merged_track_ids, desc="Re-classifying merged",
                            disable=not self.verbose):
            if track_id not in tracks:
                continue

            track = tracks[track_id]
            crops = track.get_crops()

            if not crops:
                continue

            # Sample crops for jersey recognition
            jersey_crops = crops[::SAMPLE_STRIDE][:MAX_CROPS_MERGED]

            # Re-run jersey recognition with fresh state for this track
            self.jersey_recognizer.reset_track(track_id)
            jersey_number = self.jersey_recognizer.process_tracklet(jersey_crops, track_id)
            track.jersey_number = jersey_number

            pred = self.jersey_recognizer.get_prediction(track_id)
            track.jersey_confidence = pred.confidence

            if self.verbose and track.jersey_number is not None:
                print(f"    Track {track_id}: Jersey #{track.jersey_number} (conf: {track.jersey_confidence:.2f})")

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
