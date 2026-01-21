"""
Command-line interface for the unified pipeline.

Usage:
    python -m unified_pipeline.cli --video input.mp4 --output output.mp4
    python -m unified_pipeline.cli --video input.mp4 --config config.yaml
    python -m unified_pipeline.cli --video input.mp4 --team-classifier compare
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, load_config, save_config
from .pipeline import UnifiedPipeline, create_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Ice Hockey Player Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m unified_pipeline.cli --video input.mp4 --output output.mp4

  # With custom config
  python -m unified_pipeline.cli --video input.mp4 --config config.yaml

  # Compare team classifiers
  python -m unified_pipeline.cli --video input.mp4 --team-classifier compare

  # Disable jersey/team for debugging
  python -m unified_pipeline.cli --video input.mp4 --no-jersey --no-team

  # Output only MOT format
  python -m unified_pipeline.cli --video input.mp4 --output-format mot
        """
    )

    # Required arguments
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to input video file"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output video file (default: input_tracked.mp4)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["video", "mot", "both"],
        default="both",
        help="Output format: video, mot, or both (default: both)"
    )

    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Save current configuration to file"
    )

    # Detection options
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model (uses config value if not specified)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Detection image size (default: 1280)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Detection confidence threshold (default: 0.4)"
    )

    # Tracking options
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.6,
        help="Tracking confidence threshold (default: 0.6)"
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="Track buffer for lost tracks (default: 30)"
    )

    # Multi-pass options
    parser.add_argument(
        "--no-backward",
        action="store_true",
        help="Disable backward tracking pass"
    )
    parser.add_argument(
        "--no-interpolation",
        action="store_true",
        help="Disable track interpolation"
    )

    # Team classification options
    parser.add_argument(
        "--team-classifier",
        type=str,
        choices=["hybrid", "robust", "compare"],
        default="hybrid",
        help="Team classifier type (default: hybrid)"
    )

    # Feature flags
    parser.add_argument(
        "--no-jersey",
        action="store_true",
        help="Disable jersey number recognition"
    )
    parser.add_argument(
        "--no-team",
        action="store_true",
        help="Disable team classification"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Disable mask propagation"
    )

    # General options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build configuration from arguments."""
    # Load base config if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = PipelineConfig()

    # Override with CLI arguments
    config.device = args.device
    config.verbose = not args.quiet
    config.save_debug_info = args.debug

    # Detection
    if args.model is not None:
        config.detection.model_path = args.model
    config.detection.imgsz = args.imgsz
    config.detection.player_confidence = args.conf
    config.detection.device = args.device

    # Tracking
    config.tracking.track_thresh = args.track_thresh
    config.tracking.track_buffer = args.track_buffer

    # Multi-pass
    config.multipass.enable_backward_pass = not args.no_backward
    config.multipass.enable_interpolation = not args.no_interpolation

    # Team classification
    config.team.classifier_type = args.team_classifier

    # Output format
    config.output.output_video = args.output_format in ["video", "both"]
    config.output.output_mot = args.output_format in ["mot", "both"]

    # Disable features if requested
    if args.no_masks:
        config.mask.sam2_checkpoint = None
        config.mask.sam2_model_id = None

    return config


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    # Build configuration
    config = build_config(args)

    # Save config if requested
    if args.save_config:
        save_config(config, args.save_config)
        print(f"Configuration saved to: {args.save_config}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(video_path.with_suffix('')) + "_tracked.mp4"

    # Run pipeline
    try:
        pipeline = create_pipeline(config)

        # Handle jersey/team disable
        if args.no_jersey:
            pipeline.jersey_recognizer = None
        if args.no_team:
            pipeline.team_classifier = None

        tracks = pipeline.process_video(str(video_path), output_path)

        # Print summary
        if not args.quiet:
            print(f"\n=== Summary ===")
            print(f"  Total tracks: {len(tracks)}")

            players = sum(1 for t in tracks.values() if t.class_id == 0)
            goalies = sum(1 for t in tracks.values() if t.class_id == 1)
            refs = sum(1 for t in tracks.values() if t.class_id == 2)
            pucks = sum(1 for t in tracks.values() if t.class_id == 3)

            print(f"  Players: {players}")
            print(f"  Goalies: {goalies}")
            print(f"  Referees: {refs}")
            print(f"  Pucks: {pucks}")

            jerseys = sum(1 for t in tracks.values()
                         if t.jersey_number is not None and t.class_id in [0, 1])
            print(f"  Tracks with jersey: {jerseys}")

        # Team classifier comparison
        if args.team_classifier == "compare":
            print("\n=== Team Classifier Comparison ===")
            print("  (Comparison output would be generated here)")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130

    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
