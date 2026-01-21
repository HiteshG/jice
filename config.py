"""
Configuration dataclasses for the unified pipeline.

Supports loading from YAML files and provides sensible defaults for all parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Literal
import yaml


# Class mapping for YOLO model
CLASS_NAMES: Dict[int, str] = {
    0: "center_ice",   # Scene element (ignore)
    1: "faceoff",      # Scene element (ignore)
    2: "goalpost",     # Scene element (ignore)
    3: "goaltender",   # Track
    4: "player",       # Track
    5: "puck",         # Track (low confidence)
    6: "referee",      # Track
}

# Classes to track (filter others)
TRACKABLE_CLASSES: List[int] = [3, 4, 5, 6]

# Remap YOLO classes to internal IDs
# Internal: 0=player, 1=goaltender, 2=referee, 3=puck
YOLO_TO_INTERNAL: Dict[int, int] = {
    3: 1,  # goaltender -> 1
    4: 0,  # player -> 0
    5: 3,  # puck -> 3
    6: 2,  # referee -> 2
}

INTERNAL_TO_NAME: Dict[int, str] = {
    0: "player",
    1: "goaltender",
    2: "referee",
    3: "puck",
}


@dataclass
class DetectionConfig:
    """Configuration for YOLO detection."""
    model_path: str = "hockey_yolo.pt"
    imgsz: int = 1280
    trackable_classes: List[int] = field(default_factory=lambda: [3, 4, 5, 6])
    player_confidence: float = 0.4
    goaltender_confidence: float = 0.4
    referee_confidence: float = 0.4
    puck_confidence: float = 0.2  # Lower threshold for small puck
    device: str = "cuda"
    half: bool = False  # FP16 inference (disabled by default to avoid dtype issues)


@dataclass
class TrackingConfig:
    """Configuration for McByte tracker."""
    track_thresh: float = 0.6
    track_buffer: int = 30  # Frames to keep lost tracks
    match_thresh: float = 0.8
    # Association step thresholds
    max_cost_1st_assoc: float = 0.9
    max_cost_2nd_assoc: float = 0.5
    max_cost_unconfirmed: float = 0.7
    # Mask integration thresholds
    min_mask_avg_conf: float = 0.6
    min_mm1: float = 0.9  # Mask coverage ratio
    min_mm2: float = 0.05  # Mask fill ratio
    # Frame rate (auto-detected if None)
    frame_rate: Optional[int] = None


@dataclass
class MaskConfig:
    """Configuration for SAM2 + CUTIE mask propagation."""
    # SAM2 settings (for segment-anything-2-real-time fork)
    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    sam2_checkpoint: Optional[str] = None  # Path to sam2.1_hiera_large.pt
    sam2_config: Optional[str] = None  # Path to sam2.1_hiera_l.yaml
    # CUTIE settings
    cutie_checkpoint: str = "mask_propagation/Cutie/weights/cutie-base-mega.pth"
    cutie_config: str = "mask_propagation/Cutie/cutie/config/eval_config.yaml"
    # Mask creation settings
    bbox_overlap_threshold: float = 0.6  # Don't create masks for overlapped subjects
    min_mask_avg_conf: float = 0.6
    max_internal_size: int = 540  # Reduce for memory issues (-1 for full resolution)
    device: str = "cuda"


@dataclass
class CMCConfig:
    """Configuration for Camera Motion Compensation."""
    method: Literal["orb", "ecc", "sift", "sparseOptFlow", "hybrid"] = "hybrid"
    downscale: int = 2
    # ORB settings
    orb_n_features: int = 500
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8
    # ECC settings (fallback)
    ecc_iterations: int = 200
    ecc_epsilon: float = 1e-4
    # Feature matching
    ransac_reproj_threshold: float = 3.0


@dataclass
class JerseyConfig:
    """Configuration for jersey number recognition pipeline."""
    # Model paths
    str_model: str = "models/parseq_hockey.ckpt"
    legibility_model: str = "models/legibility_resnet34_hockey.pth"
    vitpose_config: str = "pose/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_huge_coco.py"
    vitpose_checkpoint: str = "pose/ViTPose/checkpoints/vitpose-h.pth"
    # ReID settings
    reid_model: str = "res50_market"
    # Processing thresholds
    legibility_threshold: float = 0.5
    temperature_scale: float = 2.367
    filter_threshold: float = 0.2
    sum_threshold: float = 1.0
    # Temporal stability (prevents jitter)
    lock_threshold: float = 3.0  # Confidence sum to lock number
    lock_frame_count: int = 10  # Frames before considering re-evaluation
    # Gaussian outlier removal
    outlier_threshold: float = 3.5
    outlier_rounds: int = 3
    # Torso cropping
    crop_padding: int = 5
    crop_confidence_threshold: float = 0.4
    crop_min_height: int = 35
    crop_min_width: int = 30
    # Digit bias
    use_bias: bool = True
    bias_double_digit: float = 0.61
    bias_single_digit: float = 0.39
    device: str = "cuda"


@dataclass
class TeamConfig:
    """Configuration for team classification."""
    classifier_type: Literal["hybrid", "robust", "compare"] = "hybrid"
    # Hybrid classifier (MobileNet + color)
    hybrid_n_clusters: int = 2
    hybrid_history_window: int = 15
    # Robust classifier (SigLIP + HDBSCAN)
    robust_model_name: str = "google/siglip-base-patch16-224"
    robust_min_cluster_size: int = 5
    robust_min_samples: int = 3
    robust_history_window: int = 20
    robust_color_weight: float = 20.0
    # Shared settings
    white_ratio_threshold: float = 0.25
    low_saturation_threshold: int = 40
    device: str = "cuda"


@dataclass
class MultiPassConfig:
    """Configuration for multi-pass processing."""
    enable_backward_pass: bool = True
    enable_interpolation: bool = True
    # Track merging
    merge_iou_threshold: float = 0.5
    merge_reid_threshold: float = 0.6
    # Interpolation
    max_interpolation_gap: int = 30  # Max frames to interpolate
    interpolation_method: Literal["linear", "kalman"] = "kalman"


@dataclass
class AnnotationConfig:
    """Configuration for video annotation output."""
    # Bounding box
    bbox_thickness: int = 2
    bbox_alpha: float = 0.7
    # Text
    font_scale: float = 0.6
    font_thickness: int = 2
    # Team colors (BGR format)
    team_colors: Dict[int, tuple] = field(default_factory=lambda: {
        0: (255, 255, 255),  # White/away team
        1: (0, 0, 255),      # Red/home team (default)
        -1: (128, 128, 128), # Unknown team (gray)
    })
    # Referee/puck colors
    referee_color: tuple = (0, 255, 255)  # Yellow
    puck_color: tuple = (0, 165, 255)     # Orange
    # What to show
    show_jersey_number: bool = True
    show_track_id: bool = False
    show_confidence: bool = False
    show_class_label: bool = False


@dataclass
class OutputConfig:
    """Configuration for output files."""
    output_video: bool = True
    output_mot: bool = True
    video_codec: str = "mp4v"
    video_fps: Optional[int] = None  # Auto-detect if None
    mot_format: str = "frame,id,x,y,w,h,conf,class,team,jersey"


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all sub-configs."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    cmc: CMCConfig = field(default_factory=CMCConfig)
    jersey: JerseyConfig = field(default_factory=JerseyConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    multipass: MultiPassConfig = field(default_factory=MultiPassConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Global settings
    device: str = "cuda"
    verbose: bool = True
    save_debug_info: bool = False
    debug_output_dir: str = "debug_output"

    def __post_init__(self):
        """Propagate global device setting to sub-configs."""
        if self.device:
            for cfg_name in ['detection', 'mask', 'jersey', 'team']:
                cfg = getattr(self, cfg_name)
                if hasattr(cfg, 'device'):
                    cfg.device = self.device


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load configuration from a YAML file.

    If no path is provided, returns default configuration.
    """
    if config_path is None:
        return PipelineConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return _dict_to_config(data)


def _dict_to_config(data: dict) -> PipelineConfig:
    """Convert a dictionary to PipelineConfig with nested dataclasses."""
    config_classes = {
        'detection': DetectionConfig,
        'tracking': TrackingConfig,
        'mask': MaskConfig,
        'cmc': CMCConfig,
        'jersey': JerseyConfig,
        'team': TeamConfig,
        'multipass': MultiPassConfig,
        'annotation': AnnotationConfig,
        'output': OutputConfig,
    }

    kwargs = {}
    for key, value in data.items():
        if key in config_classes and isinstance(value, dict):
            kwargs[key] = config_classes[key](**value)
        else:
            kwargs[key] = value

    return PipelineConfig(**kwargs)


def save_config(config: PipelineConfig, output_path: str) -> None:
    """Save configuration to a YAML file."""
    from dataclasses import asdict

    data = asdict(config)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
