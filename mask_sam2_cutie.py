"""
SAM2 + CUTIE mask propagation for tracking.

Uses SAM2 (real-time fork) for high-quality initial mask generation and CUTIE for
temporal mask propagation. Ported from McByte/mask_propagation/mask_manager.py
with upgrade from SAM1 to SAM2.

SAM2 Real-Time Fork: https://github.com/Gy920/segment-anything-2-real-time
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from .config import MaskConfig
from .data_types import MaskMetrics, Detection


# Constants from McByte
MASK_CREATION_BBOX_OVERLAP_THRESHOLD = 0.6

# Default SAM2 paths (relative to segment-anything-2-real-time repo)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"


class SAM2CutieMaskManager:
    """
    Mask manager combining SAM2 for segmentation and CUTIE for propagation.

    SAM2 provides better mask quality than SAM1, especially for:
    - Players in motion blur
    - Partial occlusions
    - Varying lighting on ice

    CUTIE excels at temporal propagation with occlusion handling.
    """

    def __init__(self, config: MaskConfig):
        """
        Initialize the mask manager.

        Args:
            config: Mask configuration
        """
        self.config = config
        self.device = config.device

        # Mask state
        self.tracklet_mask_dict: Dict[int, int] = {}  # track_id -> mask_id
        self.mask_color_dict: Dict[int, int] = {}     # mask_id -> color_id
        self.mask_color_counter = 0
        self.num_objects = 0
        self.current_object_list_cutie: List[int] = []
        self.last_object_number_cutie = 0
        self.awaiting_mask_tracklet_ids: List[int] = []
        self.mask_prediction_prev_frame: Optional[np.ndarray] = None
        self.initialized = False
        self.init_delay_counter = 0
        self.SAM_START_FRAME = 1

        # Models (lazy loaded)
        self.sam2_predictor = None
        self.cutie = None
        self.cutie_processor = None

        self._load_models()

    def _load_models(self) -> None:
        """Load SAM2 and CUTIE models."""
        self._load_sam2()
        self._load_cutie()

    def _load_sam2(self) -> None:
        """Load SAM2 model for initial mask generation using real-time fork."""
        try:
            # Try loading SAM2 from the real-time fork
            # https://github.com/Gy920/segment-anything-2-real-time
            from sam2.build_sam import build_sam2_camera_predictor
            from hydra import initialize_config_dir, compose
            from hydra.core.global_hydra import GlobalHydra

            # Resolve paths
            sam2_checkpoint = self.config.sam2_checkpoint or DEFAULT_SAM2_CHECKPOINT
            sam2_config_path = self.config.sam2_config or DEFAULT_SAM2_CONFIG

            # If absolute path provided, extract just the config name and set up Hydra
            if Path(sam2_config_path).is_absolute():
                # Set up Hydra to search in the SAM2 configs directory
                config_dir = Path(sam2_config_path).parent
                config_name = Path(sam2_config_path).stem
            else:
                # Assume it's relative to sam2 repo root
                sam2_root = self.config.sam2_root or Path(sam2_checkpoint).parent.parent
                config_dir = Path(sam2_root) / "sam2" / "configs" / "sam2.1"
                config_name = Path(sam2_config_path).stem

            # Initialize Hydra with SAM2 config directory
            GlobalHydra.instance().clear()
            initialize_config_dir(version_base="1.2", config_dir=str(config_dir))

            # Build SAM2 camera predictor (optimized for real-time)
            self.sam2_predictor = build_sam2_camera_predictor(
                config_name,
                sam2_checkpoint,
                device=self.device
            )
            self._using_sam2_realtime = True
            print(f"Loaded SAM2 real-time from: {sam2_checkpoint}")

        except ImportError:
            try:
                # Try standard SAM2 as fallback
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                sam2_checkpoint = self.config.sam2_checkpoint or DEFAULT_SAM2_CHECKPOINT
                sam2_model = build_sam2("sam2_hiera_l", sam2_checkpoint)
                sam2_model.to(self.device)
                sam2_model.eval()
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                self._using_sam2_realtime = False
                print(f"Loaded standard SAM2 from: {sam2_checkpoint}")

            except ImportError:
                # Fall back to SAM1 if SAM2 not available
                print("Warning: SAM2 not available, falling back to SAM1")
                self._load_sam1_fallback()

    def _load_sam1_fallback(self) -> None:
        """Load SAM1 as fallback if SAM2 not available."""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            sam_checkpoint = Path(__file__).parent.parent / "McByte" / "sam_models" / "sam_vit_b_01ec64.pth"
            if not sam_checkpoint.exists():
                sam_checkpoint = "./sam_models/sam_vit_b_01ec64.pth"

            sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
            sam.to(self.device)
            self.sam2_predictor = SamPredictor(sam)
            self._using_sam1 = True

        except ImportError:
            raise ImportError(
                "Neither SAM2 nor SAM1 available. Install with: "
                "pip install segment-anything or pip install sam2"
            )

    def _load_cutie(self) -> None:
        """Load CUTIE model for temporal mask propagation."""
        try:
            import sys
            from pathlib import Path

            # Add CUTIE to path
            cutie_root = Path(__file__).parent.parent / "McByte" / "mask_propagation" / "Cutie"
            sys.path.insert(0, str(cutie_root))

            from omegaconf import open_dict
            from hydra import compose, initialize_config_dir

            from cutie.model.cutie import CUTIE
            from cutie.inference.inference_core import InferenceCore
            from cutie.inference.utils.args_utils import get_dataset_cfg

            # Initialize Hydra config
            config_path = cutie_root / "cutie" / "config"

            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    initialize_config_dir(version_base='1.3.2', config_dir=str(config_path))
                    cfg = compose(config_name="eval_config")

                    # Set weights path
                    weight_path = Path(self.config.cutie_checkpoint)
                    if not weight_path.is_absolute():
                        weight_path = Path(__file__).parent.parent / "McByte" / weight_path

                    with open_dict(cfg):
                        cfg['weights'] = str(weight_path)
                        if self.config.max_internal_size > 0:
                            cfg['max_internal_size'] = self.config.max_internal_size

                    # Required CUTIE initialization
                    _ = get_dataset_cfg(cfg)

                    # Load model
                    self.cutie = CUTIE(cfg).cuda().eval()
                    model_weights = torch.load(cfg.weights, map_location=self.device)
                    self.cutie.load_weights(model_weights)

                    torch.cuda.empty_cache()
                    self.cutie_processor = InferenceCore(self.cutie, cfg=cfg)

        except Exception as e:
            print(f"Warning: CUTIE loading failed: {e}")
            print("Mask propagation will be disabled")
            self.cutie = None
            self.cutie_processor = None

    def reset(self) -> None:
        """Reset state for new video."""
        self.tracklet_mask_dict = {}
        self.mask_color_dict = {}
        self.mask_color_counter = 0
        self.num_objects = 0
        self.current_object_list_cutie = []
        self.last_object_number_cutie = 0
        self.awaiting_mask_tracklet_ids = []
        self.mask_prediction_prev_frame = None
        self.initialized = False
        self.init_delay_counter = 0

        # Reset CUTIE processor if available
        if self.cutie_processor is not None:
            self.cutie_processor.clear_memory()

    def generate_initial_mask(self, frame: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        """
        Generate mask for a single bounding box using SAM2.

        Args:
            frame: BGR image
            bbox_xyxy: Bounding box [x1, y1, x2, y2]

        Returns:
            Binary mask array (H, W)
        """
        if self.sam2_predictor is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # Set image
        self.sam2_predictor.set_image(frame)

        # Predict mask
        if hasattr(self.sam2_predictor, 'predict'):
            # SAM2 interface
            masks, scores, _ = self.sam2_predictor.predict(
                box=bbox_xyxy,
                multimask_output=False
            )
            return masks[0].astype(np.uint8)
        else:
            # SAM1 interface (fallback)
            box_tensor = torch.tensor([bbox_xyxy], device=self.device)
            transformed_box = self.sam2_predictor.transform.apply_boxes_torch(
                box_tensor, frame.shape[:2]
            )
            masks, _, _ = self.sam2_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False
            )
            return masks[0, 0].cpu().numpy().astype(np.uint8)

    def get_updated_masks(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        frame_id: int,
        online_tlwhs: List[np.ndarray],
        online_ids: List[int],
        new_tracks: List[Any],
        removed_track_ids: List[int]
    ) -> Tuple[Optional[np.ndarray], Dict[int, int], Optional[Dict[int, float]], Optional[np.ndarray]]:
        """
        Update masks based on tracker state.

        Propagates existing masks, creates new masks for new tracks,
        and removes masks for removed tracks.

        Args:
            frame: Current BGR frame
            prev_frame: Previous BGR frame
            frame_id: Current frame number (starting from 1)
            online_tlwhs: Track positions in tlwh format
            online_ids: Track IDs
            new_tracks: Newly created tracks
            removed_track_ids: IDs of removed tracks

        Returns:
            Tuple of (prediction, tracklet_mask_dict, mask_avg_prob_dict, prediction_colors_preserved)
        """
        if self.cutie_processor is None:
            return None, {}, None, None

        # Convert to torch
        frame_torch = self._image_to_torch(frame)
        prev_frame_torch = self._image_to_torch(prev_frame)

        prediction = None

        # Initialize masks on first valid frame
        start_frame = self.SAM_START_FRAME + 1 + self.init_delay_counter
        if frame_id == start_frame and online_tlwhs:
            prediction = self._initialize_first_masks(
                frame_torch, prev_frame_torch, prev_frame,
                online_tlwhs, online_ids
            )
        elif frame_id > start_frame:
            # Add/remove masks and propagate
            self._add_new_masks(prev_frame_torch, prev_frame, online_tlwhs, online_ids, new_tracks)
            self._remove_masks(removed_track_ids)

            # Propagate to current frame
            prediction = self.cutie_processor.step(frame_torch)

        mask_avg_prob_dict = None
        prediction_colors_preserved = None

        if prediction is not None:
            prediction, mask_avg_prob_dict, prediction_colors_preserved = self._post_process_mask(prediction)

        return prediction, self.tracklet_mask_dict.copy(), mask_avg_prob_dict, prediction_colors_preserved

    def _initialize_first_masks(
        self,
        frame_torch: torch.Tensor,
        prev_frame_torch: torch.Tensor,
        prev_frame: np.ndarray,
        online_tlwhs: List[np.ndarray],
        online_ids: List[int]
    ) -> Optional[torch.Tensor]:
        """Initialize masks for first frame detections."""
        self.sam2_predictor.set_image(prev_frame)

        image_boxes_list = []
        new_tracks_ids = []

        for i, tlwh in enumerate(online_tlwhs):
            # Check for overlap with lower tracks
            track_BBs_lower = _get_tracklets_with_lower_bottom(tlwh, online_tlwhs)
            overlap = _get_overlap_with_lower_tracklets(tlwh, track_BBs_lower)

            if overlap >= MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                self.awaiting_mask_tracklet_ids.append(online_ids[i])
                continue

            # Convert tlwh to xyxy
            x1, y1, w, h = tlwh
            image_boxes_list.append([x1, y1, x1 + w, y1 + h])
            new_tracks_ids.append(online_ids[i])

        if len(image_boxes_list) == 0:
            self.init_delay_counter += 1
            return None

        # Generate masks with SAM2
        masks = self._predict_masks_batch(prev_frame, image_boxes_list)

        # Combine masks
        mask = np.zeros(masks[0].shape, dtype=np.int32)
        for mi, m in enumerate(masks):
            current_mask = m.astype(np.int32)
            current_mask[current_mask > 0] = mi + 1
            non_occupied = (mask == 0).astype(np.int32)
            mask += current_mask * non_occupied

        self.num_objects = len(masks)
        self.current_object_list_cutie = list(range(1, self.num_objects + 1))
        self.last_object_number_cutie = max(self.current_object_list_cutie, default=0)

        # Create mappings
        self.tracklet_mask_dict = dict(zip(new_tracks_ids, range(1, self.num_objects + 1)))
        self.mask_color_dict = dict(zip(range(1, self.num_objects + 1), range(1, self.num_objects + 1)))
        self.mask_color_counter = max(list(self.mask_color_dict.values()), default=0)

        # Initialize CUTIE
        mask_torch = self._index_to_one_hot(mask, self.num_objects + 1)
        _ = self.cutie_processor.step(prev_frame_torch, mask_torch[1:], idx_mask=False)
        prediction = self.cutie_processor.step(frame_torch)

        self.initialized = True
        return prediction

    def _add_new_masks(
        self,
        prev_frame_torch: torch.Tensor,
        prev_frame: np.ndarray,
        online_tlwhs: List[np.ndarray],
        online_ids: List[int],
        new_tracks: List[Any]
    ) -> None:
        """Add masks for new tracks."""
        if len(new_tracks) == 0 and len(self.awaiting_mask_tracklet_ids) == 0:
            return

        self.sam2_predictor.set_image(prev_frame)
        image_boxes_list = []
        new_tracks_ids = []

        # Process awaiting tracks
        for track_id in list(self.awaiting_mask_tracklet_ids):
            if track_id not in online_ids:
                continue

            idx = online_ids.index(track_id)
            tlwh = online_tlwhs[idx]

            track_BBs_lower = _get_tracklets_with_lower_bottom(tlwh, online_tlwhs)
            overlap = _get_overlap_with_lower_tracklets(tlwh, track_BBs_lower)

            if overlap < MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                x1, y1, w, h = tlwh
                image_boxes_list.append([x1, y1, x1 + w, y1 + h])
                new_tracks_ids.append(track_id)
                self.awaiting_mask_tracklet_ids.remove(track_id)

        # Process new tracks
        for track in new_tracks:
            tlwh = track.last_det_tlwh if hasattr(track, 'last_det_tlwh') else track.tlwh

            track_BBs_lower = _get_tracklets_with_lower_bottom(tlwh, online_tlwhs)
            overlap = _get_overlap_with_lower_tracklets(tlwh, track_BBs_lower)

            if overlap >= MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                self.awaiting_mask_tracklet_ids.append(track.track_id)
                continue

            x1, y1, w, h = tlwh
            image_boxes_list.append([x1, y1, x1 + w, y1 + h])
            new_tracks_ids.append(track.track_id)

        if len(image_boxes_list) == 0:
            return

        # Generate masks
        masks = self._predict_masks_batch(prev_frame, image_boxes_list)

        # Combine new masks
        mask_extra = np.zeros(masks[0].shape, dtype=np.int32)
        max_mask_number = max(self.tracklet_mask_dict.values(), default=0)

        new_masks_numbers = []
        new_object_numbers = []

        for mi, m in enumerate(masks):
            current_mask = m.astype(np.int32)
            next_mask_number = max_mask_number + mi + 1
            current_mask[current_mask > 0] = next_mask_number
            new_masks_numbers.append(next_mask_number)

            self.last_object_number_cutie += 1
            new_object_numbers.append(self.last_object_number_cutie)

            non_occupied = (mask_extra == 0).astype(np.int32)
            mask_extra += current_mask * non_occupied

        # Update combined mask
        if self.mask_prediction_prev_frame is not None:
            self.mask_prediction_prev_frame[mask_extra > 0] = mask_extra[mask_extra > 0]
        else:
            self.mask_prediction_prev_frame = mask_extra

        self.num_objects += len(new_tracks_ids)

        # Update CUTIE
        mask_torch = self._index_to_one_hot(self.mask_prediction_prev_frame, self.num_objects + 1)
        self.current_object_list_cutie.extend(new_object_numbers)
        _ = self.cutie_processor.step(
            prev_frame_torch, mask_torch,
            objects=self.current_object_list_cutie, idx_mask=False
        )

        # Update dictionaries
        for track_id, mask_id in zip(new_tracks_ids, new_masks_numbers):
            self.tracklet_mask_dict[track_id] = mask_id
            self.mask_color_counter += 1
            self.mask_color_dict[mask_id] = self.mask_color_counter

    def _remove_masks(self, removed_track_ids: List[int]) -> None:
        """Remove masks for removed tracks."""
        if len(removed_track_ids) == 0:
            return

        mask_ids_to_remove = [
            self.tracklet_mask_dict[tid]
            for tid in removed_track_ids
            if tid in self.tracklet_mask_dict
        ]

        if not mask_ids_to_remove:
            return

        # Purge from CUTIE
        purge_activated, tmp_keep_idx, obj_keep_idx = \
            self.cutie_processor.object_manager.purge_selected_objects(mask_ids_to_remove)

        if purge_activated:
            self.cutie_processor.memory.purge_except(obj_keep_idx)

        self.current_object_list_cutie = obj_keep_idx
        self.num_objects = len(self.current_object_list_cutie)

        # Update dictionaries
        _update_dicts_after_removal(
            self.tracklet_mask_dict,
            self.mask_color_dict,
            mask_ids_to_remove
        )

    def _predict_masks_batch(self, frame: np.ndarray, boxes: List[List[float]]) -> List[np.ndarray]:
        """Predict masks for multiple bounding boxes."""
        boxes_tensor = torch.tensor(boxes, device=self.device)

        if hasattr(self.sam2_predictor, 'predict'):
            # SAM2 batch prediction
            masks = []
            for box in boxes:
                mask, _, _ = self.sam2_predictor.predict(
                    box=np.array(box),
                    multimask_output=False
                )
                masks.append(mask[0])
            return masks
        else:
            # SAM1 batch prediction
            transformed_boxes = self.sam2_predictor.transform.apply_boxes_torch(
                boxes_tensor, frame.shape[:2]
            )
            masks, _, _ = self.sam2_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            return [m[0].cpu().numpy() for m in masks]

    def _post_process_mask(
        self, prediction: torch.Tensor
    ) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        """Post-process CUTIE prediction."""
        # Get average probabilities
        mask_avg_prob_dict = self._get_mask_avg_prob(prediction)

        # Convert to numpy
        prediction_np = self._torch_prob_to_numpy(prediction)
        self.mask_prediction_prev_frame = prediction_np.copy()

        # Preserve colors
        prediction_colors = self._adjust_mask_colors(prediction_np)

        return prediction_np, mask_avg_prob_dict, prediction_colors

    def _get_mask_avg_prob(self, prediction: torch.Tensor) -> Dict[int, float]:
        """Get average probability for each mask."""
        mask_avg_prob_dict = {}
        mask_maxes = torch.max(prediction, dim=0).indices

        for v in self.tracklet_mask_dict.values():
            if v < prediction.shape[0]:
                avg_score = (prediction[v][mask_maxes == v]).mean().item()
                if not np.isnan(avg_score):
                    mask_avg_prob_dict[v] = avg_score

        return mask_avg_prob_dict

    def _adjust_mask_colors(self, prediction: np.ndarray) -> np.ndarray:
        """Adjust mask colors to preserve consistency after removals."""
        prediction_colors = prediction.copy()
        for k in sorted(self.mask_color_dict.keys(), reverse=True):
            prediction_colors[prediction_colors == k] = self.mask_color_dict[k]
        return prediction_colors

    def _image_to_torch(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR numpy image to torch tensor."""
        # Convert BGR to RGB
        image_rgb = image[:, :, ::-1].copy()
        # Normalize and convert to tensor
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        return image_tensor.unsqueeze(0).to(self.device)

    def _index_to_one_hot(self, mask: np.ndarray, num_classes: int) -> torch.Tensor:
        """Convert index mask to one-hot tensor."""
        one_hot = np.eye(num_classes)[mask.astype(np.int32)]
        one_hot = torch.from_numpy(one_hot).permute(2, 0, 1).float()
        return one_hot.to(self.device)

    def _torch_prob_to_numpy(self, prediction: torch.Tensor) -> np.ndarray:
        """Convert torch probability tensor to numpy index mask."""
        return torch.argmax(prediction, dim=0).cpu().numpy().astype(np.int32)

    def compute_mask_metrics(
        self,
        mask: np.ndarray,
        bbox_xyxy: np.ndarray,
        avg_conf: float
    ) -> MaskMetrics:
        """
        Compute mask metrics for association conditioning.

        Args:
            mask: Binary mask array
            bbox_xyxy: Detection bounding box
            avg_conf: Average mask confidence

        Returns:
            MaskMetrics with mm1, mm2, and avg_confidence
        """
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[1], x2)
        y2 = min(mask.shape[0], y2)

        bbox_region = mask[y1:y2, x1:x2]
        mask_pixels_in_bbox = np.sum(bbox_region > 0)
        bbox_area = (x2 - x1) * (y2 - y1)
        total_mask_pixels = np.sum(mask > 0)

        # mm1: mask coverage ratio (how much of mask is in bbox)
        mm1 = mask_pixels_in_bbox / total_mask_pixels if total_mask_pixels > 0 else 0

        # mm2: mask fill ratio (how much of bbox is filled by mask)
        mm2 = mask_pixels_in_bbox / bbox_area if bbox_area > 0 else 0

        return MaskMetrics(mm1=mm1, mm2=mm2, avg_confidence=avg_conf)


# Helper functions (ported from original)

def _get_tracklets_with_lower_bottom(
    tracklet_tlwh: np.ndarray,
    all_tlwhs: List[np.ndarray]
) -> List[np.ndarray]:
    """Get tracklets with bottom edge lower than the given tracklet."""
    y, h = tracklet_tlwh[1], tracklet_tlwh[3]
    bottom = y + h

    return [
        tlwh for tlwh in all_tlwhs
        if tlwh[1] + tlwh[3] > bottom  # > to exclude self
    ]


def _get_overlap_with_lower_tracklets(
    tracklet_tlwh: np.ndarray,
    lower_tracklets: List[np.ndarray]
) -> float:
    """Compute max overlap with lower tracklets."""
    x, y, w, h = tracklet_tlwh
    max_overlap = 0

    for lt in lower_tracklets:
        lx, ly, lw, lh = lt

        # Compute intersection
        x_dist = min(x + w, lx + lw) - max(x, lx)
        y_dist = min(y + h, ly + lh) - max(y, ly)

        if x_dist <= 0 or y_dist <= 0:
            continue

        overlap_area = x_dist * y_dist
        overlap_ratio = overlap_area / (w * h)
        max_overlap = max(max_overlap, overlap_ratio)

        if max_overlap >= 1.0:
            break

    return max_overlap


def _update_dicts_after_removal(
    tracklet_mask_dict: Dict[int, int],
    mask_color_dict: Dict[int, int],
    removed_mask_ids: List[int]
) -> None:
    """Update dictionaries after mask removal."""
    # Update tracklet_mask_dict
    entries_to_remove = []
    decrement_dict = {}

    for track_id, mask_id in tracklet_mask_dict.items():
        if mask_id in removed_mask_ids:
            entries_to_remove.append(track_id)
        else:
            decrement = sum(1 for rmi in removed_mask_ids if mask_id > rmi)
            if decrement > 0:
                decrement_dict[track_id] = decrement

    for entry in entries_to_remove:
        del tracklet_mask_dict[entry]

    for track_id, dec in decrement_dict.items():
        tracklet_mask_dict[track_id] -= dec

    # Update mask_color_dict
    entries_to_remove = []
    decrement_dict = {}

    for mask_id in list(mask_color_dict.keys()):
        if mask_id in removed_mask_ids:
            entries_to_remove.append(mask_id)
        else:
            decrement = sum(1 for rmi in removed_mask_ids if mask_id > rmi)
            if decrement > 0:
                decrement_dict[mask_id] = decrement

    for entry in entries_to_remove:
        del mask_color_dict[entry]

    for mask_id in sorted(decrement_dict.keys()):
        new_id = mask_id - decrement_dict[mask_id]
        mask_color_dict[new_id] = mask_color_dict.pop(mask_id)


def create_mask_manager(config: Optional[MaskConfig] = None) -> SAM2CutieMaskManager:
    """
    Factory function to create a mask manager.

    Args:
        config: Optional mask config. Uses defaults if None.

    Returns:
        Configured SAM2CutieMaskManager instance
    """
    if config is None:
        config = MaskConfig()
    return SAM2CutieMaskManager(config)
