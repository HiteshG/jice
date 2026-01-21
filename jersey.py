"""
Jersey number recognition pipeline.

Full pipeline: ReID -> Gaussian Outlier -> Legibility -> Pose -> Crop -> PARSeq -> Vote -> Lock

Ported from jersey-number-pipeline/ with temporal locking for jitter prevention.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from scipy.special import softmax

from .config import JerseyConfig
from .data_types import JerseyPrediction


# Constants from original pipeline
PADDING = 5
CONFIDENCE_THRESHOLD = 0.4
HEIGHT_MIN = 35
WIDTH_MIN = 30
TS = 2.367  # Temperature scaling constant

# Digit bias (double digit more likely for jerseys)
BIAS_FOR_DIGITS = [0.06, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094]
TOKEN_LIST = 'E0123456789'  # E = end token

SUM_THRESHOLD = 1.0
FILTER_THRESHOLD = 0.2


class LegibilityClassifier:
    """
    Binary classifier for frame legibility.

    Filters out frames where jersey numbers are not visible/readable.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize legibility classifier.

        Args:
            model_path: Path to ResNet34 checkpoint
            device: Device to run on
        """
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load the legibility model."""
        try:
            self.model = models.resnet34(pretrained=False)

            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)

                # Handle state dict with 'model_ft.' prefix
                if all(k.startswith('model_ft.') for k in checkpoint.keys()):
                    checkpoint = {k.replace('model_ft.', ''): v for k, v in checkpoint.items()}

                # Detect output size from checkpoint
                fc_weight = checkpoint.get('fc.weight')
                if fc_weight is not None:
                    num_classes = fc_weight.shape[0]
                    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
                else:
                    # Default to binary classifier
                    self.model.fc = nn.Linear(self.model.fc.in_features, 2)

                self.model.load_state_dict(checkpoint)
            else:
                # No checkpoint, use binary classifier
                self.model.fc = nn.Linear(self.model.fc.in_features, 2)

            self.model.to(self.device)
            self.model.eval()
            self._is_binary = (self.model.fc.out_features == 2)

        except Exception as e:
            print(f"Warning: Failed to load legibility model: {e}")
            self.model = None
            self._is_binary = True

    def predict(self, crop: np.ndarray, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Predict if a crop is legible.

        Args:
            crop: BGR image crop
            threshold: Classification threshold

        Returns:
            Tuple of (is_legible, confidence)
        """
        if self.model is None:
            return True, 1.0

        # Validate crop
        if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return True, 1.0

        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)

                # Handle both binary and single-class models
                if self._is_binary:
                    # Binary classifier: [not_legible, legible]
                    probs = F.softmax(output, dim=1)
                    legible_prob = probs[0, 1].item()
                else:
                    # Single-class regression: direct confidence score
                    legible_prob = torch.sigmoid(output[0, 0]).item()

            return legible_prob >= threshold, legible_prob

        except Exception:
            return True, 1.0


class PARSeqRecognizer:
    """
    Scene text recognition using PARSeq model.

    Recognizes 1-2 digit jersey numbers from torso crops.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize PARSeq recognizer.

        Args:
            model_path: Path to PARSeq checkpoint
            device: Device to run on
        """
        self.device = device
        self.model = None
        self.img_size = (32, 128)  # PARSeq default

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load the PARSeq model."""
        try:
            # Try to load PARSeq
            from strhub.models.utils import load_from_checkpoint

            if Path(model_path).exists():
                self.model = load_from_checkpoint(model_path).to(self.device)
                self.model.eval()

        except ImportError:
            print("Warning: strhub not available, using fallback STR")
            self.model = None
        except Exception as e:
            print(f"Warning: Failed to load PARSeq model: {e}")
            self.model = None

    def predict(self, crop: np.ndarray, use_temperature: bool = True) -> Tuple[Optional[str], float, List[float]]:
        """
        Recognize text in a crop.

        Args:
            crop: BGR image crop (torso region)
            use_temperature: Apply temperature scaling for calibration

        Returns:
            Tuple of (predicted_text, confidence, per_char_confidences)
        """
        if self.model is None:
            return None, 0.0, []

        # Validate crop
        if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return None, 0.0, []

        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)

                if use_temperature:
                    logits = logits / TS

                probs = F.softmax(logits, dim=-1)

                # Greedy decode
                pred_ids = probs.argmax(dim=-1)
                confidences = probs.max(dim=-1).values

                # Convert to string
                pred_text = ""
                char_confs = []
                for i, (idx, conf) in enumerate(zip(pred_ids[0], confidences[0])):
                    if idx == 0:  # End token
                        break
                    char = TOKEN_LIST[idx.item()]
                    if char != 'E':
                        pred_text += char
                        char_confs.append(conf.item())

                # Validate: must be 1-2 digits, 0-99
                if self._is_valid_number(pred_text):
                    mean_conf = np.mean(char_confs) if char_confs else 0.0
                    return pred_text, mean_conf, char_confs

                return None, 0.0, []

        except Exception as e:
            return None, 0.0, []

    def predict_with_logits(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get raw logits for Bayesian aggregation.

        Returns:
            Tuple of (tens_logits, units_logits) as (11,) arrays
        """
        if self.model is None:
            return np.zeros(11), np.zeros(11)

        # Validate crop
        if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros(11), np.zeros(11)

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                logits = logits / TS

                # Get first two positions (tens, units)
                tens_logits = F.softmax(logits[0, 0, :11], dim=-1).cpu().numpy()
                units_logits = F.softmax(logits[0, 1, :11], dim=-1).cpu().numpy()

                return tens_logits, units_logits

        except Exception:
            return np.zeros(11), np.zeros(11)

    @staticmethod
    def _is_valid_number(text: str) -> bool:
        """Check if text is a valid jersey number (1-99)."""
        if not text or len(text) > 2:
            return False
        try:
            num = int(text)
            return 0 < num < 100
        except ValueError:
            return False


class JerseyRecognizer:
    """
    Full jersey number recognition pipeline with temporal stability.

    Processes tracklets and maintains locked predictions to prevent jitter.
    """

    def __init__(self, config: JerseyConfig):
        """
        Initialize the jersey recognizer.

        Args:
            config: Jersey recognition configuration
        """
        self.config = config
        self.device = config.device

        # Models
        self.legibility_classifier = LegibilityClassifier(
            config.legibility_model, config.device
        )
        self.parseq = PARSeqRecognizer(config.str_model, config.device)

        # Temporal state for jitter prevention
        self.track_predictions: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.locked_numbers: Dict[int, int] = {}  # track_id -> locked number
        self.lock_confidence: Dict[int, float] = {}  # track_id -> total confidence

    def reset(self) -> None:
        """Reset all temporal state."""
        self.track_predictions.clear()
        self.locked_numbers.clear()
        self.lock_confidence.clear()

    def reset_track(self, track_id: int) -> None:
        """Reset temporal state for a specific track (for re-classification)."""
        if track_id in self.track_predictions:
            del self.track_predictions[track_id]
        if track_id in self.locked_numbers:
            del self.locked_numbers[track_id]
        if track_id in self.lock_confidence:
            del self.lock_confidence[track_id]

    def process_crop(
        self,
        crop: np.ndarray,
        track_id: int,
        use_pose_crop: bool = False
    ) -> Optional[int]:
        """
        Process a single crop and update track predictions.

        Args:
            crop: BGR player crop
            track_id: Track ID for temporal aggregation
            use_pose_crop: Whether to apply torso cropping

        Returns:
            Current best prediction (may be None if not confident)
        """
        # Check if already locked
        if track_id in self.locked_numbers:
            return self.locked_numbers[track_id]

        # Check legibility
        is_legible, leg_conf = self.legibility_classifier.predict(
            crop, self.config.legibility_threshold
        )
        if not is_legible:
            return self._get_current_prediction(track_id)

        # Optionally crop to torso (simplified - no pose estimation)
        if use_pose_crop:
            crop = self._simple_torso_crop(crop)

        # Skip if crop too small
        if crop.shape[0] < HEIGHT_MIN or crop.shape[1] < WIDTH_MIN:
            return self._get_current_prediction(track_id)

        # Run PARSeq
        pred_text, confidence, char_confs = self.parseq.predict(crop)

        if pred_text and confidence >= self.config.filter_threshold:
            number = int(pred_text)
            self.track_predictions[track_id].append((number, confidence))

            # Check if should lock
            total_conf = self._compute_total_confidence(track_id, number)
            if total_conf >= self.config.lock_threshold:
                self.locked_numbers[track_id] = number
                self.lock_confidence[track_id] = total_conf

        return self._get_current_prediction(track_id)

    def process_tracklet(
        self,
        crops: List[np.ndarray],
        track_id: int
    ) -> Optional[int]:
        """
        Process all crops from a tracklet for final prediction.

        Uses Bayesian aggregation for better accuracy.

        Args:
            crops: List of BGR player crops
            track_id: Track ID

        Returns:
            Final jersey number prediction or None
        """
        if track_id in self.locked_numbers:
            return self.locked_numbers[track_id]

        if not crops:
            return None

        # Collect predictions
        all_predictions = []

        for crop in crops:
            # Check legibility
            is_legible, _ = self.legibility_classifier.predict(
                crop, self.config.legibility_threshold
            )
            if not is_legible:
                continue

            # Simple torso crop
            torso = self._simple_torso_crop(crop)
            if torso.shape[0] < HEIGHT_MIN or torso.shape[1] < WIDTH_MIN:
                continue

            # Get logits
            tens_logits, units_logits = self.parseq.predict_with_logits(torso)
            all_predictions.append((tens_logits, units_logits))

        if not all_predictions:
            return None

        # Bayesian aggregation
        number = self._bayesian_aggregate(all_predictions)

        if number is not None:
            self.locked_numbers[track_id] = number

        return number

    def _bayesian_aggregate(
        self,
        predictions: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[int]:
        """
        Aggregate predictions using log-likelihood sum.

        Args:
            predictions: List of (tens_probs, units_probs) tuples

        Returns:
            Final predicted number or None
        """
        if not predictions:
            return None

        # Initialize with bias priors
        if self.config.use_bias:
            tens_priors = np.array(BIAS_FOR_DIGITS)
            units_priors = np.array(BIAS_FOR_DIGITS)
        else:
            tens_priors = np.ones(11) / 11
            units_priors = np.ones(11) / 11

        # Sum log-likelihoods
        sum_logl_tens = np.zeros(11)
        sum_logl_units = np.zeros(11)

        for tens_prob, units_prob in predictions:
            # Add small epsilon to avoid log(0)
            sum_logl_tens += np.log(tens_prob + 1e-10)
            sum_logl_units += np.log(units_prob + 1e-10)

        # Add log priors
        sum_logl_tens += np.log(tens_priors)
        sum_logl_units += np.log(units_priors)

        # Get argmax
        tens_digit = np.argmax(sum_logl_tens)
        units_digit = np.argmax(sum_logl_units)

        # Construct number
        if tens_digit == 0:  # End token = single digit
            number = units_digit
        else:
            number = tens_digit * 10 + units_digit

        # Validate
        if 0 < number < 100:
            return number

        return None

    def _simple_torso_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Simple torso cropping (upper 60%, center 60%).

        For full accuracy, use ViTPose-based cropping.
        """
        h, w = crop.shape[:2]

        # Vertical: top 10% to 60%
        y1 = int(h * 0.1)
        y2 = int(h * 0.6)

        # Horizontal: center 60%
        x1 = int(w * 0.2)
        x2 = int(w * 0.8)

        return crop[y1:y2, x1:x2]

    def _compute_total_confidence(self, track_id: int, number: int) -> float:
        """Compute total confidence for a number across all predictions."""
        total = 0.0
        for pred_num, conf in self.track_predictions[track_id]:
            if pred_num == number:
                total += conf
        return total

    def _get_current_prediction(self, track_id: int) -> Optional[int]:
        """Get current best prediction for a track."""
        if track_id in self.locked_numbers:
            return self.locked_numbers[track_id]

        predictions = self.track_predictions.get(track_id, [])
        if not predictions:
            return None

        # Confidence-weighted voting
        votes = defaultdict(float)
        for number, conf in predictions:
            votes[number] += conf

        if not votes:
            return None

        best_number = max(votes.keys(), key=lambda k: votes[k])
        best_conf = votes[best_number]

        if best_conf >= self.config.sum_threshold:
            return best_number

        return None

    def on_track_lost(self, track_id: int) -> None:
        """
        Handle track loss event.

        Unlocks the number so it can be re-evaluated if track reappears.
        """
        if track_id in self.locked_numbers:
            del self.locked_numbers[track_id]
        if track_id in self.lock_confidence:
            del self.lock_confidence[track_id]
        self.track_predictions.pop(track_id, None)

    def get_prediction(self, track_id: int) -> JerseyPrediction:
        """
        Get jersey prediction for a track.

        Returns:
            JerseyPrediction with number and confidence
        """
        if track_id in self.locked_numbers:
            return JerseyPrediction(
                number=self.locked_numbers[track_id],
                confidence=self.lock_confidence.get(track_id, 1.0),
                is_legible=True
            )

        current = self._get_current_prediction(track_id)
        if current is not None:
            # Compute confidence
            votes = defaultdict(float)
            for number, conf in self.track_predictions.get(track_id, []):
                votes[number] += conf
            conf = votes.get(current, 0.0)

            return JerseyPrediction(
                number=current,
                confidence=conf,
                is_legible=True
            )

        return JerseyPrediction(number=None, confidence=0.0, is_legible=False)


def create_jersey_recognizer(config: Optional[JerseyConfig] = None) -> JerseyRecognizer:
    """
    Factory function to create a jersey recognizer.

    Args:
        config: Optional jersey config. Uses defaults if None.

    Returns:
        Configured JerseyRecognizer instance
    """
    if config is None:
        config = JerseyConfig()
    return JerseyRecognizer(config)
