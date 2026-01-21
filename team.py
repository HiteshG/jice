"""
Team classification module.

Implements two approaches for A/B comparison:
- HybridTeamClassifier: MobileNetV3 + color features + Spectral clustering
- RobustTeamClassifier: SigLIP + HDBSCAN

Ported from hockey-vision-analytics/hockey/common/team_hybrid.py and team_robust.py.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

from .config import TeamConfig
from .data_types import TeamAssignment


class HybridTeamClassifier:
    """
    Team classifier using MobileNetV3 deep features + color features.

    Features:
    - MobileNetV3 Small: 576 dims
    - Color features (HSV/LAB): 49 dims
    - Spectral clustering with RBF kernel
    - Temporal consistency via history window
    """

    def __init__(self, config: TeamConfig):
        """
        Initialize the hybrid classifier.

        Args:
            config: Team classification configuration
        """
        self.config = config
        self.device = config.device
        self.n_clusters = config.hybrid_n_clusters
        self.history_window = config.hybrid_history_window

        # Feature extractor
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Clustering
        self.cluster_model = None
        self.cluster_centers = None
        self.team_mapping = {}  # cluster_id -> team_id

        # Temporal history
        self.track_history: Dict[int, List[int]] = defaultdict(list)

        self._load_model()

    def _load_model(self) -> None:
        """Load MobileNetV3 feature extractor."""
        try:
            self.model = models.mobilenet_v3_small(pretrained=True)
            # Remove classifier, keep features
            self.model.classifier = nn.Identity()
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"Warning: Failed to load MobileNetV3: {e}")
            self.model = None

    def reset(self) -> None:
        """Reset classifier state."""
        self.cluster_model = None
        self.cluster_centers = None
        self.team_mapping = {}
        self.track_history.clear()

    def extract_jersey_region(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract jersey region from player crop.

        Focuses on upper 60% (10%-60% vertically, 20%-80% horizontally).
        Returns original crop if region would be empty.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return crop if crop is not None else np.zeros((1, 1, 3), dtype=np.uint8)

        h, w = crop.shape[:2]
        y1 = int(h * 0.1)
        y2 = int(h * 0.6)
        x1 = int(w * 0.2)
        x2 = int(w * 0.8)

        # Ensure region is valid
        if y2 <= y1 or x2 <= x1:
            return crop

        return crop[y1:y2, x1:x2]

    def extract_deep_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract deep features using MobileNetV3.

        Args:
            crops: List of BGR player crops

        Returns:
            (N, 576) feature array
        """
        if self.model is None or len(crops) == 0:
            return np.zeros((len(crops), 576))

        features = []
        with torch.no_grad():
            for crop in crops:
                # Skip invalid crops
                if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                    features.append(np.zeros(576))
                    continue

                try:
                    jersey = self.extract_jersey_region(crop)
                    rgb = cv2.cvtColor(jersey, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(rgb).unsqueeze(0).to(self.device)
                    feat = self.model(tensor).cpu().numpy().flatten()
                    features.append(feat)
                except Exception:
                    features.append(np.zeros(576))

        return np.array(features) if features else np.zeros((0, 576))

    def extract_color_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract color features (HSV/LAB histograms + statistics).

        Returns 49-dimensional feature vector per crop.
        """
        features = []

        for crop in crops:
            # Skip invalid crops
            if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                features.append(np.zeros(49))
                continue

            try:
                jersey = self.extract_jersey_region(crop)

                # Convert to HSV and LAB
                hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)

                # H histogram (18 bins)
                h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
                h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)

                # S histogram (8 bins)
                s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
                s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)

                # V histogram (8 bins)
                v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
                v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)

                # HSV mean and std (6 dims)
                hsv_mean = np.mean(hsv, axis=(0, 1)) / 255.0
                hsv_std = np.std(hsv, axis=(0, 1)) / 255.0

                # LAB mean and std (6 dims)
                lab_mean = np.mean(lab, axis=(0, 1)) / 255.0
                lab_std = np.std(lab, axis=(0, 1)) / 255.0

                # Saturation ratios (2 dims)
                s_channel = hsv[:, :, 1]
                low_sat_ratio = np.mean(s_channel < 40) / 255.0
                high_sat_ratio = np.mean(s_channel > 150) / 255.0

                # White ratio (1 dim)
                white_ratio = self._compute_white_ratio(jersey)

                # Concatenate all features (49 dims)
                feat = np.concatenate([
                    h_hist, s_hist, v_hist,  # 34
                    hsv_mean, hsv_std,        # 6
                    lab_mean, lab_std,        # 6
                    [low_sat_ratio, high_sat_ratio],  # 2
                    [white_ratio]             # 1
                ])

                features.append(feat)
            except Exception:
                features.append(np.zeros(49))

        return np.array(features) if features else np.zeros((0, 49))

    def _compute_white_ratio(self, crop: np.ndarray) -> float:
        """Compute ratio of white pixels."""
        if crop is None or crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
            return 0.0
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # White: low saturation, high value
            white_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
            return np.mean(white_mask)
        except Exception:
            return 0.0

    def extract_all_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract combined deep + color features.

        Returns:
            (N, 625) feature array
        """
        deep_feats = self.extract_deep_features(crops)
        color_feats = self.extract_color_features(crops)
        return np.concatenate([deep_feats, color_feats], axis=1)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit clustering model on initial crops.

        Args:
            crops: List of player crops for initial clustering
        """
        if len(crops) < self.n_clusters * 2:
            return

        try:
            from sklearn.cluster import SpectralClustering

            features = self.extract_all_features(crops)

            self.cluster_model = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity='rbf',
                gamma=1.0,
                n_init=10
            )

            labels = self.cluster_model.fit_predict(features)

            # Analyze clusters to determine white vs colored
            self._analyze_clusters(crops, labels)

        except ImportError:
            print("Warning: sklearn not available, using simple clustering")
            self.cluster_model = None

    def _analyze_clusters(self, crops: List[np.ndarray], labels: np.ndarray) -> None:
        """
        Analyze clusters to map to team IDs.

        Team 0 = white/away, Team 1 = colored/home
        """
        cluster_white_ratios = defaultdict(list)

        for i, crop in enumerate(crops):
            label = labels[i]
            jersey = self.extract_jersey_region(crop)
            white_ratio = self._compute_white_ratio(jersey)
            cluster_white_ratios[label].append(white_ratio)

        # Compute mean white ratio per cluster
        mean_ratios = {}
        for label, ratios in cluster_white_ratios.items():
            mean_ratios[label] = np.mean(ratios)

        # Map higher white ratio to team 0 (away)
        sorted_labels = sorted(mean_ratios.keys(), key=lambda k: mean_ratios[k], reverse=True)
        self.team_mapping = {sorted_labels[i]: i for i in range(len(sorted_labels))}

    def predict(
        self,
        crops: List[np.ndarray],
        tracker_ids: Optional[List[int]] = None
    ) -> List[TeamAssignment]:
        """
        Predict team assignments for crops.

        Args:
            crops: List of BGR player crops
            tracker_ids: Optional track IDs for temporal consistency

        Returns:
            List of TeamAssignment objects
        """
        if not crops:
            return []

        # If clustering not fitted, use simple method
        if self.cluster_model is None:
            return self._simple_classify(crops, tracker_ids)

        # Extract features and predict
        features = self.extract_all_features(crops)
        raw_predictions = self.cluster_model.fit_predict(features)

        # Map to team IDs
        predictions = []
        for i, label in enumerate(raw_predictions):
            team_id = self.team_mapping.get(label, 0)
            predictions.append(TeamAssignment(
                team_id=team_id,
                confidence=0.8,
                method="hybrid"
            ))

        # Apply temporal consistency
        if tracker_ids is not None:
            predictions = self._apply_temporal_consistency(predictions, tracker_ids)

        return predictions

    def _simple_classify(
        self,
        crops: List[np.ndarray],
        tracker_ids: Optional[List[int]] = None
    ) -> List[TeamAssignment]:
        """
        Simple color-based classification fallback.

        White/low-saturation -> Team 0 (away)
        Colored -> Team 1 (home)
        """
        predictions = []

        for crop in crops:
            # Skip invalid crops
            if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                predictions.append(TeamAssignment(team_id=-1, confidence=0.0, method="invalid"))
                continue

            try:
                jersey = self.extract_jersey_region(crop)
                white_ratio = self._compute_white_ratio(jersey)

                # Compute average saturation
                hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
                avg_saturation = np.mean(hsv[:, :, 1])

                if white_ratio > self.config.white_ratio_threshold or \
                   avg_saturation < self.config.low_saturation_threshold:
                    team_id = 0  # Away/white
                else:
                    team_id = 1  # Home/colored

                predictions.append(TeamAssignment(
                    team_id=team_id,
                    confidence=0.6,
                    method="fallback"
                ))
            except Exception:
                predictions.append(TeamAssignment(team_id=-1, confidence=0.0, method="error"))

        if tracker_ids is not None:
            predictions = self._apply_temporal_consistency(predictions, tracker_ids)

        return predictions

    def _apply_temporal_consistency(
        self,
        predictions: List[TeamAssignment],
        tracker_ids: List[int]
    ) -> List[TeamAssignment]:
        """
        Apply temporal consistency using history window.

        Uses majority vote over last N frames.
        """
        result = []

        for i, (pred, track_id) in enumerate(zip(predictions, tracker_ids)):
            # Update history
            self.track_history[track_id].append(pred.team_id)

            # Trim to window size
            if len(self.track_history[track_id]) > self.history_window:
                self.track_history[track_id] = \
                    self.track_history[track_id][-self.history_window:]

            # Majority vote
            history = self.track_history[track_id]
            if len(history) >= 3:
                team_id = int(np.round(np.mean(history)))
                confidence = history.count(team_id) / len(history)
            else:
                team_id = pred.team_id
                confidence = pred.confidence

            result.append(TeamAssignment(
                team_id=team_id,
                confidence=confidence,
                method=pred.method
            ))

        return result


class RobustTeamClassifier:
    """
    Robust team classifier using SigLIP + HDBSCAN.

    Features:
    - SigLIP embeddings: 768 dims
    - Color features with jersey masking: 43 dims (scaled 20x)
    - HDBSCAN for robust clustering
    - Extended temporal consistency (20-frame window)
    """

    def __init__(self, config: TeamConfig):
        """
        Initialize the robust classifier.

        Args:
            config: Team classification configuration
        """
        self.config = config
        self.device = config.device
        self.history_window = config.robust_history_window
        self.color_weight = config.robust_color_weight

        # Feature extractor
        self.model = None
        self.processor = None
        self.transform = None

        # Clustering
        self.cluster_model = None
        self.cluster_centers = None
        self.team_mapping = {}

        # Temporal history
        self.track_history: Dict[int, List[int]] = defaultdict(list)
        self.track_confidences: Dict[int, List[float]] = defaultdict(list)

        self._load_model()

    def _load_model(self) -> None:
        """Load SigLIP model."""
        try:
            from transformers import AutoProcessor, AutoModel

            self.processor = AutoProcessor.from_pretrained(
                self.config.robust_model_name
            )
            self.model = AutoModel.from_pretrained(
                self.config.robust_model_name
            ).to(self.device)
            self.model.eval()

        except ImportError:
            print("Warning: transformers not available, falling back to MobileNet")
            self._fallback_to_mobilenet()
        except Exception as e:
            print(f"Warning: Failed to load SigLIP: {e}")
            self._fallback_to_mobilenet()

    def _fallback_to_mobilenet(self) -> None:
        """Fall back to MobileNet if SigLIP unavailable."""
        try:
            self.model = models.mobilenet_v3_small(pretrained=True)
            self.model.classifier = nn.Identity()
            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            self.processor = None  # Signal fallback mode

        except Exception as e:
            print(f"Warning: Failed to load MobileNet fallback: {e}")
            self.model = None

    def reset(self) -> None:
        """Reset classifier state."""
        self.cluster_model = None
        self.cluster_centers = None
        self.team_mapping = {}
        self.track_history.clear()
        self.track_confidences.clear()

    def extract_deep_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract deep features using SigLIP or MobileNet.

        Returns:
            (N, 768) for SigLIP or (N, 576) for MobileNet
        """
        if self.model is None or len(crops) == 0:
            dim = 768 if self.processor else 576
            return np.zeros((len(crops), dim))

        dim = 768 if self.processor else 576
        features = []
        with torch.no_grad():
            for crop in crops:
                # Skip invalid crops
                if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                    features.append(np.zeros(dim))
                    continue

                try:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                    if self.processor:
                        # SigLIP
                        inputs = self.processor(images=rgb, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = self.model.get_image_features(**inputs)
                        feat = outputs.cpu().numpy().flatten()
                    else:
                        # MobileNet fallback
                        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
                        feat = self.model(tensor).cpu().numpy().flatten()

                    features.append(feat)
                except Exception:
                    features.append(np.zeros(dim))

        return np.array(features) if features else np.zeros((0, dim))

    def extract_color_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract color features with jersey focus.

        Returns 43-dimensional feature vector per crop.
        """
        features = []

        for crop in crops:
            # Skip invalid crops
            if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                features.append(np.zeros(43))
                continue

            try:
                h, w = crop.shape[:2]

                # Focus on jersey region
                y1, y2 = int(h * 0.1), int(h * 0.6)
                x1, x2 = int(w * 0.2), int(w * 0.8)

                # Ensure valid region
                if y2 <= y1 or x2 <= x1:
                    jersey = crop
                else:
                    jersey = crop[y1:y2, x1:x2]

                hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)

                # H histogram (18 bins)
                h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
                h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)

                # S histogram (16 bins)
                s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
                s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)

                # HSV mean (3 dims)
                hsv_mean = np.mean(hsv, axis=(0, 1)) / 255.0

                # LAB mean (3 dims)
                lab_mean = np.mean(lab, axis=(0, 1)) / 255.0

                # Saturation ratios (3 dims)
                s_channel = hsv[:, :, 1]
                low_sat = np.mean(s_channel < 40)
                mid_sat = np.mean((s_channel >= 40) & (s_channel < 150))
                high_sat = np.mean(s_channel >= 150)

                # Concatenate (43 dims)
                feat = np.concatenate([
                    h_hist,  # 18
                    s_hist,  # 16
                    hsv_mean,  # 3
                    lab_mean,  # 3
                    [low_sat, mid_sat, high_sat]  # 3
                ])

                features.append(feat)
            except Exception:
                features.append(np.zeros(43))

        return np.array(features) if features else np.zeros((0, 43))

    def extract_multimodal_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract combined deep + color features with scaling.

        Color features are scaled by color_weight to match deep feature scale.
        """
        deep_feats = self.extract_deep_features(crops)
        color_feats = self.extract_color_features(crops) * self.color_weight

        return np.concatenate([deep_feats, color_feats], axis=1)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit HDBSCAN clustering on initial crops.

        Args:
            crops: List of player crops for initial clustering
        """
        if len(crops) < self.config.robust_min_cluster_size * 2:
            return

        try:
            import hdbscan
            from sklearn.decomposition import PCA

            features = self.extract_multimodal_features(crops)

            # Dimensionality reduction
            n_components = min(50, features.shape[1], features.shape[0])
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(features)

            # HDBSCAN clustering
            self.cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=self.config.robust_min_cluster_size,
                min_samples=self.config.robust_min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )

            labels = self.cluster_model.fit_predict(reduced_features)

            # Analyze clusters
            self._analyze_clusters(crops, labels)

        except ImportError:
            print("Warning: hdbscan not available, using simple clustering")
            self.cluster_model = None

    def _analyze_clusters(self, crops: List[np.ndarray], labels: np.ndarray) -> None:
        """
        Analyze clusters to map to team IDs.
        """
        cluster_white_ratios = defaultdict(list)

        for i, crop in enumerate(crops):
            label = labels[i]
            if label == -1:  # Outlier
                continue

            # Skip invalid crops
            if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue

            try:
                # Compute white ratio
                h, w = crop.shape[:2]
                y1, y2 = int(h*0.1), int(h*0.6)
                x1, x2 = int(w*0.2), int(w*0.8)
                if y2 > y1 and x2 > x1:
                    jersey = crop[y1:y2, x1:x2]
                else:
                    jersey = crop
                hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
                white_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
                white_ratio = np.mean(white_mask)

                cluster_white_ratios[label].append(white_ratio)
            except Exception:
                continue

        # Map clusters to teams
        mean_ratios = {}
        for label, ratios in cluster_white_ratios.items():
            mean_ratios[label] = np.mean(ratios)

        sorted_labels = sorted(mean_ratios.keys(), key=lambda k: mean_ratios[k], reverse=True)
        self.team_mapping = {sorted_labels[i]: i for i in range(len(sorted_labels))}

    def predict(
        self,
        crops: List[np.ndarray],
        tracker_ids: Optional[List[int]] = None
    ) -> List[TeamAssignment]:
        """
        Predict team assignments for crops.

        Args:
            crops: List of BGR player crops
            tracker_ids: Optional track IDs for temporal consistency

        Returns:
            List of TeamAssignment objects
        """
        if not crops:
            return []

        # If clustering not fitted, use fallback
        if self.cluster_model is None:
            return self._fallback_classify(crops, tracker_ids)

        # Predict with clustering
        features = self.extract_multimodal_features(crops)

        # Need to reduce dimensionality same as fit
        from sklearn.decomposition import PCA
        n_components = min(50, features.shape[1], features.shape[0])
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(features)

        labels, _ = hdbscan.approximate_predict(self.cluster_model, reduced)

        predictions = []
        for i, label in enumerate(labels):
            if label == -1:
                # Outlier - use fallback
                pred = self._fallback_single(crops[i])
                pred.is_outlier = True
            else:
                team_id = self.team_mapping.get(label, 0)
                pred = TeamAssignment(
                    team_id=team_id,
                    confidence=0.85,
                    method="robust"
                )
            predictions.append(pred)

        # Apply temporal consistency
        if tracker_ids is not None:
            predictions = self._apply_temporal_consistency(predictions, tracker_ids)

        return predictions

    def _fallback_classify(
        self,
        crops: List[np.ndarray],
        tracker_ids: Optional[List[int]] = None
    ) -> List[TeamAssignment]:
        """Fallback to simple color-based classification."""
        predictions = [self._fallback_single(crop) for crop in crops]

        if tracker_ids is not None:
            predictions = self._apply_temporal_consistency(predictions, tracker_ids)

        return predictions

    def _fallback_single(self, crop: np.ndarray) -> TeamAssignment:
        """Classify single crop using color heuristics."""
        # Validate crop
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return TeamAssignment(team_id=-1, confidence=0.0, method="invalid")

        try:
            h, w = crop.shape[:2]
            y1, y2 = int(h*0.1), int(h*0.6)
            x1, x2 = int(w*0.2), int(w*0.8)
            if y2 > y1 and x2 > x1:
                jersey = crop[y1:y2, x1:x2]
            else:
                jersey = crop

            hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
            white_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
            white_ratio = np.mean(white_mask)
            avg_sat = np.mean(hsv[:, :, 1])

            if white_ratio > 0.3 or avg_sat < 40:
                team_id = 0
            else:
                team_id = 1

            return TeamAssignment(
                team_id=team_id,
                confidence=0.5,
                method="fallback"
            )
        except Exception:
            return TeamAssignment(team_id=-1, confidence=0.0, method="error")

    def _apply_temporal_consistency(
        self,
        predictions: List[TeamAssignment],
        tracker_ids: List[int]
    ) -> List[TeamAssignment]:
        """Apply temporal consistency with confidence weighting."""
        result = []

        for pred, track_id in zip(predictions, tracker_ids):
            # Update history
            self.track_history[track_id].append(pred.team_id)
            self.track_confidences[track_id].append(pred.confidence)

            # Trim to window
            if len(self.track_history[track_id]) > self.history_window:
                self.track_history[track_id] = \
                    self.track_history[track_id][-self.history_window:]
                self.track_confidences[track_id] = \
                    self.track_confidences[track_id][-self.history_window:]

            # Weighted majority vote
            history = self.track_history[track_id]
            confidences = self.track_confidences[track_id]

            if len(history) >= 5:
                # Weight by confidence
                votes = defaultdict(float)
                for team, conf in zip(history, confidences):
                    votes[team] += conf

                team_id = max(votes.keys(), key=lambda k: votes[k])
                total_conf = votes[team_id] / sum(confidences)
            else:
                team_id = pred.team_id
                total_conf = pred.confidence

            result.append(TeamAssignment(
                team_id=team_id,
                confidence=total_conf,
                is_outlier=pred.is_outlier,
                method=pred.method
            ))

        return result


class TeamClassifier:
    """
    Unified team classifier supporting both approaches.

    Provides A/B comparison capability.
    """

    def __init__(self, config: TeamConfig):
        """
        Initialize team classifier.

        Args:
            config: Team configuration
        """
        self.config = config
        self.classifier_type = config.classifier_type

        # Initialize classifiers
        if self.classifier_type in ["hybrid", "compare"]:
            self.hybrid = HybridTeamClassifier(config)
        else:
            self.hybrid = None

        if self.classifier_type in ["robust", "compare"]:
            self.robust = RobustTeamClassifier(config)
        else:
            self.robust = None

    def reset(self) -> None:
        """Reset all classifier state."""
        if self.hybrid:
            self.hybrid.reset()
        if self.robust:
            self.robust.reset()

    def fit(self, crops: List[np.ndarray]) -> None:
        """Fit clustering models."""
        if self.hybrid:
            self.hybrid.fit(crops)
        if self.robust:
            self.robust.fit(crops)

    def predict(
        self,
        crops: List[np.ndarray],
        tracker_ids: Optional[List[int]] = None
    ) -> List[TeamAssignment]:
        """
        Predict team assignments.

        If compare mode, uses hybrid by default.
        """
        if self.classifier_type == "hybrid" and self.hybrid:
            return self.hybrid.predict(crops, tracker_ids)
        elif self.classifier_type == "robust" and self.robust:
            return self.robust.predict(crops, tracker_ids)
        elif self.classifier_type == "compare" and self.hybrid:
            return self.hybrid.predict(crops, tracker_ids)
        else:
            # Fallback
            return [TeamAssignment(team_id=-1, confidence=0.0, method="none")
                    for _ in crops]

    def predict_both(
        self,
        crops: List[np.ndarray],
        tracker_ids: Optional[List[int]] = None
    ) -> Tuple[List[TeamAssignment], List[TeamAssignment]]:
        """
        Predict with both classifiers for comparison.

        Returns:
            Tuple of (hybrid_predictions, robust_predictions)
        """
        hybrid_preds = self.hybrid.predict(crops, tracker_ids) if self.hybrid else []
        robust_preds = self.robust.predict(crops, tracker_ids) if self.robust else []

        return hybrid_preds, robust_preds


def create_team_classifier(config: Optional[TeamConfig] = None) -> TeamClassifier:
    """
    Factory function to create a team classifier.

    Args:
        config: Optional team config. Uses defaults if None.

    Returns:
        Configured TeamClassifier instance
    """
    if config is None:
        config = TeamConfig()
    return TeamClassifier(config)
