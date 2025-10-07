import csv
import numpy as np
import random
import warnings
import signal
import sys
import os
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

from load_datasets import *
def detect_model_from_args():
    for i, arg in enumerate(sys.argv):
        if arg in ['--model', '-m'] and i + 1 < len(sys.argv):
            mt = sys.argv[i + 1].lower()
            if mt in ['llava', 'qwen', 'internvl']:
                return mt
        elif arg.lower() in ['llava', 'qwen', 'internvl'] and i > 0:
            return arg.lower()
    return 'llava'

def get_extractor_class(model_type):
    if model_type == 'internvl':
        from feature_extractor_internvl import HiddenStateExtractor as H
        return H
    elif model_type == 'qwen':
        from feature_extractor_qwen import HiddenStateExtractor as H
        return H
    else:
        from feature_extractor import HiddenStateExtractor as H
        return H

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
class PCAConfig:
    """Global configuration for PCA dimensionality reduction"""

    # PCA mode
    # "single_layer": Train one PCA on layer 16, use for all layers (original method)
    # "layer_specific": Train separate PCA for each layer (new method)
    PCA_MODE = "layer_specific"  # Change this to switch modes

    # Single layer mode settings
    SINGLE_LAYER_TRAINING_LAYER = 16  # Will be overridden to 20 for InternVL

    # PCA settings
    INPUT_DIM = 4096
    PCA_COMPONENTS_LIST = [32, 64, 128, 256]  # Various number of components to try

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*80)
        print("PCA CONFIGURATION")
        print("="*80)
        print(f"Mode: {cls.PCA_MODE}")
        if cls.PCA_MODE == "single_layer":
            print(f"Training layer: {cls.SINGLE_LAYER_TRAINING_LAYER}")
        print(f"Input dimension: {cls.INPUT_DIM}")
        print(f"PCA components to try: {cls.PCA_COMPONENTS_LIST}")
        print("="*80)

# Global config instance
CONFIG = PCAConfig()

def setup_gpu_environment():
    if torch.cuda.is_available():
        # Use both GPUs if available
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA devices")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")

        # Set memory fraction to avoid OOM (RTX 4090 has 24GB)
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        return torch.device('cuda:0')  # Primary GPU
    else:
        print("CUDA not available, falling back to CPU")
        return torch.device('cpu')

GPU_DEVICE = setup_gpu_environment()

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_info():
    """Get GPU memory usage information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
    return "GPU not available"


class GPUPCAProjection:
    """
    GPU-accelerated PCA-based dimensionality reduction from 4096 to specified number of components.
    Trained only on training data and applied to test data using PyTorch on GPU.
    """

    def __init__(self, n_components=256, device=None):
        """
        Initialize GPU PCA projection.

        Args:
            n_components: Number of principal components to keep
            device: GPU device to use (auto-detected if None)
        """
        self.n_components = n_components
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.is_fitted = False

        # PCA components will be stored as tensors
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

        print(f"Initializing GPU PCA on device: {self.device}")

    def fit(self, features_dict, labels_dict=None):
        """
        Fit PCA on training data only using GPU acceleration.

        Args:
            features_dict: Dict of {dataset_name: features_array}
            labels_dict: Dict of {dataset_name: labels_array} (not used for PCA)
        """
        print(f"Fitting GPU PCA with {self.n_components} components on {self.device}...")

        # Combine all training features
        all_features = []
        total_samples = 0

        for dataset_name, features in features_dict.items():
            features_array = np.array(features)
            all_features.append(features_array)
            total_samples += len(features_array)
            print(f"  {dataset_name}: {len(features_array)} samples")

        # Combine all features and move to GPU
        combined_features = np.vstack(all_features)
        print(f"  Total training samples for PCA: {len(combined_features)}")
        print(f"  Feature shape: {combined_features.shape}")
        print(f"  Moving data to {self.device}...")

        # Convert to PyTorch tensor and move to GPU
        X = torch.tensor(combined_features, dtype=torch.float32, device=self.device)

        # Center the data
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_

        print(f"  Computing SVD on GPU...")
        # Compute SVD for PCA
        # For efficiency with large datasets, we can use the covariance method
        if X_centered.shape[0] > X_centered.shape[1]:
            # More samples than features: use covariance matrix approach
            cov_matrix = torch.mm(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

            # Sort eigenvalues and eigenvectors in descending order
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        else:
            # More features than samples: use SVD directly
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
            eigenvalues = (S ** 2) / (X_centered.shape[0] - 1)
            eigenvectors = Vt.T

        # Keep only the top n_components
        self.components_ = eigenvectors[:, :self.n_components].T  # Shape: (n_components, n_features)
        eigenvalues = eigenvalues[:self.n_components]

        # Compute explained variance ratio
        total_variance = torch.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / torch.sum(torch.tensor([torch.trace(cov_matrix) if 'cov_matrix' in locals() else torch.var(X_centered, dim=0).sum()]))

        self.is_fitted = True

        # Print explained variance ratio (move to CPU for printing)
        explained_var_cpu = self.explained_variance_ratio_.cpu().numpy()
        cumulative_variance = np.cumsum(explained_var_cpu)
        print(f"  Explained variance ratio (first 10 components): {explained_var_cpu[:10]}")
        print(f"  Cumulative explained variance: {cumulative_variance[-1]:.4f}")

        # Clean up GPU memory
        del X, X_centered
        if 'cov_matrix' in locals():
            del cov_matrix
        if 'eigenvalues' in locals():
            del eigenvalues, eigenvectors
        torch.cuda.empty_cache()

        return self

    def transform(self, features):
        """
        Apply PCA transformation to features using GPU.

        Args:
            features: numpy array of features to transform

        Returns:
            transformed_features: PCA-transformed features (numpy array)
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted! Call fit() first.")

        # Convert to tensor and move to GPU
        features_array = np.array(features)
        X = torch.tensor(features_array, dtype=torch.float32, device=self.device)

        # Center and transform
        X_centered = X - self.mean_
        X_transformed = torch.mm(X_centered, self.components_.T)

        # Move back to CPU and convert to numpy
        result = X_transformed.cpu().numpy()

        # Clean up GPU memory
        del X, X_centered, X_transformed
        torch.cuda.empty_cache()

        return result


def train_pca_projection(features_dict, labels_dict=None, n_components=256, device=None):
    """
    Train GPU PCA projection on training data only.

    Args:
        features_dict: Dict of {dataset_name: features_array}
        labels_dict: Dict of {dataset_name: labels_array} (not used for PCA)
        n_components: Number of PCA components
        device: GPU device to use (auto-detected if None)

    Returns:
        pca_model: Trained GPU PCA model
    """
    if device is None:
        device = GPU_DEVICE

    print(f"Training GPU PCA projection: {CONFIG.INPUT_DIM} -> {n_components} dimensions")
    print(f"Using device: {device}")

    # Create and fit GPU PCA model
    pca_model = GPUPCAProjection(n_components=n_components, device=device)
    pca_model.fit(features_dict, labels_dict)

    return pca_model


# This function is replaced by train_pca_projection above
# Keeping for compatibility but not used in PCA version


def apply_pca_projection(pca_model, features_dict):
    """
    Apply GPU PCA projection to transform features from 4096 to n_components dimensions

    Args:
        pca_model: Trained GPU PCA model
        features_dict: Dict of {dataset_name: features_array}

    Returns:
        projected_features_dict: Dict of {dataset_name: projected_features_array}
    """
    projected_features_dict = {}

    print(f"Applying GPU PCA projection to features (n_components={pca_model.n_components})...")

    for dataset_name, features in features_dict.items():
        features_array = np.array(features)
        print(f"  Projecting {dataset_name}: {features_array.shape} -> ", end="")

        # Apply GPU PCA transformation
        projected_features = pca_model.transform(features_array)
        projected_features_dict[dataset_name] = projected_features

        print(f"{projected_features.shape}")

    return projected_features_dict

def euclidean_distance_batch_gpu(X, Y, device=None):
    """GPU-accelerated batch Euclidean distance computation between X and Y"""
    if device is None:
        device = GPU_DEVICE

    try:
        # Convert inputs to GPU tensors (use float32 for memory efficiency)
        X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
        Y_gpu = torch.tensor(Y, dtype=torch.float32, device=device)

        # Ensure proper dimensions
        if X_gpu.dim() == 1:
            X_gpu = X_gpu.unsqueeze(0)  # Add batch dimension
        if Y_gpu.dim() == 1:
            Y_gpu = Y_gpu.unsqueeze(0)  # Add batch dimension

        # Compute pairwise Euclidean distances
        # X: (n_samples, n_features), Y: (n_training, n_features)
        # Result: (n_samples, n_training)
        distances = torch.cdist(X_gpu, Y_gpu, p=2)

        # Move back to CPU
        return distances.cpu().numpy()

    except Exception as e:
        print(f"    GPU Euclidean distance failed: {e}, using CPU fallback")
        # CPU fallback
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        distances = []
        for x in X:
            row_distances = []
            for y in Y:
                dist = np.linalg.norm(x - y)
                row_distances.append(dist)
            distances.append(row_distances)

        return np.array(distances)

class KCDDetector:
    """
    Contrastive K-Nearest Neighbors (KNN) based jailbreak detection.

    This implements an improved KNN algorithm that uses BOTH benign and malicious samples:
    1. Uses L2-normalized features (critical for performance)
    2. Computes contrastive score: distance_to_malicious - distance_to_benign
    3. Leverages both classes for better decision boundary learning
    4. Enhanced with GPU acceleration for distance computation
    """

    def __init__(self, k=50, use_gpu=True, normalization=True):
        """
        Initialize KCD detector.

        Args:
            k: Number of nearest neighbors (50 for small datasets like ours)
            use_gpu: Whether to use GPU acceleration for distance computation
            normalization: Whether to L2-normalize features (critical - always True)
        """
        self.k = k
        self.normalization = normalization
        self.threshold = 0.0
        self.benign_features = None
        self.malicious_features = None
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if self.use_gpu:
            print(f"    KCDDetector initialized with GPU acceleration on {GPU_DEVICE}")
        else:
            print("    KCDDetector initialized with CPU computation")

        print(f"    Hyperparameters: k={self.k}, normalization={self.normalization}")

    def _normalize_features(self, features):
        """L2 normalize features - critical for KNN performance"""
        if not self.normalization:
            return features

        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # L2 normalization
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return features / norms

    def _euclidean_distance_batch(self, X, Y):
        """Batch Euclidean distance computation with GPU acceleration"""
        if self.use_gpu:
            return euclidean_distance_batch_gpu(X, Y)
        else:
            # CPU batch computation using broadcasting
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if Y.ndim == 1:
                Y = Y.reshape(1, -1)

            # Compute pairwise distances using broadcasting
            # X: (n_samples, n_features), Y: (n_training, n_features)
            # Result: (n_samples, n_training)
            X_expanded = X[:, np.newaxis, :]  # (n_samples, 1, n_features)
            Y_expanded = Y[np.newaxis, :, :]  # (1, n_training, n_features)
            distances = np.sqrt(np.sum((X_expanded - Y_expanded) ** 2, axis=2))
            return distances

    def fit_training_data(self, benign_data, malicious_data):
        """
        Fit KCD detector with both benign and malicious training data.

        Args:
            benign_data: Dict of {dataset_name: list_of_features} for benign samples
            malicious_data: Dict of {dataset_name: list_of_features} for malicious samples
        """
        print("Fitting KCD detector with both benign and malicious training data...")

        # Process benign data
        all_benign_features = []
        benign_total = 0

        for dataset_name, features in benign_data.items():
            if len(features) > 0:
                all_benign_features.extend(features)
                benign_total += len(features)
                print(f"  Benign {dataset_name}: {len(features)} samples")
            else:
                print(f"  Warning: Benign {dataset_name} has no samples, skipping")

        # Process malicious data
        all_malicious_features = []
        malicious_total = 0

        for dataset_name, features in malicious_data.items():
            if len(features) > 0:
                all_malicious_features.extend(features)
                malicious_total += len(features)
                print(f"  Malicious {dataset_name}: {len(features)} samples")
            else:
                print(f"  Warning: Malicious {dataset_name} has no samples, skipping")

        if benign_total == 0:
            raise ValueError("No benign training data provided!")
        if malicious_total == 0:
            raise ValueError("No malicious training data provided!")

        # Convert to numpy arrays and normalize
        self.benign_features = np.array(all_benign_features)
        self.benign_features = self._normalize_features(self.benign_features)

        self.malicious_features = np.array(all_malicious_features)
        self.malicious_features = self._normalize_features(self.malicious_features)

        print(f"  Total benign training samples: {len(self.benign_features)}")
        print(f"  Total malicious training samples: {len(self.malicious_features)}")
        print(f"  Feature dimension: {self.benign_features.shape[1]}")

        # Adjust k based on smaller training set size (more conservative)
        # min_training_size = min(len(self.benign_features), len(self.malicious_features))
        # if min_training_size < 5000:
        #     self.k = min(50, min_training_size - 1)  # Small dataset
        # elif min_training_size < 20000:
        #     self.k = min(200, min_training_size - 1)  # Medium dataset
        # else:
        #     self.k = min(1000, min_training_size - 1)  # Large dataset

        print(f"  Finalized: k={self.k}")

    def compute_kcd_score(self, x):
        """
        Compute KCD outlier score for a sample.

        Score = distance_to_k_nearest_malicious - distance_to_k_nearest_benign

        Intuition:
        - If sample is benign: distance_to_benign is small, distance_to_malicious is large → negative score
        - If sample is malicious: distance_to_benign is large, distance_to_malicious is small → positive score

        Higher scores indicate more likely to be malicious (jailbreak).
        """
        if self.benign_features is None or self.malicious_features is None:
            raise ValueError("Detector not fitted! Call fit_training_data() first.")

        # Normalize test sample
        x_normalized = self._normalize_features(x.reshape(1, -1))

        # Compute distances to benign training samples
        benign_distances = self._euclidean_distance_batch(x_normalized, self.benign_features)
        benign_distances = benign_distances.flatten()
        benign_distances_sorted = np.sort(benign_distances)
        kth_benign_distance = benign_distances_sorted[min(self.k - 1, len(benign_distances_sorted) - 1)]

        # Compute distances to malicious training samples
        malicious_distances = self._euclidean_distance_batch(x_normalized, self.malicious_features)
        malicious_distances = malicious_distances.flatten()
        malicious_distances_sorted = np.sort(malicious_distances)
        kth_malicious_distance = malicious_distances_sorted[min(self.k - 1, len(malicious_distances_sorted) - 1)]

        # Contrastive score: closer to malicious samples = higher score
        contrastive_score = kth_benign_distance - kth_malicious_distance

        return contrastive_score

    def fit_threshold(self, validation_features, validation_labels):
        """Find optimal threshold using validation data"""
        # Compute KCD scores for validation data
        _, scores = self.predict(validation_features)
        validation_labels = np.array(validation_labels)

        # Check if we have both classes
        unique_labels = np.unique(validation_labels)
        if len(unique_labels) < 2:
            print(f"Warning: Only one class in validation data: {unique_labels}")
            self.threshold = 0.0
            return 0.0

        # Find threshold that maximizes balanced accuracy
        from sklearn.metrics import f1_score, balanced_accuracy_score

        # Analyze score distributions
        benign_scores = scores[validation_labels == 0]
        malicious_scores = scores[validation_labels == 1]

        benign_mean, benign_std = np.mean(benign_scores), np.std(benign_scores)
        malicious_mean, malicious_std = np.mean(malicious_scores), np.std(malicious_scores)

        # Calculate separation between distributions
        separation = abs(malicious_mean - benign_mean) / (benign_std + malicious_std + 1e-8)

        print(f"  Validation score separation: {separation:.2f} (higher is better)")
        print(f"  Benign scores: mean={benign_mean:.4f}, std={benign_std:.4f}")
        print(f"  Malicious scores: mean={malicious_mean:.4f}, std={malicious_std:.4f}")

        # Use adaptive threshold range based on score distributions
        if separation > 2.0:  # Well-separated distributions
            # Use narrow range around the midpoint
            midpoint = (benign_mean + malicious_mean) / 2
            range_width = min(benign_std, malicious_std) * 2
            score_range = [midpoint - range_width, midpoint + range_width]
            print(f"  Using narrow threshold range around midpoint: [{score_range[0]:.4f}, {score_range[1]:.4f}]")
        else:  # Overlapping distributions - use wider range
            score_range = np.percentile(scores, [5, 95])
            range_width = score_range[1] - score_range[0]
            score_range[0] -= 0.2 * range_width
            score_range[1] += 0.2 * range_width
            print(f"  Using wide threshold range for overlapping distributions: [{score_range[0]:.4f}, {score_range[1]:.4f}]")

        thresholds = np.linspace(score_range[0], score_range[1], 200)
        best_score = 0
        best_threshold = 0.0
        best_f1 = 0
        best_balanced_acc = 0

        for threshold in thresholds:
            y_pred = (scores > threshold).astype(int)
            try:
                balanced_acc = balanced_accuracy_score(validation_labels, y_pred)
                f1 = f1_score(validation_labels, y_pred, zero_division=0)
                combined_score = 0.8 * balanced_acc + 0.2 * f1

                if combined_score > best_score:
                    best_score = combined_score
                    best_threshold = threshold
                    best_f1 = f1
                    best_balanced_acc = balanced_acc
            except:
                continue

        print(f"  Optimal threshold: {best_threshold:.4f} (Balanced Acc: {best_balanced_acc:.4f}, F1: {best_f1:.4f})")

        self.threshold = best_threshold
        return best_threshold

    def predict(self, features):
        """Predict whether samples are OOD (jailbreak) using KCD"""
        if len(features) > 100:  # Use batch processing for larger datasets
            scores = self._compute_kcd_scores_batch(features)
        else:
            scores = np.array([self.compute_kcd_score(x) for x in features])
        predictions = (scores > self.threshold).astype(int)
        return predictions, scores

    def _compute_kcd_scores_batch(self, features):
        """Batch KCD score computation with GPU acceleration"""
        features = np.array(features)

        # Normalize all test features
        features_normalized = self._normalize_features(features)

        # Compute distances to benign training samples (batch)
        benign_distances = self._euclidean_distance_batch(features_normalized, self.benign_features)
        # benign_distances shape: (n_test_samples, n_benign_training_samples)

        # Compute distances to malicious training samples (batch)
        malicious_distances = self._euclidean_distance_batch(features_normalized, self.malicious_features)
        # malicious_distances shape: (n_test_samples, n_malicious_training_samples)

        # For each test sample, compute contrastive score
        scores = []
        for i in range(len(features_normalized)):
            # k-th nearest benign distance
            benign_sample_distances = benign_distances[i]
            benign_distances_sorted = np.sort(benign_sample_distances)
            kth_benign_distance = benign_distances_sorted[min(self.k - 1, len(benign_distances_sorted) - 1)]

            # k-th nearest malicious distance
            malicious_sample_distances = malicious_distances[i]
            malicious_distances_sorted = np.sort(malicious_sample_distances)
            kth_malicious_distance = malicious_distances_sorted[min(self.k - 1, len(malicious_distances_sorted) - 1)]

            # Contrastive score
            contrastive_score = kth_benign_distance - kth_malicious_distance
            scores.append(contrastive_score)

        return np.array(scores)
    
    def evaluate(self, test_features, test_labels):
        """Evaluate the detector on test data"""
        predictions, scores = self.predict(test_features)

        accuracy = accuracy_score(test_labels, predictions)

        # Calculate F1 score
        f1 = f1_score(test_labels, predictions, zero_division=0)

        # Calculate TPR, FPR from confusion matrix and AUROC/AUPRC
        unique_classes = np.unique(test_labels)
        if len(unique_classes) > 1:
            try:
                # Calculate TPR, FPR from confusion matrix
                tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity/Recall)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

                # Calculate AUROC and AUPRC
                fpr_curve, tpr_curve, _ = roc_curve(test_labels, scores)
                auroc = auc(fpr_curve, tpr_curve)
                precision, recall, _ = precision_recall_curve(test_labels, scores)
                auprc = auc(recall, precision)
            except:
                tpr = float('nan')
                fpr = float('nan')
                auroc = float('nan')
                auprc = float('nan')
        else:
            tpr = float('nan')
            fpr = float('nan')
            auroc = float('nan')
            auprc = float('nan')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
            'auroc': auroc,
            'auprc': auprc,
            'predictions': predictions,
            'scores': scores,
            'threshold': self.threshold
        }


def analyze_dataset_composition(dataset, dataset_name):
    """Analyze and report dataset composition"""
    total = len(dataset)
    benign = sum(1 for s in dataset if s['toxicity'] == 0)
    malicious = sum(1 for s in dataset if s['toxicity'] == 1)

    has_images = sum(1 for s in dataset if s.get('img') is not None)
    text_only = total - has_images

    print(f"{dataset_name}: {total} samples")
    print(f"  Classes: {benign} benign, {malicious} malicious")
    print(f"  Modality: {has_images} multimodal, {text_only} text-only")

    if benign == 0:
        print(f"  WARNING: {dataset_name} contains ONLY malicious samples!")
    elif malicious == 0:
        print(f"  WARNING: {dataset_name} contains ONLY benign samples!")

    return {'total': total, 'benign': benign, 'malicious': malicious, 'multimodal': has_images}


def prepare_knn_data_structure(datasets_dict, hidden_states_dict, labels_dict):
    benign_data = {}
    malicious_data = {}

    for dataset_name in datasets_dict.keys():
        if dataset_name not in hidden_states_dict:
            continue

        features = hidden_states_dict[dataset_name]
        labels = labels_dict[dataset_name]

        # Separate benign and malicious samples
        benign_features = [features[i] for i, label in enumerate(labels) if label == 0]
        malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]

        if benign_features:
            benign_data[f"{dataset_name}_benign"] = benign_features
        if malicious_features:
            malicious_data[f"{dataset_name}_malicious"] = malicious_features

    return benign_data, malicious_data


def signal_handler(sig, frame):
    print('\nInterrupted by user. Exiting...')
    sys.exit(0)

def prepare_balanced_training():
    """
    Load balanced training data according to the balanced dataset configuration:
    Training Set (2,000 examples, 1:1 ratio):
    - Benign (1,000): Alpaca (500) + MM-Vet (218) + OpenAssistant (282)
    - Malicious (1,000): AdvBench (300) + JailbreakV-28K (550) + DAN variants (150)
    """
    print("Loading balanced training data...")
    benign_training = {}
    malicious_training = {}

    # === BENIGN TRAINING DATA (1,000 total) ===
    print("Loading benign training data (1,000 samples)...")

    # 1. Alpaca - 500 samples
    try:
        alpaca_samples = load_alpaca(max_samples=500)
        if alpaca_samples:
            benign_training["Alpaca"] = alpaca_samples
            print(f"  Loaded {len(alpaca_samples)} Alpaca samples")
    except Exception as e:
        print(f"Could not load Alpaca: {e}")

    # 2. MM-Vet - 218 samples
    try:
        mmvet_samples = load_mm_vet()
        if mmvet_samples:
            # Limit to 218 samples and ensure they are benign
            mmvet_benign = [s for s in mmvet_samples if s.get('toxicity', 0) == 0][:218]
            if mmvet_benign:
                benign_training["MM-Vet"] = mmvet_benign
                print(f"  Loaded {len(mmvet_benign)} MM-Vet samples")
    except Exception as e:
        print(f"Could not load MM-Vet: {e}")

    # 3. OpenAssistant - 282 samples
    try:
        openassistant_samples = load_openassistant(max_samples=282)
        if openassistant_samples:
            benign_training["OpenAssistant"] = openassistant_samples
            print(f"  Loaded {len(openassistant_samples)} OpenAssistant samples")
    except Exception as e:
        print(f"Could not load OpenAssistant: {e}")

    # === MALICIOUS TRAINING DATA (1,000 total) ===
    print("Loading malicious training data (1,000 samples)...")

    # 1. AdvBench - 300 samples
    try:
        advbench_samples = load_advbench(max_samples=300)
        if advbench_samples:
            malicious_training["AdvBench"] = advbench_samples
            print(f"  Loaded {len(advbench_samples)} AdvBench samples")
    except Exception as e:
        print(f"Could not load AdvBench: {e}")

    # 2. JailbreakV-28K - 550 samples (using llm_transfer_attack and query_related for training)
    try:
        # Use separate calls for controlled distribution to ensure we know exactly how many
        # samples come from each attack type (275 from each attack type)
        print("  Loading JailbreakV-28K with controlled attack type distribution...")

        # Load llm_transfer_attack samples (275 samples)
        llm_attack_samples = load_JailBreakV_custom(
            attack_types=["llm_transfer_attack"],
            max_samples=275
        )

        # Load query_related samples (275 samples)
        query_related_samples = load_JailBreakV_custom(
            attack_types=["query_related"],
            max_samples=275
        )

        # Combine both attack types
        jbv_samples = []
        if llm_attack_samples:
            jbv_samples.extend(llm_attack_samples)
            print(f"    Loaded {len(llm_attack_samples)} llm_transfer_attack samples")
        if query_related_samples:
            jbv_samples.extend(query_related_samples)
            print(f"    Loaded {len(query_related_samples)} query_related samples")

        if jbv_samples:
            malicious_training["JailbreakV-28K"] = jbv_samples
            print(f"  Total JailbreakV-28K training samples: {len(jbv_samples)} ({len(llm_attack_samples)} llm_transfer + {len(query_related_samples)} query_related)")
        else:
            print("  Warning: No JailbreakV-28K samples loaded")

    except Exception as e:
        print(f"Could not load JailbreakV-28K: {e}")

    # 3. DAN Prompts - 150 samples
    try:
        dan_samples = load_dan_prompts(max_samples=150)
        if dan_samples:
            malicious_training["DAN"] = dan_samples
            print(f"  Loaded {len(dan_samples)} DAN samples")
    except Exception as e:
        print(f"Could not load DAN Prompts: {e}")

    return benign_training, malicious_training

def prepare_balanced_evaluation():
    """
    Prepare balanced evaluation datasets according to the balanced dataset configuration:
    Test Set (1,800 examples, 1:1 ratio):
    - Safe (900): XSTest safe (250) + FigTxt safe (300) + VQAv2 (350)
    - Unsafe (900): XSTest unsafe (200) + FigTxt unsafe (350) + VAE (200) + JailbreakV-28K (150)
    """
    print("Loading balanced evaluation datasets...")
    test_datasets = {}

    # === SAFE TEST DATA (900 total) ===
    print("Loading safe test data (900 samples)...")

    # 1. XSTest safe - 250 samples
    try:
        xstest_samples = load_XSTest()
        if xstest_samples:
            xstest_safe = [s for s in xstest_samples if s.get('toxicity', 0) == 0][:250]
            if xstest_safe:
                test_datasets["XSTest_safe"] = xstest_safe
                print(f"  Loaded {len(xstest_safe)} XSTest safe samples")
    except Exception as e:
        print(f"Could not load XSTest: {e}")

    # 2. XSTest unsafe - 200 samples
    try:
        if 'xstest_samples' in locals():
            xstest_unsafe = [s for s in xstest_samples if s.get('toxicity', 0) == 1][:200]
            if xstest_unsafe:
                test_datasets["XSTest_unsafe"] = xstest_unsafe
                print(f"  Loaded {len(xstest_unsafe)} XSTest unsafe samples")
    except Exception as e:
        print(f"Could not load XSTest unsafe: {e}")

    # 3. FigTxt safe - 300 samples
    try:
        figtxt_samples = load_FigTxt()
        if figtxt_samples:
            figtxt_safe = [s for s in figtxt_samples if s.get('toxicity', 0) == 0][:300]
            if figtxt_safe:
                test_datasets["FigTxt_safe"] = figtxt_safe
                print(f"  Loaded {len(figtxt_safe)} FigTxt safe samples")
    except Exception as e:
        print(f"Could not load FigTxt: {e}")

    # 4. FigTxt unsafe - 350 samples
    try:
        if 'figtxt_samples' in locals():
            figtxt_unsafe = [s for s in figtxt_samples if s.get('toxicity', 0) == 1][:350]
            if figtxt_unsafe:
                test_datasets["FigTxt_unsafe"] = figtxt_unsafe
                print(f"  Loaded {len(figtxt_unsafe)} FigTxt unsafe samples")
    except Exception as e:
        print(f"Could not load FigTxt unsafe: {e}")

    # 5. VQAv2 - 350 safe samples
    try:
        vqav2_samples = load_vqav2(max_samples=350)
        if vqav2_samples:
            test_datasets["VQAv2"] = vqav2_samples
            print(f"  Loaded {len(vqav2_samples)} VQAv2 samples")
    except Exception as e:
        print(f"Could not load VQAv2: {e}")

    # === UNSAFE TEST DATA (additional) ===
    print("Loading additional unsafe test data...")

    # 6. VAE (adversarial visual attacks) - 200 samples
    try:
        vae_samples = load_adversarial_img()
        if vae_samples:
            # Limit to 200 samples
            vae_limited = vae_samples[:200]
            test_datasets["VAE"] = vae_limited
            print(f"  Loaded {len(vae_limited)} VAE samples")
    except Exception as e:
        print(f"Could not load VAE: {e}")

    # 7. JailbreakV-28K test samples - 150 samples (figstep attack only)
    try:
        jbv_test_samples = load_JailBreakV_figstep(max_samples=150)
        if jbv_test_samples:
            test_datasets["JailbreakV-28K_test"] = jbv_test_samples
            print(f"  Loaded {len(jbv_test_samples)} JailbreakV-28K test samples (figstep attack)")
    except Exception as e:
        print(f"Could not load JailbreakV-28K test samples: {e}")

    return test_datasets

def main():
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    # Set random seed for reproducibility (use consistent seed)
    MAIN_SEED = 46  # Match the seed used elsewhere in the script
    random.seed(MAIN_SEED)
    np.random.seed(MAIN_SEED)
    torch.manual_seed(MAIN_SEED)
    torch.cuda.manual_seed(MAIN_SEED)
    torch.cuda.manual_seed_all(MAIN_SEED)

    # Ensure deterministic behavior for overall pipeline
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Additional determinism settings
    os.environ['PYTHONHASHSEED'] = str(MAIN_SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CuBLAS operations
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Random seeds set for reproducibility (seed={MAIN_SEED})")

    # Model selection
    REQUESTED_MODEL = detect_model_from_args()
    if REQUESTED_MODEL == 'internvl':
        model_path = "model/internvl3-8b"
        CONFIG.SINGLE_LAYER_TRAINING_LAYER = 20
    else:
        model_path = "model/llava-v1.6-vicuna-7b/" if REQUESTED_MODEL == 'llava' else "./model/qwen2.5-vl-7b-instruct"
        CONFIG.SINGLE_LAYER_TRAINING_LAYER = 16

    # Initialize feature extractor
    HiddenStateExtractor = get_extractor_class(REQUESTED_MODEL)
    extractor = HiddenStateExtractor(model_path)

    print("="*80)
    print(f"BALANCED OOD-BASED JAILBREAK DETECTION USING KCD WITH GPU PCA DIMENSIONALITY REDUCTION ({REQUESTED_MODEL.upper()})")
    print("="*80)
    print("Approach: Use BOTH benign and malicious prompts for training")
    print("          Compute contrastive score: distance_to_malicious - distance_to_benign")
    print("          Apply L2 normalization and Euclidean distance")
    print("          Enhanced with GPU-accelerated PCA dimensionality reduction: 4096 -> various dimensions per layer")
    print("          PCA trained only on training data using PyTorch GPU acceleration")
    print("          Enhanced with GPU acceleration for both PCA and distance computation")
    print("="*80)
    print("Training Set (2,000 examples, 1:1 ratio):")
    print("  - Benign (1,000): Alpaca (500) + MM-Vet (218) + OpenAssistant (282)")
    print("  - Malicious (1,000): AdvBench (300) + JailbreakV-28K (550) + DAN variants (150)")
    print("Test Set (1,800 examples, 1:1 ratio):")
    print("  - Safe (900): XSTest safe (250) + FigTxt safe (300) + VQAv2 (350)")
    print("  - Unsafe (900): XSTest unsafe (200) + FigTxt unsafe (350) + VAE (200) + JailbreakV-28K (150)")
    print("="*80)

    # Load balanced training data
    in_dist_datasets, ood_datasets = prepare_balanced_training()

    # Analyze training data composition
    print("\n--- Training Data Analysis ---")
    total_benign = 0
    total_malicious = 0

    print("In-Distribution (Benign) Training Data:")
    for dataset_name, samples in in_dist_datasets.items():
        analyze_dataset_composition(samples, dataset_name)
        total_benign += len(samples)

    print("\nOOD (Malicious) Training Data:")
    for dataset_name, samples in ood_datasets.items():
        analyze_dataset_composition(samples, dataset_name)
        total_malicious += len(samples)

    print(f"\nTraining Data Summary:")
    print(f"  Total Benign (In-Distribution): {total_benign:,} samples")
    print(f"  Total Malicious (OOD): {total_malicious:,} samples")
    print(f"  Ratio (Benign:Malicious): {total_benign/max(total_malicious,1):.1f}:1")
    print("-"*80)

    # Load balanced evaluation data
    test_datasets = prepare_balanced_evaluation()

    print("\n--- Evaluation Data Analysis ---")
    for dataset_name, samples in test_datasets.items():
        analyze_dataset_composition(samples, dataset_name)

    # Extract hidden states for all datasets (with caching)
    print("\n--- Extracting Hidden States ---")
    all_datasets = {**in_dist_datasets, **ood_datasets, **test_datasets}
    all_hidden_states = {}
    all_labels = {}

    for dataset_name, samples in all_datasets.items():
        print(f"Extracting features for {dataset_name} ({len(samples)} samples)...")

        # Use smaller batch sizes for large datasets to manage memory
        batch_size = 25 if len(samples) > 5000 else 50
        memory_cleanup_freq = 5 if len(samples) > 5000 else 10

        hidden_states, labels, _ = extractor.extract_hidden_states(
            samples, f"{dataset_name}", layer_start=0, layer_end=31, use_cache=True,
            batch_size=batch_size, memory_cleanup_freq=memory_cleanup_freq,
            experiment_name="balanced_ml_detection"
        )
        all_hidden_states[dataset_name] = hidden_states
        all_labels[dataset_name] = labels

    layers = list(range(0, 32))  # layers 0-31
    layer_results = {}

    # === TRAIN PCA PROJECTIONS ===
    CONFIG.print_config()

    # Store PCA models for different component numbers
    pca_models = {}

    # Only use training datasets (in_dist_datasets and ood_datasets)
    training_dataset_names = set(in_dist_datasets.keys()) | set(ood_datasets.keys())

    print(f"Training datasets: {list(training_dataset_names)}")
    print("Test datasets will NOT be used for PCA training")

    # Results storage for different PCA component numbers
    all_results = {}

    for n_components in CONFIG.PCA_COMPONENTS_LIST:
        print(f"\n{'='*80}")
        print(f"EXPERIMENTING WITH {n_components} PCA COMPONENTS")
        print(f"{'='*80}")

        # Store PCA models for each layer with this component number
        layer_pca_models = {}

        if CONFIG.PCA_MODE == "single_layer":
            print(f"\n=== Training Single PCA (Layer {CONFIG.SINGLE_LAYER_TRAINING_LAYER}, {n_components} components) ===")
            print(f"Will use this PCA for all layers")

            # Prepare training data for the single training layer
            projection_features_dict = {}
            projection_labels_dict = {}

            for dataset_name in training_dataset_names:
                if dataset_name in all_hidden_states:
                    projection_features_dict[dataset_name] = all_hidden_states[dataset_name][CONFIG.SINGLE_LAYER_TRAINING_LAYER]
                    projection_labels_dict[dataset_name] = all_labels[dataset_name]

            # Train the single PCA model
            single_pca_model = train_pca_projection(
                projection_features_dict,
                projection_labels_dict,
                n_components=n_components,
                device=GPU_DEVICE
            )

            # Use the same model for all layers
            for layer_idx in layers:
                layer_pca_models[layer_idx] = single_pca_model

            print(f"Single PCA training completed! Using for all {len(layers)} layers.")

        elif CONFIG.PCA_MODE == "layer_specific":
            print(f"\n=== Training Layer-Specific PCA ({n_components} components) ===")
            print("Training separate PCA models for each layer (0-31)")

            for layer_idx in layers:
                print(f"\n--- Training PCA for Layer {layer_idx} ---")

                # Prepare training data for this layer's PCA
                projection_features_dict = {}
                projection_labels_dict = {}

                for dataset_name in training_dataset_names:
                    if dataset_name in all_hidden_states:
                        projection_features_dict[dataset_name] = all_hidden_states[dataset_name][layer_idx]
                        projection_labels_dict[dataset_name] = all_labels[dataset_name]

                # Train the PCA model for this layer (GPU PCA doesn't need layer-specific seeds)
                layer_pca_model = train_pca_projection(
                    projection_features_dict,
                    projection_labels_dict,
                    n_components=n_components,
                    device=GPU_DEVICE
                )

                # Store the trained model
                layer_pca_models[layer_idx] = layer_pca_model

                print(f"Layer {layer_idx} PCA training completed!")

            print(f"\nAll {len(layer_pca_models)} layer-specific PCA models trained successfully!")

        else:
            raise ValueError(f"Unknown PCA mode: {CONFIG.PCA_MODE}. Use 'single_layer' or 'layer_specific'")

        # Store PCA models for this component number
        pca_models[n_components] = layer_pca_models

        # Now evaluate each layer with this PCA component number
        layer_results = {}

        for layer_idx in layers:
            print(f"\n=== Evaluating Layer {layer_idx} with {n_components} PCA Components ===")

            # Prepare data for this layer
            layer_hidden_states = {}
            layer_labels = {}

            for dataset_name in all_datasets.keys():
                if dataset_name in all_hidden_states:
                    layer_hidden_states[dataset_name] = all_hidden_states[dataset_name][layer_idx]
                    layer_labels[dataset_name] = all_labels[dataset_name]

            # Apply layer-specific PCA projection to transform features from 4096 to n_components dimensions
            # Note: PCA was trained ONLY on training data, now applied to all data
            print(f"  Applying layer-specific PCA projection to layer {layer_idx} features...")
            projected_layer_hidden_states = apply_pca_projection(
                layer_pca_models[layer_idx],
                layer_hidden_states
            )

            # Use projected features instead of original features
            layer_hidden_states = projected_layer_hidden_states

            # Prepare benign training data for KCD
            benign_data, _ = prepare_knn_data_structure(
                in_dist_datasets,
                {k: v for k, v in layer_hidden_states.items() if k in in_dist_datasets},
                {k: v for k, v in layer_labels.items() if k in in_dist_datasets}
            )

            # Initialize and train KCD detector with GPU acceleration
            # Determine k based on training set size
            total_benign_samples = sum(len(features) for features in benign_data.values())
            if total_benign_samples < 5000:
                k = 50  # Small dataset (our case)
            elif total_benign_samples < 20000:
                k = 200  # Medium dataset
            else:
                k = 1000  # Large dataset

            detector = KCDDetector(k=k, use_gpu=True, normalization=True)

            # Prepare malicious training data from OOD datasets
            malicious_data = {}
            for dataset_name in ood_datasets.keys():
                if dataset_name in layer_hidden_states:
                    features = layer_hidden_states[dataset_name]
                    labels = layer_labels[dataset_name]

                    # Get malicious samples (should be most/all samples in OOD datasets)
                    malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
                    if malicious_features:
                        malicious_data[f"{dataset_name}_malicious"] = malicious_features

            # Fit KCD detector with both benign and malicious training data
            print(f"  {get_gpu_memory_info()}")
            try:
                detector.fit_training_data(benign_data, malicious_data)
                cleanup_gpu_memory()
            except ValueError as e:
                print(f"  Skipping layer {layer_idx} - {e}")
                continue

            # Debug: Check if detector was fitted
            if detector.benign_features is None or detector.malicious_features is None:
                print(f"  Skipping layer {layer_idx} - detector not fitted properly")
                continue

            print(f"  KCD detector fitted with {len(detector.benign_features)} benign + {len(detector.malicious_features)} malicious samples, k={detector.k}")

            # Create validation set from training data for threshold optimization
            val_features = []
            val_labels = []

            # Sample from benign training data
            for _, features in benign_data.items():
                sample_size = min(100, len(features))
                if sample_size > 0:
                    sampled_features = random.sample(features, sample_size)
                    val_features.extend(sampled_features)
                    val_labels.extend([0] * len(sampled_features))

            # Sample from malicious training data (from OOD datasets)
            for dataset_name in ood_datasets.keys():
                if dataset_name in layer_hidden_states:
                    features = layer_hidden_states[dataset_name]
                    labels = layer_labels[dataset_name]

                    # Get malicious samples
                    malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
                    if malicious_features:
                        sample_size = min(100, len(malicious_features))
                        if sample_size > 0:
                            sampled_features = random.sample(malicious_features, sample_size)
                            val_features.extend(sampled_features)
                            val_labels.extend([1] * len(sampled_features))

            # Fit threshold with balanced validation set
            if val_features and len(set(val_labels)) > 1:
                print(f"  Validation set: {len(val_features)} samples ({val_labels.count(0)} benign, {val_labels.count(1)} malicious)")

                # Debug: Analyze score distributions before threshold fitting (use batch computation for speed)
                print("  Computing validation scores...")
                _, val_scores = detector.predict(val_features)  # Use batch computation
                benign_scores = [val_scores[i] for i, label in enumerate(val_labels) if label == 0]
                malicious_scores = [val_scores[i] for i, label in enumerate(val_labels) if label == 1]

                print(f"  Validation score distributions:")
                print(f"    Benign: mean={np.mean(benign_scores):.2f}, std={np.std(benign_scores):.2f}, range=[{np.min(benign_scores):.2f}, {np.max(benign_scores):.2f}]")
                print(f"    Malicious: mean={np.mean(malicious_scores):.2f}, std={np.std(malicious_scores):.2f}, range=[{np.min(malicious_scores):.2f}, {np.max(malicious_scores):.2f}]")

                detector.fit_threshold(val_features, val_labels)
            else:
                print(f"  Insufficient validation data, using default threshold")
                detector.threshold = 0.0

            # === COMBINED EVALUATION (Real-world scenario) ===
            print("  Computing combined evaluation (real-world scenario)...")

            # Combine all test features and labels for unified evaluation
            combined_test_features = []
            combined_test_labels = []
            dataset_boundaries = {}  # Track which samples belong to which dataset
            current_idx = 0

            for test_dataset_name in test_datasets.keys():
                if test_dataset_name in layer_hidden_states:
                    test_features = layer_hidden_states[test_dataset_name]
                    test_labels_data = layer_labels[test_dataset_name]

                    dataset_boundaries[test_dataset_name] = (current_idx, current_idx + len(test_features))
                    combined_test_features.extend(test_features)
                    combined_test_labels.extend(test_labels_data)
                    current_idx += len(test_features)

            if combined_test_features:
                # Evaluate on combined test set (this simulates real-world usage)
                print(f"  Combined test set: {len(combined_test_features)} samples")
                combined_eval_results = detector.evaluate(combined_test_features, combined_test_labels)

                # Compute combined test score distributions for analysis
                combined_scores = combined_eval_results['scores']
                combined_benign_scores = [combined_scores[i] for i, label in enumerate(combined_test_labels) if label == 0]
                combined_malicious_scores = [combined_scores[i] for i, label in enumerate(combined_test_labels) if label == 1]

                print(f"  Combined test score distributions:")
                if combined_benign_scores:
                    print(f"    Benign: mean={np.mean(combined_benign_scores):.2f}, std={np.std(combined_benign_scores):.2f}, range=[{np.min(combined_benign_scores):.2f}, {np.max(combined_benign_scores):.2f}]")
                if combined_malicious_scores:
                    print(f"    Malicious: mean={np.mean(combined_malicious_scores):.2f}, std={np.std(combined_malicious_scores):.2f}, range=[{np.min(combined_malicious_scores):.2f}, {np.max(combined_malicious_scores):.2f}]")

                # Handle NaN values for display
                f1_str = f"{combined_eval_results['f1']:.4f}"
                tpr_str = "N/A" if np.isnan(combined_eval_results['tpr']) else f"{combined_eval_results['tpr']:.4f}"
                fpr_str = "N/A" if np.isnan(combined_eval_results['fpr']) else f"{combined_eval_results['fpr']:.4f}"
                auroc_str = "N/A" if np.isnan(combined_eval_results['auroc']) else f"{combined_eval_results['auroc']:.4f}"
                auprc_str = "N/A" if np.isnan(combined_eval_results['auprc']) else f"{combined_eval_results['auprc']:.4f}"

                print(f"  COMBINED RESULTS   : Acc={combined_eval_results['accuracy']:.4f}, F1={f1_str}, TPR={tpr_str}, FPR={fpr_str}, "
                      f"AUROC={auroc_str}, AUPRC={auprc_str}, Thresh={combined_eval_results['threshold']:.4f}")

                # Store combined results
                layer_performance = {'COMBINED': combined_eval_results}

                # === INDIVIDUAL DATASET ANALYSIS (for debugging) ===
                print("  Individual dataset analysis:")
                for test_dataset_name, (start_idx, end_idx) in dataset_boundaries.items():
                    dataset_scores = combined_scores[start_idx:end_idx]
                    dataset_labels = combined_test_labels[start_idx:end_idx]
                    dataset_predictions = combined_eval_results['predictions'][start_idx:end_idx]

                    # Calculate individual dataset metrics using the same threshold
                    dataset_accuracy = accuracy_score(dataset_labels, dataset_predictions)

                    # Separate scores by label for analysis
                    dataset_benign_scores = [dataset_scores[i] for i, label in enumerate(dataset_labels) if label == 0]
                    dataset_malicious_scores = [dataset_scores[i] for i, label in enumerate(dataset_labels) if label == 1]

                    if dataset_benign_scores:
                        benign_stats = f"mean={np.mean(dataset_benign_scores):.2f}, range=[{np.min(dataset_benign_scores):.2f}, {np.max(dataset_benign_scores):.2f}]"
                    else:
                        benign_stats = "no benign samples"

                    if dataset_malicious_scores:
                        malicious_stats = f"mean={np.mean(dataset_malicious_scores):.2f}, range=[{np.min(dataset_malicious_scores):.2f}, {np.max(dataset_malicious_scores):.2f}]"
                    else:
                        malicious_stats = "no malicious samples"

                    print(f"    {test_dataset_name:15s}: Acc={dataset_accuracy:.4f}")
                    print(f"      Benign: {benign_stats}")
                    print(f"      Malicious: {malicious_stats}")

                    # Calculate F1, TPR, FPR for individual dataset
                    f1 = f1_score(dataset_labels, dataset_predictions, zero_division=0)

                    # Calculate TPR, FPR from confusion matrix
                    unique_classes = np.unique(dataset_labels)
                    if len(unique_classes) > 1:
                        try:
                            tn, fp, fn, tp = confusion_matrix(dataset_labels, dataset_predictions).ravel()
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
                        except:
                            tpr = float('nan')
                            fpr = float('nan')
                    else:
                        tpr = float('nan')
                        fpr = float('nan')

                    # Store individual results for detailed analysis
                    layer_performance[test_dataset_name] = {
                        'accuracy': dataset_accuracy,
                        'f1': f1,
                        'tpr': tpr,
                        'fpr': fpr,
                        'auroc': float('nan'),  # Not meaningful for individual datasets with combined threshold
                        'auprc': float('nan'),  # Not meaningful for individual datasets with combined threshold
                        'predictions': dataset_predictions,
                        'scores': dataset_scores,
                        'threshold': combined_eval_results['threshold']
                    }
            else:
                print("  No test data available for evaluation")
                layer_performance = {}

            layer_results[layer_idx] = layer_performance

        # Store results for this PCA component number
        all_results[n_components] = layer_results

        # Calculate layer ranking based on COMBINED results (real-world performance) for this PCA component
        layer_combined_scores = []
        for layer_idx in layers:
            if layer_results[layer_idx] and 'COMBINED' in layer_results[layer_idx]:
                combined_result = layer_results[layer_idx]['COMBINED']
                accuracy = combined_result['accuracy']
                auroc = combined_result['auroc'] if not np.isnan(combined_result['auroc']) else 0.0
                auprc = combined_result['auprc'] if not np.isnan(combined_result['auprc']) else 0.0

                # Combined score prioritizing accuracy and AUROC for real-world performance
                combined_score = 0.5 * accuracy + 0.3 * auroc + 0.2 * auprc
                layer_combined_scores.append((layer_idx, accuracy, auroc, auprc, combined_score))
            else:
                layer_combined_scores.append((layer_idx, 0.0, 0.0, 0.0, 0.0))

        # Sort by combined score (real-world performance)
        layer_combined_scores.sort(key=lambda x: x[4], reverse=True)

        # Also calculate individual dataset averages for comparison
        layer_individual_avg_scores = []
        for layer_idx in layers:
            if layer_results[layer_idx]:
                # Exclude COMBINED from individual averages
                individual_results = {k: v for k, v in layer_results[layer_idx].items() if k != 'COMBINED'}
                if individual_results:
                    accuracies = [result['accuracy'] for result in individual_results.values()
                                 if not np.isnan(result['accuracy'])]
                    avg_accuracy = np.mean(accuracies) if accuracies else 0.0
                    layer_individual_avg_scores.append((layer_idx, avg_accuracy))
                else:
                    layer_individual_avg_scores.append((layer_idx, 0.0))
            else:
                layer_individual_avg_scores.append((layer_idx, 0.0))

        layer_individual_avg_scores.sort(key=lambda x: x[1], reverse=True)

        # Print results for this PCA component number
        print(f"\n{'='*120}")
        print(f"RESULTS SUMMARY FOR {n_components} PCA COMPONENTS")
        print(f"{'='*120}")

        # Find best performing layer for this PCA component number
        if layer_combined_scores:
            best_layer_idx, best_accuracy, best_auroc, best_auprc, best_combined_score = layer_combined_scores[0]
            print(f"Best performing layer: {best_layer_idx}")
            print(f"  Combined score: {best_combined_score:.4f}")
            print(f"  Accuracy: {best_accuracy:.4f}, AUROC: {best_auroc:.4f}, AUPRC: {best_auprc:.4f}")

        # Store summary for this PCA component
        all_results[n_components]['summary'] = {
            'best_layer': best_layer_idx if layer_combined_scores else None,
            'best_combined_score': best_combined_score if layer_combined_scores else 0.0,
            'layer_combined_scores': layer_combined_scores,
            'layer_individual_avg_scores': layer_individual_avg_scores
        }

    # Save results to CSV for all PCA experiments
    output_path = "results/balanced_kcd_pca_ablation_results.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["PCA_Components", "Layer", "Dataset", "Method", "Accuracy", "F1", "TPR", "FPR", "AUROC", "AUPRC", "Threshold", "Combined_Rank", "Individual_Rank"])

        for n_components in CONFIG.PCA_COMPONENTS_LIST:
            if n_components in all_results:
                layer_results = all_results[n_components]
                summary = layer_results.get('summary', {})
                layer_combined_scores = summary.get('layer_combined_scores', [])
                layer_individual_avg_scores = summary.get('layer_individual_avg_scores', [])

                # Create ranking based on combined performance
                layer_combined_ranking = {layer_idx: rank for rank, (layer_idx, _, _, _, _) in enumerate(layer_combined_scores, 1)}

                # Create ranking based on individual dataset average
                layer_individual_ranking = {layer_idx: rank for rank, (layer_idx, _) in enumerate(layer_individual_avg_scores, 1)}

                for layer_idx in layers:
                    if layer_idx in layer_results and layer_results[layer_idx]:
                        for dataset_name, result in layer_results[layer_idx].items():
                            if dataset_name != 'summary':  # Skip summary entry
                                # Handle NaN values for CSV
                                f1_val = f"{result['f1']:.4f}"
                                tpr_val = "N/A" if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
                                fpr_val = "N/A" if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"
                                auroc_val = "N/A" if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
                                auprc_val = "N/A" if np.isnan(result['auprc']) else f"{result['auprc']:.4f}"
                                threshold_val = f"{result['threshold']:.4f}"
                                combined_rank = layer_combined_ranking.get(layer_idx, "N/A")
                                individual_rank = layer_individual_ranking.get(layer_idx, "N/A")

                                writer.writerow([
                                    n_components,
                                    layer_idx,
                                    dataset_name,
                                    "KCD",
                                    f"{result['accuracy']:.4f}",
                                    f1_val,
                                    tpr_val,
                                    fpr_val,
                                    auroc_val,
                                    auprc_val,
                                    threshold_val,
                                    combined_rank,
                                    individual_rank
                                ])

    print(f"\nResults saved to {output_path}")

    # Print summary results for all PCA experiments
    print("\n" + "="*120)
    print("BALANCED OOD JAILBREAK DETECTION SUMMARY (KCD Algorithm with PCA Ablation)")
    print("="*120)
    print("Training Configuration (2,000 samples, 1:1 ratio):")
    print(f"  In-Distribution Datasets: {list(in_dist_datasets.keys())}")
    print(f"  OOD Datasets: {list(ood_datasets.keys())}")
    print("Test Configuration (1,800 samples, 1:1 ratio):")
    print(f"  Test Datasets: {list(test_datasets.keys())}")
    print(f"PCA Components Tested: {CONFIG.PCA_COMPONENTS_LIST}")
    print("-"*120)

    # Find overall best performing configuration
    best_overall_score = 0.0
    best_overall_config = None

    for n_components in CONFIG.PCA_COMPONENTS_LIST:
        if n_components in all_results:
            summary = all_results[n_components].get('summary', {})
            best_score = summary.get('best_combined_score', 0.0)
            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_config = (n_components, summary.get('best_layer'))

    if best_overall_config:
        print(f"\nOVERALL BEST CONFIGURATION:")
        print(f"  PCA Components: {best_overall_config[0]}")
        print(f"  Best Layer: {best_overall_config[1]}")
        print(f"  Combined Score: {best_overall_score:.4f}")
    print("-"*120)

    # COMBINED PERFORMANCE RANKING for each PCA component (Real-world scenario)
    for n_components in CONFIG.PCA_COMPONENTS_LIST:
        if n_components in all_results:
            print(f"\n{'COMBINED PERFORMANCE RANKING FOR ' + str(n_components) + ' PCA COMPONENTS (Real-world scenario)':<120}")
            print(f"{'Layer':<6} {'Accuracy':<10} {'F1':<8} {'TPR':<8} {'FPR':<8} {'AUROC':<10} {'AUPRC':<10} {'Combined':<10}")
            print("-" * 120)

            layer_results = all_results[n_components]
            summary = layer_results.get('summary', {})
            layer_combined_scores = summary.get('layer_combined_scores', [])

            for layer_idx, accuracy, auroc, auprc, combined_score in layer_combined_scores:
                if layer_idx in layer_results and layer_results[layer_idx] and 'COMBINED' in layer_results[layer_idx]:
                    combined_result = layer_results[layer_idx]['COMBINED']
                    acc_str = f"{accuracy:.3f}"
                    f1_str = f"{combined_result['f1']:.3f}"
                    tpr_str = "N/A" if np.isnan(combined_result['tpr']) else f"{combined_result['tpr']:.3f}"
                    fpr_str = "N/A" if np.isnan(combined_result['fpr']) else f"{combined_result['fpr']:.3f}"
                    auroc_str = "N/A" if auroc == 0.0 else f"{auroc:.3f}"
                    auprc_str = "N/A" if auprc == 0.0 else f"{auprc:.3f}"
                    combined_str = f"{combined_score:.3f}"

                    print(f"{layer_idx:<6} {acc_str:<10} {f1_str:<8} {tpr_str:<8} {fpr_str:<8} {auroc_str:<10} {auprc_str:<10} {combined_str:<10}")
                else:
                    print(f"{layer_idx:<6} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'0.000':<10}")

    # Show summary comparison across PCA components
    print(f"\n{'PCA COMPONENT COMPARISON (Best Layer Performance)':<120}")
    print(f"{'PCA_Comp':<10} {'Best_Layer':<12} {'Accuracy':<10} {'F1':<8} {'AUROC':<10} {'AUPRC':<10} {'Combined':<10}")
    print("-" * 120)

    for n_components in CONFIG.PCA_COMPONENTS_LIST:
        if n_components in all_results:
            summary = all_results[n_components].get('summary', {})
            best_layer = summary.get('best_layer', 'N/A')
            best_score = summary.get('best_combined_score', 0.0)

            if best_layer != 'N/A' and best_layer in all_results[n_components]:
                layer_results = all_results[n_components]
                if 'COMBINED' in layer_results[best_layer]:
                    combined_result = layer_results[best_layer]['COMBINED']
                    acc_str = f"{combined_result['accuracy']:.3f}"
                    f1_str = f"{combined_result['f1']:.3f}"
                    auroc_str = "N/A" if np.isnan(combined_result['auroc']) else f"{combined_result['auroc']:.3f}"
                    auprc_str = "N/A" if np.isnan(combined_result['auprc']) else f"{combined_result['auprc']:.3f}"
                    combined_str = f"{best_score:.3f}"

                    print(f"{n_components:<10} {best_layer:<12} {acc_str:<10} {f1_str:<8} {auroc_str:<10} {auprc_str:<10} {combined_str:<10}")
                else:
                    print(f"{n_components:<10} {best_layer:<12} {'N/A':<10} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'0.000':<10}")
            else:
                print(f"{n_components:<10} {'N/A':<12} {'N/A':<10} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'0.000':<10}")

    print("\n" + "="*120)
    print("EXPERIMENT SUMMARY")
    print("="*120)
    print("- PCA Dimensionality Reduction Ablation Study")
    print(f"- PCA Components Tested: {CONFIG.PCA_COMPONENTS_LIST}")
    print("- Balanced 1:1 training and test ratios for robust evaluation")
    print("- Training: Alpaca (500) + MM-Vet (218) + OpenAssistant (282) vs AdvBench (300) + JailbreakV-28K (550) + DAN (150)")
    print("- Testing: XSTest + FigTxt + VQAv2 (safe) vs XSTest + FigTxt + VAE + JailbreakV-28K (unsafe)")
    print("- PCA trained only on training data, applied to test data")

    if best_overall_config:
        print(f"\nKEY FINDINGS:")
        print(f"- Best overall configuration: {best_overall_config[0]} PCA components, Layer {best_overall_config[1]}")
        print(f"- Best combined score: {best_overall_score:.4f}")
        print("- Detailed results saved to CSV for further analysis")


if __name__ == "__main__":
    main()
