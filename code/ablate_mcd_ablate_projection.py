import csv
import numpy as np
import random
import warnings
import signal
import sys
import os
from scipy.linalg import inv
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
class ProjectionConfig:
    """Global configuration for projection training methods"""

    # Projection training mode
    # "single_layer": Train one projection on layer 16, use for all layers (original method)
    # "layer_specific": Train separate projection for each layer (new method)
    PROJECTION_MODE = "layer_specific"  # Change this to switch modes

    # Single layer mode settings
    SINGLE_LAYER_TRAINING_LAYER = 18  # Will be overridden (20 for InternVL, 16 for LLaVA)

    # Training hyperparameters
    PROJECTION_EPOCHS = 200
    PROJECTION_BATCH_SIZE = 64
    PROJECTION_LEARNING_RATE = 1e-3
    PROJECTION_LR_SCHEDULER = 'cosine'
    PROJECTION_MAX_PATIENCE = 15

    # Architecture settings
    INPUT_DIM = 4096
    OUTPUT_DIM = 256
    HIDDEN_DIM = 512
    DROPOUT = 0.3

    # Loss weighting settings
    DATASET_LOSS_WEIGHT = 1.0      # Weight for dataset classification loss
    TOXICITY_LOSS_WEIGHT = 5.0     # Weight for toxicity detection loss (higher to balance magnitude)

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*80)
        print("PROJECTION CONFIGURATION")
        print("="*80)
        print(f"Mode: {cls.PROJECTION_MODE}")
        if cls.PROJECTION_MODE == "single_layer":
            print(f"Training layer: {cls.SINGLE_LAYER_TRAINING_LAYER}")
        print(f"Architecture: {cls.INPUT_DIM} -> {cls.HIDDEN_DIM} -> {cls.OUTPUT_DIM}")
        print(f"Training: {cls.PROJECTION_EPOCHS} epochs, batch_size={cls.PROJECTION_BATCH_SIZE}")
        print(f"Learning rate: {cls.PROJECTION_LEARNING_RATE} ({cls.PROJECTION_LR_SCHEDULER} scheduler)")
        print(f"Loss weights: Dataset={cls.DATASET_LOSS_WEIGHT}, Toxicity={cls.TOXICITY_LOSS_WEIGHT}")
        print("="*80)

# Global config instance
CONFIG = ProjectionConfig()

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

# Initialize GPU
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


class LearnedProjection(nn.Module):
    """
    Learned projection from 4096 to 256 dimensions with multi-objective loss:
    1. Samples from same dataset should be close
    2. Samples from different datasets should be far
    3. Benign centroids should be close, malicious centroids should be close,
       but benign and malicious centroids should be far apart
    """

    def __init__(self, input_dim=None, output_dim=None, hidden_dim=None, dropout=None):
        super(LearnedProjection, self).__init__()

        # Use config defaults if not specified
        input_dim = input_dim or CONFIG.INPUT_DIM
        output_dim = output_dim or CONFIG.OUTPUT_DIM
        hidden_dim = hidden_dim or CONFIG.HIDDEN_DIM
        dropout = dropout or CONFIG.DROPOUT

        # Multi-layer projection network
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through projection network"""
        return self.projection(x)


def compute_contrastive_loss(embeddings, dataset_labels, toxicity_labels,
                           margin_dataset=1.0, margin_toxicity=2.0,
                           temperature=0.1):
    """
    Compute multi-objective contrastive loss:
    1. Dataset clustering: same dataset samples close, different dataset samples far
    2. Toxicity separation: benign centroids close, malicious centroids close,
       but benign and malicious far apart
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    # Normalize embeddings for better training stability
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # 1. Dataset clustering loss
    dataset_loss = 0.0
    unique_datasets = torch.unique(dataset_labels)

    if len(unique_datasets) > 1:
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for same/different datasets
        dataset_same_mask = (dataset_labels.unsqueeze(0) == dataset_labels.unsqueeze(1)).float()
        dataset_diff_mask = 1.0 - dataset_same_mask

        # Remove diagonal (self-comparisons)
        eye_mask = 1.0 - torch.eye(batch_size, device=device)
        dataset_same_mask *= eye_mask
        dataset_diff_mask *= eye_mask

        # Dataset clustering loss: minimize intra-dataset distances, maximize inter-dataset distances
        if dataset_same_mask.sum() > 0:
            intra_dataset_loss = (distances * dataset_same_mask).sum() / dataset_same_mask.sum()
        else:
            intra_dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if dataset_diff_mask.sum() > 0:
            inter_dataset_loss = torch.clamp(margin_dataset - distances, min=0) * dataset_diff_mask
            inter_dataset_loss = inter_dataset_loss.sum() / dataset_diff_mask.sum()
        else:
            inter_dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)

        dataset_loss = intra_dataset_loss + inter_dataset_loss
    else:
        dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # 2. Toxicity separation loss
    toxicity_loss = torch.tensor(0.0, device=device, requires_grad=True)
    unique_toxicity = torch.unique(toxicity_labels)

    if len(unique_toxicity) > 1:
        # Compute centroids for each toxicity class
        benign_mask = (toxicity_labels == 0)
        malicious_mask = (toxicity_labels == 1)

        if benign_mask.sum() > 0 and malicious_mask.sum() > 0:
            benign_embeddings = embeddings[benign_mask]
            malicious_embeddings = embeddings[malicious_mask]

            benign_centroid = benign_embeddings.mean(dim=0)
            malicious_centroid = malicious_embeddings.mean(dim=0)

            # Distance between benign and malicious centroids (should be large)
            centroid_distance = F.pairwise_distance(
                benign_centroid.unsqueeze(0),
                malicious_centroid.unsqueeze(0)
            )

            # Loss: maximize distance between centroids
            toxicity_loss = torch.clamp(margin_toxicity - centroid_distance, min=0).mean()

            # Add intra-class compactness
            if len(benign_embeddings) > 1:
                benign_distances = torch.norm(
                    benign_embeddings - benign_centroid.unsqueeze(0),
                    dim=1
                )
                toxicity_loss = toxicity_loss + benign_distances.mean()

            if len(malicious_embeddings) > 1:
                malicious_distances = torch.norm(
                    malicious_embeddings - malicious_centroid.unsqueeze(0),
                    dim=1
                )
                toxicity_loss = toxicity_loss + malicious_distances.mean()

    # Combine losses with configurable weights
    weighted_dataset_loss = CONFIG.DATASET_LOSS_WEIGHT * dataset_loss
    weighted_toxicity_loss = CONFIG.TOXICITY_LOSS_WEIGHT * toxicity_loss
    total_loss = weighted_dataset_loss + weighted_toxicity_loss

    return total_loss, dataset_loss, toxicity_loss


def train_learned_projection(features_dict, labels_dict, input_dim=None, output_dim=None,
                           epochs=None, batch_size=None, learning_rate=None, device=None,
                           lr_scheduler=None, random_seed=42):
    """
    Train the learned projection network

    Args:
        features_dict: Dict of {dataset_name: features_array}
        labels_dict: Dict of {dataset_name: labels_array}
        input_dim: Input feature dimension (4096)
        output_dim: Output feature dimension (256)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        device: GPU device
        lr_scheduler: Learning rate scheduler type ('cosine', 'exponential', 'step')
        random_seed: Random seed for reproducibility

    Returns:
        trained_model: Trained projection model
        dataset_name_to_id: Mapping of dataset names to IDs
    """
    # Use config defaults if not specified
    input_dim = input_dim or CONFIG.INPUT_DIM
    output_dim = output_dim or CONFIG.OUTPUT_DIM
    epochs = epochs or CONFIG.PROJECTION_EPOCHS
    batch_size = batch_size or CONFIG.PROJECTION_BATCH_SIZE
    learning_rate = learning_rate or CONFIG.PROJECTION_LEARNING_RATE
    lr_scheduler = lr_scheduler or CONFIG.PROJECTION_LR_SCHEDULER

    if device is None:
        device = GPU_DEVICE

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Training learned projection: {input_dim} -> {output_dim} dimensions")
    print(f"Device: {device}, Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Random seed: {random_seed} (deterministic mode enabled)")

    # Prepare training data
    all_features = []
    all_dataset_labels = []
    all_toxicity_labels = []
    dataset_name_to_id = {}

    dataset_id = 0
    for dataset_name, features in features_dict.items():
        if dataset_name not in labels_dict:
            continue

        features_array = np.array(features)
        labels_array = np.array(labels_dict[dataset_name])

        if len(features_array) != len(labels_array):
            print(f"Warning: Feature-label mismatch for {dataset_name}: {len(features_array)} vs {len(labels_array)}")
            continue

        # Assign dataset ID
        if dataset_name not in dataset_name_to_id:
            dataset_name_to_id[dataset_name] = dataset_id
            dataset_id += 1

        all_features.append(features_array)
        all_dataset_labels.extend([dataset_name_to_id[dataset_name]] * len(features_array))
        all_toxicity_labels.extend(labels_array.tolist())

        print(f"  {dataset_name}: {len(features_array)} samples, dataset_id={dataset_name_to_id[dataset_name]}")

    # Convert to tensors
    all_features = np.vstack(all_features)
    all_dataset_labels = np.array(all_dataset_labels)
    all_toxicity_labels = np.array(all_toxicity_labels)

    print(f"Total training samples: {len(all_features)}")
    print(f"Feature shape: {all_features.shape}")
    print(f"Unique datasets: {len(dataset_name_to_id)}")
    print(f"Toxicity distribution: {np.bincount(all_toxicity_labels)}")

    # Create data loader
    features_tensor = torch.FloatTensor(all_features).to(device)
    dataset_labels_tensor = torch.LongTensor(all_dataset_labels).to(device)
    toxicity_labels_tensor = torch.LongTensor(all_toxicity_labels).to(device)

    dataset = TensorDataset(features_tensor, dataset_labels_tensor, toxicity_labels_tensor)

    # Create generator with fixed seed for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                          generator=generator, worker_init_fn=lambda worker_id: np.random.seed(random_seed + worker_id))

    # Initialize model
    model = LearnedProjection(input_dim=input_dim, output_dim=output_dim).to(device)

    # Initialize optimizer with deterministic behavior
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Ensure model initialization is deterministic by re-seeding before model creation
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Multiple learning rate schedulers for better training dynamics
    # 1. Cosine annealing for smooth decay
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.01)
    # 2. ReduceLROnPlateau for adaptive reduction when loss plateaus
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    # 3. Exponential decay for consistent reduction
    exp_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # 4. Step decay
    step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Select scheduler based on parameter
    if lr_scheduler == 'cosine':
        scheduler = cosine_scheduler
        print(f"Using Cosine Annealing LR scheduler (eta_min={learning_rate*0.01:.6f})")
    elif lr_scheduler == 'exponential':
        scheduler = exp_scheduler
        print(f"Using Exponential LR scheduler (gamma=0.98)")
    elif lr_scheduler == 'step':
        scheduler = step_scheduler
        print(f"Using Step LR scheduler (step_size=30, gamma=0.5)")
    else:
        scheduler = cosine_scheduler  # Default
        print(f"Using default Cosine Annealing LR scheduler")

    # Training loop
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    prev_lr = learning_rate

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_dataset_loss = 0.0
        epoch_toxicity_loss = 0.0
        num_batches = 0

        for batch_features, batch_dataset_labels, batch_toxicity_labels in dataloader:
            optimizer.zero_grad()

            # Forward pass
            embeddings = model(batch_features)

            # Compute loss
            total_loss, dataset_loss, toxicity_loss = compute_contrastive_loss(
                embeddings, batch_dataset_labels, batch_toxicity_labels
            )

            # Backward pass
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_dataset_loss += dataset_loss if isinstance(dataset_loss, (int, float)) else dataset_loss.item()
            epoch_toxicity_loss += toxicity_loss if isinstance(toxicity_loss, (int, float)) else toxicity_loss.item()
            num_batches += 1

        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_dataset_loss = epoch_dataset_loss / num_batches
        avg_toxicity_loss = epoch_toxicity_loss / num_batches

        # Learning rate scheduling
        scheduler.step()  # Cosine annealing doesn't need loss parameter

        # Also apply plateau scheduler for adaptive reduction
        plateau_scheduler.step(avg_loss)

        # Print progress with learning rate information
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 30 == 0 or epoch == 0:
            # Calculate weighted components for display
            weighted_dataset = CONFIG.DATASET_LOSS_WEIGHT * avg_dataset_loss
            weighted_toxicity = CONFIG.TOXICITY_LOSS_WEIGHT * avg_toxicity_loss
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f} "
                  f"(Dataset={avg_dataset_loss:.4f}*{CONFIG.DATASET_LOSS_WEIGHT}={weighted_dataset:.4f}, "
                  f"Toxicity={avg_toxicity_loss:.4f}*{CONFIG.TOXICITY_LOSS_WEIGHT}={weighted_toxicity:.4f}), "
                  f"LR={current_lr:.6f}")

        # Log learning rate changes, every 20 epochs
        if epoch > 0 and abs(current_lr - prev_lr) and epoch % 20 == 0 > 1e-8:
            print(f"  Learning rate changed: {prev_lr:.6f} -> {current_lr:.6f}")
        prev_lr = current_lr

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Training completed. Best loss: {best_loss:.4f}")

    # Set model to evaluation mode
    model.eval()

    return model, dataset_name_to_id


def apply_learned_projection(model, features_dict, device=None):
    """
    Apply the learned projection to transform features from 4096 to 256 dimensions

    Args:
        model: Trained projection model
        features_dict: Dict of {dataset_name: features_array}
        device: GPU device

    Returns:
        projected_features_dict: Dict of {dataset_name: projected_features_array}
    """
    if device is None:
        device = GPU_DEVICE

    model.eval()
    projected_features_dict = {}

    # print("Applying learned projection to features...")

    with torch.no_grad():
        for dataset_name, features in features_dict.items():
            features_array = np.array(features)
            # print(f"  Projecting {dataset_name}: {features_array.shape} -> ", end="")

            # Convert to tensor and project in batches to manage memory
            batch_size = 1000
            projected_features = []

            for i in range(0, len(features_array), batch_size):
                batch_features = features_array[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_features).to(device)

                # Apply projection
                batch_projected = model(batch_tensor)
                projected_features.append(batch_projected.cpu().numpy())

                # Clean up GPU memory
                del batch_tensor, batch_projected
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine batches
            projected_features = np.vstack(projected_features)
            projected_features_dict[dataset_name] = projected_features

            # print(f"{projected_features.shape}")

    return projected_features_dict

def compute_optimal_shrinkage(X, sample_cov):
    """Compute optimal shrinkage intensity using Ledoit-Wolf formula"""
    n, d = X.shape

    if n <= 1:
        return 1.0  # Full shrinkage for very small samples

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute trace of sample covariance
    trace_cov = np.trace(sample_cov)

    # Estimate optimal shrinkage intensity
    # This is a simplified version of the Ledoit-Wolf formula
    if n < d:
        # When n < d, use stronger shrinkage
        lambda_opt = min(1.0, (d - n) / (d + n))
    else:
        # Standard Ledoit-Wolf shrinkage estimation
        # Simplified formula for computational efficiency
        lambda_opt = min(1.0, (d / n) * 0.1)

    return lambda_opt

def enhanced_ledoit_wolf_covariance(X, min_samples=50):
    """Enhanced Ledoit-Wolf covariance estimator with minimum sample size validation"""
    n, d = X.shape

    if n <= 1:
        raise ValueError(f"Cannot compute covariance with only {n} sample(s). Need at least 2 samples.")

    if n < min_samples:
        raise ValueError(
            f"Cluster has only {n} samples, which is below the minimum required {min_samples} samples. "
            f"This is insufficient for reliable covariance estimation. "
            f"Consider: (1) collecting more data, (2) using fewer/broader clusters, or "
            f"(3) reducing min_samples if you understand the risks."
        )

    # Step 1: Compute sample covariance
    sample_cov = np.cov(X.T, bias=False)

    # Step 2: Compute shrinkage target (scaled identity)
    trace_cov = np.trace(sample_cov)
    target = (trace_cov / d) * np.eye(d)

    # Step 3: Enhanced shrinkage intensity (more conservative for small samples)
    lambda_opt = compute_optimal_shrinkage(X, sample_cov)
    if n < min_samples:
        lambda_opt = max(0.2, lambda_opt)  # Minimum 20% shrinkage for small samples

    # Step 4: Shrunk estimator
    shrunk_cov = (1 - lambda_opt) * sample_cov + lambda_opt * target

    return shrunk_cov

def ledoit_wolf_covariance(X):
    """Original Ledoit-Wolf covariance estimator"""
    return enhanced_ledoit_wolf_covariance(X, min_samples=50)

def enhanced_ledoit_wolf_covariance_gpu(X, min_samples=50, device=None):
    """GPU-accelerated Enhanced Ledoit-Wolf covariance estimator with minimum sample size validation"""
    if device is None:
        device = GPU_DEVICE

    n, d = X.shape
    # print(f"    GPU covariance computation: {n} samples, {d} dimensions")

    if n <= 1:
        raise ValueError(f"Cannot compute covariance with only {n} sample(s). Need at least 2 samples.")

    if n < min_samples:
        raise ValueError(
            f"Cluster has only {n} samples, which is below the minimum required {min_samples} samples. "
            f"This is insufficient for reliable covariance estimation. "
            f"Consider: (1) collecting more data, (2) using fewer/broader clusters, or "
            f"(3) reducing min_samples if you understand the risks."
        )

    # Convert to GPU tensor (use float64 for numerical consistency with CPU)
    X_gpu = torch.tensor(X, dtype=torch.float64, device=device)

    # GPU covariance computation
    X_centered = X_gpu - torch.mean(X_gpu, dim=0, keepdim=True)
    sample_cov_gpu = torch.matmul(X_centered.T, X_centered) / (n - 1)

    # Shrinkage target (scaled identity)
    trace_cov = torch.trace(sample_cov_gpu)
    target_gpu = (trace_cov / d) * torch.eye(d, device=device)

    # Enhanced shrinkage intensity (match CPU implementation exactly)
    # Convert back to CPU for shrinkage calculation to ensure consistency
    sample_cov_cpu = sample_cov_gpu.cpu().numpy()
    X_cpu = X_gpu.cpu().numpy()
    lambda_opt = compute_optimal_shrinkage(X_cpu, sample_cov_cpu)
    if n < min_samples:
        lambda_opt = max(0.2, lambda_opt)  # Minimum 20% shrinkage for small samples

    # Shrunk estimator
    shrunk_cov_gpu = (1 - lambda_opt) * sample_cov_gpu + lambda_opt * target_gpu

    # Move back to CPU as numpy array
    return shrunk_cov_gpu.cpu().numpy()

def compute_matrix_inverse_gpu(matrix, device=None):
    """GPU-accelerated matrix inversion with fallback"""
    if device is None:
        device = GPU_DEVICE

    try:
        # Convert to GPU tensor (use float64 for numerical consistency)
        matrix_gpu = torch.tensor(matrix, dtype=torch.float64, device=device)

        # GPU matrix inversion using Cholesky decomposition for stability
        try:
            # Try Cholesky first (faster for positive definite matrices)
            L = torch.linalg.cholesky(matrix_gpu)
            inv_gpu = torch.cholesky_inverse(L)
        except:
            # Fallback to general inverse
            inv_gpu = torch.linalg.inv(matrix_gpu)

        # Move back to CPU
        return inv_gpu.cpu().numpy()

    except Exception as e:
        print(f"    GPU matrix inversion failed: {e}, using CPU pseudo-inverse")
        return np.linalg.pinv(matrix)

def mahalanobis_distance_batch_gpu(X, mean, cov_inv, device=None):
    """GPU-accelerated batch Mahalanobis distance computation"""
    if device is None:
        device = GPU_DEVICE

    try:
        # Convert inputs to GPU tensors (use float64 for numerical consistency)
        X_gpu = torch.tensor(X, dtype=torch.float64, device=device)
        mean_gpu = torch.tensor(mean, dtype=torch.float64, device=device)
        cov_inv_gpu = torch.tensor(cov_inv, dtype=torch.float64, device=device)

        # Compute differences (broadcasting)
        if X_gpu.dim() == 1:
            X_gpu = X_gpu.unsqueeze(0)  # Add batch dimension

        diff = X_gpu - mean_gpu.unsqueeze(0)  # Shape: (batch_size, feature_dim)

        # Batch Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
        # result = diff @ cov_inv @ diff.T, but we want diagonal elements
        temp = torch.matmul(diff, cov_inv_gpu)  # Shape: (batch_size, feature_dim)
        distances_squared = torch.sum(temp * diff, dim=1)  # Element-wise multiply and sum
        distances = torch.sqrt(torch.clamp(distances_squared, min=0))

        # Move back to CPU
        return distances.cpu().numpy()

    except Exception as e:
        print(f"    GPU Mahalanobis distance failed: {e}, using CPU fallback")
        # CPU fallback
        if X.ndim == 1:
            X = X.reshape(1, -1)

        distances = []
        for x in X:
            diff = x - mean
            try:
                result = np.dot(diff, cov_inv)
                result = np.dot(result, diff)
                distances.append(np.sqrt(max(0, result)))
            except:
                distances.append(np.linalg.norm(diff))

        return np.array(distances)

class MCDDetector:

    def __init__(self, use_gpu=True):
        self.in_distribution_clusters = {}  # {cluster_name: {'mean': ..., 'cov_inv': ...}}
        self.ood_clusters = {}  # {attack_type: {'mean': ..., 'cov_inv': ...}}
        self.threshold = 0.0
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            print(f"    MCDDetector initialized with GPU acceleration on {GPU_DEVICE}")
        else:
            print("    MCDDetector initialized with CPU computation")
        
    def _compute_cluster_stats(self, features):
        """Compute mean and inverse covariance for a cluster of features using GPU-accelerated Ledoit-Wolf shrinkage"""
        features = np.array(features)
        mean = np.mean(features, axis=0)

        # print(f"    Computing covariance for {len(features)} samples, {features.shape[1]} dims...")

        if self.use_gpu:
            # GPU-accelerated computation
            try:
                cov = enhanced_ledoit_wolf_covariance_gpu(features)
                cov_inv = compute_matrix_inverse_gpu(cov)
                # print(f"    Successfully computed GPU Ledoit-Wolf covariance inverse")
            except Exception as e:
                print(f"    GPU computation failed: {e}, falling back to CPU")
                cov = ledoit_wolf_covariance(features)
                cov_inv = inv(cov)
                # print(f"    Successfully computed CPU Ledoit-Wolf covariance inverse")
        else:
            # CPU computation
            try:
                cov = ledoit_wolf_covariance(features)
                cov_inv = inv(cov)
                # print(f"    Successfully computed CPU Ledoit-Wolf covariance inverse")
            except np.linalg.LinAlgError as e:
                # error out
                raise ValueError(f"    Error computing covariance: {e}, using identity")
            except Exception as e:
                print(f"    Error computing covariance: {e}, using identity")
                cov_inv = np.eye(features.shape[1])

        return mean, cov_inv
    
    def _mahalanobis_distance(self, x, mean, cov_inv):
        """Compute Mahalanobis distance with GPU acceleration option"""
        if self.use_gpu:
            # Use GPU batch computation even for single sample
            distances = mahalanobis_distance_batch_gpu(x.reshape(1, -1), mean, cov_inv)
            return distances[0]
        else:
            # CPU computation
            diff = x - mean
            try:
                # Use more stable computation
                result = np.dot(diff, cov_inv)
                result = np.dot(result, diff)
                return np.sqrt(max(0, result))  # Ensure non-negative
            except:
                # Fallback to Euclidean distance if Mahalanobis fails
                return np.linalg.norm(diff)

    def _mahalanobis_distance_batch(self, X, mean, cov_inv):
        """Batch Mahalanobis distance computation"""
        if self.use_gpu:
            return mahalanobis_distance_batch_gpu(X, mean, cov_inv)
        else:
            # CPU batch computation
            distances = []
            for x in X:
                distances.append(self._mahalanobis_distance(x, mean, cov_inv))
            return np.array(distances)
    
    def fit_in_distribution(self, in_dist_data):
        """
        Fit in-distribution clusters.
        
        Args:
            in_dist_data: Dict of {cluster_name: list_of_features}
        """
        # print("Fitting in-distribution clusters...")
        for cluster_name, features in in_dist_data.items():
            if len(features) > 1:  # Need at least 2 samples for covariance
                mean, cov_inv = self._compute_cluster_stats(features)
                self.in_distribution_clusters[cluster_name] = {
                    'mean': mean,
                    'cov_inv': cov_inv,
                    'size': len(features)
                }
                # print(f"  {cluster_name}: {len(features)} samples, dim={len(mean)}")
            else:
                print(f"  Warning: {cluster_name} has only {len(features)} samples, skipping")
    
    def fit_ood_clusters(self, ood_data):
        """
        Fit OOD clusters for different attack types.
        
        Args:
            ood_data: Dict of {attack_type: list_of_features}
        """
        # print("Fitting OOD clusters...")
        for attack_type, features in ood_data.items():
            if len(features) > 1:  # Need at least 2 samples for covariance
                mean, cov_inv = self._compute_cluster_stats(features)
                self.ood_clusters[attack_type] = {
                    'mean': mean,
                    'cov_inv': cov_inv,
                    'size': len(features)
                }
                # print(f"  {attack_type}: {len(features)} samples, dim={len(mean)}")
            else:
                print(f"  Warning: {attack_type} has only {len(features)} samples, skipping")
    
    def _compute_min_in_dist_distance(self, x):
        """Compute minimum Mahalanobis distance to any in-distribution cluster"""
        if not self.in_distribution_clusters:
            return float('inf')
            
        min_distance = float('inf')
        for cluster_name, cluster_stats in self.in_distribution_clusters.items():
            distance = self._mahalanobis_distance(x, cluster_stats['mean'], cluster_stats['cov_inv'])
            min_distance = min(min_distance, distance)
        return min_distance
    
    def _compute_min_ood_distance(self, x):
        """Compute minimum Mahalanobis distance to any OOD cluster"""
        if not self.ood_clusters:
            return float('inf')
            
        min_distance = float('inf')
        for attack_type, cluster_stats in self.ood_clusters.items():
            distance = self._mahalanobis_distance(x, cluster_stats['mean'], cluster_stats['cov_inv'])
            min_distance = min(min_distance, distance)
        return min_distance
    
    def compute_ood_score(self, x):
        """
        Compute outlier score for a sample.

        Score = D_mahal(x, Z_in) - D_mahal(x, Z_ood)

        Higher scores indicate more likely to be OOD (jailbreak).
        """
        in_dist_distance = self._compute_min_in_dist_distance(x)
        ood_distance = self._compute_min_ood_distance(x)

        # Handle edge cases
        if np.isinf(in_dist_distance) and np.isinf(ood_distance):
            return 0.0
        elif np.isinf(in_dist_distance):
            return 1000.0  # Very likely OOD
        elif np.isinf(ood_distance):
            return -1000.0  # Very likely in-distribution

        # Compute contrastive score with numerical stability
        score = in_dist_distance - ood_distance

        # Clip extreme values to prevent numerical issues
        score = np.clip(score, -10000, 10000)

        return score
    
    def fit_threshold(self, validation_features, validation_labels):
        """Find optimal threshold using validation data with domain shift awareness"""
        # Use batch computation for speed
        _, scores = self.predict(validation_features)  # This uses batch computation
        validation_labels = np.array(validation_labels)

        # Check if we have both classes
        unique_labels = np.unique(validation_labels)
        if len(unique_labels) < 2:
            print(f"Warning: Only one class in validation data: {unique_labels}")
            self.threshold = 0.0
            return 0.0

        # Find threshold that maximizes balanced accuracy
        from sklearn.metrics import f1_score, balanced_accuracy_score

        # Analyze score distributions to detect potential domain shift
        benign_scores = scores[validation_labels == 0]
        malicious_scores = scores[validation_labels == 1]

        benign_mean, benign_std = np.mean(benign_scores), np.std(benign_scores)
        malicious_mean, malicious_std = np.mean(malicious_scores), np.std(malicious_scores)

        # Calculate separation between distributions
        separation = abs(malicious_mean - benign_mean) / (benign_std + malicious_std + 1e-8)

        print(f"  Validation score separation: {separation:.2f} (higher is better)")

        # Use adaptive threshold range based on score distributions
        if separation > 5.0:  # Well-separated distributions
            # Use narrow range around the midpoint
            midpoint = (benign_mean + malicious_mean) / 2
            range_width = min(benign_std, malicious_std) * 2
            score_range = [midpoint - range_width, midpoint + range_width]
            print(f"  Using narrow threshold range around midpoint: [{score_range[0]:.2f}, {score_range[1]:.2f}]")
        else:  # Overlapping distributions - use wider range
            score_range = np.percentile(scores, [5, 95])
            range_width = score_range[1] - score_range[0]
            score_range[0] -= 0.2 * range_width
            score_range[1] += 0.2 * range_width
            print(f"  Using wide threshold range for overlapping distributions: [{score_range[0]:.2f}, {score_range[1]:.2f}]")

        thresholds = np.linspace(score_range[0], score_range[1], 200)  # More granular search
        best_score = 0
        best_threshold = 0.0
        best_f1 = 0
        best_balanced_acc = 0

        for threshold in thresholds:
            y_pred = (scores > threshold).astype(int)
            try:
                # Use balanced accuracy as primary metric (handles class imbalance better)
                balanced_acc = balanced_accuracy_score(validation_labels, y_pred)
                f1 = f1_score(validation_labels, y_pred, zero_division=0)

                # For domain shift robustness, prioritize balanced accuracy over F1
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
        """Predict whether samples are OOD (jailbreak) with GPU batch optimization"""
        if self.use_gpu and len(features) > 10:  # Use batch processing for larger datasets
            scores = self._compute_ood_scores_batch(features)
        else:
            scores = np.array([self.compute_ood_score(x) for x in features])
        predictions = (scores > self.threshold).astype(int)
        return predictions, scores

    def _compute_ood_scores_batch(self, features):
        """GPU-optimized batch OOD score computation"""
        features = np.array(features)
        batch_size = len(features)

        # Compute minimum distances to in-distribution clusters (batch)
        in_dist_distances = np.full(batch_size, float('inf'))
        for cluster_stats in self.in_distribution_clusters.values():
            distances = self._mahalanobis_distance_batch(features, cluster_stats['mean'], cluster_stats['cov_inv'])
            in_dist_distances = np.minimum(in_dist_distances, distances)

        # Compute minimum distances to OOD clusters (batch)
        ood_distances = np.full(batch_size, float('inf'))
        for cluster_stats in self.ood_clusters.values():
            distances = self._mahalanobis_distance_batch(features, cluster_stats['mean'], cluster_stats['cov_inv'])
            ood_distances = np.minimum(ood_distances, distances)

        # Handle edge cases
        scores = np.zeros(batch_size)

        # Both infinite
        both_inf = np.isinf(in_dist_distances) & np.isinf(ood_distances)
        scores[both_inf] = 0.0

        # Only in_dist infinite
        in_inf = np.isinf(in_dist_distances) & ~np.isinf(ood_distances)
        scores[in_inf] = 1000.0

        # Only ood infinite
        ood_inf = ~np.isinf(in_dist_distances) & np.isinf(ood_distances)
        scores[ood_inf] = -1000.0

        # Normal case
        normal = ~np.isinf(in_dist_distances) & ~np.isinf(ood_distances)
        scores[normal] = in_dist_distances[normal] - ood_distances[normal]

        # Clip extreme values
        scores = np.clip(scores, -10000, 10000)

        return scores
    
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


def prepare_ood_data_structure(datasets_dict, hidden_states_dict, labels_dict):
    in_dist_data = {}
    ood_data = {}
    
    for dataset_name in datasets_dict.keys():
        if dataset_name not in hidden_states_dict:
            continue
            
        features = hidden_states_dict[dataset_name]
        labels = labels_dict[dataset_name]
        
        # Separate benign and malicious samples
        benign_features = [features[i] for i, label in enumerate(labels) if label == 0]
        malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
        
        if benign_features:
            in_dist_data[f"{dataset_name}_benign"] = benign_features
        if malicious_features:
            ood_data[f"{dataset_name}_malicious"] = malicious_features
    
    return in_dist_data, ood_data


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
    MAIN_SEED = 45  # Match the seed used elsewhere in the script
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
        CONFIG.SINGLE_LAYER_TRAINING_LAYER = 16 if REQUESTED_MODEL == 'llava' else CONFIG.SINGLE_LAYER_TRAINING_LAYER

    # Initialize feature extractor
    HiddenStateExtractor = get_extractor_class(REQUESTED_MODEL)
    extractor = HiddenStateExtractor(model_path)

    print("="*80)
    print(f"BALANCED OOD-BASED JAILBREAK DETECTION USING MCD - ABLATION: NO PROJECTION ({REQUESTED_MODEL.upper()})")
    print("="*80)
    print("Approach: Model benign prompts as in-distribution clusters")
    print("          Use balanced jailbreak examples as OOD clusters")
    print("          Apply contrastive Mahalanobis distance for detection")
    print("          ABLATION: Use raw 4096-dim features directly (no learned projection)")
    print("          Enhanced with Ledoit-Wolf covariance for high-dimensional features")
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

    # === ABLATION: SKIP PROJECTION TRAINING ===
    print("\n=== ABLATION STUDY: SKIPPING PROJECTION TRAINING ===")
    print("Using raw 4096-dimensional features directly without learned projection")
    print("This ablation tests the contribution of the learned projection component")

    # Store empty projection models dict for consistency with original code structure
    projection_models = {}
    dataset_name_to_id = None

    # Only use training datasets (in_dist_datasets and ood_datasets)
    training_dataset_names = set(in_dist_datasets.keys()) | set(ood_datasets.keys())
    print(f"Training datasets: {list(training_dataset_names)}")

    # Create a dummy dataset mapping for consistency
    dataset_id = 0
    dataset_name_to_id = {}
    for dataset_name in training_dataset_names:
        dataset_name_to_id[dataset_name] = dataset_id
        dataset_id += 1

    print(f"Dataset name to ID mapping: {dataset_name_to_id}")
    print("Projection training skipped - proceeding with raw 4096-dim features")

    for layer_idx in layers:
        print(f"\n=== Evaluating Layer {layer_idx} ===")

        # Prepare data for this layer
        layer_hidden_states = {}
        layer_labels = {}

        for dataset_name in all_datasets.keys():
            if dataset_name in all_hidden_states:
                layer_hidden_states[dataset_name] = all_hidden_states[dataset_name][layer_idx]
                layer_labels[dataset_name] = all_labels[dataset_name]

        # ABLATION: Skip projection application - use raw 4096-dim features directly
        print(f"  ABLATION: Using raw 4096-dim features for layer {layer_idx} (no projection)")
        # layer_hidden_states already contains the raw features, no transformation needed
        cleanup_gpu_memory()

        # Prepare in-distribution and OOD data structures
        in_dist_data, ood_data = prepare_ood_data_structure(
            in_dist_datasets,
            {k: v for k, v in layer_hidden_states.items() if k in in_dist_datasets},
            {k: v for k, v in layer_labels.items() if k in in_dist_datasets}
        )

        # For OOD data, we want the malicious samples from OOD datasets
        ood_train_data = {}
        for dataset_name in ood_datasets.keys():
            if dataset_name in layer_hidden_states:
                features = layer_hidden_states[dataset_name]
                labels = layer_labels[dataset_name]

                # Get malicious samples (should be most/all samples in OOD datasets)
                malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
                if malicious_features:
                    ood_train_data[f"{dataset_name}_malicious"] = malicious_features

        # Initialize and train MCD detector with GPU acceleration
        detector = MCDDetector(use_gpu=True)

        # Fit in-distribution clusters with GPU monitoring
        # print(f"  {get_gpu_memory_info()}")
        detector.fit_in_distribution(in_dist_data)
        cleanup_gpu_memory()

        # Fit OOD clusters with GPU monitoring
        # print(f"  {get_gpu_memory_info()}")
        detector.fit_ood_clusters(ood_train_data)
        cleanup_gpu_memory()

        # Debug: Check if clusters were created
        # print(f"  In-distribution clusters: {len(detector.in_distribution_clusters)}")
        # print(f"  OOD clusters: {len(detector.ood_clusters)}")

        if len(detector.in_distribution_clusters) == 0 or len(detector.ood_clusters) == 0:
            print(f"  Skipping layer {layer_idx} - insufficient clusters")
            continue

        # Use a larger subset of training data for better threshold optimization
        # Create balanced validation set with more samples
        val_features = []
        val_labels = []

        # Sample from in-distribution data (increase sample size for better threshold estimation)
        # Use deterministic sampling based on indices for reproducibility
        for _, features in in_dist_data.items():
            sample_size = min(100, len(features))
            if sample_size > 0:
                # Use deterministic sampling: take evenly spaced indices
                indices = np.linspace(0, len(features)-1, sample_size, dtype=int)
                sampled_features = [features[i] for i in indices]
                val_features.extend(sampled_features)
                val_labels.extend([0] * len(sampled_features))

        # Sample from OOD data (increase sample size for better threshold estimation)
        for _, features in ood_train_data.items():
            sample_size = min(100, len(features))
            if sample_size > 0:
                # Use deterministic sampling: take evenly spaced indices
                indices = np.linspace(0, len(features)-1, sample_size, dtype=int)
                sampled_features = [features[i] for i in indices]
                val_features.extend(sampled_features)
                val_labels.extend([1] * len(sampled_features))

        # Fit threshold with balanced validation set
        if val_features and len(set(val_labels)) > 1:
            # print(f"  Validation set: {len(val_features)} samples ({val_labels.count(0)} benign, {val_labels.count(1)} malicious)")

            # Debug: Analyze score distributions before threshold fitting (use batch computation for speed)
            # print("  Computing validation scores...")
            # val_predictions, val_scores = detector.predict(val_features)  # Use batch computation
            # benign_scores = [val_scores[i] for i, label in enumerate(val_labels) if label == 0]
            # malicious_scores = [val_scores[i] for i, label in enumerate(val_labels) if label == 1]

            # print(f"  Validation score distributions:")
            # print(f"    Benign: mean={np.mean(benign_scores):.2f}, std={np.std(benign_scores):.2f}, range=[{np.min(benign_scores):.2f}, {np.max(benign_scores):.2f}]")
            # print(f"    Malicious: mean={np.mean(malicious_scores):.2f}, std={np.std(malicious_scores):.2f}, range=[{np.min(malicious_scores):.2f}, {np.max(malicious_scores):.2f}]")

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
                # print(f"      Benign: {benign_stats}")
                # print(f"      Malicious: {malicious_stats}")

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

    # Calculate layer ranking based on COMBINED results (real-world performance)
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

    # Save results to CSV
    output_path = "results/balanced_mcd_ablate_projection_results.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Layer", "Dataset", "Method", "Accuracy", "F1", "TPR", "FPR", "AUROC", "AUPRC", "Threshold", "Combined_Rank", "Individual_Rank"])

        # Create ranking based on combined performance
        layer_combined_ranking = {layer_idx: rank for rank, (layer_idx, _, _, _, _) in enumerate(layer_combined_scores, 1)}

        # Create ranking based on individual dataset average
        layer_individual_ranking = {layer_idx: rank for rank, (layer_idx, _) in enumerate(layer_individual_avg_scores, 1)}

        for layer_idx in layers:
            if layer_results[layer_idx]:
                for dataset_name, result in layer_results[layer_idx].items():
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
                        layer_idx,
                        dataset_name,
                        "MCD",
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

    # Print summary results
    print("\n" + "="*120)
    print("BALANCED OOD JAILBREAK DETECTION SUMMARY (MCD Algorithm)")
    print("="*120)
    print("Training Configuration (2,000 samples, 1:1 ratio):")
    print(f"  In-Distribution Datasets: {list(in_dist_datasets.keys())}")
    print(f"  OOD Datasets: {list(ood_datasets.keys())}")
    print("Test Configuration (1,800 samples, 1:1 ratio):")
    print(f"  Test Datasets: {list(test_datasets.keys())}")
    print("-"*120)

    # COMBINED PERFORMANCE RANKING (Real-world scenario)
    print(f"\n{'COMBINED PERFORMANCE RANKING (Real-world scenario)':<120}")
    print(f"{'Layer':<6} {'Accuracy':<10} {'F1':<8} {'TPR':<8} {'FPR':<8} {'AUROC':<10} {'AUPRC':<10} {'Combined':<10}")
    print("-" * 120)

    for layer_idx, accuracy, auroc, auprc, combined_score in layer_combined_scores:
        if layer_results[layer_idx] and 'COMBINED' in layer_results[layer_idx]:
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

    # INDIVIDUAL DATASET AVERAGE RANKING (for comparison)
    print(f"\n{'INDIVIDUAL DATASET AVERAGE RANKING (for comparison)':<100}")
    print(f"{'Layer':<6} {'Avg_Acc':<10} {'XSTest':<12} {'FigTxt':<12} {'VQAv2':<12} {'VAE':<12} {'JBV-Test':<12}")
    print("-" * 100)

    for layer_idx, avg_acc in layer_individual_avg_scores:
        layer_perf = layer_results[layer_idx]

        # Format average accuracy
        avg_str = f"{avg_acc:.3f}"

        # Format individual dataset performances (excluding COMBINED)
        dataset_strs = []
        for dataset_name in ['XSTest_safe', 'XSTest_unsafe', 'FigTxt_safe', 'FigTxt_unsafe', 'VQAv2', 'VAE', 'JailbreakV-28K_test']:
            if dataset_name in layer_perf:
                acc = layer_perf[dataset_name]['accuracy']
                dataset_strs.append(f"{acc:.3f}")
            else:
                dataset_strs.append("N/A")

        print(f"{layer_idx:<6} {avg_str:<10} {dataset_strs[0]:<12} {dataset_strs[1]:<12} {dataset_strs[2]:<12} {dataset_strs[3]:<12} {dataset_strs[4]:<12}")

    # DOMAIN SHIFT ANALYSIS
    print(f"\n{'DOMAIN SHIFT ANALYSIS':<100}")
    print("Comparing validation vs test score distributions to identify domain shift:")
    print("-" * 100)

    # Find the best performing layer for detailed analysis
    if layer_combined_scores:
        best_layer = layer_combined_scores[0][0]
        print(f"Analysis for best performing layer: {best_layer}")
        print("(Validation scores were shown during training)")
        print("(Test scores are shown in individual dataset analysis above)")
        print("Large differences between validation and test score distributions indicate domain shift.")

    print("\n" + "="*120)
    print("DETAILED LAYER PERFORMANCE")
    print("="*120)

    for layer_idx in layers:
        print(f"\nLayer {layer_idx}:")
        print("-" * 80)

        if layer_results[layer_idx]:
            for dataset_name, result in layer_results[layer_idx].items():
                # Handle NaN values in summary
                f1_str = f"{result['f1']:.4f}"
                tpr_str = "N/A" if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
                fpr_str = "N/A" if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"
                auroc_str = "N/A" if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
                auprc_str = "N/A" if np.isnan(result['auprc']) else f"{result['auprc']:.4f}"

                print(f"  {dataset_name:15s}: Acc={result['accuracy']:.4f}, F1={f1_str}, TPR={tpr_str}, FPR={fpr_str}, "
                      f"AUROC={auroc_str}, AUPRC={auprc_str}, Thresh={result['threshold']:.4f}")
        else:
            print("  No results for this layer")

    print("="*120)
    print("ABLATION STUDY: NO LEARNED PROJECTION - Using raw 4096-dim features directly")
    print("- Balanced 1:1 training and test ratios for robust evaluation")
    print("- Training: Alpaca (500) + MM-Vet (218) + OpenAssistant (282) vs AdvBench (300) + JailbreakV-28K (550) + DAN (150)")
    print("- Testing: XSTest + FigTxt + VQAv2 (safe) vs XSTest + FigTxt + VAE + JailbreakV-28K (unsafe)")
    print("- Results saved to: results/balanced_mcd_ablate_projection_results.csv")


if __name__ == "__main__":
    main()
