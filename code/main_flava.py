#!/usr/bin/env python3
"""
Balanced Jailbreak Detection with FLAVA and New Dataset Configuration

This script uses FLAVA (Foundational Language And Vision Alignment) from Meta to extract features
from the last token of the corresponding encoder (text-only or image-text pairs) and trains
classifiers to detect jailbreaking attempts.

Training Set (2,000 examples, 1:1 ratio):
- Benign (1,000): Alpaca (500) + MM-Vet (218) + OpenAssistant (282)
- Malicious (1,000): AdvBench (300) + JailbreakV-28K (550) + DAN variants (150)

Test Set (1,800 examples, 1:1 ratio):
- Safe (900): XSTest safe (250) + FigTxt safe (300) + VQAv2 (350)
- Unsafe (900): XSTest unsafe (200) + FigTxt unsafe (350) + VAE (200) + JailbreakV-28K (150)

IMPORTANT: JailbreakV-28K Train/Test Separation:
- Training: Uses llm_transfer_attack (275) + query_related (275) = 550 samples total
- Testing: Uses figstep attack only (150 samples)
This ensures different attack types between training and testing for robust evaluation.

This ensures strict train-test separation and balanced evaluation.
"""

import csv
import numpy as np
import random
import os
import json
from sklearn.metrics import accuracy_score, f1_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from transformers import FlavaProcessor, FlavaModel
from PIL import Image
import requests
from io import BytesIO

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

from load_datasets import *

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
class MLConfig:
    """Global configuration for ML experiments"""
    MAIN_SEED = 42

# Global config instance
CONFIG = MLConfig()

# GPU device setup
GPU_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {GPU_DEVICE}")

class FlavaFeatureExtractor:
    """FLAVA-based feature extractor for text and multimodal inputs"""

    def __init__(self, model_path="model/flava"):
        self.processor = None
        self.model = None
        self.device = GPU_DEVICE
        self.model_path = model_path

    def _load_model(self):
        """Load FLAVA model and processor"""
        if self.model is None:
            print(f"Loading FLAVA model from {self.model_path}...")
            self.processor = FlavaProcessor.from_pretrained(self.model_path)
            self.model = FlavaModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("FLAVA model loaded successfully")
    
    def _load_image(self, image_data):
        """Load image from path, URL, or binary data"""
        try:
            # Skip if image_data is None or empty
            if not image_data:
                return None

            # Handle binary image data
            if isinstance(image_data, bytes):
                try:
                    image = Image.open(BytesIO(image_data)).convert('RGB')
                    return image
                except Exception as e:
                    print(f"Error loading binary image data: {e}")
                    return None

            # Handle string paths/URLs
            if isinstance(image_data, str):
                # Check if it's a URL
                if image_data.startswith('http'):
                    response = requests.get(image_data, timeout=10)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    # Check if file exists
                    if not os.path.exists(image_data):
                        print(f"Image file not found: {image_data}")
                        return None

                    image = Image.open(image_data).convert('RGB')

                return image

            # Handle other types (e.g., PIL Image objects)
            if hasattr(image_data, 'convert'):
                return image_data.convert('RGB')

            print(f"Unsupported image data type: {type(image_data)}")
            return None

        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def extract_features(self, samples, batch_size=32):
        """Extract features from samples using FLAVA"""
        self._load_model()

        features = []
        labels = []
        text_only_count = 0
        multimodal_count = 0
        failed_count = 0

        print(f"Extracting features from {len(samples)} samples...")

        with torch.no_grad():
            for i, sample in enumerate(samples):
                try:
                    text = sample.get('txt', '')
                    image_path = sample.get('img', None)
                    toxicity = sample.get('toxicity', 0)

                    # Load image if available
                    image = None
                    if image_path:
                        image = self._load_image(image_path)

                    # Process sample
                    if image is not None:
                        # Multimodal processing
                        inputs = self.processor(text=[text], images=[image],
                                              return_tensors="pt", padding=True, truncation=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        outputs = self.model(**inputs)
                        # Use multimodal embeddings (combines text and image)
                        if hasattr(outputs, 'multimodal_embeddings') and outputs.multimodal_embeddings is not None:
                            feature = outputs.multimodal_embeddings[0, 0, :].cpu().numpy()  # [hidden_dim]
                            multimodal_count += 1
                        else:
                            # Fallback to text embeddings if multimodal not available
                            feature = outputs.text_embeddings[0, 0, :].cpu().numpy()
                            text_only_count += 1
                            print(f"Warning: Multimodal embeddings not available, using text embeddings")
                    else:
                        # Text-only processing
                        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        outputs = self.model(**inputs)
                        # Use text encoder's last hidden state, take the [CLS] token (first token)
                        feature = outputs.text_embeddings[0, 0, :].cpu().numpy()  # [hidden_dim]
                        text_only_count += 1

                    features.append(feature)
                    labels.append(toxicity)

                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    # Add zero features for failed samples
                    features.append(np.zeros(768))  # FLAVA hidden size
                    labels.append(0)
                    failed_count += 1

                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{len(samples)} samples")
                    print(f"  Text-only: {text_only_count}, Multimodal: {multimodal_count}, Failed: {failed_count}")

        print(f"\nFeature extraction summary:")
        print(f"  Text-only samples: {text_only_count}")
        print(f"  Multimodal samples: {multimodal_count}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Total samples: {len(features)}")

        features_array = np.array(features)
        print(f"  Final feature matrix shape: {features_array.shape}")

        return features_array, np.array(labels)

class GPULinearClassifier(nn.Module):
    """GPU-accelerated linear classifier"""
    def __init__(self, input_dim, model_type='logistic'):
        super(GPULinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.model_type = model_type
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

def train_gpu_linear_classifier(X_train, y_train, model_type='logistic', epochs=100, lr=1e-3, batch_size=256):
    """Train a GPU-accelerated linear classifier"""
    device = GPU_DEVICE
    input_dim = X_train.shape[1]

    # Convert to tensors and move to GPU
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)

    # Create model
    model = GPULinearClassifier(input_dim, model_type).to(device)

    # Setup optimizer and loss
    if model_type in ['logistic', 'svm']:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif model_type == 'ridge':
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    else:  # sgd
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)

    # Create data loader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()

            if model_type == 'ridge':
                loss = criterion(torch.sigmoid(outputs), batch_y)
            else:
                loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return model

def evaluate_linear_classifier(model, X_test, y_test):
    """Evaluate a GPU linear classifier"""
    device = GPU_DEVICE
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        raw_outputs = model(X_test_tensor).squeeze()
        y_proba = torch.sigmoid(raw_outputs).cpu().numpy()
        y_pred = (y_proba > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Calculate TPR, FPR from confusion matrix
    if len(np.unique(y_test)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Calculate AUROC and AUPRC
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
        auroc = auc(fpr_curve, tpr_curve)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auprc = auc(recall, precision)
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
        'predictions': y_pred,
        'probabilities': y_proba
    }

class ClusteringBasedClassifier:
    """
    Clustering-based classification for jailbreak detection.

    This approach uses K-means clustering to model the distribution of benign and malicious samples,
    then classifies new samples based on their distance to cluster centroids.

    Methods supported:
    - 'kmeans_distance': Uses distance to nearest cluster centroid
    - 'kmeans_ratio': Uses ratio of distances to benign vs malicious clusters
    - 'gaussian_mixture': Uses Gaussian Mixture Model for probabilistic clustering
    """

    def __init__(self, method='kmeans_distance', n_clusters_per_class=3, use_gpu=True):
        self.method = method
        self.n_clusters_per_class = n_clusters_per_class
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = GPU_DEVICE if self.use_gpu else torch.device('cpu')

        # Clustering models
        self.benign_clusters = None
        self.malicious_clusters = None
        self.scaler = StandardScaler()
        self.threshold = 0.0

        print(f"    ClusteringBasedClassifier initialized with method='{method}', n_clusters={n_clusters_per_class}")
        if self.use_gpu:
            print(f"    Using GPU acceleration on {self.device}")

    def fit(self, X_train, y_train):
        """Train clustering-based classifier"""
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Separate benign and malicious samples
        benign_mask = (y_train == 0)
        malicious_mask = (y_train == 1)

        X_benign = X_train_scaled[benign_mask]
        X_malicious = X_train_scaled[malicious_mask]

        print(f"    Training on {len(X_benign)} benign and {len(X_malicious)} malicious samples")

        if self.method in ['kmeans_distance', 'kmeans_ratio']:
            # Fit K-means clusters for each class
            if len(X_benign) >= self.n_clusters_per_class:
                self.benign_clusters = KMeans(n_clusters=self.n_clusters_per_class,
                                            random_state=CONFIG.MAIN_SEED,
                                            n_init=10)
                self.benign_clusters.fit(X_benign)
                print(f"    Fitted {self.n_clusters_per_class} benign clusters")
            else:
                print(f"    Warning: Not enough benign samples for {self.n_clusters_per_class} clusters")
                self.benign_clusters = KMeans(n_clusters=1, random_state=CONFIG.MAIN_SEED)
                self.benign_clusters.fit(X_benign)

            if len(X_malicious) >= self.n_clusters_per_class:
                self.malicious_clusters = KMeans(n_clusters=self.n_clusters_per_class,
                                               random_state=CONFIG.MAIN_SEED,
                                               n_init=10)
                self.malicious_clusters.fit(X_malicious)
                print(f"    Fitted {self.n_clusters_per_class} malicious clusters")
            else:
                print(f"    Warning: Not enough malicious samples for {self.n_clusters_per_class} clusters")
                self.malicious_clusters = KMeans(n_clusters=1, random_state=CONFIG.MAIN_SEED)
                self.malicious_clusters.fit(X_malicious)

        # Set threshold based on training data using percentile approach
        train_scores = self._compute_scores(X_train_scaled)
        benign_scores = train_scores[~malicious_mask]  # Benign scores (inverted mask)
        malicious_scores = train_scores[malicious_mask]

        # Use percentile-based threshold: Set threshold at 95th percentile of benign training scores
        percentile = 95
        if len(benign_scores) > 0:
            self.threshold = np.percentile(benign_scores, percentile)
        else:
            # Fallback: use percentile of all scores
            self.threshold = np.percentile(train_scores, percentile)

        # Display score distributions and threshold performance
        print(f"  Score distributions on training set:")
        if len(benign_scores) > 0:
            print(f"    Benign: mean={np.mean(benign_scores):.3f}, std={np.std(benign_scores):.3f}, range=[{np.min(benign_scores):.3f}, {np.max(benign_scores):.3f}]")
        if len(malicious_scores) > 0:
            print(f"    Malicious: mean={np.mean(malicious_scores):.3f}, std={np.std(malicious_scores):.3f}, range=[{np.min(malicious_scores):.3f}, {np.max(malicious_scores):.3f}]")

        # Calculate training performance metrics
        y_pred = (train_scores > self.threshold).astype(int)
        train_accuracy = accuracy_score(y_train, y_pred)
        train_f1 = f1_score(y_train, y_pred, zero_division=0)

        if len(benign_scores) > 0 and len(malicious_scores) > 0:
            tpr = np.mean(malicious_scores > self.threshold)  # True Positive Rate
            fpr = np.mean(benign_scores > self.threshold)     # False Positive Rate
            print(f"  Percentile-based threshold: {self.threshold:.4f} (percentile={percentile})")
            print(f"  Training performance: Acc={train_accuracy:.4f}, F1={train_f1:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")
        else:
            print(f"  Percentile-based threshold: {self.threshold:.4f} (percentile={percentile})")
            print(f"  Training performance: Acc={train_accuracy:.4f}, F1={train_f1:.4f}")

    def _compute_scores(self, X):
        """Compute clustering-based scores for samples"""
        if self.method == 'kmeans_distance':
            return self._compute_kmeans_distance_scores(X)
        elif self.method == 'kmeans_ratio':
            return self._compute_kmeans_ratio_scores(X)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

    def _compute_kmeans_distance_scores(self, X):
        """Compute scores based on distance to nearest cluster centroids"""
        scores = []

        for x in X:
            # Distance to nearest benign cluster
            benign_distances = self.benign_clusters.transform([x])[0]
            min_benign_dist = np.min(benign_distances)

            # Distance to nearest malicious cluster
            malicious_distances = self.malicious_clusters.transform([x])[0]
            min_malicious_dist = np.min(malicious_distances)

            # Score: closer to malicious clusters = higher score
            score = min_benign_dist - min_malicious_dist
            scores.append(score)

        return np.array(scores)

    def _compute_kmeans_ratio_scores(self, X):
        """Compute scores based on ratio of distances to benign vs malicious clusters"""
        scores = []

        for x in X:
            # Distance to nearest benign cluster
            benign_distances = self.benign_clusters.transform([x])[0]
            min_benign_dist = np.min(benign_distances)

            # Distance to nearest malicious cluster
            malicious_distances = self.malicious_clusters.transform([x])[0]
            min_malicious_dist = np.min(malicious_distances)

            # Score: ratio of benign distance to malicious distance
            # Higher ratio means closer to malicious clusters
            if min_malicious_dist > 0:
                score = min_benign_dist / min_malicious_dist
            else:
                score = float('inf') if min_benign_dist > 0 else 1.0

            scores.append(score)

        return np.array(scores)

    def predict(self, X_test):
        """Predict using clustering-based classifier"""
        # Standardize test features
        X_test_scaled = self.scaler.transform(X_test)

        # Compute scores
        scores = self._compute_scores(X_test_scaled)

        # Make predictions
        predictions = (scores > self.threshold).astype(int)

        # Convert scores to probabilities (simple sigmoid transformation)
        probabilities = 1 / (1 + np.exp(-scores))

        return predictions, probabilities, scores

def train_clustering_classifier(X_train, y_train, method='kmeans_distance', n_clusters_per_class=3):
    """Train a clustering-based classifier"""
    classifier = ClusteringBasedClassifier(method=method, n_clusters_per_class=n_clusters_per_class)
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_clustering_classifier(classifier, X_test, y_test):
    """Evaluate a clustering-based classifier"""
    predictions, probabilities, scores = classifier.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Calculate TPR, FPR from confusion matrix
    if len(np.unique(y_test)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Calculate AUROC and AUPRC
        fpr_curve, tpr_curve, _ = roc_curve(y_test, probabilities)
        auroc = auc(fpr_curve, tpr_curve)
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        auprc = auc(recall, precision)
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
        'probabilities': probabilities,
        'scores': scores
    }

def prepare_balanced_datasets_organized():
    """
    Load balanced datasets in organized format for dataset-specific caching.
    Returns training and test datasets as dictionaries with dataset names as keys.
    This matches the approach used in other balanced_* scripts.

    IMPORTANT: Train/Test Separation for JailbreakV-28K:
    - Training: Uses llm_transfer_attack (275) + query_related (275) = 550 samples total
    - Testing: Uses figstep attack only (150 samples)
    This ensures different attack types between training and testing for robust evaluation.
    Controlled distribution ensures exactly 275 samples from each training attack type.
    """
    print("Loading balanced datasets in organized format...")
    print("Note: JailbreakV-28K uses different attack types for train/test separation:")
    print("  - Training: llm_transfer_attack (275) + query_related (275) = 550 total")
    print("  - Testing: figstep attack only (150 samples)")

    # === TRAINING DATASETS ===
    training_datasets = {}

    # Benign training data
    print("Loading benign training data...")

    # 1. Alpaca - 500 samples
    try:
        alpaca_samples = load_alpaca(max_samples=500)
        if alpaca_samples:
            training_datasets["Alpaca"] = alpaca_samples
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
                training_datasets["MM-Vet"] = mmvet_benign
                print(f"  Loaded {len(mmvet_benign)} MM-Vet samples")
    except Exception as e:
        print(f"Could not load MM-Vet: {e}")

    # 3. OpenAssistant - 282 samples
    try:
        openassistant_samples = load_openassistant(max_samples=282)
        if openassistant_samples:
            training_datasets["OpenAssistant"] = openassistant_samples
            print(f"  Loaded {len(openassistant_samples)} OpenAssistant samples")
    except Exception as e:
        print(f"Could not load OpenAssistant: {e}")

    # Malicious training data
    print("Loading malicious training data...")

    # 1. AdvBench - 300 samples
    try:
        advbench_samples = load_advbench(max_samples=300)
        if advbench_samples:
            training_datasets["AdvBench"] = advbench_samples
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
            training_datasets["JailbreakV-28K"] = jbv_samples
            print(f"  Total JailbreakV-28K training samples: {len(jbv_samples)} ({len(llm_attack_samples)} llm_transfer + {len(query_related_samples)} query_related)")
        else:
            print("  Warning: No JailbreakV-28K samples loaded")

    except Exception as e:
        print(f"Could not load JailbreakV-28K: {e}")

    # 3. DAN variants - 150 samples
    try:
        dan_samples = load_dan_prompts(max_samples=150)  # Use the correct function name
        if dan_samples:
            training_datasets["DAN"] = dan_samples
            print(f"  Loaded {len(dan_samples)} DAN samples")
    except Exception as e:
        print(f"Could not load DAN variants: {e}")

    # === TEST DATASETS ===
    test_datasets = {}

    # Safe test data
    print("Loading safe test data...")

    # 1. XSTest safe - 250 samples
    try:
        xstest_samples = load_XSTest()  # Use the correct function name
        if xstest_samples:
            xstest_safe = [s for s in xstest_samples if s.get('toxicity', 0) == 0][:250]
            if xstest_safe:
                test_datasets["XSTest_safe"] = xstest_safe
                print(f"  Loaded {len(xstest_safe)} XSTest safe samples")
    except Exception as e:
        print(f"Could not load XSTest: {e}")

    # 2. FigTxt safe - 300 samples
    try:
        figtxt_samples = load_FigTxt()  # Use the correct function name
        if figtxt_samples:
            figtxt_safe = [s for s in figtxt_samples if s.get('toxicity', 0) == 0][:300]
            if figtxt_safe:
                test_datasets["FigTxt_safe"] = figtxt_safe
                print(f"  Loaded {len(figtxt_safe)} FigTxt safe samples")
    except Exception as e:
        print(f"Could not load FigTxt: {e}")

    # 3. VQAv2 - 350 samples
    try:
        vqav2_samples = load_vqav2(max_samples=350)
        if vqav2_samples:
            test_datasets["VQAv2"] = vqav2_samples
            print(f"  Loaded {len(vqav2_samples)} VQAv2 samples")
    except Exception as e:
        print(f"Could not load VQAv2: {e}")

    # Unsafe test data
    print("Loading unsafe test data...")

    # 1. XSTest unsafe - 200 samples
    try:
        if 'xstest_samples' in locals():
            xstest_unsafe = [s for s in xstest_samples if s.get('toxicity', 0) == 1][:200]
            if xstest_unsafe:
                test_datasets["XSTest_unsafe"] = xstest_unsafe
                print(f"  Loaded {len(xstest_unsafe)} XSTest unsafe samples")
    except Exception as e:
        print(f"Could not load XSTest unsafe: {e}")

    # 2. FigTxt unsafe - 350 samples
    try:
        if 'figtxt_samples' in locals():
            figtxt_unsafe = [s for s in figtxt_samples if s.get('toxicity', 0) == 1][:350]
            if figtxt_unsafe:
                test_datasets["FigTxt_unsafe"] = figtxt_unsafe
                print(f"  Loaded {len(figtxt_unsafe)} FigTxt unsafe samples")
    except Exception as e:
        print(f"Could not load FigTxt unsafe: {e}")

    # 3. VAE - 200 samples
    try:
        vae_samples = load_adversarial_img()  # Use the correct function name
        if vae_samples:
            # Limit to 200 samples
            vae_limited = vae_samples[:200]
            test_datasets["VAE"] = vae_limited
            print(f"  Loaded {len(vae_limited)} VAE samples")
    except Exception as e:
        print(f"Could not load VAE: {e}")

    # 4. JailbreakV-28K test - 150 samples (figstep attack only)
    try:
        jbv_test_samples = load_JailBreakV_figstep(max_samples=150)
        if jbv_test_samples:
            test_datasets["JailbreakV-28K_test"] = jbv_test_samples
            print(f"  Loaded {len(jbv_test_samples)} JailbreakV-28K test samples (figstep attack)")
    except Exception as e:
        print(f"Could not load JailbreakV-28K test: {e}")

    return training_datasets, test_datasets

def create_balanced_training_set():
    """Legacy function - now uses organized dataset loading"""
    training_datasets, _ = prepare_balanced_datasets_organized()

    # Combine training datasets
    benign_samples = []
    malicious_samples = []

    # Combine benign datasets
    benign_dataset_names = ["Alpaca", "MM-Vet", "OpenAssistant"]
    for dataset_name in benign_dataset_names:
        if dataset_name in training_datasets:
            samples = training_datasets[dataset_name]
            # Add debug source tag for compatibility
            for sample in samples:
                sample['debug_source'] = dataset_name
            benign_samples.extend(samples)

    # Combine malicious datasets
    malicious_dataset_names = ["AdvBench", "JailbreakV-28K", "DAN"]
    for dataset_name in malicious_dataset_names:
        if dataset_name in training_datasets:
            samples = training_datasets[dataset_name]
            # Add debug source tag for compatibility
            for sample in samples:
                sample['debug_source'] = dataset_name
            malicious_samples.extend(samples)

    print(f"Training set: {len(benign_samples)} benign, {len(malicious_samples)} malicious")
    return benign_samples, malicious_samples

def create_balanced_test_set():
    """Legacy function - now uses organized dataset loading"""
    _, test_datasets = prepare_balanced_datasets_organized()

    # Combine test datasets
    safe_samples = []
    unsafe_samples = []

    # Combine safe datasets
    safe_dataset_names = ["XSTest_safe", "FigTxt_safe", "VQAv2"]
    for dataset_name in safe_dataset_names:
        if dataset_name in test_datasets:
            samples = test_datasets[dataset_name]
            # Add debug source tag for compatibility
            for sample in samples:
                sample['debug_source'] = dataset_name
            safe_samples.extend(samples)

    # Combine unsafe datasets
    unsafe_dataset_names = ["XSTest_unsafe", "FigTxt_unsafe", "VAE", "JailbreakV-28K_test"]
    for dataset_name in unsafe_dataset_names:
        if dataset_name in test_datasets:
            samples = test_datasets[dataset_name]
            # Add debug source tag for compatibility
            for sample in samples:
                sample['debug_source'] = dataset_name
            unsafe_samples.extend(samples)

    print(f"Test set: {len(safe_samples)} safe, {len(unsafe_samples)} unsafe")
    return safe_samples, unsafe_samples

def save_results_and_summary(results, model_types, train_size, test_size):
    """Save results to CSV and print summary"""
    # Save results to CSV
    output_path = "results/balanced_ml_flava_results.csv"
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model_Type", "Split", "Accuracy", "F1", "TPR", "FPR", "AUROC", "AUPRC", "Threshold", "Train_Size", "Test_Size"])

        for model_type in model_types:
            result = results[model_type]

            # Format model name for CSV
            model_name_csv = model_type.upper().replace('_', '-') if '_' in model_type else model_type.upper()

            # Write training results
            train_result = result['train']
            train_f1_val = f"{train_result['f1']:.4f}"
            train_tpr_val = "N/A" if np.isnan(train_result['tpr']) else f"{train_result['tpr']:.4f}"
            train_fpr_val = "N/A" if np.isnan(train_result['fpr']) else f"{train_result['fpr']:.4f}"
            train_auroc_val = "N/A" if np.isnan(train_result['auroc']) else f"{train_result['auroc']:.4f}"
            train_auprc_val = "N/A" if np.isnan(train_result['auprc']) else f"{train_result['auprc']:.4f}"

            writer.writerow([
                model_name_csv, "Train",
                f"{train_result['accuracy']:.4f}",
                train_f1_val, train_tpr_val, train_fpr_val, train_auroc_val, train_auprc_val,
                "0.5000", train_size, test_size
            ])

            # Write test results
            test_result = result['test']
            test_f1_val = f"{test_result['f1']:.4f}"
            test_tpr_val = "N/A" if np.isnan(test_result['tpr']) else f"{test_result['tpr']:.4f}"
            test_fpr_val = "N/A" if np.isnan(test_result['fpr']) else f"{test_result['fpr']:.4f}"
            test_auroc_val = "N/A" if np.isnan(test_result['auroc']) else f"{test_result['auroc']:.4f}"
            test_auprc_val = "N/A" if np.isnan(test_result['auprc']) else f"{test_result['auprc']:.4f}"

            writer.writerow([
                model_name_csv, "Test",
                f"{test_result['accuracy']:.4f}",
                test_f1_val, test_tpr_val, test_fpr_val, test_auroc_val, test_auprc_val,
                "0.5000", train_size, test_size
            ])

    print(f"\nResults saved to {output_path}")

    # Print summary - find best performing model (based on test accuracy)
    best_performance = []
    for model_type in model_types:
        test_result = results[model_type]['test']
        best_performance.append((model_type, test_result['accuracy']))

    best_performance.sort(key=lambda x: x[1], reverse=True)
    best_model, best_accuracy = best_performance[0]

    print("\n" + "="*80)
    print("CHALLENGING JAILBREAK DETECTION WITH FLAVA SUMMARY")
    print("="*80)
    print(f"Model: FLAVA (Foundational Language And Vision Alignment)")
    print(f"Feature Extraction: Last token from text encoder (text-only) or multimodal embeddings (image+text)")
    print(f"Training Set: {train_size} samples (5000 benign + 2000 malicious, NO jailbreaking attempts)")
    print(f"Test Set: {test_size} samples (2000 benign + 2000 malicious, WITH jailbreaking attempts)")
    # Format best model name for display
    best_model_display = best_model.upper().replace('_', '-') if '_' in best_model else best_model.upper()
    print(f"Best Overall Performance: {best_model_display} (Accuracy: {best_accuracy:.4f})")

    # Format model names for display
    model_names_display = []
    for m in model_types:
        if '_' in m:
            model_names_display.append(m.upper().replace('_', '-'))
        else:
            model_names_display.append(m.upper())
    print(f"Models compared: {', '.join(model_names_display)}")

    print(f"\nAll Model Performance (Test Set):")
    for i, (model_type, accuracy) in enumerate(best_performance, 1):
        test_result = results[model_type]['test']
        train_result = results[model_type]['train']

        test_auroc_str = "N/A" if np.isnan(test_result['auroc']) else f"{test_result['auroc']:.4f}"
        test_f1_str = f"{test_result['f1']:.4f}"
        test_tpr_str = "N/A" if np.isnan(test_result['tpr']) else f"{test_result['tpr']:.4f}"
        test_fpr_str = "N/A" if np.isnan(test_result['fpr']) else f"{test_result['fpr']:.4f}"

        # Calculate overfitting metrics
        acc_gap = train_result['accuracy'] - test_result['accuracy']
        f1_gap = train_result['f1'] - test_result['f1']

        # Format model name for display
        model_display = model_type.upper().replace('_', '-') if '_' in model_type else model_type.upper()
        print(f"  {i:2d}. {model_display}: Test Acc={accuracy:.4f}, F1={test_f1_str}, TPR={test_tpr_str}, FPR={test_fpr_str}, AUROC={test_auroc_str}")
        print(f"      Overfitting: Acc Gap={acc_gap:+.4f}, F1 Gap={f1_gap:+.4f}, Train Acc={train_result['accuracy']:.4f}")

    print("="*80)

def main():
    print("="*80)
    print("CHALLENGING JAILBREAK DETECTION WITH FLAVA")
    print("="*80)

    # Set random seed for reproducibility
    MAIN_SEED = CONFIG.MAIN_SEED
    random.seed(MAIN_SEED)
    np.random.seed(MAIN_SEED)
    torch.manual_seed(MAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(MAIN_SEED)

    os.environ['PYTHONHASHSEED'] = str(MAIN_SEED)
    print(f"Random seeds set for reproducibility (seed={MAIN_SEED})")

    # Load datasets using the same organized approach as other balanced_* scripts
    training_datasets, test_datasets = prepare_balanced_datasets_organized()

    # Initialize FLAVA feature extractor
    extractor = FlavaFeatureExtractor()

    # Extract features for all datasets individually (with dataset-specific caching)
    print("\n--- Extracting FLAVA Features with Dataset-Specific Caching ---")
    all_datasets = {**training_datasets, **test_datasets}
    all_features = {}
    all_labels = {}

    for dataset_name, samples in all_datasets.items():
        if len(samples) > 0:  # Only process non-empty datasets
            print(f"Extracting FLAVA features for {dataset_name} ({len(samples)} samples)...")

            # Use smaller batch sizes for large datasets to manage memory
            batch_size = 8 if len(samples) > 500 else 16

            # Extract features (FLAVA doesn't have built-in caching like HiddenStateExtractor)
            # TODO: Could add caching functionality to FlavaFeatureExtractor in the future
            features, labels = extractor.extract_features(samples, batch_size=batch_size)
            all_features[dataset_name] = features
            all_labels[dataset_name] = labels
        else:
            print(f"Skipping {dataset_name} - no samples found")

    # Combine training and test data from individual datasets
    print("\n--- Combining Individual Dataset Features ---")
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    # Combine training datasets
    training_dataset_names = ["Alpaca", "MM-Vet", "OpenAssistant", "AdvBench", "JailbreakV-28K", "DAN"]
    for dataset_name in training_dataset_names:
        if dataset_name in all_features:
            dataset_features = all_features[dataset_name]
            dataset_labels = all_labels[dataset_name]
            X_train_list.append(dataset_features)
            y_train_list.extend(dataset_labels)

    # Combine test datasets
    test_dataset_names = ["XSTest_safe", "FigTxt_safe", "VQAv2", "XSTest_unsafe", "FigTxt_unsafe", "VAE", "JailbreakV-28K_test"]
    for dataset_name in test_dataset_names:
        if dataset_name in all_features:
            dataset_features = all_features[dataset_name]
            dataset_labels = all_labels[dataset_name]
            X_test_list.append(dataset_features)
            y_test_list.extend(dataset_labels)

    # Concatenate all features
    X_train = np.vstack(X_train_list) if X_train_list else np.array([])
    y_train = np.array(y_train_list)
    X_test = np.vstack(X_test_list) if X_test_list else np.array([])
    y_test = np.array(y_test_list)

    print(f"Combined training features: {len(y_train)} samples ({np.sum(y_train == 0)} benign, {np.sum(y_train == 1)} malicious)")
    print(f"Combined test features: {len(y_test)} samples ({np.sum(y_test == 0)} safe, {np.sum(y_test == 1)} unsafe)")

    print(f"\nFeature extraction completed:")
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Train and evaluate models
    results = {}
    model_types = ['logistic', 'ridge', 'svm', 'sgd']
    clustering_methods = ['kmeans_distance', 'kmeans_ratio']

    print("\n--- Training and Evaluating Linear Models ---")
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} classifier...")

        # Train linear classifier
        model = train_gpu_linear_classifier(X_train, y_train, model_type=model_type,
                                          epochs=100, lr=1e-3, batch_size=256)

        # Evaluate on training set
        train_results = evaluate_linear_classifier(model, X_train, y_train)

        # Evaluate on test set
        test_results = evaluate_linear_classifier(model, X_test, y_test)

        # Store both training and test results
        results[model_type] = {
            'train': train_results,
            'test': test_results
        }

        # Handle NaN values for display - Training
        train_f1_str = f"{train_results['f1']:.4f}"
        train_tpr_str = "N/A" if np.isnan(train_results['tpr']) else f"{train_results['tpr']:.4f}"
        train_fpr_str = "N/A" if np.isnan(train_results['fpr']) else f"{train_results['fpr']:.4f}"
        train_auroc_str = "N/A" if np.isnan(train_results['auroc']) else f"{train_results['auroc']:.4f}"
        train_auprc_str = "N/A" if np.isnan(train_results['auprc']) else f"{train_results['auprc']:.4f}"

        # Handle NaN values for display - Testing
        test_f1_str = f"{test_results['f1']:.4f}"
        test_tpr_str = "N/A" if np.isnan(test_results['tpr']) else f"{test_results['tpr']:.4f}"
        test_fpr_str = "N/A" if np.isnan(test_results['fpr']) else f"{test_results['fpr']:.4f}"
        test_auroc_str = "N/A" if np.isnan(test_results['auroc']) else f"{test_results['auroc']:.4f}"
        test_auprc_str = "N/A" if np.isnan(test_results['auprc']) else f"{test_results['auprc']:.4f}"

        print(f"    {model_type.upper()} Training  - Accuracy: {train_results['accuracy']:.4f}, F1: {train_f1_str}, TPR: {train_tpr_str}, FPR: {train_fpr_str}, AUROC: {train_auroc_str}, AUPRC: {train_auprc_str}")
        print(f"    {model_type.upper()} Testing   - Accuracy: {test_results['accuracy']:.4f}, F1: {test_f1_str}, TPR: {test_tpr_str}, FPR: {test_fpr_str}, AUROC: {test_auroc_str}, AUPRC: {test_auprc_str}")

        # Calculate and display overfitting metrics
        acc_gap = train_results['accuracy'] - test_results['accuracy']
        f1_gap = train_results['f1'] - test_results['f1']
        print(f"    {model_type.upper()} Overfitting - Acc Gap: {acc_gap:+.4f}, F1 Gap: {f1_gap:+.4f}")

    print("\n--- Training and Evaluating Clustering-Based Models ---")
    for clustering_method in clustering_methods:
        print(f"\nTraining {clustering_method.upper().replace('_', '-')} clustering classifier...")

        # Train clustering classifier
        clustering_model = train_clustering_classifier(X_train, y_train,
                                                     method=clustering_method,
                                                     n_clusters_per_class=3)

        # Evaluate on training set
        train_results = evaluate_clustering_classifier(clustering_model, X_train, y_train)

        # Evaluate on test set
        test_results = evaluate_clustering_classifier(clustering_model, X_test, y_test)

        # Store both training and test results
        results[clustering_method] = {
            'train': train_results,
            'test': test_results
        }

        # Handle NaN values for display - Training
        train_f1_str = f"{train_results['f1']:.4f}"
        train_tpr_str = "N/A" if np.isnan(train_results['tpr']) else f"{train_results['tpr']:.4f}"
        train_fpr_str = "N/A" if np.isnan(train_results['fpr']) else f"{train_results['fpr']:.4f}"
        train_auroc_str = "N/A" if np.isnan(train_results['auroc']) else f"{train_results['auroc']:.4f}"
        train_auprc_str = "N/A" if np.isnan(train_results['auprc']) else f"{train_results['auprc']:.4f}"

        # Handle NaN values for display - Testing
        test_f1_str = f"{test_results['f1']:.4f}"
        test_tpr_str = "N/A" if np.isnan(test_results['tpr']) else f"{test_results['tpr']:.4f}"
        test_fpr_str = "N/A" if np.isnan(test_results['fpr']) else f"{test_results['fpr']:.4f}"
        test_auroc_str = "N/A" if np.isnan(test_results['auroc']) else f"{test_results['auroc']:.4f}"
        test_auprc_str = "N/A" if np.isnan(test_results['auprc']) else f"{test_results['auprc']:.4f}"

        method_display = clustering_method.upper().replace('_', '-')
        print(f"    {method_display} Training  - Accuracy: {train_results['accuracy']:.4f}, F1: {train_f1_str}, TPR: {train_tpr_str}, FPR: {train_fpr_str}, AUROC: {train_auroc_str}, AUPRC: {train_auprc_str}")
        print(f"    {method_display} Testing   - Accuracy: {test_results['accuracy']:.4f}, F1: {test_f1_str}, TPR: {test_tpr_str}, FPR: {test_fpr_str}, AUROC: {test_auroc_str}, AUPRC: {test_auprc_str}")

        # Calculate and display overfitting metrics
        acc_gap = train_results['accuracy'] - test_results['accuracy']
        f1_gap = train_results['f1'] - test_results['f1']
        print(f"    {method_display} Overfitting - Acc Gap: {acc_gap:+.4f}, F1 Gap: {f1_gap:+.4f}")

    # Update model_types to include clustering methods for results saving
    all_model_types = model_types + clustering_methods

    # Save results and generate summary
    save_results_and_summary(results, all_model_types, len(y_train), len(y_test))

if __name__ == "__main__":
    main()
