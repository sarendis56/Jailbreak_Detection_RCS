#!/usr/bin/env python3
"""
Principled Layer Selection for Jailbreak Detection
EXPERIMENTS:
  1. SGXSTest (Original): Paired dataset with carefully curated benign/malicious pairs
  2. Noisy Distribution: Random unmatched samples from benign and malicious datasets
  3. Latent Neighbor: Synthetic pairs created via embedding similarity (nearest neighbors)
  4. In-the-Wild: Actual training split (multimodal + text-only, unpaired)
  5. XSTest: Similar to SGXSTest but used in training (for comparison)

METRICS EVALUATED:
  - Distributional Divergence: MMD, Wasserstein Distance, KL Divergence
  - Geometric Separation: SVM Margin, Silhouette Coefficient, Distance Ratio
  - Information-Theoretic: Mutual Information, Entropy Reduction

USAGE:
    python principled_layer_selection.py <model_type> [model_path] [experiments]
    
    Arguments:
        model_type: qwen | llava | internvl
        model_path: (optional) Path to the model directory
                    If not provided, auto-detects from model/ directory:
                    - llava -> model/llava-v1.6-vicuna-7b/
                    - qwen -> model/qwen2.5-vl-7b-instruct/
                    - internvl -> model/internvl3-8b/
        experiments: (optional) Comma-separated list of experiment numbers (1-5), e.g., "1,2,3,4,5"
                     If not specified, runs all experiments (1,2,3,4,5)
    
    Examples:
        # Run all experiments (auto-detect model path)
        python analysis/principled_layer_selection.py llava
        
        # Run only experiments 1 and 3 (auto-detect model path)
        python analysis/principled_layer_selection.py llava 1,3
        
        # Run with custom model path
        python analysis/principled_layer_selection.py llava model/llava-v1.6-vicuna-7b/ 1,3
        
        # Run experiment 5 (XSTest)
        python analysis/principled_layer_selection.py llava 5

OUTPUT:
    - CSV files with detailed results for each experiment
    - Correlation analysis comparing experiments
    - Visualization plots (if matplotlib available)
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from typing import Dict, List, Tuple, Any, Optional
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial.distance import pdist, cdist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add code directory to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from load_datasets import *

# Model paths in model/ directory (from download_models.py)
MODEL_PATHS = {
    'llava': 'model/llava-v1.6-vicuna-7b/',
    'qwen': 'model/qwen2.5-vl-7b-instruct/',
    'internvl': 'model/internvl3-8b/'
}

class LayerDiscriminativeAnalyzer:
    """
    Comprehensive analyzer for evaluating discriminative power of different layers
    in detecting jailbreaking attempts using multiple metrics.
    """
    
    def __init__(self, model_path: str, model_type: str):
        self.model_path = model_path
        model_type = model_type.lower()
        
        # Import only the specified model extractor
        if model_type == 'qwen':
            from feature_extractor_qwen import HiddenStateExtractor as QwenHiddenStateExtractor
            self.extractor = QwenHiddenStateExtractor(model_path)
        elif model_type == 'llava':
            from feature_extractor import HiddenStateExtractor as LLaVAHiddenStateExtractor
            self.extractor = LLaVAHiddenStateExtractor(model_path)
        elif model_type == 'internvl':
            try:
                from code.feature_extractor_internvl import HiddenStateExtractor as InternVLHiddenStateExtractor
            except ImportError:
                from feature_extractor_internvl import HiddenStateExtractor as InternVLHiddenStateExtractor
            self.extractor = InternVLHiddenStateExtractor(model_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use one of: 'qwen', 'llava', 'internvl'.")
        self.results = {}

    def _get_model_name_for_filename(self):
        """Extract a clean model name for use in filenames"""
        model_path = self.model_path.lower()
        if 'qwen' in model_path:
            if '3b' in model_path:
                return 'qwen25vl_3b'
            elif '7b' in model_path:
                return 'qwen25vl_7b'
            else:
                return 'qwen25vl'
        elif 'llava' in model_path:
            if '7b' in model_path:
                return 'llava_7b'
            elif '13b' in model_path:
                return 'llava_13b'
            else:
                return 'llava'
        else:
            # Fallback: use last part of path, cleaned up
            import os
            model_name = os.path.basename(model_path.rstrip('/'))
            # Clean up the name for filename use
            model_name = model_name.replace('-', '_').replace('.', '_').lower()
            return model_name

    def get_model_layer_range(self):
        """Get the appropriate layer range for the current model"""
        return self.extractor.get_default_layer_range()

    def load_sgxstest_dataset(self) -> List[Dict]:
        """
        Load the SGXSTest dataset from HuggingFace.
        
        Returns:
            List of samples with 'txt', 'img', and 'toxicity' fields
        """
        try:
            print("Loading SGXSTest dataset from HuggingFace...")
            dataset = load_dataset("walledai/SGXSTest", split="train")
            
            samples = []
            for item in dataset:
                # Convert to our standard format
                sample = {
                    "txt": item.get("prompt", ""),
                    "img": None,  # SGXSTest is text-only
                    "toxicity": 1 if item.get("label", "safe") == "unsafe" else 0,
                    "category": item.get("category", "unknown"),
                    "pair_id": item.get("pair_id", None)
                }
                samples.append(sample)
            
            print(f"Successfully loaded {len(samples)} samples from SGXSTest")
            
            # Verify we have balanced pairs
            benign_count = sum(1 for s in samples if s['toxicity'] == 0)
            malicious_count = sum(1 for s in samples if s['toxicity'] == 1)
            print(f"Dataset composition: {benign_count} benign, {malicious_count} malicious")
            
            return samples
            
        except Exception as e:
            print(f"Error loading SGXSTest dataset: {e}")
            print("Dataset is required for analysis. Please ensure:")
            print("   1. You have access to the gated dataset")
            print("   2. Your HuggingFace token is properly configured")
            raise RuntimeError(f"Failed to load required SGXSTest dataset: {e}")
    
    def load_xstest_dataset(self) -> List[Dict]:
        """
        Load the XSTest dataset (similar to SGXSTest but used in training).
        
        Returns:
            List of samples with 'txt', 'img', and 'toxicity' fields
        """
        print("Loading XSTest dataset...")
        try:
            samples = load_XSTest()
            
            print(f"Successfully loaded {len(samples)} samples from XSTest")
            
            # Verify we have balanced classes
            benign_count = sum(1 for s in samples if s['toxicity'] == 0)
            malicious_count = sum(1 for s in samples if s['toxicity'] == 1)
            print(f"Dataset composition: {benign_count} benign, {malicious_count} malicious")
            
            return samples
            
        except Exception as e:
            print(f"Error loading XSTest dataset: {e}")
            raise RuntimeError(f"Failed to load XSTest dataset: {e}")
    
    def load_noisy_distribution_dataset(self, n_samples: int = 100) -> List[Dict]:
        """
        Experiment 1: Load random, unmatched samples from benign and malicious datasets.
        This tests if layer selection works without carefully curated pairs.
        
        Args:
            n_samples: Number of samples per class (default: 100 to match SGXSTest size)
        
        Returns:
            List of samples with no semantic relationship between benign and malicious pairs
        """
        print("Loading random, unmatched samples...")
        
        samples = []
        
        # Load benign samples (random selection from Alpaca)
        try:
            benign_pool = load_alpaca(max_samples=None)
            if len(benign_pool) > n_samples:
                import random
                random.seed(42)  # For reproducibility
                benign_samples = random.sample(benign_pool, n_samples)
            else:
                benign_samples = benign_pool[:n_samples]
            
            for sample in benign_samples:
                samples.append({
                    "txt": sample.get("txt", ""),
                    "img": sample.get("img", None),
                    "toxicity": 0,
                    "experiment": "noisy_distribution"
                })
            
            print(f"  Loaded {len(benign_samples)} benign samples")
        except Exception as e:
            print(f"  Warning: Could not load Alpaca: {e}")
            # Fallback to other benign sources
            try:
                benign_pool = load_openassistant(max_samples=None)
                if len(benign_pool) > n_samples:
                    import random
                    random.seed(42)
                    benign_samples = random.sample(benign_pool, n_samples)
                else:
                    benign_samples = benign_pool[:n_samples]
                
                for sample in benign_samples:
                    samples.append({
                        "txt": sample.get("txt", ""),
                        "img": sample.get("img", None),
                        "toxicity": 0,
                        "experiment": "noisy_distribution"
                    })
                print(f"  Loaded {len(benign_samples)} benign samples")
            except Exception as e2:
                print(f"  Error loading benign samples: {e2}")
                raise RuntimeError("Failed to load benign samples for noisy distribution test")
        
        # Load malicious samples (random selection from AdvBench)
        try:
            malicious_pool = load_advbench(max_samples=None)
            if len(malicious_pool) > n_samples:
                import random
                random.seed(42)  # For reproducibility
                malicious_samples = random.sample(malicious_pool, n_samples)
            else:
                malicious_samples = malicious_pool[:n_samples]
            
            for sample in malicious_samples:
                samples.append({
                    "txt": sample.get("txt", ""),
                    "img": sample.get("img", None),
                    "toxicity": 1,
                    "experiment": "noisy_distribution"
                })
            
            print(f"  Loaded {len(malicious_samples)} malicious samples")
        except Exception as e:
            print(f"  Warning: Could not load AdvBench: {e}")
            # Fallback to DAN prompts
            try:
                malicious_pool = load_dan_prompts(max_samples=None)
                if len(malicious_pool) > n_samples:
                    import random
                    random.seed(42)
                    malicious_samples = random.sample(malicious_pool, n_samples)
                else:
                    malicious_samples = malicious_pool[:n_samples]
                
                for sample in malicious_samples:
                    samples.append({
                        "txt": sample.get("txt", ""),
                        "img": sample.get("img", None),
                        "toxicity": 1,
                        "experiment": "noisy_distribution"
                    })
                print(f"  Loaded {len(malicious_samples)} malicious samples")
            except Exception as e2:
                print(f"  Error loading malicious samples: {e2}")
                raise RuntimeError("Failed to load malicious samples for noisy distribution test")
        
        benign_count = sum(1 for s in samples if s['toxicity'] == 0)
        malicious_count = sum(1 for s in samples if s['toxicity'] == 1)
        print(f"  Total: {len(samples)} samples ({benign_count} benign, {malicious_count} malicious)")
        
        return samples
    
    def load_latent_neighbor_dataset(self, n_pairs: int = 100) -> List[Dict]:
        """
        Experiment 2: Create synthetic pairs using embedding similarity.
        For each malicious prompt, find the nearest neighbor in the benign pool.
        
        Args:
            n_pairs: Number of pairs to create (default: 100)
        
        Returns:
            List of samples with synthetic pairs based on embedding similarity
        """
        print("Creating synthetic pairs via embedding similarity...")
        
        # Load malicious prompts
        try:
            malicious_pool = load_advbench(max_samples=None)
            if len(malicious_pool) > n_pairs:
                import random
                random.seed(42)
                malicious_samples = random.sample(malicious_pool, n_pairs)
            else:
                malicious_samples = malicious_pool[:n_pairs]
            print(f"  Loaded {len(malicious_samples)} malicious prompts from AdvBench")
        except Exception as e:
            print(f"  Warning: Could not load AdvBench: {e}")
            try:
                malicious_pool = load_dan_prompts(max_samples=None)
                if len(malicious_pool) > n_pairs:
                    import random
                    random.seed(42)
                    malicious_samples = random.sample(malicious_pool, n_pairs)
                else:
                    malicious_samples = malicious_pool[:n_pairs]
                print(f"  Loaded {len(malicious_samples)} malicious prompts from DAN Prompts")
            except Exception as e2:
                raise RuntimeError(f"Failed to load malicious samples: {e2}")
        
        # Load large pool of benign prompts for nearest neighbor search
        try:
            benign_pool = load_alpaca(max_samples=None)
            if len(benign_pool) < n_pairs:
                # Add more benign sources if needed
                try:
                    openassistant_pool = load_openassistant(max_samples=None)
                    benign_pool.extend(openassistant_pool)
                except:
                    pass
            print(f"  Loaded {len(benign_pool)} benign prompts for nearest neighbor search")
        except Exception as e:
            print(f"  Warning: Could not load Alpaca: {e}")
            try:
                benign_pool = load_openassistant(max_samples=None)
                print(f"  Loaded {len(benign_pool)} benign prompts from OpenAssistant")
            except Exception as e2:
                raise RuntimeError(f"Failed to load benign pool: {e2}")
        
        # Use SentenceTransformer for embedding
        print("  Computing embeddings for nearest neighbor matching...")
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            print(f"  Error: Failed to import sentence_transformers: {e}")
            print("  Please install sentence-transformers: pip install sentence-transformers")
            raise RuntimeError(f"Failed to import sentence_transformers: {e}")
        
        try:
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("  Using sentence-transformers for embedding")
        except Exception as e:
            print(f"  Error: Failed to initialize SentenceTransformer: {e}")
            print("  This might be due to:")
            print("    1. Network connectivity issues (model download required)")
            print("    2. Insufficient disk space")
            print("    3. Corrupted model cache")
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}")
        
        samples = []
        
        # Extract text from malicious samples
        malicious_texts = [s.get("txt", "") for s in malicious_samples]
        benign_texts = [s.get("txt", "") for s in benign_pool]
        
        try:
            # Use sentence transformers
            print("  Encoding malicious prompts...")
            malicious_embeddings = embedder.encode(malicious_texts, show_progress_bar=False)
            print("  Encoding benign pool...")
            benign_embeddings = embedder.encode(benign_texts, show_progress_bar=False)
            
            # Find nearest neighbors
            from sklearn.metrics.pairwise import cosine_similarity
            print("  Finding nearest neighbors...")
            similarities = cosine_similarity(malicious_embeddings, benign_embeddings)
            
            for i, malicious_sample in enumerate(malicious_samples):
                # Find the most similar benign prompt
                nearest_idx = np.argmax(similarities[i])
                nearest_benign = benign_pool[nearest_idx]
                similarity_score = similarities[i][nearest_idx]
                
                # Add malicious sample
                samples.append({
                    "txt": malicious_sample.get("txt", ""),
                    "img": malicious_sample.get("img", None),
                    "toxicity": 1,
                    "experiment": "latent_neighbor",
                    "pair_similarity": float(similarity_score)
                })
                
                # Add matched benign sample
                samples.append({
                    "txt": nearest_benign.get("txt", ""),
                    "img": nearest_benign.get("img", None),
                    "toxicity": 0,
                    "experiment": "latent_neighbor",
                    "pair_similarity": float(similarity_score)
                })
        except Exception as e:
            print(f"  Error: Failed during embedding computation: {e}")
            print("  This might be due to:")
            print("    1. Memory issues (dataset too large)")
            print("    2. Invalid text inputs")
            raise RuntimeError(f"Failed during embedding computation: {e}")
        
        benign_count = sum(1 for s in samples if s['toxicity'] == 0)
        malicious_count = sum(1 for s in samples if s['toxicity'] == 1)
        avg_similarity = np.mean([s.get('pair_similarity', 0) for s in samples if 'pair_similarity' in s])
        
        print(f"  Total: {len(samples)} samples ({benign_count} benign, {malicious_count} malicious)")
        print(f"  Average pair similarity: {avg_similarity:.4f}")
        
        return samples
    
    def load_in_the_wild_dataset(self) -> List[Dict]:
        """
        Experiment 3: Use the actual training split from balanced dataset configuration.
        This is a mixture of multimodal and text-only, unpaired samples.
        
        Returns:
            List of samples from the actual training data used in experiments
        """
        print("Loading actual training split...")
        
        samples = []
        
        # Load balanced training data (same as in balanced_ood_kcd.py)
        # Benign: Alpaca (500) + MM-Vet (218) + OpenAssistant (282)
        # Malicious: AdvBench (300) + JailbreakV-28K (550) + DAN (150)
        
        # Benign training data
        try:
            alpaca_samples = load_alpaca(max_samples=500)
            for sample in alpaca_samples:
                samples.append({
                    "txt": sample.get("txt", ""),
                    "img": sample.get("img", None),
                    "toxicity": 0,
                    "experiment": "in_the_wild",
                    "source": "Alpaca"
                })
            print(f"  Loaded {len(alpaca_samples)} Alpaca samples")
        except Exception as e:
            print(f"  Warning: Could not load Alpaca: {e}")
        
        try:
            mmvet_samples = load_mm_vet()
            mmvet_benign = [s for s in mmvet_samples if s.get('toxicity', 0) == 0][:218]
            for sample in mmvet_benign:
                samples.append({
                    "txt": sample.get("txt", ""),
                    "img": sample.get("img", None),
                    "toxicity": 0,
                    "experiment": "in_the_wild",
                    "source": "MM-Vet"
                })
            print(f"  Loaded {len(mmvet_benign)} MM-Vet samples")
        except Exception as e:
            print(f"  Warning: Could not load MM-Vet: {e}")
        
        try:
            openassistant_samples = load_openassistant(max_samples=282)
            for sample in openassistant_samples:
                samples.append({
                    "txt": sample.get("txt", ""),
                    "img": sample.get("img", None),
                    "toxicity": 0,
                    "experiment": "in_the_wild",
                    "source": "OpenAssistant"
                })
            print(f"  Loaded {len(openassistant_samples)} OpenAssistant samples")
        except Exception as e:
            print(f"  Warning: Could not load OpenAssistant: {e}")
        
        # Malicious training data
        try:
            advbench_samples = load_advbench(max_samples=300)
            for sample in advbench_samples:
                samples.append({
                    "txt": sample.get("txt", ""),
                    "img": sample.get("img", None),
                    "toxicity": 1,
                    "experiment": "in_the_wild",
                    "source": "AdvBench"
                })
            print(f"  Loaded {len(advbench_samples)} AdvBench samples")
        except Exception as e:
            print(f"  Warning: Could not load AdvBench: {e}")
        
        try:
            # JailbreakV-28K: use llm_transfer_attack and query_related for training
            from load_datasets import load_JailBreakV_custom
            llm_attack = load_JailBreakV_custom(attack_types=["llm_transfer_attack"], max_samples=275)
            query_related = load_JailBreakV_custom(attack_types=["query_related"], max_samples=275)
            jbv_samples = (llm_attack or []) + (query_related or [])
            for sample in jbv_samples[:550]:  # Limit to 550 total
                samples.append({
                    "txt": sample.get("txt", ""),
                    "img": sample.get("img", None),
                    "toxicity": 1,
                    "experiment": "in_the_wild",
                    "source": "JailbreakV-28K"
                })
            print(f"  Loaded {min(550, len(jbv_samples))} JailbreakV-28K samples")
        except Exception as e:
            print(f"  Warning: Could not load JailbreakV-28K: {e}")
        
        try:
            dan_samples = load_dan_prompts(max_samples=150)
            for sample in dan_samples:
                samples.append({
                    "txt": sample.get("txt", ""),
                    "img": sample.get("img", None),
                    "toxicity": 1,
                    "experiment": "in_the_wild",
                    "source": "DAN"
                })
            print(f"  Loaded {len(dan_samples)} DAN samples")
        except Exception as e:
            print(f"  Warning: Could not load DAN Prompts: {e}")
        
        benign_count = sum(1 for s in samples if s['toxicity'] == 0)
        malicious_count = sum(1 for s in samples if s['toxicity'] == 1)
        multimodal_count = sum(1 for s in samples if s.get('img') is not None)
        text_only_count = len(samples) - multimodal_count
        
        print(f"  Total: {len(samples)} samples ({benign_count} benign, {malicious_count} malicious)")
        print(f"  Modality: {multimodal_count} multimodal, {text_only_count} text-only")
        
        return samples
    
    def compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD) between two distributions using RBF kernel.
        Uses the same approach as domain_shift_analysis.py for consistency.

        Args:
            X, Y: Feature matrices for two distributions
            gamma: RBF kernel parameter (if None, uses 1/n_features as in domain_shift_analysis.py)

        Returns:
            MMD value (higher = more different distributions)
        """
        from sklearn.metrics.pairwise import rbf_kernel

        X = np.array(X)
        Y = np.array(Y)

        # Use automatic gamma selection like in domain_shift_analysis.py
        if gamma is None:
            gamma = 1.0 / X.shape[1]  # Default: 1/n_features for RBF

        # Debug: Print gamma value for first few calls
        # if not hasattr(self, '_mmd_debug_count'):
        #     self._mmd_debug_count = 0
        # if self._mmd_debug_count < 3:
        #     print(f"MMD Debug: Using gamma={gamma:.6f} for features with {X.shape[1]} dimensions")
        #     self._mmd_debug_count += 1

        # Compute RBF kernel matrices using sklearn for consistency
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)

        # Use unbiased MMD estimator (exclude diagonal elements) like in domain_shift_analysis.py
        n_X, n_Y = X.shape[0], Y.shape[0]

        # E[k(x,x')] - exclude diagonal for unbiased estimate
        term1 = (np.sum(K_XX) - np.trace(K_XX)) / (n_X * (n_X - 1)) if n_X > 1 else 0

        # E[k(y,y')] - exclude diagonal for unbiased estimate
        term2 = (np.sum(K_YY) - np.trace(K_YY)) / (n_Y * (n_Y - 1)) if n_Y > 1 else 0

        # E[k(x,y)]
        term3 = np.sum(K_XY) / (n_X * n_Y)

        # MMD² = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        mmd_squared = term1 + term2 - 2 * term3

        # Return MMD (not squared) and ensure non-negative
        return np.sqrt(max(0, mmd_squared))

    def compute_bootstrap_ci(self, X: np.ndarray, Y: np.ndarray, metric_func, n_bootstrap: int = 100, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a metric.

        Args:
            X, Y: Feature matrices for two distributions
            metric_func: Function to compute the metric
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Tuple of (mean_score, lower_ci, upper_ci)
        """
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Bootstrap sample from both distributions
            X_boot = resample(X, n_samples=len(X), random_state=None)
            Y_boot = resample(Y, n_samples=len(Y), random_state=None)

            try:
                score = metric_func(X_boot, Y_boot)
                bootstrap_scores.append(score)
            except:
                continue  # Skip failed bootstrap samples

        if not bootstrap_scores:
            return 0.0, 0.0, 0.0

        bootstrap_scores = np.array(bootstrap_scores)
        mean_score = np.mean(bootstrap_scores)

        # Compute confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_ci = np.percentile(bootstrap_scores, lower_percentile)
        upper_ci = np.percentile(bootstrap_scores, upper_percentile)

        return float(mean_score), float(lower_ci), float(upper_ci)

    def compute_wasserstein_distance(self, X: np.ndarray, Y: np.ndarray, n_projections: int = 50) -> float:
        """
        Compute sliced Wasserstein distance for better high-dimensional handling.

        Args:
            X, Y: Feature matrices for two distributions
            n_projections: Number of random projections for sliced Wasserstein

        Returns:
            Sliced Wasserstein distance (higher = more different distributions)
        """
        from scipy.stats import wasserstein_distance

        try:
            # For low-dimensional data, use direct computation
            if X.shape[1] <= 2:
                if X.shape[1] == 1:
                    return wasserstein_distance(X.flatten(), Y.flatten())
                else:
                    # For 2D, average over both dimensions
                    w1 = wasserstein_distance(X[:, 0], Y[:, 0])
                    w2 = wasserstein_distance(X[:, 1], Y[:, 1])
                    return (w1 + w2) / 2

            # For high-dimensional data, use sliced Wasserstein distance
            distances = []

            for _ in range(n_projections):
                # Generate random unit vector
                theta = np.random.randn(X.shape[1])
                theta = theta / np.linalg.norm(theta)

                # Project data onto this direction
                X_proj = X @ theta
                Y_proj = Y @ theta

                # Compute 1D Wasserstein distance
                distances.append(wasserstein_distance(X_proj, Y_proj))

            # Return average sliced Wasserstein distance
            return float(np.mean(distances))

        except Exception as e:
            print(f"Error computing Wasserstein distance: {e}")
            raise RuntimeError(f"Failed to compute Wasserstein distance: {e}")
    
    def compute_kl_divergence(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions using KDE.
        JS divergence is symmetric and more stable than KL divergence.

        Args:
            X, Y: Feature matrices for two distributions

        Returns:
            JS divergence (higher = more different distributions)
        """
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KernelDensity

        try:
            # Project to 2D using PCA for better representation while keeping computational efficiency
            combined = np.vstack([X, Y])
            n_components = min(2, X.shape[1])  # Use 2D or less if features < 2
            pca = PCA(n_components=n_components)
            combined_proj = pca.fit_transform(combined)

            X_proj = combined_proj[:len(X)]
            Y_proj = combined_proj[len(X):]

            # Use KDE for density estimation with automatic bandwidth selection
            kde_X = KernelDensity(kernel='gaussian', bandwidth='scott').fit(X_proj)
            kde_Y = KernelDensity(kernel='gaussian', bandwidth='scott').fit(Y_proj)

            # Create evaluation grid
            x_min = np.minimum(X_proj.min(axis=0), Y_proj.min(axis=0))
            x_max = np.maximum(X_proj.max(axis=0), Y_proj.max(axis=0))

            if n_components == 1:
                eval_points = np.linspace(x_min, x_max, 100).reshape(-1, 1)
            else:
                # Create 2D grid
                x1 = np.linspace(x_min[0], x_max[0], 50)
                x2 = np.linspace(x_min[1], x_max[1], 50)
                X1, X2 = np.meshgrid(x1, x2)
                eval_points = np.column_stack([X1.ravel(), X2.ravel()])

            # Evaluate densities
            log_dens_X = kde_X.score_samples(eval_points)
            log_dens_Y = kde_Y.score_samples(eval_points)

            # Convert to probabilities and normalize
            dens_X = np.exp(log_dens_X)
            dens_Y = np.exp(log_dens_Y)

            dens_X = dens_X / np.sum(dens_X)
            dens_Y = dens_Y / np.sum(dens_Y)

            # Compute Jensen-Shannon divergence
            # JS(P,Q) = 0.5 * KL(P,M) + 0.5 * KL(Q,M) where M = 0.5*(P+Q)
            M = 0.5 * (dens_X + dens_Y)

            # Add small epsilon for numerical stability
            epsilon = 1e-10
            dens_X = np.maximum(dens_X, epsilon)
            dens_Y = np.maximum(dens_Y, epsilon)
            M = np.maximum(M, epsilon)

            kl_pm = np.sum(dens_X * np.log(dens_X / M))
            kl_qm = np.sum(dens_Y * np.log(dens_Y / M))

            js_divergence = 0.5 * kl_pm + 0.5 * kl_qm

            return float(js_divergence)

        except Exception as e:
            print(f"Error computing JS divergence: {e}")
            raise RuntimeError(f"Failed to compute JS divergence: {e}")
    
    def compute_svm_margin(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute SVM margin width for linear separability assessment.
        
        Args:
            X: Feature matrix
            y: Binary labels
            
        Returns:
            Margin width (higher = more linearly separable)
        """
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train linear SVM
            svm = SVC(kernel='linear', C=1.0)
            svm.fit(X_scaled, y)
            
            # Margin width = 2 / ||w||
            margin = 2.0 / np.linalg.norm(svm.coef_)
            return margin
            
        except Exception as e:
            print(f"Error computing SVM margin: {e}")
            raise RuntimeError(f"Failed to compute SVM margin: {e}")
    
    def compute_silhouette_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute average silhouette coefficient for clustering quality.
        
        Args:
            X: Feature matrix
            y: Binary labels
            
        Returns:
            Silhouette score (higher = better natural clustering)
        """
        try:
            if len(np.unique(y)) < 2:
                return 0.0
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            return silhouette_score(X_scaled, y)
            
        except Exception as e:
            print(f"Error computing silhouette score: {e}")
            raise RuntimeError(f"Failed to compute silhouette score: {e}")

    def compute_distance_ratio(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute inter-class vs intra-class distance ratio.

        Args:
            X: Feature matrix
            y: Binary labels

        Returns:
            Distance ratio (higher = better class separation)
        """
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Separate classes
            X_class0 = X_scaled[y == 0]
            X_class1 = X_scaled[y == 1]

            if len(X_class0) == 0 or len(X_class1) == 0:
                return 0.0

            # Compute centroids
            centroid0 = np.mean(X_class0, axis=0)
            centroid1 = np.mean(X_class1, axis=0)

            # Inter-class distance (distance between centroids)
            inter_class_dist = np.linalg.norm(centroid1 - centroid0)

            # Intra-class distances (average distance within each class)
            intra_class_dist0 = np.mean(pdist(X_class0)) if len(X_class0) > 1 else 0
            intra_class_dist1 = np.mean(pdist(X_class1)) if len(X_class1) > 1 else 0
            avg_intra_class_dist = (intra_class_dist0 + intra_class_dist1) / 2

            # Avoid division by zero
            if avg_intra_class_dist == 0:
                return float('inf') if inter_class_dist > 0 else 0.0

            return inter_class_dist / avg_intra_class_dist

        except Exception as e:
            print(f"Error computing distance ratio: {e}")
            raise RuntimeError(f"Failed to compute distance ratio: {e}")

    def compute_mutual_information(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information between features and labels.

        Args:
            X: Feature matrix
            y: Binary labels

        Returns:
            Mutual information (higher = more informative features)
        """
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Compute mutual information
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            return np.mean(mi_scores)

        except Exception as e:
            print(f"Error computing mutual information: {e}")
            raise RuntimeError(f"Failed to compute mutual information: {e}")

    def compute_conditional_entropy_reduction(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute conditional entropy reduction (information gain).

        Args:
            X: Feature matrix
            y: Binary labels

        Returns:
            Entropy reduction (higher = more informative features)
        """
        try:
            # Base entropy H(Y)
            _, counts = np.unique(y, return_counts=True)
            base_entropy = entropy(counts / len(y), base=2)

            # Use PCA to reduce to 1D for entropy estimation
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            X_proj = pca.fit_transform(X).flatten()

            # Discretize features into bins
            n_bins = min(10, len(np.unique(X_proj)))
            X_binned = pd.cut(X_proj, bins=n_bins, labels=False)

            # Compute conditional entropy H(Y|X)
            conditional_entropy = 0.0
            for bin_val in np.unique(X_binned):
                if pd.isna(bin_val):
                    continue

                mask = (X_binned == bin_val)
                if np.sum(mask) == 0:
                    continue

                y_subset = y[mask]
                _, counts_subset = np.unique(y_subset, return_counts=True)

                if len(counts_subset) > 0:
                    prob_bin = np.sum(mask) / len(y)
                    entropy_bin = entropy(counts_subset / len(y_subset), base=2)
                    conditional_entropy += prob_bin * entropy_bin

            # Information gain = H(Y) - H(Y|X)
            return base_entropy - conditional_entropy

        except Exception as e:
            print(f"Error computing conditional entropy reduction: {e}")
            raise RuntimeError(f"Failed to compute conditional entropy reduction: {e}")

    def analyze_layer(self, layer_idx: int, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Perform comprehensive discriminative analysis for a single layer.

        Args:
            layer_idx: Layer index
            features: Feature matrix for this layer
            labels: Binary labels (0=benign, 1=malicious)

        Returns:
            Dictionary of discriminative metrics
        """
        # Separate benign and malicious features
        benign_features = features[labels == 0]
        malicious_features = features[labels == 1]

        if len(benign_features) == 0 or len(malicious_features) == 0:
            return {metric: 0.0 for metric in [
                'mmd', 'wasserstein', 'kl_divergence', 'svm_margin',
                'silhouette', 'distance_ratio', 'mutual_info', 'entropy_reduction'
            ]}

        results = {}

        # 1. Distributional Divergence Metrics
        results['mmd'] = self.compute_mmd(benign_features, malicious_features)
        results['wasserstein'] = self.compute_wasserstein_distance(benign_features, malicious_features)
        results['kl_divergence'] = self.compute_kl_divergence(benign_features, malicious_features)

        # 2. Geometric Separation Analysis
        results['svm_margin'] = self.compute_svm_margin(features, labels)
        results['silhouette'] = self.compute_silhouette_score(features, labels)
        results['distance_ratio'] = self.compute_distance_ratio(features, labels)

        # 3. Information-Theoretic Measures
        results['mutual_info'] = self.compute_mutual_information(features, labels)
        results['entropy_reduction'] = self.compute_conditional_entropy_reduction(features, labels)

        return results

    def run_comprehensive_analysis(self, dataset: List[Dict], layer_start: int = 0, layer_end: int = 31, 
                                   dataset_name: str = None, experiment_name: str = None) -> Dict:
        """
        Run comprehensive discriminative analysis across all specified layers.

        Args:
            dataset: List of samples with paired benign/malicious prompts
            layer_start: Starting layer index
            layer_end: Ending layer index
            dataset_name: Unique dataset name for caching (default: auto-generated)
            experiment_name: Unique experiment name for caching (default: "principled_layer_selection")

        Returns:
            Dictionary containing analysis results for all layers
        """
        # Use unique identifiers for caching
        if dataset_name is None:
            dataset_name = f"layer_selection_{len(dataset)}samples"
        if experiment_name is None:
            experiment_name = "principled_layer_selection"
        
        # Extract hidden states for all layers
        hidden_states_dict, labels, _ = self.extractor.extract_hidden_states(
            dataset, dataset_name,
            layer_start=layer_start, layer_end=layer_end,
            use_cache=True, experiment_name=experiment_name
        )

        # Analyze each layer
        layer_results = {}

        for layer_idx in range(layer_start, layer_end + 1):
            features = np.array(hidden_states_dict[layer_idx])
            layer_results[layer_idx] = self.analyze_layer(layer_idx, features, np.array(labels))

        self.results = layer_results
        return layer_results

    def compute_composite_scores(self, results: Dict) -> Dict[int, Dict[str, float]]:
        """
        Compute composite discriminative scores for each layer.

        Args:
            results: Layer analysis results

        Returns:
            Dictionary with composite scores for each layer
        """
        composite_scores = {}

        # Define metric weights (can be adjusted based on importance)
        weights = {
            # Distributional divergence
            'mmd': 0.10,
            'wasserstein': 0.10,
            'kl_divergence': 0.10,

            # Geometric separation 
            'svm_margin': 0.10,
            'silhouette': 0.10,
            'distance_ratio': 0.10,

            # Information-theoretic
            'mutual_info': 0.15,
            'entropy_reduction': 0.15
        }

        # Normalize each metric across layers for fair comparison
        all_layers = list(results.keys())
        normalized_results = {}

        for metric in weights.keys():
            metric_values = [results[layer][metric] for layer in all_layers]

            # Handle edge cases
            if all(v == 0 for v in metric_values):
                normalized_values = [0.0] * len(metric_values)
            else:
                # Use robust normalization (median/IQR) for better outlier handling
                metric_array = np.array(metric_values)
                median_val = np.median(metric_array)
                q75, q25 = np.percentile(metric_array, [75, 25])
                iqr = q75 - q25

                if iqr == 0:
                    # Fallback to min-max if no variance
                    min_val = min(metric_values)
                    max_val = max(metric_values)
                    if max_val == min_val:
                        normalized_values = [1.0] * len(metric_values)
                    else:
                        normalized_values = [(v - min_val) / (max_val - min_val) for v in metric_values]
                else:
                    # Robust scaling: (x - median) / IQR, then map to [0,1]
                    robust_scaled = [(v - median_val) / iqr for v in metric_values]
                    # Map to [0,1] using sigmoid-like transformation
                    normalized_values = [1 / (1 + np.exp(-2 * rs)) for rs in robust_scaled]

            for i, layer in enumerate(all_layers):
                if layer not in normalized_results:
                    normalized_results[layer] = {}
                normalized_results[layer][metric] = normalized_values[i]

        # Compute composite scores
        for layer in all_layers:
            # Individual category scores
            distributional_score = (
                weights['mmd'] * normalized_results[layer]['mmd'] +
                weights['wasserstein'] * normalized_results[layer]['wasserstein'] +
                weights['kl_divergence'] * normalized_results[layer]['kl_divergence']
            ) / (weights['mmd'] + weights['wasserstein'] + weights['kl_divergence'])

            geometric_score = (
                weights['svm_margin'] * normalized_results[layer]['svm_margin'] +
                weights['silhouette'] * normalized_results[layer]['silhouette'] +
                weights['distance_ratio'] * normalized_results[layer]['distance_ratio']
            ) / (weights['svm_margin'] + weights['silhouette'] + weights['distance_ratio'])

            information_score = (
                weights['mutual_info'] * normalized_results[layer]['mutual_info'] +
                weights['entropy_reduction'] * normalized_results[layer]['entropy_reduction']
            ) / (weights['mutual_info'] + weights['entropy_reduction'])

            # Overall composite score
            overall_score = (
                distributional_score * 0.4 +
                geometric_score * 0.4 +
                information_score * 0.3
            )

            composite_scores[layer] = {
                'distributional_score': distributional_score,
                'geometric_score': geometric_score,
                'information_score': information_score,
                'overall_score': overall_score,
                **normalized_results[layer]  # Include normalized individual metrics
            }

        return composite_scores

    def generate_layer_ranking(self, composite_scores: Dict) -> List[Tuple[int, float, Dict]]:
        """
        Generate final layer ranking based on composite scores.

        Args:
            composite_scores: Composite scores for each layer

        Returns:
            List of (layer_idx, overall_score, detailed_scores) tuples, sorted by score
        """
        ranking = []
        for layer_idx, scores in composite_scores.items():
            ranking.append((layer_idx, scores['overall_score'], scores))

        # Sort by overall score (descending)
        ranking.sort(key=lambda x: x[1], reverse=True)

        return ranking

    def save_results(self, ranking: List[Tuple], output_path: str = None):
        """
        Save detailed results to CSV file.

        Args:
            ranking: Layer ranking results
            output_path: Output CSV file path (if None, auto-generate based on model)
        """
        if output_path is None:
            # Generate model-specific filename
            model_name = self._get_model_name_for_filename()
            output_path = f"results/principled_layer_selection_results_{model_name}.csv"

        # Prepare data for CSV
        csv_data = []
        for rank, (layer_idx, overall_score, detailed_scores) in enumerate(ranking, 1):
            row = {
                'Rank': rank,
                'Layer': layer_idx,
                'Overall_Score': f"{overall_score:.4f}",
                'Distributional_Score': f"{detailed_scores['distributional_score']:.4f}",
                'Geometric_Score': f"{detailed_scores['geometric_score']:.4f}",
                'Information_Score': f"{detailed_scores['information_score']:.4f}",
                'MMD': f"{detailed_scores['mmd']:.4f}",
                'Wasserstein': f"{detailed_scores['wasserstein']:.4f}",
                'KL_Divergence': f"{detailed_scores['kl_divergence']:.4f}",
                'SVM_Margin': f"{detailed_scores['svm_margin']:.4f}",
                'Silhouette': f"{detailed_scores['silhouette']:.4f}",
                'Distance_Ratio': f"{detailed_scores['distance_ratio']:.4f}",
                'Mutual_Info': f"{detailed_scores['mutual_info']:.4f}",
                'Entropy_Reduction': f"{detailed_scores['entropy_reduction']:.4f}"
            }
            csv_data.append(row)

        # Save to CSV
        os.makedirs('results', exist_ok=True)
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)

    def print_summary_report(self, ranking: List[Tuple]):
        """
        Print a comprehensive summary report of the layer analysis.

        Args:
            ranking: Layer ranking results
        """
        print("\n" + "="*100)
        print("PRINCIPLED LAYER SELECTION FOR JAILBREAK DETECTION - COMPREHENSIVE REPORT")
        print("="*100)

        print(f"\nDataset: SGXSTest (100 paired prompts - benign vs malicious)")
        print(f"Model: {self.model_path}")
        print(f"Analysis Method: Multi-metric discriminative power assessment")

        print(f"\nMETRICS EVALUATED:")
        print(f"  Distributional Divergence (40% weight):")
        print(f"    • Maximum Mean Discrepancy (MMD)")
        print(f"    • Wasserstein Distance")
        print(f"    • KL Divergence")
        print(f"  Geometric Separation (40% weight):")
        print(f"    • SVM Margin Width")
        print(f"    • Silhouette Coefficient")
        print(f"    • Inter/Intra-class Distance Ratio")
        print(f"  Information-Theoretic (20% weight):")
        print(f"    • Mutual Information")
        print(f"    • Conditional Entropy Reduction")

        print(f"\n" + "-"*100)
        print(f"TOP 10 LAYERS FOR JAILBREAK DETECTION")
        print(f"-"*100)
        print(f"{'Rank':<4} {'Layer':<5} {'Overall':<8} {'Distrib':<8} {'Geometric':<9} {'Info':<8} {'Recommendation'}")
        print(f"-"*100)

        for rank, (layer_idx, overall_score, detailed_scores) in enumerate(ranking[:10], 1):
            dist_score = detailed_scores['distributional_score']
            geom_score = detailed_scores['geometric_score']
            info_score = detailed_scores['information_score']

            # Generate recommendation
            if overall_score > 0.8:
                recommendation = "Excellent"
            elif overall_score > 0.6:
                recommendation = "Good"
            elif overall_score > 0.4:
                recommendation = "Fair"
            else:
                recommendation = "Poor"

            print(f"{rank:<4} {layer_idx:<5} {overall_score:.3f}    {dist_score:.3f}    {geom_score:.3f}     {info_score:.3f}    {recommendation}")

        # Best layer analysis
        best_layer, best_score, best_details = ranking[0]
        print(f"\n" + "="*100)
        print(f"RECOMMENDED LAYER: {best_layer}")
        print(f"="*100)
        print(f"Overall Discriminative Score: {best_score:.4f}")
        print(f"")
        print(f"Detailed Breakdown:")
        print(f"  Distributional Divergence Score: {best_details['distributional_score']:.4f}")
        print(f"    • MMD: {best_details['mmd']:.4f}")
        print(f"    • Wasserstein Distance: {best_details['wasserstein']:.4f}")
        print(f"    • KL Divergence: {best_details['kl_divergence']:.4f}")
        print(f"")
        print(f"  Geometric Separation Score: {best_details['geometric_score']:.4f}")
        print(f"    • SVM Margin: {best_details['svm_margin']:.4f}")
        print(f"    • Silhouette Coefficient: {best_details['silhouette']:.4f}")
        print(f"    • Distance Ratio: {best_details['distance_ratio']:.4f}")
        print(f"")
        print(f"  Information-Theoretic Score: {best_details['information_score']:.4f}")
        print(f"    • Mutual Information: {best_details['mutual_info']:.4f}")
        print(f"    • Entropy Reduction: {best_details['entropy_reduction']:.4f}")

        print(f"\n" + "="*100)
        print(f"USAGE RECOMMENDATIONS")
        print(f"="*100)
        print(f"1. PRIMARY CHOICE: Use Layer {best_layer} for jailbreak detection")
        print(f"   - Highest overall discriminative power ({best_score:.4f})")
        print(f"   - Best balance across all evaluation metrics")
        print(f"")

        # Alternative recommendations
        if len(ranking) > 1:
            second_layer, second_score, _ = ranking[1]
            print(f"2. ALTERNATIVE: Layer {second_layer} (score: {second_score:.4f})")

        if len(ranking) > 2:
            third_layer, third_score, _ = ranking[2]
            print(f"3. BACKUP OPTION: Layer {third_layer} (score: {third_score:.4f})")

        print(f"")
        print(f"4. ENSEMBLE APPROACH: Consider combining top 3-5 layers for robust detection")
        print(f"5. VALIDATION: Test selected layer(s) on your specific jailbreak detection task")
        print(f"")
        print(f"This analysis provides a principled, data-driven foundation for layer selection")
        print(f"based on comprehensive discriminative power assessment.")
        print(f"="*100)

    def print_detailed_rankings(self, composite_scores: Dict):
        """
        Print detailed rankings for all metrics and categories.

        Args:
            composite_scores: Composite scores for each layer
        """
        print("\n" + "="*120)
        print("DETAILED LAYER RANKINGS BY METRIC AND CATEGORY")
        print("="*120)

        layers = list(composite_scores.keys())

        # 1. Category Rankings
        print(f"\n{'='*40} CATEGORY RANKINGS {'='*40}")

        categories = [
            ('distributional_score', 'DISTRIBUTIONAL DIVERGENCE'),
            ('geometric_score', 'GEOMETRIC SEPARATION'),
            ('information_score', 'INFORMATION-THEORETIC')
        ]

        for score_key, category_name in categories:
            print(f"\n{category_name} RANKING:")
            print(f"{'Rank':<4} {'Layer':<5} {'Score':<8} {'Percentile':<10}")
            print("-" * 35)

            # Sort layers by this category score
            category_ranking = [(layer, composite_scores[layer][score_key]) for layer in layers]
            category_ranking.sort(key=lambda x: x[1], reverse=True)

            for rank, (layer, score) in enumerate(category_ranking[:10], 1):
                percentile = (len(layers) - rank + 1) / len(layers) * 100
                print(f"{rank:<4} {layer:<5} {score:.4f}   {percentile:.1f}%")

        # 2. Individual Metric Rankings
        print(f"\n{'='*40} INDIVIDUAL METRIC RANKINGS {'='*40}")

        metrics = [
            ('mmd', 'Maximum Mean Discrepancy'),
            ('wasserstein', 'Wasserstein Distance'),
            ('kl_divergence', 'KL Divergence'),
            ('svm_margin', 'SVM Margin Width'),
            ('silhouette', 'Silhouette Coefficient'),
            ('distance_ratio', 'Distance Ratio'),
            ('mutual_info', 'Mutual Information'),
            ('entropy_reduction', 'Entropy Reduction')
        ]

        for metric_key, metric_name in metrics:
            print(f"\n{metric_name.upper()} RANKING:")
            print(f"{'Rank':<4} {'Layer':<5} {'Score':<8} {'Category':<15}")
            print("-" * 40)

            # Sort layers by this metric
            metric_ranking = [(layer, composite_scores[layer][metric_key]) for layer in layers]
            metric_ranking.sort(key=lambda x: x[1], reverse=True)

            # Determine category for context
            category_map = {
                'mmd': 'Distributional',
                'wasserstein': 'Distributional',
                'kl_divergence': 'Distributional',
                'svm_margin': 'Geometric',
                'silhouette': 'Geometric',
                'distance_ratio': 'Geometric',
                'mutual_info': 'Information',
                'entropy_reduction': 'Information'
            }

            for rank, (layer, score) in enumerate(metric_ranking[:10], 1):
                category = category_map.get(metric_key, 'Unknown')
                print(f"{rank:<4} {layer:<5} {score:.4f}   {category:<15}")

        # 3. Cross-Metric Analysis
        print(f"\n{'='*40} CROSS-METRIC ANALYSIS {'='*40}")

        # Find layers that appear in top 5 for multiple metrics
        top_5_appearances = {}
        for metric_key, _ in metrics:
            metric_ranking = [(layer, composite_scores[layer][metric_key]) for layer in layers]
            metric_ranking.sort(key=lambda x: x[1], reverse=True)

            for rank, (layer, _) in enumerate(metric_ranking[:5], 1):
                if layer not in top_5_appearances:
                    top_5_appearances[layer] = []
                top_5_appearances[layer].append((metric_key, rank))

        # Sort by number of top-5 appearances
        consistent_layers = [(layer, appearances) for layer, appearances in top_5_appearances.items()]
        consistent_layers.sort(key=lambda x: len(x[1]), reverse=True)

        print(f"\nLAYERS WITH CONSISTENT HIGH PERFORMANCE (Top 5 in multiple metrics):")
        print(f"{'Layer':<5} {'Appearances':<12} {'Metrics (Rank)'}")
        print("-" * 60)

        for layer, appearances in consistent_layers[:10]:
            if len(appearances) >= 2:  # Only show layers that appear in top 5 of at least 2 metrics
                metrics_str = ", ".join([f"{metric}({rank})" for metric, rank in appearances[:4]])
                if len(appearances) > 4:
                    metrics_str += "..."
                print(f"{layer:<5} {len(appearances):<12} {metrics_str}")

        print(f"\n{'='*120}")
        print("INTERPRETATION GUIDE:")
        print("• Layers with high 'Appearances' are consistently good across multiple metrics")
        print("• Different metrics may favor different layers - this is expected")
        print("• The overall ranking balances all metrics with appropriate weights")
        print("• Consider top 3-5 layers for ensemble approaches")
        print("="*120)


def run_single_experiment(analyzer, exp_num: int, layer_start: int, layer_end: int):
    """
    Run a single experiment by number.
    
    Args:
        analyzer: LayerDiscriminativeAnalyzer instance
        exp_num: Experiment number (1-4)
        layer_start: Starting layer index
        layer_end: Ending layer index
    
    Returns:
        Dictionary with experiment results or None if failed
    """
    exp_configs = {
        1: {
            'name': 'sgxstest',
            'load_func': analyzer.load_sgxstest_dataset,
            'load_args': {}
        },
        2: {
            'name': 'noisy_distribution',
            'load_func': analyzer.load_noisy_distribution_dataset,
            'load_args': {'n_samples': 100}
        },
        3: {
            'name': 'latent_neighbor',
            'load_func': analyzer.load_latent_neighbor_dataset,
            'load_args': {'n_pairs': 100}
        },
        4: {
            'name': 'in_the_wild',
            'load_func': analyzer.load_in_the_wild_dataset,
            'load_args': {}
        },
        5: {
            'name': 'xstest',
            'load_func': analyzer.load_xstest_dataset,
            'load_args': {}
        }
    }
    
    if exp_num not in exp_configs:
        raise ValueError(f"Invalid experiment number: {exp_num}. Must be 1-5.")
    
    config = exp_configs[exp_num]
    exp_name = config['name']
    
    print(f"\nRunning Experiment {exp_num}: {exp_name.replace('_', ' ').title()}")
    
    try:
        # Load dataset
        dataset = config['load_func'](**config['load_args'])
        
        # Use unique experiment identifier for caching
        exp_name = config['name']
        dataset_name = f"exp{exp_num}_{exp_name}_{len(dataset)}samples"
        experiment_name = f"principled_layer_selection_exp{exp_num}"
        
        # Run analysis
        results = analyzer.run_comprehensive_analysis(
            dataset, layer_start=layer_start, layer_end=layer_end,
            dataset_name=dataset_name, experiment_name=experiment_name
        )
        composite_scores = analyzer.compute_composite_scores(results)
        ranking = analyzer.generate_layer_ranking(composite_scores)
        
        return {
            'dataset': dataset,
            'results': results,
            'composite_scores': composite_scores,
            'ranking': ranking,
            'exp_num': exp_num,
            'exp_name': exp_name
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def print_experiment_summary(exp_num: int, exp_name: str, ranking: List[Tuple]):
    """Print a brief summary for an experiment."""
    if not ranking:
        return
    best_layer, best_score, _ = ranking[0]
    print(f"  Experiment {exp_num} ({exp_name}): Best layer = {best_layer} (score: {best_score:.4f})")


def analyze_correlations(all_experiment_results):
    """
    Analyze correlations between different experiments.
    
    Args:
        all_experiment_results: List of experiment result dictionaries
    
    Returns:
        Dictionary mapping experiment names to layer scores
    """
    # Extract overall scores for each experiment
    experiment_scores = {}
    
    for exp_data in all_experiment_results:
        if exp_data is None:
            continue
        exp_name = exp_data['exp_name']
        composite_scores = exp_data['composite_scores']
        layer_scores = {}
        for layer_idx, scores in composite_scores.items():
            layer_scores[layer_idx] = scores['overall_score']
        experiment_scores[exp_name] = layer_scores
    
    if len(experiment_scores) < 2:
        return experiment_scores
    
    # Compute pairwise correlations
    from scipy.stats import spearmanr, pearsonr
    
    print("\nCorrelation Analysis:")
    print(f"{'Exp 1':<20} {'Exp 2':<20} {'Pearson r':<12} {'Spearman ρ':<12}")
    print("-" * 70)
    
    exp_names = list(experiment_scores.keys())
    for i, exp1 in enumerate(exp_names):
        for exp2 in exp_names[i+1:]:
            scores1 = [experiment_scores[exp1][layer] for layer in sorted(experiment_scores[exp1].keys())]
            scores2 = [experiment_scores[exp2][layer] for layer in sorted(experiment_scores[exp2].keys())]
            
            pearson_r, _ = pearsonr(scores1, scores2)
            spearman_r, _ = spearmanr(scores1, scores2)
            
            print(f"{exp1:<20} {exp2:<20} {pearson_r:>10.4f}   {spearman_r:>10.4f}")
    
    # Find top layers for each experiment
    print("\nTop 3 Layers by Experiment:")
    for exp_name, layer_scores in experiment_scores.items():
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_layers[:3]
        layer_str = ", ".join([f"L{layer}" for layer, _ in top_3])
        print(f"  {exp_name:<20} {layer_str}")
    
    return experiment_scores


def create_correlation_visualization(all_experiment_results, analyzer=None):
    """
    Create visualization comparing results across experiments.
    
    Args:
        all_experiment_results: List of experiment result dictionaries
        analyzer: LayerDiscriminativeAnalyzer instance for model naming
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract overall scores
        experiment_scores = {}
        for exp_data in all_experiment_results:
            if exp_data is None:
                continue
            exp_name = exp_data['exp_name']
            composite_scores = exp_data['composite_scores']
            layer_scores = {}
            for layer_idx, scores in composite_scores.items():
                layer_scores[layer_idx] = scores['overall_score']
            experiment_scores[exp_name] = layer_scores
        
        if len(experiment_scores) == 0:
            print("No experiment results available for visualization")
            return
        
        # Create figure with subplots
        n_experiments = len(experiment_scores)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Robustness Experiments: Layer Selection Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall scores by layer (all experiments)
        ax1 = axes[0, 0]
        for exp_name, layer_scores in experiment_scores.items():
            layers = sorted(layer_scores.keys())
            scores = [layer_scores[layer] for layer in layers]
            ax1.plot(layers, scores, 'o-', label=exp_name.replace('_', ' ').title(), linewidth=2, markersize=4)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Overall Discriminative Score')
        ax1.set_title('Overall Scores by Layer (All Experiments)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correlation heatmap
        ax2 = axes[0, 1]
        if len(experiment_scores) >= 2:
            from scipy.stats import pearsonr
            exp_names = list(experiment_scores.keys())
            corr_matrix = np.zeros((len(exp_names), len(exp_names)))
            
            for i, exp1 in enumerate(exp_names):
                for j, exp2 in enumerate(exp_names):
                    scores1 = [experiment_scores[exp1][layer] for layer in sorted(experiment_scores[exp1].keys())]
                    scores2 = [experiment_scores[exp2][layer] for layer in sorted(experiment_scores[exp2].keys())]
                    corr, _ = pearsonr(scores1, scores2)
                    corr_matrix[i, j] = corr
            
            im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            ax2.set_xticks(range(len(exp_names)))
            ax2.set_yticks(range(len(exp_names)))
            ax2.set_xticklabels([name.replace('_', ' ').title() for name in exp_names], rotation=45, ha='right')
            ax2.set_yticklabels([name.replace('_', ' ').title() for name in exp_names])
            ax2.set_title('Correlation Matrix (Pearson r)')
            
            # Add text annotations
            for i in range(len(exp_names)):
                for j in range(len(exp_names)):
                    text = ax2.text(j, i, f'{corr_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=ax2)
        
        # Plot 3: Top 10 layers comparison
        ax3 = axes[1, 0]
        top_n = 10
        for exp_name, layer_scores in experiment_scores.items():
            sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
            top_layers = [layer for layer, _ in sorted_layers[:top_n]]
            top_scores = [score for _, score in sorted_layers[:top_n]]
            
            x_pos = np.arange(len(top_layers))
            ax3.bar(x_pos - 0.2 * (list(experiment_scores.keys()).index(exp_name) - len(experiment_scores)/2 + 0.5),
                   top_scores, width=0.2, label=exp_name.replace('_', ' ').title(), alpha=0.7)
        
        ax3.set_xlabel('Rank Position')
        ax3.set_ylabel('Overall Score')
        ax3.set_title(f'Top {top_n} Layers Comparison')
        ax3.set_xticks(range(top_n))
        ax3.set_xticklabels([f'{i+1}' for i in range(top_n)])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Score distribution histograms
        ax4 = axes[1, 1]
        for exp_name, layer_scores in experiment_scores.items():
            scores = list(layer_scores.values())
            ax4.hist(scores, bins=15, alpha=0.5, label=exp_name.replace('_', ' ').title(), edgecolor='black')
        ax4.set_xlabel('Overall Discriminative Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Score Distribution Histograms')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('results', exist_ok=True)
        if analyzer:
            model_name = analyzer._get_model_name_for_filename()
            output_path = f"results/experiments_correlation_{model_name}.png"
        else:
            output_path = "results/experiments_correlation.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        
        # Try to show plot
        try:
            plt.show()
        except:
            pass
        
    except Exception as e:
        print(f"Error creating correlation visualization: {e}")
        import traceback
        traceback.print_exc()


def parse_experiment_numbers(experiment_arg: Optional[str]) -> List[int]:
    """
    Parse experiment numbers from command line argument.
    
    Args:
        experiment_arg: Comma-separated string of experiment numbers (e.g., "1,2,3")
    
    Returns:
        List of experiment numbers (1-5)
    """
    if experiment_arg is None:
        return [1, 2, 3, 4, 5]  # Default: run all experiments
    
    try:
        exp_nums = [int(x.strip()) for x in experiment_arg.split(',')]
        # Validate experiment numbers
        valid_nums = [n for n in exp_nums if 1 <= n <= 5]
        if not valid_nums:
            raise ValueError("No valid experiment numbers (must be 1-5)")
        return sorted(set(valid_nums))  # Remove duplicates and sort
    except ValueError as e:
        print(f"Error parsing experiment numbers: {e}")
        print("Use comma-separated numbers, e.g., '1,2,3' or '1,3,5'")
        sys.exit(1)


def save_experiment_results(exp_data: Dict, model_name: str):
    """Save experiment results to CSV file in the same format as the original."""
    if exp_data is None:
        return None
    
    ranking = exp_data['ranking']
    exp_name = exp_data['exp_name']
    exp_num = exp_data['exp_num']
    
    # Use naming scheme similar to original: principled_layer_selection_results_{exp_name}_{model_name}.csv
    # For experiment 1 (original SGXSTest), use the original naming
    if exp_num == 1:
        output_path = f"results/principled_layer_selection_results_{model_name}.csv"
    else:
        output_path = f"results/principled_layer_selection_results_{exp_name}_{model_name}.csv"
    
    csv_data = []
    for rank, (layer_idx, overall_score, detailed_scores) in enumerate(ranking, 1):
        row = {
            'Rank': rank,
            'Layer': layer_idx,
            'Overall_Score': f"{overall_score:.4f}",
            'Distributional_Score': f"{detailed_scores['distributional_score']:.4f}",
            'Geometric_Score': f"{detailed_scores['geometric_score']:.4f}",
            'Information_Score': f"{detailed_scores['information_score']:.4f}",
            'MMD': f"{detailed_scores['mmd']:.4f}",
            'Wasserstein': f"{detailed_scores['wasserstein']:.4f}",
            'KL_Divergence': f"{detailed_scores['kl_divergence']:.4f}",
            'SVM_Margin': f"{detailed_scores['svm_margin']:.4f}",
            'Silhouette': f"{detailed_scores['silhouette']:.4f}",
            'Distance_Ratio': f"{detailed_scores['distance_ratio']:.4f}",
            'Mutual_Info': f"{detailed_scores['mutual_info']:.4f}",
            'Entropy_Reduction': f"{detailed_scores['entropy_reduction']:.4f}"
        }
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    os.makedirs('results', exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main():
    """
    Main function to run principled layer selection analysis.
    """
    import sys

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python principled_layer_selection.py <model_type> [model_path] [experiments]")
        print("  model_type: qwen | llava | internvl")
        print("  model_path: (optional) path to the model directory")
        print("             If not provided, auto-detects from model/ directory")
        print("  experiments: comma-separated experiment numbers (1-4), default: all")
        print("    1 = SGXSTest (original paired dataset)")
        print("    2 = Noisy Distribution (random unmatched samples)")
        print("    3 = Latent Neighbor (synthetic pairs via embedding)")
        print("    4 = In-the-Wild (actual training split)")
        print("    5 = XSTest (similar to SGXSTest but used in training)")
        sys.exit(1)

    model_type = sys.argv[1].lower()
    
    # Validate model type
    if model_type not in MODEL_PATHS:
        print(f"Error: Unknown model type '{model_type}'. Use one of: {list(MODEL_PATHS.keys())}")
        sys.exit(1)
    
    # Auto-detect model path if not provided
    # Check if second argument looks like a path (contains '/' or exists as directory)
    if len(sys.argv) >= 3:
        potential_path = sys.argv[2]
        # If it contains '/' or exists as a directory, treat it as model_path
        if '/' in potential_path or os.path.exists(potential_path):
            model_path = potential_path
            experiment_arg = sys.argv[3] if len(sys.argv) > 3 else None
        else:
            # Second argument is experiment numbers, auto-detect model path
            model_path = MODEL_PATHS[model_type]
            experiment_arg = sys.argv[2]
    else:
        # No second argument, auto-detect model path and use default experiments
        model_path = MODEL_PATHS[model_type]
        experiment_arg = None
    
    # Verify model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found: {model_path}")
        print(f"Please download the model using download_models.py or provide the model path manually")
        sys.exit(1)
    
    print(f"Using model: {model_type} at {model_path}")
    
    # Parse experiment numbers
    exp_nums = parse_experiment_numbers(experiment_arg)
    print(f"Running experiments: {exp_nums}")

    # Initialize analyzer
    analyzer = LayerDiscriminativeAnalyzer(model_path, model_type)
    layer_start, layer_end = analyzer.get_model_layer_range()
    print(f"Layer range: {layer_start}-{layer_end} ({layer_end - layer_start + 1} layers)")

    # Run selected experiments
    all_experiment_results = []
    
    for exp_num in exp_nums:
        exp_result = run_single_experiment(analyzer, exp_num, layer_start, layer_end)
        all_experiment_results.append(exp_result)
        
        if exp_result is not None:
            print_experiment_summary(exp_num, exp_result['exp_name'], exp_result['ranking'])

    # Save results
    model_name = analyzer._get_model_name_for_filename()
    saved_files = []
    for exp_data in all_experiment_results:
        if exp_data is not None:
            output_path = save_experiment_results(exp_data, model_name)
            if output_path:
                saved_files.append(output_path)

    # Correlation analysis if multiple experiments completed
    if len([r for r in all_experiment_results if r is not None]) >= 2:
        experiment_scores = analyze_correlations(all_experiment_results)
        
        # Create visualization
        try:
            create_correlation_visualization(all_experiment_results, analyzer)
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")

    # Summary
    print(f"\nCompleted {len([r for r in all_experiment_results if r is not None])}/{len(exp_nums)} experiments")
    if saved_files:
        print(f"\nResults saved:")
        for file_path in saved_files:
            print(f"  - {file_path}")


def create_visualization(ranking: List[Tuple], composite_scores: Dict, analyzer=None):
    """
    Create comprehensive visualization of layer analysis results with detailed rankings.

    Args:
        ranking: Layer ranking results
        composite_scores: Composite scores for each layer
        analyzer: LayerDiscriminativeAnalyzer instance for model-specific naming
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        print(f"\n--- Creating Enhanced Visualization with Detailed Rankings ---")

        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with more subplots for detailed analysis
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        fig.suptitle('Comprehensive Principled Layer Selection Analysis for Jailbreak Detection',
                    fontsize=18, fontweight='bold', y=0.98)

        # Extract data for plotting
        layers = [layer for layer, _, _ in ranking]
        overall_scores = [score for _, score, _ in ranking]

        # Plot 1: Overall discriminative scores by layer (top-left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(layers, overall_scores, 'o-', linewidth=2, markersize=6, color='red')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Overall Discriminative Score')
        ax1.set_title('Overall Discriminative Power by Layer')
        ax1.grid(True, alpha=0.3)
        best_layer = layers[overall_scores.index(max(overall_scores))]
        ax1.axhline(y=max(overall_scores), color='r', linestyle='--', alpha=0.7,
                   label=f'Best: Layer {best_layer}')
        ax1.legend()

        # Plot 2: Category scores for top 10 layers (top-right, spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        top_10_layers = layers[:10]
        dist_scores = [composite_scores[layer]['distributional_score'] for layer in top_10_layers]
        geom_scores = [composite_scores[layer]['geometric_score'] for layer in top_10_layers]
        info_scores = [composite_scores[layer]['information_score'] for layer in top_10_layers]

        x = np.arange(len(top_10_layers))
        width = 0.25

        ax2.bar(x - width, dist_scores, width, label='Distributional', alpha=0.8, color='lightcoral')
        ax2.bar(x, geom_scores, width, label='Geometric', alpha=0.8, color='sandybrown')
        ax2.bar(x + width, info_scores, width, label='Information', alpha=0.8, color='lightgreen')

        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Category Score')
        ax2.set_title('Category Scores for Top 10 Layers')
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_10_layers)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Individual metrics heatmap (middle-left, spans 2x2)
        ax3 = fig.add_subplot(gs[1:3, :2])
        metrics = ['mmd', 'wasserstein', 'kl_divergence', 'svm_margin', 'silhouette', 'distance_ratio', 'mutual_info', 'entropy_reduction']
        heatmap_data = []

        for layer in top_10_layers:
            row = [composite_scores[layer][metric] for metric in metrics]
            heatmap_data.append(row)

        heatmap_data = np.array(heatmap_data)

        im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(metrics)))
        ax3.set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45, ha='right')
        ax3.set_yticks(range(len(top_10_layers)))
        ax3.set_yticklabels([f'Layer {layer}' for layer in top_10_layers])
        ax3.set_title('Individual Metrics Heatmap (Top 10 Layers)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Normalized Score')

        # Plot 4: Score distribution by rank (middle-right, top)
        ax4 = fig.add_subplot(gs[1, 2:])
        ranks = list(range(1, len(layers) + 1))
        ax4.scatter(ranks, overall_scores, alpha=0.6, s=50, color='lightblue')
        ax4.set_xlabel('Rank')
        ax4.set_ylabel('Overall Discriminative Score')
        ax4.set_title('Score Distribution by Rank\n(Shows sharp drop-off after top layers)')
        ax4.grid(True, alpha=0.3)

        # Highlight top 3
        for i in range(min(3, len(ranking))):
            layer, score, _ = ranking[i]
            ax4.annotate(f'Layer {layer}', (i+1, score), xytext=(5, 5),
                        textcoords='offset points', fontweight='bold')

        # NEW: Plot 5: Category Rankings (middle-right, bottom)
        ax5 = fig.add_subplot(gs[2, 2:])
        create_category_rankings_plot(ax5, composite_scores, layers)

        # NEW: Plot 6-13: Individual metric rankings (bottom row, 4 plots each spanning 2 rows)
        metric_axes = [
            fig.add_subplot(gs[3, 0]),  # MMD
            fig.add_subplot(gs[3, 1]),  # Wasserstein
            fig.add_subplot(gs[3, 2]),  # KL Divergence
            fig.add_subplot(gs[3, 3]),  # SVM Margin
        ]

        create_individual_metric_rankings(metric_axes, composite_scores, layers, metrics[:4])

        plt.tight_layout()

        # Save the plot with model-specific name
        if analyzer:
            model_name = analyzer._get_model_name_for_filename()
            output_path = f"results/layer_selection_analysis_{model_name}.png"
        else:
            output_path = "results/layer_selection_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive visualization saved to: {output_path}")

        # Create second figure for remaining metrics
        create_second_figure_with_remaining_metrics(composite_scores, layers, metrics[4:], analyzer)

        # Show the plot if in interactive environment
        try:
            plt.show()
        except:
            pass  # Non-interactive environment

    except Exception as e:
        print(f"Error creating visualization: {e}")


def create_category_rankings_plot(ax, composite_scores, layers):
    """Create rankings plot for the 3 metric categories."""
    # Get category scores for all layers
    dist_scores = [(layer, composite_scores[layer]['distributional_score']) for layer in layers]
    geom_scores = [(layer, composite_scores[layer]['geometric_score']) for layer in layers]
    info_scores = [(layer, composite_scores[layer]['information_score']) for layer in layers]

    # Sort by scores
    dist_scores.sort(key=lambda x: x[1], reverse=True)
    geom_scores.sort(key=lambda x: x[1], reverse=True)
    info_scores.sort(key=lambda x: x[1], reverse=True)

    # Plot top 10 for each category
    top_n = 10
    x_pos = np.arange(top_n)

    # Create grouped bar chart
    width = 0.25
    ax.bar(x_pos - width, [score for _, score in dist_scores[:top_n]], width,
           label='Distributional', alpha=0.8, color='lightcoral')
    ax.bar(x_pos, [score for _, score in geom_scores[:top_n]], width,
           label='Geometric', alpha=0.8, color='sandybrown')
    ax.bar(x_pos + width, [score for _, score in info_scores[:top_n]], width,
           label='Information', alpha=0.8, color='lightgreen')

    ax.set_xlabel('Rank Position')
    ax.set_ylabel('Category Score')
    ax.set_title('Top 10 Layers by Category\n(Different categories favor different layers)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{i+1}' for i in range(top_n)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add layer numbers as text annotations
    for i, (layer, score) in enumerate(dist_scores[:top_n]):
        if i < 5:  # Only annotate top 5 to avoid clutter
            ax.text(i - width, score + 0.01, f'L{layer}', ha='center', va='bottom', fontsize=8)
    for i, (layer, score) in enumerate(geom_scores[:top_n]):
        if i < 5:
            ax.text(i, score + 0.01, f'L{layer}', ha='center', va='bottom', fontsize=8)
    for i, (layer, score) in enumerate(info_scores[:top_n]):
        if i < 5:
            ax.text(i + width, score + 0.01, f'L{layer}', ha='center', va='bottom', fontsize=8)


def create_individual_metric_rankings(axes, composite_scores, layers, metrics):
    """Create ranking plots for individual metrics."""
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        # Get scores for this metric
        metric_scores = [(layer, composite_scores[layer][metric]) for layer in layers]
        metric_scores.sort(key=lambda x: x[1], reverse=True)

        # Plot top 10
        top_10 = metric_scores[:10]
        top_layers = [layer for layer, _ in top_10]
        top_scores = [score for _, score in top_10]

        # Use a color from a predefined list instead of colormap
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(range(len(top_10)), top_scores, alpha=0.7,
                     color=colors[i % len(colors)])

        ax.set_xlabel('Rank')
        ax.set_ylabel('Normalized Score')
        ax.set_title(f'{metric.replace("_", " ").title()}\nTop 10 Layers')
        ax.set_xticks(range(len(top_10)))
        ax.set_xticklabels([f'L{layer}' for layer in top_layers], rotation=45)
        ax.grid(True, alpha=0.3)

        # Highlight the best layer
        bars[0].set_color('red')
        bars[0].set_alpha(1.0)


def create_second_figure_with_remaining_metrics(composite_scores, layers, remaining_metrics, analyzer=None):
    """Create second figure for remaining individual metrics."""
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Individual Metric Rankings (Continued)', fontsize=14, fontweight='bold')

    axes2_flat = axes2.flatten()

    for i, (ax, metric) in enumerate(zip(axes2_flat, remaining_metrics)):
        # Get scores for this metric
        metric_scores = [(layer, composite_scores[layer][metric]) for layer in layers]
        metric_scores.sort(key=lambda x: x[1], reverse=True)

        # Plot top 10
        top_10 = metric_scores[:10]
        top_layers = [layer for layer, _ in top_10]
        top_scores = [score for _, score in top_10]

        # Use a color from a predefined list instead of colormap
        colors2 = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        bars = ax.bar(range(len(top_10)), top_scores, alpha=0.7,
                     color=colors2[i % len(colors2)])

        ax.set_xlabel('Rank')
        ax.set_ylabel('Normalized Score')
        ax.set_title(f'{metric.replace("_", " ").title()}\nTop 10 Layers')
        ax.set_xticks(range(len(top_10)))
        ax.set_xticklabels([f'L{layer}' for layer in top_layers], rotation=45)
        ax.grid(True, alpha=0.3)

        # Highlight the best layer
        bars[0].set_color('red')
        bars[0].set_alpha(1.0)

    plt.tight_layout()

    # Save second figure with model-specific name
    if analyzer:
        model_name = analyzer._get_model_name_for_filename()
        output_path2 = f"results/individual_metrics_rankings_{model_name}.png"
    else:
        output_path2 = "results/individual_metrics_rankings.png"
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Individual metrics rankings saved to: {output_path2}")


if __name__ == "__main__":
    main()
