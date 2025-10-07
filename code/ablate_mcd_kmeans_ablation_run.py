#!/usr/bin/env python3
"""
This script extends the k-means clustering approach to test different numbers of clusters
for both benign and malicious samples, comparing against the original dataset-based approach.
"""

import csv
import numpy as np
import random
import warnings
import signal
import os
import sys
from sklearn.cluster import KMeans
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

# Smart import handling based on command line arguments and available dependencies
def detect_model_from_args():
    """Detect model type from command line arguments"""
    # Look for --model or -m argument
    for i, arg in enumerate(sys.argv):
        if arg in ['--model', '-m'] and i + 1 < len(sys.argv):
            model_type = sys.argv[i + 1].lower()
            if model_type in ['qwen', 'llava', 'internvl']:
                return model_type
        # Also check for direct model arguments (backward compatibility)
        elif arg.lower() in ['qwen', 'llava', 'internvl'] and i > 0:
            return arg.lower()
    return 'llava'  # Default

# Determine which model we're using
REQUESTED_MODEL = detect_model_from_args()

# Import appropriate dependencies based on requested model
if REQUESTED_MODEL == 'qwen':
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        from feature_extractor_qwen import HiddenStateExtractor
        QWEN_AVAILABLE = True
        LLAVA_AVAILABLE = False
        INTERNVL_AVAILABLE = False
        print("Using Qwen model configuration")
    except ImportError as e:
        print(f"Error: Qwen dependencies not available: {e}")
        print("Please install: pip install qwen-vl-utils transformers>=4.37.0")
        sys.exit(1)
elif REQUESTED_MODEL == 'internvl':
    try:
        from feature_extractor_internvl import HiddenStateExtractor
        QWEN_AVAILABLE = False
        LLAVA_AVAILABLE = False
        INTERNVL_AVAILABLE = True
        print("Using InternVL model configuration")
    except ImportError as e:
        print(f"Error: InternVL dependencies not available: {e}")
        print("Please install InternVL dependencies")
        sys.exit(1)
else:
    try:
        from feature_extractor import HiddenStateExtractor
        LLAVA_AVAILABLE = True
        QWEN_AVAILABLE = False
        INTERNVL_AVAILABLE = False
        print("Using LLaVA model configuration")
    except ImportError as e:
        print(f"Error: LLaVA dependencies not available: {e}")
        print("Please install LLaVA dependencies")
        sys.exit(1)

def get_model_specific_extractor(model_path, model_type):
    """Get the appropriate feature extractor based on model type"""
    # Since we've already imported the correct HiddenStateExtractor based on model type,
    # we can just use it directly
    return HiddenStateExtractor(model_path)

# Import from the original k-means script
from main_mcd_kmeans import (
    GPU_DEVICE, cleanup_gpu_memory, train_learned_projection, apply_learned_projection,
    MCDDetector, prepare_balanced_training, prepare_balanced_evaluation,
    prepare_ood_data_structure, signal_handler
)
from load_datasets import *

def prepare_kmeans_data_structure_flexible(datasets_dict, hidden_states_dict, labels_dict, 
                                         k_benign=None, k_malicious=None, random_seed=42):
    """
    Flexible k-means data structure preparation with configurable cluster numbers.
    
    Args:
        datasets_dict: Dict of dataset names to samples
        hidden_states_dict: Dict of dataset names to features
        labels_dict: Dict of dataset names to labels
        k_benign: Number of clusters for benign samples
        k_malicious: Number of clusters for malicious samples
        random_seed: Random seed for k-means clustering
    
    Returns:
        in_dist_data: Dict of {cluster_name: list_of_features} for benign clusters
        ood_data: Dict of {cluster_name: list_of_features} for malicious clusters
        metadata: Dict with clustering information
    """
    # Aggregate all benign and malicious samples across datasets
    all_benign_features = []
    all_malicious_features = []
    
    # Count number of datasets that contribute to each class
    benign_dataset_count = 0
    malicious_dataset_count = 0
    
    for dataset_name in datasets_dict.keys():
        if dataset_name not in hidden_states_dict:
            continue
            
        features = hidden_states_dict[dataset_name]
        labels = labels_dict[dataset_name]
        
        # Separate benign and malicious samples
        benign_features = [features[i] for i, label in enumerate(labels) if label == 0]
        malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
        
        if benign_features:
            all_benign_features.extend(benign_features)
            benign_dataset_count += 1
        if malicious_features:
            all_malicious_features.extend(malicious_features)
            malicious_dataset_count += 1
    
    # Use provided k values or default to dataset counts
    if k_benign is None:
        k_benign = benign_dataset_count
    if k_malicious is None:
        k_malicious = malicious_dataset_count
    
    metadata = {
        'benign_samples': len(all_benign_features),
        'malicious_samples': len(all_malicious_features),
        'benign_dataset_count': benign_dataset_count,
        'malicious_dataset_count': malicious_dataset_count,
        'k_benign': k_benign,
        'k_malicious': k_malicious
    }
    
    in_dist_data = {}
    ood_data = {}
    
    # Apply k-means clustering to benign samples
    if len(all_benign_features) > 0 and k_benign > 0:
        if len(all_benign_features) >= k_benign:
            benign_features_array = np.array(all_benign_features)
            
            # Use k-means clustering
            kmeans_benign = KMeans(n_clusters=k_benign, random_state=random_seed, n_init=10)
            benign_cluster_labels = kmeans_benign.fit_predict(benign_features_array)
            
            # Group features by cluster
            for cluster_id in range(k_benign):
                cluster_mask = benign_cluster_labels == cluster_id
                cluster_features = benign_features_array[cluster_mask].tolist()
                if len(cluster_features) > 0:
                    in_dist_data[f"benign_cluster_{cluster_id}"] = cluster_features
        else:
            # Fallback: use all samples as one cluster
            in_dist_data["benign_cluster_0"] = all_benign_features
    
    # Apply k-means clustering to malicious samples
    if len(all_malicious_features) > 0 and k_malicious > 0:
        if len(all_malicious_features) >= k_malicious:
            malicious_features_array = np.array(all_malicious_features)
            
            # Use k-means clustering
            kmeans_malicious = KMeans(n_clusters=k_malicious, random_state=random_seed, n_init=10)
            malicious_cluster_labels = kmeans_malicious.fit_predict(malicious_features_array)
            
            # Group features by cluster
            for cluster_id in range(k_malicious):
                cluster_mask = malicious_cluster_labels == cluster_id
                cluster_features = malicious_features_array[cluster_mask].tolist()
                if len(cluster_features) > 0:
                    ood_data[f"malicious_cluster_{cluster_id}"] = cluster_features
        else:
            # Fallback: use all samples as one cluster
            ood_data["malicious_cluster_0"] = all_malicious_features
    
    return in_dist_data, ood_data, metadata

def evaluate_clustering_approach(detector, test_datasets, layer_hidden_states, layer_labels):
    """
    Evaluate a clustering approach on all test datasets.

    Args:
        detector: Trained MCDDetector
        test_datasets: Dict of test dataset names to samples
        layer_hidden_states: Dict of dataset features for this layer
        layer_labels: Dict of dataset labels for this layer

    Returns:
        results: Dict of results for each test dataset
        combined_result: Combined result across all test datasets
    """
    results = {}
    all_test_features = []
    all_test_labels = []
    
    # Evaluate on each test dataset individually
    for dataset_name in test_datasets.keys():
        if dataset_name in layer_hidden_states:
            test_features = layer_hidden_states[dataset_name]
            test_labels = layer_labels[dataset_name]

            if len(test_features) > 0:
                try:
                    # Ensure test_labels is a proper numpy array and flatten if needed
                    test_labels = np.array(test_labels).flatten()
                    test_features = np.array(test_features)

                    result = detector.evaluate(test_features, test_labels)
                    results[dataset_name] = result

                    # Accumulate for combined evaluation
                    all_test_features.extend(test_features.tolist())
                    all_test_labels.extend(test_labels.tolist())
                except Exception as e:
                    print(f"        Error evaluating {dataset_name}: {e}")
                    results[dataset_name] = None
    
    # Combined evaluation across all test datasets
    if len(all_test_features) > 0:
        try:
            # Ensure all_test_labels is a proper numpy array
            all_test_labels = np.array(all_test_labels)
            combined_result = detector.evaluate(all_test_features, all_test_labels)
            results['COMBINED'] = combined_result
        except Exception as e:
            print(f"        Error in combined evaluation: {e}")
            results['COMBINED'] = None
    else:
        results['COMBINED'] = None
    
    return results

def train_layer_projections(layer_idx, all_hidden_states, all_labels,
                           in_dist_datasets, ood_datasets, num_repetitions=5, base_seed=42):
    """
    Train multiple learned projections for a specific layer with different random seeds.

    Args:
        layer_idx: Layer index to train projections for
        all_hidden_states: Dict of all hidden states
        all_labels: Dict of all labels
        in_dist_datasets: Dict of in-distribution datasets
        ood_datasets: Dict of OOD datasets
        num_repetitions: Number of projection models to train
        base_seed: Base random seed

    Returns:
        projection_models: List of trained projection models
    """
    print(f"  Training {num_repetitions} projection models for layer {layer_idx}...")

    # Prepare training data for projection
    training_dataset_names = set(in_dist_datasets.keys()) | set(ood_datasets.keys())
    projection_features_dict = {}
    projection_labels_dict = {}

    for dataset_name in training_dataset_names:
        if dataset_name in all_hidden_states:
            projection_features_dict[dataset_name] = all_hidden_states[dataset_name][layer_idx]
            projection_labels_dict[dataset_name] = all_labels[dataset_name]

    projection_models = []
    for rep_idx in range(num_repetitions):
        # Use different seed for each repetition
        rep_seed = base_seed + layer_idx * 1000 + rep_idx * 100
        print(f"    Training projection {rep_idx + 1}/{num_repetitions} (seed={rep_seed})...")

        projection_model, _ = train_learned_projection(
            projection_features_dict, projection_labels_dict,
            device=GPU_DEVICE, random_seed=rep_seed
        )
        projection_models.append(projection_model)
        cleanup_gpu_memory()

    return projection_models

def run_single_experiment_with_repetitions(layer_idx, all_datasets, all_hidden_states, all_labels,
                                         in_dist_datasets, ood_datasets, test_datasets,
                                         projection_models, approach_name, k_benign=None, k_malicious=None,
                                         random_seed=42):
    """
    Run a single experiment with multiple projection models and average the results.

    Returns:
        averaged_results: Dict of averaged evaluation results
        metadata: Dict with experiment metadata
    """
    print(f"    Running {approach_name} (k_benign={k_benign}, k_malicious={k_malicious}) with {len(projection_models)} repetitions...")

    all_repetition_results = []

    for rep_idx, projection_model in enumerate(projection_models):
        try:
            # Prepare data for this layer
            layer_hidden_states = {}
            layer_labels = {}

            for dataset_name in all_datasets.keys():
                if dataset_name in all_hidden_states:
                    layer_hidden_states[dataset_name] = all_hidden_states[dataset_name][layer_idx]
                    layer_labels[dataset_name] = all_labels[dataset_name]

            # Apply learned projection
            projected_layer_hidden_states = apply_learned_projection(
                projection_model, layer_hidden_states, device=GPU_DEVICE
            )
            layer_hidden_states = projected_layer_hidden_states
            cleanup_gpu_memory()

            # Prepare data structures based on approach
            all_training_datasets = {**in_dist_datasets, **ood_datasets}
            training_layer_hidden_states = {k: v for k, v in layer_hidden_states.items() if k in all_training_datasets}
            training_layer_labels = {k: v for k, v in layer_labels.items() if k in all_training_datasets}

            if approach_name == "dataset_based":
                # Original dataset-based approach
                in_dist_data, ood_train_data = prepare_ood_data_structure(
                    all_training_datasets, training_layer_hidden_states, training_layer_labels
                )
                metadata = {'approach': 'dataset_based'}
            else:
                # K-means approach
                in_dist_data, ood_train_data, clustering_metadata = prepare_kmeans_data_structure_flexible(
                    all_training_datasets, training_layer_hidden_states, training_layer_labels,
                    k_benign=k_benign, k_malicious=k_malicious, random_seed=random_seed + layer_idx + rep_idx
                )
                metadata = {'approach': 'kmeans', **clustering_metadata}

            # Initialize and train MCD detector
            detector = MCDDetector(use_gpu=True)

            detector.fit_in_distribution(in_dist_data)
            detector.fit_ood_clusters(ood_train_data)

            # Use a subset of training data for threshold fitting (validation)
            validation_features = []
            validation_labels = []
            for dataset_name in all_training_datasets.keys():
                if dataset_name in layer_hidden_states:
                    features = layer_hidden_states[dataset_name]
                    labels = layer_labels[dataset_name]
                    # Use 20% of training data for validation
                    val_size = max(1, len(features) // 5)
                    validation_features.extend(features[:val_size])
                    validation_labels.extend(labels[:val_size])

            if len(validation_features) > 0:
                detector.fit_threshold(validation_features, validation_labels)

            # Evaluate on test datasets
            results = evaluate_clustering_approach(
                detector, test_datasets, layer_hidden_states, layer_labels
            )

            all_repetition_results.append(results)
            cleanup_gpu_memory()

        except Exception as e:
            print(f"        Repetition {rep_idx + 1} failed: {e}")
            cleanup_gpu_memory()
            continue

    # Average results across repetitions
    if not all_repetition_results:
        return {}, metadata

    averaged_results = average_results_across_repetitions(all_repetition_results)
    metadata['num_repetitions'] = str(len(all_repetition_results))
    metadata['num_successful_repetitions'] = str(len(all_repetition_results))

    return averaged_results, metadata

def average_results_across_repetitions(all_repetition_results):
    """
    Average evaluation results across multiple repetitions.

    Args:
        all_repetition_results: List of result dicts from different repetitions

    Returns:
        averaged_results: Dict with averaged metrics
    """
    if not all_repetition_results:
        return {}

    # Get all dataset names from the first successful result
    dataset_names = set()
    for results in all_repetition_results:
        dataset_names.update(results.keys())

    averaged_results = {}

    for dataset_name in dataset_names:
        # Collect all valid results for this dataset
        dataset_results = []
        for results in all_repetition_results:
            if dataset_name in results and results[dataset_name] is not None:
                dataset_results.append(results[dataset_name])

        if not dataset_results:
            averaged_results[dataset_name] = None
            continue

        # Average the metrics
        averaged_metrics = {}
        metric_names = dataset_results[0].keys()

        for metric_name in metric_names:
            # Skip array-like metrics (predictions, scores) - only average scalar metrics
            if metric_name in ['predictions', 'scores']:
                # For array metrics, just take the first one (they should be similar across repetitions)
                averaged_metrics[metric_name] = dataset_results[0][metric_name]
                continue

            metric_values = []
            for result in dataset_results:
                if metric_name in result:
                    value = result[metric_name]
                    # Check if it's a scalar and not NaN
                    if np.isscalar(value) and not np.isnan(value):
                        metric_values.append(value)

            if metric_values:
                averaged_metrics[metric_name] = np.mean(metric_values)
                averaged_metrics[f'{metric_name}_std'] = np.std(metric_values)
            else:
                averaged_metrics[metric_name] = np.nan
                averaged_metrics[f'{metric_name}_std'] = np.nan

        averaged_results[dataset_name] = averaged_metrics

    return averaged_results

def main():
    """Main function for k-means ablation study."""
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set random seed for reproducibility
    MAIN_SEED = 45
    random.seed(MAIN_SEED)
    np.random.seed(MAIN_SEED)
    torch.manual_seed(MAIN_SEED)
    torch.cuda.manual_seed(MAIN_SEED)
    torch.cuda.manual_seed_all(MAIN_SEED)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional determinism settings
    os.environ['PYTHONHASHSEED'] = str(MAIN_SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"Random seeds set for reproducibility (seed={MAIN_SEED})")
    
    # Support LLaVA, Qwen, and InternVL models
    model_type = REQUESTED_MODEL
    
    # Set model path based on model type
    if model_type == 'qwen':
        model_path = "./model/qwen2.5-vl-7b-instruct"
    elif model_type == 'llava':
        model_path = "model/llava-v1.6-vicuna-7b/"
    elif model_type == 'internvl':
        model_path = "model/internvl3-8b"
    else:
        print(f"Unknown model type: {model_type}. Use 'llava', 'qwen', or 'internvl'")
        sys.exit(1)
    
    # Initialize model-specific feature extractor
    extractor = get_model_specific_extractor(model_path, model_type)

    # Get the layer range for this model
    layer_start, layer_end = extractor.get_default_layer_range()
    num_layers = layer_end - layer_start + 1
    print(f"Model layer range: {layer_start}-{layer_end} ({num_layers} layers)")
    
    print("="*100)
    print(f"K-MEANS CLUSTERING ABLATION STUDY FOR OOD JAILBREAK DETECTION ({model_type.upper()})")
    print("="*100)
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print("Comparing different clustering approaches:")
    print("1. Original dataset-based clustering (baseline)")
    print("2. K-means with k=dataset_count (should be best among k-means)")
    print("3. K-means with various k values (should be inferior)")
    print("="*100)

    # Load balanced training and test data
    print("\n--- Loading Data ---")
    in_dist_datasets, ood_datasets = prepare_balanced_training()
    test_datasets = prepare_balanced_evaluation()

    # Analyze data composition
    print("\n--- Data Analysis ---")
    total_benign = sum(len(samples) for samples in in_dist_datasets.values())
    total_malicious = sum(len(samples) for samples in ood_datasets.values())
    benign_dataset_count = len(in_dist_datasets)
    malicious_dataset_count = len(ood_datasets)

    print(f"Training data: {total_benign} benign samples from {benign_dataset_count} datasets")
    print(f"Training data: {total_malicious} malicious samples from {malicious_dataset_count} datasets")
    print(f"Test datasets: {list(test_datasets.keys())}")

    # Extract hidden states for all datasets
    print("\n--- Extracting Hidden States ---")
    all_datasets = {**in_dist_datasets, **ood_datasets, **test_datasets}
    all_hidden_states = {}
    all_labels = {}

    for dataset_name, samples in all_datasets.items():
        print(f"Extracting features for {dataset_name} ({len(samples)} samples)...")

        batch_size = 25 if len(samples) > 5000 else 50
        memory_cleanup_freq = 5 if len(samples) > 5000 else 10

        hidden_states, labels, _ = extractor.extract_hidden_states(
            samples, f"{dataset_name}", use_cache=True,
            batch_size=batch_size, memory_cleanup_freq=memory_cleanup_freq,
            experiment_name="balanced_ml_detection"
        )
        all_hidden_states[dataset_name] = hidden_states
        all_labels[dataset_name] = labels

    # Define experimental configurations
    print("\n--- Experimental Setup ---")

    # Test layers (subset for efficiency) - use dynamic range based on model
    if model_type == 'internvl':
        # InternVL typically has layers 0-31, use even layers 12-24
        test_layers = [12, 14, 16, 18, 20, 22, 24]
    elif model_type == 'qwen':
        # Qwen might have different layer range, use available layers in 12-24 range
        available_layers = list(range(layer_start, layer_end + 1))
        test_layers = [l for l in available_layers if 12 <= l <= 24 and l % 2 == 0]
    else:  # llava
        # LLaVA typically has layers 0-31, use even layers 12-24
        test_layers = [12, 14, 16, 18, 20, 22, 24]
    
    print(f"Testing layers: {test_layers}")

    # Define clustering approaches to test
    clustering_approaches = []

    # 1. Baseline: Original dataset-based approach
    clustering_approaches.append({
        'name': 'dataset_based',
        'k_benign': None,
        'k_malicious': None,
        'description': 'Original dataset-based clustering (baseline)'
    })

    # 2. K-means with k=dataset_count (should be best among k-means)
    clustering_approaches.append({
        'name': f'kmeans_k{benign_dataset_count}_{malicious_dataset_count}',
        'k_benign': benign_dataset_count,
        'k_malicious': malicious_dataset_count,
        'description': f'K-means with k=dataset_count ({benign_dataset_count}, {malicious_dataset_count})'
    })

    # 3. K-means with various k values
    k_values_to_test = [1, 2, 3, 5, 8]  # Different cluster numbers
    for k_benign in k_values_to_test:
        for k_malicious in k_values_to_test:
            if k_benign <= total_benign and k_malicious <= total_malicious:  # Ensure feasible
                clustering_approaches.append({
                    'name': f'kmeans_k{k_benign}_{k_malicious}',
                    'k_benign': k_benign,
                    'k_malicious': k_malicious,
                    'description': f'K-means with k=({k_benign}, {k_malicious})'
                })

    print(f"Testing {len(clustering_approaches)} clustering approaches:")
    for i, approach in enumerate(clustering_approaches, 1):
        print(f"  {i}. {approach['description']}")

    # Train projection models for each layer with repetitions
    print(f"\n--- Training Projection Models ---")
    print(f"Training separate projection models for each layer with 5 repetitions each")

    # Number of repetitions for projection training
    NUM_PROJECTION_REPETITIONS = 5

    # Store projection models for each layer
    layer_projection_models = {}  # {layer_idx: [projection_model_1, projection_model_2, ...]}

    for layer_idx in test_layers:
        print(f"\n--- Training Projections for Layer {layer_idx} ---")
        projection_models = train_layer_projections(
            layer_idx, all_hidden_states, all_labels,
            in_dist_datasets, ood_datasets,
            num_repetitions=NUM_PROJECTION_REPETITIONS,
            base_seed=MAIN_SEED
        )
        layer_projection_models[layer_idx] = projection_models
        print(f"Completed training {len(projection_models)} projection models for layer {layer_idx}")

    print("All projection training completed!")
    cleanup_gpu_memory()

    # Run experiments
    print(f"\n--- Running Experiments ---")
    all_results = {}  # {layer_idx: {approach_name: results}}
    all_metadata = {}  # {layer_idx: {approach_name: metadata}}

    for layer_idx in test_layers:
        print(f"\n=== Evaluating Layer {layer_idx} ===")
        all_results[layer_idx] = {}
        all_metadata[layer_idx] = {}

        # Get projection models for this layer
        projection_models = layer_projection_models[layer_idx]

        for approach in clustering_approaches:
            approach_name = approach['name']
            k_benign = approach['k_benign']
            k_malicious = approach['k_malicious']

            try:
                results, metadata = run_single_experiment_with_repetitions(
                    layer_idx, all_datasets, all_hidden_states, all_labels,
                    in_dist_datasets, ood_datasets, test_datasets,
                    projection_models, approach_name,
                    k_benign=k_benign, k_malicious=k_malicious,
                    random_seed=MAIN_SEED
                )

                all_results[layer_idx][approach_name] = results
                all_metadata[layer_idx][approach_name] = metadata

                # Print summary for this approach
                if 'COMBINED' in results and results['COMBINED'] is not None:
                    combined = results['COMBINED']
                    acc_mean = combined['accuracy']
                    acc_std = combined.get('accuracy_std', 0)
                    f1_mean = combined['f1']
                    f1_std = combined.get('f1_std', 0)
                    auroc_mean = combined.get('auroc', 0)
                    auroc_std = combined.get('auroc_std', 0)

                    print(f"      {approach_name}: Acc={acc_mean:.3f}±{acc_std:.3f}, "
                          f"F1={f1_mean:.3f}±{f1_std:.3f}, AUROC={auroc_mean:.3f}±{auroc_std:.3f}")
                else:
                    print(f"      {approach_name}: Failed")

            except Exception as e:
                print(f"      {approach_name}: Error - {e}")
                all_results[layer_idx][approach_name] = {}
                all_metadata[layer_idx][approach_name] = {'error': str(e)}

    # Save detailed results
    print(f"\n--- Saving Results ---")
    output_path = f"results/kmeans_ablation_{model_type}_results.csv"
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Layer", "Approach", "K_Benign", "K_Malicious", "Dataset", "Num_Repetitions",
            "Accuracy_Mean", "Accuracy_Std", "F1_Mean", "F1_Std",
            "TPR_Mean", "TPR_Std", "FPR_Mean", "FPR_Std",
            "AUROC_Mean", "AUROC_Std", "AUPRC_Mean", "AUPRC_Std",
            "Threshold_Mean", "Threshold_Std"
        ])

        for layer_idx in test_layers:
            for approach_name, results in all_results[layer_idx].items():
                metadata = all_metadata[layer_idx][approach_name]
                k_benign = metadata.get('k_benign', 'N/A')
                k_malicious = metadata.get('k_malicious', 'N/A')
                num_reps = metadata.get('num_successful_repetitions', 'N/A')

                if results:
                    for dataset_name, result in results.items():
                        if result is not None:
                            def format_metric(metric_name):
                                mean_val = result.get(metric_name, np.nan)
                                std_val = result.get(f'{metric_name}_std', np.nan)
                                mean_str = "N/A" if np.isnan(mean_val) else f"{mean_val:.4f}"
                                std_str = "N/A" if np.isnan(std_val) else f"{std_val:.4f}"
                                return mean_str, std_str

                            acc_mean, acc_std = format_metric('accuracy')
                            f1_mean, f1_std = format_metric('f1')
                            tpr_mean, tpr_std = format_metric('tpr')
                            fpr_mean, fpr_std = format_metric('fpr')
                            auroc_mean, auroc_std = format_metric('auroc')
                            auprc_mean, auprc_std = format_metric('auprc')
                            thresh_mean, thresh_std = format_metric('threshold')

                            writer.writerow([
                                layer_idx, approach_name, k_benign, k_malicious, dataset_name, num_reps,
                                acc_mean, acc_std, f1_mean, f1_std,
                                tpr_mean, tpr_std, fpr_mean, fpr_std,
                                auroc_mean, auroc_std, auprc_mean, auprc_std,
                                thresh_mean, thresh_std
                            ])
                else:
                    writer.writerow([
                        layer_idx, approach_name, k_benign, k_malicious, "FAILED", "0",
                        "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                        "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
                    ])

    print(f"Detailed results saved to {output_path}")

    # Generate summary analysis
    print(f"\n--- Summary Analysis ---")
    generate_summary_analysis(all_results, test_layers, clustering_approaches, model_type)

def generate_summary_analysis(all_results, test_layers, clustering_approaches, model_type):
    """Generate and print summary analysis of the ablation study."""

    print("\n" + "="*100)
    print(f"K-MEANS CLUSTERING ABLATION STUDY SUMMARY ({model_type.upper()})")
    print("="*100)

    # Collect performance data for analysis
    approach_performances = {}  # {approach_name: [combined_accuracies]}

    for layer_idx in test_layers:
        for approach_name, results in all_results[layer_idx].items():
            if approach_name not in approach_performances:
                approach_performances[approach_name] = []

            if results and 'COMBINED' in results and results['COMBINED'] is not None:
                combined_acc = results['COMBINED']['accuracy']
                approach_performances[approach_name].append(combined_acc)
            else:
                approach_performances[approach_name].append(0.0)  # Failed experiment

    # Calculate average performance for each approach
    approach_avg_performance = {}
    for approach_name, performances in approach_performances.items():
        if performances:
            approach_avg_performance[approach_name] = np.mean(performances)
        else:
            approach_avg_performance[approach_name] = 0.0

    # Sort approaches by average performance (descending)
    sorted_approaches = sorted(approach_avg_performance.items(), key=lambda x: x[1], reverse=True)

    print(f"\nRanking by Average Combined Accuracy across {len(test_layers)} layers:")
    print(f"{'Rank':<4} {'Approach':<30} {'Avg Accuracy':<12} {'Description'}")
    print("-" * 80)

    for rank, (approach_name, avg_acc) in enumerate(sorted_approaches, 1):
        # Find description
        description = "Unknown"
        for approach in clustering_approaches:
            if approach['name'] == approach_name:
                description = approach['description']
                break

        print(f"{rank:<4} {approach_name:<30} {avg_acc:.4f}       {description}")

    # Detailed layer-by-layer comparison
    print(f"\nDetailed Layer-by-Layer Performance:")
    print(f"{'Approach':<30} " + " ".join([f"L{layer:<6}" for layer in test_layers]) + " Average")
    print("-" * (30 + 8 * len(test_layers) + 10))

    for approach_name, avg_acc in sorted_approaches:
        performances = approach_performances[approach_name]
        perf_str = f"{approach_name:<30} "
        for perf in performances:
            perf_str += f"{perf:.3f}   "
        perf_str += f"{avg_acc:.3f}"
        print(perf_str)

    # Analysis of k-means vs dataset-based
    print(f"\nComparative Analysis:")

    # Find dataset-based performance
    dataset_based_perf = approach_avg_performance.get('dataset_based', 0.0)

    # Find best k-means performance
    kmeans_performances = {name: perf for name, perf in approach_avg_performance.items()
                          if name.startswith('kmeans_')}

    if kmeans_performances:
        best_kmeans_name, best_kmeans_perf = max(kmeans_performances.items(), key=lambda x: x[1])

        print(f"1. Dataset-based approach (baseline): {dataset_based_perf:.4f}")
        print(f"2. Best k-means approach ({best_kmeans_name}): {best_kmeans_perf:.4f}")

        if dataset_based_perf > 0:
            performance_gap = dataset_based_perf - best_kmeans_perf
            relative_gap = (performance_gap / dataset_based_perf) * 100
            print(f"3. Performance gap: {performance_gap:.4f} ({relative_gap:.1f}% relative)")

            if performance_gap > 0:
                print("   → Dataset-based clustering is SUPERIOR to k-means clustering")
            else:
                print("   → K-means clustering performs comparably to dataset-based clustering")

        # Check if k=dataset_count is best among k-means
        dataset_count_approaches = [name for name in kmeans_performances.keys()
                                  if 'k3_3' in name]  # Assuming 3 datasets each

        if dataset_count_approaches:
            dataset_count_name = dataset_count_approaches[0]
            dataset_count_perf = kmeans_performances[dataset_count_name]

            print(f"4. K-means with k=dataset_count: {dataset_count_perf:.4f}")

            if dataset_count_name == best_kmeans_name:
                print("   → Among k-means approaches, k=dataset_count works BEST")
            else:
                print(f"   → Among k-means approaches, k=dataset_count is not optimal")
                print(f"     Best k-means: {best_kmeans_name} ({best_kmeans_perf:.4f})")

    # Statistical significance test (if we had multiple runs)
    print(f"\nKey Findings for Paper:")
    print(f"• Dataset-based clustering achieves {dataset_based_perf:.1%} average accuracy")
    print(f"• Best k-means clustering achieves {best_kmeans_perf:.1%} average accuracy")
    print(f"• K-means clustering shows {'superior' if best_kmeans_perf > dataset_based_perf else 'inferior'} performance")
    print(f"• Performance difference: {abs(dataset_based_perf - best_kmeans_perf):.1%}")

    # Save summary to file
    summary_path = f"results/kmeans_ablation_{model_type}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"K-MEANS CLUSTERING ABLATION STUDY SUMMARY ({model_type.upper()})\n")
        f.write("="*50 + "\n\n")

        f.write("Ranking by Average Combined Accuracy:\n")
        for rank, (approach_name, avg_acc) in enumerate(sorted_approaches, 1):
            description = "Unknown"
            for approach in clustering_approaches:
                if approach['name'] == approach_name:
                    description = approach['description']
                    break
            f.write(f"{rank}. {approach_name}: {avg_acc:.4f} - {description}\n")

        f.write(f"\nDataset-based vs K-means:\n")
        f.write(f"Dataset-based: {dataset_based_perf:.4f}\n")
        f.write(f"Best K-means: {best_kmeans_perf:.4f}\n")
        f.write(f"Performance gap: {abs(dataset_based_perf - best_kmeans_perf):.4f}\n")

    print(f"\nSummary saved to {summary_path}")
    print("="*100)

if __name__ == "__main__":
    main()
