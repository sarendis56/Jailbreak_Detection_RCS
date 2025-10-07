#!/usr/bin/env python3
"""
Test the balanced OOD KCD method against SafeMTData multi-turn jailbreaking dataset.
This script integrates SafeMTData with the existing KCD detection framework.
"""

import os
import sys

# Set PyTorch CUDA memory management environment variable to help with fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import json
import torch
import numpy as np
import random
import warnings
import hashlib
import pickle
import argparse
from datasets import load_dataset
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import csv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

# Add LLaVA to path
sys.path.append('src/LLaVA')

# Import existing modules
from load_datasets import *

# Central baseline data structure for different models
MODEL_BASELINES = {
    'llava': {
        'overall_accuracy': 0.914,
        'f1': 0.9211,
        'auroc': 0.9352
    },
    'qwen': {
        'overall_accuracy': 0.8915,
        'f1': 0.8994,
        'auroc': 0.9627 
    }
}

def get_model_baselines(model_type):
    """Get baseline values for the specified model type"""
    if model_type not in MODEL_BASELINES:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_BASELINES.keys())}")
    return MODEL_BASELINES[model_type]

# Smart import handling based on command line arguments and available dependencies
def detect_model_from_args():
    """Detect model type from command line arguments"""
    # Look for --model or -m argument
    for i, arg in enumerate(sys.argv):
        if arg in ['--model', '-m'] and i + 1 < len(sys.argv):
            model_type = sys.argv[i + 1].lower()
            if model_type in ['qwen', 'llava']:
                return model_type
        # Also check for direct model arguments (backward compatibility)
        elif arg.lower() in ['qwen', 'llava'] and i > 0:
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
        print("Using Qwen model configuration")
    except ImportError as e:
        print(f"Error: Qwen dependencies not available: {e}")
        print("Please install: pip install qwen-vl-utils transformers>=4.37.0")
        sys.exit(1)
else:
    try:
        from feature_extractor import HiddenStateExtractor
        LLAVA_AVAILABLE = True
        QWEN_AVAILABLE = False
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

from main_kcd import (
    KCDDetector, prepare_knn_data_structure,
    prepare_balanced_training, train_learned_projection, apply_learned_projection,
    CONFIG, GPU_DEVICE, cleanup_gpu_memory
)
from main_mcd import prepare_balanced_evaluation

def build_dynamic_multiturn_conversation_for_detection(multi_turn_queries, tokenizer, model, model_name):
    """Build multi-turn conversation with CONSISTENT hidden state extraction position"""

    # Handle different model types
    if REQUESTED_MODEL == 'qwen':
        # For Qwen models, use a simpler conversation format
        conversation_text = ""

        # Build conversation step by step with real Qwen responses for ALL turns EXCEPT the final one
        for turn_idx, query in enumerate(multi_turn_queries[:-1]):  # Process all but the last turn
            # Add user message
            conversation_text += f"User: {query}\n"

            # Generate response from Qwen - use simple text format for text-only input
            messages = [{"role": "user", "content": query}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize and generate
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Truncate if too long
            max_length = getattr(model.config, 'max_position_embeddings', 4096)
            if model_inputs.input_ids.shape[1] > max_length:
                model_inputs.input_ids = model_inputs.input_ids[:, :max_length]
                model_inputs.attention_mask = model_inputs.attention_mask[:, :max_length]

            with torch.inference_mode():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.2,
                    use_cache=True
                )

            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Add assistant response
            conversation_text += f"Assistant: {response}\n"

            print(f"\n--- Turn {turn_idx + 1} ---")
            print(f"User: {query}")
            print(f"Assistant: {response}")

        # For the FINAL turn: Add user query but NO assistant response
        final_query = multi_turn_queries[-1]
        conversation_text += f"User: {final_query}\nAssistant:"

        print(f"\n--- Turn {len(multi_turn_queries)} (FINAL - No Response Added) ---")
        print(f"User: {final_query}")
        print(f"Assistant: [EMPTY - Hidden state extracted here, same as single-turn]")

        return conversation_text

    else:
        # Original LLaVA implementation
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX

        def find_conv_mode(model_name):
            if "llama-2" in model_name.lower():
                return "llava_llama_2"
            elif "mistral" in model_name.lower():
                return "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                return "chatml_direct"
            elif "v1" in model_name.lower():
                return "llava_v1"
            elif "mpt" in model_name.lower():
                return "mpt"
            else:
                return "llava_v0"

        conv_mode = find_conv_mode(model_name)
        conv = conv_templates[conv_mode].copy()

        # Build conversation step by step with real LLaVA responses for ALL turns EXCEPT the final one
        for turn_idx, query in enumerate(multi_turn_queries[:-1]):  # Process all but the last turn
            # Add user message to conversation
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)

            # Get current prompt
            prompt = conv.get_prompt()

            # Tokenize
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # Truncate if too long
            max_length = getattr(model.config, 'max_position_embeddings', 4096)
            if input_ids.shape[1] > max_length:
                input_ids = input_ids[:, :max_length]

            # Generate response from LLaVA
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=None,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=256,  # Shorter responses for intermediate turns
                    use_cache=True,
                    stopping_criteria=None
                )

            # Decode response
            input_token_len = input_ids.shape[1]
            if output_ids.shape[1] >= input_token_len:
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            else:
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            outputs = outputs.strip()

            # Update conversation with LLaVA's actual response
            conv.messages[-1][-1] = outputs

            print(f"\n--- Turn {turn_idx + 1} ---")
            print(f"User: {query}")
            print(f"LLaVA: {outputs}")

        # For the FINAL turn: Add user query but NO assistant response (same as single-turn)
        final_query = multi_turn_queries[-1]
        conv.append_message(conv.roles[0], final_query)
        conv.append_message(conv.roles[1], None)  # CRITICAL: No response, same as single-turn

        print(f"\n--- Turn {len(multi_turn_queries)} (FINAL - No Response Added) ---")
        print(f"User: {final_query}")
        print(f"Assistant: [EMPTY - Hidden state extracted here, same as single-turn]")

        # Return the conversation object ready for feature extraction
        return conv

def get_safemtdata_cache_path():
    """Get the cache directory path for SafeMTData responses"""
    cache_dir = "cache/safemtdata_responses"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_sample_hash(sample):
    """Generate a unique hash for a SafeMTData sample"""
    # Create a deterministic hash based on the sample content
    content = json.dumps(sample, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

def cache_safemtdata_response(sample_hash, response_data):
    """Cache a SafeMTData response to avoid re-querying the model"""
    cache_dir = get_safemtdata_cache_path()
    cache_file = os.path.join(cache_dir, f"{sample_hash}.pkl")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(response_data, f)

def load_cached_safemtdata_response(sample_hash):
    """Load a cached SafeMTData response if it exists"""
    cache_dir = get_safemtdata_cache_path()
    cache_file = os.path.join(cache_dir, f"{sample_hash}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def load_safemtdata_for_adaptability_test(model_path, model_type, num_training=200, num_test=50):
    """Load SafeMTData for adaptability test: training + test with NO overlap"""
    print(f"Loading SafeMTData for adaptability test: {num_training} training + {num_test} test samples...")

    try:
        # Load Attack_600 dataset
        attack_600 = load_dataset("SafeMTData/SafeMTData", 'Attack_600')["Attack_600"]
        print(f"Loaded Attack_600: {len(attack_600)} samples")

        # Ensure we have enough samples
        total_needed = num_training + num_test
        if len(attack_600) < total_needed:
            print(f"Warning: Only {len(attack_600)} samples available, need {total_needed}")
            num_training = min(num_training, len(attack_600))
            num_test = min(num_test, len(attack_600) - num_training)
            print(f"Adjusted to: {num_training} training + {num_test} test")

        # Split into discrete sets: training [0:num_training], test [num_training:num_training+num_test]
        attack_600_list = list(attack_600)
        training_samples = attack_600_list[:num_training]
        test_samples = attack_600_list[num_training:num_training + num_test]

        print(f"Split SafeMTData: {len(training_samples)} training, {len(test_samples)} test, {len(attack_600) - total_needed} unused")

        # Check if we need to process any samples (i.e., if any are not cached)
        samples_needing_processing = []
        all_samples = training_samples + test_samples

        for sample in all_samples:
            multi_turn_queries = sample['multi_turn_queries']
            if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                sample_hash = get_sample_hash(sample)
                cached_response = load_cached_safemtdata_response(sample_hash)
                if cached_response is None:
                    samples_needing_processing.append(sample)

        # Only load model if there are samples that need processing
        model_loaded = False
        tokenizer = None
        model = None
        image_processor = None

        if samples_needing_processing:
            print(f"Loading {model_type.upper()} model and tokenizer for SafeMTData processing ({len(samples_needing_processing)} uncached samples)...")

            if model_type == 'qwen':
                # Load Qwen model
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, device_map="auto"
                )
                tokenizer = AutoProcessor.from_pretrained(model_path)
                model_name = model_path
                print(f"Qwen model loaded on device: {GPU_DEVICE}")
            else:
                # Load LLaVA model
                from llava.model.builder import load_pretrained_model
                from llava.mm_utils import get_model_name_from_path
                model_name = get_model_name_from_path(model_path)
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    device=GPU_DEVICE
                )
                print(f"LLaVA model loaded on device: {GPU_DEVICE}")

            model_loaded = True

            # Clean up after loading
            cleanup_gpu_memory()
        elif not samples_needing_processing:
            print("All samples are cached, skipping model loading")
            model_name = model_path  # Set model_name even when not loading

        # Process training samples
        print(f"\n--- Processing {len(training_samples)} SafeMTData TRAINING samples ---")
        safemt_training = []
        for idx, sample in enumerate(training_samples):

            multi_turn_queries = sample['multi_turn_queries']
            if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                # Check cache first
                sample_hash = get_sample_hash(sample)
                cached_response = load_cached_safemtdata_response(sample_hash)

                if cached_response is not None:
                    # Using cached response
                    final_conversation_text = cached_response
                else:
                    print(f"  Generating new response for sample {idx + 1}")
                    # Build REAL multi-turn conversation with CONSISTENT hidden state extraction
                    final_conv = build_dynamic_multiturn_conversation_for_detection(
                        multi_turn_queries, tokenizer, model, model_name
                    )
                    # Get the final conversation text for feature extraction
                    if REQUESTED_MODEL == 'qwen':
                        final_conversation_text = final_conv  # Already a string for Qwen
                    else:
                        final_conversation_text = final_conv.get_prompt()  # LLaVA conversation object
                    # Cache the response
                    cache_safemtdata_response(sample_hash, final_conversation_text)

                safemt_sample = {
                    "txt": final_conversation_text,  # REAL multi-turn conversation with model responses
                    "img": None,  # Text-only
                    "toxicity": 1,  # All SafeMTData samples are malicious
                    "category": sample.get('category', 'multi_turn_jailbreak'),
                    "original_id": sample.get('id', -1),
                    "num_turns": len(multi_turn_queries),
                    "final_query": multi_turn_queries[-1]  # Keep track of the final harmful query
                }
                safemt_training.append(safemt_sample)

        # Process test samples
        print(f"\n--- Processing {len(test_samples)} SafeMTData TEST samples ---")
        safemt_test = []
        for idx, sample in enumerate(test_samples):

            multi_turn_queries = sample['multi_turn_queries']
            if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                # Check cache first
                sample_hash = get_sample_hash(sample)
                cached_response = load_cached_safemtdata_response(sample_hash)

                if cached_response is not None:
                    # Using cached response
                    final_conversation_text = cached_response
                else:
                    print(f"  Generating new response for test sample {idx + 1}")
                    # Build REAL multi-turn conversation with CONSISTENT hidden state extraction
                    final_conv = build_dynamic_multiturn_conversation_for_detection(
                        multi_turn_queries, tokenizer, model, model_name
                    )
                    # Get the final conversation text for feature extraction
                    if REQUESTED_MODEL == 'qwen':
                        final_conversation_text = final_conv  # Already a string for Qwen
                    else:
                        final_conversation_text = final_conv.get_prompt()  # LLaVA conversation object
                    # Cache the response
                    cache_safemtdata_response(sample_hash, final_conversation_text)

                safemt_sample = {
                    "txt": final_conversation_text,  # REAL multi-turn conversation with model responses
                    "img": None,  # Text-only
                    "toxicity": 1,  # All SafeMTData samples are malicious
                    "category": sample.get('category', 'multi_turn_jailbreak'),
                    "original_id": sample.get('id', -1),
                    "num_turns": len(multi_turn_queries),
                    "final_query": multi_turn_queries[-1]  # Keep track of the final harmful query
                }
                safemt_test.append(safemt_sample)

        print(f"\nConverted SafeMTData: {len(safemt_training)} training + {len(safemt_test)} test samples")
        print("Sample conversation structure:")
        if safemt_training:
            sample_conv = safemt_training[0]['txt']
            print(f"First 500 chars: {sample_conv[:500]}...")

        # Clean up model memory after SafeMTData processing (only if we loaded it)
        if model_loaded and model is not None:
            if model_type == 'qwen':
                del model, tokenizer
            else:
                del model, tokenizer, image_processor
            cleanup_gpu_memory()
            print("Cleaned up SafeMTData model from memory")

        return safemt_training, safemt_test

    except Exception as e:
        print(f"Error loading SafeMTData: {e}")
        # Clean up on error too (only if we loaded the model)
        if 'model_loaded' in locals() and model_loaded and 'model' in locals() and model is not None:
            if model_type == 'qwen':
                del model, tokenizer
            else:
                del model, tokenizer, image_processor
            cleanup_gpu_memory()
        return [], []

def create_adaptability_datasets(model_path, model_type):
    """Create training and test datasets for adaptability experiment"""
    print("Creating adaptability experiment datasets...")

    # Load original training and test datasets (baseline)
    print("Loading original datasets...")
    benign_training, malicious_training = prepare_balanced_training()
    original_test_datasets = prepare_balanced_evaluation()

    # Load SafeMTData with training/test split (NO overlap)
    print("Loading SafeMTData with training/test split...")
    safemt_training, safemt_test = load_safemtdata_for_adaptability_test(
        model_path, model_type, num_training=200, num_test=50
    )

    # EXPAND malicious training by adding SafeMTData training samples
    expanded_malicious_training = malicious_training.copy()
    if safemt_training:
        expanded_malicious_training["SafeMTData_Training"] = safemt_training
        print(f"EXPANDED malicious training: added {len(safemt_training)} SafeMTData samples")

    # EXPAND test datasets by adding SafeMTData test samples
    expanded_test_datasets = original_test_datasets.copy()
    if safemt_test:
        expanded_test_datasets["SafeMTData_Test"] = safemt_test
        print(f"EXPANDED test datasets: added {len(safemt_test)} SafeMTData samples")

    print(f"\nAdaptability experiment datasets:")
    print(f"TRAINING:")
    for dataset_name, samples in benign_training.items():
        print(f"  Benign - {dataset_name}: {len(samples)} samples")
    for dataset_name, samples in expanded_malicious_training.items():
        dataset_type = "Multi-turn" if "SafeMTData" in dataset_name else "Single-turn"
        print(f"  Malicious - {dataset_name}: {len(samples)} samples [{dataset_type}]")

    total_benign = sum(len(samples) for samples in benign_training.values())
    total_malicious = sum(len(samples) for samples in expanded_malicious_training.values())
    print(f"  TOTAL TRAINING: {total_benign} benign + {total_malicious} malicious = {total_benign + total_malicious}")

    print(f"TEST:")
    for dataset_name, samples in expanded_test_datasets.items():
        dataset_type = "Multi-turn" if "SafeMTData" in dataset_name else "Single-turn"
        toxicity_info = f"({sum(1 for s in samples if s.get('toxicity', 0) == 1)} malicious, {sum(1 for s in samples if s.get('toxicity', 0) == 0)} benign)"
        print(f"  {dataset_name}: {len(samples)} samples {toxicity_info} [{dataset_type}]")

    return benign_training, expanded_malicious_training, expanded_test_datasets

def analyze_safemtdata_results_detailed(detector, test_features, test_labels, dataset_name, layer_idx):
    """Analyze SafeMTData results with detailed per-sample scoring"""
    print(f"\n=== SafeMTData Multi-turn Attack Analysis (Layer {layer_idx}) ===")

    # Get detailed predictions and scores
    predictions, scores = detector.predict(test_features)
    threshold = detector.threshold

    print(f"Detection threshold: {threshold:.4f}")
    print(f"Total samples: {len(test_features)}")
    print(f"All samples should be malicious (label=1): {set(test_labels)}")
    print(f"Prediction logic: score > {threshold:.4f} → malicious (1), score ≤ {threshold:.4f} → benign (0)")

    print(f"\n--- Per-Sample Analysis ---")
    print(f"{'Sample':<8} {'Score':<12} {'Prediction':<12} {'Label':<8} {'Correct':<8} {'Status'}")
    print("-" * 70)

    correct_count = 0
    for i, (score, pred, label) in enumerate(zip(scores, predictions, test_labels)):
        correct = (pred == label)
        if correct:
            correct_count += 1

        status = "✅ DETECTED" if (pred == 1 and label == 1) else "❌ MISSED" if (pred == 0 and label == 1) else "⚠️  OTHER"

        print(f"{i+1:<8} {score:<12.4f} {pred:<12} {label:<8} {correct:<8} {status}")

    accuracy = correct_count / len(test_labels)

    # Calculate detailed metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n--- Classification Report ---")

    # Check if we have both classes in the data
    unique_labels = set(test_labels)
    unique_preds = set(predictions)

    if len(unique_labels) == 1 and len(unique_preds) == 1:
        # Only one class present - SafeMTData only has malicious samples
        if 1 in unique_labels and 1 in unique_preds:
            print("All samples are malicious and all were correctly detected as malicious.")
            print("Perfect detection: 100% accuracy on malicious samples.")
        else:
            print("Unexpected single class scenario.")
    else:
        # Normal case with multiple classes
        available_classes = sorted(list(unique_labels.union(unique_preds)))
        target_names = ['Benign' if c == 0 else 'Malicious' for c in available_classes]
        print(classification_report(test_labels, predictions, labels=available_classes, target_names=target_names))

    print(f"\n--- Confusion Matrix ---")
    cm = confusion_matrix(test_labels, predictions)

    # Handle different matrix sizes
    if cm.shape == (1, 1):
        print(f"Single class confusion matrix:")
        if test_labels[0] == 1:  # All malicious
            print(f"Malicious samples correctly detected: {cm[0,0]}")
        else:  # All benign
            print(f"Benign samples correctly detected: {cm[0,0]}")
    else:
        print(f"Confusion Matrix:")
        print(f"                Predicted")
        print(f"Actual    Benign  Malicious")
        print(f"Benign    {cm[0,0]:<6}  {cm[0,1]:<6}")
        print(f"Malicious {cm[1,0]:<6}  {cm[1,1]:<6}")

    # Score distribution analysis
    print(f"\n--- Score Distribution Analysis ---")
    print(f"Score statistics:")
    print(f"  Min score: {min(scores):.4f}")
    print(f"  Max score: {max(scores):.4f}")
    print(f"  Mean score: {sum(scores)/len(scores):.4f}")
    print(f"  Threshold: {threshold:.4f}")

    # Count how many scores are above/below threshold
    above_threshold = sum(1 for s in scores if s > threshold)
    below_threshold = len(scores) - above_threshold

    print(f"  Scores above threshold (predicted malicious): {above_threshold}")
    print(f"  Scores below threshold (predicted benign): {below_threshold}")

    return {
        'accuracy': accuracy,
        'scores': scores,
        'predictions': predictions,
        'labels': test_labels,
        'threshold': threshold
    }

def load_safemtdata_for_variable_training_experiment(model_path, model_type, max_training=50, num_test=100, random_seed=42):
    """Load SafeMTData for variable training experiment with fixed test set and maximum training pool"""
    print(f"Loading SafeMTData pool (seed={random_seed}): up to {max_training} training + {num_test} test samples...")

    try:
        # Load Attack_600 dataset
        attack_600 = load_dataset("SafeMTData/SafeMTData", 'Attack_600')["Attack_600"]
        print(f"Loaded Attack_600: {len(attack_600)} samples")

        # Convert to list and set random seed for reproducible sampling
        attack_600_list = list(attack_600)
        random.seed(random_seed)

        # Shuffle the dataset to get different samples for each run
        shuffled_samples = attack_600_list.copy()
        random.shuffle(shuffled_samples)

        # Always use first 100 samples for test (consistent across all experiments)
        test_samples = shuffled_samples[:num_test]

        # Create a pool of training samples (up to max_training)
        remaining_samples = shuffled_samples[num_test:]
        if len(remaining_samples) < max_training:
            print(f"Warning: Only {len(remaining_samples)} samples available for training pool, need {max_training}")
            max_training = len(remaining_samples)
        training_pool = remaining_samples[:max_training]

        print(f"Created SafeMTData pool (seed={random_seed}): {len(training_pool)} training pool, {len(test_samples)} test")

        # Check if we need to process any samples (i.e., if any are not cached)
        samples_needing_processing = []
        all_samples = training_pool + test_samples  # Process the full training pool

        for sample in all_samples:
            multi_turn_queries = sample['multi_turn_queries']
            if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                sample_hash = get_sample_hash(sample)
                cached_response = load_cached_safemtdata_response(sample_hash)
                if cached_response is None:
                    samples_needing_processing.append(sample)

        # Only load model if there are samples that need processing
        model_loaded = False
        tokenizer = None
        model = None
        image_processor = None

        if samples_needing_processing:
            print(f"Loading {model_type.upper()} model and tokenizer for SafeMTData processing ({len(samples_needing_processing)} uncached samples)...")

            if model_type == 'qwen':
                # Load Qwen model
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model_name = model_path
                print(f"Qwen model loaded on device: {GPU_DEVICE}")
            else:
                # Load LLaVA model
                from llava.model.builder import load_pretrained_model
                from llava.mm_utils import get_model_name_from_path
                model_name = get_model_name_from_path(model_path)
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    device=GPU_DEVICE
                )
                print(f"LLaVA model loaded on device: {GPU_DEVICE}")

            model_loaded = True

            # Clean up after loading
            cleanup_gpu_memory()
        elif not samples_needing_processing:
            print("All samples are cached, skipping model loading")
            model_name = model_path  # Set model_name even when not loading

        # Process training pool with caching (generate all responses for the pool)
        safemt_training_pool = []
        if training_pool:
            print(f"\n--- Processing {len(training_pool)} SafeMTData TRAINING POOL samples ---")
            for idx, sample in enumerate(training_pool):

                multi_turn_queries = sample['multi_turn_queries']
                if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                    # Check cache first
                    sample_hash = get_sample_hash(sample)
                    cached_response = load_cached_safemtdata_response(sample_hash)

                    if cached_response is not None:
                        # Using cached response
                        final_conversation_text = cached_response
                    else:
                        print(f"  Generating new response for training pool sample {idx + 1}")
                        # Build REAL multi-turn conversation with CONSISTENT hidden state extraction
                        final_conv = build_dynamic_multiturn_conversation_for_detection(
                            multi_turn_queries, tokenizer, model, model_name
                        )
                        # Get the final conversation text for feature extraction
                        if REQUESTED_MODEL == 'qwen':
                            final_conversation_text = final_conv  # Already a string for Qwen
                        else:
                            final_conversation_text = final_conv.get_prompt()  # LLaVA conversation object
                        # Cache the response
                        cache_safemtdata_response(sample_hash, final_conversation_text)

                    safemt_sample = {
                        "txt": final_conversation_text,
                        "img": None,
                        "toxicity": 1,
                        "category": sample.get('category', 'multi_turn_jailbreak'),
                        "original_id": sample.get('id', -1),
                        "num_turns": len(multi_turn_queries),
                        "final_query": multi_turn_queries[-1]
                    }
                    safemt_training_pool.append(safemt_sample)

        # Process test samples with caching (consistent across all runs)
        print(f"\n--- Processing {len(test_samples)} SafeMTData TEST samples ---")
        safemt_test = []
        for idx, sample in enumerate(test_samples):

            multi_turn_queries = sample['multi_turn_queries']
            if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                # Check cache first
                sample_hash = get_sample_hash(sample)
                cached_response = load_cached_safemtdata_response(sample_hash)

                if cached_response is not None:
                    # Using cached response
                    final_conversation_text = cached_response
                else:
                    print(f"  Generating new response for test sample {idx + 1}")
                    # Build REAL multi-turn conversation with CONSISTENT hidden state extraction
                    final_conv = build_dynamic_multiturn_conversation_for_detection(
                        multi_turn_queries, tokenizer, model, model_name
                    )
                    # Get the final conversation text for feature extraction
                    if REQUESTED_MODEL == 'qwen':
                        final_conversation_text = final_conv  # Already a string for Qwen
                    else:
                        final_conversation_text = final_conv.get_prompt()  # LLaVA conversation object
                    # Cache the response
                    cache_safemtdata_response(sample_hash, final_conversation_text)

                safemt_sample = {
                    "txt": final_conversation_text,
                    "img": None,
                    "toxicity": 1,
                    "category": sample.get('category', 'multi_turn_jailbreak'),
                    "original_id": sample.get('id', -1),
                    "num_turns": len(multi_turn_queries),
                    "final_query": multi_turn_queries[-1]
                }
                safemt_test.append(safemt_sample)

        print(f"\nConverted SafeMTData (seed={random_seed}): {len(safemt_training_pool)} training pool + {len(safemt_test)} test samples")

        # Clean up model memory after SafeMTData processing (only if we loaded it)
        if model_loaded and model is not None:
            if model_type == 'qwen':
                del model, tokenizer
            else:
                del model, tokenizer, image_processor
            cleanup_gpu_memory()
            print("Cleaned up SafeMTData model from memory")

        return safemt_training_pool, safemt_test

    except Exception as e:
        print(f"Error loading SafeMTData: {e}")
        # Clean up on error too (only if we loaded the model)
        if 'model_loaded' in locals() and model_loaded and 'model' in locals() and model is not None:
            if model_type == 'qwen':
                del model, tokenizer
            else:
                del model, tokenizer, image_processor
            cleanup_gpu_memory()
        return [], []

def run_variable_training_experiment(tokenizer, model, model_name, model_type='llava', num_seeds=1):
    """Run the variable training data experiment: 0 to 50 training samples, configurable runs each"""
    print("="*80)
    print(f"VARIABLE TRAINING DATA EXPERIMENT: SAFEMTDATA ADAPTABILITY ANALYSIS ({model_type.upper()})")
    print("="*80)
    print(f"Configuration: {num_seeds} seed(s), training sizes 0-50 (step 5)")
    print("="*80)

    # Experimental configuration
    training_amounts = list(range(0, 51, 5))  # [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    all_seeds = [42, 123, 456, 789, 999]
    random_seeds = all_seeds[:num_seeds]  # Use only the requested number of seeds

    print(f"Training amounts: {training_amounts}")
    print(f"Random seeds: {random_seeds}")
    print(f"Max training samples per seed: {max(training_amounts)}")
    print(f"Test samples per seed: 100")

    # Results storage
    all_results = []

    # Load original datasets (consistent across all experiments)
    print("\nLoading original datasets...")
    benign_training, malicious_training = prepare_balanced_training()
    original_test_datasets = prepare_balanced_evaluation()

    # Initialize model-specific feature extractor ONCE for all experiments
    print("Initializing feature extractor...")
    extractor = get_model_specific_extractor(model_name, model_type)

    # Set optimal layer based on model type
    if model_type == 'qwen':
        optimal_layer = 21  # Optimal layer for Qwen
    else:
        optimal_layer = 16  # Optimal layer for LLaVA

    # Pre-extract features for all original datasets (they're consistent across runs)
    print("Pre-extracting features for original datasets...")
    original_training_datasets = {**benign_training, **malicious_training}
    original_all_datasets = {**original_training_datasets, **original_test_datasets}

    original_hidden_states = {}
    original_labels = {}

    for dataset_name, samples in original_all_datasets.items():
        print(f"Pre-extracting features for {dataset_name} ({len(samples)} samples)...")

        # Use smaller batch sizes for large datasets to manage memory
        batch_size = 25 if len(samples) > 5000 else 50
        memory_cleanup_freq = 5 if len(samples) > 5000 else 10

        hidden_states, labels, _ = extractor.extract_hidden_states(
            samples, f"{dataset_name}", layer_start=optimal_layer, layer_end=optimal_layer, use_cache=True,
            batch_size=batch_size, memory_cleanup_freq=memory_cleanup_freq,
            experiment_name="safemtdata_variable_training"
        )
        original_hidden_states[dataset_name] = hidden_states
        original_labels[dataset_name] = labels

        # Clean up GPU memory after each dataset
        cleanup_gpu_memory()

    # CRITICAL: Manually unload the model from HiddenStateExtractor to free GPU memory
    print("Unloading model from HiddenStateExtractor to free GPU memory...")
    if hasattr(extractor, '_unload_model'):
        extractor._unload_model()
    else:
        # Fallback for LLaVA extractor
        if hasattr(extractor, 'model') and extractor.model is not None:
            del extractor.model
        if hasattr(extractor, 'tokenizer') and extractor.tokenizer is not None:
            del extractor.tokenizer
        if hasattr(extractor, 'image_processor') and extractor.image_processor is not None:
            del extractor.image_processor
        extractor.model = None
        extractor.tokenizer = None
        extractor.image_processor = None
        cleanup_gpu_memory()
    print("HiddenStateExtractor model unloaded")

    # Set initial projection dimensions based on model type
    CONFIG.set_model_dimensions(model_type)

    # Update CONFIG with actual input dimensions based on extracted features
    if original_hidden_states:
        first_dataset = next(iter(original_hidden_states.values()))
        if optimal_layer in first_dataset and len(first_dataset[optimal_layer]) > 0:
            actual_input_dim = len(first_dataset[optimal_layer][0])
            CONFIG.set_model_dimensions(model_type, input_dim=actual_input_dim)
            print(f"Updated projection input dimension to {actual_input_dim} based on actual features")

    # Run experiments for each seed
    for seed_idx, random_seed in enumerate(random_seeds):
        print(f"\n{'='*100}")
        print(f"SEED {seed_idx + 1}/{len(random_seeds)}: {random_seed}")
        print(f"{'='*100}")

        # Load SafeMTData pool for this seed (training pool + test set)
        safemt_training_pool, safemt_test = load_safemtdata_for_variable_training_experiment(
            model_name, model_type, max_training=50, num_test=100, random_seed=random_seed
        )

        if not safemt_training_pool or not safemt_test:
            print(f"Failed to load SafeMTData for seed {random_seed}, skipping...")
            continue

        # Create a new extractor for SafeMTData feature extraction (since we unloaded the previous one)
        print("Creating new feature extractor for SafeMTData...")
        safemt_extractor = get_model_specific_extractor(model_name, model_type)

        # Extract features for SafeMTData samples (with caching)
        all_hidden_states = {}
        all_labels = {}

        # Add original datasets
        for dataset_name in original_hidden_states:
            all_hidden_states[dataset_name] = original_hidden_states[dataset_name]
            all_labels[dataset_name] = original_labels[dataset_name]

        # Extract features for SafeMTData training pool and test set
        if safemt_training_pool:
            print(f"Extracting features for SafeMTData_Training ({len(safemt_training_pool)} samples)...")
            hidden_states, labels, _ = safemt_extractor.extract_hidden_states(
                safemt_training_pool, "SafeMTData_Training", layer_start=optimal_layer, layer_end=optimal_layer, use_cache=True,
                batch_size=25, memory_cleanup_freq=5,
                experiment_name="safemtdata_variable_training"
            )

            # Store SafeMTData training features (projection will be applied later in the loop)
            all_hidden_states["SafeMTData_Training"] = {optimal_layer: hidden_states[optimal_layer]}
            all_labels["SafeMTData_Training"] = labels
            cleanup_gpu_memory()

        if safemt_test:
            print(f"Extracting features for SafeMTData_Test ({len(safemt_test)} samples)...")
            hidden_states, labels, _ = safemt_extractor.extract_hidden_states(
                safemt_test, "SafeMTData_Test", layer_start=optimal_layer, layer_end=optimal_layer, use_cache=True,
                batch_size=25, memory_cleanup_freq=5,
                experiment_name="safemtdata_variable_training"
            )

            # Store SafeMTData test features (projection will be applied later in the loop)
            all_hidden_states["SafeMTData_Test"] = {optimal_layer: hidden_states[optimal_layer]}
            all_labels["SafeMTData_Test"] = labels
            cleanup_gpu_memory()

        # Clean up SafeMTData extractor
        if hasattr(safemt_extractor, '_unload_model'):
            safemt_extractor._unload_model()
        else:
            # Fallback for LLaVA extractor
            if hasattr(safemt_extractor, 'model') and safemt_extractor.model is not None:
                del safemt_extractor.model
            if hasattr(safemt_extractor, 'tokenizer') and safemt_extractor.tokenizer is not None:
                del safemt_extractor.tokenizer
            if hasattr(safemt_extractor, 'image_processor') and safemt_extractor.image_processor is not None:
                del safemt_extractor.image_processor
            safemt_extractor.model = None
            safemt_extractor.tokenizer = None
            safemt_extractor.image_processor = None
            cleanup_gpu_memory()
        del safemt_extractor

        # Run experiments for different training amounts
        for training_amount in training_amounts:
            print(f"\n--- Training Amount: {training_amount} SafeMTData samples ---")

            # Create dataset configurations (same as MCD script)
            expanded_malicious_training = malicious_training.copy()
            if training_amount > 0 and "SafeMTData_Training" in all_hidden_states:
                # Take only the first 'training_amount' samples from SafeMTData
                safemt_features = all_hidden_states["SafeMTData_Training"][optimal_layer]
                safemt_labels = all_labels["SafeMTData_Training"]

                print(f"DEBUG: SafeMTData labels: {set(safemt_labels)} (first 10: {safemt_labels[:10]})")

                if len(safemt_features) >= training_amount:
                    selected_features = safemt_features[:training_amount]
                    selected_labels = safemt_labels[:training_amount]
                    print(f"Using {training_amount} SafeMTData training samples")
                else:
                    selected_features = safemt_features
                    selected_labels = safemt_labels
                    print(f"Using {len(safemt_features)} SafeMTData training samples (less than requested {training_amount})")

                # Create SafeMTData training dataset entry
                safemt_training_samples = [{"txt": f"safemt_sample_{i}", "toxicity": 1} for i in range(len(selected_features))]
                expanded_malicious_training["SafeMTData_Training"] = safemt_training_samples

                # CRITICAL: Update hidden states and labels with ONLY the selected subset for this training amount
                # This ensures projection training uses the correct amount of SafeMTData samples
                current_safemt_hidden_states = {optimal_layer: selected_features}
                current_safemt_labels = selected_labels
            else:
                print("Using 0 SafeMTData training samples (baseline)")
                # Initialize empty variables for consistency
                current_safemt_hidden_states = None
                current_safemt_labels = None

            # Create training datasets configuration
            training_datasets = {**benign_training, **expanded_malicious_training}

            print(f"Training datasets: {len(benign_training)} benign + {len(expanded_malicious_training)} malicious")

            # Separate training data for this configuration
            training_hidden_states = {k: v for k, v in original_hidden_states.items() if k in benign_training}
            training_labels = {k: v for k, v in original_labels.items() if k in benign_training}

            # Add original malicious training data
            for dataset_name in malicious_training:
                if dataset_name in original_hidden_states:
                    training_hidden_states[dataset_name] = original_hidden_states[dataset_name]
                    training_labels[dataset_name] = original_labels[dataset_name]

            # Add SafeMTData training data with the correct subset for this training amount
            if training_amount > 0 and current_safemt_hidden_states is not None:
                training_hidden_states["SafeMTData_Training"] = current_safemt_hidden_states
                training_labels["SafeMTData_Training"] = current_safemt_labels
                print(f"DEBUG: Added {len(current_safemt_labels)} SafeMTData samples to projection training")

            # CRITICAL: Train learned projection for this specific training configuration
            print("Training learned projection for this training configuration...")

            # Prepare data for projection training
            projection_features_dict = {}
            projection_labels_dict = {}

            for dataset_name in training_datasets.keys():
                if dataset_name in training_hidden_states and dataset_name in training_labels:
                    features = training_hidden_states[dataset_name]
                    labels = training_labels[dataset_name]

                    # Extract features from dict structure
                    if isinstance(features, dict) and optimal_layer in features:
                        features_array = features[optimal_layer]
                    else:
                        features_array = features

                    # Ensure features are numpy arrays
                    if isinstance(features_array, list):
                        features_array = np.array(features_array)

                    projection_features_dict[dataset_name] = features_array
                    projection_labels_dict[dataset_name] = labels

            # Train projection model for this training configuration
            projection_model, _ = train_learned_projection(
                projection_features_dict, projection_labels_dict, random_seed=random_seed
            )

            print("Applying learned projections...")

            # Apply projection to training data
            training_features_for_projection = {}
            for dataset_name, features in training_hidden_states.items():
                if isinstance(features, dict) and optimal_layer in features:
                    training_features_for_projection[dataset_name] = features[optimal_layer]
                else:
                    training_features_for_projection[dataset_name] = features

            projected_training_hidden_states = apply_learned_projection(projection_model, training_features_for_projection)

            # Apply projection to test data (use original test datasets)
            test_features_for_projection = {}
            for dataset_name in original_test_datasets:
                if dataset_name in original_hidden_states:
                    hidden_states = original_hidden_states[dataset_name]
                    if isinstance(hidden_states, dict) and optimal_layer in hidden_states:
                        test_features_for_projection[dataset_name] = hidden_states[optimal_layer]
                    else:
                        test_features_for_projection[dataset_name] = hidden_states

            # Also include SafeMTData test if available
            if "SafeMTData_Test" in all_hidden_states:
                test_features_for_projection["SafeMTData_Test"] = all_hidden_states["SafeMTData_Test"][optimal_layer]

            projected_test_hidden_states = apply_learned_projection(projection_model, test_features_for_projection)

            # Prepare data in the format expected by KCDDetector using projected features
            benign_data = {}
            malicious_data = {}

            # Process benign datasets using projected training features
            for dataset_name in benign_training:
                if dataset_name in projected_training_hidden_states:
                    features = projected_training_hidden_states[dataset_name]
                    labels = training_labels[dataset_name]

                    print(f"DEBUG: Processing {dataset_name}, labels: {set(labels)} (total: {len(labels)})")

                    # Get benign samples (label == 0)
                    benign_features = [features[i] for i, label in enumerate(labels) if label == 0]
                    if benign_features:
                        benign_data[dataset_name] = benign_features

            # Process malicious datasets using projected training features
            for dataset_name in expanded_malicious_training:
                if dataset_name in projected_training_hidden_states:
                    features = projected_training_hidden_states[dataset_name]
                    labels = training_labels[dataset_name]

                    print(f"DEBUG: Processing {dataset_name}, labels: {set(labels)} (total: {len(labels)})")

                    # Get malicious samples (label == 1)
                    malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
                    print(f"DEBUG: Found {len(malicious_features)} malicious samples in {dataset_name}")

                    if malicious_features:
                        malicious_data[f"{dataset_name}_malicious"] = malicious_features

            # Train KCD detector with k=50
            print("Training KCD detector with k=50...")
            detector = KCDDetector(k=50, use_gpu=True)
            detector.fit_training_data(benign_data, malicious_data)

            # Create validation data for threshold optimization (same as original KCD)
            print("Preparing validation data...")
            val_features = []
            val_labels = []

            # Sample from benign training data (same as original)
            for _, features in benign_data.items():
                sample_size = min(100, len(features))
                if sample_size > 0:
                    sampled_features = random.sample(features, sample_size)
                    val_features.extend(sampled_features)
                    val_labels.extend([0] * len(sampled_features))

            # Sample from malicious training data (same as original)
            for dataset_name, features in malicious_data.items():
                sample_size = min(100, len(features))
                if sample_size > 0:
                    sampled_features = random.sample(features, sample_size)
                    val_features.extend(sampled_features)
                    val_labels.extend([1] * len(sampled_features))

            print(f"Validation set: {len(val_features)} samples ({val_labels.count(0)} benign, {val_labels.count(1)} malicious)")

            # Optimize threshold
            print("Optimizing detection threshold...")
            detector.fit_threshold(val_features, val_labels)

            # Combined evaluation using ONLY original test datasets (same composition across all training amounts)
            all_test_features = []
            all_test_labels = []

            for dataset_name in original_test_datasets.keys():
                if dataset_name in projected_test_hidden_states:
                    features = projected_test_hidden_states[dataset_name]
                    labels = original_labels[dataset_name]
                    print(f"{dataset_name}: {len(features)} samples")

                    # Collect for combined evaluation
                    all_test_features.extend(features)
                    all_test_labels.extend(labels)

            # SafeMTData specific evaluation (separate from combined)
            safemt_accuracy = 0.0
            if "SafeMTData_Test" in projected_test_hidden_states:
                safemt_features = projected_test_hidden_states["SafeMTData_Test"]
                safemt_labels = all_labels["SafeMTData_Test"]
                safemt_predictions, safemt_scores = detector.predict(safemt_features)
                safemt_accuracy = accuracy_score(safemt_labels, safemt_predictions)
                print(f"SafeMTData_Test: Acc={safemt_accuracy:.4f} ({len(safemt_features)} samples)")

            # Combined evaluation (original test datasets only)
            combined_accuracy = 0.0
            combined_f1 = 0.0
            combined_auroc = 0.0

            if all_test_features and all_test_labels:
                combined_predictions, combined_scores = detector.predict(all_test_features)
                combined_accuracy = accuracy_score(all_test_labels, combined_predictions)
                combined_f1 = f1_score(all_test_labels, combined_predictions)
                combined_auroc = roc_auc_score(all_test_labels, combined_scores)
                print(f"Combined: Acc={combined_accuracy:.4f}, F1={combined_f1:.4f}, AUROC={combined_auroc:.4f}")

            # Store results in format compatible with MCD aggregation
            experiment_result = {
                'num_training': training_amount,
                'run': 1,  # Single run per seed for now
                'seed': random_seed,
                'combined_accuracy': combined_accuracy,
                'combined_f1': combined_f1,
                'combined_auroc': combined_auroc,
                'safemt_accuracy': safemt_accuracy,
                'threshold': detector.threshold
            }
            all_results.append(experiment_result)

            # Clean up detector
            del detector
            cleanup_gpu_memory()

    # Print summary results
    # Save all results with model-specific filename (convert numpy types to native Python types)
    results_file = f"results/safemt_kcd_{model_type}.json"
    os.makedirs("results", exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN and infinity values
            if np.isnan(obj) or np.isinf(obj):
                return None  # Convert NaN/inf to None for JSON compatibility
            return round(float(obj), 3)  # Round to 3 decimal places
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif obj != obj:  # Check for NaN (NaN != NaN is True)
            return None
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        elif isinstance(obj, float):
            return round(obj, 3)  # Round regular floats to 3 decimal places
        else:
            return obj

    serializable_results = convert_numpy_types(all_results)

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")

    # Print final summary
    print("\nFINAL SUMMARY:")
    for training_amount in training_amounts:
        training_results = [r for r in all_results if r['num_training'] == training_amount]
        if training_results:
            avg_combined_acc = np.mean([r['combined_accuracy'] for r in training_results])
            avg_safemt_acc = np.mean([r['safemt_accuracy'] for r in training_results])
            print(f"{training_amount:3d} training: Combined={avg_combined_acc:.3f}, SafeMT={avg_safemt_acc:.3f}")

    # Aggregate, output CSV, and plot
    aggregate_and_report_kcd_results(all_results, training_amounts, results_dir="results", model_type=model_type)

    # Final cleanup
    del original_hidden_states, original_labels
    cleanup_gpu_memory()

    return all_results

def aggregate_and_report_kcd_results(all_results, training_amounts, results_dir="results", model_type='llava'):
    """
    Aggregate KCD results by training size, print summary, write CSV, and plot trends.
    """
    os.makedirs(results_dir, exist_ok=True)
    summary = []

    for training_amount in training_amounts:
        runs = [r for r in all_results if r['num_training'] == training_amount and 'error' not in r]
        if not runs:
            continue

        # Extract metrics directly from results structure
        combined_accuracies = [r['combined_accuracy'] for r in runs]
        combined_f1s = [r['combined_f1'] for r in runs]
        combined_aurocs = [r['combined_auroc'] for r in runs]
        safemt_accuracies = [r['safemt_accuracy'] for r in runs]

        if combined_accuracies:
            # Use nanmean and nanstd to handle NaN values gracefully
            avg_combined_acc = np.nanmean(combined_accuracies)
            std_combined_acc = np.nanstd(combined_accuracies)
            avg_combined_f1 = np.nanmean(combined_f1s)
            std_combined_f1 = np.nanstd(combined_f1s)
            avg_safemt_acc = np.nanmean(safemt_accuracies)
            std_safemt_acc = np.nanstd(safemt_accuracies)
            avg_auroc = np.nanmean(combined_aurocs)
            std_auroc = np.nanstd(combined_aurocs)

            summary.append({
                'num_training': training_amount,
                'avg_combined_acc': avg_combined_acc,
                'std_combined_acc': std_combined_acc,
                'avg_combined_f1': avg_combined_f1,
                'std_combined_f1': std_combined_f1,
                'avg_safemt_acc': avg_safemt_acc,
                'std_safemt_acc': std_safemt_acc,
                'avg_auroc': avg_auroc,
                'std_auroc': std_auroc
            })

    # Print summary table
    print("\nAggregated Results (mean ± std):")
    print(f"{'Train#':>7}  {'Overall Acc':>12}  {'Overall F1':>12}  {'SafeMT Acc':>12}  {'AUROC':>10}")
    for row in summary:
        print(f"{row['num_training']:7d}  {row['avg_combined_acc']:.4f} ± {row['std_combined_acc']:.4f}  "
              f"{row['avg_combined_f1']:.4f} ± {row['std_combined_f1']:.4f}  "
              f"{row['avg_safemt_acc']:.4f} ± {row['std_safemt_acc']:.4f}  "
              f"{row['avg_auroc']:.4f} ± {row['std_auroc']:.4f}")

    # Write CSV with model-specific filename
    csv_path = os.path.join(results_dir, f"safemt_kcd_{model_type}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["num_training", "avg_combined_acc", "std_combined_acc", "avg_combined_f1", "std_combined_f1",
                        "avg_safemt_acc", "std_safemt_acc", "avg_auroc", "std_auroc"])
        for row in summary:
            writer.writerow([
                row['num_training'],
                round(row['avg_combined_acc'], 3), round(row['std_combined_acc'], 3),
                round(row['avg_combined_f1'], 3), round(row['std_combined_f1'], 3),
                round(row['avg_safemt_acc'], 3), round(row['std_safemt_acc'], 3),
                round(row['avg_auroc'], 3), round(row['std_auroc'], 3)
            ])

    print(f"\nAggregated results written to: {csv_path}")

    # Generate trend plot
    generate_kcd_trend_plot(summary, results_dir, model_type)

def generate_kcd_trend_plot(summary, results_dir, model_type='llava'):
    """Generate trend plot from summary data with broken y-axis for KCD results"""
    print("Generating KCD trend plot...")

    # Extract data for plotting
    x = [row['num_training'] for row in summary]
    y_acc = [row['avg_combined_acc'] for row in summary]
    y_acc_std = [row['std_combined_acc'] for row in summary]
    y_f1 = [row['avg_combined_f1'] for row in summary]
    y_f1_std = [row['std_combined_f1'] for row in summary]
    y_mt_acc = [row['avg_safemt_acc'] for row in summary]
    y_mt_acc_std = [row['std_safemt_acc'] for row in summary]
    y_auroc = [row['avg_auroc'] for row in summary]
    y_auroc_std = [row['std_auroc'] for row in summary]

    # Create figure with broken axis - flatter to save space
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})

    # Plot data on both axes (without error bars)
    for ax in [ax1, ax2]:
        ax.plot(x, y_acc, label='Overall Accuracy', marker='o', linewidth=4, markersize=12)
        ax.plot(x, y_f1, label='Overall F1 Score', marker='d', linewidth=4, markersize=12)
        ax.plot(x, y_auroc, label='Overall AUROC', marker='^', linewidth=4, markersize=12)
        ax.plot(x, y_mt_acc, label='SafeMTData Accuracy', marker='s', linewidth=4, markersize=12)

        # Add horizontal dashed lines for baselines (dynamically based on model type)
        baselines = get_model_baselines(model_type)
        ax.axhline(baselines['overall_accuracy'], color='tab:blue', linestyle='--', linewidth=3.5, alpha=0.7, label='Baseline Overall Acc')
        ax.axhline(baselines['f1'], color='tab:orange', linestyle='--', linewidth=3.5, alpha=0.7, label='Baseline F1')
        ax.axhline(baselines['auroc'], color='tab:green', linestyle='--', linewidth=3.5, alpha=0.7, label='Baseline AUROC')

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=22)

    # Set different y-axis limits for each subplot
    ax1.set_ylim(0.8, 1.05)  # Top subplot: high values (0.8-1.05) with space above 1.0
    ax2.set_ylim(0.15, 0.30) 

    # Set custom y-ticks to avoid overlapping and maintain consistent scale
    ax1.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax2.set_yticks([0.15, 0.20, 0.25])

    # Hide the spines between ax1 and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Add diagonal lines to indicate the break
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Set labels and title
    ax2.set_xlabel('Number of SafeMTData Training Samples', fontsize=26, fontweight='bold')
    fig.suptitle(f'KCD Performance vs SafeMTData Training Size ({model_type.upper()})', fontsize=32, fontweight='bold', y=0.98)
    ax1.set_ylabel('Performance Score', fontsize=26, fontweight='bold')

    # Add legend to the bottom subplot (same as MCD script)
    legend = ax2.legend(fontsize=18, frameon=True, fancybox=True, shadow=True, loc='lower right')
    legend.get_frame().set_alpha(0.7)  # Make legend semi-transparent

    # Set x-axis ticks and labels
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i) for i in x])

    # Adjust layout and save
    plt.tight_layout()

    # Save plot with model-specific filename
    plot_path = os.path.join(results_dir, f"safemt_kcd_{model_type}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Trend plot saved to: {plot_path}")

    # Also save as PNG for easier viewing
    png_path = os.path.join(results_dir, f"safemt_kcd_{model_type}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Trend plot also saved as: {png_path}")

    plt.close()

def replot_kcd_from_csv(model_type='llava'):
    """Replot the KCD trend chart from existing CSV data without running experiments"""
    print("="*80)
    print(f"REPLOTTING KCD FROM EXISTING CSV DATA ({model_type.upper()})")
    print("="*80)

    # Look for existing results directory
    possible_dirs = [
        f"variable_training_results_{model_type}",
        "variable_training_results",
        "results",
        "../results"  # Look in parent directory as well
    ]

    results_dir = None
    for dir_name in possible_dirs:
        if os.path.exists(dir_name):
            results_dir = dir_name
            break

    if results_dir is None:
        print("Error: No results directory found. Looking for:")
        for dir_name in possible_dirs:
            print(f"  - {dir_name}")
        print("Please run the experiment first before using --replot")
        return

    # Look for summary CSV file (try multiple possible names)
    possible_summary_files = [
        f"safemt_kcd_{model_type}.csv",  # Model-specific file
    ]

    summary_file = None
    for filename in possible_summary_files:
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            summary_file = file_path
            break

    if summary_file is None:
        print("Error: No summary file found. Looking for:")
        for filename in possible_summary_files:
            print(f"  - {os.path.join(results_dir, filename)}")
        print("Please run the experiment first to generate the summary data")
        return

    print(f"Found results directory: {results_dir}")
    print(f"Found summary file: {summary_file}")

    # Load and replot
    try:
        import pandas as pd
        summary_df = pd.read_csv(summary_file)
        print(f"Loaded summary data with {len(summary_df)} rows")

        # Convert DataFrame back to list of dictionaries for compatibility
        summary = summary_df.to_dict('records')

        # Generate the plot
        generate_kcd_trend_plot(summary, results_dir, model_type)
        print("KCD replotting completed successfully!")

    except Exception as e:
        print(f"Error during replotting: {e}")
        print("Make sure the summary CSV file is valid and contains the expected columns")

def main():
    """Main function for SafeMTData variable training experiment with KCD"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='KCD SafeMTData Variable Training Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_kcd_mtdata.py --model llava --seeds 1    # LLaVA with 1 seed (preliminary)
  python main_kcd_mtdata.py --model qwen --seeds 3     # Qwen with 3 seeds
  python main_kcd_mtdata.py --model llava --seeds 5    # LLaVA with 5 seeds (full-scale)
  python main_kcd_mtdata.py -m llava -s 1             # Short form
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['llava', 'qwen'],
        default='llava',
        help='Model type to use (default: llava)'
    )

    parser.add_argument(
        '--seeds', '-s',
        type=int,
        default=1,
        choices=range(1, 6),
        metavar='[1-5]',
        help='Number of random seeds to run (1-5, default: 1 for preliminary investigation)'
    )

    parser.add_argument(
        '--replot',
        action='store_true',
        help='Only regenerate plots from existing CSV data (no training)'
    )

    args = parser.parse_args()

    model_type = args.model
    num_seeds = args.seeds
    replot_only = args.replot

    # Handle replot mode
    if replot_only:
        replot_kcd_from_csv(model_type)
        sys.exit(0)

    # Set model path based on model type
    if model_type == 'qwen':
        model_path = "model/qwen2.5-vl-7b-instruct"
    else:
        model_path = "model/llava-v1.6-vicuna-7b/"

    print(f"Configuration:")
    print(f"  Model: {model_type.upper()} ({model_path})")
    print(f"  Seeds: {num_seeds} ({'preliminary' if num_seeds == 1 else 'full-scale'})")
    print(f"  Training sizes: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50")
    print(f"  Expected experiments: ~{11 * num_seeds} (11 per seed)")
    print()

    # For SafeMTData processing, we need tokenizer and model
    # We'll load them lazily inside the SafeMTData functions to avoid memory issues

    # Run the variable training experiment
    run_variable_training_experiment(None, None, model_path, model_type, num_seeds)

if __name__ == "__main__":
    main()
