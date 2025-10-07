#!/usr/bin/env python3
"""
Test the balanced OOD MCD k-means method against SafeMTData multi-turn jailbreaking dataset.
This script integrates SafeMTData with the existing detection framework.
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
        'overall_accuracy': 0.8706,
        'f1': 0.9193,
        'auroc': 0.9708
    },
    'qwen': {
        'overall_accuracy': 0.8664,
        'f1': 0.8803, 
        'auroc': 0.9773 
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

from main_mcd_kmeans import (
    MCDDetector, prepare_kmeans_data_structure,
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

def load_safemtdata_for_adaptability_test(tokenizer, model, model_name, num_training=200, num_test=50):
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

        return safemt_training, safemt_test

    except Exception as e:
        print(f"Error loading SafeMTData: {e}")
        return [], []

def load_safemtdata_for_variable_training_experiment(tokenizer, model, model_name, max_training=50, num_test=100, random_seed=42):
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
        if samples_needing_processing and (tokenizer is None or model is None):
            print(f"Loading {model_type.upper()} model and tokenizer for SafeMTData processing ({len(samples_needing_processing)} uncached samples)...")

            if model_type == 'qwen':
                # Load Qwen model
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.float16, device_map="auto"
                )
                tokenizer = AutoProcessor.from_pretrained(model_name)
                print(f"Qwen model loaded on device: {GPU_DEVICE}")
            else:
                # Load LLaVA model
                from llava.model.builder import load_pretrained_model
                from llava.mm_utils import get_model_name_from_path
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=model_name,
                    model_base=None,
                    model_name=get_model_name_from_path(model_name),
                    device=GPU_DEVICE
                )
                print(f"LLaVA model loaded on device: {GPU_DEVICE}")

            model_loaded = True

            # Clean up after loading
            cleanup_gpu_memory()
        elif not samples_needing_processing:
            print("All samples are cached, skipping model loading")

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
        if model_loaded and 'model' in locals():
            del model, tokenizer, image_processor
            cleanup_gpu_memory()
            print("Cleaned up SafeMTData model from memory")

        return safemt_training_pool, safemt_test

    except Exception as e:
        print(f"Error loading SafeMTData: {e}")
        # Clean up on error too (only if we loaded the model)
        if 'model_loaded' in locals() and model_loaded and 'model' in locals():
            del model, tokenizer, image_processor
            cleanup_gpu_memory()
        return [], []

def create_adaptability_datasets(tokenizer, model, model_name):
    """Create training and test datasets for adaptability experiment"""
    print("Creating adaptability experiment datasets...")

    # Load original training and test datasets (baseline)
    print("Loading original datasets...")
    benign_training, malicious_training = prepare_balanced_training()
    original_test_datasets = prepare_balanced_evaluation()

    # Load SafeMTData with training/test split (NO overlap)
    print("Loading SafeMTData with training/test split...")
    safemt_training, safemt_test = load_safemtdata_for_adaptability_test(
        tokenizer, model, model_name, num_training=200, num_test=50
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

def aggregate_and_report_results(all_results, training_amounts, results_dir="results"):
    """
    Aggregate results by training size, print summary, write CSV, and plot trends.
    """
    os.makedirs(results_dir, exist_ok=True)
    summary = []
    for num_training in training_amounts:
        runs = [r for r in all_results if r['num_training'] == num_training and 'error' not in r]
        if not runs:
            continue
        avg_combined_acc = np.mean([r['combined_accuracy'] for r in runs])
        std_combined_acc = np.std([r['combined_accuracy'] for r in runs])
        avg_combined_f1 = np.mean([r['combined_f1'] for r in runs])
        std_combined_f1 = np.std([r['combined_f1'] for r in runs])
        avg_safemt_acc = np.mean([r['safemt_accuracy'] for r in runs])
        std_safemt_acc = np.std([r['safemt_accuracy'] for r in runs])
        avg_auroc = np.mean([r['combined_auroc'] for r in runs])
        std_auroc = np.std([r['combined_auroc'] for r in runs])
        summary.append({
            'num_training': num_training,
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
    csv_path = os.path.join(results_dir, f"safemt_mcd_{model_type}.csv")
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
    generate_trend_plot(summary, results_dir, model_type)

def generate_trend_plot(summary, results_dir, model_type='llava'):
    """Generate trend plot from summary data with broken y-axis"""
    print("Generating trend plot...")

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

    # Plot data on both axes
    for ax in [ax1, ax2]:
        ax.errorbar(x, y_acc, yerr=y_acc_std, label='Overall Accuracy', marker='o', capsize=8, linewidth=4, markersize=12)
        ax.errorbar(x, y_f1, yerr=y_f1_std, label='Overall F1 Score', marker='d', capsize=8, linewidth=4, markersize=12)
        ax.errorbar(x, y_auroc, yerr=y_auroc_std, label='Overall AUROC', marker='^', capsize=8, linewidth=4, markersize=12)
        ax.errorbar(x, y_mt_acc, yerr=y_mt_acc_std, label='SafeMTData Accuracy', marker='s', capsize=8, linewidth=4, markersize=12)

        # Add horizontal dashed lines for baselines (dynamically based on model type)
        baselines = get_model_baselines(model_type)
        ax.axhline(baselines['overall_accuracy'], color='tab:blue', linestyle='--', linewidth=3.5, alpha=0.7, label='Baseline Overall Acc')
        ax.axhline(baselines['f1'], color='tab:orange', linestyle='--', linewidth=3.5, alpha=0.7, label='Baseline F1')
        ax.axhline(baselines['auroc'], color='tab:green', linestyle='--', linewidth=3.5, alpha=0.7, label='Baseline AUROC')

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=22)

    # Set different y-axis limits for each subplot
    ax1.set_ylim(0.8, 1.05)  # Top subplot: high values (0.8-1.05) with space above 1.0
    ax2.set_ylim(0.15, 0.3)

    # Set custom y-ticks to avoid overlapping and maintain consistent scale
    # Top subplot: 0.8 to 1.0 in 0.05 increments (avoiding 1.05)
    ax1.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    # Bottom subplot: 0.0 to 0.15 in 0.05 increments (avoiding 0.15 to prevent overlap)
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

    # Formatting
    ax2.set_xlabel('Number of SafeMTData Training Samples', fontsize=28, fontweight='bold')
    fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical', fontsize=28, fontweight='bold')
    fig.suptitle(f'MCD Performance vs. SafeMTData Training Size ({model_type.upper()})', fontsize=32, fontweight='bold', y=0.95)

    # Add legend to the bottom right of the bottom subplot - smaller and transparent
    legend = ax2.legend(fontsize=18, frameon=True, fancybox=True, shadow=True, loc='lower right')
    legend.get_frame().set_alpha(0.7)  # Make legend semi-transparent

    # Save both PNG and PDF versions with model-specific filenames
    fig_path_png = os.path.join(results_dir, f"safemt_mcd_{model_type}.png")
    fig_path_pdf = os.path.join(results_dir, f"safemt_mcd_{model_type}.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Trend plot saved to: {fig_path_png}")
    print(f"Trend plot (PDF) saved to: {fig_path_pdf}")
    plt.close()

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
    del extractor
    print("Model unloaded, GPU memory freed")
    
    # Pre-generate SafeMTData pools for each seed (once per seed, not per training size)
    safemt_pools = {}
    safemt_tests = {}

    for seed in random_seeds:
        print(f"\n--- Generating SafeMTData pool for seed {seed} ---")
        safemt_training_pool, safemt_test = load_safemtdata_for_variable_training_experiment(
            tokenizer, model, model_name, max_training=max(training_amounts), num_test=100, random_seed=seed
        )
        safemt_pools[seed] = safemt_training_pool
        safemt_tests[seed] = safemt_test
        print(f"Generated pool for seed {seed}: {len(safemt_training_pool)} training, {len(safemt_test)} test")

    for num_training in training_amounts:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {num_training} SAFEMTDATA TRAINING SAMPLES")
        print(f"{'='*60}")

        run_results = []

        for run_idx, seed in enumerate(random_seeds):
            print(f"\n--- Run {run_idx + 1}/{len(random_seeds)} (seed={seed}) ---")

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            try:
                safemt_training_pool = safemt_pools[seed]
                safemt_test = safemt_tests[seed]

                if num_training > 0:
                    safemt_training = safemt_training_pool[:num_training]
                    print(f"Using {num_training} training samples from pool")
                else:
                    safemt_training = []
                    print("Using 0 training samples (baseline)")

                print(f"Configuration: {len(safemt_training)} training, {len(safemt_test)} test")
                
                # Extract features ONLY for SafeMTData (original features already extracted)
                all_hidden_states = original_hidden_states.copy()
                all_labels = original_labels.copy()
                
                # Create a new extractor for SafeMTData features (since we unloaded the previous one)
                safemt_extractor = None
                if safemt_training or safemt_test:
                    print("Creating new extractor for SafeMTData feature extraction...")
                    safemt_extractor = get_model_specific_extractor(model_name, model_type)
                
                # Extract features for SafeMTData training samples (if any)
                if safemt_training:
                    print(f"Extracting features for SafeMTData_Training ({len(safemt_training)} samples)...")
                    hidden_states, labels, _ = safemt_extractor.extract_hidden_states(
                        safemt_training, "SafeMTData_Training", layer_start=optimal_layer, layer_end=optimal_layer, use_cache=True,
                        batch_size=25, memory_cleanup_freq=5,
                        experiment_name="safemtdata_variable_training"
                    )
                    all_hidden_states["SafeMTData_Training"] = hidden_states
                    all_labels["SafeMTData_Training"] = labels
                    cleanup_gpu_memory()
                
                # Extract features for SafeMTData test samples
                if safemt_test:
                    print(f"Extracting features for SafeMTData_Test ({len(safemt_test)} samples)...")
                    hidden_states, labels, _ = safemt_extractor.extract_hidden_states(
                        safemt_test, "SafeMTData_Test", layer_start=optimal_layer, layer_end=optimal_layer, use_cache=True,
                        batch_size=25, memory_cleanup_freq=5,
                        experiment_name="safemtdata_variable_training"
                    )
                    all_hidden_states["SafeMTData_Test"] = hidden_states
                    all_labels["SafeMTData_Test"] = labels
                    cleanup_gpu_memory()
                
                # Clean up SafeMTData extractor
                if safemt_extractor is not None:
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
                        cleanup_gpu_memory()
                    del safemt_extractor
                
                # Create dataset configurations
                expanded_malicious_training = malicious_training.copy()
                if safemt_training:
                    expanded_malicious_training["SafeMTData_Training"] = safemt_training
                
                test_datasets = original_test_datasets.copy()
                if safemt_test:
                    test_datasets["SafeMTData_Test"] = safemt_test
                
                training_datasets = {**benign_training, **expanded_malicious_training}
                
                print(f"Training datasets: {len(benign_training)} benign + {len(expanded_malicious_training)} malicious")
                print(f"Test datasets: {len(test_datasets)} total")
                
                # Separate training and test data
                training_hidden_states = {k: v for k, v in all_hidden_states.items() if k in training_datasets}
                training_labels = {k: v for k, v in all_labels.items() if k in training_datasets}
                
                test_hidden_states = {k: v for k, v in all_hidden_states.items() if k in test_datasets}
                test_labels = {k: v for k, v in all_labels.items() if k in test_datasets}
                
                # Apply learned projections (critical for performance!)
                print("Training learned projections...")

                # Prepare data for projection training (combine all training datasets)
                projection_features_dict = {}
                projection_labels_dict = {}

                for dataset_name in training_datasets.keys():
                    if dataset_name in training_hidden_states and dataset_name in training_labels:
                        features = training_hidden_states[dataset_name]
                        labels = training_labels[dataset_name]

                        # Extract features from dict structure (features are likely {layer_16: array})
                        if isinstance(features, dict):
                            # Use layer 16 features (the layer we extracted)
                            if 16 in features:
                                features_array = features[16]
                            else:
                                # Fallback: use first available layer
                                features_array = list(features.values())[0]
                        else:
                            features_array = features

                        # Ensure features are numpy arrays
                        if isinstance(features_array, list):
                            features_array = np.array(features_array)

                        projection_features_dict[dataset_name] = features_array
                        projection_labels_dict[dataset_name] = labels

                # CRITICAL: Update CONFIG with actual input dimensions based on extracted features
                if projection_features_dict:
                    first_dataset = next(iter(projection_features_dict.values()))
                    if len(first_dataset) > 0:
                        actual_input_dim = len(first_dataset[0])
                        CONFIG.set_model_dimensions(model_type, input_dim=actual_input_dim)
                        print(f"Updated projection input dimension to {actual_input_dim} based on actual features")

                projection_model, dataset_name_to_id = train_learned_projection(
                    projection_features_dict, projection_labels_dict, random_seed=seed
                )
                
                print("Applying learned projections...")
                
                # Extract arrays from dict structure for apply_learned_projection
                training_features_for_projection = {}
                test_features_for_projection = {}

                # Use the correct optimal layer based on model type
                optimal_layer_key = 21 if model_type == 'qwen' else 16

                for dataset_name, features in training_hidden_states.items():
                    if isinstance(features, dict) and optimal_layer_key in features:
                        training_features_for_projection[dataset_name] = features[optimal_layer_key]
                    else:
                        training_features_for_projection[dataset_name] = features

                for dataset_name, features in test_hidden_states.items():
                    if isinstance(features, dict) and optimal_layer_key in features:
                        test_features_for_projection[dataset_name] = features[optimal_layer_key]
                    else:
                        test_features_for_projection[dataset_name] = features
                
                projected_training_hidden_states = apply_learned_projection(projection_model, training_features_for_projection)
                projected_test_hidden_states = apply_learned_projection(projection_model, test_features_for_projection)
                
                # Prepare k-means data structure
                print("Preparing k-means clustering...")
                in_dist_data, ood_data = prepare_kmeans_data_structure(
                    training_datasets, projected_training_hidden_states, training_labels,
                    random_seed=seed, k_benign=8, k_malicious=1
                )
                
                # Train MCD detector
                print("Training MCD detector...")
                detector = MCDDetector(use_gpu=True)
                detector.fit_in_distribution(in_dist_data)
                detector.fit_ood_clusters(ood_data)
                
                # Create validation data for threshold optimization
                print("Optimizing threshold...")
                val_features = []
                val_labels = []
                
                # Sample from clusters (same as original script)
                for cluster_data in in_dist_data.values():
                    if len(cluster_data) > 0:
                        sample_size = min(100, len(cluster_data))
                        indices = np.linspace(0, len(cluster_data)-1, sample_size, dtype=int)
                        sampled_features = [cluster_data[i] for i in indices]
                        val_features.extend(sampled_features)
                        val_labels.extend([0] * len(sampled_features))
                
                for cluster_data in ood_data.values():
                    if len(cluster_data) > 0:
                        sample_size = min(100, len(cluster_data))
                        indices = np.linspace(0, len(cluster_data)-1, sample_size, dtype=int)
                        sampled_features = [cluster_data[i] for i in indices]
                        val_features.extend(sampled_features)
                        val_labels.extend([1] * len(sampled_features))
                
                # Fit threshold
                if val_features and len(set(val_labels)) > 1:
                    detector.fit_threshold(val_features, val_labels)
                else:
                    detector.threshold = 0.0
                
                # Evaluate on all test datasets
                print("Evaluating performance...")
                
                # Combined evaluation
                all_test_features = []
                all_test_labels = []
                for dataset_name, features in projected_test_hidden_states.items():
                    labels = test_labels[dataset_name]
                    all_test_features.extend(features)
                    all_test_labels.extend(labels)
                
                combined_predictions, combined_scores = detector.predict(all_test_features)
                
                combined_accuracy = accuracy_score(all_test_labels, combined_predictions)
                combined_f1 = f1_score(all_test_labels, combined_predictions)
                combined_auroc = roc_auc_score(all_test_labels, combined_scores)
                
                # SafeMTData specific evaluation
                safemt_accuracy = 0.0
                if "SafeMTData_Test" in projected_test_hidden_states:
                    safemt_features = projected_test_hidden_states["SafeMTData_Test"]
                    safemt_labels = test_labels["SafeMTData_Test"]
                    safemt_predictions, safemt_scores = detector.predict(safemt_features)
                    safemt_accuracy = accuracy_score(safemt_labels, safemt_predictions)
                
                # Store results
                result = {
                    'num_training': num_training,
                    'run': run_idx + 1,
                    'seed': seed,
                    'combined_accuracy': combined_accuracy,
                    'combined_f1': combined_f1,
                    'combined_auroc': combined_auroc,
                    'safemt_accuracy': safemt_accuracy,
                    'threshold': detector.threshold
                }
                run_results.append(result)
                
                print(f"Results: Combined Acc={combined_accuracy:.4f}, SafeMT Acc={safemt_accuracy:.4f}, AUROC={combined_auroc:.4f}")
                
                # Clean up large variables and GPU memory after each run
                del projected_training_hidden_states, projected_test_hidden_states
                del training_hidden_states, test_hidden_states, training_labels, test_labels
                del detector, projection_model, in_dist_data, ood_data
                del all_test_features, all_test_labels, combined_scores, combined_predictions
                if 'safemt_features' in locals():
                    del safemt_features, safemt_scores, safemt_predictions
                cleanup_gpu_memory()
                
            except Exception as e:
                print(f"Error in run {run_idx + 1}: {e}")
                
                # Clean up memory even on error
                cleanup_gpu_memory()
                
                result = {
                    'num_training': num_training,
                    'run': run_idx + 1,
                    'seed': seed,
                    'combined_accuracy': 0.0,
                    'combined_f1': 0.0,
                    'combined_auroc': 0.0,
                    'safemt_accuracy': 0.0,
                    'threshold': 0.0,
                    'error': str(e)
                }
                run_results.append(result)
        
        all_results.extend(run_results)
        
        # Print summary for this training amount
        if run_results:
            avg_combined_acc = np.mean([r['combined_accuracy'] for r in run_results])
            avg_safemt_acc = np.mean([r['safemt_accuracy'] for r in run_results])
            avg_auroc = np.mean([r['combined_auroc'] for r in run_results])
            
            std_combined_acc = np.std([r['combined_accuracy'] for r in run_results])
            std_safemt_acc = np.std([r['safemt_accuracy'] for r in run_results])
            std_auroc = np.std([r['combined_auroc'] for r in run_results])
            
            print(f"\n*** SUMMARY FOR {num_training} TRAINING SAMPLES ***")
            print(f"Combined Accuracy: {avg_combined_acc:.4f} ± {std_combined_acc:.4f}")
            print(f"SafeMTData Accuracy: {avg_safemt_acc:.4f} ± {std_safemt_acc:.4f}")
            print(f"AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")
        
        # Clean up after each training amount
        cleanup_gpu_memory()
    
    # Save all results with model-specific filename (round floating point values to 3 decimal places)
    results_file = f"results/safemt_mcd_{model_type}.json"
    os.makedirs("results", exist_ok=True)

    # Function to round floating point values to 3 decimal places
    def round_floats(obj):
        if isinstance(obj, float):
            return round(obj, 3)
        elif isinstance(obj, dict):
            return {key: round_floats(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [round_floats(item) for item in obj]
        else:
            return obj

    rounded_results = round_floats(all_results)

    with open(results_file, 'w') as f:
        json.dump(rounded_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    for num_training in training_amounts:
        training_results = [r for r in all_results if r['num_training'] == num_training]
        if training_results:
            avg_combined_acc = np.mean([r['combined_accuracy'] for r in training_results])
            avg_safemt_acc = np.mean([r['safemt_accuracy'] for r in training_results])
            print(f"{num_training:3d} training: Combined={avg_combined_acc:.3f}, SafeMT={avg_safemt_acc:.3f}")
    
    # Aggregate, output CSV, and plot
    aggregate_and_report_results(all_results, training_amounts, results_dir="results")
    # Final cleanup
    del original_hidden_states, original_labels
    cleanup_gpu_memory()
    
    return all_results

def replot_from_csv(model_type='llava'):
    """Replot the trend chart from existing CSV data without running experiments"""
    print("="*80)
    print(f"REPLOTTING FROM EXISTING CSV DATA ({model_type.upper()})")
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
        f"safemt_mcd_{model_type}.csv",  
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
        generate_trend_plot(summary, results_dir, model_type)
        print("Replotting completed successfully!")

    except Exception as e:
        print(f"Error during replotting: {e}")
        print("Make sure the summary CSV file is valid and contains the expected columns")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='MCD SafeMTData Variable Training Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_mcd_kmeans_mtdata.py --model llava --seeds 1    # LLaVA with 1 seed (preliminary)
  python main_mcd_kmeans_mtdata.py --model qwen --seeds 3     # Qwen with 3 seeds
  python main_mcd_kmeans_mtdata.py --model llava --seeds 5    # LLaVA with 5 seeds (full-scale)
  python main_mcd_kmeans_mtdata.py -m llava -s 1             # Short form
  python main_mcd_kmeans_mtdata.py --model llava --replot     # Replot existing LLaVA results
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
        replot_from_csv(model_type)
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
    print(f"  Expected responses: ~{150 * num_seeds} (150 per seed)")
    print()

    # Run the variable training experiment
    run_variable_training_experiment(None, None, model_path, model_type, num_seeds)
