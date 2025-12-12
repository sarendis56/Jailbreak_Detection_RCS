#!/usr/bin/env python3
"""
Model Download Script for VLM Jailbreak Detection Project

Configure which models to download by modifying the MODELS_TO_DOWNLOAD dictionary below.
"""

import os
import shutil
from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModel
import torch

# Set to True to download the model, False to skip
MODELS_TO_DOWNLOAD = {
    'llava_vicuna_7b': True,
    'flava': False,                  # FLAVA (~2GB) - Facebook's multimodal model, baseline and more visualization
    'clip_base': False,              # CLIP ViT Base (~600MB) - For JailDAM baseline
    'clip_large': False,             # CLIP ViT Large (~1.7GB) - LLaVA vision tower
    'qwen25_vl_3b': False,          # Qwen2.5-VL-3B-Instruct (~6GB) - Qwen model
    'qwen25_vl_7b': True,          # Qwen2.5-VL-7B-Instruct (~13GB) - Larger Qwen model
    'internvl3_8b': False,          # InternVL3-8B (~15GB) - OpenGVLab model
}

# Estimated total sizes (approximate)
MODEL_SIZES = {
    'llava_vicuna_7b': 13.0,
    'flava': 2.0,
    'clip_base': 0.6,
    'clip_large': 1.7,
    'qwen25_vl_3b': 6.0,
    'qwen25_vl_7b': 13.0,
    'internvl3_8b': 15.0,
}

def print_download_plan():

    total_size = 0
    models_to_download = []
    models_to_skip = []

    for model_key, should_download in MODELS_TO_DOWNLOAD.items():
        size = MODEL_SIZES.get(model_key, 0)
        if should_download:
            models_to_download.append((model_key, size))
            total_size += size
        else:
            models_to_skip.append((model_key, size))

    if models_to_download:
        print("Models to download:")
        for model_key, size in models_to_download:
            print(f"  [DOWNLOAD] {model_key}: ~{size}GB")
        print(f"\nEstimated total download size: ~{total_size}GB")

    if models_to_skip:
        print(f"\nModels to skip:")
        for model_key, size in models_to_skip:
            print(f"  [SKIP] {model_key}: ~{size}GB")

    print("\nTo modify this selection, edit MODELS_TO_DOWNLOAD in this script.")
    print("="*80)

# =============================================================================

def create_model_directory():
    """Create the model directory structure"""
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Created model directory: {model_dir}")
    return model_dir

def model_exists(local_dir):
    """Check if model directory exists and is non-empty"""
    if not os.path.exists(local_dir):
        return False

    # Check if directory is non-empty
    try:
        return len(os.listdir(local_dir)) > 0
    except OSError:
        return False

def download_llava_v16_vicuna_7b(model_dir):
    """Download LLaVA-v1.6-Vicuna-7B model"""
    if not MODELS_TO_DOWNLOAD.get('llava_vicuna_7b', False):
        print("[SKIP] LLaVA-v1.6-Vicuna-7B disabled in configuration")
        return

    print("\n" + "="*60)
    print("DOWNLOADING LLAVA-V1.6-VICUNA-7B MODEL")
    print("="*60)

    model_name = "liuhaotian/llava-v1.6-vicuna-7b"
    local_dir = os.path.join(model_dir, "llava-v1.6-vicuna-7b")

    # Check if model already exists
    if model_exists(local_dir):
        print(f"[SKIP] Model already exists at {local_dir}")
        return

    try:
        print(f"Downloading {model_name} to {local_dir}...")
        print("This may take a while as the model is large (~13GB)")
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"[SUCCESS] Successfully downloaded LLaVA-v1.6-Vicuna-7B to {local_dir}")

        # Verify the download by checking key files
        key_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        missing_files = []

        for file in key_files:
            if not os.path.exists(os.path.join(local_dir, file)):
                # Check for alternative file names
                if file == "pytorch_model.bin":
                    # Check for safetensors or sharded models
                    safetensors_files = [f for f in os.listdir(local_dir) if f.endswith('.safetensors')]
                    bin_files = [f for f in os.listdir(local_dir) if f.startswith('pytorch_model') and f.endswith('.bin')]
                    if not safetensors_files and not bin_files:
                        missing_files.append(file)
                else:
                    missing_files.append(file)

        if missing_files:
            print(f"[WARNING] Some expected files are missing: {missing_files}")
        else:
            print("[SUCCESS] All key model files are present")
            
    except Exception as e:
        print(f"[ERROR] Error downloading LLaVA-v1.6-Vicuna-7B: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Insufficient disk space")
        print("3. HuggingFace Hub access issues")

def download_flava_model(model_dir):
    """Download FLAVA model"""
    if not MODELS_TO_DOWNLOAD.get('flava', False):
        print("[SKIP] FLAVA disabled in configuration")
        return

    print("\n" + "="*60)
    print("DOWNLOADING FLAVA MODEL")
    print("="*60)

    model_name = "facebook/flava-full"
    local_dir = os.path.join(model_dir, "flava")

    # Check if model already exists
    if model_exists(local_dir):
        print(f"[SKIP] Model already exists at {local_dir}")
        return

    try:
        print(f"Downloading {model_name} to {local_dir}...")
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"[SUCCESS] Successfully downloaded FLAVA to {local_dir}")

        # Verify the download
        key_files = ["config.json", "pytorch_model.bin"]
        missing_files = []

        for file in key_files:
            if not os.path.exists(os.path.join(local_dir, file)):
                if file == "pytorch_model.bin":
                    # Check for safetensors
                    safetensors_files = [f for f in os.listdir(local_dir) if f.endswith('.safetensors')]
                    if not safetensors_files:
                        missing_files.append(file)
                else:
                    missing_files.append(file)

        if missing_files:
            print(f"[WARNING] Some expected files are missing: {missing_files}")
        else:
            print("[SUCCESS] All key model files are present")

    except Exception as e:
        print(f"[ERROR] Error downloading FLAVA: {e}")

def download_clip_model(model_dir):
    """Download CLIP model (often used as a baseline)"""
    if not MODELS_TO_DOWNLOAD.get('clip_base', False):
        print("[SKIP] CLIP Base disabled in configuration")
        return

    print("\n" + "="*60)
    print("DOWNLOADING CLIP MODEL")
    print("="*60)

    model_name = "openai/clip-vit-base-patch32"
    local_dir = os.path.join(model_dir, "clip-vit-base-patch32")

    # Check if model already exists
    if model_exists(local_dir):
        print(f"[SKIP] Model already exists at {local_dir}")
        return

    try:
        print(f"Downloading {model_name} to {local_dir}...")
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"[SUCCESS] Successfully downloaded CLIP to {local_dir}")

    except Exception as e:
        print(f"[ERROR] Error downloading CLIP: {e}")

def download_clip_large_model(model_dir):
    """Download CLIP ViT Large model (used by LLaVA as vision tower)"""
    if not MODELS_TO_DOWNLOAD.get('clip_large', False):
        print("[SKIP] CLIP Large disabled in configuration")
        return

    print("\n" + "="*60)
    print("DOWNLOADING CLIP VIT LARGE MODEL")
    print("="*60)

    model_name = "openai/clip-vit-large-patch14-336"
    local_dir = os.path.join(model_dir, "clip-vit-large-patch14-336")

    # Check if model already exists
    if model_exists(local_dir):
        print(f"[SKIP] Model already exists at {local_dir}")
        return

    try:
        print(f"Downloading {model_name} to {local_dir}...")
        print("This model is used as the vision tower for LLaVA")

        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"[SUCCESS] Successfully downloaded CLIP ViT Large to {local_dir}")

        # Verify the download by checking key files
        key_files = ["config.json", "pytorch_model.bin"]
        missing_files = []

        for file in key_files:
            if not os.path.exists(os.path.join(local_dir, file)):
                if file == "pytorch_model.bin":
                    # Check for safetensors
                    safetensors_files = [f for f in os.listdir(local_dir) if f.endswith('.safetensors')]
                    if not safetensors_files:
                        missing_files.append(file)
                else:
                    missing_files.append(file)

        if missing_files:
            print(f"[WARNING] Some expected files are missing: {missing_files}")
        else:
            print("[SUCCESS] All key model files are present")

    except Exception as e:
        print(f"[ERROR] Error downloading CLIP ViT Large: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Insufficient disk space")
        print("3. HuggingFace Hub access issues")

def download_qwen25_vl_3b_model(model_dir):
    """Download Qwen2.5-VL-3B-Instruct model"""
    if not MODELS_TO_DOWNLOAD.get('qwen25_vl_3b', False):
        print("[SKIP] Qwen2.5-VL-3B disabled in configuration")
        return

    print("\n" + "="*60)
    print("DOWNLOADING QWEN2.5-VL-3B-INSTRUCT MODEL")
    print("="*60)

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    local_dir = os.path.join(model_dir, "qwen2.5-vl-3b-instruct")

    # Check if model already exists
    if model_exists(local_dir):
        print(f"[SKIP] Model already exists at {local_dir}")
        return

    try:
        print(f"Downloading {model_name} to {local_dir}...")
        print("This may take a while as the model is large (~6GB)")

        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"[SUCCESS] Successfully downloaded Qwen2.5-VL-3B-Instruct to {local_dir}")

        # Verify the download by checking key files
        key_files = ["config.json", "tokenizer.json"]
        missing_files = []

        for file in key_files:
            if not os.path.exists(os.path.join(local_dir, file)):
                missing_files.append(file)

        if missing_files:
            print(f"[WARNING] Some expected files are missing: {missing_files}")
        else:
            print("[SUCCESS] All key model files are present")

    except Exception as e:
        print(f"[ERROR] Error downloading Qwen2.5-VL-3B-Instruct: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Insufficient disk space")
        print("3. HuggingFace Hub access issues")

def download_qwen25_vl_7b_model(model_dir):
    """Download Qwen2.5-VL-7B-Instruct model"""
    if not MODELS_TO_DOWNLOAD.get('qwen25_vl_7b', False):
        print("[SKIP] Qwen2.5-VL-7B disabled in configuration")
        return

    print("\n" + "="*60)
    print("DOWNLOADING QWEN2.5-VL-7B-INSTRUCT MODEL")
    print("="*60)

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    local_dir = os.path.join(model_dir, "qwen2.5-vl-7b-instruct")

    # Check if model already exists
    if model_exists(local_dir):
        print(f"[SKIP] Model already exists at {local_dir}")
        return

    try:
        print(f"Downloading {model_name} to {local_dir}...")
        print("This may take a while as the model is large (~15GB)")

        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"[SUCCESS] Successfully downloaded Qwen2.5-VL-7B-Instruct to {local_dir}")

        # Verify the download by checking key files
        key_files = ["config.json", "tokenizer.json"]
        missing_files = []

        for file in key_files:
            if not os.path.exists(os.path.join(local_dir, file)):
                missing_files.append(file)

        if missing_files:
            print(f"[WARNING] Some expected files are missing: {missing_files}")
        else:
            print("[SUCCESS] All key model files are present")

    except Exception as e:
        print(f"[ERROR] Error downloading Qwen2.5-VL-7B-Instruct: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Insufficient disk space")
        print("3. HuggingFace Hub access issues")

def download_internvl3_8b_model(model_dir):
    """Download InternVL3-8B model"""
    if not MODELS_TO_DOWNLOAD.get('internvl3_8b', False):
        print("[SKIP] InternVL3-8B disabled in configuration")
        return

    print("\n" + "="*60)
    print("DOWNLOADING INTERNVL3-8B MODEL")
    print("="*60)

    model_name = "OpenGVLab/InternVL3-8B"
    local_dir = os.path.join(model_dir, "internvl3-8b")

    # Check if model already exists
    if model_exists(local_dir):
        print(f"[SKIP] Model already exists at {local_dir}")
        return

    try:
        print(f"Downloading {model_name} to {local_dir}...")
        print("This may take a while as the model is large (~20GB)")

        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"[SUCCESS] Successfully downloaded InternVL3-8B to {local_dir}")

        # Verify the download by checking key files
        key_files = ["config.json", "tokenizer.json"]
        missing_files = []

        for file in key_files:
            if not os.path.exists(os.path.join(local_dir, file)):
                missing_files.append(file)

        # Also check that at least some weights exist
        weight_files = [f for f in os.listdir(local_dir) if f.endswith(('.bin', '.safetensors'))]
        if not weight_files:
            print("[WARNING] No model weight files found in directory; ensure download completed correctly")

        if missing_files:
            print(f"[WARNING] Some expected files are missing: {missing_files}")
        else:
            print("[SUCCESS] Key config/tokenizer files are present")

    except Exception as e:
        print(f"[ERROR] Error downloading InternVL3-8B: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Insufficient disk space")
        print("3. HuggingFace Hub access issues")

def check_disk_space():
    """Check available disk space"""
    
    try:
        # Get disk usage statistics
        statvfs = os.statvfs('.')
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        print(f"Available disk space: {free_space_gb:.2f} GB")
        
        # Calculate required space based on selected models
        required_space_gb = sum(
            MODEL_SIZES.get(model_key, 0)
            for model_key, should_download in MODELS_TO_DOWNLOAD.items()
            if should_download
        )

        if free_space_gb < required_space_gb:
            print(f"[WARNING] You may need at least {required_space_gb:.1f} GB of free space for selected models")
            print("Consider freeing up disk space before proceeding")
            return False
        else:
            print(f"[SUCCESS] Sufficient disk space available ({required_space_gb:.1f} GB needed)")
            return True
            
    except Exception as e:
        print(f"[WARNING] Could not check disk space: {e}")
        return True
    
def print_usage_instructions():
    """Print instructions for using the downloaded models"""
    print("\n" + "="*80)
    print("MODEL USAGE INSTRUCTIONS")
    print("="*80)
    
    print("\nThe models have been downloaded to the 'model/' directory:")
    print("├── model/")
    print("│   ├── llava-v1.6-vicuna-7b/        # LLaVA multimodal model")
    print("│   ├── flava/                       # FLAVA multimodal model")
    print("│   ├── clip-vit-base-patch32/       # CLIP baseline model")
    print("│   ├── clip-vit-large-patch14-336/  # CLIP Large (LLaVA vision tower)")
    print("│   ├── qwen2.5-vl-3b-instruct/      # Qwen2.5-VL-3B-Instruct model")
    print("│   ├── qwen2.5-vl-7b-instruct/      # Qwen2.5-VL-7B-Instruct model")
    print("│   └── internvl3-8b/                # InternVL3-8B model")

def main():
    """Main function to download selected models"""
    print("VLM JAILBREAK DETECTION - MODEL DOWNLOADER")
    print("="*80)

    # Show download plan
    print_download_plan()

    # Check system requirements
    if not check_disk_space():
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Create model directory
    model_dir = create_model_directory()
    
    # Download models
    download_llava_v16_vicuna_7b(model_dir)
    download_flava_model(model_dir)
    download_clip_model(model_dir)
    download_clip_large_model(model_dir)
    download_qwen25_vl_3b_model(model_dir)
    download_qwen25_vl_7b_model(model_dir)
    download_internvl3_8b_model(model_dir)
    
    # Print usage instructions
    print_usage_instructions()
    
    print("\n" + "="*80)
    print("[SUCCESS] All models have been downloaded to the 'model/' directory")

    # Final verification
    total_size = 0
    for root, _, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)

    total_size_gb = total_size / (1024**3)
    print(f"[INFO] Total model size: {total_size_gb:.2f} GB")

if __name__ == "__main__":
    main()
