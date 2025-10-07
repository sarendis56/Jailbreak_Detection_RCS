#!/usr/bin/env python3
import os
import json
import sys
from pathlib import Path

def get_models_to_download():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("download_models", "download_models.py")
        if spec is None or spec.loader is None:
            raise ImportError("Could not load download_models.py")
        download_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(download_models)

        return getattr(download_models, 'MODELS_TO_DOWNLOAD', {}), getattr(download_models, 'MODEL_SIZES', {})
    except Exception as e:
        print(f"Warning: Could not read download_models.py configuration: {e}")
        return {
            'llava_vicuna_7b': True,
            'flava': True,
            'clip_base': False,
            'clip_large': True,
            'minigpt4': True,
            'qwen25_vl_3b': False,
            'qwen25_vl_7b': True,
        }, {
            'llava_vicuna_7b': 13.0,
            'flava': 2.0,
            'clip_base': 0.6,
            'clip_large': 1.7,
            'minigpt4': 7.0,
            'qwen25_vl_3b': 6.0,
            'qwen25_vl_7b': 13.0,
            'internvl3_8b': 15.0,
        }

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

def check_file_exists(filepath, description=""):
    """Check if a file exists and return status"""
    exists = os.path.exists(filepath)
    size = ""
    if exists:
        try:
            file_size = os.path.getsize(filepath)
            if file_size > 1024**3:  # GB
                size = f" ({file_size / (1024**3):.1f} GB)"
            elif file_size > 1024**2:  # MB
                size = f" ({file_size / (1024**2):.1f} MB)"
            elif file_size > 1024:  # KB
                size = f" ({file_size / 1024:.1f} KB)"
            else:
                size = f" ({file_size} bytes)"
        except:
            size = ""

    status = "[SUCCESS]" if exists else "[MISSING]"
    print(f"  {status} {filepath}{size}")
    return exists

def check_directory_exists(dirpath, description=""):
    """Check if a directory exists and count files"""
    exists = os.path.exists(dirpath)
    count = ""
    if exists and os.path.isdir(dirpath):
        try:
            file_count = len([f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))])
            count = f" ({file_count} files)"
        except:
            count = ""

    status = "[SUCCESS]" if exists else "[MISSING]"
    print(f"  {status} {dirpath}{count}")
    return exists

def check_xstest_format():
    """Special check for XSTest data format compatibility"""
    print("  [CHECK] XSTest data format compatibility:")

    if not PANDAS_AVAILABLE:
        print("    [WARNING] pandas not available - limited format checking")
        json_path = "data/XSTest/xstest_samples.json"
        if os.path.exists(json_path):
            print(f"    [SUCCESS] JSON format found: {json_path}")
            print(f"    [INFO] Install pandas for full format verification: pip install pandas")
            return True
        else:
            print(f"    [MISSING] No XSTest data found")
            return False

    # Check for the expected parquet/CSV format (primary)
    parquet_path = "data/XSTest/data/gpt4-00000-of-00001.parquet"
    csv_path = "data/XSTest/data/gpt4-00000-of-00001.csv"
    json_path = "data/XSTest/xstest_samples.json"

    parquet_exists = os.path.exists(parquet_path)
    csv_exists = os.path.exists(csv_path)
    json_exists = os.path.exists(json_path)

    if parquet_exists:
        print(f"    [SUCCESS] Parquet format: {parquet_path}")
        try:
            df = pd.read_parquet(parquet_path)
            required_cols = ['prompt', 'type']
            if all(col in df.columns for col in required_cols):
                print(f"    [SUCCESS] Required columns present: {required_cols}")
                print(f"    [SUCCESS] Sample count: {len(df)} samples")

                # Check toxicity distribution
                contrast_count = len(df[df['type'].str.contains('contrast', na=False)])
                safe_count = len(df) - contrast_count
                print(f"    [SUCCESS] Distribution: {safe_count} safe, {contrast_count} unsafe samples")
                return True
            else:
                print(f"    [ERROR] Missing required columns. Found: {list(df.columns)}")
                return False
        except Exception as e:
            print(f"    [ERROR] Error reading parquet: {e}")
            return False

    elif csv_exists:
        print(f"    [SUCCESS] CSV format: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            required_cols = ['prompt', 'type']
            if all(col in df.columns for col in required_cols):
                print(f"    [SUCCESS] Required columns present: {required_cols}")
                print(f"    [SUCCESS] Sample count: {len(df)} samples")

                # Check toxicity distribution
                contrast_count = len(df[df['type'].str.contains('contrast', na=False)])
                safe_count = len(df) - contrast_count
                print(f"    [SUCCESS] Distribution: {safe_count} safe, {contrast_count} unsafe samples")
                return True
            else:
                print(f"    [ERROR] Missing required columns. Found: {list(df.columns)}")
                return False
        except Exception as e:
            print(f"    [ERROR] Error reading CSV: {e}")
            return False

    elif json_exists:
        print(f"    [WARNING] Only JSON format found: {json_path}")
        print(f"    [WARNING] load_XSTest() expects parquet/CSV format")
        return False

    else:
        print(f"    [MISSING] No XSTest data found in any format")
        return False

def verify_datasets():
    """Verify all required datasets"""
    print("="*60)
    print("VERIFYING DATASETS")
    print("="*60)

    datasets_status = {}

    # Automatic datasets (from HuggingFace)
    print("\n[AUTO] HuggingFace Datasets:")

    auto_datasets = [
        ("data/Alpaca/alpaca_samples.json", "Alpaca instruction following dataset"),
        ("data/AdvBench/advbench_samples.json", "AdvBench harmful prompts dataset"),
        ("data/DAN_Prompts/dan_prompts_samples.json", "DAN jailbreak prompts dataset"),
        ("data/OpenAssistant/openassistant_samples.json", "OpenAssistant conversational dataset"),
        ("data/MM-SafetyBench/data", "MM-SafetyBench multimodal safety dataset")
    ]

    auto_count = 0
    for filepath, description in auto_datasets:
        print(f"\n  {description}:")
        if filepath.endswith("/data"):  # Directory check for MM-SafetyBench
            if check_directory_exists(filepath):
                auto_count += 1
                # Check for parquet files in subdirectories
                try:
                    parquet_count = 0
                    for root, dirs, files in os.walk(filepath):
                        parquet_count += len([f for f in files if f.endswith('.parquet')])
                    if parquet_count > 0:
                        print(f"    [SUCCESS] Found {parquet_count} parquet files")
                    else:
                        print(f"    [WARNING] Directory exists but no parquet files found")
                except Exception as e:
                    print(f"    [WARNING] Could not check parquet files: {e}")
        else:
            if check_file_exists(filepath):
                auto_count += 1
        datasets_status[filepath] = os.path.exists(filepath)

    # VQAv2 dataset (special structure check)
    print(f"\n[VQAv2] Visual Question Answering Dataset:")
    
    # Check for original VQAv2 format
    original_vqav2_files = [
        "data/VQAv2/v2_mscoco_train2014_annotations.json",
        "data/VQAv2/v2_mscoco_val2014_annotations.json",
        "data/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json",
        "data/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json",
        "data/VQAv2/images"
    ]
    
    # Check for VizWiz-VQA format (our current implementation)
    vizwiz_files = [
        "data/VQAv2/vqav2_samples.json",
        "data/VQAv2/images"
    ]

    # Check original VQAv2 format
    original_vqav2_count = 0
    for filepath in original_vqav2_files:
        if filepath.endswith("images"):
            if check_directory_exists(filepath):
                original_vqav2_count += 1
        else:
            if check_file_exists(filepath):
                original_vqav2_count += 1

    # Check VizWiz-VQA format
    vizwiz_count = 0
    for filepath in vizwiz_files:
        if filepath.endswith("images"):
            if check_directory_exists(filepath):
                vizwiz_count += 1
        else:
            if check_file_exists(filepath):
                vizwiz_count += 1

    # Determine which format is available
    if original_vqav2_count == len(original_vqav2_files):
        print(f"      [SUCCESS] Original VQAv2 format complete ({original_vqav2_count}/{len(original_vqav2_files)} components)")
        datasets_status["VQAv2"] = True
        auto_count += 1
    elif vizwiz_count == len(vizwiz_files):
        print(f"      [SUCCESS] VizWiz-VQA format complete ({vizwiz_count}/{len(vizwiz_files)} components)")
        # Check sample count in VizWiz format
        try:
            with open("data/VQAv2/vqav2_samples.json", 'r') as f:
                vizwiz_data = json.load(f)
                print(f"      [INFO] VizWiz-VQA samples: {len(vizwiz_data)}")
        except:
            pass
        datasets_status["VQAv2"] = True
        auto_count += 1
    else:
        print(f"      [MISSING] Neither format complete")
        print(f"        Original VQAv2: {original_vqav2_count}/{len(original_vqav2_files)} components")
        print(f"        VizWiz-VQA: {vizwiz_count}/{len(vizwiz_files)} components")
        datasets_status["VQAv2"] = False

    # Store status for both formats
    for filepath in original_vqav2_files:
        datasets_status[filepath] = os.path.exists(filepath)
    for filepath in vizwiz_files:
        datasets_status[filepath] = os.path.exists(filepath)

    # Special handling for XSTest (multiple format support)
    print(f"\n[XSTEST] Safety evaluation dataset:")
    xstest_ok = check_xstest_format()
    if xstest_ok:
        auto_count += 1
    datasets_status["XSTest"] = xstest_ok

    # Manual datasets
    print("\n[MANUAL] Manual Downloads Required:")

    manual_datasets = [
        ("data/mm-vet/mm-vet.json", "MM-Vet multimodal reasoning dataset"),
        ("data/FigTxt/benign_questions.csv", "FigTxt benign questions"),
        ("data/FigTxt/safebench.csv", "FigTxt safety benchmark"),
        ("data/JailBreakV-28k/mini_JailBreakV_28K.csv", "JailBreakV-28k text queries"),
        ("data/VAE/manual_harmful_instructions.csv", "VAE harmful instructions")
    ]

    manual_count = 0
    for filepath, description in manual_datasets:
        if check_file_exists(filepath):
            manual_count += 1
        datasets_status[filepath] = os.path.exists(filepath)

    # Manual directories
    print("\n[DIRS] Manual Directories Required:")

    manual_dirs = [
        ("data/JailBreakV-28k/figstep", "JailBreakV-28k figstep images"),
        ("data/JailBreakV-28k/llm_transfer_attack", "JailBreakV-28k LLM transfer attack images"),
        ("data/JailBreakV-28k/query_related", "JailBreakV-28k query related images"),
        ("data/VAE/adversarial_images", "VAE adversarial images")
    ]

    manual_dir_count = 0
    for dirpath, description in manual_dirs:
        if check_directory_exists(dirpath):
            manual_dir_count += 1
        datasets_status[dirpath] = os.path.exists(dirpath)

    print(f"\n[SUMMARY] Dataset Summary:")
    print(f"  Automatic datasets: {auto_count}/{len(auto_datasets) + 2} [SUCCESS]")  # +2 for VQAv2 and XSTest
    print(f"  Manual files: {manual_count}/{len(manual_datasets)} [SUCCESS]")
    print(f"  Manual directories: {manual_dir_count}/{len(manual_dirs)} [SUCCESS]")

    total_available = auto_count + manual_count + manual_dir_count
    total_required = len(auto_datasets) + 2 + len(manual_datasets) + len(manual_dirs)  # +2 for VQAv2 and XSTest
    
    return total_available, total_required, datasets_status

def verify_models():
    """Verify models based on download_models.py configuration"""
    print("\n" + "="*60)
    print("VERIFYING MODELS")
    print("="*60)

    # Get configuration from download_models.py
    models_to_download, model_sizes = get_models_to_download()

    # Map model keys to their paths and descriptions
    model_mapping = {
        'llava_vicuna_7b': ("model/llava-v1.6-vicuna-7b", "LLaVA-v1.6-Vicuna-7B multimodal model"),
        'flava': ("model/flava", "FLAVA multimodal model"),
        'clip_base': ("model/clip-vit-base-patch32", "CLIP ViT Base model"),
        'clip_large': ("model/clip-vit-large-patch14-336", "CLIP ViT Large model (LLaVA vision tower)"),
        'minigpt4': ("model/minigpt-4", "MiniGPT-4 multimodal model"),
        'qwen25_vl_3b': ("model/qwen2.5-vl-3b-instruct", "Qwen2.5-VL-3B-Instruct model"),
        'qwen25_vl_7b': ("model/qwen2.5-vl-7b-instruct", "Qwen2.5-VL-7B-Instruct model"),
        'internvl3_8b': ("model/internvl3-8b", "InternVL3-8B model"),
    }

    models_status = {}
    enabled_models = []
    disabled_models = []

    # Separate enabled and disabled models
    for model_key, should_download in models_to_download.items():
        if model_key in model_mapping:
            model_path, description = model_mapping[model_key]
            size = model_sizes.get(model_key, 0)
            if should_download:
                enabled_models.append((model_key, model_path, description, size))
            else:
                disabled_models.append((model_key, model_path, description, size))

    print(f"\nEnabled Models (checking {len(enabled_models)} models):")

    model_count = 0
    for model_key, model_path, description, size in enabled_models:
        print(f"\n  {description} (~{size}GB)")
        if check_directory_exists(model_path, f"Run: python download_models.py"):
            model_count += 1

            # Check for key model files
            model_files = []

            if os.path.exists(model_path):
                for file in os.listdir(model_path):
                    if file.endswith(('.bin', '.safetensors')):
                        model_files.append(file)

                if model_files:
                    print(f"      Found model files: {len(model_files)} files")
                else:
                    print(f"      No model weight files found")

        models_status[model_path] = os.path.exists(model_path)

    if disabled_models:
        print(f"\nDisabled Models (skipping {len(disabled_models)} models):")
        for model_key, model_path, description, size in disabled_models:
            print(f"  {description} (~{size}GB) - disabled in configuration")
            models_status[model_path] = "disabled"

    print(f"\nModel Summary:")
    print(f"  Available models: {model_count}/{len(enabled_models)}")
    if disabled_models:
        print(f"  Disabled models: {len(disabled_models)} (edit download_models.py to enable)")

    total_enabled_size = sum(size for _, _, _, size in enabled_models)
    print(f"  Total size of enabled models: ~{total_enabled_size}GB")

    return model_count, len(enabled_models), models_status

def test_basic_loading():
    """Test basic loading of key components"""
    print("\n" + "="*60)
    print("TESTING BASIC LOADING")
    print("="*60)
    
    # Test dataset loading
    print("\n[DATASETS] Testing Dataset Loading:")
    try:
        sys.path.append('code')
        from load_datasets import (
            load_alpaca, load_mm_vet, load_XSTest, load_dan_prompts, 
            load_vqav2, load_advbench, load_openassistant, load_FigTxt,
            load_JailBreakV_figstep, load_adversarial_img, load_mm_safety_bench_all
        )

        # Test automatic datasets (from HuggingFace)
        print("  Testing automatic datasets:")

        # Test Alpaca loading
        try:
            alpaca_samples = load_alpaca(max_samples=5)
            if alpaca_samples:
                print(f"    [SUCCESS] Alpaca: Loaded {len(alpaca_samples)} test samples")
            else:
                print(f"    [ERROR] Alpaca: No samples loaded")
        except Exception as e:
            print(f"    [ERROR] Alpaca: Error loading - {e}")

        # Test AdvBench loading
        try:
            advbench_samples = load_advbench(max_samples=5)
            if advbench_samples:
                print(f"    [SUCCESS] AdvBench: Loaded {len(advbench_samples)} test samples")
            else:
                print(f"    [ERROR] AdvBench: No samples loaded")
        except Exception as e:
            print(f"    [ERROR] AdvBench: Error loading - {e}")

        # Test DAN prompts loading
        try:
            dan_samples = load_dan_prompts(max_samples=5)
            if dan_samples:
                print(f"    [SUCCESS] DAN Prompts: Loaded {len(dan_samples)} test samples")
            else:
                print(f"    [ERROR] DAN Prompts: No samples loaded")
        except Exception as e:
            print(f"    [ERROR] DAN Prompts: Error loading - {e}")

        # Test OpenAssistant loading
        try:
            oa_samples = load_openassistant(max_samples=5)
            if oa_samples:
                print(f"    [SUCCESS] OpenAssistant: Loaded {len(oa_samples)} test samples")
            else:
                print(f"    [ERROR] OpenAssistant: No samples loaded")
        except Exception as e:
            print(f"    [ERROR] OpenAssistant: Error loading - {e}")

        # Test VQAv2 loading (supports both original and VizWiz formats)
        try:
            vqav2_samples = load_vqav2(max_samples=5)
            if vqav2_samples:
                print(f"    [SUCCESS] VQAv2: Loaded {len(vqav2_samples)} test samples")
            else:
                print(f"    [ERROR] VQAv2: No samples loaded")
        except Exception as e:
            print(f"    [ERROR] VQAv2: Error loading - {e}")

        # Test XSTest loading (new format)
        try:
            xstest_samples = load_XSTest()
            if xstest_samples:
                safe_count = len([s for s in xstest_samples if s.get('toxicity', 0) == 0])
                unsafe_count = len([s for s in xstest_samples if s.get('toxicity', 0) == 1])
                print(f"    [SUCCESS] XSTest: Loaded {len(xstest_samples)} samples ({safe_count} safe, {unsafe_count} unsafe)")
            else:
                print(f"    [ERROR] XSTest: No samples loaded")
        except Exception as e:
            print(f"    [ERROR] XSTest: Error loading - {e}")

        # Test manual datasets
        print("  Testing manual datasets:")

        # Test MM-Vet loading
        try:
            mmvet_samples = load_mm_vet()
            if mmvet_samples:
                print(f"    [SUCCESS] MM-Vet: Loaded {len(mmvet_samples)} samples")
            else:
                print(f"    [ERROR] MM-Vet: No samples loaded")
        except Exception as e:
            print(f"    [ERROR] MM-Vet: Error loading - {e}")

        # Test FigTxt loading
        try:
            figtxt_samples = load_FigTxt()
            if figtxt_samples:
                safe_count = len([s for s in figtxt_samples if s.get('toxicity', 0) == 0])
                unsafe_count = len([s for s in figtxt_samples if s.get('toxicity', 0) == 1])
                print(f"    [SUCCESS] FigTxt: Loaded {len(figtxt_samples)} samples ({safe_count} safe, {unsafe_count} unsafe)")
            else:
                print(f"    [ERROR] FigTxt: No samples loaded")
        except Exception as e:
            print(f"    [ERROR] FigTxt: Error loading - {e}")

        # Test JailbreakV-28K loading (figstep variant)
        try:
            jbv_samples = load_JailBreakV_figstep(max_samples=5)
            if jbv_samples:
                print(f"    [SUCCESS] JailbreakV-28K (figstep): Loaded {len(jbv_samples)} test samples")
            else:
                print(f"    [ERROR] JailbreakV-28K (figstep): No samples loaded")
        except Exception as e:
            print(f"    [ERROR] JailbreakV-28K (figstep): Error loading - {e}")

        # Test VAE adversarial images loading
        try:
            vae_samples = load_adversarial_img()
            if vae_samples:
                print(f"    [SUCCESS] VAE (adversarial images): Loaded {len(vae_samples)} samples")
            else:
                print(f"    [ERROR] VAE (adversarial images): No samples loaded")
        except Exception as e:
            print(f"    [ERROR] VAE (adversarial images): Error loading - {e}")

        # Test MM-SafetyBench loading
        try:
            mm_safety_samples = load_mm_safety_bench_all(max_samples=10)
            if mm_safety_samples:
                print(f"    [SUCCESS] MM-SafetyBench: Loaded {len(mm_safety_samples)} test samples")
            else:
                print(f"    [ERROR] MM-SafetyBench: No samples loaded")
        except Exception as e:
            print(f"    [ERROR] MM-SafetyBench: Error loading - {e}")

    except ImportError as e:
        print(f"  [ERROR] Could not import dataset loading functions: {e}")

    # Test model loading (basic check)
    print("\n[MODELS] Testing Model Accessibility:")

    # Get configuration to test only enabled models
    models_to_download, _ = get_models_to_download()

    # Test LLaVA if enabled
    if models_to_download.get('llava_vicuna_7b', False):
        llava_path = "model/llava-v1.6-vicuna-7b"
        if os.path.exists(llava_path):
            config_path = os.path.join(llava_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        model_type = config.get('model_type', 'unknown')
                        mm_vision_tower = config.get('mm_vision_tower', 'not specified')
                        print(f"  [SUCCESS] LLaVA config accessible (type: {model_type})")
                        print(f"      Vision tower: {mm_vision_tower}")
                except Exception as e:
                    print(f"  [ERROR] LLaVA config error: {e}")
            else:
                print(f"  [ERROR] LLaVA config.json not found")
        else:
            print(f"  [ERROR] LLaVA model directory not found")
    else:
        print(f"  [SKIP] LLaVA testing skipped (disabled in configuration)")

    # Test Qwen models if enabled
    for model_key, model_path in [
        ('qwen25_vl_3b', 'model/qwen2.5-vl-3b-instruct'),
        ('qwen25_vl_7b', 'model/qwen2.5-vl-7b-instruct')
    ]:
        if models_to_download.get(model_key, False):
            if os.path.exists(model_path):
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            model_type = config.get('model_type', 'unknown')
                            print(f"  [SUCCESS] {model_key.upper()} config accessible (type: {model_type})")
                    except Exception as e:
                        print(f"  [ERROR] {model_key.upper()} config error: {e}")
                else:
                    print(f"  [ERROR] {model_key.upper()} config.json not found")
            else:
                print(f"  [ERROR] {model_key.upper()} model directory not found")
        else:
            print(f"  [SKIP] {model_key.upper()} testing skipped (disabled in configuration)")

    # Test InternVL3-8B if enabled (lightweight load check)
    if models_to_download.get('internvl3_8b', False):
        internvl_path = 'model/internvl3-8b'
        if os.path.exists(internvl_path):
            config_path = os.path.join(internvl_path, 'config.json')
            if os.path.exists(config_path):
                try:
                    from transformers import AutoConfig, AutoModel, AutoTokenizer
                    config = AutoConfig.from_pretrained(internvl_path, trust_remote_code=True)
                    print(f"  [SUCCESS] InternVL3-8B config accessible (model_type: {getattr(config, 'model_type', 'unknown')})")

                    try:
                        tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)
                        print("  [SUCCESS] InternVL3-8B tokenizer accessible")
                    except Exception as e:
                        print(f"  [WARNING] InternVL3-8B tokenizer load failed: {e}")
                except Exception as e:
                    print(f"  [ERROR] InternVL3-8B config/model access error: {e}")
            else:
                print("  [ERROR] InternVL3-8B config.json not found")
        else:
            print("  [ERROR] InternVL3-8B model directory not found")
    else:
        print("  [SKIP] INTERNVL3_8B testing skipped (disabled in configuration)")

def main():
    """Main verification function"""
    print("VLM JAILBREAK DETECTION - SETUP VERIFICATION")
    print("="*80)
    
    # Verify datasets
    dataset_available, dataset_total, dataset_status = verify_datasets()
    
    # Verify models  
    model_available, model_total, model_status = verify_models()
    
    # Test basic loading
    test_basic_loading()
    
    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    dataset_percent = (dataset_available / dataset_total) * 100 if dataset_total > 0 else 0
    model_percent = (model_available / model_total) * 100 if model_total > 0 else 0
    
    print(f"[DATASETS] {dataset_available}/{dataset_total} ({dataset_percent:.0f}%) available")
    print(f"[MODELS] {model_available}/{model_total} ({model_percent:.0f}%) available")

    if dataset_available == dataset_total and model_available == model_total:
        print("\n[SUCCESS] All enabled datasets and models are available!")
        print("[READY] You can now run your experiments:")
        print("   python code/main_mcd_kmeans.py --model llava")
    else:
        print("\n[INCOMPLETE] Setup incomplete:")
        if dataset_available < dataset_total:
            print("   Missing datasets - run: python download_datasets.py")
        if model_available < model_total:
            print("   Missing models - run: python download_models.py")
        print("\n[INFO] To enable/disable models, edit MODELS_TO_DOWNLOAD in download_models.py")

    # Disk space check
    try:
        statvfs = os.statvfs('.')
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        print(f"\n[DISK] Available disk space: {free_space_gb:.1f} GB")
    except:
        pass

if __name__ == "__main__":
    main()
