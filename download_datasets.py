#!/usr/bin/env python3
"""
Loaded from HuggingFace:
- Alpaca (instruction following)
- AdvBench (harmful prompts)  
- DAN Prompts (jailbreak prompts)
- OpenAssistant (conversational)
- VQAv2 (visual question answering)
- XSTest (safety evaluation)

Manual datasets (need to be obtained separately):
- MM-Vet (multimodal reasoning)
- FigTxt (figure-text attacks)
- JailbreakV-28K (multimodal jailbreaks)
- VAE adversarial images
"""

import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm
import requests
import zipfile
import shutil
import pandas as pd

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        "data/Alpaca",
        "data/AdvBench",
        "data/DAN_Prompts",
        "data/OpenAssistant",
        "data/VQAv2",
        "data/XSTest",
        "data/MM-Vet",
        "data/FigTxt",
        "data/JailbreakV-28K",
        "data/VAE"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_alpaca():
    """Download Alpaca instruction following dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING ALPACA DATASET")
    print("="*60)

    output_path = "data/Alpaca/alpaca_samples.json"
    if os.path.exists(output_path):
        print(f"✅ Alpaca dataset already exists at {output_path}, skipping download")
        return

    # Load the dataset from HuggingFace
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    samples = []
    for item in tqdm(dataset, desc="Processing Alpaca samples"):
        # Convert to our standard format
        sample = {
            "txt": item["instruction"] + " " + (item["input"] if item["input"] else ""),
            "img": None,  # Text-only dataset
            "toxicity": 0,  # Benign samples
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        }
        samples.append(sample)

    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Successfully downloaded {len(samples)} Alpaca samples to {output_path}")

def download_advbench():
    """Download AdvBench harmful prompts dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING ADVBENCH DATASET")
    print("="*60)

    output_path = "data/AdvBench/advbench_samples.json"
    if os.path.exists(output_path):
        print(f"✅ AdvBench dataset already exists at {output_path}, skipping download")
        return

    # Load the dataset from HuggingFace
    dataset = load_dataset("walledai/AdvBench", split="train")

    samples = []
    for item in tqdm(dataset, desc="Processing AdvBench samples"):
        # Check available keys and use appropriate field
        text_field = None
        target_field = None

        # Try different possible field names
        for key in item.keys():
            if key.lower() in ['goal', 'prompt', 'text', 'instruction']:
                text_field = key
            if key.lower() in ['target', 'response', 'output']:
                target_field = key

        if text_field:
            sample = {
                "txt": item[text_field],
                "img": None,  # Text-only dataset
                "toxicity": 1,  # Malicious samples
                "source_field": text_field
            }
            if target_field:
                sample["target"] = item[target_field]
            samples.append(sample)

    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Successfully downloaded {len(samples)} AdvBench samples to {output_path}")

def download_dan_prompts():
    """Download In-the-Wild Jailbreak Prompts dataset (better source than original DAN)"""
    print("\n" + "="*60)
    print("DOWNLOADING DAN PROMPTS DATASET (IMPROVED SOURCE)")
    print("="*60)

    output_path = "data/DAN_Prompts/dan_prompts_samples.json"
    if os.path.exists(output_path):
        print(f"✅ Jailbreak Prompts dataset already exists at {output_path}, skipping download")
        return

    # Load the dataset from HuggingFace - use the most recent jailbreak config
    dataset = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", split="train")

    samples = []
    for item in tqdm(dataset, desc="Processing jailbreak prompts"):
        # Check available keys and use appropriate field
        text_field = None

        # Try different possible field names for jailbreak prompts
        for key in item.keys():
            if key.lower() in ['prompt', 'text', 'jailbreak', 'content', 'input']:
                text_field = key
                break

        if text_field and item[text_field]:
            sample = {
                "txt": item[text_field],
                "img": None,  # Text-only dataset
                "toxicity": 1,  # Malicious samples (jailbreak prompts)
                "source_field": text_field
            }
            # Add any additional fields
            for key, value in item.items():
                if key != text_field and key not in sample:
                    sample[key] = value
            samples.append(sample)

    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Successfully downloaded {len(samples)} jailbreak prompt samples to {output_path}")

def download_openassistant():
    """Download OpenAssistant conversational dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING OPENASSISTANT DATASET")
    print("="*60)

    output_path = "data/OpenAssistant/openassistant_samples.json"
    if os.path.exists(output_path):
        print(f"✅ OpenAssistant dataset already exists at {output_path}, skipping download")
        return

    # Load the dataset from HuggingFace
    dataset = load_dataset("OpenAssistant/oasst2", split="train")

    samples = []
    for item in tqdm(dataset, desc="Processing OpenAssistant samples"):
        # Only use assistant responses that are not deleted and have good quality
        if (item["role"] == "assistant" and
            not item.get("deleted", False) and
            item.get("rank", 0) == 0):  # Best ranked responses

            sample = {
                "txt": item["text"],
                "img": None,  # Text-only dataset
                "toxicity": 0,  # Benign samples
                "message_id": item["message_id"],
                "parent_id": item["parent_id"],
                "lang": item["lang"]
            }
            samples.append(sample)

    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Successfully downloaded {len(samples)} OpenAssistant samples to {output_path}")

def download_vqav2():
    """Download VizWiz-VQA visual question answering dataset with images"""
    print("\n" + "="*60)
    print("DOWNLOADING VIZWIZ-VQA DATASET")
    print("="*60)

    output_path = "data/VQAv2/vqav2_samples.json"
    images_dir = "data/VQAv2/images"
    
    if os.path.exists(output_path):
        print(f"✅ VizWiz-VQA dataset already exists at {output_path}, skipping download")
        return

    # Create images directory
    os.makedirs(images_dir, exist_ok=True)

    # Load the dataset from HuggingFace with images - use "val" split
    dataset = load_dataset("lmms-lab/VizWiz-VQA", split="val")

    samples = []
    max_samples = 1000  # Download up to 1000 samples

    for idx, item in enumerate(tqdm(dataset, desc="Processing VizWiz-VQA samples")):
        if idx >= max_samples:
            break

        # Save image if available
        image_filename = None
        if item.get("image") is not None:
            try:
                image_filename = f"vizwiz_{idx:06d}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                # Save the PIL image
                item["image"].save(image_path, "JPEG")
            except Exception as e:
                print(f"Warning: Could not save image {idx}: {e}")
                image_filename = None

        # Create sample
        sample = {
            "txt": item.get("question", ""),
            "img": image_filename,  # Store relative path to image
            "toxicity": 0,  # Benign samples
            "question_id": item.get("question_id", f"vizwiz_{idx}"),
            "image_id": item.get("image_id", f"img_{idx}"),
            "answers": item.get("answers", [])
        }
        samples.append(sample)

    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Successfully downloaded VizWiz-VQA with {len(samples)} samples to {output_path}")
    print(f"Images saved to {images_dir}/")

def download_xstest():
    """Download XSTest safety evaluation dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING XSTEST DATASET")
    print("="*60)

    # Check if any of the expected output files exist
    parquet_output_path = "data/XSTest/data/gpt4-00000-of-00001.parquet"
    csv_output_path = "data/XSTest/data/gpt4-00000-of-00001.csv"
    json_output_path = "data/XSTest/xstest_samples.json"

    if os.path.exists(parquet_output_path) or os.path.exists(csv_output_path):
        print(f"✅ XSTest dataset already exists, skipping download")
        return

    # Load the dataset from HuggingFace
    dataset = load_dataset("walledai/XSTest", split="test")

    # Create the data directory structure expected by load_XSTest
    data_dir = "data/XSTest/data"
    os.makedirs(data_dir, exist_ok=True)

    # Prepare data for CSV format (matching load_XSTest expectations)
    csv_data = []
    json_samples = []  # Keep JSON version for compatibility

    for item in tqdm(dataset, desc="Processing XSTest samples"):
        # CSV row for load_XSTest function
        csv_row = {
            "prompt": item["prompt"],
            "type": item["type"],
            "focus": item.get("focus", ""),
            "note": item.get("note", "")
        }
        csv_data.append(csv_row)

        # JSON sample for compatibility
        is_toxic = 1 if "contrast" in item.get("type", "") else 0
        json_sample = {
            "txt": item["prompt"],
            "img": None,  # Text-only dataset
            "toxicity": is_toxic,
            "type": item["type"],
            "focus": item.get("focus", ""),
            "note": item.get("note", "")
        }
        json_samples.append(json_sample)

    # Save as CSV file (primary format expected by load_XSTest)
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_output_path, index=False)
    print(f"Successfully saved {len(csv_data)} XSTest samples to {csv_output_path} (CSV format)")

    # Also save as parquet (alternative format supported by load_XSTest)
    df.to_parquet(parquet_output_path, index=False)
    print(f"Successfully saved {len(csv_data)} XSTest samples to {parquet_output_path} (Parquet format)")

    # Keep JSON version for backward compatibility
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_samples, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved {len(json_samples)} XSTest samples to {json_output_path} (JSON format)")

    print(f"XSTest dataset ready - load_XSTest() will use the parquet/CSV format automatically")

def download_mm_safety_bench():
    """Download MM-SafetyBench dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING MM-SAFETYBENCH DATASET")
    print("="*60)

    output_dir = "data/MM-SafetyBench"
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"✅ MM-SafetyBench dataset already exists at {output_dir}, skipping download")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        configs = [
            'EconomicHarm', 'Financial_Advice', 'Fraud', 'Gov_Decision',
            'HateSpeech', 'Health_Consultation', 'Illegal_Activitiy', 'Legal_Opinion',
            'Malware_Generation', 'Physical_Harm', 'Political_Lobbying',
            'Privacy_Violence', 'Sex'
        ]

        all_samples = []
        total_samples = 0
        splits = ['SD', 'SD_TYPO', 'TYPO', 'Text_only']

        for config in configs:
            print(f"  Downloading config: {config}")
            config_samples = []

            images_dir = os.path.join(output_dir, "images", config)
            os.makedirs(images_dir, exist_ok=True)

            for split in splits:
                try:
                    dataset = load_dataset("PKU-Alignment/MM-SafetyBench", config, split=split)

                    for idx, item in enumerate(tqdm(dataset, desc=f"Processing {config}-{split} samples")):
                        image_filename = None
                        if item.get('image') is not None:
                            try:
                                image_filename = f"{config}_{split}_{idx:06d}.jpg"
                                image_path = os.path.join(images_dir, image_filename)

                                item['image'].save(image_path, "JPEG")
                                image_filename = f"images/{config}/{image_filename}"
                            except Exception as e:
                                print(f"      Warning: Could not save image {config}-{split}-{idx}: {e}")
                                image_filename = None

                        sample = {
                            'id': item.get('id', ''),
                            'question': item.get('question', ''),
                            'image': image_filename,
                            'category': config,
                            'split': split,
                            'attack_type': split
                        }
                        config_samples.append(sample)
                        all_samples.append(sample)

                except Exception as e:
                    print(f"    ❌ Failed to download {config}-{split}: {e}")
                    continue
            if config_samples:
                config_dir = os.path.join(output_dir, "data")
                os.makedirs(config_dir, exist_ok=True)

                df = pd.DataFrame(config_samples)
                output_file = os.path.join(config_dir, f"{config}.parquet")
                df.to_parquet(output_file, index=False)

                print(f"  ✅ Saved {len(config_samples)} samples from {config} (all splits) to {output_file}")
                total_samples += len(config_samples)
            else:
                print(f"  ❌ No samples downloaded for config {config}")

        # Save combined dataset
        if all_samples:
            combined_df = pd.DataFrame(all_samples)
            combined_output = os.path.join(output_dir, "mm_safety_bench_combined.parquet")
            combined_df.to_parquet(combined_output, index=False)
            print(f"  ✅ Saved combined dataset with {len(all_samples)} samples to {combined_output}")

        print(f"Successfully downloaded MM-SafetyBench dataset to {output_dir}")
        print(f"Total samples downloaded: {total_samples}")
        print(f"Total configs processed: {len([c for c in configs if any(s['category'] == c for s in all_samples)])}")

    except Exception as e:
        print(f"Error downloading MM-SafetyBench: {e}")
        print("This dataset may need to be downloaded manually from:")
        print("https://github.com/isXinLiu/MM-SafetyBench")

def print_manual_instructions():
    """Print instructions for manually downloading datasets"""
    print("\n" + "="*80)
    print("MANUAL DATASET DOWNLOAD INSTRUCTIONS")
    print("="*80)
    
    print("\nThe following datasets need to be downloaded manually:")
    print("\n1. MM-Vet (Multimodal Reasoning)")
    print("   - Download dataset from: https://github.com/yuweihao/MM-Vet")
    print("   - JSON file should be at: data/mm-vet/mm-vet.json")
    
    print("\n2. FigTxt (Figure-Text Attacks)")
    print("   - Download SafeBench from: https://github.com/ThuCCSLab/FigStep")
    print("   - Download benign questions from: https://github.com/leigest519/HiddenDetect/tree/main/data/FigStep")
    print("   - Place files at:")
    print("     - data/FigTxt/benign_questions.csv")
    print("     - data/FigTxt/safebench.csv")
    
    print("\n3. JailbreakV-28K (Multimodal Jailbreaks)")
    print("   - Download from: https://github.com/SaFoLab-WISC/JailBreakV_28K")
    print("   - Place files at:")
    print("     - data/JailbreakV-28K/mini_JailBreakV_28K.csv")
    print("     - data/JailbreakV-28K/figstep/")
    print("     - data/JailbreakV-28K/llm_transfer_attack/")
    print("     - data/JailbreakV-28K/query_related/")
    
    print("\n4. VAE Adversarial Images")
    print("   - Download from: https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models")
    print("   - Place files at:")
    print("     - data/VAE/manual_harmful_instructions.csv")
    print("     - data/VAE/adversarial_images/")

def main():
    """Main function to download all datasets"""
    print("="*80)
    print("VLM JAILBREAK DETECTION - DATASET DOWNLOADER")
    print("="*80)
    
    # Track success/failure
    success_count = 0
    total_count = 7  # Updated count including MM-SafetyBench

    # Download functions
    download_functions = [
        ("Alpaca", download_alpaca),
        ("AdvBench", download_advbench),
        ("DAN Prompts", download_dan_prompts),
        ("OpenAssistant", download_openassistant),
        ("VQAv2/VizWiz-VQA", download_vqav2),
        ("XSTest", download_xstest),
        ("MM-SafetyBench", download_mm_safety_bench)
    ]

    for name, func in download_functions:
        try:
            print(f"\n[{name}] Starting download...")
            func()
            success_count += 1
            print(f"[{name}] ✅ Completed successfully")
        except Exception as e:
            print(f"[{name}] ❌ Failed: {e}")

    # Print manual download instructions
    print_manual_instructions()

    print("\n" + "="*80)
    print("DATASET DOWNLOAD COMPLETE")
    print("="*80)
    print(f"✅ Successfully processed: {success_count}/{total_count} automatic datasets")
    print("⚠️  Manual download still required: MM-Vet, FigTxt, JailbreakV-28K, VAE")
    print("\nRun 'python verify_setup.py' to check your setup status.")

if __name__ == "__main__":
    main()
