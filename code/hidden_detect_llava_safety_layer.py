import torch
import torch.nn.functional as N
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import re
import json
import time
import os
import warnings
import csv

# Suppress PyTorch model loading warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


refusal_lst = [
        "Sorry", "sorry",
        "unfortunately", "unfortunate", "sadly",
        "explicit", "deadly", "crim", "criminal", "illegal", "dangerous", "harmful", "warning", "alarm", "caution",
        "shame", "conspiracy",
        "Subject", "contrary", "shouldn"
    ]
vocab_size = 32000

# few-shot safe and unsafe datasets to locate the most safety awareness layers
def locate_most_safety_aware_layers(model_path):    
    model_name = get_model_name_from_path(model_path)

    # Try float32 first to avoid numerical instability
    try:
        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float32
        }
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            **kwargs
        )
        print("Loaded model with float32 precision")
    except Exception as e:
        print(f"Float32 failed ({e}), falling back to float16")
        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16
        }
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            **kwargs
        )

            
    def find_conv_mode(model_name):
        # select conversation mode based on the model name
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"  
        return conv_mode    
        
    def adjust_query_for_images(qs, num_images=1):   
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        # Build the repeated image token prefix according to number of images
        if model.config.mm_use_im_start_end:
            token_str = image_token_se
        else:
            token_str = DEFAULT_IMAGE_TOKEN

        prefix = (token_str + "\n") * max(0, int(num_images))

        if IMAGE_PLACEHOLDER in qs:
            # Replace placeholders with the correct repeated tokens
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, (image_token_se), qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            qs = prefix + qs

        return qs

    def construct_conv_prompt(sample, num_images=0):        
        conv = conv_templates[find_conv_mode(model_name)].copy()  
        if (sample['img'] != None):     
            qs = adjust_query_for_images(sample['txt'], num_images=num_images)
        else:
            qs = sample['txt']
        conv.append_message(conv.roles[0], qs)  
        conv.append_message(conv.roles[1], None)       
        prompt = conv.get_prompt()
        return prompt

    def load_image(image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(image_files):
        out = []
        for image_file in image_files:
            image = load_image(image_file)
            out.append(image)
        return out
    
    def load_image_from_bytes(image_data):    
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def load_images_from_bytes(image_data_list):
       return [load_image_from_bytes(data) for data in image_data_list]

    def prepare_imgs_tensor_both_cases(sample):       
        try:
            # Case 1: Comma-separated file paths
            if isinstance(sample['img'], str):
                image_files_path = sample['img'].split(",")
                img_prompt = load_images(image_files_path)
                num_images = len(img_prompt)
            # Case 2: Single binary image
            elif isinstance(sample['img'], bytes):
                img_prompt = [load_image_from_bytes(sample['img'])]
                num_images = 1
            # Case 3: List of binary images
            elif isinstance(sample['img'], list):
                # Check if all elements in the list are bytes
                if all(isinstance(item, bytes) for item in sample['img']):
                    img_prompt = load_images_from_bytes(sample['img'])
                    num_images = len(img_prompt)
                else:
                    raise ValueError("List contains non-bytes data.")

            else:
                raise ValueError("Unsupported data type in sample['img']. "
                                "Expected str, bytes, or list of bytes.")
            # Compute sizes
            images_size = [img.size for img in img_prompt if img is not None]
            # Process images into tensor
            images_tensor = process_images(img_prompt, image_processor, model.config)
            images_tensor = images_tensor.to(model.device, dtype=model.dtype)
            return images_tensor, images_size, num_images

        except Exception as e:
            print(f"Error preparing image tensors: {e}")
            return None, None, 0
        
    few_shot_safe = []
    few_shot_unsafe = []
    json_file_path = "data/few_shot/few_shot.json"

    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found. Please extract few_shot.zip first.")
        return None

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for sample in data:
        if sample.get("img") == "null":
            sample["img"] = None
        elif sample.get("img") is not None:
            if isinstance(sample["img"], str):
                # Ensure images are referenced under data/few_shot/
                if sample["img"].startswith("few_shot/"):
                    sample["img"] = sample["img"].replace("few_shot/", "data/few_shot/")
                elif not sample["img"].startswith(("http://", "https://", "/", "data/few_shot/")):
                    sample["img"] = f"data/few_shot/{sample['img']}"

        if sample.get("toxicity", 0) == 1:
            few_shot_unsafe.append(sample)
        else:
            few_shot_safe.append(sample)

    print(f"Loaded few-shot dataset: {len(few_shot_safe)} safe, {len(few_shot_unsafe)} unsafe samples")

    if len(few_shot_safe) == 0 or len(few_shot_unsafe) == 0:
        print("Error: Could not create few-shot datasets")
        return None

    # Encode refusal tokens with better error handling
    refusal_token_ids = []
    for token in refusal_lst:
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if token_ids:  # Make sure we got valid token IDs
                refusal_token_ids.extend(token_ids)
        except:
            continue

    # Remove duplicates while preserving order
    seen = set()
    refusal_token_ids = [x for x in refusal_token_ids if not (x in seen or seen.add(x))]

    print(f"Using {len(refusal_token_ids)} refusal tokens for LLaVA")
    print(f"Sample refusal token IDs: {refusal_token_ids[:10]}")

    # Use the actual model vocab size for the one-hot tensor
    actual_vocab_size = model.lm_head.out_features
    token_one_hot = torch.zeros(actual_vocab_size)
    valid_tokens = 0
    for token_id in refusal_token_ids:
        if token_id < actual_vocab_size:  # Make sure token ID is within bounds
            token_one_hot[token_id] = 1.0
            valid_tokens += 1

    print(f"Vocab size: {actual_vocab_size}, Valid refusal tokens: {valid_tokens}")
    print(f"Refusal token one-hot sum: {token_one_hot.sum().item()}")

    if valid_tokens == 0:
        print("Error: No valid refusal tokens found!")
        return None

    lm_head = model.lm_head    
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        norm = model.transformer.ln_f
    else:
        raise ValueError(f"Incorrect Model") 
    
    F_safe = []
    F_unsafe = []

    print("Processing safe samples...")
    for sample in few_shot_safe:
        F = [] 
        if sample['img'] == None:
            prompt = construct_conv_prompt(sample)
            input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .to(model.device)
                )   
            with torch.no_grad():   
                outputs = model(input_ids, images=None, image_sizes=None, output_hidden_states=True)        
        else:
            # Prepare images and align image tokens
            images_tensor, images_size, num_images = prepare_imgs_tensor_both_cases(sample)
            if images_tensor is None or images_size is None or num_images == 0:
                print("Skipping sample due to image preparation failure.")
                continue
            prompt = construct_conv_prompt(sample, num_images=num_images)
            input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .to(model.device)
                )      
            with torch.no_grad(): 
                outputs = model(input_ids, images=images_tensor, image_sizes=images_size, output_hidden_states=True)  
              

        for i, r in enumerate(outputs.hidden_states[1:]):                   
            layer_output = norm(r)                    
            logits = lm_head(layer_output)                 
            next_token_logits = logits[:, -1, :]
            reference_logits = token_one_hot.to(next_token_logits.device)                 
            cos_sim = N.cosine_similarity(next_token_logits, reference_logits)
            F.append(cos_sim.item())                  
        F_safe.append(F)

    # Average over the few-shot safe dataset
    if not F_safe:
        print("Error: No safe samples processed successfully!")
        return None

    F_safe_arr = np.mean(F_safe, axis=0)
    F_safe = F_safe_arr.tolist()
    print(f"Safe samples processed: {len(F_safe)} layers, sample values: {F_safe[:5]}")

    print("Processing unsafe samples...")
    for sample in few_shot_unsafe:
        F = []
        if sample['img'] == None:
            prompt = construct_conv_prompt(sample)
            input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .to(model.device)
                )   
            with torch.no_grad():   
                outputs = model(input_ids, images=None, image_sizes=None, output_hidden_states=True)       
        else:
            images_tensor, images_size, num_images = prepare_imgs_tensor_both_cases(sample)
            if images_tensor is None or images_size is None or num_images == 0:
                print("Skipping sample due to image preparation failure.")
                continue
            prompt = construct_conv_prompt(sample, num_images=num_images)
            input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .to(model.device)
                )      
            with torch.no_grad(): 
                outputs = model(input_ids, images=images_tensor, image_sizes=images_size, output_hidden_states=True)      

        for i, r in enumerate(outputs.hidden_states[1:]):                
            layer_output = norm(r)                   
            logits = lm_head(layer_output)              
            next_token_logits = logits[:, -1, :]          
            reference_logits = token_one_hot.to(next_token_logits.device)          
            cos_sim = N.cosine_similarity(next_token_logits, reference_logits)
            F.append(cos_sim.item())       
        F_unsafe.append(F)

    # Average over the few-shot unsafe dataset
    if not F_unsafe:
        print("Error: No unsafe samples processed successfully!")
        return None, None

    F_unsafe_arr = np.mean(F_unsafe, axis=0)
    F_unsafe = F_unsafe_arr.tolist()
    print(f"Unsafe samples processed: {len(F_unsafe)} layers, sample values: {F_unsafe[:5]}")

    # Refusal discrepancy vector
    FDV_arr = F_unsafe_arr - F_safe_arr
    FDV = FDV_arr.tolist()

    # Safety awareness exists broadly
    positive_layers = [str(i) for i, item in enumerate(FDV) if item > 0]
    print("Safety awareness broadly exists at layer: " + ", ".join(positive_layers) + ".")

    # Locate the most safety-aware layers by using the last layer as baseline
    most_aware_layers = [str(i) for i, item in enumerate(FDV) if item > FDV[-1]]
    print("The most safety-aware layers are: " + ", ".join(most_aware_layers) + ".")

    return FDV, most_aware_layers


if __name__ == "__main__":
    model_path = "model/llava-v1.6-vicuna-7b/"

    print("="*80)
    print("LLAVA SAFETY LAYER DETECTION")
    print("="*80)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please download the model first.")
        exit(1)

    result = locate_most_safety_aware_layers(model_path)

    if result is not None:
        FDV, most_aware_layers = result
        print(f"\nFull FDV (Refusal Discrepancy Vector): {FDV}")
        print(f"Most safety-aware layers: {most_aware_layers}")

        # Save results
        results = {
            "FDV": FDV,
            "most_aware_layers": most_aware_layers,
            "model_path": model_path,
            "method": "predefined_refusal_tokens"
        }

        output_file = "results/llava_safety_layers.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

        # Also save a CSV with per-layer FDV and whether it's among the most-aware layers
        output_csv = "results/llava_safety_layers.csv"
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "fdv", "is_most_aware"]) 
            most_aware_set = set(most_aware_layers)
            for layer_idx, fdv_value in enumerate(FDV):
                is_most = str(layer_idx) in most_aware_set
                writer.writerow([layer_idx, fdv_value, is_most])
        print(f"CSV saved to {output_csv}")
    else:
        print("Failed to compute safety layers")
   