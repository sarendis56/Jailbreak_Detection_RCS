import os
import json
import re
import warnings
import csv
import requests
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, AutoConfig

warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")


refusal_lst = [
        "Sorry", "sorry",
        "unfortunately", "unfortunate", "sadly",
        "explicit", "deadly", "crim", "criminal", "illegal", "dangerous", "harmful", "warning", "alarm", "caution",
        "shame", "conspiracy",
        "Subject", "contrary", "shouldn"
    ]


def _load_internvl_model(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return tokenizer, model, config


def _tokenize(tokenizer, text: str):
    encoded = tokenizer(text, return_tensors="pt")
    return encoded


# Image tiling utilities adapted from InternVL3 HF examples
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_to_pixel_values(image, input_size=448, max_num=12, device=None, dtype=None):
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(tile) for tile in tiles]
    pixel_values = torch.stack(pixel_values)
    if dtype is not None:
        pixel_values = pixel_values.to(dtype)
    if device is not None:
        pixel_values = pixel_values.to(device)
    return pixel_values


def load_image_from_path_or_url(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert('RGB')
    return Image.open(path_or_url).convert('RGB')


def locate_most_safety_aware_layers(model_path: str):
    tokenizer, model, config = _load_internvl_model(model_path)

    # Load few-shot data (supports multimodal)
    json_file_path = "data/few_shot/few_shot.json"
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found. Please extract few_shot.zip first.")
        return None

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    few_shot_safe = []
    few_shot_unsafe = []
    for sample in data:
        img_field = sample.get("img")
        entry = {"txt": sample.get("txt", ""), "toxicity": sample.get("toxicity", 0), "img": None}
        if img_field and img_field != "null":
            # map relative paths to data/few_shot
            if isinstance(img_field, str) and not img_field.startswith(("http://", "https://", "/", "data/few_shot/")):
                entry["img"] = os.path.join("data/few_shot", img_field)
            else:
                entry["img"] = img_field
        if entry["toxicity"] == 1:
            few_shot_unsafe.append(entry)
        else:
            few_shot_safe.append(entry)

    print(f"Loaded few-shot dataset: {len(few_shot_safe)} safe, {len(few_shot_unsafe)} unsafe samples")
    if len(few_shot_safe) == 0 or len(few_shot_unsafe) == 0:
        print("Error: Not enough text-only samples for InternVL safety layer baseline")
        return None

    # Refusal tokens → ids
    refusal_token_ids = []
    for token in refusal_lst:
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                refusal_token_ids.extend(token_ids)
        except Exception:
            continue
    # Deduplicate
    seen = set()
    refusal_token_ids = [x for x in refusal_token_ids if not (x in seen or seen.add(x))]

    # Determine vocab size and lm_head
    lm = getattr(model, 'language_model', None)
    lm_head = None
    vocab_size = None
    if lm is not None and hasattr(lm, 'lm_head'):
        lm_head = lm.lm_head
        vocab_size = lm_head.weight.shape[0]
    else:
        try:
            out_emb = model.get_output_embeddings()
            if out_emb is not None:
                lm_head = out_emb
                vocab_size = lm_head.weight.shape[0]
        except Exception:
            pass
    if lm_head is None or vocab_size is None:
        print("Error: Could not access language model head for InternVL3")
        return None

    token_one_hot = torch.zeros(vocab_size)
    valid_tokens = 0
    for token_id in refusal_token_ids:
        if token_id < vocab_size:
            token_one_hot[token_id] = 1.0
            valid_tokens += 1
    print(f"Vocab size: {vocab_size}, Valid refusal tokens: {valid_tokens}")
    if valid_tokens == 0:
        print("Error: No valid refusal tokens found!")
        return None

    def forward_hidden(text: str, image_path: str | None):
        # For stability during hidden-state extraction, run text-only forward even if an image is present.
        batch = _tokenize(tokenizer, text)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        with torch.no_grad():
            if image_path:
                image = load_image_from_path_or_url(image_path)
                pixel_values = load_image_to_pixel_values(
                    image,
                    input_size=448,
                    max_num=12,
                    device=model.device,
                    dtype=getattr(model, 'dtype', torch.bfloat16)
                )
                # Build multimodal prompt with explicit IMG tokens matching InternVL3 forward
                num_patches = pixel_values.size(0)
                IMG_START_TOKEN = '<img>'
                IMG_END_TOKEN = '</img>'
                IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
                model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (getattr(model, 'num_image_token') * num_patches)) + IMG_END_TOKEN
                query = (text or '')
                if '<image>' in query:
                    query = query.replace('<image>', image_tokens, 1)
                else:
                    query = image_tokens + '\n' + query
                batch2 = _tokenize(tokenizer, query)
                input_ids = batch2["input_ids"].to(model.device)
                attention_mask = batch2.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(model.device)
                image_flags = torch.ones((num_patches, 1), device=model.device, dtype=torch.long)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_flags=image_flags,
                    output_hidden_states=True
                )
            else:
                # Text-only path: use underlying language model
                if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
                    out = model.language_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                else:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
        return out

    # Collect per-layer similarities for safe/unsafe
    F_safe = []
    F_unsafe = []

    for sample in few_shot_safe:
        text = sample['txt']
        img_path = sample.get('img')
        outputs = forward_hidden(text, img_path)
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            print("Error: InternVL outputs.hidden_states is None")
            return None
        # Skip embedding layer at index 0 to match LLaVA logic
        F_layers = []
        for hs in outputs.hidden_states[1:]:
            # Project last token to logits (disable autograd for inference tensors)
            with torch.no_grad():
                logits = lm_head(hs)
                next_token_logits = logits[:, -1, :]
                ref = token_one_hot.to(next_token_logits.device)
                cos_sim = F.cosine_similarity(next_token_logits, ref)
                F_layers.append(cos_sim.item())
        F_safe.append(F_layers)

    for sample in few_shot_unsafe:
        text = sample['txt']
        img_path = sample.get('img')
        outputs = forward_hidden(text, img_path)
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            print("Error: InternVL outputs.hidden_states is None")
            return None
        F_layers = []
        for hs in outputs.hidden_states[1:]:
            with torch.no_grad():
                logits = lm_head(hs)
                next_token_logits = logits[:, -1, :]
                ref = token_one_hot.to(next_token_logits.device)
                cos_sim = F.cosine_similarity(next_token_logits, ref)
                F_layers.append(cos_sim.item())
        F_unsafe.append(F_layers)

    if not F_safe or not F_unsafe:
        print("Error: Failed to compute layer similarities")
        return None

    F_safe_arr = np.mean(F_safe, axis=0)
    F_unsafe_arr = np.mean(F_unsafe, axis=0)
    FDV_arr = F_unsafe_arr - F_safe_arr

    positive_layers = [str(i) for i, v in enumerate(FDV_arr.tolist()) if v > 0]
    # Mark the true maximum FDV layer(s)
    max_fdv = float(np.max(FDV_arr))
    most_aware_layers = [str(i) for i, v in enumerate(FDV_arr.tolist()) if v == max_fdv]
    print("Safety awareness broadly exists at layer: " + ", ".join(positive_layers) + ".")
    print("The most safety-aware layers are: " + ", ".join(most_aware_layers) + ".")

    return FDV_arr.tolist(), most_aware_layers


if __name__ == "__main__":
    model_path = "model/internvl3-8b"

    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please download the model first.")
        exit(1)

    result = locate_most_safety_aware_layers(model_path)
    if result is not None:
        FDV, most_aware_layers = result

        results = {
            "FDV": FDV,
            "most_aware_layers": most_aware_layers,
            "model_path": model_path,
            "method": "predefined_refusal_tokens_text_only"
        }

        os.makedirs("results", exist_ok=True)
        output_file = "results/internvl3_safety_layers.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

        output_csv = "results/internvl3_safety_layers.csv"
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
