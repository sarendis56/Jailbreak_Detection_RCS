import os
import re
import gc
import psutil
import requests
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel, AutoConfig

from feature_cache import FeatureCache


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
    if isinstance(path_or_url, str) and (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert('RGB')
    return Image.open(path_or_url).convert('RGB')


class HiddenStateExtractor:
    """InternVL hidden state extractor with caching support (API-compatible)."""
    
    def __init__(self, model_path, cache_dir="cache"):
        self.model_path = model_path
        self.cache = FeatureCache(cache_dir)
        self.model = None
        self.tokenizer = None
        self.config = None

    def _get_memory_info(self):
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
        else:
            gpu_memory = gpu_memory_cached = 0
        cpu_memory = psutil.virtual_memory().used / 1024**3
        return {
            'gpu_allocated': gpu_memory,
            'gpu_cached': gpu_memory_cached,
            'cpu_used': cpu_memory
        }

    def _aggressive_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.ipc_collect()
            except Exception:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _load_model(self):
        if self.model is None:
            print(f"Loading InternVL model from {self.model_path}...")
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            # Use float16 on CUDA to avoid unsupported bfloat16 kernels on some builds
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=True,
                device_map="auto"
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)

    def _get_model_layers(self):
        self._load_model()
        num_layers = None
        # Prefer language model's transformer depth if accessible
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
            num_layers = len(self.model.language_model.model.layers)
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
            num_layers = self.model.config.num_hidden_layers
        elif hasattr(self.config, 'num_hidden_layers'):
            num_layers = self.config.num_hidden_layers
        else:
            # Conservative default for 8B backbones
            num_layers = 32
        print(f"Detected {num_layers} layers in InternVL model")
        return num_layers

    def get_default_layer_range(self):
        num_layers = self._get_model_layers()
        # Include embedding layer index 0; transformer layers follow
        return 0, num_layers

    def _tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt")

    def _prepare_multimodal_inputs(self, text: str, image_path: str):
        image = load_image_from_path_or_url(image_path)
        pixel_values = load_image_to_pixel_values(
            image,
            input_size=448,
            max_num=12,
            device=self.model.device,
            dtype=getattr(self.model, 'dtype', torch.bfloat16)
        )
        num_patches = pixel_values.size(0)
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (getattr(self.model, 'num_image_token') * num_patches)) + IMG_END_TOKEN
        query = (text or '')
        if '<image>' in query:
            query = query.replace('<image>', image_tokens, 1)
        else:
            query = image_tokens + '\n' + query
        enc = self._tokenize(query)
        input_ids = enc['input_ids'].to(self.model.device)
        attention_mask = enc.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        image_flags = torch.ones((num_patches, 1), device=self.model.device, dtype=torch.long)
        return input_ids, attention_mask, pixel_values, image_flags

    def extract_hidden_states(self, dataset, dataset_name, layer_start=None, layer_end=None,
                              use_cache=True, label_key='toxicity', batch_size=10, memory_cleanup_freq=5, experiment_name=None):
        if layer_start is None or layer_end is None:
            default_start, default_end = self.get_default_layer_range()
            if layer_start is None:
                layer_start = default_start
            if layer_end is None:
                layer_end = default_end
            print(f"Using automatic layer range: {layer_start}-{layer_end}")

        layer_range = (layer_start, layer_end)
        dataset_size = len(dataset)
        if use_cache and self.cache.exists(dataset_name, self.model_path, layer_range, dataset_size, experiment_name):
            print(f"Loading cached features for {dataset_name} (size: {dataset_size}, layers: {layer_start}-{layer_end})...")
            return self.cache.load(dataset_name, self.model_path, layer_range, dataset_size, experiment_name)

        self._load_model()

        total_samples = len(dataset)
        print(f"Processing {total_samples} samples directly...")

        all_hidden_states = {i: [] for i in range(layer_start, layer_end+1)}
        labels = []

        pbar = tqdm(total=total_samples, desc=f"Extracting hidden states for {dataset_name}", unit="sample")

        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch = dataset[batch_start:batch_end]

            batch_hidden_states = {i: [] for i in range(layer_start, layer_end+1)}
            batch_labels = []

            for idx, sample in enumerate(batch):
                try:
                    if (batch_start + idx) % memory_cleanup_freq == 0:
                        self._aggressive_cleanup()
                        mem = self._get_memory_info()
                        if mem['gpu_allocated'] > 20:
                            print(f"High GPU memory usage: {mem['gpu_allocated']:.1f}GB allocated, {mem['gpu_cached']:.1f}GB cached")

                    text = sample.get('txt', '')
                    img_path = sample.get('img')

                    with torch.inference_mode():
                        if img_path is not None:
                            input_ids, attention_mask, pixel_values, image_flags = self._prepare_multimodal_inputs(text, img_path)
                            out = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                image_flags=image_flags,
                                output_hidden_states=True
                            )
                        else:
                            enc = self._tokenize(text)
                            input_ids = enc['input_ids'].to(self.model.device)
                            attention_mask = enc.get('attention_mask')
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.model.device)
                            if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
                                out = self.model.language_model.model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True
                                )
                            else:
                                out = self.model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True
                                )

                    # Collect last-token hidden state for each requested layer
                    for layer_idx in range(layer_start, layer_end+1):
                        if layer_idx < len(out.hidden_states):
                            hidden_state = out.hidden_states[layer_idx]
                            last_token_hidden = hidden_state[:, -1, :].cpu().numpy().flatten()
                            batch_hidden_states[layer_idx].append(last_token_hidden)
                        else:
                            if len(batch_hidden_states[layer_idx]) > 0:
                                zero_vector = np.zeros_like(batch_hidden_states[layer_idx][0])
                            else:
                                zero_vector = np.zeros(4096)
                            batch_hidden_states[layer_idx].append(zero_vector)

                    batch_labels.append(sample.get(label_key, 0))

                    # Cleanup tensors
                    del out
                    del input_ids
                    if 'pixel_values' in locals():
                        del pixel_values

                    processed = len(labels) + len(batch_labels)
                    has_image = "📷" if img_path is not None else "📝"
                    pbar.set_postfix({
                        'processed': f"{processed}/{total_samples}",
                        'type': has_image,
                        label_key: sample.get(label_key, 0),
                        'batch': f"{batch_start//batch_size + 1}/{(total_samples-1)//batch_size + 1}"
                    })
                    pbar.update(1)

                except Exception as e:
                    print(f"Error processing sample {batch_start + idx}: {e}")
                    self._aggressive_cleanup()
                    continue

            # Merge batch
            for layer_idx in range(layer_start, layer_end+1):
                all_hidden_states[layer_idx].extend(batch_hidden_states[layer_idx])
            labels.extend(batch_labels)

            del batch_hidden_states
            del batch_labels
            self._aggressive_cleanup()

        pbar.close()

        if use_cache:
            metadata = {
                'dataset_size': len(dataset),
                'label_key': label_key,
                'processed_samples': len(labels)
            }
            self.cache.save(dataset_name, self.model_path, layer_range, all_hidden_states, labels,
                            metadata, dataset_size, experiment_name)

        self._aggressive_cleanup()
        final_mem = self._get_memory_info()
        print(f"Feature extraction completed. GPU: {final_mem['gpu_allocated']:.1f}GB alloc, {final_mem['gpu_cached']:.1f}GB cached; CPU: {final_mem['cpu_used']:.1f}GB used")

        return all_hidden_states, labels, {}
