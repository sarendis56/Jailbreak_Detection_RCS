import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os
import tempfile
from tqdm import tqdm
from feature_cache import FeatureCache
import gc
import psutil

# Import Qwen dependencies
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class HiddenStateExtractor:
    """Qwen-specific hidden state extractor with caching support"""
    
    def __init__(self, model_path, cache_dir="cache"):
        self.model_path = model_path
        self.cache = FeatureCache(cache_dir)
        self.model = None
        self.processor = None
        self.tokenizer = None

    def _get_memory_info(self):
        """Get current memory usage information"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        else:
            gpu_memory = gpu_memory_cached = gpu_max_memory = 0

        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        return {
            'gpu_allocated': gpu_memory,
            'gpu_cached': gpu_memory_cached,
            'gpu_max': gpu_max_memory,
            'cpu_used': cpu_memory
        }

    def _print_memory_info(self, prefix=""):
        """Print current memory usage"""
        info = self._get_memory_info()
        print(f"{prefix}GPU: {info['gpu_allocated']:.2f}GB allocated, {info['gpu_cached']:.2f}GB cached, {info['gpu_max']:.2f}GB max")

    def _get_model_layers(self):
        """Get the number of layers in the model"""
        if self.model is None:
            self._load_model()

        # For Qwen2.5-VL models, get the number of layers from the language model
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
            num_layers = len(self.model.language_model.model.layers)
        else:
            # Fallback: try to get from config
            if hasattr(self.model.config, 'num_hidden_layers'):
                num_layers = self.model.config.num_hidden_layers
            else:
                # Default fallback based on model name
                if '3B' in self.model_path or '3b' in self.model_path:
                    num_layers = 36
                elif '7B' in self.model_path or '7b' in self.model_path:
                    num_layers = 28
                else:
                    num_layers = 32

        print(f"Detected {num_layers} layers in model")
        return num_layers

    def get_default_layer_range(self):
        """Get the default layer range for this model (all layers)"""
        num_layers = self._get_model_layers()
        # For Qwen, the model returns embedding layer + transformer layers
        # So if model has 36 transformer layers, we get 37 hidden states (0-36)
        return 0, num_layers  # Include the extra embedding layer

    def _resize_image_if_needed(self, image_path, max_size=448):
        """Resize image if it's too large to prevent over-tokenization"""
        try:
            if image_path.startswith(('http://', 'https://')):
                # Handle URL images
                import requests
                from PIL import Image
                from io import BytesIO

                response = requests.get(image_path, timeout=10)
                img = Image.open(BytesIO(response.content))
                original_path = image_path
            else:
                # Handle local file images
                from PIL import Image
                img = Image.open(image_path)
                original_path = image_path

            # Check if image is too large and resize if needed
            width, height = img.size
            if max(width, height) > max_size:
                # Calculate new size maintaining aspect ratio
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)

                # Additional check: if total pixels still too large, resize more aggressively
                total_pixels = new_width * new_height
                if total_pixels > 200_000:  # ~448x448 = 200k pixels
                    # Force to square aspect ratio for very large images
                    square_size = min(max_size, 336)  # Even smaller for problematic images
                    new_width = square_size
                    new_height = square_size
                    # print(f"Aggressive resize: {width}x{height} -> {new_width}x{new_height} (forced square)")
                else:
                    # print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
                    pass

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert RGBA to RGB if necessary (for JPEG compatibility)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create a white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                # Save resized image temporarily
                import tempfile
                temp_path = tempfile.mktemp(suffix='.jpg')
                img.save(temp_path, 'JPEG', quality=95)

                # Store temp path for cleanup
                if not hasattr(self, '_temp_files'):
                    self._temp_files = []
                self._temp_files.append(temp_path)

                img.close()
                return temp_path
            else:
                # Image is reasonable size, use as-is
                img.close()
                return original_path

        except Exception as e:
            print(f"Warning: Error resizing image {image_path}: {e}")
            return image_path  # Return original path if resizing fails

    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                # Additional cleanup attempts
                torch.cuda.ipc_collect()
                # Force another garbage collection after CUDA cleanup
                gc.collect()
            except RuntimeError as e:
                print(f"Warning: CUDA cleanup failed: {e}")
                # Try basic cleanup even if advanced cleanup fails
                try:
                    torch.cuda.empty_cache()
                except:
                    pass

    def _load_model(self):
        """Load Qwen model only when needed"""
        if self.model is None:
            print(f"Loading Qwen model from {self.model_path}...")

            # Clear any existing GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load with more memory-efficient settings
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            # Load processor
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=False)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(self.model_path)

            # Get tokenizer from processor
            self.tokenizer = self.processor.tokenizer

            # Set model to eval mode to save memory
            self.model.eval()

            print(f"Model loaded successfully. Device: {self.model.device}")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def _unload_model(self):
        """Unload model to free GPU memory"""
        if self.model is not None:
            print("Unloading Qwen model from GPU memory...")
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Aggressive cleanup
        self._aggressive_cleanup()
        print("Qwen model unloaded successfully")

    def _prepare_qwen_messages(self, sample):
        """Prepare messages for Qwen2.5-VL based on sample format with memory optimization"""
        content = []

        # Handle different image formats with memory optimization
        if sample.get("img") is not None:
            img_data = sample["img"]

            # Case 1: File path(s) as string
            if isinstance(img_data, str):
                if "," in img_data:
                    # Multiple comma-separated paths
                    image_files = [path.strip() for path in img_data.split(",")]
                    for img_path in image_files:
                        # Resize each image if needed
                        resized_path = self._resize_image_if_needed(img_path)
                        if not resized_path.startswith(("http://", "https://", "file://")):
                            resized_path = f"file://{os.path.abspath(resized_path)}"
                        content.append({"type": "image", "image": resized_path})
                else:
                    # Single path - resize if too large
                    img_path = self._resize_image_if_needed(img_data)
                    if not img_path.startswith(("http://", "https://", "file://")):
                        img_path = f"file://{os.path.abspath(img_path)}"
                    content.append({"type": "image", "image": img_path})

            # Case 2: Binary image data - save to temp file and clean up immediately
            elif isinstance(img_data, bytes):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(img_data)
                    tmp_path = tmp_file.name
                content.append({"type": "image", "image": f"file://{tmp_path}"})
                # Note: temp file will be cleaned up by OS eventually

            # Case 3: List of images
            elif isinstance(img_data, list):
                for item in img_data:
                    if isinstance(item, str):
                        # Resize each image if needed
                        resized_item = self._resize_image_if_needed(item)
                        if not resized_item.startswith(("http://", "https://", "file://")):
                            resized_item = f"file://{os.path.abspath(resized_item)}"
                        content.append({"type": "image", "image": resized_item})
                    elif isinstance(item, bytes):
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                            tmp_file.write(item)
                            tmp_path = tmp_file.name
                        content.append({"type": "image", "image": f"file://{tmp_path}"})

        # Add text
        content.append({"type": "text", "text": sample["txt"]})

        # Return messages in the format expected by Qwen2.5-VL
        messages = [{"role": "user", "content": content}]
        return messages

    def extract_hidden_states(self, dataset, dataset_name, layer_start=None, layer_end=None,
                            use_cache=True, label_key='toxicity', batch_size=50,
                            memory_cleanup_freq=5, experiment_name=None):
        """Extract hidden states from Qwen model"""
        
        # Set default layer ranges if not provided
        if layer_start is None or layer_end is None:
            default_start, default_end = self.get_default_layer_range()
            if layer_start is None:
                layer_start = default_start
            if layer_end is None:
                layer_end = default_end
            print(f"Using automatic layer range: {layer_start}-{layer_end}")

        # Check cache first
        layer_range = (layer_start, layer_end)
        dataset_size = len(dataset)

        if use_cache and self.cache.exists(dataset_name, self.model_path, layer_range, dataset_size, experiment_name):
            print(f"Loading cached features for {dataset_name}...")
            return self.cache.load(dataset_name, self.model_path, layer_range, dataset_size, experiment_name)

        # Load model if not already loaded
        self._load_model()

        # Extract features
        all_hidden_states = {i: [] for i in range(layer_start, layer_end+1)}
        labels = []

        pbar = tqdm(dataset, desc=f"Extracting features from {dataset_name}", unit="sample")
        for sample_idx, sample in enumerate(pbar):
            try:
                # Periodic memory cleanup and monitoring
                if sample_idx % memory_cleanup_freq == 0:
                    self._aggressive_cleanup()
                    # if sample_idx % 20 == 0:  # Print memory info every 20 samples
                    #     self._print_memory_info(f"Sample {sample_idx}: ")

                # Prepare messages for Qwen2.5-VL
                messages = self._prepare_qwen_messages(sample)

                # Process the messages using the processor
                try:
                    # Check if this is a text-only sample (no images)
                    has_images = sample.get("img") is not None

                    if has_images:
                        # Process with vision info for multimodal samples
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        vision_info = process_vision_info(messages)
                        image_inputs = vision_info[0] if len(vision_info) > 0 else None
                        video_inputs = vision_info[1] if len(vision_info) > 1 else None
                    else:
                        # Text-only sample - use simple tokenizer approach to avoid vision processing
                        # For text-only samples, just use the text directly without vision processing
                        text_content = sample["txt"]
                        inputs = self.tokenizer(
                            text_content,
                            padding=True,
                            return_tensors="pt",
                            truncation=True,
                            max_length=8192
                        )
                        inputs = inputs.to(self.model.device)

                        # Skip the processor entirely for text-only samples
                        # and go directly to model inference
                        with torch.no_grad():
                            outputs = self.model(**inputs, output_hidden_states=True)

                        # Extract hidden states for the specified layers
                        hidden_states = outputs.hidden_states

                        # Get the last token's hidden state (similar to other processing)
                        last_token_idx = inputs.attention_mask.sum(dim=1) - 1

                        for layer_idx in range(layer_start, layer_end+1):
                            layer_hidden_states = hidden_states[layer_idx]
                            # Extract last token hidden state for each sample in batch
                            last_token_hidden = layer_hidden_states[0, last_token_idx[0], :].cpu().numpy()
                            all_hidden_states[layer_idx].append(last_token_hidden)

                        labels.append(sample.get(label_key, 0))
                        continue  # Skip the rest of the processing for text-only samples

                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    # Log pixel tensor size for monitoring (no fallback)
                    if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                        pixel_size = inputs.pixel_values.numel()  # Total number of elements
                        if pixel_size > 10_000_000:  # Just log large tensors
                            print(f"Info: Large pixel tensor ({pixel_size:,} elements) - processing with images")
                except Exception as e:
                    print(f"Error in processor for sample {sample_idx}: {e}")
                    # Re-raise the exception instead of fallback processing
                    raise e
                inputs = inputs.to(self.model.device)

                # Relax truncation to 8k tokens as Qwen2.5-VL can handle 32k
                seq_len = inputs.input_ids.shape[1]
                max_length = 8192  # Increased limit to 8k tokens
                if seq_len > max_length:
                    print(f"Warning: Truncating sequence from {seq_len} to {max_length} tokens")
                    inputs.input_ids = inputs.input_ids[:, :max_length]
                    if hasattr(inputs, 'attention_mask'):
                        inputs.attention_mask = inputs.attention_mask[:, :max_length]
                    # Also check if we have pixel_values and limit them
                    if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                        # If pixel values are too large, this might be causing issues
                        pixel_shape = inputs.pixel_values.shape
                        print(f"  Pixel values shape: {pixel_shape}")
                        # Check for extreme values in pixel data
                        if torch.isnan(inputs.pixel_values).any() or torch.isinf(inputs.pixel_values).any():
                            print(f"  Warning: NaN/Inf detected in pixel values, skipping sample")
                            # Add zero vectors for all layers
                            for layer_idx in range(layer_start, layer_end+1):
                                if len(all_hidden_states[layer_idx]) > 0:
                                    zero_vector = np.zeros_like(all_hidden_states[layer_idx][0])
                                else:
                                    zero_vector = np.zeros(2048)
                                all_hidden_states[layer_idx].append(zero_vector)
                            labels.append(sample.get(label_key, 0))
                            continue

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                    # Extract hidden states for specified layers (no NaN fallback)
                    for layer_idx in range(layer_start, layer_end+1):
                        if layer_idx < len(outputs.hidden_states):
                            hidden_state = outputs.hidden_states[layer_idx]
                            # Use last token representation and immediately move to CPU
                            last_token_hidden = hidden_state[:, -1, :].cpu().numpy().flatten()
                            all_hidden_states[layer_idx].append(last_token_hidden)
                            # Clear the hidden state from GPU memory
                            del hidden_state
                        else:
                            print(f"Warning: Layer {layer_idx} not found in outputs (only {len(outputs.hidden_states)} layers available)")
                            # This should not happen with correct layer detection, but if it does, raise an error
                            raise ValueError(f"Layer {layer_idx} not found in model outputs")

                # Extract label
                labels.append(sample.get(label_key, 0))

                # Explicit cleanup of large tensors
                del outputs, inputs
                if 'text' in locals():
                    del text
                if 'image_inputs' in locals() and image_inputs is not None:
                    del image_inputs
                if 'video_inputs' in locals() and video_inputs is not None:
                    del video_inputs

                # More frequent cleanup for Qwen (every sample)
                if sample_idx % 1 == 0:  # Clean up after every sample
                    self._aggressive_cleanup()

                # Update progress
                pbar.set_postfix({
                    'processed': f"{len(labels)}/{len(dataset)}",
                    'type': "📷" if sample.get('img') is not None else "📝"
                })

            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")

                # Emergency cleanup on error
                self._aggressive_cleanup()

                # Re-raise the exception instead of creating zero vectors
                raise e

        # Convert to numpy arrays
        for layer_idx in range(layer_start, layer_end+1):
            all_hidden_states[layer_idx] = np.array(all_hidden_states[layer_idx])

        # Cache results
        if use_cache:
            metadata = {
                'dataset_size': len(dataset),
                'label_key': label_key,
                'processed_samples': len(labels),
                'layer_start': layer_start,
                'layer_end': layer_end
            }
            self.cache.save(dataset_name, self.model_path, layer_range, all_hidden_states, 
                          labels, metadata, dataset_size, experiment_name)

        self._aggressive_cleanup()
        # Clean up temporary files
        self._cleanup_temp_files()
        return all_hidden_states, labels, len(labels)

    def _cleanup_temp_files(self):
        """Clean up temporary resized image files"""
        if hasattr(self, '_temp_files'):
            for temp_file in self._temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_file}: {e}")
            self._temp_files = []
