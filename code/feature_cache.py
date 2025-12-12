import pickle
import os
import re

class FeatureCache:
    """Feature caching system to avoid repeated model inference"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, dataset_name, model_path, layer_range, dataset_size=None, experiment_name=None):
        """Generate readable cache key based on dataset and model parameters"""
        # Extract model name from path
        model_name = os.path.basename(model_path.rstrip('/'))
        if not model_name:
            model_name = os.path.basename(os.path.dirname(model_path))

        # Clean dataset name (remove special characters)
        clean_dataset_name = re.sub(r'[^\w\-_]', '_', dataset_name)

        # Build readable cache key components
        components = [
            clean_dataset_name,
            model_name,
            f"layers_{layer_range[0]}-{layer_range[1]}"
        ]

        # Add dataset size if provided
        if dataset_size is not None:
            components.append(f"size_{dataset_size}")

        # Add experiment name if provided
        if experiment_name is not None:
            clean_exp_name = re.sub(r'[^\w\-_]', '_', experiment_name)
            components.append(f"exp_{clean_exp_name}")

        # Join components with underscores
        readable_key = "_".join(components)
        return readable_key

    def _get_cache_path(self, cache_key):
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def exists(self, dataset_name, model_path, layer_range, dataset_size=None, experiment_name=None):
        """Check if cached features exist"""
        cache_key = self._get_cache_key(dataset_name, model_path, layer_range, dataset_size, experiment_name)
        cache_path = self._get_cache_path(cache_key)
        return os.path.exists(cache_path)

    def save(self, dataset_name, model_path, layer_range, hidden_states, labels, metadata=None, dataset_size=None, experiment_name=None):
        """Save extracted features to cache"""
        # Get dataset size from the data if not provided
        if dataset_size is None and hasattr(labels, '__len__'):
            dataset_size = len(labels)

        cache_key = self._get_cache_key(dataset_name, model_path, layer_range, dataset_size, experiment_name)
        cache_path = self._get_cache_path(cache_key)

        cache_data = {
            'hidden_states': hidden_states,
            'labels': labels,
            'metadata': metadata or {},
            'dataset_name': dataset_name,
            'model_path': model_path,
            'layer_range': layer_range,
            'dataset_size': dataset_size,
            'experiment_name': experiment_name
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"Features cached to {cache_path}")
        print(f"Cache key: {cache_key}")

    def load(self, dataset_name, model_path, layer_range, dataset_size=None, experiment_name=None):
        """Load cached features"""
        cache_key = self._get_cache_key(dataset_name, model_path, layer_range, dataset_size, experiment_name)
        cache_path = self._get_cache_path(cache_key)

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        print(f"Features loaded from cache: {cache_path}")
        print(f"Cache key: {cache_key}")
        return cache_data['hidden_states'], cache_data['labels'], cache_data.get('metadata', {})
    
    def get_cache_entry(self, dataset_name=None, model_path=None, dataset_size=None, experiment_name=None):
        """
        Retrieve cached metadata for a specific dataset/model combination without knowing layer range.
        Returns the cache data dict if a matching entry is found, otherwise None.
        """
        if not os.path.exists(self.cache_dir):
            return None

        for file in os.listdir(self.cache_dir):
            if not file.endswith('.pkl'):
                continue
            cache_path = os.path.join(self.cache_dir, file)
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
            except Exception:
                continue

            if dataset_name is not None and cache_data.get('dataset_name') != dataset_name:
                continue
            if model_path is not None and cache_data.get('model_path') != model_path:
                continue
            if dataset_size is not None and cache_data.get('dataset_size') != dataset_size:
                continue
            if experiment_name is not None and cache_data.get('experiment_name') != experiment_name:
                continue

            cache_data['cache_path'] = cache_path
            return cache_data

        return None
    
    def list_cache_files(self):
        """List all cached files with readable information"""
        cache_files = []
        if not os.path.exists(self.cache_dir):
            print("Cache directory does not exist")
            return cache_files

        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, file)
                try:
                    # Get file size
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

                    # Try to load metadata
                    with open(file_path, 'rb') as f:
                        cache_data = pickle.load(f)

                    cache_info = {
                        'filename': file,
                        'size_mb': file_size,
                        'dataset_name': cache_data.get('dataset_name', 'unknown'),
                        'model_path': cache_data.get('model_path', 'unknown'),
                        'layer_range': cache_data.get('layer_range', 'unknown'),
                        'dataset_size': cache_data.get('dataset_size', 'unknown'),
                        'experiment_name': cache_data.get('experiment_name', 'none'),
                        'processed_samples': cache_data.get('metadata', {}).get('processed_samples', 'unknown')
                    }
                    cache_files.append(cache_info)
                except Exception as e:
                    print(f"Error reading cache file {file}: {e}")

        # Sort by filename for consistent ordering
        cache_files.sort(key=lambda x: x['filename'])

        if cache_files:
            print(f"\nFound {len(cache_files)} cache files:")
            print("-" * 120)
            print(f"{'Filename':<50} {'Size(MB)':<10} {'Dataset':<20} {'Layers':<12} {'Samples':<10} {'Experiment':<15}")
            print("-" * 120)
            for info in cache_files:
                print(f"{info['filename']:<50} {info['size_mb']:<10.1f} {info['dataset_name']:<20} "
                      f"{info['layer_range']:<12} {info['processed_samples']:<10} {info['experiment_name']:<15}")
        else:
            print("No cache files found")

        return cache_files

    def clear(self, dataset_name=None, model_path=None, pattern=None):
        """Clear cache files"""
        if dataset_name is None and model_path is None and pattern is None:
            # Clear all cache
            count = 0
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
                    count += 1
            print(f"Cleared {count} cache files")
        elif pattern is not None:
            # Clear files matching pattern
            count = 0
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl') and pattern in file:
                    os.remove(os.path.join(self.cache_dir, file))
                    count += 1
            print(f"Cleared {count} cache files matching pattern '{pattern}'")
        else:
            # Clear specific cache (implementation can be extended)
            print("Specific cache clearing not implemented yet")
