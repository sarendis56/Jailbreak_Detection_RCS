import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
import re
import hashlib
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, classification_report, f1_score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

class GenericMLPClassifier(nn.Module):
    """Generic MLP classifier for both binary and multi-class classification"""
    
    def __init__(self, input_dim, output_dim=1, hidden_dims=[512, 256], 
                 dropout_rate=0.3, activation='relu', output_activation=None):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (1 for binary, num_classes for multi-class)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'tanh', 'gelu')
            output_activation: Output activation ('sigmoid' for binary, None for multi-class)
        """
        super(GenericMLPClassifier, self).__init__()

        self.output_dim = output_dim
        self.output_activation = output_activation
        self.hidden_dims = hidden_dims

        # Build layers separately to access hidden activations
        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.activations.append(self._get_activation(activation))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Output activation
        self.final_activation = None
        if output_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)

        # Store hidden activations for regularization
        self.hidden_activations = []
    
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        # Clear previous activations
        self.hidden_activations = []

        # Forward through hidden layers
        for i, (linear, activation, dropout) in enumerate(zip(self.hidden_layers, self.activations, self.dropouts)):
            x = linear(x)
            x = activation(x)
            # Store hidden activations for regularization
            self.hidden_activations.append(x)
            x = dropout(x)

        # Output layer
        x = self.output_layer(x)

        # Final activation
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def get_linearity_regularization(self):
        """Calculate linearity regularization term (L2 norm of hidden activations)"""
        if not self.hidden_activations:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        reg_loss = 0.0
        for activation in self.hidden_activations:
            # L2 norm of activations encourages smaller, more linear responses
            reg_loss += torch.mean(activation ** 2)

        return reg_loss / len(self.hidden_activations)

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

        # Ensure filename is not too long (max 255 chars for most filesystems)
        if len(readable_key) > 200:
            # If too long, use hash for the dataset name part only
            dataset_hash = hashlib.md5(clean_dataset_name.encode()).hexdigest()[:8]
            components[0] = f"dataset_{dataset_hash}"
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

def train_generic_classifier(X_train, y_train, input_dim, output_dim=1,
                           hidden_dims=[512, 256], dropout_rate=0.3,
                           epochs=50, batch_size=32, lr=0.001,
                           task_type='binary', verbose=True, optimize_threshold=True,
                           linearity_reg=0.0, class_weight='balanced'):
    """
    Train generic MLP classifier with optional threshold optimization and linearity regularization

    Args:
        X_train, y_train: Training data
        input_dim: Input feature dimension
        output_dim: Output dimension (1 for binary, num_classes for multi-class)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        task_type: 'binary' or 'multiclass'
        verbose: Print training progress
        optimize_threshold: Whether to optimize threshold for binary classification
        linearity_reg: Linearity regularization strength (penalizes hidden activations)
        class_weight: Class weighting strategy ('balanced', 'none', or dict of weights)
                     'balanced' automatically adjusts weights inversely proportional to class frequencies
    """
    # Use multiple GPUs if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_multi_gpu = torch.cuda.device_count() > 1
    if verbose and torch.cuda.is_available():
        print(f"Using device: {device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        if use_multi_gpu:
            print("Multi-GPU training enabled")

    # Compute class weights if needed
    class_weights = None
    pos_weight = None

    if class_weight == 'balanced':
        unique_classes = np.unique(y_train)
        if task_type == 'binary':
            # For binary classification, compute pos_weight for BCELoss
            n_samples = len(y_train)
            n_positive = np.sum(y_train == 1)
            n_negative = np.sum(y_train == 0)
            if n_positive > 0 and n_negative > 0:
                pos_weight = torch.FloatTensor([n_negative / n_positive]).to(device)
                if verbose:
                    print(f"Class imbalance detected: {n_negative}:{n_positive} (negative:positive)")
                    print(f"Using pos_weight: {pos_weight.item():.4f}")
        else:
            # For multiclass, compute class weights for CrossEntropyLoss
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weights = torch.FloatTensor(class_weights).to(device)
            if verbose:
                print(f"Using balanced class weights: {class_weights.cpu().numpy()}")
    elif isinstance(class_weight, dict):
        # Custom class weights provided
        if task_type == 'binary':
            if 1 in class_weight and 0 in class_weight:
                pos_weight = torch.FloatTensor([class_weight[1] / class_weight[0]]).to(device)
        else:
            unique_classes = np.unique(y_train)
            class_weights = torch.FloatTensor([class_weight.get(cls, 1.0) for cls in unique_classes]).to(device)

    # Determine output activation and loss function
    use_logits_loss = pos_weight is not None
    if task_type == 'binary':
        if use_logits_loss:
            output_activation = None  # BCEWithLogitsLoss includes sigmoid
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            output_activation = 'sigmoid'
            criterion = nn.BCELoss()
    else:  # multiclass
        output_activation = None
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create model
    model = GenericMLPClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        output_activation=output_activation
    ).to(device)

    # Enable multi-GPU training if available
    if use_multi_gpu:
        model = nn.DataParallel(model)
        if verbose:
            print(f"Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optimize batch size for multi-GPU training
    effective_batch_size = batch_size * torch.cuda.device_count() if use_multi_gpu else batch_size
    if verbose and use_multi_gpu:
        print(f"Effective batch size: {effective_batch_size} (base: {batch_size} × {torch.cuda.device_count()} GPUs)")

    # Create data loaders with optimized settings
    # Keep data on CPU and move to GPU during training for better memory management
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),  # Keep on CPU
        torch.FloatTensor(y_train) if task_type == 'binary' else torch.LongTensor(y_train)  # Keep on CPU
    )

    # Use multiple workers for data loading to utilize CPU cores
    num_workers = min(8, torch.get_num_threads())  # Use up to 8 workers or available threads
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False
    )

    if verbose:
        print(f"DataLoader: batch_size={effective_batch_size}, num_workers={num_workers}, pin_memory={torch.cuda.is_available()}")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Move data to GPU
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_X)

            if task_type == 'binary':
                outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)

                # Handle predictions differently for BCELoss vs BCEWithLogitsLoss
                if use_logits_loss:
                    # BCEWithLogitsLoss: apply sigmoid then threshold
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    # BCELoss: outputs are already sigmoid, just threshold
                    predicted = (outputs > 0.5).float()
            else:  # multiclass
                loss = criterion(outputs, batch_y)
                _, predicted = torch.max(outputs.data, 1)

            # Add linearity regularization
            if linearity_reg > 0.0:
                reg_loss = model.get_linearity_regularization()
                loss = loss + linearity_reg * reg_loss

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            train_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Optimize threshold for binary classification
    optimal_threshold = 0.5  # default
    if task_type == 'binary' and optimize_threshold:
        optimal_threshold = _find_optimal_threshold(model, X_train, y_train, device, use_logits_loss)
        if verbose:
            print(f"Optimal threshold: {optimal_threshold:.4f}")

    # Store threshold in model for later use
    model.optimal_threshold = optimal_threshold

    return model

def _find_optimal_threshold(model, X_train, y_train, device, use_logits_loss=False):
    """Find optimal threshold using F1 score on training data"""
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        raw_outputs = model(X_train_tensor).squeeze()

        # Apply sigmoid if using logits loss (outputs are raw logits)
        if use_logits_loss:
            train_predictions = torch.sigmoid(raw_outputs).cpu().numpy()
        else:
            train_predictions = raw_outputs.cpu().numpy()

    # Try different thresholds and find the one with best F1 score
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (train_predictions > threshold).astype(int)

        # Calculate F1 score
        tp = np.sum((y_train == 1) & (y_pred == 1))
        fp = np.sum((y_train == 0) & (y_pred == 1))
        fn = np.sum((y_train == 1) & (y_pred == 0))

        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

    model.train()
    return best_threshold

def evaluate_generic_classifier(model, X_test, y_test, task_type='binary', categories=None, use_optimal_threshold=True):
    """
    Evaluate generic classifier with proper AUROC/AUPRC calculation

    Args:
        model: Trained model
        X_test, y_test: Test data
        task_type: 'binary' or 'multiclass'
        categories: List of category names for multiclass
        use_optimal_threshold: Whether to use model's optimal threshold for binary classification
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)

        if task_type == 'binary':
            predictions = outputs.squeeze().cpu().numpy()
            # Use optimal threshold if available and requested
            threshold = getattr(model, 'optimal_threshold', 0.5) if use_optimal_threshold else 0.5
            y_pred_binary = (predictions > threshold).astype(int)
        else:  # multiclass
            predictions = F.softmax(outputs, dim=1).cpu().numpy()
            y_pred_binary = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    # Calculate AUROC, AUPRC, TPR, FPR for binary classification
    if task_type == 'binary':
        # Check if we have both classes in test set
        unique_classes = np.unique(y_test)
        if len(unique_classes) > 1:
            try:
                # Calculate TPR, FPR from confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity/Recall)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

                # Ensure predictions are probabilities, not binary
                if np.all((predictions == 0) | (predictions == 1)):
                    print("Warning: Predictions are binary, AUROC/AUPRC may not be meaningful")

                fpr_curve, tpr_curve, _ = roc_curve(y_test, predictions)
                auroc = auc(fpr_curve, tpr_curve)

                precision, recall, _ = precision_recall_curve(y_test, predictions)
                auprc = auc(recall, precision)

                # Validate AUC values
                if np.isnan(auroc) or auroc < 0 or auroc > 1:
                    print(f"Warning: Invalid AUROC value: {auroc}")
                    auroc = 0.0
                if np.isnan(auprc) or auprc < 0 or auprc > 1:
                    print(f"Warning: Invalid AUPRC value: {auprc}")
                    auprc = 0.0

            except Exception as e:
                print(f"Warning: Could not calculate metrics: {e}")
                print(f"Predictions range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
                print(f"Unique predictions: {len(np.unique(predictions))}")
                tpr = float('nan')
                fpr = float('nan')
                auroc = 0.0
                auprc = 0.0
        else:
            # Single class case - provide more informative metrics
            single_class = unique_classes[0]
            class_name = "malicious" if single_class == 1 else "benign"

            # Calculate prediction confidence for the single class
            if single_class == 1:  # All malicious
                # For malicious samples, higher prediction = better
                avg_confidence = np.mean(predictions)
                detection_rate = np.mean(predictions > 0.5)
            else:  # All benign
                # For benign samples, lower prediction = better
                avg_confidence = np.mean(1 - predictions)
                detection_rate = np.mean(predictions < 0.5)

            print(f"Single-class test set: {len(y_test)} {class_name} samples")
            print(f"Average confidence: {avg_confidence:.4f}, Detection rate: {detection_rate:.4f}")
            print(f"TPR/FPR/AUROC/AUPRC set to N/A (not applicable for single-class)")

            tpr = float('nan')  # Use NaN to indicate not applicable
            fpr = float('nan')
            auroc = float('nan')
            auprc = float('nan')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
            'auroc': auroc,
            'auprc': auprc,
            'predictions': predictions,
            'threshold': getattr(model, 'optimal_threshold', 0.5) if use_optimal_threshold else 0.5
        }
    
    else:  # multiclass
        # Generate classification report
        if categories is None:
            categories = [f"Class_{i}" for i in range(len(np.unique(y_test)))]
        
        report = classification_report(y_test, y_pred_binary, target_names=categories, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_binary)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': predictions
        }
