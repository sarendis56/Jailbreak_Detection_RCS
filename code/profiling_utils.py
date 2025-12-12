"""
Profiling utilities for measuring runtime and memory overhead
"""
import time
import torch
import psutil
import numpy as np
from contextlib import contextmanager
from collections import defaultdict


class PerformanceProfiler:
    """Profiler for measuring runtime and memory overhead"""
    
    def __init__(self):
        self.measurements = defaultdict(list)
        self.current_measurements = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        memory_info = {}
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            memory_info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        else:
            memory_info['gpu_allocated_mb'] = 0
            memory_info['gpu_reserved_mb'] = 0
            memory_info['gpu_max_allocated_mb'] = 0
        
        # CPU memory
        process = psutil.Process()
        memory_info['cpu_mb'] = process.memory_info().rss / 1024**2
        
        return memory_info
    
    @contextmanager
    def profile(self, operation_name, num_samples=None):
        """Context manager for profiling an operation"""
        # Reset peak memory stats before operation
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Get initial memory
        mem_before = self._get_memory_usage()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            mem_after = self._get_memory_usage()
            
            # Calculate metrics
            elapsed_time = end_time - start_time
            gpu_memory_delta = mem_after['gpu_allocated_mb'] - mem_before['gpu_allocated_mb']
            gpu_peak_memory = mem_after['gpu_max_allocated_mb']
            cpu_memory_delta = mem_after['cpu_mb'] - mem_before['cpu_mb']
            
            # Store measurement
            measurement = {
                'operation': operation_name,
                'time_seconds': elapsed_time,
                'time_per_sample_seconds': elapsed_time / num_samples if num_samples else None,
                'gpu_memory_delta_mb': gpu_memory_delta,
                'gpu_peak_memory_mb': gpu_peak_memory,
                'cpu_memory_delta_mb': cpu_memory_delta,
                'num_samples': num_samples
            }
            
            self.measurements[operation_name].append(measurement)
            self.current_measurements[operation_name] = measurement
    
    def get_summary(self, operation_name):
        """Get summary statistics for an operation"""
        if operation_name not in self.measurements:
            return None
        
        measurements = self.measurements[operation_name]
        if not measurements:
            return None
        
        times = [m['time_seconds'] for m in measurements]
        gpu_deltas = [m['gpu_memory_delta_mb'] for m in measurements]
        cpu_deltas = [m['cpu_memory_delta_mb'] for m in measurements]
        gpu_peaks = [m.get('gpu_peak_memory_mb', 0) for m in measurements]
        
        return {
            'operation': operation_name,
            'total_time_seconds': sum(times),
            'mean_time_seconds': np.mean(times),
            'std_time_seconds': np.std(times),
            'min_time_seconds': np.min(times),
            'max_time_seconds': np.max(times),
            'total_gpu_memory_mb': sum(gpu_deltas),
            'mean_gpu_memory_mb': np.mean(gpu_deltas),
            'peak_gpu_memory_mb': max(gpu_peaks) if gpu_peaks else 0,
            'total_cpu_memory_mb': sum(cpu_deltas),
            'mean_cpu_memory_mb': np.mean(cpu_deltas),
            'num_operations': len(measurements)
        }
    
    def reset(self):
        """Reset all measurements"""
        self.measurements.clear()
        self.current_measurements.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def save_profiling_results(profiler, output_path, layer_idx, dataset_name, method_name, feature_extraction_summary=None, baseline_summary=None):
    """Save profiling results to CSV file matching benchmark format
    
    Args:
        profiler: PerformanceProfiler instance for projection and detection
        output_path: Path to CSV output file
        layer_idx: Layer index
        dataset_name: Dataset name
        method_name: Method name (KCD or MCD)
        feature_extraction_summary: Optional summary from feature extraction profiler
    """
    import csv
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(output_path)
    
    with open(output_path, 'a', newline='') as csvfile:
        # Match benchmark CSV format: Method, Component, Time_Total_s, Time_Mean_s, Time_Std_s, 
        # Peak_GPU_GB, Peak_CPU_GB, Throughput_Samples_per_sec, Time_per_Sample_ms
        fieldnames = [
            'Method', 'Component', 'Layer', 'Time_Total_s', 'Time_Mean_s', 'Time_Std_s',
            'Peak_GPU_GB', 'Peak_CPU_GB', 'Throughput_Samples_per_sec', 'Time_per_Sample_ms'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Get summaries for each operation
        feature_extraction = feature_extraction_summary or profiler.get_summary('feature_extraction')
        projection = profiler.get_summary('projection')
        detection = profiler.get_summary('detection')
        
        # Write separate rows for each component (matching benchmark format)
        rows_written = []
        
        # 1. MLP Projection component
        if projection:
            # Get num_samples from measurements if available
            proj_measurements = profiler.measurements.get('projection', [])
            num_samples = 0
            if proj_measurements:
                num_samples = sum(m.get('num_samples', 0) or 0 for m in proj_measurements)
            if num_samples == 0 and projection.get('mean_time_seconds', 0) > 0:
                # Fallback: estimate from time
                num_samples = int(projection['total_time_seconds'] / projection['mean_time_seconds'])
            throughput = num_samples / projection['total_time_seconds'] if projection['total_time_seconds'] > 0 and num_samples > 0 else 0
            time_per_sample_ms = projection['mean_time_seconds'] * 1000 if projection['mean_time_seconds'] else 0
            peak_gpu_gb = projection.get('peak_gpu_memory_mb', projection.get('mean_gpu_memory_mb', 0)) / 1024  # Convert MB to GB
            peak_cpu_gb = projection.get('mean_cpu_memory_mb', 0) / 1024  # Convert MB to GB
            
            row = {
                'Method': method_name,
                'Component': 'mlp_projection',
                'Layer': layer_idx,
                'Time_Total_s': f"{projection['total_time_seconds']:.6f}",
                'Time_Mean_s': f"{projection['mean_time_seconds']:.6f}",
                'Time_Std_s': f"{projection['std_time_seconds']:.6f}",
                'Peak_GPU_GB': f"{peak_gpu_gb:.6f}" if peak_gpu_gb > 0 else "",
                'Peak_CPU_GB': f"{peak_cpu_gb:.6f}" if peak_cpu_gb > 0 else "",
                'Throughput_Samples_per_sec': f"{throughput:.6f}" if throughput > 0 else "",
                'Time_per_Sample_ms': f"{time_per_sample_ms:.6f}" if time_per_sample_ms > 0 else ""
            }
            writer.writerow(row)
            rows_written.append(row)
        
        # 2. Detection component (KNN for KCD, Mahal for MCD)
        if detection:
            # Get num_samples from measurements if available
            det_measurements = profiler.measurements.get('detection', [])
            num_samples = 0
            if det_measurements:
                num_samples = sum(m.get('num_samples', 0) or 0 for m in det_measurements)
            if num_samples == 0 and detection.get('mean_time_seconds', 0) > 0:
                # Fallback: estimate from time
                num_samples = int(detection['total_time_seconds'] / detection['mean_time_seconds'])
            throughput = num_samples / detection['total_time_seconds'] if detection['total_time_seconds'] > 0 and num_samples > 0 else 0
            time_per_sample_ms = detection['mean_time_seconds'] * 1000 if detection['mean_time_seconds'] else 0
            peak_gpu_gb = detection.get('peak_gpu_memory_mb', detection.get('mean_gpu_memory_mb', 0)) / 1024  # Convert MB to GB
            peak_cpu_gb = detection.get('mean_cpu_memory_mb', 0) / 1024  # Convert MB to GB
            
            component_name = 'knn_distance' if method_name == 'KCD' else 'mahal_distance'
            row = {
                'Method': method_name,
                'Component': component_name,
                'Layer': layer_idx,
                'Time_Total_s': f"{detection['total_time_seconds']:.6f}",
                'Time_Mean_s': f"{detection['mean_time_seconds']:.6f}",
                'Time_Std_s': f"{detection['std_time_seconds']:.6f}",
                'Peak_GPU_GB': f"{peak_gpu_gb:.6f}" if peak_gpu_gb > 0 else "",
                'Peak_CPU_GB': f"{peak_cpu_gb:.6f}" if peak_cpu_gb > 0 else "",
                'Throughput_Samples_per_sec': f"{throughput:.6f}" if throughput > 0 else "",
                'Time_per_Sample_ms': f"{time_per_sample_ms:.6f}" if time_per_sample_ms > 0 else ""
            }
            writer.writerow(row)
            rows_written.append(row)
        
        # 3. Baseline forward pass (if available, for fair comparison)
        if baseline_summary:
            num_samples = baseline_summary.get('num_operations', 0)
            throughput = num_samples / baseline_summary['total_time_seconds'] if baseline_summary['total_time_seconds'] > 0 and num_samples > 0 else 0
            time_per_sample_ms = baseline_summary['mean_time_seconds'] * 1000 if baseline_summary['mean_time_seconds'] else 0
            peak_gpu_gb = baseline_summary.get('peak_gpu_memory_mb', 0) / 1024  # Convert MB to GB
            peak_cpu_gb = baseline_summary.get('mean_cpu_memory_mb', 0) / 1024  # Convert MB to GB
            
            row = {
                'Method': method_name,
                'Component': 'baseline_forward_pass',
                'Layer': layer_idx,
                'Time_Total_s': f"{baseline_summary['total_time_seconds']:.6f}",
                'Time_Mean_s': f"{baseline_summary['mean_time_seconds']:.6f}",
                'Time_Std_s': f"{baseline_summary['std_time_seconds']:.6f}",
                'Peak_GPU_GB': f"{peak_gpu_gb:.6f}" if peak_gpu_gb > 0 else "",
                'Peak_CPU_GB': f"{peak_cpu_gb:.6f}" if peak_cpu_gb > 0 else "",
                'Throughput_Samples_per_sec': f"{throughput:.6f}" if throughput > 0 else "",
                'Time_per_Sample_ms': f"{time_per_sample_ms:.6f}" if time_per_sample_ms > 0 else ""
            }
            writer.writerow(row)
            rows_written.append(row)
        
        # 4. Feature extraction (if available, for reference)
        if feature_extraction:
            num_samples = feature_extraction.get('num_operations', 0)
            throughput = num_samples / feature_extraction['total_time_seconds'] if feature_extraction['total_time_seconds'] > 0 and num_samples > 0 else 0
            time_per_sample_ms = feature_extraction['mean_time_seconds'] * 1000 if feature_extraction['mean_time_seconds'] else 0
            peak_gpu_gb = feature_extraction.get('peak_gpu_memory_mb', feature_extraction.get('mean_gpu_memory_mb', 0)) / 1024  # Convert MB to GB
            peak_cpu_gb = feature_extraction.get('mean_cpu_memory_mb', 0) / 1024  # Convert MB to GB
            
            row = {
                'Method': method_name,
                'Component': 'feature_extraction',
                'Layer': layer_idx,
                'Time_Total_s': f"{feature_extraction['total_time_seconds']:.6f}",
                'Time_Mean_s': f"{feature_extraction['mean_time_seconds']:.6f}",
                'Time_Std_s': f"{feature_extraction['std_time_seconds']:.6f}",
                'Peak_GPU_GB': f"{peak_gpu_gb:.6f}" if peak_gpu_gb > 0 else "",
                'Peak_CPU_GB': f"{peak_cpu_gb:.6f}" if peak_cpu_gb > 0 else "",
                'Throughput_Samples_per_sec': f"{throughput:.6f}" if throughput > 0 else "",
                'Time_per_Sample_ms': f"{time_per_sample_ms:.6f}" if time_per_sample_ms > 0 else ""
            }
            writer.writerow(row)
            rows_written.append(row)
    
    return rows_written
