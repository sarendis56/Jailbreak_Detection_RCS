#!/usr/bin/env python3
"""
Analyze correlation between principled layer selection scores and actual performance metrics.

This script compares the theoretical layer rankings from principled_layer_selection_results.csv
with the empirical performance from multi-run experiments (F1 and AUROC).

If multi-run results are not available, it will automatically fall back to single-run results
from files like balanced_kcd_qwen_results.csv.

Supports LLaVA, Qwen, and InternVL.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
import sys
import warnings

def load_principled_scores(model_type=None):
    """Load the principled layer selection results"""
    # Try different possible paths depending on where script is run from
    base_paths = [
        "results/",  # Run from project directory
        "HiddenDetect/results/"  # Run from parent directory
    ]

    # Model-specific filenames
    if model_type:
        # If specific model type provided, try multiple variations
        if model_type == "qwen":
            model_filenames = [
                "principled_layer_selection_results_qwen25vl_3b.csv",
                "principled_layer_selection_results_qwen25vl_7b.csv",
                "principled_layer_selection_results_qwen25vl.csv",
                "principled_layer_selection_results_qwen.csv"
            ]
        elif model_type == "llava":
            model_filenames = [
                "principled_layer_selection_results_llava_7b.csv",
                "principled_layer_selection_results_llava_13b.csv",
                "principled_layer_selection_results_llava.csv"
            ]
        elif model_type == "internvl":
            model_filenames = [
                "principled_layer_selection_results_internvl3_8b.csv",
                "principled_layer_selection_results_internvl.csv",
                f"principled_layer_selection_results_{model_type}.csv"
            ]
        else:
            model_filenames = [f"principled_layer_selection_results_{model_type}.csv"]
    else:
        # Try all possible model-specific files and fallback
        model_filenames = [
            "principled_layer_selection_results_qwen25vl_3b.csv",
            "principled_layer_selection_results_qwen25vl_7b.csv",
            "principled_layer_selection_results_qwen25vl.csv",
            "principled_layer_selection_results_llava_7b.csv",
            "principled_layer_selection_results_llava_13b.csv",
            "principled_layer_selection_results_llava.csv",
            "principled_layer_selection_results.csv"  # Fallback to old naming
        ]

    scores_path = None
    for base_path in base_paths:
        for filename in model_filenames:
            path = base_path + filename
            if os.path.exists(path):
                scores_path = path
                # Extract model type from filename for later use
                if "qwen" in filename.lower():
                    detected_model = "qwen"
                elif "llava" in filename.lower():
                    detected_model = "llava"
                elif "internvl" in filename.lower():
                    detected_model = "internvl"
                else:
                    detected_model = "unknown"
                break
        if scores_path:
            break

    if scores_path is None:
        print(f"Error: No principled layer selection results found!")
        print("Looked for files:")
        for base_path in base_paths:
            for filename in model_filenames:
                print(f"  - {base_path + filename}")
        return None, None

    df = pd.read_csv(scores_path)
    print(f"Loaded principled scores for {len(df)} layers from: {scores_path}")
    print(f"Detected model type: {detected_model}")
    return df, detected_model

def find_latest_multi_run_results(model_type="unknown"):
    """Find the latest multi-run results for each method, filtered by model type"""
    # Try different possible base directories depending on where script is run from
    possible_base_dirs = [
        "multi_run_results",  # Run from HiddenDetect directory
        "HiddenDetect/multi_run_results"  # Run from parent directory
    ]

    base_dir = None
    for dir_path in possible_base_dirs:
        if os.path.exists(dir_path):
            base_dir = dir_path
            break

    if base_dir is None:
        print(f"Error: multi_run_results directory not found in any of these locations:")
        for dir_path in possible_base_dirs:
            print(f"  - {dir_path}")
        return {}

    methods = ['mcd', 'kcd']
    latest_results = {}

    for method in methods:
        # Find all directories for this method, optionally filtered by model type
        if model_type != "unknown":
            # Look for model-specific results first
            pattern = f"{base_dir}/{method}_{model_type}_*runs_*"
            dirs = glob.glob(pattern)

            if not dirs:
                # If no model-specific results found, try generic pattern as fallback
                print(f"Warning: No {model_type}-specific results found for {method}, trying generic pattern...")
                pattern = f"{base_dir}/{method}_*runs_*"
                dirs = glob.glob(pattern)
        else:
            # If model type is unknown, use generic pattern
            pattern = f"{base_dir}/{method}_*runs_*"
            dirs = glob.glob(pattern)

        if dirs:
            # Get the latest directory by extracting and comparing timestamps
            def extract_timestamp(dir_path):
                # Extract timestamp from pattern like "mcd_qwen_20runs_20250814_181114"
                import re
                match = re.search(r'_(\d{8}_\d{6})$', dir_path)
                return match.group(1) if match else '00000000_000000'

            # If we have a specific model type, filter directories to match it
            if model_type != "unknown":
                model_filtered_dirs = []
                for dir_path in dirs:
                    # Extract model from directory name (e.g., "mcd_qwen_20runs_..." -> "qwen")
                    import re
                    match = re.search(rf'{method}_([^_]+)_\d+runs_', dir_path)
                    if match:
                        dir_model = match.group(1)
                        if dir_model == model_type:
                            model_filtered_dirs.append(dir_path)

                if model_filtered_dirs:
                    dirs = model_filtered_dirs
                else:
                    print(f"Warning: No {model_type}-specific directories found for {method}, using all available")

            # Sort by timestamp (newest first)
            latest_dir = sorted(dirs, key=extract_timestamp)[-1]
            latest_results[method] = latest_dir
            print(f"Found latest {method.upper()} results: {latest_dir}")
        else:
            print(f"Warning: No results found for {method}")

    return latest_results

def find_single_run_results(model_type="unknown"):
    """Find single run results files as fallback when multi-run results are not available"""
    # Try different possible paths depending on where script is run from
    base_paths = [
        "results/",  # Run from project directory
        "HiddenDetect/results/"  # Run from parent directory
    ]

    # Look for single run results files
    single_run_files = {}

    # Common patterns for single run results files
    patterns = [
        f"balanced_kcd_{model_type}_results.csv",
        f"balanced_mcd_{model_type}_results.csv",
        "balanced_kcd_qwen_results.csv",
        "balanced_kcd_llava_results.csv",
        "balanced_mcd_qwen_results.csv",
        "balanced_mcd_llava_results.csv",
        # Generic fallbacks
        "balanced_kcd_results.csv",
        "balanced_mcd_results.csv"
    ]

    for base_path in base_paths:
        for pattern in patterns:
            file_path = base_path + pattern
            if os.path.exists(file_path):
                # Extract method from filename
                if "kcd" in pattern:
                    method = "kcd"
                elif "mcd" in pattern:
                    method = "mcd"
                else:
                    continue

                if method not in single_run_files:
                    single_run_files[method] = file_path
                    print(f"Found single run {method.upper()} results: {file_path}")

    return single_run_files

def load_performance_data(results_dirs):
    """Load F1 and AUROC data from multi-run results"""
    all_data = {}

    for method, result_dir in results_dirs.items():
        print(f"\nLoading {method.upper()} data from {result_dir}")

        # Load classification metrics (F1)
        f1_path = f"{result_dir}/aggregated_{method}_classification.csv"
        auroc_path = f"{result_dir}/aggregated_{method}_performance.csv"

        # For MCD/KCD methods, use original logic
        method_data = {}

        # Load F1 data
        if os.path.exists(f1_path):
            f1_df = pd.read_csv(f1_path)
            # Filter for COMBINED dataset only
            f1_combined = f1_df[f1_df['Dataset'] == 'COMBINED'].copy()
            method_data['f1'] = f1_combined[['Layer', 'F1_Mean', 'F1_Std']].copy()
            print(f"  Loaded F1 data for {len(method_data['f1'])} layers")
        else:
            print(f"  Warning: {f1_path} not found")

        # Load AUROC data
        if os.path.exists(auroc_path):
            auroc_df = pd.read_csv(auroc_path)
            # Filter for COMBINED dataset only
            auroc_combined = auroc_df[auroc_df['Dataset'] == 'COMBINED'].copy()
            method_data['auroc'] = auroc_combined[['Layer', 'AUROC_Mean', 'AUROC_Std']].copy()
            print(f"  Loaded AUROC data for {len(method_data['auroc'])} layers")
        else:
            print(f"  Warning: {auroc_path} not found")

        all_data[method] = method_data

    return all_data

def load_single_run_performance_data(single_run_files):
    """Load F1 and AUROC data from single run results files"""
    all_data = {}

    for method, file_path in single_run_files.items():
        print(f"\nLoading {method.upper()} data from {file_path}")

        try:
            df = pd.read_csv(file_path)

            # Filter for COMBINED dataset only
            combined_data = df[df['Dataset'] == 'COMBINED'].copy()

            if combined_data.empty:
                print(f"  Warning: No COMBINED dataset found in {file_path}")
                continue

            method_data = {}

            # For single run data, we don't have standard deviations, so we'll set them to 0
            if 'F1' in combined_data.columns:
                f1_data = combined_data[['Layer', 'F1']].copy()
                f1_data['F1_Mean'] = f1_data['F1']
                f1_data['F1_Std'] = 0.0  # No std for single run
                f1_data = f1_data[['Layer', 'F1_Mean', 'F1_Std']]
                method_data['f1'] = f1_data
                print(f"  Loaded F1 data for {len(f1_data)} layers")

            if 'AUROC' in combined_data.columns:
                auroc_data = combined_data[['Layer', 'AUROC']].copy()
                auroc_data['AUROC_Mean'] = auroc_data['AUROC']
                auroc_data['AUROC_Std'] = 0.0  # No std for single run
                auroc_data = auroc_data[['Layer', 'AUROC_Mean', 'AUROC_Std']]
                method_data['auroc'] = auroc_data
                print(f"  Loaded AUROC data for {len(auroc_data)} layers")

            # For non-ML methods
            all_data[method] = method_data

        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            continue

    return all_data



def create_f1_correlation_plot(principled_scores, performance_data, model_type="unknown", output_dir=None):
    """Create F1 correlation plot"""

    # Determine output directory based on current working directory
    if output_dir is None:
        if os.path.exists("results"):
            output_dir = "results"  # Run from project directory
        else:
            output_dir = "HiddenDetect/results"  # Run from parent directory

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create single plot for F1
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 12))
    ax1_twin = ax1.twinx()

    # Bar chart for Separation Score (left y-axis)
    bars = ax1.bar(principled_scores['Layer'], principled_scores['Overall_Score'],
                   alpha=0.6, color='lightblue', label='Separation Score', width=0.6)
    ax1.set_xlabel('Layer', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Separation Score', color='blue', fontsize=24, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.set_ylim(0, 1.0)

    # Set x-axis limits based on aligned data
    max_layer = principled_scores['Layer'].max()
    ax1.set_xlim(-1, max_layer + 1)

    # Line plots for F1 scores (right y-axis)
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (method, data) in enumerate(performance_data.items()):
        if 'f1' in data and not data['f1'].empty:
            f1_data = data['f1'].sort_values('Layer')
            # Only show error bars if we have non-zero standard deviations
            yerr = f1_data['F1_Std'] if f1_data['F1_Std'].sum() > 0 else None
            capsize = 4 if yerr is not None else 0
            ax1_twin.errorbar(f1_data['Layer'], f1_data['F1_Mean'],
                            yerr=yerr,
                            color=colors[i % len(colors)], marker=markers[i % len(markers)],
                            label=f'{method.upper()} F1', linewidth=4, markersize=12,
                            capsize=capsize, capthick=3)

    ax1_twin.set_ylabel('F1 Score', color='red', fontsize=24, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='red', labelsize=20)
    ax1_twin.set_ylim(0, 1.0)

    # Return axes for optional baseline overlay and legend handling in caller
    return fig, ax1, ax1_twin, bars

    # Update title to include model type
    model_title = model_type.upper() if model_type != "unknown" else ""
    title = f'F1 Score vs Separation Score by Layer ({model_title})' if model_title else 'F1 Score vs Separation Score by Layer'
    ax1.set_title(title, fontsize=28, fontweight='bold', pad=30)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save as PDF with model-specific name
    if model_type != "unknown":
        output_path = f"{output_dir}/f1_correlation_plot_{model_type}.pdf"
    else:
        output_path = f"{output_dir}/f1_correlation_plot.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"F1 correlation plot saved to: {output_path}")

    plt.close()

def create_auroc_correlation_plot(principled_scores, performance_data, model_type="unknown", output_dir=None):
    """Create AUROC correlation plot"""

    # Determine output directory based on current working directory
    if output_dir is None:
        if os.path.exists("results"):
            output_dir = "results"  # Run from project directory
        else:
            output_dir = "HiddenDetect/results"  # Run from parent directory

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create single plot for AUROC
    fig, ax2 = plt.subplots(1, 1, figsize=(16, 12))
    ax2_twin = ax2.twinx()

    # Bar chart for Separation Score (left y-axis)
    bars = ax2.bar(principled_scores['Layer'], principled_scores['Overall_Score'],
                   alpha=0.6, color='lightblue', label='Separation Score', width=0.6)
    ax2.set_xlabel('Layer', fontsize=24, fontweight='bold')
    ax2.set_ylabel('Separation Score', color='blue', fontsize=24, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.set_ylim(0, 1.0)

    # Set x-axis limits based on aligned data
    max_layer = principled_scores['Layer'].max()
    ax2.set_xlim(-1, max_layer + 1)

    # Line plots for AUROC scores (right y-axis)
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (method, data) in enumerate(performance_data.items()):
        if 'auroc' in data and not data['auroc'].empty:
            auroc_data = data['auroc'].sort_values('Layer')
            # Only show error bars if we have non-zero standard deviations
            yerr = auroc_data['AUROC_Std'] if auroc_data['AUROC_Std'].sum() > 0 else None
            capsize = 4 if yerr is not None else 0
            ax2_twin.errorbar(auroc_data['Layer'], auroc_data['AUROC_Mean'],
                            yerr=yerr,
                            color=colors[i % len(colors)], marker=markers[i % len(markers)],
                            label=f'{method.upper()} AUROC', linewidth=4, markersize=12,
                            capsize=capsize, capthick=3)

    ax2_twin.set_ylabel('AUROC', color='red', fontsize=24, fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='red', labelsize=20)
    ax2_twin.set_ylim(0, 1.0)

    # Update title to include model type
    model_title = model_type.upper() if model_type != "unknown" else ""
    title = f'AUROC vs Separation Score by Layer ({model_title})' if model_title else 'AUROC vs Separation Score by Layer'
    ax2.set_title(title, fontsize=28, fontweight='bold', pad=30)
    ax2.grid(True, alpha=0.3)

    # Return axes for optional baseline overlay and legend handling in caller
    return fig, ax2, ax2_twin, bars

def load_fdv_baseline(model_type="unknown"):
    """Load FDV baseline CSV and return a DataFrame with normalized values by layer."""
    if model_type == "qwen":
        path = "results/qwen25vl_adaptive_safety_layers.csv"
    elif model_type == "llava":
        path = "results/llava_safety_layers.csv"
    else:
        # Try to infer either; prefer Qwen if both exist
        qwen_path = "results/qwen25vl_adaptive_safety_layers.csv"
        llava_path = "results/llava_safety_layers.csv"
        path = qwen_path if os.path.exists(qwen_path) else (llava_path if os.path.exists(llava_path) else None)

    if path is None or not os.path.exists(path):
        warnings.warn("FDV baseline CSV not found; skipping baseline overlay")
        return None

    import pandas as pd
    df = pd.read_csv(path)
    if not {'layer', 'fdv'}.issubset(df.columns):
        warnings.warn(f"FDV baseline CSV at {path} missing required columns; skipping baseline overlay")
        return None

    # Normalize FDV to [0,1] to share the left axis with Separation Score
    fdv_min = df['fdv'].min()
    fdv_max = df['fdv'].max()
    if fdv_max - fdv_min > 1e-12:
        df['fdv_norm'] = (df['fdv'] - fdv_min) / (fdv_max - fdv_min)
    else:
        df['fdv_norm'] = 0.0

    df.rename(columns={'layer': 'Layer'}, inplace=True)
    return df[['Layer', 'fdv_norm']]

def compute_correlations(principled_scores, performance_data):
    """Compute correlation coefficients between scores and performance metrics"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    correlations = {}

    for method, data in performance_data.items():
        print(f"\n{method.upper()} Method:")
        method_corr = {}

        # Merge with principled scores
        if 'f1' in data and not data['f1'].empty:
            merged_f1 = pd.merge(principled_scores, data['f1'], on='Layer', how='inner')
            if len(merged_f1) > 1:
                f1_corr = merged_f1['Overall_Score'].corr(merged_f1['F1_Mean'])
                method_corr['f1'] = f1_corr
                print(f"  F1 vs Separation Score correlation: {f1_corr:.4f}")

        if 'auroc' in data and not data['auroc'].empty:
            merged_auroc = pd.merge(principled_scores, data['auroc'], on='Layer', how='inner')
            if len(merged_auroc) > 1:
                auroc_corr = merged_auroc['Overall_Score'].corr(merged_auroc['AUROC_Mean'])
                method_corr['auroc'] = auroc_corr
                print(f"  AUROC vs Separation Score correlation: {auroc_corr:.4f}")

        correlations[method] = method_corr

    return correlations

def print_top_layers_comparison(principled_scores, performance_data, top_n=10):
    """Compare top layers from principled scores vs actual performance"""
    print("\n" + "="*80)
    print(f"TOP {top_n} LAYERS COMPARISON")
    print("="*80)

    # Top layers by principled score
    top_principled = principled_scores.nlargest(top_n, 'Overall_Score')['Layer'].tolist()
    print(f"Top {top_n} layers by Separation Score: {top_principled}")

    # Top layers by performance metrics
    for method, data in performance_data.items():
        print(f"\n{method.upper()} Method:")

        if 'f1' in data and not data['f1'].empty:
            top_f1 = data['f1'].nlargest(top_n, 'F1_Mean')['Layer'].tolist()
            print(f"  Top {top_n} layers by F1: {top_f1}")

            # Calculate overlap
            overlap_f1 = len(set(top_principled) & set(top_f1))
            print(f"  F1 overlap with principled: {overlap_f1}/{top_n} ({overlap_f1/top_n*100:.1f}%)")

        if 'auroc' in data and not data['auroc'].empty:
            top_auroc = data['auroc'].nlargest(top_n, 'AUROC_Mean')['Layer'].tolist()
            print(f"  Top {top_n} layers by AUROC: {top_auroc}")

            # Calculate overlap
            overlap_auroc = len(set(top_principled) & set(top_auroc))
            print(f"  AUROC overlap with principled: {overlap_auroc}/{top_n} ({overlap_auroc/top_n*100:.1f}%)")

def create_summary_table(correlations, performance_data, model_type="unknown", output_dir=None):
    """Create a summary table of the analysis results"""

    # Determine output directory based on current working directory
    if output_dir is None:
        if os.path.exists("results"):
            output_dir = "results"  # Run from project directory
        else:
            output_dir = "HiddenDetect/results"  # Run from parent directory

    # Create summary data
    summary_data = []

    for method, corr_data in correlations.items():
        row = {
            'Method': method.upper(),
            'F1_Correlation': corr_data.get('f1', 'N/A'),
            'AUROC_Correlation': corr_data.get('auroc', 'N/A'),
        }

        # Add performance statistics
        if 'f1' in performance_data[method]:
            f1_data = performance_data[method]['f1']
            row['F1_Mean_Avg'] = f1_data['F1_Mean'].mean()
            row['F1_Mean_Std'] = f1_data['F1_Mean'].std()

        if 'auroc' in performance_data[method]:
            auroc_data = performance_data[method]['auroc']
            row['AUROC_Mean_Avg'] = auroc_data['AUROC_Mean'].mean()
            row['AUROC_Mean_Std'] = auroc_data['AUROC_Mean'].std()

        summary_data.append(row)

    # Create DataFrame and save with model-specific name
    summary_df = pd.DataFrame(summary_data)
    if model_type != "unknown":
        summary_path = f"{output_dir}/layer_correlation_summary_{model_type}.csv"
    else:
        summary_path = f"{output_dir}/layer_correlation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSummary table saved to: {summary_path}")
    print("\nSUMMARY TABLE:")
    print(summary_df.to_string(index=False, float_format='%.4f'))

def main():
    print("="*80)
    print("LAYER PERFORMANCE CORRELATION ANALYSIS")
    print("="*80)
    print("Usage: python analyze_layer_performance_correlation.py [qwen|llava|internvl]")
    print("If no model type is specified, will auto-detect from available files.")
    print("="*80)

    # Parse command line arguments for model type
    model_type_arg = None
    if len(sys.argv) > 1:
        model_type_arg = sys.argv[1].lower()
        if model_type_arg not in ['qwen', 'llava', 'internvl']:
            print(f"Warning: Unknown model type '{model_type_arg}'. Will auto-detect from files.")
            model_type_arg = None

    # Load principled layer selection scores
    principled_scores, detected_model = load_principled_scores(model_type_arg)
    if principled_scores is None:
        return

    # Use command line argument if provided, otherwise use detected model, fallback to "unknown"
    model_type = model_type_arg if model_type_arg else (detected_model if detected_model else "unknown")

    # Find latest multi-run results, filtered by model type
    results_dirs = find_latest_multi_run_results(model_type)
    performance_data = {}

    if results_dirs:
        # Load performance data from multi-run results
        performance_data = load_performance_data(results_dirs)
    else:
        print("No multi-run results found. Trying single run results as fallback...")

        # Try to find single run results
        single_run_files = find_single_run_results(model_type)

        if single_run_files:
            print("Found single run results files. Using as fallback.")
            performance_data = load_single_run_performance_data(single_run_files)
        else:
            print("Error: No performance results found (neither multi-run nor single run)!")
            return

    if not performance_data:
        print("Error: No performance data could be loaded!")
        return

    # Load FDV baseline (optional)
    baseline_df = load_fdv_baseline(model_type)

    # Create F1 correlation plot and overlay baseline if available
    fig_f1, ax1, ax1_twin, bars_f1 = create_f1_correlation_plot(principled_scores, performance_data, model_type)

    # Overlay FDV baseline on left axis (normalized), aligned by layer
    if baseline_df is not None and not baseline_df.empty:
        import pandas as pd
        aligned = pd.merge(principled_scores[['Layer']], baseline_df, on='Layer', how='left').sort_values('Layer')
        ax1.plot(aligned['Layer'], aligned['fdv_norm'], color='black', linestyle='--', linewidth=3, label='FDV Baseline (norm)')

    # Build a single combined legend at bottom right
    handles_left, labels_left = ax1.get_legend_handles_labels()
    handles_right, labels_right = ax1_twin.get_legend_handles_labels()
    handles = handles_left + handles_right
    labels = labels_left + labels_right
    ax1.legend(handles, labels, loc='lower right', fontsize=20, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    # Save as PDF with model-specific name
    if model_type != "unknown":
        output_path = f"results/f1_correlation_plot_{model_type}.pdf"
    else:
        output_path = f"results/f1_correlation_plot.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"F1 correlation plot saved to: {output_path}")
    plt.close(fig_f1)

    # Create AUROC correlation plot and overlay baseline if available
    fig_auc, ax2, ax2_twin, bars_auc = create_auroc_correlation_plot(principled_scores, performance_data, model_type)

    if baseline_df is not None and not baseline_df.empty:
        import pandas as pd
        aligned = pd.merge(principled_scores[['Layer']], baseline_df, on='Layer', how='left').sort_values('Layer')
        ax2.plot(aligned['Layer'], aligned['fdv_norm'], color='black', linestyle='--', linewidth=3, label='FDV Baseline (norm)')

    handles_left, labels_left = ax2.get_legend_handles_labels()
    handles_right, labels_right = ax2_twin.get_legend_handles_labels()
    handles = handles_left + handles_right
    labels = labels_left + labels_right
    ax2.legend(handles, labels, loc='lower right', fontsize=20, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if model_type != "unknown":
        output_path = f"results/auroc_correlation_plot_{model_type}.pdf"
    else:
        output_path = f"results/auroc_correlation_plot.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"AUROC correlation plot saved to: {output_path}")
    plt.close(fig_auc)

    # Compute correlations
    correlations = compute_correlations(principled_scores, performance_data)

    # Compare top layers
    print_top_layers_comparison(principled_scores, performance_data, top_n=10)

    # Create summary table
    create_summary_table(correlations, performance_data, model_type)

    # Determine output directory for final messages
    if os.path.exists("results"):
        results_dir = "results"  # Run from project directory
    else:
        results_dir = "HiddenDetect/results"  # Run from parent directory

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Model type: {model_type.upper()}")
    print("Check the generated plots:")

    # Show model-specific filenames
    if model_type != "unknown":
        print(f"  - F1 correlation: {results_dir}/f1_correlation_plot_{model_type}.pdf")
        print(f"  - AUROC correlation: {results_dir}/auroc_correlation_plot_{model_type}.pdf")
        print(f"Check the summary table: {results_dir}/layer_correlation_summary_{model_type}.csv")
    else:
        print(f"  - F1 correlation: {results_dir}/f1_correlation_plot.pdf")
        print(f"  - AUROC correlation: {results_dir}/auroc_correlation_plot.pdf")
        print(f"Check the summary table: {results_dir}/layer_correlation_summary.csv")

if __name__ == "__main__":
    main()
