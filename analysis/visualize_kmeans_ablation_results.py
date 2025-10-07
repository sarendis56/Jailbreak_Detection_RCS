import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

def detect_model_from_args():
    """Detect model type from command line arguments"""
    # Look for --model or -m argument
    for i, arg in enumerate(sys.argv):
        if arg in ['--model', '-m'] and i + 1 < len(sys.argv):
            model_type = sys.argv[i + 1].lower()
            if model_type in ['qwen', 'llava', 'internvl']:
                return model_type
        # Also check for direct model arguments (backward compatibility)
        elif arg.lower() in ['qwen', 'llava', 'internvl'] and i > 0:
            return arg.lower()
    return 'llava'  # Default

# Detect focus layer from args (default 16)
def detect_focus_layer_from_args(default_layer=16):
    focus_layer = default_layer
    for i, arg in enumerate(sys.argv):
        if arg in ['--layer', '-l'] and i + 1 < len(sys.argv):
            try:
                focus_layer = int(sys.argv[i + 1])
            except ValueError:
                print(f"Warning: Invalid layer value '{sys.argv[i + 1]}'. Falling back to {default_layer}.")
            break
    return focus_layer

# Determine which model we're using
REQUESTED_MODEL = detect_model_from_args()
FOCUS_LAYER = detect_focus_layer_from_args(16)
print(f"Processing visualization for {REQUESTED_MODEL.upper()} model (focus layer: {FOCUS_LAYER})")

# Path to the CSV file based on model type
if REQUESTED_MODEL == 'internvl':
    csv_path = os.path.join(os.path.dirname(__file__), f'../results/kmeans_ablation_{REQUESTED_MODEL}_results.csv')
else:
    csv_path = os.path.join(os.path.dirname(__file__), '../results/kmeans_ablation_results.csv')

# Load data
if not os.path.exists(csv_path):
    print(f"Error: Results file not found: {csv_path}")
    print(f"Available model types: llava, qwen, internvl")
    print(f"Usage: python visualize_kmeans_ablation_results.py [--model MODEL_TYPE]")
    sys.exit(1)

with open(csv_path, 'r') as f:
    header = f.readline().strip().split(',')

df = pd.read_csv(csv_path, skiprows=1, names=header)
df['Layer'] = pd.to_numeric(df['Layer'], errors='coerce')

# Filter for COMBINED dataset and determine appropriate layer range
available_layers = sorted(df[df['Dataset'] == 'COMBINED']['Layer'].unique())
print(f"Available layers for {REQUESTED_MODEL}: {available_layers}")

# Use different layer ranges based on model type
if REQUESTED_MODEL == 'internvl':
    # InternVL typically has layers 0-31, use even layers 12-24
    layers = list(range(12, 25, 2))
elif REQUESTED_MODEL == 'qwen':
    # Qwen might have different layer range, use available layers in 12-24 range
    layers = [l for l in available_layers if 12 <= l <= 24 and l % 2 == 0]
else:  # llava
    # LLaVA typically has layers 0-31, use even layers 12-24
    layers = list(range(12, 25, 2))

# Filter to only include layers that exist in the data
layers = [l for l in layers if l in available_layers]
print(f"Using layers: {layers}")

df = df[(df['Dataset'] == 'COMBINED') & (df['Layer'].isin(layers))]

for col in ['F1_Mean', 'F1_Std', 'Accuracy_Mean', 'Accuracy_Std', 'AUROC_Mean', 'AUROC_Std']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Main Figure (all layers, no legend) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
metrics = ['AUROC_Mean', 'Accuracy_Mean']
titles = ['AUROC vs Layer', 'Accuracy vs Layer']

for i, metric in enumerate(metrics):
    for approach in df['Approach'].unique():
        sub = df[df['Approach'] == approach]
        axes[i].plot(sub['Layer'], sub[metric], marker='o', label=approach)
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Layer')
    axes[i].set_ylabel(metric.replace('_', ' '))
    axes[i].set_xticks(layers)
    # axes[i].legend()  # Remove legend as requested
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.suptitle(f'KMeans Ablation: AUROC and Accuracy by Layer and Approach ({REQUESTED_MODEL.upper()})', fontsize=16, y=1.05)
plt.subplots_adjust(top=0.85)

# Create figures directory if it doesn't exist
figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(figures_dir, exist_ok=True)

out_path = os.path.join(figures_dir, f'kmeans_ablation_auroc_accuracy_{REQUESTED_MODEL}.png')
plt.savefig(out_path, bbox_inches='tight')
print(f"Figure saved to {out_path}")

# --- New Figure: Focus on selected layer with Error Bars ---
# Choose requested layer; if missing, pick nearest available
layer_focus = df[df['Layer'] == FOCUS_LAYER].copy()
if layer_focus.empty:
    if layers:
        nearest = min(layers, key=lambda x: abs(x - FOCUS_LAYER))
        print(f"Warning: Layer {FOCUS_LAYER} not found in data. Using nearest available layer {nearest}.")
        layer_focus = df[df['Layer'] == nearest].copy()
        FOCUS_LAYER = nearest
    else:
        print("Error: No layers available to visualize.")
        sys.exit(1)

# Shorten approach labels for x-axis and prepare for sorting
short_labels = []
order_keys = []
for approach in layer_focus['Approach']:
    if approach.startswith('kmeans_k'):
        parts = approach.replace('kmeans_k', '').split('_')
        if len(parts) == 2:
            short_labels.append(f"{parts[0]}/{parts[1]}")
            # Use tuple for sorting: (1, int, int)
            order_keys.append((1, int(parts[0]), int(parts[1])))
        else:
            short_labels.append(approach)
            order_keys.append((2, 0, 0))
    elif approach == 'dataset_based':
        short_labels.append('dataset')
        order_keys.append((0, 0, 0))
    else:
        short_labels.append(approach)
        order_keys.append((2, 0, 0))

# Add short_labels and order_keys to DataFrame for sorting
layer_focus['short_label'] = short_labels
layer_focus['order_key'] = order_keys
layer_focus = layer_focus.sort_values('order_key').reset_index(drop=True)

# Print the data used for visualization
print(f"\nData used for Layer {FOCUS_LAYER} visualization:")
print(layer_focus[['short_label', 'F1_Mean', 'F1_Std', 'Accuracy_Mean', 'Accuracy_Std', 'AUROC_Mean', 'AUROC_Std']])

fig2, ax2 = plt.subplots(figsize=(12, 6))  # Further increased width for 3 bars
bar_width = 0.25
index = range(len(layer_focus))

# Set font sizes
title_fontsize = 20
label_fontsize = 12
tick_fontsize = 10
legend_fontsize = 12

ax2.bar([i - bar_width for i in index], layer_focus['F1_Mean'],
        yerr=layer_focus['F1_Std'], width=bar_width, label='F1', capsize=5, color='#1f77b4')
ax2.bar(index, layer_focus['Accuracy_Mean'],
        yerr=layer_focus['Accuracy_Std'], width=bar_width, label='Accuracy', capsize=5, color='#ff7f0e')
ax2.bar([i + bar_width for i in index], layer_focus['AUROC_Mean'],
        yerr=layer_focus['AUROC_Std'], width=bar_width, label='AUROC', capsize=5, color='#2ca02c')

ax2.set_xticks(index)
ax2.set_xticklabels(layer_focus['short_label'], rotation=30, ha='right', fontsize=tick_fontsize)
ax2.set_ylabel('Score', fontsize=label_fontsize)
ax2.set_title(f'Layer {FOCUS_LAYER}: F1, Accuracy, and AUROC by Clustering Strategy (with Std) - {REQUESTED_MODEL.upper()}', fontsize=title_fontsize)
ax2.legend(fontsize=legend_fontsize)
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
ax2.set_ylim([0.8, 1.0])
ax2.tick_params(axis='y', labelsize=tick_fontsize)
ax2.tick_params(axis='x', labelsize=tick_fontsize)

plt.tight_layout()
layer_out_path_png = os.path.join(figures_dir, f'layer{FOCUS_LAYER}_f1_accuracy_auroc_{REQUESTED_MODEL}.png')
layer_out_path_pdf = os.path.join(figures_dir, f'layer{FOCUS_LAYER}_f1_accuracy_auroc_{REQUESTED_MODEL}.pdf')
plt.savefig(layer_out_path_png, bbox_inches='tight')
plt.savefig(layer_out_path_pdf, bbox_inches='tight')
print(f"Layer {FOCUS_LAYER} figure saved to {layer_out_path_png} and {layer_out_path_pdf}")
