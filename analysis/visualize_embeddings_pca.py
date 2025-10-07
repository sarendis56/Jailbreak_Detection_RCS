#!/usr/bin/env python3
"""
PCA Visualization of Llava and Flava Embeddings for Dataset Separability Analysis

This script creates PCA visualizations to show the separability of different datasets
using embeddings from Llava (layer 16) and Flava models. It starts with MM-Vet and 
JailbreakV (Figstep subset) and then adds more datasets to observe how separability changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import warnings

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from load_datasets import *
from feature_extractor import HiddenStateExtractor
from balanced_flava import FlavaFeatureExtractor
import random

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EmbeddingVisualizer:
    """Class to handle PCA visualization of embeddings"""
    
    def __init__(self, llava_model_path="model/llava-v1.6-vicuna-7b/", flava_model_path="model/flava"):
        self.llava_model_path = llava_model_path
        self.flava_model_path = flava_model_path
        self.llava_extractor = None
        self.flava_extractor = None
        self.random_seed = 42

        # Set random seeds for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
    def _load_extractors(self):
        """Load the feature extractors"""
        if self.llava_extractor is None:
            print("Initializing Llava extractor...")
            self.llava_extractor = HiddenStateExtractor(self.llava_model_path)

        if self.flava_extractor is None:
            print("Initializing Flava extractor...")
            self.flava_extractor = FlavaFeatureExtractor(self.flava_model_path)
    
    def load_datasets_for_visualization(self):
        """Load datasets for visualization"""
        print("Loading datasets for visualization...")
        datasets = {}
        
        # Load MM-Vet (benign)
        try:
            mmvet_samples = load_mm_vet()
            if mmvet_samples:
                mmvet_subset = mmvet_samples[:300]
                datasets["MM-Vet"] = {"samples": mmvet_subset, "label": 0, "color": "blue"}
                print(f"Loaded {len(mmvet_subset)} MM-Vet samples")
        except Exception as e:
            print(f"Could not load MM-Vet: {e}")
        
        # Load JailbreakV Figstep (malicious)
        try:
            jbv_figstep_samples = load_JailBreakV_figstep(max_samples=300)
            if jbv_figstep_samples:
                datasets["FigStep"] = {"samples": jbv_figstep_samples, "label": 1, "color": "red"}
                print(f"Loaded {len(jbv_figstep_samples)} JailbreakV Figstep samples")
        except Exception as e:
            print(f"Could not load JailbreakV Figstep: {e}")
            
        # Load additional datasets for extended analysis
        try:
            vqav2_samples = load_vqav2(max_samples=300)
            if vqav2_samples:
                datasets["VQAv2"] = {"samples": vqav2_samples, "label": 0, "color": "lightblue"}
                print(f"Loaded {len(vqav2_samples)} VQAv2 samples")
        except Exception as e:
            print(f"Could not load VQAv2: {e}")
            
        try:
            advbench_samples = load_advbench(max_samples=300)
            if advbench_samples:
                datasets["AdvBench"] = {"samples": advbench_samples, "label": 1, "color": "orange"}
                print(f"Loaded {len(advbench_samples)} AdvBench samples")
        except Exception as e:
            print(f"Could not load AdvBench: {e}")
            
        try:
            alpaca_samples = load_alpaca(max_samples=300)
            if alpaca_samples:
                datasets["Alpaca"] = {"samples": alpaca_samples, "label": 0, "color": "green"}
                print(f"Loaded {len(alpaca_samples)} Alpaca samples")
        except Exception as e:
            print(f"Could not load Alpaca: {e}")

        # Load OpenAssistant (benign)
        try:
            openassistant_samples = load_openassistant(max_samples=300)
            if openassistant_samples:
                datasets["OpenAssistant"] = {"samples": openassistant_samples, "label": 0, "color": "lightgreen"}
                print(f"Loaded {len(openassistant_samples)} OpenAssistant samples")
        except Exception as e:
            print(f"Could not load OpenAssistant: {e}")

        # Load JailbreakV llm_transfer_attack + query_related (malicious)
        try:
            jbv_llm_samples = load_JailBreakV_custom(
                attack_types=["llm_transfer_attack", "query_related"],
                max_samples=300
            )
            if jbv_llm_samples:
                datasets["JailbreakV-LLM"] = {"samples": jbv_llm_samples, "label": 1, "color": "darkred"}
                print(f"Loaded {len(jbv_llm_samples)} JailbreakV LLM+Query samples")
        except Exception as e:
            print(f"Could not load JailbreakV LLM+Query: {e}")

        # Load DAN variants (malicious)
        try:
            dan_samples = load_dan_prompts(max_samples=150)
            if dan_samples:
                datasets["DAN"] = {"samples": dan_samples, "label": 1, "color": "purple"}
                print(f"Loaded {len(dan_samples)} DAN samples")
        except Exception as e:
            print(f"Could not load DAN: {e}")

        # Load XSTest safe (benign)
        try:
            xstest_samples = load_XSTest()
            if xstest_samples:
                xstest_safe = [s for s in xstest_samples if s.get('toxicity', 0) == 0][:150]
                if xstest_safe:
                    datasets["XSTest-Safe"] = {"samples": xstest_safe, "label": 0, "color": "cyan"}
                    print(f"Loaded {len(xstest_safe)} XSTest safe samples")
        except Exception as e:
            print(f"Could not load XSTest safe: {e}")

        # Load XSTest unsafe (malicious)
        try:
            if 'xstest_samples' in locals():
                xstest_unsafe = [s for s in xstest_samples if s.get('toxicity', 0) == 1][:150]
                if xstest_unsafe:
                    datasets["XSTest-Unsafe"] = {"samples": xstest_unsafe, "label": 1, "color": "magenta"}
                    print(f"Loaded {len(xstest_unsafe)} XSTest unsafe samples")
        except Exception as e:
            print(f"Could not load XSTest unsafe: {e}")

        try:
            mm_safety_samples = load_mm_safety_bench_all(max_samples=300)
            if mm_safety_samples:
                datasets["MM-SafetyBench"] = {"samples": mm_safety_samples, "label": 1, "color": "brown"}
                print(f"Loaded {len(mm_safety_samples)} MM-SafetyBench samples")
        except Exception as e:
            print(f"Could not load MM-SafetyBench: {e}")

        return datasets
    
    def extract_embeddings(self, datasets, model_type="llava"):
        """Extract embeddings for all datasets"""
        self._load_extractors()
        
        embeddings_dict = {}
        
        for dataset_name, dataset_info in datasets.items():
            samples = dataset_info["samples"]
            print(f"\nExtracting {model_type} embeddings for {dataset_name} ({len(samples)} samples)...")
            
            try:
                if model_type == "llava":
                    # Extract layer 16 embeddings from Llava
                    hidden_states, labels, _ = self.llava_extractor.extract_hidden_states(
                        samples, f"{dataset_name}_viz", layer_start=16, layer_end=16,
                        use_cache=True, batch_size=25, memory_cleanup_freq=5,
                        experiment_name="pca_visualization"
                    )
                    embeddings = np.array(hidden_states[16])  # Get layer 16 embeddings and convert to numpy array

                elif model_type == "flava":
                    # Extract Flava embeddings
                    embeddings, labels = self.flava_extractor.extract_features(samples, batch_size=16)
                    embeddings = np.array(embeddings)  # Ensure it's a numpy array

                embeddings_dict[dataset_name] = {
                    "embeddings": embeddings,
                    "labels": labels,
                    "color": dataset_info["color"],
                    "dataset_label": dataset_info["label"]
                }
                print(f"Extracted embeddings shape: {embeddings.shape}")
                
            except Exception as e:
                print(f"Error extracting embeddings for {dataset_name}: {e}")
                continue
                
        return embeddings_dict
    
    def create_pca_visualization(self, embeddings_dict, model_type, dataset_combinations, output_dir="visualizations"):
        """Create PCA visualizations for different dataset combinations"""
        os.makedirs(output_dir, exist_ok=True)
        
        for combo_name, dataset_names in dataset_combinations.items():
            print(f"\nCreating PCA visualization for {combo_name} using {model_type}...")
            
            # Collect embeddings and labels for this combination
            all_embeddings = []
            all_labels = []
            all_colors = []
            all_dataset_names = []
            
            for dataset_name in dataset_names:
                if dataset_name in embeddings_dict:
                    data = embeddings_dict[dataset_name]
                    all_embeddings.append(data["embeddings"])
                    all_labels.extend([dataset_name] * len(data["embeddings"]))
                    all_colors.extend([data["color"]] * len(data["embeddings"]))
                    all_dataset_names.extend([dataset_name] * len(data["embeddings"]))
            
            if len(all_embeddings) == 0:
                print(f"No embeddings found for combination {combo_name}")
                continue
                
            # Combine all embeddings
            X = np.vstack(all_embeddings)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=2, random_state=self.random_seed)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create the plot with larger figure size
            plt.figure(figsize=(16, 12))

            # Set larger font sizes
            plt.rcParams.update({'font.size': 36})

            # Plot each dataset with different colors
            unique_datasets = list(set(all_dataset_names))
            for dataset_name in unique_datasets:
                mask = np.array(all_dataset_names) == dataset_name
                color = embeddings_dict[dataset_name]["color"]
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=color, label=dataset_name, alpha=0.7, s=80)

            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=42)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=42)
            plt.title(f'PCA Visualization: {combo_name}\n{model_type.upper()} Embeddings (Layer 16)' if model_type == 'llava' else f'PCA Visualization: {combo_name}\n{model_type.upper()} Embeddings', fontsize=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=33, frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)

            # Add explained variance info with larger font
            total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
            plt.figtext(0.02, 0.02, f'Total variance explained: {total_variance:.1%}', fontsize=30)
            
            plt.tight_layout()

            # Save the plot in both PDF and PNG formats
            base_filename = f"{model_type}_{combo_name.lower().replace(' ', '_').replace('&', 'and')}_pca"

            # Save PDF version
            pdf_filepath = os.path.join(output_dir, f"{base_filename}.pdf")
            plt.savefig(pdf_filepath, format='pdf', dpi=300, bbox_inches='tight')
            print(f"Saved PCA plot (PDF): {pdf_filepath}")

            # Save PNG version
            png_filepath = os.path.join(output_dir, f"{base_filename}.png")
            plt.savefig(png_filepath, format='png', dpi=300, bbox_inches='tight')
            print(f"Saved PCA plot (PNG): {png_filepath}")

            plt.close()

    def create_combined_pca_visualization(self, llava_embeddings_dict, flava_embeddings_dict, dataset_combinations, output_dir="visualizations"):
        """Create a combined 2x4 subplot figure with all models and cases"""
        os.makedirs(output_dir, exist_ok=True)

        # Create a unified color mapping for all datasets
        all_datasets = set()
        for dataset_names in dataset_combinations.values():
            all_datasets.update(dataset_names)

        # Create a consistent color mapping
        unified_colors = {
            "MM-Vet": "blue",
            "FigStep": "red",
            "VQAv2": "lightblue",
            "AdvBench": "orange",
            "Alpaca": "green",
            "OpenAssistant": "lightgreen",
            "JailbreakV-LLM": "darkred",
            "DAN": "purple",
            "XSTest-Safe": "cyan",
            "XSTest-Unsafe": "magenta",
            "MM-SafetyBench": "brown"
        }

        # Create a large figure with 2 rows (models) and 4 columns (cases)
        # Extra width to accommodate legends that extend outside
        fig, axes = plt.subplots(2, 4, figsize=(32, 14))
        plt.rcParams.update({'font.size': 36})

        model_data = [
            ("Llava", llava_embeddings_dict),
            ("Flava", flava_embeddings_dict)
        ]

        combo_names = list(dataset_combinations.keys())

        for row_idx, (model_name, embeddings_dict) in enumerate(model_data):
            for col_idx, combo_name in enumerate(combo_names):
                ax = axes[row_idx, col_idx]

                dataset_names = dataset_combinations[combo_name]

                # Collect embeddings and labels for this combination
                all_embeddings = []
                all_dataset_names = []

                for dataset_name in dataset_names:
                    if dataset_name in embeddings_dict:
                        data = embeddings_dict[dataset_name]
                        all_embeddings.append(data["embeddings"])
                        all_dataset_names.extend([dataset_name] * len(data["embeddings"]))

                if len(all_embeddings) == 0:
                    ax.text(0.5, 0.5, f'No data\nfor {combo_name}',
                           ha='center', va='center', transform=ax.transAxes, fontsize=27)
                    ax.set_title(f'{model_name}: {combo_name}', fontsize=30, fontweight='bold')
                    continue

                # Combine all embeddings
                X = np.vstack(all_embeddings)

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Apply PCA
                pca = PCA(n_components=2, random_state=self.random_seed)
                X_pca = pca.fit_transform(X_scaled)

                # Plot each dataset with unified colors
                unique_datasets = list(set(all_dataset_names))
                for dataset_name in unique_datasets:
                    mask = np.array(all_dataset_names) == dataset_name
                    # Use unified color scheme
                    color = unified_colors.get(dataset_name, "gray")
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                             c=color, label=dataset_name, alpha=0.7, s=50)

                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=27, fontweight='bold')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=27, fontweight='bold')
                ax.set_title(f'{model_name}: {combo_name}', fontsize=30, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Make tick labels larger
                ax.tick_params(axis='both', which='major', labelsize=24)

        # Create a unified legend with all datasets that appear in any subplot
        all_unique_datasets = set()
        for dataset_names in dataset_combinations.values():
            for dataset_name in dataset_names:
                if dataset_name in llava_embeddings_dict or dataset_name in flava_embeddings_dict:
                    all_unique_datasets.add(dataset_name)

        # Create legend handles manually using the unified color scheme
        legend_handles = []
        legend_labels = []
        for dataset_name in sorted(all_unique_datasets):
            color = unified_colors.get(dataset_name, "gray")
            handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                              markersize=10, alpha=0.7, linestyle='None')
            legend_handles.append(handle)
            legend_labels.append(dataset_name)

        # Add the unified legend to the figure
        fig.legend(legend_handles, legend_labels, loc='center right',
                  bbox_to_anchor=(0.98, 0.5), fontsize=21, frameon=True,
                  fancybox=True, shadow=True)

        # Adjust layout with more spacing and room for external legend
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.4, right=0.82)

        # Save the combined plot in both PDF and PNG formats
        base_filename = "combined_pca_visualization"

        # Save PDF version
        pdf_filepath = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_filepath, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved combined PCA plot (PDF): {pdf_filepath}")

        # Save PNG version
        png_filepath = os.path.join(output_dir, f"{base_filename}.png")
        plt.savefig(png_filepath, format='png', dpi=300, bbox_inches='tight')
        print(f"Saved combined PCA plot (PNG): {png_filepath}")

        plt.close()
    
    def run_visualization_analysis(self):
        """Run the complete visualization analysis"""
        print("="*80)
        print("PCA VISUALIZATION OF LLAVA AND FLAVA EMBEDDINGS")
        print("="*80)
        
        # Load datasets
        datasets = self.load_datasets_for_visualization()
        
        if len(datasets) == 0:
            print("No datasets loaded. Exiting.")
            return
        
        # Define dataset combinations for visualization
        combinations = {
            "MM-Vet & FigStep": ["MM-Vet", "FigStep"],
            "Basic Datasets": ["MM-Vet", "FigStep", "Alpaca", "AdvBench"],  # One text-only and one multimodal for each category
            "JailDAM": ["MM-Vet", "MM-SafetyBench", "FigStep", "JailbreakV-LLM"],  # JailDAM experiment setup
            "All Datasets": ["MM-Vet", "FigStep", "VQAv2", "AdvBench", "Alpaca",
                           "OpenAssistant", "JailbreakV-LLM", "DAN", "XSTest-Safe", "XSTest-Unsafe", "MM-SafetyBench"]
        }
        
        # Extract embeddings for both models
        llava_embeddings_dict = {}
        flava_embeddings_dict = {}

        for model_type in ["llava", "flava"]:
            print(f"\n{'='*60}")
            print(f"PROCESSING {model_type.upper()} EMBEDDINGS")
            print(f"{'='*60}")

            # Extract embeddings
            embeddings_dict = self.extract_embeddings(datasets, model_type)

            if len(embeddings_dict) == 0:
                print(f"No embeddings extracted for {model_type}. Skipping.")
                continue

            # Store embeddings for combined visualization
            if model_type == "llava":
                llava_embeddings_dict = embeddings_dict
            else:
                flava_embeddings_dict = embeddings_dict

            # Create individual visualizations
            self.create_pca_visualization(embeddings_dict, model_type, combinations)

        # Create combined visualization if we have both models
        if len(llava_embeddings_dict) > 0 and len(flava_embeddings_dict) > 0:
            print(f"\n{'='*60}")
            print("CREATING COMBINED VISUALIZATION")
            print(f"{'='*60}")
            self.create_combined_pca_visualization(llava_embeddings_dict, flava_embeddings_dict, combinations)
        
        print("\n" + "="*80)
        print("VISUALIZATION ANALYSIS COMPLETE")
        print("="*80)
        print("Check the 'visualizations' directory for both PDF and PNG plots.")
        print("\nIndividual plots for each model and case (saved in both PDF and PNG formats):")
        print("1. MM-Vet & FigStep: Basic separability between benign and malicious")
        print("2. Basic Datasets: How separability changes with more datasets")
        print("3. JailDAM: Previous work setup (MM-Vet vs MM-SafetyBench, FigStep, JailbreakV-28k)")
        print("4. All Datasets: Full complexity with all available datasets")
        print("\nCombined visualization (saved in both PDF and PNG formats):")
        print("- combined_pca_visualization.pdf/.png: 2x4 subplot showing all models and cases in one figure")

def main():
    visualizer = EmbeddingVisualizer()
    visualizer.run_visualization_analysis()

if __name__ == "__main__":
    main()
