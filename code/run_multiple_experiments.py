#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
import pandas as pd
import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime

def backup_original_file(filepath):
    """Create a backup of the original file"""
    backup_path = str(filepath) + '.backup'
    with open(filepath, 'r') as original:
        content = original.read()
    with open(backup_path, 'w') as backup:
        backup.write(content)
    return backup_path

def restore_from_backup(filepath, backup_path):
    """Restore file from backup"""
    with open(backup_path, 'r') as backup:
        content = backup.read()
    with open(filepath, 'w') as original:
        original.write(content)

def modify_seed_in_content(content, new_seed):
    """Modify the MAIN_SEED variable in file content (centralized seed management)"""
    import re

    # Target the centralized MAIN_SEED variable (both direct and config-based)
    main_seed_pattern = r'MAIN_SEED = \d+'
    replacement = f'MAIN_SEED = {new_seed}'

    # Check if the new centralized seed system is being used
    if re.search(main_seed_pattern, content):
        # Use the new centralized approach - this handles both direct MAIN_SEED and config-based MAIN_SEED
        content = re.sub(main_seed_pattern, replacement, content)
        print(f"  Using centralized MAIN_SEED approach")

        # Count how many replacements were made
        seed_count = len(re.findall(main_seed_pattern.replace(r'\d+', str(new_seed)), content))
        if seed_count > 1:
            print(f"  Updated {seed_count} MAIN_SEED occurrences (including config classes)")
    else:
        # Fallback to old approach for backward compatibility
        print(f"  Warning: MAIN_SEED not found, using legacy seed replacement")

        # Replace individual seed calls (legacy approach)
        content = re.sub(r'random\.seed\(\d+\)', f'random.seed({new_seed})', content)
        content = re.sub(r'np\.random\.seed\(\d+\)', f'np.random.seed({new_seed})', content)
        content = re.sub(r'torch\.manual_seed\(\d+\)', f'torch.manual_seed({new_seed})', content)
        content = re.sub(r'torch\.cuda\.manual_seed\(\d+\)', f'torch.cuda.manual_seed({new_seed})', content)
        content = re.sub(r'torch\.cuda\.manual_seed_all\(\d+\)', f'torch.cuda.manual_seed_all({new_seed})', content)

        # Replace layer seed calculation (any_number + layer_idx pattern)
        content = re.sub(r'layer_seed = \d+ \+ layer_idx', f'layer_seed = {new_seed} + layer_idx', content)

        # Replace the print statement (both literal numbers and variable references)
        content = re.sub(r'Random seeds set for reproducibility \(seed=\d+\)',
                        f'Random seeds set for reproducibility (seed={new_seed})', content)
        # Also handle cases where the print statement uses a variable like {MAIN_SEED}
        content = re.sub(r'Random seeds set for reproducibility \(seed=\{MAIN_SEED\}\)',
                        f'Random seeds set for reproducibility (seed={new_seed})', content)

        # Replace the default parameter in function signature
        content = re.sub(r'random_seed=\d+\)', f'random_seed={new_seed})', content)

    return content

def modify_seed_in_file(filepath, new_seed):
    """Modify the random seed in a Python file"""
    print(f"Updating seed to {new_seed} in {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    modified_content = modify_seed_in_content(content, new_seed)

    # Verify that changes were made
    if content == modified_content:
        # Check if seed is already correct
        import re
        if re.search(f'MAIN_SEED = {new_seed}', content):
            print(f"  MAIN_SEED already set to {new_seed} - no changes needed")
        else:
            print(f"  Warning: No seed replacements made in {filepath}")
            print("  This might indicate the seed patterns were not found or script needs updating")
    else:
        print(f"  Successfully updated seed to {new_seed}")

    with open(filepath, 'w') as f:
        f.write(modified_content)

def run_single_experiment(script_path, seed, run_id, total_runs, working_dir,
                          model_type=None, script_name=None, layer_subset=None):
    """Run a single experiment with given seed"""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT {run_id}/{total_runs} WITH SEED {seed}")
    print(f"{'='*80}")

    # Create backup of original file
    backup_path = backup_original_file(script_path)

    try:
        # Modify seed in the script
        modify_seed_in_file(script_path, seed)

        # Run the experiment from the correct working directory
        # Use relative path from working_dir to script
        script_relative = os.path.relpath(script_path, working_dir)

        print(f"Working directory: {working_dir}")
        print(f"Running script: {script_relative}")
        print(f"Using seed: {seed}")

        # Build command - add model argument for MCD and KCD scripts
        cmd = [sys.executable, script_relative]
        if model_type and script_name in ['mcd', 'kcd']:
            cmd.extend(['--model', model_type])
            print(f"Using model: {model_type}")
        if layer_subset:
            cmd.extend(['--layers', layer_subset])
            print(f"Restricting layers to {layer_subset}")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=working_dir)

        if result.returncode != 0:
            print(f"ERROR in run {run_id}:")
            print("STDOUT:", result.stdout[-1000:] if result.stdout else "None")  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:] if result.stderr else "None")  # Last 1000 chars
            return False
        else:
            print(f"Run {run_id} completed successfully")
            return True

    except Exception as e:
        print(f"ERROR running experiment {run_id}: {e}")
        return False
    finally:
        # Always restore from backup after each run
        restore_from_backup(script_path, backup_path)
        # Clean up backup file
        try:
            os.remove(backup_path)
        except:
            pass  # Ignore cleanup errors

def load_results_csv(csv_path):
    """Load results from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def aggregate_results_group(results_list, method_name, metric_group, group_name):
    """Aggregate results for a specific group of metrics"""
    if not results_list:
        return None

    # Combine all dataframes
    all_results = pd.concat(results_list, ignore_index=True)

    # Filter out non-numeric values (like "N/A")
    for col in metric_group:
        if col in all_results.columns:
            all_results[col] = pd.to_numeric(all_results[col], errors='coerce')

    # Group and aggregate - include Method for ML experiments to preserve individual methods
    aggregated_results = []

    if 'ML_' in method_name:
        # For ML experiments, group by Layer, Dataset, and Method to preserve individual ML methods
        grouped = all_results.groupby(['Layer', 'Dataset', 'Method'])

        for (layer, dataset, method), group in grouped:
            result_row = {
                'Layer': layer,
                'Dataset': dataset,
                'Method': method,  # Use the actual method name (ML_LOGISTIC, ML_RIDGE, etc.)
                'Group': group_name,
                'Runs': len(group)
            }

            # Compute mean, std, and max/min for each metric
            for col in metric_group:
                if col in group.columns:
                    values = group[col].dropna()
                    if len(values) > 0:
                        result_row[f'{col}_Mean'] = round(values.mean(), 4)
                        result_row[f'{col}_Std'] = round(values.std() if len(values) > 1 else 0.0, 4)

                        # For FPR, we want minimum (lower is better)
                        # For other metrics, we want maximum (higher is better)
                        if col == 'FPR':
                            result_row[f'{col}_Min'] = round(values.min(), 4)
                        else:
                            result_row[f'{col}_Max'] = round(values.max(), 4)
                    else:
                        # Use 0.0000 instead of NaN for missing/non-applicable metrics
                        result_row[f'{col}_Mean'] = 0.0000
                        result_row[f'{col}_Std'] = 0.0000

                        if col == 'FPR':
                            result_row[f'{col}_Min'] = 0.0000
                        else:
                            result_row[f'{col}_Max'] = 0.0000

            aggregated_results.append(result_row)
    else:
        # For MCD/KCD experiments, group by Layer and Dataset (original behavior)
        grouped = all_results.groupby(['Layer', 'Dataset'])

        for (layer, dataset), group in grouped:
            result_row = {
                'Layer': layer,
                'Dataset': dataset,
                'Method': method_name,  # Use the provided method name
                'Group': group_name,
                'Runs': len(group)
            }

            # Compute mean, std, and max/min for each metric
            for col in metric_group:
                if col in group.columns:
                    values = group[col].dropna()
                    if len(values) > 0:
                        result_row[f'{col}_Mean'] = round(values.mean(), 4)
                        result_row[f'{col}_Std'] = round(values.std() if len(values) > 1 else 0.0, 4)

                        # For FPR, we want minimum (lower is better)
                        # For other metrics, we want maximum (higher is better)
                        if col == 'FPR':
                            result_row[f'{col}_Min'] = round(values.min(), 4)
                        else:
                            result_row[f'{col}_Max'] = round(values.max(), 4)
                    else:
                        # Use 0.0000 instead of NaN for missing/non-applicable metrics
                        result_row[f'{col}_Mean'] = 0.0000
                        result_row[f'{col}_Std'] = 0.0000

                        if col == 'FPR':
                            result_row[f'{col}_Min'] = 0.0000
                        else:
                            result_row[f'{col}_Max'] = 0.0000

            aggregated_results.append(result_row)



    df = pd.DataFrame(aggregated_results)

    # Add individual rankings within each group based on primary metric
    # Group 1 (Performance): rank by Accuracy_Mean (higher is better)
    # Group 2 (Classification): rank by F1_Mean (higher is better)
    if group_name == "Performance":
        primary_metric = 'Accuracy_Mean'
        ascending = False  # Higher accuracy is better
    else:  # Classification group
        primary_metric = 'F1_Mean'
        ascending = False  # Higher F1 is better

    # Add ranking for COMBINED results only
    combined_df = df[df['Dataset'] == 'COMBINED'].copy()
    if not combined_df.empty and primary_metric in combined_df.columns:
        combined_df = combined_df.sort_values(primary_metric, ascending=ascending)
        combined_df[f'{group_name}_Rank'] = range(1, len(combined_df) + 1)

        # Merge rankings back to main dataframe
        # For ML experiments, we need to use both Layer and Method for mapping
        if 'ML_' in combined_df['Method'].iloc[0] if len(combined_df) > 0 else False:
            # ML experiments: create mapping using (Layer, Method) tuple
            rank_mapping = dict(zip(zip(combined_df['Layer'], combined_df['Method']), combined_df[f'{group_name}_Rank']))
            df[f'{group_name}_Rank'] = df.apply(lambda row: rank_mapping.get((row['Layer'], row['Method']), 0), axis=1)
        else:
            # MCD/KCD experiments: use Layer only (original behavior)
            rank_mapping = dict(zip(combined_df['Layer'], combined_df[f'{group_name}_Rank']))
            df[f'{group_name}_Rank'] = df['Layer'].map(rank_mapping)
            df[f'{group_name}_Rank'] = df[f'{group_name}_Rank'].fillna(0)  # Fill NaN with 0 for non-COMBINED
    else:
        df[f'{group_name}_Rank'] = 0

    return df


def aggregate_results(results_list, method_name):
    """Aggregate results into two focused groups"""
    if not results_list:
        return None, None

    # Define metric groups
    group1_metrics = ['Accuracy', 'AUROC', 'AUPRC']  # Performance metrics
    group2_metrics = ['TPR', 'FPR', 'F1']            # Classification metrics

    # Create aggregated results for each group
    group1_df = aggregate_results_group(results_list, method_name, group1_metrics, "Performance")
    group2_df = aggregate_results_group(results_list, method_name, group2_metrics, "Classification")

    return group1_df, group2_df

def find_newest_matching_directory(output_dir, script, model=None):
    """Find the newest directory that matches the script and model combination"""
    if not output_dir.exists():
        return None

    # Build the expected prefix for directory names
    if model:
        prefix = f"{script}_{model}_"
    else:
        prefix = f"{script}_"

    # Find all matching directories
    matching_dirs = []
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            # Extract timestamp from directory name (format: script_model_Nruns_YYYYMMDD_HHMMSS)
            parts = item.name.split('_')
            if len(parts) >= 4:  # At least script_model_Nruns_YYYYMMDD_HHMMSS
                try:
                    # The timestamp should be the last two parts: YYYYMMDD_HHMMSS
                    date_part = parts[-2]  # YYYYMMDD
                    time_part = parts[-1]  # HHMMSS
                    if len(date_part) == 8 and date_part.isdigit() and len(time_part) == 6 and time_part.isdigit():
                        timestamp = f"{date_part}_{time_part}"
                        matching_dirs.append((item, timestamp))
                except:
                    continue

    if not matching_dirs:
        return None

    # Sort by timestamp (newest first)
    matching_dirs.sort(key=lambda x: x[1], reverse=True)
    return matching_dirs[0][0]  # Return the directory path

def count_successful_runs(run_dir, expected_csv_pattern):
    """Count the number of successful runs in a directory"""
    if not run_dir.exists():
        return 0, []

    # Count run_XX_seed_YY_*.csv files
    run_files = []
    for item in run_dir.iterdir():
        if item.is_file() and item.name.startswith('run_') and item.name.endswith('.csv'):
            # Extract run number and seed from filename
            parts = item.name.split('_')
            if len(parts) >= 4:  # run_XX_seed_YY_...
                try:
                    run_num = int(parts[1])
                    seed = int(parts[3])
                    run_files.append((run_num, seed, item))
                except ValueError:
                    continue

    # Sort by run number
    run_files.sort(key=lambda x: x[0])
    return len(run_files), run_files

def load_existing_metadata(run_dir):
    """Load existing experiment metadata if it exists"""
    metadata_file = run_dir / 'experiment_metadata.json'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_file}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments with different seeds')
    parser.add_argument('--script', required=True, choices=['mcd', 'kcd'],
                       help='Which script to run (mcd or kcd)')
    parser.add_argument('--model', choices=['llava', 'qwen', 'internvl'], default='llava',
                       help='Model type to use (llava, qwen, or internvl, default: llava). Only applies to mcd and kcd scripts.')
    parser.add_argument('--runs', type=int, default=50,
                       help='Number of runs (default: 50)')
    parser.add_argument('--seeds', nargs='+', type=int,
                       help='Custom seeds to use (if not provided, will use consecutive seeds starting from 42)')
    parser.add_argument('--output-dir', default='multi_run_results',
                       help='Output directory for results (default: multi_run_results)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from the newest matching experiment directory if it has fewer runs than requested')
    parser.add_argument('--layers', default=None,
                       help='Comma-separated layer indices to train/evaluate (e.g., 12,16,20)')

    # Add usage examples in help
    parser.epilog = """
Examples:
  python code/run_multiple_experiments.py --script mcd --model qwen --runs 10
  python code/run_multiple_experiments.py --script kcd --model qwen --runs 5 --seeds 42 43 44 45 46
  python code/run_multiple_experiments.py --script mcd --model internvl --runs 20
  python code/run_multiple_experiments.py --script kcd --model internvl --runs 10
  python code/run_multiple_experiments.py --script mcd --model llava --runs 50 --resume  (resume from newest matching directory)
"""
    
    args = parser.parse_args()
    
    # Determine script path and working directory
    script_dir = Path(__file__).parent  # This is the 'code' directory
    working_dir = script_dir.parent      # This is the parent directory (where experiments should run)

    if args.script == 'mcd':
        script_path = script_dir / 'balanced_ood_mcd.py'
        expected_csv = f'results/balanced_mcd_{args.model}_results.csv'
        method_name = f'MCD_k_MultiSeed_{args.model.upper()}'
    elif args.script == 'kcd':
        script_path = script_dir / 'balanced_ood_kcd.py'
        expected_csv = f'results/balanced_kcd_{args.model}_results.csv'
        method_name = f'KCD_MultiSeed_{args.model.upper()}'

    if not script_path.exists():
        print(f"Error: Script {script_path} not found!")
        return

    print(f"Script directory: {script_dir}")
    print(f"Working directory: {working_dir}")
    print(f"Script path: {script_path}")

    layer_subset = None
    layer_subset_str = None
    if args.layers:
        try:
            parsed_layers = [int(part.strip()) for part in args.layers.split(',') if part.strip()]
        except ValueError:
            print("Error: --layers must be a comma-separated list of integers.")
            return
        if parsed_layers:
            layer_subset = sorted(set(parsed_layers))
            layer_subset_str = ','.join(str(layer) for layer in layer_subset)
            print(f"Layer subset requested: {layer_subset}")
        else:
            print("Warning: --layers provided but no valid indices found. Ignoring.")
    
    # Create output directory (relative to working directory)
    output_dir = working_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Handle resume functionality
    run_dir = None
    existing_runs = 0
    existing_seeds = []
    resume_from_run = 1

    if args.resume:
        print(f"Looking for existing experiments to resume...")
        existing_dir = find_newest_matching_directory(output_dir, args.script, args.model)

        if existing_dir:
            print(f"Found existing directory: {existing_dir}")

            # Count existing successful runs
            existing_runs, run_files = count_successful_runs(existing_dir, expected_csv)
            print(f"Found {existing_runs} existing successful runs")

            # Load existing metadata to get the original seeds
            metadata = load_existing_metadata(existing_dir)
            if metadata:
                existing_seeds = metadata.get('seeds', [])[:existing_runs]  # Only count successful runs
                print(f"Existing seeds: {existing_seeds}")
                stored_layers = metadata.get('layer_subset')
                if layer_subset is None and stored_layers:
                    layer_subset = stored_layers
                    if layer_subset:
                        layer_subset_str = ','.join(str(layer) for layer in layer_subset)
                    print(f"Resuming with stored layer subset: {layer_subset}")
                elif layer_subset is not None and stored_layers and sorted(stored_layers) != sorted(layer_subset):
                    print("Warning: Provided --layers differs from stored layer subset in the existing experiment.")

            if existing_runs < args.runs:
                print(f"Resuming from run {existing_runs + 1} (need {args.runs - existing_runs} more runs)")
                run_dir = existing_dir
                resume_from_run = existing_runs + 1
            else:
                print(f"Existing directory already has {existing_runs} runs (>= requested {args.runs})")
                print("No additional runs needed. Use a different --runs value or create a new experiment.")
                return
        else:
            print("No existing matching directory found. Starting new experiment.")

    # Determine seeds
    if args.seeds:
        if args.resume and existing_runs > 0:
            # When resuming with custom seeds, use the remaining seeds
            remaining_seeds = args.seeds[existing_runs:][:args.runs - existing_runs]
            if len(remaining_seeds) < (args.runs - existing_runs):
                print(f"Warning: Not enough custom seeds for remaining runs. Need {args.runs - existing_runs}, got {len(remaining_seeds)}")
                args.runs = existing_runs + len(remaining_seeds)
            seeds = remaining_seeds
        else:
            seeds = args.seeds[:args.runs]  # Use provided seeds
            if len(seeds) < args.runs:
                print(f"Warning: Only {len(seeds)} seeds provided, but {args.runs} runs requested")
                args.runs = len(seeds)
    else:
        if args.resume and existing_runs > 0:
            # Continue with consecutive seeds from where we left off
            last_seed = existing_seeds[-1] if existing_seeds else 41  # Default to 41 so next is 42
            seeds = list(range(last_seed + 1, last_seed + 1 + (args.runs - existing_runs)))
        else:
            seeds = list(range(42, 42 + args.runs))  # Use consecutive seeds starting from 42

    # Create run-specific subdirectory if not resuming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Always define timestamp
    if run_dir is None:
        run_dir = output_dir / f"{args.script}_{args.model}_{args.runs}runs_{timestamp}"
        run_dir.mkdir(exist_ok=True)
    
    if args.resume and existing_runs > 0:
        print(f"Resuming experiment: running {len(seeds)} additional experiments")
        print(f"Total target runs: {args.runs} (already have {existing_runs})")
    else:
        print(f"Running {args.runs} experiments with {args.script.upper()} method")

    print(f"Model: {args.model.upper()}")
    print(f"Seeds: {seeds}")
    print(f"Results will be saved to: {run_dir}")

    # Store individual run results
    individual_results = []
    successful_runs = existing_runs  # Start with existing successful runs

    # Load existing results if resuming
    if args.resume and existing_runs > 0:
        print(f"Loading {existing_runs} existing results...")
        existing_run_count, existing_run_files = count_successful_runs(run_dir, expected_csv)
        for run_num, seed, csv_file in existing_run_files:
            df = load_results_csv(csv_file)
            if df is not None:
                df['Run'] = run_num
                df['Seed'] = seed
                individual_results.append(df)
                print(f"Loaded existing run {run_num} (seed {seed})")

    for i, seed in enumerate(seeds, resume_from_run):
        success = run_single_experiment(script_path, seed, i, args.runs, working_dir,
                                        args.model, args.script, layer_subset_str)

        if success:
            # Move the generated CSV to run directory (CSV is generated in working_dir)
            csv_source = working_dir / expected_csv

            # Create results directory if it doesn't exist
            results_dir = working_dir / "results"
            if not results_dir.exists():
                results_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created results directory: {results_dir}")

            # Check if the CSV file actually exists
            if not csv_source.exists():
                print(f"Warning: Expected CSV file not found at {csv_source}")
                print(f"Looking for alternative locations...")

                # Try alternative paths
                alt_paths = [
                    working_dir / expected_csv.split('/')[-1],  # Just filename in working_dir
                    script_dir / expected_csv.split('/')[-1],   # Just filename in script_dir
                    script_dir / expected_csv                   # Full path in script_dir
                ]

                for alt_path in alt_paths:
                    if alt_path.exists():
                        csv_source = alt_path
                        print(f"Found CSV at alternative location: {csv_source}")
                        break
                else:
                    print(f"ERROR: Could not find CSV file for run {i}")
                    continue
            # Create destination filename (just the base filename, not the full path)
            csv_filename = expected_csv.split('/')[-1]  # Get just the filename
            csv_dest = run_dir / f"run_{i:02d}_seed_{seed}_{csv_filename}"

            # Move/copy the file
            try:
                csv_source.rename(csv_dest)
                print(f"Results moved: {csv_source} -> {csv_dest}")

                # Load and store results
                df = load_results_csv(csv_dest)
                if df is not None:
                    df['Run'] = i
                    df['Seed'] = seed
                    individual_results.append(df)
                    successful_runs += 1
                    print(f"Results processed successfully for run {i}")
                else:
                    print(f"Warning: Could not load CSV data for run {i}")
            except Exception as e:
                print(f"Error moving results file for run {i}: {e}")
        
        current_run_in_sequence = i - resume_from_run + 1
        total_new_runs = len(seeds)
        print(f"Completed {current_run_in_sequence}/{total_new_runs} new runs ({successful_runs} total successful)")
    
    if successful_runs == 0:
        print("ERROR: No successful runs completed!")
        return
    
    print(f"\n{'='*80}")
    print(f"AGGREGATING RESULTS FROM {successful_runs} SUCCESSFUL RUNS")
    print(f"{'='*80}")
    
    # Aggregate results for MCD and KCD methods
    group1_df, group2_df = aggregate_results(individual_results, method_name)

    if group1_df is not None and group2_df is not None:
        # Save aggregated results for both groups
        group1_csv = run_dir / f"aggregated_{args.script}_performance.csv"
        group2_csv = run_dir / f"aggregated_{args.script}_classification.csv"

        group1_df.to_csv(group1_csv, index=False)
        group2_df.to_csv(group2_csv, index=False)

        print(f"Performance metrics saved: {group1_csv}")
        print(f"Classification metrics saved: {group2_csv}")

        # Combine all seeds (existing + new)
        all_seeds = existing_seeds + seeds

        # Save experiment metadata
        metadata = {
            'script': args.script,
            'model': args.model,
            'total_runs': args.runs,
            'successful_runs': successful_runs,
            'seeds': all_seeds,
            'timestamp': run_dir.name.split('_')[-1] if '_' in run_dir.name else datetime.now().strftime("%Y%m%d_%H%M%S"),
            'method': method_name,
            'resumed': args.resume and existing_runs > 0,
            'original_runs': existing_runs,
            'layer_subset': layer_subset or [],
            'output_files': {
                'performance_metrics': str(group1_csv.name),
                'classification_metrics': str(group2_csv.name)
            }
        }

        metadata_file = run_dir / 'experiment_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Experiment metadata saved: {metadata_file}")

        # Print summary statistics
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        print(f"Method: {method_name}")
        print(f"Model: {args.model.upper()}")
        print(f"Successful runs: {successful_runs}/{args.runs}")
        print(f"Seeds used: {seeds[:successful_runs]}")
        print(f"Results directory: {run_dir}")
        print(f"Performance metrics (Accuracy, AUROC, AUPRC): {group1_csv}")
        print(f"Classification metrics (TPR, FPR, F1): {group2_csv}")

        # Show top 5 layers for each group
        combined_perf = group1_df[group1_df['Dataset'] == 'COMBINED'].sort_values('Performance_Rank').head(5)
        combined_class = group2_df[group2_df['Dataset'] == 'COMBINED'].sort_values('Classification_Rank').head(5)

        if not combined_perf.empty:
            print(f"\nTop 5 Performance Layers (by Accuracy):")
            for _, row in combined_perf.iterrows():
                print(f"  Layer {int(row['Layer'])}: Accuracy={row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.4f}")

        if not combined_class.empty:
            print(f"\nTop 5 Classification Layers (by F1):")
            for _, row in combined_class.iterrows():
                print(f"  Layer {int(row['Layer'])}: F1={row['F1_Mean']:.4f}±{row['F1_Std']:.4f}")
    else:
        print("ERROR: Failed to aggregate results!")

if __name__ == "__main__":
    main()
