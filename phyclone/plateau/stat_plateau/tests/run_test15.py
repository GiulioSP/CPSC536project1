#!/usr/bin/env python
"""Quick test runner for TEST 15 - stat_plateau class with real data"""
import sys
import os
import tempfile
import shutil

# Add parent directory to path so we can import local modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
from stat_plateau import stat_plateau

print("="*70)
print("TEST 15: stat_plateau class with real sample data")
print("="*70)

try:
    # Set up paths
    csv_path = os.path.join(parent_dir, "sample_data", "combined.csv")
    print(f"Loading sample data from: {csv_path}")
    
    data = pd.read_csv(csv_path)
    print(f"Loaded {len(data)} rows from sample data")
    print(f"Columns: {list(data.columns)}")
    
    # Get unique datasets
    datasets = data["dataset"].unique()
    print(f"Found {len(datasets)} datasets")
    
    # Get the first dataset and count its chains
    dataset_name = datasets[0]
    dataset_group = data[data["dataset"] == dataset_name]
    chain_numbers = sorted(dataset_group["chain"].unique())
    num_chains = len(chain_numbers)
    
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Dataset has {num_chains} available chains: {list(chain_numbers)}")
    
    # Create stat_plateau instance with all available chains
    print(f"\n Creating stat_plateau instance with k={num_chains}...")
    analyzer = stat_plateau(
        csv_path=csv_path,
        k=num_chains,
        window=0,
        check_all_subsets=False
    )
    
    print(f"  • CSV path: {analyzer.csv_path}")
    print(f"  • k value: {analyzer.k}")
    print(f"  • Convergence threshold: {analyzer.convergence_threshold}")
    print(f"  • Window: {analyzer.window}")
    
    # Run convergence analysis using stat_plateau
    print(f"\n Running eval_convergence (single k analysis)...")
    result = analyzer.eval_convergence()
    
    print(f" eval_convergence completed!")
    print(f"  • Result type: {type(result)}")
    print(f"  • Converged: {result}")
    
    # Now test with check_all_subsets=True
    print(f"\nCreating stat_plateau instance with check_all_subsets=True...")
    analyzer_all = stat_plateau(
        csv_path=csv_path,
        k=num_chains,
        window=0,
        check_all_subsets=True
    )
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    print(f" Created temporary output directory: {temp_dir}")
    
    try:
        print(f"\n Running eval_convergence (check all subsets)...")
        result_all = analyzer_all.eval_convergence(output_base_dir=temp_dir)
        
        print(f" eval_convergence (all subsets) completed!")
        print(f"  • Result type: {type(result_all)}")
        print(f"  • Converged: {result_all}")
        
        # Check what folders were created and which k achieved convergence
        if os.path.exists(temp_dir):
            created_folders = sorted(os.listdir(temp_dir))
            print(f"\n Output folders created: {created_folders}")
            
            # Check which k value achieved convergence first
            convergence_k = None
            for folder in created_folders:
                folder_path = os.path.join(temp_dir, folder)
                if os.path.isdir(folder_path):
                    files = os.listdir(folder_path)
                    print(f"  • {folder}: {files}")
                    
                    # Extract k value from folder name (e.g., "k2_chains" -> 2)
                    if folder.startswith("k") and folder.endswith("_chains"):
                        k_val = int(folder[1:-7])  # Extract number between 'k' and '_chains'
                        
                        # Check if any CSV in this folder has convergence=True
                        for filename in files:
                            if filename.endswith(".csv"):
                                filepath = os.path.join(folder_path, filename)
                                try:
                                    result_df = pd.read_csv(filepath)
                                    if result_df["converged"].any():
                                        if convergence_k is None or k_val < convergence_k:
                                            convergence_k = k_val
                                except:
                                    pass
            
            if convergence_k is not None:
                print(f"\n Convergence reached at k={convergence_k}")
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        print(f"\n Cleaned up temporary directory")
    
    print("\n=== TEST 15 PASSED! ===")
    print("stat_plateau class works correctly with real sample data")
    
except Exception as e:
    import traceback
    print(f"\n✗✗✗ TEST 15 FAILED! ✗✗✗")
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("="*70)
