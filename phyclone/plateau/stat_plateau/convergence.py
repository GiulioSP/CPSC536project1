import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from .convergenceMetrics import PSRF_between_chain_ESS, calculate_rhat

# Python script for convergence analysis
def analyze_convergence(
    chains_list,
    ess_threshold=None,
    lag_window=10,
    k=None,
):
    """
    Assess convergence of multiple MCMC chains using between-chain ESS and Rhat diagnostics.
    
    Iterates through the chains at intervals of lag_window iterations, computing both
    PSRF_between_chain_ESS and Rhat at each interval. Stops and returns when the between-chain
    ESS reaches the threshold or only one chain remains.
    
    When chains have different lengths, the algorithm continues with remaining chains after
    shorter chains are exhausted. For example, if chains have lengths [100, 100, 50, 50],
    the algorithm will:
    - Use all 4 chains until iteration 50
    - Use first 2 chains from iteration 60 onward
    
    Args:
      chains_list: list[array-like] of logP value sequences, one array per chain.
                   Each element should be array-like (list, np.ndarray, pd.Series, etc.)
                   containing float logP values from a single chain.
                   Example: [np.array([...]), np.array([...]), np.array([...])]
      ess_threshold: ESS threshold to reach. If None, defaults to 20 * number of chains.
      lag_window: Interval (in iterations) at which to compute diagnostics. Default = 10.
      k: Number of chains to use in ESS calculation. If None, uses all chains.
         If specified, computes ESS for all possible k-sized subsets and returns True
         if ANY subset's ESS exceeds threshold.
    
    Returns:
      pd.DataFrame with columns:
        - 'iteration': iteration number (multiple of lag_window) at which diagnostics computed
        - 'between_chain_ess': between-chain ESS value at that iteration
        - 'rhat': Rhat convergence diagnostic at that iteration
        - 'converged': boolean indicating if threshold was reached at this iteration
      
      The dataframe will contain rows for each lag_window interval, stopping when ESS exceeds
      threshold, only one chain remains, or all iterations are exhausted.
    """
    # Convert all chains to numpy arrays
    chains = [np.asarray(chain, dtype=float) for chain in chains_list]
    M = len(chains)  # total number of chains
    
    if M < 2:
        raise ValueError("Need at least 2 chains for convergence analysis")
    
    # Find maximum chain length
    max_length = max(len(chain) for chain in chains)
    if max_length < lag_window:
        raise ValueError(f"Chains must have at least {lag_window} samples (lag_window)")
    
    # Set k to M if not specified (use all chains)
    if k is None:
        k = M
    elif k < 2 or k > M:
        raise ValueError(f"k must be between 2 and {M} (number of chains)")
    
    # Set default threshold based on total number of chains (not k)
    if ess_threshold is None:
        ess_threshold = 20 * M
    
    rows = []
    
    # Iterate at lag_window intervals with progress bar
    for N in tqdm(range(lag_window, max_length + 1, lag_window), desc="Computing diagnostics"):
        # Identify which chains have at least N samples
        valid_chain_indices = [i for i, chain in enumerate(chains) if len(chain) >= N]
        
        # Stop if fewer than 2 chains remain
        if len(valid_chain_indices) < 2:
            print(f"Stopping at iteration {N}: only {len(valid_chain_indices)} chain(s) remaining")
            break
        
        # Extract first N samples from valid chains
        chains_at_N = [chains[i][:N] for i in valid_chain_indices]
        
        try:
            # Generate all k-sized subsets of valid chain indices
            # k is based on current number of valid chains
            current_k = min(k, len(valid_chain_indices))
            if current_k < 2:
                current_k = 2
            
            k_subsets = list(combinations(range(len(chains_at_N)), current_k))
            
            # Compute ESS for all k-sized subsets
            max_ess_across_subsets = -np.inf
            rhat_list = []
            
            for subset_indices in k_subsets:
                # Extract chains for this subset
                subset_chains = [chains_at_N[i] for i in subset_indices]
                
                # Compute between-chain ESS for this subset
                subset_ess = PSRF_between_chain_ESS(subset_chains, threshold=ess_threshold)
                
                # Handle return type: could be int (threshold reached) or tuple (threshold not reached)
                if isinstance(subset_ess, tuple):
                    subset_ess_value = subset_ess[1]
                else:
                    subset_ess_value = subset_ess
                
                # Track maximum ESS across subsets
                if subset_ess_value > max_ess_across_subsets:
                    max_ess_across_subsets = subset_ess_value
                
                # Compute Rhat for this subset (only on first subset to reduce computation)
                if not rhat_list:
                    rhat = calculate_rhat(subset_chains)
                    rhat_list.append(rhat)
            
            # Use the maximum ESS from all subsets
            between_chain_ess_value = max_ess_across_subsets
            rhat = rhat_list[0]
            
            # Check if threshold reached (if ANY subset exceeded it)
            converged = between_chain_ess_value >= ess_threshold
            
            rows.append({
                "iteration": N,
                "between_chain_ess": between_chain_ess_value,
                "rhat": rhat,
                "converged": converged,
            })
            
            # Stop if threshold reached
            if converged:
                break
        
        except Exception as e:
            # Log detailed error information
            import traceback
            print(f"Error at iteration {N}: {type(e).__name__}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            rows.append({
                "iteration": N,
                "between_chain_ess": np.nan,
                "rhat": np.nan,
                "converged": False,
            })
    
    result_df = pd.DataFrame(rows, columns=["iteration", "between_chain_ess", "rhat", "converged"])
    return result_df


def run_convergence_analysis(
    output_base_dir,
    csv_path="data/processed/combined.csv",
    k_values=None,
    window=0,
    lag_window=10
):
    """
    Run convergence analysis on combined CSV data across multiple k-subset sizes.
    
    Args:
        csv_path: Path to combined CSV file with columns: chain, iter, log_p, dataset
        output_base_dir: Base directory for output folders (half_chains, three_fourth_chains, all_chains)
        k_values: List of k values to analyze. Default [2, 3, 4]
        window: Window size for sequential chain subsetting. If 0 (default), uses entire chains.
                If > 0, splits each chain into sequential windows of size `window` and analyzes
                them progressively until convergence is achieved.
    """
    if k_values is None:
        k_values = [2, 3, 4]
    
    k_to_folder = {
        2: "half_chains",
        3: "three_fourth_chains",
        4: "all_chains"
    }
    
    # Read combined.csv
    data = pd.read_csv(csv_path)
    
    # Check required columns
    required = {"chain", "iter", "log_p", "dataset"}
    if not required.issubset(data.columns):
        raise ValueError(f"Input CSV must contain columns: {required}")
    
    # Group by dataset
    for dataset_name, dataset_group in data.groupby("dataset"):
        # Get unique chain numbers and sort them
        chain_numbers = sorted(dataset_group["chain"].unique())
        
        # Create chains_list where index corresponds to chain number
        chains_list = []
        for chain_num in chain_numbers:
            chain_data = dataset_group[dataset_group["chain"] == chain_num]
            # Sort by iteration (smallest to largest) within each chain
            chain_data = chain_data.sort_values("iter")
            log_p_values = chain_data["log_p"].values
            chains_list.append(log_p_values)
        
        # Determine if using windowed analysis
        if window > 0:
            # Split each chain into sequential windows
            num_windows = [len(chain) // window for chain in chains_list]
            max_windows = max(num_windows)
            
            window_idx = 0
            converged = False
            
            # Storage for results from all windows for this dataset
            all_results_by_k = {k: [] for k in k_values}
            
            while window_idx < max_windows and not converged:
                # Extract window from each chain
                windowed_chains = []
                for chain in chains_list:
                    start_idx = window_idx * window
                    end_idx = min((window_idx + 1) * window, len(chain))
                    if start_idx < len(chain):
                        windowed_chains.append(chain[start_idx:end_idx])
                
                # Only proceed if we have valid windows
                if all(len(wc) > 0 for wc in windowed_chains):
                    # Run convergence analysis for each k value on this window
                    for k in k_values:
                        print(f"Analyzing convergence for dataset: {dataset_name}, k={k}, window={window_idx + 1}")
                        
                        try:
                            result_df = analyze_convergence(windowed_chains, k=k, lag_window=lag_window)
                            
                            # Add window column (1-indexed)
                            result_df["window"] = window_idx + 1
                            
                            # Store results
                            all_results_by_k[k].append(result_df)
                            
                            # Check if converged
                            if result_df["converged"].any():
                                converged = True
                                print(f"  Convergence achieved at window {window_idx + 1}")
                        except Exception as e:
                            import traceback
                            print(f"  Error analyzing {dataset_name} (k={k}, window={window_idx + 1}): {type(e).__name__}: {e}")
                            print(f"  Traceback: {traceback.format_exc()}")
                    
                    if converged:
                        print(f"  Convergence achieved for dataset {dataset_name}")
                        break
                
                window_idx += 1
            
            # Save combined results for each k value
            for k in k_values:
                if all_results_by_k[k]:  # Only save if there are results
                    # Combine all window results
                    combined_df = pd.concat(all_results_by_k[k], ignore_index=True)
                    
                    # Create output folder if it doesn't exist
                    output_folder = os.path.join(output_base_dir, k_to_folder[k])
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    
                    # Save output CSV
                    output_filename = f"{k}_chains_{dataset_name}.csv"
                    output_path = os.path.join(output_folder, output_filename)
                    combined_df.to_csv(output_path, index=False)
                    print(f"  Saved combined results to {output_path}")
        else:
            # Original behavior: analyze entire chains
            # Run convergence analysis for each k value
            for k in k_values:
                print(f"Analyzing convergence for dataset: {dataset_name}, k={k}")
                
                # Create output folder if it doesn't exist
                output_folder = os.path.join(output_base_dir, k_to_folder[k])
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                try:
                    result_df = analyze_convergence(chains_list, k=k)
                    
                    # Save output CSV
                    output_filename = f"{k}_chains_{dataset_name}.csv"
                    output_path = os.path.join(output_folder, output_filename)
                    result_df.to_csv(output_path, index=False)
                    print(f"  Saved to {output_path}")
                except Exception as e:
                    import traceback
                    print(f"  Error analyzing {dataset_name} (k={k}): {type(e).__name__}: {e}")
                    print(f"  Traceback: {traceback.format_exc()}")
    
    print("Convergence analysis complete.")


if __name__ == "__main__":
    # run_convergence_analysis(output_base_dir="output/entire_chain", lag_window=10)
    # run_convergence_analysis(window=20, output_base_dir="output/windows/size20", lag_window=2)
    run_convergence_analysis(window=50, output_base_dir="output/windows/size50", lag_window=2)
