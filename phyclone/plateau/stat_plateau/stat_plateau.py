import os
import numpy as np
import pandas as pd
from convergence import analyze_convergence, run_convergence_analysis


class stat_plateau:
    def __init__(
        self,
        csv_path="phyclone/plateau/stat_plateau/sample_data/combined.csv",
        k=2,
        check_all_subsets=False,
        window=0,
        convergence_threshold=None
    ):
        """
        Initialize stat_plateau object with default parameters.
        
        Args:
            csv_path: Path to combined CSV file with columns: chain, iter, log_p, dataset
            k: Single k value to analyze for subset sizes
            check_all_subsets: If True, uses run_convergence_analysis to check all k values up to k.
                               If False, only analyzes the specified k value using analyze_convergence.
            window: Window size for sequential chain subsetting. If 0 (default), uses entire chains with lag_window=10.
                    If > 0, splits each chain into sequential windows of size `window` and uses that as the lag_window parameter.
            convergence_threshold: Threshold for convergence detection. Default is 20*k if not provided.
        """
        self.csv_path = csv_path
        self.k = k
        self.check_all_subsets = check_all_subsets
        self.window = window
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else 20 * k
    
    def eval_convergence(self, output_base_dir="output"):
        """
        Evaluate convergence for this stat_plateau configuration.
        
        If check_all_subsets=False, analyzes only the specified k value using analyze_convergence.
        If check_all_subsets=True, uses run_convergence_analysis to analyze all k values up to k.
        
        Args:
            output_base_dir: Base directory where output folders will be created (only used if check_all_subsets=True)
        
        Returns:
            bool: True if convergence has been reached, False otherwise
        """
        if not self.check_all_subsets:
            # Use analyze_convergence for single k value analysis
            return self._eval_convergence_single_k()
        else:
            # Use run_convergence_analysis for multiple k values
            return self._eval_convergence_all_subsets(output_base_dir)
    
    def _eval_convergence_single_k(self):
        """
        Analyze convergence for a single k value using analyze_convergence.
        
        Returns:
            bool: True if convergence has been reached, False otherwise
        """
        # Read combined.csv
        data = pd.read_csv(self.csv_path)
        
        # Check required columns
        required = {"chain", "iter", "log_p", "dataset"}
        if not required.issubset(data.columns):
            raise ValueError(f"Input CSV must contain columns: {required}")
        
        # Track overall convergence across all datasets
        overall_converged = False
        
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
            if self.window > 0:
                # Split each chain into sequential windows
                num_windows = [len(chain) // self.window for chain in chains_list]
                max_windows = max(num_windows)
                
                window_idx = 0
                
                while window_idx < max_windows:
                    # Extract window from each chain
                    windowed_chains = []
                    for chain in chains_list:
                        start_idx = window_idx * self.window
                        end_idx = min((window_idx + 1) * self.window, len(chain))
                        if start_idx < len(chain):
                            windowed_chains.append(chain[start_idx:end_idx])
                    
                    # Only proceed if we have valid windows
                    if all(len(wc) > 0 for wc in windowed_chains):
                        try:
                            # Use window as lag_window when window > 0, else use default 10
                            lag_window_val = self.window if self.window > 0 else 10
                            result_df = analyze_convergence(windowed_chains, k=self.k, lag_window=lag_window_val)
                            
                            # Check if converged
                            if result_df["converged"].any():
                                overall_converged = True
                                print(f"  Convergence achieved for dataset {dataset_name} at window {window_idx + 1}")
                                break
                        except Exception as e:
                            import traceback
                            print(f"  Error analyzing {dataset_name} (k={self.k}, window={window_idx + 1}): {type(e).__name__}: {e}")
                            print(f"  Traceback: {traceback.format_exc()}")
                    
                    window_idx += 1
            else:
                # Analyze entire chains
                try:
                    # Use window as lag_window when window > 0, else use default 10
                    lag_window_val = self.window if self.window > 0 else 10
                    result_df = analyze_convergence(chains_list, k=self.k, lag_window=lag_window_val)
                    
                    # Check if converged
                    if result_df["converged"].any():
                        overall_converged = True
                        print(f"  Convergence achieved for dataset {dataset_name}")
                except Exception as e:
                    import traceback
                    print(f"  Error analyzing {dataset_name} (k={self.k}): {type(e).__name__}: {e}")
                    print(f"  Traceback: {traceback.format_exc()}")
        
        return overall_converged
    
    def _eval_convergence_all_subsets(self, output_base_dir):
        """
        Analyze convergence for multiple k values using run_convergence_analysis.
        
        Analyzes all k values from 2 to self.k (inclusive).
        
        Args:
            output_base_dir: Base directory where output folders will be created
        
        Returns:
            bool: True if convergence has been reached, False otherwise
        """
        # Generate list of k values from 2 to self.k
        k_values = list(range(2, self.k + 1))
        
        # Use window as lag_window when window > 0, else use default 10
        lag_window_val = self.window if self.window > 0 else 10
        
        # Call run_convergence_analysis with all k values
        run_convergence_analysis(
            output_base_dir=output_base_dir,
            csv_path=self.csv_path,
            k_values=k_values,
            window=self.window,
            lag_window=lag_window_val
        )
        
        # Analyze the results to determine if convergence was achieved
        # Check if any output files were created and contain convergence=True
        converged = False
        
        # Check all k folders that were created (k2_chains, k3_chains, ..., k{self.k}_chains)
        for k in range(2, self.k + 1):
            output_folder = os.path.join(output_base_dir, f"k{k}_chains")
            if not os.path.exists(output_folder):
                continue
            
            for filename in os.listdir(output_folder):
                if filename.endswith(".csv"):
                    filepath = os.path.join(output_folder, filename)
                    try:
                        result_df = pd.read_csv(filepath)
                        if result_df["converged"].any():
                            converged = True
                            print(f"  Convergence achieved in {filepath} for k={k}")
                            break
                    except Exception as e:
                        print(f"Warning: Could not read {filepath}: {e}")
            
            if converged:
                break
        
        return converged

