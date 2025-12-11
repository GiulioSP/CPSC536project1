import numpy as np
import pandas as pd
from convergence import analyze_convergence


class stat_plateau:
    def __init__(
        self,
        csv_path="data/processed/combined.csv",
        k=2,
        window=0,
        lag_window=10,
        convergence_threshold=None
    ):
        """
        Initialize stat_plateau object with default parameters.
        
        Args:
            csv_path: Path to combined CSV file with columns: chain, iter, log_p, dataset
            k: Single k value to analyze for subset sizes
            window: Window size for sequential chain subsetting. If 0 (default), uses entire chains.
                    If > 0, splits each chain into sequential windows of size `window` and analyzes
                    them progressively until convergence is achieved.
            lag_window: Lag window parameter for convergence analysis
            convergence_threshold: Threshold for convergence detection. Default is 20*k if not provided.
        """
        self.csv_path = csv_path
        self.k = k
        self.window = window
        self.lag_window = lag_window
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else 20 * k
    
    def run_convergence_analysis(self):
        """
        Run convergence analysis on combined CSV data for a single k-subset size.
        
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
                            result_df = analyze_convergence(windowed_chains, k=self.k, lag_window=self.lag_window)
                            
                            # Check if converged using the threshold
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
                    result_df = analyze_convergence(chains_list, k=self.k, lag_window=self.lag_window)
                    
                    # Check if converged using the threshold
                    if result_df["converged"].any():
                        overall_converged = True
                        print(f"  Convergence achieved for dataset {dataset_name}")
                except Exception as e:
                    import traceback
                    print(f"  Error analyzing {dataset_name} (k={self.k}): {type(e).__name__}: {e}")
                    print(f"  Traceback: {traceback.format_exc()}")
        
        return overall_converged

