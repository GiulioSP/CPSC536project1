# Convergence Detection - Statistical Methods

Folder containing logic for assessing statistical convergence. Used mainly to assess bulk-ESS and $\hat R$ metrics. Contains two main files: 

- `convergenceMetrics.py`: implementation of $\hat R$ and ESS calculations. Functions found in this file: 
  - `calculate_rhat`: calculates $\hat R$ given a list of chains
  - `PSRF_bulk_ESS`: calculates ESS of a given chain
  - `PSRF_between_chain_ESS`: calculates ESS across chains
- `convergence.py`: applies functions within `convergenceMetrics.py` onto log-likelihood chains
  - `analyze_convergence`: Calculates between-chain ESS and $\hat R$, saves as dataframe
  - `run_convergence_analysis`: Assesses convergence for a range of different K values. 

## Stat Plateau class
Main class to be used by PhyClone for assessing early stopping. 
- `csv_path` (str, default: `"data/processed/combined.csv"`): Path to combined CSV file containing chain data with columns: `chain`, `iter`, `log_p`, `dataset`
- `k` (int, default: `2`): Single k value for subset size analysis
- `check_all_subsets`: If True, uses run_convergence_analysis to check all k values up to k.
                       If False, only analyzes the specified k value using analyze_convergence.
- `window` (int, default: `0`): Window size for sequential chain subsetting. If 0, uses entire chains. If > 0, splits chains into sequential windows and analyzes progressively
- `lag_window` (int, default: `10`): Lag window parameter for convergence analysis
- `convergence_threshold` (integer, optional): Threshold for convergence detection. If not provided, defaults to `20 * k`

### Methods:
- `eval_convergence(output_base_dir="output")`: Main method to evaluate convergence. Routes to either single k or all subsets analysis based on `check_all_subsets` parameter. Returns a boolean indicating whether convergence has been reached (True/False)
- `_eval_convergence_single_k()`: Helper method for analyzing convergence with a single k value using `analyze_convergence()`. Returns True if convergence is reached, False otherwise.
- `_eval_convergence_all_subsets(output_base_dir)`: Helper method for analyzing convergence across all k values from 2 to k using `run_convergence_analysis()`. Creates output folders and CSV files for each k value, then checks all folders to determine if convergence was achieved. Returns True if convergence is reached at any k value, False otherwise. 

### Usage:

Declare StatPlateau Object: 

```
    analyzer = stat_plateau(
        csv_path=your_csv_path,
        k=your_num_chains,
        check_all_subsets=False,
        window=your_window_size,
        convergence_threshold=your_convergence_threshold
    )
```

Analyze convergence for a $k$ number of chains. For example: if number of chains is 8 but we only want to look at subset = 4, we make an analyzer with $k=4$ and `check_all_subsets=FALSE`:

```
    four_analyzer = stat_plateau(
      csv_path=your_csv_path,
      k=4,
      check_all_subsets=False,
      window=your_window_size,
      convergence_threshold=your_convergence_threshold
    )

    # check convergence
    four_analyzer.evaluate_convergence()
```

Alternatively, if we want to look at all subsets of size $\{2,3,\cdots,8\}$, we make the analyzer with $k=8$ and `check_all_subsets=TRUE`:

```
    all_analyzer = stat_plateau(
      csv_path=your_csv_path,
      k=8,
      check_all_subsets=TRUE,
      window=your_window_size,
      convergence_threshold=your_convergence_threshold
    )

    # check convergence
    all_analyzer.evaluate_convergence()
```

