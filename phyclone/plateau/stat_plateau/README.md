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
- `window` (int, default: `0`): Window size for sequential chain subsetting. If 0, uses entire chains. If > 0, splits chains into sequential windows and analyzes progressively
- `lag_window` (int, default: `10`): Lag window parameter for convergence analysis
- `convergence_threshold` (float, optional): Threshold for convergence detection. If not provided, defaults to `20 * k`

### Methods:
- `run_convergence_analysis()`: Runs convergence analysis on the CSV data and returns a boolean indicating whether convergence has been reached (True/False) 

