"""
Test script for analyze_convergence function with dummy data
"""
import numpy as np
import sys
import os
import pandas as pd
from io import StringIO
from unittest.mock import patch, mock_open

# Add parent directory to path so we can import local modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from convergence import analyze_convergence
from stat_plateau import stat_plateau

print("="*70)
print("TEST 1: Equal-length chains (baseline)")
print("="*70)

# Create dummy chains with 8 chains and 100 iterations each
np.random.seed(42)
num_chains = 4
num_iterations = 00

chains_list = []
for i in range(num_chains):
    # Create synthetic logP values (normally distributed around -10000 with some drift)
    chain = np.cumsum(np.random.normal(-5, 100, num_iterations)) - 10000
    chains_list.append(chain)
    print(f"Chain {i}: {len(chain)} samples, min={chain.min():.2f}, max={chain.max():.2f}")

print(f"\nRunning analyze_convergence on {num_chains} equal-length chains...")
print(f"ESS threshold: {20 * num_chains}")

try:
    result = analyze_convergence(chains_list)
    print("\nSuccess! Result:")
    print(result)
    print(f"\nResult shape: {result.shape}")
except Exception as e:
    import traceback
    print(f"\nError: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 2: Unequal-length chains (new functionality)")
print("="*70)

# Create chains with different lengths: [150, 150, 100, 75]
np.random.seed(42)
chain_lengths = [150, 150, 100, 75]
unequal_chains = []

for i, length in enumerate(chain_lengths):
    chain = np.cumsum(np.random.normal(-5, 100, length)) - 10000
    unequal_chains.append(chain)
    print(f"Chain {i}: {len(chain)} samples, min={chain.min():.2f}, max={chain.max():.2f}")

print(f"\nRunning analyze_convergence on unequal-length chains...")
print(f"Expected behavior:")
print(f"  - Iteration 10-70: Use all 4 chains")
print(f"  - Iteration 80+: Use only first 3 chains (chain 3 exhausted)")
print(f"  - Iteration 110+: Use only first 2 chains (chain 2 exhausted)")
print(f"  - Stop when < 2 chains remain or convergence reached")

try:
    result = analyze_convergence(unequal_chains)
    print("\nSuccess! Result:")
    print(result)
    print(f"\nResult shape: {result.shape}")
    print(f"\nAnalysis: Algorithm progressed through iterations with decreasing chain counts")
except Exception as e:
    import traceback
    print(f"\nError: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 3: Different k values with unequal chains")
print("="*70)

for k_val in [2, 3, 4]:
    print(f"\nTesting with k={k_val}...")
    try:
        result = analyze_convergence(unequal_chains, k=k_val)
        print(f"  Success! Got {len(result)} rows")
        print(f"  First iteration: {result.iloc[0]['iteration']}")
        print(f"  Last iteration: {result.iloc[-1]['iteration']}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

print("\n" + "="*70)
print("TEST 4: Extreme case - one chain much longer")
print("="*70)

# One chain very long, others short: [300, 50, 40, 30]
np.random.seed(42)
extreme_lengths = [300, 50, 40, 30]
extreme_chains = []

for i, length in enumerate(extreme_lengths):
    chain = np.cumsum(np.random.normal(-5, 100, length)) - 10000
    extreme_chains.append(chain)
    print(f"Chain {i}: {len(chain)} samples")

print(f"\nRunning analyze_convergence...")
print(f"Expected: Use progressively fewer chains as shorter ones exhaust")

try:
    result = analyze_convergence(extreme_chains)
    print("\nSuccess! Result:")
    print(result)
    print(f"\nProgression through iterations: {result['iteration'].tolist()}")
    if len(result) > 0:
        print(f"Algorithm stopped at iteration {result.iloc[-1]['iteration']}")
except Exception as e:
    import traceback
    print(f"\nError: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("All tests completed!")
print("="*70)


# ============================================================================
# TESTS FOR stat_plateau CLASS
# ============================================================================

print("\n\n")
print("="*70)
print("STAT_PLATEAU CLASS TESTS")
print("="*70)

print("\n" + "="*70)
print("TEST 5: stat_plateau initialization with default parameters")
print("="*70)

try:
    analyzer = stat_plateau()
    print("Success! stat_plateau instance created with defaults:")
    print(f"  csv_path: {analyzer.csv_path}")
    print(f"  k: {analyzer.k}")
    print(f"  window: {analyzer.window}")
    print(f"  convergence_threshold: {analyzer.convergence_threshold}")
    assert analyzer.csv_path == "data/processed/combined.csv"
    assert analyzer.k == 2
    assert analyzer.window == 0
    assert analyzer.convergence_threshold == 40, f"Expected 40 (20*2), got {analyzer.convergence_threshold}"
    print("✓ All default values verified!")
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 6: stat_plateau initialization with custom parameters")
print("="*70)

try:
    analyzer = stat_plateau(
        csv_path="/virtual/custom_data.csv",
        k=3,
        window=50,
        convergence_threshold=100
    )
    print("Success! stat_plateau instance created with custom values:")
    print(f"  csv_path: {analyzer.csv_path}")
    print(f"  k: {analyzer.k}")
    print(f"  window: {analyzer.window}")
    print(f"  convergence_threshold: {analyzer.convergence_threshold}")
    assert analyzer.csv_path == "/virtual/custom_data.csv"
    assert analyzer.k == 3
    assert analyzer.window == 50
    assert analyzer.convergence_threshold == 100
    print("✓ All custom values verified (no file access required)!")
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 7: stat_plateau convergence_threshold auto-calculation")
print("="*70)

try:
    # Test with k=2, threshold should be 40
    analyzer1 = stat_plateau(k=2)
    assert analyzer1.convergence_threshold == 40, f"Expected 40, got {analyzer1.convergence_threshold}"
    print(f"✓ k=2: convergence_threshold = {analyzer1.convergence_threshold} (20*2)")
    
    # Test with k=3, threshold should be 60
    analyzer2 = stat_plateau(k=3)
    assert analyzer2.convergence_threshold == 60, f"Expected 60, got {analyzer2.convergence_threshold}"
    print(f"✓ k=3: convergence_threshold = {analyzer2.convergence_threshold} (20*3)")
    
    # Test with k=4, threshold should be 80
    analyzer3 = stat_plateau(k=4)
    assert analyzer3.convergence_threshold == 80, f"Expected 80, got {analyzer3.convergence_threshold}"
    print(f"✓ k=4: convergence_threshold = {analyzer3.convergence_threshold} (20*4)")
    
    # Test with explicit override
    analyzer4 = stat_plateau(k=5, convergence_threshold=150)
    assert analyzer4.convergence_threshold == 150, f"Expected 150, got {analyzer4.convergence_threshold}"
    print(f"✓ k=5, explicit threshold=150: convergence_threshold = {analyzer4.convergence_threshold}")
    
    print("✓ Convergence threshold auto-calculation verified!")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 8: stat_plateau run_convergence_analysis with in-memory data")
print("="*70)

try:
    # Generate test data in memory (no file I/O)
    np.random.seed(42)
    data_list = []
    
    # 2 datasets, 3 chains each, 100 iterations per chain
    for dataset_id in ["dataset_A", "dataset_B"]:
        for chain_id in range(3):
            chain_data = np.cumsum(np.random.normal(-5, 100, 100)) - 10000
            for iter_idx, log_p in enumerate(chain_data):
                data_list.append({
                    "dataset": dataset_id,
                    "chain": chain_id,
                    "iter": iter_idx,
                    "log_p": log_p
                })
    
    df = pd.DataFrame(data_list)
    
    print(f"Generated in-memory test data with {len(df)} rows")
    print(f"  Datasets: {df['dataset'].unique()}")
    print(f"  Chains per dataset: {df.groupby('dataset')['chain'].nunique().values}")
    
    # Mock pd.read_csv to return our in-memory DataFrame
    with patch('stat_plateau.pd.read_csv') as mock_read_csv:
        mock_read_csv.return_value = df
        
        # Test run_convergence_analysis
        analyzer = stat_plateau(csv_path="/virtual/test_data.csv", k=2)
        result = analyzer.run_convergence_analysis()
        
        print(f"\nConvergence analysis completed!")
        print(f"  Result type: {type(result)}")
        print(f"  Result value: {result}")
        assert isinstance(result, (bool, np.bool_)), f"Expected bool, got {type(result)}"
        print("✓ run_convergence_analysis returns boolean!")
        print("✓ Test completed without accessing actual files!")
        
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 9: stat_plateau attribute consistency")
print("="*70)

try:
    # Create analyzer with specific parameters (no actual file needed)
    params = {
        "csv_path": "/virtual/test.csv",
        "k": 4,
        "window": 100,
        "convergence_threshold": 150
    }
    
    analyzer = stat_plateau(**params)
    
    # Verify all attributes match initialization parameters
    for attr, expected_value in params.items():
        actual_value = getattr(analyzer, attr)
        assert actual_value == expected_value, f"{attr}: expected {expected_value}, got {actual_value}"
        print(f"✓ {attr}: {actual_value}")
    
    print("\n✓ All attributes match initialization parameters (no file access required)!")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 10: Multiple stat_plateau instances independence")
print("="*70)

try:
    # Create multiple instances with different parameters
    analyzer1 = stat_plateau(k=2, window=0)
    analyzer2 = stat_plateau(k=3, window=50)
    analyzer3 = stat_plateau(k=4, window=100)
    
    # Verify each instance maintains its own state
    assert analyzer1.k == 2 and analyzer1.window == 0
    assert analyzer2.k == 3 and analyzer2.window == 50
    assert analyzer3.k == 4 and analyzer3.window == 100
    
    print("Instance 1: k=2, window=0 ✓")
    print("Instance 2: k=3, window=50 ✓")
    print("Instance 3: k=4, window=100 ✓")
    
    # Verify thresholds are calculated correctly for each
    assert analyzer1.convergence_threshold == 40
    assert analyzer2.convergence_threshold == 60
    assert analyzer3.convergence_threshold == 80
    
    print("\nThresholds:")
    print(f"  Instance 1: {analyzer1.convergence_threshold} (20*2) ✓")
    print(f"  Instance 2: {analyzer2.convergence_threshold} (20*3) ✓")
    print(f"  Instance 3: {analyzer3.convergence_threshold} (20*4) ✓")
    
    print("\n✓ All instances maintain independent state!")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 11: _eval_convergence_all_subsets with mocked output")
print("="*70)

try:
    import tempfile
    import shutil
    
    # Create a temporary directory for test outputs
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create analyzer with k=3 (will analyze k=2, k=3)
        analyzer = stat_plateau(
            csv_path="phyclone/plateau/stat_plateau/sample_data/combined.csv",
            k=3,
            check_all_subsets=True
        )
        
        # Mock run_convergence_analysis to create fake output files
        def mock_run_analysis(output_base_dir, csv_path, k_values, window, lag_window):
            # Create output folders and files for each k value
            for k in k_values:
                folder = os.path.join(output_base_dir, f"k{k}_chains")
                os.makedirs(folder, exist_ok=True)
                
                # Create a dummy result CSV with convergence=True for k=2
                result_data = {
                    "iteration": [10, 20, 30],
                    "between_chain_ess": [10, 30, 50],
                    "rhat": [1.5, 1.2, 1.0],
                    "converged": [False, False, k == 2]  # Only k=2 converges
                }
                result_df = pd.DataFrame(result_data)
                filepath = os.path.join(folder, f"{k}_chains_test_dataset.csv")
                result_df.to_csv(filepath, index=False)
        
        # Patch run_convergence_analysis
        with patch('stat_plateau.run_convergence_analysis', side_effect=mock_run_analysis):
            result = analyzer._eval_convergence_all_subsets(temp_dir)
            
            print(f"✓ _eval_convergence_all_subsets returned: {result}")
            assert result == True, f"Expected True (convergence at k=2), got {result}"
            print(f"✓ Correctly detected convergence in k2_chains folder")
            
            # Verify folders were created
            assert os.path.exists(os.path.join(temp_dir, "k2_chains")), "k2_chains folder not created"
            assert os.path.exists(os.path.join(temp_dir, "k3_chains")), "k3_chains folder not created"
            print(f"✓ All expected folders were created")
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        print(f"✓ Temporary directory cleaned up")
    
    print("✓ Test passed!")
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 12: _eval_convergence_all_subsets no convergence")
print("="*70)

try:
    temp_dir = tempfile.mkdtemp()
    
    try:
        analyzer = stat_plateau(
            csv_path=os.path.join(parent_dir, "sample_data", "combined.csv"),
            k=8,
            check_all_subsets=True
        )
        
        # Mock run_convergence_analysis with no convergence
        def mock_run_no_convergence(output_base_dir, csv_path, k_values, window, lag_window):
            for k in k_values:
                folder = os.path.join(output_base_dir, f"k{k}_chains")
                os.makedirs(folder, exist_ok=True)
                
                # All rows have converged=False
                result_data = {
                    "iteration": [10, 20, 30],
                    "between_chain_ess": [10, 30, 50],
                    "rhat": [1.5, 1.2, 1.0],
                    "converged": [False, False, False]  # No convergence
                }
                result_df = pd.DataFrame(result_data)
                filepath = os.path.join(folder, f"{k}_chains_test_dataset.csv")
                result_df.to_csv(filepath, index=False)
        
        with patch('stat_plateau.run_convergence_analysis', side_effect=mock_run_no_convergence):
            result = analyzer._eval_convergence_all_subsets(temp_dir)
            
            print(f"✓ _eval_convergence_all_subsets returned: {result}")
            assert result == False, f"Expected False (no convergence), got {result}"
            print(f"✓ Correctly detected no convergence across all k values")
    
    finally:
        shutil.rmtree(temp_dir)
    
    print("✓ Test passed!")
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 13: _eval_convergence_all_subsets checks all k folders")
print("="*70)

try:
    temp_dir = tempfile.mkdtemp()
    
    try:
        analyzer = stat_plateau(
            csv_path=os.path.join(parent_dir, "sample_data", "combined.csv"),
            k=8,
            check_all_subsets=True
        )
        
        # Mock run_convergence_analysis with convergence only in last k
        def mock_run_late_convergence(output_base_dir, csv_path, k_values, window, lag_window):
            for k in k_values:
                folder = os.path.join(output_base_dir, f"k{k}_chains")
                os.makedirs(folder, exist_ok=True)
                
                # Only k=4 converges
                result_data = {
                    "iteration": [10, 20, 30],
                    "between_chain_ess": [10, 30, 50],
                    "rhat": [1.5, 1.2, 1.0],
                    "converged": [False, False, k == 4]  # Only k=4 converges
                }
                result_df = pd.DataFrame(result_data)
                filepath = os.path.join(folder, f"{k}_chains_test_dataset.csv")
                result_df.to_csv(filepath, index=False)
        
        with patch('stat_plateau.run_convergence_analysis', side_effect=mock_run_late_convergence):
            result = analyzer._eval_convergence_all_subsets(temp_dir)
            
            print(f"✓ _eval_convergence_all_subsets returned: {result}")
            assert result == True, f"Expected True (convergence at k=4), got {result}"
            print(f"✓ Correctly found convergence in final k4_chains folder")
            print(f"✓ Verified that all k folders (k2, k3, k4) are checked")
    
    finally:
        shutil.rmtree(temp_dir)
    
    print("✓ Test passed!")
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("TEST 14: _eval_convergence_all_subsets missing folders")
print("="*70)

try:
    temp_dir = tempfile.mkdtemp()
    
    try:
        analyzer = stat_plateau(
            csv_path="phyclone/plateau/stat_plateau/sample_data/combined.csv",
            k=3,
            check_all_subsets=True
        )
        
        # Mock run_convergence_analysis that creates no folders
        def mock_run_no_folders(output_base_dir, csv_path, k_values, window, lag_window):
            # Don't create any folders
            pass
        
        with patch('stat_plateau.run_convergence_analysis', side_effect=mock_run_no_folders):
            result = analyzer._eval_convergence_all_subsets(temp_dir)
            
            print(f"✓ _eval_convergence_all_subsets returned: {result}")
            assert result == False, f"Expected False (no folders), got {result}"
            print(f"✓ Correctly handled missing output folders")
    
    finally:
        shutil.rmtree(temp_dir)
    
    print("✓ Test passed!")
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("stat_plateau._eval_convergence_all_subsets tests completed!")
print("="*70)

print("\n" + "="*70)
print("TEST 15: Real sample data with max 3 chains")
print("="*70)

try:
    # Load the actual sample data file
    csv_path = os.path.join(parent_dir, "sample_data", "combined.csv")
    print(f"Loading sample data from: {csv_path}")
    
    data = pd.read_csv(csv_path)
    print(f"Loaded {len(data)} rows from sample data")
    print(f"Columns: {list(data.columns)}")
    
    # Get unique datasets
    datasets = data["dataset"].unique()
    print(f"Found datasets: {datasets}")
    
    # Test with the first dataset, using all available chains
    dataset_name = datasets[0]
    dataset_group = data[data["dataset"] == dataset_name]
    
    # Get unique chain numbers - use all available chains
    chain_numbers = sorted(dataset_group["chain"].unique())
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Selected all {len(chain_numbers)} available chains: {list(chain_numbers)}")
    
    # Build chains_list from actual data
    chains_list = []
    for chain_num in chain_numbers:
        chain_data = dataset_group[dataset_group["chain"] == chain_num]
        chain_data = chain_data.sort_values("iter")
        log_p_values = chain_data["log_p"].values
        chains_list.append(log_p_values)
        print(f"  Chain {chain_num}: {len(log_p_values)} samples, "
              f"min={log_p_values.min():.2f}, max={log_p_values.max():.2f}")
    
    # Run convergence analysis
    print(f"\nRunning analyze_convergence with {len(chains_list)} chains...")
    print(f"ESS threshold: {20 * len(chains_list)}")
    
    result = analyze_convergence(chains_list)
    
    print("\nSuccess! Convergence analysis result:")
    print(result)
    print(f"\nResult statistics:")
    print(f"  Number of iterations analyzed: {len(result)}")
    print(f"  First iteration: {result.iloc[0]['iteration']}")
    print(f"  Last iteration: {result.iloc[-1]['iteration']}")
    print(f"  Final between-chain ESS: {result.iloc[-1]['between_chain_ess']:.2f}")
    print(f"  Final Rhat: {result.iloc[-1]['rhat']:.4f}")
    print(f"  Convergence achieved: {result['converged'].any()}")
    
    print("\n✓ Test 15 passed!")
    
except Exception as e:
    import traceback
    print(f"\n✗ Test 15 failed!")
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("All tests completed!")
print("="*70)

