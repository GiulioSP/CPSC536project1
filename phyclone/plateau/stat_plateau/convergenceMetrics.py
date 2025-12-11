# Script to calculate bulk ESS from CSV with logP values
# Assume that order of logP values corresponds to order of sampling
# INPUT: combined.csv file in data/processed directory
#        should have the columns: 'iter', 'log_P'
# OUTPUT: iteration number when ESS hits threshould 
import pandas as pd
import numpy as np
from scipy import special

# calculate rhat 
def calculate_rhat(chains_list):
    """
    Compute the rank-normalized Rhat convergence diagnostic following Vehtari et al. (2021).
    
    Rhat assesses whether multiple MCMC chains have converged to the same stationary distribution.
    The improved rank-normalized version is more sensitive to non-stationarity and chain-mixing issues.
    
    The method involves:
    1. Rank-normalize each chain independently
    2. Fold the ranks around the midpoint
    3. Convert to z-scores using inverse normal CDF
    4. Compute between-chain variance (B) and within-chain variance (W)
    5. Compute Rhat = sqrt((N-1)/N + (M+1)/(M*N) * B/W)
    
    Rhat values close to 1.0 indicate good convergence (typically Rhat < 1.01).
    Values substantially above 1.0 (e.g., > 1.1) suggest chains have not converged.
    
    References:
    - Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021). 
      "Rank-normalization, folding, and localization: An improved Rhat for assessing 
      convergence of MCMC". Bayesian Analysis 16(2):667-718.
    
    Args:
      chains_list: list[array-like] of sample sequences, one array per chain.
                   Each element should be array-like (list, np.ndarray, pd.Series, etc.)
                   containing float values. All chains should have the same length N.
                   Example: [np.array([...]), np.array([...]), np.array([...])]
    
    Returns:
      float rhat: The rank-normalized Rhat statistic. Values near 1.0 indicate convergence.
    """
    # Convert all chains to numpy arrays
    chains = [np.asarray(chain, dtype=float) for chain in chains_list]
    M = len(chains)  # number of chains
    
    if M < 2:
        raise ValueError("Need at least 2 chains to compute Rhat")
    
    # All chains should have the same length
    N = len(chains[0])
    if any(len(chain) != N for chain in chains):
        raise ValueError("All chains must have the same length for Rhat computation")
    
    if N < 2:
        raise ValueError("Each chain must have at least 2 samples")
    
    # Step 1-3: Rank-normalize and transform each chain
    z_chains = []
    for chain in chains:
        # Rank-normalize within this chain (1-indexed)
        ranks = np.argsort(np.argsort(chain)) + 1
        
        # Fold ranks around midpoint
        z_folded = np.abs(ranks - (N + 1) / 2.0) / (N + 1)
        
        # Convert to z-scores using inverse normal CDF
        # Clip to ensure input to erfinv is in (-1, 1)
        z_input = np.clip(2 * z_folded - 1, -0.9999, 0.9999)
        z_scores = np.sqrt(2) * special.erfinv(z_input)
        z_chains.append(z_scores)
    
    z_chains = np.asarray(z_chains, dtype=float)  # shape: (M, N)
    
    # Step 4: Compute within-chain and between-chain variances
    # Within-chain variance: average variance of each chain
    chain_means = np.mean(z_chains, axis=1)  # shape: (M,)
    chain_vars = np.var(z_chains, axis=1, ddof=1)  # unbiased variance, shape: (M,)
    W = np.mean(chain_vars)  # within-chain variance
    
    # Between-chain variance: variance of chain means
    global_mean = np.mean(z_chains)
    # B = N / (M - 1) * np.sum((chain_means - global_mean) ** 2)
    # terms simplified into var_hat calculation, but full equation shown here for clarity
    
    # Step 5: Compute Rhat
    # Rhat = sqrt(Var_hat / W) where Var_hat = (N-1)/N * W + 1/N * B
    if W == 0:
        # All chains are constant; they have converged trivially
        return 1.0
    
    # var_hat = (N - 1) / N * W + 1.0 / N * B
    var_hat = (N-1)*W/N + np.sum((chain_means - global_mean) ** 2)/(M - 1)
    rhat = np.sqrt(var_hat / W)
    
    return rhat


# calculate bulk ESS using rank-normalized method
def PSRF_bulk_ESS(logP_values, threshold=400):
    """
    Walk through prefixes of logP_values and return the prefix length (iteration)
    at which the bulk ESS first reaches or exceeds `threshold`.

    Uses the rank-normalized approach following Vehtari et al. (2021) to compute 
    effective sample size (ESS).

    Returns:
      int iteration (number of samples) when ESS >= threshold, or
      (max_iter, max_ess) tuple if threshold is never reached, where max_iter is
      the iteration with the maximum ESS observed and max_ess is that ESS value.
    """
    x_all = np.asarray(logP_values, dtype=float)
    n_total = x_all.size
    if n_total == 0:
        return None

    max_ess = -np.inf
    max_iter = 0

    for N in range(2, n_total + 1):
        x = x_all[:N]

        # Step 1: Convert to ranks (1-indexed)
        ranks = np.argsort(np.argsort(x)) + 1

        # Step 2: Fold ranks around midpoint to [0, 0.5]
        z_folded = np.abs(ranks - (N + 1) / 2.0) / (N + 1)

        # Step 3: Convert folded ranks to standard normal z-scores (quantile transform)
        # Using inverse normal CDF on folded ranks via scipy.special.erfinv
        # Clip to ensure input to erfinv is in (-1, 1)
        z_input = np.clip(2 * z_folded - 1, -0.9999, 0.9999)
        z_scores = np.sqrt(2) * special.erfinv(z_input)
        z_scores = z_scores - np.mean(z_scores)

        # Step 4: Compute autocovariance on z-scored ranked data
        acov_full = np.correlate(z_scores, z_scores, mode='full')
        acov = acov_full[N - 1:]  # lags 0..N-1

        acov0 = acov[0]
        if acov0 == 0 or not np.isfinite(acov0):
            ess = float(N)
        else:
            rho = acov / acov0

            # Step 5: Compute ESS from autocorrelation:
            # ESS = N / (1 + 2 * sum of positive autocorrelations up to first negative or N/2)
            max_lag = min(N // 2, len(rho) - 1)
            acf_sum = 0.0
            for lag in range(1, max_lag + 1):
                if rho[lag] > 0:
                    acf_sum += rho[lag]
                else:
                    break

            tau_hat = 1.0 + 2.0 * acf_sum
            if tau_hat < 1.0:
                tau_hat = 1.0
            ess = N / tau_hat
        
        # update current iteration index each loop
        max_iter = N
        # update max ESS if current ESS is larger
        if ess > max_ess:
            max_ess = ess

        # return early if threshold reached
        if ess >= threshold:
            return (max_iter, max_ess)

    # threshold never reached: return iteration and value of max ESS observed
    return (max_iter, max_ess)


def PSRF_between_chain_ESS(chains_list, threshold=400):
    """
    Compute between-chain bulk ESS following Vehtari et al. (2021).
    
    Walk through prefixes of chain samples and return the prefix length (iteration)
    at which the between-chain bulk ESS first reaches or exceeds `threshold`.
    
    Uses the rank-normalized approach on pooled samples across chains to compute 
    effective sample size (ESS). This diagnostic is important for assessing whether
    multiple chains have mixed well and reached the same stationary distribution.
    
    The method involves:
    1. For each iteration N:
       a. Extract first N samples from each chain
       b. Rank-normalize each chain independently
       c. Fold ranks and convert to z-scores
       d. Pool all z-scored ranked data across chains
    2. Compute autocorrelation on the pooled data
    3. Compute between-chain ESS as: ESS = (M*N) / (1 + 2 * sum of positive autocorrelations)
       where M is the number of chains and N is samples per chain
    
    This approach is robust to non-stationarity and assesses convergence across chains.
    
    References:
    - Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021). 
      "Rank-normalization, folding, and localization: An improved Rhat for assessing 
      convergence of MCMC". Bayesian Analysis 16(2):667-718.
    
    Args:
      chains_list: list[array-like] of logP value sequences, one array per chain.
                   Each element should be array-like (list, np.ndarray, pd.Series, etc.)
                   containing float logP values. All chains can have different lengths;
                   the function will use the shortest chain length.
                   Example: [np.array([...]), np.array([...]), np.array([...])]
      threshold: ESS threshold to reach (default 400, following MCMC diagnostics guidelines).
    
    Returns:
      int iteration (number of samples per chain) when ESS >= threshold, or
      (max_iter, max_ess) tuple if threshold is never reached, where max_iter is
      the iteration with the maximum ESS observed and max_ess is that ESS value.
    """
    # Convert all chains to numpy arrays and find minimum length
    chains = [np.asarray(chain, dtype=float) for chain in chains_list]
    M = len(chains)  # number of chains
    
    if M == 0:
        return None
    
    max_chain_length = min(len(chain) for chain in chains)
    if max_chain_length == 0:
        return None
    
    max_ess = -np.inf
    max_iter = 0
    
    for N in range(2, max_chain_length + 1):
        # Extract first N samples from each chain and process
        pooled_z_scores = []
        
        for chain in chains:
            x = chain[:N]
            
            # Step 1: Convert to ranks (1-indexed, within-chain)
            ranks = np.argsort(np.argsort(x)) + 1
            
            # Step 2: Fold ranks around midpoint to [0, 0.5]
            z_folded = np.abs(ranks - (N + 1) / 2.0) / (N + 1)
            
            # Step 3: Convert folded ranks to standard normal z-scores
            # Clip to ensure input to erfinv is in (-1, 1)
            z_input = np.clip(2 * z_folded - 1, -0.9999, 0.9999)
            z_scores = np.sqrt(2) * special.erfinv(z_input)
            z_scores = z_scores - np.mean(z_scores)
            
            pooled_z_scores.extend(z_scores)
        
        # Step 4: Convert to numpy array for autocovariance computation
        pooled_data = np.asarray(pooled_z_scores, dtype=float)
        total_samples = M * N
        
        # Step 5: Compute autocovariance on pooled data
        acov_full = np.correlate(pooled_data, pooled_data, mode='full')
        acov = acov_full[total_samples - 1:]  # lags 0..total_samples-1
        
        acov0 = acov[0]
        if acov0 == 0 or not np.isfinite(acov0):
            ess = float(total_samples)
        else:
            rho = acov / acov0
            
            # Step 6: Compute ESS from autocorrelation
            # ESS = (M*N) / (1 + 2 * sum of positive autocorrelations)
            max_lag = min(total_samples // 2, len(rho) - 1)
            acf_sum = 0.0
            for lag in range(1, max_lag + 1):
                if rho[lag] > 0:
                    acf_sum += rho[lag]
                else:
                    break
            
            tau_hat = 1.0 + 2.0 * acf_sum
            if tau_hat < 1.0:
                tau_hat = 1.0
                
            ess = total_samples / tau_hat
        
        # Update max ESS tracking
        max_iter = N
        if ess > max_ess:
            max_ess = ess
        
        # Return early if threshold reached
        if ess >= threshold:
            return N
    
    # Threshold never reached: return iteration and value of max ESS observed
    return (max_iter, max_ess)
