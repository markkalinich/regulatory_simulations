#!/usr/bin/env python3
"""
Shared statistical utility functions.

This module provides reusable statistical methods for confidence intervals,
error estimation, and other common statistical calculations used throughout
the codebase.
"""

from scipy import stats
import numpy as np
from typing import Tuple, Optional


def clopper_pearson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute Clopper-Pearson exact binomial confidence interval.
    
    Args:
        successes: Number of successes (int)
        n: Total number of trials (int)
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        tuple: (lower_bound, upper_bound) as proportions in [0, 1]
        
    Example:
        >>> lower, upper = clopper_pearson_ci(95, 100)  # 95% success rate
        >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
        95% CI: [0.887, 0.983]
    """
    if n == 0:
        return (0.0, 1.0)
    
    # Lower bound: Beta quantile (or 0 if no successes)
    if successes == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha/2, successes, n - successes + 1)
    
    # Upper bound: Beta quantile (or 1 if all successes)
    if successes == n:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha/2, successes + 1, n - successes)
    
    return (lower, upper)


def bootstrap_f1_ci(
    tp: int, fp: int, fn: int, tn: int,
    n_bootstrap: int = 10000, 
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute non-parametric bootstrap confidence interval for F1 score.
    
    F1 = 2*TP / (2*TP + FP + FN)
    This uses non-parametric bootstrap by resampling the observed outcomes
    with replacement and recomputing F1 for each bootstrap sample. 
    This is equivalent to bootstrapping the dataset rows (given only the aggregated counts).
    
    Args:
        tp: True positives
        fp: False positives  
        fn: False negatives
        tn: True negatives
        n_bootstrap: Number of bootstrap samples (default 10000)
        alpha: Significance level (default 0.05 for 95% CI)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (lower_bound, upper_bound) F1 scores, or (NaN, NaN) if N=0
    """
    # Total samples (full confusion matrix)
    n_total = tp + fp + fn + tn
    if n_total == 0:
        return (float('nan'), float('nan'))
    
    # Initialize RNG
    rng = np.random.default_rng(random_state)
    
    # Reconstruct observed outcomes: 0=TP, 1=FP, 2=FN, 3=TN
    observations = np.array([0]*tp + [1]*fp + [2]*fn + [3]*tn)
    
    f1_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample observations with replacement
        resampled = rng.choice(observations, size=n_total, replace=True)
        
        # Count outcomes in resampled data
        tp_boot = np.sum(resampled == 0)
        fp_boot = np.sum(resampled == 1)
        fn_boot = np.sum(resampled == 2)
        
        # Calculate F1
        denom = 2 * tp_boot + fp_boot + fn_boot
        f1_samples[i] = (2 * tp_boot / denom) if denom > 0 else 0.0
    
    lower = np.percentile(f1_samples, 100 * alpha/2)
    upper = np.percentile(f1_samples, 100 * (1 - alpha/2))
    
    return (lower, upper)


def calculate_lod(n_samples: int, ci_level: float = 0.50) -> float:
    """
    Calculate Limit of Detection (LOD) for false negative rate.
    
    Given 0 failures observed in n_samples tests, returns the upper bound
    on FNR at the specified confidence level using Clopper-Pearson exact method.
    
    This prevents models with 100% observed sensitivity from appearing to have
    zero risk, while acknowledging the statistical limit of what we can detect.
    
    Args:
        n_samples: Number of test samples
        ci_level: Confidence level (0.50 = median, 0.95 = conservative)
        
    Returns:
        float: Upper bound on FNR (limit of detection)
        
    Example:
        >>> lod = calculate_lod(450, ci_level=0.50)
        >>> print(f"LOD for n=450: {lod:.6f}")
        LOD for n=450: 0.001539
    """
    # For 0 observed failures, lower bound is always 0
    # Upper bound at ci_level gives us the LOD
    return stats.beta.ppf(ci_level, 1, n_samples)

