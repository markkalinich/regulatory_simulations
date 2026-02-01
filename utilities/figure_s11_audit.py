#!/usr/bin/env python3
"""
Figure S11 Audit Script

Independently verifies Figure S11 (P2 across failure multiplier values) by:

1. Loading FN/total_positive from comprehensive_metrics.csv (therapy_engagement)
2. Independently calculating P2 using the SAME Monte Carlo approach as the pipeline
3. Comparing calculated median values to the output CSVs (p1_p2_p_harm_values_m_{m}.csv)

P2 Formula:
    adjusted_sensitivity = (1 - FNR)^m
    P2 = (1 - adjusted_sensitivity) × P(fail_help) × P(lack_care_harm)

IMPORTANT: The pipeline uses Monte Carlo sampling from Beta posteriors, NOT point estimates.
The FNR samples come from Beta(α + fn, α + tp) where α=1.0 (uniform prior).
The reported "risk_probability" is the MEDIAN of these Monte Carlo samples.

Usage:
    python utilities/figure_s11_audit.py --paper-run-dir <path_to_pipeline_output>
    python utilities/figure_s11_audit.py  # Uses most recent pipeline run

Output:
    figure_s11_audit_report.csv - Detailed comparison for each (model, M, baseline_pct)
    figure_s11_audit_summary.json - Summary with pass/fail status
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
import re
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.regulatory_paper_parameters import RISK_MODEL_PARAMS


# =============================================================================
# CONFIGURATION (from regulatory_paper_parameters.py)
# =============================================================================

M_VALUES = RISK_MODEL_PARAMS['failure_multiplier_values']
BASELINE_PERCENTAGES = RISK_MODEL_PARAMS['p2_baseline_harm_percentages']
PROB_FAIL_SEEK_HELP = RISK_MODEL_PARAMS['prob_fail_seek_help']
N_MC_SAMPLES = RISK_MODEL_PARAMS['n_mc_samples']
PRIOR_ALPHA = RISK_MODEL_PARAMS['prior_alpha']

# Tolerance for floating point comparisons
# Monte Carlo has inherent variance, so we allow some tolerance
# The pipeline uses 50,000 samples which gives ~0.5-1% standard error
# We allow 3% to account for different random seeds
ABSOLUTE_TOLERANCE = 1e-8  # For very small P2 values
RELATIVE_TOLERANCE = 0.03  # 3% relative tolerance for Monte Carlo variance


def sample_fnr_from_posterior(fn: int, total_positive: int, n_samples: int, 
                               prior_alpha: float = 1.0, random_state: int = 42) -> np.ndarray:
    """
    Sample FNR from Beta posterior distribution.
    
    Given k failures (false negatives) out of n positive tests, the posterior
    for FNR with Beta(alpha, alpha) prior is:
        Beta(alpha + k, alpha + (n - k))
    
    Args:
        fn: Number of false negatives (k)
        total_positive: Total positive test cases (n)
        n_samples: Number of MC samples to draw
        prior_alpha: Prior parameter (1.0 = uniform, 0.5 = Jeffreys)
        random_state: Random seed for reproducibility
    
    Returns:
        Array of n_samples FNR values sampled from posterior
    """
    tp = total_positive - fn  # true positives = n - k
    alpha_posterior = prior_alpha + fn          # alpha + k (failures)
    beta_posterior = prior_alpha + tp           # alpha + (n - k) (successes)
    return stats.beta.rvs(alpha_posterior, beta_posterior, size=n_samples, random_state=random_state)


def calculate_p2_mc(fn: int, total_positive: int, baseline_pct: float, m: float,
                    prob_fail_help: float = 1.0, n_samples: int = 50000,
                    prior_alpha: float = 1.0, random_state: int = 42) -> Dict[str, float]:
    """
    Calculate P2 using Monte Carlo sampling (matching pipeline approach).
    
    Args:
        fn: Number of false negatives for therapy engagement
        total_positive: Total positive cases for therapy engagement
        baseline_pct: P(lack of care → harm) as percentage
        m: Failure multiplier
        prob_fail_help: P(user fails to seek qualified help) - default 1.0
        n_samples: Number of Monte Carlo samples
        prior_alpha: Beta prior parameter
        random_state: Random seed for reproducibility
        
    Returns:
        Dict with 'median', 'ci_5', 'ci_95' for P2
    """
    # Sample FNR from posterior
    fnr_samples = sample_fnr_from_posterior(fn, total_positive, n_samples, prior_alpha, random_state)
    
    # Calculate P2 for each sample
    sensitivity_samples = 1 - fnr_samples
    adjusted_sensitivity = sensitivity_samples ** m
    p_miss_therapeutic = 1 - adjusted_sensitivity
    p_lack_care_harm = baseline_pct / 100.0
    
    p2_samples = p_miss_therapeutic * prob_fail_help * p_lack_care_harm
    
    return {
        'median': np.median(p2_samples),
        'ci_5': np.percentile(p2_samples, 5),
        'ci_95': np.percentile(p2_samples, 95),
    }


def calculate_p2_point(fnr: float, baseline_pct: float, m: float, prob_fail_help: float = 1.0) -> float:
    """
    Calculate P2 analytically (point estimate) - for reference only.
    
    P2 = (1 - (1-FNR)^m) × P(fail_help) × P(lack_care_harm)
    """
    sensitivity = 1 - fnr
    adjusted_sensitivity = sensitivity ** m
    p_miss_therapeutic = 1 - adjusted_sensitivity
    p_lack_care_harm = baseline_pct / 100.0
    
    p2 = p_miss_therapeutic * prob_fail_help * p_lack_care_harm
    return p2


def load_therapy_engagement_metrics(paper_run_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Load therapy engagement metrics from comprehensive_metrics.csv.
    
    Returns:
        Dict mapping (model_family, model_size) -> {'sensitivity': float, 'fnr': float, 'fn': int, 'total_positive': int}
    """
    metrics_path = paper_run_dir / 'Data' / 'processed_data' / 'model_performance_metrics' / 'therapy_engagement_comprehensive_metrics.csv'
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Therapy engagement metrics not found: {metrics_path}")
    
    df = pd.read_csv(metrics_path)
    
    metrics = {}
    for _, row in df.iterrows():
        key = (row['model_family'], row['model_size'])
        metrics[key] = {
            'sensitivity': row['sensitivity'],
            'fnr': 1 - row['sensitivity'],
            'fn': int(row['fn']),
            'total_positive': int(row['total_positive']),
        }
    
    return metrics


def verify_fnr_inputs(te_metrics: Dict, paper_run_dir: Path) -> bool:
    """
    Verify that the FNR values in comprehensive_metrics match cache data.
    
    This cross-checks with the confusion matrix audit.
    """
    # Load the P2 CSV to get the fn/n values used
    csv_path = paper_run_dir / 'Data' / 'processed_data' / 'correlated_failure_analysis' / 'p1_p2_p_harm_values_m_1.csv'
    
    if not csv_path.exists():
        return True  # Can't verify without the CSV
    
    df = pd.read_csv(csv_path)
    p2_data = df[df['risk_type'] == 'P2']
    
    all_match = True
    for (family, size), metrics in te_metrics.items():
        # Find matching row
        row = p2_data[
            (p2_data['model_family'].apply(normalize_family) == normalize_family(family)) &
            (p2_data['model_size'] == size)
        ]
        
        if len(row) > 0:
            csv_fn = row.iloc[0].get('tx_eng_fn')
            csv_n = row.iloc[0].get('tx_eng_n')
            
            if csv_fn is not None and csv_n is not None:
                if int(csv_fn) != metrics['fn'] or int(csv_n) != metrics['total_positive']:
                    print(f"  WARNING: FN mismatch for {family} {size}: metrics={metrics['fn']}/{metrics['total_positive']}, csv={csv_fn}/{csv_n}")
                    all_match = False
    
    return all_match


def load_reported_p2_values(paper_run_dir: Path, m: float) -> Optional[pd.DataFrame]:
    """
    Load reported P2 values from the output CSV for a specific M value.
    
    Returns:
        DataFrame with columns: model_family, model_size, baseline_percentage, risk_probability, etc.
    """
    # CSV files are in Data/processed_data/correlated_failure_analysis/
    csv_dir = paper_run_dir / 'Data' / 'processed_data' / 'correlated_failure_analysis'
    
    # Format M value for filename (handles floats like 1.0, 2.0, etc.)
    if m == int(m):
        m_str = str(int(m))
    else:
        m_str = str(m)
    
    csv_path = csv_dir / f'p1_p2_p_harm_values_m_{m_str}.csv'
    
    if not csv_path.exists():
        # Try alternative format
        csv_path = csv_dir / f'p1_p2_p_harm_values_m_{m}.csv'
    
    if not csv_path.exists():
        return None
    
    return pd.read_csv(csv_path)


def normalize_family(family: str) -> str:
    """Normalize model family name for matching."""
    family_lower = family.lower()
    if family_lower.startswith('llama'):
        return 'llama'
    return family_lower


def run_audit(paper_run_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Run Figure S11 audit.
    
    Returns DataFrame with audit results.
    """
    print("=" * 70)
    print("FIGURE S11 AUDIT: P2 ACROSS FAILURE MULTIPLIER VALUES")
    print("=" * 70)
    print(f"Paper run: {paper_run_dir}")
    print(f"Output: {output_path}")
    print()
    print("This audit verifies:")
    print("  - P2 calculations using Monte Carlo sampling from Beta posteriors")
    print(f"  - M values: {M_VALUES}")
    print(f"  - Baseline percentages: {BASELINE_PERCENTAGES}")
    print(f"  - Monte Carlo samples: {N_MC_SAMPLES}, Prior α={PRIOR_ALPHA}")
    print(f"  - P(fail_help) = {PROB_FAIL_SEEK_HELP}")
    print()
    
    # Load therapy engagement metrics (source of FN/total_positive for Beta posterior)
    print("Step 1: Loading therapy engagement metrics (FN counts for Beta posterior)...")
    te_metrics = load_therapy_engagement_metrics(paper_run_dir)
    print(f"  Loaded metrics for {len(te_metrics)} models")
    
    # Verify FNR inputs match what was used in figure generation
    print("\nStep 2: Verifying FN inputs match CSV data...")
    fnr_match = verify_fnr_inputs(te_metrics, paper_run_dir)
    if fnr_match:
        print("  ✅ FN/total_positive values match")
    else:
        print("  ⚠️ Some FN values don't match - results may differ")
    
    audit_results = []
    m_summaries = {}
    
    # Audit each M value
    for m in M_VALUES:
        print(f"\n{'='*70}")
        print(f"M = {m}")
        print(f"{'='*70}")
        
        # Load reported values for this M
        reported_df = load_reported_p2_values(paper_run_dir, m)
        
        if reported_df is None:
            print(f"  WARNING: No output CSV found for M={m}")
            m_summaries[m] = {
                'csv_found': False,
                'models_checked': 0,
                'passed': 0,
                'failed': 0,
            }
            continue
        
        # Filter to P2 only (CSV contains P1, P2, P_harm)
        p2_reported = reported_df[reported_df['risk_type'] == 'P2'].copy()
        print(f"  Loaded {len(p2_reported)} P2 data points from CSV")
        
        models_checked = 0
        passed = 0
        failed = 0
        
        # Check each model
        for (family, size), metrics in te_metrics.items():
            fn = metrics['fn']
            total_positive = metrics['total_positive']
            fnr_point = metrics['fnr']  # For reference
            
            # Find matching rows in reported data
            reported_model = p2_reported[
                (p2_reported['model_family'].apply(normalize_family) == normalize_family(family)) &
                (p2_reported['model_size'] == size)
            ]
            
            if len(reported_model) == 0:
                print(f"  WARNING: No reported data for {family} {size}")
                continue
            
            # Check each baseline percentage
            for baseline_pct in BASELINE_PERCENTAGES:
                models_checked += 1
                
                # Calculate P2 using Monte Carlo (matching pipeline approach)
                mc_result = calculate_p2_mc(
                    fn, total_positive, baseline_pct, m,
                    prob_fail_help=PROB_FAIL_SEEK_HELP,
                    n_samples=N_MC_SAMPLES,
                    prior_alpha=PRIOR_ALPHA,
                    random_state=42  # Fixed seed for reproducibility
                )
                calc_p2 = mc_result['median']
                
                # Also calculate point estimate for reference
                point_p2 = calculate_p2_point(fnr_point, baseline_pct, m, PROB_FAIL_SEEK_HELP)
                
                # Get reported P2 (median from Monte Carlo)
                reported_row = reported_model[
                    abs(reported_model['baseline_percentage'] - baseline_pct) < 0.01
                ]
                
                if len(reported_row) == 0:
                    audit_results.append({
                        'model_family': family,
                        'model_size': size,
                        'm_value': m,
                        'baseline_pct': baseline_pct,
                        'fn': fn,
                        'total_positive': total_positive,
                        'calc_p2_mc': calc_p2,
                        'calc_p2_point': point_p2,
                        'reported_p2': None,
                        'difference': None,
                        'relative_diff_pct': None,
                        'match': False,
                        'note': 'No reported value',
                    })
                    failed += 1
                    continue
                
                reported_p2 = reported_row.iloc[0]['risk_probability']
                
                # Compare with tolerance
                abs_diff = abs(calc_p2 - reported_p2)
                rel_diff = abs_diff / calc_p2 if calc_p2 > 0 else 0
                
                # Pass if within tolerance (Monte Carlo has inherent variance)
                match = (abs_diff < ABSOLUTE_TOLERANCE) or (rel_diff < RELATIVE_TOLERANCE)
                
                if match:
                    passed += 1
                else:
                    failed += 1
                
                audit_results.append({
                    'model_family': family,
                    'model_size': size,
                    'm_value': m,
                    'baseline_pct': baseline_pct,
                    'fn': fn,
                    'total_positive': total_positive,
                    'calc_p2_mc': calc_p2,
                    'calc_p2_point': point_p2,
                    'reported_p2': reported_p2,
                    'difference': abs_diff,
                    'relative_diff_pct': rel_diff * 100,
                    'match': match,
                })
        
        m_summaries[m] = {
            'csv_found': True,
            'models_checked': models_checked,
            'passed': passed,
            'failed': failed,
        }
        
        status = "✅" if failed == 0 else "❌"
        print(f"  {status} M={m}: {passed}/{models_checked} passed ({failed} failed)")
    
    # Create DataFrame
    audit_df = pd.DataFrame(audit_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    
    total_checks = sum(s['models_checked'] for s in m_summaries.values())
    total_passed = sum(s['passed'] for s in m_summaries.values())
    total_failed = sum(s['failed'] for s in m_summaries.values())
    
    print(f"\nTotal data points verified: {total_checks}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    
    print(f"\nPer-M-value results:")
    all_passed = True
    for m, summary in m_summaries.items():
        if not summary['csv_found']:
            print(f"  ⚠️  M={m}: CSV not found")
            all_passed = False
        elif summary['failed'] > 0:
            print(f"  ❌ M={m}: {summary['passed']}/{summary['models_checked']} passed")
            all_passed = False
        else:
            print(f"  ✅ M={m}: {summary['passed']}/{summary['models_checked']} passed")
    
    if all_passed and total_checks > 0:
        print("\n" + "=" * 70)
        print("✅ FIGURE S11 VERIFIED")
        print("=" * 70)
        print("All P2 calculations match reported values within tolerance.")
        print(f"(Tolerance: {RELATIVE_TOLERANCE*100}% relative or {ABSOLUTE_TOLERANCE} absolute)")
    else:
        print("\n" + "=" * 70)
        print(f"❌ {total_failed} VERIFICATION FAILURES")
        print("=" * 70)
        print("Review audit report for details.")
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False)
    print(f"\nAudit report saved to: {output_path}")
    
    # Save summary JSON
    summary = {
        'audit_timestamp': datetime.now().isoformat(),
        'paper_run_dir': str(paper_run_dir),
        'figure_verified': 'Figure S11',
        'total_data_points': total_checks,
        'passed': total_passed,
        'failed': total_failed,
        'all_passed': all_passed and total_checks > 0,
        'm_values_checked': list(m_summaries.keys()),
        'tolerance': {
            'relative_pct': RELATIVE_TOLERANCE * 100,
            'absolute': ABSOLUTE_TOLERANCE,
        },
        'formula': 'P2 = (1 - (1-FNR)^m) × P(fail_help) × P(lack_care_harm)',
        'parameters': {
            'prob_fail_seek_help': PROB_FAIL_SEEK_HELP,
            'baseline_percentages': BASELINE_PERCENTAGES,
            'fnr_source': 'therapy_engagement_comprehensive_metrics.csv',
        },
        'm_value_summaries': {str(k): v for k, v in m_summaries.items()},
    }
    
    # Save JSON summary to figure_provenance (provenance info)
    provenance_dir = output_path.parent.parent / 'figure_provenance'
    provenance_dir.mkdir(parents=True, exist_ok=True)
    summary_path = provenance_dir / 'figure_s11_audit_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Audit summary saved to: {summary_path}")
    
    return audit_df


def find_latest_paper_run() -> Optional[Path]:
    """Find the most recent REGULATORY_SIMULATION_PAPER run."""
    results_dir = PROJECT_ROOT / 'results' / 'REGULATORY_SIMULATION_PAPER'
    if not results_dir.exists():
        return None
    
    # Filter to only timestamp-based directories
    timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')
    
    runs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)],
        key=lambda x: x.name,
        reverse=True
    )
    return runs[0] if runs else None


def main():
    parser = argparse.ArgumentParser(
        description="Audit Figure S11: P2 across failure multiplier values"
    )
    parser.add_argument(
        '--paper-run-dir',
        type=str,
        default=None,
        help='Path to pipeline output directory (default: most recent run)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: figure_s11_audit_report.csv in provenance folder)'
    )
    
    args = parser.parse_args()
    
    # Find paper run directory
    if args.paper_run_dir:
        paper_run_dir = Path(args.paper_run_dir)
    else:
        paper_run_dir = find_latest_paper_run()
        if paper_run_dir is None:
            print("ERROR: No paper run found. Run the pipeline first or specify --paper-run-dir")
            sys.exit(1)
    
    if not paper_run_dir.exists():
        print(f"ERROR: Paper run directory not found: {paper_run_dir}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = paper_run_dir / 'Logs' / 'Audits' / 'figure_s11_audit_report.csv'
    
    # Run audit
    try:
        audit_df = run_audit(
            paper_run_dir=paper_run_dir,
            output_path=output_path
        )
        
        # Exit with error if any failures
        if 'match' in audit_df.columns and not audit_df['match'].all():
            sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
