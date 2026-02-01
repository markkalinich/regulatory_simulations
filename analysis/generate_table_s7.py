#!/usr/bin/env python3
"""
Generate Table S7: Parameters for P₁/P₂ Risk Model

This script generates a comprehensive summary table of all parameters used in the
P₁ and P₂ calculations, addressing reviewer concerns about model transparency.

Output: CSV table with parameter definitions, sources, tested ranges, and justifications.
"""

import sys
import os
import csv
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.regulatory_paper_parameters import RISK_MODEL_PARAMS


def generate_table_s7(output_path: str = None):
    """
    Generate Table S7 with all P₁/P₂ parameters.
    
    Args:
        output_path: Path to save CSV. If None, prints to stdout.
    """
    
    # Define table structure with all parameters
    table_rows = [
        # Header row
        ["Parameter", "Symbol/Notation", "Value/Range", "Source", "Tested Range (Sensitivity)", "Justification"],
        
        # === BEHAVIORAL PROBABILITIES ===
        [
            "Baseline SI Prevalence",
            "P(SI)",
            "0.1% - 10%",
            "Clinical literature (general population to crisis settings)",
            "0.1%, 1%, 2%, ..., 10%",
            "Range spans general population (~0.1%) to high-risk clinical settings (>10%). Used as x-axis in P₁ calculations."
        ],
        [
            "Therapy-Seeking Rate",
            "P(therapy_request | conversation)",
            f"{RISK_MODEL_PARAMS['therapy_request_rate']:.1%} (2.9%)",
            f"Anthropic (2025): {RISK_MODEL_PARAMS['empirical_therapy_request_k']:,} affective conversations out of {RISK_MODEL_PARAMS['empirical_therapy_request_n']:,} total",
            "Sampled from Beta posterior in Monte Carlo",
            "Empirically derived from large-scale usage data. 'Affective' conversations equated with therapy-seeking behavior. Beta({k}+1, {n}-{k}+1) posterior used for uncertainty propagation.".format(
                k=RISK_MODEL_PARAMS['empirical_therapy_request_k'],
                n=RISK_MODEL_PARAMS['empirical_therapy_request_n']
            )
        ],
        [
            "Model Compliance Rate",
            "P(model_engages | therapy_request)",
            f"{RISK_MODEL_PARAMS['model_comply_rate']:.0%} (90%)",
            f"Anthropic (2025): '<10% of coaching/counseling conversations involve resistance'",
            "Sampled from Beta posterior in Monte Carlo",
            "Assumes ~90% of therapy-like requests proceed without model pushback. Beta({k}+1, {n}-{k}+1) posterior used for uncertainty propagation.".format(
                k=RISK_MODEL_PARAMS['empirical_model_comply_k'],
                n=RISK_MODEL_PARAMS['empirical_model_comply_n']
            )
        ],
        [
            "Probability User Fails to Seek Help",
            "P(no_help | harmful_interaction)",
            f"{RISK_MODEL_PARAMS['prob_fail_seek_help']:.1f} (100%)",
            "Conservative assumption (worst case)",
            "Fixed at 1.0 in main analysis",
            "No empirical data available. We use worst-case assumption (user never seeks help elsewhere after harmful AI interaction) for conservative safety analysis. Could be relaxed with data."
        ],
        [
            "Baseline Harm Rate",
            "P(harm | SI, lack_of_care)",
            "0.1% - 10%",
            "Varied parameter (no specific empirical source)",
            "0.1%, 1%, 2%, ..., 10%",
            "Represents probability that lack of appropriate care leads to harm (self-harm, attempt). Used as x-axis in P₂ calculations. Range chosen to span plausible severity scenarios."
        ],
        
        # === MODEL FALSE NEGATIVE RATES ===
        [
            "SI Detection False Negative Rate",
            "FNR_SI(model)",
            "Model-specific (empirical)",
            "Experimental results: missed SI cases / total SI cases",
            "Varied across models; uncertainty via Beta posteriors",
            "Empirically measured for each model. When 0 failures observed, use 50th percentile Clopper-Pearson upper bound as conservative floor. Beta(α+FN, β+TN) posterior used in Monte Carlo."
        ],
        [
            "Therapy Request Detection FNR",
            "FNR_TR(model)",
            "Model-specific (empirical)",
            "Experimental results: missed therapy requests / total requests",
            "Varied across models; uncertainty via Beta posteriors",
            "Empirically measured for each model. When 0 failures observed, use 50th percentile Clopper-Pearson upper bound as conservative floor. Beta(α+FN, β+TN) posterior used in Monte Carlo."
        ],
        [
            "Therapy Engagement Detection FNR",
            "FNR_TE(model)",
            "Model-specific (empirical)",
            "Experimental results: missed engagement / total engagement",
            "Varied across models; adjusted by failure multiplier (m)",
            "Empirically measured for each model. In P₂ dependence analysis, adjusted via FNR_adj = 1-(1-FNR)^m to model correlated failures conditional on upstream detection failures."
        ],
        
        # === DEPENDENCE/CORRELATION PARAMETERS ===
        [
            "Failure Multiplier",
            "m",
            f"{RISK_MODEL_PARAMS['default_failure_multiplier']:.1f} (independence)",
            "Modeling assumption",
            ", ".join([f"{m:.0f}" if m < 1000 else f"{m:.0e}" for m in RISK_MODEL_PARAMS['failure_multiplier_values']]),
            "Power transformation parameter modeling conditional dependence: FNR_adj = 1-(1-FNR)^m. m=1 assumes independence; m>1 models increased failure rates conditional on prior failures. Used in sensitivity analysis to show impact of correlated detection errors."
        ],
        
        # === MONTE CARLO / UNCERTAINTY QUANTIFICATION ===
        [
            "Monte Carlo Sample Size",
            "N_MC",
            f"{RISK_MODEL_PARAMS['n_mc_samples']:,}",
            "Simulation parameter",
            "Fixed (50,000 for production, 1,000 for testing)",
            "Number of Monte Carlo draws for uncertainty propagation through P₁ and P₂. Samples drawn from Beta posteriors for FNRs and behavioral rates."
        ],
        [
            "Beta Prior Parameter",
            "α_prior = β_prior",
            f"{RISK_MODEL_PARAMS['prior_alpha']:.1f}",
            "Bayesian prior choice",
            "Fixed at 1.0 (uniform prior)",
            "Uniform prior used for FNR Bayesian estimation."
        ],
    ]
    
    # Write to CSV
    if output_path:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table_rows)
        print(f"✅ Table S7 written to: {output_path}")
        print(f"   ({len(table_rows)-1} parameters documented)")  # Subtract 1 for header row
    else:
        # Print to stdout
        writer = csv.writer(sys.stdout)
        writer.writerows(table_rows)
    
    return table_rows


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Table S7: P₁/P₂ Risk Model Parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Write to default location
  python analysis/generate_table_s7.py
  
  # Write to custom path
  python analysis/generate_table_s7.py -o results/supplementary_table_s7.csv
  
  # Print to stdout
  python analysis/generate_table_s7.py --stdout
        """
    )
    
    parser.add_argument(
        '-o', '--output',
        default='results/supplementary_tables/table_s7_parameters.csv',
        help='Output CSV path (default: results/supplementary_tables/table_s7_parameters.csv)'
    )
    
    parser.add_argument(
        '--stdout',
        action='store_true',
        help='Print to stdout instead of file'
    )
    
    args = parser.parse_args()
    
    if args.stdout:
        generate_table_s7(output_path=None)
    else:
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generate_table_s7(output_path=str(output_path))


if __name__ == "__main__":
    main()

