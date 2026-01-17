#!/usr/bin/env python3
"""
Figure S11: P2 by Harm Prevalence Across Failure Multiplier Values

Shows how P2 varies with harm prevalence at different failure multiplier (M) values.
Each row shows a different M value, demonstrating convergence behavior as 
correlation between detection failures increases.

8 rows × 3 columns facet plot - identical to middle row of Figure 5, repeated at different M values
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import functions from the existing Figure 5 script
from analysis.comparative_analysis.p1_and_p2_plot_provenance import (
    load_experiment_metrics,
    prepare_plot_data,
    normalize_family,
    format_size_label,
    get_alpha_for_param_billions,
    DEFAULT_PARAMS
)
from config.regulatory_paper_parameters import RISK_MODEL_PARAMS
from utilities.input_validation import InputValidationError

# M values to show (imported from centralized config)
M_VALUES = [int(m) for m in RISK_MODEL_PARAMS['failure_multiplier_values']]

# Model families
MODEL_FAMILIES = {'gemma': 'Gemma', 'qwen': 'Qwen', 'llama': 'LLaMA'}
FAMILY_COLORS = {'gemma': '#1f77b4', 'qwen': '#ff7f0e', 'llama': '#2ca02c'}


def create_figure_s11(si_csv, tr_csv, te_csv, output_path, params=None):
    """Create Figure S11: P2 facet plot across M values"""
    
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Load metrics
    suicide_metrics = load_experiment_metrics(si_csv)
    therapy_request_metrics = load_experiment_metrics(tr_csv)
    therapy_engagement_metrics = load_experiment_metrics(te_csv)
    
    # Set up the figure
    n_rows = len(M_VALUES)
    n_cols = len(MODEL_FAMILIES)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 32), sharey='row', sharex='col')
    
    # Font sizes
    title_size = 28
    tick_size = 22
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': title_size,
        'axes.labelsize': 24,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'legend.fontsize': 16
    })
    
    # Generate plot data for each M value
    for row_idx, m in enumerate(M_VALUES):
        print(f"Generating data for M={m}...")
        
        # Set failure multiplier for this row
        params['failure_multiplier'] = m
        
        # Prepare plot data (with Monte Carlo uncertainty)
        plot_data = prepare_plot_data(
            suicide_metrics, 
            therapy_request_metrics, 
            therapy_engagement_metrics, 
            params
        )
        
        # Add normalized family column
        plot_data['normalized_family'] = plot_data['model_family'].apply(normalize_family)
        
        # Filter to P2 only
        p2_data = plot_data[plot_data['risk_type'] == 'P2']
        
        # Plot each family in columns
        for col_idx, (family_key, family_name) in enumerate(MODEL_FAMILIES.items()):
            ax = axes[row_idx, col_idx]
            
            # Filter to this family
            family_data = p2_data[p2_data['normalized_family'] == family_key]
            
            if len(family_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes)
                continue
            
            # Get unique param_billions values, sorted
            unique_sizes = family_data[['param_billions']].drop_duplicates().sort_values('param_billions')
            
            # Check if we have uncertainty data
            has_uncertainty = 'risk_ci_5' in family_data.columns and \
                            (family_data['risk_ci_5'] != family_data['risk_probability']).any()
            
            uncertainty_style = params.get('uncertainty_style', 'both')
            
            # Plot each model size
            for _, size_row in unique_sizes.iterrows():
                param_b = size_row['param_billions']
                model_data = family_data[family_data['param_billions'] == param_b].sort_values('baseline_percentage')
                alpha = get_alpha_for_param_billions(param_b, family_data)
                size_label = format_size_label(param_b)
                
                x = model_data['baseline_percentage'].values
                y = model_data['risk_probability'].values
                
                if has_uncertainty and uncertainty_style != 'none':
                    ci_5 = model_data['risk_ci_5'].values
                    ci_95 = model_data['risk_ci_95'].values
                    yerr_lower = y - ci_5
                    yerr_upper = ci_95 - y
                    
                    # Draw ribbon if requested
                    if uncertainty_style in ('ribbon', 'both'):
                        ax.fill_between(x, ci_5, ci_95,
                                       color=FAMILY_COLORS[family_key], alpha=alpha * 0.15)
                    
                    # Draw error bars if requested (black for visibility)
                    if uncertainty_style in ('errorbar', 'both'):
                        ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper],
                                   marker='o', linestyle='-', color=FAMILY_COLORS[family_key], 
                                   alpha=alpha, linewidth=2, markersize=6, 
                                   capsize=3, capthick=1, elinewidth=1,
                                   ecolor='black',
                                   label=size_label)
                    elif uncertainty_style == 'ribbon':
                        ax.plot(x, y, marker='o', linestyle='-', color=FAMILY_COLORS[family_key], 
                               alpha=alpha, linewidth=2, markersize=6, label=size_label)
                else:
                    # No uncertainty
                    ax.plot(x, y, marker='o', linestyle='-', color=FAMILY_COLORS[family_key], 
                           alpha=alpha, linewidth=2, markersize=8, label=size_label)
            
            # Formatting
            ax.set_yscale('log')
            ax.set_ylim(1e-5, 1e-1)
            ax.grid(True, alpha=0.3)
            
            # Column titles (only top row)
            if row_idx == 0:
                ax.set_title(family_name)
            
            # Row labels (only left column)
            if col_idx == 0:
                ax.set_ylabel(f'P$_2$; M = {m:,}')
            
            # X-axis labels (only bottom row)
            if row_idx == n_rows - 1:
                ax.set_xlabel('P(Lack of Care → Harm) %')
            
            # Legend
            ax.legend(title='Model Size', loc='lower right')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure S11: P2 across M values'
    )
    parser.add_argument('--si-metrics', type=str, required=True,
                       help='Path to SI comprehensive_metrics.csv')
    parser.add_argument('--tr-metrics', type=str, required=True,
                       help='Path to therapy request comprehensive_metrics.csv')
    parser.add_argument('--te-metrics', type=str, required=True,
                       help='Path to therapy engagement comprehensive_metrics.csv')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for figure')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    output_path = output_dir / 'figure_s11_p2_across_m_values.png'
    create_figure_s11(args.si_metrics, args.tr_metrics, args.te_metrics, output_path)


if __name__ == '__main__':
    main()
