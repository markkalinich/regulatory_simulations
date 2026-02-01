#!/usr/bin/env python3
"""
Multi-experiment performance comparison plot 
Rows = metrics, Columns = classification tasks
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import argparse
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add project root
from utilities.figure_provenance import FigureProvenanceTracker
from utilities.statistics import clopper_pearson_ci, bootstrap_f1_ci
from utilities.input_validation import validate_comprehensive_metrics, InputValidationError

# Initialize provenance tracker
tracker = FigureProvenanceTracker(
    figure_name="multi_model_performance",
    base_dir=Path(__file__).parent.parent / "results" / "model_performance_comparison"
)

def map_model_size_to_billions(model_size):
    """Convert model size strings to billions for plotting.
    
    Handles sizes with suffixes like '1b-it', '8b-pt' by extracting the core size.
    """
    import re
    
    size_mapping = {
        '270m': 0.27, '1b': 1.0, '4b': 4.0, '8b': 8.0, '12b': 12.0, '27b': 27.0,
        '0.6b': 0.6, '1.7b': 1.7, '14b': 14.0, '32b': 32.0, '70b': 70.0
    }
    
    # Extract core size (e.g., '1b-it' -> '1b', '270m-pt' -> '270m')
    size_lower = model_size.lower()
    match = re.match(r'^(\d+\.?\d*[bm])', size_lower)
    if match:
        core_size = match.group(1)
        return size_mapping.get(core_size, float('nan'))
    
    return size_mapping.get(size_lower, float('nan'))

def load_comprehensive_metrics(csv_path, experiment_name):
    """Load pre-calculated metrics from comprehensive_metrics.csv and compute CIs
    
    Args:
        csv_path: Path to comprehensive_metrics.csv
        experiment_name: Name of experiment (e.g., "Suicidal Ideation")
        
    Returns:
        DataFrame with metrics and computed confidence intervals
        
    Raises:
        InputValidationError: If input data fails validation
    """
    # Validate input file before processing
    df = validate_comprehensive_metrics(
        Path(csv_path), 
        experiment_name=experiment_name,
        min_models=1
    )
    
    # Add experiment name and convert model size
    df['experiment'] = experiment_name
    df['model_size_billions'] = df['model_size'].apply(map_model_size_to_billions)
    
    # Calculate Clopper-Pearson CIs for each metric
    # Parse success rate: successful_parses / total_samples
    parse_cis = df.apply(lambda row: clopper_pearson_ci(
        int(row['successful_parses']), int(row['total_samples'])), axis=1)
    df['parse_success_rate_ci_lower'] = parse_cis.apply(lambda x: x[0])
    df['parse_success_rate_ci_upper'] = parse_cis.apply(lambda x: x[1])
    
    # Sensitivity: tp / total_positive
    sens_cis = df.apply(lambda row: clopper_pearson_ci(
        int(row['tp']), int(row['total_positive'])), axis=1)
    df['sensitivity_ci_lower'] = sens_cis.apply(lambda x: x[0])
    df['sensitivity_ci_upper'] = sens_cis.apply(lambda x: x[1])
    
    # Specificity: tn / total_negative
    spec_cis = df.apply(lambda row: clopper_pearson_ci(
        int(row['tn']), int(row['total_negative'])), axis=1)
    df['specificity_ci_lower'] = spec_cis.apply(lambda x: x[0])
    df['specificity_ci_upper'] = spec_cis.apply(lambda x: x[1])
    
    # Accuracy: (tp + tn) / total_samples
    acc_cis = df.apply(lambda row: clopper_pearson_ci(
        int(row['tp'] + row['tn']), int(row['total_samples'])), axis=1)
    df['accuracy_ci_lower'] = acc_cis.apply(lambda x: x[0])
    df['accuracy_ci_upper'] = acc_cis.apply(lambda x: x[1])
    
    # F1 score: bootstrap CI using 4-cell multinomial over full confusion matrix
    f1_cis = df.apply(lambda row: bootstrap_f1_ci(
        int(row['tp']), int(row['fp']), int(row['fn']), int(row['tn']), n_bootstrap=5000), axis=1)
    df['f1_score_ci_lower'] = f1_cis.apply(lambda x: x[0])
    df['f1_score_ci_upper'] = f1_cis.apply(lambda x: x[1])
    
    return df

def calculate_marker_size(model_size_billions):
    """Calculate marker size based on log of model parameters"""
    # Use log scale to make differences more visible
    # Base size + log scaling factor
    base_size = 30
    log_size = np.log10(model_size_billions) * 40  # Scale factor for visibility
    return base_size + log_size

def create_multi_experiment_plot(si_csv, tr_csv, td_csv, figsize=(15, 25), log_x=True):
    """Create transposed faceted comparison plot with provenance tracking"""
    
    # Track input datasets
    tracker.add_input_dataset(
        si_csv,
        description="Suicide ideation detection comprehensive metrics",
        columns_used=['model_family', 'model_size', 'parse_success_rate', 'sensitivity', 'specificity', 'accuracy', 'f1_score']
    )
    
    tracker.add_input_dataset(
        tr_csv,
        description="Therapy request detection comprehensive metrics",
        columns_used=['model_family', 'model_size', 'parse_success_rate', 'sensitivity', 'specificity', 'accuracy', 'f1_score']
    )
    
    tracker.add_input_dataset(
        td_csv,
        description="Therapy engagement detection comprehensive metrics",
        columns_used=['model_family', 'model_size', 'parse_success_rate', 'sensitivity', 'specificity', 'accuracy', 'f1_score']
    )
    
    # Much larger fonts for manuscript quality
    plt.rcParams.update({
        'font.size': 20,             # Increased from 15.625
        'axes.titlesize': 24,        # Increased from 18.75
        'axes.labelsize': 22,        # Increased from 18.75
        'xtick.labelsize': 18,       # Increased from 14.0625
        'ytick.labelsize': 18,       # Increased from 14.0625
        'legend.fontsize': 18,       # Increased from 14.0625
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
    })
    
    # Load all three experiments
    si_data = load_comprehensive_metrics(si_csv, "Suicidal Ideation")
    tr_data = load_comprehensive_metrics(tr_csv, "Therapy Request") 
    td_data = load_comprehensive_metrics(td_csv, "Therapeutic Interaction")
    
    # Combine all data
    all_data = pd.concat([si_data, tr_data, td_data], ignore_index=True)
    
    print(f"Loaded {len(all_data)} data points across {len(all_data['experiment'].unique())} experiments")
    
    # Define metrics to plot (rows)
    metrics = ['parse_success_rate', 'sensitivity', 'specificity', 'accuracy', 'f1_score']
    metric_labels = ['Parsing Success Rate', 'Sensitivity', 'Specificity', 'Accuracy', 'F1 Score']
    
    # Experiments (columns)
    experiments = ['Suicidal Ideation', 'Therapy Request', 'Therapeutic Interaction']
    
    # Create the TRANSPOSED plot: rows = metrics, columns = experiments
    fig, axes = plt.subplots(5, 3, figsize=figsize, sharey=True)
    
    # Colorblind-friendly palette similar to NEJM style
    # Using distinct colors that work for colorblind readers
    family_colors = {
        'gemma': '#0173B2',   # Blue
        'qwen': '#DE8F05',    # Orange  
        'llama': '#029E73'    # Green/teal
    }
    
    # Calculate marker sizes for all models
    all_data['marker_size'] = all_data['model_size_billions'].apply(calculate_marker_size)
    
    # Track if we've added the legend
    legend_added = False
    
    for metric_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        for exp_idx, experiment in enumerate(experiments):
            ax = axes[metric_idx, exp_idx]
            exp_data = all_data[all_data['experiment'] == experiment]
            
            # Plot each model family with different colors (uniform marker size)
            # Use substring matching to handle variants like 'llama3.2', 'qwen2', 'gemma2'
            for family in ['gemma', 'qwen', 'llama']:
                family_data = exp_data[exp_data['model_family'].str.lower().str.startswith(family)].sort_values('model_size_billions')
                if len(family_data) > 0:
                    x_vals = family_data['model_size_billions'].values
                    y_vals = family_data[metric].values
                    
                    # Get CI bounds for this metric
                    ci_lower = family_data[f'{metric}_ci_lower'].values
                    ci_upper = family_data[f'{metric}_ci_upper'].values
                    yerr_lower = y_vals - ci_lower
                    yerr_upper = ci_upper - y_vals
                    
                    # Plot error bars first (behind points) - black for visibility
                    ax.errorbar(x_vals, y_vals,
                               yerr=[yerr_lower, yerr_upper],
                               fmt='none',
                               color='black',
                               alpha=0.6,
                               capsize=3,
                               capthick=1.5,
                               elinewidth=1.5)
                    
                    # Plot with uniform marker size
                    ax.scatter(x_vals, y_vals,
                             color=family_colors[family], 
                             s=150,  # Increased from 100 by 50%
                             alpha=0.7,
                             edgecolors='black',
                             linewidths=0.5,
                             zorder=5,  # Ensure points are on top of error bars
                             label=family.capitalize() if not legend_added else '')
                    
                    # Connect points with thicker lines
                    ax.plot(x_vals, y_vals,
                           color=family_colors[family], 
                           linewidth=3,  # Increased from 2
                           alpha=0.6,
                           zorder=4)
            
            # Add legend to top-right panel only (metric 0, experiment 2) in bottom-right corner
            if metric_idx == 0 and exp_idx == 2 and not legend_added:
                handles = [mlines.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=family_colors[family], 
                                     markersize=10, label=family.capitalize(),
                                     markeredgecolor='black', markeredgewidth=0.5)
                          for family in ['gemma', 'qwen', 'llama']]
                # Fix display names in legend
                for h, name in zip(handles, ['Gemma', 'Qwen', 'LLaMA']):
                    h.set_label(name)
                ax.legend(handles=handles, loc='lower right', framealpha=0.9)
                legend_added = True
            
            # Column labels (experiments) - only on top row
            if metric_idx == 0:
                ax.set_title(experiment, fontweight='bold', pad=10)
            
            # Row labels (metrics) - only on leftmost column
            if exp_idx == 0:
                ax.set_ylabel(metric_label, fontweight='bold')
            
            # X-axis labels - only on bottom row
            if metric_idx == 4:  # Last row
                ax.set_xlabel('Parameters (Bn)', fontweight='bold')
            
            # Darker gridlines (25% darker = alpha increased from 0.3 to ~0.5)
            ax.grid(True, alpha=0.5, linewidth=0.8)
            
            # Use log or linear scale for x-axis based on parameter
            ax.set_xscale('log' if log_x else 'linear')
            ax.set_xlim(0.1, max(all_data['model_size_billions'].max() * 1.2, 100))
            ax.set_xticks([0.1, 1, 10, 100])
            ax.set_xticklabels(['0.1', '1', '10', '100'])
            
            # All metrics use same y-axis: 0 to 1.05
            ax.set_ylim(-0.02, 1.05)
            
            # Y-axis ticks
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    # Set analysis parameters
    tracker.set_analysis_parameters(
        layout='transposed',
        experiments=experiments,
        metrics=metrics,
        metric_labels=metric_labels,
        model_families=['gemma', 'qwen', 'llama'],
        family_colors=family_colors,
        log_x_scale=log_x,
        figsize=figsize,
        y_limit=[-0.02, 1.05],
        marker_size=150,
        line_width=3
    )
    
    plt.tight_layout()
    
    # Save outputs using provenance tracker
    output_png = tracker.get_output_path("multi_model_performance.png")
    plt.savefig(str(output_png), dpi=300, bbox_inches='tight')
    tracker.add_output_file(output_png, file_type="figure")
    
    print(f"  Saved PNG: {output_png}")
    
    # Save provenance metadata
    tracker.save_provenance()
    
    print(f"✅ Multi-experiment performance plot (transposed) saved with provenance tracking")
    print(f"   Output directory: {tracker.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create multi-experiment performance comparison plot (transposed) with provenance')
    parser.add_argument('--suicide-ideation-csv', required=True, 
                       help='Path to suicide ideation comprehensive_metrics.csv')
    parser.add_argument('--therapy-request-csv', required=True,
                       help='Path to therapy request comprehensive_metrics.csv') 
    parser.add_argument('--therapy-detection-csv', required=True,
                       help='Path to therapy detection comprehensive_metrics.csv')
    parser.add_argument('--figsize', nargs=2, type=float, default=[15, 25],
                       help='Figure size (width height)')
    parser.add_argument('--linear-x', action='store_true',
                       help='Use linear x-axis instead of logarithmic (default: logarithmic)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTI-EXPERIMENT PERFORMANCE COMPARISON (TRANSPOSED)")
    print("(WITH PROVENANCE TRACKING)")
    print("="*80)
    
    # Verify all CSV files exist
    for csv_path in [args.suicide_ideation_csv, args.therapy_request_csv, args.therapy_detection_csv]:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    create_multi_experiment_plot(
        args.suicide_ideation_csv,
        args.therapy_request_csv, 
        args.therapy_detection_csv,
        tuple(args.figsize),
        not args.linear_x  # Invert: linear_x flag OFF means log scale ON
    )
    
    print("="*80)
    print("COMPLETE!")
    print(f"All outputs saved to: {tracker.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
