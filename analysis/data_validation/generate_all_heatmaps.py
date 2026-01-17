#!/usr/bin/env python3
"""
Generate heatmaps for all three tasks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.figure_provenance import FigureProvenanceTracker

# Base directory for results
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"


def get_si_severity_color(category):
    """Get color for SI category based on severity"""
    si_severity = {
        'passive_si': '#ffcccc',
        'active_si_no_plan': '#ff9999',
        'active_si_plan_no_intent': '#ff6666',
        'active_si_plan_with_intent': '#ff3333',
        'active_si_plan_with_intent_prep': '#cc0000',
    }
    return si_severity.get(category, '#ffffff')


def get_model_family_color(model_name):
    """Get color for model based on family"""
    model_lower = model_name.lower()
    if 'gemma' in model_lower:
        return '#1f77b4'
    elif 'qwen' in model_lower:
        return '#ff7f0e'
    elif 'llama' in model_lower:
        return '#2ca02c'
    return '#7f7f7f'


def add_legend(fig, items, title, location, bbox_anchor):
    """Helper to add a legend to the figure
    
    Args:
        fig: Matplotlib figure
        items: List of (label, color) tuples
        title: Legend title
        location: Legend location (e.g., 'upper left')
        bbox_anchor: Tuple for bbox_to_anchor positioning
    """
    legend_elements = [Rectangle((0, 0), 1, 1, fc=c, ec='black', lw=0.5) for _, c in items]
    legend_labels = [l for l, _ in items]
    legend = fig.legend(legend_elements, legend_labels,
                       title=title,
                       loc=location,
                       bbox_to_anchor=bbox_anchor,
                       frameon=True,
                       fontsize=8)
    legend.get_title().set_fontsize(9)
    legend.get_title().set_fontweight('bold')
    return legend


def create_heatmap(matrix_file, statement_info_file, title, output_name, row_color_func, row_legend_items=None):
    """
    Create a heatmap ordered by miss rate
    
    Args:
        matrix_file: Path to correctness matrix CSV
        statement_info_file: Path to statement info CSV
        title: Plot title
        output_name: Output filename (without extension)
        row_color_func: Function to map ground_truth categories to colors
        row_legend_items: Optional list of (label, color) tuples for row color legend
    """
    tracker = FigureProvenanceTracker(
        figure_name=output_name,
        base_dir=RESULTS_DIR / "model_performance_analysis"
    )
    
    # Load data
    matrix = pd.read_csv(matrix_file, index_col=0)
    matrix.columns = matrix.columns.astype(int)
    statement_info = pd.read_csv(statement_info_file)
    
    tracker.add_input_dataset(str(matrix_file), 'Correctness matrix')
    tracker.add_input_dataset(str(statement_info_file), 'Statement info')
    
    # Transpose: statements as rows, models as columns
    matrix_T = matrix.T
    
    # Sort rows by miss rate (most to least missed)
    miss_rates = 1 - matrix_T.mean(axis=1)
    sorted_indices = miss_rates.sort_values(ascending=False).index
    matrix_T_sorted = matrix_T.loc[sorted_indices]
    
    # Sort columns by accuracy (most to least accurate)
    model_accuracies = matrix_T_sorted.mean(axis=0).sort_values(ascending=False)
    matrix_T_sorted = matrix_T_sorted[model_accuracies.index]
    
    # Create colors
    row_colors = [row_color_func(statement_info.loc[idx, 'ground_truth']) for idx in matrix_T_sorted.index]
    col_colors = [get_model_family_color(model) for model in matrix_T_sorted.columns]
    
    # Create clustermap
    g = sns.clustermap(
        matrix_T_sorted,
        cmap='binary_r',
        vmin=0,
        vmax=1,
        row_colors=row_colors,
        col_colors=col_colors,
        row_cluster=False,
        col_cluster=False,  # Don't cluster columns - keep sorted by accuracy
        figsize=(12, 18),
        cbar_pos=None,
        xticklabels=True,
        yticklabels=False,
        linewidths=0,
        rasterized=True,
        dendrogram_ratio=(0.05, 0.05),  # Small space for legends (5% on each side)
    )
    
    # Add red line at 50% miss rate cutoff
    # Find where miss rate drops below 50%
    miss_rates_sorted = 1 - matrix_T_sorted.mean(axis=1)
    cutoff_idx = (miss_rates_sorted > 0.5).sum()
    if cutoff_idx > 0:
        g.ax_heatmap.axhline(y=cutoff_idx, color='red', linewidth=1.5, linestyle='-', zorder=10)
    
    g.ax_heatmap.set_xlabel('Model', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('Statement', fontsize=12, fontweight='bold')
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')
    g.fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Add legends using helper function
    families = [('Gemma', '#1f77b4'), ('Qwen', '#ff7f0e'), ('Llama', '#2ca02c')]
    add_legend(g.fig, families, 'Model Family (Column Colors)', 'upper left', (0.0, 0.97))
    
    cells = [('Correct', '#ffffff'), ('Miss', '#000000')]
    add_legend(g.fig, cells, 'Prediction Outcome', 'upper center', (0.5, 0.97))
    
    if row_legend_items:
        add_legend(g.fig, row_legend_items, 'Statement Type (Row Colors)', 'upper right', (1.0, 0.97))
    
    # Hide dendrogram axes
    g.ax_col_dendrogram.axis('off')
    g.ax_row_dendrogram.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    
    # Save
    output_path = tracker.get_output_path(f'{output_name}.png')
    g.savefig(output_path, dpi=300, bbox_inches='tight')
    tracker.add_output_file(f'{output_name}.png', 'Heatmap visualization')
    tracker.save_provenance()
    
    print(f"Saved {output_name} to: {output_path}")
    print(f"  Overall accuracy: {matrix_T.values.mean():.1%}\n")


def main():
    review_stats_dir = RESULTS_DIR / "review_statistics"
    
    # SI heatmap
    def si_row_colors(category):
        return get_si_severity_color(category)
    
    si_legend = [
        ('Non-SI', '#ffffff'),
        ('Passive SI', '#ffcccc'),
        ('Active SI (No Plan)', '#ff9999'),
        ('Active SI (Plan, No Intent)', '#ff6666'),
        ('Active SI (Plan with Intent)', '#ff3333'),
        ('Active SI (Plan, Intent, Prep)', '#cc0000'),
    ]
    
    create_heatmap(
        review_stats_dir / 'si_model_statement_correctness_matrix.csv',
        review_stats_dir / 'si_statement_info.csv',
        'SI Statement Accuracy',
        'si_correctness_heatmap',
        si_row_colors,
        si_legend
    )
    
    # Therapy request heatmap  
    def tr_row_colors(category):
        # Binary: therapy request vs non-therapy request
        if 'Explicit Requests for Therapy' in str(category):
            return '#ffcccc'  # Light red for therapy request
        return '#ccccff'  # Light blue for non-therapy request
    
    tr_legend = [
        ('Therapy Request', '#ffcccc'),
        ('Non-Therapy Request', '#ccccff'),
    ]
    
    create_heatmap(
        review_stats_dir / 'therapy_request_model_statement_correctness_matrix.csv',
        review_stats_dir / 'therapy_request_statement_info.csv',
        'Therapy Request Statement Accuracy',
        'therapy_request_correctness_heatmap',
        tr_row_colors,
        tr_legend
    )
    
    # Therapy engagement heatmap
    def te_row_colors(category):
        # Binary: therapeutic engagement vs non-engagement
        if 'clear_engagement' in str(category).lower():
            return '#ccffcc'  # Light green for therapeutic engagement
        return '#ffcccc'  # Light red for non-engagement
    
    te_legend = [
        ('Therapeutic Engagement', '#ccffcc'),
        ('Non-Engagement', '#ffcccc'),
    ]
    
    create_heatmap(
        review_stats_dir / 'therapy_engagement_model_conversation_correctness_matrix.csv',
        review_stats_dir / 'therapy_engagement_conversation_info.csv',
        'Therapy Engagement Conversation Accuracy',
        'therapy_engagement_correctness_heatmap',
        te_row_colors,
        te_legend
    )


if __name__ == '__main__':
    main()
