#!/usr/bin/env python3
"""
Visualization Module - Chart and plot generation for experiment results.

This module provides functions to create performance plots and visualization components,
extracted from batch_results_analyzer.py for reusability and testing.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def create_performance_plots(metrics_df: pd.DataFrame, 
                           output_file: Path,
                           title: str = "Model Performance Across Families") -> None:
    """
    Create comprehensive performance visualization plots.
    
    Args:
        metrics_df: DataFrame with model metrics (must have columns: model_family, model_size, 
                   parse_success_rate, sensitivity, specificity, accuracy, f1_score)
        output_file: Path where to save the plot PNG file
        title: Main title for the plot
    """
    print("Generating performance plots...")
    
    if metrics_df is None or metrics_df.empty:
        raise ValueError("Metrics DataFrame is empty or None")
    
    # Define family colors and size-based transparency
    family_colors = {'gemma': '#2E86AB', 'qwen': '#A23B72', 'llama': '#F18F01'}
    
    # Create subplots for 5 metrics x 3 families
    fig, axes = plt.subplots(5, 3, figsize=(18, 24))
    fig.suptitle(title, fontsize=20, y=0.98)
    metrics = ['parse_success_rate', 'sensitivity', 'specificity', 'accuracy', 'f1_score']
    metric_labels = ['Parse Success Rate', 'Sensitivity', 'Specificity', 'Accuracy', 'F1 Score']
    families = ['gemma', 'qwen', 'llama']
    
    for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        for family_idx, family in enumerate(families):
            ax = axes[metric_idx, family_idx]
            family_data = metrics_df[metrics_df['model_family'] == family].copy()
            if not family_data.empty:
                # Create size-based transparency (smaller models more transparent)
                size_order = ['270m', '0.6b', '1b', '1.7b', '4b', '8b', '12b', '14b', '27b', '32b', '70b']
                family_data['size_idx'] = family_data['model_size'].map(
                    lambda x: size_order.index(x) if x in size_order else len(size_order)
                )
                family_data = family_data.sort_values('size_idx')
                
                # Calculate transparency: smaller models = 0.4, larger models = 1.0
                alphas = []
                for size in family_data['model_size']:
                    if size in size_order:
                        idx = size_order.index(size)
                        alpha = 0.4 + (0.6 * idx / (len(size_order) - 1))
                    else:
                        alpha = 1.0
                    alphas.append(alpha)
                
                # Create bars with individual transparency
                bars_x = []
                for bar_idx, (size, value, alpha) in enumerate(zip(family_data['model_size'], family_data[metric], alphas)):
                    bar = ax.bar(bar_idx, value, color=family_colors[family], alpha=alpha, width=0.8)
                    bars_x.append(size)
                    # Add value label
                    ax.text(bar_idx, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_xticks(range(len(bars_x)))
                ax.set_xticklabels(bars_x, rotation=45)
                ax.set_ylim(0, 1.1)
                ax.set_title(f'{family.upper()} - {label}', fontsize=12)
                ax.set_xlabel('Model Size')
                ax.set_ylabel(label)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{family.upper()} - {label}', fontsize=12)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance plots: {output_file}")


def get_family_colors() -> Dict[str, str]:
    """
    Get the standard color mapping for model families.
    
    Returns:
        Dictionary mapping family names to color codes
    """
    return {'gemma': '#2E86AB', 'qwen': '#A23B72', 'llama': '#F18F01'}


def get_size_order() -> List[str]:
    """
    Get the standard ordering of model sizes for consistent plotting.
    
    Returns:
        List of model sizes in order from smallest to largest
    """
    return ['270m', '0.6b', '1b', '1.7b', '4b', '8b', '12b', '14b', '27b', '32b', '70b']