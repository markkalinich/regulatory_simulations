#!/usr/bin/env python3
"""
Base facet plot creation functionality.

This module provides the core plotting logic that is shared across
all family-specific facet plots (Gemma, Llama, Qwen, combined).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any

try:
    from .facet_plot_utils import (
        MODEL_TYPE_COLORS,
        MODEL_TYPE_MARKERS,
        MODEL_TYPE_DISPLAY_LABELS,
        apply_guard_metrics_to_df,
    )
except ImportError:
    from facet_plot_utils import (
        MODEL_TYPE_COLORS,
        MODEL_TYPE_MARKERS,
        MODEL_TYPE_DISPLAY_LABELS,
        apply_guard_metrics_to_df,
    )


@dataclass
class FacetPlotConfig:
    """Configuration for a facet plot.
    
    This dataclass captures all the family-specific settings needed
    to generate a facet plot, allowing the plotting logic to be reused.
    """
    # Data settings
    family_filter_fn: Callable[[pd.DataFrame], pd.DataFrame]
    """Function to filter dataframe to only include models for this plot."""
    
    version_fn: Callable[[str, str], float]
    """Function to determine version from (family, size) -> numeric version."""
    
    model_type_fn: Callable[[str, str], str]
    """Function to determine model type from (family, size) -> type string."""
    
    param_fn: Callable[[str, str], float]
    """Function to get parameter count from (family, size) -> billions."""
    
    # Version/column settings
    versions: List[float]
    """List of numeric version values for columns."""
    
    version_labels: List[str]
    """Display labels for each version column."""
    
    # Metrics to plot (rows)
    metrics: List[str] = field(default_factory=lambda: [
        'parse_success_rate', 'sensitivity', 'specificity', 'accuracy', 'f1_score'
    ])
    metric_labels: List[str] = field(default_factory=lambda: [
        'Parse Success Rate', 'Sensitivity', 'Specificity', 'Accuracy', 'F1 Score'
    ])
    
    # Plot styling
    figsize: Tuple[float, float] = (14, 18)
    x_lim: Tuple[float, float] = (0.15, 50)
    x_ticks: List[float] = field(default_factory=lambda: [0.3, 1, 3, 10, 30])
    x_tick_labels: List[str] = field(default_factory=lambda: ['0.3', '1', '3', '10', '30'])
    
    # Model type styling (can override defaults)
    type_colors: Dict[str, str] = field(default_factory=lambda: MODEL_TYPE_COLORS.copy())
    type_markers: Dict[str, str] = field(default_factory=lambda: MODEL_TYPE_MARKERS.copy())
    type_display_labels: Dict[str, str] = field(default_factory=lambda: MODEL_TYPE_DISPLAY_LABELS.copy())
    
    # Optional guard metrics computation
    guard_metrics_fn: Optional[Callable[[], Dict]] = None
    """Optional function to compute guard model metrics from cache."""
    
    # Plot name for logging
    plot_name: str = "Facet Plot"


def create_facet_plot(
    df: pd.DataFrame,
    config: FacetPlotConfig,
    output_path: str,
    title: Optional[str] = None,
    show_summary: bool = True
) -> None:
    """Create a facet plot using the provided configuration.
    
    Args:
        df: DataFrame with model metrics (from comprehensive_metrics.csv)
        config: FacetPlotConfig with all plot settings
        output_path: Path to save the output PNG
        title: Optional overall title for the plot
        show_summary: Whether to print summary of models per version
    """
    # Apply guard metrics if function provided
    if config.guard_metrics_fn is not None:
        guard_metrics = config.guard_metrics_fn()
        df = apply_guard_metrics_to_df(df, guard_metrics)
        if guard_metrics:
            print(f"Applied guard model corrections for {len(guard_metrics)} models")
    
    # Filter to relevant models
    df = config.family_filter_fn(df).copy()
    
    if len(df) == 0:
        print(f"No models found for {config.plot_name}!")
        return
    
    # Add computed columns
    df['size_billions'] = df.apply(
        lambda r: config.param_fn(r['model_family'], r['model_size']), axis=1
    )
    df['model_type'] = df.apply(
        lambda r: config.model_type_fn(r['model_family'], r['model_size']), axis=1
    )
    df['version'] = df.apply(
        lambda r: config.version_fn(r['model_family'], r['model_size']), axis=1
    )
    
    print(f"Loaded {len(df)} models for {config.plot_name}")
    print(f"Model types: {df['model_type'].value_counts().to_dict()}")
    print(f"Versions: {df['version'].value_counts().to_dict()}")
    
    # Setup plot styling
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',
    })
    
    # Create figure
    fig, axes = plt.subplots(
        len(config.metrics), len(config.versions), 
        figsize=config.figsize, 
        sharex=True, sharey='row'
    )
    
    # Handle single row/column case
    if len(config.metrics) == 1:
        axes = axes.reshape(1, -1)
    if len(config.versions) == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each cell
    for metric_idx, (metric, metric_label) in enumerate(zip(config.metrics, config.metric_labels)):
        for ver_idx, (version, ver_label) in enumerate(zip(config.versions, config.version_labels)):
            ax = axes[metric_idx, ver_idx]
            
            # Get data for this version
            ver_data = df[df['version'] == version].copy()
            
            if len(ver_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='gray')
                _style_empty_ax(ax, config, metric_idx, ver_idx, metric_label, ver_label)
                continue
            
            # Plot each model type
            for model_type in ver_data['model_type'].unique():
                type_data = ver_data[ver_data['model_type'] == model_type]
                
                # IT and PT are translucent, others are opaque
                alpha = 0.5 if model_type in ['IT', 'PT'] else 0.85
                
                for _, row in type_data.iterrows():
                    ax.scatter(
                        row['size_billions'], 
                        row[metric],
                        c=config.type_colors.get(model_type, '#888888'),
                        marker=config.type_markers.get(model_type, 'o'),
                        s=120,
                        alpha=alpha,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=3
                    )
            
            # Connect IT models with lines
            it_data = ver_data[ver_data['model_type'] == 'IT'].sort_values('size_billions')
            if len(it_data) > 1:
                ax.plot(it_data['size_billions'], it_data[metric],
                       color=config.type_colors.get('IT', '#2E86AB'), 
                       linewidth=1.5, alpha=0.5, zorder=2)
            
            # Connect PT models with lines
            pt_data = ver_data[ver_data['model_type'] == 'PT'].sort_values('size_billions')
            if len(pt_data) > 1:
                ax.plot(pt_data['size_billions'], pt_data[metric],
                       color=config.type_colors.get('PT', '#52B788'), 
                       linewidth=1.5, alpha=0.5, zorder=2)
            
            # Style the axis
            _style_ax(ax, config, metric_idx, ver_idx, metric_label, ver_label)
    
    # Create legend
    _add_legend(fig, config)
    
    # Add title and adjust layout
    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 0.88, 0.99])
    else:
        plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")
    
    # Save PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to: {pdf_path}")
    
    plt.close()
    
    # Print summary
    if show_summary:
        _print_summary(df, config)


def _style_ax(ax, config: FacetPlotConfig, metric_idx: int, ver_idx: int, 
              metric_label: str, ver_label: str) -> None:
    """Apply standard styling to an axis."""
    ax.set_xscale('log')
    ax.set_xlim(config.x_lim)
    ax.set_xticks(config.x_ticks)
    ax.set_xticklabels(config.x_tick_labels)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(-0.02, 1.05)
    
    # Column titles (top row only)
    if metric_idx == 0:
        ax.set_title(ver_label, fontweight='bold', pad=10)
    
    # Row labels (left column only)
    if ver_idx == 0:
        ax.set_ylabel(metric_label, fontweight='bold')
    
    # X-axis label (bottom row only)
    if metric_idx == len(config.metrics) - 1:
        ax.set_xlabel('Parameters (B)', fontweight='bold')


def _style_empty_ax(ax, config: FacetPlotConfig, metric_idx: int, ver_idx: int,
                    metric_label: str, ver_label: str) -> None:
    """Apply styling to an empty axis."""
    ax.set_xscale('log')
    ax.set_xlim(config.x_lim)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    if metric_idx == 0:
        ax.set_title(ver_label, fontweight='bold', pad=10)
    if ver_idx == 0:
        ax.set_ylabel(metric_label, fontweight='bold')
    if metric_idx == len(config.metrics) - 1:
        ax.set_xlabel('Parameters (B)', fontweight='bold')


def _add_legend(fig, config: FacetPlotConfig) -> None:
    """Add legend to the figure."""
    # Get unique types that are actually in the data
    type_handles = [
        mlines.Line2D(
            [0], [0], 
            marker=config.type_markers.get(model_type, 'o'), 
            color=color, 
            linestyle='None',
            markersize=10, 
            label=config.type_display_labels.get(model_type, model_type), 
            markeredgecolor='black', 
            markeredgewidth=0.5
        )
        for model_type, color in config.type_colors.items()
    ]
    
    fig.legend(
        handles=type_handles, 
        title='Model Type', 
        loc='upper right', 
        bbox_to_anchor=(0.99, 0.98), 
        framealpha=0.9
    )


def _print_summary(df: pd.DataFrame, config: FacetPlotConfig) -> None:
    """Print summary of models per version."""
    print(f"\n=== Models per {config.plot_name} Version ===")
    for version, ver_label in zip(config.versions, config.version_labels):
        ver_data = df[df['version'] == version]
        print(f"\n{ver_label}:")
        if len(ver_data) == 0:
            print("  (no models)")
        else:
            for _, row in ver_data.iterrows():
                parse_rate = row.get('parse_success_rate', 0) * 100
                print(f"  {row['model_family']:20} {row['model_size']:30} ({row['model_type']:15}) - Parse: {parse_rate:.1f}%")
