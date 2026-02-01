#!/usr/bin/env python3
"""
Llama Version Facet Plot - SI Detection Performance

Columns = Llama version (1, 2, 3.0, 3.1+)
Rows = Metrics (Parse%, Sensitivity, Specificity, Accuracy, F1)
X-axis = Log10 model parameters (billions)
Shape = Model type (IT, PT, Mental Health, Guard, Medical)

Only includes Llama-family models

Model metadata is loaded from config/models_config.csv
"""

import pandas as pd
from pathlib import Path
import argparse

from facet_plot_base import FacetPlotConfig, create_facet_plot
from facet_plot_utils import (
    get_model_metadata,
    get_param_billions_from_config,
    compute_llama_guard_metrics,
)


# =============================================================================
# Llama-Specific Configuration
# =============================================================================

def filter_llama_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only Llama-family models."""
    return df[df['model_family'].str.lower().str.contains('llama', na=False)]


def determine_llama_version(family: str, size: str) -> float:
    """Determine Llama generation from CSV config or family name.
    
    Returns:
        Version number. 3.1, 3.2, 3.3 are grouped into 3.5 for the '3.1+' column.
    """
    metadata = get_model_metadata(family, size)
    if metadata is not None and pd.notna(metadata.get('version')):
        version = float(metadata['version'])
        # Group 3.1, 3.2, 3.3 into 3.5 column
        if version >= 3.1:
            return 3.5
        return version
    
    # Fallback: try to infer from family name
    family_lower = family.lower()
    if 'llama4' in family_lower:
        return 4.0
    elif 'llama3.3' in family_lower or 'llama3.2' in family_lower or 'llama3.1' in family_lower:
        return 3.5  # Group into 3.1+ column
    elif 'llama3' in family_lower:
        return 3.0
    elif 'llama2' in family_lower:
        return 2.0
    elif 'llama1' in family_lower:
        return 1.0
    
    return 0  # Unknown


def determine_llama_model_type(family: str, size: str) -> str:
    """Determine model type from CSV config, with fallback logic."""
    metadata = get_model_metadata(family, size)
    if metadata is not None and pd.notna(metadata.get('model_type')):
        return metadata['model_type']
    
    # Fallback logic based on family name
    family_lower = family.lower()
    if 'therapy' in family_lower:
        return 'Therapy'
    elif 'mental' in family_lower:
        return 'Mental Health'
    elif 'guard' in family_lower:
        return 'Guard'
    elif 'medical' in family_lower or 'med' in family_lower:
        return 'Medical'
    elif '-it' in size or 'instruct' in size.lower() or 'chat' in size.lower():
        return 'IT'
    else:
        return 'PT'


def get_llama_config(figsize=(16, 18)) -> FacetPlotConfig:
    """Get the configuration for Llama facet plots."""
    return FacetPlotConfig(
        family_filter_fn=filter_llama_models,
        version_fn=determine_llama_version,
        model_type_fn=determine_llama_model_type,
        param_fn=get_param_billions_from_config,
        versions=[1.0, 2.0, 3.0, 3.5],
        version_labels=['Llama 1', 'Llama 2', 'Llama 3.0', 'Llama 3.1+'],
        figsize=figsize,
        x_lim=(0.5, 100),
        x_ticks=[1, 3, 8, 13, 70],
        x_tick_labels=['1', '3', '8', '13', '70'],
        type_colors={
            'IT': '#2E86AB',
            'PT': '#52B788',
            'Mental Health': '#C73E1D',
            'Guard': '#7B2D8E',
            'Medical': '#F18F01',
        },
        type_markers={
            'IT': 'o',
            'PT': 's',
            'Mental Health': 'P',
            'Guard': '^',
            'Medical': 'h',
        },
        type_display_labels={
            'IT': 'Instruct Tune',
            'PT': 'Base Model',
            'Mental Health': 'Mental Health',
            'Guard': 'Guard*',  # Asterisk indicates modified parsing
            'Medical': 'Medical',
        },
        guard_metrics_fn=compute_llama_guard_metrics,
        plot_name='Llama',
    )


def create_llama_version_plot(metrics_csv: str, output_path: str, 
                               figsize=(16, 18), title=None) -> None:
    """Create the Llama version faceted plot.
    
    Args:
        metrics_csv: Path to comprehensive metrics CSV
        output_path: Path to save the plot
        figsize: Figure size tuple (width, height)
        title: Optional overall title for the plot
    """
    df = pd.read_csv(metrics_csv)
    config = get_llama_config(figsize=figsize)
    create_facet_plot(df, config, output_path, title=title)


def main():
    parser = argparse.ArgumentParser(description='Create Llama version facet plot')
    parser.add_argument('--metrics-csv', required=True, help='Path to comprehensive_metrics.csv')
    parser.add_argument('--output', default='llama_version_facet_plot.png', help='Output path')
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 18], help='Figure size')
    parser.add_argument('--title', type=str, default=None, help='Overall plot title')
    
    args = parser.parse_args()
    create_llama_version_plot(args.metrics_csv, args.output, tuple(args.figsize), title=args.title)


if __name__ == "__main__":
    main()
