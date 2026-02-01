#!/usr/bin/env python3
"""
Qwen Version Facet Plot - SI Detection Performance

Columns = Qwen version (1.x, 2, 3)
Rows = Metrics (Parse%, Sensitivity, Specificity, Accuracy, F1)
X-axis = Log10 model parameters (billions)
Shape = Model type (IT, PT, Medical, Mental Health, Guard)

Only includes Qwen-family models

Model metadata is loaded from config/models_config.csv
"""

import pandas as pd
from pathlib import Path
import argparse

from facet_plot_base import FacetPlotConfig, create_facet_plot
from facet_plot_utils import (
    get_model_metadata,
    get_param_billions_from_config,
    compute_qwen_guard_metrics,
)


# =============================================================================
# Qwen-Specific Configuration
# =============================================================================

def filter_qwen_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only Qwen-family models."""
    return df[df['model_family'].str.lower().str.contains('qwen', na=False)]


def determine_qwen_version(family: str, size: str) -> float:
    """Determine Qwen generation from CSV config or family name."""
    metadata = get_model_metadata(family, size)
    if metadata is not None and pd.notna(metadata.get('version')):
        version = metadata['version']
        # Extract major version number
        if isinstance(version, (int, float)):
            return int(version)
        version_str = str(version)
        if version_str.startswith('3'):
            return 3
        elif version_str.startswith('2'):
            return 2
        elif version_str.startswith('1'):
            return 1
    
    # Fallback: try to infer from family name
    family_lower = family.lower()
    if 'qwen3' in family_lower or 'qwen-3' in family_lower:
        return 3
    elif 'qwen2' in family_lower or 'qwen-2' in family_lower:
        return 2
    elif 'qwen1' in family_lower or 'qwen-1' in family_lower or 'qwen1.5' in family_lower:
        return 1
    
    # Default: check gemma_generation column (reused for qwen in some cases)
    if metadata is not None and pd.notna(metadata.get('gemma_generation')):
        return int(metadata['gemma_generation'])
    
    return 0  # Unknown


def determine_qwen_model_type(family: str, size: str) -> str:
    """Determine model type from CSV config, with fallback logic."""
    metadata = get_model_metadata(family, size)
    if metadata is not None and pd.notna(metadata.get('model_type')):
        return metadata['model_type']
    
    # Fallback logic based on family name
    family_lower = family.lower()
    if 'medical' in family_lower or 'med' in family_lower:
        return 'Medical'
    elif 'mental' in family_lower or 'therapy' in family_lower or 'depression' in family_lower:
        return 'Mental Health'
    elif 'guard' in family_lower:
        return 'Guard'
    elif '-it' in size or 'instruct' in size.lower() or 'chat' in size.lower():
        return 'IT'
    else:
        return 'PT'


def get_qwen_config(figsize=(14, 18)) -> FacetPlotConfig:
    """Get the configuration for Qwen facet plots."""
    return FacetPlotConfig(
        family_filter_fn=filter_qwen_models,
        version_fn=determine_qwen_version,
        model_type_fn=determine_qwen_model_type,
        param_fn=get_param_billions_from_config,
        versions=[1, 2, 3],
        version_labels=['Qwen 1.x', 'Qwen 2', 'Qwen 3'],
        figsize=figsize,
        x_lim=(0.3, 100),
        x_ticks=[0.5, 1, 3, 10, 30],
        x_tick_labels=['0.5', '1', '3', '10', '30'],
        type_colors={
            'IT': '#2E86AB',
            'PT': '#52B788',
            'Medical': '#F18F01',
            'Guard': '#7B2D8E',
            'Mental Health': '#C73E1D',
        },
        type_markers={
            'IT': 'o',
            'PT': 's',
            'Medical': 'D',
            'Guard': '^',
            'Mental Health': 'P',
        },
        type_display_labels={
            'IT': 'Instruct Tune',
            'PT': 'Base Model',
            'Medical': 'Medical',
            'Guard': 'Guard*',  # Asterisk indicates modified parsing
            'Mental Health': 'Mental Health',
        },
        guard_metrics_fn=compute_qwen_guard_metrics,
        plot_name='Qwen',
    )


def create_qwen_version_plot(metrics_csv: str, output_path: str, 
                              figsize=(14, 18), title=None) -> None:
    """Create the Qwen version faceted plot.
    
    Args:
        metrics_csv: Path to comprehensive metrics CSV
        output_path: Path to save the plot
        figsize: Figure size tuple (width, height)
        title: Optional overall title for the plot
    """
    df = pd.read_csv(metrics_csv)
    config = get_qwen_config(figsize=figsize)
    create_facet_plot(df, config, output_path, title=title)


def main():
    parser = argparse.ArgumentParser(description='Create Qwen version facet plot')
    parser.add_argument('--metrics-csv', required=True, help='Path to comprehensive_metrics.csv')
    parser.add_argument('--output', default='qwen_version_facet_plot.png', help='Output path')
    parser.add_argument('--figsize', nargs=2, type=float, default=[14, 18], help='Figure size')
    parser.add_argument('--title', type=str, default=None, help='Overall plot title')
    
    args = parser.parse_args()
    create_qwen_version_plot(args.metrics_csv, args.output, tuple(args.figsize), title=args.title)


if __name__ == "__main__":
    main()
