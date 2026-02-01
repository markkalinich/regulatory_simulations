#!/usr/bin/env python3
"""
Gemma Version Facet Plot - SI Detection Performance

Columns = Gemma version (1, 2, 3, 3n)
Rows = Metrics (Parse%, Sensitivity, Specificity, Accuracy, F1)
X-axis = Log10 model parameters (billions)
Shape = Model type (IT, PT, MedGemma, ShieldGemma, Mental Health)

Only includes Gemma-family models (gemma, gemma2, medgemma, shieldgemma, mental_health)

Model metadata (gemma_generation, model_type) is loaded from config/models_config.csv
"""

import pandas as pd
from pathlib import Path
import argparse

from facet_plot_base import FacetPlotConfig, create_facet_plot
from facet_plot_utils import (
    get_model_metadata,
    get_param_billions_from_config,
    compute_shieldgemma_metrics,
)


# =============================================================================
# Gemma-Specific Configuration
# =============================================================================

GEMMA_FAMILIES = ['gemma', 'gemma1', 'gemma2', 'gemma3n', 'medgemma', 'shieldgemma', 'mental_health', 'gemma_therapy']

# Mental health model version mapping (fallback when config lookup fails)
MENTAL_HEALTH_VERSION_MAP = {
    '270m': 3,           # Gemma 3 270M based
    '1b': 3,             # Gemma 3 1B based  
    '2b_v1': 1,          # Gemma 1 2B based
    '2b_v2': 1,          # Gemma 1 2B based
    '2b_v3': 1,          # Gemma 1 2B based
    '2b_v4': 1,          # Gemma 1 2B based
    '2b_v5': 1,          # Gemma 1 2B based
    '2b_therapist': 2,   # Gemma 2 2B based
}


def filter_gemma_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only Gemma-family models."""
    return df[df['model_family'].isin(GEMMA_FAMILIES)]


def determine_gemma_version(family: str, size: str) -> float:
    """Determine Gemma generation from CSV config, with fallback logic."""
    # Check family name first - gemma3n gets its own column
    if family == 'gemma3n':
        return 3.5  # Gemma 3n gets its own column
    
    metadata = get_model_metadata(family, size)
    if metadata is not None and pd.notna(metadata.get('gemma_generation')):
        gen = metadata['gemma_generation']
        # Handle string values like '3n' for Gemma 3n models
        if isinstance(gen, str):
            if gen == '3n':
                return 3.5  # Give Gemma 3n its own column
            try:
                return int(float(gen))
            except:
                return 0
        return int(float(gen))
    
    # Fallback for mental_health models with short size names
    if family == 'mental_health' and size in MENTAL_HEALTH_VERSION_MAP:
        return MENTAL_HEALTH_VERSION_MAP[size]
    
    # Fallback: return 0 (unknown) if not in CSV
    return 0


def determine_gemma_model_type(family: str, size: str) -> str:
    """Determine model type from CSV config, with fallback logic."""
    metadata = get_model_metadata(family, size)
    if metadata is not None and pd.notna(metadata.get('model_type')):
        return metadata['model_type']
    
    # Fallback logic if not in CSV
    if family == 'medgemma':
        return 'MedGemma'
    elif family == 'shieldgemma':
        return 'ShieldGemma'
    elif family in ['mental_health', 'gemma_therapy']:
        return 'Mental Health'
    elif '-pt' in size or size.endswith('-pt') or size == '4b':
        return 'PT'
    else:
        return 'IT'


def get_gemma_config(figsize=(18, 18)) -> FacetPlotConfig:
    """Get the configuration for Gemma facet plots."""
    return FacetPlotConfig(
        family_filter_fn=filter_gemma_models,
        version_fn=determine_gemma_version,
        model_type_fn=determine_gemma_model_type,
        param_fn=get_param_billions_from_config,
        versions=[1, 2, 3, 3.5],
        version_labels=['Gemma 1', 'Gemma 2', 'Gemma 3', 'Gemma 3n'],
        figsize=figsize,
        x_lim=(0.15, 50),
        x_ticks=[0.3, 1, 3, 10, 30],
        x_tick_labels=['0.3', '1', '3', '10', '30'],
        type_colors={
            'IT': '#2E86AB',
            'PT': '#52B788',
            'MedGemma': '#F18F01',
            'ShieldGemma': '#7B2D8E',
            'Mental Health': '#C73E1D',
        },
        type_markers={
            'IT': 'o',
            'PT': 's',
            'MedGemma': 'D',
            'ShieldGemma': '^',
            'Mental Health': 'P',
        },
        type_display_labels={
            'IT': 'Instruct Tune',
            'PT': 'Base Model',
            'MedGemma': 'MedGemma',
            'ShieldGemma': 'ShieldGemma',
            'Mental Health': 'Mental Health',
        },
        guard_metrics_fn=compute_shieldgemma_metrics,
        plot_name='Gemma',
    )


def create_gemma_version_plot(metrics_csv: str, output_path: str, 
                               figsize=(18, 18), title=None) -> None:
    """Create the Gemma version faceted plot.
    
    Args:
        metrics_csv: Path to comprehensive metrics CSV
        output_path: Path to save the plot
        figsize: Figure size tuple (width, height)
        title: Optional overall title for the plot
    """
    df = pd.read_csv(metrics_csv)
    config = get_gemma_config(figsize=figsize)
    create_facet_plot(df, config, output_path, title=title)


def main():
    parser = argparse.ArgumentParser(description='Create Gemma version facet plot')
    parser.add_argument('--metrics-csv', required=True, help='Path to comprehensive_metrics.csv')
    parser.add_argument('--output', default='gemma_version_facet_plot.png', help='Output path')
    parser.add_argument('--figsize', nargs=2, type=float, default=[18, 18], help='Figure size')
    parser.add_argument('--title', type=str, default=None, help='Overall plot title')
    
    args = parser.parse_args()
    create_gemma_version_plot(args.metrics_csv, args.output, tuple(args.figsize), title=args.title)


if __name__ == "__main__":
    main()
