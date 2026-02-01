"""
Analysis module for safety simulation experiments.

This module is organized into subfolders:
- data_validation/ - Psychiatrist review validation visualizations
- model_performance/ - Individual model analysis and metrics
- comparative_analysis/ - Cross-model comparisons and plots
- statistics/ - Statistical calculations
"""

# Importing from submodules for backwards compatibility
from .model_performance.metrics_calculator import (
    calculate_model_metrics,
    generate_metrics_for_all_models,
    determine_multiclass_labels
)

__all__ = [
    'calculate_model_metrics',
    'generate_metrics_for_all_models', 
    'determine_multiclass_labels'
]