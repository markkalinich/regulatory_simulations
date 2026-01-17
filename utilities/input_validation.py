#!/usr/bin/env python3
"""
Input Validation Module for Figure Generation Scripts

Provides validation functions to catch common data issues before they cause
silent failures in figure generation.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any


class InputValidationError(Exception):
    """Raised when input data fails validation."""
    pass


# =============================================================================
# Expected Columns for Key File Types
# =============================================================================

# comprehensive_metrics.csv from batch_results_analyzer.py
COMPREHENSIVE_METRICS_REQUIRED = [
    'model_family',
    'model_size', 
    'total_samples',
    'successful_parses',
    'parse_success_rate',
    'sensitivity',
    'specificity',
    'accuracy',
    'f1_score',
    'tp', 'tn', 'fp', 'fn',
    'total_positive',
    'total_negative',
]

# For risk analysis (P1/P2 plots) - subset of comprehensive_metrics
RISK_ANALYSIS_REQUIRED = [
    'model_family',
    'model_size',
    'sensitivity',
    'fn',
    'total_positive',
]


# =============================================================================
# Validation Functions
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    source_file: str = "unknown",
    check_empty: bool = True,
    check_nan_in_columns: Optional[List[str]] = None,
) -> None:
    """
    Validate a DataFrame has required columns and no critical issues.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must exist
        source_file: Description of source file (for error messages)
        check_empty: If True, raise error if DataFrame is empty
        check_nan_in_columns: List of columns where NaN is not allowed
        
    Raises:
        InputValidationError: If validation fails
    """
    # Check for empty DataFrame
    if check_empty and len(df) == 0:
        raise InputValidationError(f"Empty DataFrame from {source_file}")
    
    # Check required columns exist
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise InputValidationError(
            f"Missing required columns in {source_file}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Check for NaN in specified columns
    if check_nan_in_columns:
        for col in check_nan_in_columns:
            if col in df.columns and df[col].isna().any():
                nan_count = df[col].isna().sum()
                raise InputValidationError(
                    f"Found {nan_count} NaN values in column '{col}' from {source_file}"
                )


def validate_comprehensive_metrics(
    csv_path: Path,
    experiment_name: str = "",
    min_models: int = 1,
) -> pd.DataFrame:
    """
    Load and validate comprehensive_metrics.csv file.
    
    Args:
        csv_path: Path to comprehensive_metrics.csv
        experiment_name: Name of experiment (for error messages)
        min_models: Minimum number of models expected
        
    Returns:
        Validated DataFrame
        
    Raises:
        InputValidationError: If validation fails
    """
    source = f"{experiment_name} ({csv_path})" if experiment_name else str(csv_path)
    
    # Check file exists
    if not csv_path.exists():
        raise InputValidationError(f"File not found: {csv_path}")
    
    # Load and validate
    df = pd.read_csv(csv_path)
    
    validate_dataframe(
        df,
        required_columns=COMPREHENSIVE_METRICS_REQUIRED,
        source_file=source,
        check_empty=True,
        check_nan_in_columns=['sensitivity', 'specificity', 'accuracy', 'f1_score'],
    )
    
    # Check minimum models
    if len(df) < min_models:
        raise InputValidationError(
            f"Expected at least {min_models} models in {source}, found {len(df)}"
        )
    
    # Validate metric ranges (0-1)
    for col in ['parse_success_rate', 'sensitivity', 'specificity', 'accuracy']:
        if col in df.columns:
            if (df[col] < 0).any() or (df[col] > 1).any():
                raise InputValidationError(
                    f"Column '{col}' contains values outside [0, 1] range in {source}"
                )
    
    # Validate counts are non-negative integers
    for col in ['tp', 'tn', 'fp', 'fn', 'total_positive', 'total_negative']:
        if col in df.columns:
            if (df[col] < 0).any():
                raise InputValidationError(
                    f"Column '{col}' contains negative values in {source}"
                )
    
    return df


def validate_models_config(
    csv_path: Path,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load and validate models_config.csv file.
    
    Args:
        csv_path: Path to models_config.csv
        required_columns: Custom required columns (uses default if None)
        
    Returns:
        Validated DataFrame
    """
    if required_columns is None:
        required_columns = ['model_family', 'model_size', 'param_billions', 'path']
    
    if not csv_path.exists():
        raise InputValidationError(f"Models config not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    validate_dataframe(
        df,
        required_columns=required_columns,
        source_file=str(csv_path),
    )
    
    return df


def validate_figure_inputs(
    input_files: Dict[str, Path],
    min_models_per_experiment: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to validate all inputs for multi-experiment figures.
    
    Args:
        input_files: Dict mapping experiment name to comprehensive_metrics.csv path
        min_models_per_experiment: Minimum models expected per experiment
        
    Returns:
        Dict mapping experiment name to validated DataFrame
        
    Raises:
        InputValidationError: If any validation fails
    """
    validated = {}
    
    for exp_name, csv_path in input_files.items():
        validated[exp_name] = validate_comprehensive_metrics(
            csv_path,
            experiment_name=exp_name,
            min_models=min_models_per_experiment,
        )
    
    return validated


# =============================================================================
# Utility Functions
# =============================================================================

def summarize_validation(df: pd.DataFrame, name: str = "") -> str:
    """
    Generate a summary string of DataFrame for logging.
    
    Args:
        df: Validated DataFrame
        name: Name for display
        
    Returns:
        Summary string
    """
    prefix = f"[{name}] " if name else ""
    n_models = len(df)
    families = df['model_family'].nunique() if 'model_family' in df.columns else 0
    
    return f"{prefix}{n_models} models from {families} families"

