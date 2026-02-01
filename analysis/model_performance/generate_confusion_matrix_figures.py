#!/usr/bin/env python3
"""
Confusion Matrix Figure Generator with Provenance Tracking

Generates binary confusion matrix grids for each task type:
- Suicidal Ideation (SI vs non-SI)
- Therapy Request (explicit request vs non-request)
- Therapy Engagement (therapeutic vs non-therapeutic)

This script wraps the existing confusion matrix functions and adds
provenance tracking for reproducibility.

Usage:
    python generate_confusion_matrix_figures.py \\
        --experiment-dir <path_to_experiment_dir> \\
        --task-type suicidal_ideation \\
        --output-dir <output_path>
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utilities.figure_provenance import FigureProvenanceTracker
from analysis.model_performance.confusion_matrices import (
    create_binary_confusion_matrix_grid,
    create_all_confusion_matrices,
)
# Import centralized category definitions (single source of truth)
from config.regulatory_paper_parameters import (
    BINARY_POSITIVE_CATEGORIES,
    GROUND_TRUTH_POSITIVE_CATEGORIES,
)

# Task-specific configurations
# NOTE: binary_positive_categories and ground_truth_positive_categories are
# imported from config/regulatory_paper_parameters.py for consistency
TASK_CONFIGS = {
    'suicidal_ideation': {
        'binary_positive_categories': BINARY_POSITIVE_CATEGORIES['suicidal_ideation'],
        'ground_truth_positive_categories': GROUND_TRUTH_POSITIVE_CATEGORIES['suicidal_ideation'],
        'binary_classification_name': 'SI',
        'multiclass_labels': ['clearly_neutral', 'clearly_happy', 'clearly_sad',
                              'clearly_angry', 'ambiguous_emotional',
                              'passive_si', 'active_si_abstract',
                              'active_si_plan_no_intent', 'active_si_plan_with_intent',
                              'preparatory_si', 'parse_fail'],
    },
    'therapy_request': {
        'binary_positive_categories': BINARY_POSITIVE_CATEGORIES['therapy_request'],
        'ground_truth_positive_categories': GROUND_TRUTH_POSITIVE_CATEGORIES['therapy_request'],
        'binary_classification_name': 'Therapy Request',
        'multiclass_labels': ['declarative', 'non_therapeutic_question',
                              'explicit_therapy_request', 'parse_fail'],
    },
    'therapy_engagement': {
        'binary_positive_categories': BINARY_POSITIVE_CATEGORIES['therapy_engagement'],
        'ground_truth_positive_categories': GROUND_TRUTH_POSITIVE_CATEGORIES['therapy_engagement'],
        'binary_classification_name': 'Therapy Engagement',
        'multiclass_labels': ['non_therapeutic', 'ambiguous_engagement',
                              'simulated_therapy', 'parse_fail'],
    },
}


def load_experiment_results(experiment_dir: Path, models_config: Optional[str] = None) -> pd.DataFrame:
    """
    Load model outputs from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory containing model_outputs/
        models_config: Optional path to models config CSV. If provided, filters to those models.
                      If None, loads all models found.
    """
    model_outputs_dir = experiment_dir / "model_outputs"
    
    if not model_outputs_dir.exists():
        raise FileNotFoundError(f"Model outputs directory not found: {model_outputs_dir}")
    
    # Build allowed models set from config if provided
    allowed_models = None
    allowed_models_base_sizes = None  # For backwards compatibility with old naming
    if models_config:
        import pandas as pd
        config_df = pd.read_csv(models_config)
        config_df = config_df[config_df.get('enabled', True) != False]  # Only enabled models
        allowed_models = set()
        allowed_models_base_sizes = {}  # Maps (family, base_size) -> full_size for fallback
        for _, row in config_df.iterrows():
            # Normalize family name for matching (llama3.2 -> llama)
            family = normalize_family_name(row['family'])
            allowed_models.add((family, row['size']))
            # Also store base size for backwards compatibility (e.g., "270m-it" -> "270m")
            base_size = row['size'].split('-')[0]
            allowed_models_base_sizes[(family, base_size)] = row['size']
        print(f"Filtering to {len(allowed_models)} models from {models_config}")
    
    all_results = []
    
    for csv_file in sorted(model_outputs_dir.glob("*.csv")):
        # Extract model family and size from filename
        # Format: family_size_*.csv or family_size.csv
        parts = csv_file.stem.split('_')
        if len(parts) >= 2:
            model_family = parts[0]
            model_size = parts[1]
            
            # Filter if config provided (normalize family for matching)
            if allowed_models:
                normalized_family = normalize_family_name(model_family)
                # Try exact match first
                if (normalized_family, model_size) in allowed_models:
                    pass  # Matched exactly
                # Fallback: check if this is an old naming style (e.g., "270m" matches "270m-it")
                elif allowed_models_base_sizes and (normalized_family, model_size) in allowed_models_base_sizes:
                    pass  # Matched via base size
                else:
                    continue  # No match, skip this model
            
            df = pd.read_csv(csv_file)
            df['model_family'] = model_family  # Keep original for display
            df['model_size'] = model_size
            all_results.append(df)
    
    if not all_results:
        raise ValueError(f"No matching CSV files found in {model_outputs_dir}")
    
    print(f"Loaded {len(all_results)} models")
    return pd.concat(all_results, ignore_index=True)


def normalize_family_name(family: str) -> str:
    """Normalize model family names (llama2, llama3.1, llama3.2 → llama)."""
    family_lower = family.lower()
    if family_lower.startswith('llama'):
        return 'llama'
    elif family_lower.startswith('qwen'):
        return 'qwen'
    elif family_lower.startswith('gemma'):
        return 'gemma'
    return family_lower


def size_to_billions(s: str) -> float:
    """Convert model size string to billions for sorting (270m → 0.27, 1b → 1.0)."""
    import re
    s_clean = re.sub(r'-(it|pt)$', '', s.lower())  # Strip -it/-pt suffixes
    
    match = re.search(r'(\d+(?:\.\d+)?)\s*(m|b)?', s_clean)
    if match:
        num = float(match.group(1))
        unit = match.group(2) if match.group(2) else 'b'
        if unit == 'm':
            return num / 1000.0  # Convert millions to billions
        return num
    return 0


def get_model_families_from_results(results_df: pd.DataFrame) -> dict:
    """Extract model families and sizes from results DataFrame, with normalized family names."""
    model_families = {}
    
    for (family, size), _ in results_df.groupby(['model_family', 'model_size']):
        # Normalize family name (llama2, llama3.1 → llama)
        normalized_family = normalize_family_name(family)
        
        if normalized_family not in model_families:
            model_families[normalized_family] = []
        if size not in model_families[normalized_family]:
            model_families[normalized_family].append(size)
    
    # Sort sizes by billions (smallest to largest)
    for family in model_families:
        model_families[family] = sorted(model_families[family], key=size_to_billions)
    
    return model_families


def generate_confusion_matrix_figure(
    experiment_dir: Path,
    task_type: str,
    output_dir: Path,
    models_config: Optional[str] = None,
) -> Optional[Path]:
    """
    Generate binary confusion matrix grid for a task.
    
    Args:
        experiment_dir: Path to experiment directory with model_outputs/
        task_type: One of 'suicidal_ideation', 'therapy_request', 'therapy_engagement'
        output_dir: Directory to save output figures
        models_config: Optional path to models config CSV to filter models
        
    Returns:
        Path to the generated figure
    """
    if task_type not in TASK_CONFIGS:
        raise ValueError(f"Unknown task type: {task_type}. "
                        f"Must be one of: {list(TASK_CONFIGS.keys())}")
    
    config = TASK_CONFIGS[task_type]
    
    # Initialize provenance tracker
    tracker = FigureProvenanceTracker(
        figure_name=f"{task_type}_confusion_matrix",
        base_dir=output_dir,
    )
    
    # Load results
    print(f"Loading results from: {experiment_dir}")
    results_df = load_experiment_results(experiment_dir, models_config=models_config)
    
    # Track input (use experiment dir, not model_outputs which is a directory)
    # Just record the path as metadata rather than trying to read directory
    tracker.set_analysis_parameters(
        input_experiment_dir=str(experiment_dir / "model_outputs"),
        input_description=f"Model outputs for {task_type} task",
    )
    
    # Get model families
    model_families = get_model_families_from_results(results_df)
    print(f"Found {len(model_families)} model families:")
    for family, sizes in model_families.items():
        print(f"  {family}: {sizes}")
    
    # Set analysis parameters
    tracker.set_analysis_parameters(
        task_type=task_type,
        experiment_dir=str(experiment_dir),
        binary_classification_name=config['binary_classification_name'],
        model_families=model_families,
        total_predictions=len(results_df),
    )
    
    # Generate confusion matrix grid
    print(f"Generating confusion matrix grid...")
    
    # Use the output directory directly (not a subdirectory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_binary_confusion_matrix_grid(
        results_df=results_df,
        model_families=model_families,
        experiment_type=task_type.replace('suicidal_ideation', 'suicide_detection'),
        binary_positive_categories=config['binary_positive_categories'],
        ground_truth_positive_categories=config['ground_truth_positive_categories'],
        binary_classification_name=config['binary_classification_name'],
        output_dir=output_dir,
    )
    
    # Track output
    output_file = output_dir / "confusion_matrices" / "binary_confusion_matrix_grid.png"
    if output_file.exists():
        # Move to output directory root with better name
        final_output = output_dir / f"{task_type}_binary_confusion_matrix_grid.png"
        import shutil
        shutil.copy(output_file, final_output)
        
        # Clean up the redundant subfolder created by underlying function
        confusion_subdir = output_dir / "confusion_matrices"
        if confusion_subdir.exists():
            shutil.rmtree(confusion_subdir)
        
        tracker.add_output_file(
            final_output,
            file_type="figure",
            dpi=300,
        )
        
        # Save provenance
        tracker.save_provenance()
        
        print(f"✓ Saved: {final_output}")
        return final_output
    else:
        print(f"⚠ Output file not found: {output_file}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix figures with provenance tracking"
    )
    
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment directory containing model_outputs/"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=['suicidal_ideation', 'therapy_request', 'therapy_engagement'],
        help="Type of classification task"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output figures"
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default=None,
        help="Path to models config CSV to filter to specific models"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONFUSION MATRIX FIGURE GENERATOR")
    print("=" * 70)
    
    try:
        output_path = generate_confusion_matrix_figure(
            experiment_dir=Path(args.experiment_dir),
            task_type=args.task_type,
            output_dir=Path(args.output_dir),
            models_config=args.models_config,
        )
        
        if output_path:
            print("\n" + "=" * 70)
            print("COMPLETE!")
            print(f"Output: {output_path}")
            print("=" * 70)
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

