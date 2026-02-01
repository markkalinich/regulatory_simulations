#!/usr/bin/env python3
"""Combine metrics from all three tasks into a single spreadsheet.

This script finds the most recent experiment results for each task and combines them.
"""
import pandas as pd
from pathlib import Path
import argparse

def get_latest_experiment_dir(base_path: Path, task_prefix: str) -> Path:
    """Find the most recent experiment directory for a task."""
    import re
    task_dir = base_path / task_prefix
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    # Find all experiment directories (format: YYYYMMDD_HHMMSS_*)
    # Only include directories that start with a date pattern
    date_pattern = re.compile(r'^\d{8}_\d{6}')
    exp_dirs = sorted(
        [d for d in task_dir.iterdir() if d.is_dir() and date_pattern.match(d.name)],
        key=lambda x: x.name,
        reverse=True
    )
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {task_dir}")
    
    return exp_dirs[0]

def main():
    parser = argparse.ArgumentParser(description='Combine metrics from all three tasks')
    parser.add_argument('--si-dir', type=str, help='Override path to SI experiment dir')
    parser.add_argument('--tr-dir', type=str, help='Override path to TR experiment dir')
    parser.add_argument('--te-dir', type=str, help='Override path to TE experiment dir')
    parser.add_argument('--output', type=str, default='data/inputs/model_results/all_models_all_tasks.csv',
                       help='Output path for combined CSV')
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    results_base = root / 'results' / 'individual_prediction_performance'
    
    # Find or use provided experiment directories
    if args.si_dir:
        si_dir = Path(args.si_dir)
    else:
        si_dir = get_latest_experiment_dir(results_base, 'suicidal_ideation')
    
    if args.tr_dir:
        tr_dir = Path(args.tr_dir)
    else:
        tr_dir = get_latest_experiment_dir(results_base, 'therapy_request')
    
    if args.te_dir:
        te_dir = Path(args.te_dir)
    else:
        te_dir = get_latest_experiment_dir(results_base, 'therapy_engagement')
    
    files = {
        'suicidal_ideation': si_dir / 'tables' / 'comprehensive_metrics.csv',
        'therapy_request': tr_dir / 'tables' / 'comprehensive_metrics.csv',
        'therapy_engagement': te_dir / 'tables' / 'comprehensive_metrics.csv',
    }
    
    print("Using experiment directories:")
    for task, path in files.items():
        print(f"  {task}: {path.parent.parent.name}")

    # Columns to keep
    keep_cols = ['model_family', 'model_size', 'parse_success_rate', 'sensitivity', 'specificity', 'accuracy', 'f1_score']
    
    # Load and combine
    dfs = []
    for task, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        df = pd.read_csv(path, usecols=keep_cols)
        df.insert(0, 'task', task)
        dfs.append(df)
        print(f"  Loaded {len(df)} models for {task}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(['model_family', 'model_size', 'task']).reset_index(drop=True)
    
    # Save
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows to {out_path}")
    
    # Summary
    unique_models = combined.groupby(['model_family', 'model_size']).size().reset_index()
    print(f"Total unique models: {len(unique_models)}")

if __name__ == '__main__':
    main()
