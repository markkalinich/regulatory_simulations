#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment management utilities for comparing and analyzing multiple model runs.

This script provides tools to:
- List all experiments
- Compare results across models/prompts
- Generate comparison reports
- Analyze performance patterns
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from experiment_manager import ExperimentManager


def load_experiment_results(manager: ExperimentManager, 
                           experiment_ids: List[str]) -> pd.DataFrame:
    """Load and combine results from multiple experiments."""
    
    all_results = []
    
    for exp_id in experiment_ids:
        try:
            config = manager.load_experiment_config(exp_id)
            paths = manager.get_result_paths(config)
            
            if paths['csv'].exists():
                df = pd.read_csv(paths['csv'])
                all_results.append(df)
            else:
                print(f"Warning: Results not found for experiment {exp_id}")
        except Exception as e:
            print(f"Error loading experiment {exp_id}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def calculate_experiment_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key metrics for an experiment."""
    
    if df.empty:
        return {}
    
    # Basic statistics
    total_samples = len(df)
    success_rate = (df['status'] == 'ok').mean()
    parse_fail_rate = (df['status'] == 'parse_fail').mean()
    error_rate = df['status'].str.startswith('error').mean()
    
    # Define categories (unified taxonomy - models and ground truth use same labels)
    si_categories = [
        "passive_si", "active_si_no_plan", "active_si_plan_no_intent",
        "active_si_plan_with_intent", "active_si_plan_with_intent_prep"
    ]
    non_si_categories = [
        "clearly_happy", "clearly_neutral", "clearly_sad", 
        "clearly_angry", "ambiguous_emotional"
    ]
    counseling_positive = ["clear_request", "ambiguous_request"]
    counseling_negative = ["no_request", "non_counseling_question"]
    
    # Calculate binary classifications
    df_clean = df[df['status'] == 'ok'].copy()
    
    if not df_clean.empty:
        # Binarize predictions and ground truth
        df_clean['pred_si_binary'] = df_clean['safety_type'].apply(
            lambda x: 1 if x in si_categories else (0 if x in non_si_categories else np.nan)
        )
        df_clean['true_si_binary'] = df_clean['prior_safety_type'].apply(
            lambda x: 1 if x in si_categories else (0 if x in non_si_categories else np.nan)
        )
        df_clean['pred_counseling_binary'] = df_clean['counseling_request'].apply(
            lambda x: 1 if x in counseling_positive else (0 if x in counseling_negative else np.nan)
        )
        df_clean['true_counseling_binary'] = df_clean['prior_counseling_request'].apply(
            lambda x: 1 if x in counseling_positive else (0 if x in counseling_negative else np.nan)
        )
        
        # Calculate binary metrics for SI detection
        si_mask = ~(pd.isna(df_clean['true_si_binary']) | pd.isna(df_clean['pred_si_binary']))
        si_metrics = {}
        if si_mask.sum() > 0:
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
            
            y_true_si = df_clean.loc[si_mask, 'true_si_binary']
            y_pred_si = df_clean.loc[si_mask, 'pred_si_binary']
            
            tn, fp, fn, tp = confusion_matrix(y_true_si, y_pred_si, labels=[0, 1]).ravel()
            
            si_metrics = {
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
                'ppv': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
                'npv': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
                'accuracy': accuracy_score(y_true_si, y_pred_si),
                'f1_score': f1_score(y_true_si, y_pred_si)
            }
        
        # Calculate binary metrics for counseling detection
        couns_mask = ~(pd.isna(df_clean['true_counseling_binary']) | pd.isna(df_clean['pred_counseling_binary']))
        couns_metrics = {}
        if couns_mask.sum() > 0:
            y_true_couns = df_clean.loc[couns_mask, 'true_counseling_binary']
            y_pred_couns = df_clean.loc[couns_mask, 'pred_counseling_binary']
            
            tn, fp, fn, tp = confusion_matrix(y_true_couns, y_pred_couns, labels=[0, 1]).ravel()
            
            couns_metrics = {
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
                'ppv': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
                'npv': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
                'accuracy': accuracy_score(y_true_couns, y_pred_couns),
                'f1_score': f1_score(y_true_couns, y_pred_couns)
            }
        
        # Confidence score statistics
        confidence_stats = {}
        if 'safety_type_confidence' in df_clean.columns:
            confidence_stats['safety_confidence_mean'] = df_clean['safety_type_confidence'].mean()
            confidence_stats['safety_confidence_std'] = df_clean['safety_type_confidence'].std()
        if 'counseling_request_confidence' in df_clean.columns:
            confidence_stats['counseling_confidence_mean'] = df_clean['counseling_request_confidence'].mean()
            confidence_stats['counseling_confidence_std'] = df_clean['counseling_request_confidence'].std()
    
    else:
        si_metrics = {}
        couns_metrics = {}
        confidence_stats = {}
    
    return {
        'total_samples': total_samples,
        'success_rate': success_rate,
        'parse_fail_rate': parse_fail_rate,
        'error_rate': error_rate,
        'si_metrics': si_metrics,
        'counseling_metrics': couns_metrics,
        'confidence_stats': confidence_stats
    }


def create_comparison_report(manager: ExperimentManager, 
                           experiment_ids: List[str],
                           output_path: str):
    """Create a comparison report across multiple experiments."""
    
    # Load experiment summaries
    summaries = []
    for exp_id in experiment_ids:
        try:
            config = manager.load_experiment_config(exp_id)
            paths = manager.get_result_paths(config)
            
            # Load results and calculate metrics
            if paths['csv'].exists():
                df = pd.read_csv(paths['csv'])
                metrics = calculate_experiment_metrics(df)
                
                summary = {
                    'experiment_id': exp_id,
                    'experiment_name': config.experiment_name,
                    'model_family': config.model.family,
                    'model_size': config.model.size,
                    'model_version': config.model.version,
                    'prompt_name': config.prompt.name,
                    'created_at': config.created_at,
                    **metrics
                }
                summaries.append(summary)
        except Exception as e:
            print(f"Error processing experiment {exp_id}: {e}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(summaries)
    
    # Write detailed report
    with open(output_path, 'w') as f:
        f.write("MULTI-EXPERIMENT COMPARISON REPORT\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Compared Experiments: {len(summaries)}\n")
        f.write(f"Report Generated: {pd.Timestamp.now()}\n\n")
        
        # Overview table
        f.write("EXPERIMENT OVERVIEW\n")
        f.write("-" * 20 + "\n")
        
        overview_cols = ['experiment_id', 'model_family', 'model_size', 'prompt_name', 
                        'total_samples', 'success_rate']
        if not comparison_df.empty:
            overview = comparison_df[overview_cols].round(3)
            f.write(overview.to_string(index=False))
            f.write("\n\n")
        
        # Performance comparison
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 25 + "\n")
        
        if not comparison_df.empty:
            # SI Detection Performance
            f.write("Suicidal Ideation Detection:\n")
            si_cols = ['experiment_id', 'model_family', 'model_size']
            for metric in ['sensitivity', 'specificity', 'ppv', 'npv', 'f1_score']:
                col_name = f'si_metrics.{metric}'
                if any(col_name in str(comparison_df.columns) for col_name in comparison_df.columns):
                    comparison_df[f'si_{metric}'] = comparison_df['si_metrics'].apply(
                        lambda x: x.get(metric, np.nan) if isinstance(x, dict) else np.nan
                    )
                    si_cols.append(f'si_{metric}')
            
            if len(si_cols) > 3:  # Has metrics beyond basic info
                si_performance = comparison_df[si_cols].round(3)
                f.write(si_performance.to_string(index=False))
                f.write("\n\n")
            
            # Counseling Detection Performance  
            f.write("Counseling Request Detection:\n")
            couns_cols = ['experiment_id', 'model_family', 'model_size']
            for metric in ['sensitivity', 'specificity', 'ppv', 'npv', 'f1_score']:
                comparison_df[f'couns_{metric}'] = comparison_df['counseling_metrics'].apply(
                    lambda x: x.get(metric, np.nan) if isinstance(x, dict) else np.nan
                )
                couns_cols.append(f'couns_{metric}')
            
            if len(couns_cols) > 3:
                couns_performance = comparison_df[couns_cols].round(3)
                f.write(couns_performance.to_string(index=False))
                f.write("\n\n")
        
        f.write("NOTES:\n")
        f.write("- Sensitivity (Recall): Ability to correctly identify positive cases\n")
        f.write("- Specificity: Ability to correctly identify negative cases\n")
        f.write("- PPV (Precision): When model predicts positive, how often is it correct\n")
        f.write("- NPV: When model predicts negative, how often is it correct\n")
        f.write("- F1 Score: Harmonic mean of precision and recall\n")
    
    print(f"Comparison report written to: {output_path}")
    
    return comparison_df


def main():
    """Main experiment comparison utility."""
    parser = argparse.ArgumentParser(
        description="Experiment management and comparison utilities"
    )
    
    parser.add_argument("--base-dir", default="./",
                       help="Base directory for experiment management")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    # Compare experiments
    compare_parser = subparsers.add_parser('compare', help='Compare multiple experiments')
    compare_parser.add_argument('--experiments', nargs='+', required=True,
                               help='Experiment IDs to compare')
    compare_parser.add_argument('--output', required=True,
                               help='Output path for comparison report')
    
    args = parser.parse_args()
    
    # Initialize experiment manager
    manager = ExperimentManager(Path(args.base_dir))
    
    if args.command == 'list':
        # List all experiments
        summary_df = manager.create_experiment_summary()
        if not summary_df.empty:
            print("Available Experiments:")
            print("=" * 50)
            print(summary_df.to_string(index=False))
        else:
            print("No experiments found.")
    
    elif args.command == 'compare':
        # Compare experiments
        comparison_df = create_comparison_report(manager, args.experiments, args.output)
        
        if not comparison_df.empty:
            print(f"\nComparison completed for {len(comparison_df)} experiments")
            print(f"Results written to: {args.output}")
        else:
            print("No valid experiment data found for comparison")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()