#!/usr/bin/env python3
"""
Report Generator Module - Text report generation for analysis results.

This module provides functions to generate comprehensive analysis reports,
extracted from batch_results_analyzer.py for reusability.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def generate_comprehensive_report(metrics_df: pd.DataFrame,
                                report_title: str,
                                analysis_info: Dict[str, Any],
                                output_file: Path) -> None:
    """
    Generate comprehensive text analysis report.
    
    Args:
        metrics_df: DataFrame with model performance metrics
        report_title: Title for the report
        analysis_info: Dictionary with analysis metadata (timestamp, prompt_file, input_data)
        output_file: Path where to save the report
    """
    print("Generating analysis report...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(report_title)
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Analysis ID: {analysis_info.get('timestamp', 'N/A')}")
    report_lines.append(f"Prompt: {Path(analysis_info.get('prompt_file', '')).stem}")
    report_lines.append(f"Dataset: {Path(analysis_info.get('input_data', '')).stem}")
    report_lines.append("")
    
    # Executive Summary
    report_lines.extend(generate_executive_summary(metrics_df))
    report_lines.append("")
    
    # Family-wise analysis
    report_lines.extend(generate_family_analysis(metrics_df))
    
    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved analysis report: {output_file}")


def generate_executive_summary(metrics_df: pd.DataFrame) -> List[str]:
    """
    Generate executive summary section of the report.
    
    Args:
        metrics_df: DataFrame with model performance metrics
        
    Returns:
        List of report lines for executive summary
    """
    lines = []
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total models analyzed: {len(metrics_df)}")
    lines.append(f"Total samples per model: {metrics_df['total_samples'].iloc[0] if not metrics_df.empty else 'N/A'}")
    
    # Best performing models
    if not metrics_df.empty:
        best_accuracy = metrics_df.loc[metrics_df['accuracy'].idxmax()]
        best_f1 = metrics_df.loc[metrics_df['f1_score'].idxmax()]
        best_parse = metrics_df.loc[metrics_df['parse_success_rate'].idxmax()]
        
        lines.append(f"Best accuracy: {best_accuracy['model_family']} {best_accuracy['model_size']} ({best_accuracy['accuracy']:.3f})")
        lines.append(f"Best F1 score: {best_f1['model_family']} {best_f1['model_size']} ({best_f1['f1_score']:.3f})")
        lines.append(f"Best parsing: {best_parse['model_family']} {best_parse['model_size']} ({best_parse['parse_success_rate']:.3f})")
    
    return lines


def generate_family_analysis(metrics_df: pd.DataFrame) -> List[str]:
    """
    Generate family-wise analysis section of the report.
    
    Args:
        metrics_df: DataFrame with model performance metrics
        
    Returns:
        List of report lines for family analysis
    """
    lines = []
    
    # Family-wise analysis
    for family in ['gemma', 'qwen', 'llama']:
        family_data = metrics_df[metrics_df['model_family'] == family]
        if family_data.empty:
            continue
            
        lines.append(f"{family.upper()} FAMILY ANALYSIS")
        lines.append("-" * 40)
        
        for _, row in family_data.iterrows():
            lines.append(f"{row['model_size']}:")
            lines.append(f"  Parse Success: {row['parse_success_rate']:.1%}")
            lines.append(f"  Accuracy: {row['accuracy']:.1%}")
            lines.append(f"  Sensitivity: {row['sensitivity']:.1%}")
            lines.append(f"  Specificity: {row['specificity']:.1%}")
            lines.append(f"  F1 Score: {row['f1_score']:.1%}")
            lines.append("")
    
    return lines


def format_model_metrics(row: pd.Series) -> str:
    """
    Format model metrics for report display.
    
    Args:
        row: Single row from metrics DataFrame
        
    Returns:
        Formatted string with model metrics
    """
    return (f"{row['model_family'].upper()} {row['model_size']}: "
           f"Acc={row['accuracy']:.2%}, "
           f"F1={row['f1_score']:.2%}, "
           f"Parse={row['parse_success_rate']:.2%}")


def get_top_performers(metrics_df: pd.DataFrame, metric: str, n: int = 3) -> pd.DataFrame:
    """
    Get top N performing models for a specific metric.
    
    Args:
        metrics_df: DataFrame with model performance metrics
        metric: Metric column name to sort by
        n: Number of top performers to return
        
    Returns:
        DataFrame with top N performers
    """
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame")
    
    return metrics_df.nlargest(n, metric)