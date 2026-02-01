#!/usr/bin/env python3
"""
Heatmap and Difficult Statement Analysis Audit Script

Independently verifies Figures S8-S10 (per-statement accuracy heatmaps) and 
the difficult statement analysis by:

1. Recalculating correctness matrices from model output CSVs
2. Verifying difficult statement counts match summary files
3. Cross-checking with comprehensive_metrics.csv accuracy values
4. Validating figure provenance files

This script provides an audit trail for:
- Figures S8-S10: Per-statement/conversation accuracy heatmaps
- Difficult statement analysis used in manuscript Section 2

Usage:
    python utilities/heatmap_audit.py --paper-run-dir <path_to_pipeline_output>
    python utilities/heatmap_audit.py  # Uses most recent pipeline run

Output:
    heatmap_audit_report.csv - Detailed verification results
    heatmap_audit_summary.json - Summary with pass/fail status
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.classification_utils import (
    to_binary_si,
    to_binary_therapy_request,
    to_binary_therapy_engagement,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TASK_CONFIG = {
    'suicidal_ideation': {
        'ground_truth_col': 'prior_safety_type',
        'prediction_col': 'safety_type',
        'binary_converter': to_binary_si,
        'matrix_file': 'si_model_statement_correctness_matrix.csv',
        'info_file': 'si_statement_info.csv',
        'breakdown_file': 'suicidal_ideation_difficult_statements_breakdown.csv',
        'figure_name': 'figure_S8',
        'expected_samples': 450,
    },
    'therapy_request': {
        'ground_truth_col': 'prior_therapy_request',
        'prediction_col': 'therapy_request',
        'binary_converter': to_binary_therapy_request,
        'matrix_file': 'therapy_request_model_statement_correctness_matrix.csv',
        'info_file': 'therapy_request_statement_info.csv',
        'breakdown_file': 'therapy_request_difficult_statements_breakdown.csv',
        'figure_name': 'figure_S9',
        'expected_samples': 780,
    },
    'therapy_engagement': {
        'ground_truth_col': 'prior_therapy_engagement',
        'prediction_col': 'therapy_engagement',
        'binary_converter': to_binary_therapy_engagement,
        'matrix_file': 'therapy_engagement_model_conversation_correctness_matrix.csv',
        'info_file': 'therapy_engagement_conversation_info.csv',
        'breakdown_file': 'therapy_engagement_difficult_statements_breakdown.csv',
        'figure_name': 'figure_S10',
        'expected_samples': 420,
    },
}


def load_model_outputs(experiment_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all model output CSVs from experiment directory."""
    model_outputs = {}
    output_dir = experiment_dir / 'model_outputs'
    
    if not output_dir.exists():
        return model_outputs
    
    for csv_file in sorted(output_dir.glob('*.csv')):
        filename = csv_file.stem
        # Extract model name (first two underscore-separated parts)
        model_name = '_'.join(filename.split('_')[:2])
        
        if model_name not in model_outputs:
            df = pd.read_csv(csv_file)
            model_outputs[model_name] = df
    
    return model_outputs


def calculate_correctness_matrix(
    model_outputs: Dict[str, pd.DataFrame],
    ground_truth_col: str,
    prediction_col: str,
    binary_converter
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Independently calculate correctness matrix from model outputs.
    
    Returns:
        Tuple of (correctness_matrix, statement_info)
    """
    if not model_outputs:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get statement texts from first model (in order)
    first_model_df = list(model_outputs.values())[0]
    statement_texts = first_model_df['text'].tolist()
    
    results = {}
    
    for model_name, df in sorted(model_outputs.items()):
        correctness = []
        
        # Create lookup by text
        model_text_lookup = {row['text']: row for _, row in df.iterrows()}
        
        for stmt_text in statement_texts:
            if stmt_text not in model_text_lookup:
                correctness.append(np.nan)
                continue
            
            stmt_row = model_text_lookup[stmt_text]
            
            # Check for parse failure
            status = stmt_row.get('status', 'ok')
            if status != 'ok':
                correctness.append(0)  # Parse failure = incorrect
                continue
            
            ground_truth = stmt_row[ground_truth_col]
            prediction = stmt_row[prediction_col]
            
            # Convert to binary
            gt_binary = binary_converter(ground_truth)
            pred_binary = binary_converter(prediction)
            
            # 1 if correct, 0 if incorrect
            correctness.append(1 if pred_binary == gt_binary else 0)
        
        results[model_name] = correctness
    
    # Create matrix with statement indices as columns
    matrix_df = pd.DataFrame(results, index=range(len(statement_texts))).T
    
    # Create statement info
    info_data = []
    for idx, row in first_model_df.iterrows():
        info_data.append({
            'statement_index': len(info_data),
            'text': row['text'][:200] if len(row['text']) > 200 else row['text'],
            'ground_truth': row[ground_truth_col]
        })
    
    info_df = pd.DataFrame(info_data)
    
    return matrix_df, info_df


def calculate_difficult_statements(
    matrix_df: pd.DataFrame,
    info_df: pd.DataFrame,
    threshold: float = 0.5
) -> Tuple[int, float, Dict[str, int]]:
    """
    Calculate difficult statements (missed by >threshold of models).
    
    Returns:
        Tuple of (difficult_count, difficult_pct, category_breakdown)
    """
    if matrix_df.empty:
        return 0, 0.0, {}
    
    # Calculate miss rate per statement
    miss_rate = 1 - matrix_df.mean(axis=0)
    
    # Difficult = miss rate > threshold (strictly greater than)
    difficult_mask = miss_rate > threshold
    difficult_count = int(difficult_mask.sum())
    difficult_pct = difficult_count / matrix_df.shape[1] * 100
    
    # Get category breakdown
    difficult_ids = [int(x) for x in miss_rate[difficult_mask].index]
    difficult_info = info_df[info_df['statement_index'].isin(difficult_ids)]
    # Convert to int to avoid np.int64 comparison issues
    category_breakdown = {k: int(v) for k, v in difficult_info['ground_truth'].value_counts().items()}
    
    return difficult_count, difficult_pct, category_breakdown


def find_experiment_dirs(paper_run_dir: Path) -> Dict[str, Optional[Path]]:
    """Find experiment directories used for each task."""
    # These are stored in results/individual_prediction_performance/
    base_dir = PROJECT_ROOT / 'results' / 'individual_prediction_performance'
    
    task_subdirs = {
        'suicidal_ideation': 'suicidal_ideation',
        'therapy_request': 'therapy_request',
        'therapy_engagement': 'therapy_engagement',
    }
    
    experiment_dirs = {}
    
    for task_name, subdir in task_subdirs.items():
        task_base = base_dir / subdir
        if not task_base.exists():
            experiment_dirs[task_name] = None
            continue
        
        # Find the most recent _paper directory
        paper_dirs = sorted(
            [d for d in task_base.iterdir() if d.is_dir() and '_paper' in d.name],
            key=lambda x: x.name,
            reverse=True
        )
        
        if paper_dirs:
            experiment_dirs[task_name] = paper_dirs[0]
        else:
            # Fall back to most recent directory with model_outputs
            all_dirs = sorted(
                [d for d in task_base.iterdir() if d.is_dir() and (d / 'model_outputs').exists()],
                key=lambda x: x.name,
                reverse=True
            )
            experiment_dirs[task_name] = all_dirs[0] if all_dirs else None
    
    return experiment_dirs


def load_reported_matrix(review_stats_dir: Path, task_name: str) -> Optional[pd.DataFrame]:
    """Load reported correctness matrix from review_statistics."""
    config = TASK_CONFIG[task_name]
    matrix_path = review_stats_dir / config['matrix_file']
    
    if matrix_path.exists():
        return pd.read_csv(matrix_path, index_col=0)
    return None


def load_reported_info(review_stats_dir: Path, task_name: str) -> Optional[pd.DataFrame]:
    """Load reported statement info from review_statistics."""
    config = TASK_CONFIG[task_name]
    info_path = review_stats_dir / config['info_file']
    
    if info_path.exists():
        return pd.read_csv(info_path)
    return None


def load_reported_summary(difficult_dir: Path, task_name: str) -> Optional[pd.Series]:
    """Load reported difficult statements summary from breakdown file.
    
    The breakdown file now contains metadata columns (total_statements, total_models,
    difficult_count, difficult_pct) that were previously in a separate summary file.
    """
    config = TASK_CONFIG[task_name]
    breakdown_path = difficult_dir / config['breakdown_file']
    
    if breakdown_path.exists():
        df = pd.read_csv(breakdown_path)
        if len(df) > 0:
            # Extract summary info from first row (all rows have same metadata)
            first_row = df.iloc[0]
            return pd.Series({
                'task': first_row.get('task', task_name),
                'total_statements': first_row.get('total_statements'),
                'total_models': first_row.get('total_models'),
                'difficult_count': first_row.get('difficult_count'),
                'difficult_pct': first_row.get('difficult_pct'),
            })
    return None


def load_reported_breakdown(difficult_dir: Path, task_name: str) -> Optional[pd.DataFrame]:
    """Load reported difficult statements breakdown."""
    config = TASK_CONFIG[task_name]
    breakdown_path = difficult_dir / config['breakdown_file']
    
    if breakdown_path.exists():
        return pd.read_csv(breakdown_path)
    return None


def load_comprehensive_metrics(paper_run_dir: Path, task_name: str) -> Optional[pd.DataFrame]:
    """Load comprehensive metrics for accuracy comparison."""
    metrics_path = paper_run_dir / 'Data' / 'processed_data' / 'model_performance_metrics' / f'{task_name}_comprehensive_metrics.csv'
    
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    import hashlib
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_figure_provenance(output_dir: Path, figure_name: str) -> Optional[Dict]:
    """Load provenance JSON for a heatmap figure."""
    # Heatmaps are in results/model_performance_analysis/{date}/{timestamp}_{name}/
    base_dir = PROJECT_ROOT / 'results' / 'model_performance_analysis'
    
    if not base_dir.exists():
        return None
    
    # Find most recent date directory
    date_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    if not date_dirs:
        return None
    
    # Find the heatmap directory
    heatmap_name = figure_name.replace('figure_', '').lower() + '_correctness_heatmap'
    
    for date_dir in date_dirs:
        for subdir in date_dir.iterdir():
            if subdir.is_dir() and heatmap_name in subdir.name:
                provenance_file = subdir / f'{heatmap_name}_provenance.json'
                if provenance_file.exists():
                    with open(provenance_file, 'r') as f:
                        return json.load(f)
    
    return None


def run_audit(
    paper_run_dir: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Run full audit for Figures S8-S10 and difficult statement analysis.
    
    Returns DataFrame with audit results.
    """
    print("=" * 70)
    print("FIGURES S8-S10 & DIFFICULT STATEMENT ANALYSIS AUDIT")
    print("=" * 70)
    print(f"Paper run: {paper_run_dir}")
    print(f"Output: {output_path}")
    print()
    print("This audit verifies:")
    print("  - Figures S8-S10: Per-statement/conversation accuracy heatmaps")
    print("  - Difficult statement counts and category breakdowns")
    print("  - Threshold: miss_rate > 0.5 (missed by 8+ of 14 models)")
    print()
    
    # Find experiment directories
    experiment_dirs = find_experiment_dirs(paper_run_dir)
    
    # Review statistics directory
    review_stats_dir = PROJECT_ROOT / 'results' / 'review_statistics'
    
    # Difficult statement analysis directory
    difficult_dir = paper_run_dir / 'Data' / 'processed_data' / 'difficult_statement_analysis'
    
    audit_results = []
    task_summaries = {}
    
    for task_name, config in TASK_CONFIG.items():
        print(f"\n{'='*70}")
        print(f"TASK: {task_name.upper()} ({config['figure_name'].upper()})")
        print(f"{'='*70}")
        
        exp_dir = experiment_dirs.get(task_name)
        if exp_dir is None:
            print(f"  ERROR: No experiment directory found for {task_name}")
            continue
        
        print(f"Experiment dir: {exp_dir.name}")
        
        # Step 1: Load model outputs and calculate correctness matrix
        print("\nStep 1: Loading model outputs and calculating correctness matrix...")
        model_outputs = load_model_outputs(exp_dir)
        
        if not model_outputs:
            print(f"  ERROR: No model outputs found in {exp_dir / 'model_outputs'}")
            continue
        
        print(f"  Loaded {len(model_outputs)} models")
        
        # Calculate correctness matrix from raw model outputs
        calc_matrix, calc_info = calculate_correctness_matrix(
            model_outputs,
            config['ground_truth_col'],
            config['prediction_col'],
            config['binary_converter']
        )
        
        print(f"  Calculated matrix shape: {calc_matrix.shape} (models x statements)")
        
        # Step 2: Load reported matrix and compare
        print("\nStep 2: Comparing to reported correctness matrix...")
        reported_matrix = load_reported_matrix(review_stats_dir, task_name)
        
        matrix_match = False
        matrix_shape_match = False
        matrix_values_match = False
        
        if reported_matrix is not None:
            matrix_shape_match = calc_matrix.shape == reported_matrix.shape
            print(f"  Reported matrix shape: {reported_matrix.shape}")
            print(f"  Shape match: {'✅' if matrix_shape_match else '❌'}")
            
            if matrix_shape_match:
                # Compare values (both have same column types)
                calc_matrix.columns = calc_matrix.columns.astype(str)
                reported_matrix.columns = reported_matrix.columns.astype(str)
                
                # Sort both by index for comparison
                calc_sorted = calc_matrix.sort_index()
                reported_sorted = reported_matrix.sort_index()
                
                # Compare numeric values
                try:
                    matrix_values_match = np.allclose(
                        calc_sorted.values, 
                        reported_sorted.values, 
                        equal_nan=True
                    )
                except (ValueError, TypeError):
                    matrix_values_match = calc_sorted.equals(reported_sorted)
                
                print(f"  Values match: {'✅' if matrix_values_match else '❌'}")
                
                matrix_match = matrix_shape_match and matrix_values_match
        else:
            print("  ⚠️  Reported matrix not found")
        
        # Step 3: Verify model accuracies against comprehensive_metrics
        print("\nStep 3: Verifying model accuracies against comprehensive_metrics.csv...")
        comp_metrics = load_comprehensive_metrics(paper_run_dir, task_name)
        
        accuracy_checks = []
        if comp_metrics is not None and not calc_matrix.empty:
            for model_name in calc_matrix.index:
                matrix_accuracy = calc_matrix.loc[model_name].mean()
                
                # Find in comprehensive metrics
                parts = model_name.split('_', 1)
                if len(parts) == 2:
                    family, size = parts
                    metrics_row = comp_metrics[
                        (comp_metrics['model_family'] == family) & 
                        (comp_metrics['model_size'] == size)
                    ]
                    
                    if len(metrics_row) > 0:
                        metrics_accuracy = metrics_row.iloc[0]['accuracy']
                        match = abs(matrix_accuracy - metrics_accuracy) < 0.0001
                        accuracy_checks.append({
                            'model': model_name,
                            'matrix_accuracy': matrix_accuracy,
                            'metrics_accuracy': metrics_accuracy,
                            'match': match
                        })
                        
                        status = "✅" if match else "❌"
                        print(f"  {status} {model_name:25} matrix={matrix_accuracy:.4f} metrics={metrics_accuracy:.4f}")
        
        all_accuracies_match = all(c['match'] for c in accuracy_checks) if accuracy_checks else False
        
        # Step 4: Calculate difficult statements
        print("\nStep 4: Calculating difficult statements (miss_rate > 0.5)...")
        calc_difficult_count, calc_difficult_pct, calc_breakdown = calculate_difficult_statements(
            calc_matrix, calc_info
        )
        
        print(f"  Calculated difficult count: {calc_difficult_count}")
        print(f"  Calculated difficult pct: {calc_difficult_pct:.2f}%")
        print(f"  Calculated breakdown: {calc_breakdown}")
        
        # Step 5: Compare to reported difficult statements
        print("\nStep 5: Comparing to reported difficult statement summary...")
        reported_summary = load_reported_summary(difficult_dir, task_name)
        reported_breakdown = load_reported_breakdown(difficult_dir, task_name)
        
        summary_match = False
        breakdown_match = False
        
        if reported_summary is not None:
            reported_count = int(reported_summary['difficult_count'])
            reported_pct = float(reported_summary['difficult_pct'])
            
            count_match = calc_difficult_count == reported_count
            pct_match = abs(calc_difficult_pct - reported_pct) < 0.01
            
            print(f"  Reported difficult count: {reported_count}")
            print(f"  Count match: {'✅' if count_match else '❌'}")
            
            summary_match = count_match and pct_match
        else:
            print("  ⚠️  Reported summary not found")
        
        if reported_breakdown is not None:
            reported_breakdown_dict = dict(zip(
                reported_breakdown['category'], 
                reported_breakdown['count']
            ))
            
            breakdown_match = calc_breakdown == reported_breakdown_dict
            print(f"  Reported breakdown: {reported_breakdown_dict}")
            print(f"  Breakdown match: {'✅' if breakdown_match else '❌'}")
        else:
            print("  ⚠️  Reported breakdown not found")
        
        # Step 6: Check provenance
        print("\nStep 6: Checking figure provenance...")
        heatmap_name = config['figure_name'].replace('figure_', '') + '_correctness_heatmap'
        provenance = load_figure_provenance(paper_run_dir, config['figure_name'])
        
        provenance_found = provenance is not None
        if provenance_found:
            print(f"  ✅ Provenance found for {config['figure_name']}")
            print(f"     Generated: {provenance.get('generated_at', 'Unknown')}")
        else:
            print(f"  ⚠️  Provenance not found for {config['figure_name']}")
        
        # Calculate boundary statistics
        if not calc_matrix.empty:
            miss_rate = 1 - calc_matrix.mean(axis=0)
            at_boundary = (miss_rate == 0.5).sum()
            above_boundary = (miss_rate > 0.5).sum()
            print(f"\n  Boundary analysis:")
            print(f"    Statements above 50% threshold (difficult): {above_boundary}")
            print(f"    Statements exactly at 50% boundary: {at_boundary}")
        
        # Overall task result
        task_passed = (
            matrix_match and 
            all_accuracies_match and 
            summary_match and 
            breakdown_match
        )
        
        task_summaries[task_name] = {
            'figure': config['figure_name'],
            'matrix_match': matrix_match,
            'accuracy_match': all_accuracies_match,
            'summary_match': summary_match,
            'breakdown_match': breakdown_match,
            'provenance_found': provenance_found,
            'difficult_count': calc_difficult_count,
            'at_boundary': at_boundary if not calc_matrix.empty else 0,
            'passed': task_passed,
        }
        
        # Store detailed audit result
        audit_results.append({
            'task': task_name,
            'figure': config['figure_name'],
            'experiment_dir': str(exp_dir.name) if exp_dir else None,
            
            # Matrix verification
            'num_models': len(model_outputs),
            'num_statements': calc_matrix.shape[1] if not calc_matrix.empty else 0,
            'matrix_shape_match': matrix_shape_match,
            'matrix_values_match': matrix_values_match,
            'matrix_match': matrix_match,
            
            # Accuracy verification
            'all_accuracies_match': all_accuracies_match,
            'num_accuracy_checks': len(accuracy_checks),
            
            # Difficult statements
            'calc_difficult_count': calc_difficult_count,
            'calc_difficult_pct': calc_difficult_pct,
            'reported_difficult_count': int(reported_summary['difficult_count']) if reported_summary is not None else None,
            'summary_match': summary_match,
            'breakdown_match': breakdown_match,
            
            # Boundary analysis
            'statements_at_boundary': at_boundary if not calc_matrix.empty else 0,
            
            # Provenance
            'provenance_found': provenance_found,
            
            # Overall
            'all_match': task_passed,
        })
        
        status = "✅ PASS" if task_passed else "❌ FAIL"
        print(f"\n{status} - {task_name.upper()}")
    
    # Create DataFrame
    audit_df = pd.DataFrame(audit_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    
    total_tasks = len(task_summaries)
    passed_tasks = sum(1 for t in task_summaries.values() if t['passed'])
    
    print(f"\nTasks audited: {total_tasks}")
    print(f"  Passed: {passed_tasks}")
    print(f"  Failed: {total_tasks - passed_tasks}")
    
    print(f"\nPer-task results:")
    for task_name, summary in task_summaries.items():
        status = "✅" if summary['passed'] else "❌"
        print(f"  {status} {summary['figure']}: {task_name}")
        print(f"       Matrix match: {'✅' if summary['matrix_match'] else '❌'}")
        print(f"       Accuracy match: {'✅' if summary['accuracy_match'] else '❌'}")
        print(f"       Difficult count match: {'✅' if summary['summary_match'] else '❌'}")
        print(f"       Breakdown match: {'✅' if summary['breakdown_match'] else '❌'}")
        print(f"       Difficult statements: {summary['difficult_count']} (+ {summary['at_boundary']} at boundary)")
    
    if passed_tasks == total_tasks:
        print("\n" + "=" * 70)
        print("✅ ALL FIGURES S8-S10 VERIFIED")
        print("=" * 70)
        print("Correctness matrices and difficult statement analysis match exactly.")
    else:
        print("\n" + "=" * 70)
        print(f"❌ {total_tasks - passed_tasks} TASK(S) FAILED VERIFICATION")
        print("=" * 70)
        print("Review audit report for details.")
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False)
    print(f"\nAudit report saved to: {output_path}")
    
    # Save summary JSON
    summary = {
        'audit_timestamp': datetime.now().isoformat(),
        'paper_run_dir': str(paper_run_dir),
        'figures_verified': ['Figure S8', 'Figure S9', 'Figure S10'],
        'total_tasks': total_tasks,
        'passed': passed_tasks,
        'failed': total_tasks - passed_tasks,
        'all_passed': passed_tasks == total_tasks,
        'threshold': '> 0.5 (missed by 8+ of 14 models)',
        'task_summaries': task_summaries,
        'verifications_performed': [
            'Correctness matrix values',
            'Model accuracy vs comprehensive_metrics.csv',
            'Difficult statement count',
            'Difficult statement category breakdown',
            'Figure provenance files',
        ],
    }
    
    # Save JSON summary to figure_provenance (provenance info)
    provenance_dir = output_path.parent.parent / 'figure_provenance'
    provenance_dir.mkdir(parents=True, exist_ok=True)
    summary_path = provenance_dir / 'heatmap_audit_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Audit summary saved to: {summary_path}")
    
    return audit_df


def find_latest_paper_run() -> Optional[Path]:
    """Find the most recent REGULATORY_SIMULATION_PAPER run."""
    results_dir = PROJECT_ROOT / 'results' / 'REGULATORY_SIMULATION_PAPER'
    if not results_dir.exists():
        return None
    
    # Filter to only timestamp-based directories (YYYYMMDD_HHMMSS format)
    # Exclude directories like 'zOLD', 'backup', etc.
    import re
    timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')
    
    runs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)],
        key=lambda x: x.name,
        reverse=True
    )
    return runs[0] if runs else None


def main():
    parser = argparse.ArgumentParser(
        description="Audit Figures S8-S10 heatmaps and difficult statement analysis"
    )
    parser.add_argument(
        '--paper-run-dir',
        type=str,
        default=None,
        help='Path to pipeline output directory (default: most recent run)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: heatmap_audit_report.csv in paper run dir)'
    )
    
    args = parser.parse_args()
    
    # Find paper run directory
    if args.paper_run_dir:
        paper_run_dir = Path(args.paper_run_dir)
    else:
        paper_run_dir = find_latest_paper_run()
        if paper_run_dir is None:
            print("ERROR: No paper run found. Run the pipeline first or specify --paper-run-dir")
            sys.exit(1)
    
    if not paper_run_dir.exists():
        print(f"ERROR: Paper run directory not found: {paper_run_dir}")
        sys.exit(1)
    
    # Set output path (audit files go in Logs/Audits/)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = paper_run_dir / 'Logs' / 'Audits' / 'heatmap_audit_report.csv'
    
    # Run audit
    try:
        audit_df = run_audit(
            paper_run_dir=paper_run_dir,
            output_path=output_path
        )
        
        # Exit with error if any failures
        if 'all_match' in audit_df.columns and not audit_df['all_match'].all():
            sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
