#!/usr/bin/env python3
"""
Regulatory Simulation Paper Pipeline - Generate All Figures and Data for Publication

This script orchestrates the complete pipeline for generating all figures
and data needed for the regulatory simulation paper (medrxiv preprint revision).

Main Figures:
    Figure 3: Expert review breakdown (approved/modified/removed) - 3-panel barplot
    Figure 4: Model performance metrics (parse rate, sens/spec/accuracy/f1)
    Figure 5: P1/P2/P_harm risk analysis (failure_multiplier m values from config)

Supplementary Figures:
    Figure S3: Sankey diagrams (SI, therapy request, therapy engagement)
    Figures S4-S6: Binary confusion matrices (SI, Therapy Request, Therapy Engagement)
    Figures S7-S9: Per-statement accuracy heatmaps (SI, Therapy Request, Therapy Engagement)
    Figure S10: P2 by Harm Prevalence Across Failure Multiplier Values

Data Outputs:
    - raw_data/: Original finalized datasets and raw model results
    - processed_data/: Psychiatrist review files, comprehensive metrics
    - model_outputs.tar.gz: Compressed model prediction CSVs
    - prompts/: Classification prompts and Gemini generation prompts
    - model_info/: Model configuration CSV

Output Structure:
    results/REGULATORY_SIMULATION_PAPER/[YYYYMMDD_HHMMSS]/
        Figures/
            figure_3.png
            figure_4.png
            figure_5/
                p1_p2_p_harm_risk_analysis_m_1.0.png
                p1_p2_p_harm_risk_analysis_m_2.0.png
                ...
        Supplementary_Figures/
            figure_S3/
                si_expert_review_sankey.png
                therapy_request_expert_review_sankey.png
                therapy_engagement_expert_review_sankey.png
            figures_S4-S6/
                suicidal_ideation_binary_confusion_matrix_grid.png
                therapy_request_binary_confusion_matrix_grid.png
                therapy_engagement_binary_confusion_matrix_grid.png
            figures_S7-S9/
                si_correctness_heatmap.png
                therapy_request_correctness_heatmap.png
                therapy_engagement_correctness_heatmap.png
            figure_S10/
                figure_s10_p2_across_m_values.png
        Data/
            raw_data/
            processed_data/
            model_outputs.tar.gz
            prompts/
            model_info/

Usage:
    python run_regulatory_simulation_paper_pipeline.py
    python run_regulatory_simulation_paper_pipeline.py --models-config config/regulatory_paper_models.csv
    python run_regulatory_simulation_paper_pipeline.py --figures-only
    python run_regulatory_simulation_paper_pipeline.py --dry-run
    
    Options:
        --models-config       Path to CSV file specifying which models to use
                              (default: config/regulatory_paper_models.csv)
        --figures-only        Only generate figures, skip data collection
        --dry-run             Show what would be done without executing
        --si-experiment-dir   Override SI experiment directory (use existing results)
        --tr-experiment-dir   Override Therapy Request experiment directory
        --te-experiment-dir   Override Therapy Engagement experiment directory

Author: Mark Kalinich
"""

import subprocess
import sys
import logging
import argparse
import shutil
import tarfile
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import os

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
DEFAULT_MODELS_CONFIG = ROOT / "config" / "regulatory_paper_models.csv"

# Output directories with timestamp
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PAPER_OUTPUT_BASE = RESULTS_DIR / "REGULATORY_SIMULATION_PAPER" / RUN_TIMESTAMP
FIGURES_DIR = PAPER_OUTPUT_BASE / "Figures"
SUPP_FIGURES_DIR = PAPER_OUTPUT_BASE / "Supplementary_Figures"
DATA_OUTPUT_DIR = PAPER_OUTPUT_BASE / "Data"


def load_models_from_config(config_path: Path) -> dict:
    """
    Load models from a CSV config file.
    
    Args:
        config_path: Path to CSV with columns: family, size, enabled
        
    Returns:
        Dict mapping family -> list of sizes (only enabled models)
    """
    import pandas as pd
    
    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found: {config_path}")
    
    df = pd.read_csv(config_path)
    
    # Filter to enabled models only
    if 'enabled' in df.columns:
        df = df[df['enabled'] == True]
    
    # Build family -> sizes dict
    models = {}
    for _, row in df.iterrows():
        family = row['family']
        size = row['size']
        if family not in models:
            models[family] = []
        if size not in models[family]:
            models[family].append(size)
    
    return models

# Failure multiplier values for P1/P2 sensitivity analysis (Figure 5)
# m=1: independent failures, m>1: FNR approximately m× higher, m→∞: certain failure
FAILURE_MULTIPLIER_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0, 100.0, 1000.0, 10000.0, 100000.0]

# Task configurations with input data and prompts
TASKS = {
    "suicidal_ideation": {
        "short_name": "SI",
        "results_subdir": "suicidal_ideation",
        "input_data": "data/inputs/finalized_input_data/SI_finalized_sentences.csv",
        "prompt_file": "data/prompts/system_suicide_detection_v2.txt",
    },
    "therapy_request": {
        "short_name": "TR",
        "results_subdir": "therapy_request",
        "input_data": "data/inputs/finalized_input_data/therapy_request_finalized_sentences.csv",
        "prompt_file": "data/prompts/therapy_request_classifier_v3.txt",
    },
    "therapy_engagement": {
        "short_name": "TE",
        "results_subdir": "therapy_engagement",
        "input_data": "data/inputs/finalized_input_data/therapy_engagement_finalized_sentences.csv",
        "prompt_file": "data/prompts/therapy_engagement_conversation_prompt_v2.txt",
    },
}


def get_models_filter_string(models: dict) -> str:
    """Convert models dict to --models argument format: 'family:size,family:size,...'"""
    models_list = []
    for family, sizes in models.items():
        for size in sizes:
            models_list.append(f"{family}:{size}")
    return ",".join(models_list)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging with both console and file handlers."""
    logger = logging.getLogger("regulatory_simulation_pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG level)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def log_section(logger: logging.Logger, title: str, char: str = "═") -> None:
    """Log a section header."""
    width = 70
    logger.info("")
    logger.info(char * width)
    logger.info(f"  {title}")
    logger.info(char * width)


def log_subsection(logger: logging.Logger, title: str) -> None:
    """Log a subsection header."""
    logger.info("")
    logger.info(f"─── {title} ───")


# =============================================================================
# Utility Functions
# =============================================================================



def filter_comprehensive_metrics(
    input_csv: Path, 
    output_csv: Path,
    paper_models: dict,
    logger: logging.Logger
) -> bool:
    """
    Filter comprehensive_metrics.csv to only include specified models.
    
    Args:
        input_csv: Path to original comprehensive_metrics.csv (with all models)
        output_csv: Path to save filtered CSV (only paper models)
        paper_models: Dict mapping family -> list of sizes
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    import pandas as pd
    
    if not input_csv.exists():
        logger.error(f"  Input CSV not found: {input_csv}")
        return False
    
    df = pd.read_csv(input_csv)
    original_count = len(df)
    
    # Filter to paper models only
    filtered_rows = []
    for family, sizes in paper_models.items():
        for size in sizes:
            # Try exact match first
            mask = (df['model_family'] == family) & (df['model_size'] == size)
            
            # Fallback 1: match if data's model_size starts with configured size (e.g., "270m" matches "270m-it")
            if mask.sum() == 0:
                base_size = size.split('-')[0]
                mask = (df['model_family'] == family) & (df['model_size'].str.startswith(base_size))
            
            # Fallback 2: handle llama family variants (llama3.2, llama3.1, llama3.3 -> llama)
            if mask.sum() == 0 and family.startswith('llama'):
                base_size = size.split('-')[0]
                mask = (df['model_family'] == 'llama') & (df['model_size'].str.startswith(base_size))
            
            filtered_rows.append(df[mask])
    
    if filtered_rows:
        filtered_df = pd.concat(filtered_rows, ignore_index=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_csv, index=False)
        logger.info(f"  Filtered: {original_count} → {len(filtered_df)} models")
        return True
    else:
        logger.error(f"  No paper models found in {input_csv}")
        return False


def regenerate_experiment_from_cache(
    task_name: str,
    paper_models: dict,
    logger: logging.Logger,
    dry_run: bool = False,
    cache_dir: str = "cache"
) -> Optional[Path]:
    """
    Regenerate experiment results from cache for a specific task.
    
    Args:
        task_name: Name of the task (suicidal_ideation, therapy_request, therapy_engagement)
        paper_models: Dict mapping family -> list of sizes
        logger: Logger instance
        dry_run: If True, don't actually run
        cache_dir: Cache directory (default: cache)
    
    Returns:
        Path to the generated experiment directory, or None if failed
    """
    task_config = TASKS[task_name]
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / "individual_prediction_performance" / task_config["results_subdir"] / f"{timestamp}_{task_config['short_name']}_paper"
    
    logger.info(f"  Regenerating {task_name} from cache ({cache_dir})...")
    logger.info(f"    Output: {output_dir}")
    
    if dry_run:
        logger.info(f"    [DRY RUN] Would regenerate from cache")
        return output_dir
    
    # Build batch_results_analyzer command
    script = ROOT / "analysis" / "model_performance" / "batch_results_analyzer.py"
    models_filter = get_models_filter_string(paper_models)
    
    args = [
        "--output-dir", str(output_dir),
        "--input-data", str(ROOT / task_config["input_data"]),
        "--prompt-file", str(ROOT / task_config["prompt_file"]),
        "--timestamp", timestamp,
        "--experiment-type", task_name,
        "--models", models_filter,
        "--cache-dir", cache_dir,
    ]
    
    logger.info(f"    Models: {len(paper_models)} families, {sum(len(v) for v in paper_models.values())} total")
    
    success = run_python_script(script, args, logger, dry_run=dry_run)
    
    if success and output_dir.exists():
        logger.info(f"    ✓ Generated: {output_dir.name}")
        return output_dir
    else:
        logger.error(f"    ✗ Failed to regenerate {task_name}")
        return None


def find_latest_experiment_dir(task_name: str) -> Optional[Path]:
    """Find the most recent experiment directory for a task."""
    task_config = TASKS[task_name]
    results_base = RESULTS_DIR / "individual_prediction_performance" / task_config["results_subdir"]
    
    if not results_base.exists():
        return None
    
    # Find directories with comprehensive_metrics.csv
    valid_dirs = []
    for task_dir in results_base.iterdir():
        if task_dir.is_dir():
            metrics_file = task_dir / "tables" / "comprehensive_metrics.csv"
            if metrics_file.exists():
                valid_dirs.append(task_dir)
    
    if not valid_dirs:
        return None
    
    # Sort by directory name (timestamp) and return most recent
    valid_dirs.sort(key=lambda x: x.name, reverse=True)
    return valid_dirs[0]


def run_python_script(
    script_path: Path, 
    args: List[str], 
    logger: logging.Logger,
    cwd: Optional[Path] = None,
    dry_run: bool = False
) -> bool:
    """Run a Python script and return success status."""
    if cwd is None:
        cwd = ROOT
    
    cmd = [sys.executable, str(script_path)] + args
    
    logger.debug(f"  Command: {' '.join(cmd)}")
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would run: {script_path.name}")
        return True
    
    try:
        # Inherit current environment (including venv activation)
        # and add project root to PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT) + ':' + env.get('PYTHONPATH', '')
        
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min timeout
            env=env,
            shell=False,
        )
        
        if result.returncode != 0:
            logger.error(f"  ✗ Script failed: {script_path.name}")
            logger.debug(f"  STDERR: {result.stderr[-1000:] if result.stderr else 'None'}")
            return False
        
        # Log any output
        if result.stdout:
            for line in result.stdout.strip().split('\n')[-5:]:
                logger.debug(f"    {line}")
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ Script timed out: {script_path.name}")
        return False
    except Exception as e:
        logger.error(f"  ✗ Script failed with exception: {e}")
        return False


# =============================================================================
# Figure Generation
# =============================================================================

def generate_figure_3(logger: logging.Logger, dry_run: bool = False) -> bool:
    """Generate Figure 3: Expert Review Breakdown (3-panel barplot)."""
    log_subsection(logger, "Figure 3: Expert Review Breakdown")
    
    script = ROOT / "analysis" / "data_validation" / "combined_three_panel_review_provenance.py"
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would run: {script.name}")
        return True
    
    success = run_python_script(script, [], logger, dry_run=dry_run)
    
    if success:
        # Find and copy the output - structure is {date}/{timestamp_name}/file.png
        src_dir = RESULTS_DIR / "data_validation" / "psychiatrist_review_barplots"
        if src_dir.exists():
            # Find most recent date directory, then most recent timestamped subdirectory
            date_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            if date_dirs:
                timestamp_dirs = sorted([d for d in date_dirs[0].iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                if timestamp_dirs:
                    src_file = timestamp_dirs[0] / "combined_three_panel_review.png"
                    if src_file.exists():
                        dst_file = FIGURES_DIR / "figure_3.png"
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(src_file, dst_file)
                        logger.info(f"  ✓ Saved: Figures/figure_3.png")
                        return True
        
        logger.warning(f"  ⚠ Output not found after script completion")
        return False
    
    return False


def generate_figure_4(
    filtered_csvs: Dict[str, Path],
    logger: logging.Logger, 
    dry_run: bool = False
) -> bool:
    """Generate Figure 4: Model Performance Metrics."""
    log_subsection(logger, "Figure 4: Model Performance Metrics")
    
    script = ROOT / "analysis" / "comparative_analysis" / "multi_experiment_plot_transposed_provenance.py"
    
    # Get paths to filtered comprehensive_metrics.csv for each task
    si_csv = filtered_csvs['suicidal_ideation']
    tr_csv = filtered_csvs['therapy_request']
    te_csv = filtered_csvs['therapy_engagement']
    
    logger.info(f"  SI metrics:  {si_csv}")
    logger.info(f"  TR metrics:  {tr_csv}")
    logger.info(f"  TE metrics:  {te_csv}")
    
    args = [
        "--suicide-ideation-csv", str(si_csv),
        "--therapy-request-csv", str(tr_csv),
        "--therapy-detection-csv", str(te_csv),
    ]
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would run: {script.name}")
        return True
    
    success = run_python_script(script, args, logger, dry_run=dry_run)
    
    if success:
        # Find and copy the output - structure is {date}/{timestamp_name}/file.png
        src_dir = ROOT / "analysis" / "results" / "model_performance_comparison"
        if src_dir.exists():
            # Find most recent date directory, then most recent timestamped subdirectory
            date_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            if date_dirs:
                timestamp_dirs = sorted([d for d in date_dirs[0].iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                if timestamp_dirs:
                    src_file = timestamp_dirs[0] / "multi_model_performance.png"
                    if src_file.exists():
                        dst_file = FIGURES_DIR / "figure_4.png"
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(src_file, dst_file)
                        logger.info(f"  ✓ Saved: Figures/figure_4.png")
                        return True
        
        logger.warning(f"  ⚠ Output not found after script completion")
        return False
    
    return False


def generate_figure_5(
    filtered_csvs: Dict[str, Path],
    logger: logging.Logger, 
    dry_run: bool = False
) -> bool:
    """Generate Figure 5: P1/P2/P_harm Risk Analysis for multiple failure_multiplier values."""
    log_subsection(logger, "Figure 5: P1/P2/P_harm Risk Analysis")
    
    script = ROOT / "analysis" / "comparative_analysis" / "p1_and_p2_plot_provenance.py"
    output_dir = FIGURES_DIR / "figure_5"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get paths to filtered comprehensive_metrics.csv for each task
    si_csv = filtered_csvs['suicidal_ideation']
    tr_csv = filtered_csvs['therapy_request']
    te_csv = filtered_csvs['therapy_engagement']
    
    all_success = True
    
    for m in FAILURE_MULTIPLIER_VALUES:
        logger.info(f"  Generating m = {m}...")
        
        args = [
            "--suicide-csv", str(si_csv),
            "--therapy-request-csv", str(tr_csv),
            "--therapy-engagement-csv", str(te_csv),
            "--failure-multiplier", str(m),
            "--n-mc-samples", "50000",
            "--uncertainty-style", "both",
        ]
        
        if dry_run:
            logger.info(f"    [DRY RUN] Would generate m={m}")
            continue
        
        success = run_python_script(script, args, logger, dry_run=dry_run)
        
        if success:
            # Find and copy the output - structure is {date}/{timestamp_name}/file.png
            src_dir = RESULTS_DIR / "risk_analysis"
            if src_dir.exists():
                date_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                if date_dirs:
                    timestamp_dirs = sorted([d for d in date_dirs[0].iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                    if timestamp_dirs:
                        # Look for the multiplier-specific file
                        m_str = f"_m_{m}"
                        for f in timestamp_dirs[0].iterdir():
                            if f.is_file() and f.name.endswith('.png') and m_str in f.name:
                                dst_file = output_dir / f.name
                                shutil.copy(f, dst_file)
                                logger.info(f"    ✓ Saved: figure_5/{f.name}")
                                break
        else:
            all_success = False
    
    return all_success


def generate_figure_s3(logger: logging.Logger, dry_run: bool = False) -> bool:
    """Generate Figure S3: Sankey Diagrams."""
    log_subsection(logger, "Figure S3: Sankey Diagrams")
    
    script = ROOT / "analysis" / "data_validation" / "sankey_diagram_configs.py"
    output_dir = SUPP_FIGURES_DIR / "figure_S3"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = ['si', 'therapy_request', 'therapy_engagement']
    all_success = True
    
    for exp in experiments:
        logger.info(f"  Generating {exp} Sankey diagram...")
        
        if dry_run:
            logger.info(f"    [DRY RUN] Would generate {exp}")
            continue
        
        success = run_python_script(script, [exp], logger, dry_run=dry_run)
        
        if success:
            # Find and copy outputs - structure is {date}/{prefix_timestamp_name}/file.png
            # Prefixes: SI_, tx_request_, tx_engagement_
            prefix_map = {'si': 'SI_', 'therapy_request': 'tx_request_', 'therapy_engagement': 'tx_engagement_'}
            prefix = prefix_map.get(exp, exp + '_')
            
            src_dir = RESULTS_DIR / "data_validation" / "psychiatrist_review_sankey_diagrams"
            if src_dir.exists():
                date_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                if date_dirs:
                    # Find the most recent directory for this specific experiment
                    exp_dirs = sorted([d for d in date_dirs[0].iterdir() if d.is_dir() and d.name.startswith(prefix)], key=lambda x: x.name, reverse=True)
                    if exp_dirs:
                        for f in exp_dirs[0].iterdir():
                            if f.is_file() and f.name.endswith('.png'):
                                dst_file = output_dir / f.name
                                shutil.copy(f, dst_file)
                                logger.info(f"    ✓ Saved: figure_S3/{f.name}")
        else:
            all_success = False
    
    return all_success


def generate_confusion_matrices(
    experiment_dirs: Dict[str, Path],
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """Generate Figures S4-S6: Binary Confusion Matrices (consolidated into one folder)."""
    log_subsection(logger, "Figures S4-S6: Confusion Matrices")
    
    script = ROOT / "analysis" / "model_performance" / "generate_confusion_matrix_figures.py"
    
    # Consolidated output directory for all confusion matrices
    output_dir = SUPP_FIGURES_DIR / "figures_S4-S6"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task_display_map = {
        'suicidal_ideation': 'SI',
        'therapy_request': 'Therapy Request',
        'therapy_engagement': 'Therapy Engagement',
    }
    
    all_success = True
    
    for task_name, display_name in task_display_map.items():
        logger.info(f"  Generating {display_name} Confusion Matrix...")
        
        exp_dir = experiment_dirs[task_name]
        
        args = [
            "--experiment-dir", str(exp_dir),
            "--task-type", task_name,
            "--output-dir", str(output_dir),
            "--models-config", str(ROOT / "config" / "regulatory_paper_models.csv"),
        ]
        
        if dry_run:
            logger.info(f"    [DRY RUN] Would generate {display_name}")
            continue
        
        success = run_python_script(script, args, logger, dry_run=dry_run)
        
        if success:
            logger.info(f"    ✓ Saved: {task_name}_binary_confusion_matrix_grid.png")
        else:
            all_success = False
    
    if all_success and not dry_run:
        logger.info(f"  ✓ All confusion matrices saved to: figures_S4-S6/")
    
    return all_success


def generate_heatmaps(
    experiment_dirs: Dict[str, Path],
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """Generate Figures S7-S9: Per-statement Accuracy Heatmaps (consolidated into one folder)."""
    log_subsection(logger, "Figures S7-S9: Accuracy Heatmaps")
    
    # First generate the correctness matrices
    matrix_script = ROOT / "analysis" / "model_performance" / "generate_model_statement_matrices.py"
    heatmap_script = ROOT / "analysis" / "data_validation" / "generate_all_heatmaps.py"
    
    # Consolidated output directory
    output_dir = SUPP_FIGURES_DIR / "figures_S7-S9"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map task names to filename prefixes (for finding heatmap output files)
    task_filename_map = {
        'suicidal_ideation': 'si_correctness_heatmap',
        'therapy_request': 'therapy_request_correctness_heatmap',
        'therapy_engagement': 'therapy_engagement_correctness_heatmap',
    }
    
    if dry_run:
        for task_name in task_filename_map.keys():
            logger.info(f"    [DRY RUN] Would generate heatmap for: {task_name}")
        return True
    
    # Step 1: Generate correctness matrices (one call with all experiment dirs)
    logger.info("  Step 1: Generating correctness matrices...")
    
    matrix_args = []
    for task_name, exp_dir in experiment_dirs.items():
        matrix_args.extend([f"--{task_name.replace('_', '-')}-dir", str(exp_dir)])
    
    matrix_success = run_python_script(matrix_script, matrix_args, logger, dry_run=dry_run)
    if not matrix_success:
        logger.warning("  ⚠ Matrix generation may have failed")
        return False
    
    # Step 2: Generate heatmaps (one call - it reads from results/review_statistics/)
    logger.info("  Step 2: Generating heatmaps...")
    
    heatmap_success = run_python_script(heatmap_script, [], logger, dry_run=dry_run)
    if not heatmap_success:
        logger.warning("  ⚠ Heatmap generation may have failed")
        return False
    
    # Step 3: Copy outputs to consolidated supplementary figure directory
    logger.info("  Step 3: Copying outputs to figures_S7-S9/...")
    
    # Heatmaps are saved to results/model_performance_analysis/{date}/{timestamp_heatmapname}/
    # Each heatmap gets its own timestamped directory
    src_base = RESULTS_DIR / "model_performance_analysis"
    
    if not src_base.exists():
        logger.error(f"  ✗ Heatmap output directory not found: {src_base}")
        return False
    
    # Find the most recent date directory
    date_dirs = sorted([d for d in src_base.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
    if not date_dirs:
        logger.error(f"  ✗ No date directories in {src_base}")
        return False
    
    date_dir = date_dirs[0]
    
    # Copy each heatmap to the consolidated figure directory
    # Each heatmap has its own timestamped subdirectory like: 20260101_2030_si_correctness_heatmap
    all_success = True
    for task_name, filename_prefix in task_filename_map.items():
        # Find the most recent directory for this specific heatmap
        heatmap_dirs = sorted(
            [d for d in date_dir.iterdir() if d.is_dir() and filename_prefix in d.name],
            key=lambda x: x.name, 
            reverse=True
        )
        
        if not heatmap_dirs:
            logger.warning(f"    ⚠ No directory found for: {filename_prefix}")
            all_success = False
            continue
        
        heatmap_src_dir = heatmap_dirs[0]
        
        # Look for the PNG file in this directory
        found = False
        for f in heatmap_src_dir.iterdir():
            if f.is_file() and f.name.endswith('.png'):
                dst_file = output_dir / f.name
                shutil.copy(f, dst_file)
                logger.info(f"    ✓ Saved: {f.name}")
                found = True
                break
        
        if not found:
            logger.warning(f"    ⚠ No PNG found in: {heatmap_src_dir}")
            all_success = False
    
    if all_success:
        logger.info(f"  ✓ All heatmaps saved to: figures_S7-S9/")
    
    return all_success


def generate_figure_s10(
    filtered_csvs: Dict[str, Path],
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """Generate Figure S10: P2 by Harm Prevalence Across Failure Multiplier Values.
    
    Shows how P2 varies with harm prevalence at different failure multiplier (M) values.
    Uses the same underlying data as Figure 5 but shows all M values in a single facet plot.
    """
    log_subsection(logger, "Figure S10: P2 Across M Values")
    
    script = ROOT / "analysis" / "comparative_analysis" / "figure_s10_p2_by_model_size_across_m.py"
    output_dir = SUPP_FIGURES_DIR / "figure_S10"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    args = [
        "--si-metrics", str(filtered_csvs["suicidal_ideation"]),
        "--tr-metrics", str(filtered_csvs["therapy_request"]),
        "--te-metrics", str(filtered_csvs["therapy_engagement"]),
        "--output-dir", str(output_dir),
    ]
    
    success = run_python_script(script, args, logger, dry_run=dry_run)
    
    if success and not dry_run:
        logger.info(f"  ✓ Saved: figure_S10/figure_s10_p2_across_m_values.png")
    
    return success


# =============================================================================
# Data Collection
# =============================================================================

def collect_raw_data(logger: logging.Logger, dry_run: bool = False) -> bool:
    """Collect raw datasets."""
    log_subsection(logger, "Collecting Raw Data")
    
    raw_data_dir = DATA_OUTPUT_DIR / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Finalized datasets
    finalized_files = [
        "SI_finalized_sentences.csv",
        "therapy_request_finalized_sentences.csv",
        "therapy_engagement_finalized_sentences.csv",
    ]
    
    # Raw model results (before expert review)
    raw_model_files = [
        "SI_balanced_100_per_category_ordered_input.csv",
        "therapy_request_100_per_category_reformatted.csv",
        "therapy_engagement_conversations_downsampled_150.csv",
    ]
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would copy {len(finalized_files) + len(raw_model_files)} files")
        return True
    
    # Copy finalized datasets
    src_dir = DATA_DIR / "inputs" / "finalized_input_data"
    for filename in finalized_files:
        src = src_dir / filename
        if src.exists():
            shutil.copy(src, raw_data_dir / filename)
            logger.info(f"  ✓ Copied: {filename}")
        else:
            logger.warning(f"  ⚠ Not found: {filename}")
    
    # Copy raw model results
    src_dir = DATA_DIR / "inputs" / "raw_model_results"
    for filename in raw_model_files:
        src = src_dir / filename
        if src.exists():
            shutil.copy(src, raw_data_dir / filename)
            logger.info(f"  ✓ Copied: {filename}")
        else:
            logger.warning(f"  ⚠ Not found: {filename}")
    
    return True


def collect_processed_data(
    experiment_dirs: Dict[str, Path],
    logger: logging.Logger, 
    dry_run: bool = False
) -> bool:
    """Collect processed data files."""
    log_subsection(logger, "Collecting Processed Data")
    
    processed_dir = DATA_OUTPUT_DIR / "processed_data"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Psychiatrist review files
    review_files = [
        "SI_psychiatrist_01_and_02_scores.csv",
        "therapy_request_psychiatrist_01_and_02_scores.csv",
        "therapy_engagement_psychiatrist_01_and_02_scores.csv",
    ]
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would copy review files and comprehensive metrics")
        return True
    
    # Copy review files
    src_dir = DATA_DIR / "inputs" / "intermediate_files"
    for filename in review_files:
        src = src_dir / filename
        if src.exists():
            shutil.copy(src, processed_dir / filename)
            logger.info(f"  ✓ Copied: {filename}")
        else:
            logger.warning(f"  ⚠ Not found: {filename}")
    
    # Copy comprehensive_metrics.csv from each experiment
    for task_name, exp_dir in experiment_dirs.items():
        metrics_file = exp_dir / "tables" / "comprehensive_metrics.csv"
        if metrics_file.exists():
            dst_name = f"{task_name}_comprehensive_metrics.csv"
            shutil.copy(metrics_file, processed_dir / dst_name)
            logger.info(f"  ✓ Copied: {dst_name}")
        else:
            logger.warning(f"  ⚠ Not found: {task_name} comprehensive_metrics.csv")
    
    return True


def collect_model_outputs(
    experiment_dirs: Dict[str, Path],
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """Collect and compress model outputs."""
    log_subsection(logger, "Collecting Model Outputs")
    
    tar_path = DATA_OUTPUT_DIR / "model_outputs.tar.gz"
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would create: model_outputs.tar.gz")
        return True
    
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(tar_path, "w:gz") as tar:
        for task_name, exp_dir in experiment_dirs.items():
            outputs_dir = exp_dir / "model_outputs"
            if outputs_dir.exists():
                for csv_file in outputs_dir.glob("*.csv"):
                    arcname = f"{task_name}/{csv_file.name}"
                    tar.add(csv_file, arcname=arcname)
                logger.info(f"  ✓ Added {task_name} model outputs")
            else:
                logger.warning(f"  ⚠ Not found: {task_name} model_outputs/")
    
    logger.info(f"  ✓ Created: model_outputs.tar.gz")
    return True


def collect_prompts(logger: logging.Logger, dry_run: bool = False) -> bool:
    """Collect prompt files."""
    log_subsection(logger, "Collecting Prompts")
    
    prompts_dir = DATA_OUTPUT_DIR / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Classification prompts
    classification_prompts = [
        "system_suicide_detection_v2.txt",
        "therapy_request_classifier_v3.txt",
        "therapy_engagement_conversation_prompt_v2.txt",
    ]
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would copy prompts")
        return True
    
    # Copy classification prompts
    src_dir = DATA_DIR / "prompts"
    for filename in classification_prompts:
        src = src_dir / filename
        if src.exists():
            shutil.copy(src, prompts_dir / filename)
            logger.info(f"  ✓ Copied: {filename}")
        else:
            logger.warning(f"  ⚠ Not found: {filename}")
    
    # Copy Gemini prompts
    gemini_src_dir = src_dir / "gemini_prompts"
    gemini_dst_dir = prompts_dir / "gemini_prompts"
    if gemini_src_dir.exists():
        shutil.copytree(gemini_src_dir, gemini_dst_dir)
        logger.info(f"  ✓ Copied: gemini_prompts/")
    
    return True


def collect_model_info(
    paper_models: dict,
    models_config_path: Path,
    logger: logging.Logger, 
    dry_run: bool = False
) -> bool:
    """Extract model information to CSV."""
    log_subsection(logger, "Collecting Model Information")
    
    import pandas as pd
    
    model_info_dir = DATA_OUTPUT_DIR / "model_info"
    model_info_dir.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would create model_info.csv")
        return True
    
    # Read full config
    config_path = ROOT / "config" / "models_config.csv"
    if not config_path.exists():
        logger.warning(f"  ⚠ Config not found: {config_path}")
        return False
    
    df = pd.read_csv(config_path)
    
    # Filter to paper models
    paper_model_rows = []
    for family, sizes in paper_models.items():
        for size in sizes:
            match = df[(df['family'] == family) & (df['size'] == size)]
            if not match.empty:
                paper_model_rows.append(match.iloc[0])
    
    if paper_model_rows:
        paper_df = pd.DataFrame(paper_model_rows)
        output_path = model_info_dir / "paper_models_config.csv"
        paper_df.to_csv(output_path, index=False)
        logger.info(f"  ✓ Created: paper_models_config.csv ({len(paper_df)} models)")
    
    # Copy the models config used for this run
    shutil.copy(models_config_path, model_info_dir / "models_config_used.csv")
    logger.info(f"  ✓ Copied: models_config_used.csv")
    
    # Also copy full config for reference
    shutil.copy(config_path, model_info_dir / "models_config_full.csv")
    logger.info(f"  ✓ Copied: models_config_full.csv")
    
    # Create a manifest
    manifest = {
        "paper_models": paper_models,
        "models_config_file": str(models_config_path),
        "generated": datetime.now().isoformat(),
    }
    manifest_path = model_info_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  ✓ Created: manifest.json")
    
    return True


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> int:
    """Run the complete paper pipeline."""
    
    # Load models from config file
    models_config_path = Path(args.models_config)
    paper_models = load_models_from_config(models_config_path)
    
    # Setup - put log in the output directory
    PAPER_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    log_file = PAPER_OUTPUT_BASE / "pipeline.log"
    
    logger = setup_logging(log_file)
    
    # Header
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════════════╗")
    logger.info("║       REGULATORY SIMULATION PAPER - FIGURE & DATA GENERATION         ║")
    logger.info("╚══════════════════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info(f"  Timestamp:  {RUN_TIMESTAMP}")
    logger.info(f"  Models:     {models_config_path}")
    logger.info(f"  Log file:   {log_file}")
    logger.info(f"  Dry run:    {args.dry_run}")
    logger.info(f"  Output:     {PAPER_OUTPUT_BASE}")
    logger.info("")
    
    if args.dry_run:
        logger.info("  *** DRY RUN MODE - No changes will be made ***")
    
    # Get experiment directories
    # If override provided: use that existing directory
    # If no override: regenerate from cache with only paper_models
    log_section(logger, "EXPERIMENT DATA")
    
    logger.info(f"  Models config: {models_config_path}")
    logger.info(f"  Total models: {sum(len(v) for v in paper_models.values())}")
    for family, sizes in paper_models.items():
        logger.info(f"    {family}: {sizes}")
    logger.info("")
    
    experiment_dirs = {}
    needs_filtering = {}  # Track which dirs need CSV filtering (overrides may have all models)
    
    # Suicidal Ideation
    if args.si_experiment_dir:
        experiment_dirs['suicidal_ideation'] = Path(args.si_experiment_dir)
        needs_filtering['suicidal_ideation'] = True  # Override may have all models
        logger.info(f"  suicidal_ideation: Using override: {args.si_experiment_dir}")
    else:
        exp_dir = regenerate_experiment_from_cache('suicidal_ideation', paper_models, logger, args.dry_run, args.cache_dir)
        if exp_dir:
            experiment_dirs['suicidal_ideation'] = exp_dir
            needs_filtering['suicidal_ideation'] = False  # Fresh generation only has paper models
        else:
            logger.error(f"  ✗ Failed to generate suicidal_ideation")
            return 1
    
    # Therapy Request
    if args.tr_experiment_dir:
        experiment_dirs['therapy_request'] = Path(args.tr_experiment_dir)
        needs_filtering['therapy_request'] = True
        logger.info(f"  therapy_request: Using override: {args.tr_experiment_dir}")
    else:
        exp_dir = regenerate_experiment_from_cache('therapy_request', paper_models, logger, args.dry_run, args.cache_dir)
        if exp_dir:
            experiment_dirs['therapy_request'] = exp_dir
            needs_filtering['therapy_request'] = False
        else:
            logger.error(f"  ✗ Failed to generate therapy_request")
            return 1
    
    # Therapy Engagement
    if args.te_experiment_dir:
        experiment_dirs['therapy_engagement'] = Path(args.te_experiment_dir)
        needs_filtering['therapy_engagement'] = True
        logger.info(f"  therapy_engagement: Using override: {args.te_experiment_dir}")
    else:
        exp_dir = regenerate_experiment_from_cache('therapy_engagement', paper_models, logger, args.dry_run, args.cache_dir)
        if exp_dir:
            experiment_dirs['therapy_engagement'] = exp_dir
            needs_filtering['therapy_engagement'] = False
        else:
            logger.error(f"  ✗ Failed to generate therapy_engagement")
            return 1
    
    # Verify all directories exist (for overrides)
    for task_name, exp_dir in experiment_dirs.items():
        if not args.dry_run and not exp_dir.exists():
            logger.error(f"  ✗ Directory not found: {exp_dir}")
            return 1
    
    success = True
    
    # Create filtered CSVs only for overrides (regenerated data already has only paper models)
    filtered_csvs = {}
    
    for task_name, exp_dir in experiment_dirs.items():
        if needs_filtering.get(task_name, False):
            # Override provided - need to filter
            log_subsection(logger, f"Filtering {task_name} to paper models")
            filtered_csv_dir = DATA_OUTPUT_DIR / "filtered_metrics"
            filtered_csv_dir.mkdir(parents=True, exist_ok=True)
            
            input_csv = exp_dir / "tables" / "comprehensive_metrics.csv"
            output_csv = filtered_csv_dir / f"{task_name}_comprehensive_metrics.csv"
            
            if not args.dry_run:
                if filter_comprehensive_metrics(input_csv, output_csv, paper_models, logger):
                    filtered_csvs[task_name] = output_csv
                else:
                    logger.error(f"    Failed to filter {task_name}")
                    success = False
            else:
                logger.info(f"    [DRY RUN] Would filter to paper models")
                filtered_csvs[task_name] = output_csv
        else:
            # Regenerated - already has only paper models
            filtered_csvs[task_name] = exp_dir / "tables" / "comprehensive_metrics.csv"
    
    # Generate figures
    log_section(logger, "GENERATING MAIN FIGURES")
    
    if not generate_figure_3(logger, args.dry_run):
        success = False
    
    if not generate_figure_4(filtered_csvs, logger, args.dry_run):
        success = False
    
    if not generate_figure_5(filtered_csvs, logger, args.dry_run):
        success = False
    
    # Generate supplementary figures
    log_section(logger, "GENERATING SUPPLEMENTARY FIGURES")
    
    if not generate_figure_s3(logger, args.dry_run):
        success = False
    
    if not generate_confusion_matrices(experiment_dirs, logger, args.dry_run):
        success = False
    
    if not generate_heatmaps(experiment_dirs, logger, args.dry_run):
        success = False
    
    if not generate_figure_s10(filtered_csvs, logger, args.dry_run):
        success = False
    
    # Collect data (unless figures-only)
    if not args.figures_only:
        log_section(logger, "COLLECTING DATA")
        
        if not collect_raw_data(logger, args.dry_run):
            success = False
        
        if not collect_processed_data(experiment_dirs, logger, args.dry_run):
            success = False
        
        if not collect_model_outputs(experiment_dirs, logger, args.dry_run):
            success = False
        
        if not collect_prompts(logger, args.dry_run):
            success = False
        
        if not collect_model_info(paper_models, models_config_path, logger, args.dry_run):
            success = False
    else:
        logger.info("")
        logger.info("⏭  Skipping data collection (--figures-only)")
    
    # Generate manuscript claims verification report
    log_section(logger, "GENERATING MANUSCRIPT CLAIMS VERIFICATION")
    
    verification_script = ROOT / "analysis" / "manuscript_claims_verification.py"
    verification_output = PAPER_OUTPUT_BASE / "MANUSCRIPT_CLAIMS_VERIFICATION.md"
    
    verification_args = [
        "--paper-run-dir", str(PAPER_OUTPUT_BASE),
        "--output", str(verification_output),
    ]
    
    if not args.dry_run:
        logger.info("  Generating manuscript claims verification report...")
        verification_success = run_python_script(verification_script, verification_args, logger, dry_run=args.dry_run)
        
        if verification_success:
            logger.info(f"  ✓ Verification report: {verification_output}")
        else:
            logger.warning(f"  ⚠ Verification report generation failed (non-critical)")
    else:
        logger.info(f"  [DRY RUN] Would generate verification report")
    
    # Summary
    log_section(logger, "PIPELINE COMPLETE")
    
    logger.info("")
    logger.info(f"  Figures:              {FIGURES_DIR}")
    logger.info(f"  Supplementary:        {SUPP_FIGURES_DIR}")
    if not args.figures_only:
        logger.info(f"  Data:                 {DATA_OUTPUT_DIR}")
    logger.info(f"  Verification Report:  {verification_output}")
    logger.info(f"  Log file:             {log_file}")
    logger.info("")
    
    if success:
        logger.info("✓ Pipeline completed successfully")
        return 0
    else:
        logger.warning("⚠ Pipeline completed with some failures - check logs")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures and data for the regulatory simulation paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--models-config",
        type=str,
        default=str(DEFAULT_MODELS_CONFIG),
        help=f"Path to CSV file specifying which models to use (default: {DEFAULT_MODELS_CONFIG})"
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only generate figures, skip data collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--si-experiment-dir",
        type=str,
        help="Override SI experiment directory (default: auto-detect latest)"
    )
    parser.add_argument(
        "--tr-experiment-dir",
        type=str,
        help="Override Therapy Request experiment directory (default: auto-detect latest)"
    )
    parser.add_argument(
        "--te-experiment-dir",
        type=str,
        help="Override Therapy Engagement experiment directory (default: auto-detect latest)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Cache directory (default: cache). Use cache_v2 for V2 cache."
    )
    
    args = parser.parse_args()
    
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())

