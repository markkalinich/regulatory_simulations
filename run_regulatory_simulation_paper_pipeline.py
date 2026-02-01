#!/usr/bin/env python3
"""
Regulatory Simulation Paper Pipeline - Generate All Figures and Data for Publication

This script orchestrates the complete pipeline for generating all figures
and data needed for the regulatory simulation paper revisions to address reviewer feedback..

Main Figures:
    Figure 3: Psychiatrist review breakdown (approved/modified/removed) - 3-panel barplot
    Figure 4: Model performance metrics (parse rate, sens/spec/accuracy/f1)
    Figure 5: P1/P2/P_harm risk analysis (failure_multiplier m values from config)

Supplementary Figures:
    Figure S4: Sankey diagrams (SI, therapy request, therapy engagement)
    Figures S5-S7: Binary confusion matrices (SI, Therapy Request, Therapy Engagement)
    Figures S8-S10: Per-statement accuracy heatmaps (SI, Therapy Request, Therapy Engagement)
    Figure S11: P2 vs P(lack of care leading to harm) Across Failure Multiplier Values

Output Structure:
    results/REGULATORY_SIMULATION_PAPER/[YYYYMMDD_HHMMSS]/
        README.md                        # Explains directory structure
        Figures/
            figure_3.png                 # Psychiatrist review breakdown
            figure_4.png                 # Model performance metrics
            figure_5/                    # P1/P2/P_harm risk analysis
        Supplementary_Figures/
            figure_S4/                   # Sankey diagrams
            figures_S5-S7/               # Binary confusion matrices
            figures_S8-S10/              # Per-statement accuracy heatmaps  
            figure_S11/                  # P2 across failure multiplier values
        Data/
            raw_data/
                model_info/              # Model configuration (paper_models_config.csv + manifest.json)
                model_inputs/
                    prompts/             # Classification prompts + Gemini prompts
                    statements/          # Finalized input datasets (SI, TR, TE)
                    table_s7_parameters.csv  # P1/P2 parameter summary
                model_outputs/           # Model prediction CSVs by task
            processed_data/              # Psychiatrist review files, comprehensive metrics
        Logs/
            pipeline.log
            manuscript_claims_verification.md
            figure_provenance/           # Provenance JSON files for all figures
            Audits/                      # Audit reports (CSV + JSON summaries)

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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config.regulatory_paper_parameters import RISK_MODEL_PARAMS
from utilities.cache_audit import audit_cache

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
LOGS_DIR = PAPER_OUTPUT_BASE / "Logs"


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
    """Generate Figure 3: Psychiatrist Review Breakdown (3-panel barplot)."""
    log_subsection(logger, "Figure 3: Psychiatrist Review Breakdown")
    
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
    """Generate Figure 5: P1/P2/P_harm Risk Analysis for m=1.0 (baseline).
    
    Note: Figure S11 shows P2 across all failure multiplier values.
    """
    log_subsection(logger, "Figure 5: P1/P2/P_harm Risk Analysis")
    
    script = ROOT / "analysis" / "comparative_analysis" / "p1_and_p2_plot_provenance.py"
    
    # Get paths to filtered comprehensive_metrics.csv for each task
    si_csv = filtered_csvs['suicidal_ideation']
    tr_csv = filtered_csvs['therapy_request']
    te_csv = filtered_csvs['therapy_engagement']
    
    m = 1.0  # Only generate m=1.0 for Figure 5 (Figure S11 shows all M values)
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
        return True
    
    success = run_python_script(script, args, logger, dry_run=dry_run)
    
    if success:
        # Find and copy the output - structure is {date}/{timestamp_name}/file.png
        src_dir = RESULTS_DIR / "risk_analysis"
        if src_dir.exists():
            date_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            if date_dirs:
                timestamp_dirs = sorted([d for d in date_dirs[0].iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                if timestamp_dirs:
                    src_output_dir = timestamp_dirs[0]
                    # Copy figure_5.png
                    src_png = src_output_dir / 'figure_5.png'
                    if src_png.exists():
                        shutil.copy(src_png, FIGURES_DIR / src_png.name)
                        logger.info(f"    ✓ Saved: Figures/{src_png.name}")
                    
                    # Copy CSV to Data/processed_data/correlated_failure_analysis/
                    for csv_file in src_output_dir.glob('p1_p2*.csv'):
                        corr_dir = DATA_OUTPUT_DIR / "processed_data" / "correlated_failure_analysis"
                        corr_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy(csv_file, corr_dir / csv_file.name)
                        logger.info(f"    ✓ Saved: Data/processed_data/correlated_failure_analysis/{csv_file.name}")
    
    return success


def generate_figure_s4(logger: logging.Logger, dry_run: bool = False) -> bool:
    """Generate Figure S4: Sankey Diagrams."""
    log_subsection(logger, "Figure S4: Sankey Diagrams")
    
    script = ROOT / "analysis" / "data_validation" / "sankey_diagram_configs.py"
    output_dir = SUPP_FIGURES_DIR / "figure_S4"
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
                        src_exp_dir = exp_dirs[0]
                        for f in src_exp_dir.iterdir():
                            if f.is_file() and f.name.endswith('.png'):
                                dst_file = output_dir / f.name
                                shutil.copy(f, dst_file)
                                logger.info(f"    ✓ Saved: figure_S4/{f.name}")
        else:
            all_success = False
    
    return all_success


def generate_confusion_matrices(
    experiment_dirs: Dict[str, Path],
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """Generate Figures S5-S7: Binary Confusion Matrices (consolidated into one folder)."""
    log_subsection(logger, "Figures S5-S7: Confusion Matrices")
    
    script = ROOT / "analysis" / "model_performance" / "generate_confusion_matrix_figures.py"
    
    # Consolidated output directory for all confusion matrices
    output_dir = SUPP_FIGURES_DIR / "figures_S5-S7"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map task to figure number: SI=S5, TR=S6, TE=S7
    task_figure_map = {
        'suicidal_ideation': ('SI', 'figure_S5.png'),
        'therapy_request': ('Therapy Request', 'figure_S6.png'),
        'therapy_engagement': ('Therapy Engagement', 'figure_S7.png'),
    }
    
    all_success = True
    
    for task_name, (display_name, figure_name) in task_figure_map.items():
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
            # Rename to figure_S5/S6/S7.png
            old_name = output_dir / f"{task_name}_binary_confusion_matrix_grid.png"
            new_name = output_dir / figure_name
            if old_name.exists():
                old_name.rename(new_name)
                logger.info(f"    ✓ Saved: {figure_name}")
            else:
                logger.warning(f"    ⚠ Output not found: {old_name.name}")
        else:
            all_success = False
    
    # Clean up any timestamp subfolders created by the script
    for subdir in output_dir.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            shutil.rmtree(subdir)
    
    if all_success and not dry_run:
        logger.info(f"  ✓ All confusion matrices saved to: figures_S5-S7/")
    
    return all_success


def generate_heatmaps(
    experiment_dirs: Dict[str, Path],
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """Generate Figures S8-S10: Per-statement Accuracy Heatmaps (consolidated into one folder)."""
    log_subsection(logger, "Figures S8-S10: Accuracy Heatmaps")
    
    # First generate the correctness matrices
    matrix_script = ROOT / "analysis" / "model_performance" / "generate_model_statement_matrices.py"
    heatmap_script = ROOT / "analysis" / "model_performance" / "generate_all_heatmaps.py"
    
    # Consolidated output directory
    output_dir = SUPP_FIGURES_DIR / "figures_S8-S10"
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
    logger.info("  Step 3: Copying outputs to figures_S8-S10/...")
    
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
    
    # Map task names to final figure names: SI=S8, TR=S9, TE=S10
    task_to_figure = {
        'suicidal_ideation': 'figure_S8.png',
        'therapy_request': 'figure_S9.png',
        'therapy_engagement': 'figure_S10.png',
    }
    
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
        figure_name = task_to_figure[task_name]
        
        # Look for the PNG file in this directory
        found = False
        for f in heatmap_src_dir.iterdir():
            if f.is_file() and f.name.endswith('.png'):
                dst_file = output_dir / figure_name
                shutil.copy(f, dst_file)
                logger.info(f"    ✓ Saved: {figure_name}")
                found = True
                break
        
        if not found:
            logger.warning(f"    ⚠ No PNG found in: {heatmap_src_dir}")
            all_success = False
    
    if all_success:
        logger.info(f"  ✓ All heatmaps saved to: figures_S8-S10/")
    
    return all_success


def generate_figure_s11(
    filtered_csvs: Dict[str, Path],
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """Generate Figure S11: P2 by Harm Prevalence Across Failure Multiplier Values.
    
    Shows how P2 varies with harm prevalence at different failure multiplier (M) values.
    Uses the same underlying data as Figure 5 but shows all M values in a single facet plot.
    """
    log_subsection(logger, "Figure S11: P2 Across M Values")
    
    script = ROOT / "analysis" / "comparative_analysis" / "figure_s11_p2_by_model_size_across_m.py"
    output_dir = SUPP_FIGURES_DIR / "figure_S11"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    args = [
        "--si-metrics", str(filtered_csvs["suicidal_ideation"]),
        "--tr-metrics", str(filtered_csvs["therapy_request"]),
        "--te-metrics", str(filtered_csvs["therapy_engagement"]),
        "--output-dir", str(output_dir),
    ]
    
    success = run_python_script(script, args, logger, dry_run=dry_run)
    
    if success and not dry_run:
        # Rename the output file to figure_S11.png
        old_name = output_dir / "figure_s11_p2_across_m_values.png"
        new_name = output_dir / "figure_S11.png"
        if old_name.exists():
            old_name.rename(new_name)
            logger.info(f"  ✓ Saved: figure_S11/figure_S11.png")
        else:
            logger.warning(f"  ⚠ Output not found: {old_name.name}")
        
        # Move CSV outputs to Data/processed_data/correlated_failure_analysis/
        corr_dir = DATA_OUTPUT_DIR / "processed_data" / "correlated_failure_analysis"
        corr_dir.mkdir(parents=True, exist_ok=True)
        
        for csv_file in list(output_dir.glob("p1_p2_p_harm_values_m_*.csv")):
            dst_file = corr_dir / csv_file.name
            shutil.move(str(csv_file), str(dst_file))  # Move instead of copy
            logger.info(f"  ✓ Saved: Data/processed_data/correlated_failure_analysis/{csv_file.name}")
        
        # Clean up any timestamp subfolders created by the script
        for subdir in output_dir.iterdir():
            if subdir.is_dir() and subdir.name.isdigit():
                shutil.rmtree(subdir)
    
    return success

# =============================================================================
# Data Collection
# =============================================================================

def collect_raw_data(
    experiment_dirs: Dict[str, Path],
    logger: logging.Logger, 
    dry_run: bool = False
) -> bool:
    """Collect raw datasets and model outputs."""
    log_subsection(logger, "Collecting Raw Data")
    
    raw_data_dir = DATA_OUTPUT_DIR / "raw_data"
    
    # Statements go under model_inputs/statements/
    statements_dir = raw_data_dir / "model_inputs" / "statements"
    statements_dir.mkdir(parents=True, exist_ok=True)
    
    # Finalized datasets (model inputs) - only the 3 actual input files used
    finalized_files = [
        "SI_finalized_sentences.csv",
        "therapy_request_finalized_sentences.csv",
        "therapy_engagement_finalized_sentences.csv",
    ]
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would copy input files and model outputs")
        return True
    
    # Copy finalized datasets (inputs) to model_inputs/statements/
    src_dir = DATA_DIR / "inputs" / "finalized_input_data"
    for filename in finalized_files:
        src = src_dir / filename
        if src.exists():
            shutil.copy(src, statements_dir / filename)
            logger.info(f"  ✓ Copied: model_inputs/statements/{filename}")
        else:
            logger.warning(f"  ⚠ Not found: {filename}")
    
    # Copy model output CSVs (actual model predictions)
    model_outputs_dir = raw_data_dir / "model_outputs"
    model_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    for task_name, exp_dir in experiment_dirs.items():
        task_outputs_dir = model_outputs_dir / task_name
        task_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        outputs_src = exp_dir / "model_outputs"
        if outputs_src.exists():
            csv_count = 0
            for csv_file in outputs_src.glob("*.csv"):
                shutil.copy(csv_file, task_outputs_dir / csv_file.name)
                csv_count += 1
            logger.info(f"  ✓ Copied: {task_name} model outputs ({csv_count} files)")
        else:
            logger.warning(f"  ⚠ Not found: {task_name} model_outputs/")
    
    return True


def collect_processed_data(
    experiment_dirs: Dict[str, Path],
    logger: logging.Logger, 
    dry_run: bool = False
) -> bool:
    """Collect processed data files into organized subdirectories."""
    log_subsection(logger, "Collecting Processed Data")
    
    processed_dir = DATA_OUTPUT_DIR / "processed_data"
    
    # Create organized subdirectories
    psychiatrist_dir = processed_dir / "psychiatrist_statement_review"
    metrics_dir = processed_dir / "model_performance_metrics"
    difficult_dir = processed_dir / "difficult_statement_analysis"
    # Note: correlated_failure_analysis/ is created during figure generation
    
    for d in [psychiatrist_dir, metrics_dir, difficult_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would copy review files and comprehensive metrics")
        return True
    
    # Copy psychiatrist review files to psychiatrist_statement_review/
    review_files = [
        "SI_psychiatrist_01_and_02_scores.csv",
        "therapy_request_psychiatrist_01_and_02_scores.csv",
        "therapy_engagement_psychiatrist_01_and_02_scores.csv",
    ]
    
    src_dir = DATA_DIR / "inputs" / "intermediate_files"
    for filename in review_files:
        src = src_dir / filename
        if src.exists():
            shutil.copy(src, psychiatrist_dir / filename)
            logger.info(f"  ✓ Copied: psychiatrist_statement_review/{filename}")
        else:
            logger.warning(f"  ⚠ Not found: {filename}")
    
    # Copy comprehensive_metrics.csv to model_performance_metrics/ (skip if already filtered)
    for task_name, exp_dir in experiment_dirs.items():
        dst_name = f"{task_name}_comprehensive_metrics.csv"
        dst_path = metrics_dir / dst_name
        
        # Skip if already exists (from filtering step)
        if dst_path.exists():
            logger.info(f"  ✓ Already exists: model_performance_metrics/{dst_name} (filtered)")
            continue
            
        metrics_file = exp_dir / "tables" / "comprehensive_metrics.csv"
        if metrics_file.exists():
            shutil.copy(metrics_file, dst_path)
            logger.info(f"  ✓ Copied: model_performance_metrics/{dst_name}")
        else:
            logger.warning(f"  ⚠ Not found: {task_name} comprehensive_metrics.csv")
    
    # Copy difficult statements breakdown to difficult_statement_analysis/
    # (Summary files removed - breakdown contains all needed info including metadata)
    review_stats_dir = RESULTS_DIR / "review_statistics"
    difficult_stmt_files = [
        "suicidal_ideation_difficult_statements_breakdown.csv",
        "therapy_request_difficult_statements_breakdown.csv",
        "therapy_engagement_difficult_statements_breakdown.csv",
    ]
    
    for filename in difficult_stmt_files:
        src = review_stats_dir / filename
        if src.exists():
            shutil.copy(src, difficult_dir / filename)
            logger.info(f"  ✓ Copied: difficult_statement_analysis/{filename}")
        else:
            logger.warning(f"  ⚠ Not found: {filename}")
    
    return True


def collect_prompts(logger: logging.Logger, dry_run: bool = False) -> bool:
    """Collect prompt files to raw_data/model_inputs/prompts/."""
    log_subsection(logger, "Collecting Prompts")
    
    # Prompts go under raw_data/model_inputs/prompts/
    prompts_dir = DATA_OUTPUT_DIR / "raw_data" / "model_inputs" / "prompts"
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
            logger.info(f"  ✓ Copied: model_inputs/prompts/{filename}")
        else:
            logger.warning(f"  ⚠ Not found: {filename}")
    
    # Copy Gemini prompts
    gemini_src_dir = src_dir / "gemini_prompts"
    gemini_dst_dir = prompts_dir / "gemini_prompts"
    if gemini_src_dir.exists():
        shutil.copytree(gemini_src_dir, gemini_dst_dir)
        logger.info(f"  ✓ Copied: model_inputs/prompts/gemini_prompts/")
    
    return True


def generate_table_s7(logger: logging.Logger, dry_run: bool = False) -> bool:
    """Generate Table S7: P1/P2 Parameter Summary to raw_data/model_inputs/."""
    log_subsection(logger, "Generating Table S7: P1/P2 Parameters")
    
    script = ROOT / "analysis" / "generate_table_s7.py"
    # Table S7 goes under raw_data/model_inputs/
    output_dir = DATA_OUTPUT_DIR / "raw_data" / "model_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "table_s7_parameters.csv"
    
    args = [
        "-o", str(output_file),
    ]
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would generate Table S7")
        return True
    
    success = run_python_script(script, args, logger, dry_run=dry_run)
    
    if success:
        logger.info(f"  ✓ Saved: raw_data/model_inputs/table_s7_parameters.csv")
    
    return success


def collect_model_info(
    paper_models: dict,
    models_config_path: Path,
    logger: logging.Logger, 
    dry_run: bool = False
) -> bool:
    """Extract model information to raw_data/model_info/."""
    log_subsection(logger, "Collecting Model Information")
    
    import pandas as pd
    
    # Model info goes under raw_data/model_info/
    model_info_dir = DATA_OUTPUT_DIR / "raw_data" / "model_info"
    model_info_dir.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would create model_info.csv")
        return True
    
    # Read from the models config that was passed in (regulatory_paper_models.csv)
    if not models_config_path.exists():
        logger.warning(f"  ⚠ Config not found: {models_config_path}")
        return False
    
    df = pd.read_csv(models_config_path)
    
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
        logger.info(f"  ✓ Created: raw_data/model_info/paper_models_config.csv ({len(paper_df)} models)")
    
    # Create a manifest
    manifest = {
        "paper_models": paper_models,
        "models_config_file": str(models_config_path),
        "generated": datetime.now().isoformat(),
    }
    manifest_path = model_info_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  ✓ Created: raw_data/model_info/manifest.json")
    
    return True


def extract_provenance_from_pngs(logger: logging.Logger, dry_run: bool = False) -> bool:
    """Extract provenance JSON from all PNG figures to Logs/figure_provenance/."""
    log_subsection(logger, "Extracting Provenance from PNGs")
    
    try:
        from PIL import Image
    except ImportError:
        logger.error("  ✗ PIL/Pillow not available - cannot extract provenance")
        return False
    
    # Provenance goes under Logs/figure_provenance/
    provenance_dir = LOGS_DIR / "figure_provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would extract provenance from PNGs")
        return True
    
    # Define all PNG files to extract from
    png_files = [
        # Main figures
        (FIGURES_DIR / "figure_3.png", "figure_3"),
        (FIGURES_DIR / "figure_4.png", "figure_4"),
        (FIGURES_DIR / "figure_5.png", "figure_5"),
        # Supplementary figures
        (SUPP_FIGURES_DIR / "figure_S4" / "si_psychiatrist_review_sankey.png", "figure_S4_si"),
        (SUPP_FIGURES_DIR / "figure_S4" / "therapy_request_psychiatrist_review_sankey.png", "figure_S4_therapy_request"),
        (SUPP_FIGURES_DIR / "figure_S4" / "therapy_engagement_psychiatrist_review_sankey.png", "figure_S4_therapy_engagement"),
        (SUPP_FIGURES_DIR / "figures_S5-S7" / "figure_S5.png", "figure_S5"),
        (SUPP_FIGURES_DIR / "figures_S5-S7" / "figure_S6.png", "figure_S6"),
        (SUPP_FIGURES_DIR / "figures_S5-S7" / "figure_S7.png", "figure_S7"),
        (SUPP_FIGURES_DIR / "figures_S8-S10" / "figure_S8.png", "figure_S8"),
        (SUPP_FIGURES_DIR / "figures_S8-S10" / "figure_S9.png", "figure_S9"),
        (SUPP_FIGURES_DIR / "figures_S8-S10" / "figure_S10.png", "figure_S10"),
        (SUPP_FIGURES_DIR / "figure_S11" / "figure_S11.png", "figure_S11"),
    ]
    
    extracted_count = 0
    missing_count = 0
    no_provenance_count = 0
    
    for png_path, figure_name in png_files:
        if not png_path.exists():
            logger.warning(f"  ⚠ PNG not found: {png_path.name}")
            missing_count += 1
            continue
        
        try:
            img = Image.open(png_path)
            # Try both lowercase and uppercase keys
            prov_json = img.text.get('provenance') or img.text.get('Provenance')
            
            if prov_json:
                # Save extracted provenance to JSON file
                json_path = provenance_dir / f"{figure_name}_provenance.json"
                prov_data = json.loads(prov_json)
                with open(json_path, 'w') as f:
                    json.dump(prov_data, f, indent=2)
                logger.info(f"  ✓ Extracted: {figure_name}_provenance.json")
                extracted_count += 1
            else:
                logger.warning(f"  ⚠ No provenance in: {png_path.name}")
                no_provenance_count += 1
        except Exception as e:
            logger.error(f"  ✗ Error extracting from {png_path.name}: {e}")
            no_provenance_count += 1
    
    logger.info(f"")
    logger.info(f"  Summary: {extracted_count} extracted, {missing_count} missing, {no_provenance_count} without provenance")
    
    return no_provenance_count == 0  # Return False if any PNGs are missing provenance


def generate_readme(logger: logging.Logger, dry_run: bool = False) -> bool:
    """Generate README.md explaining the directory structure."""
    log_subsection(logger, "Generating README")
    
    readme_path = PAPER_OUTPUT_BASE / "README.md"
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would generate README.md")
        return True
    
    readme_content = f"""# Regulatory Simulation Paper - Pipeline Output

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This directory contains all figures, data, and logs generated by the regulatory simulation paper pipeline.

## Directory Structure

```
{PAPER_OUTPUT_BASE.name}/
├── README.md                    # This file
├── Figures/                     # Main manuscript figures
│   ├── figure_3.png            # Psychiatrist review breakdown (approved/modified/removed)
│   ├── figure_4.png            # Model performance metrics (parse rate, sensitivity, specificity, accuracy, F1)
│   └── figure_5/               # P1/P2/P_harm risk analysis across failure multiplier values
│
├── Supplementary_Figures/       # Supplementary figures
│   ├── figure_S4/              # Sankey diagrams showing ground truth → model prediction flows
│   ├── figures_S5-S7/          # Binary confusion matrices (SI, Therapy Request, Therapy Engagement)
│   ├── figures_S8-S10/         # Per-statement accuracy heatmaps showing model × statement performance
│   └── figure_S11/             # P2 across failure multiplier (M) values
│
├── Data/
│   ├── raw_data/
│   │   ├── model_info/         # Model configuration used in this run
│   │   │   ├── paper_models_config.csv   # Models included in analysis
│   │   │   └── manifest.json             # Run metadata
│   │   ├── model_inputs/
│   │   │   ├── prompts/        # Classification prompts sent to models
│   │   │   │   ├── system_suicide_detection_v2.txt
│   │   │   │   ├── therapy_request_classifier_v3.txt
│   │   │   │   ├── therapy_engagement_conversation_prompt_v2.txt
│   │   │   │   └── gemini_prompts/       # Prompts used to generate synthetic data
│   │   │   ├── statements/     # Input datasets (ground truth statements)
│   │   │   │   ├── SI_finalized_sentences.csv
│   │   │   │   ├── therapy_request_finalized_sentences.csv
│   │   │   │   └── therapy_engagement_finalized_sentences.csv
│   │   │   └── table_s7_parameters.csv   # P1/P2 risk model parameters (Supplementary Table 7)
│   │   └── model_outputs/      # Raw model predictions by task
│   │       ├── suicidal_ideation/
│   │       ├── therapy_request/
│   │       └── therapy_engagement/
│   │
│   └── processed_data/         # Analysis outputs
│       ├── model_performance_metrics/    # Comprehensive metrics CSVs
│       ├── psychiatrist_statement_review/  # Psychiatrist review data
│       ├── correlated_failure_analysis/    # P1/P2 data across M values
│       └── difficult_statement_analysis/   # Statements difficult for models
│
└── Logs/
    ├── pipeline.log            # Full pipeline execution log
    ├── manuscript_claims_verification.md   # Verification of manuscript claims vs data
    ├── figure_provenance/      # Provenance JSON files
    │   ├── figure_*_provenance.json       # Figure provenance (input hashes, timestamps)
    │   └── *_audit_summary.json           # Audit provenance (verification summaries)
    └── Audits/                 # Detailed audit reports (CSV)
        ├── cache_audit_report.csv              # Cache integrity check
        ├── confusion_matrix_audit_report.csv   # Figures 4, S5-S7 verification
        ├── heatmap_audit_report.csv            # Figures S8-S10 verification
        └── figure_s11_audit_report.csv         # Figure S11 verification
```

## Key Files

### Figures
- **figure_3.png**: Shows psychiatrist review outcomes for each task
- **figure_4.png**: Model performance comparison with binary classification metrics
- **figure_5/**: Risk analysis showing P1, P2, and P_harm across model sizes

### Verification & Provenance
- **Logs/figure_provenance/**: JSON files with complete provenance
  - `figure_*_provenance.json`: SHA-256 hashes of input files, timestamps, parameters
  - `*_audit_summary.json`: Audit verification summaries (what was checked, pass/fail)
- **Logs/Audits/**: Detailed CSV audit reports
  - Each audit recalculates metrics from raw data and compares to reported values
  - All audits should show 100% pass rate for a valid pipeline run

## Reproducibility

To regenerate these results:

```bash
python run_regulatory_simulation_paper_pipeline.py --models-config config/regulatory_paper_models.csv
```

The pipeline uses cached LLM responses from `regulatory_paper_cache_v3/results.db` to ensure exact reproducibility.
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"  ✓ Created: README.md")
    return True


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> int:
    """Run the complete paper pipeline."""
    
    # Load models from config file
    models_config_path = Path(args.models_config)
    paper_models = load_models_from_config(models_config_path)
    
    # Setup - create output directory and Logs folder
    PAPER_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    logs_dir = PAPER_OUTPUT_BASE / "Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"
    
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
    
    # Verify all models used Q8_0 quantization
    log_subsection(logger, "Verifying Q8_0 Quantization")
    if not args.dry_run:
        import sqlite3
        cache_db = ROOT / args.cache_dir / "results.db"
        if cache_db.exists():
            conn = sqlite3.connect(cache_db)
            cursor = conn.execute('''
                SELECT DISTINCT mf.model_key, ck.quantization_name
                FROM cache_keys ck
                JOIN model_files mf ON ck.model_path = mf.model_path
                ORDER BY mf.model_key
            ''')
            quants = cursor.fetchall()
            conn.close()
            
            non_q8_models = [(model, quant) for model, quant in quants if quant != 'Q8_0']
            
            if non_q8_models:
                logger.error(f"  ✗ Found {len(non_q8_models)} models with non-Q8_0 quantization:")
                for model, quant in non_q8_models:
                    logger.error(f"    {model}: {quant}")
                logger.error(f"  All models must be Q8_0 for paper consistency")
                return 1
            else:
                logger.info(f"  ✓ All {len(quants)} models verified as Q8_0")
        else:
            logger.warning(f"  ⚠ Cache database not found: {cache_db}")
            logger.warning(f"    Cannot verify quantizations - proceeding anyway")
    else:
        logger.info(f"  [DRY RUN] Would verify Q8_0 quantization")
    
    success = True
    
    # Create filtered CSVs only for overrides (regenerated data already has only paper models)
    filtered_csvs = {}
    
    for task_name, exp_dir in experiment_dirs.items():
        if needs_filtering.get(task_name, False):
            # Override provided - need to filter
            log_subsection(logger, f"Filtering {task_name} to paper models")
            metrics_dir = DATA_OUTPUT_DIR / "processed_data" / "model_performance_metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            input_csv = exp_dir / "tables" / "comprehensive_metrics.csv"
            output_csv = metrics_dir / f"{task_name}_comprehensive_metrics.csv"
            
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
    
    if not generate_figure_s4(logger, args.dry_run):
        success = False
    
    if not generate_confusion_matrices(experiment_dirs, logger, args.dry_run):
        success = False
    
    if not generate_heatmaps(experiment_dirs, logger, args.dry_run):
        success = False
    
    if not generate_figure_s11(filtered_csvs, logger, args.dry_run):
        success = False
    
    # Collect data (unless figures-only)
    if not args.figures_only:
        log_section(logger, "COLLECTING DATA")
        
        if not collect_raw_data(experiment_dirs, logger, args.dry_run):
            success = False
        
        if not collect_processed_data(experiment_dirs, logger, args.dry_run):
            success = False
        
        
        if not collect_prompts(logger, args.dry_run):
            success = False
        
        if not collect_model_info(paper_models, models_config_path, logger, args.dry_run):
            success = False
        
        if not generate_table_s7(logger, args.dry_run):
            success = False
    else:
        logger.info("")
        logger.info("⏭  Skipping data collection (--figures-only)")
    
    # Extract provenance from PNGs to verify embedding and create JSON files
    log_section(logger, "EXTRACTING FIGURE PROVENANCE FROM PNGS")
    if not extract_provenance_from_pngs(logger, args.dry_run):
        logger.warning("  ⚠ Some figures missing embedded provenance (non-critical)")
        # Don't fail pipeline for missing provenance
    
    # Generate review statistics (required for manuscript claims verification)
    log_section(logger, "GENERATING REVIEW STATISTICS")
    
    if not args.dry_run:
        logger.info("  Generating psychiatrist review statistics...")
        
        # Generate review statistics (si, therapy_request, therapy_engagement)
        review_stats_script = ROOT / "analysis" / "statistics" / "generate_review_statistics.py"
        review_stats_success = run_python_script(review_stats_script, [], logger, dry_run=args.dry_run)
        
        if review_stats_success:
            logger.info(f"  ✓ Review statistics generated")
            
            # Generate chi-squared tests (depends on review statistics)
            logger.info("  Generating chi-squared tests...")
            chi_sq_script = ROOT / "analysis" / "statistics" / "calculate_review_statistics.py"
            chi_sq_success = run_python_script(chi_sq_script, [], logger, dry_run=args.dry_run)
            
            if chi_sq_success:
                logger.info(f"  ✓ Chi-squared tests generated")
            else:
                logger.warning(f"  ⚠ Chi-squared tests generation failed")
        else:
            logger.warning(f"  ⚠ Review statistics generation failed")
    else:
        logger.info(f"  [DRY RUN] Would generate review statistics and chi-squared tests")
    
    # Generate manuscript claims verification report
    log_section(logger, "GENERATING MANUSCRIPT CLAIMS VERIFICATION")
    
    verification_script = ROOT / "analysis" / "manuscript_claims_verification.py"
    verification_output = logs_dir / "manuscript_claims_verification.md"
    
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
    
    # Audits - all go to Logs/Audits/
    log_section(logger, "AUDITS")
    audits_dir = LOGS_DIR / "Audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.dry_run and not args.figures_only:
        # Cache Audit
        logger.info("  Running cache audit...")
        try:
            audit_output = audits_dir / "cache_audit_report.csv"
            audit_df = audit_cache(args.cache_dir, str(models_config_path), verbose=False)
            audit_df.to_csv(audit_output, index=False)
            
            # Log summary stats
            total_models = len(audit_df)
            models_with_issues = len(audit_df[
                (audit_df['SI_success'] < audit_df['SI_total']) |
                (audit_df['TR_success'] < audit_df['TR_total']) |
                (audit_df['TE_success'] < audit_df['TE_total']) |
                (audit_df['wrong_params'] > 0)
            ])
            
            logger.info(f"  ✓ Cache audit complete: {total_models} models audited")
            if models_with_issues > 0:
                logger.info(f"    ⚠ {models_with_issues} models have parse failures or parameter issues")
        except Exception as e:
            logger.warning(f"  ⚠ Cache audit failed (non-critical): {e}")
        
        # Confusion Matrix Audit (Figures 4, S5-S7)
        logger.info("  Running confusion matrix audit (Figures 4, S5-S7)...")
        cm_audit_script = ROOT / "utilities" / "confusion_matrix_audit.py"
        cm_audit_args = [
            "--paper-run-dir", str(PAPER_OUTPUT_BASE),
            "--cache-dir", args.cache_dir,
            "--output", str(audits_dir / "confusion_matrix_audit_report.csv"),
        ]
        cm_audit_success = run_python_script(cm_audit_script, cm_audit_args, logger, dry_run=args.dry_run)
        if cm_audit_success:
            logger.info(f"  ✓ Confusion matrix audit passed")
        else:
            logger.warning(f"  ⚠ Confusion matrix audit FAILED - review report")
        
        # Heatmap Audit (Figures S8-S10)
        logger.info("  Running heatmap audit (Figures S8-S10)...")
        hm_audit_script = ROOT / "utilities" / "heatmap_audit.py"
        hm_audit_args = [
            "--paper-run-dir", str(PAPER_OUTPUT_BASE),
            "--output", str(audits_dir / "heatmap_audit_report.csv"),
        ]
        hm_audit_success = run_python_script(hm_audit_script, hm_audit_args, logger, dry_run=args.dry_run)
        if hm_audit_success:
            logger.info(f"  ✓ Heatmap audit passed")
        else:
            logger.warning(f"  ⚠ Heatmap audit FAILED - review report")
        
        # Figure S11 Audit (P2 across failure multiplier values)
        logger.info("  Running Figure S11 audit (P2 across M values)...")
        s11_audit_script = ROOT / "utilities" / "figure_s11_audit.py"
        s11_audit_args = [
            "--paper-run-dir", str(PAPER_OUTPUT_BASE),
            "--output", str(audits_dir / "figure_s11_audit_report.csv"),
        ]
        s11_audit_success = run_python_script(s11_audit_script, s11_audit_args, logger, dry_run=args.dry_run)
        if s11_audit_success:
            logger.info(f"  ✓ Figure S11 audit passed")
        else:
            logger.warning(f"  ⚠ Figure S11 audit FAILED - review report")
        
        logger.info(f"  ✓ Audit reports saved to: {audits_dir}")
    else:
        logger.info(f"  [SKIPPED] Audits (dry-run or figures-only mode)")
    
    # Generate README
    generate_readme(logger, args.dry_run)
    
    # Summary
    log_section(logger, "PIPELINE COMPLETE")
    
    logger.info("")
    logger.info(f"  Figures:              {FIGURES_DIR}")
    logger.info(f"  Supplementary:        {SUPP_FIGURES_DIR}")
    if not args.figures_only:
        logger.info(f"  Data:                 {DATA_OUTPUT_DIR}")
    logger.info(f"  Logs:                 {logs_dir}")
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
        default="regulatory_paper_cache_v3",
        help="Cache directory (default: regulatory_paper_cache_v3)"
    )
    
    args = parser.parse_args()
    
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())

