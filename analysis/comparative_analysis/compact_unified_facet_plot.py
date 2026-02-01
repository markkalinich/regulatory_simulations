#!/usr/bin/env python3
"""
Compact Unified Facet Plot - All Classification Tasks on a Single Figure

Layout:
    Rows = Model families (Gemma, Llama, Qwen)
    Columns = Classification tasks (Suicidal Ideation, Therapy Request, Therapy Engagement)
    Y-axis = F1 Score
    
Generates 7 plot variants:
    Base plots (no trendlines):
        1. F1 vs Version: x=version (v1-v3, 3.1+), color=fine-tune, size=params
        2. F1 vs Parameters: x=params (log scale), color=fine-tune, shape=version, size=version
    
    With overall trendlines (1 per subplot):
        3. F1 vs Version + overall trend
        4. F1 vs Parameters + overall trend
    
    With per-type trendlines (up to 5 per subplot, min 3 points):
        5. F1 vs Version + by-type trends
        6. F1 vs Parameters + by-type trends
    
    With per-version trendlines (up to 4 per subplot, min 3 points):
        7. F1 vs Parameters + by-version trends

Version Grouping:
    - Version "3.1+" represents grouped later versions for Llama and Gemma:
        * Llama: 3.1, 3.2, 3.3 → internally 3.5 → displayed as "3.1+"
        * Gemma: 2.27, 2.9 → internally 3.5 → displayed as "3.1+"
    - This grouping treats architecturally similar later versions as a single generation

Safety Model Corrections:
    - ShieldGemma, Llama Guard, and Qwen Guard use non-JSON output formats
    - Metrics are recomputed from cache using native parsing for these 9 models
    - This corrects parse rates and improves performance accuracy for safety models

Trendlines & Statistics:
    - All trendlines are linear regression (scipy.stats.linregress)
    - For log-scale plots, regression is performed on log10(x) to produce straight lines
    - Minimum 3 data points required to draw a trendline
    - Bonferroni correction applied per-plot (each plot is independent experiment):
        * Overall trends: α = 0.05/9 (9 subplots)
        * By-type trends: α = 0.05/45 (9 subplots × 5 types)
        * By-version trends: α = 0.05/36 (9 subplots × 4 versions)
    - Line style indicates Bonferroni-corrected significance: solid (p < α), dashed (p ≥ α)
    - Overall trends only:
        * 95% confidence interval band shown in light gray (similar to seaborn regplot)
        * R² value annotated in upper left corner of each subplot
    - Overall trends: gray
    - By-type trends: match fine-tune type colors
    - By-version trends: brown gradient (v1=dark, 3.1+=light)

Common styling:
    - Alpha = 0.35 (transparent, no edges)
    - Increased jitter on version plot for visibility

Usage:
    .venv/bin/python analysis/comparative_analysis/compact_unified_facet_plot.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
import argparse
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Suppress polyfit warnings for cleaner output
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# Import utilities from existing modules
from facet_plot_utils import (
    get_model_metadata,
    get_param_billions_from_config,
)
from gemma_version_facet_plot import (
    GEMMA_FAMILIES,
    determine_gemma_version,
    determine_gemma_model_type,
)
from llama_version_facet_plot import (
    determine_llama_version,
    determine_llama_model_type,
)
from qwen_version_facet_plot import (
    determine_qwen_version,
    determine_qwen_model_type,
)


# =============================================================================
# Configuration
# =============================================================================

FAMILIES = ['gemma', 'llama', 'qwen']
FAMILY_LABELS = ['Gemma', 'Llama', 'Qwen']

TASKS = ['suicidal_ideation', 'therapy_request', 'therapy_engagement']
TASK_LABELS = ['Suicidal Ideation', 'Therapy Request', 'Therapy Engagement']

# Colors for fine-tune type
TYPE_COLORS = {
    'IT': '#2E86AB',           # Blue (instruct tuned)
    'PT': '#52B788',           # Sea Green (pretrained/base)
    'Medical': '#F18F01',      # Orange
    'MedGemma': '#F18F01',     # Orange
    'Safety': '#7B2D8E',       # Purple
    'ShieldGemma': '#7B2D8E',  # Purple
    'Guard': '#7B2D8E',        # Purple
    'Mental Health': '#C73E1D', # Red
}

TYPE_DISPLAY_LABELS = {
    'IT': 'Instruct',
    'PT': 'Base',
    'Medical': 'Medical',
    'Safety': 'Safety',
    'Mental Health': 'Mental Health',
}

# Shapes for version (when using shape encoding)
VERSION_MARKERS = {
    1: 'o',   # Circle - v1
    2: 's',   # Square - v2
    3: '^',   # Triangle up - v3
    4: 'D',   # Diamond - v4
}


@dataclass
class PlotConfig:
    """Configuration for a single plot variant."""
    x_type: str  # 'version' or 'params'
    use_version_shape: bool  # Only for params plot
    trendline_type: Optional[str]  # None, 'overall', 'by_type', 'by_version'
    output_suffix: str
    title_suffix: str


# =============================================================================
# Data Processing Functions
# =============================================================================

def get_canonical_type(model_type: str) -> str:
    """Map model type to canonical type for coloring and legend."""
    if model_type in ['MedGemma']:
        return 'Medical'
    elif model_type in ['ShieldGemma', 'Guard']:
        return 'Safety'
    return model_type


def determine_model_family_and_version(family: str, size: str) -> Tuple[str, float]:
    """Determine base family and version using family-specific logic."""
    family_lower = family.lower()
    
    if family in GEMMA_FAMILIES or family_lower.startswith('gemma'):
        return ('gemma', determine_gemma_version(family, size))
    elif 'llama' in family_lower:
        return ('llama', determine_llama_version(family, size))
    elif 'qwen' in family_lower:
        return ('qwen', determine_qwen_version(family, size))
    
    return ('unknown', 0)


def determine_model_type(family: str, size: str) -> str:
    """Determine model type using family-specific logic."""
    family_lower = family.lower()
    
    if family in GEMMA_FAMILIES or family_lower.startswith('gemma'):
        model_type = determine_gemma_model_type(family, size)
    elif 'llama' in family_lower:
        model_type = determine_llama_model_type(family, size)
    elif 'qwen' in family_lower:
        model_type = determine_qwen_model_type(family, size)
    else:
        model_type = 'IT'
    
    return get_canonical_type(model_type)


def load_unified_data(csv_path: str) -> pd.DataFrame:
    """Load the unified all_models_all_tasks.csv file."""
    df = pd.read_csv(csv_path)
    
    # Apply safety model corrections (ShieldGemma, Llama Guard, Qwen Guard)
    # These models use different output formats that need special parsing
    from facet_plot_utils import (
        compute_shieldgemma_metrics,
        compute_llama_guard_metrics,
        compute_qwen_guard_metrics,
        apply_guard_metrics_to_df,
    )
    
    safety_metrics = {}
    try:
        shield = compute_shieldgemma_metrics()
        safety_metrics.update(shield)
    except:
        pass
    
    try:
        llama_guard = compute_llama_guard_metrics()
        safety_metrics.update(llama_guard)
    except:
        pass
    
    try:
        qwen_guard = compute_qwen_guard_metrics()
        safety_metrics.update(qwen_guard)
    except:
        pass
    
    if safety_metrics:
        df = apply_guard_metrics_to_df(df, safety_metrics)
        print(f"Applied safety model corrections for {len(safety_metrics)} model configurations")
    
    # Add computed columns
    df['base_family'], df['version'] = zip(*df.apply(
        lambda r: determine_model_family_and_version(r['model_family'], r['model_size']), axis=1
    ))
    df['size_billions'] = df.apply(
        lambda r: get_param_billions_from_config(r['model_family'], r['model_size']), axis=1
    )
    df['model_type'] = df.apply(
        lambda r: determine_model_type(r['model_family'], r['model_size']), axis=1
    )
    
    # Filter to known families
    df = df[df['base_family'].isin(FAMILIES)].copy()
    
    return df


def normalize_version_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """Create a normalized version column for ordinal x-axis plotting.
    
    Note: Version 3.5 is an internal representation for grouped later versions:
        - Llama 3.5 represents 3.1, 3.2, 3.3 (displayed as "3.1+")
        - Gemma 3.5 represents 2.27, 2.9 (displayed as "3.1+")
    These are mapped to ordinal position 4 for consistent plotting.
    """
    df = df.copy()
    
    version_maps = {
        'gemma': {1: 1, 2: 2, 3: 3, 3.5: 4},
        'llama': {1: 1, 2: 2, 3: 3, 3.5: 4},
        'qwen': {1: 1, 2: 2, 3: 3},
    }
    
    def map_version(row):
        family = row['base_family']
        version = row['version']
        if family in version_maps and version in version_maps[family]:
            return version_maps[family][version]
        return version
    
    df['version_ordinal'] = df.apply(map_version, axis=1)
    return df


def add_jitter(values: np.ndarray, jitter_amount: float = 0.1) -> np.ndarray:
    """Add small random jitter to values to reduce overlap."""
    np.random.seed(42)
    return values + np.random.uniform(-jitter_amount, jitter_amount, size=len(values))


# =============================================================================
# Unified Plotting Function
# =============================================================================

def create_facet_plot(
    df: pd.DataFrame,
    config: PlotConfig,
    output_path: str,
    figsize: Tuple[float, float] = (14, 10),
    save_pdf: bool = False
) -> List[Dict]:
    """Create a single facet plot based on configuration.
    
    This is the core plotting function - all plot variants use this.
    
    Args:
        df: DataFrame with model performance data
        config: PlotConfig specifying x-axis type and shape encoding
        output_path: Path to save the PNG
        figsize: Figure size (width, height)
        save_pdf: Whether to also save PDF version
    
    Returns:
        List of regression statistics dictionaries
    """
    # Always normalize version column (needed for shapes even if not x-axis)
    df = normalize_version_for_plotting(df)
    
    # Get unique tasks
    tasks_in_data = [t for t in TASKS if t in df['task'].unique()]
    task_labels_in_data = [TASK_LABELS[TASKS.index(t)] for t in tasks_in_data]
    
    if len(tasks_in_data) == 0:
        print("No valid tasks found!")
        return []
    
    # Collect regression statistics
    regression_stats = []
    
    # Calculate Bonferroni-corrected significance threshold for this plot
    # Based on maximum possible number of regressions (conservative approach)
    n_subplots = len(FAMILIES) * len(tasks_in_data)  # 3 families × 3 tasks = 9
    if config.trendline_type == 'overall':
        max_regressions = n_subplots * 1  # 1 regression per subplot
    elif config.trendline_type == 'by_type':
        max_regressions = n_subplots * 5  # Up to 5 types per subplot
    elif config.trendline_type == 'by_version':
        max_regressions = n_subplots * 4  # Up to 4 versions per subplot
    else:
        max_regressions = 1  # No trendlines
    
    bonferroni_alpha = 0.05 / max_regressions if config.trendline_type else 0.05
    
    # Setup plot styling
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
    })
    
    # Create figure
    n_rows = len(FAMILIES)
    n_cols = len(tasks_in_data)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Size mapping for version plots (log scale: 0.3B -> 40, 70B -> 400)
    def size_to_marker_size(size_b):
        if pd.isna(size_b) or size_b <= 0:
            return 60
        min_size, max_size = 0.3, 80
        min_marker, max_marker = 40, 350
        log_ratio = (np.log10(size_b) - np.log10(min_size)) / (np.log10(max_size) - np.log10(min_size))
        log_ratio = np.clip(log_ratio, 0, 1)
        return min_marker + log_ratio * (max_marker - min_marker)
    
    # Size mapping for version (linear scale for params plot)
    def version_to_marker_size(version_ordinal):
        # v1=60, v2=100, v3=140, v4=180
        size_map = {1: 60, 2: 100, 3: 140, 4: 180}
        return size_map.get(version_ordinal, 100)
    
    # Plot each cell
    for fam_idx, (family, fam_label) in enumerate(zip(FAMILIES, FAMILY_LABELS)):
        for task_idx, (task, task_label) in enumerate(zip(tasks_in_data, task_labels_in_data)):
            ax = axes[fam_idx, task_idx]
            cell_data = df[(df['base_family'] == family) & (df['task'] == task)].copy()
            
            if len(cell_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
            else:
                # Group by model_type
                groups = cell_data.groupby('model_type')
                
                for model_type, type_data in groups:
                    # Get x values
                    if config.x_type == 'version':
                        x_raw = type_data['version_ordinal'].values
                        x_vals = add_jitter(x_raw, jitter_amount=0.25)  # More jitter for version
                    else:
                        x_vals = type_data['size_billions'].values
                    
                    y_vals = type_data['f1_score'].values
                    
                    # Get sizes
                    if config.x_type == 'version':
                        # Version plot: size = parameters
                        sizes = type_data['size_billions'].apply(size_to_marker_size).values
                    elif config.use_version_shape:
                        # Params plot with shapes: size = version
                        sizes = type_data['version_ordinal'].apply(version_to_marker_size).values
                    else:
                        # Params plot without shapes: fixed size
                        sizes = 100
                    
                    # Use version-based shapes for params plot, circles for version plot
                    if config.use_version_shape:
                        # Plot each point individually with version-based marker and size
                        for i, (_, row) in enumerate(type_data.iterrows()):
                            marker = VERSION_MARKERS.get(row['version_ordinal'], 'o')
                            s = sizes[i] if isinstance(sizes, np.ndarray) else sizes
                            ax.scatter(
                                x_vals[i] if isinstance(x_vals, np.ndarray) else x_vals,
                                y_vals[i],
                                c=TYPE_COLORS.get(model_type, '#888888'),
                                marker=marker,
                                s=s,
                                alpha=0.35,
                                edgecolors='none',
                                zorder=3
                            )
                    else:
                        # Plot all at once with circle marker
                        ax.scatter(
                            x_vals,
                            y_vals,
                            c=TYPE_COLORS.get(model_type, '#888888'),
                            marker='o',
                            s=sizes,
                            alpha=0.35,
                            edgecolors='none',
                            zorder=3
                        )
                
                # Draw trendlines after all scatter plots
                r_squared = None
                if config.trendline_type and len(cell_data) >= 3:
                    r_squared = _draw_trendlines(ax, cell_data, config, family, task, regression_stats, 
                                                 sig_threshold=bonferroni_alpha)
            
            # Style the axis
            _style_axis(ax, config, fam_idx, task_idx, fam_label, task_label, 
                       len(FAMILIES), len(tasks_in_data))
            
            # Add R² annotation for overall trends (upper left corner)
            if r_squared is not None:
                ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                       transform=ax.transAxes, 
                       fontsize=11, 
                       verticalalignment='top',
                       horizontalalignment='left')
    
    # Add legend
    _add_legend(fig, config)
    
    # Layout (leave room for legends below with minimal spacing)
    # Use smaller bottom margin for tighter layout
    if config.trendline_type:
        # Stacked legends for trendline plots
        plt.tight_layout(rect=[0, 0.07, 1, 1])
    else:
        # Horizontal legends for base plots
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    if save_pdf:
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {pdf_path}")
    
    plt.close()
    
    return regression_stats


def _draw_trendlines(ax, cell_data: pd.DataFrame, config: PlotConfig, 
                     family: str, task: str, regression_stats: list, sig_threshold: float = 0.05):
    """Draw trendlines based on configuration and collect statistics.
    
    Args:
        sig_threshold: Bonferroni-corrected significance threshold for this plot
    
    Returns:
        R² value if this is an overall trend (for annotation), otherwise None
    """
    r_squared_for_annotation = None
    
    if config.trendline_type == 'overall':
        # Single trendline for all data in this subplot
        x_col = 'version_ordinal' if config.x_type == 'version' else 'size_billions'
        x_data = cell_data[x_col].values
        y_data = cell_data['f1_score'].values
        
        if len(x_data) >= 3:
            use_log = config.x_type == 'params'
            stats = _add_linear_trendline(ax, x_data, y_data, use_log=use_log,
                                         sig_threshold=sig_threshold,
                                         draw_ci=True,  # Draw CI band for overall trends
                                         color='#333333', linewidth=2, alpha=0.6)
            if stats:
                stats.update({
                    'plot_type': config.output_suffix,
                    'trendline_type': 'overall',
                    'family': family,
                    'task': task,
                    'group': 'all',
                })
                regression_stats.append(stats)
                r_squared_for_annotation = stats['r_squared']
    
    elif config.trendline_type == 'by_type':
        # One trendline per fine-tune type (if >= 3 points)
        x_col = 'version_ordinal' if config.x_type == 'version' else 'size_billions'
        
        for model_type in cell_data['model_type'].unique():
            type_data = cell_data[cell_data['model_type'] == model_type]
            if len(type_data) >= 3:
                x_data = type_data[x_col].values
                y_data = type_data['f1_score'].values
                color = TYPE_COLORS.get(model_type, '#888888')
                use_log = config.x_type == 'params'
                stats = _add_linear_trendline(ax, x_data, y_data, use_log=use_log,
                                            sig_threshold=sig_threshold,
                                            color=color, linewidth=1.5, alpha=0.7)
                if stats:
                    stats.update({
                        'plot_type': config.output_suffix,
                        'trendline_type': 'by_type',
                        'family': family,
                        'task': task,
                        'group': model_type,
                    })
                    regression_stats.append(stats)
    
    elif config.trendline_type == 'by_version':
        # One trendline per version (only for params plot)
        if config.x_type == 'params':
            for version in sorted(cell_data['version_ordinal'].unique()):
                version_data = cell_data[cell_data['version_ordinal'] == version]
                if len(version_data) >= 3:
                    x_data = version_data['size_billions'].values
                    y_data = version_data['f1_score'].values
                    # Use a color gradient for versions
                    version_colors = {1: '#8B4513', 2: '#CD853F', 3: '#DEB887', 4: '#F5DEB3'}
                    color = version_colors.get(version, '#888888')
                    stats = _add_linear_trendline(ax, x_data, y_data, use_log=True,
                                                sig_threshold=sig_threshold,
                                                color=color, linewidth=1.5, alpha=0.7)
                    if stats:
                        stats.update({
                            'plot_type': config.output_suffix,
                            'trendline_type': 'by_version',
                            'family': family,
                            'task': task,
                            'group': f'v{version}',
                        })
                        regression_stats.append(stats)
    
    return r_squared_for_annotation


def _add_linear_trendline(ax, x_data, y_data, use_log=False, sig_threshold=0.05, 
                          draw_ci=False, **plot_kwargs):
    """Add a linear regression trendline and return statistics.
    
    Args:
        use_log: If True, perform regression on log-transformed x values
                (for log-scale plots to get straight lines)
        sig_threshold: Significance threshold for line style (default 0.05, use Bonferroni-corrected value)
        draw_ci: If True, draw 95% confidence interval band around the line
    
    Returns:
        dict with regression statistics or None if insufficient data
    """
    from scipy import stats as scipy_stats
    
    # Remove NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    if use_log:
        mask = mask & (x_data > 0)  # Can't log transform non-positive values
    
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if len(x_clean) < 2:
        return None
    
    # Transform x if needed (for log-scale plots)
    if use_log:
        x_fit = np.log10(x_clean)
    else:
        x_fit = x_clean
    
    # Check if all x values are identical (can't fit a line)
    if np.allclose(x_fit, x_fit[0]):
        return None
    
    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_fit, y_clean)
    
    # Calculate additional statistics
    r_squared = r_value ** 2
    n = len(x_clean)
    
    # Determine if significant using provided threshold
    is_significant = p_value < sig_threshold
    linestyle = '-' if is_significant else '--'
    
    # Create line points for plotting
    x_min, x_max = x_fit.min(), x_fit.max()
    x_line_fit = np.linspace(x_min, x_max, 100)
    y_line = slope * x_line_fit + intercept
    
    # Calculate 95% confidence interval if requested
    if draw_ci:
        # Calculate residuals and standard error of regression
        y_pred_fit = slope * x_fit + intercept
        residuals = y_clean - y_pred_fit
        se_regression = np.sqrt(np.sum(residuals**2) / (n - 2))
        
        # Calculate standard error of prediction at each point along the line
        x_mean = np.mean(x_fit)
        x_var = np.sum((x_fit - x_mean)**2)
        
        # For each x in the line, calculate prediction SE
        se_pred = se_regression * np.sqrt(1/n + (x_line_fit - x_mean)**2 / x_var)
        
        # Use t-distribution critical value for 95% CI
        from scipy.stats import t as t_dist
        t_critical = t_dist.ppf(0.975, n - 2)  # 95% CI, two-tailed
        
        # Calculate CI bounds
        ci_upper = y_line + t_critical * se_pred
        ci_lower = y_line - t_critical * se_pred
        
        # Transform back to original scale for plotting
        if use_log:
            x_line_plot = 10 ** x_line_fit
        else:
            x_line_plot = x_line_fit
        
        # Draw CI band (very light gray, high transparency)
        ax.fill_between(x_line_plot, ci_lower, ci_upper, 
                       color='#CCCCCC', alpha=0.3, zorder=1, linewidth=0)
    else:
        # Transform back to original scale for plotting
        if use_log:
            x_line_plot = 10 ** x_line_fit
        else:
            x_line_plot = x_line_fit
    
    # Draw line with appropriate style
    ax.plot(x_line_plot, y_line, linestyle=linestyle, zorder=2, **plot_kwargs)
    
    # Return statistics
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
        'n_points': n,
        'significant': is_significant,
        'bonferroni_alpha': sig_threshold,
    }


def _style_axis(ax, config: PlotConfig, fam_idx: int, task_idx: int,
                fam_label: str, task_label: str, n_rows: int, n_cols: int) -> None:
    """Apply styling to an axis based on plot type."""
    
    if config.x_type == 'version':
        ax.set_xlim(0.3, 4.7)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['v1', 'v2', 'v3', '3.1+'])
        x_label = 'Model Version'
    else:
        ax.set_xscale('log')
        ax.set_xlim(0.15, 100)
        ax.set_xticks([0.3, 1, 3, 10, 30, 70])
        ax.set_xticklabels(['0.3', '1', '3', '10', '30', '70'])
        x_label = 'Parameters (B)'
    
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Column titles (top row only)
    if fam_idx == 0:
        ax.set_title(task_label, fontweight='bold', pad=8)
    
    # Row labels (left column only)
    if task_idx == 0:
        ax.set_ylabel(f'{fam_label}\nF1 Score', fontweight='bold')
    
    # X-axis label (bottom row only)
    if fam_idx == n_rows - 1:
        ax.set_xlabel(x_label, fontweight='bold')


def _add_legend(fig, config: PlotConfig) -> None:
    """Add appropriate legend based on plot configuration.
    
    Legends are positioned horizontally below the plot grid with equal spacing:
    - Left: Fine-tune Type
    - Center: Model Version or Model Size
    - Right: Significance (if applicable)
    """
    
    legend_types = ['IT', 'PT', 'Medical', 'Safety', 'Mental Health']
    
    # Calculate positions for equal spacing
    # For 3 legends: divide into 4 equal sections, place legends in sections 1, 2, 3
    if config.trendline_type:
        # Three legends: Fine-tune Type, Version/Size, Significance
        positions = [0.20, 0.50, 0.80]
        # Use vertical stacking (ncol=2) to make legends narrower
        type_ncol = 2
        version_ncol = 2
        sig_ncol = 1
        # Position closer to plot for trendline plots
        y_pos = 0.03
    else:
        # Two legends: Fine-tune Type, Version/Size
        positions = [0.30, 0.70, None]
        # Keep horizontal layout for 2-legend plots
        type_ncol = 5
        version_ncol = 4
        sig_ncol = 2
        # Position for base plots
        y_pos = 0.01
    
    # Type legend (color) - positioned on the left
    type_handles = [
        mlines.Line2D([0], [0], marker='o', color=TYPE_COLORS.get(t, '#888888'),
                      linestyle='None', markersize=8, alpha=0.6,
                      label=TYPE_DISPLAY_LABELS.get(t, t))
        for t in legend_types if t in TYPE_COLORS
    ]
    
    leg1 = fig.legend(handles=type_handles, title='Fine-tune Type',
                      loc='upper center', bbox_to_anchor=(positions[0], y_pos),
                      framealpha=0.9, fontsize=9, ncol=type_ncol)
    fig.add_artist(leg1)
    
    # Version shape+size legend (for params plot with shapes) - positioned in the center
    if config.use_version_shape:
        version_labels = ['v1', 'v2', 'v3', 'v3.1+']
        # Size mapping: v1=60, v2=100, v3=140, 3.1+=180
        size_scale = [60, 100, 140, 180]
        version_handles = [
            mlines.Line2D([0], [0], marker=VERSION_MARKERS[i+1], color='gray',
                          linestyle='None', markersize=np.sqrt(size_scale[i])*0.35, 
                          alpha=0.6, label=version_labels[i])
            for i in range(4)
        ]
        leg_version = fig.legend(handles=version_handles, title='Model Version',
                                loc='upper center', bbox_to_anchor=(positions[1], y_pos),
                                framealpha=0.9, fontsize=9, ncol=version_ncol)
        fig.add_artist(leg_version)
    
    # Size legend (for version plots without trendlines) - positioned in the center
    if config.x_type == 'version' and not config.trendline_type:
        def size_to_marker_size(size_b):
            min_size, max_size = 0.3, 80
            min_marker, max_marker = 40, 350
            log_ratio = (np.log10(size_b) - np.log10(min_size)) / (np.log10(max_size) - np.log10(min_size))
            return min_marker + log_ratio * (max_marker - min_marker)
        
        size_examples = [(1, '~1B'), (10, '~10B'), (70, '~70B')]
        size_handles = [
            mlines.Line2D([0], [0], marker='o', color='gray',
                          linestyle='None', markersize=np.sqrt(size_to_marker_size(s))*0.35, 
                          alpha=0.5, label=label)
            for s, label in size_examples
        ]
        leg_size = fig.legend(handles=size_handles, title='Model Size',
                             loc='upper center', bbox_to_anchor=(positions[1], y_pos),
                             framealpha=0.9, fontsize=9, ncol=3)
        fig.add_artist(leg_size)
    
    # Trendline significance legend (if plot has trendlines) - positioned on the right
    if config.trendline_type:
        sig_handles = [
            mlines.Line2D([0], [0], color='#333333', linestyle='-', linewidth=2,
                          label='p < 0.05 (significant)'),
            mlines.Line2D([0], [0], color='#333333', linestyle='--', linewidth=2,
                          label='p ≥ 0.05 (n.s.)'),
        ]
        leg_sig = fig.legend(handles=sig_handles, title='Significance',
                            loc='upper center', bbox_to_anchor=(positions[2], y_pos),
                            framealpha=0.9, fontsize=9, ncol=sig_ncol)
        fig.add_artist(leg_sig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create compact unified facet plots for all classification tasks'
    )
    parser.add_argument('--unified-csv', type=str, default=None,
                       help='Path to all_models_all_tasks.csv')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory for output files')
    parser.add_argument('--prefix', type=str, default='compact',
                       help='Output file prefix')
    parser.add_argument('--figsize', nargs=2, type=float, default=[14, 10],
                       help='Figure size (width height)')
    parser.add_argument('--save-pdf', action='store_true',
                       help='Also save PDF versions')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if args.unified_csv:
        csv_path = args.unified_csv
    else:
        csv_path = Path(__file__).parent.parent.parent / 'data' / 'inputs' / 'model_results' / 'all_models_all_tasks.csv'
    
    if not Path(csv_path).exists():
        print(f"Data file not found: {csv_path}")
        return
    
    df = load_unified_data(str(csv_path))
    print(f"Loaded {len(df)} rows from: {csv_path}")
    print(f"Tasks: {df['task'].unique().tolist()}")
    print(f"Families: {df['base_family'].value_counts().to_dict()}")
    print(f"Model types: {df['model_type'].value_counts().to_dict()}")
    
    figsize = tuple(args.figsize)
    
    # Define all 7 plot configurations
    plot_configs = [
        # Original 2 plots (no trendlines)
        PlotConfig(
            x_type='version',
            use_version_shape=False,
            trendline_type=None,
            output_suffix='f1_vs_version',
            title_suffix='F1 vs Version'
        ),
        PlotConfig(
            x_type='params',
            use_version_shape=True,
            trendline_type=None,
            output_suffix='f1_vs_params',
            title_suffix='F1 vs Parameters'
        ),
        
        # Request 1: Overall trendlines (1 per subplot)
        PlotConfig(
            x_type='version',
            use_version_shape=False,
            trendline_type='overall',
            output_suffix='f1_vs_version_overall_trend',
            title_suffix='F1 vs Version (Overall Trend)'
        ),
        PlotConfig(
            x_type='params',
            use_version_shape=True,
            trendline_type='overall',
            output_suffix='f1_vs_params_overall_trend',
            title_suffix='F1 vs Parameters (Overall Trend)'
        ),
        
        # Request 2: Per-type trendlines (up to 5 per subplot)
        PlotConfig(
            x_type='version',
            use_version_shape=False,
            trendline_type='by_type',
            output_suffix='f1_vs_version_by_type_trend',
            title_suffix='F1 vs Version (By Type Trends)'
        ),
        PlotConfig(
            x_type='params',
            use_version_shape=True,
            trendline_type='by_type',
            output_suffix='f1_vs_params_by_type_trend',
            title_suffix='F1 vs Parameters (By Type Trends)'
        ),
        
        # Request 3: Per-version trendlines (only for params plot)
        PlotConfig(
            x_type='params',
            use_version_shape=True,
            trendline_type='by_version',
            output_suffix='f1_vs_params_by_version_trend',
            title_suffix='F1 vs Parameters (By Version Trends)'
        ),
    ]
    
    # Generate all plots and collect regression statistics
    print(f"\n=== Generating {len(plot_configs)} plots ===")
    all_regression_stats = []
    
    for config in plot_configs:
        output_path = output_dir / f"{args.prefix}_{config.output_suffix}.png"
        stats = create_facet_plot(df, config, str(output_path), figsize=figsize, save_pdf=args.save_pdf)
        all_regression_stats.extend(stats)
    
    # Save regression statistics to CSV
    if all_regression_stats:
        stats_df = pd.DataFrame(all_regression_stats)
        
        # Reorder columns for readability
        col_order = ['plot_type', 'trendline_type', 'family', 'task', 'group', 
                     'n_points', 'slope', 'intercept', 'r_squared', 'p_value', 
                     'std_err', 'significant', 'bonferroni_alpha']
        stats_df = stats_df[col_order]
        
        stats_path = output_dir / f"{args.prefix}_regression_statistics.csv"
        stats_df.to_csv(stats_path, index=False, float_format='%.6f')
        print(f"\nSaved regression statistics to: {stats_path}")
        
        # Print summary
        n_sig = stats_df['significant'].sum()
        n_total = len(stats_df)
        print(f"  Total regressions: {n_total}")
        print(f"  Significant (Bonferroni per-plot): {n_sig} ({100*n_sig/n_total:.1f}%)")
        
        # Show per-plot breakdown
        print(f"\n  Per-plot breakdown:")
        for plot_type in stats_df['plot_type'].unique():
            plot_stats = stats_df[stats_df['plot_type'] == plot_type]
            n_sig_plot = plot_stats['significant'].sum()
            n_plot = len(plot_stats)
            alpha_plot = plot_stats['bonferroni_alpha'].iloc[0]
            print(f"    {plot_type}: {n_sig_plot}/{n_plot} significant (α={alpha_plot:.6f})")
    
    print(f"\n=== Done! ===")


if __name__ == "__main__":
    main()
