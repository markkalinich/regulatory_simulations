#!/usr/bin/env python3
"""
Combined All-Families F1 Facet Plot - SI Detection Performance

Rows = Model families (Gemma, Llama, Qwen)
Columns = Versions (1, 2, 3, 3.5) with family-specific labels
    - Gemma: Gemma 1, Gemma 2, Gemma 3, Gemma 3n
    - Llama: Llama 1, Llama 2, Llama 3.0, Llama 3.1+
    - Qwen: Qwen 1.x, Qwen 2, Qwen 3, (empty)
X-axis = Log10 model parameters (billions)
Y-axis = F1 Score only

Uses EXACT same grouping logic as individual family plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import argparse

from facet_plot_utils import (
    get_model_metadata,
    get_param_billions_from_config,
    compute_shieldgemma_metrics,
    compute_llama_guard_metrics,
    compute_qwen_guard_metrics,
    apply_guard_metrics_to_df,
    MODEL_TYPE_COLORS,
    MODEL_TYPE_MARKERS,
)

# Import version/type functions from family-specific modules
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
# Combined Plot Configuration
# =============================================================================

FAMILIES = ['gemma', 'llama', 'qwen']
FAMILY_LABELS = ['Gemma', 'Llama', 'Qwen']

# Numeric versions for columns (internal use)
VERSIONS = [1, 2, 3, 3.5]

# Family-specific column labels
VERSION_LABELS_BY_FAMILY = {
    'gemma': ['Gemma 1', 'Gemma 2', 'Gemma 3', 'Gemma 3n'],
    'llama': ['Llama 1', 'Llama 2', 'Llama 3.0', 'Llama 3.1+'],
    'qwen': ['Qwen 1.x', 'Qwen 2', 'Qwen 3', ''],  # No Qwen 3.5
}

# Combined type colors (union of all family types)
TYPE_COLORS = {
    'IT': '#2E86AB',
    'PT': '#52B788',
    'MedGemma': '#F18F01',
    'Medical': '#F18F01',
    'ShieldGemma': '#7B2D8E',
    'Safety': '#7B2D8E',
    'Guard': '#7B2D8E',
    'Mental Health': '#C73E1D',
}

TYPE_MARKERS = {
    'IT': 'o',
    'PT': 's',
    'MedGemma': 'D',
    'Medical': 'h',
    'ShieldGemma': '^',
    'Safety': '^',
    'Guard': '^',
    'Mental Health': 'P',
}

TYPE_DISPLAY_LABELS = {
    'IT': 'Instruct Tune',
    'PT': 'Base Model',
    'MedGemma': 'MedGemma',
    'Medical': 'Medical',
    'Safety': 'Safety*',
    'Mental Health': 'Mental Health',
}


def determine_model_family_and_version(family: str, size: str):
    """
    Determine base family and version using EXACT logic from individual plots.
    
    Returns:
        (base_family, version_number) tuple
    """
    family_lower = family.lower()
    
    # Gemma family - use exact family list matching (not substring)
    if family in GEMMA_FAMILIES or family_lower.startswith('gemma'):
        return ('gemma', determine_gemma_version(family, size))
    
    # Llama family
    elif 'llama' in family_lower:
        return ('llama', determine_llama_version(family, size))
    
    # Qwen family
    elif 'qwen' in family_lower:
        return ('qwen', determine_qwen_version(family, size))
    
    return ('unknown', 0)


def determine_model_type(family: str, size: str) -> str:
    """Determine model type using family-specific logic."""
    family_lower = family.lower()
    
    if family in GEMMA_FAMILIES or family_lower.startswith('gemma'):
        model_type = determine_gemma_model_type(family, size)
        # Rename ShieldGemma to Safety for consistency
        if model_type == 'ShieldGemma':
            return 'Safety'
        return model_type
    elif 'llama' in family_lower:
        model_type = determine_llama_model_type(family, size)
        if model_type == 'Guard':
            return 'Safety'
        return model_type
    elif 'qwen' in family_lower:
        model_type = determine_qwen_model_type(family, size)
        if model_type == 'Guard':
            return 'Safety'
        return model_type
    
    return 'IT'


def compute_all_safety_model_metrics():
    """Compute metrics for all safety models (ShieldGemma, Llama Guard, Qwen Guard)."""
    all_metrics = {}
    
    shield_metrics = compute_shieldgemma_metrics()
    all_metrics.update(shield_metrics)
    
    llama_metrics = compute_llama_guard_metrics()
    all_metrics.update(llama_metrics)
    
    qwen_metrics = compute_qwen_guard_metrics()
    all_metrics.update(qwen_metrics)
    
    return all_metrics


def create_all_families_f1_plot(metrics_csv: str, output_path: str, 
                                 figsize=(16, 10), title=None) -> None:
    """Create combined F1 facet plot for all model families.
    
    Args:
        metrics_csv: Path to comprehensive metrics CSV
        output_path: Path to save the plot
        figsize: Figure size tuple (width, height)
        title: Optional overall title for the plot
    """
    df = pd.read_csv(metrics_csv)
    
    # Apply safety model corrections
    safety_metrics = compute_all_safety_model_metrics()
    df = apply_guard_metrics_to_df(df, safety_metrics)
    if safety_metrics:
        print(f"Applied safety model corrections for {len(safety_metrics)} models")
    
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
    
    print(f"Loaded {len(df)} models total")
    print(f"Base families: {df['base_family'].value_counts().to_dict()}")
    print(f"Versions: {sorted(df['version'].unique().tolist())}")
    print(f"Model types: {df['model_type'].value_counts().to_dict()}")
    
    # Setup plot styling
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',
    })
    
    # Create figure: rows = families, cols = versions
    fig, axes = plt.subplots(len(FAMILIES), len(VERSIONS), figsize=figsize, 
                              sharex=True, sharey=True)
    
    for fam_idx, (family, fam_label) in enumerate(zip(FAMILIES, FAMILY_LABELS)):
        for ver_idx, version in enumerate(VERSIONS):
            ax = axes[fam_idx, ver_idx]
            
            # Get data for this family and version
            ver_data = df[(df['base_family'] == family) & (df['version'] == version)].copy()
            
            if len(ver_data) == 0:
                # Empty cell styling
                ver_label = VERSION_LABELS_BY_FAMILY[family][ver_idx]
                if ver_label:  # Don't show "No data" for intentionally empty columns
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_xscale('log')
                ax.set_xlim(0.15, 100)
                ax.set_ylim(-0.02, 1.05)
                ax.set_title(ver_label, fontweight='bold', pad=10)
                if ver_idx == 0:
                    ax.set_ylabel(f'{fam_label}\nF1 Score', fontweight='bold')
                if fam_idx == len(FAMILIES) - 1:
                    ax.set_xlabel('Parameters (B)', fontweight='bold')
                ax.grid(True, alpha=0.3, linewidth=0.5)
                continue
            
            # Plot each model type
            for model_type in ver_data['model_type'].unique():
                type_data = ver_data[ver_data['model_type'] == model_type]
                
                alpha = 0.5 if model_type in ['IT', 'PT'] else 0.85
                
                for _, row in type_data.iterrows():
                    ax.scatter(
                        row['size_billions'], 
                        row['f1_score'],
                        c=TYPE_COLORS.get(model_type, '#888888'),
                        marker=TYPE_MARKERS.get(model_type, 'o'),
                        s=120,
                        alpha=alpha,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=3
                    )
            
            # Connect IT models with lines
            it_data = ver_data[ver_data['model_type'] == 'IT'].sort_values('size_billions')
            if len(it_data) > 1:
                ax.plot(it_data['size_billions'], it_data['f1_score'],
                       color=TYPE_COLORS['IT'], linewidth=1.5, alpha=0.5, zorder=2)
            
            # Connect PT models with lines
            pt_data = ver_data[ver_data['model_type'] == 'PT'].sort_values('size_billions')
            if len(pt_data) > 1:
                ax.plot(pt_data['size_billions'], pt_data['f1_score'],
                       color=TYPE_COLORS['PT'], linewidth=1.5, alpha=0.5, zorder=2)
            
            # Styling
            ax.set_xscale('log')
            ax.set_xlim(0.15, 100)
            ax.set_xticks([0.3, 1, 3, 10, 30])
            ax.set_xticklabels(['0.3', '1', '3', '10', '30'])
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_ylim(-0.02, 1.05)
            
            # Column titles (all rows) - use family-specific labels
            ver_label = VERSION_LABELS_BY_FAMILY[family][ver_idx]
            ax.set_title(ver_label, fontweight='bold', pad=10)
            
            # Row labels (left column only)
            if ver_idx == 0:
                ax.set_ylabel(f'{fam_label}\nF1 Score', fontweight='bold')
            
            # X-axis label (bottom row only)
            if fam_idx == len(FAMILIES) - 1:
                ax.set_xlabel('Parameters (B)', fontweight='bold')
    
    # Create legend
    type_handles = [
        mlines.Line2D([0], [0], marker=TYPE_MARKERS.get(model_type, 'o'), 
                      color=color, linestyle='None', markersize=10, 
                      label=TYPE_DISPLAY_LABELS.get(model_type, model_type), 
                      markeredgecolor='black', markeredgewidth=0.5)
        for model_type, color in TYPE_COLORS.items()
        if model_type in TYPE_DISPLAY_LABELS  # Only include types with display labels
    ]
    
    fig.legend(handles=type_handles, title='Model Type', 
               loc='upper right', bbox_to_anchor=(0.99, 0.98), framealpha=0.9)
    
    # Add title and adjust layout
    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 0.88, 0.99])
    else:
        plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to: {pdf_path}")
    
    plt.close()
    
    # Print summary
    print("\n=== Models per Family + Version ===")
    for family, fam_label in zip(FAMILIES, FAMILY_LABELS):
        print(f"\n{fam_label.upper()}:")
        total = 0
        for version, ver_label in zip(VERSIONS, VERSION_LABELS_BY_FAMILY[family]):
            if not ver_label:  # Skip empty column labels
                continue
            ver_data = df[(df['base_family'] == family) & (df['version'] == version)]
            count = len(ver_data)
            total += count
            print(f"  {ver_label}: {count} models")
        print(f"  {fam_label.upper()} Total: {total} models")
    
    print(f"\n=== TOTAL MODELS PLOTTED: {len(df)} ===")


def main():
    parser = argparse.ArgumentParser(description='Create combined all-families F1 facet plot')
    parser.add_argument('--metrics-csv', required=True, help='Path to comprehensive_metrics.csv')
    parser.add_argument('--output', default='all_families_f1_facet_plot.png', help='Output path')
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 10], help='Figure size')
    parser.add_argument('--title', type=str, default=None, help='Overall plot title')
    
    args = parser.parse_args()
    create_all_families_f1_plot(args.metrics_csv, args.output, tuple(args.figsize), title=args.title)


if __name__ == "__main__":
    main()
