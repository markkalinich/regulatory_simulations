#!/usr/bin/env python3
"""
Model Family Facet Plot - SI Detection Performance
Columns = Model families (Gemma, Qwen, Llama, GPT)
Rows = Metrics (Parse%, Sensitivity, Specificity, Accuracy, F1)
X-axis = Log10 model parameters (billions)
Color = Model family
Shape = Model type (IT, PT, MedGemma, ShieldGemma, Mental Health)
Alpha = Model version (v1=light, v2=medium, v3=dark)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
import argparse


def map_model_size_to_billions(model_size):
    """Convert model size strings to billions for plotting"""
    # Remove suffixes like -it, -pt, _v1, etc.
    size_clean = model_size.lower()
    for suffix in ['-it', '-pt', '_v1', '_v2', '_v3', '_v4', '_v5', '_therapist']:
        size_clean = size_clean.replace(suffix, '')
    
    size_mapping = {
        '270m': 0.27, 
        '0.6b': 0.6, 
        '1b': 1.0, 
        '1.7b': 1.7,
        '2b': 2.0,
        '4b': 4.0, 
        '7b': 7.0,
        '8b': 8.0, 
        '9b': 9.0,
        '12b': 12.0, 
        '14b': 14.0,
        '20b': 20.0,
        '27b': 27.0,
        '32b': 32.0, 
        '70b': 70.0,
        '120b': 120.0
    }
    return size_mapping.get(size_clean, float('nan'))


def determine_model_type(family, size):
    """Determine model type for marker shape"""
    if family == 'medgemma':
        return 'MedGemma'
    elif family == 'shieldgemma':
        return 'ShieldGemma'
    elif family == 'mental_health':
        return 'Mental Health'
    elif '-pt' in size or size.endswith('-pt'):
        return 'PT'
    else:
        return 'IT'


def determine_base_family(family):
    """Map all model families to base family for column assignment"""
    gemma_derived = ['gemma', 'gemma1', 'gemma2', 'gemma3n', 'medgemma', 'shieldgemma', 'mental_health', 'gemma_therapy']
    if family in gemma_derived:
        return 'gemma'
    return family


def determine_version_alpha(family, model_size, model_version):
    """Determine alpha based on model version (only for gemma family)"""
    # For non-gemma families, use full opacity
    base_family = determine_base_family(family)
    if base_family != 'gemma':
        return 0.9
    
    # For gemma-derived models, use version-based alpha
    # v1 = 0.3 (light), v2 = 0.6 (medium), v3 = 0.9 (dark)
    if model_version.startswith('1'):
        return 0.35
    elif model_version.startswith('2'):
        return 0.6
    elif model_version.startswith('3'):
        return 0.9
    else:
        return 0.9


def create_facet_plot(metrics_csv, output_path, figsize=(16, 20)):
    """Create the faceted plot"""
    
    # Load data
    df = pd.read_csv(metrics_csv)
    
    # Add computed columns
    df['size_billions'] = df['model_size'].apply(map_model_size_to_billions)
    df['model_type'] = df.apply(lambda r: determine_model_type(r['model_family'], r['model_size']), axis=1)
    df['base_family'] = df['model_family'].apply(determine_base_family)
    
    # Filter out models with 0% parse rate (failed completely)
    df = df[df['parse_success_rate'] > 0]
    
    # Get Gemma generation based on actual model architecture
    # Gemma 1 = original (2b, 7b sizes), Gemma 2 = second gen, Gemma 3 = third gen
    # This is determined by the actual base model, NOT the version field in cache
    version_map = {
        # Gemma 3 models (270m, 1b, 4b, 12b, 27b sizes - new architecture)
        ('gemma', '270m-it'): '3', ('gemma', '270m-pt'): '3',
        ('gemma', '1b-it'): '3', ('gemma', '1b-pt'): '3',
        ('gemma', '4b-it'): '3', ('gemma', '4b-pt'): '3',
        ('gemma', '12b-it'): '3', ('gemma', '12b-pt'): '3',
        ('gemma', '27b-it'): '3', ('gemma', '27b-pt'): '3',
        # Gemma 2 models (2b, 9b, 27b sizes)
        ('gemma2', '2b-it'): '2', ('gemma2', '2b-pt'): '2',
        ('gemma2', '9b-it'): '2', ('gemma2', '9b-pt'): '2',
        ('gemma2', '27b-it'): '2', ('gemma2', '27b-pt'): '2',
        # MedGemma - based on Gemma 2
        ('medgemma', '4b-it'): '2', ('medgemma', '27b-it'): '2',
        # ShieldGemma - all based on Gemma 2
        ('shieldgemma', '2b'): '2', ('shieldgemma', '4b-it'): '2',
        ('shieldgemma', '9b'): '2', ('shieldgemma', '27b'): '2',
        # Mental health fine-tunes - base model varies:
        # gemma3-* = Gemma 3
        ('mental_health', '270m'): '3', ('mental_health', '1b'): '3',
        # gemma-2-2b-it-therapist = Gemma 2
        ('mental_health', '2b_therapist'): '2',
        # gemma-2b-* = Gemma 1 (original 2B model)
        ('mental_health', '2b_v1'): '1', ('mental_health', '2b_v2'): '1',
        ('mental_health', '2b_v3'): '1', ('mental_health', '2b_v4'): '1',
        ('mental_health', '2b_v5'): '1',
        # Qwen 3
        ('qwen', '0.6b'): '3', ('qwen', '1.7b'): '3', ('qwen', '4b'): '3',
        ('qwen', '8b'): '3', ('qwen', '14b'): '3', ('qwen', '32b'): '3',
        # Llama 3.x
        ('llama', '1b-it'): '3', ('llama', '8b-it'): '3', ('llama', '70b-it'): '3',
        # GPT-OSS 
        ('gpt_oss', '20b'): '1', ('gpt_oss', '120b'): '1',
    }
    
    df['version'] = df.apply(lambda r: version_map.get((r['model_family'], r['model_size']), '3'), axis=1)
    df['alpha'] = df.apply(lambda r: determine_version_alpha(r['model_family'], r['model_size'], r['version']), axis=1)
    
    print(f"Loaded {len(df)} models after filtering")
    print(f"Base families: {df['base_family'].unique()}")
    print(f"Model types: {df['model_type'].unique()}")
    
    # Setup plot
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'font.family': 'sans-serif',
    })
    
    # Define layout
    metrics = ['parse_success_rate', 'sensitivity', 'specificity', 'accuracy', 'f1_score']
    metric_labels = ['Parse Success Rate', 'Sensitivity', 'Specificity', 'Accuracy', 'F1 Score']
    families = ['gemma', 'qwen', 'llama', 'gpt_oss']
    family_labels = ['Gemma Family', 'Qwen', 'Llama', 'GPT-OSS']
    
    # Colors for base families
    family_colors = {
        'gemma': '#E64B35',    # Red
        'qwen': '#4DBBD5',     # Cyan
        'llama': '#00A087',    # Green
        'gpt_oss': '#3C5488'   # Blue
    }
    
    # Markers for model types
    type_markers = {
        'IT': 'o',              # Circle
        'PT': 's',              # Square
        'MedGemma': 'D',        # Diamond
        'ShieldGemma': '^',     # Triangle up
        'Mental Health': 'P'    # Plus (filled)
    }
    
    # Create figure: rows = metrics, cols = families
    fig, axes = plt.subplots(len(metrics), len(families), figsize=figsize, 
                              sharex=True, sharey='row')
    
    for metric_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        for fam_idx, (family, fam_label) in enumerate(zip(families, family_labels)):
            ax = axes[metric_idx, fam_idx]
            
            # Get data for this family column
            fam_data = df[df['base_family'] == family].copy()
            
            if len(fam_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xscale('log')
                continue
            
            # Plot each model type with different markers
            for model_type in fam_data['model_type'].unique():
                type_data = fam_data[fam_data['model_type'] == model_type]
                
                for _, row in type_data.iterrows():
                    ax.scatter(
                        row['size_billions'], 
                        row[metric],
                        c=family_colors[family],
                        marker=type_markers.get(model_type, 'o'),
                        s=120,
                        alpha=row['alpha'],
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=3
                    )
            
            # Connect IT models with lines (main trend)
            it_data = fam_data[fam_data['model_type'] == 'IT'].sort_values('size_billions')
            if len(it_data) > 1:
                ax.plot(it_data['size_billions'], it_data[metric],
                       color=family_colors[family], linewidth=1.5, alpha=0.5, zorder=2)
            
            # Styling
            ax.set_xscale('log')
            ax.set_xlim(0.15, 200)
            ax.set_xticks([0.3, 1, 3, 10, 30, 100])
            ax.set_xticklabels(['0.3', '1', '3', '10', '30', '100'])
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Y-axis: 0 to 1
            if metric_idx == 0:  # Parse rate might exceed 1
                ax.set_ylim(-0.02, 1.05)
            else:
                ax.set_ylim(-0.02, 1.05)
            
            # Column titles (top row only)
            if metric_idx == 0:
                ax.set_title(fam_label, fontweight='bold', pad=10)
            
            # Row labels (left column only)
            if fam_idx == 0:
                ax.set_ylabel(metric_label, fontweight='bold')
            
            # X-axis label (bottom row only)
            if metric_idx == len(metrics) - 1:
                ax.set_xlabel('Parameters (B)', fontweight='bold')
    
    # Create legend
    # Model type legend (shapes)
    type_handles = [
        mlines.Line2D([0], [0], marker=marker, color='gray', linestyle='None',
                      markersize=10, label=model_type, markeredgecolor='black', markeredgewidth=0.5)
        for model_type, marker in type_markers.items()
    ]
    
    # Version legend (alpha) - only for Gemma
    version_handles = [
        mlines.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                      markersize=10, alpha=alpha, label=f'v{ver}', 
                      markeredgecolor='black', markeredgewidth=0.5)
        for ver, alpha in [('1', 0.35), ('2', 0.6), ('3', 0.9)]
    ]
    
    # Add legends to the figure
    fig.legend(handles=type_handles, title='Model Type', 
               loc='upper right', bbox_to_anchor=(0.99, 0.98), framealpha=0.9)
    fig.legend(handles=version_handles, title='Gemma Version',
               loc='upper right', bbox_to_anchor=(0.99, 0.78), framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave room for legends
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create model family facet plot')
    parser.add_argument('--metrics-csv', required=True, help='Path to comprehensive_metrics.csv')
    parser.add_argument('--output', default='model_family_facet_plot.png', help='Output path')
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 20], help='Figure size')
    
    args = parser.parse_args()
    
    create_facet_plot(args.metrics_csv, args.output, tuple(args.figsize))


if __name__ == "__main__":
    main()

