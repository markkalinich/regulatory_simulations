#!/usr/bin/env python3
"""
Generate combined three-panel barplot showing expert review status across all three datasets 
with full data provenance tracking.

Panel 1: SI sentences (10 categories)
Panel 2: Therapy request sentences (12 categories)
Panel 3: Therapy engagement conversations (13 categories)

Key features:
- Normalized y-axes (0-100%) for consistent aspect ratios
- Larger font sizes for readability
- Vertical layout (single column, 3 rows)
- Consistent color scheme across all panels (green=kept, yellow=modified, red=removed)

Data provenance features:
- Tracks all three input datasets with SHA256 hashes
- Automatic source data copying with timestamps
- JSON metadata with analysis parameters for each panel
- Output to: results/data_validation/psychiatrist_review_barplots/YYYYMMDD/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add utilities to path - get workspace root
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root / 'utilities'))
sys.path.insert(0, str(workspace_root / 'config'))
from figure_provenance import FigureProvenanceTracker
from constants import SI_LABELS, THERAPY_REQUEST_LABELS, get_therapy_engagement_label


# ============================================================================
# Helper Functions for Review Status Processing
# ============================================================================

def calculate_review_status(p1_status, p2_status):
    """
    Determine final review status based on P1 and P2 review decisions.
    
    Review workflow:
    - P1 reviews first: KEPT_exact_match, KEPT_with_changes, or REMOVED
    - P2 reviews P1's work: KEPT or REMOVED
    
    Args:
        p1_status: Psychiatrist 1's review decision
        p2_status: Psychiatrist 2's review decision
        
    Returns:
        str: 'removed', 'modified', or 'kept'
    """
    if p1_status == 'REMOVED' or p2_status == 'REMOVED':
        return 'removed'
    elif p1_status == 'KEPT_with_changes':
        return 'modified'
    else:
        return 'kept'


def calculate_status_percentages(df, category_column, categories):
    """
    Calculate kept/modified/removed percentages for each category.
    
    Args:
        df: DataFrame with 'status' column already calculated
        category_column: Name of the column containing categories
        categories: List of category values to process
        
    Returns:
        dict: {category: {'kept': %, 'modified': %, 'removed': %}}
    """
    status_percentages = {}
    for cat in categories:
        cat_df = df[df[category_column] == cat]
        total = len(cat_df)
        status_percentages[cat] = {
            'kept': 100 * len(cat_df[cat_df['status'] == 'kept']) / total if total > 0 else 0,
            'modified': 100 * len(cat_df[cat_df['status'] == 'modified']) / total if total > 0 else 0,
            'removed': 100 * len(cat_df[cat_df['status'] == 'removed']) / total if total > 0 else 0
        }
    return status_percentages


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_si_data():
    """Load and process SI sentence data."""
    data_path = Path(__file__).parent.parent.parent / 'data/inputs/intermediate_files/SI_psychiatrist_01_and_02_scores.csv'
    df = pd.read_csv(data_path)
    
    # Define ordering
    non_si_categories = ['clearly_neutral', 'clearly_happy', 'clearly_sad', 'ambiguous_emotional', 'clearly_angry']
    si_categories = ['passive_si', 'active_si_abstract', 'active_si_plan_no_intent', 
                     'active_si_plan_with_intent', 'preparatory_si']
    
    # Use centralized labels with plot-specific overrides for old taxonomy
    category_labels = {
        **{k: SI_LABELS[k] for k in ['clearly_happy', 'clearly_neutral', 'clearly_sad', 'clearly_angry', 'ambiguous_emotional', 'passive_si']},
        'active_si_abstract': 'Active SI\nNo Plan',
        'active_si_plan_no_intent': 'Active SI\nWith Plan',
        'active_si_plan_with_intent': 'Active SI\nPlan &\nIntent',
        'preparatory_si': 'Active SI\nPlan, Intent\n& Prep'
    }
    
    all_categories = non_si_categories + si_categories
    
    # Calculate review status using shared helper
    df['status'] = df.apply(lambda row: calculate_review_status(row['Psychiatrist_01'], row['Psychiatrist_02']), axis=1)
    
    # Calculate status percentages using shared helper
    status_percentages = calculate_status_percentages(df, 'Safety type', all_categories)
    
    return data_path, df, all_categories, category_labels, status_percentages, len(non_si_categories)


def load_therapy_request_data():
    """Load and process therapy request sentence data."""
    data_path = Path(__file__).parent.parent.parent / 'data/inputs/intermediate_files/therapy_request_psychiatrist_01_and_02_scores.csv'
    df = pd.read_csv(data_path)
    
    # Define ordering
    no_request_categories = [
        'Neutral Declarative Statements',
        'Affect-Containing Declarative Statements - Clearly Happy',
        'Affect-Containing Declarative Statements - Clearly Sad',
        'Affect-Containing Declarative Statements - Clearly Angry',
        'Neutral Non-Therapeutic Questions',
        'Affect-Containing Non-Therapeutic Questions - Clearly Happy',
        'Affect-Containing Non-Therapeutic Questions - Clearly Sad',
        'Affect-Containing Non-Therapeutic Questions - Clearly Angry',
    ]
    
    therapy_request_categories = [
        'Neutral Explicit Requests for Therapy',
        'Affect-Containing Explicit Requests for Therapy - Clearly Happy',
        'Affect-Containing Explicit Requests for Therapy - Clearly Sad',
        'Affect-Containing Explicit Requests for Therapy - Clearly Angry',
    ]
    
    # Use centralized labels
    category_labels = THERAPY_REQUEST_LABELS
    
    all_categories = no_request_categories + therapy_request_categories
    
    # Calculate review status using shared helper
    df['status'] = df.apply(lambda row: calculate_review_status(row['Psychiatrist_01'], row['Psychiatrist_02']), axis=1)
    
    # Calculate status percentages using shared helper
    status_percentages = calculate_status_percentages(df, 'Counseling Request', all_categories)
    
    return data_path, df, all_categories, category_labels, status_percentages, len(no_request_categories)


def load_therapy_engagement_data():
    """Load and process therapy engagement conversation data."""
    data_path = Path(__file__).parent.parent.parent / 'data/inputs/intermediate_files/therapy_engagement_psychiatrist_01_and_02_scores.csv'
    df = pd.read_csv(data_path)
    
    # Get one row per conversation
    conversations_df = df.groupby('Example_ID').agg({
        'SubCategory': 'first',
        'Psychiatrist_01': 'first',
        'Psychiatrist_02': 'first'
    }).reset_index()
    
    # Aggregate subcategories
    def aggregate_subcategory(subcategory):
        if subcategory.startswith('SimulatedTherapy_'):
            parts = subcategory.replace('SimulatedTherapy_', '').split('_')
            if len(parts) >= 2:
                therapy_type = parts[1]
                return f'SimulatedTherapy_{therapy_type}'
        elif subcategory.startswith('Ambiguous_'):
            parts = subcategory.replace('Ambiguous_', '').split('_')
            if len(parts) >= 1:
                return f'Ambiguous_{parts[0]}'
        return subcategory
    
    conversations_df['AggregatedSubCategory'] = conversations_df['SubCategory'].apply(aggregate_subcategory)
    
    # Get categories in desired order
    non_therapeutic_cats = sorted([cat for cat in conversations_df['AggregatedSubCategory'].unique() 
                                   if cat.startswith('NonTherapeutic_')])
    ambiguous_cats = sorted([cat for cat in conversations_df['AggregatedSubCategory'].unique() 
                            if cat.startswith('Ambiguous_')])
    
    simulated_therapy_order = [
        'SimulatedTherapy_CognitiveTechniqueConcept',
        'SimulatedTherapy_SkillConcept',
        'SimulatedTherapy_PsychoanalyticConcept',
        'SimulatedTherapy_DiagnosisSuggestion',
        'SimulatedTherapy_MedicationMention'
    ]
    simulated_therapy_cats = [cat for cat in simulated_therapy_order 
                             if cat in conversations_df['AggregatedSubCategory'].unique()]
    
    # Use centralized label function
    all_categories = non_therapeutic_cats + ambiguous_cats + simulated_therapy_cats
    category_labels = {cat: get_therapy_engagement_label(cat) for cat in all_categories}
    
    # Calculate review status using shared helper
    conversations_df['status'] = conversations_df.apply(
        lambda row: calculate_review_status(row['Psychiatrist_01'], row['Psychiatrist_02']), axis=1
    )
    
    # Calculate status percentages using shared helper
    status_percentages = calculate_status_percentages(conversations_df, 'AggregatedSubCategory', all_categories)
    
    split_indices = [len(non_therapeutic_cats), len(non_therapeutic_cats) + len(ambiguous_cats)]
    return data_path, conversations_df, all_categories, category_labels, status_percentages, split_indices


def plot_panel(ax, categories, category_labels, status_percentages, panel_label, 
               group_info, ylabel='% Items'):
    """Plot a single panel with normalized percentages."""
    x = np.arange(len(categories))
    width = 0.8
    
    # Get percentage data
    kept_pcts = [status_percentages[cat]['kept'] for cat in categories]
    modified_pcts = [status_percentages[cat]['modified'] for cat in categories]
    removed_pcts = [status_percentages[cat]['removed'] for cat in categories]
    
    # Plot stacked bars
    ax.bar(x, kept_pcts, width, label='Kept', color='#2ecc71')
    ax.bar(x, modified_pcts, width, bottom=kept_pcts, label='Modified', color='#f1c40f')
    ax.bar(x, removed_pcts, width, 
           bottom=np.array(kept_pcts) + np.array(modified_pcts),
           label='Removed', color='#e74c3c')
    
    # Add category labels - larger font with 90 degree rotation
    ax.set_xticks(x)
    ax.set_xticklabels([category_labels[cat] for cat in categories], 
                       fontsize=11, rotation=90, ha='center')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    
    # Add panel label
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top')
    
    # Add group demarcations and labels
    if 'divider_positions' in group_info:
        for pos in group_info['divider_positions']:
            ax.axvline(x=pos, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    if 'group_labels' in group_info:
        for label_info in group_info['group_labels']:
            ax.text(label_info['x'], 108, label_info['text'], 
                   ha='center', fontsize=13, fontweight='bold', style='italic')
            
            # Add horizontal bar
            ax.plot(label_info['bar_range'], [105, 105], 'k-', linewidth=3, clip_on=False)
    
    # Customize
    ax.set_ylim(0, 112)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def main():
    """Generate combined three-panel figure with provenance tracking."""
    
    # Initialize provenance tracker
    base_dir = Path(__file__).parent.parent.parent / 'results/data_validation/psychiatrist_review_barplots'
    tracker = FigureProvenanceTracker(
        figure_name='combined_three_panel_review',
        base_dir=base_dir
    )
    
    # Load all datasets
    si_path, si_df, si_cats, si_labels, si_pcts, si_split = load_si_data()
    tr_path, tr_df, tr_cats, tr_labels, tr_pcts, tr_split = load_therapy_request_data()
    te_path, te_df, te_cats, te_labels, te_pcts, te_splits = load_therapy_engagement_data()
    
    # Track all input datasets
    tracker.add_input_dataset(
        file_path=si_path,
        description='SI statements with psychiatrist 1 and 2 review decisions (Panel A)',
        columns_used=['Safety type', 'Psychiatrist_01', 'Psychiatrist_02'],
        copy_to_source=True
    )
    
    tracker.add_input_dataset(
        file_path=tr_path,
        description='Therapy request statements with psychiatrist 1 and 2 review decisions (Panel B)',
        columns_used=['Counseling Request', 'Psychiatrist_01', 'Psychiatrist_02'],
        copy_to_source=True
    )
    
    tracker.add_input_dataset(
        file_path=te_path,
        description='Therapy engagement conversations with psychiatrist 1 and 2 review decisions (Panel C)',
        columns_used=['Example_ID', 'SubCategory', 'Psychiatrist_01', 'Psychiatrist_02'],
        copy_to_source=True
    )
    
    # Record analysis parameters
    tracker.set_analysis_parameters(
        plot_type='three_panel_stacked_bar',
        y_axis_format='normalized_percentage',
        figure_size_inches='10x14',
        panel_a_task='suicide_ideation',
        panel_a_total_statements=len(si_df),
        panel_a_categories=len(si_cats),
        panel_b_task='therapy_request',
        panel_b_total_statements=len(tr_df),
        panel_b_categories=len(tr_cats),
        panel_c_task='therapy_engagement',
        panel_c_total_conversations=len(te_df),
        panel_c_categories=len(te_cats),
        status_categories=['kept', 'modified', 'removed'],
        color_scheme={'kept': '#2ecc71', 'modified': '#f1c40f', 'removed': '#e74c3c'}
    )
    
    # Create figure with 3 vertically stacked panels - reduced size
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))
    
    # Panel 1: SI sentences
    si_group_info = {
        'divider_positions': [si_split - 0.5],
        'group_labels': [
            {
                'x': si_split / 2 - 0.5,
                'text': 'Non-SI Statements',
                'bar_range': [-0.5, si_split - 0.6]
            },
            {
                'x': si_split + (len(si_cats) - si_split) / 2 - 0.5,
                'text': 'SI Statements',
                'bar_range': [si_split - 0.4, len(si_cats) - 0.5]
            }
        ]
    }
    plot_panel(ax1, si_cats, si_labels, si_pcts, 'A', si_group_info)
    
    # Panel 2: Therapy request sentences
    tr_group_info = {
        'divider_positions': [tr_split - 0.5],
        'group_labels': [
            {
                'x': tr_split / 2 - 0.5,
                'text': 'No Request',
                'bar_range': [-0.5, tr_split - 0.6]
            },
            {
                'x': tr_split + (len(tr_cats) - tr_split) / 2 - 0.5,
                'text': 'Therapy Request',
                'bar_range': [tr_split - 0.4, len(tr_cats) - 0.5]
            }
        ]
    }
    plot_panel(ax2, tr_cats, tr_labels, tr_pcts, 'B', tr_group_info)
    
    # Panel 3: Therapy engagement conversations
    te_group_info = {
        'divider_positions': [te_splits[1] - 0.5],  # Only line between Ambiguous and Therapeutic
        'group_labels': [
            {
                'x': te_splits[0] / 2 - 0.5,
                'text': 'Non-Therapeutic',
                'bar_range': [-0.5, te_splits[0] - 0.1]
            },
            {
                'x': (te_splits[0] + te_splits[1]) / 2 - 0.5,
                'text': 'Ambiguous',
                'bar_range': [te_splits[0] + 0.1, te_splits[1] - 0.6]
            },
            {
                'x': te_splits[1] + (len(te_cats) - te_splits[1]) / 2 - 0.5,
                'text': 'Therapeutic Conversation',
                'bar_range': [te_splits[1] - 0.4, len(te_cats) - 0.5]
            }
        ]
    }
    plot_panel(ax3, te_cats, te_labels, te_pcts, 'C', te_group_info, 
              ylabel='% Items')
    
    # Add single legend for all panels (bottom left of first panel)
    ax1.legend(loc='lower left', fontsize=13, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the figure with provenance tracking
    output_filename = tracker.get_output_path('combined_three_panel_review.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    # Track output file
    tracker.add_output_file(
        file_path=output_filename,
        file_type='figure',
        dpi=300,
        figure_size_inches='10x14',
        format='png',
        panels=3
    )
    
    # Save provenance metadata
    provenance_file = tracker.save_provenance()
    
    print(f"\n{'='*70}")
    print("COMBINED THREE-PANEL REVIEW BARPLOT - GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nFigure saved to: {output_filename}")
    print(f"Provenance metadata: {provenance_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("THREE-PANEL FIGURE SUMMARY")
    print("="*60)
    print(f"Panel A: SI Sentences - {len(si_cats)} categories")
    print(f"Panel B: Therapy Request Sentences - {len(tr_cats)} categories")
    print(f"Panel C: Therapy Engagement Conversations - {len(te_cats)} categories")
    print(f"\nAll y-axes normalized to 0-100% for consistent aspect ratios")
    print(f"Figure size: 10 x 14 inches")
    print(f"Resolution: 300 dpi")
    
    plt.close()
    
    return output_filename, provenance_file


if __name__ == '__main__':
    main()
