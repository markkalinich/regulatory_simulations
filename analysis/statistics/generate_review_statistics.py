#!/usr/bin/env python3
"""
Generate Review Statistics for All Three Datasets

This script generates comprehensive review statistics for:
1. SI statements
2. Therapy request statements
3. Therapy engagement conversations

For each dataset, it produces a CSV table with:
- Total number of initial sentences/conversations
- Number approved (no changes), approved (with modifications), and removed
- Statistics broken down by sub-categories (matching barplot categories)

Output files:
- analysis/review_statistics/si_review_statistics.csv
- analysis/review_statistics/therapy_request_review_statistics.csv
- analysis/review_statistics/therapy_engagement_review_statistics.csv
"""

import pandas as pd
from pathlib import Path
import sys


def generate_si_statistics():
    """Generate review statistics for SI statements."""
    print("\n" + "="*80)
    print("GENERATING SI REVIEW STATISTICS")
    print("="*80)
    
    # Load the combined psychiatrist scores
    data_file = Path('data/inputs/intermediate_files/SI_psychiatrist_01_and_02_scores.csv')
    df = pd.read_csv(data_file)
    
    print(f"Loaded {len(df)} statements from {data_file}")
    
    # Define category groupings and display names (matching barplot)
    non_si_categories = {
        'clearly_neutral': 'Non-SI: Neutral',
        'clearly_happy': 'Non-SI: Happy',
        'clearly_sad': 'Non-SI: Sad',
        'ambiguous_emotional': 'Non-SI: Ambiguous Emotional',
        'clearly_angry': 'Non-SI: Angry'
    }
    
    si_categories = {
        'passive_si': 'SI: Passive',
        'active_si_abstract': 'SI: Active (no plan)',
        'active_si_plan_no_intent': 'SI: Active (plan, no intent)',
        'active_si_plan_with_intent': 'SI: Active (plan + intent)',
        'preparatory_si': 'SI: Preparatory'
    }
    
    # Determine status for each statement
    # Approved (No Changes): KEPT_exact_match by P1 AND KEPT by P2
    # Approved (Modified): KEPT_with_changes by P1 AND KEPT by P2
    # Not Approved (Removed): Everything else
    def get_status(row):
        p1 = row['Psychiatrist_01']
        p2 = row['Psychiatrist_02']
        
        if p1 == 'KEPT_exact_match' and p2 == 'KEPT':
            return 'approved_no_changes'
        elif p1 == 'KEPT_with_changes' and p2 == 'KEPT':
            return 'approved_modified'
        else:
            return 'removed'
    
    df['status'] = df.apply(get_status, axis=1)
    
    # Calculate statistics
    results = []
    
    # Process Non-SI categories
    for internal_name, display_name in non_si_categories.items():
        cat_df = df[df['Safety type'] == internal_name]
        results.append({
            'Category': display_name,
            'Generated': len(cat_df),
            'Approved (No Changes)': len(cat_df[cat_df['status'] == 'approved_no_changes']),
            'Approved (Modified)': len(cat_df[cat_df['status'] == 'approved_modified']),
            'Not Approved (Removed)': len(cat_df[cat_df['status'] == 'removed'])
        })
    
    # Non-SI subtotal
    non_si_df = df[df['Safety type'].isin(non_si_categories.keys())]
    results.append({
        'Category': 'SUBTOTAL: Non-SI',
        'Generated': len(non_si_df),
        'Approved (No Changes)': len(non_si_df[non_si_df['status'] == 'approved_no_changes']),
        'Approved (Modified)': len(non_si_df[non_si_df['status'] == 'approved_modified']),
        'Not Approved (Removed)': len(non_si_df[non_si_df['status'] == 'removed'])
    })
    
    # Process SI categories
    for internal_name, display_name in si_categories.items():
        cat_df = df[df['Safety type'] == internal_name]
        results.append({
            'Category': display_name,
            'Generated': len(cat_df),
            'Approved (No Changes)': len(cat_df[cat_df['status'] == 'approved_no_changes']),
            'Approved (Modified)': len(cat_df[cat_df['status'] == 'approved_modified']),
            'Not Approved (Removed)': len(cat_df[cat_df['status'] == 'removed'])
        })
    
    # SI subtotal
    si_df = df[df['Safety type'].isin(si_categories.keys())]
    results.append({
        'Category': 'SUBTOTAL: SI',
        'Generated': len(si_df),
        'Approved (No Changes)': len(si_df[si_df['status'] == 'approved_no_changes']),
        'Approved (Modified)': len(si_df[si_df['status'] == 'approved_modified']),
        'Not Approved (Removed)': len(si_df[si_df['status'] == 'removed'])
    })
    
    # Total
    results.append({
        'Category': 'TOTAL',
        'Generated': len(df),
        'Approved (No Changes)': len(df[df['status'] == 'approved_no_changes']),
        'Approved (Modified)': len(df[df['status'] == 'approved_modified']),
        'Not Approved (Removed)': len(df[df['status'] == 'removed'])
    })
    
    return pd.DataFrame(results)


def generate_therapy_request_statistics():
    """Generate review statistics for therapy request statements."""
    print("\n" + "="*80)
    print("GENERATING THERAPY REQUEST REVIEW STATISTICS")
    print("="*80)
    
    # Load the psychiatrist scores
    data_file = Path('data/inputs/intermediate_files/therapy_request_psychiatrist_01_and_02_scores.csv')
    df = pd.read_csv(data_file)
    
    print(f"Loaded {len(df)} statements from {data_file}")
    
    # Define category groupings (matching barplot)
    declarative_categories = {
        'Neutral Declarative Statements': 'Declarative: Neutral Declarative Statements',
        'Affect-Containing Declarative Statements - Clearly Happy': 'Declarative: Clearly Happy',
        'Affect-Containing Declarative Statements - Clearly Sad': 'Declarative: Clearly Sad',
        'Affect-Containing Declarative Statements - Clearly Angry': 'Declarative: Clearly Angry'
    }
    
    non_therapeutic_categories = {
        'Neutral Non-Therapeutic Questions': 'Non-Therapeutic: Neutral Non-Therapeutic Questions',
        'Affect-Containing Non-Therapeutic Questions - Clearly Happy': 'Non-Therapeutic: Clearly Happy',
        'Affect-Containing Non-Therapeutic Questions - Clearly Sad': 'Non-Therapeutic: Clearly Sad',
        'Affect-Containing Non-Therapeutic Questions - Clearly Angry': 'Non-Therapeutic: Clearly Angry'
    }
    
    explicit_therapy_categories = {
        'Neutral Explicit Requests for Therapy': 'Explicit Therapy: Neutral Explicit Requests for Therapy',
        'Affect-Containing Explicit Requests for Therapy - Clearly Happy': 'Explicit Therapy: Clearly Happy',
        'Affect-Containing Explicit Requests for Therapy - Clearly Sad': 'Explicit Therapy: Clearly Sad',
        'Affect-Containing Explicit Requests for Therapy - Clearly Angry': 'Explicit Therapy: Clearly Angry'
    }
    
    # Determine status for each statement (P1+P2 combined)
    def get_status(row):
        p1 = row['Psychiatrist_01']
        p2 = row['Psychiatrist_02']
        
        # If either reviewer removed it, count as removed
        if p1 == 'REMOVED' or p2 == 'REMOVED':
            return 'removed'
        # If P1 made changes, count as modified (even if P2 approved)
        elif p1 == 'KEPT_with_changes':
            return 'approved_modified'
        # Both approved without changes
        else:  # p1 == 'KEPT_exact_match' and p2 == 'KEPT'
            return 'approved_no_changes'
    
    df['status'] = df.apply(get_status, axis=1)
    
    # Calculate statistics
    results = []
    
    # Helper function to process category group
    def process_category_group(category_map, group_name):
        group_results = []
        
        for internal_name, display_name in category_map.items():
            cat_df = df[df['Counseling Request'] == internal_name]
            if len(cat_df) > 0:
                group_results.append({
                    'Category': display_name,
                    'Generated': len(cat_df),
                    'Approved (No Changes)': len(cat_df[cat_df['status'] == 'approved_no_changes']),
                    'Approved (Modified)': len(cat_df[cat_df['status'] == 'approved_modified']),
                    'Not Approved (Removed)': len(cat_df[cat_df['status'] == 'removed'])
                })
        
        # Subtotal
        group_df = df[df['Counseling Request'].isin(category_map.keys())]
        if len(group_df) > 0:
            group_results.append({
                'Category': f'SUBTOTAL: {group_name}',
                'Generated': len(group_df),
                'Approved (No Changes)': len(group_df[group_df['status'] == 'approved_no_changes']),
                'Approved (Modified)': len(group_df[group_df['status'] == 'approved_modified']),
                'Not Approved (Removed)': len(group_df[group_df['status'] == 'removed'])
            })
        
        return group_results
    
    # Process each group
    results.extend(process_category_group(declarative_categories, 'Declarative Statements'))
    results.extend(process_category_group(non_therapeutic_categories, 'Non-Therapeutic Questions'))
    results.extend(process_category_group(explicit_therapy_categories, 'Explicit Therapy Requests'))
    
    # Total
    results.append({
        'Category': 'TOTAL',
        'Generated': len(df),
        'Approved (No Changes)': len(df[df['status'] == 'approved_no_changes']),
        'Approved (Modified)': len(df[df['status'] == 'approved_modified']),
        'Not Approved (Removed)': len(df[df['status'] == 'removed'])
    })
    
    return pd.DataFrame(results)


def generate_therapy_engagement_statistics():
    """Generate review statistics for therapy engagement conversations."""
    print("\n" + "="*80)
    print("GENERATING THERAPY ENGAGEMENT REVIEW STATISTICS")
    print("="*80)
    
    # Load the P1+P2 combined psychiatrist scores
    data_file = Path('data/inputs/intermediate_files/therapy_engagement_psychiatrist_01_and_02_scores.csv')
    df = pd.read_csv(data_file)
    
    print(f"Loaded {len(df)} rows from {data_file}")
    
    # Group by Example_ID to get conversation-level statistics
    conversations = df.groupby('Example_ID').first().reset_index()
    print(f"Total conversations: {len(conversations)}")
    
    # Define category groupings (matching barplot/sankey)
    # These categories are aggregated from the detailed subcategories
    non_therapeutic_categories = [
        'NonTherapeutic_CreativeWriting',
        'NonTherapeutic_InfoSeeking',
        'NonTherapeutic_PlanningOrg',
        'NonTherapeutic_PracticalTask',
        'NonTherapeutic_TechnicalCoding'
    ]
    
    # Ambiguous: 3 aggregated groups (combined across disorders)
    ambiguous_categories = [
        'Ambiguous_DisclosureBoundary',  # Detected Disclosure
        'Ambiguous_InfoPathology',        # Info: Pathology
        'Ambiguous_InfoTherapy'           # Info: Therapy
    ]
    
    # Therapeutic Conversation: 5 therapy techniques (combined across disorders)
    simulated_therapy_categories = [
        'SimulatedTherapy_CognitiveTechniqueConcept',  # Cognitive Technique
        'SimulatedTherapy_SkillConcept',               # CBT/DBT Skill
        'SimulatedTherapy_PsychoanalyticConcept',      # Psychodynamic
        'SimulatedTherapy_DiagnosisSuggestion',        # Diagnosis
        'SimulatedTherapy_MedicationMention'           # Med Recommendation
    ]
    
    # Aggregate subcategories to match barplot groupings
    def aggregate_subcategory(subcategory):
        """
        Aggregate detailed subcategories to match barplot visualization.
        - SimulatedTherapy: Combine across disorders, keep therapy type
        - Ambiguous: Combine across disorders/concepts, keep major type
        - NonTherapeutic: Keep as-is
        """
        if subcategory.startswith('SimulatedTherapy_'):
            parts = subcategory.replace('SimulatedTherapy_', '').split('_')
            if len(parts) >= 2:
                # parts[0] is disorder (Anxiety, Depression, etc.)
                # parts[1] is therapy type (SkillConcept, MedicationMention, etc.)
                therapy_type = parts[1]
                return f'SimulatedTherapy_{therapy_type}'
        elif subcategory.startswith('Ambiguous_'):
            parts = subcategory.replace('Ambiguous_', '').split('_')
            if len(parts) >= 1:
                # parts[0] is major type (DisclosureBoundary, InfoPathology, InfoTherapy)
                return f'Ambiguous_{parts[0]}'
        # NonTherapeutic stays as-is
        return subcategory
    
    conversations['AggregatedSubCategory'] = conversations['SubCategory'].apply(aggregate_subcategory)
    
    # Determine status for each conversation (P1+P2 combined)
    def get_status(row):
        p1 = row['Psychiatrist_01']
        p2 = row['Psychiatrist_02']
        
        # If either reviewer removed it, count as removed
        if p1 == 'REMOVED' or p2 == 'REMOVED':
            return 'removed'
        # If P1 made changes, count as modified (even if P2 approved)
        elif p1 == 'KEPT_with_changes':
            return 'approved_modified'
        # Both approved without changes
        else:  # p1 == 'KEPT_exact_match' and p2 == 'KEPT'
            return 'approved_no_changes'
    
    conversations['status'] = conversations.apply(get_status, axis=1)
    
    # Calculate statistics
    results = []
    
    # Non-Therapeutic conversations (keep individual categories)
    for category in non_therapeutic_categories:
        cat_df = conversations[conversations['AggregatedSubCategory'] == category]
        if len(cat_df) > 0:
            display_name = f"Non-Therapeutic: {category.replace('NonTherapeutic_', '')}"
            results.append({
                'Category': display_name,
                'Generated': len(cat_df),
                'Approved (No Changes)': len(cat_df[cat_df['status'] == 'approved_no_changes']),
                'Approved (Modified)': len(cat_df[cat_df['status'] == 'approved_modified']),
                'Not Approved (Removed)': len(cat_df[cat_df['status'] == 'removed'])
            })
    
    # Non-Therapeutic subtotal
    nt_df = conversations[conversations['AggregatedSubCategory'].isin(non_therapeutic_categories)]
    results.append({
        'Category': 'SUBTOTAL: Non-Therapeutic Conversations',
        'Generated': len(nt_df),
        'Approved (No Changes)': len(nt_df[nt_df['status'] == 'approved_no_changes']),
        'Approved (Modified)': len(nt_df[nt_df['status'] == 'approved_modified']),
        'Not Approved (Removed)': len(nt_df[nt_df['status'] == 'removed'])
    })
    
    # Ambiguous conversations (aggregated groups)
    display_names = {
        'Ambiguous_DisclosureBoundary': 'Ambiguous: Detected Disclosure',
        'Ambiguous_InfoPathology': 'Ambiguous: Info - Pathology',
        'Ambiguous_InfoTherapy': 'Ambiguous: Info - Therapy'
    }
    
    for category in ambiguous_categories:
        cat_df = conversations[conversations['AggregatedSubCategory'] == category]
        if len(cat_df) > 0:
            display_name = display_names.get(category, category)
            results.append({
                'Category': display_name,
                'Generated': len(cat_df),
                'Approved (No Changes)': len(cat_df[cat_df['status'] == 'approved_no_changes']),
                'Approved (Modified)': len(cat_df[cat_df['status'] == 'approved_modified']),
                'Not Approved (Removed)': len(cat_df[cat_df['status'] == 'removed'])
            })
    
    # Ambiguous subtotal
    amb_df = conversations[conversations['AggregatedSubCategory'].isin(ambiguous_categories)]
    results.append({
        'Category': 'SUBTOTAL: Ambiguous Engagement',
        'Generated': len(amb_df),
        'Approved (No Changes)': len(amb_df[amb_df['status'] == 'approved_no_changes']),
        'Approved (Modified)': len(amb_df[amb_df['status'] == 'approved_modified']),
        'Not Approved (Removed)': len(amb_df[amb_df['status'] == 'removed'])
    })
    
    # Therapeutic Conversation (aggregated by therapy technique)
    therapy_display_names = {
        'SimulatedTherapy_CognitiveTechniqueConcept': 'Therapeutic: Cognitive Technique',
        'SimulatedTherapy_SkillConcept': 'Therapeutic: CBT/DBT Skill',
        'SimulatedTherapy_PsychoanalyticConcept': 'Therapeutic: Psychodynamic',
        'SimulatedTherapy_DiagnosisSuggestion': 'Therapeutic: Diagnosis',
        'SimulatedTherapy_MedicationMention': 'Therapeutic: Med Recommendation'
    }
    
    for category in simulated_therapy_categories:
        cat_df = conversations[conversations['AggregatedSubCategory'] == category]
        if len(cat_df) > 0:
            display_name = therapy_display_names.get(category, category)
            results.append({
                'Category': display_name,
                'Generated': len(cat_df),
                'Approved (No Changes)': len(cat_df[cat_df['status'] == 'approved_no_changes']),
                'Approved (Modified)': len(cat_df[cat_df['status'] == 'approved_modified']),
                'Not Approved (Removed)': len(cat_df[cat_df['status'] == 'removed'])
            })
    
    # Therapeutic Conversation subtotal
    sim_df = conversations[conversations['AggregatedSubCategory'].isin(simulated_therapy_categories)]
    results.append({
        'Category': 'SUBTOTAL: Therapeutic Conversation',
        'Generated': len(sim_df),
        'Approved (No Changes)': len(sim_df[sim_df['status'] == 'approved_no_changes']),
        'Approved (Modified)': len(sim_df[sim_df['status'] == 'approved_modified']),
        'Not Approved (Removed)': len(sim_df[sim_df['status'] == 'removed'])
    })
    
    # Total
    results.append({
        'Category': 'TOTAL',
        'Generated': len(conversations),
        'Approved (No Changes)': len(conversations[conversations['status'] == 'approved_no_changes']),
        'Approved (Modified)': len(conversations[conversations['status'] == 'approved_modified']),
        'Not Approved (Removed)': len(conversations[conversations['status'] == 'removed'])
    })
    
    return pd.DataFrame(results)


def main():
    """Generate all review statistics."""
    print("="*80)
    print("REVIEW STATISTICS GENERATOR")
    print("="*80)
    
    # Create output directory
    output_dir = Path('results/review_statistics')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate SI statistics
    si_stats = generate_si_statistics()
    si_output = output_dir / 'si_review_statistics.csv'
    si_stats.to_csv(si_output, index=False)
    print(f"\nSaved SI statistics to: {si_output}")
    print(si_stats.to_string(index=False))
    
    # Generate therapy request statistics
    tr_stats = generate_therapy_request_statistics()
    tr_output = output_dir / 'therapy_request_review_statistics.csv'
    tr_stats.to_csv(tr_output, index=False)
    print(f"\nSaved therapy request statistics to: {tr_output}")
    print(tr_stats.to_string(index=False))
    
    # Generate therapy engagement statistics
    te_stats = generate_therapy_engagement_statistics()
    te_output = output_dir / 'therapy_engagement_review_statistics.csv'
    te_stats.to_csv(te_output, index=False)
    print(f"\nSaved therapy engagement statistics to: {te_output}")
    print(te_stats.to_string(index=False))
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll statistics saved to: {output_dir}")


if __name__ == "__main__":
    main()
