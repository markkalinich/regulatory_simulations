#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert Review Sankey Diagram Generator with Configurations
Consolidated generator and configs for all experiment types.

Author: Mark Kalinich
Date: October 26, 2025
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.graph_objects as go
from utilities.figure_provenance import FigureProvenanceTracker
from config.constants import SI_LABELS, THERAPY_REQUEST_LABELS


@dataclass
class SankeyConfig:
    """Configuration for a specific dataset's Sankey diagram."""
    
    # Identity
    experiment_type: str  # 'si', 'therapy_request', 'therapy_engagement'
    figure_name: str
    prefix: str  # For provenance tracker
    
    # File paths
    raw_data_path: Path
    review_data_path: Path
    final_data_path: Path
    
    # Column names
    category_column: str  # 'Safety type', 'Counseling Request', 'SubCategory'
    id_column: Optional[str] = None  # For therapy_engagement (Example_ID)
    
    # Categories & labels
    category_order: List[str] = field(default_factory=list)
    category_labels: Dict[str, str] = field(default_factory=dict)
    
    # Review workflow
    has_p2_review: bool = True  # False for therapy_engagement (not yet)
    has_modification_edits: bool = False  # True for therapy_engagement
    edit_status_column: Optional[str] = None  # For therapy_engagement
    
    # Node colors (RGBA strings)
    category_color_map: Dict[str, str] = field(default_factory=dict)
    action_colors: Dict[str, str] = field(default_factory=dict)
    
    # Output metadata
    downsampling_note: str = ""
    
    def __post_init__(self):
        if not self.action_colors:
            self.action_colors = {
                'Kept': 'rgba(46, 204, 113, 0.9)',
                'Modified': 'rgba(241, 196, 15, 0.9)',
                'Removed': 'rgba(231, 76, 60, 0.9)',
                'Final': 'rgba(26, 188, 156, 0.9)',
                'Not Used': 'rgba(149, 165, 166, 0.9)'
            }


class ExpertReviewSankeyGenerator:
    """Generate expert review Sankey diagrams for any dataset."""
    
    def __init__(self, config: SankeyConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        
        # Initialize provenance tracker
        self.tracker = FigureProvenanceTracker(
            figure_name=config.figure_name,
            base_dir=project_root / "results/data_validation/psychiatrist_review_sankey_diagrams",
            prefix=config.prefix
        )
    
    def load_data(self) -> tuple:
        """Load raw, review, and final datasets."""
        print("Loading datasets...")
        raw_data = pd.read_csv(self.config.raw_data_path)
        
        # Handle encoding for therapy_engagement review file
        try:
            review_data = pd.read_csv(self.config.review_data_path)
        except UnicodeDecodeError:
            review_data = pd.read_csv(self.config.review_data_path, encoding='latin-1')
        
        final_data = pd.read_csv(self.config.final_data_path)
        
        # For therapy_engagement, count conversations not turns
        if self.config.id_column:
            raw_count = raw_data[self.config.id_column].nunique()
            review_count = review_data[self.config.id_column].nunique()
            final_count = final_data[self.config.id_column].nunique() if self.config.id_column in final_data.columns else len(final_data)
            unit = "conversations"
        else:
            raw_count = len(raw_data)
            review_count = len(review_data)
            final_count = len(final_data)
            unit = "statements"
        
        print(f"  Raw: {raw_count} {unit}")
        print(f"  Review: {review_count} {unit}")
        print(f"  Final: {final_count} {unit}")
        
        # Track inputs
        self.tracker.add_input_dataset(
            self.config.raw_data_path,
            description=f"Raw {self.config.experiment_type} data",
            columns_used=[self.config.category_column]
        )
        self.tracker.add_input_dataset(
            self.config.review_data_path,
            description=f"Psychiatrist review data",
            columns_used=[self.config.category_column]
        )
        self.tracker.add_input_dataset(
            self.config.final_data_path,
            description=f"Final dataset after review",
            columns_used=[self.config.category_column]
        )
        
        return raw_data, review_data, final_data
    
    def prepare_review_data(self, review_data: pd.DataFrame, raw_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add P1/P2 action columns."""
        print("\nPreparing review data...")
        
        # For therapy_engagement with conversation-level data, simplify subcategories first
        if self.config.id_column and self.config.experiment_type == 'therapy_engagement':
            # Simplify subcategories to base groups
            def simplify_subcategory(subcat):
                parts = subcat.split('_')
                
                # NonTherapeutic: keep as-is (already at right level)
                if parts[0] == 'NonTherapeutic':
                    return subcat
                
                # Ambiguous: parts 0 and 1 (skip the disorder suffix in part 2)
                # e.g., Ambiguous_DisclosureBoundary_Anxiety → Ambiguous_DisclosureBoundary
                elif parts[0] == 'Ambiguous' and len(parts) >= 2:
                    return f"{parts[0]}_{parts[1]}"
                
                # SimulatedTherapy: parts 0 + parts[2:] (skip the disorder in the middle, keep technique)
                # e.g., SimulatedTherapy_Anxiety_CognitiveTechniqueConcept → SimulatedTherapy_CognitiveTechniqueConcept
                elif parts[0] == 'SimulatedTherapy' and len(parts) >= 3:
                    technique = '_'.join(parts[2:])  # Everything after the disorder
                    return f"{parts[0]}_{technique}"
                
                # Default fallback
                else:
                    return subcat
            
            # Simplify subcategories in review data
            review_data = review_data.copy()
            review_data['base_subcategory'] = review_data[self.config.category_column].apply(simplify_subcategory)
            
            # Group by Example_ID to get one row per conversation with P1 and P2 status
            review_grouped = review_data.groupby(self.config.id_column).agg({
                'base_subcategory': 'first',
                'Psychiatrist_01': 'first',
                'Psychiatrist_02': 'first'
            }).reset_index()
            review_grouped.rename(columns={'base_subcategory': self.config.category_column}, inplace=True)
            
            # Use the standard P1/P2 processing below
            review_data = review_grouped
        
        # Standard processing for SI and therapy_request
        # Simplify P1 actions
        def simplify_p1_action(action):
            if pd.isna(action):
                return 'Removed'
            action_str = str(action).upper()
            if 'KEPT_EXACT_MATCH' in action_str:
                return 'Kept'
            elif 'KEPT_WITH_CHANGES' in action_str:
                return 'Modified'
            elif 'REMOVED' in action_str:
                return 'Removed'
            else:
                # Fallback for any other values
                return 'Removed'
        
        review_data['P1_Action'] = review_data['Psychiatrist_01'].apply(simplify_p1_action)
        
        # P2 actions (if available)
        if self.config.has_p2_review and 'Psychiatrist_02' in review_data.columns:
            def simplify_p2_action(row):
                # P2 only reviews items that P1 kept or modified
                if row['P1_Action'] == 'Removed':
                    return None  # P2 didn't review this
                elif pd.isna(row['Psychiatrist_02']):
                    return None  # No P2 review
                
                p2_status = str(row['Psychiatrist_02']).upper()
                if 'KEPT_EXACT_MATCH' in p2_status:
                    return 'Kept'
                elif 'KEPT_WITH_CHANGES' in p2_status:
                    return 'Modified'
                elif p2_status == 'KEPT':
                    # Handle short format used in SI and therapy_request P2 columns
                    return 'Kept'
                elif 'REMOVED' in p2_status:
                    return 'Removed'
                else:
                    return None
            review_data['P2_Action'] = review_data.apply(simplify_p2_action, axis=1)
        
        return review_data
    
    def create_flow_data(self, review_data: pd.DataFrame, 
                        final_data: pd.DataFrame) -> pd.DataFrame:
        """Build flow connections for Sankey."""
        print("\nCreating flow connections...")
        
        # Calculate actual final dataset size from the data
        if self.config.id_column:
            actual_final_size = final_data[self.config.id_column].nunique() if self.config.id_column in final_data.columns else len(final_data)
        else:
            actual_final_size = len(final_data)
        
        flows = []
        
        # Categories → P1 Actions
        for category in self.config.category_order:
            cat_data = review_data[review_data[self.config.category_column] == category]
            
            for p1_action in ['Kept', 'Modified', 'Removed']:
                action_col = 'P1_Action_Detail' if self.config.has_modification_edits else 'P1_Action'
                count = len(cat_data[cat_data[action_col] == p1_action])
                if count > 0:
                    flows.append({
                        'source': self.config.category_labels[category],
                        'target': f"P1: {p1_action}",
                        'value': count
                    })
        
        # P1 Actions → P2 Actions (if P2 review exists)
        if self.config.has_p2_review:
            for p1_action in ['Kept', 'Modified']:
                p1_filtered = review_data[review_data['P1_Action'] == p1_action]
                for p2_action in ['Kept', 'Modified', 'Removed']:
                    count = len(p1_filtered[p1_filtered['P2_Action'] == p2_action])
                    if count > 0:
                        flows.append({
                            'source': f"P1: {p1_action}",
                            'target': f"P2: {p2_action}",
                            'value': count
                        })
            
            # P2 Kept/Modified → Final + Not Used
            p2_approved_count = len(review_data[
                (review_data['P1_Action'].isin(['Kept', 'Modified'])) &
                (review_data['P2_Action'].isin(['Kept', 'Modified']))
            ])
            not_used_total = p2_approved_count - actual_final_size
            
            # Get counts for each P2 action
            p2_kept_total = len(review_data[review_data['P2_Action'] == 'Kept'])
            p2_modified_total = len(review_data[review_data['P2_Action'] == 'Modified'])
            
            # If there's downsampling, distribute Not Used proportionally
            if not_used_total > 0:
                # Proportional allocation of Not Used using round() for better accuracy
                kept_ratio = p2_kept_total / p2_approved_count
                p2_kept_not_used = round(not_used_total * kept_ratio)
                
                # Ensure total Not Used = not_used_total (adjust for any rounding error)
                p2_modified_not_used = not_used_total - p2_kept_not_used
                
                # Calculate flows to Final
                p2_kept_to_final = p2_kept_total - p2_kept_not_used
                p2_modified_to_final = p2_modified_total - p2_modified_not_used
                
                # P2: Kept flows
                if p2_kept_to_final > 0:
                    flows.append({'source': 'P2: Kept', 'target': 'Final', 'value': p2_kept_to_final})
                if p2_kept_not_used > 0:
                    flows.append({'source': 'P2: Kept', 'target': 'Not Used', 'value': p2_kept_not_used})
                
                # P2: Modified flows
                if p2_modified_to_final > 0:
                    flows.append({'source': 'P2: Modified', 'target': 'Final', 'value': p2_modified_to_final})
                if p2_modified_not_used > 0:
                    flows.append({'source': 'P2: Modified', 'target': 'Not Used', 'value': p2_modified_not_used})
            else:
                # No downsampling - all P2 approved items go to Final
                if p2_kept_total > 0:
                    flows.append({'source': 'P2: Kept', 'target': 'Final', 'value': p2_kept_total})
                if p2_modified_total > 0:
                    flows.append({'source': 'P2: Modified', 'target': 'Final', 'value': p2_modified_total})
        else:
            # No P2 review - some P1:Kept items are downsampled to Not Used
            action_col = 'P1_Action_Detail' if self.config.has_modification_edits else 'P1_Action'
            p1_kept_count = len(review_data[review_data[action_col] == 'Kept'])
            p1_modified_count = len(review_data[review_data[action_col] == 'Modified'])
            
            # Calculate how many from P1:Kept go to Not Used vs P1 Approved
            total_approved = p1_kept_count + p1_modified_count
            not_used = total_approved - actual_final_size
            p1_kept_to_approved = p1_kept_count - not_used
            
            # P1: Kept splits to P1 Approved and Not Used
            if p1_kept_to_approved > 0:
                flows.append({
                    'source': 'P1: Kept',
                    'target': 'P1 Approved',
                    'value': p1_kept_to_approved
                })
            if not_used > 0:
                flows.append({
                    'source': 'P1: Kept',
                    'target': 'Not Used',
                    'value': not_used
                })
            
            # P1: Modified goes entirely to P1 Approved
            if p1_modified_count > 0:
                flows.append({
                    'source': 'P1: Modified',
                    'target': 'P1 Approved',
                    'value': p1_modified_count
                })
            
            # P1 Approved → Final (all of it)
            flows.append({
                'source': 'P1 Approved',
                'target': 'Final',
                'value': actual_final_size
            })
        
        flows_df = pd.DataFrame(flows)
        print(f"  Created {len(flows_df)} flow connections")
        
        return flows_df
    
    def create_sankey_diagram(self, flows_df: pd.DataFrame) -> go.Figure:
        """Create Plotly Sankey diagram."""
        print("\nGenerating Sankey diagram...")
        
        # Aggregate flows
        agg_flows = flows_df.groupby(['source', 'target'])['value'].sum().reset_index()
        
        # Build ordered node list
        category_nodes = [self.config.category_labels[cat] 
                         for cat in self.config.category_order]
        p1_nodes = ['P1: Kept', 'P1: Modified', 'P1: Removed']
        p2_nodes = ['P2: Kept', 'P2: Modified', 'P2: Removed'] if self.config.has_p2_review else ['P1 Approved']
        final_nodes = ['Final', 'Not Used']
        
        # Filter to only include nodes that appear in the flows
        all_sources = set(agg_flows['source'].unique())
        all_targets = set(agg_flows['target'].unique())
        all_flow_nodes = all_sources | all_targets
        
        all_nodes = (
            [n for n in category_nodes if n in all_flow_nodes] +
            [n for n in p1_nodes if n in all_flow_nodes] +
            [n for n in p2_nodes if n in all_flow_nodes] +
            [n for n in final_nodes if n in all_flow_nodes]
        )
        
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Add counts to labels
        node_labels = []
        for node in all_nodes:
            inflow = agg_flows[agg_flows['target'] == node]['value'].sum()
            outflow = agg_flows[agg_flows['source'] == node]['value'].sum()
            total = max(inflow, outflow) if inflow > 0 or outflow > 0 else 0
            node_labels.append(f"{node} ({int(total)})" if total > 0 else node)
        
        # Map indices
        agg_flows['source_idx'] = agg_flows['source'].map(node_dict)
        agg_flows['target_idx'] = agg_flows['target'].map(node_dict)
        
        # Assign colors
        node_colors = []
        for node in all_nodes:
            # Category nodes - use config's color map
            if node in category_nodes:
                color_assigned = False
                for pattern, color in self.config.category_color_map.items():
                    if pattern.lower() in node.lower():
                        node_colors.append(color)
                        color_assigned = True
                        break
                if not color_assigned:
                    # Use default color
                    node_colors.append(self.config.category_color_map.get('default', 'rgba(52, 152, 219, 0.9)'))
            # Action nodes - use action colors
            else:
                color_assigned = False
                for action, color in self.config.action_colors.items():
                    if action in node:
                        node_colors.append(color)
                        color_assigned = True
                        break
                if not color_assigned:
                    node_colors.append('rgba(149, 165, 166, 0.9)')  # Default gray
        
        # Assign link colors based on target node
        link_colors = []
        for _, row in agg_flows.iterrows():
            target_node = all_nodes[row['target_idx']]
            # Color flows based on destination
            if 'Kept' in target_node or 'Final' in target_node or 'Approved' in target_node:
                link_colors.append('rgba(46, 204, 113, 0.3)')  # Green for kept/final/approved
            elif 'Removed' in target_node:
                link_colors.append('rgba(231, 76, 60, 0.3)')  # Red for removed
            elif 'Modified' in target_node:
                link_colors.append('rgba(241, 196, 15, 0.3)')  # Yellow for modified
            elif 'Not Used' in target_node:
                link_colors.append('rgba(149, 165, 166, 0.3)')  # Gray for not used
            else:
                link_colors.append('rgba(0, 0, 0, 0.2)')  # Default transparent black
        
        # Create figure
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=agg_flows['source_idx'].tolist(),
                target=agg_flows['target_idx'].tolist(),
                value=agg_flows['value'].tolist(),
                color=link_colors
            )
        ))
        
        # Set title based on experiment type
        title_map = {
            'si': 'Suicidal Ideation',
            'therapy_request': 'Therapy Request Expert Review Flow',
            'therapy_engagement': 'Therapy Engagement Expert Review Flow'
        }
        title = title_map.get(self.config.experiment_type, 
                              f"{self.config.experiment_type.replace('_', ' ').title()} Expert Review Flow")
        
        fig.update_layout(
            title=title,
            font_size=12,
            height=600
        )
        
        # Save outputs
        output_html = self.tracker.get_output_path(f"{self.config.figure_name}.html")
        output_png = self.tracker.get_output_path(f"{self.config.figure_name}.png")
        
        fig.write_html(str(output_html))
        fig.write_image(str(output_png), width=1000, height=600, scale=3)
        
        self.tracker.add_output_file(output_html, file_type="interactive HTML")
        self.tracker.add_output_file(output_png, file_type="static PNG")
        
        print(f"  Saved: {output_html.name}, {output_png.name}")
        
        return fig
    
    def generate(self):
        """Main generation workflow."""
        print("="*80)
        print(f"{self.config.experiment_type.replace('_', ' ').upper()} EXPERT REVIEW SANKEY DIAGRAM")
        print("="*80)
        
        # Execute workflow
        raw_data, review_data, final_data = self.load_data()
        
        # Pass raw_data for therapy_engagement
        if self.config.has_modification_edits:
            review_data = self.prepare_review_data(review_data, raw_data)
        else:
            review_data = self.prepare_review_data(review_data)
        
        # Calculate actual final dataset size from the data
        if self.config.id_column:
            actual_final_size = final_data[self.config.id_column].nunique() if self.config.id_column in final_data.columns else len(final_data)
        else:
            actual_final_size = len(final_data)
        
        flows_df = self.create_flow_data(review_data, final_data)
        fig = self.create_sankey_diagram(flows_df)
        
        # Save provenance
        self.tracker.set_analysis_parameters(
            experiment_type=self.config.experiment_type,
            final_dataset_size=actual_final_size,
            downsampling=self.config.downsampling_note
        )
        self.tracker.save_provenance()
        
        print("\n" + "="*80)
        print(f"COMPLETE! Outputs: {self.tracker.output_dir}")
        print("="*80)
        
        return fig

# Project root and shared resources
PROJECT_ROOT = project_root
COLORS = {
    'kept': 'rgba(46, 204, 113, 0.9)', 'modified': 'rgba(241, 196, 15, 0.9)',
    'removed': 'rgba(231, 76, 60, 0.9)', 'final': 'rgba(26, 188, 156, 0.9)',
    'not_used': 'rgba(149, 165, 166, 0.9)', 'blue': 'rgba(52, 152, 219, 0.9)',
    'red': 'rgba(231, 76, 60, 0.9)', 'orange': 'rgba(230, 126, 34, 0.9)'
}

# Helper to build config with common defaults
def _config(exp_type, prefix, raw_file, review_file, final_file, cat_col, cat_order, cat_labels, 
            color_map, notes, id_col=None):
    return SankeyConfig(
        experiment_type=exp_type, figure_name=f'{exp_type}_expert_review_sankey', prefix=prefix,
        raw_data_path=PROJECT_ROOT / f'data/inputs/raw_model_results/{raw_file}',
        review_data_path=PROJECT_ROOT / f'data/inputs/intermediate_files/{review_file}',
        final_data_path=PROJECT_ROOT / f'data/inputs/finalized_input_data/{final_file}',
        category_column=cat_col, id_column=id_col, category_order=cat_order, 
        category_labels=cat_labels, category_color_map=color_map,
        has_p2_review=True, downsampling_note=notes
    )

# SI Configuration
SI_CONFIG = _config(
    'si', 'SI', 
    'SI_balanced_100_per_category_ordered_input.csv',
    'SI_psychiatrist_01_and_02_scores.csv',
    'SI_finalized_sentences.csv',
    'Safety type',
    ['clearly_happy', 'clearly_neutral', 'clearly_sad', 'clearly_angry', 'ambiguous_emotional',
     'passive_si', 'active_si_abstract', 'active_si_plan_no_intent', 'active_si_plan_with_intent', 'preparatory_si'],
    {**{k: SI_LABELS[k] for k in ['clearly_happy', 'clearly_neutral', 'clearly_sad', 'clearly_angry', 'ambiguous_emotional']},
     'passive_si': 'Passive SI', 'active_si_abstract': 'Active SI (abstract)',
     'active_si_plan_no_intent': 'Active SI w/ plan (no intent)', 
     'active_si_plan_with_intent': 'Active SI w/ plan & intent', 'preparatory_si': 'Preparatory SI'},
    {'si': COLORS['red'], 'default': COLORS['blue']},
    '45 per category'
)

# Therapy Request Configuration
THERAPY_REQUEST_CONFIG = _config(
    'therapy_request', 'tx_request',
    'therapy_request_100_per_category_reformatted.csv',
    'therapy_request_psychiatrist_01_and_02_scores.csv',
    'therapy_request_finalized_sentences.csv',
    'Counseling Request',
    ['Neutral Declarative Statements', 'Neutral Non-Therapeutic Questions', 'Neutral Explicit Requests for Therapy',
     'Affect-Containing Declarative Statements - Clearly Happy', 'Affect-Containing Declarative Statements - Clearly Sad',
     'Affect-Containing Declarative Statements - Clearly Angry', 'Affect-Containing Non-Therapeutic Questions - Clearly Happy',
     'Affect-Containing Non-Therapeutic Questions - Clearly Sad', 'Affect-Containing Non-Therapeutic Questions - Clearly Angry',
     'Affect-Containing Explicit Requests for Therapy - Clearly Happy', 'Affect-Containing Explicit Requests for Therapy - Clearly Sad',
     'Affect-Containing Explicit Requests for Therapy - Clearly Angry'],
    {k: v.replace('\n', ' ') for k, v in THERAPY_REQUEST_LABELS.items()},
    {'request': COLORS['orange'], 'default': COLORS['blue']},
    '65 per category'
)

# Therapy Engagement Configuration
THERAPY_ENGAGEMENT_CONFIG = _config(
    'therapy_engagement', 'tx_engagement',
    'therapy_engagement_conversations_downsampled_150.csv',
    'therapy_engagement_psychiatrist_01_and_02_scores.csv',
    'therapy_engagement_finalized_sentences.csv',
    'SubCategory',
    ['NonTherapeutic_CreativeWriting', 'NonTherapeutic_InfoSeeking', 'NonTherapeutic_PlanningOrg',
     'NonTherapeutic_PracticalTask', 'NonTherapeutic_TechnicalCoding', 'Ambiguous_DisclosureBoundary',
     'Ambiguous_InfoPathology', 'Ambiguous_InfoTherapy', 'SimulatedTherapy_CognitiveTechniqueConcept',
     'SimulatedTherapy_DiagnosisSuggestion', 'SimulatedTherapy_MedicationMention',
     'SimulatedTherapy_PsychoanalyticConcept', 'SimulatedTherapy_SkillConcept'],
    {'NonTherapeutic_CreativeWriting': 'Creative Writing', 'NonTherapeutic_InfoSeeking': 'Info Seeking',
     'NonTherapeutic_PlanningOrg': 'Planning/Org', 'NonTherapeutic_PracticalTask': 'Practical Task',
     'NonTherapeutic_TechnicalCoding': 'Technical/Coding', 'Ambiguous_DisclosureBoundary': 'Disclosure',
     'Ambiguous_InfoPathology': 'Disease Info', 'Ambiguous_InfoTherapy': 'Therapy Info',
     'SimulatedTherapy_CognitiveTechniqueConcept': 'Tx - Cog. Technique', 'SimulatedTherapy_DiagnosisSuggestion': 'Tx - Diagnosis',
     'SimulatedTherapy_MedicationMention': 'Tx - Medications', 'SimulatedTherapy_PsychoanalyticConcept': 'Tx - Psychodynamics',
     'SimulatedTherapy_SkillConcept': 'Tx - Skills'},
    {'tx -': COLORS['red'], 'disclosure': COLORS['blue'], 'disease info': COLORS['blue'], 
     'therapy info': COLORS['blue'], 'default': COLORS['blue']},
    'P1 and P2 review complete',
    id_col='Example_ID'
)


if __name__ == '__main__':
    import sys
    
    # Determine which config to use based on command line argument
    if len(sys.argv) > 1:
        experiment_type = sys.argv[1].lower()
        if experiment_type == 'si':
            config = SI_CONFIG
        elif experiment_type == 'therapy_request':
            config = THERAPY_REQUEST_CONFIG
        elif experiment_type == 'therapy_engagement':
            config = THERAPY_ENGAGEMENT_CONFIG
        else:
            print(f"Unknown experiment type: {experiment_type}")
            print("Usage: python sankey_diagram_configs.py [si|therapy_request|therapy_engagement]")
            sys.exit(1)
    else:
        # Default to SI if no argument provided
        print("No experiment type specified. Running all three...")
        for config in [SI_CONFIG, THERAPY_REQUEST_CONFIG, THERAPY_ENGAGEMENT_CONFIG]:
            generator = ExpertReviewSankeyGenerator(config, PROJECT_ROOT)
            generator.generate()
            print("\n" + "="*80 + "\n")
        sys.exit(0)
    
    # Run the specified config
    generator = ExpertReviewSankeyGenerator(config, PROJECT_ROOT)
    generator.generate()
