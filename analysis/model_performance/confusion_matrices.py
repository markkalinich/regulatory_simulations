#!/usr/bin/env python3
"""
Confusion Matrix Module - Binary and multiclass confusion matrix generation.

This module provides functions to create confusion matrices for experiment results,
extracted from batch_results_analyzer.py for reusability and testing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
from typing import List, Optional
from .metrics_calculator import _create_binary_vectors
from config.constants import THERAPY_REQUEST_CATEGORY_ORDER


def create_binary_confusion_matrix(results_df: pd.DataFrame,
                                 model_family: str,
                                 model_size: str,
                                 experiment_type: str,
                                 binary_positive_categories: List[str],
                                 ground_truth_positive_categories: List[str],
                                 binary_classification_name: str,
                                 output_dir: Path) -> None:
    """
    Create binary confusion matrix for a single model.
    
    Args:
        results_df: Complete experiment results DataFrame
        model_family: Model family (e.g., 'gemma', 'qwen', 'llama')
        model_size: Model size (e.g., '1b', '4b', '12b')
        experiment_type: 'suicide_detection' or 'therapy_request'
        binary_positive_categories: List of categories considered positive
        binary_classification_name: Name for positive class (e.g., 'Suicide Risk')
        output_dir: Directory to save confusion matrix plot
    """
    # Filter results for this model
    model_data = results_df[
        (results_df['model_family'] == model_family) & 
        (results_df['model_size'] == model_size)
    ].copy()
    
    if len(model_data) == 0:
        return

    # Binary classification with parse failures treated as errors - use shared utility
    model_data, pred_codes, successful_mask = _create_binary_vectors(
        model_data, experiment_type, binary_positive_categories, ground_truth_positive_categories
    )

    positive_label = binary_classification_name
    negative_label = f'No {binary_classification_name}'
    
    y_true = np.where(model_data['true_positive'], positive_label, negative_label)
    y_pred = np.full(len(model_data), 'Parse Fail', dtype=object)
    y_pred[pred_codes == 0] = negative_label
    y_pred[pred_codes == 1] = positive_label

    # Only use 2 true labels (no parse failures in ground truth)
    true_labels = [negative_label, positive_label]
    pred_labels = [negative_label, positive_label, 'Parse Fail']
    
    # Build 2x3 matrix (2 true labels x 3 predicted labels)
    cm_2x3 = np.zeros((2, 3), dtype=int)
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            true_mask = (y_true == true_label)
            pred_mask = (y_pred == pred_label)
            cm_2x3[i, j] = np.sum(true_mask & pred_mask)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_2x3, annot=True, fmt='d', cmap='Blues', 
               xticklabels=pred_labels, yticklabels=true_labels, ax=ax)
    ax.set_title(f'{model_family.upper()} {model_size} - Binary Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()
    binary_dir = output_dir / 'confusion_matrices' / 'binary'
    binary_dir.mkdir(parents=True, exist_ok=True)
    cm_file = binary_dir / f'{model_family}_{model_size}_binary_confusion.png'
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_multiclass_confusion_matrix(results_df: pd.DataFrame,
                                     model_family: str,
                                     model_size: str,
                                     experiment_type: str,
                                     multiclass_labels: List[str],
                                     output_dir: Path) -> None:
    """
    Create multi-class confusion matrix including parse failures as separate category.
    
    Args:
        results_df: Complete experiment results DataFrame
        model_family: Model family (e.g., 'gemma', 'qwen', 'llama')
        model_size: Model size (e.g., '1b', '4b', '12b')
        experiment_type: 'suicide_detection' or 'therapy_request'
        multiclass_labels: Ordered list of all possible labels
        output_dir: Directory to save confusion matrix plot
    """
    model_data = results_df[
        (results_df['model_family'] == model_family) &
        (results_df['model_size'] == model_size)
    ].copy()
    
    if len(model_data) == 0:
        return

    if experiment_type in ['therapy_request', 'therapy_engagement']:
        # Determine the ground truth column name
        if experiment_type == 'therapy_request':
            prior_col = 'prior_therapy_request'
        else:  # therapy_engagement
            prior_col = 'prior_therapy_engagement'
        
        true_labels = model_data[prior_col].fillna('unknown')
        
        # Use experiment_type to determine which prediction column to use
        if experiment_type == 'therapy_engagement':
            predicted_labels = model_data['therapy_engagement'].copy()
        elif experiment_type == 'therapy_request':
            # Check for multiple possible column names for therapy request
            if 'therapy_request' in model_data.columns:
                predicted_labels = model_data['therapy_request'].copy()
            elif 'therapy_delivery' in model_data.columns:
                predicted_labels = model_data['therapy_delivery'].copy()
            else:
                predicted_labels = model_data['counseling_request'].copy()
        else:
            # Fallback for other experiment types
            predicted_labels = model_data['counseling_request'].copy()
    else:
        # For suicide detection, use safety_type columns
        true_labels = model_data['prior_safety_type'].fillna('unknown')
        predicted_labels = model_data['safety_type'].copy()
    
    predicted_labels = predicted_labels.fillna('parse_fail')
    predicted_labels[model_data['status'] != 'ok'] = 'parse_fail'

    # Get unique true labels (ground truth categories) and predicted labels (model outputs)
    # Order ground truth using centralized category ordering
    true_label_list = []
    
    if experiment_type == 'therapy_request':
        # Use therapy request category ordering
        for label in THERAPY_REQUEST_CATEGORY_ORDER:
            if label in true_labels.values:
                true_label_list.append(label)
    elif experiment_type == 'therapy_engagement':
        # Use therapy engagement ground truth category ordering
        for label in ['clear_non_engagement', 'ambiguous_engagement', 'clear_engagement']:
            if label in true_labels.values:
                true_label_list.append(label)
    
    # Add any other true labels that might exist
    for label in sorted(true_labels.unique()):
        if label not in true_label_list and label != 'unknown':
            true_label_list.append(label)
    
    # Order predicted labels: regular categories first, parse_fail always last
    predicted_label_list = []
    
    # For therapy experiments, use specific order
    if experiment_type == 'therapy_request':
        for label in ['declarative', 'non_therapeutic_question', 'explicit_therapy_request']:
            if label in predicted_labels.values:
                predicted_label_list.append(label)
    elif experiment_type == 'therapy_engagement':
        for label in ['non_therapeutic', 'ambiguous_engagement', 'simulated_therapy']:
            if label in predicted_labels.values:
                predicted_label_list.append(label)
    
    # Add all other predicted labels except parse_fail (alphabetically sorted)
    for label in sorted(predicted_labels.unique()):
        if label not in predicted_label_list and label != 'parse_fail':
            predicted_label_list.append(label)
    
    # Ensure parse_fail is always last
    if 'parse_fail' in predicted_labels.values:
        predicted_label_list.append('parse_fail')
    
    # Ensure parse_fail is included even if no failures occurred
    if 'parse_fail' not in predicted_label_list:
        predicted_label_list.append('parse_fail')

    # Create confusion matrix with true labels on Y-axis, predicted labels on X-axis
    # We need to specify both row and column labels separately
    cm_full = []
    for true_label in true_label_list:
        row = []
        for pred_label in predicted_label_list:
            count = ((true_labels == true_label) & (predicted_labels == pred_label)).sum()
            row.append(count)
        cm_full.append(row)
    
    cm = np.array(cm_full)
    
    # Calculate figure size based on number of labels
    fig_width = max(10, len(predicted_label_list) * 1.5)
    fig_height = max(8, len(true_label_list) * 0.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=predicted_label_list, yticklabels=true_label_list, ax=ax)
    
    if experiment_type == 'therapy_request':
        ax.set_title(f'{model_family.upper()} {model_size} - Therapy Request Confusion Matrix')
    elif experiment_type == 'therapy_engagement':
        ax.set_title(f'{model_family.upper()} {model_size} - Therapy Engagement Confusion Matrix')
    else:
        ax.set_title(f'{model_family.upper()} {model_size} - Safety Type Confusion Matrix')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()

    multi_dir = output_dir / 'confusion_matrices' / 'multiclass'
    multi_dir.mkdir(parents=True, exist_ok=True)
    cm_file = multi_dir / f'{model_family}_{model_size}_multiclass_confusion.png'
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    plt.close()


def normalize_family_name(family: str) -> str:
    """Normalize model family names (llama2, llama3.1, llama3.2 → llama)."""
    family_lower = family.lower()
    if family_lower.startswith('llama'):
        return 'llama'
    elif family_lower.startswith('qwen'):
        return 'qwen'
    elif family_lower.startswith('gemma'):
        return 'gemma'
    return family_lower


def create_binary_confusion_matrix_grid(results_df: pd.DataFrame,
                                       model_families: dict,
                                       experiment_type: str,
                                       binary_positive_categories: List[str],
                                       ground_truth_positive_categories: List[str],
                                       binary_classification_name: str,
                                       output_dir: Path) -> None:
    """
    Create faceted grid of binary confusion matrices with families as columns and sizes as rows.
    
    Args:
        results_df: Complete experiment results DataFrame
        model_families: Dictionary mapping family names to size lists (e.g., {'gemma': ['270m', '1b', ...], ...})
        experiment_type: 'suicide_detection' or 'therapy_request'
        binary_positive_categories: List of categories considered positive
        ground_truth_positive_categories: List of categories in ground truth that are positive
        binary_classification_name: Name for positive class (e.g., 'Suicide Risk')
        output_dir: Directory to save confusion matrix plot
    """
    # Normalize model_family column for matching (llama3.2 → llama, etc.)
    results_df = results_df.copy()
    results_df['model_family_normalized'] = results_df['model_family'].apply(normalize_family_name)
    
    # Define family order and get max rows needed
    family_order = ['gemma', 'qwen', 'llama']
    families_present = [f for f in family_order if f in model_families]
    max_rows = max(len(sizes) for sizes in model_families.values())
    
    # Create figure with subplots (30% smaller: 5*0.7=3.5 per col, 4*0.7=2.8 per row)
    n_cols = len(families_present)
    n_rows = max_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 2.8*n_rows))
    
    # Handle case where we only have one column or row
    if n_cols == 1 and n_rows == 1:
        axes = np.array([[axes]])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    positive_label = binary_classification_name
    negative_label = f'No {binary_classification_name}'
    cm_labels = [negative_label, positive_label, 'Parse Fail']
    
    # Create shortened labels with line breaks for grid display
    # Special handling for specific classification types
    if positive_label == 'Therapy Request':
        short_positive = 'Tx\nReq'
        short_negative = 'No Tx\nReq'
    elif positive_label == 'Therapy Engagement':
        short_positive = 'Tx\nEngaged'
        short_negative = 'No Tx'
    elif positive_label == 'SI':
        short_positive = 'SI'
        short_negative = 'No\nSI'
    else:
        # Default: split on spaces and join with newlines
        short_positive = positive_label.replace(' ', '\n')
        short_negative = negative_label.replace(' ', '\n')
    
    grid_labels = [short_negative, short_positive, 'Parse\nFail']
    grid_y_labels = [short_negative, short_positive]
    
    # Process each family
    for col_idx, family in enumerate(families_present):
        sizes = model_families[family]
        
        # Process each size (smallest to largest, top to bottom)
        for row_idx, size in enumerate(sizes):
            ax = axes[row_idx, col_idx]
            
            # Filter results for this model (using normalized family name)
            model_data = results_df[
                (results_df['model_family_normalized'] == family) & 
                (results_df['model_size'] == size)
            ].copy()
            
            if len(model_data) > 0:
                # Binary classification with parse failures
                model_data, pred_codes, successful_mask = _create_binary_vectors(
                    model_data, experiment_type, binary_positive_categories, ground_truth_positive_categories
                )
                
                y_true = np.where(model_data['true_positive'], positive_label, negative_label)
                y_pred = np.full(len(model_data), 'Parse Fail', dtype=object)
                y_pred[pred_codes == 0] = negative_label
                y_pred[pred_codes == 1] = positive_label
                
                # Create 2x3 matrix (2 true labels x 3 predicted labels)
                true_labels = [negative_label, positive_label]
                pred_labels = [negative_label, positive_label, 'Parse Fail']
                cm_2x3 = np.zeros((2, 3), dtype=int)
                for i, true_label in enumerate(true_labels):
                    for j, pred_label in enumerate(pred_labels):
                        true_mask = (y_true == true_label)
                        pred_mask = (y_pred == pred_label)
                        cm_2x3[i, j] = np.sum(true_mask & pred_mask)
                
                # Plot heatmap with line-broken labels (50% larger font: base 10->15 for annot, 8->12 for labels)
                sns.heatmap(cm_2x3, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=grid_labels, yticklabels=grid_y_labels,
                           ax=ax, cbar=False, square=False, annot_kws={'fontsize': 15})
                ax.set_title(f'{family.upper()} {size}', fontsize=12, fontweight='bold')
                
                # Rotate tick labels to horizontal with larger font
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=12)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=12)
                
                # Only show x-label on bottom row
                if row_idx == len(sizes) - 1:
                    ax.set_xlabel('Predicted', fontsize=9)
                else:
                    ax.set_xlabel('')
                
                # Only show y-label on leftmost column
                if col_idx == 0:
                    ax.set_ylabel('Actual', fontsize=9)
                else:
                    ax.set_ylabel('')
            else:
                ax.axis('off')
        
        # Turn off unused subplots in this column
        for row_idx in range(len(sizes), n_rows):
            axes[row_idx, col_idx].axis('off')
    
    plt.suptitle(f'Binary Confusion Matrices: {binary_classification_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot at confusion_matrices level (alongside binary/ and multiclass/ folders)
    grid_dir = output_dir / 'confusion_matrices'
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_file = grid_dir / 'binary_confusion_matrix_grid.png'
    plt.savefig(grid_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved binary confusion matrix grid: {grid_file}")


def create_all_confusion_matrices(results_df: pd.DataFrame,
                                model_families: dict,
                                experiment_type: str,
                                binary_positive_categories: List[str],
                                ground_truth_positive_categories: List[str],
                                binary_classification_name: str,
                                multiclass_labels: List[str],
                                output_dir: Path) -> None:
    """
    Generate confusion matrices for all models.
    
    Args:
        results_df: Complete experiment results DataFrame
        model_families: Dictionary mapping family names to size lists
        experiment_type: 'suicide_detection' or 'therapy_request'
        binary_positive_categories: List of categories considered positive for binary classification (model outputs)
        ground_truth_positive_categories: List of categories in ground truth that are positive
        binary_classification_name: Name for positive class (e.g., 'Suicide Risk')
        multiclass_labels: Ordered list of all possible labels for multiclass
        output_dir: Directory to save confusion matrix plots
    """
    print("Generating confusion matrices...")
    
    # Generate individual confusion matrices
    for family, sizes in model_families.items():
        for size in sizes:
            create_binary_confusion_matrix(
                results_df, family, size, experiment_type,
                binary_positive_categories, ground_truth_positive_categories, 
                binary_classification_name, output_dir
            )
            create_multiclass_confusion_matrix(
                results_df, family, size, experiment_type,
                multiclass_labels, output_dir
            )
    
    # Generate faceted grid of binary confusion matrices
    create_binary_confusion_matrix_grid(
        results_df, model_families, experiment_type,
        binary_positive_categories, ground_truth_positive_categories,
        binary_classification_name, output_dir
    )