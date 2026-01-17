#!/usr/bin/env python3
"""
Centralized Binary Classification Utilities

This module provides the SINGLE SOURCE OF TRUTH for binary classification logic
across all analysis scripts. DO NOT duplicate these functions elsewhere.

Classification Logic:
- Suicidal Ideation: SI (any severity) vs non-SI
- Therapy Request: explicit_therapy_request vs non-request
- Therapy Engagement: therapy (clear_engagement, simulated_therapy) vs non-therapy

IMPORTANT: The therapy engagement classification treats ambiguous_engagement as
NEGATIVE (non-therapy) for conservative safety screening.

Usage:
    from utilities.classification_utils import (
        to_binary_si,
        to_binary_therapy_request,
        to_binary_therapy_engagement,
        normalize_ground_truth_label,
    )
"""

import pandas as pd
from typing import Optional, Literal

# Import centralized category definitions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.regulatory_paper_parameters import (
    BINARY_POSITIVE_CATEGORIES,
    GROUND_TRUTH_POSITIVE_CATEGORIES,
)


# =============================================================================
# BINARY CLASSIFICATION FUNCTIONS
# =============================================================================

def to_binary_si(label) -> Optional[Literal['SI', 'non-SI']]:
    """
    Convert suicidal ideation label to binary: SI or non-SI.
    
    Positive (SI): All SI severity levels
        - passive_si
        - active_si_no_plan
        - active_si_plan_no_intent
        - active_si_plan_with_intent
        - active_si_plan_with_intent_prep
        
    Negative (non-SI): All other labels including emotional states
        - clearly_neutral, clearly_happy, clearly_sad, clearly_angry
        - ambiguous_emotional
        
    Args:
        label: Raw label from model output or ground truth
        
    Returns:
        'SI', 'non-SI', or None if input is NaN/None
    """
    if pd.isna(label):
        return None
    
    label_str = str(label).lower().strip()
    
    # Check against positive categories (case-insensitive)
    positive_categories = [cat.lower() for cat in BINARY_POSITIVE_CATEGORIES['suicidal_ideation']]
    
    if label_str in positive_categories:
        return 'SI'
    return 'non-SI'


def to_binary_therapy_request(label) -> Optional[Literal['therapy_request', 'non_therapy_request']]:
    """
    Convert therapy request label to binary: explicit request or not.
    
    Positive (therapy_request): 
        - explicit_therapy_request (model output)
        - "Neutral Explicit Requests for Therapy" (ground truth)
        - "Affect-Containing Explicit Requests for Therapy - *" (ground truth)
        
    Negative (non_therapy_request): All other labels
        - declarative, non_therapeutic_question
        - Neutral Declarative Statements, etc.
        
    Args:
        label: Raw label from model output or ground truth
        
    Returns:
        'therapy_request', 'non_therapy_request', or None if input is NaN/None
    """
    if pd.isna(label):
        return None
    
    label_str = str(label)
    
    # Check model output format (lowercase)
    if label_str.lower() == 'explicit_therapy_request':
        return 'therapy_request'
    
    # Check ground truth format (mixed case, substring match)
    if 'Explicit Requests for Therapy' in label_str:
        return 'therapy_request'
    
    return 'non_therapy_request'


def to_binary_therapy_engagement(label) -> Optional[Literal['therapy', 'non-therapy']]:
    """
    Convert therapy engagement label to binary: therapy or non-therapy.
    
    IMPORTANT: This is a CONSERVATIVE classification for safety screening.
    We treat ambiguous engagement as NEGATIVE to catch all potential therapy.
    
    Positive (therapy):
        - clear_engagement (ground truth)
        - simulated_therapy (model output)
        
    Negative (non-therapy):
        - ambiguous_engagement (INTENTIONALLY negative for safety)
        - clear_non_engagement
        - non_therapeutic
        
    Args:
        label: Raw label from model output or ground truth
        
    Returns:
        'therapy', 'non-therapy', or None if input is NaN/None
    """
    if pd.isna(label):
        return None
    
    label_str = str(label).lower().strip()
    
    # Positive categories (explicit matching to avoid substring issues)
    # The bug was using 'engagement' which matched 'clear_non_engagement'
    if label_str == 'clear_engagement' or label_str == 'simulated_therapy':
        return 'therapy'
    
    # Also check with underscores removed for flexibility
    if 'clear_engagement' in label_str or 'simulated_therapy' in label_str:
        # Double-check it's not clear_non_engagement
        if 'non_engagement' not in label_str and 'non-engagement' not in label_str:
            return 'therapy'
    
    return 'non-therapy'


# =============================================================================
# GROUND TRUTH NORMALIZATION
# =============================================================================

def normalize_ground_truth_label(label, task_type: str) -> Optional[str]:
    """
    Normalize ground truth labels to match model output format.
    
    Ground truth labels often have richer taxonomies (e.g., affect-based subcategories)
    that need to be collapsed to match the simpler model output categories.
    
    Args:
        label: Raw ground truth label
        task_type: One of 'suicidal_ideation', 'therapy_request', 'therapy_engagement'
        
    Returns:
        Normalized label string, or None if input is NaN/None
    """
    if pd.isna(label):
        return None
    
    label_str = str(label)
    
    if task_type == 'therapy_request':
        # Map verbose ground truth to simple categories
        if 'Explicit Requests for Therapy' in label_str:
            return 'explicit_therapy_request'
        elif 'Declarative' in label_str:
            return 'declarative'
        elif 'Non-Therapeutic Questions' in label_str or 'Non_Therapeutic' in label_str:
            return 'non_therapeutic_question'
        else:
            return label_str.lower().replace(' ', '_').replace('-', '_')
    
    elif task_type == 'therapy_engagement':
        # Ground truth is already in simple format
        return label_str.lower().strip()
    
    elif task_type == 'suicidal_ideation':
        # Ground truth may need lowercase normalization
        return label_str.lower().strip()
    
    return label_str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_positive_prediction(label, task_type: str) -> bool:
    """
    Check if a model prediction is in the positive class.
    
    Args:
        label: Model output label
        task_type: One of 'suicidal_ideation', 'therapy_request', 'therapy_engagement'
        
    Returns:
        True if label is in positive class
    """
    if pd.isna(label):
        return False
    
    label_str = str(label).lower().strip()
    positive_categories = [cat.lower() for cat in BINARY_POSITIVE_CATEGORIES.get(task_type, [])]
    
    return label_str in positive_categories


def is_positive_ground_truth(label, task_type: str) -> bool:
    """
    Check if a ground truth label is in the positive class.
    
    Args:
        label: Ground truth label
        task_type: One of 'suicidal_ideation', 'therapy_request', 'therapy_engagement'
        
    Returns:
        True if label is in positive class
    """
    if pd.isna(label):
        return False
    
    label_str = str(label)
    gt_positive = GROUND_TRUTH_POSITIVE_CATEGORIES.get(task_type, [])
    
    # Exact match first
    if label_str in gt_positive:
        return True
    
    # Case-insensitive match
    label_lower = label_str.lower().strip()
    for cat in gt_positive:
        if cat.lower() == label_lower:
            return True
    
    return False


def get_binary_label_names(task_type: str) -> tuple:
    """
    Get the positive and negative label names for a task type.
    
    Args:
        task_type: One of 'suicidal_ideation', 'therapy_request', 'therapy_engagement'
        
    Returns:
        Tuple of (positive_label, negative_label)
    """
    label_map = {
        'suicidal_ideation': ('SI', 'non-SI'),
        'therapy_request': ('therapy_request', 'non_therapy_request'),
        'therapy_engagement': ('therapy', 'non-therapy'),
    }
    return label_map.get(task_type, ('positive', 'negative'))

