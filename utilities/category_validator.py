#!/usr/bin/env python3
"""
Category Validation - Prevent prompt/config mismatches

This module validates that prompt output categories match config expected categories
before running expensive experiments.

Prevents taxonomy mismatch bugs like:
- DESIGN-1 issue: Prompt outputs new taxonomy, config expects old taxonomy
- Result: 20-25 point F1 score underestimation (Llama 70b: actual 0.94, reported 0.70)

Key Functions:
- extract_categories_from_prompt(): Parse allowed values from prompt text
- validate_prompt_config_match(): Ensure prompt and config categories align
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from config.experiment_config import ExperimentConfig


class CategoryValidationError(Exception):
    """Raised when prompt categories don't match config categories."""
    pass


def extract_categories_from_prompt(prompt_path: str) -> Dict[str, List[str]]:
    """
    Extract allowed category values from prompt file.
    
    Parses prompts that specify allowed values like:
        - safety_type ∈ ["category1", "category2", ...]
        - therapy_request ∈ ["category1", "category2", ...]
    
    Args:
        prompt_path: Path to prompt file
        
    Returns:
        Dictionary mapping field names to lists of allowed categories
        Example: {"safety_type": ["passive_si", ...], "therapy_request": [...]}
        
    Raises:
        CategoryValidationError: If prompt format cannot be parsed
    """
    prompt_file = Path(prompt_path)
    if not prompt_file.exists():
        raise CategoryValidationError(f"Prompt file not found: {prompt_path}")
    
    prompt_text = prompt_file.read_text()
    categories = {}
    
    # Pattern matches lines like:
    # - safety_type ∈ [
    # - therapy_request ∈ [
    # - therapy_engagement ∈ [
    field_pattern = r'-\s+(\w+)\s+∈\s+\['
    
    # Find all field declarations
    lines = prompt_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.search(field_pattern, line)
        
        if match:
            field_name = match.group(1)
            field_categories = []
            
            # Collect categories until we hit the closing bracket
            i += 1
            while i < len(lines):
                cat_line = lines[i].strip()
                
                # Stop at closing bracket
                if cat_line.startswith(']'):
                    break
                
                # Extract quoted category name
                cat_match = re.search(r'"([^"]+)"', cat_line)
                if cat_match:
                    field_categories.append(cat_match.group(1))
                
                i += 1
            
            if field_categories:
                categories[field_name] = field_categories
        
        i += 1
    
    if not categories:
        raise CategoryValidationError(
            f"Could not extract categories from prompt: {prompt_path}\n"
            f"Expected format: '- field_name ∈ [\\n  \"category1\",\\n  \"category2\"\\n]'"
        )
    
    return categories


def validate_prompt_config_match(prompt_path: str, 
                                 config: ExperimentConfig,
                                 strict: bool = True) -> Dict[str, Any]:
    """
    Validate that prompt output categories match config expected categories.
    
    This prevents expensive experiment failures due to taxonomy mismatches.
    
    Args:
        prompt_path: Path to prompt file
        config: Experiment configuration with expected categories
        strict: If True, raise exception on mismatch; if False, return warnings
        
    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "prompt_categories": dict,
            "config_categories": dict,
            "mismatches": list of error descriptions,
            "warnings": list of warning messages
        }
        
    Raises:
        CategoryValidationError: If strict=True and categories don't match
        
    Example:
        >>> config = SuicideDetectionConfig()
        >>> result = validate_prompt_config_match("prompts/si_v2.txt", config)
        >>> if not result["valid"]:
        ...     print(f"Validation failed: {result['mismatches']}")
    """
    # Extract categories from prompt
    try:
        prompt_categories = extract_categories_from_prompt(prompt_path)
    except CategoryValidationError as e:
        if strict:
            raise
        return {
            "valid": False,
            "prompt_categories": {},
            "config_categories": {},
            "mismatches": [str(e)],
            "warnings": []
        }
    
    # Get expected categories from config based on experiment type
    experiment_type = config.experiment_type
    config_categories = {}
    mismatches = []
    warnings = []
    
    # Map experiment types to their primary prediction fields
    field_mapping = {
        'suicidal_ideation': 'safety_type',
        'therapy_request': 'therapy_request', 
        'therapy_engagement': 'therapy_engagement'
    }
    
    primary_field = field_mapping.get(experiment_type)
    if not primary_field:
        warning = f"Unknown experiment type: {experiment_type}. Skipping validation."
        warnings.append(warning)
        return {
            "valid": not strict,
            "prompt_categories": prompt_categories,
            "config_categories": config_categories,
            "mismatches": [],
            "warnings": [warning]
        }
    
    # Check if prompt declares the expected field
    if primary_field not in prompt_categories:
        mismatch = (
            f"Prompt missing expected field '{primary_field}' for experiment type '{experiment_type}'. "
            f"Prompt declares fields: {list(prompt_categories.keys())}"
        )
        mismatches.append(mismatch)
    
    # For experiments with multiclass categories, validate those
    if hasattr(config, 'binary_positive_categories'):
        # Get the positive categories (used for binary classification)
        positive_cats = set(config.binary_positive_categories)
        
        # Get all categories from prompt for the primary field
        if primary_field in prompt_categories:
            prompt_cats = set(prompt_categories[primary_field])
            config_categories[primary_field] = list(positive_cats)
            
            # Check if positive categories are subset of prompt categories
            missing_in_prompt = positive_cats - prompt_cats
            if missing_in_prompt:
                mismatch = (
                    f"Config expects positive categories {missing_in_prompt} "
                    f"but they're not in prompt's allowed values for '{primary_field}'. "
                    f"Prompt allows: {prompt_cats}"
                )
                mismatches.append(mismatch)
            
            # Warn if config doesn't recognize some prompt categories
            extra_in_prompt = prompt_cats - positive_cats
            if extra_in_prompt:
                # Check if these are valid negative categories
                warning = (
                    f"Prompt defines categories {extra_in_prompt} for '{primary_field}' "
                    f"that are not in config's positive categories. This is OK if they're valid negatives, "
                    f"but verify this is intentional."
                )
                warnings.append(warning)
    
    # Determine overall validity
    valid = len(mismatches) == 0
    
    result = {
        "valid": valid,
        "prompt_categories": prompt_categories,
        "config_categories": config_categories,
        "mismatches": mismatches,
        "warnings": warnings
    }
    
    # Raise exception if strict mode and validation failed
    if strict and not valid:
        error_msg = (
            f"\n{'='*70}\n"
            f"CATEGORY VALIDATION FAILED\n"
            f"{'='*70}\n"
            f"Prompt: {prompt_path}\n"
            f"Experiment type: {experiment_type}\n"
            f"\nMismatches:\n"
        )
        for i, mismatch in enumerate(mismatches, 1):
            error_msg += f"  {i}. {mismatch}\n"
        
        if warnings:
            error_msg += f"\nWarnings:\n"
            for i, warning in enumerate(warnings, 1):
                error_msg += f"  {i}. {warning}\n"
        
        error_msg += (
            f"\nThis validation prevents expensive experiment failures due to taxonomy mismatches.\n"
            f"Update either the prompt or config to align categories before running experiments.\n"
            f"{'='*70}\n"
        )
        raise CategoryValidationError(error_msg)
    
    return result


def print_validation_report(result: Dict[str, Any], prompt_path: str) -> None:
    """
    Print a formatted validation report.
    
    Args:
        result: Validation result from validate_prompt_config_match()
        prompt_path: Path to prompt file (for display)
    """
    status = "✅ VALID" if result["valid"] else "❌ INVALID"
    print(f"\n{'='*70}")
    print(f"Category Validation Report: {status}")
    print(f"{'='*70}")
    print(f"Prompt: {prompt_path}")
    
    if result["prompt_categories"]:
        print(f"\nPrompt declares these fields:")
        for field, cats in result["prompt_categories"].items():
            print(f"  {field}: {len(cats)} categories")
            for cat in cats:
                print(f"    - {cat}")
    
    if result["config_categories"]:
        print(f"\nConfig expects these positive categories:")
        for field, cats in result["config_categories"].items():
            print(f"  {field}: {cats}")
    
    if result["mismatches"]:
        print(f"\n❌ Mismatches found:")
        for i, mismatch in enumerate(result["mismatches"], 1):
            print(f"  {i}. {mismatch}")
    
    if result["warnings"]:
        print(f"\n⚠️  Warnings:")
        for i, warning in enumerate(result["warnings"], 1):
            print(f"  {i}. {warning}")
    
    if result["valid"]:
        print(f"\n✅ Validation passed! Prompt and config categories align.")
    else:
        print(f"\n❌ Validation failed! Fix mismatches before running experiments.")
    
    print(f"{'='*70}\n")
