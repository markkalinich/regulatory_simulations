#!/usr/bin/env python3
"""
Experiment Data Loader for Safety Simulations

Loads experiment data from the CACHE DATABASE (single source of truth).
Combines cached LLM results with ground truth labels from input CSV files.

Key Functions:
- load_and_validate_results(): Load from cache + join with ground truth
- Data validation and error checking for analysis pipeline

CRITICAL: This module ALWAYS loads LLM results from cache, never from CSV files.
CSV files in data/model_outputs/ are for provenance only and are NOT data sources.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from .metrics_calculator import determine_multiclass_labels
from cache.result_cache import ResultCache
from cache.result_cache_v2 import ResultCacheV2
from orchestration.experiment_manager import ExperimentConfig, ModelConfig, PromptConfig
from config.utils import load_system_prompt
from config.experiment_config import get_experiment_config
from config.constants import MODEL_SPECIFIC_PROMPT_SUFFIXES
from utilities.category_validator import validate_prompt_config_match, CategoryValidationError


def detect_cache_version(cache_dir: str) -> int:
    """Detect if cache directory is V1 or V2 format.
    
    V2 caches have model_files table, V1 caches don't.
    
    Returns:
        1 for V1 cache, 2 for V2 cache
    """
    import sqlite3
    db_path = Path(cache_dir) / "results.db"
    
    if not db_path.exists():
        # Default to V1 for backwards compatibility
        return 1
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='model_files'"
            )
            if cursor.fetchone():
                return 2
            return 1
    except Exception:
        return 1


def get_prompt_suffix(model_family: str) -> Optional[str]:
    """Get model-specific prompt suffix."""
    if not model_family:
        return None
    model_family = model_family.lower()
    if model_family in MODEL_SPECIFIC_PROMPT_SUFFIXES:
        return MODEL_SPECIFIC_PROMPT_SUFFIXES[model_family]
    for prefix, suffix in MODEL_SPECIFIC_PROMPT_SUFFIXES.items():
        if model_family.startswith(prefix):
            return suffix
    return None


def load_experiment_results(input_data_path: str, prompt_file_path: str, 
                           model_families: dict, experiment_type: str,
                           cache_dir: str = "cache") -> Tuple[pd.DataFrame, List[str]]:
    """
    Load experiment results from CACHE DATABASE (single source of truth).
    Joins cached LLM results with ground truth labels from input CSV.
    
    CRITICAL: This function ALWAYS loads from cache, never from CSV result files.
    
    Args:
        input_data_path: Path to input CSV with ground truth labels
        prompt_file_path: Path to prompt file
        model_families: Dictionary of model family configurations
        experiment_type: 'suicidal_ideation', 'therapy_request', or 'therapy_engagement'
        cache_dir: Path to cache directory
        
    Returns:
        Tuple of (combined_results_df, multiclass_labels_list)
        
    Raises:
        ValueError: If input data not found or cache query fails
    """
    print("Loading experiment results from CACHE...")
    
    # 1. Load input CSV to get input texts + ground truth labels
    input_df = pd.read_csv(input_data_path)
    print(f"Loaded {len(input_df)} inputs from {input_data_path}")
    
    # Determine ground truth column names based on experiment type
    if experiment_type == 'suicidal_ideation':
        ground_truth_col = 'Safety type'  # Note: capitalized in CSV
        # Input CSV uses 'statement' column for text
        text_col = 'statement' if 'statement' in input_df.columns else 'text'
    elif experiment_type == 'therapy_request':
        ground_truth_col = 'therapy_request'
        text_col = 'statement' if 'statement' in input_df.columns else 'text'
    elif experiment_type == 'therapy_engagement':
        ground_truth_col = 'therapy_engagement'
        text_col = 'statement' if 'statement' in input_df.columns else 'conversation'
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    if text_col not in input_df.columns:
        raise ValueError(f"Input CSV must have '{text_col}' column")
    if ground_truth_col not in input_df.columns:
        raise ValueError(f"Input CSV must have '{ground_truth_col}' column for ground truth labels")
    
    # Check for duplicate texts with conflicting ground truth labels
    duplicates = input_df[input_df.duplicated(subset=[text_col], keep=False)]
    if len(duplicates) > 0:
        dup_conflicts = duplicates.groupby(text_col)[ground_truth_col].nunique()
        conflicts = dup_conflicts[dup_conflicts > 1]
        if len(conflicts) > 0:
            conflict_examples = conflicts.head(3).index.tolist()
            raise ValueError(
                f"Found {len(conflicts)} duplicate texts with conflicting ground truth labels! "
                f"Examples: {conflict_examples[:3]}... "
                f"Fix input data before running analysis."
            )
        else:
            print(f"  Note: {len(duplicates)} duplicate texts found, but all have consistent labels.")
    
    input_texts = input_df[text_col].tolist()
    
    # 2. Read prompt file and get prompt name
    prompt_name = Path(prompt_file_path).stem
    print(f"Loaded prompt: {prompt_name}")
    
    # 2.5. Validate prompt categories match config (DESIGN-11 protection)
    print(f"\nValidating prompt categories for {experiment_type}...")
    try:
        analysis_config = get_experiment_config(experiment_type)
        validation_result = validate_prompt_config_match(
            prompt_file_path,
            analysis_config,
            strict=False  # Warn but don't fail for analysis (data might be old)
        )
        
        if not validation_result['valid']:
            print(f"⚠️  WARNING: Category validation issues detected:")
            for mismatch in validation_result['mismatches']:
                print(f"   - {mismatch}")
            print(f"   Continuing analysis, but results may be incorrect if categories don't align.")
        else:
            print(f"✅ Category validation passed")
        
        # Log warnings instead of printing (they're verbose and expected)
        # Warnings are logged for debugging but not shown in console
    except CategoryValidationError as e:
        print(f"⚠️  WARNING: Could not validate categories: {e}")
        print(f"   Continuing analysis, but verify prompt/config alignment manually.")
    print()
    
    # 3. Detect cache version and initialize appropriate cache
    cache_version = detect_cache_version(cache_dir)
    print(f"Using cache V{cache_version} at {cache_dir}")
    
    if cache_version == 2:
        cache = ResultCacheV2(cache_dir, read_only=True)  # Read-only: don't modify database during analysis
    else:
        cache = ResultCache(cache_dir)
    
    stats = cache.get_statistics()
    print(f"Cache contains {stats['unique_entries']} unique entries, {stats['total_results']} total results")
    
    # 4. Query cache for each model
    all_model_results = []
    
    # Load models_config.csv (single source of truth for model metadata)
    from config.experiment_config import load_models_config
    models_config_df = load_models_config()
    
    for family_name, model_sizes in model_families.items():
        for model_size in model_sizes:
            # Look up model info from models_config.csv (single source of truth)
            config_match = models_config_df[
                (models_config_df['family'] == family_name) & 
                (models_config_df['size'] == model_size)
            ]
            
            if len(config_match) == 0:
                # Fail loudly - model must be in config for correct cache lookup
                raise ValueError(
                    f"Model {family_name} {model_size} not found in models_config.csv. "
                    f"Cannot determine correct version for cache lookup. "
                    f"Add this model to the config or remove it from model_families."
                )
            
            model_version = str(config_match.iloc[0]['version'])
            lm_studio_id = config_match.iloc[0].get('lm_studio_id', '')
            
            print(f"Loading cached results for {family_name} {model_size}...")
            
            # Get prompt suffix for this model family
            prompt_suffix = get_prompt_suffix(family_name)
            
            # Query cache for this model
            try:
                if cache_version == 2:
                    # V2 cache uses lm_studio_id (model_key)
                    # V2 expects BASE prompt content (suffix applied separately)
                    if not lm_studio_id:
                        print(f"  WARNING: No lm_studio_id for {family_name} {model_size}, skipping")
                        continue
                    
                    # Read raw prompt content (V2 hashes base prompt, not modified)
                    with open(prompt_file_path, 'r') as f:
                        base_prompt_content = f.read().strip()
                    
                    model_df = cache.load_results_for_analysis(
                        model_key=lm_studio_id,
                        input_texts=input_texts,
                        prompt_content=base_prompt_content,
                        prompt_name=prompt_name,
                        temperature=0.0,
                        max_tokens=256,
                        top_p=1.0,
                        prompt_suffix=prompt_suffix,
                        model_family=family_name,
                        model_size=model_size,
                        model_version=model_version
                    )
                else:
                    # V1 cache uses ExperimentConfig with modified prompt
                    # Create a simple model config object for load_system_prompt
                    class SimpleModelConfig:
                        def __init__(self, family):
                            self.family = family
                    
                    model_config = SimpleModelConfig(family_name)
                    prompt_content = load_system_prompt(prompt_file_path, model_config)
                    
                    model = ModelConfig(
                        family=family_name,
                        size=model_size,
                        version=model_version
                    )
                    
                    prompt = PromptConfig(
                        name=prompt_name,
                        description=f"Prompt: {prompt_name}",
                        file_path=prompt_file_path,
                        version="1.0"
                    )
                    
                    config = ExperimentConfig(
                        experiment_name=f"{family_name}_{model_size}_{prompt_name}_analysis",
                        model=model,
                        prompt=prompt,
                        input_dataset=input_data_path,
                        temperature=0.0,
                        max_tokens=256,
                        top_p=1.0
                    )
                    
                    model_df = cache.load_results_for_analysis(config, input_texts, prompt_content)
                
                if len(model_df) == 0:
                    print(f"  WARNING: No cached results found for {family_name} {model_size}")
                    continue
                
                print(f"  Loaded {len(model_df)} cached results")
                all_model_results.append(model_df)
                
            except Exception as e:
                print(f"  ERROR loading cache for {family_name} {model_size}: {e}")
                continue
    
    if not all_model_results:
        raise ValueError("No cached results found for any models! Run experiments first to populate cache.")
    
    # 5. Combine all model results
    results_df = pd.concat(all_model_results, ignore_index=True)
    print(f"Combined {len(results_df)} total results from {len(all_model_results)} models")
    
    # 5.5. Generate cache manifest for traceability (V2 only)
    cache_manifest = None
    if cache_version == 2 and 'cache_id' in results_df.columns:
        cache_manifest = cache.generate_cache_manifest(results_df)
        print(f"  Cache manifest: {cache_manifest['total_entries']} unique cache entries")
    
    # 6. Join with ground truth labels
    # Create a mapping from input_text to ground truth
    ground_truth_map = dict(zip(input_df[text_col], input_df[ground_truth_col]))
    
    # Add ground truth column based on input_text
    if experiment_type == 'suicidal_ideation':
        results_df['prior_safety_type'] = results_df['input_text'].map(ground_truth_map)
    elif experiment_type == 'therapy_request':
        results_df['prior_therapy_request'] = results_df['input_text'].map(ground_truth_map)
    elif experiment_type == 'therapy_engagement':
        results_df['prior_therapy_engagement'] = results_df['input_text'].map(ground_truth_map)
    
    # Add other prior columns if they exist in input_df
    for col in input_df.columns:
        if col.startswith('prior_') or col in ['therapy_request', 'therapy_engagement', 'safety_type']:
            if col != ground_truth_col and col not in results_df.columns:
                col_map = dict(zip(input_df[text_col], input_df[col]))
                results_df[f'prior_{col}'] = results_df['input_text'].map(col_map)
    
    # Rename input_text to text for consistency with old format
    results_df.rename(columns={'input_text': 'text'}, inplace=True)
    
    # Add row IDs
    results_df.insert(0, 'id', range(1, len(results_df) + 1))
    
    # Determine multi-class label ordering
    multiclass_labels = determine_multiclass_labels(results_df, experiment_type)
    
    print(f"✓ Successfully loaded {len(results_df)} results from cache")
    
    # Store manifest as attribute for later retrieval
    results_df.attrs['cache_manifest'] = cache_manifest
    
    return results_df, multiclass_labels


def save_cache_manifest(results_df: pd.DataFrame, output_path: str) -> bool:
    """
    Save cache manifest to a JSON file for traceability.
    
    Args:
        results_df: DataFrame with cache_manifest in attrs (from load_experiment_results)
        output_path: Path to save the manifest JSON
        
    Returns:
        True if manifest was saved, False if no manifest available
    """
    import json
    
    manifest = results_df.attrs.get('cache_manifest')
    if not manifest:
        return False
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"  Saved cache manifest: {output_path}")
    return True


def get_experiment_result_files(results_pattern: str) -> List[str]:
    """
    DEPRECATED: This function is no longer used.
    Analysis loads directly from cache, not from CSV files.
    Kept for backward compatibility only.
    """
    import glob
    return glob.glob(results_pattern)


def validate_results_dataframe(results_df: pd.DataFrame, experiment_type: str) -> bool:
    """
    Validate that the results DataFrame has required columns for the experiment type.
    
    Args:
        results_df: Combined experiment results DataFrame
        experiment_type: 'suicide_detection', 'therapy_request', or 'therapy_engagement'
        
    Returns:
        True if valid, raises ValueError if invalid
        
    Raises:
        ValueError: If required columns are missing
    """
    required_base_columns = ['model_family', 'model_size', 'status']
    
    if experiment_type == 'therapy_request':
        # Therapy request experiments
        required_columns = required_base_columns + ['prior_therapy_request', 'therapy_request']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {experiment_type}: {missing_columns}")
    elif experiment_type == 'therapy_engagement':
        # Therapy engagement experiments
        required_columns = required_base_columns + ['prior_therapy_engagement', 'therapy_engagement']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {experiment_type}: {missing_columns}")
    else:  # suicidal_ideation
        required_columns = required_base_columns + ['prior_safety_type', 'safety_type']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for {experiment_type}: {missing_columns}")
    
    return True


def load_and_validate_results(input_data_path: str, prompt_file_path: str,
                             model_families: dict, experiment_type: str,
                             cache_dir: str = "cache") -> Tuple[pd.DataFrame, List[str]]:
    """
    Load experiment results from cache and validate they have required columns.
    
    Args:
        input_data_path: Path to input CSV with ground truth labels
        prompt_file_path: Path to prompt file
        model_families: Dictionary of model family configurations
        experiment_type: 'suicidal_ideation', 'therapy_request', or 'therapy_engagement'
        cache_dir: Path to cache directory
        
    Returns:
        Tuple of (validated_results_df, multiclass_labels_list)
        
    Raises:
        ValueError: If no results found or required columns missing
    """
    results_df, multiclass_labels = load_experiment_results(
        input_data_path, prompt_file_path, model_families, experiment_type, cache_dir
    )
    validate_results_dataframe(results_df, experiment_type)
    return results_df, multiclass_labels