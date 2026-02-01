#!/usr/bin/env python3
"""
Batch Cache Checker V2 - Check cache status for all enabled models using V2 cache.

Uses path-based model identification for accurate cache checking.

Usage:
    python -m utilities.batch_cache_checker_v2 \
        --prompt-name system_suicide_detection_v2 \
        --prompt-file data/prompts/system_suicide_detection_v2.txt \
        --input-data data/inputs/finalized_input_data/SI_finalized_sentences.csv \
        --cache-dir cache_v2
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.result_cache_v2 import ResultCacheV2
from orchestration.data_processor import DataProcessor
from config.constants import MODEL_SPECIFIC_PROMPT_SUFFIXES


def get_prompt_suffix(model_family: str) -> str:
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


def check_all_models_cache_v2(
    prompt_name: str,
    prompt_file: str,
    input_data: str,
    num_replicates: int = 1,
    cache_dir: str = "cache_v2",
    config_path: str = "config/models_config.csv",
    models_filter: str = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Check V2 cache status for all enabled models.
    
    Args:
        models_filter: Optional comma-separated list of "family:size" pairs to check.
                      If None, checks all enabled models.
    
    Returns:
        Tuple of (complete_models, incomplete_models)
    """
    # Suppress V2 cache logging
    logging.getLogger('cache.result_cache_v2').setLevel(logging.ERROR)
    
    # Load models config
    models_df = pd.read_csv(config_path)
    enabled_models = models_df[models_df['enabled'] == True]
    
    # Filter to specific models if requested
    if models_filter:
        filter_pairs = set()
        for pair in models_filter.split(','):
            pair = pair.strip()
            if ':' in pair:
                family, size = pair.split(':', 1)
                filter_pairs.add((family.strip(), size.strip()))
        
        if filter_pairs:
            enabled_models = enabled_models[
                enabled_models.apply(
                    lambda row: (row['family'], row['size']) in filter_pairs, 
                    axis=1
                )
            ]
    
    # Load and process input data
    processor = DataProcessor()
    processed_df = processor.load_input_data(input_data)
    total_samples = len(processed_df)
    total_expected = total_samples * num_replicates
    
    if total_samples == 0:
        return [], []
    
    # Initialize V2 cache
    cache = ResultCacheV2(cache_dir)
    
    # Load base prompt content
    with open(prompt_file, 'r') as f:
        base_prompt_content = f.read().strip()
    
    complete = []
    incomplete = []
    
    # Check each enabled model
    for _, model_row in enabled_models.iterrows():
        family = model_row['family']
        size = model_row['size']
        
        # Get the LM Studio model key
        lm_studio_id = model_row.get('lm_studio_id', f'{family}:{size}')
        
        # Check if model exists in LM Studio
        model_info = cache.get_model_info(lm_studio_id)
        if not model_info:
            # Model not in LM Studio - mark as incomplete with 0%
            incomplete.append({
                'family': family,
                'size': size,
                'lm_studio_id': lm_studio_id,
                'cached': 0,
                'total': total_expected,
                'percentage': 0.0,
                'note': 'not in LM Studio'
            })
            continue
        
        # Get prompt suffix for this model family
        prompt_suffix = get_prompt_suffix(family)
        
        # Count cached results for this model
        total_cached = 0
        total_errors = 0
        for idx, row in processed_df.iterrows():
            text = str(row['text'])
            
            # Create cache key
            cache_key = cache.create_cache_key(
                model_key=lm_studio_id,
                prompt_content=base_prompt_content,
                input_text=text,
                temperature=0.0,
                max_tokens=256,
                top_p=1.0,
                prompt_name=prompt_name,
                prompt_suffix=prompt_suffix
            )
            
            if cache_key:
                cached_results, missing = cache.check_cache(cache_key, num_replicates)
                # Count ALL cached results (including errors) for preflight purposes
                total_cached += min(len(cached_results), num_replicates)
                # Track errors separately for data quality reporting
                error_count = sum(1 for r in cached_results if r.status_type in ('api_error', 'error'))
                total_errors += error_count
        
        percentage = (total_cached / total_expected) * 100
        
        model_result = {
            'family': family,
            'size': size,
            'lm_studio_id': lm_studio_id,
            'cached': total_cached,
            'total': total_expected,
            'percentage': percentage,
            'errors': total_errors,
        }
        
        if percentage >= 100.0:
            complete.append(model_result)
        else:
            incomplete.append(model_result)
    
    return complete, incomplete


def main():
    parser = argparse.ArgumentParser(
        description="Check V2 cache status for all enabled models",
    )
    
    parser.add_argument('--prompt-name', '-p', required=True,
                        help='Prompt name')
    parser.add_argument('--prompt-file', required=True,
                        help='Path to prompt file')
    parser.add_argument('--input-data', '-i', required=True,
                        help='Path to input CSV file')
    parser.add_argument('--num-replicates', '-n', type=int, default=1,
                        help='Number of replicates (default: 1)')
    parser.add_argument('--cache-dir', default='cache_v2',
                        help='V2 cache directory (default: cache_v2)')
    parser.add_argument('--config', default='config/models_config.csv',
                        help='Models config CSV path')
    parser.add_argument('--models', '-m', default=None,
                        help='Comma-separated list of "family:size" pairs to check (default: all enabled)')
    parser.add_argument('--json', action='store_true',
                        help='Output in JSON format')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Only output if there are incomplete models')
    
    args = parser.parse_args()
    
    try:
        complete, incomplete = check_all_models_cache_v2(
            prompt_name=args.prompt_name,
            prompt_file=args.prompt_file,
            models_filter=args.models,
            input_data=args.input_data,
            num_replicates=args.num_replicates,
            cache_dir=args.cache_dir,
            config_path=args.config,
        )
        
        if args.json:
            import json
            result = {
                'complete_count': len(complete),
                'incomplete_count': len(incomplete),
                'complete_models': complete,
                'incomplete_models': incomplete,
            }
            print(json.dumps(result, indent=2))
        else:
            total_models = len(complete) + len(incomplete)
            
            if not args.quiet or len(incomplete) > 0:
                print(f"COMPLETE: {len(complete)}/{total_models} models at 100%")
                
                # Show complete models with error counts if any have errors
                complete_with_errors = [m for m in complete if m.get('errors', 0) > 0]
                if complete_with_errors:
                    print()
                    print("Complete models with errors (for data quality review):")
                    for model in complete_with_errors:
                        lm_studio_id = model['lm_studio_id']
                        cached = model['cached']
                        total = model['total']
                        errors = model.get('errors', 0)
                        print(f"  {lm_studio_id:<60} {cached}/{total} ({errors} errors)")
            
            if len(incomplete) > 0:
                print(f"INCOMPLETE: {len(incomplete)} models below 100%")
                print()
                print("Incomplete models:")
                
                # Sort by percentage (lowest first)
                incomplete_sorted = sorted(incomplete, key=lambda x: x['percentage'])
                
                for model in incomplete_sorted:
                    lm_studio_id = model['lm_studio_id']
                    cached = model['cached']
                    total = model['total']
                    pct = model['percentage']
                    errors = model.get('errors', 0)
                    note = model.get('note', '')
                    note_str = f" ({note})" if note else ""
                    error_str = f", {errors} errors" if errors > 0 else ""
                    print(f"  {lm_studio_id:<60} {cached}/{total} ({pct:.1f}%{error_str}){note_str}")
            elif not args.quiet:
                print("All models fully cached!")
        
        sys.exit(0 if len(incomplete) == 0 else 1)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()

