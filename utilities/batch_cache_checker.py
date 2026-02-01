#!/usr/bin/env python3
"""
Batch Cache Checker - Check cache status for all enabled models at once.

This is more efficient than checking each model individually from bash,
as it loads the cache database once and reuses it.

Counts only valid (non-error) cached results, excluding api_error entries.
This ensures accurate reporting of models that need inference vs those
that are fully cached with successful results.

Usage:
    python -m utilities.batch_cache_checker \
        --prompt-name system_suicide_detection_v2 \
        --prompt-file data/prompts/system_suicide_detection_v2.txt \
        --input-data data/inputs/finalized_input_data/SI_finalized_sentences.csv

Output format (to stdout):
    COMPLETE: 125/127 models at 100%
    INCOMPLETE: 2 models below 100%
    
    Incomplete models:
      llama1:30b                      378/450 (84.0%)
      qwen:7b-medical                 225/450 (50.0%)

Note: Models with api_error entries are counted as incomplete since those
      entries will be retried on the next run.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.result_cache import ResultCache
from orchestration.experiment_manager import ExperimentConfig, ModelConfig, PromptConfig
from orchestration.data_processor import DataProcessor
from config import load_system_prompt


def check_all_models_cache(
    prompt_name: str,
    prompt_file: str,
    input_data: str,
    num_replicates: int = 1,
    cache_dir: str = "cache",
    config_path: str = "config/models_config.csv",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Check cache status for all enabled models.
    
    Uses count_valid_results() to exclude api_error entries from the count,
    ensuring accurate reporting of what work remains to be done.
    
    Args:
        prompt_name: Name of the prompt (e.g., 'system_suicide_detection_v2')
        prompt_file: Path to the prompt file
        input_data: Path to input CSV file
        num_replicates: Number of replicates per sample (default: 1)
        cache_dir: Cache directory (default: 'cache')
        config_path: Path to models config CSV (default: 'config/models_config.csv')
    
    Returns:
        Tuple of (complete_models, incomplete_models)
        Each model dict contains: family, size, version, lm_studio_id, cached, total, percentage
    """
    # Load models config
    models_df = pd.read_csv(config_path)
    enabled_models = models_df[models_df['enabled'] == True]
    
    # Load and process input data
    processor = DataProcessor()
    processed_df = processor.load_input_data(input_data)
    total_samples = len(processed_df)
    total_expected = total_samples * num_replicates
    
    if total_samples == 0:
        return [], []
    
    # Initialize cache
    cache = ResultCache(cache_dir)
    
    complete = []
    incomplete = []
    
    # Check each enabled model
    for _, model_row in enabled_models.iterrows():
        family = model_row['family']
        size = model_row['size']
        version = str(model_row['version'])
        
        # Create config for this model
        model_config = ModelConfig(
            family=family,
            size=size,
            version=version
        )
        
        prompt_config = PromptConfig(
            name=prompt_name,
            description=f'Prompt: {prompt_name}',
            file_path=prompt_file,
            version='1.0'
        )
        
        config = ExperimentConfig(
            experiment_name=f'{family}_{size}_{prompt_name}_analysis',
            model=model_config,
            prompt=prompt_config,
            input_dataset=input_data
        )
        
        # Load prompt content
        prompt_content = load_system_prompt(prompt_file, model_config)
        
        # Count cached results for this model (excluding api errors)
        total_cached = 0
        for idx, row in processed_df.iterrows():
            text = str(row['text'])
            # Use count_valid_results to exclude api_error entries
            valid_count = cache.count_valid_results(config, text, prompt_content)
            total_cached += min(valid_count, num_replicates)
        
        percentage = (total_cached / total_expected) * 100
        
        # Get the full LM Studio model ID
        lm_studio_id = model_row.get('lm_studio_id', f'{family}:{size}')
        
        model_info = {
            'family': family,
            'size': size,
            'version': version,
            'lm_studio_id': lm_studio_id,
            'cached': total_cached,
            'total': total_expected,
            'percentage': percentage,
        }
        
        if percentage >= 100.0:
            complete.append(model_info)
        else:
            incomplete.append(model_info)
    
    return complete, incomplete


def main():
    parser = argparse.ArgumentParser(
        description="Check cache status for all enabled models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--prompt-name', '-p', required=True,
                        help='Prompt name')
    parser.add_argument('--prompt-file', required=True,
                        help='Path to prompt file')
    parser.add_argument('--input-data', '-i', required=True,
                        help='Path to input CSV file')
    parser.add_argument('--num-replicates', '-n', type=int, default=1,
                        help='Number of replicates (default: 1)')
    parser.add_argument('--cache-dir', default='cache',
                        help='Cache directory (default: cache)')
    parser.add_argument('--config', default='config/models_config.csv',
                        help='Models config CSV path')
    parser.add_argument('--json', action='store_true',
                        help='Output in JSON format')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Only output if there are incomplete models')
    
    args = parser.parse_args()
    
    try:
        complete, incomplete = check_all_models_cache(
            prompt_name=args.prompt_name,
            prompt_file=args.prompt_file,
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
                'incomplete_models': incomplete,
            }
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            total_models = len(complete) + len(incomplete)
            
            if not args.quiet or len(incomplete) > 0:
                print(f"COMPLETE: {len(complete)}/{total_models} models at 100%")
            
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
                    print(f"  {lm_studio_id:<60} {cached}/{total} ({pct:.1f}%)")
            elif not args.quiet:
                print("All models fully cached!")
        
        # Exit code: 0 if all complete, 1 if any incomplete
        sys.exit(0 if len(incomplete) == 0 else 1)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

