#!/usr/bin/env python3
"""
Cache Checker - Check experiment cache status without loading models.

This module provides functionality to check what percentage of an experiment's
results are already cached, allowing the orchestration script to skip model
loading when results are fully cached.

Usage from Python:
    from utilities.cache_checker import check_cache_percentage
    
    percentage = check_cache_percentage(
        model_family="gemma",
        model_size="270m-it", 
        model_version="3.0",
        prompt_name="system_suicide_detection_v2",
        prompt_file="data/prompts/system_suicide_detection_v2.txt",
        input_data="data/inputs/finalized_input_data/SI_finalized_sentences.csv",
        num_replicates=1
    )

Usage from bash:
    python -m utilities.cache_checker \\
        --model-family gemma \\
        --model-size 270m-it \\
        --model-version 3.0 \\
        --prompt-name system_suicide_detection_v2 \\
        --prompt-file data/prompts/system_suicide_detection_v2.txt \\
        --input-data data/inputs/finalized_input_data/SI_finalized_sentences.csv \\
        --num-replicates 1
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.result_cache import ResultCache
from orchestration.experiment_manager import ExperimentConfig, ModelConfig, PromptConfig
from orchestration.data_processor import DataProcessor
from config import load_system_prompt


def check_cache_percentage(
    model_family: str,
    model_size: str,
    model_version: str,
    prompt_name: str,
    prompt_file: str,
    input_data: str,
    num_replicates: int = 1,
    cache_dir: str = "cache",
    quiet: bool = False
) -> float:
    """
    Check what percentage of experiment results are already cached.
    
    This allows the orchestration script to skip model loading when
    results are fully cached (100%).
    
    Args:
        model_family: Model family (e.g., 'gemma', 'qwen')
        model_size: Model size (e.g., '270m-it', '4b')
        model_version: Model version (e.g., '3.0')
        prompt_name: Name of the prompt configuration
        prompt_file: Path to the prompt file
        input_data: Path to the input CSV file
        num_replicates: Number of replicates per sample
        cache_dir: Path to cache directory
        quiet: If True, suppress progress output
        
    Returns:
        Cache percentage (0.0 to 100.0)
    """
    # Load and process input data
    processor = DataProcessor()
    processed_df = processor.load_input_data(input_data)
    total_samples = len(processed_df)
    
    if total_samples == 0:
        return 100.0  # No samples means "fully cached" (nothing to do)
    
    # Create experiment config with proper structure
    model_config = ModelConfig(
        family=model_family,
        size=model_size,
        version=model_version
    )
    
    prompt_config = PromptConfig(
        name=prompt_name,
        description=f'Prompt: {prompt_name}',
        file_path=prompt_file,
        version='1.0'
    )
    
    config = ExperimentConfig(
        experiment_name=f'{model_family}_{model_size}_{prompt_name}_analysis',
        model=model_config,
        prompt=prompt_config,
        input_dataset=input_data
    )
    
    # Load prompt content using the same modification logic as run_experiment.py
    prompt_content = load_system_prompt(prompt_file, model_config)
    
    # Check cache for each sample
    cache = ResultCache(cache_dir)
    total_cached = 0
    
    for idx, row in processed_df.iterrows():
        text = str(row['text'])
        cached_results, missing = cache.check_cache(config, text, prompt_content, num_replicates)
        total_cached += len(cached_results)
    
    # Calculate percentage
    total_expected = total_samples * num_replicates
    cache_percentage = (total_cached / total_expected) * 100
    
    if not quiet:
        # Log to stderr so stdout only has the percentage
        print(f"Cache check: {total_cached}/{total_expected} results cached ({cache_percentage:.1f}%)", 
              file=sys.stderr)
    
    return cache_percentage


def main():
    """CLI interface for cache checking."""
    parser = argparse.ArgumentParser(
        description="Check experiment cache status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m utilities.cache_checker \\
        --model-family gemma \\
        --model-size 270m-it \\
        --model-version 3.0 \\
        --prompt-name system_suicide_detection_v2 \\
        --prompt-file data/prompts/system_suicide_detection_v2.txt \\
        --input-data data/inputs/finalized_input_data/SI_finalized_sentences.csv

Output:
    Prints cache percentage (e.g., "85.5") to stdout.
    Prints diagnostic info to stderr (can be suppressed with --quiet).
        """
    )
    
    parser.add_argument('--model-family', '-f', required=True,
                        help='Model family (e.g., gemma, qwen)')
    parser.add_argument('--model-size', '-s', required=True,
                        help='Model size (e.g., 270m-it, 4b)')
    parser.add_argument('--model-version', '-v', required=True,
                        help='Model version (e.g., 3.0)')
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
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress diagnostic output to stderr')
    
    args = parser.parse_args()
    
    try:
        percentage = check_cache_percentage(
            model_family=args.model_family,
            model_size=args.model_size,
            model_version=args.model_version,
            prompt_name=args.prompt_name,
            prompt_file=args.prompt_file,
            input_data=args.input_data,
            num_replicates=args.num_replicates,
            cache_dir=args.cache_dir,
            quiet=args.quiet
        )
        
        # Print just the percentage to stdout (for bash to capture)
        print(f"{percentage:.1f}")
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("0.0")  # Default to 0% on error (forces model loading - safe)
        sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("0.0")  # Default to 0% on error (forces model loading - safe)
        sys.exit(1)


if __name__ == "__main__":
    main()
