#!/usr/bin/env python3
"""
Cache Checker V2 - Check experiment cache status using path-based model identification.

This module checks what percentage of an experiment's results are already cached
in the V2 cache system, allowing the orchestration script to skip model loading
when results are fully cached.

Usage from bash:
    python -m utilities.cache_checker_v2 \
        --model-key "google.gemma-3-270m-it" \
        --prompt-file data/prompts/system_suicide_detection_v2.txt \
        --input-data data/inputs/finalized_input_data/SI_finalized_sentences.csv \
        --cache-dir cache_v2
"""

import argparse
import sys
import logging
from pathlib import Path

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


def check_cache_percentage_v2(
    model_key: str,
    prompt_file: str,
    input_data: str,
    model_family: str = None,
    num_replicates: int = 1,
    cache_dir: str = "cache_v2",
    temperature: float = 0.0,
    max_tokens: int = 256,
    top_p: float = 1.0,
    quiet: bool = False
) -> float:
    """
    Check what percentage of experiment results are already cached in V2 cache.
    
    Args:
        model_key: LM Studio model key (e.g., 'google.gemma-3-270m-it')
        prompt_file: Path to the prompt file
        input_data: Path to the input CSV file
        model_family: Model family for prompt suffix detection (optional)
        num_replicates: Number of replicates per sample
        cache_dir: Path to V2 cache directory
        temperature: Temperature setting
        max_tokens: Max tokens setting
        top_p: Top-p setting
        quiet: If True, suppress progress output
        
    Returns:
        Cache percentage (0.0 to 100.0)
    """
    # Suppress V2 cache logging
    logging.getLogger('cache.result_cache_v2').setLevel(logging.ERROR)
    
    # Load input data
    processor = DataProcessor()
    processed_df = processor.load_input_data(input_data)
    total_samples = len(processed_df)
    
    if total_samples == 0:
        return 100.0  # No samples means "fully cached"
    
    # Initialize V2 cache
    cache = ResultCacheV2(cache_dir)
    
    # Check if model exists in LM Studio
    model_info = cache.get_model_info(model_key)
    if not model_info:
        if not quiet:
            print(f"Model {model_key} not found in LM Studio", file=sys.stderr)
        return 0.0
    
    # Load base prompt content
    with open(prompt_file, 'r') as f:
        base_prompt_content = f.read().strip()
    
    # Get prompt suffix for this model family
    prompt_suffix = get_prompt_suffix(model_family)
    prompt_name = Path(prompt_file).stem
    
    # Check cache for each sample
    total_cached = 0
    
    for idx, row in processed_df.iterrows():
        text = str(row['text'])
        
        # Create cache key
        cache_key = cache.create_cache_key(
            model_key=model_key,
            prompt_content=base_prompt_content,
            input_text=text,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            prompt_name=prompt_name,
            prompt_suffix=prompt_suffix
        )
        
        if cache_key:
            cached_results, missing = cache.check_cache(cache_key, num_replicates)
            total_cached += len(cached_results)
    
    # Calculate percentage
    total_expected = total_samples * num_replicates
    cache_percentage = (total_cached / total_expected) * 100
    
    if not quiet:
        print(f"V2 Cache check: {total_cached}/{total_expected} cached ({cache_percentage:.1f}%)", 
              file=sys.stderr)
    
    return cache_percentage


def main():
    """CLI interface for V2 cache checking."""
    parser = argparse.ArgumentParser(
        description="Check experiment cache status (V2 - path-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model-key', '-m', required=True,
                        help='LM Studio model key (e.g., google.gemma-3-270m-it)')
    parser.add_argument('--model-family', '-f', default=None,
                        help='Model family for prompt suffix (e.g., gemma, qwen)')
    parser.add_argument('--prompt-file', '-p', required=True,
                        help='Path to prompt file')
    parser.add_argument('--input-data', '-i', required=True,
                        help='Path to input CSV file')
    parser.add_argument('--num-replicates', '-n', type=int, default=1,
                        help='Number of replicates (default: 1)')
    parser.add_argument('--cache-dir', default='cache_v2',
                        help='V2 cache directory (default: cache_v2)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress diagnostic output')
    
    args = parser.parse_args()
    
    try:
        percentage = check_cache_percentage_v2(
            model_key=args.model_key,
            prompt_file=args.prompt_file,
            input_data=args.input_data,
            model_family=args.model_family,
            num_replicates=args.num_replicates,
            cache_dir=args.cache_dir,
            quiet=args.quiet
        )
        
        # Print just the percentage to stdout (for bash to capture)
        print(f"{percentage:.1f}")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("0.0")  # Default to 0% on error (forces model loading - safe)
        sys.exit(1)


if __name__ == "__main__":
    main()

