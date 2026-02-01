#!/usr/bin/env python3
"""
Pilot test for Cache V2 with path-based model identification.

This script:
1. Tests the new cache structure
2. Verifies cache hits on re-run
3. Confirms different models get different cache keys
"""

import argparse
import time
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cache.result_cache_v2 import ResultCacheV2, CacheKeyV2
from orchestration.api_client import LMStudioClient, LMStudioAPIError
from orchestration.data_processor import DataProcessor
from config import load_system_prompt, EvaluationConfig
from config.constants import MODEL_SPECIFIC_PROMPT_SUFFIXES
from utilities.schemas import extract_first_json_object, validate_and_coerce
import pandas as pd


def get_prompt_suffix(model_family: str) -> str:
    """Get model-specific prompt suffix."""
    if model_family in MODEL_SPECIFIC_PROMPT_SUFFIXES:
        return MODEL_SPECIFIC_PROMPT_SUFFIXES[model_family]
    for prefix, suffix in MODEL_SPECIFIC_PROMPT_SUFFIXES.items():
        if model_family.startswith(prefix):
            return suffix
    return None


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def run_pilot_experiment(
    model_key: str,
    model_family: str,
    input_file: str,
    prompt_file: str,
    cache_dir: str = "cache_v2_test",
    temperature: float = 0.0,
    max_tokens: int = 256,
    top_p: float = 1.0,
    log_file: str = None,
    verbose: bool = False,
):
    """Run pilot experiment with V2 cache.
    
    Returns dict with results or None on failure.
    """
    start_time = datetime.now()
    
    # Setup logging
    logger = logging.getLogger("pilot_test")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler - minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    # File handler - detailed output
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        logger.addHandler(file_handler)
    
    # Initialize V2 cache (suppress its logging)
    logging.getLogger("cache.result_cache_v2").setLevel(logging.WARNING)
    cache = ResultCacheV2(cache_dir)
    
    # Get model info from LM Studio
    model_info = cache.get_model_info(model_key)
    if not model_info:
        logger.error(f"Model {model_key} not found in LM Studio!")
        return None
    
    model_path = model_info.get('path', model_key)
    
    # Log file gets detailed info, console just gets final result
    logger.debug(f"Model: {model_path}")
    logger.debug(f"Input: {input_file}")
    logger.debug(f"Prompt: {prompt_file}")
    logger.debug(f"Cache: {cache_dir}")
    
    # Log file: Detailed info
    logger.debug(f"Model key: {model_key}")
    logger.debug(f"Model info: {json.dumps(model_info, default=str)}")
    logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens}, Top P: {top_p}")
    
    # Load prompt (without model-specific modifications for hashing)
    with open(prompt_file, 'r') as f:
        base_prompt_content = f.read().strip()
    
    # Get model-specific suffix
    prompt_suffix = get_prompt_suffix(model_family)
    actual_prompt = base_prompt_content
    if prompt_suffix:
        actual_prompt = f"{base_prompt_content}\n\n{prompt_suffix}"
        logger.debug(f"Prompt suffix applied: {prompt_suffix}")
    
    # Load input data
    processor = DataProcessor()
    df = processor.load_input_data(input_file)
    total_inputs = len(df)
    
    # Initialize API client
    eval_config = EvaluationConfig(
        model=model_key,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    client = LMStudioClient(eval_config)
    
    # Process each input
    cache_hits = 0
    api_calls = 0
    api_times = []
    errors = []
    parse_failures = 0
    
    for idx, row in df.iterrows():
        input_text = str(row["text"])
        
        # Create cache key
        cache_key = cache.create_cache_key(
            model_key=model_key,
            prompt_content=base_prompt_content,
            input_text=input_text,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            prompt_name=Path(prompt_file).stem,
            prompt_suffix=prompt_suffix
        )
        
        if not cache_key:
            logger.error(f"Could not create cache key for row {idx}")
            continue
        
        # Check cache
        cached_results, missing = cache.check_cache(cache_key, num_replicates=1)
        
        if cached_results:
            cache_hits += 1
            logger.debug(f"Row {idx}: CACHE HIT")
        else:
            # Make API call
            api_calls += 1
            call_start = time.time()
            
            try:
                raw_response = client.call_chat_completion(actual_prompt, input_text)
                processing_time = time.time() - call_start
                api_times.append(processing_time)
                
                # Parse response
                content = client.extract_content(raw_response)
                parsed_json = extract_first_json_object(content) if content else None
                parsed_result = validate_and_coerce(parsed_json) if parsed_json else None
                
                status_type = "ok" if parsed_result else "parse_fail"
                if status_type == "parse_fail":
                    parse_failures += 1
                    logger.debug(f"Row {idx}: PARSE FAIL - {content[:100] if content else 'empty'}")
                
                # Store in cache
                cache.store_result(
                    cache_key=cache_key,
                    input_text=input_text,
                    raw_response=raw_response,
                    parsed_result=parsed_result,
                    status_type=status_type,
                    processing_time=processing_time,
                    prompt_suffix=prompt_suffix
                )
                
                logger.debug(f"Row {idx}: API CALL ({status_type}, {processing_time:.2f}s)")
                
            except LMStudioAPIError as e:
                processing_time = time.time() - call_start
                api_times.append(processing_time)
                error_msg = str(e)[:200]
                errors.append(error_msg)
                
                cache.store_result(
                    cache_key=cache_key,
                    input_text=input_text,
                    raw_response={"error": str(e)},
                    parsed_result=None,
                    status_type="api_error",
                    processing_time=processing_time,
                    error_details=str(e)[:500],
                    prompt_suffix=prompt_suffix
                )
                
                logger.warning(f"Row {idx}: API ERROR - {error_msg}")
        
    # No progress bar - just process silently
    
    # Calculate timing stats
    total_time = (datetime.now() - start_time).total_seconds()
    avg_api_time = sum(api_times) / len(api_times) if api_times else 0
    min_api_time = min(api_times) if api_times else 0
    max_api_time = max(api_times) if api_times else 0
    
    # Results summary
    hit_rate = cache_hits / total_inputs * 100 if total_inputs > 0 else 0
    
    # Console: Single line summary
    status_parts = [f"{cache_hits}/{total_inputs} cached ({hit_rate:.0f}%)"]
    if api_calls > 0:
        status_parts.append(f"{api_calls} API calls")
    if api_times:
        status_parts.append(f"avg {avg_api_time:.1f}s")
    status_parts.append(format_time(total_time))
    if errors:
        status_parts.append(f"{len(errors)} errors")
    
    logger.info(f"  → {' | '.join(status_parts)}")
    
    # Log file: detailed breakdown
    logger.debug(f"API times: avg={avg_api_time:.2f}s, min={min_api_time:.2f}s, max={max_api_time:.2f}s")
    if errors:
        for err in errors[:3]:
            logger.debug(f"Error: {err}")
    if parse_failures:
        logger.debug(f"Parse failures: {parse_failures}")
    
    # Log file: detailed final stats
    logger.debug(f"Final stats - cache_hits={cache_hits}, api_calls={api_calls}, errors={len(errors)}, parse_failures={parse_failures}")
    
    return {
        "model_key": model_key,
        "model_path": model_path,
        "inputs": total_inputs,
        "cache_hits": cache_hits,
        "api_calls": api_calls,
        "cache_hit_rate": hit_rate,
        "errors": len(errors),
        "parse_failures": parse_failures,
        "avg_api_time": avg_api_time,
        "total_time": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Cache V2 Pilot Test")
    parser.add_argument("--model-key", required=True, help="LM Studio model key")
    parser.add_argument("--model-family", required=True, help="Model family (for prompt modifications)")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--prompt", required=True, help="Prompt file")
    parser.add_argument("--cache-dir", default="cache_v2_test", help="Cache directory")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--log-file", default=None, help="Log file for detailed output")
    parser.add_argument("--verbose", action="store_true", help="Verbose console output")
    
    args = parser.parse_args()
    
    result = run_pilot_experiment(
        model_key=args.model_key,
        model_family=args.model_family,
        input_file=args.input,
        prompt_file=args.prompt,
        cache_dir=args.cache_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_file=args.log_file,
        verbose=args.verbose,
    )
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
