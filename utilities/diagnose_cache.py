#!/usr/bin/env python3
"""
Diagnose cache state for all enabled models.
Shows exactly why models might hit GPU.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from cache.result_cache import ResultCache
from orchestration.experiment_manager import ExperimentConfig, ModelConfig, PromptConfig
from orchestration.data_processor import DataProcessor
from config import load_system_prompt

# Configuration
TASKS = [
    ("data/inputs/finalized_input_data/SI_finalized_sentences.csv", "data/prompts/system_suicide_detection_v2.txt", "system_suicide_detection_v2"),
    ("data/inputs/finalized_input_data/therapy_request_finalized_sentences.csv", "data/prompts/therapy_request_classifier_v3.txt", "therapy_request_classifier_v3"),
    ("data/inputs/finalized_input_data/therapy_engagement_finalized_sentences.csv", "data/prompts/therapy_engagement_conversation_prompt_v2.txt", "therapy_engagement_conversation_prompt_v2"),
]
NUM_REPLICATES = 1
CONFIG_PATH = "config/models_config.csv"

def count_cache_status(cache, config, prompt_content, df, num_replicates, include_errors):
    """Count cache status with specific include_errors setting."""
    total_cached = 0
    api_errors = 0
    successes = 0
    
    for idx, row in df.iterrows():
        text = str(row['text'])
        cached_results, missing = cache.check_cache(config, text, prompt_content, num_replicates, include_errors=include_errors)
        total_cached += len(cached_results)
        for result in cached_results:
            if 'api_error' in result.status:
                api_errors += 1
            else:
                successes += 1
    
    return total_cached, successes, api_errors

def main():
    # Load enabled models
    models_df = pd.read_csv(CONFIG_PATH)
    enabled = models_df[models_df['enabled'] == True]
    
    print(f"Checking {len(enabled)} enabled models across {len(TASKS)} tasks")
    print("="*100)
    
    cache = ResultCache("cache")
    processor = DataProcessor()
    
    incomplete_models = []
    
    for input_data, prompt_file, prompt_name in TASKS:
        df = processor.load_input_data(input_data)
        total_expected = len(df) * NUM_REPLICATES
        
        print(f"\n📁 {prompt_name}")
        print(f"   Expected per model: {total_expected} results")
        print("-"*100)
        
        for idx, model_row in enabled.iterrows():
            family = model_row['family']
            size = model_row['size']
            version = str(model_row['version'])
            lmstudio_id = model_row.get('lmstudio_id', f"{family}:{size}")
            
            model_config = ModelConfig(family=family, size=size, version=version)
            prompt_config = PromptConfig(name=prompt_name, description="", file_path=prompt_file, version='1.0')
            config = ExperimentConfig(
                experiment_name=f'{family}_{size}_{prompt_name}_analysis',
                model=model_config, prompt=prompt_config, input_dataset=input_data
            )
            
            prompt_content = load_system_prompt(prompt_file, model_config)
            
            # Check with include_errors=True (what run_experiment.py uses by default)
            total_with_errors, successes, api_errors = count_cache_status(cache, config, prompt_content, df, NUM_REPLICATES, include_errors=True)
            
            # Check with include_errors=False (what would exclude errors)
            total_without_errors, _, _ = count_cache_status(cache, config, prompt_content, df, NUM_REPLICATES, include_errors=False)
            
            pct_with_errors = (total_with_errors / total_expected) * 100 if total_expected > 0 else 100
            pct_without_errors = (total_without_errors / total_expected) * 100 if total_expected > 0 else 100
            
            if pct_with_errors < 100:
                incomplete_models.append({
                    'model': f"{family}:{size}:{version}",
                    'lmstudio_id': lmstudio_id,
                    'task': prompt_name.split('_')[-2].upper(),
                    'pct_with_errors': pct_with_errors,
                    'pct_without_errors': pct_without_errors,
                    'successes': successes,
                    'api_errors': api_errors,
                    'missing': total_expected - total_with_errors
                })
                print(f"   ❌ {family}:{size}:{version}")
                print(f"      With errors: {pct_with_errors:.1f}%, Without: {pct_without_errors:.1f}%")
                print(f"      Successes: {successes}, API Errors: {api_errors}, Missing: {total_expected - total_with_errors}")
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    if incomplete_models:
        print(f"\n⚠️  {len(incomplete_models)} model/task combinations are NOT 100% cached (include_errors=True):")
        print("\nThese WILL hit the GPU when run_experiment.py is called:\n")
        for m in incomplete_models:
            print(f"  - {m['model']} ({m['task']}): {m['pct_with_errors']:.1f}% cached, {m['missing']} missing")
    else:
        print("\n✅ All models are 100% cached (including errors).")
        print("   No models should hit the GPU under normal conditions.")

if __name__ == "__main__":
    main()

