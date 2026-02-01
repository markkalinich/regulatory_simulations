#!/usr/bin/env python3
"""
Quantization Validation Utility

Validates that the currently loaded model in LM Studio matches the expected
quantization from models_config.csv. Called during preflight checks.

Usage:
    python utilities/validate_quantization.py <model_family> <model_size> <model_key>

Examples:
    python utilities/validate_quantization.py llama3.3 70b-q8 meta/llama-3.3-70b
    
Exit codes:
    0: Quantization matches expected
    1: Quantization mismatch (wrong quant loaded)
    2: Configuration error (model not found in config, LM Studio unavailable, etc.)
"""

import sys
import pandas as pd
from pathlib import Path
from cache.result_cache_v2 import get_lm_studio_models


def get_expected_quantization(model_family: str, model_size: str) -> str:
    """Load expected quantization from models_config.csv."""
    config_path = Path(__file__).parent.parent / 'config' / 'models_config.csv'
    
    if not config_path.exists():
        print(f"❌ ERROR: models_config.csv not found at {config_path}", file=sys.stderr)
        sys.exit(2)
    
    df = pd.read_csv(config_path)
    
    # Match by family and size
    mask = (df['family'] == model_family) & (df['size'] == model_size)
    matches = df[mask]
    
    if len(matches) == 0:
        print(f"❌ ERROR: No config found for {model_family}:{model_size}", file=sys.stderr)
        sys.exit(2)
    
    if len(matches) > 1:
        print(f"⚠️  WARNING: Multiple configs found for {model_family}:{model_size}, using first", file=sys.stderr)
    
    quant = matches.iloc[0]['quantization']
    
    if pd.isna(quant):
        print(f"⚠️  WARNING: No quantization specified for {model_family}:{model_size}", file=sys.stderr)
        return None
    
    return quant


def get_actual_quantization(model_key: str) -> str:
    """Get currently loaded quantization from LM Studio."""
    try:
        models = get_lm_studio_models()
    except Exception as e:
        print(f"❌ ERROR: Failed to query LM Studio: {e}", file=sys.stderr)
        sys.exit(2)
    
    if model_key not in models:
        print(f"❌ ERROR: Model {model_key} not found in LM Studio", file=sys.stderr)
        sys.exit(2)
    
    model_info = models[model_key]
    quant = model_info.get('quantization', {})
    
    if isinstance(quant, dict):
        quant_name = quant.get('name')
    else:
        quant_name = None
    
    if not quant_name:
        print(f"⚠️  WARNING: No quantization info available from LM Studio for {model_key}", file=sys.stderr)
        return None
    
    return quant_name


def main():
    if len(sys.argv) != 4:
        print("Usage: validate_quantization.py <model_family> <model_size> <model_key>", file=sys.stderr)
        print("Example: validate_quantization.py llama3.3 70b-q8 meta/llama-3.3-70b", file=sys.stderr)
        sys.exit(2)
    
    model_family = sys.argv[1]
    model_size = sys.argv[2]
    model_key = sys.argv[3]
    
    # Get expected quantization from config
    expected_quant = get_expected_quantization(model_family, model_size)
    
    # Get actual quantization from LM Studio
    actual_quant = get_actual_quantization(model_key)
    
    # If no expected quantization configured, skip validation
    if expected_quant is None:
        print(f"  ⚠️  No quantization configured for {model_family}:{model_size} - skipping validation")
        sys.exit(0)
    
    # If LM Studio doesn't report quantization, warn but allow
    if actual_quant is None:
        print(f"  ⚠️  LM Studio did not report quantization - cannot validate")
        sys.exit(0)
    
    # Validate match
    if expected_quant != actual_quant:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"❌ QUANTIZATION MISMATCH DETECTED", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        print(f"  Model:            {model_family}:{model_size}", file=sys.stderr)
        print(f"  LM Studio Key:    {model_key}", file=sys.stderr)
        print(f"  Expected:         {expected_quant}", file=sys.stderr)
        print(f"  Actually Loaded:  {actual_quant}", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"  Please load the correct quantization in LM Studio before running.", file=sys.stderr)
        print(f"  This prevents accidentally caching results from the wrong quantization.", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        print(f"", file=sys.stderr)
        sys.exit(1)
    
    # Success
    print(f"  ✓ Quantization verified: {actual_quant} (as expected)")
    sys.exit(0)


if __name__ == "__main__":
    main()
