#!/usr/bin/env python3
"""
Shared utilities for facet plot generation.

This module contains common functionality used across all facet plot scripts:
- Model configuration loading
- Metadata helpers
- Color/marker definitions
- Guard model metrics computation from cache
"""

import pandas as pd
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# Configuration Loading
# =============================================================================

def load_models_config() -> Optional[pd.DataFrame]:
    """Load model configuration from CSV (single source of truth).
    
    Note: LM Studio has metadata bugs for some Qwen2 models:
      - qwen2-0.5b-instruct: Reports 5B, actually 0.5B
      - qwen2-1.5b: Reports 7B, actually 1.5B
      - qwen2-1.5b-instruct: Reports 5B, actually 1.5B
    These are manually corrected in models_config.csv.
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "models_config.csv"
    if config_path.exists():
        return pd.read_csv(config_path)
    return None


# Module-level config cache
_MODELS_CONFIG: Optional[pd.DataFrame] = None


def get_models_config() -> Optional[pd.DataFrame]:
    """Get cached models config, loading if necessary."""
    global _MODELS_CONFIG
    if _MODELS_CONFIG is None:
        _MODELS_CONFIG = load_models_config()
    return _MODELS_CONFIG


def get_model_metadata(family: str, size: str) -> Optional[pd.Series]:
    """Get model metadata from CSV config."""
    config = get_models_config()
    if config is None:
        return None
    match = config[(config['family'] == family) & (config['size'] == size)]
    if len(match) > 0:
        return match.iloc[0]
    return None


def get_param_billions_from_config(family: str, size: str) -> float:
    """Get param_billions directly from models_config.csv."""
    config = get_models_config()
    if config is None:
        return float('nan')
    
    match = config[(config['family'] == family) & (config['size'] == size)]
    if len(match) > 0 and pd.notna(match.iloc[0].get('param_billions')):
        return float(match.iloc[0]['param_billions'])
    
    return float('nan')


# =============================================================================
# Color and Marker Definitions
# =============================================================================

# Consistent colors across all plots
# Base models = cool colors (blue/green), Fine-tunes = warm colors (red/orange/purple)
MODEL_TYPE_COLORS = {
    'IT': '#2E86AB',           # Blue (base instruct)
    'PT': '#52B788',           # Sea Green (base pretrain)
    'MedGemma': '#F18F01',     # Orange (medical fine-tune)
    'Medical': '#F18F01',      # Orange (medical fine-tune)
    'ShieldGemma': '#7B2D8E',  # Dark Purple (safety fine-tune)
    'Safety': '#7B2D8E',       # Dark Purple (safety fine-tune)
    'Guard': '#7B2D8E',        # Dark Purple (safety fine-tune)
    'Mental Health': '#C73E1D', # Red (mental health fine-tune)
}

# Consistent markers across all plots
MODEL_TYPE_MARKERS = {
    'IT': 'o',              # Circle
    'PT': 's',              # Square
    'MedGemma': 'D',        # Diamond
    'Medical': 'h',         # Hexagon (or 'D' for diamond)
    'ShieldGemma': '^',     # Triangle up
    'Safety': '^',          # Triangle up
    'Guard': '^',           # Triangle up
    'Mental Health': 'P',   # Plus (filled)
}

# Display labels for legend
MODEL_TYPE_DISPLAY_LABELS = {
    'IT': 'Instruct Tune',
    'PT': 'Base Model',
    'MedGemma': 'MedGemma',
    'Medical': 'Medical',
    'ShieldGemma': 'ShieldGemma',
    'Safety': 'Safety*',     # Asterisk indicates modified parsing
    'Guard': 'Guard*',       # Asterisk indicates modified parsing
    'Mental Health': 'Mental Health',
}


# =============================================================================
# Guard Model Metrics Computation
# =============================================================================

def compute_guard_metrics_from_cache(
    model_family: str,
    sizes: List[str],
    parse_response_fn,
    cache_path: str = None,
    input_data_path: str = None
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compute metrics for guard models using their NATIVE output format.
    
    Guard models output simple formats instead of JSON:
    - ShieldGemma: JSON with is_suicidal/safety_type
    - Llama Guard: "safe" or "unsafe\\nS11"
    - Qwen Guard: "Safety: X\\nCategories: Y"
    
    Args:
        model_family: The model family name (e.g., 'shieldgemma', 'llama_guard', 'qwen_guard')
        sizes: List of model sizes to process
        parse_response_fn: Function to parse response content -> binary prediction (0 or 1, or None to skip)
        cache_path: Path to cache database (defaults to cache/results.db)
        input_data_path: Path to input data CSV (defaults to SI_finalized_sentences.csv)
    
    Returns:
        dict: {(family, size): {metric: value, ...}} for each model
    """
    if cache_path is None:
        cache_path = str(Path(__file__).parent.parent.parent / "cache" / "results.db")
    if input_data_path is None:
        input_data_path = str(Path(__file__).parent.parent.parent / "data" / "inputs" / "finalized_input_data" / "SI_finalized_sentences.csv")
    
    # Load ground truth
    input_df = pd.read_csv(input_data_path)
    si_categories = ['passive_si', 'active_si_no_plan', 'active_si_plan_no_intent', 
                     'active_si_plan_with_intent', 'active_si_plan_with_intent_prep']
    input_df['ground_truth_binary'] = input_df['Safety type'].apply(
        lambda x: 1 if x in si_categories else 0
    )
    
    results = {}
    
    try:
        conn = sqlite3.connect(cache_path)
        
        for size in sizes:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT it.input_text, cr.raw_response
                FROM cached_results cr
                JOIN cache_keys ck ON cr.cache_id = ck.cache_id
                JOIN input_texts it ON ck.input_hash = it.input_hash
                WHERE ck.model_family = ? AND ck.model_size = ?
            """, (model_family, size))
            
            model_results = []
            for input_text, raw_response in cursor.fetchall():
                if raw_response:
                    try:
                        data = json.loads(raw_response)
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        pred_binary = parse_response_fn(content)
                        if pred_binary is not None:
                            model_results.append({
                                'input_text': input_text,
                                'pred_binary': pred_binary
                            })
                    except:
                        pass
            
            if model_results:
                model_df = pd.DataFrame(model_results)
                merged = input_df.merge(model_df, left_on='statement', right_on='input_text', how='inner')
                
                tp = ((merged['pred_binary'] == 1) & (merged['ground_truth_binary'] == 1)).sum()
                tn = ((merged['pred_binary'] == 0) & (merged['ground_truth_binary'] == 0)).sum()
                fp = ((merged['pred_binary'] == 1) & (merged['ground_truth_binary'] == 0)).sum()
                fn = ((merged['pred_binary'] == 0) & (merged['ground_truth_binary'] == 1)).sum()
                
                total = tp + tn + fp + fn
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                accuracy = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                
                results[(model_family, size)] = {
                    'parse_success_rate': 1.0,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                    'total_samples': total,
                    'successful_parses': total
                }
        
        conn.close()
    except Exception as e:
        print(f"Warning: Could not compute guard model metrics from cache: {e}")
    
    return results


# =============================================================================
# Response Parsers for Different Guard Models
# =============================================================================

def parse_shieldgemma_response(content: str) -> Optional[int]:
    """Parse ShieldGemma native response format."""
    try:
        parsed = json.loads(content)
        if 'is_suicidal' in parsed:
            return 1 if parsed['is_suicidal'].lower() == 'yes' else 0
        elif 'safety_type' in parsed:
            st = parsed['safety_type'].lower()
            return 1 if any(x in st for x in ['suicid', 'si_', 'active_si', 'passive_si']) else 0
    except:
        pass
    return None


def parse_llama_guard_response(content: str) -> Optional[int]:
    """Parse Llama Guard native response format: 'safe' or 'unsafe\\nS11'."""
    content = content.strip().lower()
    if content:
        return 0 if content == 'safe' else 1
    return None


def parse_qwen_guard_response(content: str) -> Optional[int]:
    """Parse Qwen Guard native response format: 'Safety: X\\nCategories: Y'."""
    if 'Safety:' in content:
        for line in content.strip().split('\n'):
            if line.startswith('Safety:'):
                safety = line.replace('Safety:', '').strip()
                return 0 if safety == 'Safe' else 1
    return None


# =============================================================================
# Convenience Functions for Computing All Guard Metrics
# =============================================================================

def compute_shieldgemma_metrics(cache_path: str = None, input_data_path: str = None):
    """Compute metrics for ShieldGemma models."""
    return compute_guard_metrics_from_cache(
        model_family='shieldgemma',
        sizes=['2b', '4b-it', '9b', '27b'],
        parse_response_fn=parse_shieldgemma_response,
        cache_path=cache_path,
        input_data_path=input_data_path
    )


def compute_llama_guard_metrics(cache_path: str = None, input_data_path: str = None):
    """Compute metrics for Llama Guard models."""
    return compute_guard_metrics_from_cache(
        model_family='llama_guard',
        sizes=['1b', '8b'],
        parse_response_fn=parse_llama_guard_response,
        cache_path=cache_path,
        input_data_path=input_data_path
    )


def compute_qwen_guard_metrics(cache_path: str = None, input_data_path: str = None):
    """Compute metrics for Qwen Guard models."""
    return compute_guard_metrics_from_cache(
        model_family='qwen_guard',
        sizes=['0.6b', '4b', '8b'],
        parse_response_fn=parse_qwen_guard_response,
        cache_path=cache_path,
        input_data_path=input_data_path
    )


def apply_guard_metrics_to_df(df: pd.DataFrame, guard_metrics: Dict) -> pd.DataFrame:
    """Apply computed guard metrics to a dataframe."""
    for (family, size), metrics in guard_metrics.items():
        mask = (df['model_family'] == family) & (df['model_size'] == size)
        for col, val in metrics.items():
            if col in df.columns:
                df.loc[mask, col] = val
    return df
