#!/usr/bin/env python3
"""
Model Name Collision Detection Script

This script detects potential model name collisions between:
1. The lm_studio_id values in config/models_config.csv 
2. The modelKey values available in LM Studio (queried live via 'lms ls --json')

LM Studio uses substring/partial matching when loading models, which means:
- If you request "gemma-3n-e2b", it may load "gemma-3n-e2b-it" if that's found first
- If you request "llama-3.1-8b", it may load any of several matching models

This can lead to INCORRECT models being loaded and used in experiments!

Usage:
    python utilities/detect_model_collisions.py
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def load_config_models() -> pd.DataFrame:
    """Load models from config/models_config.csv."""
    config_path = Path(__file__).parent.parent / "config" / "models_config.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return pd.read_csv(config_path)


def load_lm_studio_inventory() -> List[Dict]:
    """Load available models directly from LM Studio via 'lms ls --json'."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["lms", "ls", "--json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"lms ls --json failed: {result.stderr}")
        
        return json.loads(result.stdout)
        
    except FileNotFoundError:
        raise RuntimeError("'lms' command not found. Is LM Studio CLI installed?")
    except subprocess.TimeoutExpired:
        raise RuntimeError("'lms ls --json' timed out. Is LM Studio running?")


def extract_model_keys(inventory: List[Dict]) -> Set[str]:
    """Extract all modelKey values from the inventory."""
    keys = set()
    for model in inventory:
        if 'modelKey' in model:
            keys.add(model['modelKey'])
        elif 'model_key' in model:
            keys.add(model['model_key'])
    return keys


def find_partial_matches(config_id: str, inventory_keys: Set[str]) -> List[str]:
    """
    Find models that might match a config_id through partial/substring matching.
    
    LM Studio's matching is case-insensitive and does substring matching,
    so we need to check for models where:
    1. The config_id is a substring of an inventory key
    2. The inventory key is a substring of the config_id
    3. Case-insensitive matches
    """
    matches = []
    config_lower = config_id.lower()
    
    for key in inventory_keys:
        key_lower = key.lower()
        
        # Exact match (case insensitive)
        if config_lower == key_lower:
            matches.append(('exact', key))
        # Config is substring of inventory key (e.g., "gemma-3n-e2b" matches "gemma-3n-e2b-it")
        elif config_lower in key_lower:
            matches.append(('config_in_key', key))
        # Inventory key is substring of config (less common but possible)
        elif key_lower in config_lower:
            matches.append(('key_in_config', key))
        # Check for common basename matching (e.g., stripping publisher prefix)
        elif config_id.split('/')[-1].lower() == key.split('/')[-1].lower():
            matches.append(('basename_match', key))
        elif config_lower.replace('_', '-') == key_lower.replace('_', '-'):
            matches.append(('normalized_match', key))
    
    return matches


def is_pt_it_confusion(config_id: str, other_matches: List[Tuple[str, str]]) -> bool:
    """
    Check if any matches could cause PT/IT model confusion.
    
    This is a HIGH RISK scenario because:
    - PT (pre-trained) and IT (instruction-tuned) models behave very differently
    - Loading IT instead of PT (or vice versa) will corrupt experiment results
    """
    config_lower = config_id.lower()
    config_is_pt = config_lower.endswith('-pt') or '-pt-' in config_lower or not any(
        x in config_lower for x in ['-it', '-instruct', '-chat']
    )
    config_is_it = any(x in config_lower for x in ['-it', '-instruct', '-chat'])
    
    for match_type, match_key in other_matches:
        match_lower = match_key.lower()
        match_is_pt = match_lower.endswith('-pt') or '-pt-' in match_lower
        match_is_it = any(x in match_lower for x in ['-it', '-instruct', '-chat'])
        
        # Check for PT config matching IT model or vice versa
        if config_is_pt and match_is_it:
            return True
        if config_is_it and match_is_pt:
            return True
        
        # Also check for base model matching instruction-tuned variant
        # e.g., "gemma-3-270m" could match "gemma-3-270m-it"
        if config_id in match_key and match_is_it and not config_is_it:
            return True
    
    return False


def detect_collisions() -> Dict[str, any]:
    """
    Main collision detection function.
    
    Returns a report with:
    - exact_matches: Config IDs that exactly match an inventory key (GOOD)
    - no_matches: Config IDs with no matches at all (BAD - model won't load)
    - ambiguous_matches: Config IDs with multiple potential matches (DANGEROUS)
    - partial_matches: Config IDs where only partial matches exist (RISKY)
    - pt_it_confusion: Config IDs that could load wrong PT/IT variant (CRITICAL)
    """
    # Load data
    config_df = load_config_models()
    inventory = load_lm_studio_inventory()
    inventory_keys = extract_model_keys(inventory)
    
    print(f"\n{'='*80}")
    print("MODEL NAME COLLISION DETECTION REPORT")
    print(f"{'='*80}")
    print(f"\nLoaded {len(config_df)} models from config/models_config.csv")
    print(f"Loaded {len(inventory_keys)} models from LM Studio inventory")
    
    # Results tracking
    results = {
        'exact_matches': [],
        'no_matches': [],
        'ambiguous_matches': [],
        'partial_only_matches': [],
        'pt_it_confusion': [],  # NEW: High-risk PT/IT confusion cases
        'disabled_models': [],
    }
    
    # Analyze each enabled config model
    for _, row in config_df.iterrows():
        config_id = row['lm_studio_id']
        family = row['family']
        size = row['size']
        model_type = row.get('model_type', '')
        enabled = row.get('enabled', True)
        
        if not enabled or str(enabled).lower() == 'false':
            results['disabled_models'].append({
                'family': family,
                'size': size,
                'config_id': config_id
            })
            continue
        
        # Find all potential matches
        matches = find_partial_matches(config_id, inventory_keys)
        
        # Categorize the result
        exact = [m for m in matches if m[0] == 'exact']
        partial = [m for m in matches if m[0] != 'exact']
        
        if len(exact) == 1 and len(partial) == 0:
            results['exact_matches'].append({
                'family': family,
                'size': size,
                'config_id': config_id,
                'matched_to': exact[0][1]
            })
        elif len(exact) == 1 and len(partial) > 0:
            # Has exact match but also potential collisions
            # Check for PT/IT confusion - this is HIGH RISK even with exact match
            if is_pt_it_confusion(config_id, partial):
                results['pt_it_confusion'].append({
                    'family': family,
                    'size': size,
                    'model_type': model_type,
                    'config_id': config_id,
                    'exact_match': exact[0][1],
                    'confusing_matches': [(m[0], m[1]) for m in partial],
                    'risk': 'HIGH - PT/IT confusion possible! May load instruction-tuned instead of pre-trained or vice versa!'
                })
            else:
                results['ambiguous_matches'].append({
                    'family': family,
                    'size': size,
                    'config_id': config_id,
                    'exact_match': exact[0][1],
                    'other_matches': [(m[0], m[1]) for m in partial],
                    'risk': 'MEDIUM - exact match exists but other similar models could interfere'
                })
        elif len(exact) == 0 and len(partial) > 0:
            # No exact match, only partial - VERY RISKY
            # Check if this is also PT/IT confusion
            if is_pt_it_confusion(config_id, partial):
                results['pt_it_confusion'].append({
                    'family': family,
                    'size': size,
                    'model_type': model_type,
                    'config_id': config_id,
                    'exact_match': None,
                    'confusing_matches': [(m[0], m[1]) for m in partial],
                    'risk': 'CRITICAL - No exact match AND PT/IT confusion! WILL load wrong model type!'
                })
            else:
                results['partial_only_matches'].append({
                    'family': family,
                    'size': size,
                    'config_id': config_id,
                    'potential_matches': [(m[0], m[1]) for m in partial],
                    'risk': 'HIGH - no exact match, LM Studio may load wrong model!'
                })
        elif len(exact) == 0 and len(partial) == 0:
            # No matches at all
            results['no_matches'].append({
                'family': family,
                'size': size,
                'config_id': config_id,
                'risk': 'CRITICAL - model not found in LM Studio!'
            })
        elif len(exact) > 1:
            # Multiple exact matches (shouldn't happen but check anyway)
            results['ambiguous_matches'].append({
                'family': family,
                'size': size,
                'config_id': config_id,
                'all_exact_matches': [m[1] for m in exact],
                'risk': 'HIGH - multiple exact matches!'
            })
    
    return results


def print_report(results: Dict) -> None:
    """Print a formatted report of collision detection results."""
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Exact matches (SAFE):          {len(results['exact_matches'])}")
    print(f"✗ PT/IT confusion (CRITICAL):    {len(results.get('pt_it_confusion', []))}")
    print(f"⚠ Partial-only matches (HIGH):   {len(results['partial_only_matches'])}")
    print(f"⚠ Ambiguous matches (MEDIUM):    {len(results['ambiguous_matches'])}")
    print(f"✗ No matches (CRITICAL):         {len(results['no_matches'])}")
    print(f"○ Disabled models (skipped):     {len(results['disabled_models'])}")
    
    # CRITICAL: PT/IT Confusion - May load wrong model type!
    if results.get('pt_it_confusion'):
        print(f"\n{'='*80}")
        print("🚨 CRITICAL: PT/IT MODEL CONFUSION DETECTED")
        print(f"{'='*80}")
        print("These models may load the WRONG variant (IT instead of PT or vice versa)!")
        print("This WILL corrupt experiment results - IT and PT models behave differently!\n")
        for item in results['pt_it_confusion']:
            print(f"  • {item['family']}:{item['size']} (configured as: {item.get('model_type', 'unknown')})")
            print(f"    Config ID: {item['config_id']}")
            if item.get('exact_match'):
                print(f"    ✓ Has exact match: {item['exact_match']}")
            else:
                print(f"    ✗ NO exact match!")
            print(f"    ⚠️  Could load these instead:")
            for match_type, match_key in item['confusing_matches'][:3]:
                print(f"       - [{match_type}] {match_key}")
            print()
    
    # Critical issues - No matches
    if results['no_matches']:
        print(f"\n{'='*80}")
        print("❌ CRITICAL: MODELS NOT FOUND IN LM STUDIO")
        print(f"{'='*80}")
        print("These models are in config but don't exist in LM Studio inventory:")
        for item in results['no_matches']:
            print(f"  • {item['family']}:{item['size']}")
            print(f"    Config ID: {item['config_id']}")
            print()
    
    # High risk - Partial only matches
    if results['partial_only_matches']:
        print(f"\n{'='*80}")
        print("⚠️  HIGH RISK: PARTIAL MATCHES ONLY (WRONG MODEL MAY LOAD)")
        print(f"{'='*80}")
        print("These config IDs don't have exact matches and may load incorrect models:")
        for item in results['partial_only_matches']:
            print(f"\n  • {item['family']}:{item['size']}")
            print(f"    Config ID: {item['config_id']}")
            print(f"    Potential matches (may load ANY of these!):")
            for match_type, match_key in item['potential_matches']:
                print(f"      - [{match_type}] {match_key}")
    
    # Medium risk - Ambiguous
    if results['ambiguous_matches']:
        print(f"\n{'='*80}")
        print("⚠️  MEDIUM RISK: AMBIGUOUS MATCHES")
        print(f"{'='*80}")
        for item in results['ambiguous_matches']:
            print(f"\n  • {item['family']}:{item['size']}")
            print(f"    Config ID: {item['config_id']}")
            if 'exact_match' in item:
                print(f"    Exact match: {item['exact_match']}")
                print(f"    Similar models that could interfere:")
                for match_type, match_key in item['other_matches'][:5]:
                    print(f"      - [{match_type}] {match_key}")
    
    # Safe models (condensed)
    print(f"\n{'='*80}")
    print("✅ SAFE: EXACT MATCHES")
    print(f"{'='*80}")
    print(f"These {len(results['exact_matches'])} models have exact matches and are safe.")


def generate_fix_recommendations(results: Dict) -> List[str]:
    """Generate specific recommendations for fixing issues."""
    recommendations = []
    
    recommendations.append("\n" + "="*80)
    recommendations.append("RECOMMENDATIONS TO FIX ISSUES")
    recommendations.append("="*80)
    
    # For partial matches, suggest using exact inventory keys
    if results['partial_only_matches']:
        recommendations.append("\n1. UPDATE CONFIG WITH EXACT MODEL KEYS:")
        recommendations.append("   Edit config/models_config.csv and replace the lm_studio_id values:")
        for item in results['partial_only_matches']:
            recommendations.append(f"\n   {item['family']}:{item['size']}:")
            recommendations.append(f"   Current:  {item['config_id']}")
            if item['potential_matches']:
                # Suggest the most likely correct match
                best_match = item['potential_matches'][0][1]
                recommendations.append(f"   Replace with: {best_match}")
    
    # For no matches
    if results['no_matches']:
        recommendations.append("\n2. DOWNLOAD MISSING MODELS OR REMOVE FROM CONFIG:")
        for item in results['no_matches']:
            recommendations.append(f"   • {item['family']}:{item['size']} - {item['config_id']}")
            recommendations.append(f"     Either: 'lms get {item['config_id']}' or set enabled=False in CSV")
    
    # Code improvements
    recommendations.append("\n3. CODE IMPROVEMENTS:")
    recommendations.append("   a) Add exact match validation before loading models")
    recommendations.append("   b) Query LM Studio's loaded model after loading to verify correct model")
    recommendations.append("   c) Use the 'path' field from inventory instead of modelKey for disambiguation")
    
    return recommendations


def main():
    """Run the collision detection analysis."""
    try:
        results = detect_collisions()
        print_report(results)
        recommendations = generate_fix_recommendations(results)
        for rec in recommendations:
            print(rec)
        
        # Return exit code based on severity
        if results['no_matches'] or results['partial_only_matches']:
            print("\n⚠️  ISSUES FOUND - Please review and fix before running experiments!")
            return 1
        else:
            print("\n✅ No critical issues found.")
            return 0
            
    except Exception as e:
        print(f"\n❌ Error running collision detection: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit(main())
