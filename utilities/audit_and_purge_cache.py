#!/usr/bin/env python3
"""
Audit cache for PT/IT collision and optionally purge affected entries.

This script checks cached results to see if the model that ACTUALLY ran
(from the API response) matches what was REQUESTED (from config).

Usage:
    python utilities/audit_and_purge_cache.py --audit          # Just check for mismatches
    python utilities/audit_and_purge_cache.py --purge-risky    # Delete all PT/IT collision-risk entries
    python utilities/audit_and_purge_cache.py --purge-mismatched  # Delete only entries with actual mismatches
"""

import sqlite3
import json
import argparse
from pathlib import Path
from datetime import datetime

# Models with PT/IT collision risk
RISKY_MODELS = [
    # Gemma 3 - PT versions that could load IT
    'google.gemma-3-270m',
    'google.gemma-3-1b-pt', 
    'google.gemma-3-4b-pt',
    'google.gemma-3-12b-pt',
    'google.gemma-3-27b-pt',
    # And IT versions that could load PT
    'google.gemma-3-270m-it',
    'google.gemma-3-1b-it',
    'google.gemma-3-4b-it', 
    'google.gemma-3-12b-it',
    'google.gemma-3-27b-it',
    # Gemma 3n
    'gemma-3n-e2b',
    'gemma-3n-e2b-it',
    'gemma-3n-e4b',
    'gemma-3n-e4b-it',
    # Gemma 2 PT/IT pairs
    'gemma-2-2b', 'gemma-2-2b-it',
    'gemma-2-9b', 'gemma-2-9b-it',
    'gemma-2-27b', 'gemma-2-27b-it',
    # Gemma 1 
    'gemma-2b', 'gemma-2b-it',
    # Llama 3.2
    'llama-3.2-1b', 'llama-3.2-1b-instruct',
    'llama-3.2-3b', 'llama-3.2-3b-instruct',
    # Llama 3.1
    'meta-llama-3.1-8b', 'meta-llama-3.1-8b-instruct',
    # Llama 3
    'meta-llama-3-8b', 'meta-llama-3-8b-instruct',
    # Llama 2
    'llama-2-7b', 'llama-2-7b-chat',
    'llama-2-13b', 'llama-2-13b-chat',
    'llama-2-70b', 'llama-2-70b-chat',
    # Qwen 2
    'qwen2-1.5b', 'qwen2-1.5b-instruct',
    'qwen2-7b', 'qwen2-7b-instruct',
]


def is_pt_it_mismatch(requested: str, actual: str) -> bool:
    """Check if there's a PT/IT type mismatch."""
    if actual in ('NO MODEL FIELD', 'ERROR'):
        return False
    
    req_lower = requested.lower()
    act_lower = actual.lower()
    
    # Check if one is IT and other is PT
    req_is_it = any(x in req_lower for x in ['-it', '-instruct', '-chat'])
    act_is_it = any(x in act_lower for x in ['-it', '-instruct', '-chat'])
    
    # Mismatch if one is IT and the other isn't
    return req_is_it != act_is_it


def audit_cache(db_path: str) -> dict:
    """Audit cache for mismatches between requested and actual models."""
    conn = sqlite3.connect(db_path)
    
    results = {
        'total_checked': 0,
        'mismatches': [],
        'by_model': {},
    }
    
    print("=" * 70)
    print("AUDITING CACHE FOR PT/IT COLLISIONS")
    print("=" * 70)
    
    for model in RISKY_MODELS:
        cursor = conn.execute('''
            SELECT ck.cache_id, ck.model_full_name, cr.raw_response, cr.created_at
            FROM cache_keys ck
            JOIN cached_results cr ON ck.cache_id = cr.cache_id
            WHERE ck.model_full_name = ?
        ''', (model,))
        
        model_results = {'total': 0, 'mismatched': 0, 'examples': []}
        
        for row in cursor.fetchall():
            results['total_checked'] += 1
            model_results['total'] += 1
            
            cache_id = row[0]
            requested = row[1]
            created_at = row[3]
            
            try:
                response = json.loads(row[2])
                actual = response.get('model', 'NO MODEL FIELD')
                
                if is_pt_it_mismatch(requested, actual):
                    model_results['mismatched'] += 1
                    mismatch = {
                        'cache_id': cache_id,
                        'requested': requested,
                        'actual': actual,
                        'created_at': created_at,
                    }
                    results['mismatches'].append(mismatch)
                    if len(model_results['examples']) < 3:
                        model_results['examples'].append(mismatch)
            except:
                pass
        
        if model_results['total'] > 0:
            results['by_model'][model] = model_results
            status = "✓" if model_results['mismatched'] == 0 else "❌"
            print(f"  {status} {model}: {model_results['total']} entries, {model_results['mismatched']} mismatched")
    
    conn.close()
    return results


def purge_models(db_path: str, models: list, dry_run: bool = True) -> int:
    """Delete cache entries for specified models."""
    conn = sqlite3.connect(db_path)
    
    total_deleted = 0
    
    for model in models:
        # First count
        cursor = conn.execute('''
            SELECT COUNT(*) FROM cached_results 
            WHERE cache_id IN (SELECT cache_id FROM cache_keys WHERE model_full_name = ?)
        ''', (model,))
        count = cursor.fetchone()[0]
        
        if count > 0:
            if dry_run:
                print(f"  Would delete {count} results for {model}")
            else:
                # Delete results
                conn.execute('''
                    DELETE FROM cached_results 
                    WHERE cache_id IN (SELECT cache_id FROM cache_keys WHERE model_full_name = ?)
                ''', (model,))
                # Delete cache keys
                conn.execute('''
                    DELETE FROM cache_keys WHERE model_full_name = ?
                ''', (model,))
                print(f"  Deleted {count} results for {model}")
            total_deleted += count
    
    if not dry_run:
        conn.commit()
    conn.close()
    
    return total_deleted


def main():
    parser = argparse.ArgumentParser(description='Audit and purge cache for PT/IT collision')
    parser.add_argument('--audit', action='store_true', help='Audit cache for mismatches')
    parser.add_argument('--purge-risky', action='store_true', help='Delete ALL entries for risky models')
    parser.add_argument('--purge-mismatched', action='store_true', help='Delete only mismatched entries')
    parser.add_argument('--confirm', action='store_true', help='Actually perform deletion (default is dry-run)')
    parser.add_argument('--db', default='cache/results.db', help='Path to cache database')
    
    args = parser.parse_args()
    
    if not any([args.audit, args.purge_risky, args.purge_mismatched]):
        args.audit = True  # Default to audit
    
    db_path = args.db
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        return 1
    
    if args.audit or args.purge_mismatched:
        results = audit_cache(db_path)
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Total entries checked: {results['total_checked']:,}")
        print(f"  Entries with PT/IT mismatch: {len(results['mismatches']):,}")
        
        if results['mismatches']:
            print()
            print("Mismatched entries (showing first 10):")
            for m in results['mismatches'][:10]:
                print(f"  - Requested: {m['requested']}")
                print(f"    Actual:    {m['actual']}")
                print(f"    Date:      {m['created_at']}")
                print()
    
    if args.purge_risky:
        print()
        print("=" * 70)
        print("PURGING ALL RISKY MODEL ENTRIES")
        print("=" * 70)
        dry_run = not args.confirm
        if dry_run:
            print("(DRY RUN - add --confirm to actually delete)")
        total = purge_models(db_path, RISKY_MODELS, dry_run=dry_run)
        print()
        action = "Would delete" if dry_run else "Deleted"
        print(f"{action} {total:,} total entries")
        
    elif args.purge_mismatched and results['mismatches']:
        print()
        print("=" * 70)
        print("PURGING MISMATCHED ENTRIES ONLY")
        print("=" * 70)
        dry_run = not args.confirm
        if dry_run:
            print("(DRY RUN - add --confirm to actually delete)")
        
        conn = sqlite3.connect(db_path)
        deleted = 0
        for m in results['mismatches']:
            if dry_run:
                deleted += 1
            else:
                conn.execute('DELETE FROM cached_results WHERE cache_id = ?', (m['cache_id'],))
                deleted += 1
        
        if not dry_run:
            conn.commit()
        conn.close()
        
        action = "Would delete" if dry_run else "Deleted"
        print(f"{action} {deleted:,} mismatched entries")
    
    return 0


if __name__ == '__main__':
    exit(main())
