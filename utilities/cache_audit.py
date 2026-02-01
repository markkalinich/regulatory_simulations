#!/usr/bin/env python3
"""
Lightweight cache audit script for regulatory_paper_cache_v3.

Verifies for each model:
1. Full name and quantization
2. Total and successful entries for SI, TR, TE tasks
3. Entries with non-standard parameters
"""

import sqlite3
import pandas as pd
from pathlib import Path
import sys

# Expected parameters from config/regulatory_paper_parameters.py
EXPECTED_TEMP = 0.0
EXPECTED_MAX_TOKENS = 256
EXPECTED_TOP_P = 1.0

# Prompt hash to task mapping (from database inspection)
PROMPT_TASKS = {
    'a90ba1cf384df10ba115584f7199cf4c22ff7c6c24b9495d523066db424fb189': 'SI',
    '43827a7d351f31c814d71d8f120ee588a7b287e864682bcf4473471def565011': 'TR',
    '07511101a183573b0e920a4712648d696ef93e644a6f29f6cfe6b64daa5564b7': 'TE'
}

def audit_cache(cache_dir: str, models_config: str, verbose: bool = True):
    """Audit cache database for all models.
    
    Args:
        cache_dir: Path to cache directory
        models_config: Path to models config CSV
        verbose: If True, print detailed output. If False, only return dataframe.
    
    Returns:
        DataFrame with audit results
    """
    
    cache_path = Path(cache_dir)
    db_path = cache_path / "results.db"
    
    if not db_path.exists():
        if verbose:
            print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    # Load expected models
    models_df = pd.read_csv(models_config)
    if verbose:
        print(f"📋 Auditing {len(models_df)} models from {Path(models_config).name}")
        print(f"📁 Cache: {cache_dir}\n")
    
    conn = sqlite3.connect(db_path)
    
    results = []
    
    for _, row in models_df.iterrows():
        # Use lm_studio_id as model_key (this is what's stored in database)
        model_key = row['lm_studio_id']
        model_display = f"{row['family']}/{row['size']}"
        
        # Get model metadata from database
        cursor = conn.execute("""
            SELECT model_path, quantization_name, display_name, format
            FROM model_files
            WHERE model_key = ?
        """, (model_key,))
        
        db_row = cursor.fetchone()
        
        if not db_row:
            if verbose:
                print(f"⚠️  Model not in database: {model_display} ({model_key})")
            results.append({
                'model': model_display,
                'SI_total': 0,
                'SI_success': 0,
                'TR_total': 0,
                'TR_success': 0,
                'TE_total': 0,
                'TE_success': 0,
                'wrong_params': 0,
                'lm_studio_id': model_key,
                'quantization': 'NOT_IN_DB',
                'format': 'N/A',
                'hostnames': 'N/A',
                'model_path': 'N/A'
            })
            continue
        
        model_path, quantization, display_name, format_type = db_row
        
        # Get unique hostnames that created cache entries for this model
        cursor = conn.execute("""
            SELECT DISTINCT ec.hostname
            FROM cached_results cr
            JOIN cache_keys ck ON cr.cache_id = ck.cache_id
            JOIN execution_contexts ec ON cr.context_id = ec.context_id
            WHERE ck.model_path = ?
            ORDER BY ec.hostname
        """, (model_path,))
        hostnames = [row[0] for row in cursor.fetchall()]
        hostnames_str = ', '.join(hostnames) if hostnames else 'N/A'
        
        # Count entries for each task
        task_counts = {}
        for prompt_hash, task in PROMPT_TASKS.items():
            # Total entries for this model+task
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM cache_keys ck
                JOIN cached_results cr ON ck.cache_id = cr.cache_id
                WHERE ck.model_path = ? AND ck.prompt_hash = ?
            """, (model_path, prompt_hash))
            total = cursor.fetchone()[0]
            
            # Successful entries (status_type = 'ok')
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM cache_keys ck
                JOIN cached_results cr ON ck.cache_id = cr.cache_id
                WHERE ck.model_path = ? 
                  AND ck.prompt_hash = ?
                  AND cr.status_type = 'ok'
            """, (model_path, prompt_hash))
            success = cursor.fetchone()[0]
            
            task_counts[task] = {'total': total, 'success': success}
        
        # Count entries with wrong parameters
        cursor = conn.execute("""
            SELECT COUNT(*)
            FROM cache_keys ck
            WHERE ck.model_path = ?
              AND (ck.temperature != ? OR ck.max_tokens != ? OR ck.top_p != ?)
        """, (model_path, EXPECTED_TEMP, EXPECTED_MAX_TOKENS, EXPECTED_TOP_P))
        wrong_params = cursor.fetchone()[0]
        
        results.append({
            'model': model_display,
            'SI_total': task_counts['SI']['total'],
            'SI_success': task_counts['SI']['success'],
            'TR_total': task_counts['TR']['total'],
            'TR_success': task_counts['TR']['success'],
            'TE_total': task_counts['TE']['total'],
            'TE_success': task_counts['TE']['success'],
            'wrong_params': wrong_params,
            'lm_studio_id': model_key,
            'quantization': quantization,
            'format': format_type,
            'hostnames': hostnames_str,
            'model_path': model_path
        })
    
    conn.close()
    
    # Create DataFrame and display
    df = pd.DataFrame(results)
    
    if verbose:
        # Print summary
        print("=" * 150)
        print(f"{'Model':<30} {'SI_Tot':<7} {'SI_OK':<7} {'TR_Tot':<7} {'TR_OK':<7} {'TE_Tot':<7} {'TE_OK':<7} {'BadP':<5} {'Quant':<7} {'Format':<7} {'Hostname(s)':<25}")
        print("=" * 150)
        
        for _, row in df.iterrows():
            print(f"{row['model']:<30} "
                  f"{row['SI_total']:<7} {row['SI_success']:<7} "
                  f"{row['TR_total']:<7} {row['TR_success']:<7} "
                  f"{row['TE_total']:<7} {row['TE_success']:<7} "
                  f"{row['wrong_params']:<5} "
                  f"{row['quantization']:<7} {row['format']:<7} {row['hostnames']:<25}")
        
        print("=" * 150)
        
        # Summary statistics
        print("\n📊 SUMMARY:")
        print(f"   Total models: {len(df)}")
        print(f"   Models in database: {len(df[df['quantization'] != 'NOT_IN_DB'])}")
        print(f"   Models missing from database: {len(df[df['quantization'] == 'NOT_IN_DB'])}")
    
    if verbose:
        # Check for issues
        issues = []
        
        # Check if all models have expected quantization (Q8_0)
        non_q8 = df[df['quantization'] != 'Q8_0']
        if len(non_q8) > 0:
            issues.append(f"   ⚠️  {len(non_q8)} models not Q8_0: {list(non_q8['model'])}")
        
        # Check for missing entries
        for task in ['SI', 'TR', 'TE']:
            total_col = f'{task}_total'
            missing = df[df[total_col] == 0]
            if len(missing) > 0:
                issues.append(f"   ⚠️  {len(missing)} models missing {task} data: {list(missing['model'])}")
        
        # Check for parse failures
        for task in ['SI', 'TR', 'TE']:
            total_col = f'{task}_total'
            success_col = f'{task}_success'
            failed = df[(df[total_col] > 0) & (df[success_col] < df[total_col])]
            if len(failed) > 0:
                for _, row in failed.iterrows():
                    fail_count = row[total_col] - row[success_col]
                    issues.append(f"   ⚠️  {row['model']}: {fail_count} {task} parse failures")
        
        # Check for wrong parameters
        wrong_param_models = df[df['wrong_params'] > 0]
        if len(wrong_param_models) > 0:
            for _, row in wrong_param_models.iterrows():
                issues.append(f"   ⚠️  {row['model']}: {row['wrong_params']} entries with wrong parameters")
        
        if issues:
            print("\n⚠️  ISSUES FOUND:")
            for issue in issues:
                print(issue)
        else:
            print("\n✅ No issues found - all models have complete data with correct parameters!")
        
        # Save detailed report
        output_file = cache_path.parent / "cache_audit_report.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed report saved: {output_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit cache database")
    parser.add_argument(
        '--cache-dir',
        default='regulatory_paper_cache_v3',
        help='Cache directory (default: regulatory_paper_cache_v3)'
    )
    parser.add_argument(
        '--models-config',
        default='config/regulatory_paper_models.csv',
        help='Models config CSV (default: config/regulatory_paper_models.csv)'
    )
    
    args = parser.parse_args()
    
    audit_cache(args.cache_dir, args.models_config)
