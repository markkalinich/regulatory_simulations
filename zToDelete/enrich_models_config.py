#!/usr/bin/env python3
"""
Enrich models_config.csv with metadata from LM Studio.

This script adds columns from LM Studio's model inventory:
- architecture: Model architecture (llama, gemma, etc.)
- quantization: Quantization level (Q4_K_M, Q8_0, etc.)
- publisher: Model publisher

Usage:
    python -m utilities.enrich_models_config [--dry-run]
    
Options:
    --dry-run    Show what would be added without modifying the CSV
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def fetch_lm_studio_inventory() -> Dict[str, dict]:
    """Fetch model inventory from LM Studio and index by modelKey."""
    try:
        result = subprocess.run(
            ["lms", "ls", "--json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"❌ Error: lms ls --json failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        
        models = json.loads(result.stdout)
        
        # Index by modelKey for fast lookup
        return {m["modelKey"]: m for m in models}
        
    except FileNotFoundError:
        print("❌ Error: 'lms' command not found. Is LM Studio CLI installed?", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("❌ Error: 'lms ls --json' timed out. Is LM Studio running?", file=sys.stderr)
        sys.exit(1)


def enrich_csv(dry_run: bool = False) -> None:
    """Enrich models_config.csv with LM Studio metadata."""
    
    config_path = Path(__file__).parent.parent / "config" / "models_config.csv"
    
    if not config_path.exists():
        print(f"❌ Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)
    
    print(f"📖 Reading {config_path}")
    
    # Read existing CSV
    with open(config_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames.copy()
        rows = list(reader)
    
    print(f"   Found {len(rows)} models in config")
    
    # Fetch LM Studio inventory
    print("🔍 Fetching LM Studio inventory...")
    inventory = fetch_lm_studio_inventory()
    print(f"   Found {len(inventory)} models in LM Studio")
    
    # Define new columns
    new_columns = ['architecture', 'quantization', 'publisher']
    
    # Check which columns already exist
    existing_new = [c for c in new_columns if c in original_fieldnames]
    if existing_new:
        print(f"   ℹ️  Columns already exist: {existing_new}")
    
    # Final fieldnames
    fieldnames = list(original_fieldnames)
    for col in new_columns:
        if col not in fieldnames:
            fieldnames.append(col)
    
    # Enrich rows
    matched = 0
    unmatched = []
    
    for row in rows:
        lm_studio_id = row.get('lm_studio_id', '')
        
        if lm_studio_id in inventory:
            model_info = inventory[lm_studio_id]
            matched += 1
            
            # Add new columns
            row['architecture'] = model_info.get('architecture', '')
            row['publisher'] = model_info.get('publisher', '')
            
            # Handle quantization (nested object)
            quant = model_info.get('quantization', {})
            if isinstance(quant, dict):
                row['quantization'] = quant.get('name', '')
            else:
                row['quantization'] = str(quant) if quant else ''
        else:
            unmatched.append(lm_studio_id)
            # Set empty values for new columns
            for col in new_columns:
                if col not in row:
                    row[col] = ''
    
    print(f"\n📊 Results:")
    print(f"   ✓ Matched: {matched}/{len(rows)}")
    
    if unmatched:
        print(f"   ⚠️  Unmatched ({len(unmatched)}):")
        for model_id in unmatched[:10]:  # Show first 10
            print(f"      - {model_id}")
        if len(unmatched) > 10:
            print(f"      ... and {len(unmatched) - 10} more")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - Would add columns: {[c for c in new_columns if c not in original_fieldnames]}")
        print("   Run without --dry-run to apply changes")
        
        # Show sample
        print("\n   Sample enriched row:")
        sample = next((r for r in rows if r.get('architecture')), rows[0])
        for col in new_columns:
            print(f"      {col}: {sample.get(col, '(empty)')}")
    else:
        # Backup original
        backup_path = config_path.with_suffix('.csv.backup')
        print(f"\n💾 Creating backup: {backup_path}")
        with open(backup_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=original_fieldnames)
            writer.writeheader()
            # Re-read original for backup
            with open(config_path, 'r', newline='') as orig:
                for line in orig:
                    f.write(line)
                    break  # Just header
            with open(config_path, 'r', newline='') as orig:
                reader = csv.DictReader(orig)
                writer = csv.DictWriter(f, fieldnames=original_fieldnames)
                for row in reader:
                    writer.writerow(row)
        
        # Write enriched CSV
        print(f"✍️  Writing enriched CSV: {config_path}")
        with open(config_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✅ Done! Added columns: {[c for c in new_columns if c not in original_fieldnames]}")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich models_config.csv with LM Studio metadata"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be added without modifying the CSV'
    )
    
    args = parser.parse_args()
    enrich_csv(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
