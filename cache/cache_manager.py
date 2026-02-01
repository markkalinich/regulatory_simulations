#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache management utilities for the experiment system.

Simple tools to:
- View cache statistics
- Clear cache entries (with confirmation)
- Query cache contents
"""

import argparse
from pathlib import Path

from cache.result_cache import ResultCache


def print_cache_stats(cache: ResultCache):
    """Print detailed cache statistics."""
    stats = cache.get_statistics()
    
    print("\nCACHE STATISTICS")
    print("=" * 50)
    print(f"Unique cache entries: {stats['unique_entries']:,}")
    print(f"Total results (with replicates): {stats['total_results']:,}")
    print(f"Average replicates: {stats['total_results'] / max(stats['unique_entries'], 1):.2f}")
    print(f"Database size: {stats['database_size_mb']:.2f} MB")
    print(f"Database path: {stats['database_path']}")
    print()


def clear_cache_with_confirmation(cache: ResultCache):
    """Clear all cache entries after confirmation."""
    stats = cache.get_statistics()
    
    print(f"\nCurrent cache contains:")
    print(f"  - {stats['unique_entries']:,} unique cache entries")
    print(f"  - {stats['total_results']:,} total results")
    print(f"  - {stats['database_size_mb']:.2f} MB of data")
    
    confirm = input("\n⚠️  Are you SURE you want to clear ALL cache entries? Type 'YES' to confirm: ").strip()
    
    if confirm == "YES":
        cache.clear_cache(confirm=True)
        print("✓ All cache entries cleared successfully.")
    else:
        print("✗ Cache clear cancelled.")


def show_sample_inputs(cache: ResultCache, limit: int = 10):
    """Show sample cached input texts."""
    inputs = cache.get_all_input_texts(limit=limit)
    
    print(f"\nSAMPLE CACHED INPUTS (showing {len(inputs)} of many):")
    print("=" * 50)
    
    for i, item in enumerate(inputs, 1):
        text = item['input_text']
        # Truncate long texts
        if len(text) > 100:
            text = text[:97] + "..."
        print(f"{i}. {text}")
        print(f"   (Used in {item['config_count']} configs, {item['total_results']} total results)")
    print()


def export_cache_to_csv(cache: ResultCache, output_path: Path):
    """Export all cache data to CSV."""
    print(f"\nExporting cache to {output_path}...")
    
    df = cache.get_all_cached_results_dataframe()
    df.to_csv(output_path, index=False)
    
    print(f"✓ Exported {len(df):,} results to {output_path}")
    print(f"  Columns: {', '.join(df.columns)}")


def main():
    """Main cache management utility."""
    parser = argparse.ArgumentParser(
        description="Cache management utilities for safety simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cache/cache_manager_new.py stats
  python cache/cache_manager_new.py clear
  python cache/cache_manager_new.py sample --limit 20
  python cache/cache_manager_new.py export --output cache_export.csv
        """
    )
    
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Directory containing cache database (default: ./cache)"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)
    
    # Stats command
    subparsers.add_parser('stats', help='Show cache statistics')
    
    # Clear command
    subparsers.add_parser('clear', help='Clear all cache entries (with confirmation)')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Show sample cached inputs')
    sample_parser.add_argument('--limit', type=int, default=10, help='Number of samples to show')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export cache to CSV')
    export_parser.add_argument('--output', default='cache_export.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Initialize cache
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        return 1
    
    cache = ResultCache(str(cache_dir))
    
    # Execute command
    if args.command == 'stats':
        print_cache_stats(cache)
    
    elif args.command == 'clear':
        clear_cache_with_confirmation(cache)
    
    elif args.command == 'sample':
        show_sample_inputs(cache, limit=args.limit)
    
    elif args.command == 'export':
        output_path = Path(args.output)
        export_cache_to_csv(cache, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
