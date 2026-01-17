#!/usr/bin/env python3
"""
File Management Utilities for Safety Simulations

Provides file system operations for safety simulation experiment management.
Handles result file copying, directory creation, and storage of analysis outputs.

Key Functions:
- copy_raw_results(): Archive raw experiment result files
- ensure_directory_structure(): Create output directories
- File pattern matching and bulk operations

Supports storage of experiment results, analysis outputs, and report generation.
"""

import shutil
from pathlib import Path
from typing import List
import glob


def copy_raw_results(results_pattern: str, output_dir: Path) -> int:
    """
    Copy raw experiment result files to analysis directory.
    
    Args:
        results_pattern: Glob pattern to match experiment result CSV files
        output_dir: Base output directory for the analysis
        
    Returns:
        Number of files copied
    """
    print("Copying raw result files...")
    raw_dir = output_dir / 'raw_results'
    raw_dir.mkdir(exist_ok=True)
    
    # Get all relevant experiment CSV files
    result_files = glob.glob(results_pattern)
    
    for file_path in result_files:
        file_name = Path(file_path).name
        dest_path = raw_dir / file_name
        # Copy file with metadata preservation
        shutil.copy2(file_path, dest_path)
    
    print(f"Copied {len(result_files)} raw result files to {raw_dir}")
    return len(result_files)


def ensure_directory_structure(output_dir: Path) -> None:
    """
    Create the complete directory structure for analysis outputs.
    
    Args:
        output_dir: Base output directory for the analysis
    """
    # Create all required subdirectories
    subdirs = [
        'tables',
        'plots', 
        'confusion_matrices/binary',
        'confusion_matrices/multiclass',
        'model_outputs',  # Renamed from raw_results
        'reports'
    ]
    
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def copy_file_safely(source_path: Path, dest_path: Path, create_dirs: bool = True) -> bool:
    """
    Copy a file safely with error handling and optional directory creation.
    
    Args:
        source_path: Path to source file
        dest_path: Path to destination file
        create_dirs: Whether to create destination directories if they don't exist
        
    Returns:
        True if successful, False if failed
    """
    try:
        if create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source_path, dest_path)
        return True
    except Exception as e:
        print(f"Error copying {source_path} to {dest_path}: {e}")
        return False


def get_file_list(pattern: str, sort: bool = True) -> List[str]:
    """
    Get list of files matching a glob pattern.
    
    Args:
        pattern: Glob pattern to match files
        sort: Whether to sort the results
        
    Returns:
        List of file paths matching the pattern
    """
    files = glob.glob(pattern)
    if sort:
        files.sort()
    return files


def clean_directory(directory: Path, pattern: str = "*") -> int:
    """
    Remove files matching pattern from directory.
    
    Args:
        directory: Directory to clean
        pattern: Glob pattern for files to remove (default: all files)
        
    Returns:
        Number of files removed
    """
    if not directory.exists():
        return 0
    
    files_to_remove = list(directory.glob(pattern))
    removed_count = 0
    
    for file_path in files_to_remove:
        try:
            if file_path.is_file():
                file_path.unlink()
                removed_count += 1
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    return removed_count


def ensure_output_directory(output_dir: Path) -> Path:
    """
    Create output directory and return the path.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        Path to the created output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir