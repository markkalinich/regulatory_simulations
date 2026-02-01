#!/usr/bin/env python3
"""
LM Studio Manager - Handle model loading/unloading with validation.

This module provides safe model loading for LM Studio with:
- Full path loading to avoid substring matching issues
- Verification that the correct model was loaded
- Automatic unloading of incorrect models

Usage from Python:
    from utilities.lms_manager import ensure_model_loaded, unload_all_models
    
    success = ensure_model_loaded("google.gemma-3-270m-it")
    unload_all_models()

Usage from bash:
    # Load a model safely
    python -m utilities.lms_manager --load google.gemma-3-270m-it
    
    # Unload all models
    python -m utilities.lms_manager --unload-all
    
    # Check what's currently loaded
    python -m utilities.lms_manager --status
"""

import argparse
import subprocess
import sys
import time
from typing import Optional, Tuple

# Add parent directory to path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.model_validator import ModelValidator


def run_lms_command(args: list, timeout: int = 30) -> Tuple[bool, str]:
    """
    Run an lms command and return success status and output.
    
    Args:
        args: Command arguments (e.g., ['ps'] for 'lms ps')
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, output)
    """
    try:
        result = subprocess.run(
            ['lms'] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "lms command not found - is LM Studio CLI installed?"


def get_loaded_model() -> Optional[str]:
    """
    Get the currently loaded model key.
    
    Returns:
        Model key string, or None if no model is loaded.
    """
    success, output = run_lms_command(['ps'])
    if not success:
        return None
    
    # Check for "no models loaded" message
    output_lower = output.lower()
    if 'no models' in output_lower or 'not currently loaded' in output_lower:
        return None
    
    # Parse output - look for model key
    # lms ps output format:
    # IDENTIFIER                  MODEL                       STATUS    SIZE       CONTEXT    TTL
    # llama-3-8b-therapy-model    llama-3-8b-therapy-model    IDLE      8.54 GB    4096
    
    lines = output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip header line (starts with IDENTIFIER or contains column headers)
        line_lower = line.lower()
        if line_lower.startswith('identifier') or 'context' in line_lower and 'ttl' in line_lower:
            continue
        
        # Skip error/instruction lines
        if any(x in line_lower for x in ['error', 'to load', 'lms load', '─', '│']):
            continue
        
        # First word is typically the model key
        parts = line.split()
        if parts and len(parts[0]) > 3:  # Model keys are typically longer than 3 chars
            return parts[0]
    
    return None


def unload_all_models(quiet: bool = False) -> bool:
    """
    Unload all currently loaded models.
    
    Args:
        quiet: If True, suppress output
        
    Returns:
        True if successful (or no models to unload)
    """
    success, output = run_lms_command(['unload', '--all'], timeout=60)
    
    if not quiet:
        if success:
            print("✓ All models unloaded")
        else:
            print(f"⚠️  Could not unload models: {output}")
    
    time.sleep(2)  # Brief pause after unloading
    return success


def ensure_model_loaded(model_name: str, quiet: bool = False) -> Tuple[bool, str]:
    """
    Ensure the specified model is loaded, with validation.
    
    This function:
    1. Checks if the correct model is already loaded
    2. Unloads any incorrect models
    3. Gets the full path for unambiguous loading
    4. Loads the model
    5. Verifies the correct model was loaded
    
    Args:
        model_name: The model key (e.g., "google.gemma-3-270m-it")
        quiet: If True, suppress progress output
        
    Returns:
        Tuple of (success, message)
    """
    def log(msg: str):
        if not quiet:
            print(msg)
    
    # Check if model is already loaded
    loaded_model = get_loaded_model()
    
    if loaded_model:
        # Check for exact match (case-insensitive)
        if loaded_model.lower() == model_name.lower():
            log(f"  ✓ {model_name} already loaded (exact match)")
            return True, "already_loaded"
        
        # Check for substring match (risky)
        if model_name.lower() in loaded_model.lower():
            log(f"  ⚠️  A model containing '{model_name}' is loaded: '{loaded_model}'")
            log("     Unloading to ensure correct model...")
            unload_all_models(quiet=True)
        elif loaded_model.lower() in model_name.lower():
            log(f"  ⚠️  Loaded model '{loaded_model}' is substring of requested '{model_name}'")
            log("     Unloading to ensure correct model...")
            unload_all_models(quiet=True)
        else:
            # Different model loaded
            log(f"  -> Different model loaded ({loaded_model}), unloading...")
            unload_all_models(quiet=True)
    
    # Get the full unambiguous path
    validator = ModelValidator()
    full_path = validator.get_model_path(model_name)
    
    if not full_path:
        msg = f"Model '{model_name}' not found in LM Studio"
        log(f"  ❌ CRITICAL: {msg}")
        log("     Cannot load model without unambiguous path (risk of loading wrong model).")
        log("     To fix:")
        log("       1. Ensure the model is downloaded in LM Studio")
        log("       2. Verify models_config.csv has the correct lm_studio_id")
        return False, msg
    
    log(f"  -> Using full path: {full_path}")
    log("  -> Loading...", )
    
    # Load using full path
    success, output = run_lms_command(['load', full_path], timeout=120)
    
    if not success:
        log(" ✗ failed to load")
        log(f"     Error: {output}")
        log(f"     Try: lms load '{full_path}' manually to see error details.")
        return False, f"Failed to load: {output}"
    
    log(" ✓ loaded")
    time.sleep(2)  # Brief pause after loading
    
    # Verify correct model was loaded
    loaded_after = get_loaded_model()
    
    if not loaded_after:
        msg = "Could not verify loaded model (lms ps returned nothing)"
        log(f"  ❌ CRITICAL: {msg}")
        return False, msg
    
    # Check if loaded model matches (case-insensitive substring)
    if model_name.lower() in loaded_after.lower() or loaded_after.lower() in model_name.lower():
        log("  ✓ Verified: correct model loaded")
        return True, "loaded_successfully"
    else:
        msg = f"Model mismatch! Requested '{model_name}', loaded '{loaded_after}'"
        log(f"  ❌ CRITICAL: {msg}")
        log("     This could produce incorrect results. Aborting to prevent data corruption.")
        log("     Please verify models_config.csv has the correct lm_studio_id for this model.")
        # Unload the wrong model
        unload_all_models(quiet=True)
        return False, msg


def check_lm_studio_running() -> bool:
    """Check if LM Studio is running and responsive."""
    success, _ = run_lms_command(['ps'], timeout=10)
    return success


def main():
    """CLI interface for LM Studio management."""
    parser = argparse.ArgumentParser(
        description="LM Studio Manager - Safe model loading with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load a model safely
    python -m utilities.lms_manager --load google.gemma-3-270m-it
    
    # Unload all models
    python -m utilities.lms_manager --unload-all
    
    # Check status
    python -m utilities.lms_manager --status
    
    # Check if LM Studio is running
    python -m utilities.lms_manager --check
        """
    )
    
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--load', '-l', metavar='MODEL',
                        help='Load a model (with validation)')
    action.add_argument('--unload-all', '-u', action='store_true',
                        help='Unload all models')
    action.add_argument('--status', '-s', action='store_true',
                        help='Show currently loaded model')
    action.add_argument('--check', '-c', action='store_true',
                        help='Check if LM Studio is running')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    if args.load:
        success, msg = ensure_model_loaded(args.load, quiet=args.quiet)
        if not args.quiet:
            if success:
                print(f"Result: {msg}")
        sys.exit(0 if success else 1)
        
    elif args.unload_all:
        success = unload_all_models(quiet=args.quiet)
        sys.exit(0 if success else 1)
        
    elif args.status:
        loaded = get_loaded_model()
        if loaded:
            print(f"Loaded: {loaded}")
        else:
            print("No model loaded")
        sys.exit(0)
        
    elif args.check:
        if check_lm_studio_running():
            print("LM Studio is running")
            sys.exit(0)
        else:
            print("LM Studio is not running or not responding")
            sys.exit(1)


if __name__ == "__main__":
    main()
