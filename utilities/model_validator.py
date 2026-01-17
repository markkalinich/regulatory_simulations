#!/usr/bin/env python3
"""
Model Validator - Pre-flight validation for LM Studio model loading.

This module provides validation to ensure the correct model is loaded before
running experiments. It uses the full model path (not just modelKey) to 
eliminate ambiguity from LM Studio's substring matching behavior.

Usage:
    from utilities.model_validator import ModelValidator
    
    validator = ModelValidator()
    
    # Validate before experiment
    if not validator.validate_model(config.model.full_name):
        raise RuntimeError("Model validation failed")
    
    # Get the unambiguous path for loading
    path = validator.get_model_path(config.model.full_name)
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an available model."""
    model_key: str
    path: str
    display_name: str
    publisher: str
    architecture: str
    params: str
    size_bytes: int
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelInfo':
        return cls(
            model_key=data.get('modelKey', data.get('model_key', '')),
            path=data.get('path', ''),
            display_name=data.get('displayName', data.get('display_name', '')),
            publisher=data.get('publisher', ''),
            architecture=data.get('architecture', ''),
            params=data.get('paramsString', data.get('params', '')),
            size_bytes=data.get('sizeBytes', data.get('size_bytes', 0))
        )


class ModelValidationError(Exception):
    """Raised when model validation fails."""
    pass


class ModelNotFoundError(ModelValidationError):
    """Raised when a requested model is not found."""
    pass


class AmbiguousModelError(ModelValidationError):
    """Raised when a model identifier matches multiple models."""
    pass


class ModelValidator:
    """
    Validates model availability and provides unambiguous model paths.
    
    LM Studio uses substring matching for model loading, which can cause
    the wrong model to be loaded. This validator:
    
    1. Checks if a model exists EXACTLY as specified
    2. Detects potential ambiguities (multiple substring matches)
    3. Provides the full unambiguous path for loading
    4. Caches inventory for performance
    """
    
    def __init__(self):
        """Initialize the validator."""
        self._inventory: Dict[str, ModelInfo] = {}
        self._path_to_key: Dict[str, str] = {}
        self._loaded = False
    
    def _fetch_inventory_from_lms(self) -> List[Dict]:
        """Fetch model inventory directly from LM Studio via 'lms ls --json'."""
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
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse 'lms ls --json' output: {e}")
    
    def _load_inventory(self, force_refresh: bool = False) -> None:
        """Load the model inventory from LM Studio."""
        if self._loaded and not force_refresh:
            return
        
        data = self._fetch_inventory_from_lms()
        
        self._inventory.clear()
        self._path_to_key.clear()
        
        for model_data in data:
            info = ModelInfo.from_dict(model_data)
            if info.model_key:
                self._inventory[info.model_key] = info
                if info.path:
                    self._path_to_key[info.path] = info.model_key
        
        self._loaded = True
    
    def refresh_inventory(self) -> bool:
        """
        Refresh inventory from LM Studio.
        
        Returns:
            True if refresh succeeded, False otherwise.
        """
        try:
            result = subprocess.run(
                ['lms', 'ls', '--json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return False
            
            # Save to file
            inventory_path = self._find_inventory_path()
            with open(inventory_path, 'w') as f:
                f.write(result.stdout)
            
            # Reload
            self._loaded = False
            self._load_inventory()
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Get all available models."""
        self._load_inventory()
        return self._inventory.copy()
    
    def find_matches(self, model_id: str) -> List[Tuple[str, str, ModelInfo]]:
        """
        Find all models that could match a given identifier.
        
        This simulates LM Studio's substring matching behavior.
        
        Args:
            model_id: The model identifier to search for.
            
        Returns:
            List of (match_type, model_key, ModelInfo) tuples.
            match_type is one of: 'exact', 'substring', 'superstring'
        """
        self._load_inventory()
        matches = []
        model_id_lower = model_id.lower()
        
        for key, info in self._inventory.items():
            key_lower = key.lower()
            
            if model_id_lower == key_lower:
                matches.append(('exact', key, info))
            elif model_id_lower in key_lower:
                matches.append(('substring', key, info))
            elif key_lower in model_id_lower:
                matches.append(('superstring', key, info))
        
        return matches
    
    def validate_model(self, model_id: str, strict: bool = True) -> Tuple[bool, Optional[ModelInfo], List[str]]:
        """
        Validate that a model can be loaded unambiguously.
        
        Args:
            model_id: The model identifier from config.
            strict: If True, fail on any ambiguity. If False, allow exact matches
                   even if other substring matches exist.
        
        Returns:
            Tuple of (is_valid, model_info, warnings)
            - is_valid: True if model can be loaded safely
            - model_info: ModelInfo if found, None otherwise
            - warnings: List of warning messages
        """
        self._load_inventory()
        
        matches = self.find_matches(model_id)
        warnings = []
        
        if not matches:
            return False, None, [f"Model '{model_id}' not found in LM Studio inventory"]
        
        exact_matches = [m for m in matches if m[0] == 'exact']
        other_matches = [m for m in matches if m[0] != 'exact']
        
        # Best case: exactly one exact match, no ambiguity
        if len(exact_matches) == 1 and not other_matches:
            return True, exact_matches[0][2], []
        
        # Has exact match but also substring matches
        if len(exact_matches) == 1 and other_matches:
            if strict:
                warnings.append(
                    f"Model '{model_id}' has exact match but similar models exist that "
                    f"could be loaded instead: {[m[1] for m in other_matches[:3]]}"
                )
                # Still valid in strict mode if we have exact match
                return True, exact_matches[0][2], warnings
            else:
                return True, exact_matches[0][2], warnings
        
        # No exact match, only partial matches - DANGEROUS
        if not exact_matches and other_matches:
            match_list = [m[1] for m in other_matches[:5]]
            return False, None, [
                f"Model '{model_id}' has NO exact match! "
                f"LM Studio may load any of these instead: {match_list}"
            ]
        
        # Multiple exact matches (shouldn't happen)
        if len(exact_matches) > 1:
            return False, None, [
                f"Model '{model_id}' has multiple exact matches: {[m[1] for m in exact_matches]}"
            ]
        
        return False, None, ["Unknown validation state"]
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """
        Get the unambiguous full path for a model.
        
        The path is unique and should be used for loading to avoid
        any substring matching issues.
        
        Args:
            model_id: The model identifier.
            
        Returns:
            Full path string, or None if not found.
        """
        is_valid, info, _ = self.validate_model(model_id, strict=False)
        if is_valid and info:
            return info.path
        return None
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get full model info for a model identifier."""
        is_valid, info, _ = self.validate_model(model_id, strict=False)
        return info if is_valid else None


def validate_experiment_model(model_full_name: str, fail_on_warning: bool = False) -> Tuple[bool, str, List[str]]:
    """
    Convenience function to validate a model before running an experiment.
    
    Args:
        model_full_name: The model name from ExperimentConfig.model.full_name
        fail_on_warning: If True, treat warnings as failures
        
    Returns:
        Tuple of (success, model_path, messages)
    """
    validator = ModelValidator()
    is_valid, info, warnings = validator.validate_model(model_full_name, strict=True)
    
    messages = []
    
    if not is_valid:
        messages.extend(warnings)
        return False, "", messages
    
    if warnings:
        messages.extend(warnings)
        if fail_on_warning:
            return False, "", messages
    
    return True, info.path if info else "", messages


def preflight_check(model_id: str) -> bool:
    """
    Run preflight validation check for a model.
    
    Prints results to stdout and returns success status.
    
    Args:
        model_id: Model identifier to validate.
        
    Returns:
        True if validation passed, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"PREFLIGHT MODEL VALIDATION")
    print(f"{'='*60}")
    print(f"Validating: {model_id}")
    
    validator = ModelValidator()
    is_valid, info, warnings = validator.validate_model(model_id, strict=True)
    
    if is_valid:
        print(f"\n✅ VALIDATION PASSED")
        print(f"   Model key: {info.model_key}")
        print(f"   Full path: {info.path}")
        print(f"   Publisher: {info.publisher}")
        print(f"   Architecture: {info.architecture}")
        print(f"   Parameters: {info.params}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS:")
            for w in warnings:
                print(f"   - {w}")
        
        print(f"\n💡 Use this path for unambiguous loading:")
        print(f"   {info.path}")
        
    else:
        print(f"\n❌ VALIDATION FAILED")
        for msg in warnings:
            print(f"   - {msg}")
        
        # Show potential matches
        matches = validator.find_matches(model_id)
        if matches:
            print(f"\n   Potential matches found:")
            for match_type, key, match_info in matches[:5]:
                print(f"     [{match_type}] {key}")
                print(f"              Path: {match_info.path}")
    
    print(f"{'='*60}\n")
    return is_valid


def main():
    """CLI interface for model validation."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Model Validator - Validate and look up LM Studio models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full preflight validation
    python -m utilities.model_validator google.gemma-3-270m-it
    
    # Get just the model path (for bash scripts)
    python -m utilities.model_validator --get-path google.gemma-3-270m-it
    
    # Validate without verbose output
    python -m utilities.model_validator --validate google.gemma-3-270m-it
        """
    )
    
    parser.add_argument('model_id', help='Model identifier to look up')
    parser.add_argument('--get-path', action='store_true',
                        help='Print just the model path (for bash scripts)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate model exists (exit 0 if valid, 1 if not)')
    
    args = parser.parse_args()
    
    validator = ModelValidator()
    
    if args.get_path:
        # Just print the path - for bash scripts
        path = validator.get_model_path(args.model_id)
        if path:
            print(path)
            sys.exit(0)
        else:
            sys.exit(1)
            
    elif args.validate:
        # Quick validation without verbose output
        is_valid, info, warnings = validator.validate_model(args.model_id, strict=True)
        if is_valid:
            print(f"valid:{info.model_key}")
            sys.exit(0)
        else:
            print(f"invalid:{';'.join(warnings)}")
            sys.exit(1)
            
    else:
        # Default: full preflight check
        success = preflight_check(args.model_id)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
