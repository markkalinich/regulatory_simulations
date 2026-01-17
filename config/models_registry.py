#!/usr/bin/env python3
"""
Model Registry - Single source of truth for model configuration.

This module provides:
1. Loading and parsing of models_config.csv
2. Query functions for model families, sizes, and specs
3. CLI interface for bash scripts to call

Usage from Python:
    from config.models_registry import ModelsRegistry
    
    registry = ModelsRegistry()
    families = registry.get_enabled_families()
    sizes = registry.get_family_sizes("gemma")
    spec = registry.get_model_spec("gemma", "270m-it")

Usage from bash:
    # List all enabled families
    python -m config.models_registry --list-families
    
    # List sizes for a family
    python -m config.models_registry --family gemma --list-sizes
    
    # Get model spec (returns version|lm_studio_id)
    python -m config.models_registry --family gemma --size 270m-it --get-spec
    
    # Validate a model exists
    python -m config.models_registry --family gemma --size 270m-it --validate
"""

import csv
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelSpec:
    """Complete specification for a model."""
    family: str
    size: str
    version: str
    lm_studio_id: str
    gemma_generation: str
    model_type: str
    enabled: bool
    param_billions: float
    
    @classmethod
    def from_row(cls, row: Dict[str, str]) -> 'ModelSpec':
        """Create ModelSpec from CSV row dictionary."""
        return cls(
            family=row['family'],
            size=row['size'],
            version=row['version'],
            lm_studio_id=row['lm_studio_id'],
            gemma_generation=row.get('gemma_generation', ''),
            model_type=row.get('model_type', ''),
            enabled=row.get('enabled', 'True').lower() == 'true',
            param_billions=float(row.get('param_billions', 0))
        )
    
    def to_bash_spec(self) -> str:
        """Return version|lm_studio_id format for bash compatibility."""
        return f"{self.version}|{self.lm_studio_id}"


class ModelsRegistry:
    """
    Registry for model configurations loaded from CSV.
    
    This is the single source of truth for model definitions.
    Both Python code and bash scripts should use this class.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the registry.
        
        Args:
            config_path: Path to models_config.csv. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "models_config.csv"
        
        self.config_path = config_path
        self._models: Dict[Tuple[str, str], ModelSpec] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and parse the CSV configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Models config not found: {self.config_path}")
        
        with open(self.config_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                spec = ModelSpec.from_row(row)
                key = (spec.family, spec.size)
                self._models[key] = spec
    
    def get_all_models(self) -> List[ModelSpec]:
        """Get all models (enabled and disabled)."""
        return list(self._models.values())
    
    def get_enabled_models(self) -> List[ModelSpec]:
        """Get only enabled models."""
        return [m for m in self._models.values() if m.enabled]
    
    def get_enabled_families(self) -> List[str]:
        """Get list of unique families that have at least one enabled model."""
        families = set()
        for spec in self._models.values():
            if spec.enabled:
                families.add(spec.family)
        return sorted(families)
    
    def get_family_sizes(self, family: str, enabled_only: bool = True) -> List[str]:
        """
        Get all sizes for a given family.
        
        Args:
            family: Model family name
            enabled_only: If True, only return sizes for enabled models
            
        Returns:
            List of size strings
        """
        sizes = []
        for (fam, size), spec in self._models.items():
            if fam == family:
                if not enabled_only or spec.enabled:
                    sizes.append(size)
        return sizes
    
    def get_model_spec(self, family: str, size: str) -> Optional[ModelSpec]:
        """
        Get the full specification for a model.
        
        Args:
            family: Model family name
            size: Model size identifier
            
        Returns:
            ModelSpec if found, None otherwise
        """
        return self._models.get((family, size))
    
    def validate_model(self, family: str, size: str) -> Tuple[bool, str]:
        """
        Validate that a model exists and is enabled.
        
        Args:
            family: Model family name
            size: Model size identifier
            
        Returns:
            Tuple of (is_valid, message)
        """
        spec = self.get_model_spec(family, size)
        
        if spec is None:
            return False, f"Model {family}:{size} not found in registry"
        
        if not spec.enabled:
            return False, f"Model {family}:{size} exists but is disabled"
        
        return True, f"Model {family}:{size} is valid and enabled"
    
    def get_models_by_type(self, model_type: str, enabled_only: bool = True) -> List[ModelSpec]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: Type to filter by (e.g., 'IT', 'PT', 'Medical', 'Mental Health')
            enabled_only: If True, only return enabled models
            
        Returns:
            List of matching ModelSpec objects
        """
        models = []
        for spec in self._models.values():
            if spec.model_type == model_type:
                if not enabled_only or spec.enabled:
                    models.append(spec)
        return models
    
    def get_statistics(self) -> Dict[str, any]:
        """Get registry statistics."""
        all_models = self.get_all_models()
        enabled = [m for m in all_models if m.enabled]
        
        families = self.get_enabled_families()
        types = set(m.model_type for m in enabled)
        
        return {
            'total_models': len(all_models),
            'enabled_models': len(enabled),
            'disabled_models': len(all_models) - len(enabled),
            'families': len(families),
            'model_types': sorted(types),
            'config_path': str(self.config_path)
        }


def main():
    """CLI interface for bash scripts."""
    parser = argparse.ArgumentParser(
        description="Model Registry - Query model configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all enabled families (space-separated for bash)
  python -m config.models_registry --list-families
  
  # List sizes for a family
  python -m config.models_registry --family gemma --list-sizes
  
  # Get model spec (version|lm_studio_id)
  python -m config.models_registry --family gemma --size 270m-it --get-spec
  
  # Validate a model exists and is enabled
  python -m config.models_registry --family gemma --size 270m-it --validate
  
  # Show registry statistics
  python -m config.models_registry --stats
        """
    )
    
    parser.add_argument('--family', '-f', help='Model family name')
    parser.add_argument('--size', '-s', help='Model size identifier')
    
    # Actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--list-families', action='store_true',
                              help='List all enabled families (space-separated)')
    action_group.add_argument('--list-sizes', action='store_true',
                              help='List sizes for a family (requires --family)')
    action_group.add_argument('--get-spec', action='store_true',
                              help='Get model spec as version|lm_studio_id (requires --family and --size)')
    action_group.add_argument('--validate', action='store_true',
                              help='Validate model exists and is enabled (requires --family and --size)')
    action_group.add_argument('--stats', action='store_true',
                              help='Show registry statistics')
    action_group.add_argument('--list-all', action='store_true',
                              help='List all enabled models as family:size pairs')
    
    args = parser.parse_args()
    
    try:
        registry = ModelsRegistry()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Handle actions
    if args.list_families:
        families = registry.get_enabled_families()
        print(' '.join(families))
        
    elif args.list_sizes:
        if not args.family:
            print("ERROR: --list-sizes requires --family", file=sys.stderr)
            sys.exit(1)
        sizes = registry.get_family_sizes(args.family)
        if not sizes:
            print(f"ERROR: No enabled models found for family '{args.family}'", file=sys.stderr)
            sys.exit(1)
        print(' '.join(sizes))
        
    elif args.get_spec:
        if not args.family or not args.size:
            print("ERROR: --get-spec requires both --family and --size", file=sys.stderr)
            sys.exit(1)
        spec = registry.get_model_spec(args.family, args.size)
        if spec is None:
            print(f"ERROR: Model {args.family}:{args.size} not found", file=sys.stderr)
            sys.exit(1)
        print(spec.to_bash_spec())
        
    elif args.validate:
        if not args.family or not args.size:
            print("ERROR: --validate requires both --family and --size", file=sys.stderr)
            sys.exit(1)
        is_valid, message = registry.validate_model(args.family, args.size)
        print(message)
        sys.exit(0 if is_valid else 1)
        
    elif args.stats:
        stats = registry.get_statistics()
        print(f"Models Registry Statistics")
        print(f"=" * 40)
        print(f"Config file: {stats['config_path']}")
        print(f"Total models: {stats['total_models']}")
        print(f"Enabled: {stats['enabled_models']}")
        print(f"Disabled: {stats['disabled_models']}")
        print(f"Families: {stats['families']}")
        print(f"Model types: {', '.join(stats['model_types'])}")
        
    elif args.list_all:
        models = registry.get_enabled_models()
        for spec in models:
            print(f"{spec.family}:{spec.size}")
            
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
