#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced configuration and experiment management system for multi-model evaluations.

This system handles:
- Multiple model families (Gemma, LLaMA, Mistral, etc.)
- Multiple model sizes within families
- Multiple prompt variations
- Structured result storage and organization
- Experiment tracking and metadata management
"""

import yaml
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.regulatory_paper_parameters import API_PARAMS


def load_models_config() -> pd.DataFrame:
    """Load model configuration from CSV (single source of truth)."""
    config_path = Path(__file__).parent.parent / "config" / "models_config.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found: {config_path}")
    return pd.read_csv(config_path)


def get_model_name_map() -> Dict[Tuple[str, str, str], str]:
    """Build MODEL_NAME_MAP from CSV config."""
    df = load_models_config()
    result = {}
    for _, row in df.iterrows():
        if row.get('enabled', True):
            # Ensure version is string without trailing .0
            version = str(row['version'])
            if version.endswith('.0'):
                version = version[:-2]
            result[(row['family'], row['size'], version)] = row['lm_studio_id']
    return result


def get_model_metadata(family: str, size: str) -> Dict[str, Any]:
    """Get full metadata for a model from CSV config."""
    df = load_models_config()
    match = df[(df['family'] == family) & (df['size'] == size)]
    if len(match) == 0:
        return {}
    row = match.iloc[0]
    return row.to_dict()


# Load MODEL_NAME_MAP from CSV (single source of truth)
MODEL_NAME_MAP = get_model_name_map()

# Note: MODEL_NAME_MAP is now loaded from config/models_config.csv
# To add a new model, just add a row to that CSV file.


def normalize_version(version: str) -> str:
    """Normalize version string for consistent lookup (e.g., '1.0' -> '1', '3.0' -> '3')."""
    version = str(version)
    if version.endswith('.0'):
        version = version[:-2]
    return version


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    family: str  # e.g., "gemma", "llama", "mistral"
    size: str    # e.g., "270m", "4b", "12b", "27b"
    version: str = "latest"  # e.g., "3", "3.1", "2.5"
    full_name: str = ""  # Full model name as used by LM Studio
    
    def __post_init__(self):
        if not self.full_name:
            # Normalize version for consistent lookup
            normalized_version = normalize_version(self.version)
            
            # Look up the model name in our mapping table
            key = (self.family, self.size, normalized_version)
            if key in MODEL_NAME_MAP:
                self.full_name = MODEL_NAME_MAP[key]
            else:
                # Also try with original version in case it's something like "1.1"
                key_original = (self.family, self.size, self.version)
                if key_original in MODEL_NAME_MAP:
                    self.full_name = MODEL_NAME_MAP[key_original]
                else:
                    # FAIL LOUDLY instead of using a bad fallback
                    # This prevents silent failures with invalid model identifiers
                    available_keys = [k for k in MODEL_NAME_MAP.keys() if k[0] == self.family]
                    raise ValueError(
                        f"Model ({self.family}, {self.size}, {self.version}) not found in MODEL_NAME_MAP. "
                        f"Available models for family '{self.family}': {available_keys}. "
                        f"Please add this model to config/models_config.csv with the correct lm_studio_id."
                    )


@dataclass
class PromptConfig:
    """Configuration for a specific prompt."""
    name: str  # e.g., "system_suicide_detection", "enhanced_safety_prompt"
    description: str
    file_path: str
    version: str = "1.0"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.
    
    API settings imported from config/regulatory_paper_parameters.py for consistency.
    """
    experiment_name: str
    model: ModelConfig
    prompt: PromptConfig
    input_dataset: str
    
    # API settings (defaults from centralized config)
    base_url: str = "http://localhost:1234/v1"
    temperature: float = API_PARAMS['temperature']
    max_tokens: int = API_PARAMS['max_tokens']
    top_p: float = API_PARAMS['top_p']
    request_timeout: int = API_PARAMS['request_timeout']
    request_delay: float = API_PARAMS['request_delay']
    
    # Metadata
    created_at: str = ""
    description: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment ID based on key parameters."""
        key_components = [
            self.model.family,
            self.model.size,
            self.model.version,
            self.prompt.name,
            self.prompt.version,
            Path(self.input_dataset).stem
        ]
        key_string = "_".join(str(c) for c in key_components)
        # Add hash for uniqueness while keeping readability
        hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
        return f"{key_string}_{hash_suffix}"


class ExperimentManager:
    """Manages experiment configurations and result storage."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.configs_dir = self.base_dir / "data" / "model_outputs" / "configs"
        self.results_dir = self.base_dir / "data" / "model_outputs"  # Fixed: was "results"
        
        # Directory creation DISABLED - no longer writing configs or results to data/model_outputs
        # Create directory structure
        # for dir_path in [self.configs_dir, self.results_dir]:
        #     dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_experiment_config(self, 
                               experiment_name: str,
                               model_family: str,
                               model_size: str,
                               model_version: str,
                               prompt_name: str,
                               prompt_file: str,
                               input_dataset: str,
                               description: str = "",
                               **api_kwargs) -> ExperimentConfig:
        """Create a new experiment configuration."""
        
        model = ModelConfig(
            family=model_family,
            size=model_size,
            version=model_version
        )
        
        prompt = PromptConfig(
            name=prompt_name,
            description=f"Prompt configuration: {prompt_name}",
            file_path=prompt_file
        )
        
        config = ExperimentConfig(
            experiment_name=experiment_name,
            model=model,
            prompt=prompt,
            input_dataset=input_dataset,
            description=description,
            **api_kwargs
        )
        
        return config
    
    def save_experiment_config(self, config: ExperimentConfig) -> Path:
        """Save experiment configuration to file."""
        config_file = self.configs_dir / f"{config.get_experiment_id()}.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
        
        return config_file
    
    def load_experiment_config(self, experiment_id: str) -> ExperimentConfig:
        """Load experiment configuration from file."""
        config_file = self.configs_dir / f"{experiment_id}.yaml"
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested objects
        model_dict = config_dict.pop('model')
        prompt_dict = config_dict.pop('prompt')
        
        config = ExperimentConfig(
            model=ModelConfig(**model_dict),
            prompt=PromptConfig(**prompt_dict),
            **config_dict
        )
        
        return config
    
    def get_result_paths(self, config: ExperimentConfig) -> Dict[str, Path]:
        """Get standardized paths for experiment results."""
        exp_id = config.get_experiment_id()
        
        # Directory creation DISABLED - no longer writing results to data/model_outputs
        # Results go directly to data/model_outputs (not in a subdirectory)
        # self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths for analysis and visualizations (created on demand)
        analysis_dir = self.base_dir / "results" / "analysis_reports"
        viz_dir = self.base_dir / "results" / "visualizations"
        
        return {
            'csv': self.results_dir / f"{exp_id}_results.csv",
            'jsonl': self.results_dir / f"{exp_id}_results.jsonl",
            'analysis': analysis_dir / f"{exp_id}_analysis.txt",
            'visualizations': viz_dir / f"{exp_id}_visualizations"
        }
    
    def list_experiments(self) -> List[str]:
        """List all available experiments."""
        config_files = list(self.configs_dir.glob("*.yaml"))
        return [f.stem for f in config_files]
    
    def create_experiment_summary(self) -> pd.DataFrame:
        """Create a summary DataFrame of all experiments."""
        experiments = []
        
        for exp_id in self.list_experiments():
            try:
                config = self.load_experiment_config(exp_id)
                paths = self.get_result_paths(config)
                
                # Check if results exist
                results_exist = paths['csv'].exists()
                
                experiments.append({
                    'experiment_id': exp_id,
                    'experiment_name': config.experiment_name,
                    'model_family': config.model.family,
                    'model_size': config.model.size,
                    'model_version': config.model.version,
                    'prompt_name': config.prompt.name,
                    'input_dataset': Path(config.input_dataset).name,
                    'created_at': config.created_at,
                    'results_exist': results_exist,
                    'description': config.description
                })
            except Exception as e:
                print(f"Error loading experiment {exp_id}: {e}")
        
        return pd.DataFrame(experiments)


def create_model_configs() -> List[ModelConfig]:
    """Create standard model configurations for common models."""
    configs = []
    
    # Gemma models
    for size in ["270m", "4b", "12b", "27b"]:
        configs.append(ModelConfig(
            family="gemma",
            size=size,
            version="3",
            full_name=f"google/gemma-3-{size}"
        ))
    
    # LLaMA models
    for size in ["1b", "3b", "8b", "70b"]:
        configs.append(ModelConfig(
            family="llama",
            size=size,
            version="3.2",
            full_name=f"meta-llama/llama-3.2-{size}"
        ))
    
    # Mistral models
    for size in ["7b", "22b"]:
        configs.append(ModelConfig(
            family="mistral",
            size=size,
            version="0.3",
            full_name=f"mistralai/mistral-{size}-v0.3"
        ))
    
    return configs


def create_prompt_configs(prompts_dir: Path) -> List[PromptConfig]:
    """Create prompt configurations for available prompt files."""
    configs = []
    
    # Standard prompts
    prompt_definitions = [
        {
            "name": "system_suicide_detection",
            "description": "Standard suicide and counseling detection prompt",
            "file": "system_suicide_detection.txt",
            "version": "1.0"
        },
        {
            "name": "enhanced_safety_prompt", 
            "description": "Enhanced safety-focused prompt with additional context",
            "file": "enhanced_safety_prompt.txt",
            "version": "1.0"
        },
        {
            "name": "clinical_screening_prompt",
            "description": "Clinical-style screening prompt",
            "file": "clinical_screening_prompt.txt", 
            "version": "1.0"
        }
    ]
    
    for prompt_def in prompt_definitions:
        file_path = prompts_dir / prompt_def["file"]
        if file_path.exists():
            configs.append(PromptConfig(
                name=prompt_def["name"],
                description=prompt_def["description"],
                file_path=str(file_path),
                version=prompt_def["version"]
            ))
    
    return configs