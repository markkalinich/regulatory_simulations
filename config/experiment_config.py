#!/usr/bin/env python3
"""
Safety Simulation Experiment Configuration Management

Defines configuration classes and parameters for safety simulation experiments.
Provides type-safe configuration management with experiment-specific parameter sets.

Key Classes:
- AnalysisConfiguration: Main configuration container for analysis pipeline
- ExperimentConfig: Experiment-specific parameters (binary categories, models, etc.)
- ModelConfig: Model family and size configuration

Supports suicide detection and therapy request classification experiments
with parameter sets and validation for each type.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd

# Import centralized category definitions
from .regulatory_paper_parameters import (
    BINARY_POSITIVE_CATEGORIES,
    GROUND_TRUTH_POSITIVE_CATEGORIES
)


def load_models_config() -> pd.DataFrame:
    """Load model configuration from CSV (single source of truth)."""
    config_path = Path(__file__).parent / "models_config.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found: {config_path}")
    return pd.read_csv(config_path)


def get_model_families() -> Dict[str, List[str]]:
    """Build families dict from CSV config."""
    df = load_models_config()
    df = df[df.get('enabled', True) != False]  # Only enabled models
    families = {}
    for family in df['family'].unique():
        family_df = df[df['family'] == family]
        families[family] = family_df['size'].tolist()
    return families


@dataclass
class ModelConfiguration:
    """Configuration for model families and sizes."""
    families: Dict[str, List[str]]
    
    @classmethod
    def get_default(cls, models_filter: Optional[Dict[str, List[str]]] = None) -> 'ModelConfiguration':
        """
        Get model configuration, optionally filtered to specific models.
        
        Args:
            models_filter: Optional dict of {family: [sizes]} to analyze.
                          If provided, only these models are included.
                          If None, all enabled models from CSV are included.
        """
        if models_filter:
            return cls(families=models_filter)
        return cls(families=get_model_families())


class ExperimentConfig(ABC):
    """Abstract base class for experiment configurations."""
    
    def __init__(self, models_filter: Optional[Dict[str, List[str]]] = None):
        """
        Initialize experiment config.
        
        Args:
            models_filter: Optional dict of {family: [sizes]} to analyze.
                          If provided, only these models are included.
        """
        self._models_filter = models_filter
    
    @property
    @abstractmethod
    def experiment_type(self) -> str:
        """Experiment type identifier."""
        pass
    
    @property
    @abstractmethod
    def binary_positive_categories(self) -> List[str]:
        """Categories considered positive for binary classification (model outputs)."""
        pass
    
    @property
    def ground_truth_positive_categories(self) -> List[str]:
        """Ground truth categories considered positive. Defaults to binary_positive_categories."""
        return self.binary_positive_categories
    
    @property
    @abstractmethod
    def binary_classification_name(self) -> str:
        """Name for the positive class in binary classification."""
        pass
    
    @property
    @abstractmethod
    def results_pattern(self) -> str:
        """Glob pattern to match experiment result files."""
        pass
    
    @property
    @abstractmethod
    def report_title(self) -> str:
        """Title for the analysis report."""
        pass
    
    @property
    def model_config(self) -> ModelConfiguration:
        """Model configuration (filtered if models_filter was provided)."""
        return ModelConfiguration.get_default(self._models_filter)


class SuicideDetectionConfig(ExperimentConfig):
    """Configuration for suicide detection experiments."""
    
    def __init__(self, positive_categories: Optional[List[str]] = None,
                 models_filter: Optional[Dict[str, List[str]]] = None):
        super().__init__(models_filter)
        # Imported from centralized config (config/regulatory_paper_parameters.py)
        self._positive_categories = positive_categories or \
            BINARY_POSITIVE_CATEGORIES['suicidal_ideation']
    
    @property
    def experiment_type(self) -> str:
        return 'suicidal_ideation'
    
    @property
    def binary_positive_categories(self) -> List[str]:
        """Categories that indicate SI (unified taxonomy for models and ground truth)."""
        return self._positive_categories
    
    @property
    def binary_classification_name(self) -> str:
        return 'SI'
    
    @property
    def results_pattern(self) -> str:
        return "data/model_outputs/*system_suicide_detection*.csv"
    
    @property
    def report_title(self) -> str:
        return "COMPREHENSIVE SUICIDE IDEATION DETECTION ANALYSIS REPORT"


class TherapyRequestConfig(ExperimentConfig):
    """Configuration for therapy request classification experiments."""
    
    def __init__(self, 
                 positive_categories: Optional[List[str]] = None,
                 ground_truth_positive_categories: Optional[List[str]] = None,
                 models_filter: Optional[Dict[str, List[str]]] = None):
        super().__init__(models_filter)
        # Model output categories that indicate a therapy request
        # Imported from centralized config (config/regulatory_paper_parameters.py)
        self._positive_categories = positive_categories or \
            BINARY_POSITIVE_CATEGORIES['therapy_request']
        # Ground truth categories that represent therapy requests
        self._ground_truth_positive_categories = ground_truth_positive_categories or \
            GROUND_TRUTH_POSITIVE_CATEGORIES['therapy_request']
    
    @property
    def experiment_type(self) -> str:
        return 'therapy_request'
    
    @property
    def binary_positive_categories(self) -> List[str]:
        return self._positive_categories
    
    @property
    def ground_truth_positive_categories(self) -> List[str]:
        return self._ground_truth_positive_categories
    
    @property
    def binary_classification_name(self) -> str:
        return 'Therapy Request'
    
    @property
    def results_pattern(self) -> str:
        return "data/model_outputs/*therapy_*_prompt*.csv"
    
    @property
    def report_title(self) -> str:
        return "COMPREHENSIVE THERAPY REQUEST CLASSIFICATION ANALYSIS REPORT"


class TherapyEngagementConfig(ExperimentConfig):
    """Configuration for therapy engagement detection experiments.
    
    This evaluates whether the CHATBOT is engaging in therapeutic behavior
    across a conversation (3-class classification).
    
    Binary classification:
    - Ground truth positive: simulated_therapy (clear therapeutic behavior)
    - Ground truth negative: clear_non_engagement and ambiguous_engagement
    - Prediction positive: simulated_therapy
    - Prediction negative: non_therapeutic and ambiguous_engagement
    
    Multiclass classification:
    - clear_non_engagement: No therapeutic elements (creative writing, general help)
    - ambiguous_engagement: Boundary cases that could be supportive but not therapy
    - simulated_therapy: Clear therapeutic techniques (diagnosis, medication, therapy methods)
    """
    
    def __init__(self, 
                 positive_categories: Optional[List[str]] = None,
                 ground_truth_positive_categories: Optional[List[str]] = None,
                 models_filter: Optional[Dict[str, List[str]]] = None):
        super().__init__(models_filter)
        # Model output categories that indicate therapeutic engagement
        # Imported from centralized config (config/regulatory_paper_parameters.py)
        self._positive_categories = positive_categories or \
            BINARY_POSITIVE_CATEGORIES['therapy_engagement']
        # Ground truth categories that represent therapeutic engagement
        # NOTE: Ground truth uses "clear_engagement" while model outputs "simulated_therapy"
        self._ground_truth_positive_categories = ground_truth_positive_categories or \
            GROUND_TRUTH_POSITIVE_CATEGORIES['therapy_engagement']
    
    @property
    def experiment_type(self) -> str:
        return 'therapy_engagement'
    
    @property
    def binary_positive_categories(self) -> List[str]:
        return self._positive_categories
    
    @property
    def ground_truth_positive_categories(self) -> List[str]:
        return self._ground_truth_positive_categories
    
    @property
    def binary_classification_name(self) -> str:
        return 'Therapy Engagement'
    
    @property
    def results_pattern(self) -> str:
        return "data/model_outputs/*therapy_engagement*.csv"
    
    @property
    def report_title(self) -> str:
        return "COMPREHENSIVE THERAPY ENGAGEMENT DETECTION ANALYSIS REPORT"


@dataclass
class AnalysisConfiguration:
    """Complete configuration for an analysis run."""
    output_dir: Path
    input_data: str
    prompt_file: str
    timestamp: str
    experiment_config: ExperimentConfig
    cache_dir: str = "cache"
    
    @classmethod
    def create(cls, 
               output_dir: str,
               input_data: str, 
               prompt_file: str,
               timestamp: str,
               experiment_type: str,
               models_filter: Optional[Dict[str, List[str]]] = None,
               cache_dir: str = "cache") -> 'AnalysisConfiguration':
        """
        Create analysis configuration from experiment type string.
        
        Args:
            output_dir: Directory for analysis outputs
            input_data: Path to input data file
            prompt_file: Path to prompt file
            timestamp: Analysis timestamp
            experiment_type: Type of experiment ('suicidal_ideation', 'therapy_request', or 'therapy_engagement')
            models_filter: Optional dict of {family: [sizes]} to analyze.
                          If provided, only these models are included in analysis.
            
        Returns:
            AnalysisConfiguration instance
            
        Raises:
            ValueError: If experiment_type is unknown
        """
        if experiment_type == 'suicidal_ideation':
            experiment_config = SuicideDetectionConfig(models_filter=models_filter)
        elif experiment_type == 'therapy_engagement':
            experiment_config = TherapyEngagementConfig(models_filter=models_filter)
        elif experiment_type == 'therapy_request':
            experiment_config = TherapyRequestConfig(models_filter=models_filter)
        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")
        
        return cls(
            output_dir=Path(output_dir),
            input_data=input_data,
            prompt_file=prompt_file,
            timestamp=timestamp,
            experiment_config=experiment_config,
            cache_dir=cache_dir
        )


def get_experiment_config(experiment_type: str, 
                         models_filter: Optional[Dict[str, List[str]]] = None) -> ExperimentConfig:
    """
    Get experiment configuration for the specified type.
    
    Args:
        experiment_type: Type of experiment
        models_filter: Optional dict of {family: [sizes]} to analyze.
        
    Returns:
        ExperimentConfig instance
        
    Raises:
        ValueError: If experiment_type is unknown
    """
    if experiment_type == 'suicidal_ideation':
        return SuicideDetectionConfig(models_filter=models_filter)
    elif experiment_type == 'therapy_engagement':
        return TherapyEngagementConfig(models_filter=models_filter)
    elif experiment_type == 'therapy_request':
        return TherapyRequestConfig(models_filter=models_filter)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")