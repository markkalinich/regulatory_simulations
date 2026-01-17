"""
Type definitions and protocols for the safety simulation system.

This module provides comprehensive type definitions, protocols, and type aliases
for improved type safety and IDE support throughout the analysis pipeline.
"""

from typing import (
    Protocol, TypeVar, Dict, List, Tuple, Any, Optional, Union, 
    Callable, Iterator, NamedTuple
)
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod


# Type aliases for common data structures
MetricValue = Union[float, int]
ModelName = str
ExperimentType = str  # 'suicide_detection' or 'therapy_request'
ClassificationLabel = str
TimestampStr = str
FilePath = Union[str, Path]

# Result data structures
class ModelMetrics(NamedTuple):
    """Metrics for a single model."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    support: int
    
    
class ExperimentResults(NamedTuple):
    """Results from a complete experiment."""
    metrics: Dict[ModelName, ModelMetrics]
    confusion_matrices: Dict[ModelName, pd.DataFrame]
    raw_data: pd.DataFrame
    metadata: Dict[str, Any]


# Configuration protocols
class ExperimentConfigProtocol(Protocol):
    """Protocol defining experiment configuration interface."""
    
    @property
    def experiment_type(self) -> ExperimentType:
        """Type of experiment."""
        ...
    
    @property
    def binary_positive_categories(self) -> List[str]:
        """Categories considered positive for binary classification."""
        ...
    
    @property
    def binary_classification_name(self) -> str:
        """Name for the positive class."""
        ...
    
    @property
    def results_pattern(self) -> str:
        """Pattern to match result files."""
        ...
    
    @property
    def report_title(self) -> str:
        """Title for analysis reports."""
        ...


class AnalysisConfigProtocol(Protocol):
    """Protocol for complete analysis configuration."""
    
    output_dir: Path
    input_data: str
    prompt_file: str
    timestamp: str
    experiment_config: ExperimentConfigProtocol


# Data processing protocols
class DataLoaderProtocol(Protocol):
    """Protocol for data loading operations."""
    
    def load_experiment_results(self, results_pattern: str) -> pd.DataFrame:
        """Load experiment results from files matching pattern."""
        ...
    
    def filter_by_experiment_type(self, 
                                  df: pd.DataFrame, 
                                  experiment_type: str) -> pd.DataFrame:
        """Filter results by experiment type."""
        ...


class MetricsCalculatorProtocol(Protocol):
    """Protocol for metrics calculation."""
    
    def calculate_comprehensive_metrics(self, 
                                        df: pd.DataFrame,
                                        positive_categories: List[str]) -> pd.DataFrame:
        """Calculate comprehensive metrics for all models."""
        ...
    
    def calculate_model_metrics(self,
                                df: pd.DataFrame,
                                model_name: str,
                                positive_categories: List[str]) -> ModelMetrics:
        """Calculate metrics for a single model."""
        ...


class VisualizationProtocol(Protocol):
    """Protocol for visualization operations."""
    
    def create_performance_plots(self,
                                 metrics_df: pd.DataFrame,
                                 output_path: FilePath,
                                 title: str) -> None:
        """Create performance visualization plots."""
        ...
    
    def create_confusion_matrices(self,
                                  df: pd.DataFrame,
                                  models: List[str],
                                  positive_categories: List[str],
                                  output_dir: Path) -> None:
        """Create confusion matrix visualizations."""
        ...


class ReportGeneratorProtocol(Protocol):
    """Protocol for report generation."""
    
    def generate_analysis_report(self,
                                 config: AnalysisConfigProtocol,
                                 metrics_df: pd.DataFrame,
                                 total_results: int,
                                 num_experiments: int,
                                 output_path: FilePath) -> None:
        """Generate comprehensive analysis report."""
        ...


# File management protocol
class FileManagerProtocol(Protocol):
    """Protocol for file management operations."""
    
    def ensure_directory_structure(self, output_dir: Path) -> Dict[str, Path]:
        """Ensure output directory structure exists."""
        ...
    
    def copy_raw_results(self,
                         results_pattern: str,
                         raw_results_dir: Path) -> int:
        """Copy raw result files to output directory."""
        ...


# Analysis orchestration protocols
class ExperimentOrchestratorProtocol(Protocol):
    """Protocol for experiment orchestration."""
    
    def execute_experiment(self,
                           output_dir: str,
                           input_data: str,
                           prompt_file: str,
                           experiment_type: str,
                           timestamp: Optional[str] = None) -> Dict[str, Any]:
        """Execute a safety simulation experiment."""
        ...
    
    def execute_batch_experiments(self,
                                  experiments: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """Execute multiple experiments in sequence."""
        ...


class ComprehensiveAnalyzerProtocol(Protocol):
    """Protocol for the main analysis pipeline."""
    
    def run_analysis(self) -> None:
        """Run complete analysis pipeline."""
        ...


# Generic types for data processing
T = TypeVar('T')
DataFrameProcessor = Callable[[pd.DataFrame], pd.DataFrame]
MetricsProcessor = Callable[[pd.DataFrame], pd.DataFrame]
ValidationFunction = Callable[[T], bool]

# Error types for better error handling
class SafetySimulationError(Exception):
    """Base exception for safety simulation errors."""
    pass


class ConfigurationError(SafetySimulationError):
    """Raised when configuration is invalid."""
    pass


class DataProcessingError(SafetySimulationError):
    """Raised when data processing fails."""
    pass


class VisualizationError(SafetySimulationError):
    """Raised when visualization generation fails."""
    pass


class FileOperationError(SafetySimulationError):
    """Raised when file operations fail."""
    pass


# Result validation functions
def validate_experiment_results(df: pd.DataFrame) -> bool:
    """Validate that a DataFrame contains required experiment result columns."""
    required_columns = ['model_full_name', 'status']
    return all(col in df.columns for col in required_columns)