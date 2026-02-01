"""
Experiment Orchestrator for Safety Simulations

Provides a Python interface for executing safety simulation experiments with 
configuration validation and result processing. Supports suicide detection 
and therapy request classification experiment types.

Handles:
- Experiment configuration validation and setup
- Execution of the analysis pipeline
- Result compilation and organization
- Error handling and logging
- Reporting across experiment types
"""

from typing import Protocol, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from config.experiment_config import AnalysisConfiguration
from analysis.model_performance.batch_results_analyzer import BatchResultsAnalyzer


class ExperimentOrchestrator(Protocol):
    """Protocol defining the interface for experiment orchestrators."""
    
    def execute(self, config: AnalysisConfiguration) -> Dict[str, Any]:
        """Execute an experiment with the given configuration."""
        ...


class SafetySimulationOrchestrator:
    """
    Orchestrator for safety simulation experiments.
    
    Provides a programmatic interface for executing safety simulation experiments
    with automatic configuration of experiment-specific parameters. Handles:
    
    - Experiment type detection (suicide detection vs therapy request classification)
    - Input data validation and preprocessing  
    - Analysis configuration setup based on experiment type
    - Execution of the analysis pipeline
    - Result validation and organization
    - Error handling and logging
    
    Supports suicide detection and therapy request classification experiments
    with appropriate model configurations and evaluation metrics.
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 validate_inputs: bool = True):
        """
        Initialize the orchestrator.
        
        Args:
            logger: Optional logger for experiment tracking
            validate_inputs: Whether to validate configuration inputs
        """
        self.logger = logger or logging.getLogger(__name__)
        self.validate_inputs = validate_inputs
        
    def execute_experiment(self, 
                          output_dir: str,
                          input_data: str, 
                          prompt_file: str,
                          experiment_type: str,
                          timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a safety simulation experiment.
        
        Args:
            output_dir: Directory for analysis outputs
            input_data: Path to input data file
            prompt_file: Path to prompt template file
            experiment_type: Type of experiment ('suicide_detection' or 'therapy_request')
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            Dictionary containing execution results and metadata
        """
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create configuration
        config = AnalysisConfiguration.create(
            output_dir=output_dir,
            input_data=input_data,
            prompt_file=prompt_file,
            timestamp=timestamp,
            experiment_type=experiment_type
        )
        
        # Validate configuration if requested
        if self.validate_inputs:
            self._validate_configuration(config)
            
        self.logger.info(f"Starting {experiment_type} experiment with timestamp {timestamp}")
        self.logger.info(f"Input data: {input_data}")
        self.logger.info(f"Output directory: {output_dir}")
        
        try:
            # Execute analysis
            analyzer = BatchResultsAnalyzer(config)
            analyzer.run_analysis()
            
            # Collect results metadata
            results = {
                'status': 'success',
                'experiment_type': config.experiment_config.experiment_type,
                'timestamp': timestamp,
                'output_directory': output_dir,
                'input_data_file': input_data,
                'prompt_file': prompt_file,
                'configuration': {
                    'binary_positive_categories': list(config.experiment_config.binary_positive_categories),
                    'binary_classification_name': config.experiment_config.binary_classification_name,
                    'results_pattern': config.experiment_config.results_pattern,
                    'report_title': config.experiment_config.report_title
                }
            }
            
            self.logger.info("Experiment completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'experiment_type': experiment_type,
                'timestamp': timestamp
            }
    
    def execute_batch_experiments(self, 
                                 experiments: list[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Execute multiple experiments in sequence.
        
        Args:
            experiments: List of experiment configurations, each containing:
                - output_dir, input_data, prompt_file, experiment_type
                - Optional: timestamp
                
        Returns:
            Dictionary mapping experiment IDs to results
        """
        results = {}
        
        for i, exp_config in enumerate(experiments):
            exp_id = f"experiment_{i+1}"
            self.logger.info(f"Starting batch experiment {exp_id}")
            
            result = self.execute_experiment(**exp_config)
            results[exp_id] = result
            
            if result['status'] == 'error':
                self.logger.warning(f"Experiment {exp_id} failed, continuing with next")
            
        return results
    
    def _validate_configuration(self, config: AnalysisConfiguration) -> None:
        """
        Validate experiment configuration.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check that input files exist
        input_path = Path(config.input_data)
        if not input_path.exists():
            raise ValueError(f"Input data file not found: {config.input_data}")
            
        prompt_path = Path(config.prompt_file)
        if not prompt_path.exists():
            raise ValueError(f"Prompt file not found: {config.prompt_file}")
            
        # Validate experiment type
        valid_types = {'suicide_detection', 'therapy_request'}
        if config.experiment_config.experiment_type not in valid_types:
            raise ValueError(f"Invalid experiment type: {config.experiment_config.experiment_type}. "
                           f"Must be one of: {valid_types}")
            
        # Check that configuration has required attributes  
        if not hasattr(config.experiment_config, 'binary_positive_categories'):
            raise ValueError("Configuration missing required attribute: binary_positive_categories")
        
        if not config.experiment_config.binary_positive_categories:
            raise ValueError("binary_positive_categories cannot be empty")


def create_orchestrator(logger: Optional[logging.Logger] = None,
                       validate_inputs: bool = True) -> SafetySimulationOrchestrator:
    """
    Factory function to create a configured orchestrator instance.
    
    Args:
        logger: Optional logger for experiment tracking
        validate_inputs: Whether to validate configuration inputs
        
    Returns:
        Configured SafetySimulationOrchestrator instance
    """
    return SafetySimulationOrchestrator(logger=logger, validate_inputs=validate_inputs)