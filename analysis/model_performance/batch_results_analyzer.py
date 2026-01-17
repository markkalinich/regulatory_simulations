#!/usr/bin/env python3
"""
Batch Results Analyzer for Safety Simulations

Processes completed experiment result files to generate multi-model analysis reports. 
Takes CSV result files from multiple model experiments and produces:

- Performance comparison tables and plots across models
- Confusion matrices for classification analysis  
- Statistical summaries and rankings
- Text reports
- Organized file archives

Works on completed experiment result files, not live experiments.
For running new experiments, use orchestration/run_experiment.py or bash_scripts/run_all_models.sh.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
from pathlib import Path
import glob
from datetime import datetime
import warnings
from typing import Dict, List, Optional, Any
import sys

# Add parent directory to path for absolute imports when running as script
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration constants are now handled by experiment_config module
try:
    from .metrics_calculator import calculate_model_metrics, generate_metrics_for_all_models, determine_multiclass_labels
    from .visualization import create_performance_plots
    from .confusion_matrices import create_all_confusion_matrices
    from .data_loader import load_and_validate_results, get_experiment_result_files, save_cache_manifest
    from .single_experiment_report_generator import generate_comprehensive_report
except ImportError:
    # Running as script directly
    from analysis.model_performance.metrics_calculator import calculate_model_metrics, generate_metrics_for_all_models, determine_multiclass_labels
    from analysis.model_performance.visualization import create_performance_plots
    from analysis.model_performance.confusion_matrices import create_all_confusion_matrices
    from analysis.model_performance.data_loader import load_and_validate_results, get_experiment_result_files, save_cache_manifest
    from analysis.model_performance.single_experiment_report_generator import generate_comprehensive_report

from utilities.file_manager import copy_raw_results, ensure_directory_structure
from config.experiment_config import AnalysisConfiguration
from utilities.types_definitions import (
    AnalysisConfigProtocol, ComprehensiveAnalyzerProtocol, 
    DataProcessingError, ConfigurationError
)
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class BatchResultsAnalyzer:
    """
    Batch results analyzer for safety simulation experiments.
    
    Loads and analyzes completed experiment result files from multiple models.
    Coordinates the post-experiment analysis pipeline:
    
    - Data loading and validation from CSV result files
    - Performance metrics calculation (accuracy, precision, recall, F1)
    - Multi-model comparison visualizations
    - Confusion matrix generation for all models
    - Text report generation
    - Raw result file organization and archival
    """
    
    def __init__(self, config: AnalysisConfiguration, results_timestamp: Optional[str] = None) -> None:
        """
        Initialize analyzer with configuration object.
        
        Args:
            config: Complete analysis configuration
            results_timestamp: Filter results to only those from this specific timestamp
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(config, AnalysisConfiguration):
            raise ConfigurationError("Invalid configuration type")
            
        self.config = config
        
        # Extract frequently used properties for backward compatibility
        self.output_dir = config.output_dir
        self.input_data = config.input_data
        self.prompt_file = config.prompt_file
        self.timestamp = config.timestamp
        self.experiment_type = config.experiment_config.experiment_type
        
        # Extract experiment-specific configuration
        self.binary_positive_categories = config.experiment_config.binary_positive_categories
        self.ground_truth_positive_categories = config.experiment_config.ground_truth_positive_categories
        self.binary_classification_name = config.experiment_config.binary_classification_name
        self.report_title = config.experiment_config.report_title
        
        # Model configurations
        self.model_families = config.experiment_config.model_config.families
        
        # Results storage
        self.results_df = None
        self.metrics_summary = []
        
        # Ensure output directory structure exists using extracted function
        ensure_directory_structure(self.output_dir)
        
    def generate_analysis_id(self) -> str:
        """Generate enhanced analysis ID with prompt and dataset names."""
        # Extract prompt name (filename without extension)
        prompt_name = Path(self.prompt_file).stem
        
        # Extract dataset name (filename without extension) 
        dataset_name = Path(self.input_data).stem
        
        return f"{self.timestamp}_{prompt_name}_{dataset_name}"
        
    def load_experiment_results(self) -> None:
        """Load all experiment results from CACHE DATABASE."""
        # Use extracted data loading function - loads from cache, not CSV files
        self.results_df, self.multiclass_labels = load_and_validate_results(
            self.input_data,
            self.prompt_file,
            self.model_families,
            self.experiment_type,
            cache_dir=self.config.cache_dir
        )
        
        # Save cache manifest for traceability (if available)
        manifest_path = self.output_dir / 'cache_manifest.json'
        if hasattr(self.results_df, 'attrs') and self.results_df.attrs.get('cache_manifest'):
            save_cache_manifest(self.results_df, str(manifest_path))
        
    def calculate_metrics_for_model(self, model_family: str, model_size: str) -> Optional[Dict[str, Any]]:
        """Calculate comprehensive metrics for a single model."""
        # Ensure results_df is loaded
        if self.results_df is None:
            self.load_experiment_results()
        
        # Type checker assurance - load_experiment_results sets results_df
        assert self.results_df is not None
        
        return calculate_model_metrics(
            self.results_df, 
            model_family, 
            model_size, 
            self.experiment_type, 
            self.binary_positive_categories,
            self.ground_truth_positive_categories
        )
    
    def generate_metrics_summary(self) -> None:
        """Generate metrics for all models."""
        print("Calculating metrics for all models...")
        # Ensure results are loaded
        if self.results_df is None:
            self.load_experiment_results()
        
        # Type checker assurance - load_experiment_results sets results_df
        assert self.results_df is not None
        
        self.metrics_summary = generate_metrics_for_all_models(
            self.results_df,
            self.model_families,
            self.experiment_type,
            self.binary_positive_categories,
            self.ground_truth_positive_categories
        )
        
        self.metrics_df = pd.DataFrame(self.metrics_summary)
        # Save metrics table
        metrics_file = self.output_dir / 'tables' / 'comprehensive_metrics.csv'
        self.metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics table: {metrics_file}")
        
    def create_performance_plots(self):
        """Create comprehensive performance visualization"""
        # Ensure metrics are loaded
        if not hasattr(self, 'metrics_df') or self.metrics_df is None or self.metrics_df.empty:
            self.generate_metrics_summary()
        
        # Use extracted visualization function
        plot_file = self.output_dir / 'plots' / 'comprehensive_performance.png'
        create_performance_plots(
            self.metrics_df,
            plot_file,
            title='Model Performance Across Families'
        )
    
    def create_confusion_matrices(self):
        """Generate confusion matrices for each model"""
        # Ensure results are loaded
        if self.results_df is None:
            self.load_experiment_results()
        
        # Type checker assurance - load_experiment_results sets results_df
        assert self.results_df is not None
        
        # Use extracted confusion matrix function
        create_all_confusion_matrices(
            self.results_df,
            self.model_families,
            self.experiment_type,
            self.binary_positive_categories,
            self.ground_truth_positive_categories,
            self.binary_classification_name,
            self.multiclass_labels,
            self.output_dir
        )
    

    
    def generate_analysis_report(self):
        """Generate comprehensive text analysis report"""
        # Prepare analysis information
        analysis_info = {
            'timestamp': self.timestamp,
            'prompt_file': self.prompt_file,
            'input_data': self.input_data
        }
        
        # Use extracted report generation function
        report_file = self.output_dir / 'reports' / 'comprehensive_analysis_report.txt'
        generate_comprehensive_report(
            self.metrics_df,
            self.report_title,
            analysis_info,
            report_file
        )
    
    def write_model_output_csvs(self):
        """Write model output CSV files from cache data (for provenance/archival)."""
        print("Writing model output CSV files...")
        model_outputs_dir = self.output_dir / 'model_outputs'
        model_outputs_dir.mkdir(exist_ok=True)
        
        # Ensure results_df is loaded
        if self.results_df is None:
            self.load_experiment_results()
        
        # Type checker assurance
        assert self.results_df is not None
        
        # Group results by model
        grouped = self.results_df.groupby(['model_family', 'model_size'])
        for (model_family, model_size), group_df in grouped:
            # Generate filename
            prompt_name = Path(self.prompt_file).stem
            dataset_name = Path(self.input_data).stem
            filename = f"{model_family}_{model_size}_{prompt_name}_{dataset_name}_results.csv"
            filepath = model_outputs_dir / filename
            
            # Write CSV
            group_df.to_csv(filepath, index=False)
        
        print(f"Wrote {len(grouped)} model output files to {model_outputs_dir}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive analysis...")
        
        # Load and analyze data first
        self.load_experiment_results()
        self.generate_metrics_summary()
        
        # Generate outputs
        self.create_performance_plots()
        self.create_confusion_matrices()
        self.write_model_output_csvs()
        self.generate_analysis_report()
        
        print(f"Analysis complete! Results saved to: {self.output_dir}")

def parse_models_filter(models_str: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """
    Parse models filter string into family:sizes dict.
    
    Args:
        models_str: Comma-separated "family:size" pairs, e.g., "gemma:270m-it,llama:8b-it"
        
    Returns:
        Dict mapping family names to list of sizes, or None if no filter
    """
    if not models_str:
        return None
    
    families: Dict[str, List[str]] = {}
    for pair in models_str.split(','):
        pair = pair.strip()
        if ':' not in pair:
            print(f"⚠️  Invalid model format '{pair}', expected 'family:size'. Skipping.")
            continue
        family, size = pair.split(':', 1)
        family = family.strip()
        size = size.strip()
        if family not in families:
            families[family] = []
        if size not in families[family]:
            families[family].append(size)
    
    return families if families else None


def main():
    parser = argparse.ArgumentParser(description='Analyze completed safety simulation experiment results across multiple models')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--input-data', required=True, help='Input data file')
    parser.add_argument('--prompt-file', required=True, help='Prompt file')
    parser.add_argument('--timestamp', required=True, help='Analysis timestamp')
    parser.add_argument('--experiment-type', required=True, 
                       choices=['suicidal_ideation', 'therapy_request', 'therapy_engagement'], 
                       help='Type of experiment: suicidal_ideation, therapy_request, or therapy_engagement')
    parser.add_argument('--results-timestamp', help='Filter results to only those created during this timestamp (YYYYMMDD_HHMMSS)')
    parser.add_argument('--models', help='Comma-separated list of models to analyze (family:size format). If omitted, analyzes all models.')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory (default: cache). Use cache_v2 for V2 cache.')
    
    args = parser.parse_args()
    
    # Parse models filter
    models_filter = parse_models_filter(args.models)
    if models_filter:
        print(f"📋 Analyzing specific models: {args.models}")
    else:
        print(f"📋 Analyzing all models from config")
    
    # Create configuration object
    config = AnalysisConfiguration.create(
        args.output_dir, 
        args.input_data, 
        args.prompt_file, 
        args.timestamp, 
        args.experiment_type,
        models_filter=models_filter,
        cache_dir=args.cache_dir
    )
    
    analyzer = BatchResultsAnalyzer(config, args.results_timestamp)
    analyzer.run_analysis()

if __name__ == '__main__':
    main()