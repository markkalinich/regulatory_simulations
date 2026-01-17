#!/usr/bin/env python3
"""
Figure Provenance Tracker for Reproducible Science

Automatically tracks data sources, file hashes, and analysis parameters for all figures.
Ensures complete reproducibility by creating timestamped snapshots of source data.

Usage:
    tracker = FigureProvenanceTracker(
        figure_name="SI Review Barplot",
        base_dir="results/data_validation/psychiatrist_review_barplots"
    )
    
    tracker.add_input_dataset(
        "data/inputs/finalized_input_data/SI_finalized_sentences.csv",
        description="Final validated SI dataset",
        columns_used=["Safety type", "Counseling Request", "statement"]
    )
    
    tracker.set_analysis_parameters(
        categories=["Kept", "Modified", "Removed"],
        grouping="by_safety_type"
    )
    
    output_path = tracker.get_output_path("SI_review_barplot.png")
    plt.savefig(output_path, dpi=300)
    tracker.add_output_file(output_path)
    
    tracker.save_provenance()
"""

from pathlib import Path
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import shutil


class FigureProvenanceTracker:
    """
    Track data provenance for reproducible figures.
    
    Creates timestamped directories organized by date, with files timestamped to the minute.
    Each run gets its own set of files within the day's folder.
    
    Directory structure:
        results/{base_dir}/
        ├── YYYYMMDD/                                    # Date folder
        │   ├── YYYYMMDD_HHMM_figure_name.png
        │   ├── YYYYMMDD_HHMM_data_provenance.json
        │   └── source_data/
        │       └── YYYYMMDD_HHMM_dataset.csv
        └── YYYYMMDD+1/
            └── ...
    """
    
    def __init__(self, figure_name: str, base_dir: Union[str, Path], prefix: Optional[str] = None):
        """
        Initialize the provenance tracker.
        
        Args:
            figure_name: Name of the figure being generated
            base_dir: Base directory for output (will create date-based subdirectory)
            prefix: Optional prefix for the folder name (e.g., "SI", "tx_request")
        """
        self.figure_name = figure_name
        self.base_dir = Path(base_dir)
        
        # Create date-based output directory (YYYYMMDD format)
        self.date = datetime.now().strftime('%Y%m%d')
        
        # Generate timestamp for this run (YYYYMMDD_HHMM format - military time)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Create figure-specific subdirectory with optional prefix
        # Format: YYYYMMDD/[prefix_]YYYYMMDD_HHMM_figurename/
        if prefix:
            self.figure_subdir = f"{prefix}_{self.timestamp}_{figure_name}"
        else:
            self.figure_subdir = f"{self.timestamp}_{figure_name}"
        self.output_dir = self.base_dir / self.date / self.figure_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking structures
        self.input_datasets: List[Dict[str, Any]] = []
        self.analysis_parameters: Dict[str, Any] = {}
        self.output_files: List[Dict[str, Any]] = []
        self.reproducibility_command: Optional[str] = None
        
        # Create source_data subdirectory within figure folder
        self.source_data_dir = self.output_dir / 'source_data'
        self.source_data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized provenance tracker:")
        print(f"  Figure: {figure_name}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Timestamp: {self.timestamp}")
    
    def add_input_dataset(self, 
                         file_path: Union[str, Path], 
                         description: str,
                         columns_used: Optional[List[str]] = None,
                         copy_to_source: bool = True):
        """
        Add an input dataset with automatic hashing and optional copying.
        
        Args:
            file_path: Path to input dataset
            description: Human-readable description of the dataset
            columns_used: List of column names used from this dataset (optional)
            copy_to_source: If True, copy file to source_data/ folder (default: True)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {file_path}")
        
        # Calculate file hash for verification
        file_hash = self._hash_file(file_path)
        
        # Get row count if CSV
        row_count = None
        if file_path.suffix == '.csv':
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
            except Exception as e:
                print(f"Warning: Could not read CSV for row count: {e}")
        
        # Record dataset metadata
        dataset_info = {
            "file_path": str(file_path),
            "file_hash_sha256": file_hash,
            "row_count": row_count,
            "columns_used": columns_used,
            "description": description,
            "recorded_at": datetime.now().isoformat()
        }
        
        # Copy to source_data folder with descriptive naming
        if copy_to_source:
            # Always use descriptive naming with grandparent_parent_filename
            # e.g., "suicide_ideation_tables_comprehensive_metrics.csv"
            grandparent = file_path.parent.parent.name if len(file_path.parents) >= 2 else ""
            parent_name = file_path.parent.name
            stem = file_path.stem
            suffix = file_path.suffix
            
            if grandparent:
                dest_filename = f"{grandparent}_{parent_name}_{stem}{suffix}"
            else:
                dest_filename = f"{parent_name}_{stem}{suffix}"
            
            dest_path = self.source_data_dir / dest_filename
            
            # Only copy if file doesn't exist or has different hash
            if dest_path.exists():
                existing_hash = self._hash_file(dest_path)
                if existing_hash == file_hash:
                    # Same file already copied, skip
                    dataset_info["source_data_copy"] = f"source_data/{dest_path.name}"
                    print(f"  Added dataset: {file_path.name} ({row_count} rows)")
                    print(f"    Already copied: source_data/{dest_path.name}")
                    self.input_datasets.append(dataset_info)
                    return
            
            shutil.copy2(file_path, dest_path)
            dataset_info["source_data_copy"] = f"source_data/{dest_path.name}"
            print(f"  Added dataset: {file_path.name} ({row_count} rows)")
            print(f"    Copied to: source_data/{dest_path.name}")
        else:
            print(f"  Added dataset: {file_path.name} (not copied)")
        
        self.input_datasets.append(dataset_info)
    
    def set_analysis_parameters(self, **kwargs):
        """
        Record analysis parameters used to generate the figure.
        
        Args:
            **kwargs: Any key-value pairs describing analysis parameters
        """
        self.analysis_parameters.update(kwargs)
        print(f"Set analysis parameters: {len(kwargs)} parameters recorded")
    
    def add_output_file(self, file_path: Union[str, Path], file_type: str = "figure", **metadata):
        """
        Record an output file generated by the analysis.
        
        Args:
            file_path: Path to output file (can be relative or absolute)
            file_type: Type of file (e.g., "figure", "table", "data")
            **metadata: Additional metadata about the file
        """
        file_path = Path(file_path)
        
        output_info = {
            "file_path": str(file_path),
            "file_type": file_type,
            "relative_path": str(file_path.relative_to(self.output_dir)) if file_path.is_relative_to(self.output_dir) else str(file_path.name),
            **metadata
        }
        self.output_files.append(output_info)
        print(f"  Added output: {file_path.name}")
    
    def set_reproducibility_command(self, command: str):
        """
        Set the command to reproduce this figure.
        
        Args:
            command: Shell command or Python script invocation
        
        Example:
            tracker.set_reproducibility_command(
                "python analysis/psychiatrist_review_barplots.py --task SI"
            )
        """
        self.reproducibility_command = command
    
    def get_output_path(self, filename: str) -> Path:
        """
        Get the full output path for a file (no timestamp prefix needed now).
        
        Args:
            filename: Base filename (e.g., "combined_three_panel_review.png")
        
        Returns:
            Full path within figure subfolder
        """
        return self.output_dir / filename
    
    def save_provenance(self):
        """Save provenance metadata to JSON file."""
        provenance_path = self.output_dir / "data_provenance.json"
        
        provenance_data = {
            "figure_name": self.figure_name,
            "generated_at": datetime.now().isoformat(),
            "date": self.date,
            "time": datetime.now().strftime('%H:%M:%S'),
            "input_datasets": self.input_datasets,
            "analysis_parameters": self.analysis_parameters,
            "output_files": self.output_files,
            "reproducibility_command": self.reproducibility_command
        }
        
        with open(provenance_path, 'w') as f:
            json.dump(provenance_data, f, indent=2)
        
        print(f"✓ Saved provenance: {provenance_path.relative_to(self.base_dir)}")
        print(f"  Total inputs: {len(self.input_datasets)}")
        print(f"  Total outputs: {len(self.output_files)}")
        
        return provenance_path
    
    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """
        Calculate SHA256 hash of file for verification.
        
        Args:
            file_path: Path to file
        
        Returns:
            Hexadecimal SHA256 hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def verify_dataset(provenance_file: str, dataset_index: int = 0) -> bool:
        """
        Verify that a dataset hasn't changed since figure was generated.
        
        Args:
            provenance_file: Path to data_provenance.json
            dataset_index: Index of dataset in input_datasets list
        
        Returns:
            True if hash matches, False if changed
        """
        with open(provenance_file, 'r') as f:
            provenance = json.load(f)
        
        dataset = provenance["input_datasets"][dataset_index]
        original_hash = dataset["file_hash_sha256"]
        current_hash = FigureProvenanceTracker._hash_file(Path(dataset["file_path"]))
        
        if original_hash == current_hash:
            print(f"✓ Dataset verified: {dataset['file_path']}")
            return True
        else:
            print(f"✗ Dataset changed: {dataset['file_path']}")
            print(f"  Original hash: {original_hash}")
            print(f"  Current hash:  {current_hash}")
            return False
    
    @staticmethod
    def list_runs(base_dir: str, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all figure generation runs in a directory.
        
        Args:
            base_dir: Base directory to search
            date: Optional date string (YYYYMMDD) to filter by
        
        Returns:
            List of run information dictionaries
        """
        base_path = Path(base_dir)
        runs = []
        
        # Get all date folders
        if date:
            date_folders = [base_path / date] if (base_path / date).exists() else []
        else:
            date_folders = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
        
        # Find all provenance files
        for date_folder in sorted(date_folders):
            provenance_files = list(date_folder.glob("*_data_provenance.json"))
            for prov_file in sorted(provenance_files):
                with open(prov_file, 'r') as f:
                    provenance = json.load(f)
                
                runs.append({
                    "date": date_folder.name,
                    "timestamp": provenance.get("generated_at"),
                    "figure_name": provenance.get("figure_name"),
                    "provenance_file": str(prov_file),
                    "num_inputs": len(provenance.get("input_datasets", [])),
                    "num_outputs": len(provenance.get("output_files", []))
                })
        
        return runs


def example_usage():
    """Example usage of FigureProvenanceTracker."""
    # Initialize tracker
    tracker = FigureProvenanceTracker(
        figure_name="SI Psychiatrist Review Barplot",
        base_dir="results/data_validation/psychiatrist_review_barplots"
    )
    
    # Add input datasets
    tracker.add_input_dataset(
        "data/inputs/finalized_input_data/SI_finalized_sentences.csv",
        description="Final validated SI dataset after psychiatrist review",
        columns_used=["Safety type", "Counseling Request", "statement"]
    )
    
    tracker.add_input_dataset(
        "data/inputs/manual_review/SI_psychiatrist_01_reviewed.csv",
        description="Psychiatrist 01 review decisions",
        columns_used=["Example_ID", "Action", "Reason"]
    )
    
    # Set analysis parameters
    tracker.set_analysis_parameters(
        categories=["Kept", "Modified", "Removed"],
        grouping="by_safety_type",
        visualization_type="bar_plot"
    )
    
    # Generate figure (your plotting code here)
    output_path = tracker.get_output_path("SI_review_barplot.png")
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Record output
    tracker.add_output_file(output_path, file_type="figure", dpi=300)
    
    # Set reproducibility command
    tracker.set_reproducibility_command(
        "python analysis/psychiatrist_review_barplots.py --task SI"
    )
    
    # Save provenance
    tracker.save_provenance()


if __name__ == "__main__":
    print("Figure Provenance Tracker")
    print("=" * 50)
    print("\nRun example_usage() to see demo")
    print("\nTo use in your scripts:")
    print("  from utilities.figure_provenance import FigureProvenanceTracker")
