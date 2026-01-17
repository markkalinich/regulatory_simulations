#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing utilities for input/output handling.
"""
import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from config.constants import SI_POSITIVE_CATEGORIES


class DataProcessor:
    """Handles input data loading and output data writing."""
    
    def __init__(self):
        """Initialize the data processor.""" 
        pass
    
    def load_input_data(self, input_path: str) -> pd.DataFrame:
        """
        Load and normalize input CSV data.
        
        Supports two formats:
        1. Requested format: ["Safety type", "Counseling Request", "statement"]
        2. Legacy format: ["text"] (+ optional "id")
        
        Args:
            input_path: Path to input CSV file
            
        Returns:
            Normalized DataFrame with standardized column names
            
        Raises:
            ValueError: If required columns are missing
        """
        df = pd.read_csv(input_path)
        
        # Normalize column names (handle case, spaces, BOM)
        normalized_cols = {}
        for c in df.columns:
            c_norm = c.strip().lower().lstrip("\ufeff")
            if c_norm in normalized_cols:
                raise ValueError(f"Duplicate column names after normalization: '{c}' and '{normalized_cols[c_norm]}'")
            normalized_cols[c_norm] = c
        
        # Finalized format: has 'statement' column
        # Rename 'statement' → 'text' for processing
        if "statement" in normalized_cols:
            df = df.rename(columns={normalized_cols["statement"]: "text"})
            
            # Preserve ground truth 'Safety type' column
            if "safety type" in normalized_cols:
                df = df.rename(columns={normalized_cols["safety type"]: "prior_safety_type"})
        elif "text" not in df.columns:
            raise ValueError(
                "Input CSV must contain either a 'statement' column or a 'text' column."
            )
        
        # Ensure ID column exists
        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)
            
        return df
    
    def create_output_row(self, 
                         uid: Any, 
                         text: str, 
                         parsed_result: Optional[Dict[str, Any]], 
                         status: str,
                         row: pd.Series,
                         df_columns: List[str]) -> Dict[str, Any]:
        """
        Create output row for CSV results.
        
        Args:
            uid: Unique identifier for the row
            text: Input text
            parsed_result: Parsed and validated model response
            status: Processing status ("ok", "parse_fail", or error message)
            row: Original input row
            df_columns: Column names from input DataFrame
            
        Returns:
            Dictionary representing output row
        """
        if parsed_result is None:
            parsed_result = {
                "safety_type": None,
                "safety_type_confidence": None,
                "therapy_request": None,
                "therapy_request_confidence": None,
                "therapy_engagement": None,
                "therapy_engagement_confidence": None,
            }
        
        # Determine if suicidal ideation is present
        si_present = None
        if parsed_result["safety_type"]:
            si_present = parsed_result["safety_type"] in SI_POSITIVE_CATEGORIES
        
        return {
            "id": uid,
            "text": text,
            **parsed_result,
            "si_present": si_present,
            "status": status,
            "prior_safety_type": (
                row.get("prior_safety_type") 
                if "prior_safety_type" in df_columns 
                else None
            ),
            "prior_therapy_request": (
                row.get("therapy_request") if "therapy_request" in df_columns 
                else row.get("Counseling Request") if "Counseling Request" in df_columns
                else None
            ),
            "prior_therapy_engagement": (
                row.get("therapy_engagement") 
                if "therapy_engagement" in df_columns 
                else None
            )
        }
    
    def create_jsonl_entry(self,
                          uid: Any,
                          text: str, 
                          raw_response: Dict[str, Any],
                          parsed_result: Optional[Dict[str, Any]],
                          status: str,
                          row: pd.Series,
                          df_columns: List[str]) -> Dict[str, Any]:
        """
        Create JSONL entry for detailed logging.
        
        Args:
            uid: Unique identifier
            text: Input text  
            raw_response: Raw API response
            parsed_result: Parsed and validated response
            status: Processing status
            row: Original input row
            df_columns: Column names from input DataFrame
            
        Returns:
            Dictionary for JSONL output
        """
        return {
            "id": uid,
            "input": text,
            "raw_response": raw_response,
            "parsed": parsed_result,
            "status": status,
            "prior_safety_type": (
                row.get("prior_safety_type") 
                if "prior_safety_type" in df_columns 
                else None
            ),
            "prior_therapy_request": (
                row.get("therapy_request") 
                if "therapy_request" in df_columns 
                else None
            ),
            "prior_therapy_engagement": (
                row.get("therapy_engagement") 
                if "therapy_engagement" in df_columns 
                else None
            )
        }
    
    def create_error_row(self,
                        uid: Any,
                        text: str,
                        error: Exception,
                        row: pd.Series,
                        df_columns: List[str]) -> Dict[str, Any]:
        """
        Create output row for error cases.
        
        Args:
            uid: Unique identifier
            text: Input text
            error: Exception that occurred
            row: Original input row
            df_columns: Column names from input DataFrame
            
        Returns:
            Dictionary representing error row
        """
        return {
            "id": uid,
            "text": text,
            "safety_type": None,
            "safety_type_confidence": None,
            "therapy_request": None, 
            "therapy_request_confidence": None,
            "therapy_engagement": None,
            "therapy_engagement_confidence": None,
            "si_present": None,
            "status": f"error:{type(error).__name__}:{str(error)[:200]}",
            "prior_safety_type": (
                row.get("prior_safety_type") 
                if "prior_safety_type" in df_columns 
                else None
            ),
            "prior_therapy_request": (
                row.get("therapy_request")
                if "therapy_request" in df_columns 
                else None
            ),
            "prior_therapy_engagement": (
                row.get("therapy_engagement")
                if "therapy_engagement" in df_columns 
                else None
            )
        }
    
    def write_outputs(self, 
                     output_rows: List[Dict[str, Any]],
                     jsonl_entries: List[Dict[str, Any]], 
                     csv_path: str,
                     jsonl_path: str) -> None:
        """
        Write output data to CSV and JSONL files.
        
        Args:
            output_rows: List of dictionaries for CSV output
            jsonl_entries: List of dictionaries for JSONL output
            csv_path: Path to write CSV file
            jsonl_path: Path to write JSONL file
        """
        # Ensure output directories exist
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        
        # Write CSV
        pd.DataFrame(output_rows).to_csv(csv_path, index=False)
        
        # Write JSONL  
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")