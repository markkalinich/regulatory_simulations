#!/usr/bin/env python3
"""
Generate correctness matrices from experiment results - SIMPLE VERSION

This is a simpler, standalone script with hardcoded paths for quick one-off analysis.
For the full-featured version with CLI arguments and heatmap generation, use:
    generate_model_statement_matrices.py

Both scripts create binary correctness matrices (rows=models, cols=statements, values=0/1)
using the same centralized classification utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use centralized classification utilities (DO NOT duplicate these functions)
from utilities.classification_utils import (
    to_binary_si,
    to_binary_therapy_request,
    to_binary_therapy_engagement,
)


def load_results(results_dir):
    """Load all model output CSVs from a directory"""
    outputs = {}
    for csv_file in sorted(Path(results_dir).glob('*.csv')):
        model_name = '_'.join(csv_file.stem.split('_')[:2])
        if model_name not in outputs:
            outputs[model_name] = pd.read_csv(csv_file)
    return outputs


def create_matrix(model_outputs, ground_truth_col, prediction_col, normalizer=None):
    """
    Create correctness matrix: rows=models, cols=statements, values=1 (correct) or 0 (miss)
    
    Args:
        model_outputs: dict of model_name -> DataFrame
        ground_truth_col: column name for ground truth
        prediction_col: column name for predictions
        normalizer: optional function to normalize labels before comparison
    """
    # Get statement texts from first model
    first_df = list(model_outputs.values())[0]
    statements = first_df['text'].tolist()
    
    # Build matrix
    matrix = {}
    for model_name, df in sorted(model_outputs.items()):
        text_lookup = {row['text']: row for _, row in df.iterrows()}
        
        correctness = []
        for text in statements:
            if text not in text_lookup:
                correctness.append(np.nan)
            else:
                row = text_lookup[text]
                gt = row[ground_truth_col]
                pred = row[prediction_col]
                
                if normalizer:
                    gt = normalizer(gt)
                    pred = normalizer(pred)
                
                correctness.append(1 if pred == gt else 0)
        
        matrix[model_name] = correctness
    
    return pd.DataFrame(matrix, index=range(len(statements))).T


# NOTE: Binary classification functions (to_binary_si, to_binary_therapy_request,
# to_binary_therapy_engagement) are now imported from utilities/classification_utils.py
# DO NOT add duplicate implementations here.


def main():
    # SI
    print("Processing SI...")
    si_outputs = load_results('results/individual_prediction_performance/suicidal_ideation/20251027_182837_SI/model_outputs')
    si_matrix = create_matrix(si_outputs, 'prior_safety_type', 'safety_type', to_binary_si)
    si_matrix.to_csv('results/review_statistics/si_model_statement_correctness_matrix.csv')
    print(f"  SI matrix: {si_matrix.shape}, accuracy: {si_matrix.values.mean():.1%}")
    
    # Therapy Request
    print("Processing therapy request...")
    tr_outputs = load_results('results/individual_prediction_performance/therapy_request/20251027_182856_tx_request/model_outputs')
    tr_matrix = create_matrix(tr_outputs, 'prior_therapy_request', 'therapy_request', to_binary_therapy_request)
    tr_matrix.to_csv('results/review_statistics/therapy_request_model_statement_correctness_matrix.csv')
    print(f"  Therapy request matrix: {tr_matrix.shape}, accuracy: {tr_matrix.values.mean():.1%}")
    
    # Therapy Engagement
    print("Processing therapy engagement...")
    te_outputs = load_results('results/individual_prediction_performance/therapy_engagement/20251030_210744_tx_engagement/model_outputs')
    te_matrix = create_matrix(te_outputs, 'prior_therapy_engagement', 'therapy_engagement', to_binary_therapy_engagement)
    te_matrix.to_csv('results/review_statistics/therapy_engagement_model_conversation_correctness_matrix.csv')
    print(f"  Therapy engagement matrix: {te_matrix.shape}, accuracy: {te_matrix.values.mean():.1%}")


if __name__ == '__main__':
    main()
