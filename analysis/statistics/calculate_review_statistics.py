#!/usr/bin/env python3
"""
Calculate chi-squared tests for review statistics.
Reads CSV files and compares approval rates between groups.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load data
si = pd.read_csv('results/review_statistics/si_review_statistics.csv')
tr = pd.read_csv('results/review_statistics/therapy_request_review_statistics.csv')
te = pd.read_csv('results/review_statistics/therapy_engagement_review_statistics.csv')

def get_stats(df, labels):
    """Extract approved/other counts for specified rows."""
    rows = df[df['Category'].isin(labels)]
    approved = rows['Approved (No Changes)'].sum()
    other = rows['Approved (Modified)'].sum() + rows['Not Approved (Removed)'].sum()
    total = rows['Generated'].sum()
    return approved, other, total

def chi2_test(name1, stats1, name2, stats2):
    """Run chi-squared test and return results."""
    app1, oth1, tot1 = stats1
    app2, oth2, tot2 = stats2
    
    chi2, p, dof, _ = chi2_contingency([[app1, oth1], [app2, oth2]])
    
    print(f"\n{name1}: {app1}/{tot1} ({100*app1/tot1:.1f}%) vs {name2}: {app2}/{tot2} ({100*app2/tot2:.1f}%)")
    print(f"  χ² = {chi2:.4f}, p = {p:.4e}")
    
    return {
        'Comparison': f"{name1} vs {name2}",
        'Group1': name1,
        'Group1_Approved': app1,
        'Group1_Total': tot1,
        'Group1_Percent': 100*app1/tot1,
        'Group2': name2,
        'Group2_Approved': app2,
        'Group2_Total': tot2,
        'Group2_Percent': 100*app2/tot2,
        'Chi_Squared': chi2,
        'P_Value': p
    }

print("="*80)
print("REVIEW STATISTICS - CHI-SQUARED TESTS")
print("="*80)

results = []

# Test 1: Non-SI vs SI
print("\n1. Non-SI vs SI")
non_si = get_stats(si, ['SUBTOTAL: Non-SI'])
si_group = get_stats(si, ['SUBTOTAL: SI'])
results.append(chi2_test('Non-SI', non_si, 'SI', si_group))

# Test 2: No Therapy Request vs Therapy Request
print("\n2. No Therapy Request vs Therapy Request")
no_tr = get_stats(tr, ['SUBTOTAL: Declarative Statements', 'SUBTOTAL: Non-Therapeutic Questions'])
yes_tr = get_stats(tr, ['SUBTOTAL: Explicit Therapy Requests'])
results.append(chi2_test('No Therapy Request', no_tr, 'Therapy Request', yes_tr))

# Test 3: Non-Therapeutic Interaction vs Therapeutic Conversation
print("\n3. Non-Therapeutic Interaction vs Therapeutic Conversation")
non_ther = get_stats(te, ['SUBTOTAL: Non-Therapeutic Conversations', 'SUBTOTAL: Ambiguous Engagement'])
ther = get_stats(te, ['SUBTOTAL: Therapeutic Conversation'])
results.append(chi2_test('Non-Therapeutic Interaction', non_ther, 'Therapeutic Conversation', ther))

print("\n" + "="*80)

# Save results to CSV
results_df = pd.DataFrame(results)
output_path = 'results/review_statistics/chi_squared_tests.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

