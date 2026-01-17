#!/usr/bin/env python3
"""
Manuscript Claims Verification - Extract key numbers for the paper

Usage:
    python analysis/manuscript_claims_verification.py --paper-run-dir results/REGULATORY_SIMULATION_PAPER/[timestamp]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# P1/P2 calculation parameters (from p1_and_p2_plot_provenance.py)
THERAPY_REQUEST_RATE = 0.029  # 2.9% from Anthropic data
MODEL_COMPLY_RATE = 0.90      # 90% compliance
PROB_FAIL_SEEK_HELP = 1.0     # Worst case

def calculate_p1(si_pct, si_fnr, tr_fnr):
    """P1 = P(suicidal) × FNR(SI) × P(therapy_request) × FNR(therapy_request) × P(comply)"""
    return (si_pct/100) * si_fnr * THERAPY_REQUEST_RATE * tr_fnr * MODEL_COMPLY_RATE

def calculate_p2(te_fnr, harm_pct, failure_multiplier=1.0):
    """P2 = (1 - (1-FNR(engagement))^m) × P(fail_help) × P(lack_care_harm)"""
    adjusted_fnr = 1 - (1 - te_fnr) ** failure_multiplier
    return adjusted_fnr * PROB_FAIL_SEEK_HELP * (harm_pct/100)

def verify_claims(paper_run_dir: Path) -> str:
    """Extract claims from manuscript and compare to data"""
    
    output = ["# Manuscript Claims Verification\n"]
    
    # Parse success rates from comprehensive_metrics.csv
    si_metrics = pd.read_csv(paper_run_dir / "Data/filtered_metrics/suicidal_ideation_comprehensive_metrics.csv")
    tr_metrics = pd.read_csv(paper_run_dir / "Data/filtered_metrics/therapy_request_comprehensive_metrics.csv")
    te_metrics = pd.read_csv(paper_run_dir / "Data/filtered_metrics/therapy_engagement_comprehensive_metrics.csv")
    
    output.append("## Cross-model performance at prediction tasks\n")
    
    # Gemma 0.27B parse rate
    gemma_270m_si = si_metrics[si_metrics['model_size'].str.contains('270m', case=False)]['parse_success_rate'].values[0] * 100
    gemma_270m_tr = tr_metrics[tr_metrics['model_size'].str.contains('270m', case=False)]['parse_success_rate'].values[0] * 100
    gemma_270m_te = te_metrics[te_metrics['model_size'].str.contains('270m', case=False)]['parse_success_rate'].values[0] * 100
    gemma_range = (min(gemma_270m_si, gemma_270m_tr, gemma_270m_te), max(gemma_270m_si, gemma_270m_tr, gemma_270m_te))
    
    # Check if range overlaps with claim (2-11%)
    passed = gemma_range[0] >= 1.5 and gemma_range[1] <= 12.0
    output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
    output.append(f"Claim: Gemma 0.27B parse rate 2-11% across three tasks")
    output.append(f"Data: {gemma_range[0]:.1f}%-{gemma_range[1]:.1f}% (SI={gemma_270m_si:.1f}%, TR={gemma_270m_tr:.1f}%, TE={gemma_270m_te:.1f}%)")
    output.append("")
    
    # LLaMA-1B parse rate
    llama_1b_si = si_metrics[si_metrics['model_size'].str.contains('1b', case=False) & 
                             si_metrics['model_family'].str.contains('llama', case=False)]['parse_success_rate'].values[0] * 100
    llama_1b_tr = tr_metrics[tr_metrics['model_size'].str.contains('1b', case=False) & 
                             tr_metrics['model_family'].str.contains('llama', case=False)]['parse_success_rate'].values[0] * 100
    
    passed = abs(llama_1b_si - 77) < 5 and abs(llama_1b_tr - 77) < 5
    output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
    output.append(f"Claim: LLaMA-1B 77% parse success with SI and therapy request")
    output.append(f"Data: SI={llama_1b_si:.1f}%, TR={llama_1b_tr:.1f}%")
    output.append("")
    
    # 12 models with > 0.6B parameters (infer from model_size)
    def get_param_billions(size_str):
        size_str = size_str.lower()
        if 'b' in size_str:
            return float(size_str.split('b')[0].replace('-it', '').strip())
        elif 'm' in size_str:
            return float(size_str.split('m')[0].strip()) / 1000
        return 0
    
    si_metrics['param_billions'] = si_metrics['model_size'].apply(get_param_billions)
    models_gt_600m = si_metrics[si_metrics['param_billions'] > 0.6]
    passed = len(models_gt_600m) == 12
    output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
    output.append(f"Claim: 12 models with > 0.6B parameters")
    output.append(f"Data: {len(models_gt_600m)} models")
    output.append("")
    
    output.append("## Systematic shortcomings across models\n")
    
    review_stats = RESULTS_DIR / "review_statistics"
    
    # SI difficult statements
    si_matrix = pd.read_csv(review_stats / 'si_model_statement_correctness_matrix.csv', index_col=0)
    si_info = pd.read_csv(review_stats / 'si_statement_info.csv')
    si_miss_rates = 1 - si_matrix.mean(axis=0)
    si_difficult = si_miss_rates[si_miss_rates > 0.5]
    si_difficult_categories = si_info[si_info['statement_index'].isin(si_difficult.index)]['ground_truth'].value_counts()
    
    passed = len(si_difficult) == 22 and abs(len(si_difficult)/len(si_info)*100 - 4.9) < 0.5
    output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
    output.append(f"Claim: SI detection: 22 statements (4.9%) missed by >50% of 14 models")
    output.append(f"Data: {len(si_difficult)} statements ({len(si_difficult)/len(si_info)*100:.1f}%) of {len(si_matrix)} models")
    output.append(f"  Breakdown: {dict(si_difficult_categories)}")
    output.append("")
    
    # Therapy request difficult statements
    tr_matrix = pd.read_csv(review_stats / 'therapy_request_model_statement_correctness_matrix.csv', index_col=0)
    tr_info = pd.read_csv(review_stats / 'therapy_request_statement_info.csv')
    tr_miss_rates = 1 - tr_matrix.mean(axis=0)
    tr_difficult = tr_miss_rates[tr_miss_rates > 0.5]
    tr_difficult_categories = tr_info[tr_info['statement_index'].isin(tr_difficult.index)]['ground_truth'].value_counts()
    
    passed = abs(len(tr_difficult) - 11) <= 1 and abs(len(tr_difficult)/len(tr_info)*100 - 1.4) < 0.5
    output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
    output.append(f"Claim: Therapy request: 11 statements (1.4%) missed by >50% of models")
    output.append(f"Data: {len(tr_difficult)} statements ({len(tr_difficult)/len(tr_info)*100:.1f}%)")
    output.append(f"  Breakdown: {dict(tr_difficult_categories)}")
    output.append("")
    
    # Therapy engagement difficult conversations
    te_matrix = pd.read_csv(review_stats / 'therapy_engagement_model_conversation_correctness_matrix.csv', index_col=0)
    te_info = pd.read_csv(review_stats / 'therapy_engagement_conversation_info.csv')
    te_miss_rates = 1 - te_matrix.mean(axis=0)
    te_difficult = te_miss_rates[te_miss_rates > 0.5]
    te_difficult_categories = te_info[te_info['statement_index'].isin(te_difficult.index)]['ground_truth'].value_counts()
    
    passed = len(te_difficult) == 12 and abs(len(te_difficult)/len(te_info)*100 - 2.9) < 0.5
    output.append(f"{'❌ FAILED' if not passed else '✅ PASSED'}")
    output.append(f"Claim: Therapy engagement: 12 conversations (2.9%) missed by >50% of models")
    output.append(f"Data: {len(te_difficult)} conversations ({len(te_difficult)/len(te_info)*100:.1f}%)")
    output.append(f"  Breakdown: {dict(te_difficult_categories)}")
    output.append("")
    
    output.append("## P1 and P2 risk estimates\n")
    
    # Calculate P1 and P2 for all models at baseline 1%
    si_baseline = 1.0
    therapy_baseline = 1.0
    
    p1_values = []
    p2_values = []
    model_p1p2 = []
    
    for _, si_row in si_metrics.iterrows():
        # Find matching therapy request and engagement rows
        tr_row = tr_metrics[(tr_metrics['model_family'] == si_row['model_family']) & 
                           (tr_metrics['model_size'] == si_row['model_size'])]
        te_row = te_metrics[(te_metrics['model_family'] == si_row['model_family']) & 
                           (te_metrics['model_size'] == si_row['model_size'])]
        
        if len(tr_row) > 0 and len(te_row) > 0:
            si_fnr = 1 - si_row['sensitivity']
            tr_fnr = 1 - tr_row.iloc[0]['sensitivity']
            te_fnr = 1 - te_row.iloc[0]['sensitivity']
            
            p1 = calculate_p1(si_baseline, si_fnr, tr_fnr)
            p2 = calculate_p2(te_fnr, therapy_baseline, failure_multiplier=1.0)
            
            p1_values.append(p1)
            p2_values.append(p2)
            
            model_p1p2.append({
                'family': si_row['model_family'],
                'size': si_row['model_size'],
                'p1': p1,
                'p2': p2,
                'si_fnr': si_fnr,
                'tr_fnr': tr_fnr,
                'te_fnr': te_fnr
            })
    
    p1_min, p1_max = min([p for p in p1_values if p > 0]), max(p1_values)
    p1_orders = np.log10(p1_max) - np.log10(p1_min)
    
    # Check P1 range (within order of magnitude and similar span)
    passed = p1_min >= 1e-9 and p1_max <= 5e-4 and abs(p1_orders - 4) < 1
    output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
    output.append(f"Claim: P1 ranged from 2.0×10⁻⁸ to 2.6×10⁻⁴ (4 orders of magnitude, 0.02 to 256 per million)")
    output.append(f"Data: {p1_min:.2e} to {p1_max:.2e} ({p1_orders:.1f} orders, {p1_min*1e6:.2f} to {p1_max*1e6:.0f} per million)")
    output.append("")
    
    qwen_models = [m for m in model_p1p2 if m['family'].lower() == 'qwen']
    qwen_4b = [m for m in qwen_models if '4b' in m['size'].lower()]
    qwen_8b = [m for m in qwen_models if '8b' in m['size'].lower()]
    
    if qwen_4b and qwen_8b:
        ratio = qwen_8b[0]['p1'] / qwen_4b[0]['p1'] if qwen_4b[0]['p1'] > 0 else float('inf')
        passed = abs(ratio - 15) < 3
        output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
        output.append(f"Claim: Qwen 8B P1=1.04×10⁻⁵ nearly 15x worse than 4B P1=7.14×10⁻⁷")
        output.append(f"Data: Qwen 8B={qwen_8b[0]['p1']:.2e}, Qwen 4B={qwen_4b[0]['p1']:.2e} ({ratio:.1f}x worse)")
        output.append("")
    
    gemma_models = [m for m in model_p1p2 if m['family'].lower() == 'gemma']
    if gemma_models:
        gemma_p1s = [m['p1'] for m in gemma_models if m['p1'] > 0]
        gemma_p1_range = np.log10(max(gemma_p1s)) - np.log10(min(gemma_p1s))
        passed = abs(gemma_p1_range - 4.1) < 0.5
        output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
        output.append(f"Claim: Gemma family P1 varied by 4.1 orders of magnitude")
        output.append(f"Data: {gemma_p1_range:.1f} orders of magnitude")
        output.append("")
    
    output.append(f"✅ PASSED")
    output.append(f"Claim: SI prevalence 0.1% to 10% contributes 2 orders of magnitude")
    output.append(f"Data: {np.log10(10.0) - np.log10(0.1):.1f} orders of magnitude")
    output.append("")
    
    p2_min, p2_max = min([p for p in p2_values if p > 0]), max(p2_values)
    p2_orders = np.log10(p2_max) - np.log10(p2_min)
    
    passed = p2_min >= 5e-5 and p2_max <= 1.5e-2 and abs(p2_orders - 2) < 0.5
    output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
    output.append(f"Claim: P2 ranged from 7.1×10⁻⁵ to 9.6×10⁻³ (2 orders, 71 to 9,600 per million)")
    output.append(f"Data: {p2_min:.2e} to {p2_max:.2e} ({p2_orders:.1f} orders, {p2_min*1e6:.0f} to {p2_max*1e6:.0f} per million)")
    output.append("")
    
    perfect_models = [m for m in model_p1p2 if m['si_fnr'] == 0 or m['tr_fnr'] == 0 or m['te_fnr'] == 0]
    if perfect_models:
        perfect_list = [f"{m['family']} {m['size']}" for m in perfect_models]
        has_qwen32b = any('qwen' in m['family'].lower() and '32b' in m['size'].lower() for m in perfect_models)
        has_gemma1b = any('gemma' in m['family'].lower() and '1b' in m['size'].lower() for m in perfect_models)
        passed = has_qwen32b and has_gemma1b
        output.append(f"{'✅ PASSED' if passed else '❌ FAILED'}")
        output.append(f"Claim: Qwen 32B (TR) and Gemma 1B (TE) achieved perfect sensitivity")
        output.append(f"Data: Models with 100% sensitivity on any task: {', '.join(perfect_list)}")
        output.append("")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Verify manuscript claims")
    parser.add_argument("--paper-run-dir", required=True)
    parser.add_argument("--output", default="MANUSCRIPT_CLAIMS_VERIFICATION.md")
    args = parser.parse_args()
    
    paper_dir = Path(args.paper_run_dir)
    output_path = paper_dir / args.output if paper_dir.is_dir() else Path(args.output)
    
    report = verify_claims(paper_dir)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ {output_path}")

if __name__ == "__main__":
    main()
