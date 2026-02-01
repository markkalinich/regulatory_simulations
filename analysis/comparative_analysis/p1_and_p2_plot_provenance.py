#!/usr/bin/env python3
"""
P1, P2, P_harm Risk Analysis Facet Plot

Generates a 3×3 facet plot showing:
- Row 1 (P1): Probability of Hazard → Hazardous Situation
- Row 2 (P2): Probability of Hazardous Situation → Harm  
- Row 3 (P_harm): Combined probability of Hazard → Harm

Failure Multiplier (m) for conditional dependence:
Models the conditional dependence of engagement detection failure on prior failures
(SI detection and therapy request detection) using a power transformation:
    FNR_adjusted = 1 - (1 - FNR_observed)^m

- m = 1: Independent failures. P2 uses observed FNR.
- m > 1: FNR is approximately m× higher (for small FNR), modeling that prior
         failures make subsequent failures more likely.
- m → ∞: FNR → 1, representing certain failure given prior failures.

For small FNR values, this approximates multiplicative scaling: m=2 ≈ 2× FNR.
For larger FNR, the transformation approaches 1.0 (certain failure).

P2 presupposes we are IN the hazardous situation, so P(therapeutic) = 1.
P2 = FNR_adjusted × P(fail help) × P(lack care → harm)

Where P(lack care → harm) is varied on the x-axis (analogous to SI% for P1).
P_harm = P1 × P2

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from pathlib import Path
import sys
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add project root for config imports
from utilities.figure_provenance import FigureProvenanceTracker
from utilities.input_validation import validate_dataframe, RISK_ANALYSIS_REQUIRED, InputValidationError
from config.regulatory_paper_parameters import RISK_MODEL_PARAMS

# =============================================================================
# CONFIGURABLE PARAMETERS (imported from centralized config)
# =============================================================================
EMPIRICAL_DATA = {
    'therapy_request': {
        'k': RISK_MODEL_PARAMS['empirical_therapy_request_k'],
        'n': RISK_MODEL_PARAMS['empirical_therapy_request_n']
    },
    'model_comply': {
        'k': RISK_MODEL_PARAMS['empirical_model_comply_k'],
        'n': RISK_MODEL_PARAMS['empirical_model_comply_n']
    },
}

# Default values for assumed parameters (can be overridden via command line)
# Imported from centralized config for consistency
DEFAULT_PARAMS = {
    'therapy_request_rate': RISK_MODEL_PARAMS['therapy_request_rate'],
    'model_comply_rate': RISK_MODEL_PARAMS['model_comply_rate'],
    'prob_fail_seek_help': RISK_MODEL_PARAMS['prob_fail_seek_help'],
    'lod_ci_level': RISK_MODEL_PARAMS['lod_ci_level'],
    'failure_multiplier': RISK_MODEL_PARAMS['default_failure_multiplier'],
    # Monte Carlo error propagation settings
    'n_mc_samples': RISK_MODEL_PARAMS['n_mc_samples'],
    'prior_alpha': RISK_MODEL_PARAMS['prior_alpha'],
    'random_state': 42,              # Fixed seed for reproducibility
    'uncertainty_style': 'both',     # Plotting decision - kept local
    'sample_params': RISK_MODEL_PARAMS['mc_sample_empirical_rates'],  # Legacy name for backward compat
}

# Limit of Detection CI level (can be overridden via command line)
LOD_CI_LEVEL = DEFAULT_PARAMS['lod_ci_level']

def calculate_lod(n_samples, ci_level=LOD_CI_LEVEL):
    """
    Calculate Limit of Detection for FNR using Clopper-Pearson exact method.
    
    Given 0 failures observed in n_samples tests, returns the upper bound
    of the confidence interval for the true failure rate.
    
    Args:
        n_samples: Number of tests conducted
        ci_level: Confidence level (0.50 = median, 0.95 = 95% upper bound)
    
    Returns:
        Upper bound for true FNR (used as floor for point estimates)
    """
    # For k=0 failures, upper bound is Beta.ppf(ci_level, 1, n)
    return stats.beta.ppf(ci_level, 1, n_samples)

# Initialize provenance tracker
# Note: .parent.parent.parent gets us from analysis/comparative_analysis/ to project root
tracker = FigureProvenanceTracker(
    figure_name="p1_p2_risk_analysis",
    base_dir=Path(__file__).parent.parent.parent / "results" / "risk_analysis"
)

def load_models_config():
    """Load models config to get param_billions and normalized family names."""
    config_path = Path(__file__).parent.parent.parent / "config" / "models_config.csv"
    return pd.read_csv(config_path)


def load_experiment_metrics(csv_path):
    """Load sensitivity metrics from comprehensive_metrics.csv
    
    Returns dict with:
        - sensitivity: observed sensitivity (point estimate)
        - fnr: observed false negative rate (1 - sensitivity)
        - fn: number of false negatives (k for Beta posterior)
        - total_positive: number of positive test cases (n for Beta posterior)
        
    Raises:
        InputValidationError: If input data fails validation
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns for risk analysis
    validate_dataframe(
        df,
        required_columns=RISK_ANALYSIS_REQUIRED,
        source_file=str(csv_path),
        check_empty=True,
        check_nan_in_columns=['sensitivity', 'fn', 'total_positive'],
    )
    
    metrics = {}
    for _, row in df.iterrows():
        key = (row['model_family'], row['model_size'])
        metrics[key] = {
            'sensitivity': row['sensitivity'],
            'fnr': 1 - row['sensitivity'],
            'fn': int(row['fn']),                    # k = number of failures
            'total_positive': int(row['total_positive'])  # n = total positive cases
        }
    return metrics


def get_sample_size_from_metrics(metrics):
    """Extract sample size (total_positive) from loaded metrics.
    
    All models are evaluated on the same dataset, so total_positive should be
    identical across all models. We take the first one.
    
    Args:
        metrics: Dict returned by load_experiment_metrics()
        
    Returns:
        int: Total positive cases in the dataset
    """
    if not metrics:
        raise ValueError("No metrics loaded - cannot determine sample size")
    first_model = next(iter(metrics.values()))
    return first_model['total_positive']


def sample_fnr_from_posterior(fn, total_positive, n_samples, prior_alpha=1.0, random_state=None):
    """
    Sample FNR from Beta posterior distribution.
    
    Given k failures (false negatives) out of n positive tests, the posterior
    for FNR with Beta(alpha, alpha) prior is:
        Beta(alpha + k, alpha + (n - k))
    
    Args:
        fn: Number of false negatives (k)
        total_positive: Total positive test cases (n)
        n_samples: Number of MC samples to draw
        prior_alpha: Prior parameter (1.0 = uniform, 0.5 = Jeffreys)
        random_state: Random seed for reproducibility
    
    Returns:
        Array of n_samples FNR values sampled from posterior
    
    Raises:
        ValueError: If inputs are invalid (fn < 0, fn > total_positive, total_positive <= 0)
    """
    # Validate inputs
    if total_positive <= 0:
        raise ValueError(f"total_positive must be > 0, got {total_positive}")
    if fn < 0:
        raise ValueError(f"fn (false negatives) must be >= 0, got {fn}")
    if fn > total_positive:
        raise ValueError(f"fn ({fn}) cannot exceed total_positive ({total_positive})")
    
    tp = total_positive - fn  # true positives = n - k
    alpha_posterior = prior_alpha + fn          # alpha + k (failures)
    beta_posterior = prior_alpha + tp           # alpha + (n - k) (successes)
    return stats.beta.rvs(alpha_posterior, beta_posterior, size=n_samples, random_state=random_state)


def monte_carlo_risk_estimation(si_metrics, therapy_request_metrics, therapy_engagement_metrics,
                                 si_prevalence_pct, harm_given_no_care_pct, params):
    """
    Estimate P1, P2, P_harm with uncertainty via Monte Carlo sampling.
    
    Samples FNRs from Beta posteriors and optionally samples assumed parameters
    from their empirical distributions, propagating all uncertainty through
    the risk calculation.
    
    Args:
        si_metrics: Dict with 'fn', 'total_positive' for SI detection
        therapy_request_metrics: Dict with 'fn', 'total_positive' for therapy request
        therapy_engagement_metrics: Dict with 'fn', 'total_positive' for therapy engagement
        si_prevalence_pct: Baseline SI prevalence percentage (x-axis for P1)
        harm_given_no_care_pct: P(lack of care leads to harm) percentage (x-axis for P2)
        params: Analysis parameters dict
    
    Returns:
        Dict with 'p1', 'p2', 'p_harm' each containing 'median', 'ci_5', 'ci_95'
    """
    # Validate empirical data
    for key in ['therapy_request', 'model_comply']:
        d = EMPIRICAL_DATA[key]
        if d['n'] <= 0 or d['k'] < 0 or d['k'] > d['n']:
            raise ValueError(f"Invalid empirical data for {key}: k={d['k']}, n={d['n']}")
    
    n_samples = params['n_mc_samples']
    prior_alpha = params['prior_alpha']
    failure_multiplier = params['failure_multiplier']
    sample_params = params.get('sample_params', True)
    
    # Sample FNRs from Beta posteriors
    si_fnr_samples = sample_fnr_from_posterior(
        si_metrics['fn'], si_metrics['total_positive'], n_samples, prior_alpha, params.get('random_state'))
    tr_fnr_samples = sample_fnr_from_posterior(
        therapy_request_metrics['fn'], therapy_request_metrics['total_positive'], n_samples, prior_alpha, params.get('random_state'))
    te_fnr_samples = sample_fnr_from_posterior(
        therapy_engagement_metrics['fn'], therapy_engagement_metrics['total_positive'], n_samples, prior_alpha, params.get('random_state'))
    
    # Sample assumed parameters from empirical posteriors (or use point estimates)
    if sample_params:
        d = EMPIRICAL_DATA['therapy_request']
        therapy_rate_samples = stats.beta.rvs(prior_alpha + d['k'], prior_alpha + d['n'] - d['k'], size=n_samples, random_state=params.get('random_state'))
        
        d = EMPIRICAL_DATA['model_comply']
        comply_rate_samples = stats.beta.rvs(prior_alpha + d['k'], prior_alpha + d['n'] - d['k'], size=n_samples, random_state=params.get('random_state'))
        
        fail_help_samples = np.full(n_samples, params['prob_fail_seek_help'])  # No data, fixed
    else:
        therapy_rate_samples = np.full(n_samples, params['therapy_request_rate'])
        comply_rate_samples = np.full(n_samples, params['model_comply_rate'])
        fail_help_samples = np.full(n_samples, params['prob_fail_seek_help'])
    
    # Calculate P1 for each sample
    p_suicidal = si_prevalence_pct / 100.0
    p1_samples = (p_suicidal * 
                  si_fnr_samples * 
                  therapy_rate_samples * 
                  tr_fnr_samples * 
                  comply_rate_samples)
    
    # Calculate P2 for each sample
    # FNR adjusted via power transformation: approximates m× FNR for small FNR,
    # gracefully saturates toward 1.0 for larger FNR or high multiplier
    sensitivity_samples = 1 - te_fnr_samples
    adjusted_sensitivity = sensitivity_samples ** failure_multiplier
    p_miss_therapeutic = 1 - adjusted_sensitivity
    p_lack_care_harm = harm_given_no_care_pct / 100.0
    p2_samples = (p_miss_therapeutic * 
                  fail_help_samples * 
                  p_lack_care_harm)
    
    # Calculate P_harm for each sample
    p_harm_samples = p1_samples * p2_samples
    
    # Return percentiles
    return {
        'p1': {
            'median': np.median(p1_samples),
            'ci_5': np.percentile(p1_samples, 5),
            'ci_95': np.percentile(p1_samples, 95),
            'samples': p1_samples  # Keep for potential further analysis
        },
        'p2': {
            'median': np.median(p2_samples),
            'ci_5': np.percentile(p2_samples, 5),
            'ci_95': np.percentile(p2_samples, 95),
            'samples': p2_samples
        },
        'p_harm': {
            'median': np.median(p_harm_samples),
            'ci_5': np.percentile(p_harm_samples, 5),
            'ci_95': np.percentile(p_harm_samples, 95),
            'samples': p_harm_samples
        }
    }

def calculate_p1(si_percentage, suicide_fnr, therapy_request_fnr, 
                therapy_request_rate=0.029, model_comply_rate=1.0):
    """
    Calculate P1: Probability of Hazard → Hazardous Situation
    
    P1 = P(patient suicidal) × P(suicide detection miss) × P(requests therapy) × 
         P(therapy request detection miss) × P(model complies)
    """
    p_suicidal = si_percentage / 100.0
    p_suicide_miss = suicide_fnr
    p_therapy_request = therapy_request_rate
    p_therapy_miss = therapy_request_fnr
    p_comply = model_comply_rate
    
    p1 = p_suicidal * p_suicide_miss * p_therapy_request * p_therapy_miss * p_comply
    
    return p1

def calculate_p2(therapy_detection_fnr, prob_lack_care_harm_percentage,
                prob_fail_seek_help=1.0, failure_multiplier=1.0):
    """
    Calculate P2: Probability of Hazardous Situation → Harm
    
    P2 presupposes we are already IN the hazardous situation (therapeutic interaction
    is happening), so P(therapeutic) = 1.
    
    Uses power transformation: FNR_adjusted = 1 - (1 - FNR_observed)^m
    
    Args:
        therapy_detection_fnr: Observed false negative rate for engagement detection
        prob_lack_care_harm_percentage: P(lack of care leads to harm) as % (x-axis)
        prob_fail_seek_help: P(user fails to seek qualified help) - default 1.0 worst case
        failure_multiplier: FNR multiplier (1 = independent, higher = correlated failures)
            - m=1: FNR unchanged (independent failures)
            - m=2: FNR approximately doubles (for small FNR)
            - m→∞: FNR → 1 (certain failure given prior failures)
    """
    # Power transformation: approximates m× FNR for small FNR, saturates toward 1.0
    sensitivity = 1 - therapy_detection_fnr
    adjusted_sensitivity = sensitivity ** failure_multiplier
    p_miss_therapeutic = 1 - adjusted_sensitivity
    p_fail_help = prob_fail_seek_help
    p_lack_care_harm = prob_lack_care_harm_percentage / 100.0
    
    p2 = p_miss_therapeutic * p_fail_help * p_lack_care_harm
    
    return p2


def calculate_p_harm(p1, p2):
    """
    Calculate P_harm: Overall probability of Hazard → Harm
    
    P_harm = P1 × P2
    
    Note: The failure multiplier is incorporated directly into P2,
    so P_harm is simply the product of the two probabilities.
    
    Args:
        p1: Probability of Hazard → Hazardous Situation
        p2: Probability of Hazardous Situation → Harm (already incorporates failure_multiplier)
    
    Returns:
        P_harm: Combined probability of harm
    """
    p_harm = p1 * p2
    
    return p_harm

def prepare_plot_data(suicide_metrics, therapy_request_metrics, therapy_engagement_metrics, 
                      params=None):
    """Prepare data for plotting P1, P2, and P_harm risk vs baseline percentages
    
    Uses Monte Carlo sampling from Beta posteriors to propagate uncertainty
    from binomial FNR estimates through the risk calculation.
    
    Args:
        suicide_metrics: Dict of (family, size) -> metrics for SI detection
        therapy_request_metrics: Dict of (family, size) -> metrics for therapy request
        therapy_engagement_metrics: Dict of (family, size) -> metrics for therapy engagement
        params: Dict of analysis parameters (uses DEFAULT_PARAMS if not provided)
    """
    # Use defaults for any missing parameters
    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        # Merge with defaults to ensure all params are present
        merged = DEFAULT_PARAMS.copy()
        merged.update(params)
        params = merged
    
    # Extract parameters
    n_mc_samples = params['n_mc_samples']
    prior_alpha = params['prior_alpha']
    use_mc = n_mc_samples > 0
    
    # Load config to get param_billions for each model
    config_df = load_models_config()
    
    suicide_models = set(suicide_metrics.keys())
    therapy_request_models = set(therapy_request_metrics.keys())
    therapy_engagement_models = set(therapy_engagement_metrics.keys())
    common_models = suicide_models.intersection(therapy_request_models).intersection(therapy_engagement_models)
    
    print(f"Found {len(common_models)} models in all three experiments")
    print(f"Parameters: failure_multiplier={params['failure_multiplier']}, therapy_request_rate={params['therapy_request_rate']}, "
          f"model_comply_rate={params['model_comply_rate']}, prob_fail_seek_help={params['prob_fail_seek_help']}")
    if use_mc:
        prior_name = "Jeffreys" if prior_alpha == 0.5 else "Uniform" if prior_alpha == 1.0 else f"Beta({prior_alpha},{prior_alpha})"
        print(f"Monte Carlo: {n_mc_samples} samples, {prior_name} prior (α={prior_alpha})")
    else:
        print("Monte Carlo: DISABLED (point estimates only)")
    
    plot_data = []
    baseline_percentages = [0.1] + list(range(1, 11))
    
    for model_family, model_size in common_models:
        # Get metrics for this model
        si_m = suicide_metrics[(model_family, model_size)]
        tx_req_m = therapy_request_metrics[(model_family, model_size)]
        tx_eng_m = therapy_engagement_metrics[(model_family, model_size)]
        
        # Look up param_billions from config
        config_match = config_df[(config_df['family'] == model_family) & (config_df['size'] == model_size)]
        
        # Fallback 1: Try prefix matching on size (e.g., "1b" matches "1b-it")
        if len(config_match) == 0:
            base_size = model_size.split('-')[0]
            config_match = config_df[(config_df['family'] == model_family) & (config_df['size'].str.startswith(base_size))]
        
        # Fallback 2: For llama family, try matching any llama variant
        if len(config_match) == 0 and model_family == 'llama':
            base_size = model_size.split('-')[0]
            llama_families = config_df[config_df['family'].str.startswith('llama')]
            config_match = llama_families[llama_families['size'].str.startswith(base_size)]
        
        if len(config_match) > 0:
            param_billions = float(config_match.iloc[0]['param_billions'])
        else:
            print(f"  Warning: No config found for {model_family}/{model_size}, skipping")
            continue
        
        for baseline_pct in baseline_percentages:
            base_data = {
                'model_family': model_family,
                'model_size': model_size,
                'param_billions': param_billions,
                'baseline_percentage': baseline_pct,
                # Store raw counts for provenance
                'si_fn': si_m['fn'],
                'si_n': si_m['total_positive'],
                'tx_req_fn': tx_req_m['fn'],
                'tx_req_n': tx_req_m['total_positive'],
                'tx_eng_fn': tx_eng_m['fn'],
                'tx_eng_n': tx_eng_m['total_positive'],
            }
            
            if use_mc:
                # Monte Carlo estimation with uncertainty
                # Pass baseline_pct for both parameters (same x-axis value for visual comparison)
                mc_results = monte_carlo_risk_estimation(
                    si_m, tx_req_m, tx_eng_m, baseline_pct, baseline_pct, params)
                
                for risk_type in ['p1', 'p2', 'p_harm']:
                    plot_data.append({
                        **base_data, 
                        'risk_probability': mc_results[risk_type]['median'],
                        'risk_ci_5': mc_results[risk_type]['ci_5'],
                        'risk_ci_95': mc_results[risk_type]['ci_95'],
                        'risk_type': risk_type.upper() if risk_type != 'p_harm' else 'P_harm'
                    })
            else:
                # Point estimates (backward compatibility)
                p1 = calculate_p1(baseline_pct, si_m['fnr'], tx_req_m['fnr'],
                                therapy_request_rate=params['therapy_request_rate'],
                                model_comply_rate=params['model_comply_rate'])
                p2 = calculate_p2(tx_eng_m['fnr'], baseline_pct, 
                                prob_fail_seek_help=params['prob_fail_seek_help'], 
                                failure_multiplier=params['failure_multiplier'])
                p_harm = calculate_p_harm(p1, p2)
                
                for risk_type, risk_val in [('P1', p1), ('P2', p2), ('P_harm', p_harm)]:
                    plot_data.append({
                        **base_data, 
                        'risk_probability': risk_val,
                        'risk_ci_5': risk_val,  # No uncertainty
                        'risk_ci_95': risk_val,
                        'risk_type': risk_type
                    })
    
    return pd.DataFrame(plot_data)

def get_alpha_for_param_billions(param_billions, family_data):
    """Calculate alpha transparency based on param_billions within family"""
    family_sizes = family_data['param_billions'].unique()
    
    min_size = min(family_sizes)
    max_size = max(family_sizes)
    
    if max_size == min_size:
        return 1.0
    
    alpha = 0.25 + 0.75 * (param_billions - min_size) / (max_size - min_size)
    return alpha


def normalize_family(family):
    """Normalize family names (llama2, llama3.1, etc. → llama)"""
    family_lower = family.lower()
    if family_lower.startswith('llama'):
        return 'llama'
    elif family_lower.startswith('qwen'):
        return 'qwen'
    elif family_lower.startswith('gemma'):
        return 'gemma'
    return family_lower


def format_size_label(param_billions):
    """Format param_billions for legend display (e.g., 0.27 → '0.27B', 70 → '70B')"""
    if param_billions < 1:
        return f'{param_billions:.2f}B'
    elif param_billions == int(param_billions):
        return f'{int(param_billions)}B'
    else:
        return f'{param_billions}B'

def create_p1_p2_risk_plot(suicide_csv, therapy_request_csv, therapy_engagement_csv, 
                           figsize=(20, 24), log_y=True, params=None):
    """Create P1, P2, and P_harm risk analysis facet plot with provenance tracking
    
    Args:
        suicide_csv: Path to SI detection comprehensive_metrics.csv
        therapy_request_csv: Path to therapy request comprehensive_metrics.csv
        therapy_engagement_csv: Path to therapy engagement comprehensive_metrics.csv
        figsize: Figure size tuple (width, height)
        log_y: Use logarithmic y-axis (default True)
        params: Dict of analysis parameters (uses DEFAULT_PARAMS if not provided)
    """
    # Use defaults for any missing parameters
    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        # Merge with defaults
        merged = DEFAULT_PARAMS.copy()
        merged.update(params)
        params = merged
    
    failure_multiplier = params['failure_multiplier']
    
    # Track input datasets
    tracker.add_input_dataset(
        suicide_csv,
        description="Suicide ideation detection comprehensive metrics",
        columns_used=['model_family', 'model_size', 'sensitivity']
    )
    
    tracker.add_input_dataset(
        therapy_request_csv,
        description="Therapy request detection comprehensive metrics",
        columns_used=['model_family', 'model_size', 'sensitivity']
    )
    
    tracker.add_input_dataset(
        therapy_engagement_csv,
        description="Therapy engagement detection comprehensive metrics",
        columns_used=['model_family', 'model_size', 'sensitivity']
    )
    
    title_size = 28
    tick_size = 22
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': title_size,
        'axes.labelsize': 24,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'legend.fontsize': 16
    })
    
    suicide_metrics = load_experiment_metrics(suicide_csv)
    therapy_request_metrics = load_experiment_metrics(therapy_request_csv)
    therapy_engagement_metrics = load_experiment_metrics(therapy_engagement_csv)
    
    # Calculate sample sizes dynamically from loaded data (not hardcoded)
    sample_sizes = {
        'suicidal_ideation': get_sample_size_from_metrics(suicide_metrics),
        'therapy_request': get_sample_size_from_metrics(therapy_request_metrics),
        'therapy_engagement': get_sample_size_from_metrics(therapy_engagement_metrics)
    }
    
    plot_data = prepare_plot_data(suicide_metrics, therapy_request_metrics, therapy_engagement_metrics, params)
    
    print(f"Prepared {len(plot_data)} data points for plotting")
    
    # Add normalized family column for filtering
    plot_data['normalized_family'] = plot_data['model_family'].apply(normalize_family)
    
    # Save computed P1/P2/P_harm values to CSV for reproducibility and verification
    output_name_base = params.get('output_name', 'figure_5')
    csv_filename = f"p1_p2_p_harm_values_m_{failure_multiplier}.csv"
    csv_output_path = tracker.get_output_path(csv_filename)
    plot_data.to_csv(csv_output_path, index=False)
    tracker.add_output_file(csv_output_path, file_type="data", 
                           description="Computed P1, P2, P_harm values for all models and baseline percentages")
    print(f"  Saved computed values: {csv_filename}")
    
    # Determine number of rows based on include_p_harm parameter
    include_p_harm = params.get('include_p_harm', False)
    if include_p_harm:
        n_rows = 3
        risk_types = ['P1', 'P2', 'P_harm']
        # Adjust figsize for 3 rows if using default 2-row size
        if figsize == (20, 16):
            figsize = (20, 24)
    else:
        n_rows = 2
        risk_types = ['P1', 'P2']
    
    # Create subplot grid: n_rows × 3 columns (model families)
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize, sharey='row')
    family_colors = {'gemma': '#1f77b4', 'qwen': '#ff7f0e', 'llama': '#2ca02c'}
    family_display_names = {'gemma': 'Gemma', 'qwen': 'Qwen', 'llama': 'LLaMA'}
    
    risk_labels = {
        'P1': 'P$_1$ (Hazard → Hazardous Situation)',
        'P2': 'P$_2$ (Hazardous Situation → Harm)',
        'P_harm': f'P$_{{harm}}$ (Hazard → Harm, m={failure_multiplier})'
    }
    baseline_labels = {
        'P1': 'SI % in User Base', 
        'P2': 'P(Lack of Care → Harm) %',
        'P_harm': 'Baseline Prevalence %'
    }
    
    # Plot each risk type (P1, P2, P_harm) in rows and each family in columns
    for row_idx, risk_type in enumerate(risk_types):
        for col_idx, family in enumerate(['gemma', 'qwen', 'llama']):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this risk type and normalized family
            risk_family_data = plot_data[
                (plot_data['risk_type'] == risk_type) & 
                (plot_data['normalized_family'] == family)
            ]
            
            if len(risk_family_data) == 0:
                ax.set_title(family_display_names[family])
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get unique param_billions values, sorted
            unique_sizes = risk_family_data[['param_billions']].drop_duplicates().sort_values('param_billions')
            
            # Check if we have uncertainty data
            has_uncertainty = 'risk_ci_5' in risk_family_data.columns and \
                            (risk_family_data['risk_ci_5'] != risk_family_data['risk_probability']).any()
            
            # Get uncertainty style from params
            uncertainty_style = params.get('uncertainty_style', 'both')
            
            # Plot each model size with different alpha, ordered by param_billions
            for _, size_row in unique_sizes.iterrows():
                param_b = size_row['param_billions']
                model_data = risk_family_data[risk_family_data['param_billions'] == param_b].sort_values('baseline_percentage')
                alpha = get_alpha_for_param_billions(param_b, risk_family_data)
                size_label = format_size_label(param_b)
                
                x = model_data['baseline_percentage'].values
                y = model_data['risk_probability'].values
                
                if has_uncertainty and uncertainty_style != 'none':
                    ci_5 = model_data['risk_ci_5'].values
                    ci_95 = model_data['risk_ci_95'].values
                    yerr_lower = y - ci_5
                    yerr_upper = ci_95 - y
                    
                    # Draw ribbon if requested
                    if uncertainty_style in ('ribbon', 'both'):
                        ax.fill_between(x, ci_5, ci_95,
                                       color=family_colors[family], alpha=alpha * 0.15)
                    
                    # Draw error bars if requested (black for visibility)
                    if uncertainty_style in ('errorbar', 'both'):
                        ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper],
                                   marker='o', linestyle='-', color=family_colors[family], 
                                   alpha=alpha, linewidth=2, markersize=6, 
                                   capsize=3, capthick=1, elinewidth=1,
                                   ecolor='black',  # Black error bars
                                   label=size_label)
                    elif uncertainty_style == 'ribbon':
                        # Just the line on top of ribbon
                        ax.plot(x, y, marker='o', linestyle='-', color=family_colors[family], 
                               alpha=alpha, linewidth=2, markersize=6, label=size_label)
                else:
                    # No uncertainty or style is 'none' - just plot median line
                    ax.plot(x, y, marker='o', linestyle='-', color=family_colors[family], 
                           alpha=alpha, linewidth=2, markersize=8, label=size_label)
            
            # Set titles and labels
            if row_idx == 0:  # Top row
                ax.set_title(family_display_names[family])
            
            if col_idx == 0:  # Leftmost column
                ax.set_ylabel(risk_labels[risk_type])
            
            # Both rows need x-axis labels since they're different
            ax.set_xlabel(baseline_labels[risk_type])
            
            ax.grid(True, alpha=0.3)
            
            # Legend is already sorted by param_billions from plotting order
            ax.legend(title='Model Size', loc='lower right')
            
            # Set y-axis scale with FIXED limits for consistency across multiplier values
            if log_y:
                ax.set_yscale('log')
                # Fixed y-axis limits for each risk type to enable comparison across plots
                # Limits chosen to ensure no clipping across all scenarios
                if risk_type == 'P1':
                    ax.set_ylim(1e-9, 1e-2)  # Max ~2.6e-3 with worst case
                elif risk_type == 'P2':
                    ax.set_ylim(1e-5, 1)  # Max ~0.1 with high m and 10% baseline
                elif risk_type == 'P_harm':
                    ax.set_ylim(1e-12, 1e-3)  # Max ~2.6e-4 (P1_max × P2_max)
            else:
                risk_data = plot_data[plot_data['risk_type'] == risk_type]
                ax.set_ylim(0, max(risk_data['risk_probability']) * 1.1)
    
    # Set analysis parameters (use actual values from params dict)
    lod_ci = params['lod_ci_level']
    prior_name = "Jeffreys" if params['prior_alpha'] == 0.5 else "Uniform" if params['prior_alpha'] == 1.0 else f"Beta({params['prior_alpha']},{params['prior_alpha']})"
    tracker.set_analysis_parameters(
        therapy_request_rate=params['therapy_request_rate'],
        model_comply_rate=params['model_comply_rate'],
        prob_fail_seek_help=params['prob_fail_seek_help'],
        failure_multiplier=params['failure_multiplier'],
        lod_ci_level=lod_ci,
        lod_si=calculate_lod(sample_sizes['suicidal_ideation'], lod_ci),
        lod_therapy_request=calculate_lod(sample_sizes['therapy_request'], lod_ci),
        lod_therapy_engagement=calculate_lod(sample_sizes['therapy_engagement'], lod_ci),
        sample_sizes=sample_sizes,  # Dynamically calculated from input data
        n_mc_samples=params['n_mc_samples'],
        prior_alpha=params['prior_alpha'],
        prior_type=prior_name,
        baseline_percentages=[0.1] + list(range(1, 11)),
        log_y_scale=log_y,
        figsize=figsize,
        model_families=['gemma', 'qwen', 'llama'],
        risk_types=risk_types,
        include_p_harm=include_p_harm
    )
    
    plt.tight_layout()
    
    # Save outputs using provenance tracker
    output_name = params.get('output_name', 'figure_5')
    output_filename = f"{output_name}.png"
    
    output_png = tracker.get_output_path(output_filename)
    plt.savefig(str(output_png), dpi=300, bbox_inches='tight')
    tracker.add_output_file(output_png, file_type="figure")
    
    print(f"  Saved PNG: {output_png}")
    
    # Save provenance metadata
    tracker.save_provenance()
    
    if include_p_harm:
        print(f"✅ P1, P2, P_harm Risk Analysis plot saved with provenance tracking")
    else:
        print(f"✅ P1, P2 Risk Analysis plot saved with provenance tracking")
    print(f"   Failure multiplier m = {failure_multiplier}")
    print(f"   Output directory: {tracker.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create P1, P2, and P_harm risk analysis facet plot with provenance')
    parser.add_argument('--suicide-csv', required=True,
                       help='Path to suicide ideation comprehensive_metrics.csv')
    parser.add_argument('--therapy-request-csv', required=True,
                       help='Path to therapy request comprehensive_metrics.csv')
    parser.add_argument('--therapy-engagement-csv', required=True,
                       help='Path to therapy engagement comprehensive_metrics.csv')
    parser.add_argument('--figsize', nargs=2, type=float, default=[20, 16],
                       help='Figure size (width height). Default: 20 16 for 2-row layout, use 20 24 with --include-p-harm')
    parser.add_argument('--linear-y', action='store_true',
                       help='Use linear y-axis instead of logarithmic (default: logarithmic)')
    
    # Configurable analysis parameters (for sensitivity analysis)
    parser.add_argument('--failure-multiplier', type=float, default=DEFAULT_PARAMS['failure_multiplier'],
                       help=f"FNR multiplier for conditional dependence. "
                            f"m=1: independent failures (default), m>1: FNR ≈ m× higher, m→∞: certain failure. "
                            f"(default: {DEFAULT_PARAMS['failure_multiplier']})")
    parser.add_argument('--therapy-request-rate', type=float, 
                       default=DEFAULT_PARAMS['therapy_request_rate'],
                       help=f"P(requests therapy | suicidal) - from literature. "
                            f"(default: {DEFAULT_PARAMS['therapy_request_rate']})")
    parser.add_argument('--model-comply-rate', type=float,
                       default=DEFAULT_PARAMS['model_comply_rate'],
                       help=f"P(model complies with therapy request). "
                            f"(default: {DEFAULT_PARAMS['model_comply_rate']})")
    parser.add_argument('--prob-fail-seek-help', type=float,
                       default=DEFAULT_PARAMS['prob_fail_seek_help'],
                       help=f"P(user fails to seek qualified help elsewhere). "
                            f"(default: {DEFAULT_PARAMS['prob_fail_seek_help']})")
    parser.add_argument('--lod-ci-level', type=float,
                       default=DEFAULT_PARAMS['lod_ci_level'],
                       help=f"Confidence level for LOD (0.5=median, 0.95=conservative). "
                            f"(default: {DEFAULT_PARAMS['lod_ci_level']})")
    
    # Monte Carlo error propagation settings
    parser.add_argument('--n-mc-samples', type=int,
                       default=DEFAULT_PARAMS['n_mc_samples'],
                       help=f"Number of Monte Carlo samples for uncertainty propagation. "
                            f"Set to 0 to disable MC and use point estimates. "
                            f"(default: {DEFAULT_PARAMS['n_mc_samples']}, use 50000 for production)")
    parser.add_argument('--prior-alpha', type=float,
                       default=DEFAULT_PARAMS['prior_alpha'],
                       help=f"Beta prior parameter for FNR posteriors. "
                            f"1.0 = uniform prior, 0.5 = Jeffreys prior. "
                            f"(default: {DEFAULT_PARAMS['prior_alpha']})")
    parser.add_argument('--uncertainty-style', type=str,
                       choices=['ribbon', 'errorbar', 'both', 'none'],
                       default=DEFAULT_PARAMS['uncertainty_style'],
                       help=f"How to display uncertainty: 'ribbon' (shaded area), "
                            f"'errorbar' (vertical bars), 'both', or 'none'. "
                            f"(default: {DEFAULT_PARAMS['uncertainty_style']})")
    parser.add_argument('--include-p-harm', action='store_true',
                       help="Include P_harm (3rd row) in the figure. Default: only P1 and P2 rows.")
    parser.add_argument('--output-name', type=str, default='figure_5',
                       help="Base name for output files (default: figure_5)")
    
    args = parser.parse_args()
    
    # Build params dict from command-line args
    params = {
        'failure_multiplier': args.failure_multiplier,
        'therapy_request_rate': args.therapy_request_rate,
        'model_comply_rate': args.model_comply_rate,
        'prob_fail_seek_help': args.prob_fail_seek_help,
        'lod_ci_level': args.lod_ci_level,
        'n_mc_samples': args.n_mc_samples,
        'prior_alpha': args.prior_alpha,
        'uncertainty_style': args.uncertainty_style,
        'include_p_harm': args.include_p_harm,
        'output_name': args.output_name,
    }
    
    print("="*80)
    if args.include_p_harm:
        print("P1, P2, AND P_HARM RISK ANALYSIS FACET PLOT")
    else:
        print("P1 AND P2 RISK ANALYSIS FACET PLOT")
    print("(WITH PROVENANCE TRACKING)")
    print("="*80)
    print(f"Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    for csv_path in [args.suicide_csv, args.therapy_request_csv, args.therapy_engagement_csv]:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    create_p1_p2_risk_plot(
        args.suicide_csv,
        args.therapy_request_csv,
        args.therapy_engagement_csv,
        tuple(args.figsize),
        not args.linear_y,  # Invert: linear_y flag OFF means log scale ON
        params
    )
    
    print("="*80)
    print("COMPLETE!")
    print(f"All outputs saved to: {tracker.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
