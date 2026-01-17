# Naming Conventions Guide

This document establishes naming conventions for the safety simulations codebase.

## Task Names

| Full Name | Short Form | Usage |
|-----------|------------|-------|
| `suicidal_ideation` | `SI` | Config keys, variable names, directory names |
| `therapy_request` | `TR` or `tx_request` | Config keys, variable names, directory names |
| `therapy_engagement` | `TE` or `tx_engagement` | Config keys, variable names, directory names |

### Preferred Convention

- **Config keys**: Use full snake_case: `suicidal_ideation`, `therapy_request`, `therapy_engagement`
- **Variable names**: Use full snake_case: `si_metrics`, `therapy_request_metrics`, `therapy_engagement_metrics`
- **Directory names**: Use full snake_case: `suicidal_ideation/`, `therapy_request/`, `therapy_engagement/`
- **Short prefixes** (for brevity in logs/filenames): `SI_`, `TR_`, `TE_`

### Legacy Naming

The following legacy naming patterns exist in the codebase and should NOT be changed (breaking):
- Historic result directories may use `tx_request` or `tx_engagement` (e.g., `20251027_182856_tx_request/`)
- Some older scripts reference these paths

### Migration Status

| Location | Current | Target | Status |
|----------|---------|--------|--------|
| `config/regulatory_paper_parameters.py` | `therapy_*` | `therapy_*` | ✅ Correct |
| `config/experiment_config.py` | `therapy_*` | `therapy_*` | ✅ Correct |
| `run_regulatory_simulation_paper_pipeline.py` | Mixed | Keep as-is | ⚠️ Legacy paths |
| `analysis/comparative_analysis/*.py` | `tx_*` vars | `therapy_*` vars | 🔄 Optional |

## File Naming

### Python Files
- Use snake_case: `generate_confusion_matrix_figures.py`
- Suffix with `_provenance` if using FigureProvenanceTracker

### Data Files
- CSVs: snake_case with descriptive names
- Results: `YYYYMMDD_HHMMSS_{task_short}_{description}/`

### Figure Outputs
- Use lowercase with underscores: `figure_3.png`, `si_correctness_heatmap.png`

## Column Names

### Standard DataFrame Columns
| Column | Description |
|--------|-------------|
| `model_family` | Model family (gemma, qwen, llama) |
| `model_size` | Size string (e.g., "1b", "270m-it") |
| `model_version` | Version identifier |
| `sensitivity` | True positive rate |
| `specificity` | True negative rate |
| `accuracy` | Overall accuracy |
| `f1_score` | F1 score |
| `tp`, `tn`, `fp`, `fn` | Confusion matrix counts |

### Task-Specific Columns
| Task | Ground Truth Col | Prediction Col |
|------|------------------|----------------|
| suicidal_ideation | `prior_safety_type` | `safety_type` |
| therapy_request | `prior_therapy_request` | `therapy_request` |
| therapy_engagement | `prior_therapy_engagement` | `therapy_engagement` |

---

*Last Updated: 2026-01-06*

