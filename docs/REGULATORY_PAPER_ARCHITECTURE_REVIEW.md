# Regulatory Simulation Paper - Architecture Review & Improvement Plan

**Date:** 2026-01-05  
**Status:** In Progress  

This document captures the architecture review of the regulatory simulation paper pipeline, completed work, and planned improvements.

---

## Table of Contents
1. [Completed Work](#completed-work)
2. [Remaining Tasks](#remaining-tasks)
3. [Figure Dependency Map](#figure-dependency-map)
4. [zLegacy Assessment](#zlegacy-assessment)
5. [Known Issues](#known-issues)
6. [Architecture Notes](#architecture-notes)

---

## Completed Work

### ✅ D1+D2: Parameter Centralization
**Date Completed:** 2026-01-05

Created `config/regulatory_paper_parameters.py` as single source of truth for all quantitative assumptions:

- **API Parameters**: temperature, max_tokens, top_p, request_timeout, request_delay, concurrency settings
- **Risk Model Parameters**: therapy_request_rate, model_comply_rate, prob_fail_seek_help, failure_multiplier values, Monte Carlo settings
- **Binary Classification Categories**: Centralized definitions for SI, therapy request, and therapy engagement

**Files Migrated:**
- `analysis/comparative_analysis/p1_and_p2_plot_provenance.py`
- `analysis/comparative_analysis/figure_s10_p2_by_model_size_across_m.py`
- `config/experiment_config.py`
- `orchestration/run_experiment.py`
- `orchestration/experiment_manager.py`
- `config/utils.py`

**Key Decisions:**
- Sample sizes are NOT parameters - they're calculated dynamically from input data via `get_sample_size_from_metrics()`
- Visualization parameters (colors, fonts, figure sizes) remain in plotting scripts
- `request_timeout` standardized to 60s (was inconsistent: 120s in utils.py, 60s in experiment_manager.py)
- Renamed `sample_params` → `mc_sample_empirical_rates` for clarity

See `config/PARAMETERS_MIGRATION_MAP.md` for detailed migration tracking.

---

## Remaining Tasks

### ✅ A1+A2: Cache Traceability Logging
**Priority:** High  
**Status:** Completed (2026-01-05)

Added cache traceability so each analysis can be traced back to specific cache entries:

**Changes made:**
- `cache/result_cache_v2.py`: Added `cache_id` and `created_at` to returned results
- `cache/result_cache_v2.py`: Added `generate_cache_manifest()` method
- `analysis/model_performance/data_loader.py`: Added `save_cache_manifest()` function
- `analysis/model_performance/batch_results_analyzer.py`: Auto-saves manifest to `cache_manifest.json`

**Manifest contents:**
- `timestamp`: When analysis was run
- `cache_dir`: Which cache was used
- `total_entries`: Number of unique cache entries
- `models`: Per-model breakdown with cache IDs
- `cache_ids`: Complete list of all cache IDs used

**Example output:** `results/.../20260105_185726_SI_paper/cache_manifest.json`

### ✅ A4: Create Run Manifests
**Priority:** High  
**Status:** Completed (2026-01-05)

Implemented as part of A1+A2. The `cache_manifest.json` serves as the run manifest, documenting:
- Cache entry IDs used
- Model metadata (family, size, version, path)
- Timestamps for each entry

### ✅ B1: Document Two-Phase Architecture
**Priority:** Medium  
**Status:** Completed (2026-01-06)

See [Two-Phase Architecture](#two-phase-architecture) section for full documentation.

### ✅ C1: Create Figure Dependency Map
**Priority:** Medium  
**Status:** Completed (2026-01-05)

See [Figure Dependency Map](#figure-dependency-map) section above.

### ✅ C2: Add Input Validation Assertions
**Priority:** Medium  
**Status:** Completed (2026-01-06)

Created `utilities/input_validation.py` with validation functions:
- `validate_comprehensive_metrics()` - Validates metrics CSVs for figure generation
- `validate_dataframe()` - Generic validation with column/NaN checks
- `validate_models_config()` - Validates models_config.csv

Integrated into key figure scripts:
- `multi_experiment_plot_transposed_provenance.py` (Figure 4)
- `p1_and_p2_plot_provenance.py` (Figure 5)

Validations catch:
- Missing required columns
- Empty DataFrames
- NaN values in metric columns
- Metric values outside [0,1] range
- Negative count values

### E: Package Extraction Evaluation
**Priority:** Low  
**Status:** Not Started

Evaluate separating the LLM prompting/running infrastructure into a separate package/repo:
- **Pros**: Shared across projects, reduces coupling
- **Cons**: Dependency management complexity
- **Recommendation**: Consider after papers are published

### ✅ F1: Verify zLegacy is Dead Code
**Priority:** Low  
**Status:** Completed (2026-01-05)

See [zLegacy Assessment](#zlegacy-assessment) section above. All zLegacy directories verified as dead code (~415 MB total, safe to delete).

### ✅ F2+F3: Document Paper Associations
**Priority:** Medium  
**Status:** Completed (2026-01-06)

See [Paper Associations](#paper-associations) section for full documentation.

---

## Figure Dependency Map

### Main Figures

#### Figure 3: Expert Review Breakdown
- **Script**: `analysis/data_validation/combined_three_panel_review_provenance.py`
- **Input Data**:
  - `data/inputs/finalized_input_data/SI_finalized_sentences.csv` (ground truth with approval status)
  - `data/inputs/finalized_input_data/therapy_request_finalized_sentences.csv`
  - `data/inputs/finalized_input_data/therapy_engagement_finalized_sentences.csv`
- **Output**: 3-panel barplot showing approved/modified/removed breakdown per task
- **Dependencies**: None (uses raw input data only)

#### Figure 4: Model Performance Metrics
- **Script**: `analysis/comparative_analysis/multi_experiment_plot_transposed_provenance.py`
- **Input Data**:
  - `comprehensive_metrics.csv` from SI experiment results
  - `comprehensive_metrics.csv` from TR experiment results  
  - `comprehensive_metrics.csv` from TE experiment results
- **Output**: Multi-panel scatter plot (parse rate, sensitivity, specificity, accuracy, F1)
- **Dependencies**: Requires experiment results generated from cache

#### Figure 5: P1/P2/P_harm Risk Analysis
- **Script**: `analysis/comparative_analysis/p1_and_p2_plot_provenance.py`
- **Input Data**:
  - Same `comprehensive_metrics.csv` files as Figure 4
- **Parameters** (from `config/regulatory_paper_parameters.py`):
  - `therapy_request_rate`: 0.029 (from Anthropic data)
  - `model_comply_rate`: 0.90 (from Anthropic data)
  - `prob_fail_seek_help`: 1.0 (worst case)
  - `failure_multiplier_values`: [1, 2, 5, 10, 20, 100, 1000, 10000, 100000]
  - `n_mc_samples`: 50000
- **Output**: 3×3 facet plot for each m value (9 figures total)
- **Dependencies**: Requires experiment results generated from cache

### Supplementary Figures

#### Figure S3: Sankey Diagrams
- **Script**: `analysis/data_validation/sankey_diagram_configs.py`
- **Input Data**: Same finalized input files as Figure 3
- **Output**: 3 Sankey diagrams (SI, TR, TE) showing data flow

#### Figures S4-S6: Binary Confusion Matrices
- **Script**: `analysis/model_performance/generate_confusion_matrix_figures.py`
- **Input Data**: Experiment results (model predictions + ground truth)
- **Output**: Grid of confusion matrices per model per task

#### Figures S7-S9: Per-Statement Accuracy Heatmaps  
- **Scripts**: 
  - `analysis/model_performance/generate_correctness_matrices.py`
  - `analysis/model_performance/generate_model_statement_heatmaps.py`
- **Input Data**: Experiment results (model predictions + ground truth)
- **Output**: Heatmaps showing correctness per statement per model

#### Figure S10: P2 Across Failure Multiplier Values
- **Script**: `analysis/comparative_analysis/figure_s10_p2_by_model_size_across_m.py`
- **Input Data**: Same `comprehensive_metrics.csv` files as Figure 5
- **Parameters** (from `config/regulatory_paper_parameters.py`):
  - `failure_multiplier_values`: All M values shown in facet rows
  - Same risk model parameters as Figure 5
- **Output**: 8×3 facet plot showing P2 vs harm prevalence for each M value

### Data Flow Summary

```
Input Data (finalized_sentences.csv)
       │
       ├──► Figure 3 (direct)
       │
       ▼
    Cache (V2)
       │
       ├──► batch_results_analyzer.py
       │           │
       │           ▼
       │    comprehensive_metrics.csv
       │           │
       │           ├──► Figure 4
       │           │
       │           └──► Figure 5 (+ risk model params)
       │
       └──► Figures S4-S9 (model predictions)
```

---

## zLegacy Assessment

**Status:** ✅ Verified as Dead Code - Safe to Delete

### Directories Found

| Directory | Size | Contents | Referenced? |
|-----------|------|----------|-------------|
| `zLegacy/` (top-level) | 275M | Old analysis scripts, figure plots, hazard analysis | ❌ No |
| `data/inputs/finalized_input_data/zLegacy/` | 1.6M | Backup/original data files | ❌ No |
| `data/inputs/zlegacy/` | 3.6M | Old intermediate data files | ❌ No |
| `data/prompts/zlegacy/` | 24K | Old prompt versions | ❌ No |
| `utilities/zLegacy/` | 8K | Old utility script | ❌ No |
| `results/zLegacy/` | 6.1M | Old result plots | ❌ No |
| `results/individual_prediction_performance/suicidal_ideation/zLegacy/` | 129M | Old experiment results | ❌ No |

**Total zLegacy Size:** ~415 MB

### Verification

Searched entire codebase for references to zLegacy files:
- **Active code importing from zLegacy**: None
- **Active code reading zLegacy data**: None
- **Only reference**: `utilities/migrate_results_structure.sh` which explicitly *excludes* zlegacy

### Recommendation

All zLegacy directories are safe to delete. Before deletion:
1. Ensure all current analyses pass (✅ verified - pipeline runs successfully)
2. Consider a backup if historical context may be needed
3. Delete to reclaim ~415 MB of space

```bash
# Command to delete all zLegacy (run from project root):
rm -rf zLegacy \
       data/inputs/finalized_input_data/zLegacy \
       data/inputs/zlegacy \
       data/prompts/zlegacy \
       utilities/zLegacy \
       results/zLegacy \
       results/individual_prediction_performance/suicidal_ideation/zLegacy
```

---

## Two-Phase Architecture

The regulatory simulation pipeline has a conceptual two-phase architecture, though the phases can be executed together or separately.

### Phase 1: Cache Population

**Purpose:** Run prompts through LLMs and store results in the V2 cache.

**Entry Points:**
1. `bash_scripts/run_experiments.sh` - Manual experiment runner
2. `bash_scripts/run_experiments_scheduled.sh` - Batch scheduler for multiple experiments
3. Direct Python: `orchestration/run_experiment.py`

**Data Flow:**
```
Input Data (finalized_sentences.csv)
       │
       ▼
   Prompts (data/prompts/*.txt)
       │
       ▼
   LM Studio API (localhost:1234)
       │
       ▼
   Cache V2 (cache/v2_lmstudio_results.sqlite)
```

**Key Components:**
| Component | File | Purpose |
|-----------|------|---------|
| API Client | `orchestration/api_client.py` | Sends requests to LM Studio |
| Experiment Runner | `orchestration/run_experiment.py` | Coordinates prompt→response flow |
| Result Cache | `cache/result_cache_v2.py` | SQLite storage with deduplication |
| Experiment Config | `config/experiment_config.py` | Task-specific settings |

**Cache Key (V2):** LM Studio's model `path` field + prompt hash + input text hash

**When to Run Phase 1:**
- When running experiments for the first time
- When adding new models to the analysis
- When changing prompts (will create new cache entries)
- NOT needed if only changing visualization or analysis code

### Phase 2: Analysis

**Purpose:** Read from cache, compute metrics, generate figures.

**Entry Point:** `run_regulatory_simulation_paper_pipeline.py`

**Data Flow:**
```
Cache V2 (sqlite)
       │
       ▼
   batch_results_analyzer.py
       │
       ├──► comprehensive_metrics.csv (per-model performance)
       │            │
       │            ├──► Figure 4 (performance scatter)
       │            │
       │            └──► Figure 5 (risk analysis)
       │
       └──► cache_manifest.json (traceability)
```

**Key Components:**
| Component | File | Purpose |
|-----------|------|---------|
| Data Loader | `analysis/model_performance/data_loader.py` | Reads cache, joins with ground truth |
| Batch Analyzer | `analysis/model_performance/batch_results_analyzer.py` | Computes metrics per model |
| Risk Plotter | `analysis/comparative_analysis/p1_and_p2_plot_provenance.py` | Figure 5 generation |
| Performance Plotter | `analysis/comparative_analysis/multi_experiment_plot_transposed_provenance.py` | Figure 4 generation |

**When to Run Phase 2:**
- After Phase 1 has populated the cache
- When changing analysis parameters (in `regulatory_paper_parameters.py`)
- When regenerating figures with different visualization options
- Can run with `--figures-only` if comprehensive_metrics.csv already exists

### Phase Execution Modes

| Mode | Command | Phase 1 | Phase 2 |
|------|---------|---------|---------|
| Full Pipeline | `python run_regulatory_simulation_paper_pipeline.py` | ✓ (from cache) | ✓ |
| Figures Only | `python run_regulatory_simulation_paper_pipeline.py --figures-only` | ✗ | ✓ |
| Populate Cache | `./bash_scripts/run_experiments.sh ...` | ✓ | ✗ |
| Dry Run | `python run_regulatory_simulation_paper_pipeline.py --dry-run` | ✗ | ✗ (shows plan) |

### Important Notes

1. **Phase 1 is Idempotent:** Running experiments with the same inputs will hit cache, not re-query the LLM.

2. **Phase 2 Reads from Cache:** Even the "full pipeline" doesn't re-run LLM queries. It just reads from cache and generates analysis outputs.

3. **Cache Traceability:** `cache_manifest.json` (generated during Phase 2) documents exactly which cache entries were used, allowing you to trace any figure back to specific cache rows.

4. **No Direct LLM Calls in Figure Scripts:** All figure generation scripts read from pre-computed `comprehensive_metrics.csv`, never directly from the LLM.

---

## Paper Associations

This repository contains code for **two papers**. This section documents which files belong to which paper.

### Paper 1: Regulatory Simulation Paper

**Main Pipeline:** `run_regulatory_simulation_paper_pipeline.py`  
**Output Directory:** `results/REGULATORY_SIMULATION_PAPER/[timestamp]/`  
**Status:** In major revision (medrXiv preprint)

#### Files Used

| Category | Files |
|----------|-------|
| **Pipeline** | `run_regulatory_simulation_paper_pipeline.py` |
| **Config** | `config/regulatory_paper_parameters.py`<br>`config/regulatory_paper_models.csv` |
| **Main Figures** | `analysis/data_validation/combined_three_panel_review_provenance.py` (Fig 3)<br>`analysis/comparative_analysis/multi_experiment_plot_transposed_provenance.py` (Fig 4)<br>`analysis/comparative_analysis/p1_and_p2_plot_provenance.py` (Fig 5) |
| **Supp Figures** | `analysis/data_validation/sankey_diagram_configs.py` (S3)<br>`analysis/model_performance/generate_confusion_matrix_figures.py` (S4-S6)<br>`analysis/model_performance/generate_correctness_matrices.py` (S7-S9)<br>`analysis/comparative_analysis/figure_s10_p2_by_model_size_across_m.py` (S10) |
| **Verification** | `analysis/manuscript_claims_verification.py` |

#### Figure Outputs

- **Figure 3:** Expert review breakdown (3-panel barplot)
- **Figure 4:** Model performance metrics (scatter plots)
- **Figure 5:** P1/P2/P_harm risk analysis (3×3 facet per M value)
- **Figure S3:** Sankey diagrams
- **Figures S4-S6:** Binary confusion matrices
- **Figures S7-S9:** Per-statement accuracy heatmaps
- **Figure S10:** P2 across all M values

---

### Paper 2: Fine-tune Paper (Mental Health LLM Fine-tuning)

**Main Pipeline:** `run_paper_pipeline.py`  
**Output Directory:** `results/FINETUNE_PAPER_FIGURES/[timestamp]/`  
**Status:** Published (frozen branch: `mental_health_finetune_paper`)

#### Files Used

| Category | Files |
|----------|-------|
| **Pipeline** | `run_paper_pipeline.py` |
| **Figures** | `analysis/comparative_analysis/coverage_facet_plot.py` (Fig 1)<br>`analysis/comparative_analysis/f1_vs_params_facet.py` (Fig 2)<br>`analysis/comparative_analysis/delta_f1_facet_plot.py` (Fig 3) |
| **Table** | `analysis/comparative_analysis/combined_regression_table.py` (Table 1) |
| **Supp Figures** | `analysis/comparative_analysis/family_task_facet_plots.py` |

#### Figure Outputs

- **Figure 1:** Model coverage facet
- **Figure 2:** F1 vs parameters overall trend
- **Figure 3:** Delta F1 facet plot
- **Table 1:** Regression table (Bonferroni corrected)
- **Supplementary:** 9 family×task facet plots

---

### Shared Components

Both papers share these foundational components:

| Component | Files | Description |
|-----------|-------|-------------|
| **LLM Infrastructure** | `orchestration/api_client.py`<br>`orchestration/run_experiment.py`<br>`orchestration/experiment_manager.py` | API calls, experiment execution |
| **Caching** | `cache/result_cache_v2.py`<br>`cache/cache_manager.py` | V2 SQLite cache |
| **Data Loading** | `analysis/model_performance/data_loader.py`<br>`analysis/model_performance/batch_results_analyzer.py` | Load from cache, compute metrics |
| **Statistics** | `utilities/statistics.py` | CI calculations, bootstrap |
| **Config** | `config/experiment_config.py`<br>`config/constants.py`<br>`config/models_config.csv` | Shared experiment settings |
| **Input Data** | `data/inputs/finalized_input_data/*.csv` | Same datasets for both papers |

### Key Differences

| Aspect | Regulatory Simulation | Fine-tune |
|--------|----------------------|-----------|
| **Focus** | Risk analysis (P1/P2/P_harm) | F1 performance across versions |
| **Models** | Subset (16 models) | All available models |
| **Key Analysis** | Failure multiplier (M) impact | Fine-tuning version impact |
| **Verification** | Manuscript claims verification | N/A |

---

## Known Issues

### 1. LLaMA Family Naming
**Status:** Known, Documented  
**Impact:** Low (handled in code)

In `config/regulatory_paper_models.csv`, LLaMA models are split into separate families:
- `llama3.1`
- `llama3.2`
- `llama3.3`

This is handled by `normalize_family()` in analysis scripts which groups them all as `llama` for visualization.

**TODO:** Consider overhauling config to use unified `llama` family with version column for consistency.

### 2. Manuscript Claims Verification Requires Full Pipeline
**Status:** Known  
**Impact:** Low

`manuscript_claims_verification.py` requires `Data/filtered_metrics/` which is only generated when running full pipeline (without `--figures-only`).

### 3. Historical request_timeout Inconsistency
**Status:** Resolved  
**Resolution:** Standardized to 60s (what was actually used)

---

## Architecture Notes

### Pipeline Structure
```
run_regulatory_simulation_paper_pipeline.py
├── Experiment Data Generation (from cache_v2)
│   ├── suicidal_ideation → comprehensive_metrics.csv
│   ├── therapy_request → comprehensive_metrics.csv
│   └── therapy_engagement → comprehensive_metrics.csv
├── Main Figures
│   ├── Figure 3: Expert Review Breakdown
│   ├── Figure 4: Model Performance Metrics
│   └── Figure 5: P1/P2/P_harm Risk Analysis (multiple m values)
├── Supplementary Figures
│   ├── Figure S3: Sankey Diagrams
│   ├── Figures S4-S6: Confusion Matrices
│   └── Figures S7-S9: Accuracy Heatmaps
└── Data Collection (optional)
```

### Key Configuration Files
- `config/regulatory_paper_parameters.py` - All quantitative parameters
- `config/regulatory_paper_models.csv` - Which models to use
- `config/models_config.csv` - Master model metadata

### Cache Architecture
- **V1 Cache**: Uses `model_family + model_size + model_version` as key
- **V2 Cache**: Uses LM Studio's `path` as unique model identifier (current, preferred)

---

## References

- `config/PARAMETERS_MIGRATION_MAP.md` - Detailed parameter migration tracking
- `TODOs.md` - Technical debt items
- `run_regulatory_simulation_paper_pipeline.py` - Main pipeline script

---

*Last Updated: 2026-01-06*

