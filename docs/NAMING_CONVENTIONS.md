# Naming Conventions

## Task Names

| Full Name | Short Form |
|-----------|------------|
| `suicidal_ideation` | `SI` |
| `therapy_request` | `TR` |
| `therapy_engagement` | `TE` |

- **Config keys/variables/directories**: Use full snake_case (`suicidal_ideation`, `therapy_request`)
- **Short prefixes** (logs/filenames): `SI_`, `TR_`, `TE_`

## File Naming

- **Python files**: snake_case (`generate_confusion_matrix_figures.py`)
- **Data files**: snake_case with descriptive names
- **Figures**: lowercase with underscores (`figure_3.png`, `si_correctness_heatmap.png`)

## Standard DataFrame Columns

| Column | Description |
|--------|-------------|
| `model_family` | Model family (gemma, qwen, llama) |
| `model_size` | Size string (e.g., "1b", "270m-it") |
| `sensitivity` | True positive rate |
| `specificity` | True negative rate |
| `accuracy` | Overall accuracy |
| `f1_score` | F1 score |
| `tp`, `tn`, `fp`, `fn` | Confusion matrix counts |

## Ground Truth / Prediction Columns

| Task | Ground Truth | Prediction |
|------|--------------|------------|
| suicidal_ideation | `prior_safety_type` | `safety_type` |
| therapy_request | `prior_therapy_request` | `therapy_request` |
| therapy_engagement | `prior_therapy_engagement` | `therapy_engagement` |
