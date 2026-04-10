# Supplementary figure naming (manuscript S5–S13)

**Status:** Pipeline outputs, provenance slugs, and scripts are aligned with manuscript supplementary numbering. Main manuscript figures in the pipeline are **2–4** (see root README Figure Guide).

## Layout (current)

| Manuscript | Role                      | Pipeline folder                         | Primary file(s)                               | Provenance JSON slug(s) (extract step)                                         |
| ---------- | ------------------------- | --------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------ |
| Fig 2      | Expert review breakdown   | `Figures/`                              | `figure_2.png`                                | `figure_2`                                                                     |
| Fig 3      | Model metrics             | `Figures/`                              | `figure_3.png`                                | `figure_3`                                                                     |
| Fig 4      | P1/P2 risk                | `Figures/`                              | `figure_4.png`                                | `figure_4`                                                                     |
| **S5**     | Sankey (review flow)      | `Supplementary_Figures/figure_S5/`      | `figure_S5_si.png`, `figure_S5_therapy_request.png`, `figure_S5_therapy_engagement.png` | `figure_S5_si`, `figure_S5_therapy_request`, `figure_S5_therapy_engagement` |
| **S6–S8**  | Binary confusion matrices | `Supplementary_Figures/figures_S6-S8/` | `figure_S6.png` … `figure_S8.png`             | `figure_S6` … `figure_S8`                                                      |
| **S9–S11** | Statement heatmaps        | `Supplementary_Figures/figures_S9-S11/` | `figure_S9.png` … `figure_S11.png`           | `figure_S9` … `figure_S11`                                                     |
| **S12**    | FNR adjustment heatmap    | `Supplementary_Figures/figure_S12/`     | `figure_S12.png`                              | `figure_S12`                                                                   |
| **S13**    | P2 across *M*             | `Supplementary_Figures/figure_S13/`     | `figure_S13.png` (renamed from `figure_s13_p2_across_m_values.png`) | `figure_S13`                                                          |

**Main consumers:** `run_regulatory_simulation_paper_pipeline.py`, root `README.md` Figure Guide, `utilities/confusion_matrix_audit.py`, `utilities/heatmap_audit.py`, `utilities/figure_s13_audit.py`.

Heatmap **generation** still writes under `results/model_performance_analysis/` with timestamped directories named after the task heatmap (`si_correctness_heatmap`, etc.); the pipeline copies consolidated PNGs into `figures_S9-S11/`. The heatmap audit resolves provenance via `heatmap_provenance_basename` in `utilities/heatmap_audit.py`.
