# Figure naming: Phase A (inventory) and Phase B (proposed targets)

**Status:** For alignment before Phase C (code/output renames).  
**Frozen:** No renames applied in this document.

## Phase A — Inventory (manuscript ↔ pipeline today)

| Manuscript | Role | Pipeline folder | Primary file(s) | Provenance JSON slug (extract step) | Main consumers |
|------------|------|-----------------|-----------------|--------------------------------------|----------------|
| Fig 3 | Expert review breakdown | `Figures/` | `figure_3.png` | `figure_3` | `combined_three_panel_review_provenance.py`, pipeline |
| Fig 4 | Model metrics | `Figures/` | `figure_4.png` | `figure_4` | `multi_experiment_plot_transposed_provenance.py`, `confusion_matrix_audit.py` (Fig 4 + S6–S8) |
| Fig 5 | P1/P2 risk | `Figures/` | `figure_5.png` | `figure_5` | `p1_and_p2_plot_provenance.py`, pipeline |
| **S5** | Sankey (review flow) | `Supplementary_Figures/figure_S4/` | `si_*sankey.png`, `therapy_*sankey.png` (×3) | `figure_S4_si`, `figure_S4_therapy_request`, `figure_S4_therapy_engagement` | `sankey_diagram_configs.py`, pipeline `generate_figure_s4`, `extract_provenance_from_pngs` |
| **S6–S8** | Binary confusion matrices | `Supplementary_Figures/figures_S5-S7/` | `figure_S5.png` … `figure_S7.png` | `figure_S5` … `figure_S7` | `generate_confusion_matrix_figures.py`, pipeline, audits |
| **S9–S11** | Statement heatmaps | `Supplementary_Figures/figures_S8-S10/` | `figure_S8.png` … `figure_S10.png` | `figure_S8` … `figure_S10` | `generate_model_statement_matrices.py`, `heatmap_audit.py`, pipeline |
| **S12** | FNR adjustment heatmap | `Supplementary_Figures/figure_S12/` | `figure_S12.png` | `figure_S12` | `figure_s12_failure_multiplier_heatmap.py`, pipeline |
| **S13** | P2 across *M* | `Supplementary_Figures/figure_S11/` | `figure_S11.png` (renamed from script output) | `figure_S11` (not `figure_S13`) | `figure_s11_p2_by_model_size_across_m.py`, pipeline `generate_figure_s11`, `utilities/figure_s11_audit.py` |

**Other references:** `run_regulatory_simulation_paper_pipeline.py` (paths, `extract_provenance_from_pngs` list, embedded run `README.md`, audits block), root `README.md` Figure Guide, `config/manuscript_claims.json` (if any paths), `utilities/figure_provenance.py` (generic).

**Note:** `figure_s11_audit.py` text still says “Figure S11” in places; manuscript table calls that output **S13**.

## Phase B — Proposed canonical layout (for sign-off)

**Principle:** Folder and flagship PNG names use **manuscript** supplementary numbers **S5–S13**; main figures **3–5** unchanged unless you choose otherwise.

| Manuscript | Proposed folder | Proposed primary file(s) | Proposed provenance slug(s) |
|------------|-----------------|--------------------------|----------------------------|
| S5 | `Supplementary_Figures/figure_S5/` | keep three sankey basenames or rename to `figure_S5_si.png` etc. (TBD) | `figure_S5_si`, … (align with files) |
| S6–S8 | `Supplementary_Figures/figure_S6/` … `figure_S8/` **or** one `figures_S6-S8/` with `figure_S6.png` … `figure_S8.png` (TBD) | `figure_S6.png`, `figure_S7.png`, `figure_S8.png` | `figure_S6`, `figure_S7`, `figure_S8` |
| S9–S11 | Same pattern as above | `figure_S9.png`, `figure_S10.png`, `figure_S11.png` | `figure_S9`, `figure_S10`, `figure_S11` |
| S12 | `Supplementary_Figures/figure_S12/` | `figure_S12.png` | `figure_S12` (already matches) |
| S13 | `Supplementary_Figures/figure_S13/` | `figure_S13.png` | `figure_S13` |

**Open choices before Phase C**

1. **Sankeys (S5):** one directory with three PNGs vs three subfolders; exact filenames.  
2. **S6–S8 and S9–S11:** separate folder per figure vs shared “batch” folders (mirrors current `figures_S5-S7` style or fully flat `figure_S6/` …).  
3. **Script renames:** e.g. `figure_s11_p2_by_model_size_across_m.py` → `figure_s13_…` vs keep internal module names and only change outputs.  
4. **Audit / claims / manuscript JSON:** rename `figure_s11_audit.py` and user-facing strings to S13 or keep filename with doc note.

After you confirm (2) and (3), Phase C can be a single mechanical pass: pipeline outputs, `extract_provenance_from_pngs`, audits, README Figure Guide, and optional script renames.
