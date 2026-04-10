#!/usr/bin/env python3
"""
Figure S12: Adjusted false-negative rate vs. observed FNR and failure multiplier M

Heatmap of FNR_adjusted = 1 - (1 - FNR_observed)^M. Larger M drives adjusted FNR toward
100%, with smaller baseline FNR requiring larger M to approach 100% failure rate.

Rows: illustrative FNR_observed values (percent).
Columns: M values from config/regulatory_paper_parameters.py (same grid as main sensitivity analysis).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.regulatory_paper_parameters import RISK_MODEL_PARAMS
from utilities.figure_provenance import FigureProvenanceTracker

# Baseline observed FNR (fractions); display as percent on row labels
FNR_OBSERVED_FRAC = np.array(
    [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    dtype=float,
)

M_VALUES = np.array(RISK_MODEL_PARAMS["failure_multiplier_values"], dtype=float)


def observed_fnr_row_label(p: float) -> str:
    """Y-tick label from the same numeric row value p used in the heatmap (p = FNR_observed as a fraction).

    No lookup table: label is p×100%. Whole-number percents snap to integers to drop binary float noise
    (e.g. 10.0 not 9.9999999); other values use general format.
    """
    pct = float(p) * 100.0
    if p == 0.0:
        return "0%"
    r = float(np.round(pct, 12))
    nearest_int = int(round(r))
    if math.isclose(r, nearest_int, rel_tol=0.0, abs_tol=1e-6):
        return f"{nearest_int}%"
    return f"{format(r, 'g')}%"


def adjusted_fnr(fnr_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
    """FNR_adjusted = 1 - (1 - FNR_observed)^M; broadcasting over 2D grid."""
    return 1.0 - np.power(1.0 - fnr_obs[:, None], m[None, :])


def cell_annotation_text(v: float) -> str:
    """Format adjusted FNR as a percentage with two significant figures."""
    pct = v * 100.0
    s = f"{pct:.2g}"
    # Python's .2g uses scientific notation for e.g. 100 → "1e+02"; round for plain percent text.
    if "e" in s or "E" in s:
        s = format(round(pct), "g")
    return f"{s}%"


def white_to_red_cmap() -> LinearSegmentedColormap:
    """Light (low risk) → saturated red (high adjusted FNR)."""
    return LinearSegmentedColormap.from_list(
        "white_red_risk",
        ["#ffffff", "#fee0d2", "#fc9272", "#de2d26", "#a50f15"],
    )


def create_figure_s12(output_path: Path, tracker: FigureProvenanceTracker | None = None) -> None:
    fnr_obs = FNR_OBSERVED_FRAC
    m_vals = M_VALUES
    z = adjusted_fnr(fnr_obs, m_vals)

    row_labels = [observed_fnr_row_label(float(p)) for p in fnr_obs]
    col_labels = [f"{int(m):,}" if m == int(m) else str(m) for m in m_vals]

    fig, ax = plt.subplots(figsize=(11.5, 9.0), layout="constrained")

    cmap = white_to_red_cmap()
    cmap.set_bad(color="0.85")

    im = ax.imshow(
        z,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(m_vals)))
    ax.set_yticks(np.arange(len(fnr_obs)))
    tick_fs = 16.5  # +50% from 11
    axis_label_fs = 19.5  # +50% from 13
    cbar_label_fs = 18.0  # +50% from 12
    cbar_tick_fs = 15.0  # +50% from 10
    title_fs = 20

    ax.set_xticklabels(col_labels, rotation=0, fontsize=tick_fs)
    ax.set_yticklabels(row_labels, fontsize=tick_fs)
    ax.set_xlabel("Failure-multiplier parameter (M)", fontsize=axis_label_fs, labelpad=10)
    ax.set_ylabel(r"FNR$_{\mathrm{observed}}$ (%)", fontsize=axis_label_fs, labelpad=10)
    ax.set_title(
        r"Relationship between M, FNR$_{\mathrm{observed}}$, and FNR$_{\mathrm{adjusted}}$",
        fontsize=title_fs,
        pad=14,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"FNR$_{\mathrm{adjusted}}$ (%)", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)
    cbar.set_ticks(np.linspace(0.0, 1.0, 6))
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _p: f"{x * 100:.0f}%")
    )

    # Annotate cells (readability for print)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z[i, j]
            txt = cell_annotation_text(val)
            # White→red: light background at low FNR, dark red at high
            text_color = "white" if val > 0.42 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=13.5)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if tracker:
        tracker.add_output_file(output_path, file_type="figure", dpi=300)
        tracker.set_analysis_parameters(
            formula="FNR_adjusted = 1 - (1 - FNR_observed)**M",
            fnr_observed_fractions=fnr_obs.tolist(),
            m_values=m_vals.tolist(),
            reference="analysis/comparative_analysis/p1_and_p2_plot_provenance.py (P2 scaling)",
        )
        tracker.set_reproducibility_command(
            "python analysis/revisions/figure_s12_failure_multiplier_heatmap.py "
            "--output-dir <Supplementary_Figures/figure_S12>"
        )
        tracker.save_provenance()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure S12: FNR adjustment heatmap (M vs observed FNR)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for figure_S12.png (and provenance subfolders)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = FigureProvenanceTracker(
        figure_name="figure_s12_failure_multiplier_fnr_heatmap",
        base_dir=output_dir,
    )

    out_png = output_dir / "figure_s12_failure_multiplier_fnr_heatmap.png"
    create_figure_s12(out_png, tracker=tracker)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
