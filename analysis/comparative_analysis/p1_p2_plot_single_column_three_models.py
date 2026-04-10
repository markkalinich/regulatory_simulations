#!/usr/bin/env python3
"""
Optional P1/P2 layout (not a main figure): 1 column, Gemma 12B + Qwen 8B + LLaMA 8B only.
Same logic as Figure 5 (`p1_and_p2_plot_provenance.py`). Also supports `--figure-3-summary`
(Keep/Remove bar sketch — not the pipeline’s Figure 3).

Example (after running the paper pipeline; paths under results/REGULATORY_SIMULATION_PAPER/<timestamp>/):
  python analysis/comparative_analysis/p1_p2_plot_single_column_three_models.py \\
    --suicide-csv results/REGULATORY_SIMULATION_PAPER/<timestamp>/Data/processed_data/model_performance_metrics/suicidal_ideation_comprehensive_metrics.csv \\
    --therapy-request-csv results/.../therapy_request_comprehensive_metrics.csv \\
    --therapy-engagement-csv results/.../therapy_engagement_comprehensive_metrics.csv \\
    --output results/.../Figures/figure_5_single_column_three_models.png
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Project root and imports from main Figure 5 script
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.comparative_analysis.p1_and_p2_plot_provenance import (
    DEFAULT_PARAMS,
    load_experiment_metrics,
    get_sample_size_from_metrics,
    prepare_plot_data,
    normalize_family,
    get_alpha_for_param_billions,
)

# Subset: (normalized_family, param_billions) for Gemma 12B, Qwen 8B, LLaMA 8B
MODEL_SUBSET = [
    ("gemma", 12.0),
    ("qwen", 8.0),
    ("llama", 8.0),
]

FAMILY_COLORS = {"gemma": "#1f77b4", "qwen": "#ff7f0e", "llama": "#2ca02c"}
FAMILY_DISPLAY = {"gemma": "Gemma", "qwen": "Qwen", "llama": "LLaMA"}


def model_legend_label(normalized_family: str) -> str:
    """Family name only for legend, e.g. 'Gemma', 'Qwen', 'LLaMA'."""
    return FAMILY_DISPLAY[normalized_family]


def create_single_column_plot(
    suicide_csv: Path,
    therapy_request_csv: Path,
    therapy_engagement_csv: Path,
    output_path: Path,
    figsize=(10, 12),
    log_y: bool = True,
    params: dict | None = None,
    include_p_harm: bool = False,
) -> None:
    """Build 2-row × 1-column P1/P2 plot with only Gemma 12B, Qwen 8B, LLaMA 8B."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        merged = DEFAULT_PARAMS.copy()
        merged.update(params)
        params = merged

    failure_multiplier = params["failure_multiplier"]
    uncertainty_style = params.get("uncertainty_style", "both")

    suicide_metrics = load_experiment_metrics(suicide_csv)
    therapy_request_metrics = load_experiment_metrics(therapy_request_csv)
    therapy_engagement_metrics = load_experiment_metrics(therapy_engagement_csv)

    plot_data = prepare_plot_data(
        suicide_metrics,
        therapy_request_metrics,
        therapy_engagement_metrics,
        params,
    )
    plot_data["normalized_family"] = plot_data["model_family"].apply(normalize_family)

    # Restrict to the three models (by normalized family + param_billions)
    subset_mask = plot_data.apply(
        lambda r: (r["normalized_family"], r["param_billions"]) in MODEL_SUBSET,
        axis=1,
    )
    plot_data = plot_data.loc[subset_mask].copy()
    if plot_data.empty:
        raise ValueError(
            "No data left after filtering to Gemma 12B, Qwen 8B, LLaMA 8B. "
            "Check that the input CSVs contain these models."
        )

    # Order for legend: Gemma 12B, Qwen 8B, LLaMA 8B
    def sort_key(row):
        nf, pb = row["normalized_family"], row["param_billions"]
        order = {"gemma": 0, "qwen": 1, "llama": 2}
        return (order.get(nf, 99), -pb)

    plot_data["_sort"] = plot_data.apply(sort_key, axis=1)
    plot_data = plot_data.sort_values("_sort").drop(columns=["_sort"])

    # Styling (match Figure 5)
    title_size = 28
    tick_size = 22
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": title_size,
        "axes.labelsize": 24,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": 16,
    })

    risk_types = ["P1", "P2", "P_harm"] if include_p_harm else ["P1", "P2"]
    # Multi-line y-labels so "Hazardous Situation" is not cut off and labels don't overlap
    risk_labels = {
        "P1": "P$_1$ (Hazard $\\rightarrow$\nHazardous Situation)",
        "P2": "P$_2$ (Hazardous Situation\n$\\rightarrow$ Harm)",
        "P_harm": f"P$_{{harm}}$ (Hazard $\\rightarrow$ Harm, m={failure_multiplier})",
    }
    baseline_labels = {
        "P1": "SI % in User Base",
        "P2": "P(Lack of Care → Harm) %",
        "P_harm": "Baseline Prevalence %",
    }

    n_rows = len(risk_types)
    fig, axes = plt.subplots(n_rows, 1, figsize=(figsize[0], 5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]
    has_uncertainty = (
        "risk_ci_5" in plot_data.columns
        and (plot_data["risk_ci_5"] != plot_data["risk_probability"]).any()
    )

    for row_idx, risk_type in enumerate(risk_types):
        ax = axes[row_idx]
        risk_data = plot_data[plot_data["risk_type"] == risk_type]

        # One line per model (3 models)
        for (nf, pb), grp in risk_data.groupby(["normalized_family", "param_billions"]):
            grp = grp.sort_values("baseline_percentage")
            color = FAMILY_COLORS[nf]
            label = model_legend_label(nf)
            alpha = get_alpha_for_param_billions(pb, risk_data)
            x = grp["baseline_percentage"].values
            y = grp["risk_probability"].values

            if has_uncertainty and uncertainty_style != "none":
                ci_5 = grp["risk_ci_5"].values
                ci_95 = grp["risk_ci_95"].values
                if uncertainty_style in ("ribbon", "both"):
                    ax.fill_between(x, ci_5, ci_95, color=color, alpha=alpha * 0.15)
                if uncertainty_style in ("errorbar", "both"):
                    yerr_lower = y - ci_5
                    yerr_upper = ci_95 - y
                    ax.errorbar(
                        x, y,
                        yerr=[yerr_lower, yerr_upper],
                        marker="o",
                        linestyle="-",
                        color=color,
                        alpha=alpha,
                        linewidth=2,
                        markersize=6,
                        capsize=3,
                        capthick=1,
                        elinewidth=1,
                        ecolor="black",
                        label=label,
                    )
                elif uncertainty_style == "ribbon":
                    ax.plot(
                        x, y,
                        marker="o",
                        linestyle="-",
                        color=color,
                        alpha=alpha,
                        linewidth=2,
                        markersize=6,
                        label=label,
                    )
            else:
                ax.plot(
                    x, y,
                    marker="o",
                    linestyle="-",
                    color=color,
                    alpha=alpha,
                    linewidth=2,
                    markersize=8,
                    label=label,
                )

        ax.set_ylabel(risk_labels[risk_type])
        ax.set_xlabel(baseline_labels[risk_type])
        ax.grid(True, alpha=0.3)
        # Legend only on bottom panel
        if row_idx == n_rows - 1:
            ax.legend(title="Model", loc="lower right")

        if log_y:
            ax.set_yscale("log")
            if risk_type == "P1":
                ax.set_ylim(1e-9, 1e-2)
            elif risk_type == "P2":
                ax.set_ylim(1e-5, 1)
            elif risk_type == "P_harm":
                ax.set_ylim(1e-12, 1e-3)

    plt.tight_layout(h_pad=3.0, rect=(0.12, 0, 1, 1))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# Figure 3 summary: single figure, 3 columns, Keep (green) / Remove (red)
# -----------------------------------------------------------------------------
KEEP_COLOR = "#2ecc71"   # green
REMOVE_COLOR = "#e74c3c"  # red


def create_figure3_summary_plot(output_path: Path, figsize=(8, 3.6)) -> None:
    """
    Single plot, 3 bars: one bar per task (SI, Therapy Request, Therapy Interaction).
    Each bar is one stacked column: Keep (green) vs Remove (red), aggregated across all subcategories.
    """
    import numpy as np
    from analysis.data_validation.combined_three_panel_review_provenance import (
        load_si_data,
        load_therapy_request_data,
        load_therapy_engagement_data,
    )

    _, si_df, si_cats, _, si_pcts, _ = load_si_data()
    _, tr_df, tr_cats, _, tr_pcts, _ = load_therapy_request_data()
    _, te_df, te_cats, _, te_pcts, _ = load_therapy_engagement_data()

    # Aggregate to single Keep % and Remove % per task (weighted by category counts)
    def task_keep_remove_pcts(df, category_col, categories, status_percentages):
        total = len(df)
        if total == 0:
            return 0.0, 0.0
        keep_sum = 0.0
        remove_sum = 0.0
        for cat in categories:
            n = len(df[df[category_col] == cat])
            pct = 100.0 * n / total
            keep_sum += pct * (status_percentages[cat]["kept"] / 100.0)
            remove_sum += pct * ((status_percentages[cat]["modified"] + status_percentages[cat]["removed"]) / 100.0)
        return keep_sum, remove_sum

    # Category column names from the loaders
    si_keep, si_remove = task_keep_remove_pcts(si_df, "Safety type", si_cats, si_pcts)
    tr_keep, tr_remove = task_keep_remove_pcts(tr_df, "Counseling Request", tr_cats, tr_pcts)
    te_keep, te_remove = task_keep_remove_pcts(te_df, "AggregatedSubCategory", te_cats, te_pcts)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(3)
    width = 0.6

    keep_vals = [si_keep, tr_keep, te_keep]
    remove_vals = [si_remove, tr_remove, te_remove]

    ax.bar(x, keep_vals, width, label="Keep", color=KEEP_COLOR)
    ax.bar(x, remove_vals, width, bottom=keep_vals, label="Remove", color=REMOVE_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([])  # Bar labels removed for now
    ax.set_ylabel("% Items", fontsize=18, fontweight="bold")  # +50% from 12
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=15)  # +50% from default ~10
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=22, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="P1/P2 risk plot (single column, 3 models) or Figure 3 summary (Keep/Remove, 3 columns)."
    )
    parser.add_argument("--figure-3-summary", action="store_true",
                        help="Generate Figure 3 summary: 3 columns (SI, therapy request, therapy engagement), Keep (green) / Remove (red). Uses data from data/inputs/intermediate_files/.")
    parser.add_argument("--suicide-csv", type=Path, default=None,
                        help="Path to suicide ideation comprehensive_metrics.csv (required unless --figure-3-summary)")
    parser.add_argument("--therapy-request-csv", type=Path, default=None,
                        help="Path to therapy request comprehensive_metrics.csv (required unless --figure-3-summary)")
    parser.add_argument("--therapy-engagement-csv", type=Path, default=None,
                        help="Path to therapy engagement comprehensive_metrics.csv (required unless --figure-3-summary)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path")
    parser.add_argument("--failure-multiplier", type=float,
                        default=DEFAULT_PARAMS["failure_multiplier"],
                        help="FNR multiplier m (default: 1.0; only for P1/P2 plot)")
    parser.add_argument("--n-mc-samples", type=int,
                        default=DEFAULT_PARAMS["n_mc_samples"],
                        help="Monte Carlo samples for uncertainty (only for P1/P2 plot)")
    parser.add_argument("--uncertainty-style", choices=["ribbon", "errorbar", "both", "none"],
                        default=DEFAULT_PARAMS["uncertainty_style"])
    parser.add_argument("--linear-y", action="store_true", help="Linear y-axis instead of log (P1/P2 only)")
    parser.add_argument("--include-p-harm", action="store_true", help="Add P_harm row (P1/P2 only)")

    args = parser.parse_args()

    if args.figure_3_summary:
        out = args.output or ROOT / "results" / "figure_3_summary_keep_remove.png"
        create_figure3_summary_plot(out)
        return

    # P1/P2 plot: require the three metrics CSVs
    if args.suicide_csv is None or args.therapy_request_csv is None or args.therapy_engagement_csv is None:
        parser.error("--suicide-csv, --therapy-request-csv, and --therapy-engagement-csv are required for the P1/P2 plot (omit --figure-3-summary).")
    for p in [args.suicide_csv, args.therapy_request_csv, args.therapy_engagement_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    params = {
        "failure_multiplier": args.failure_multiplier,
        "n_mc_samples": args.n_mc_samples,
        "uncertainty_style": args.uncertainty_style,
    }
    out = args.output
    if out is None:
        out_dir = ROOT / "results" / "risk_analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / "figure_5_single_column_three_models.png"

    create_single_column_plot(
        args.suicide_csv,
        args.therapy_request_csv,
        args.therapy_engagement_csv,
        out,
        params=params,
        log_y=not args.linear_y,
        include_p_harm=args.include_p_harm,
    )


if __name__ == "__main__":
    main()
