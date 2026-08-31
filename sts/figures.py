"""The 8 project figures. Every figure function takes already-computed
data (never re-derives results); example-selection functions take an
explicit `criterion` string so visualization examples are never picked
just because they look good."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig1_pipeline_diagram(out_path: Path) -> None:
    """Teacher -> student -> graph pipeline schematic."""
    fig, ax = plt.subplots(figsize=(10, 3))
    stages = ["CPTAC H&E\npatch", "GenBio-PathFM\n(1.1B, frozen)", "ConvNeXt V2-Atto\nstudent (3.7M)",
              "320-D\nembedding", "PNA graph\ncorrection", "y_final =\ny_patch + dy"]
    for i, s in enumerate(stages):
        ax.add_patch(plt.Rectangle((i * 2, 0), 1.6, 1, fill=True, facecolor="#e8eef7", edgecolor="black"))
        ax.text(i * 2 + 0.8, 0.5, s, ha="center", va="center", fontsize=8)
        if i < len(stages) - 1:
            ax.annotate("", xy=(i * 2 + 2, 0.5), xytext=(i * 2 + 1.6, 0.5),
                        arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.5, len(stages) * 2)
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig2_wsi_patch_examples(thumbnails: list[np.ndarray], patch_examples: list[np.ndarray], out_path: Path) -> None:
    n = min(4, len(thumbnails))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        axes[0, i].imshow(thumbnails[i])
        axes[0, i].set_title(f"WSI {i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(patch_examples[i])
        axes[1, i].set_title("sampled patch")
        axes[1, i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig3_representation_agreement(cosine_sims: dict[str, np.ndarray], out_path: Path) -> None:
    """GenBio vs student cosine-similarity distribution, split by cancer."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([v for v in cosine_sims.values()], labels=list(cosine_sims.keys()))
    ax.set_ylabel("cosine similarity (student proj vs teacher)")
    ax.set_title("Student-teacher representation agreement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig4_ablation_bars(results_df: pd.DataFrame, out_path: Path, metric_col: str = "fisher_z_mean_patient_R") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(results_df["model"], results_df[metric_col])
    ax.set_ylabel(metric_col)
    ax.set_title("Patch vs shuffled vs local vs multiscale")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig5_destruction_dose_response(q_values: list[float], r_mean: list[float], r_ci_low: list[float],
                                    r_ci_high: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q_values, r_mean, marker="o")
    ax.fill_between(q_values, r_ci_low, r_ci_high, alpha=0.2)
    ax.set_xlabel("fraction of physical edges replaced (q)")
    ax.set_ylabel("patient-level R")
    ax.set_title("Spatial destruction dose-response")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig6_morans_vs_improvement(morans_i: np.ndarray, delta_r: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(morans_i, delta_r, alpha=0.4, s=10)
    ax.set_xlabel("ground-truth Moran's I")
    ax.set_ylabel("delta r (multiscale - patch)")
    ax.set_title("Spatial autocorrelation vs graph improvement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig7_efficiency_vs_performance(param_counts: list[float], performance: list[float], labels: list[str],
                                    out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(param_counts, performance)
    for x, y, l in zip(param_counts, performance, labels):
        ax.annotate(l, (x, y), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("parameter count (log scale)")
    ax.set_ylabel("performance (patient-level R)")
    ax.set_title("Compute vs prediction performance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig8_spatial_maps(coords_um: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, gene_name: str,
                       selection_criterion: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sc0 = axes[0].scatter(coords_um[:, 0], coords_um[:, 1], c=y_true, s=8, cmap="viridis")
    axes[0].set_title(f"{gene_name} -- ground truth")
    axes[0].invert_yaxis()
    plt.colorbar(sc0, ax=axes[0])
    sc1 = axes[1].scatter(coords_um[:, 0], coords_um[:, 1], c=y_pred, s=8, cmap="viridis")
    axes[1].set_title(f"{gene_name} -- predicted")
    axes[1].invert_yaxis()
    plt.colorbar(sc1, ax=axes[1])
    fig.suptitle(f"Selection criterion: {selection_criterion}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def representative_gene_selection(per_gene_r: pd.Series, criterion: str = "median") -> str:
    """Predetermined, documented example selection (never cherry-picked):
    median-performing gene by default; also supports 'best' / 'worst' /
    'top_quartile_median'."""
    s = per_gene_r.dropna().sort_values()
    if len(s) == 0:
        return None
    if criterion == "median":
        return s.index[len(s) // 2]
    if criterion == "best":
        return s.index[-1]
    if criterion == "worst":
        return s.index[0]
    if criterion == "top_quartile_median":
        q75_idx = int(len(s) * 0.75)
        return s.index[q75_idx]
    raise ValueError(criterion)
