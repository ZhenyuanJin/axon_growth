from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf


def plot_lambda_correlation(summary, output_path, best_lambda):
    """Plot seed correlations and mean plus standard deviation by lambda."""
    grouped = summary.groupby("Lambda")["correlation"].agg(["mean", "std"]).reset_index().sort_values("Lambda")
    fig, ax = cf.get_fig_ax()
    for _, seed_data in summary.groupby("seed"):
        seed_data = seed_data.sort_values("Lambda")
        cf.plt_line(ax, seed_data["Lambda"], seed_data["correlation"], color="0.75", linewidth=0.7, alpha=0.7)
    cf.plt_line(ax, grouped["Lambda"], grouped["mean"], label="Mean", color=cf.BLUE, linewidth=1.8)
    ax.fill_between(
        grouped["Lambda"], grouped["mean"] - grouped["std"].fillna(0),
        grouped["mean"] + grouped["std"].fillna(0), color=cf.BLUE, alpha=0.18
    )
    best_row = grouped.iloc[np.argmin(np.abs(grouped["Lambda"].values - best_lambda))]
    cf.plt_scatter(ax, [best_row["Lambda"]], [best_row["mean"]], label="Best", color=cf.RED, s=28, zorder=5)
    cf.set_ax(ax, xlabel="Lambda (mm)", ylabel="Pearson correlation", legend_loc="upper left", legend_size=9)
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_connectivity_comparison(model_connectivity, output_path):
    """Plot measured and corrected-model human connection matrices."""
    info = uf.AnimalInfo("human", "lariviere")
    measured = info.SC_data.loc[info.source_name, info.target_name]
    model = model_connectivity.loc[info.source_name, info.target_name]
    correlation = uf.data_model_correlation(measured, model, "human", "lariviere", diagonal=False)
    fig, axes = cf.get_fig_ax(nrows=1, ncols=2, ax_width=4.0, ax_height=4.0, fig_width=11.0)
    vmax = max(float(np.nanpercentile(measured.values, 99)), float(np.nanpercentile(model.values, 99)))
    images = []
    for ax, matrix, title in zip(axes, [measured, model], ["Measured CCMe", "Binned 3D EDR CCMe"]):
        images.append(ax.imshow(matrix.values, aspect="auto", origin="upper", vmin=0, vmax=vmax, cmap="viridis"))
        cf.set_ax(ax, xlabel="Target area", ylabel="Source area", title=title, legend=False)
    cf.add_text(axes[1], text=f"Corr={correlation:.3f}", x=0.98, y=0.98, transform="ax", ha="right", va="top", color="white")
    fig.colorbar(images[-1], ax=list(axes), fraction=0.025, pad=0.03, label="CCMe")
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def _aggregate_histogram(centers, expected, empirical, display_bin_count):
    occupied = expected > 0
    native_width = centers[1] - centers[0]
    lower = max(0.0, float(centers[occupied][0] - native_width / 2))
    upper = float(centers[occupied][-1] + native_width / 2)
    edges = np.linspace(lower, upper, display_bin_count + 1)
    display_index = np.minimum(np.searchsorted(edges, centers, side="right") - 1, display_bin_count - 1)
    display_index = np.maximum(display_index, 0)
    expected_display = np.bincount(display_index, weights=expected, minlength=display_bin_count)
    empirical_display = np.bincount(display_index, weights=empirical, minlength=display_bin_count)
    display_centers = (edges[:-1] + edges[1:]) / 2
    return display_centers, expected_display, empirical_display


def plot_distance_histogram(centers, expected, empirical, output_path, display_bin_count=100):
    """Plot aggregated target EDR and empirical distance probabilities."""
    centers, expected, empirical = _aggregate_histogram(
        centers, expected, empirical, display_bin_count
    )
    fig, ax = cf.get_fig_ax()
    cf.plt_line(ax, centers, expected, label="EDR target", color=cf.BLACK, linewidth=1.5)
    cf.plt_scatter(ax, centers, empirical, label="Samples", color=cf.BLUE, s=11)
    cf.set_ax(
        ax, xlabel="Euclidean distance (mm)", ylabel="Probability per display bin",
        legend_loc="upper right", legend_size=9
    )
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def load_connectivity(run_dir):
    return pd.read_csv(Path(run_dir) / "connectivity" / "CCMe_sym_normalized.csv", index_col=0)
