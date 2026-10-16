from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf


def plot_lambda_correlation(summary, output_path, best_lambda):
    """Plot seed correlations and mean plus standard deviation by lambda."""
    grouped = summary.groupby("Lambda")["correlation"].agg(["mean", "std"]).reset_index().sort_values("Lambda")
    fig, ax = cf.get_fig_ax()
    for seed, seed_data in summary.groupby("seed"):
        seed_data = seed_data.sort_values("Lambda")
        cf.plt_line(ax, seed_data["Lambda"], seed_data["correlation"], color="0.75", linewidth=0.7, alpha=0.7)
    cf.plt_line(ax, grouped["Lambda"], grouped["mean"], label="Mean", color=cf.BLUE, linewidth=1.8)
    ax.fill_between(grouped["Lambda"], grouped["mean"] - grouped["std"].fillna(0), grouped["mean"] + grouped["std"].fillna(0), color=cf.BLUE, alpha=0.18)
    best_row = grouped.iloc[np.argmin(np.abs(grouped["Lambda"].values - best_lambda))]
    cf.plt_scatter(ax, [best_row["Lambda"]], [best_row["mean"]], label="Best", color=cf.RED, s=28, zorder=5)
    cf.set_ax(ax, xlabel="Lambda (mm)", ylabel="Pearson correlation", xlog=True, legend_loc="upper left", legend_size=9)
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_connectivity_comparison(model_flne, output_path, animal="marmoset", dataset_key="majka"):
    """Plot measured and best-model connection matrices."""
    info = uf.AnimalInfo(animal, dataset_key)
    measured = info.SC_data.loc[info.source_name, info.target_name]
    model = model_flne.loc[info.source_name, info.target_name]
    correlation = uf.data_model_correlation(measured, model, animal, dataset_key, diagonal=False)
    fig, axes = cf.get_fig_ax(nrows=1, ncols=2, ax_width=4.0, ax_height=4.0, fig_width=11.0)
    vmax = max(float(np.nanpercentile(measured.values, 99)), float(np.nanpercentile(model.values, 99)))
    images = []
    for ax, matrix, title in zip(axes, [measured, model], ["Measured FLNe", "3D EDR FLNe"]):
        images.append(ax.imshow(matrix.values, aspect="auto", origin="upper", vmin=0, vmax=vmax, cmap="viridis"))
        cf.set_ax(ax, xlabel="Target area", ylabel="Source area", title=title, legend=False)
    cf.add_text(axes[1], text=f"Corr={correlation:.3f}", x=0.98, y=0.98, transform="ax", ha="right", va="top", color="white")
    fig.colorbar(images[-1], ax=list(axes), fraction=0.025, pad=0.03, label="FLNe")
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_distance_distribution(catalog, source_indices, target_indices, Lambda, output_path):
    """Plot sampled distances against the normalized target EDR distance law."""
    coordinates = np.asarray(catalog["coordinates"])
    distances = np.linalg.norm(coordinates[target_indices] - coordinates[source_indices], axis=1)
    max_distance = float(np.max(catalog["class_distances"]))
    bins = np.linspace(0, max_distance, 80)
    empirical, edges = np.histogram(distances, bins=bins, density=False)
    empirical = empirical / np.sum(empirical)
    class_distances = np.asarray(catalog["class_distances"])
    class_weights = np.exp(-class_distances / Lambda)
    class_weights /= np.sum(class_weights)
    expected, _ = np.histogram(class_distances, bins=bins, weights=class_weights)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = cf.get_fig_ax()
    cf.plt_line(ax, centers, expected, label="EDR target", color=cf.BLACK, linewidth=1.5)
    cf.plt_scatter(ax, centers, empirical, label="Samples", color=cf.BLUE, s=11)
    cf.set_ax(
        ax,
        xlabel="Euclidean distance (mm)",
        ylabel="Probability per bin",
        legend_loc="upper right",
        legend_size=9,
    )
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def load_flne(run_dir):
    return pd.read_csv(Path(run_dir) / "connectivity" / "FLNe.csv", index_col=0)
