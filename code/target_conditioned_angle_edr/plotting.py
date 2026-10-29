from pathlib import Path

import numpy as np
import scipy.special

from target_conditioned_angle_edr import parameters
from utils_function import cf


COLORS = {"random": cf.GRAY, "SC_angle_kde": cf.RED}
LABELS = {"random": "Random", "SC_angle_kde": "SC angle KDE"}


def get_angle_density(angle_grid, centers, weights, bandwidth_degrees):
    """Return peak-normalized deterministic von Mises mixture values, shape (grid_size,)."""
    concentration = 1.0 / np.radians(bandwidth_degrees) ** 2
    kernels = np.exp(concentration * (np.cos(angle_grid[:, None] - centers[None, :]) - 1))
    density = kernels @ weights / (2 * np.pi * scipy.special.i0e(concentration))
    if not np.all(np.isfinite(density)) or density.max() <= 0:
        raise ValueError("Invalid circular density.")
    return density / density.max()


def plot_angle_distribution(ax, target, grid, density):
    """Draw one precomputed KDE curve without empirical scatter or an obstructing legend."""
    cf.plt_line(ax, np.degrees(grid), density, color=cf.RED)
    cf.set_ax(ax, xlabel="Target-to-source angle (degree)", ylabel="Peak-normalized KDE",
              title=f"{target} angle distribution", xlim=[-180, 180], ylim=[0, 1.05],
              xtick=[-180, -90, 0, 90, 180], legend=False)


def plot_all_angle_distributions(angle_data, output_dir, bandwidth_degrees):
    """Save one PNG/PDF per target and numeric densities; curves have no random inputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-np.pi, np.pi, 721)
    densities = {}
    for index, (target, (centers, weights)) in enumerate(angle_data.items()):
        density = get_angle_density(grid, centers, weights, bandwidth_degrees)
        densities[target] = density
        fig, ax = cf.get_fig_ax()
        plot_angle_distribution(ax, target, grid, density)
        cf.save_fig(fig, str(output_dir / f"{index:03d}_{target.replace('/', '_')}"),
                    formats=["png", "pdf"], pkl=False, bbox_inches="tight")
    np.savez_compressed(output_dir / "angle_densities.npz", angle_degrees=np.degrees(grid),
                        target_names=np.asarray(list(densities)), density=np.array(list(densities.values())))


def plot_scan_metric(ax, scan, metric, ylabel, legend_anchor=(0.5, 1.18)):
    """Draw mean and sample SD for five independently evaluated runs at every Lambda."""
    for method in parameters.DIRECTION_METHODS:
        subset = scan.loc[scan.direction_method == method].sort_values("Lambda")
        x, y = subset.Lambda.to_numpy(), subset[f"{metric}_mean"].to_numpy()
        sd = subset[f"{metric}_std"].to_numpy()
        cf.plt_line(ax, x, y, color=COLORS[method], label=LABELS[method])
        ax.fill_between(x, y - sd, y + sd, color=COLORS[method], alpha=0.17)
    cf.set_ax(ax, xlabel="Lambda", ylabel=ylabel, legend_loc="upper center",
              bbox_to_anchor=legend_anchor, legend_kwargs={"ncols": 2, "frameon": False})


def plot_correlation_improvement(scan, optimal_rows, comparison, output_dir, matrix_label):
    """Plot repeated overall correlations and target means with SD error bars."""
    fig, axes = cf.get_fig_ax(ncols=2)
    plot_scan_metric(axes[0], scan, "correlation", f"{matrix_label} correlation", legend_anchor=(0.5, 1.4))
    for _, row in optimal_rows.iterrows():
        cf.plt_scatter(axes[0], np.atleast_1d(row.Lambda), np.atleast_1d(row.correlation_mean),
                       color=COLORS[row.direction_method])
    gains = optimal_rows.set_index("direction_method").correlation_mean
    cf.set_ax(axes[0], title=f"Overall correlation\nGain = {gains['SC_angle_kde'] - gains['random']:.3f}",
              legend=False)
    x = comparison.correlation_random_mean.to_numpy()
    y = comparison.correlation_SC_angle_kde_mean.to_numpy()
    xsd = comparison.correlation_random_std.to_numpy()
    ysd = comparison.correlation_SC_angle_kde_std.to_numpy()
    lower = max(-1.05, min(np.min(x - xsd), np.min(y - ysd)) - 0.06)
    upper = min(1.06, max(np.max(x + xsd), np.max(y + ysd)) + 0.06)
    cf.plt_line(axes[1], [lower, upper], [lower, upper], color=cf.GRAY, linestyle="--")
    cf.add_errorbar(axes[1], x, y, ysd, xerr=xsd, fmt="none", color=cf.GRAY, alpha=0.35)
    cf.plt_scatter(axes[1], x, y, color=cf.BLUE)
    improved = int((comparison.correlation_difference_mean > 0).sum())
    cf.set_ax(axes[1], xlabel="Random target correlation", ylabel="SC angle KDE target correlation",
              title=f"Target means\nImproved {improved}/{len(comparison)}", legend=False,
              xlim=[lower, upper], ylim=[lower, upper])
    axes[1].set_aspect("equal", adjustable="box")
    cf.save_fig(fig, str(Path(output_dir) / "correlation_improvement"), formats=["png", "pdf"],
                pkl=False, bbox_inches="tight")


def plot_metric_scans(scan, output_dir, matrix_label):
    """Plot the remaining precomputed fit and acceptance metrics with mean +/- sample SD."""
    metrics = [("log_correlation", "Log correlation"), ("rmae_both", "RMAE"),
               ("median_target_correlation", "Median target correlation"),
               ("mean_acceptance_rate", "Mean acceptance rate")]
    fig, axes = cf.get_fig_ax(nrows=2, ncols=2)
    for ax, (metric, label) in zip(axes.flat, metrics):
        plot_scan_metric(ax, scan, metric, label)
    cf.save_fig(fig, str(Path(output_dir) / "metric_scans"), formats=["png", "pdf"],
                pkl=False, bbox_inches="tight")
