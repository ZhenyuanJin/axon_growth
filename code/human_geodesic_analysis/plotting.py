import numpy as np
from utils_function import cf


def _plot_finite_segments(ax, x, y, label, **kwargs):
    finite = np.isfinite(y)
    starts = np.flatnonzero(finite & ~np.r_[False, finite[:-1]])
    stops = np.flatnonzero(finite & ~np.r_[finite[1:], False]) + 1
    for segment_index, (start, stop) in enumerate(zip(starts, stops)):
        cf.plt_line(
            ax, x[start:stop], y[start:stop],
            label=label if segment_index == 0 else None, **kwargs
        )


def plot_ratio_binned_rmae(binned_rmae, output_path):
    """Plot 2D and 3D RMAE against the shared physical distance ratio."""
    fig, ax = cf.get_fig_ax()
    styles = [
        ("existing", "o", "-"),
        ("non-link", "s", "--"),
    ]
    for category, marker, linestyle in styles:
        table = binned_rmae[binned_rmae["link_category"] == category]
        _plot_finite_segments(
            ax, table["ratio_mean"].to_numpy(), table["rmae_2d"].to_numpy(),
            f"2D {category}", color=cf.BLUE, marker=marker, linestyle=linestyle
        )
        _plot_finite_segments(
            ax, table["ratio_mean"].to_numpy(), table["rmae_3d"].to_numpy(),
            f"3D {category}", color=cf.ORANGE, marker=marker, linestyle=linestyle
        )
    cf.set_ax(
        ax, xlabel="Mean geodesic / Euclidean distance", ylabel="RMAE",
        legend_loc="lower center", bbox_to_anchor=(0.5, 1.0),
        legend_kwargs={"ncols": 2}
    )
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)
