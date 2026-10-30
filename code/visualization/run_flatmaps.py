from pathlib import Path

from utils_function import cf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "flatmaps"
DATASETS = [("mouse", "gamanut"), ("marmoset", "majka"),
            ("macaque", "markov40"), ("human", "lariviere")]


def plot_flatmap(ax, coords, title):
    """Draw deterministic region outlines from a dict of (N, 2) float arrays."""
    for vertices in coords.values():
        cf.plt_polygon(ax, vertices, color=cf.BLACK, fill=False, adjust_lim=False)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    cf.set_ax(ax, title=title, legend=False)


def main():
    for animal, dataset in DATASETS:
        path = PROJECT_ROOT / "data" / f"{animal}_{dataset}" / "area_coords" / "coords_dict"
        coords = cf.load_pkl(str(path))
        fig, ax = cf.get_fig_ax()
        plot_flatmap(ax, coords, animal.capitalize())
        cf.save_fig(fig, str(OUTPUT_DIR / animal), formats=["png"], pkl=False)
        print(f"Saved {OUTPUT_DIR / animal}.png", flush=True)


if __name__ == "__main__":
    main()
