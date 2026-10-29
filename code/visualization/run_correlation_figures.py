import gc

from visualization.run_paper_figures import get_figure_generator


FIGURE_NAMES = [
    "cross_species_connectivity_fit",
    "long_range_connection_analysis",
    "cross_species_scaling",
    "marmoset_atlas_comparison",
    "marmoset_area_performance_and_retrograde",
    "area_performance_all_species",
    "direction_preference_performance",
]


def main():
    figures = get_figure_generator()
    for name in FIGURE_NAMES:
        print(f"Plotting {name}", flush=True)
        getattr(figures, f"plot_{name}")()
        gc.collect()
    print(f"Figures saved to {figures.figdir}", flush=True)


if __name__ == "__main__":
    main()
