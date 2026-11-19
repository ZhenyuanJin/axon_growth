"""Generate the angle KDE kernel schematic without loading data or running models."""

from target_conditioned_angle_edr import parameters
from target_conditioned_angle_edr import plotting


def main():
    output_dir = parameters.OUTPUT_DIR / "schematics"
    plotting.plot_angle_kernel_schematic(output_dir)
    print(f"Saved angle KDE kernel schematic to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
