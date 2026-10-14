import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "results" / "reference_figs"
GENERATED_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_FILE = PROJECT_ROOT / "results" / "visual_regression.json"
MINIMUM_SIMILARITY = 0.95
MAXIMUM_PAGE_SIZE_RELATIVE_ERROR = 0.01
SIMILARITY_OVERRIDES = {
    "SI_fig_5.pdf": 0.70,
    "SI_fig_6.pdf": 0.90,
}
REFERENCE_NOTES = {
    "SI_fig_5.pdf": "Panel E contains manually added data points that are not part of the automated figure output.",
}
REFERENCE_MAPPING = {
    "fig_1.pdf": "connection_generation_method_overview.pdf",
    "fig_2.pdf": "cross_species_connectivity_fit.pdf",
    "fig_3.pdf": "direction_distribution_comparison.pdf",
    "fig_4.pdf": "marmoset_connection_statistics.pdf",
    "fig_5.pdf": "long_range_connection_analysis.pdf",
    "fig_6.pdf": "cross_species_scaling.pdf",
    "SI_fig_1.pdf": "distance_2d_3d_and_edr.pdf",
    "SI_fig_2.pdf": "angle_distance_dependence.pdf",
    "SI_fig_3.pdf": "marmoset_atlas_comparison.pdf",
    "SI_fig_4.pdf": "connection_statistics_other_species.pdf",
    "SI_fig_5.pdf": "marmoset_area_performance_and_retrograde.pdf",
    "SI_fig_6.pdf": "area_performance_all_species.pdf",
    "SI_fig_7.pdf": "distance_relative_error.pdf",
    "SI_fig_8.pdf": "angle_uniformity_by_distance_threshold.pdf",
    "SI_fig_10.pdf": "marmoset_direction_and_injection_bias.pdf",
    "SI_fig_11.pdf": "direction_preference_performance.pdf",
}


def get_page_size(filename):
    output = subprocess.check_output(["pdfinfo", str(filename)], text=True)
    for line in output.splitlines():
        if line.startswith("Page size:"):
            fields = line.split()
            return float(fields[2]), float(fields[4])
    raise ValueError(f"Page size was not reported for {filename}")


def load_rendered_pdf(filename, temporary_dir, stem):
    output_prefix = Path(temporary_dir) / stem
    subprocess.run(
        ["pdftoppm", "-singlefile", "-png", "-r", "100", str(filename), str(output_prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return np.asarray(Image.open(f"{output_prefix}.png").convert("L"), dtype=float) / 255.0


def get_global_similarity(reference, generated):
    if reference.shape != generated.shape:
        return 0.0
    mean_reference = reference.mean()
    mean_generated = generated.mean()
    variance_reference = reference.var()
    variance_generated = generated.var()
    covariance = np.mean((reference - mean_reference) * (generated - mean_generated))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    return float(
        ((2 * mean_reference * mean_generated + c1) * (2 * covariance + c2))
        / ((mean_reference ** 2 + mean_generated ** 2 + c1) * (variance_reference + variance_generated + c2))
    )


def get_normalized_similarity(reference, generated):
    normalized = []
    for image in [reference, generated]:
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.resize((512, 256), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.GaussianBlur(0.5))
        normalized.append(np.asarray(image, dtype=float) / 255.0)
    return get_global_similarity(*normalized)


def get_page_size_relative_error(reference_size, generated_size):
    return max(abs(generated / reference - 1) for reference, generated in zip(reference_size, generated_size))


def main():
    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(f"Missing local reference directory: {REFERENCE_DIR}")
    report = {}
    with tempfile.TemporaryDirectory() as temporary_dir:
        for reference_name, generated_name in REFERENCE_MAPPING.items():
            reference_path = REFERENCE_DIR / reference_name
            generated_path = GENERATED_DIR / generated_name
            if not reference_path.exists() or not generated_path.exists():
                raise FileNotFoundError(f"Missing comparison input: {reference_path} or {generated_path}")
            reference_size = get_page_size(reference_path)
            generated_size = get_page_size(generated_path)
            reference = load_rendered_pdf(reference_path, temporary_dir, f"reference_{reference_path.stem}")
            generated = load_rendered_pdf(generated_path, temporary_dir, f"generated_{reference_path.stem}")
            exact_similarity = get_global_similarity(reference, generated)
            normalized_similarity = get_normalized_similarity(reference, generated)
            page_size_relative_error = get_page_size_relative_error(reference_size, generated_size)
            minimum_similarity = SIMILARITY_OVERRIDES.get(reference_name, MINIMUM_SIMILARITY)
            report[reference_name] = {
                "generated": generated_name,
                "note": REFERENCE_NOTES.get(reference_name),
                "reference_page_size": reference_size,
                "generated_page_size": generated_size,
                "page_size_relative_error": page_size_relative_error,
                "exact_similarity": exact_similarity,
                "normalized_similarity": normalized_similarity,
                "minimum_similarity": minimum_similarity,
                "passed": page_size_relative_error <= MAXIMUM_PAGE_SIZE_RELATIVE_ERROR and normalized_similarity >= minimum_similarity,
            }
    REPORT_FILE.write_text(json.dumps(report, indent=2, sort_keys=True))
    failures = [name for name, result in report.items() if not result["passed"]]
    if failures:
        raise AssertionError(f"Visual regression failed: {failures}. See {REPORT_FILE}")
    print(f"Visual regression passed. Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
