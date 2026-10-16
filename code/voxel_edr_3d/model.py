import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.signal
import scipy.special
from utils_function import cf
from utils_function import utils_function as uf


def _save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def _load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _get_voxel_data(project_root, animal, dataset_key):
    """Load voxel coordinates and area ids for a supported dataset.

    Returns coordinates with shape (voxel_count, 3), float64 dtype, an int16
    area-id vector, ordered area names, and physical grid spacing in mm. No
    randomness is used.
    """
    project_root = Path(project_root)
    dataset_name = f"{animal}_{dataset_key}"
    if dataset_name not in {"marmoset_majka", "macaque_markov40"}:
        raise ValueError(f"Unsupported voxel EDR dataset: {dataset_name}")
    info = uf.AnimalInfo(animal, dataset_key)
    voxel_path = project_root / "data" / dataset_name / "threeD_data" / "area_voxel_dict_3D.joblib"
    voxel_dict = cf.load_joblib(str(voxel_path))
    missing = sorted(set(info.full_name) - set(voxel_dict))
    extra = sorted(set(voxel_dict) - set(info.full_name))
    if missing or extra:
        raise ValueError(f"Voxel-area mismatch. Missing={missing}, extra={extra}")

    coordinates = []
    area_ids = []
    for area_id, area_name in enumerate(info.full_name):
        area_coordinates = np.asarray(voxel_dict[area_name], dtype=np.float64)
        if area_coordinates.ndim != 2 or area_coordinates.shape[1] != 3:
            raise ValueError(f"Invalid voxel coordinates for {area_name}: {area_coordinates.shape}")
        coordinates.append(area_coordinates)
        area_ids.append(np.full(len(area_coordinates), area_id, dtype=np.int16))
    coordinates = np.concatenate(coordinates, axis=0)
    area_ids = np.concatenate(area_ids)

    if dataset_name == "marmoset_majka":
        atlas_path = project_root / "raw_data" / dataset_name / "threeD_data" / "atlas_segmentation_200um.nii.gz"
        spacing = np.asarray(nib.load(atlas_path).header.get_zooms()[:3], dtype=np.float64)
    else:
        spacing = []
        for axis in range(3):
            unique_coordinates = np.unique(coordinates[:, axis])
            spacing.append(round(float(np.median(np.diff(unique_coordinates))), 9))
        spacing = np.asarray(spacing, dtype=np.float64)
    return coordinates, area_ids, list(info.full_name), spacing


def _get_grid_indices(coordinates, spacing):
    origin = np.min(coordinates, axis=0)
    grid_indices = np.rint((coordinates - origin) / spacing).astype(np.int16)
    reconstructed = origin + grid_indices.astype(np.float64) * spacing
    if not np.allclose(coordinates, reconstructed, rtol=0.0, atol=2e-6):
        max_error = float(np.max(np.abs(coordinates - reconstructed)))
        raise ValueError(f"Coordinates do not lie on the expected voxel grid; max error={max_error}")
    if len(np.unique(grid_indices, axis=0)) != len(grid_indices):
        raise ValueError("Duplicate voxel grid coordinates were found.")
    return grid_indices, origin


def _get_integer_spacing(spacing, max_displacement):
    ratios = [float(value).as_integer_ratio() for value in spacing]
    denominator = max(item[1] for item in ratios)
    if any(denominator % item[1] != 0 for item in ratios):
        raise ValueError("Voxel spacings do not have a common binary denominator.")
    integer_spacing = np.asarray([num * (denominator // den) for num, den in ratios], dtype=np.int64)
    max_key = sum((int(integer_spacing[i]) * int(max_displacement[i])) ** 2 for i in range(3))
    if max_key > np.iinfo(np.int64).max:
        denominator = 1_000_000
        integer_spacing = np.rint(spacing * denominator).astype(np.int64)
    return integer_spacing, denominator


def _get_exact_distance_keys(displacements, integer_spacing, denominator):
    max_displacement = np.max(np.abs(displacements), axis=0).astype(object)
    max_key = sum((int(integer_spacing[i]) * int(max_displacement[i])) ** 2 for i in range(3))
    if max_key > np.iinfo(np.int64).max:
        raise OverflowError("Exact squared-distance keys exceed int64.")
    scaled = displacements.astype(np.int64) * integer_spacing.reshape(1, 3)
    keys = np.sum(scaled * scaled, axis=1, dtype=np.int64)
    distances = np.sqrt(keys.astype(np.float64)) / denominator
    return keys, distances


def build_displacement_catalog(project_root, cache_dir, animal="marmoset", dataset_key="majka", coordinates=None, area_ids=None, area_names=None, spacing=None):
    """Build the exact implicit pair-distance catalog.

    The catalog represents every ordered pair of distinct occupied voxels.
    Equal physical distances share an exact integer squared-distance key based
    on the stored grid spacings. Derived arrays are saved independently so
    later workers can memory-map them. No randomness is used.
    """
    project_root = Path(project_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if coordinates is None:
        coordinates, area_ids, area_names, spacing = _get_voxel_data(project_root, animal, dataset_key)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    area_ids = np.asarray(area_ids, dtype=np.int16)
    spacing = np.asarray(spacing, dtype=np.float64)
    if len(coordinates) != len(area_ids):
        raise ValueError("coordinates and area_ids must have the same length.")

    grid_indices, origin = _get_grid_indices(coordinates, spacing)
    grid_shape = tuple((np.max(grid_indices, axis=0) + 1).astype(int))
    voxel_map = np.full(grid_shape, -1, dtype=np.int32)
    voxel_map[tuple(grid_indices.T)] = np.arange(len(grid_indices), dtype=np.int32)
    occupancy = (voxel_map >= 0).astype(np.float64)

    pair_counts_grid = scipy.signal.fftconvolve(occupancy, occupancy[::-1, ::-1, ::-1], mode="full")
    pair_counts_grid = np.rint(pair_counts_grid).astype(np.int64)
    if np.min(pair_counts_grid) < 0:
        raise ValueError("FFT autocorrelation produced negative pair counts.")
    expected_all_pairs = len(grid_indices) ** 2
    if int(np.sum(pair_counts_grid)) != expected_all_pairs:
        raise ValueError("FFT autocorrelation pair count does not match voxel_count squared.")

    positions = np.argwhere(pair_counts_grid > 0)
    center = np.asarray(grid_shape, dtype=np.int64) - 1
    displacements = positions.astype(np.int64) - center.reshape(1, 3)
    displacement_counts = pair_counts_grid[tuple(positions.T)]
    nonzero = np.any(displacements != 0, axis=1)
    displacements = displacements[nonzero]
    displacement_counts = displacement_counts[nonzero]
    expected_distinct_pairs = len(grid_indices) * (len(grid_indices) - 1)
    if int(np.sum(displacement_counts)) != expected_distinct_pairs:
        raise ValueError("Distinct-pair count does not match voxel_count * (voxel_count - 1).")

    integer_spacing, spacing_denominator = _get_integer_spacing(spacing, np.asarray(grid_shape) - 1)
    distance_keys, _ = _get_exact_distance_keys(displacements, integer_spacing, spacing_denominator)
    order = np.argsort(distance_keys, kind="stable")
    distance_keys = distance_keys[order]
    displacements = displacements[order].astype(np.int16)
    displacement_counts = displacement_counts[order]
    class_keys, class_starts, class_sizes = np.unique(distance_keys, return_index=True, return_counts=True)
    class_pair_counts = np.add.reduceat(displacement_counts, class_starts)
    class_ids = np.repeat(np.arange(len(class_keys), dtype=np.int32), class_sizes)
    class_distances = np.sqrt(class_keys.astype(np.float64)) / spacing_denominator

    arrays = {
        "coordinates": coordinates,
        "grid_indices": grid_indices,
        "area_ids": area_ids,
        "voxel_map": voxel_map,
        "displacements": displacements,
        "displacement_counts": displacement_counts,
        "distance_class_ids": class_ids,
        "class_distances": class_distances,
        "class_pair_counts": class_pair_counts,
    }
    for name, value in arrays.items():
        np.save(cache_dir / f"{name}.npy", value, allow_pickle=False)
    metadata = {
        "animal": animal,
        "dataset_key": dataset_key,
        "voxel_count": len(grid_indices),
        "area_names": list(area_names),
        "spacing_mm": spacing.tolist(),
        "integer_spacing": integer_spacing.tolist(),
        "spacing_denominator": int(spacing_denominator),
        "grid_origin_mm": origin.tolist(),
        "grid_shape": [int(value) for value in grid_shape],
        "displacement_count": len(displacements),
        "distance_class_count": len(class_distances),
        "ordered_distinct_pair_count": int(expected_distinct_pairs),
    }
    _save_json(metadata, cache_dir / "metadata.json")
    return load_displacement_catalog(cache_dir)


def load_displacement_catalog(cache_dir, mmap_mode="r"):
    """Load an implicit distance catalog, memory-mapping numeric arrays."""
    cache_dir = Path(cache_dir)
    metadata = _load_json(cache_dir / "metadata.json")
    names = [
        "coordinates",
        "grid_indices",
        "area_ids",
        "voxel_map",
        "displacements",
        "displacement_counts",
        "distance_class_ids",
        "class_distances",
        "class_pair_counts",
    ]
    catalog = {name: np.load(cache_dir / f"{name}.npy", mmap_mode=mmap_mode, allow_pickle=False) for name in names}
    catalog["metadata"] = metadata
    return catalog


def get_displacement_probabilities(catalog, Lambda):
    """Return normalized oriented-displacement probabilities for one EDR scale.

    Each point pair receives exp(-distance / Lambda) divided by the total
    number of ordered pairs in its exact distance class. The returned float64
    probabilities sum to one. No randomness is used.
    """
    if not np.isfinite(Lambda) or Lambda <= 0:
        raise ValueError("Lambda must be a positive finite value.")
    class_distances = np.asarray(catalog["class_distances"])
    class_pair_counts = np.asarray(catalog["class_pair_counts"])
    class_ids = np.asarray(catalog["distance_class_ids"])
    displacement_counts = np.asarray(catalog["displacement_counts"])
    log_class_weights = -class_distances / Lambda
    log_normalizer = scipy.special.logsumexp(log_class_weights)
    class_probabilities = np.exp(log_class_weights - log_normalizer)
    probabilities = class_probabilities[class_ids] * displacement_counts / class_pair_counts[class_ids]
    probability_sum = float(np.sum(probabilities, dtype=np.float64))
    if not np.isclose(probability_sum, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Displacement probabilities do not sum to one: {probability_sum}")
    probabilities /= probability_sum
    return probabilities


def _get_valid_pair_indices(catalog, displacement):
    grid_indices = np.asarray(catalog["grid_indices"])
    voxel_map = np.asarray(catalog["voxel_map"])
    target_grid = grid_indices.astype(np.int32) + displacement.astype(np.int32)
    shape = np.asarray(voxel_map.shape, dtype=np.int32)
    in_bounds = np.all((target_grid >= 0) & (target_grid < shape), axis=1)
    source_indices = np.flatnonzero(in_bounds)
    bounded_targets = target_grid[in_bounds]
    target_indices = voxel_map[tuple(bounded_targets.T)]
    occupied = target_indices >= 0
    return source_indices[occupied].astype(np.uint32), target_indices[occupied].astype(np.uint32)


def _sample_pairs_for_displacement(catalog, displacement, pair_count, sample_count, rng):
    voxel_count = len(catalog["grid_indices"])
    expected_proposals = sample_count * voxel_count / pair_count
    if expected_proposals >= voxel_count / 2:
        valid_sources, valid_targets = _get_valid_pair_indices(catalog, displacement)
        if len(valid_sources) != pair_count:
            raise ValueError(f"Catalog count mismatch for displacement {displacement.tolist()}")
        selected = rng.randint(0, pair_count, size=sample_count)
        return valid_sources[selected], valid_targets[selected]

    voxel_map = np.asarray(catalog["voxel_map"])
    grid_indices = np.asarray(catalog["grid_indices"])
    shape = np.asarray(voxel_map.shape, dtype=np.int32)
    source_output = np.empty(sample_count, dtype=np.uint32)
    target_output = np.empty(sample_count, dtype=np.uint32)
    filled = 0
    acceptance_rate = pair_count / voxel_count
    while filled < sample_count:
        remaining = sample_count - filled
        proposal_count = max(16, int(math.ceil(remaining / acceptance_rate * 1.2)))
        candidates = rng.randint(0, voxel_count, size=proposal_count)
        target_grid = grid_indices[candidates].astype(np.int32) + displacement.astype(np.int32)
        in_bounds = np.all((target_grid >= 0) & (target_grid < shape), axis=1)
        candidates = candidates[in_bounds]
        target_grid = target_grid[in_bounds]
        targets = voxel_map[tuple(target_grid.T)]
        occupied = targets >= 0
        candidates = candidates[occupied]
        targets = targets[occupied]
        take = min(remaining, len(candidates))
        if take:
            source_output[filled:filled + take] = candidates[:take]
            target_output[filled:filled + take] = targets[:take]
            filled += take
    return source_output, target_output


def get_connection_samples(catalog, Lambda, connection_count, seed):
    """Sample ordered voxel connections from the corrected EDR distribution.

    Returns uint32 source and target voxel-index arrays of shape
    (connection_count,). Randomness is local and determined only by seed.
    """
    if int(connection_count) != connection_count or connection_count <= 0:
        raise ValueError("connection_count must be a positive integer.")
    connection_count = int(connection_count)
    rng = cf.get_local_rng(seed=seed)
    probabilities = get_displacement_probabilities(catalog, Lambda)
    chosen_displacements = rng.choice(len(probabilities), size=connection_count, p=probabilities)
    order = np.argsort(chosen_displacements, kind="stable")
    sorted_choices = chosen_displacements[order]
    unique_choices, starts, sizes = np.unique(sorted_choices, return_index=True, return_counts=True)
    sorted_sources = np.empty(connection_count, dtype=np.uint32)
    sorted_targets = np.empty(connection_count, dtype=np.uint32)
    for displacement_index, start, size in zip(unique_choices, starts, sizes):
        displacement = np.asarray(catalog["displacements"][displacement_index])
        pair_count = int(catalog["displacement_counts"][displacement_index])
        sources, targets = _sample_pairs_for_displacement(catalog, displacement, pair_count, int(size), rng)
        sorted_sources[start:start + size] = sources
        sorted_targets[start:start + size] = targets
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(connection_count)
    return sorted_sources[inverse_order], sorted_targets[inverse_order]


def get_connection_matrix(catalog, source_indices, target_indices):
    """Aggregate sampled voxel pairs into an area-level directed count matrix."""
    source_indices = np.asarray(source_indices)
    target_indices = np.asarray(target_indices)
    if source_indices.shape != target_indices.shape:
        raise ValueError("source_indices and target_indices must have identical shapes.")
    area_ids = np.asarray(catalog["area_ids"])
    area_names = catalog["metadata"]["area_names"]
    area_count = len(area_names)
    flat_ids = area_ids[source_indices] * area_count + area_ids[target_indices]
    matrix = np.bincount(flat_ids, minlength=area_count * area_count).reshape(area_count, area_count)
    if int(np.sum(matrix)) != len(source_indices):
        raise ValueError("Area connection counts do not sum to the sampled connection count.")
    return pd.DataFrame(matrix, index=area_names, columns=area_names)


def run_connection_task(task, cache_dir, output_dir, save_pairs=False, force_run=False):
    """Run and save one deterministic voxel-EDR sampling task."""
    Lambda = float(task["Lambda"])
    seed = int(task["seed"])
    connection_count = int(task["connection_count"])
    task_id = task["task_id"]
    animal = task.get("animal", "marmoset")
    dataset_key = task.get("dataset_key", "majka")
    run_dir = Path(output_dir) / task_id
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force_run and (not save_pairs or (run_dir / "source_voxel_indices.npy").exists()):
        result = _load_json(metrics_path)
        result["run_dir"] = str(run_dir)
        return result

    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_displacement_catalog(cache_dir)
    source_indices, target_indices = get_connection_samples(catalog, Lambda, connection_count, seed)
    connection_matrix = get_connection_matrix(catalog, source_indices, target_indices)
    connection_matrix.to_csv(run_dir / "connection_counts.csv")
    connectivity = uf.get_SC_model_dict_from_CCMt_model(connection_matrix, animal, dataset_key, str(run_dir))
    info = uf.AnimalInfo(animal, dataset_key)
    correlation = uf.data_model_correlation(
        SC_data=info.SC_data,
        SC_model=connectivity["FLNe"],
        animal=animal,
        dataset_key=dataset_key,
        diagonal=False,
    )
    if save_pairs:
        np.save(run_dir / "source_voxel_indices.npy", source_indices, allow_pickle=False)
        np.save(run_dir / "target_voxel_indices.npy", target_indices, allow_pickle=False)
    config = {
        "animal": animal,
        "dataset_key": dataset_key,
        "Lambda": Lambda,
        "seed": seed,
        "connection_count": connection_count,
        "task_id": task_id,
        "save_pairs": bool(save_pairs),
    }
    metrics = {**config, "correlation": float(correlation)}
    _save_json(config, run_dir / "config.json")
    _save_json(metrics, metrics_path)
    metrics["run_dir"] = str(run_dir)
    return metrics
