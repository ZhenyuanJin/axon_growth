import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.signal
import scipy.special
import scipy.spatial.distance
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


# ============================================================
# Human binned vertex model
# ============================================================

def _load_human_vertices(project_root):
    project_root = Path(project_root)
    info = uf.AnimalInfo("human", "lariviere")
    vertex_path = project_root / "data" / "human_lariviere" / "threeD_data" / "area_voxel_dict_3D.joblib"
    vertex_dict = cf.load_joblib(str(vertex_path))
    missing = sorted(set(info.full_name) - set(vertex_dict))
    extra = sorted(set(vertex_dict) - set(info.full_name))
    if missing or extra:
        raise ValueError(f"Vertex-area mismatch. Missing={missing}, extra={extra}")
    coordinates = []
    area_ids = []
    area_starts = []
    start = 0
    for area_id, area_name in enumerate(info.full_name):
        area_coordinates = np.asarray(vertex_dict[area_name], dtype=np.float64)
        if area_coordinates.ndim != 2 or area_coordinates.shape[1] != 3:
            raise ValueError(f"Invalid vertex coordinates for {area_name}: {area_coordinates.shape}")
        if len(area_coordinates) == 0:
            raise ValueError(f"Area {area_name} has no vertices.")
        area_starts.append(start)
        coordinates.append(area_coordinates)
        area_ids.append(np.full(len(area_coordinates), area_id, dtype=np.int16))
        start += len(area_coordinates)
    coordinates = np.concatenate(coordinates)
    area_ids = np.concatenate(area_ids)
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("Duplicate human surface vertices were found.")
    return coordinates, area_ids, np.asarray(area_starts, dtype=np.int32), list(info.full_name)


def _validate_input_arrays(coordinates, area_ids, area_names):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    area_ids = np.asarray(area_ids, dtype=np.int16)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (point_count, 3).")
    if area_ids.shape != (len(coordinates),):
        raise ValueError("area_ids must have shape (point_count,).")
    if len(coordinates) != len(np.unique(coordinates, axis=0)):
        raise ValueError("coordinates must be unique.")
    unique_ids = np.unique(area_ids)
    if not np.array_equal(unique_ids, np.arange(len(area_names))):
        raise ValueError("area_ids must be contiguous from zero and match area_names.")
    order = np.argsort(area_ids, kind="stable")
    coordinates = coordinates[order]
    area_ids = area_ids[order]
    counts = np.bincount(area_ids, minlength=len(area_names))
    area_starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)
    return coordinates, area_ids, area_starts, list(area_names)


def _get_area_bounds(area_starts, vertex_count):
    starts = np.asarray(area_starts, dtype=np.int64)
    stops = np.concatenate((starts[1:], [vertex_count]))
    return starts, stops


def _get_bin_index(distances, bin_width, bin_count):
    indices = np.floor(np.asarray(distances) / bin_width).astype(np.int64)
    return np.minimum(indices, bin_count - 1)


def _get_area_pair_bin_counts(coordinates, area_starts, bin_width):
    vertex_count = len(coordinates)
    starts, stops = _get_area_bounds(area_starts, vertex_count)
    extent = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
    upper_bound = float(np.linalg.norm(extent))
    bin_count = int(np.ceil(upper_bound / bin_width))
    counts = np.zeros((bin_count, len(starts), len(starts)), dtype=np.uint32)
    max_distance = 0.0
    for source_area in range(len(starts)):
        source_coordinates = coordinates[starts[source_area]:stops[source_area]]
        for target_area in range(source_area, len(starts)):
            target_coordinates = coordinates[starts[target_area]:stops[target_area]]
            distances = scipy.spatial.distance.cdist(source_coordinates, target_coordinates)
            max_distance = max(max_distance, float(np.max(distances)))
            if source_area == target_area:
                distances = distances[~np.eye(len(distances), dtype=bool)]
            bin_indices = _get_bin_index(distances.ravel(), bin_width, bin_count)
            pair_counts = np.bincount(bin_indices, minlength=bin_count).astype(np.uint32)
            counts[:, source_area, target_area] = pair_counts
            if target_area != source_area:
                counts[:, target_area, source_area] = pair_counts
    global_counts = np.sum(counts, axis=(1, 2), dtype=np.int64)
    occupied = np.flatnonzero(global_counts)
    if len(occupied) == 0:
        raise ValueError("No non-self vertex pairs were found.")
    final_bin_count = int(occupied[-1] + 1)
    counts = counts[:final_bin_count]
    edges = np.arange(final_bin_count + 1, dtype=np.float64) * bin_width
    return counts, edges, max_distance


def build_human_bin_catalog(project_root, cache_dir, bin_width=0.1, coordinates=None, area_ids=None, area_names=None):
    """Build exact area-pair counts for fixed-width Euclidean distance bins.

    Inputs are point coordinates with shape (N, 3), contiguous integer area ids,
    and area names. When arrays are omitted, human Lariviere vertices are loaded.
    Outputs contain every ordered non-self pair exactly once. No randomness is used.
    """
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be a positive finite value.")
    if coordinates is None or area_ids is None or area_names is None:
        if not (coordinates is None and area_ids is None and area_names is None):
            raise ValueError("coordinates, area_ids, and area_names must be supplied together.")
        coordinates, area_ids, area_starts, area_names = _load_human_vertices(project_root)
    else:
        coordinates, area_ids, area_starts, area_names = _validate_input_arrays(
            coordinates, area_ids, area_names
        )
    counts, edges, max_distance = _get_area_pair_bin_counts(coordinates, area_starts, float(bin_width))
    global_counts = np.sum(counts, axis=(1, 2), dtype=np.int64)
    expected_pair_count = len(coordinates) * (len(coordinates) - 1)
    if int(np.sum(global_counts)) != expected_pair_count:
        raise ValueError("Binned counts do not cover every ordered non-self pair exactly once.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "coordinates.npy", coordinates, allow_pickle=False)
    np.save(cache_dir / "area_ids.npy", area_ids, allow_pickle=False)
    np.save(cache_dir / "area_starts.npy", area_starts, allow_pickle=False)
    np.save(cache_dir / "bin_edges.npy", edges, allow_pickle=False)
    np.save(cache_dir / "global_bin_counts.npy", global_counts, allow_pickle=False)
    np.save(cache_dir / "area_pair_bin_counts.npy", counts, allow_pickle=False)
    metadata = {
        "animal": "human",
        "dataset_key": "lariviere",
        "bin_width_mm": float(bin_width),
        "bin_count": len(global_counts),
        "occupied_bin_count": int(np.count_nonzero(global_counts)),
        "vertex_count": len(coordinates),
        "ordered_nonself_pair_count": int(expected_pair_count),
        "maximum_distance_mm": max_distance,
        "area_count": len(area_names),
        "area_names": area_names,
    }
    _save_json(metadata, cache_dir / "metadata.json")
    return load_human_bin_catalog(cache_dir)


def load_human_bin_catalog(cache_dir, mmap_mode="r"):
    """Load a binned-distance catalog using memory-mapped arrays."""
    cache_dir = Path(cache_dir)
    names = [
        "coordinates",
        "area_ids",
        "area_starts",
        "bin_edges",
        "global_bin_counts",
        "area_pair_bin_counts",
    ]
    catalog = {name: np.load(cache_dir / f"{name}.npy", mmap_mode=mmap_mode) for name in names}
    catalog["metadata"] = _load_json(cache_dir / "metadata.json")
    return catalog


def get_human_bin_probabilities(catalog, Lambda):
    """Return truncated exponential probability mass for every distance bin."""
    if not np.isfinite(Lambda) or Lambda <= 0:
        raise ValueError("Lambda must be a positive finite value.")
    edges = np.asarray(catalog["bin_edges"])
    counts = np.asarray(catalog["global_bin_counts"])
    probabilities = np.exp(-edges[:-1] / Lambda) - np.exp(-edges[1:] / Lambda)
    probabilities[counts == 0] = 0.0
    probability_sum = float(np.sum(probabilities))
    if not np.isfinite(probability_sum) or probability_sum <= 0:
        raise ValueError("Invalid total bin probability.")
    probabilities /= probability_sum
    if not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Bin probabilities do not sum to one.")
    return probabilities


def get_human_pair_bin_weights(catalog, Lambda):
    """Return per-pair probability for each distance bin."""
    probabilities = get_human_bin_probabilities(catalog, Lambda)
    counts = np.asarray(catalog["global_bin_counts"])
    weights = np.zeros_like(probabilities)
    occupied = counts > 0
    weights[occupied] = probabilities[occupied] / counts[occupied]
    return weights


def get_human_area_probability_matrix(catalog, Lambda):
    """Aggregate corrected pair probabilities into an area-pair matrix."""
    pair_weights = get_human_pair_bin_weights(catalog, Lambda)
    counts = np.asarray(catalog["area_pair_bin_counts"], dtype=np.float64)
    probabilities = np.tensordot(pair_weights, counts, axes=(0, 0))
    probability_sum = float(np.sum(probabilities))
    if not np.isclose(probability_sum, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Area probabilities sum to {probability_sum}, not one.")
    return probabilities


def save_human_area_probability_matrix(cache_dir, Lambda, output_path):
    """Compute and save one deterministic area-probability matrix."""
    output_path = Path(output_path)
    if output_path.exists():
        return str(output_path)
    catalog = load_human_bin_catalog(cache_dir)
    probabilities = get_human_area_probability_matrix(catalog, Lambda)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, probabilities, allow_pickle=False)
    return str(output_path)


def get_human_area_connection_matrix(catalog, area_probabilities, connection_count, seed):
    """Sample a directed area connection-count matrix with a local seed."""
    if int(connection_count) != connection_count or connection_count <= 0:
        raise ValueError("connection_count must be a positive integer.")
    probabilities = np.asarray(area_probabilities)
    if probabilities.shape != (catalog["metadata"]["area_count"],) * 2:
        raise ValueError("area_probabilities has the wrong shape.")
    rng = cf.get_local_rng(seed=seed)
    counts = rng.multinomial(int(connection_count), probabilities.ravel())
    area_count = catalog["metadata"]["area_count"]
    names = catalog["metadata"]["area_names"]
    return pd.DataFrame(counts.reshape(area_count, area_count), index=names, columns=names)


def get_human_vertex_samples(catalog, area_connection_matrix, Lambda, seed):
    """Sample raw vertex pairs conditional on an area connection matrix.

    Returns uint32 source and target indices. Randomness is local and determined
    only by seed. Aggregating the returned pairs exactly reproduces the supplied
    area connection-count matrix.
    """
    rng = cf.get_local_rng(seed=seed)
    coordinates = np.asarray(catalog["coordinates"])
    bin_width = catalog["metadata"]["bin_width_mm"]
    bin_count = catalog["metadata"]["bin_count"]
    pair_bin_weights = get_human_pair_bin_weights(catalog, Lambda)
    starts, stops = _get_area_bounds(catalog["area_starts"], len(coordinates))
    total = int(np.sum(area_connection_matrix.values))
    source_output = np.empty(total, dtype=np.uint32)
    target_output = np.empty(total, dtype=np.uint32)
    offset = 0
    for source_area in range(len(starts)):
        source_indices = np.arange(starts[source_area], stops[source_area], dtype=np.uint32)
        for target_area in range(len(starts)):
            sample_count = int(area_connection_matrix.iat[source_area, target_area])
            if sample_count == 0:
                continue
            target_indices = np.arange(starts[target_area], stops[target_area], dtype=np.uint32)
            distances = scipy.spatial.distance.cdist(coordinates[source_indices], coordinates[target_indices])
            bin_indices = _get_bin_index(distances, bin_width, bin_count)
            weights = pair_bin_weights[bin_indices]
            if source_area == target_area:
                np.fill_diagonal(weights, 0.0)
            flat_weights = weights.ravel()
            weight_sum = float(np.sum(flat_weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError(f"Invalid conditional weights for area pair {source_area}, {target_area}")
            selected = rng.choice(len(flat_weights), size=sample_count, p=flat_weights / weight_sum)
            local_source, local_target = np.unravel_index(selected, weights.shape)
            source_output[offset:offset + sample_count] = source_indices[local_source]
            target_output[offset:offset + sample_count] = target_indices[local_target]
            offset += sample_count
    if offset != total:
        raise ValueError("Raw vertex sample count does not match the area count matrix.")
    return source_output, target_output


def run_human_connection_task(task, cache_dir, output_dir, save_pairs=False, force_run=False):
    """Run and save one deterministic corrected human EDR task."""
    run_dir = Path(output_dir) / task["task_id"]
    metrics_path = run_dir / "metrics.json"
    pairs_exist = (run_dir / "source_vertex_indices.npy").exists()
    if metrics_path.exists() and not force_run and (not save_pairs or pairs_exist):
        result = _load_json(metrics_path)
        result["run_dir"] = str(run_dir)
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_human_bin_catalog(cache_dir)
    area_probabilities = np.load(task["probability_path"], mmap_mode="r")
    connection_matrix = get_human_area_connection_matrix(catalog, area_probabilities, task["connection_count"], task["seed"])
    connection_matrix.to_csv(run_dir / "connection_counts.csv")
    connectivity = uf.get_SC_model_dict_from_CCMt_model(connection_matrix, "human", "lariviere", str(run_dir))
    info = uf.AnimalInfo("human", "lariviere")
    correlation = uf.data_model_correlation(
        SC_data=info.SC_data,
        SC_model=connectivity["CCMe_sym_normalized"],
        animal="human",
        dataset_key="lariviere",
        diagonal=False,
    )
    if save_pairs:
        pair_seed = int(task["seed"]) + 1_000_003
        sources, targets = get_human_vertex_samples(catalog, connection_matrix, task["Lambda"], pair_seed)
        np.save(run_dir / "source_vertex_indices.npy", sources, allow_pickle=False)
        np.save(run_dir / "target_vertex_indices.npy", targets, allow_pickle=False)
    config = {
        "animal": "human",
        "dataset_key": "lariviere",
        "bin_width_mm": catalog["metadata"]["bin_width_mm"],
        "Lambda": float(task["Lambda"]),
        "seed": int(task["seed"]),
        "connection_count": int(task["connection_count"]),
        "task_id": task["task_id"],
        "save_pairs": bool(save_pairs),
    }
    metrics = {**config, "correlation": float(correlation)}
    _save_json(config, run_dir / "config.json")
    _save_json(metrics, metrics_path)
    metrics["run_dir"] = str(run_dir)
    return metrics


def get_distance_histograms(catalog, source_indices, target_indices, Lambda, chunk_size=1_000_000):
    """Return native-bin target and sampled distance probability vectors."""
    expected = get_human_bin_probabilities(catalog, Lambda)
    empirical_counts = np.zeros(len(expected), dtype=np.int64)
    coordinates = np.asarray(catalog["coordinates"])
    bin_width = catalog["metadata"]["bin_width_mm"]
    for start in range(0, len(source_indices), chunk_size):
        stop = min(start + chunk_size, len(source_indices))
        distances = np.linalg.norm(
            coordinates[source_indices[start:stop]] - coordinates[target_indices[start:stop]], axis=1
        )
        indices = _get_bin_index(distances, bin_width, len(expected))
        empirical_counts += np.bincount(indices, minlength=len(expected))
    empirical = empirical_counts / np.sum(empirical_counts)
    edges = np.asarray(catalog["bin_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, expected, empirical


# ============================================================
# Mouse common-region model
# ============================================================

def get_common_region_flne(connection_counts, source_names, target_names):
    """Normalize Gamanut-Allen common-region counts by target input totals."""
    common_region_counts = connection_counts.loc[source_names, target_names].astype(float)
    column_sums = common_region_counts.sum(axis=0)
    if np.any(column_sums <= 0):
        missing = column_sums.index[column_sums <= 0].tolist()
        raise ValueError(f"No generated input connections for targets: {missing}")
    return common_region_counts.div(column_sums, axis=1)


def get_empirical_common_regions(source_names, target_names):
    """Return empirical FLNe conditioned on Gamanut-Allen common regions."""
    measured = uf.AnimalInfo("mouse", "gamanut").SC_data.loc[source_names, target_names].astype(float)
    column_sums = measured.sum(axis=0)
    if np.any(column_sums <= 0):
        missing = column_sums.index[column_sums <= 0].tolist()
        raise ValueError(f"No empirical input connections for targets: {missing}")
    return measured.div(column_sums, axis=1)


def get_common_region_correlation(measured, generated, diagonal=False):
    """Return Pearson correlation across common source-target region pairs."""
    if measured.index.tolist() != generated.index.tolist() or measured.columns.tolist() != generated.columns.tolist():
        raise ValueError("Measured and generated common-region matrices are not aligned.")
    mask = np.ones(measured.shape, dtype=bool)
    if not diagonal:
        for row, source in enumerate(measured.index):
            for column, target in enumerate(measured.columns):
                if source == target:
                    mask[row, column] = False
    return float(cf.get_corr(measured.values[mask], generated.values[mask]))


def get_2d_common_region_correlation(project_root, source_names, target_names):
    """Return accepted 2D correlation on identical Gamanut-Allen regions."""
    path = Path(project_root) / "results" / "models" / "connection_generation" / "mouse_gamanut" / "89849cf471ed" / "connectivity" / "FLNe.csv"
    model_2d = pd.read_csv(path, index_col=0)
    model_2d = model_2d.loc[source_names, target_names]
    model_2d = model_2d.div(model_2d.sum(axis=0), axis=1)
    measured = get_empirical_common_regions(source_names, target_names)
    return get_common_region_correlation(measured, model_2d, diagonal=False)


def run_mouse_connection_task(task, cache_dir, output_dir, source_names, target_names, save_pairs=False, force_run=False):
    """Generate one deterministic common-region sample and save its metrics."""
    run_dir = Path(output_dir) / task["task_id"]
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force_run and (not save_pairs or (run_dir / "source_voxel_indices.npy").exists()):
        result = _load_json(metrics_path)
        result["run_dir"] = str(run_dir)
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_displacement_catalog(cache_dir)
    sources, targets = get_connection_samples(catalog, task["Lambda"], task["connection_count"], task["seed"])
    counts = get_connection_matrix(catalog, sources, targets)
    flne = get_common_region_flne(counts, source_names, target_names)
    measured = get_empirical_common_regions(source_names, target_names)
    correlation = get_common_region_correlation(measured, flne, diagonal=False)
    counts.to_csv(run_dir / "connection_counts.csv")
    flne.to_csv(run_dir / "gamanut_allen_common_region_FLNe.csv")
    if save_pairs:
        np.save(run_dir / "source_voxel_indices.npy", sources, allow_pickle=False)
        np.save(run_dir / "target_voxel_indices.npy", targets, allow_pickle=False)
    metrics = {**task, "correlation": correlation}
    _save_json(task, run_dir / "config.json")
    _save_json(metrics, metrics_path)
    metrics["run_dir"] = str(run_dir)
    return metrics


# ============================================================
# Cross-species model workflow
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "cross_species_3d"
PREPROCESSING_DIR = OUTPUT_DIR / "preprocessing"
MODELS_DIR = OUTPUT_DIR / "models"
WORKER_COUNT = 10
COARSE_CONNECTION_COUNT = 1_000_000
FINAL_CONNECTION_COUNT = 8_000_000
COARSE_SEEDS = range(3)
FINAL_SEEDS = range(5)
COARSE_POINT_COUNT = 20
FINE_POINT_COUNT = 11
DATASETS = [
    ("mouse", "gamanut"),
    ("marmoset", "majka"),
    ("macaque", "markov40"),
    ("human", "lariviere"),
]


def get_task_table(dataset_name, lambda_values, seeds, connection_count, stage):
    """Return deterministic tasks whose seeds do not depend on worker order."""
    animal, dataset_key = dataset_name.split("_", 1)
    tasks = []
    for lambda_index, Lambda in enumerate(lambda_values):
        for seed in seeds:
            tasks.append({
                "task_id": f"{stage}_lambda_{lambda_index:03d}_seed_{seed:02d}",
                "dataset_name": dataset_name,
                "animal": animal,
                "dataset_key": dataset_key,
                "Lambda": float(Lambda),
                "seed": int(seed),
                "connection_count": int(connection_count),
            })
    return tasks


def get_best_lambda(summary):
    """Select the highest mean-correlation lambda, breaking ties low."""
    grouped = summary.groupby("Lambda")["correlation"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values(["mean", "Lambda"], ascending=[False, True]).reset_index(drop=True)
    return float(grouped.loc[0, "Lambda"]), grouped


def get_coarse_lambdas(animal, dataset_key):
    """Return the established coarse scan for one species."""
    if animal == "mouse":
        return np.geomspace(0.1, 6.0, COARSE_POINT_COUNT)
    info = uf.AnimalInfo(animal, dataset_key)
    return np.geomspace(0.25, float(np.max(info.distance_3D.values) / 2), COARSE_POINT_COUNT)


def get_fine_lambdas(coarse_lambdas, best_lambda):
    """Return an eleven-point linear refinement around a coarse optimum."""
    values = np.unique(np.sort(np.asarray(coarse_lambdas, dtype=np.float64)))
    index = int(np.argmin(np.abs(values - best_lambda)))
    lower = values[max(0, index - 1)]
    upper = values[min(len(values) - 1, index + 1)]
    if lower == upper:
        raise ValueError("A fine scan requires at least two distinct coarse lambda values.")
    return np.linspace(lower, upper, FINE_POINT_COUNT)


def _run_3d_task(task, cache_dir, stage_dir, source_names=None, target_names=None):
    dataset_name = task["dataset_name"]
    if dataset_name == "human_lariviere":
        return run_human_connection_task(task, cache_dir, stage_dir)
    if dataset_name == "mouse_gamanut":
        return run_mouse_connection_task(task, cache_dir, stage_dir, source_names, target_names)
    return run_connection_task(task, cache_dir, stage_dir)


def prepare_human_probabilities(tasks, cache_dir, stage_dir):
    """Create one corrected area-probability matrix per lambda."""
    paths = {}
    for task in tasks:
        Lambda = task["Lambda"]
        if Lambda not in paths:
            path = stage_dir / "probabilities" / f"lambda_{len(paths):03d}.npy"
            paths[Lambda] = save_human_area_probability_matrix(cache_dir, Lambda, path)
        task["probability_path"] = paths[Lambda]


def run_3d_stage(dataset_name, tasks, cache_dir, stage_dir, source_names=None, target_names=None):
    """Run, sort, and save one cross-species 3D task stage."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    if dataset_name == "human_lariviere":
        prepare_human_probabilities(tasks, cache_dir, stage_dir)
    kwargs_list = [{
        "task": task,
        "cache_dir": cache_dir,
        "stage_dir": stage_dir,
        "source_names": source_names,
        "target_names": target_names,
    } for task in tasks]
    results = cf.multi_process(
        process_num=min(WORKER_COUNT, len(tasks)), func=_run_3d_task,
        kwargs_list=kwargs_list, func_name=f"{dataset_name} 3D {stage_dir.name}",
    )
    summary = pd.DataFrame(sorted(results, key=lambda item: item["task_id"]))
    summary.to_csv(stage_dir / "summary.csv", index=False)
    return summary


def run_3d_dataset(animal, dataset_key):
    """Scan and run the optimal 3D model for one dataset."""
    dataset_name = f"{animal}_{dataset_key}"
    dataset_dir = MODELS_DIR / "3d" / dataset_name
    cache_dir = PREPROCESSING_DIR / dataset_name / "distance_catalog"
    source_names = None
    target_names = None
    if dataset_name == "mouse_gamanut":
        common_dir = PREPROCESSING_DIR / dataset_name / "common_regions"
        with (common_dir / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        source_names = metadata["source_names"]
        target_names = metadata["target_names"]
    coarse_lambdas = get_coarse_lambdas(animal, dataset_key)
    coarse_tasks = get_task_table(dataset_name, coarse_lambdas, COARSE_SEEDS, COARSE_CONNECTION_COUNT, "coarse")
    coarse = run_3d_stage(dataset_name, coarse_tasks, cache_dir, dataset_dir / "coarse", source_names, target_names)
    coarse_best, _ = get_best_lambda(coarse)
    fine_lambdas = get_fine_lambdas(coarse_lambdas, coarse_best)
    fine_tasks = get_task_table(dataset_name, fine_lambdas, FINAL_SEEDS, FINAL_CONNECTION_COUNT, "fine")
    fine = run_3d_stage(dataset_name, fine_tasks, cache_dir, dataset_dir / "fine", source_names, target_names)
    best_lambda, grouped = get_best_lambda(fine)
    grouped.to_csv(dataset_dir / "lambda_summary.csv", index=False)
    best_tasks = get_task_table(dataset_name, [best_lambda], FINAL_SEEDS, FINAL_CONNECTION_COUNT, "best")
    best = run_3d_stage(dataset_name, best_tasks, cache_dir, dataset_dir / "best", source_names, target_names)
    best["animal"] = animal
    best["dataset_key"] = dataset_key
    best["dimension"] = "3D"
    selected = {
        "animal": animal,
        "dataset_key": dataset_key,
        "best_lambda": best_lambda,
        "mean_correlation": float(best["correlation"].mean()),
        "std_correlation": float(best["correlation"].std()),
        "seed_correlations": {str(int(row.seed)): float(row.correlation) for row in best.itertuples()},
    }
    _save_json(selected, dataset_dir / "optimal_parameters.json")
    return best


def _run_2d_task(task):
    model = uf.get_connection_generation_results(task["config"], MODELS_DIR / "2d_runs")
    animal = task["config"]["animal"]
    dataset_key = task["config"]["dataset_key"]
    if animal == "mouse":
        common_dir = PREPROCESSING_DIR / "mouse_gamanut" / "common_regions"
        with (common_dir / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        generated = model.get_suitable_SC_model().loc[metadata["source_names"], metadata["target_names"]]
        generated = generated.div(generated.sum(axis=0), axis=1)
        measured = get_empirical_common_regions(metadata["source_names"], metadata["target_names"])
        correlation = get_common_region_correlation(measured, generated, diagonal=False)
    else:
        correlation = model.get_data_model_correlation()
    return {
        "task_id": task["task_id"],
        "animal": animal,
        "dataset_key": dataset_key,
        "dimension": "2D",
        "Lambda": task["config"]["length_params"]["Lambda"],
        "seed": task["config"]["the_seed"],
        "correlation": float(correlation),
        "run_dir": str(model.run_dir),
    }


def run_2d_models():
    """Run five seeds at each established optimal 2D lambda."""
    tasks = []
    for animal, dataset_key in DATASETS:
        dataset_name = f"{animal}_{dataset_key}"
        for seed in FINAL_SEEDS:
            tasks.append({
                "task_id": f"{dataset_name}_seed_{seed:02d}",
                "config": {
                    "animal": animal,
                    "dataset_key": dataset_key,
                    "animal_and_dataset_key": dataset_name,
                    "direction_method": "random",
                    "length_method": "exponential",
                    "direction_params": {},
                    "length_params": {"Lambda": uf.OPTIMAL_PARAMS[dataset_name]["Lambda"]},
                    "na": FINAL_CONNECTION_COUNT,
                    "process_num": 1,
                    "worker_count": 1,
                    "the_seed": int(seed),
                },
            })
    results = cf.multi_process(
        process_num=WORKER_COUNT, func=_run_2d_task,
        kwargs_list=[{"task": task} for task in tasks], func_name="cross-species optimal 2D",
    )
    summary = pd.DataFrame(sorted(results, key=lambda item: item["task_id"]))
    summary.to_csv(MODELS_DIR / "2d_summary.csv", index=False)
    return summary


def main():
    """Run optimal 2D repeats and all four 3D parameter searches."""
    missing = [
        PREPROCESSING_DIR / f"{animal}_{dataset_key}" / "distance_catalog" / "metadata.json"
        for animal, dataset_key in DATASETS
    ]
    missing = [path for path in missing if not path.exists()]
    if missing:
        raise FileNotFoundError("Run preprocess.py first. Missing:\n" + "\n".join(str(path) for path in missing))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary_2d = run_2d_models()
    summaries = [summary_2d]
    for animal, dataset_key in DATASETS:
        best = run_3d_dataset(animal, dataset_key).copy()
        summaries.append(best)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(MODELS_DIR / "optimal_seed_correlations.csv", index=False)


if __name__ == "__main__":
    main()
