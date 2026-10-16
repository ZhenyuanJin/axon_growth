import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def build_bin_catalog(project_root, cache_dir, bin_width=0.1, coordinates=None, area_ids=None, area_names=None):
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
    return load_bin_catalog(cache_dir)


def load_bin_catalog(cache_dir, mmap_mode="r"):
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


def get_bin_probabilities(catalog, Lambda):
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


def get_pair_bin_weights(catalog, Lambda):
    """Return per-pair probability for each distance bin."""
    probabilities = get_bin_probabilities(catalog, Lambda)
    counts = np.asarray(catalog["global_bin_counts"])
    weights = np.zeros_like(probabilities)
    occupied = counts > 0
    weights[occupied] = probabilities[occupied] / counts[occupied]
    return weights


def get_area_probability_matrix(catalog, Lambda):
    """Aggregate corrected pair probabilities into an area-pair matrix."""
    pair_weights = get_pair_bin_weights(catalog, Lambda)
    counts = np.asarray(catalog["area_pair_bin_counts"], dtype=np.float64)
    probabilities = np.tensordot(pair_weights, counts, axes=(0, 0))
    probability_sum = float(np.sum(probabilities))
    if not np.isclose(probability_sum, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Area probabilities sum to {probability_sum}, not one.")
    return probabilities


def save_area_probability_matrix(cache_dir, Lambda, output_path):
    """Compute and save one deterministic area-probability matrix."""
    output_path = Path(output_path)
    if output_path.exists():
        return str(output_path)
    catalog = load_bin_catalog(cache_dir)
    probabilities = get_area_probability_matrix(catalog, Lambda)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, probabilities, allow_pickle=False)
    return str(output_path)


def get_area_connection_matrix(catalog, area_probabilities, connection_count, seed):
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


def get_vertex_samples(catalog, area_connection_matrix, Lambda, seed):
    """Sample raw vertex pairs conditional on an area connection matrix.

    Returns uint32 source and target indices. Randomness is local and determined
    only by seed. Aggregating the returned pairs exactly reproduces the supplied
    area connection-count matrix.
    """
    rng = cf.get_local_rng(seed=seed)
    coordinates = np.asarray(catalog["coordinates"])
    bin_width = catalog["metadata"]["bin_width_mm"]
    bin_count = catalog["metadata"]["bin_count"]
    pair_bin_weights = get_pair_bin_weights(catalog, Lambda)
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


def run_connection_task(task, cache_dir, output_dir, save_pairs=False, force_run=False):
    """Run and save one deterministic corrected human EDR task."""
    run_dir = Path(output_dir) / task["task_id"]
    metrics_path = run_dir / "metrics.json"
    pairs_exist = (run_dir / "source_vertex_indices.npy").exists()
    if metrics_path.exists() and not force_run and (not save_pairs or pairs_exist):
        result = _load_json(metrics_path)
        result["run_dir"] = str(run_dir)
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_bin_catalog(cache_dir)
    area_probabilities = np.load(task["probability_path"], mmap_mode="r")
    connection_matrix = get_area_connection_matrix(catalog, area_probabilities, task["connection_count"], task["seed"])
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
        sources, targets = get_vertex_samples(catalog, connection_matrix, task["Lambda"], pair_seed)
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
    expected = get_bin_probabilities(catalog, Lambda)
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
