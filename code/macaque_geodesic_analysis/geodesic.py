import itertools

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial.distance
from utils_function import cf


def _get_positive_offsets():
    offsets = []
    for offset in itertools.product([-1, 0, 1], repeat=3):
        if offset == (0, 0, 0):
            continue
        first_nonzero = next(value for value in offset if value != 0)
        if first_nonzero > 0:
            offsets.append(np.asarray(offset, dtype=np.int16))
    return offsets


def _get_required_offsets(offset):
    axes = np.flatnonzero(offset)
    if len(axes) == 1:
        return []
    required = []
    for subset_size in range(1, len(axes)):
        for subset in itertools.combinations(axes, subset_size):
            intermediate = np.zeros(3, dtype=np.int16)
            intermediate[list(subset)] = offset[list(subset)]
            required.append(intermediate)
    return required


def get_no_corner_cutting_graph(grid_indices, voxel_map, spacing):
    """Return a weighted 26-neighbor graph that cannot cross empty corners.

    Face moves require an occupied endpoint. Edge-diagonal moves additionally
    require both intermediate face voxels. Corner-diagonal moves require all
    six intermediate voxels in the local 2x2x2 block. Edge weights are the
    physical Euclidean lengths implied by spacing. No randomness is used.
    """
    grid_indices = np.asarray(grid_indices, dtype=np.int32)
    voxel_map = np.asarray(voxel_map)
    spacing = np.asarray(spacing, dtype=np.float64)
    if grid_indices.ndim != 2 or grid_indices.shape[1] != 3:
        raise ValueError("grid_indices must have shape (voxel_count, 3).")
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing must contain three positive values.")
    shape = np.asarray(voxel_map.shape, dtype=np.int32)
    rows = []
    columns = []
    weights = []
    source_ids = np.arange(len(grid_indices), dtype=np.int32)
    for offset in _get_positive_offsets():
        target_grid = grid_indices + offset
        valid = np.all((target_grid >= 0) & (target_grid < shape), axis=1)
        candidate_ids = source_ids[valid]
        candidate_targets = target_grid[valid]
        target_ids = voxel_map[tuple(candidate_targets.T)]
        occupied = target_ids >= 0
        candidate_ids = candidate_ids[occupied]
        target_ids = target_ids[occupied]
        for intermediate_offset in _get_required_offsets(offset):
            intermediate_grid = grid_indices[candidate_ids] + intermediate_offset
            intermediate_ids = voxel_map[tuple(intermediate_grid.T)]
            occupied = intermediate_ids >= 0
            candidate_ids = candidate_ids[occupied]
            target_ids = target_ids[occupied]
        if len(candidate_ids) == 0:
            continue
        rows.append(candidate_ids)
        columns.append(target_ids.astype(np.int32, copy=False))
        weights.append(np.full(len(candidate_ids), np.linalg.norm(offset * spacing), dtype=np.float64))
    if not rows:
        return scipy.sparse.csr_matrix((len(grid_indices), len(grid_indices)), dtype=np.float64)
    rows = np.concatenate(rows)
    columns = np.concatenate(columns)
    weights = np.concatenate(weights)
    graph = scipy.sparse.coo_matrix(
        (np.r_[weights, weights], (np.r_[rows, columns], np.r_[columns, rows])),
        shape=(len(grid_indices), len(grid_indices)),
    ).tocsr()
    graph.sort_indices()
    return graph


def get_largest_component_mask(graph):
    """Return the node mask for the largest undirected graph component."""
    component_count, labels = scipy.sparse.csgraph.connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=component_count)
    return labels == int(np.argmax(sizes)), sizes


def get_sampled_source_indices(area_ids, sampled_area_ids, valid_mask, samples_per_area, seed):
    """Return deterministic sampled graph nodes for each requested area.

    Randomness is local and fully determined by seed. Sampling is without
    replacement and each requested area contributes samples_per_area nodes.
    """
    if int(samples_per_area) != samples_per_area or samples_per_area <= 0:
        raise ValueError("samples_per_area must be a positive integer.")
    rng = cf.get_local_rng(seed=seed)
    sampled_indices = []
    sampled_labels = []
    for area_id in sampled_area_ids:
        candidates = np.flatnonzero((area_ids == area_id) & valid_mask)
        if len(candidates) < samples_per_area:
            raise ValueError(f"Area {area_id} has fewer valid voxels than requested samples.")
        selected = rng.choice(candidates, size=int(samples_per_area), replace=False)
        sampled_indices.append(selected)
        sampled_labels.append(np.full(int(samples_per_area), area_id, dtype=np.int16))
    return np.concatenate(sampled_indices), np.concatenate(sampled_labels)


def get_sampled_area_distance_matrices(graph, coordinates, area_ids, sampled_area_ids, source_indices, source_area_ids, batch_size):
    """Estimate area-pair geodesic and Euclidean means from sampled sources.

    Each sampled source is compared with every valid target voxel. The two
    returned arrays have shape (area_count, sampled_area_count), float64 dtype,
    and use identical source-target samples. No randomness occurs here.
    """
    if int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    area_ids = np.asarray(area_ids)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    source_area_ids = np.asarray(source_area_ids)
    largest_mask, component_sizes = get_largest_component_mask(graph)
    valid_indices = np.flatnonzero(largest_mask)
    valid_area_ids = area_ids[valid_indices]
    valid_coordinates = coordinates[valid_indices]
    area_count = int(np.max(area_ids) + 1)
    area_voxel_counts = np.bincount(valid_area_ids, minlength=area_count)
    if np.any(area_voxel_counts == 0):
        raise ValueError("At least one area has no voxel in the largest graph component.")
    sampled_area_ids = np.asarray(sampled_area_ids, dtype=np.int64)
    sampled_lookup = {int(area_id): index for index, area_id in enumerate(sampled_area_ids)}
    geodesic_sum = np.zeros((area_count, len(sampled_area_ids)), dtype=np.float64)
    euclidean_sum = np.zeros_like(geodesic_sum)
    sample_counts = np.zeros(len(sampled_area_ids), dtype=np.int64)
    for start in range(0, len(source_indices), int(batch_size)):
        stop = min(start + int(batch_size), len(source_indices))
        local_sources = source_indices[start:stop]
        geodesic = scipy.sparse.csgraph.dijkstra(
            graph, directed=False, indices=local_sources, return_predecessors=False
        )[:, valid_indices]
        if not np.all(np.isfinite(geodesic)):
            raise ValueError("Non-finite distance occurred inside the largest component.")
        euclidean = scipy.spatial.distance.cdist(coordinates[local_sources], valid_coordinates)
        for local_index, source_area_id in enumerate(source_area_ids[start:stop]):
            sampled_index = sampled_lookup[int(source_area_id)]
            geodesic_sum[:, sampled_index] += np.bincount(
                valid_area_ids, weights=geodesic[local_index], minlength=area_count
            ) / area_voxel_counts
            euclidean_sum[:, sampled_index] += np.bincount(
                valid_area_ids, weights=euclidean[local_index], minlength=area_count
            ) / area_voxel_counts
            sample_counts[sampled_index] += 1
    if np.any(sample_counts == 0):
        raise ValueError("At least one requested source area received no samples.")
    geodesic_mean = geodesic_sum / sample_counts[np.newaxis, :]
    euclidean_mean = euclidean_sum / sample_counts[np.newaxis, :]
    return geodesic_mean, euclidean_mean, component_sizes
