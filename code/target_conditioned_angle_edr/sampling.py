import hashlib

import numpy as np
import shapely
from shapely.strtree import STRtree

from utils_function import cf


def get_stage_seed(seed, *identity):
    """Return a task-specific uint32 seed compatible with the local RNG helper."""
    text = "|".join([str(seed)] + [str(item) for item in identity])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def get_points_in_polygon(polygon, point_count, seed):
    """Return uniform float64 points, shape (point_count, 2), using an explicit seed."""
    if point_count < 0 or polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Invalid polygon or point count.")
    rng = cf.get_local_rng(seed)
    min_x, min_y, max_x, max_y = polygon.bounds
    acceptance = polygon.area / ((max_x - min_x) * (max_y - min_y))
    points = np.empty((point_count, 2), dtype=float)
    filled = 0
    while filled < point_count:
        batch_size = max(64, int(np.ceil((point_count - filled) / acceptance * 1.1)))
        candidates = rng.uniform([min_x, min_y], [max_x, max_y], (batch_size, 2))
        inside = shapely.contains_xy(polygon, candidates[:, 0], candidates[:, 1])
        accepted = candidates[inside][:point_count - filled]
        points[filled:filled + len(accepted)] = accepted
        filled += len(accepted)
    return points


def get_area_components(animal_info, polygon_dict):
    """Return deterministic float64 centroids (n, 2) and area fractions (n,) per area."""
    components = {}
    for name in animal_info.full_name:
        polygons = [polygon_dict[split] for split in animal_info.name_to_split_name_dict[name]]
        areas = np.array([polygon.area for polygon in polygons], dtype=float)
        centers = np.array([[polygon.centroid.x, polygon.centroid.y] for polygon in polygons])
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(areas)) or np.any(areas <= 0):
            raise ValueError(f"Invalid components for area {name}.")
        fractions = np.ones(1) if len(polygons) == 1 else areas / areas.sum()
        components[name] = (centers, fractions)
    return components


def get_angle_centers_and_weights(target_name, animal_info, SC_data, area_components):
    """Return float64 target-to-source angles and normalized component-pair weights (n,)."""
    target_centers, target_fractions = area_components[target_name]
    angles, weights = [], []
    for source_name in animal_info.source_name:
        weight = float(SC_data.loc[source_name, target_name])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid angle weight: {source_name}, {target_name}.")
        if source_name == target_name or weight == 0:
            continue
        source_centers, source_fractions = area_components[source_name]
        displacement = source_centers[:, None, :] - target_centers[None, :, :]
        if np.any(np.linalg.norm(displacement, axis=-1) == 0):
            raise ValueError(f"Coincident component centers for {source_name}, {target_name}.")
        angles.extend(np.arctan2(displacement[..., 1], displacement[..., 0]).ravel())
        weights.extend((weight * source_fractions[:, None] * target_fractions[None, :]).ravel())
    angles, weights = np.asarray(angles), np.asarray(weights)
    if not np.all(np.isfinite(angles)) or not len(weights) or weights.sum() <= 0:
        raise ValueError(f"Invalid angle weights for target {target_name}.")
    return angles, weights / weights.sum()


def get_target_points(target_name, point_count, animal_info, polygon_dict, seed):
    """Return seeded uniform target points (n, 2) and int32 split IDs (n,)."""
    split_names = animal_info.name_to_split_name_dict[target_name]
    split_areas = np.array([polygon_dict[name].area for name in split_names], dtype=float)
    rng = cf.get_local_rng(seed)
    selected = rng.choice(len(split_names), size=point_count, p=split_areas / split_areas.sum())
    points = np.empty((point_count, 2), dtype=float)
    split_ids = np.empty(point_count, dtype=np.int32)
    for index, split_name in enumerate(split_names):
        mask = selected == index
        points[mask] = get_points_in_polygon(
            polygon_dict[split_name], int(mask.sum()), get_stage_seed(seed, "split", split_name),
        )
        split_ids[mask] = animal_info.full_split_name.index(split_name)
    return points, split_ids


def get_context(animal_info):
    """Load deterministic geometry, empirical weights, and split-to-area integer mappings."""
    polygons = animal_info.polygon_dict
    split_names = animal_info.full_split_name
    lookup_names = list(polygons)
    area_components = get_area_components(animal_info, polygons)
    SC_data = animal_info.SC_data.loc[animal_info.source_name, animal_info.target_name]
    values = SC_data.to_numpy()
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Empirical connection weights must be finite and nonnegative.")
    area_indices = {name: index for index, name in enumerate(animal_info.full_name)}
    split_area_ids = np.array([
        area_indices[animal_info.split_name_to_name_dict[name]] for name in split_names
    ])
    whole_cortex = animal_info.whole_cortex_polygon
    shapely.prepare(whole_cortex)
    for polygon in polygons.values():
        shapely.prepare(polygon)
    return {
        "info": animal_info,
        "polygons": polygons,
        "tree": STRtree([polygons[name] for name in lookup_names]),
        "tree_split_ids": np.array([split_names.index(name) for name in lookup_names], dtype=np.int32),
        "whole_cortex": whole_cortex,
        "split_area_ids": split_area_ids,
        "SC_data": SC_data,
        "angles": {
            target: get_angle_centers_and_weights(target, animal_info, SC_data, area_components)
            for target in animal_info.target_name
        },
    }


def get_split_ids(points, context):
    """Return int32 split IDs (n,) using polygon insertion order and whole-cortex membership."""
    geometries = shapely.points(points)
    matches = context["tree"].query(geometries, predicate="intersects")
    split_ids = np.full(len(points), -1, dtype=np.int32)
    if matches.shape[1]:
        point_ids, polygon_ids = matches
        first_match = np.full(len(points), len(context["tree_split_ids"]), dtype=np.int32)
        np.minimum.at(first_match, point_ids, polygon_ids)
        inside = (first_match < len(context["tree_split_ids"])) & shapely.intersects(
            context["whole_cortex"], geometries,
        )
        split_ids[inside] = context["tree_split_ids"][first_match[inside]]
    return split_ids


def get_angles(method, point_count, centers, weights, bandwidth_degrees, seed):
    """Return seeded float64 angles (point_count,) from a uniform or von Mises mixture."""
    rng = cf.get_local_rng(seed)
    if method == "random":
        return rng.uniform(-np.pi, np.pi, point_count)
    if method != "SC_angle_kde" or not 0 < bandwidth_degrees < 180:
        raise ValueError("Invalid direction method or bandwidth.")
    selected = rng.choice(len(centers), size=point_count, p=weights)
    return rng.vonmises(centers[selected], 1.0 / np.radians(bandwidth_degrees) ** 2)


def get_target_samples(context, target_name, target_points, target_split_ids, Lambda, method, bandwidth_degrees, max_iteration, seed):
    """Return seeded source/target coordinates, split IDs, sample IDs, and rejection diagnostics."""
    if Lambda <= 0 or max_iteration < 1 or len(target_points) == 0:
        raise ValueError("Lambda, max_iteration and target sample count must be positive.")
    centers, weights = context["angles"][target_name]
    point_count = len(target_points)
    unresolved = np.arange(point_count)
    source_points = np.full_like(target_points, np.nan)
    source_split_ids = np.full(point_count, -1, dtype=np.int32)
    proposals, self_rejections, outside_rejections = 0, 0, 0
    for iteration in range(max_iteration):
        if not len(unresolved):
            break
        distance_seed = get_stage_seed(seed, target_name, iteration, "distance")
        distances = cf.get_local_rng(distance_seed).exponential(Lambda, point_count)[unresolved]
        angles = get_angles(
            method, len(unresolved), centers, weights, bandwidth_degrees,
            get_stage_seed(seed, target_name, iteration, method, "angle"),
        )
        candidates = target_points[unresolved] + distances[:, None] * np.column_stack([
            np.cos(angles), np.sin(angles),
        ])
        split_ids = get_split_ids(candidates, context)
        inside = split_ids >= 0
        same_area = np.zeros(len(split_ids), dtype=bool)
        same_area[inside] = (
            context["split_area_ids"][split_ids[inside]]
            == context["split_area_ids"][target_split_ids[unresolved[inside]]]
        )
        valid = inside & ~same_area
        accepted_ids = unresolved[valid]
        source_points[accepted_ids] = candidates[valid]
        source_split_ids[accepted_ids] = split_ids[valid]
        proposals += len(unresolved)
        self_rejections += int(same_area.sum())
        outside_rejections += int((~inside).sum())
        unresolved = unresolved[~valid]
    accepted = source_split_ids >= 0
    sample_ids = np.flatnonzero(accepted).astype(np.int32)
    diagnostics = {
        "target": target_name, "requested_count": point_count,
        "accepted_count": int(accepted.sum()), "unresolved_count": len(unresolved),
        "acceptance_rate": float(accepted.mean()), "proposal_count": proposals,
        "self_rejections": self_rejections, "outside_rejections": outside_rejections,
    }
    if not accepted.any():
        raise ValueError(f"No valid source samples for target {target_name}.")
    return {
        "source_points": source_points[accepted], "target_points": target_points[accepted],
        "source_split_ids": source_split_ids[accepted], "target_split_ids": target_split_ids[accepted],
        "sample_ids": sample_ids,
    }, diagnostics
