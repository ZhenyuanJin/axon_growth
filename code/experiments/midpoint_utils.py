import numpy as np


def get_largest_split_name(area_name, animal_info):
    """Return the largest 2D component name for one area."""
    split_names = animal_info.name_to_split_name_dict[area_name]
    return max(split_names, key=lambda name: animal_info.polygon_dict[name].area)


def get_area_midpoint(area_name, animal_info):
    """Return the float64 centroid of an area's largest 2D component, shape (2,)."""
    split_name = get_largest_split_name(area_name, animal_info)
    centroid = animal_info.polygon_dict[split_name].centroid
    midpoint = np.array([centroid.x, centroid.y], dtype=float)
    if not np.all(np.isfinite(midpoint)):
        raise ValueError(f"Invalid midpoint for area {area_name}.")
    return midpoint


def get_area_size(area_name, animal_info):
    """Return the summed flat-map area across all components of one area."""
    return sum(animal_info.polygon_dict[name].area for name in animal_info.name_to_split_name_dict[area_name])
