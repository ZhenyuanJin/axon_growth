import hashlib
import itertools
import json
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.csgraph
import scipy.stats
import SimpleITK as sitk
from cross_species_3d import model
from utils_function import cf
from utils_function import utils_function as uf


ANNOTATION_URL = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd"
STRUCTURE_GRAPH_URL = "https://api.brain-map.org/api/v2/structure_graph_download/1.json"
FINE_SPACING_MM = 0.025
HEMISPHERE_AXIS = 2
HEMISPHERE_SIDE = "first"
MODEL_SPACING_MM = 0.1
ALLEN_TO_MODEL_DOWNSAMPLE_FACTOR = 4

VISUAL_AREA_MAP = {
    "A": ["VISa"],
    "AL": ["VISal"],
    "AM": ["VISam"],
    "LI": ["VISli"],
    "LM": ["VISl"],
    "P": ["VISpl"],
    "PM": ["VISpm"],
    "POR": ["VISpor"],
    "RL": ["VISrl"],
    "V1": ["VISp"],
}
COMPOSITE_AREA_MAP = {"SSp-nm": ["SSp-n", "SSp-m"]}
EXCLUDED_AREAS = {
    "DP": "No Allen CCF2017 homolog for the dataset's dorsal-posterior visual field; Allen DP is olfactory cortex.",
    "LLA": "VISlla is absent from the CCF2017 annotation volume.",
    "MM": "VISm is absent from the CCF2017 annotation volume.",
    "PORa": "No distinct CCF2017 subdivision corresponding to anterior POR.",
    "SSp-lj": "No Allen CCF2017 lower-jaw subdivision.",
    "TEp": "Allen CCF2017 has TEa but no separable posterior TE field.",
}


def _save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def _load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _get_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_allen_files(raw_dir):
    """Download the official CCF2017 annotation and ontology if absent.

    Returns the two local paths. Existing files are preserved. No randomness
    is used.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "annotation": raw_dir / "annotation_25.nrrd",
        "structure_graph": raw_dir / "structure_graph_1.json",
    }
    urls = {"annotation": ANNOTATION_URL, "structure_graph": STRUCTURE_GRAPH_URL}
    for key, path in paths.items():
        if not path.exists():
            urllib.request.urlretrieve(urls[key], path)
    return paths


def load_structure_nodes(structure_graph_path):
    """Load Allen ontology nodes and return dictionaries indexed by id/acronym."""
    with Path(structure_graph_path).open(encoding="utf-8") as file:
        root = json.load(file)["msg"][0]
    by_id = {}
    by_acronym = {}

    def visit(node, path):
        node_copy = {key: value for key, value in node.items() if key != "children"}
        node_copy["path_acronyms"] = path + [node["acronym"]]
        by_id[node["id"]] = node_copy
        by_acronym.setdefault(node["acronym"], []).append(node_copy)
        for child in node.get("children", []):
            visit(child, node_copy["path_acronyms"])

    visit(root, [])
    children_by_id = {node_id: [] for node_id in by_id}
    for node in by_id.values():
        parent_id = node["parent_structure_id"]
        if parent_id in children_by_id:
            children_by_id[parent_id].append(node["id"])
    return by_id, by_acronym, children_by_id


def get_descendant_ids(structure_id, children_by_id):
    """Return a structure id and all descendant ids in deterministic order."""
    output = []
    stack = [int(structure_id)]
    while stack:
        node_id = stack.pop()
        output.append(node_id)
        stack.extend(reversed(sorted(children_by_id[node_id])))
    return output


def get_area_mapping(structure_graph_path, annotated_ids=None):
    """Return the conservative Gamanut-to-CCF2017 mapping table.

    The output has one row for each of the 47 dataset regions. Standard names
    are accepted only when their Allen ontology path is within Isocortex.
    Visual and composite aliases are explicitly defined. If annotated_ids is
    supplied, mappings without any annotated descendant are excluded.
    """
    _, by_acronym, children_by_id = load_structure_nodes(structure_graph_path)
    info = uf.AnimalInfo("mouse", "gamanut")
    rows = []
    used_structure_ids = set()
    annotated_ids = None if annotated_ids is None else set(int(value) for value in annotated_ids)
    for area in info.full_name:
        if area in EXCLUDED_AREAS:
            rows.append({"gamanut_area": area, "allen_acronyms": "", "mapping_type": "excluded", "status": "excluded", "reason": EXCLUDED_AREAS[area], "structure_ids": ""})
            continue
        if area in VISUAL_AREA_MAP:
            acronyms = VISUAL_AREA_MAP[area]
            mapping_type = "visual_alias"
        elif area in COMPOSITE_AREA_MAP:
            acronyms = COMPOSITE_AREA_MAP[area]
            mapping_type = "composite"
        else:
            acronyms = [area]
            mapping_type = "exact_isocortex"

        roots = []
        reason = ""
        for acronym in acronyms:
            candidates = [node for node in by_acronym.get(acronym, []) if "Isocortex" in node["path_acronyms"]]
            if len(candidates) != 1:
                reason = f"Expected one Isocortex node for {acronym}; found {len(candidates)}."
                break
            roots.append(candidates[0])
        structure_ids = []
        if not reason:
            for root in roots:
                structure_ids.extend(get_descendant_ids(root["id"], children_by_id))
            structure_ids = sorted(set(structure_ids))
            if annotated_ids is not None and not annotated_ids.intersection(structure_ids):
                reason = "Mapped Allen structure has no voxels in CCF2017 annotation."
        duplicate_structure_ids = used_structure_ids.intersection(structure_ids)
        if duplicate_structure_ids and not reason:
            reason = f"Allen descendant ids are already assigned to an earlier mapping: {sorted(duplicate_structure_ids)}"
        status = "included" if not reason else "excluded"
        if status == "included":
            used_structure_ids.update(structure_ids)
        rows.append({
            "gamanut_area": area,
            "allen_acronyms": ";".join(acronyms),
            "mapping_type": mapping_type,
            "status": status,
            "reason": reason,
            "structure_ids": ";".join(str(value) for value in structure_ids) if status == "included" else "",
        })
    return pd.DataFrame(rows)


def load_annotation(annotation_path):
    """Load the Allen NRRD in its native (AP, DV, ML) index order."""
    image = sitk.ReadImage(str(annotation_path))
    spacing_um = np.asarray(image.GetSpacing(), dtype=float)
    if not np.allclose(spacing_um, 25.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"Expected 25 um Allen annotation spacing, found {spacing_um.tolist()}.")
    annotation = np.asarray(sitk.GetArrayFromImage(image)).transpose(2, 1, 0)
    if annotation.ndim != 3:
        raise ValueError(f"Expected a 3D annotation, found shape {annotation.shape}.")
    return annotation


def get_hemisphere(annotation, side=HEMISPHERE_SIDE):
    """Return one CCF hemisphere along the Allen mediolateral index axis."""
    if annotation.shape[HEMISPHERE_AXIS] % 2:
        raise ValueError("The Allen mediolateral dimension must be even.")
    midpoint = annotation.shape[HEMISPHERE_AXIS] // 2
    slices = [slice(None)] * 3
    if side == "first":
        slices[HEMISPHERE_AXIS] = slice(0, midpoint)
    elif side == "second":
        slices[HEMISPHERE_AXIS] = slice(midpoint, None)
    else:
        raise ValueError("side must be 'first' or 'second'.")
    return annotation[tuple(slices)]


def get_mapped_annotation(annotation, mapping):
    """Map Allen structure ids to consecutive included-area ids.

    Returns an int16 volume where zero is outside the Gamanut-Allen common
    region subset and included
    areas are numbered from one in mapping-table order. No randomness is used.
    """
    included = mapping[mapping["status"] == "included"].reset_index(drop=True)
    structure_to_area = {}
    for area_id, row in included.iterrows():
        for structure_id in str(row["structure_ids"]).split(";"):
            if structure_id:
                structure_to_area[int(structure_id)] = area_id + 1
    keys = np.asarray(sorted(structure_to_area), dtype=np.uint32)
    values = np.asarray([structure_to_area[int(key)] for key in keys], dtype=np.int16)
    output = np.zeros(annotation.shape, dtype=np.int16)
    for start in range(0, annotation.shape[0], 32):
        chunk = annotation[start:start + 32]
        positions = np.searchsorted(keys, chunk)
        valid = positions < len(keys)
        valid[valid] &= keys[positions[valid]] == chunk[valid]
        mapped = np.zeros(chunk.shape, dtype=np.int16)
        mapped[valid] = values[positions[valid]]
        output[start:start + 32] = mapped
    return output, included


def get_coarse_voxels(mapped_annotation, included_mapping):
    """Aggregate fine labels to a disjoint coarse voxel grid.

    Each coarse voxel receives the modal fine label across its complete block;
    background and excluded structures remain label zero. Returns coordinates
    in mm, zero-based area ids, area names, and isotropic spacing. No randomness
    is used.
    """
    factor = ALLEN_TO_MODEL_DOWNSAMPLE_FACTOR
    expected_spacing = FINE_SPACING_MM * factor
    if not np.isclose(MODEL_SPACING_MM, expected_spacing, rtol=0.0, atol=1e-12):
        raise ValueError("Fixed model spacing is inconsistent with the Allen downsampling factor.")
    shape = np.asarray(mapped_annotation.shape)
    if np.any(shape % factor):
        raise ValueError(f"Annotation shape {tuple(shape)} is not divisible by factor {factor}.")
    coarse_shape = shape // factor
    blocks = mapped_annotation.reshape(
        coarse_shape[0], factor, coarse_shape[1], factor, coarse_shape[2], factor
    ).transpose(0, 2, 4, 1, 3, 5).reshape(-1, factor ** 3)
    coarse_labels = scipy.stats.mode(blocks, axis=1, keepdims=False).mode.astype(np.int16)
    coarse_labels = coarse_labels.reshape(tuple(coarse_shape))
    grid_indices = np.argwhere(coarse_labels > 0)
    area_ids = coarse_labels[tuple(grid_indices.T)] - 1
    spacing = np.repeat(MODEL_SPACING_MM, 3)
    coordinates = grid_indices.astype(np.float64) * spacing
    area_names = included_mapping["gamanut_area"].tolist()
    counts = np.bincount(area_ids, minlength=len(area_names))
    if np.any(counts == 0):
        missing = [area_names[index] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f"Included regions disappeared during downsampling: {missing}")
    return coordinates, area_ids.astype(np.int16), area_names, spacing


def prepare_gamanut_allen_common_regions(raw_dir, output_dir):
    """Prepare one-hemisphere Allen voxels for Gamanut-Allen common regions."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = download_allen_files(raw_dir)
    annotation = load_annotation(paths["annotation"])
    hemisphere = get_hemisphere(annotation)
    mapping = get_area_mapping(paths["structure_graph"], annotated_ids=np.unique(hemisphere))
    mapped_annotation, included = get_mapped_annotation(hemisphere, mapping)
    coordinates, area_ids, area_names, spacing = get_coarse_voxels(mapped_annotation, included)
    mapping.to_csv(output_dir / "area_mapping.csv", index=False)
    np.save(output_dir / "coordinates_mm.npy", coordinates, allow_pickle=False)
    np.save(output_dir / "area_ids.npy", area_ids, allow_pickle=False)
    info = uf.AnimalInfo("mouse", "gamanut")
    included_set = set(area_names)
    common_source_names = [name for name in info.source_name if name in included_set]
    common_target_names = [name for name in info.target_name if name in included_set]
    metadata = {
        "annotation_url": ANNOTATION_URL,
        "structure_graph_url": STRUCTURE_GRAPH_URL,
        "annotation_sha256": _get_sha256(paths["annotation"]),
        "structure_graph_sha256": _get_sha256(paths["structure_graph"]),
        "annotation_shape": list(annotation.shape),
        "hemisphere_axis": HEMISPHERE_AXIS,
        "hemisphere_side": HEMISPHERE_SIDE,
        "hemisphere_shape": list(hemisphere.shape),
        "fine_spacing_mm": FINE_SPACING_MM,
        "downsample_factor": ALLEN_TO_MODEL_DOWNSAMPLE_FACTOR,
        "spacing_mm": spacing.tolist(),
        "area_names": area_names,
        "source_names": common_source_names,
        "target_names": common_target_names,
        "excluded_names": mapping.loc[mapping["status"] == "excluded", "gamanut_area"].tolist(),
        "voxel_count": int(len(coordinates)),
        "grid_shape": (np.asarray(hemisphere.shape) // ALLEN_TO_MODEL_DOWNSAMPLE_FACTOR).astype(int).tolist(),
    }
    _save_json(metadata, output_dir / "metadata.json")
    return mapping, metadata


def load_gamanut_allen_common_regions(output_dir):
    """Load prepared arrays for the Gamanut-Allen common region subset."""
    output_dir = Path(output_dir)
    with (output_dir / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    coordinates = np.load(output_dir / "coordinates_mm.npy", allow_pickle=False)
    area_ids = np.load(output_dir / "area_ids.npy", allow_pickle=False)
    return coordinates, area_ids, metadata["area_names"], np.asarray(metadata["spacing_mm"], dtype=float), metadata


# ============================================================
# Surface and voxel geodesic preprocessing
# ============================================================



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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ALLEN_DIR = PROJECT_ROOT / "raw_data" / "mouse_allen_ccf2017"
OUTPUT_DIR = PROJECT_ROOT / "results" / "cross_species_3d"
PREPROCESSING_DIR = OUTPUT_DIR / "preprocessing"
FULL_DISTANCE_WORKER_COUNT = 64
FULL_DISTANCE_SOURCE_CHUNK_SIZE = 256
FULL_DISTANCE_DIJKSTRA_BATCH_SIZE = 16
GEODESIC_BIN_WIDTH_MM = {"macaque_markov40": 0.02, "human_lariviere": 0.1}


def get_human_surface_data():
    """Return the labeled human midthickness mesh and area ids."""
    surface_path = PROJECT_ROOT / "raw_data" / "human_lariviere" / "threeD_data" / "Q1-Q6_RelatedParcellation210.L.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii"
    voxel_path = PROJECT_ROOT / "data" / "human_lariviere" / "threeD_data" / "area_voxel_dict_3D.joblib"
    surface = nib.load(str(surface_path))
    vertices = np.asarray(surface.darrays[0].data, dtype=np.float64)
    faces = np.asarray(surface.darrays[1].data, dtype=np.int32)
    vertex_dict = cf.load_joblib(str(voxel_path))
    area_names = list(uf.AnimalInfo("human", "lariviere").full_name)
    tree = scipy.spatial.cKDTree(vertices)
    selected_indices = []
    area_ids = []
    for area_id, area_name in enumerate(area_names):
        coordinates = np.asarray(vertex_dict[area_name], dtype=np.float64)
        distances, indices = tree.query(coordinates, k=1)
        if np.any(distances != 0):
            raise ValueError(f"Human area {area_name} does not align exactly with the midthickness mesh.")
        selected_indices.append(indices)
        area_ids.append(np.full(len(indices), area_id, dtype=np.int16))
    selected_indices = np.concatenate(selected_indices).astype(np.int32)
    area_ids = np.concatenate(area_ids)
    if len(np.unique(selected_indices)) != len(selected_indices):
        raise ValueError("Human labeled surface vertices are not unique.")
    full_to_selected = np.full(len(vertices), -1, dtype=np.int32)
    full_to_selected[selected_indices] = np.arange(len(selected_indices), dtype=np.int32)
    selected_faces = full_to_selected[faces]
    selected_faces = selected_faces[np.all(selected_faces >= 0, axis=1)]
    return vertices[selected_indices], selected_faces, area_ids, area_names


def get_surface_graph(vertices, faces):
    """Return an undirected mesh-edge graph with physical edge lengths."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = scipy.sparse.coo_matrix(
        (np.r_[weights, weights], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    graph.sort_indices()
    component_count, _ = scipy.sparse.csgraph.connected_components(graph, directed=False)
    if component_count != 1:
        raise ValueError(f"Human labeled surface graph has {component_count} components.")
    return graph


def get_shortest_path(predecessors, source, target):
    """Reconstruct one scipy shortest path as vertex indices."""
    path = [int(target)]
    while path[-1] != source:
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise ValueError("Could not reconstruct the selected human geodesic path.")
        path.append(predecessor)
    return np.asarray(path[::-1], dtype=np.int32)


def build_human_shortcut_cache(full_distance_dir, output_dir):
    """Build one human shortcut example from the saved full-pair distances."""
    full_distance_dir = Path(full_distance_dir)
    output_dir = Path(output_dir)
    required = [
        output_dir / "metadata.json",
        output_dir / "surface_vertices.npy",
        output_dir / "surface_faces.npy",
        output_dir / "shortcut_path_indices.npy",
    ]
    if all(path.exists() for path in required):
        return _load_json(output_dir / "metadata.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces, _, _ = get_human_surface_data()
    graph = get_surface_graph(vertices, faces)
    best_ratio = -np.inf
    best_pair = None
    chunk_metadata = sorted(
        full_distance_dir.glob("chunks/*/metadata.json"),
        key=lambda path: path.parent.name,
    )
    if not chunk_metadata:
        raise FileNotFoundError(f"No full-pair distance chunks found under {full_distance_dir}.")
    for metadata_path in chunk_metadata:
        metadata = _load_json(metadata_path)
        packed = np.load(metadata_path.parent / "geodesic_upper_float32.npy", mmap_mode="r")
        offset = 0
        for source in range(metadata["source_start"], metadata["source_stop"]):
            targets = np.arange(source + 1, len(vertices), dtype=np.int32)
            row_count = len(targets)
            geodesic = np.asarray(packed[offset:offset + row_count], dtype=np.float64)
            offset += row_count
            euclidean = np.linalg.norm(vertices[targets] - vertices[source], axis=1)
            eligible = (euclidean >= 2.0) & (euclidean <= 15.0)
            ratios = np.divide(geodesic, euclidean, out=np.zeros_like(geodesic), where=eligible)
            target_position = int(np.argmax(ratios))
            if ratios[target_position] > best_ratio:
                best_ratio = float(ratios[target_position])
                best_pair = (source, int(targets[target_position]))
        if offset != len(packed):
            raise ValueError(f"Packed distance count mismatch in {metadata_path.parent.name}.")
    if best_pair is None:
        raise ValueError("No eligible human shortcut pair was found.")
    source, target = best_pair
    distances, predecessors = scipy.sparse.csgraph.dijkstra(
        graph, directed=False, indices=source, return_predecessors=True
    )
    path = get_shortest_path(predecessors, source, target)
    np.save(output_dir / "surface_vertices.npy", vertices, allow_pickle=False)
    np.save(output_dir / "surface_faces.npy", faces, allow_pickle=False)
    np.save(output_dir / "shortcut_path_indices.npy", path, allow_pickle=False)
    metadata = {
        "animal": "human",
        "dataset_key": "lariviere",
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "selection_euclidean_range_mm": [2.0, 15.0],
        "selection_distance_dtype": "float32",
        "shortcut_source_index": source,
        "shortcut_target_index": target,
        "shortcut_euclidean_mm": float(np.linalg.norm(vertices[source] - vertices[target])),
        "shortcut_geodesic_mm": float(distances[target]),
        "shortcut_ratio": float(distances[target] / np.linalg.norm(vertices[source] - vertices[target])),
    }
    _save_json(metadata, output_dir / "metadata.json")
    return metadata


def _extend_bin_counts(counts, bin_count):
    if counts.shape[1] >= bin_count:
        return counts
    extended = np.zeros((counts.shape[0], bin_count, counts.shape[2]), dtype=np.uint64)
    extended[:, :counts.shape[1]] = counts
    return extended


def _add_forward_bin_counts(counts, source_slot, target_area_ids, distances, bin_width):
    bin_indices = np.floor(np.asarray(distances) / bin_width).astype(np.int64)
    required_bin_count = int(np.max(bin_indices)) + 1
    counts = _extend_bin_counts(counts, required_bin_count)
    area_count = counts.shape[2]
    flat_indices = bin_indices * area_count + target_area_ids
    local = np.bincount(flat_indices, minlength=required_bin_count * area_count).astype(np.uint64)
    counts[source_slot, :required_bin_count] += local.reshape(required_bin_count, area_count)
    return counts


def get_full_distance_chunk(graph, coordinates, area_ids, task, output_dir, bin_width, dijkstra_batch_size):
    """Save one packed upper-triangle distance chunk and its pair-bin counts.

    The chunk contains every pair whose lower point index lies in the task
    interval and whose upper index is larger. Geodesic and Euclidean counts use
    the same unordered point pairs. No randomness is used.
    """
    output_dir = Path(output_dir)
    chunk_dir = output_dir / "chunks" / task["task_id"]
    metadata_path = chunk_dir / "metadata.json"
    distance_path = chunk_dir / "geodesic_upper_float32.npy"
    counts_path = chunk_dir / "forward_bin_counts.npz"
    if metadata_path.exists() and distance_path.exists() and counts_path.exists():
        return _load_json(metadata_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    source_start = int(task["source_start"])
    source_stop = int(task["source_stop"])
    point_count = len(coordinates)
    source_indices = np.arange(source_start, source_stop, dtype=np.int64)
    pair_count = int(np.sum(point_count - source_indices - 1, dtype=np.int64))
    packed = np.lib.format.open_memmap(distance_path, mode="w+", dtype=np.float32, shape=(pair_count,))
    source_area_ids = np.unique(area_ids[source_start:source_stop])
    source_lookup = {int(area_id): index for index, area_id in enumerate(source_area_ids)}
    geodesic_counts = np.zeros((len(source_area_ids), 1, len(np.unique(area_ids))), dtype=np.uint64)
    euclidean_counts = np.zeros_like(geodesic_counts)
    geodesic_max = 0.0
    euclidean_max = 0.0
    offset = 0
    for batch_start in range(source_start, source_stop, int(dijkstra_batch_size)):
        batch_stop = min(batch_start + int(dijkstra_batch_size), source_stop)
        batch_sources = np.arange(batch_start, batch_stop, dtype=np.int64)
        geodesic_batch = scipy.sparse.csgraph.dijkstra(
            graph, directed=False, indices=batch_sources, return_predecessors=False
        )
        for local_index, source_index in enumerate(batch_sources):
            target_slice = slice(source_index + 1, point_count)
            geodesic = geodesic_batch[local_index, target_slice]
            if not np.all(np.isfinite(geodesic)):
                raise ValueError(f"Non-finite geodesic distance in {task['task_id']}.")
            euclidean = np.linalg.norm(coordinates[target_slice] - coordinates[source_index], axis=1)
            row_count = len(geodesic)
            packed[offset:offset + row_count] = geodesic.astype(np.float32)
            offset += row_count
            if row_count == 0:
                continue
            geodesic_max = max(geodesic_max, float(np.max(geodesic)))
            euclidean_max = max(euclidean_max, float(np.max(euclidean)))
            source_slot = source_lookup[int(area_ids[source_index])]
            target_area_ids = area_ids[target_slice].astype(np.int64, copy=False)
            geodesic_counts = _add_forward_bin_counts(
                geodesic_counts, source_slot, target_area_ids, geodesic, bin_width
            )
            euclidean_counts = _add_forward_bin_counts(
                euclidean_counts, source_slot, target_area_ids, euclidean, bin_width
            )
    if offset != pair_count:
        raise ValueError(f"Packed distance count mismatch in {task['task_id']}.")
    packed.flush()
    if int(np.sum(geodesic_counts)) != pair_count or int(np.sum(euclidean_counts)) != pair_count:
        raise ValueError(f"Pair-bin count mismatch in {task['task_id']}.")
    np.savez(
        counts_path, source_area_ids=source_area_ids, geodesic_counts=geodesic_counts,
        euclidean_counts=euclidean_counts,
    )
    metadata = {
        **task,
        "point_count": point_count,
        "unordered_pair_count": pair_count,
        "geodesic_maximum_distance_mm": geodesic_max,
        "euclidean_maximum_distance_mm": euclidean_max,
        "distance_dtype": "float32",
        "bin_width_mm": float(bin_width),
    }
    _save_json(metadata, metadata_path)
    return metadata


def get_full_distance_chunks_worker(task_list, graph_path, coordinates_path, area_ids_path, output_dir, bin_width, dijkstra_batch_size):
    """Process deterministic distance chunks after loading shared inputs once."""
    graph = scipy.sparse.load_npz(graph_path)
    coordinates = np.load(coordinates_path, mmap_mode="r")
    area_ids = np.load(area_ids_path, mmap_mode="r")
    return [
        get_full_distance_chunk(
            graph, coordinates, area_ids, task, output_dir, bin_width, dijkstra_batch_size
        )
        for task in task_list
    ]


def _save_full_binned_catalog(output_dir, metric, chunk_metadata, area_names, point_count, bin_width):
    maximum_distance = max(item[f"{metric}_maximum_distance_mm"] for item in chunk_metadata)
    bin_count = int(np.floor(maximum_distance / bin_width)) + 1
    area_count = len(area_names)
    counts = np.zeros((bin_count, area_count, area_count), dtype=np.uint64)
    for item in sorted(chunk_metadata, key=lambda value: value["task_id"]):
        counts_path = Path(output_dir) / "chunks" / item["task_id"] / "forward_bin_counts.npz"
        with np.load(counts_path) as local:
            source_area_ids = local["source_area_ids"]
            forward = local[f"{metric}_counts"]
        for source_slot, source_area_id in enumerate(source_area_ids):
            local_bin_count = forward.shape[1]
            counts[:local_bin_count, source_area_id, :] += forward[source_slot]
            counts[:local_bin_count, :, source_area_id] += forward[source_slot]
    expected_pair_count = point_count * (point_count - 1)
    if int(np.sum(counts, dtype=np.uint64)) != expected_pair_count:
        raise ValueError(f"{metric} catalog does not contain every ordered non-self pair.")
    global_counts = np.sum(counts, axis=(1, 2), dtype=np.uint64)
    occupied = np.flatnonzero(global_counts)
    final_bin_count = int(occupied[-1]) + 1
    counts = counts[:final_bin_count]
    global_counts = global_counts[:final_bin_count]
    bin_edges = np.arange(final_bin_count + 1, dtype=np.float64) * bin_width
    catalog_dir = Path(output_dir) / f"{metric}_catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    np.save(catalog_dir / "bin_edges.npy", bin_edges, allow_pickle=False)
    np.save(catalog_dir / "global_bin_counts.npy", global_counts, allow_pickle=False)
    np.save(catalog_dir / "area_pair_bin_counts.npy", counts, allow_pickle=False)
    metadata = {
        "distance_metric": metric,
        "bin_width_mm": float(bin_width),
        "bin_count": final_bin_count,
        "occupied_bin_count": int(np.count_nonzero(global_counts)),
        "point_count": int(point_count),
        "ordered_nonself_pair_count": int(expected_pair_count),
        "maximum_distance_mm": float(maximum_distance),
        "area_count": area_count,
        "area_names": list(area_names),
    }
    _save_json(metadata, catalog_dir / "metadata.json")
    return metadata


def build_full_distance_catalogs(graph, coordinates, area_ids, area_names, output_dir, bin_width, worker_count=FULL_DISTANCE_WORKER_COUNT, source_chunk_size=FULL_DISTANCE_SOURCE_CHUNK_SIZE, dijkstra_batch_size=FULL_DISTANCE_DIJKSTRA_BATCH_SIZE):
    """Build exact Euclidean/geodesic bin catalogs and packed geodesic chunks.

    Only the largest graph component is retained. Every unordered point pair is
    saved once as float32 geodesic distance; both orientations are represented
    exactly in the two area-pair/bin catalogs. Work is deterministic and
    resumable at source-chunk boundaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    largest_mask, component_sizes = get_largest_component_mask(graph)
    component_indices = np.flatnonzero(largest_mask)
    component_graph = graph[component_indices][:, component_indices].tocsr()
    component_coordinates = np.asarray(coordinates)[component_indices]
    component_area_ids = np.asarray(area_ids)[component_indices]
    if not np.array_equal(np.unique(component_area_ids), np.arange(len(area_names))):
        raise ValueError("Every area must occur in the largest graph component.")
    graph_path = output_dir / "largest_component_graph.npz"
    coordinates_path = output_dir / "coordinates.npy"
    area_ids_path = output_dir / "area_ids.npy"
    component_indices_path = output_dir / "original_point_indices.npy"
    scipy.sparse.save_npz(graph_path, component_graph)
    np.save(coordinates_path, component_coordinates, allow_pickle=False)
    np.save(area_ids_path, component_area_ids, allow_pickle=False)
    np.save(component_indices_path, component_indices, allow_pickle=False)
    point_count = len(component_coordinates)
    tasks = []
    for task_index, source_start in enumerate(range(0, point_count - 1, int(source_chunk_size))):
        tasks.append({
            "task_id": f"chunk_{task_index:05d}",
            "source_start": int(source_start),
            "source_stop": int(min(source_start + int(source_chunk_size), point_count - 1)),
        })
    worker_count = min(int(worker_count), len(tasks))
    task_groups = [tasks[index::worker_count] for index in range(worker_count)]
    results = cf.multi_process(
        process_num=worker_count,
        func=get_full_distance_chunks_worker,
        kwargs_list=[{
            "task_list": task_group,
            "graph_path": graph_path,
            "coordinates_path": coordinates_path,
            "area_ids_path": area_ids_path,
            "output_dir": output_dir,
            "bin_width": float(bin_width),
            "dijkstra_batch_size": int(dijkstra_batch_size),
        } for task_group in task_groups],
        func_name=f"full geodesic distances: {output_dir.parent.name}",
    )
    chunk_metadata = sorted([item for group in results for item in group], key=lambda item: item["task_id"])
    if [item["task_id"] for item in chunk_metadata] != [task["task_id"] for task in tasks]:
        raise ValueError("Full-distance chunks are missing or duplicated.")
    euclidean = _save_full_binned_catalog(
        output_dir, "euclidean", chunk_metadata, area_names, point_count, float(bin_width)
    )
    geodesic = _save_full_binned_catalog(
        output_dir, "geodesic", chunk_metadata, area_names, point_count, float(bin_width)
    )
    metadata = {
        "bin_width_mm": float(bin_width),
        "point_count": point_count,
        "original_point_count": len(coordinates),
        "graph_component_count": len(component_sizes),
        "largest_component_point_count": int(np.max(component_sizes)),
        "source_chunk_size": int(source_chunk_size),
        "dijkstra_batch_size": int(dijkstra_batch_size),
        "distance_chunk_count": len(tasks),
        "packed_upper_triangle_pair_count": int(point_count * (point_count - 1) // 2),
        "packed_distance_dtype": "float32",
        "area_names": list(area_names),
        "euclidean_bin_count": euclidean["bin_count"],
        "geodesic_bin_count": geodesic["bin_count"],
    }
    _save_json(metadata, output_dir / "metadata.json")
    return metadata


def build_macaque_full_distance_catalogs(worker_count=FULL_DISTANCE_WORKER_COUNT):
    """Build full macaque Euclidean and no-corner-cutting geodesic catalogs."""
    dataset_dir = PREPROCESSING_DIR / "macaque_markov40"
    catalog = model.load_displacement_catalog(dataset_dir / "distance_catalog")
    graph = get_no_corner_cutting_graph(
        catalog["grid_indices"], catalog["voxel_map"], catalog["metadata"]["spacing_mm"]
    )
    return build_full_distance_catalogs(
        graph, catalog["coordinates"], catalog["area_ids"], catalog["metadata"]["area_names"],
        dataset_dir / "full_pair_distances", GEODESIC_BIN_WIDTH_MM["macaque_markov40"],
        worker_count=worker_count,
    )


def build_human_full_distance_catalogs(worker_count=FULL_DISTANCE_WORKER_COUNT):
    """Build full human Euclidean and surface-geodesic distance catalogs."""
    vertices, faces, area_ids, area_names = get_human_surface_data()
    graph = get_surface_graph(vertices, faces)
    return build_full_distance_catalogs(
        graph, vertices, area_ids, area_names,
        PREPROCESSING_DIR / "human_lariviere" / "full_pair_distances",
        GEODESIC_BIN_WIDTH_MM["human_lariviere"], worker_count=worker_count,
    )


def main():
    """Prepare all four species under one isolated result root."""
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    mouse_regions = PREPROCESSING_DIR / "mouse_gamanut" / "common_regions"
    prepare_gamanut_allen_common_regions(RAW_ALLEN_DIR, mouse_regions)
    mouse_coordinates, mouse_area_ids, mouse_names, mouse_spacing, _ = load_gamanut_allen_common_regions(mouse_regions)
    model.build_displacement_catalog(
        PROJECT_ROOT, PREPROCESSING_DIR / "mouse_gamanut" / "distance_catalog",
        animal="mouse", dataset_key="gamanut", coordinates=mouse_coordinates,
        area_ids=mouse_area_ids, area_names=mouse_names, spacing=mouse_spacing,
    )
    for animal, dataset_key in [("marmoset", "majka"), ("macaque", "markov40")]:
        model.build_displacement_catalog(
            PROJECT_ROOT, PREPROCESSING_DIR / f"{animal}_{dataset_key}" / "distance_catalog",
            animal=animal, dataset_key=dataset_key,
        )
    model.build_human_bin_catalog(
        PROJECT_ROOT, PREPROCESSING_DIR / "human_lariviere" / "distance_catalog",
        bin_width=GEODESIC_BIN_WIDTH_MM["human_lariviere"],
    )
    build_macaque_full_distance_catalogs()
    build_human_full_distance_catalogs()
    build_human_shortcut_cache(
        PREPROCESSING_DIR / "human_lariviere" / "full_pair_distances",
        PREPROCESSING_DIR / "human_lariviere" / "shortcut",
    )


if __name__ == "__main__":
    main()
