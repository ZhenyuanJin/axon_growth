from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import utils_function as uf


def load_macaque_matrices(project_root):
    """Load aligned measured, accepted 2D, and five-seed mean 3D matrices."""
    project_root = Path(project_root)
    info = uf.AnimalInfo("macaque", "markov40")
    names = info.full_name
    measured = info.SC_data.loc[names, names]
    model_2d_path = project_root / "results" / "models" / "connection_generation" / "macaque_markov40" / "4ff50ccd68d0" / "connectivity" / "FLNe.csv"
    model_2d = pd.read_csv(model_2d_path, index_col=0).loc[names, names]
    model_3d_paths = sorted(
        (project_root / "results" / "voxel_edr_3d_macaque_markov40" / "best").glob(
            "best_lambda_000_seed_*/connectivity/FLNe.csv"
        )
    )
    if len(model_3d_paths) != 5:
        raise ValueError(f"Expected five 3D best-seed matrices, found {len(model_3d_paths)}.")
    arrays = [pd.read_csv(path, index_col=0).loc[names, names].values for path in model_3d_paths]
    model_3d = pd.DataFrame(np.mean(arrays, axis=0), index=names, columns=names)
    return measured, model_2d, model_3d


def get_ratio_pair_table(ratio_matrix, row_names, column_names):
    """Return one row for every valid directed macaque comparison pair."""
    ratio_matrix = np.asarray(ratio_matrix, dtype=np.float64)
    if ratio_matrix.shape != (len(row_names), len(column_names)):
        raise ValueError("ratio_matrix shape does not match row and column names.")
    row_id, column_id = np.indices(ratio_matrix.shape)
    off_diagonal = np.asarray(row_names)[:, None] != np.asarray(column_names)[None, :]
    return pd.DataFrame({
        "source_area": np.asarray(row_names)[row_id[off_diagonal]],
        "target_area": np.asarray(column_names)[column_id[off_diagonal]],
        "geodesic_to_euclidean_ratio": ratio_matrix[off_diagonal],
    })
