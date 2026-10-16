import numpy as np
import pandas as pd
from human_geodesic_analysis import analysis


def test_binned_rmae_uses_shared_pooled_ratio_bins(monkeypatch):
    names = ["A", "B", "C", "D", "E"]
    row, col = np.triu_indices(len(names), k=1)
    pair_table = pd.DataFrame({
        "source_area": np.asarray(names)[row],
        "target_area": np.asarray(names)[col],
        "geodesic_to_euclidean_ratio": np.linspace(1.0, 3.0, len(row)),
    })
    matrix = pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)
    relative_error_2d = pd.DataFrame(
        np.arange(len(names) ** 2, dtype=np.float64).reshape(len(names), len(names)),
        index=names, columns=names
    )
    relative_error_3d = relative_error_2d + 1.0
    returned = iter([relative_error_2d, relative_error_3d])
    monkeypatch.setattr(analysis.uf, "data_model_relative_error", lambda *args, **kwargs: next(returned))
    existing = pd.DataFrame(False, index=names, columns=names)
    existing.values[row[:5], col[:5]] = True
    non_link = pd.DataFrame(False, index=names, columns=names)
    non_link.values[row[5:], col[5:]] = True
    monkeypatch.setattr(
        analysis.uf, "get_weight_category_mask",
        lambda *args, **kwargs: {"all": existing, "non": non_link}
    )
    result = analysis.get_ratio_binned_rmae(
        pair_table, matrix, matrix, matrix, 4, "human", "lariviere"
    )
    existing_result = result[result["link_category"] == "existing"]
    non_link_result = result[result["link_category"] == "non-link"]
    assert np.allclose(existing_result["ratio_mean"], non_link_result["ratio_mean"])
    assert np.all(np.diff(existing_result["ratio_mean"]) > 0)
    assert existing_result["pair_count"].sum() == 5
    assert non_link_result["pair_count"].sum() == 5
    occupied = result["pair_count"] > 0
    assert np.allclose(
        result.loc[occupied, "rmae_3d"] - result.loc[occupied, "rmae_2d"], 1.0
    )
