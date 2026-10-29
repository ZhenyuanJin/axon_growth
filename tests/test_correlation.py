import numpy as np
import pandas as pd
import pytest

from utils_function import cf


@pytest.mark.parametrize('rows,columns', [(['A', 'B', 'C', 'D'], ['C', 'B']),
                                        (['A', 'B', 'C', 'D'], ['C']),
                                        (['C'], ['A', 'B', 'C', 'D']),
                                        (['C', 'A', 'B'], ['B', 'C', 'A'])])
def test_correlation_excludes_matching_area_names(rows, columns):
    first = np.arange(len(rows) * len(columns), dtype=float).reshape(len(rows), len(columns)) + 1
    second = first ** 2 + np.cos(first)
    data = pd.DataFrame(first, index=rows, columns=columns)
    generated = pd.DataFrame(second, index=rows, columns=columns)
    mask = np.asarray(rows)[:, None] != np.asarray(columns)[None, :]
    expected = np.corrcoef(first[mask], second[mask])[0, 1]
    shuffled = generated.iloc[::-1, ::-1]
    assert cf.get_pd_rectangle_corr(data, shuffled, diagonal=False) == pytest.approx(expected)
    assert cf.get_pd_rectangle_corr(data, shuffled, diagonal=True) == pytest.approx(
        np.corrcoef(first.ravel(), second.ravel())[0, 1]
    )


def test_correlation_without_matching_names_uses_all_entries():
    data = pd.DataFrame([[1., 2.], [4., 8.]], index=['A', 'B'], columns=['X', 'Y'])
    generated = pd.DataFrame([[8., 2.], [3., 1.]], index=data.index, columns=data.columns)
    expected = np.corrcoef(data.values.ravel(), generated.values.ravel())[0, 1]
    assert cf.get_pd_rectangle_corr(data, generated, diagonal=False) == pytest.approx(expected)


def test_correlation_rejects_unmatched_or_duplicate_labels():
    data = pd.DataFrame([[1., 2.], [4., 8.]], index=['A', 'B'], columns=['A', 'B'])
    other = data.rename(index={'A': 'C'})
    with pytest.raises(ValueError, match='same row and column labels'):
        cf.get_pd_rectangle_corr(data, other)
    duplicated = data.copy()
    duplicated.index = ['A', 'A']
    with pytest.raises(ValueError, match='unique'):
        cf.get_pd_rectangle_corr(duplicated, duplicated)
