import numpy as np
import scipy.sparse.csgraph
from macaque_geodesic_analysis import geodesic


def _get_voxel_map(points):
    points = np.asarray(points, dtype=np.int16)
    shape = tuple(np.max(points, axis=0) + 1)
    voxel_map = np.full(shape, -1, dtype=np.int32)
    voxel_map[tuple(points.T)] = np.arange(len(points), dtype=np.int32)
    return points, voxel_map


def test_diagonal_edge_cannot_cut_empty_corner():
    points, voxel_map = _get_voxel_map([[0, 0, 0], [1, 1, 0]])
    graph = geodesic.get_no_corner_cutting_graph(points, voxel_map, np.ones(3))
    assert graph.nnz == 0


def test_diagonal_edge_requires_filled_local_block():
    points, voxel_map = _get_voxel_map([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    graph = geodesic.get_no_corner_cutting_graph(points, voxel_map, np.ones(3))
    distance = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=0)[3]
    assert np.isclose(distance, np.sqrt(2))


def test_body_diagonal_requires_every_intermediate_voxel():
    points, voxel_map = _get_voxel_map([
        [0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0],
        [0, 0, 1], [1, 1, 0], [1, 0, 1],
    ])
    graph = geodesic.get_no_corner_cutting_graph(points, voxel_map, np.ones(3))
    assert graph[0, 1] == 0
