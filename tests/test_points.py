import numpy as np
import polars as pl

from analysis.products.points import DISTANCE_THRESHOLD_NM, map_root_points

# Two triangles tiling the z=0 square [0, 1000] x [0, 1000] nm. Vertices 0 and 1
# are in domain 7, vertex 2 in domain 8, and vertex 3 in no domain.
MESH = (
    np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [1000.0, 1000.0, 0.0]]),
    np.array([[0, 1, 2], [1, 3, 2]]),
)
VERTEX_DOMAINS = pl.DataFrame({"mesh_index": [0, 1, 2, 3], "domain_id": [7, 7, 8, None]})


def test_map_root_points():
    points = pl.DataFrame({
        "point_id": [10, 11, 12, 13, 14],
        "x": [100.0, 900.0, 100.0, 900.0, 500.0],
        "y": [100.0, 100.0, 900.0, 900.0, 500.0],
        # The last point sits 100 nm past the threshold above the surface.
        "z": [10.0, 0.0, -20.0, 0.0, DISTANCE_THRESHOLD_NM + 100.0],
    })
    mapped = map_root_points(points, MESH, VERTEX_DOMAINS)
    assert mapped["point_id"].to_list() == [10, 11, 12, 13, 14]
    assert mapped["mesh_index"].to_list() == [0, 1, 2, 3, None]
    assert mapped["domain_id"].to_list() == [7, 7, 8, None, None]
    assert mapped["distance_nm"].to_list()[:3] == [10.0, 0.0, 20.0]


def test_map_root_points_single_point():
    points = pl.DataFrame({"point_id": [1], "x": [900.0], "y": [100.0], "z": [5.0]})
    mapped = map_root_points(points, MESH, VERTEX_DOMAINS)
    assert mapped["mesh_index"].to_list() == [1]
    assert mapped["domain_id"].to_list() == [7]
