"""The dendrite geometry of the column cells, from the local skeleton files.

The dendrite is the `dendrite_mask` of the skeleton: the vertices that are not axon,
and the axon vertices on the path from them to the root. Path distances and cable come
from the native skeleton (nm), and are given in µm.
Heights and the coordinates for plots are in a post-transform frame (`FRAMES`):

- "streamline": the minnie streamline field. Each cell uses the one streamline
  through its soma. y is the depth along that streamline, and x and z are radial
  around it.
- "affine": the rigid minnie transform (a rotation, a translation and a 1/1000
  scaling).
"""

import numpy as np
import polars as pl
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from standard_transform import minnie_ds

from .io import CELL_TYPE_CATEGORIES
from .products.catalog import Tables
from .products.skeleton import cached_skeleton, dendrite_mask
from .sources import VERSION
from .stash import Stash

EXCITATORY_TYPES = [
    c for c in CELL_TYPE_CATEGORIES if c in ["23P", "4P", "5P-IT", "5P-ET", "5P-NP", "6P-IT", "6P-CT"]
]
FRAMES = ("streamline", "affine")
# The labels of `dendrite_regions`. "apical" holds the obliques and the tuft.
REGIONS = ["soma", "trunk", "apical", "basal"]
# The path radii (µm) of the density windows in `scripts/spine_density/spine_density.py`.
RADII_UM = [5, 10, 20]


def spine_density_stash(name: str) -> Stash:
    """The `spines` or `vertices` output of `scripts/spine_density/spine_density.py`."""
    return Stash(
        f"spine_density/{name}",
        inputs=[Tables["column_spines"], Tables["column_post_synapses"]],
        radii_um=RADII_UM,
        # The rule of `dendrite_mask`.
        dendrite_mask="keep_axon_on_root_paths",
        version=VERSION,
    )


def column_cells(cell_types: list[str] = EXCITATORY_TYPES) -> pl.DataFrame:
    """`root_id` and `cell_type` of the column cells of `cell_types`.

    NOTE: cell_info and cell_selection, not proofread_cell_info, so that a rebuild of
    column_post_synapses does not trigger a rebuild here.
    """
    column_roots = Tables["cell_selection"].lazy().filter(pl.col("is_column")).select("pt_root_id")
    return (
        Tables["cell_info"]
        .lazy()
        .join(column_roots, on="pt_root_id", how="semi")
        .filter(pl.col("cell_type").cast(pl.String).is_in(cell_types))
        .select(pl.col("pt_root_id").alias("root_id"), pl.col("cell_type").cast(pl.String))
        .unique("root_id")
        .sort("root_id")
        .collect()
    )


def frame_points(points: np.ndarray, soma: np.ndarray, frame: str = "streamline") -> np.ndarray:
    """`points` (n x 3, nm) in `frame` (µm). The streamline frame uses the streamline
    through `soma` (nm)."""
    points = np.asarray(points, dtype=float)
    if frame == "streamline":
        streamline = minnie_ds.streamline().streamline_at_point(np.asarray(soma, dtype=float))
        return streamline.transformer()(points)
    if frame == "affine":
        return minnie_ds.transform().apply(points)
    raise ValueError(f"frame must be one of {FRAMES}, not {frame!r}")


def dendrite_skeleton(root_id: int):
    """The cached skeleton of `root_id` masked to `dendrite_mask`, in nm. It is one
    connected tree with the soma at its root."""
    cell = cached_skeleton(root_id)
    skel = cell.apply_mask("skeleton", dendrite_mask(cell.skeleton)).skeleton
    assert (skel.parent_node_array < 0).sum() == 1, f"{root_id}: the dendrite is not one tree"
    return skel


def vertex_geometry(skel, frame: str = "streamline") -> pl.DataFrame:
    """One row per vertex of the dendrite skeleton `skel` (from `dendrite_skeleton`),
    in the positional order of `skel`."""
    native = np.asarray(skel.vertices, dtype=float)
    soma_nm = np.asarray(skel.root_location, dtype=float)
    xyz = frame_points(native, soma_nm, frame)
    soma = xyz[skel.root_positional]
    return pl.DataFrame(
        {
            "skeleton_vertex_id": np.asarray(skel.vertex_index, dtype=np.int64),
            "compartment": np.asarray(skel.get_feature("compartment")),
            "path_distance_to_soma_um": skel.distance_to_root(as_positional=True) / 1000,
            "half_edge_length_um": np.asarray(skel.half_edge_length) / 1000,
            "euc_distance_to_soma_um": np.linalg.norm(native - soma_nm, axis=1) / 1000,
            "x_um": xyz[:, 0],
            "y_um": xyz[:, 1],
            "z_um": xyz[:, 2],
            # y grows toward the white matter, so a positive height is toward the pia.
            "height_above_soma_um": soma[1] - xyz[:, 1],
            "lateral_distance_um": np.linalg.norm(xyz[:, [0, 2]] - soma[[0, 2]], axis=1),
        }
    )


def tip_path(skel, geometry: pl.DataFrame, rule: str = "highest") -> np.ndarray:
    """Positional vertices of `skel` from one dendrite tip to the root.

    "highest": the tip nearest the pia, in the frame of `geometry` (`vertex_geometry`).
    "farthest": the tip with the longest native path to the soma.
    """
    tips = skel.end_points_positional
    if rule == "highest":
        tip = tips[np.argmin(geometry["y_um"].to_numpy()[tips])]
    elif rule == "farthest":
        tip = tips[np.argmax(geometry["path_distance_to_soma_um"].to_numpy()[tips])]
    else:
        raise ValueError(rule)
    parent = skel.parent_node_array
    path = [tip]
    while path[-1] != skel.root_positional:
        path.append(parent[path[-1]])
        assert path[-1] >= 0, "path left the tree before the root"
    return np.array(path)


def top_tips(skel, geometry: pl.DataFrame, window_um: float) -> np.ndarray:
    """The positional dendrite tips of `skel` within `window_um` of depth of the
    highest tip, in the frame of `geometry` (`vertex_geometry`)."""
    tips = skel.end_points_positional
    y = geometry["y_um"].to_numpy()[tips]
    return tips[y <= y.min() + window_um]


def union_path(skel, tips: np.ndarray) -> np.ndarray:
    """The positional vertices on the paths from `tips` to the root."""
    parent = skel.parent_node_array
    on_path = np.zeros(skel.n_vertices, dtype=bool)
    for tip in tips:
        vertex = tip
        # A walk stops at the root or at a vertex that an earlier walk marked.
        while vertex >= 0 and not on_path[vertex]:
            on_path[vertex] = True
            vertex = parent[vertex]
    assert on_path[skel.root_positional]
    return np.flatnonzero(on_path)


def dendrite_regions(
    skel, geometry: pl.DataFrame, top_tip_window_um: float = 25, trunk_start_um: float = 25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The region (`REGIONS`) of each positional vertex of the dendrite skeleton `skel`,
    the trunk vertices and the top tips. `geometry` is the `vertex_geometry` of `skel`.

    The rule:

    1. The top tips are the tips within `top_tip_window_um` of depth of the highest
       tip. The union of their paths to the soma, other than the soma, is the "trunk".
    2. Remove the edges of the trunk that are at least `trunk_start_um` of path from
       the soma, and find the connected components of the rest.
    3. A vertex in the component of the soma is "basal". A vertex in the component of
       a trunk vertex is "apical". So a branch that leaves the trunk within
       `trunk_start_um` of the soma is basal.
    """
    tips = top_tips(skel, geometry, top_tip_window_um)
    path = union_path(skel, tips)
    distance = geometry["path_distance_to_soma_um"].to_numpy()
    n = skel.n_vertices
    parent = skel.parent_node_array
    child = np.flatnonzero(parent >= 0)
    on_path = np.zeros(n, dtype=bool)
    on_path[path] = True
    # The trunk is a subtree that holds the root, so an edge with both ends on it is a
    # trunk edge. The edge is named by its child, so the cut is at the first trunk
    # vertex past `trunk_start_um`.
    is_path_edge = on_path[child] & on_path[parent[child]]
    keep = ~is_path_edge | (distance[child] < trunk_start_um)
    graph = csr_array((np.ones(keep.sum()), (child[keep], parent[child[keep]])), shape=(n, n))
    _, component = connected_components(graph, directed=False)
    regions = np.where(component == component[skel.root_positional], "basal", "apical")
    regions = regions.astype(object)
    regions[on_path] = "trunk"
    regions[skel.root_positional] = "soma"
    return regions, path, tips
