"""Mask the axon out of a neuron mesh, with the axon label of its skeleton.

The skeleton comes as an ossify `Cell` from the skeleton service: a `graph` layer
of level2 points, linked to a `skeleton` layer with a bool feature (`is_axon`).
A vertex goes only if it is clearly axon, so an unclear vertex stays.

The rule:
1. Each level2 point anchors to its nearest mesh vertex in its own PCG chunk. A
   level2 node never leaves its chunk, so this keeps an anchor off a close
   surface of another branch.
2. A vertex goes if its nearest anchor along the mesh surface is axon, and the
   nearest other anchor is more than `margin_nm` farther. A path along the
   surface from an axon to a close dendrite must go through the real junction.
3. A mesh piece with no anchor uses the nearest points of the skeleton, resampled
   every 250 nm, with `fallback_margin_nm` in place of `margin_nm`.
4. With `vertex_domain`, a domain goes only if at least `domain_frac` of its
   vertices go.

This module imports nothing from `analysis`, so that it can move elsewhere.
All units are nm, and all indices are positional.
"""

from functools import lru_cache

import numpy as np
from meshmash import interpret_mesh, mesh_to_adjacency, subset_mesh_by_indices
from meshmash.types import Mesh
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

# Offset per chunk index, larger than any distance inside a chunk, so that a KD
# tree on the shifted points never matches across chunks.
CHUNK_SHIFT_NM = 1e7
FALLBACK_SPACING_NM = 250.0


@lru_cache
def chunk_grid_from_cave(datastack: str) -> tuple[np.ndarray, np.ndarray]:
    """(origin_nm, size_nm) of the level2 chunk grid of the PCG of `datastack`."""
    from caveclient import CAVEclient

    meta = CAVEclient(datastack).info.segmentation_cloudvolume(progress=False).meta
    resolution = np.asarray(meta.resolution(0), dtype=float)
    size = np.asarray(meta.graph_chunk_size, dtype=float) * resolution
    origin = np.asarray(meta.voxel_offset(0), dtype=float) * resolution
    if not meta.chunks_start_at_voxel_offset:
        origin = np.zeros(3)
    return origin, size


def _chunk_index(xyz, chunk_grid, nudge=0.0):
    origin, size = chunk_grid
    return np.floor((xyz + nudge - origin) / size).astype(np.int64)


def _anchor_level2(vertices, l2_xyz, chunk_grid):
    """The mesh vertex of each level2 point: the nearest one in the same chunk, or
    the nearest one of all if its chunk has no vertex."""
    # NOTE: a mesh vertex on a chunk face is in both chunks, so it goes into the
    # tree once with the chunk of each side.
    tree_xyz = np.concatenate([vertices, vertices])
    tree_chunks = np.concatenate(
        [_chunk_index(vertices, chunk_grid, -1.0), _chunk_index(vertices, chunk_grid, 1.0)]
    )
    tree = cKDTree(tree_xyz + tree_chunks * CHUNK_SHIFT_NM)
    d, i = tree.query(l2_xyz + _chunk_index(l2_xyz, chunk_grid) * CHUNK_SHIFT_NM)
    anchor = i % len(vertices)
    far = d >= CHUNK_SHIFT_NM / 2
    if far.any():
        anchor[far] = cKDTree(vertices).query(l2_xyz[far])[1]
    return anchor


def _surface_distance(graph, sources):
    """Distance along `graph` from each vertex to the nearest of `sources`."""
    if len(sources) == 0:
        return np.full(graph.shape[0], np.inf)
    return dijkstra(graph, directed=False, indices=np.unique(sources), min_only=True)


def _resample_skeleton(xyz, edges, spacing):
    """Points every `spacing` nm along each edge, with the edge end nearer to each
    point. Includes the skeleton vertices."""
    a, b = edges[:, 0], edges[:, 1]
    length = np.linalg.norm(xyz[a] - xyz[b], axis=1)
    n_inner = np.maximum(np.ceil(length / spacing).astype(int) - 1, 0)
    edge = np.repeat(np.arange(len(a)), n_inner)
    start = np.cumsum(n_inner) - n_inner
    t = (np.arange(len(edge)) - np.repeat(start, n_inner) + 1) / np.repeat(n_inner + 1, n_inner)
    points = xyz[a[edge]] * (1 - t)[:, None] + xyz[b[edge]] * t[:, None]
    owner = np.where(t < 0.5, a[edge], b[edge])
    return np.concatenate([xyz, points]), np.concatenate([np.arange(len(xyz)), owner])


def _euclidean_axon(query, skel_xyz, skel_edges, skel_label, margin_nm):
    """True where the nearest skeleton point is axon, and the nearest other point
    is more than `margin_nm` farther."""
    xyz, owner = _resample_skeleton(skel_xyz, skel_edges, FALLBACK_SPACING_NM)
    label = skel_label[owner]
    d_axon = cKDTree(xyz[label]).query(query)[0]
    d_other = cKDTree(xyz[~label]).query(query)[0]
    return d_other - d_axon > margin_nm


def _domain_vote(remove, vertex_domain, frac):
    """A domain goes if at least `frac` of its vertices go. A vertex in no domain
    (-1) keeps its own value."""
    in_domain = vertex_domain >= 0
    _, domain = np.unique(vertex_domain[in_domain], return_inverse=True)
    n_all = np.bincount(domain)
    n_remove = np.bincount(domain, weights=remove[in_domain])
    out = remove.copy()
    out[in_domain] = (n_remove >= frac * n_all)[domain]
    return out


def axon_vertex_mask(
    mesh: Mesh,
    cell,
    *,
    vertex_domain: np.ndarray | None = None,
    chunk_grid: tuple[np.ndarray, np.ndarray] | None = None,
    label: str = "is_axon",
    # NOTE: the scratch evaluation (scratch/axon_mask) ran at 3000 on the ossify
    # mesh graph, which counts most edges twice. That is 1500 nm of surface.
    margin_nm: float = 1500.0,
    domain_frac: float = 0.9,
    fallback_margin_nm: float = 1500.0,
) -> np.ndarray:
    """A bool array, one per mesh vertex: True where the vertex is axon and goes.

    `cell` is an ossify `Cell` with a `graph` layer of level2 points, linked to a
    `skeleton` layer with the bool feature `label`. `vertex_domain` is a domain id
    per vertex, -1 for none. `chunk_grid` is (origin_nm, size_nm) of the level2
    chunks. If it is None, it comes from CAVE for `cell.meta["datastack"]`.
    """
    vertices, faces = interpret_mesh(mesh)
    vertices = np.asarray(vertices, dtype=float)
    skel = cell.skeleton
    skel_label = np.asarray(skel.get_feature(label), dtype=bool)
    if not skel_label.any():
        return np.zeros(len(vertices), dtype=bool)
    if skel_label.all():
        return np.ones(len(vertices), dtype=bool)
    if chunk_grid is None:
        chunk_grid = chunk_grid_from_cave(cell.meta["datastack"])

    l2_xyz = np.asarray(cell.graph.vertices, dtype=float)
    l2_label = skel_label[cell.graph.map_index_to_layer("skeleton", as_positional=True)]
    anchor = _anchor_level2(vertices, l2_xyz, chunk_grid)
    graph = mesh_to_adjacency((vertices, faces))
    d_axon = _surface_distance(graph, anchor[l2_label])
    d_other = _surface_distance(graph, anchor[~l2_label])
    with np.errstate(invalid="ignore"):
        remove = d_other - d_axon > margin_nm

    unreached = np.isinf(d_axon) & np.isinf(d_other)
    if unreached.any():
        skel_xyz = np.asarray(skel.vertices, dtype=float)
        remove[unreached] = _euclidean_axon(
            vertices[unreached], skel_xyz, skel.edges_positional, skel_label, fallback_margin_nm
        )

    if vertex_domain is not None:
        if len(vertex_domain) != len(vertices):
            raise ValueError(f"vertex_domain has {len(vertex_domain)} rows for {len(vertices)} vertices")
        remove = _domain_vote(remove, np.asarray(vertex_domain), domain_frac)
    return remove


def mask_axon(mesh: Mesh, cell, **kwargs) -> tuple[Mesh, np.ndarray]:
    """The mesh without its axon, and the mask of `axon_vertex_mask` (True where a
    vertex went). A face goes if any of its vertices goes. `kwargs` go to
    `axon_vertex_mask`."""
    remove = axon_vertex_mask(mesh, cell, **kwargs)
    return subset_mesh_by_indices(interpret_mesh(mesh), ~remove), remove
