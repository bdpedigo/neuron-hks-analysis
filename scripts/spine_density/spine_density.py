"""Spines per µm along the skeletons of the column cells (R2.3).

It writes two stashes (`analysis.dendrites.spine_density_stash`):

- `spines`: one row per spine structure, with the skeleton vertex it sits on.
- `vertices`: one row per dendrite or soma vertex, with its cable and the spine
  density in a window of path radius `r` around the vertex.

A synaptic spine (a structure with a synapse in the skeleton file) sits on the vertex
of its synapse that is nearest the root. Any other spine sits on the vertex of the
level-2 node nearest its representative point. The dendrite is the `dendrite_mask` of
the skeleton, which keeps the axon vertices on the path from the dendrite to the root.
Spines on a vertex outside it (`is_axon_vertex`) are kept in `spines` but do not count
in the densities. All lengths are native.

The stashes hold no distances or heights. The plotting scripts get those from
`analysis.dendrites.vertex_geometry`, so that the geometry can change without a rerun.

    uv run python scripts/spine_density/spine_density.py
"""

# %%
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from analysis import Tables
from analysis.dendrites import RADII_UM, spine_density_stash
from analysis.products.skeleton import cached_skeleton, dendrite_mask

# Set to an int to run only the first few cells. The stashes are written only for a
# full run.
ROOT_LIMIT = None
MAX_WORKERS = 8

# %% INPUTS
spine_table = (
    Tables["column_spines"]
    .lazy()
    .select(
        "root_id",
        "structure_id",
        "rep_x",
        "rep_y",
        "rep_z",
        "unitary_pred_label",
        "posterior_unitary",
    )
    .collect()
)
synapse_structures = (
    Tables["column_post_synapses"]
    .lazy()
    .filter(pl.col("structure_id").is_not_null())
    .select(pl.col("post_pt_root_id").alias("root_id"), "synapse_id", "structure_id")
    .collect()
)
spines_by_root = spine_table.partition_by("root_id", as_dict=True)
synapses_by_root = synapse_structures.partition_by("root_id", as_dict=True)
root_ids = sorted(root for (root,) in spines_by_root)
if ROOT_LIMIT is not None:
    root_ids = root_ids[:ROOT_LIMIT]
print(f"{len(root_ids)} roots, {spine_table.height:,} spines")


# %% PER CELL
def place_spines(cell, spines: pl.DataFrame, synapses: pl.DataFrame) -> pl.DataFrame:
    """`spines` with the positional skeleton vertex of each spine and the method.

    The skeleton must have `vertex_index == arange(n)`, so that a position is also
    the vertex id.
    """
    skel = cell.skeleton
    d2r = skel.distance_to_root(as_positional=True)

    post = cell.annotations["post_syn"]
    synapse_vertex = post.map_index_to_layer("skeleton", as_positional=True)
    synapse_vertex = pl.DataFrame(
        {
            "synapse_id": np.asarray(post.vertex_df.index, dtype=np.int64),
            "vertex": synapse_vertex.astype(np.int64),
            "d2r": d2r[synapse_vertex],
        }
    )
    by_synapse = (
        synapses.join(synapse_vertex, on="synapse_id", how="inner")
        .group_by("structure_id")
        .agg(pl.col("vertex").sort_by("d2r").first())
    )

    points = spines.select("rep_x", "rep_y", "rep_z").to_numpy()
    graph_to_skeleton = cell.graph.map_index_to_layer("skeleton", as_positional=True)
    l2_distance, nearest = cKDTree(cell.graph.vertices).query(points)

    placed = (
        spines.with_columns(
            l2_vertex=graph_to_skeleton[nearest].astype(np.int64),
            l2_distance_nm=l2_distance.astype(np.float32),
        )
        .join(by_synapse, on="structure_id", how="left")
        .with_columns(
            method=pl.when(pl.col("vertex").is_not_null())
            .then(pl.lit("synapse"))
            .otherwise(pl.lit("nearest_l2")),
            vertex=pl.coalesce("vertex", "l2_vertex"),
        )
    )
    vertex = placed["vertex"].to_numpy()
    return placed.with_columns(is_axon_vertex=~dendrite_mask(skel)[vertex])


def window_densities(cell, placed: pl.DataFrame) -> pl.DataFrame:
    """One row per dendrite or soma vertex, with the spine counts and the windowed
    densities (spines/µm) of all spines and of the synaptic spines."""
    skel = cell.skeleton
    dendrite = cell.apply_mask("skeleton", dendrite_mask(skel)).skeleton
    ids = np.asarray(dendrite.vertex_index)

    # Per-vertex arrays indexed by vertex id. The half-edge lengths are those of
    # the masked skeleton, so a vertex next to the axon counts only its dendrite
    # edges.
    n = skel.n_vertices
    cable = np.zeros(n)
    cable[ids] = dendrite.half_edge_length
    on_dendrite = placed.filter(~pl.col("is_axon_vertex"))
    counts = {
        "spines": np.bincount(on_dendrite["vertex"].to_numpy(), minlength=n),
        "synaptic_spines": np.bincount(
            on_dendrite.filter(pl.col("method") == "synapse")["vertex"].to_numpy(), minlength=n
        ),
    }

    columns = {
        "skeleton_vertex_id": ids,
        "half_edge_length_um": cable[ids] / 1000,
        "n_spines": counts["spines"][ids],
        "n_synaptic_spines": counts["synaptic_spines"][ids],
    }
    for r in RADII_UM:
        pairs = dendrite.proximity_mapping(distance_threshold=r * 1000)
        center, member = pairs["idx"].to_numpy(), pairs["prox_idx"].to_numpy()
        window_um = np.bincount(center, weights=cable[member], minlength=n)[ids] / 1000
        for name, count in counts.items():
            window_n = np.bincount(center, weights=count[member], minlength=n)[ids]
            prefix = "" if name == "spines" else "synaptic_"
            columns[f"{prefix}density_r{r}"] = window_n / window_um
        columns[f"window_um_r{r}"] = window_um
    return pl.DataFrame(columns)


def run_root(root_id: int) -> tuple[pl.DataFrame, pl.DataFrame, float] | None:
    start = time.perf_counter()
    try:
        cell = cached_skeleton(root_id)
    except FileNotFoundError:
        return None
    skel = cell.skeleton
    assert np.array_equal(skel.vertex_index, np.arange(skel.n_vertices)), root_id
    empty = synapse_structures.clear()
    placed = place_spines(
        cell, spines_by_root[(root_id,)], synapses_by_root.get((root_id,), empty)
    )
    vertices = window_densities(cell, placed).with_columns(root_id=pl.lit(root_id, pl.Int64))
    spines = placed.select(
        "root_id",
        "structure_id",
        pl.col("vertex").alias("skeleton_vertex_id"),
        "method",
        "is_axon_vertex",
        pl.col("l2_vertex").alias("l2_vertex_id"),
        "l2_distance_nm",
        "unitary_pred_label",
        "posterior_unitary",
    )
    return spines, vertices, time.perf_counter() - start


# %% RUN
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    results = dict(zip(root_ids, tqdm(pool.map(run_root, root_ids), total=len(root_ids))))
wall = time.perf_counter() - start

missing = [root for root, result in results.items() if result is None]
done = {root: result for root, result in results.items() if result is not None}
spines = pl.concat([result[0] for result in done.values()])
vertices = pl.concat([result[1] for result in done.values()]).select(
    "root_id", pl.exclude("root_id")
)
seconds = pl.Series("seconds", [result[2] for result in done.values()])
print(f"no skeleton: {missing}")
print(f"wall {wall:.0f} s for {len(done)} roots with {MAX_WORKERS} threads")
print(seconds.describe())

# %% CHECK: THE CABLE-WEIGHTED MEAN OF THE WINDOWED DENSITY IS CLOSE TO SPINES / CABLE
check = vertices.group_by("root_id").agg(
    ratio=pl.col("n_spines").sum() / pl.col("half_edge_length_um").sum(),
    **{
        f"weighted_r{r}": (pl.col(f"density_r{r}") * pl.col("half_edge_length_um")).sum()
        / pl.col("half_edge_length_um").sum()
        for r in RADII_UM
    },
)
print(
    check.select(
        **{
            f"median_rel_diff_r{r}": ((pl.col(f"weighted_r{r}") / pl.col("ratio")) - 1).abs().median()
            for r in RADII_UM
        }
    )
)

# %% SAVE
if ROOT_LIMIT is None:
    for name, table in [("spines", spines), ("vertices", vertices)]:
        stash = spine_density_stash(name)
        stash.save(table)
        print(f"{name}: {stash.path}, {stash.path.stat().st_size / 1e6:.1f} MB")
