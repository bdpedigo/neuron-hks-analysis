"""P8: skeleton features for each synapse, from the ossify skeletons (S5)."""

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import ossify as osy
import polars as pl
from ossify.algorithms import subtree_aggregate
from tqdm.auto import tqdm

from ..io import DATA_PATH, FROZEN_PATH
from ..sources import SKELETON_PATH, VERSION, VERSION_TIMESTAMP
from . import cell_selection  # noqa: F401 -- registers "cell_selection", whose cache key P8 reads
from . import nuclei  # noqa: F401 -- registers "root_nuclei", whose cache key P8 reads
from .catalog import Tables
from .features import add_distance_to_nucleus
from .partitions import hash_partition

SIDES = {"pre": "pre_syn", "post": "post_syn"}
COMPARTMENTS = ["axon", "dendrite", "soma"]

# A tip counts as a real tip if no branch point is within this many hops above it.
REAL_TIP_MIN_HOPS = 3
# Vertices with no real tip below them take the value of their parent, for up to
# this many rounds.
TIP_FILL_ROUNDS = 4

# A local copy of the skeleton files that `cached_skeleton` reads.
SKELETON_CACHE = DATA_PATH / "skeletons"


def _check_version(cell: osy.Cell, root_id: int) -> osy.Cell:
    timestamp = datetime.fromisoformat(cell.meta["timestamp"])
    assert abs((timestamp - VERSION_TIMESTAMP).total_seconds()) < 1, (
        f"skeleton of {root_id} is at {timestamp}, not at v{VERSION}"
    )
    return cell


def load_skeleton(root_id: int) -> osy.Cell:
    cell = osy.load_cell(str(SKELETON_PATH / f"{root_id}.osy"))
    return _check_version(cell, root_id)


def cached_skeleton(root_id: int) -> osy.Cell:
    """The skeleton of `root_id`, from a local copy of the file in `SKELETON_PATH`.
    The first call copies the file to `SKELETON_CACHE`."""
    path = SKELETON_CACHE / f"{root_id}.osy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes((SKELETON_PATH / f"{root_id}.osy").read_bytes())
        tmp.rename(path)
    return _check_version(osy.load_cell(str(path)), root_id)


def dendrite_mask(skel) -> np.ndarray:
    """Per skeleton vertex (positional), True for the dendrite and soma.

    These are the vertices that are not axon, and also the axon vertices on the path
    from a vertex that is not axon to the root. So the mask is one connected piece
    that holds the root, and a dendrite that leaves from a mislabeled stretch of axon
    stays attached to the soma.
    """
    parent = skel.parent_node_array
    mask = ~skel.get_feature("is_axon").astype(bool)
    frontier = np.unique(parent[mask & (parent >= 0)])
    frontier = frontier[~mask[frontier]]
    while len(frontier):
        mask[frontier] = True
        frontier = np.unique(parent[frontier])
        frontier = frontier[(frontier >= 0)]
        frontier = frontier[~mask[frontier]]
    assert mask[skel.root_positional], "the root is not in the dendrite mask"
    return mask


def max_dist_to_tip(skel) -> np.ndarray:
    """Per skeleton vertex (positional), the path distance (nm) to the farthest
    real tip below it, counting only tips that are not axon.

    The axon vertices themselves are not masked, so their values are not
    meaningful. Read the value only for dendrite and soma vertices.
    """
    # NOTE: the old code (meshrep tools/pull_csm_skeletons_more_dists.py) removed
    # the axon before it found the tips. There, the point where an axon leaves a
    # dendrite was not a branch point. Here the topology is kept, and only the
    # axon tips are excluded. The effect on the real-tip test is small.
    parent = skel.parent_node_array
    is_branch = np.zeros(skel.n_vertices, dtype=bool)
    is_branch[skel.branch_points_positional] = True
    is_axon = skel.get_feature("is_axon").astype(bool)

    tips = skel.end_points_positional
    tips = tips[~is_axon[tips]]
    real = np.ones(len(tips), dtype=bool)
    above = tips.copy()
    for _ in range(REAL_TIP_MIN_HOPS):
        above = np.where(above >= 0, parent[np.maximum(above, 0)], -1)
        real &= ~((above >= 0) & is_branch[np.maximum(above, 0)])
    real_tips = tips[real]

    d2r = skel.distance_to_root(as_positional=True)
    # NOTE: -inf, not NaN, because the "max" aggregate does not carry NaN upward
    tip_d2r = np.full(skel.n_vertices, -np.inf)
    tip_d2r[real_tips] = d2r[real_tips]
    farthest = subtree_aggregate(skel, tip_d2r, "max")
    for _ in range(TIP_FILL_ROUNDS):
        missing = np.isneginf(farthest) & (parent >= 0)
        if not missing.any():
            break
        farthest[missing] = farthest[parent[missing]]
    farthest[np.isneginf(farthest)] = np.nan
    return farthest - d2r


def skeleton_synapse_features(cell: osy.Cell) -> pl.DataFrame:
    """One row per synapse and side of `cell`, with its skeleton features.

    The compartment comes from the skeleton vertex that the synapse maps to:
    axon if the vertex is axon, soma if it is the root, and dendrite otherwise.
    """
    skel = cell.skeleton
    is_axon = skel.get_feature("is_axon").astype(bool)
    tip_dists = max_dist_to_tip(skel)
    root_id = int(cell.meta["root_id"])
    frames = []
    for side, layer_name in SIDES.items():
        layer = cell.annotations[layer_name]
        verts = layer.map_index_to_layer("skeleton", as_positional=True)
        on_axon = is_axon[verts]
        compartment = np.where(
            on_axon, "axon", np.where(verts == skel.root_positional, "soma", "dendrite")
        )
        tip = tip_dists[verts] if side == "post" else np.full(len(verts), np.nan)
        tip = np.where(on_axon, np.nan, tip)
        positions = layer.vertex_df
        frames.append(
            pl.DataFrame(
                {
                    "synapse_id": np.asarray(positions.index, dtype=np.int64),
                    "root_id": np.full(len(verts), root_id, dtype=np.int64),
                    "side": np.full(len(verts), side),
                    "compartment": compartment,
                    "path_distance_to_nuc": layer.distance_to_root(),
                    "max_dist_to_tip": tip,
                    **{
                        f"ctr_pt_position_{c}": positions[f"ctr_pt_position_{c}"].to_numpy()
                        for c in "xyz"
                    },
                }
            )
        )
    return pl.concat(frames).with_columns(
        pl.col("side").cast(pl.Enum(list(SIDES))),
        pl.col("compartment").cast(pl.Enum(COMPARTMENTS)),
        pl.col("path_distance_to_nuc", "max_dist_to_tip").fill_nan(None),
    )


def _root_features(root_id: int) -> pl.DataFrame | None:
    try:
        cell = load_skeleton(root_id)
    except FileNotFoundError:
        return None
    return skeleton_synapse_features(cell)


def _write_part(root_ids: list[int], part: Path, nuclei: pl.LazyFrame, max_workers: int) -> list[int]:
    """Write the features of `root_ids` to `part`, and return the roots with no
    skeleton file."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = dict(zip(root_ids, pool.map(_root_features, root_ids)))
    missing = [root_id for root_id, frame in frames.items() if frame is None]
    features = pl.concat([frame for frame in frames.values() if frame is not None])
    features = add_distance_to_nucleus(
        features.lazy(),
        nuclei,
        position=("ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"),
        name="euc_distance_to_nuc",
    ).collect()
    tmp = part.with_suffix(".tmp")
    features.write_parquet(tmp, compression="snappy")
    tmp.rename(part)
    return missing


def write_skeleton_features(
    root_ids: list[int], out: Path, *, n_chunks: int = 32, max_workers: int = 8
) -> None:
    """Write `skeleton_synapse_features` for every root in `root_ids` to `out`.

    Each chunk of roots is written as a part, and the parts are combined into
    `out` at the end. An interrupted run resumes from the parts it finished.
    """
    nuclei = Tables["root_nuclei"].lazy()
    chunks: dict[int, list[int]] = {}
    for root_id in root_ids:
        chunks.setdefault(hash_partition(root_id, n_chunks), []).append(root_id)
    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    todo = [
        (chunk_roots, partial / f"part-{chunk:04d}.parquet")
        for chunk, chunk_roots in sorted(chunks.items())
    ]
    todo = [(chunk_roots, part) for chunk_roots, part in todo if not part.exists()]
    missing = []
    for chunk_roots, part in tqdm(todo):
        missing += _write_part(chunk_roots, part, nuclei, max_workers)
    if missing:
        print(f"skeleton_synapse_features: {len(missing)} roots have no skeleton: {missing}")
    tmp = out.with_name(out.name + ".tmp")
    pl.scan_parquet(partial / "*.parquet").sink_parquet(tmp, compression="snappy")
    tmp.rename(out)
    shutil.rmtree(partial)


def _proofread_roots() -> list[int]:
    selection = Tables["cell_selection"].lazy()
    return selection.filter(pl.col("is_proofread")).collect()["pt_root_id"].to_list()


Tables.register(
    "skeleton_synapse_features",
    description=(
        "One row per synapse and side (`pre` or `post`) on each proofread root, from "
        "the skeleton files. It holds the compartment (axon, dendrite or soma), the "
        "path distance to the nucleus, the distance to the farthest tip and the "
        "straight distance to the nearest nucleus, all in nm."
    ),
    materialize=lambda path: write_skeleton_features(_proofread_roots(), path),
    cache_params=lambda: {
        "version": VERSION,
        "cell_selection": Tables["cell_selection"].cache_key,
        "root_nuclei": Tables["root_nuclei"].cache_key,
        "skeleton_path": str(SKELETON_PATH),
        "real_tip_min_hops": REAL_TIP_MIN_HOPS,
        "tip_fill_rounds": TIP_FILL_ROUNDS,
    },
)


# A copy of gs://csm-skel/cellpax/minnie65_v1822/tables/processed/table--r122.parquet.
CELLPAX_FEATURES_PATH = FROZEN_PATH / "minnie65_v1822_cellpax_r122.parquet"

Tables.register(
    "cellpax_features",
    description=(
        "One row per root at v1822, with about 160 skeleton and synapse features per "
        "cell from the cellpax tables (gs://csm-skel/cellpax, r122). "
        "`dendrite_cable_length_um` is the dendrite cable with the axon masked off. It "
        "is 0.988 of the `dendrite_mask` cable on the column cells. About 1% of the "
        "neurons have no row or a null cable."
    ),
    build=lambda: pl.scan_parquet(CELLPAX_FEATURES_PATH),
    frozen=CELLPAX_FEATURES_PATH,
)
