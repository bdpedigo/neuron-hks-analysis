"""Spine density against path distance, per dendrite region and cell type.

One column per excitatory cell type. The top row is the trunk and the apical branches
(obliques and tuft), the bottom row the basal branches, from
`analysis.dendrites.dendrite_regions`. Each cell is a tree of line segments: one
segment per skeleton edge, from the child vertex to its parent, at the windowed spine
density of each vertex (`DENSITY`, from the `vertices` stash of `spine_density.py`).
So a cell is one line that branches where its dendrite branches. The black line is the
cable-weighted mean of the cells in `BIN_UM` bins, for bins with `MIN_BIN_CELLS` cells.

NOTE: the density window is a path radius around a vertex, and it does not stop at a
region border. So near the soma, the window of a basal vertex can count spines on the
trunk, and the reverse.

    uv run python scripts/spine_density/region_density.py
"""

# %%
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from tqdm.auto import tqdm

from analysis import CELL_TYPE_PALETTE, save_matplotlib_figure, set_matplotlib_theme
from analysis.dendrites import (
    EXCITATORY_TYPES,
    column_cells,
    dendrite_regions,
    dendrite_skeleton,
    spine_density_stash,
    vertex_geometry,
)

set_matplotlib_theme()

# "streamline" or "affine", see `analysis.dendrites`. It sets the top tips.
FRAME = "streamline"
# The settings of `dendrite_regions`.
TOP_TIP_WINDOW_UM = 25
TRUNK_START_UM = 25
# A column of the `vertices` stash: density_r{5,10,20} (all dendrite spines) or
# synaptic_density_r{5,10,20} (synaptic spines only).
DENSITY = "density_r20"
BIN_UM = 20
# A bin of the mean line needs cable from at least this many cells.
MIN_BIN_CELLS = 3
# The alpha of a cell line is LINE_ALPHA_SCALE / (cells of its type), within LINE_ALPHA.
LINE_ALPHA_SCALE = 3
LINE_ALPHA = (0.01, 0.3)
MAX_WORKERS = 8

ROWS = {"Trunk and apical": ["trunk", "apical"], "Basal": ["basal"]}
LABELS = {"density": "Spines per µm", "synaptic_density": "Synaptic spines per µm"}


def save(fig, name):
    save_matplotlib_figure(fig, name, subfolder="spine_density", formats=["png"], dpi=200)


# %% INPUTS
cells = column_cells(EXCITATORY_TYPES)
vertices = (
    spine_density_stash("vertices")
    .load()
    .select("root_id", "skeleton_vertex_id", DENSITY)
    .join(cells, on="root_id", how="semi")
)
cells = cells.join(vertices.select(pl.col("root_id").unique()), on="root_id", how="semi")
vertices_by_root = vertices.partition_by("root_id", as_dict=True)
print(cells["cell_type"].value_counts().sort("cell_type"))


# %% PER CELL
def cell_lines(root_id: int) -> pl.DataFrame:
    """One row per edge of the dendrite of `root_id` (named by its child vertex), with
    the region, path distance, density and cable of the child and its parent."""
    skel = dendrite_skeleton(root_id)
    geometry = vertex_geometry(skel, FRAME).join(
        vertices_by_root[(root_id,)], on="skeleton_vertex_id", how="left"
    )
    regions, _, _ = dendrite_regions(
        skel, geometry, top_tip_window_um=TOP_TIP_WINDOW_UM, trunk_start_um=TRUNK_START_UM
    )
    parent = skel.parent_node_array
    child = np.flatnonzero(parent >= 0)
    distance = geometry["path_distance_to_soma_um"].to_numpy()
    density = geometry[DENSITY].to_numpy()
    return pl.DataFrame(
        {
            "root_id": np.full(len(child), root_id, dtype=np.int64),
            "region": regions[child].astype(str),
            "distance_um": distance[child],
            "parent_distance_um": distance[parent[child]],
            "density": density[child],
            "parent_density": density[parent[child]],
            "cable_um": geometry["half_edge_length_um"].to_numpy()[child],
        }
    )


root_ids = cells["root_id"].to_list()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    lines = pl.concat(list(tqdm(pool.map(cell_lines, root_ids), total=len(root_ids))))
lines = lines.join(cells, on="root_id")

# %% THE CABLE-WEIGHTED MEAN PER BIN
means = (
    lines.with_columns(
        row=pl.col("region").replace_strict(
            {region: row for row, regions in ROWS.items() for region in regions}
        ),
        bin_um=pl.col("distance_um") // BIN_UM * BIN_UM + BIN_UM / 2,
    )
    .group_by("cell_type", "row", "bin_um")
    .agg(
        n_cells=pl.col("root_id").n_unique(),
        density=(pl.col("density") * pl.col("cable_um")).sum() / pl.col("cable_um").sum(),
    )
    .filter(pl.col("n_cells") >= MIN_BIN_CELLS)
    .sort("cell_type", "row", "bin_um")
)

# %% THE FIGURE
ymax = lines["density"].quantile(0.995)
fig, axs = plt.subplots(
    len(ROWS),
    len(EXCITATORY_TYPES),
    figsize=(3 * len(EXCITATORY_TYPES), 3.2 * len(ROWS)),
    sharey=True,
    sharex="col",
    layout="constrained",
    squeeze=False,
)
for col, cell_type in enumerate(EXCITATORY_TYPES):
    of_type = lines.filter(pl.col("cell_type") == cell_type)
    n_cells = of_type["root_id"].n_unique()
    alpha = float(np.clip(LINE_ALPHA_SCALE / n_cells, *LINE_ALPHA))
    for row, (row_label, regions) in enumerate(ROWS.items()):
        ax = axs[row, col]
        data = of_type.filter(pl.col("region").is_in(regions))
        segments = np.stack(
            [
                data.select("distance_um", "density").to_numpy(),
                data.select("parent_distance_um", "parent_density").to_numpy(),
            ],
            axis=1,
        )
        ax.add_collection(
            LineCollection(
                segments,
                colors=CELL_TYPE_PALETTE[cell_type],
                linewidths=0.4,
                alpha=alpha,
                rasterized=True,
            )
        )
        mean = means.filter((pl.col("cell_type") == cell_type) & (pl.col("row") == row_label))
        ax.plot(mean["bin_um"], mean["density"], color="black", lw=1.5)
        ax.autoscale(axis="x")
        ax.set_xlim(left=0)
        ax.set_ylim(0, ymax)
        if row == 0:
            ax.set_title(f"{cell_type} (n = {n_cells})")
        if col == 0:
            ax.set_ylabel(f"{row_label}\n{LABELS[DENSITY.rsplit('_r', 1)[0]]}")
fig.supxlabel("Path distance to soma (µm)")
radius = DENSITY.rsplit("_r", 1)[1]
fig.suptitle(
    f"Spine density by region (r = {radius} µm). Top tips within {TOP_TIP_WINDOW_UM} µm, "
    f"trunk from {TRUNK_START_UM} µm, {FRAME} frame"
)
save(fig, f"region_density_{DENSITY}_top{TOP_TIP_WINDOW_UM}um_trunk{TRUNK_START_UM}um_{FRAME}")
