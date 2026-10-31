"""Apical and basal regions of the dendrite, from the topological rule of
`analysis.dendrites.dendrite_regions`.

The figure has one column per excitatory cell type, with one sampled cell each. Row 1
is the whole dendrite with the trunk in bold and a dot on each top tip, row 2 the trunk
and the apical branches, and row 3 the basal branches. Each run samples new cells, unless `SEED` is set.

    uv run python scripts/spine_density/dendrite_regions.py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis import CELL_TYPE_PALETTE, save_matplotlib_figure, set_matplotlib_theme
from analysis.dendrites import (
    EXCITATORY_TYPES,
    REGIONS,
    column_cells,
    dendrite_regions,
    dendrite_skeleton,
    vertex_geometry,
)
from analysis.plots import add_skeleton
from analysis.products.skeleton import SKELETON_CACHE

set_matplotlib_theme()

# "streamline" or "affine", see `analysis.dendrites`.
FRAME = "streamline"
# The settings of `dendrite_regions`.
TOP_TIP_WINDOW_UM = 25
TRUNK_START_UM = 25
# None draws a new sample on each run. Set it to the printed seed to redraw a sample.
SEED = None
CONTEXT_COLOR = "0.88"


def save(fig, name):
    save_matplotlib_figure(
        fig, name, subfolder="spine_density/dendrite_regions", formats=["png"], dpi=200
    )


def cell_regions(root_id: int) -> dict:
    """The 2D positions (soma at the origin, pia up), the edges with the region of
    their child vertex, the trunk edges and the top tips of `root_id`."""
    skel = dendrite_skeleton(root_id)
    geometry = vertex_geometry(skel, FRAME)
    regions, path, tips = dendrite_regions(
        skel, geometry, top_tip_window_um=TOP_TIP_WINDOW_UM, trunk_start_um=TRUNK_START_UM
    )
    soma_x = geometry["x_um"].to_numpy()[skel.root_positional]
    xy = np.column_stack(
        [geometry["x_um"].to_numpy() - soma_x, geometry["height_above_soma_um"].to_numpy()]
    )
    parent = skel.parent_node_array
    child = np.flatnonzero(parent >= 0)
    cable = geometry["half_edge_length_um"].to_numpy()
    return {
        "xy": xy,
        "edges": np.column_stack([child, parent[child]]),
        "edge_regions": regions[child],
        "is_trunk_edge": np.isin(child, path) & np.isin(parent[child], path),
        "tips": tips,
        "cable_um": {r: cable[regions == r].sum() for r in REGIONS},
    }


# %% SAMPLE ONE CELL PER TYPE
seed = SEED if SEED is not None else int(np.random.SeedSequence().generate_state(1)[0])
cached = pl.DataFrame({"root_id": [int(p.stem) for p in SKELETON_CACHE.glob("*.osy")]})
cells = column_cells(EXCITATORY_TYPES).join(cached, on="root_id", how="semi")
sample = {
    cell_type: cells.filter(pl.col("cell_type") == cell_type)
    .sample(1, seed=seed)["root_id"]
    .item()
    for cell_type in EXCITATORY_TYPES
}
print(f"seed {seed}")
shots = {cell_type: cell_regions(root_id) for cell_type, root_id in sample.items()}
print(
    pl.DataFrame(
        [
            {"cell_type": t, "root_id": sample[t], **{k: round(v) for k, v in s["cable_um"].items()}}
            for t, s in shots.items()
        ]
    )
)

# %% THE FIGURE
ROWS = [
    ("Whole dendrite", None),
    ("Trunk and apical", {"trunk", "apical"}),
    ("Basal", {"basal"}),
]
fig, axs = plt.subplots(
    len(ROWS),
    len(EXCITATORY_TYPES),
    figsize=(2.6 * len(EXCITATORY_TYPES), 3.6 * len(ROWS)),
    sharex="col",
    sharey="col",
    layout="constrained",
)
for col, cell_type in enumerate(EXCITATORY_TYPES):
    shot = shots[cell_type]
    color = CELL_TYPE_PALETTE[cell_type]
    xy, edges, edge_regions = shot["xy"], shot["edges"], shot["edge_regions"]
    for row, (label, keep) in enumerate(ROWS):
        ax = axs[row, col]
        if keep is None:
            add_skeleton(ax, edges[~shot["is_trunk_edge"]], xy, color="0.55", linewidths=0.5)
            add_skeleton(ax, edges[shot["is_trunk_edge"]], xy, color=color, linewidths=2)
            ax.scatter(
                *xy[shot["tips"]].T, s=40, color=color, edgecolor="black", lw=0.6, zorder=10
            )
        else:
            mask = np.isin(edge_regions, list(keep))
            add_skeleton(ax, edges[~mask], xy, color=CONTEXT_COLOR, linewidths=0.4)
            add_skeleton(ax, edges[mask], xy, color=color, linewidths=0.6)
            if "trunk" in keep:
                add_skeleton(ax, edges[shot["is_trunk_edge"]], xy, color=color, linewidths=1.5)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")
        if row == 0:
            n_tips = len(shot["tips"])
            ax.set_title(
                f"{cell_type}\n{sample[cell_type]}\n{n_tips} top tip{'s' if n_tips > 1 else ''}",
                fontsize="small",
            )
        if col == 0:
            ax.text(
                -0.08, 0.5, label, rotation=90, ha="right", va="center", transform=ax.transAxes
            )
fig.suptitle(
    f"Dendrite regions: soma to the tips within {TOP_TIP_WINDOW_UM} µm of the highest, "
    f"trunk from {TRUNK_START_UM} µm, "
    f"{FRAME} frame (seed {seed})",
    fontsize="medium",
)
save(fig, f"dendrite_regions_top{TOP_TIP_WINDOW_UM}um_trunk{TRUNK_START_UM}um_{FRAME}_{seed}")
