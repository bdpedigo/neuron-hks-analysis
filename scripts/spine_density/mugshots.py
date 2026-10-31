"""A gallery of dendrite "mugshots" per cell type, colored by spine density.

Each panel is the dendrite and soma skeleton of one column cell (axon masked off) in
the post-transform `FRAME`. Each edge has the windowed spine density of its two
vertices, from the `vertices` stash of `spine_density.py`. The horizontal axis is the
first principal axis of the cell in the frame x and z, centered on the soma. The
vertical axis is the height above the soma in the frame, on the same scale for all
panels of a cell type.

    uv run python scripts/spine_density/mugshots.py
"""

# %%
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from analysis import save_matplotlib_figure, set_matplotlib_theme
from analysis.dendrites import (
    EXCITATORY_TYPES,
    column_cells,
    dendrite_skeleton,
    spine_density_stash,
    vertex_geometry,
)
from analysis.plots import add_skeleton

set_matplotlib_theme()

# "streamline" or "affine", see `analysis.dendrites`.
FRAME = "streamline"
# A column of the `vertices` stash: density_r{5,10,20} (all dendrite spines) or
# synaptic_density_r{5,10,20} (synaptic spines only).
DENSITY = "density_r10"
N_PER_TYPE = 50
# The grid has the number of columns that brings the figure closest to this aspect.
FIG_ASPECT = 16 / 9
FIG_WIDTH_IN = 16
# The panel size of a cell type fits this quantile of the cell extents. Larger cells
# draw past the edge of their panel.
EXTENT_QUANTILE = 0.95
SEED = 0
# The color scales to draw a gallery for. "type": one color scale per cell type, with a
# colorbar. "cell": one color scale per cell (1st to 99th percentile), as in the old
# meshrep gallery, with no colorbar.
CLIMS = ["type", "cell"]
CMAP = "coolwarm"
MAX_WORKERS = 8

LABELS = {
    "density": "Spines per µm",
    "synaptic_density": "Synaptic spines per µm",
}


def save(fig, name, bbox="tight"):
    save_matplotlib_figure(
        fig, name, subfolder="spine_density/mugshots", formats=["png"], dpi=300, bbox_inches=bbox
    )


# %% INPUTS
cells = column_cells(EXCITATORY_TYPES)
vertices = (
    spine_density_stash("vertices")
    .load()
    .select("root_id", "skeleton_vertex_id", DENSITY)
    .join(cells, on="root_id", how="semi")
)
cells = cells.join(vertices.select(pl.col("root_id").unique()), on="root_id", how="semi")
sample = pl.concat(
    [
        group.sample(min(N_PER_TYPE, group.height), seed=SEED)
        for (_,), group in cells.partition_by("cell_type", as_dict=True).items()
    ]
)
vertices_by_root = vertices.join(sample, on="root_id", how="semi").partition_by(
    "root_id", as_dict=True
)
print(sample["cell_type"].value_counts().sort("cell_type"))


# %% PER CELL
def mugshot(root_id: int) -> dict:
    """The 2D vertex positions and the edges of the dendrite of `root_id`, the
    density at each vertex, and the depth of the soma in the frame."""
    skel = dendrite_skeleton(root_id)
    geometry = vertex_geometry(skel, FRAME).join(
        vertices_by_root[(root_id,)], on="skeleton_vertex_id", how="left"
    )
    xyz = geometry.select("x_um", "y_um", "z_um").to_numpy()
    soma = xyz[skel.root_positional]
    horizontal = PCA(n_components=1).fit_transform(xyz[:, [0, 2]] - soma[[0, 2]])[:, 0]
    # The PCA sign is arbitrary. Put the bigger half of the arbor on the right.
    if np.median(horizontal) < 0:
        horizontal = -horizontal
    # The vertical axis is down = deeper, as the frame y.
    xy = np.column_stack(
        [horizontal - horizontal[skel.root_positional], -geometry["height_above_soma_um"].to_numpy()]
    )
    return {
        "edges": skel.edges_positional,
        "xy": xy,
        "values": geometry[DENSITY].to_numpy(),
        "soma_depth_um": soma[1],
    }


root_ids = sample["root_id"].to_list()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    shots = dict(zip(root_ids, pool.map(mugshot, root_ids)))


# %% ONE GALLERY PER CELL TYPE
def grid_shape(n: int, panel_aspect: float) -> tuple[int, int]:
    """The rows and columns for `n` panels that bring the figure closest to
    `FIG_ASPECT`."""
    best = min(
        range(1, n + 1),
        key=lambda cols: abs(np.log(cols * panel_aspect / np.ceil(n / cols) / FIG_ASPECT)),
    )
    return int(np.ceil(n / best)), best


# The page area of the "type" gallery of each cell type. The "cell" gallery reuses it, so
# that it keeps the space of the colorbar and the two versions line up.
page_areas = {}
for CLIM, cell_type in [(c, t) for c in CLIMS for t in EXCITATORY_TYPES]:
    group = sample.filter(pl.col("cell_type") == cell_type)
    if group.height == 0:
        continue
    # Sort the panels by the depth of the soma.
    group_ids = sorted(group["root_id"].to_list(), key=lambda r: shots[r]["soma_depth_um"])
    group_shots = [shots[r] for r in group_ids]
    points = [s["xy"] for s in group_shots]
    half_width = np.quantile([np.abs(xy[:, 0]).max() for xy in points], EXTENT_QUANTILE)
    up = np.quantile([-xy[:, 1].min() for xy in points], EXTENT_QUANTILE)
    down = np.quantile([xy[:, 1].max() for xy in points], EXTENT_QUANTILE)
    values = np.concatenate([s["values"] for s in group_shots])
    type_clim = (0, np.nanquantile(values, 0.99))

    n_rows, n_cols = grid_shape(len(group_ids), 2 * half_width / (up + down))
    panel_w = FIG_WIDTH_IN / n_cols
    panel_h = panel_w * (up + down) / (2 * half_width)
    title_in = 0.4
    fig_h = n_rows * panel_h + title_in
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH_IN, fig_h), squeeze=False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - title_in / fig_h, wspace=0, hspace=0)
    for ax in axs.flat:
        ax.axis("off")
    for ax, root_id, shot, xy in zip(axs.flat, group_ids, group_shots, points):
        if CLIM == "type":
            clim = type_clim
        else:
            clim = tuple(np.nanquantile(shot["values"], [0.01, 0.99]))
        lines = add_skeleton(
            ax, shot["edges"], shot["xy"], shot["values"], cmap=CMAP, clim=clim, clip_on=False
        )
        ax.set(xlim=(-half_width, half_width), ylim=(down, -up), aspect="equal")
        # The label sits just below the lowest point of the cell.
        bottom = min(xy[:, 1].max(), down)
        ax.text(0, bottom, str(root_id)[-6:], ha="center", va="top", fontsize=4, color="0.5", clip_on=False)
    radius = DENSITY.split("_r")[-1]
    label = f"{LABELS[DENSITY.rsplit('_r', 1)[0]]} (r = {radius} µm)"
    if CLIM == "type":
        cax = fig.add_axes([1.005, 0.25, 0.006, 0.5])
        fig.colorbar(lines, cax=cax, label=label)
    scale = "" if CLIM == "type" else ", normalized within each cell"
    fig.suptitle(f"{cell_type} (n = {len(group_ids)}), {label}{scale}, {FRAME} frame", y=1 - 0.05 / fig_h, va="top", fontsize="medium")
    if CLIM == "type":
        fig.canvas.draw()
        page_areas[cell_type] = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    save(fig, f"mugshots_{cell_type}_{DENSITY}_{CLIM}_{FRAME}", bbox=page_areas.get(cell_type, "tight"))
    plt.close(fig)
    print(f"{CLIM} {cell_type}: {n_rows} x {n_cols} grid, type color scale {type_clim[1]:.2f} spines/µm")
