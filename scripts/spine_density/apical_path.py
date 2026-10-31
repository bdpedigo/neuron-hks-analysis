"""Spines per µm along the apical path of the excitatory column cells.

A comparison with Egashira et al. 2026 (Sci. Adv. 12, eadw8458), who map spines along
the apical trunk of L5 ET cells. For each cell, the path goes from the soma to one
dendrite tip (`TIP_RULE`). The spine counts come from the `vertices` stash of
`spine_density.py`. Path lengths are native. The tip choice, the height and the
example figure use the post-transform `FRAME`.

    uv run python scripts/spine_density/apical_path.py
"""

# %%
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
from tqdm.auto import tqdm

from analysis import CELL_TYPE_PALETTE, save_matplotlib_figure, set_matplotlib_theme
from analysis.dendrites import (
    EXCITATORY_TYPES,
    column_cells,
    dendrite_skeleton,
    spine_density_stash,
    tip_path,
    vertex_geometry,
)
from analysis.plots import add_skeleton

set_matplotlib_theme()

# "streamline" or "affine", see `analysis.dendrites`.
FRAME = "streamline"
# "highest": the dendrite tip nearest the pia in `FRAME`. "farthest": the dendrite tip
# with the longest native path to the soma.
TIP_RULE = "highest"
BIN_UM = 20
# "path": path distance to the soma. "depth": height above the soma in `FRAME`.
X = "path"
# Set to an int to run only the first few cells.
ROOT_LIMIT = None
MAX_WORKERS = 8

# The paper's "middle" and "distal" compartments of the apical dendrite (µm from soma).
MIDDLE_UM = (200, 500)
DISTAL_UM = 500
# The path distances (µm) over which the radius of the apical dendrite near the soma is
# the median of the skeleton radius. The soma itself inflates the radius up to ~20 µm.
RADIUS_WINDOW_UM = (20, 50)

SPINE_SETS = {
    "n_synaptic_spines": "Synaptic spines (lower bound)",
    "n_spines": "All dendrite spines (upper bound)",
}


def save(fig, name):
    save_matplotlib_figure(fig, name, subfolder="spine_density", formats=["png"], dpi=200)


# %% INPUTS
cell_types = EXCITATORY_TYPES
cells = column_cells(cell_types)
vertices = spine_density_stash("vertices").load().join(cells, on="root_id", how="semi")
root_ids = sorted(vertices["root_id"].unique().to_list())
if ROOT_LIMIT is not None:
    root_ids = root_ids[:ROOT_LIMIT]
vertices_by_root = vertices.partition_by("root_id", as_dict=True)
print(f"{len(root_ids)} cells: {cells['cell_type'].value_counts().sort('cell_type').rows()}")


# %% PER CELL
def path_vertices(root_id: int) -> pl.DataFrame:
    """One row per vertex on the path of `root_id`, with its path cable and its share
    of the spines on its vertex."""
    skel = dendrite_skeleton(root_id)
    geometry = vertex_geometry(skel, FRAME)
    path = tip_path(skel, geometry, TIP_RULE)
    d2r = geometry["path_distance_to_soma_um"].to_numpy()
    # Half of the edge to the parent plus half of the edge from the child on the path.
    to_parent = np.append(d2r[path[:-1]] - d2r[path[1:]], 0.0)
    path_cable = (to_parent + np.concatenate([[0.0], to_parent[:-1]])) / 2
    assert np.isclose(path_cable.sum(), d2r[path[0]]), root_id

    rows = (
        geometry[path]
        .select(
            "skeleton_vertex_id",
            "path_distance_to_soma_um",
            "height_above_soma_um",
            pl.col("half_edge_length_um").alias("vertex_cable_um"),
        )
        .with_columns(
            path_cable_um=path_cable,
            is_soma=path == skel.root_positional,
            radius_um=np.asarray(skel.get_feature("radius"))[path] / 1000,
        )
        .join(
            vertices_by_root[(root_id,)].select("root_id", "skeleton_vertex_id", *SPINE_SETS),
            on="skeleton_vertex_id",
            how="left",
        )
    )
    assert rows["n_spines"].null_count() == 0, f"{root_id}: a path vertex is not in vertices"
    # A branch-point vertex also holds the spines of its side branches. The path takes
    # the share of the spines that its cable has of the vertex cable.
    share = (pl.col("path_cable_um") / pl.col("vertex_cable_um")).fill_nan(0.0)
    return rows.with_columns(
        **{f"path_{name}": pl.col(name) * share for name in SPINE_SETS},
        tip_rule=pl.lit(TIP_RULE),
    )


start = time.perf_counter()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    paths = list(tqdm(pool.map(path_vertices, root_ids), total=len(root_ids)))
print(f"{time.perf_counter() - start:.0f} s for {len(root_ids)} cells")
# The soma vertex is not dendrite: its radius is the soma radius.
paths = pl.concat(paths).join(cells, on="root_id").filter(~pl.col("is_soma"))

# %% CHECK: THE BINNED DENSITIES ADD UP
x_col = {"path": "path_distance_to_soma_um", "depth": "height_above_soma_um"}[X]
x_label = {"path": "Path distance to soma (µm)", "depth": f"Height above soma (µm, {FRAME})"}[X]
binned = (
    paths.with_columns(bin_um=pl.col(x_col) // BIN_UM * BIN_UM + BIN_UM / 2)
    .group_by("root_id", "cell_type", "bin_um")
    .agg(
        cable_um=pl.col("path_cable_um").sum(),
        radius_cable=(pl.col("radius_um") * pl.col("path_cable_um")).sum(),
        **{name: pl.col(f"path_{name}").sum() for name in SPINE_SETS},
    )
    .sort("root_id", "bin_um")
)
totals = paths.group_by("root_id").agg(
    pl.col("path_cable_um").sum(), pl.col("path_n_spines").sum()
)
check = binned.group_by("root_id").agg(pl.col("cable_um").sum(), pl.col("n_spines").sum()).join(
    totals, on="root_id"
)
assert np.allclose(check["cable_um"], check["path_cable_um"])
assert np.allclose(check["n_spines"], check["path_n_spines"])
# Bins with little cable are noise.
binned = binned.filter(pl.col("cable_um") >= BIN_UM / 2).with_columns(
    radius_um=pl.col("radius_cable") / pl.col("cable_um"),
    **{f"{name}_per_um": pl.col(name) / pl.col("cable_um") for name in SPINE_SETS},
)
# The peak of each cell: the bin with the highest density, per spine set.
peaks = {
    name: binned.sort(f"{name}_per_um", descending=True).unique("root_id", keep="first")
    for name in SPINE_SETS
}
print(
    paths.group_by("cell_type")
    .agg(
        n_cells=pl.col("root_id").n_unique(),
        median_path_um=pl.col("path_cable_um").sum().over("root_id").median(),
    )
    .sort("cell_type")
)

# %% 1. ONE LINE PER CELL, WITH A DOT AT ITS PEAK
def draw_paths(
    ax,
    cell_type: str,
    name: str,
    lw: float = 0.6,
    dot: float = 6,
    value: str | None = None,
    log: bool = False,
) -> int:
    """One line per cell of `cell_type`, and the cable-weighted mean in black. Returns
    the number of cells.

    By default the lines are the density of spine set `name`, with a dot at the peak of
    each line. With `value`, the lines are the binned column `value`, `name` is its
    cable-weighted sum, and there are no dots. With `log`, the lines break at bins with a
    value of 0, which a log axis cannot show.
    """
    data = binned.filter(pl.col("cell_type") == cell_type)
    color = CELL_TYPE_PALETTE[cell_type]
    y = value or f"{name}_per_um"
    for (root_id,), group in data.partition_by("root_id", as_dict=True).items():
        values = group[y].to_numpy().copy()
        if log:
            values[values <= 0] = np.nan
        ax.plot(group["bin_um"], values, color=color, lw=lw, alpha=0.5)
    if value is None:
        peak = peaks[name].filter(pl.col("cell_type") == cell_type)
        ax.scatter(peak["bin_um"], peak[y], color="black", s=dot, zorder=10, lw=0)
    mean = (
        data.group_by("bin_um")
        .agg(pl.col(name).sum() / pl.col("cable_um").sum(), n=pl.len())
        .filter(pl.col("n") >= 3)
        .sort("bin_um")
    )
    ax.plot(mean["bin_um"], mean[name], color="black", lw=2)
    return data["root_id"].n_unique()


types_present = [c for c in cell_types if c in binned["cell_type"].unique()]
# Top row: all dendrite spines. Bottom row: the skeleton radius along the path.
fig, axs = plt.subplots(
    2,
    len(types_present),
    figsize=(3.2 * len(types_present), 7),
    sharex=True,
    sharey="row",
    layout="constrained",
    squeeze=False,
)
for col, cell_type in enumerate(types_present):
    n = draw_paths(axs[0, col], cell_type, "n_spines", log=True)
    draw_paths(axs[1, col], cell_type, "radius_cable", value="radius_um", log=True)
    axs[0, col].set_title(f"{cell_type} (n = {n})")
axs[0, 0].set_ylabel("Spines per µm")
axs[1, 0].set_ylabel("Skeleton radius (µm)")
# Log y axes. Bins with no spines do not show.
axs[0, 0].set_yscale("log")
axs[1, 0].set_yscale("log")
for ax, ticks in [(axs[0, 0], [0.1, 0.3, 1, 3]), (axs[1, 0], [0.2, 0.5, 1, 2])]:
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
fig.supxlabel(x_label)
fig.suptitle(f"Soma to the {TIP_RULE} dendrite tip, {BIN_UM} µm bins, all dendrite spines")
save(fig, f"apical_path_density_{TIP_RULE}_{X}_{BIN_UM}um_{FRAME}")

# %% 1B. ONE FIGURE PER CELL TYPE
for cell_type in types_present:
    fig, axs = plt.subplots(1, len(SPINE_SETS), figsize=(12, 4.5), sharex=True, layout="constrained")
    for ax, (name, label) in zip(axs, SPINE_SETS.items()):
        n = draw_paths(ax, cell_type, name, lw=0.8, dot=12)
        ax.set(title=label, xlabel=x_label, ylabel="Spines per µm")
        ax.set_ylim(bottom=0)
    fig.suptitle(f"{cell_type} (n = {n}), soma to the {TIP_RULE} dendrite tip, {BIN_UM} µm bins")
    save(fig, f"apical_path_density_{cell_type}_{TIP_RULE}_{X}_{BIN_UM}um_{FRAME}")

with pl.Config(tbl_rows=30):
    print(
        peaks["n_spines"]
        .group_by("cell_type")
        .agg(
            median_peak_um=pl.col("bin_um").median(),
            median_peak_density=pl.col("n_spines_per_um").median(),
        )
        .sort("cell_type")
    )

# %% 1C. PEAK DENSITY AGAINST THE RADIUS OF THE APICAL DENDRITE NEAR THE SOMA
near_radius = (
    paths.filter(pl.col("path_distance_to_soma_um").is_between(*RADIUS_WINDOW_UM))
    .group_by("root_id")
    .agg(near_soma_radius_um=pl.col("radius_um").median())
)
fig, axs = plt.subplots(1, len(SPINE_SETS), figsize=(12, 5.5), layout="constrained")
for ax, (name, label) in zip(axs, SPINE_SETS.items()):
    data = peaks[name].join(near_radius, on="root_id")
    for cell_type in types_present:
        group = data.filter(pl.col("cell_type") == cell_type)
        ax.scatter(
            group["near_soma_radius_um"],
            group[f"{name}_per_um"],
            color=CELL_TYPE_PALETTE[cell_type],
            s=14,
            alpha=0.8,
            lw=0,
            label=cell_type,
        )
    rho = data.select(pl.corr("near_soma_radius_um", f"{name}_per_um", method="spearman")).item()
    ax.set(
        xscale="log",
        yscale="log",
        title=label,
        xlabel=f"Apical radius {RADIUS_WINDOW_UM[0]}-{RADIUS_WINDOW_UM[1]} µm from soma (µm)",
        ylabel="Peak spines per µm",
    )
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize="small")
    for axis, ticks in [(ax.xaxis, [0.2, 0.5, 1, 2]), (ax.yaxis, [0.5, 1, 2, 5])]:
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FormatStrFormatter("%g"))
        axis.set_minor_formatter(NullFormatter())
axs[-1].legend(frameon=False, fontsize="x-small", markerscale=1.5, loc="lower right")
save(fig, f"apical_peak_vs_radius_{TIP_RULE}_{BIN_UM}um_{FRAME}")

# %% 2. MIDDLE AND DISTAL COMPARTMENTS
# Paper values for adult V1 L5 ET cells, read from their Fig. 2E (spines per µm).
PAPER_V1_ET = {"middle": 2.2, "distal": 0.6}
compartments = (
    paths.with_columns(
        compartment=pl.when(pl.col("path_distance_to_soma_um").is_between(*MIDDLE_UM))
        .then(pl.lit("middle"))
        .when(pl.col("path_distance_to_soma_um") > DISTAL_UM)
        .then(pl.lit("distal"))
    )
    .drop_nulls("compartment")
    .group_by("root_id", "cell_type", "compartment")
    .agg(
        cable_um=pl.col("path_cable_um").sum(),
        **{name: pl.col(f"path_{name}").sum() for name in SPINE_SETS},
    )
    .filter(pl.col("cable_um") >= 50)
    .with_columns(**{name: pl.col(name) / pl.col("cable_um") for name in SPINE_SETS})
    .group_by("cell_type", "compartment")
    .agg(
        n_cells=pl.len(),
        **{f"median_{name}_per_um": pl.col(name).median() for name in SPINE_SETS},
    )
    .with_columns(paper_v1_l5et=pl.col("compartment").replace_strict(PAPER_V1_ET, return_dtype=pl.Float64))
    .sort("cell_type", "compartment", descending=[False, True])
)
with pl.Config(tbl_rows=30, tbl_cols=10):
    print(compartments)

# %% 3. ONE 5P-ET CELL WITH ITS PATH
example = binned.filter(pl.col("cell_type") == "5P-ET")["root_id"]
if len(example):
    example = example[0]
    skel = dendrite_skeleton(example)
    geometry = vertex_geometry(skel, FRAME)
    path = tip_path(skel, geometry, TIP_RULE)
    xy = geometry.select("x_um", "y_um").to_numpy()
    fig, ax = plt.subplots(figsize=(5, 8), layout="constrained")
    add_skeleton(ax, skel.edges_positional, xy, color="0.75", linewidths=0.6)
    ax.plot(xy[path, 0], xy[path, 1], color=CELL_TYPE_PALETTE["5P-ET"], lw=2)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set(xlabel=f"x (µm, {FRAME})", ylabel=f"Depth (µm, {FRAME})", title=f"{example} (5P-ET)")
    save(fig, f"apical_path_example_{TIP_RULE}_{FRAME}")
