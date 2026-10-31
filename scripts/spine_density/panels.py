"""Panels from the outputs of `spine_density.py` (R2.3).

Path distances are native. The example cell is drawn in the post-transform `FRAME`.

    uv run python scripts/spine_density/panels.py
"""

# %%
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from analysis import CELL_TYPE_PALETTE, Tables, save_matplotlib_figure, set_matplotlib_theme
from analysis.dendrites import (
    RADII_UM,
    dendrite_skeleton,
    frame_points,
    spine_density_stash,
    vertex_geometry,
)
from analysis.plots import add_cell_type_y, add_skeleton, sorted_stripplot_with_means
from analysis.products.skeleton import cached_skeleton

set_matplotlib_theme()
# "streamline" or "affine", see `analysis.dendrites`.
FRAME = "streamline"
BROAD_TYPES = ["excitatory", "inhibitory"]
MAX_WORKERS = 8

spines = spine_density_stash("spines").load()
vertices = spine_density_stash("vertices").load()


def save(fig, name):
    save_matplotlib_figure(fig, name, subfolder="spine_density", formats=["png"], dpi=200)


def root_geometry(root_id: int) -> pl.DataFrame:
    geometry = vertex_geometry(dendrite_skeleton(root_id), FRAME)
    return geometry.select(
        pl.lit(root_id, pl.Int64).alias("root_id"), "skeleton_vertex_id", "path_distance_to_soma_um"
    )


root_ids = vertices["root_id"].unique().sort().to_list()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    geometry = pl.concat(pool.map(root_geometry, root_ids))
vertices = vertices.join(geometry, on=["root_id", "skeleton_vertex_id"], how="left")
assert vertices["path_distance_to_soma_um"].null_count() == 0


# %% PER-CELL SUMMARY
# The density of a cell is its spine count over its dendrite and soma cable. The
# synaptic spines give the lower bound, all spines on the dendrite the upper bound.
per_cell = (
    vertices.group_by("root_id")
    .agg(
        cable_um=pl.col("half_edge_length_um").sum(),
        n_spines=pl.col("n_spines").sum(),
        n_synaptic_spines=pl.col("n_synaptic_spines").sum(),
    )
    .with_columns(
        density=pl.col("n_spines") / pl.col("cable_um"),
        synaptic_density=pl.col("n_synaptic_spines") / pl.col("cable_um"),
    )
)
axon_share = (
    spines.filter(pl.col("method") == "nearest_l2")
    .group_by("root_id")
    .agg(
        n_free=pl.len(),
        p_free_on_axon=pl.col("is_axon_vertex").mean(),
    )
)
cells = add_cell_type_y(
    Tables["proofread_cell_info"]
    .lazy()
    .filter(pl.col("is_column"))
    .collect()
    .join(per_cell, left_on="pt_root_id", right_on="root_id", how="inner")
    .join(axon_share, left_on="pt_root_id", right_on="root_id", how="left")
    .to_pandas()
    .set_index("pt_root_id")
)
summary = (
    cells.groupby("cell_type", observed=True)
    .apply(
        lambda g: pd.Series(
            {
                "n_cells": len(g),
                "synaptic_density": g["n_synaptic_spines"].sum() / g["cable_um"].sum(),
                "density": g["n_spines"].sum() / g["cable_um"].sum(),
                "median_cell_synaptic_density": g["synaptic_density"].median(),
                "median_cell_density": g["density"].median(),
                "p_free_on_axon": g["n_free"].mul(g["p_free_on_axon"]).sum() / g["n_free"].sum(),
            }
        )
    )
    .round(3)
)
print(summary)

# %% 1. THE BOUNDS PER CELL TYPE
xmax = np.ceil(cells["density"].quantile(0.995) * 2) / 2
for x, label in [
    ("synaptic_density", "Synaptic spines per µm (lower bound)"),
    ("density", "Spines per µm (upper bound)"),
]:
    fig, axs, means = sorted_stripplot_with_means(
        data=cells,
        x=x,
        weight_x="cable_um",
        size="cable_um",
        size_title="Dendrite\ncable (µm)",
        size_norm=(1000, 10000),
        legend_sizes=(1000, 5000, 10000),
        xlabel=label,
        xlim=(0, xmax),
    )
    axs[1].set_xlim(0, xmax)
    save(fig, f"cell_type_{x}")

# %% 3. DENSITY AGAINST PATH DISTANCE TO THE SOMA (r = 10 µm)
BIN_UM = 20
binned = (
    vertices.join(
        pl.from_pandas(cells.reset_index()[["pt_root_id", "cell_type", "broad_type"]]),
        left_on="root_id",
        right_on="pt_root_id",
    )
    .with_columns(
        distance_bin=(pl.col("path_distance_to_soma_um") // BIN_UM * BIN_UM + BIN_UM / 2)
    )
    .group_by("cell_type", "broad_type", "distance_bin")
    .agg(
        cable_um=pl.col("half_edge_length_um").sum(),
        **{
            name: (pl.col(f"{name}_r10") * pl.col("half_edge_length_um")).sum()
            / pl.col("half_edge_length_um").sum()
            for name in ["density", "synaptic_density"]
        },
    )
    # Bins with little cable are noise.
    .filter(pl.col("cable_um") > 500)
    .sort("cell_type", "distance_bin")
    .to_pandas()
)
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True, layout="constrained")
for row, broad_type in enumerate(BROAD_TYPES):
    for col, (name, label) in enumerate(
        [("synaptic_density", "Synaptic spines"), ("density", "All spines")]
    ):
        ax = axs[row, col]
        data = binned.query("broad_type == @broad_type")
        for cell_type, group in data.groupby("cell_type", observed=True):
            ax.plot(
                group["distance_bin"],
                group[name],
                color=CELL_TYPE_PALETTE[cell_type],
                linewidth=2,
                label=cell_type,
            )
        ax.set(title=f"{label}, {broad_type}", xlabel="Path distance to soma (µm)")
        if col == 0:
            ax.set_ylabel("Spines per µm (r = 10 µm)")
        if col == 1:
            ax.legend(fontsize="x-small", frameon=False, loc="upper right")
save(fig, "density_by_path_distance")

# %% 4. THE EFFECT OF r ON THE VERTEX DENSITIES
vertex_types = vertices.join(
    pl.from_pandas(cells.reset_index()[["pt_root_id", "broad_type"]]),
    left_on="root_id",
    right_on="pt_root_id",
)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
bins = np.linspace(0, xmax * 1.5, 61)
colors = sns.color_palette("Greys", len(RADII_UM) + 1)[1:]
for ax, broad_type in zip(axs, BROAD_TYPES):
    data = vertex_types.filter(pl.col("broad_type") == broad_type)
    for r, color in zip(RADII_UM, colors):
        ax.hist(
            data[f"density_r{r}"].to_numpy(),
            bins=bins,
            weights=data["half_edge_length_um"].to_numpy(),
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"r = {r} µm",
        )
    ax.set(title=broad_type.capitalize(), xlabel="Spines per µm (all spines)")
axs[0].set_ylabel("Density (weighted by cable)")
axs[1].legend(frameon=False)
save(fig, "vertex_density_by_radius")

# %% 5. THE SYNAPSE AND LEVEL-2 PLACEMENTS OF THE SYNAPTIC SPINES
# A check of the level-2 rule on the spines where both rules apply.
synaptic = spines.filter(pl.col("method") == "synapse")
distances = vertices.select("root_id", "skeleton_vertex_id", "path_distance_to_soma_um")
check = (
    synaptic.join(distances, on=["root_id", "skeleton_vertex_id"], how="left")
    .join(
        distances.rename({"skeleton_vertex_id": "l2_vertex_id", "path_distance_to_soma_um": "l2_d"}),
        on=["root_id", "l2_vertex_id"],
        how="left",
    )
    .with_columns(gap_um=(pl.col("l2_d") - pl.col("path_distance_to_soma_um")).abs())
)
print(
    check.select(
        same_vertex=(pl.col("l2_vertex_id") == pl.col("skeleton_vertex_id")).mean(),
        l2_on_axon=pl.col("l2_d").is_null().mean(),
        gap_under_5um=(pl.col("gap_um") < 5).mean(),
        gap_p99_um=pl.col("gap_um").quantile(0.99),
    )
)
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.hist(check["gap_um"].drop_nulls().clip(upper_bound=50).to_numpy(), bins=50, color="0.3")
ax.set(
    yscale="log",
    xlabel="|Δ path distance to soma| (µm, clipped at 50)",
    ylabel="Synaptic spines",
    title="Level-2 placement against synapse placement",
)
save(fig, "placement_check")

# %% 6. ONE EXAMPLE CELL
example = cells.query("cell_type == '23P'")["cable_um"].idxmax()
skel = cached_skeleton(example).skeleton
xy = frame_points(skel.vertices, skel.root_location, FRAME)[:, :2]
density = np.full(skel.n_vertices, np.nan)
rows = vertices.filter(pl.col("root_id") == example)
density[rows["skeleton_vertex_id"].to_numpy()] = rows["density_r10"].to_numpy()
fig, ax = plt.subplots(figsize=(7, 8), layout="constrained")
# The axon has no density, so it is gray.
lines = add_skeleton(ax, skel.edges_positional, xy, density, linewidths=1.5)
ax.autoscale()
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set(
    xlabel=f"x (µm, {FRAME})",
    ylabel=f"Depth (µm, {FRAME})",
    title=f"{example} (23P), r = 10 µm",
)
fig.colorbar(lines, ax=ax, label="Spines per µm", shrink=0.6)
save(fig, f"example_cell_{FRAME}")
