"""Is the bimodal mean spine volume of the 4P-IT cells real, and why does the
mean size of the synapses onto their spines not show it? The S-figure for reviewer
comment R2.11 shows, per cell type, the per-cell mean synapse size and mean spine
volume. The mean spine volume of 4P has two modes there.

In that figure the two rows count different things:

- The mean synapse size (`post_mean_spine_size`) is over every spine-tagged synapse:
  the multi-input spines and the structures that the unitary-spine model does not call
  one spine count too, and a spine counts once per synapse.
- The mean spine volume is over the unitary spines with a synapse, once per spine.

This script puts both measures on the same spines. The cells are 23P, 4P, 5P-IT and
6P-IT with `MIN_SYNAPSES` or more synapses (2000 in the S-figure) and with a soma depth
in the 1st to 99th percentile of their type. Two flags change the cells:

- `USE_MESO_TYPE`: the types are the IT meso types (L2 IT to L6 IT), and the figures go
  to a `meso_type` subfolder, not `cell_type`. The figures of 4P use L4 IT.
- `USE_LDA_FILTER`: only the cells that the LDA of the S-figure calls excitatory.

The per-cell rollups and the synapses on the unitary spines with one synapse are
stashed, so a new filter only sums the stash again:

- `synapse_rollup`: per cell and per (`tag`, `unitary`, `n_synapses_class`,
  `is_multi`), the count and the sums of synapse size and spine volume (each plain
  and log). `n_synapses_class` is the number of synapses on the spine, with 3 for 3
  or more.
- `spine_rollup`: the same per spine, from `spine_census_features`.
- `one_synapse_spines`: every synapse on a unitary spine with one synapse, with its
  size and spine volume. The counts per cell and bin of log10 spine volume
  (`volume_counts`) come from it.

Unless a label says otherwise, the spines are the unitary spines with one synapse.

Figures, in `figures/size_distributions/4P_bimodality/`:

1. `single_spine_joint`: spine volume vs synapse size, per synapse, on unitary spines
   with one synapse, with marginals.
2. `single_spine_joint_pre`: the same for the synapses from proofread excitatory and
   inhibitory axons, with their marginals next to the marginal of all synapses.
3. `multi_input_pre_sizes`: synapse size from proofread excitatory and inhibitory
   axons, on one-synapse and on multi-input spines.
4. `cell_means`: per cell, the mean spine volume and mean synapse size, with each
   filter.
5. `cell_single_volume_vs_size`: per cell, mean spine volume vs mean synapse size on
   the same spines (unitary, one synapse).
6. `4P_depth_area`: 4P mean spine volume against depth and visual area.
7. Printed only: the within-cell and between-cell spread, and the two 4P groups.
   `cell_means_null`: per type, the mean spine volume per cell (unitary, one synapse)
   against the shared-pool null, simulated and from the central limit theorem.
8. `gmm_pairs_{type}`: per type, a 2-component Gaussian mixture on the log of mean
    spine volume, mean synapse size (both on the unitary spines with one synapse) and
    the share of multi-input spines, as a pair plot colored by cluster.
    `mixture_test`: per type, one or two groups of cells in the log of mean spine
    volume. A 2-component Gaussian mixture against one Gaussian and one skew-normal,
    each tested by a parametric bootstrap. The bootstrap is stashed.
    `weights_by_type`, `weights_cell_fits_{type}`, `weights_predicted_moments`: per
    cell, the weight of the large of two fixed Gaussians of log spine volume, fit on
    counts per bin that are stashed, and how well that one weight describes the cell.
    `weights_sd_example_{type}`: one cell, to show the SD of log volume of a cell.
    `model_ladder`: per type, five per-cell models of log spine volume, from the
    weight alone to a free mixture per cell, scored by held-out likelihood within
    each cell. The scores are stashed.
    `model_fits_cells_{type}`: the five models on example cells of the focus type.
9. `4P_matched_thalamic`: 4P cells of the two clusters, matched 1:1 by soma
    distance, and their input from the proofread thalamic axons, compared within
    pairs with the Wilcoxon signed-rank test.
10. `4P_matched_soma_shaft`: the same pairs, and their soma and shaft synapses.
11. `4P_class_thalamic`, `4P_class_soma_shaft`: the outcomes of 9 and 10 per
    cluster, over all the 4P cells with a cluster posterior of 0.9 or more.
    `4P_class_thalamic_column`: the outcomes of 9 for the column cells only.
    `4P_class_thalamic_all_spines`, `4P_class_thalamic_all_spines_column`: the same
    two, over all the spine synapses.
    `4P_class_cell_volume_hists_column`: per cell, the histogram of spine volume on
    the unitary spines with one synapse, for 30 random column cells per cluster.
    `4P_class_cell_joints_column`: for the same cells, spine volume vs synapse size
    per synapse, with marginals.
12. `4P_class_cell_features`: the same, for depth, synapse counts and shares, and
    nucleus volume.
13. `4P_class_top_down`: the soma positions of the 4P cells in the x-z plane, by
    cluster.
14. `4P_class_single_spine_joint`: spine volume vs synapse size per synapse, on the
    unitary spines with one synapse, for the cells of each cluster.
    `4P_class_single_spine_joint_cell_fits`: the same, with a least-squares line per
    cell and the pooled line per cluster.
15. `4P_class_mugshots`, `4P_class_mugshots_column`: the dendrite skeletons of 30
    cells per cluster, from all the cells and from the column cells, as in
    `scripts/spine_density/mugshots.py`. The 2D points are stashed.
16. `4P_class_spine_density`: spines per µm of dendrite against the path
    distance to the soma, one line per cell, by cluster. The cells are those with a
    local skeleton file: the column cells and the mugshot cells. The same per µm for
    the shaft synapses (`4P_class_shaft_density`), the spine volume
    (`4P_class_spine_volume_density`), the spine surface area
    (`4P_class_spine_area_density`), and the summed synapse size on the spines
    (`4P_class_spine_size_density`) and on the shafts (`4P_class_shaft_size_density`).
17. `4P_class_putative_thalamic_column`: per cluster, the synapses from putative
    thalamic axons (the predicted pre types of `scratch/per_cell_counts.parquet`),
    for the column cells.
18. Printed only: a Neuroglancer link with segment properties for the L3 and L4 cells
    (23P and 4P, or L3 IT and L4 IT).
19. `class_column_output_share`: per proofread column or thalamic cell, the share
    of its synapses onto the clusters of L3 and L4 together (23P and 4P, or L3 IT and
    L4 IT) that land on the small-spine cells, one row per cell type.

    uv run python scripts/size_distributions/bimodality.py
"""

# %%
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from caveclient import CAVEclient
from nglui.statebuilder import ViewerState
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed, parallel_config
from scipy.stats import norm, pearsonr, skewnorm, wilcoxon
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer

from analysis import (
    CELL_TYPE_PALETTE,
    FIG_PATH,
    SYNAPSE_SIZE_LABEL,
    Stash,
    Tables,
    save_matplotlib_figure,
    set_matplotlib_theme,
    synapse_size_nm3,
)
from analysis.census_inputs import post_cells
from analysis.dendrites import dendrite_skeleton, vertex_geometry
from analysis.plots import add_skeleton, remapper
from analysis.products.census import census_synapses
from analysis.products.skeleton import (
    L4IT_SKELETON_CACHE,
    SKELETON_CACHE,
    cached_skeleton,
    dendrite_mask,
)
from analysis.sources import DATASTACK, VERSION

set_matplotlib_theme()
out_path = FIG_PATH / "size_distributions" / "4P_bimodality"
USE_MESO_TYPE = True
USE_LDA_FILTER = False
if USE_MESO_TYPE:
    TYPE_NAMES = {f"L{n}IT": f"L{n} IT" for n in range(2, 7)}
    FOCUS_TYPE = "L4IT"
    out_path = out_path / "meso_type"
else:
    TYPE_NAMES = {t: remapper(t) for t in ["23P", "4P", "5P-IT", "6P-IT"]}
    FOCUS_TYPE = "4P"
    out_path = out_path / "cell_type"
CELL_TYPES = list(TYPE_NAMES)
FOCUS = TYPE_NAMES[FOCUS_TYPE]
MIN_SYNAPSES = 1000
VOLUME_LABEL = "Spine volume (nm³)"
SIZE_LABEL = SYNAPSE_SIZE_LABEL[0].upper() + SYNAPSE_SIZE_LABEL[1:]
EI_PALETTE = {
    "excitatory": CELL_TYPE_PALETTE["excitatory"],
    "inhibitory": CELL_TYPE_PALETTE["inhibitory"],
}

# %% CELLS, WITH THE OPTIONAL LDA FILTER OF THE S-FIGURE
cells = post_cells(False).to_pandas().set_index("pt_root_id")
if USE_MESO_TYPE:
    cells["cell_type"] = cells["meso_type"]
cells["cell_type"] = cells["cell_type"].astype(str)
if USE_LDA_FILTER:
    features = [
        "post_p_spine_synapse",
        "post_p_shaft_synapse",
        "post_p_soma_synapse",
        "post_mean_spine_inputs",
        "post_p_spine_site_is_multi",
        "post_p_spine_synapse_is_multi",
        "nuc_volume",
    ]
    X = cells[features]
    mask = X.notna().all(axis=1)
    train = cells[
        mask & cells["is_column"] & cells["broad_type"].isin(["excitatory", "inhibitory"])
    ]
    transformer = QuantileTransformer(output_distribution="normal", random_state=0)
    transformer.fit(X[mask])
    lda = LinearDiscriminantAnalysis()
    lda.fit(transformer.transform(train[features]), train["broad_type"].astype(str))
    cells.loc[mask, "broad_type_lda_prediction"] = lda.predict(
        transformer.transform(X[mask])
    )
# The rollups hold every cell of `CELL_TYPES` with enough synapses, so the LDA filter
# can change without a new rollup.
rollup_ids = cells.query(
    "cell_type in @CELL_TYPES and post_total_synapses >= @MIN_SYNAPSES"
).index.to_numpy()
cells = cells.loc[rollup_ids]
if USE_LDA_FILTER:
    cells = cells.query("broad_type_lda_prediction == 'excitatory'")
# Per type, only the cells with a soma depth in the 1st to 99th percentile.
depth = cells.groupby("cell_type")["pt_position_um_y"]
cells = cells[
    cells["pt_position_um_y"].between(
        depth.transform("quantile", 0.01), depth.transform("quantile", 0.99)
    )
].copy()
cells["post_mean_spine_size"] = synapse_size_nm3(cells["post_mean_spine_size"])
print(cells["cell_type"].value_counts())

# %% STASHED ROLLUPS
stash_inputs = [
    Tables[name]
    for name in (
        "microns_synapses",
        "synapse_census_features",
        "spine_census_features",
        "post_aggregation",
        "cell_info",
    )
]
stash_params = dict(cell_types=CELL_TYPES, min_synapses=MIN_SYNAPSES)
synapse_stash = Stash("r2_11_4P_bimodality/synapse_rollup", stash_inputs, **stash_params)
spine_stash = Stash("r2_11_4P_bimodality/spine_rollup", stash_inputs, **stash_params)
singles_stash = Stash("r2_11_4P_bimodality/one_synapse_spines", stash_inputs, **stash_params)
SPINE_KEYS = ["unitary", "n_synapses_class", "is_multi"]


def spine_flags(prefix: str) -> list[pl.Expr]:
    """The filter flags of a spine, from its columns with `prefix`. They are null
    for a synapse on no structure."""
    return [
        (pl.col(f"{prefix}unitary_pred_label") == "unitary").alias("unitary"),
        pl.col(f"{prefix}n_synapses").clip(upper_bound=3).alias("n_synapses_class"),
        pl.col(f"{prefix}is_multi").alias("is_multi"),
    ]


def sums(value: str, name: str) -> list[pl.Expr]:
    return [
        pl.col(value).sum().alias(f"{name}_sum"),
        pl.col(value).log().sum().alias(f"log_{name}_sum"),
    ]


def roll_synapses(part: pl.LazyFrame) -> pl.LazyFrame:
    return (
        part.with_columns(spine_flags("spine_"))
        .group_by("post_pt_root_id", "tag", *SPINE_KEYS)
        .agg(pl.len().alias("n"), *sums("size", "size"), *sums("spine_size_nm3", "volume"))
    )


def one_synapse_spines(part: pl.LazyFrame) -> pl.LazyFrame:
    return (
        part.with_columns(spine_flags("spine_"))
        .filter(
            pl.col("tag") == "spine",
            pl.col("unitary").fill_null(False),
            pl.col("n_synapses_class") == 1,
            pl.col("spine_size_nm3") > 0,
        )
        .select(
            "post_pt_root_id",
            pl.col("size").cast(pl.Float32),
            pl.col("spine_size_nm3").cast(pl.Float32).alias("volume"),
        )
    )


where = pl.col("post_pt_root_id").is_in(pl.Series(rollup_ids).implode())
if not synapse_stash.exists():
    synapse_stash.save(census_synapses(where, then=roll_synapses))
if not singles_stash.exists():
    singles_stash.save(census_synapses(where, then=one_synapse_spines))
if not spine_stash.exists():
    spine_stash.save(
        Tables["spine_census_features"]
        .lazy()
        .filter(where)
        .with_columns(spine_flags(""))
        .group_by("post_pt_root_id", *SPINE_KEYS)
        .agg(pl.len().alias("n"), *sums("size_nm3", "volume"))
        .collect(engine="streaming")
    )
synapse_rollup = synapse_stash.load().with_columns(
    pl.col("size_sum") * synapse_size_nm3(1),
    pl.col("log_size_sum") + pl.col("n") * np.log(synapse_size_nm3(1)),
)
spine_rollup = spine_stash.load()
cell_types = pl.DataFrame(
    {"post_pt_root_id": cells.index.to_numpy(), "cell_type": cells["cell_type"].values}
)
# Every synapse on a unitary spine with one synapse, about 7 × 10^7 of them. The cell
# type is an Enum, so it costs 4 bytes per synapse.
cell_type_enum = cell_types.with_columns(pl.col("cell_type").cast(pl.Enum(CELL_TYPES)))
singles_frame = (
    singles_stash.load()
    .with_columns(size=synapse_size_nm3(pl.col("size")).cast(pl.Float32))
    .join(cell_type_enum, on="post_pt_root_id", how="inner")
)
# Per cell, the counts of the synapses per bin of log10 spine volume. The end bins are
# open, so they hold the spines past the range.
VOLUME_RANGE = (6.0, 9.5)
N_VOLUME_BINS = 100
volume_edges = np.linspace(*VOLUME_RANGE, N_VOLUME_BINS + 1)
volume_counts = (
    singles_frame.group_by(
        "post_pt_root_id",
        (
            (pl.col("volume").cast(pl.Float64).log10() - VOLUME_RANGE[0])
            / (volume_edges[1] - volume_edges[0])
        )
        .floor()
        .clip(0, N_VOLUME_BINS - 1)
        .cast(pl.Int32)
        .alias("bin"),
    )
    .len()
    .to_pandas()
    .pivot_table(index="post_pt_root_id", columns="bin", values="len", fill_value=0)
    .reindex(columns=range(N_VOLUME_BINS), fill_value=0)
    .astype(np.int64)
)
singles = singles_frame.to_pandas()
del singles_frame


# %% PER-CELL MEANS UNDER EACH FILTER
def cell_mean(rollup: pl.DataFrame, keep: pl.Expr, value: str) -> pd.Series:
    """Per cell, the mean of `value` over the rows of `rollup` that `keep` holds."""
    means = (
        rollup.filter(keep)
        .group_by("post_pt_root_id")
        .agg((pl.col(f"{value}_sum").sum() / pl.col("n").sum()).alias("mean"))
    )
    return means.to_pandas().set_index("post_pt_root_id")["mean"]


on_spine = pl.col("tag") == "spine"
unitary = pl.col("unitary").fill_null(False)
synaptic = pl.col("n_synapses_class") >= 1
not_multi = ~pl.col("is_multi").fill_null(False)
one_synapse = pl.col("n_synapses_class") == 1
# name: (label, rollup, filter, value)
MEASURES = {
    "volume_unitary": (
        "Spine volume:\nunitary spines with a synapse",
        spine_rollup,
        unitary & synaptic,
        "volume",
    ),
    "volume_unitary_not_multi": (
        "Spine volume:\nunitary, not multi-input",
        spine_rollup,
        unitary & synaptic & not_multi,
        "volume",
    ),
    "volume_one_synapse": (
        "Spine volume:\nunitary, one synapse",
        synapse_rollup,
        on_spine & unitary & one_synapse,
        "volume",
    ),
    "size_spine": (
        "Synapse size:\nall spine synapses",
        synapse_rollup,
        on_spine,
        "size",
    ),
    "size_unitary": (
        "Synapse size:\nunitary spines",
        synapse_rollup,
        on_spine & unitary,
        "size",
    ),
    "size_unitary_not_multi": (
        "Synapse size:\nunitary, not multi-input",
        synapse_rollup,
        on_spine & unitary & not_multi,
        "size",
    ),
    "size_one_synapse": (
        "Synapse size:\nunitary, one synapse",
        synapse_rollup,
        on_spine & unitary & one_synapse,
        "size",
    ),
    "size_multi": (
        "Synapse size:\nunitary, multi-input",
        synapse_rollup,
        on_spine & unitary & ~not_multi,
        "size",
    ),
}
for name, (_, rollup, keep, value) in MEASURES.items():
    cells[name] = cell_mean(rollup, keep, value)
cells["n_one_synapse"] = (
    synapse_rollup.filter(on_spine & unitary & one_synapse)
    .group_by("post_pt_root_id")
    .agg(pl.col("n").sum())
    .to_pandas()
    .set_index("post_pt_root_id")["n"]
    .reindex(cells.index, fill_value=0)
)
cells["p_one_synapse"] = cells["n_one_synapse"] / cells["post_total_known_synapses"]
# Sanity check: the rollup gives the mean of the top row.
ratio = cells["size_spine"] / cells["post_mean_spine_size"]
print(f"size_spine / post_mean_spine_size: {ratio.min():.4f} to {ratio.max():.4f}")
print(cells.groupby("cell_type")[list(MEASURES)].median().T)


# %% PROOFREAD INPUTS ONTO THE SPINES OF THESE CELLS
# The thalamic axons count as excitatory.
proofread = (
    Tables["proofread_pre_synapses"]
    .lazy()
    .filter(where, pl.col("tag") == "spine")
    .select(
        "post_pt_root_id",
        "size",
        pl.col("spine_size_nm3").alias("volume"),
        *spine_flags("spine_"),
        pl.col("pre_broad_type")
        .cast(pl.String)
        .replace({"thalamic": "excitatory"})
        .alias("pre_type"),
    )
    .filter(pl.col("pre_type").is_in(["excitatory", "inhibitory"]))
    .collect()
    .with_columns(size=synapse_size_nm3(pl.col("size")))
    .join(cell_types, on="post_pt_root_id", how="inner")
)
print(
    proofread.group_by("cell_type", "pre_type", "n_synapses_class", "is_multi")
    .len()
    .sort("cell_type", "pre_type", "n_synapses_class", "is_multi")
)


# %% JOINT PLOT HELPERS
def joint_axes(n: int, width: float = 4):
    """`n` joint axes in a row on shared x and y, each with a top and a right
    marginal axes."""
    fig, axs = plt.subplots(
        1, n, figsize=(width * n, width), sharex=True, sharey=True, squeeze=False
    )
    fig.subplots_adjust(wspace=0.5)
    joints = []
    for ax in axs[0]:
        top = ax.inset_axes([0, 1.04, 1, 0.25], sharex=ax)
        right = ax.inset_axes([1.04, 0, 0.25, 1], sharey=ax)
        for marginal in (top, right):
            marginal.tick_params(labelbottom=False, labelleft=False)
            marginal.spines[["top", "right"]].set_visible(False)
        joints.append((ax, top, right))
    return fig, joints


def hide_marginal_labels(joints):
    """Hide the tick labels of the marginals. Call it after the scales are set,
    because a new scale makes new tick labels."""
    for _, top, right in joints:
        for marginal in (top, right):
            marginal.tick_params(which="both", labelbottom=False, labelleft=False)


def log_edges(values, n: int = 60) -> np.ndarray:
    low, high = np.quantile(values, [0.001, 0.999])
    return np.geomspace(low, high, n + 1)


def hide_count_axis(ax, vertical=False):
    """Drop the spine, ticks and label of the count axis of a histogram on `ax`."""
    ax.spines["bottom" if vertical else "left"].set_visible(False)
    if vertical:
        ax.set(xticks=[], xlabel="")
    else:
        ax.set(yticks=[], ylabel="")


def line_hist(ax, values, bins, vertical=False, log=True, **kwargs):
    """A density histogram of `values` on `bins`, as a line through the bin centers,
    with no count axis. With `log`, `bins` are log-spaced edges. The label of the
    value axis of `ax` stays."""
    label = ax.get_ylabel() if vertical else ax.get_xlabel()
    kwargs.setdefault("fill", False)
    kwargs.setdefault("stat", "density")
    sns.histplot(
        **({"y": values} if vertical else {"x": values}),
        bins=np.log10(bins) if log else bins,
        log_scale=log,
        element="poly",
        ax=ax,
        **kwargs,
    )
    (ax.set_ylabel if vertical else ax.set_xlabel)(label)
    hide_count_axis(ax, vertical)


def log_r(x, y) -> float:
    return pearsonr(np.log(x), np.log(y))[0]


# %% 1: SPINE VOLUME VS SYNAPSE SIZE, ONE SYNAPSE PER SPINE
x_edges = log_edges(singles["volume"])
y_edges = log_edges(singles["size"])
fig, joints = joint_axes(len(CELL_TYPES))
for (ax, top, right), cell_type in zip(joints, CELL_TYPES):
    data = singles.query("cell_type == @cell_type")
    ax.hist2d(
        data["volume"], data["size"], bins=[x_edges, y_edges], cmap="Greys", rasterized=True
    )
    line_hist(top, data["volume"], x_edges, color="black", fill=True, alpha=0.4)
    line_hist(right, data["size"], y_edges, vertical=True, color="black", fill=True, alpha=0.4)
    r = log_r(data["volume"], data["size"])
    print(f"1 {cell_type}: n={len(data):,} synapses, log-log r={r:.2f}")
    ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes, va="top")
    ax.set(xscale="log", yscale="log")
    top.set_title(TYPE_NAMES[cell_type], pad=4)
hide_marginal_labels(joints)
joints[0][0].set_ylabel(SIZE_LABEL)
fig.supxlabel(VOLUME_LABEL, y=-0.09)
save_matplotlib_figure(fig, "single_spine_joint", out_path, formats=["png"])

# %% 2: THE SAME, BY THE TYPE OF THE PROOFREAD PRE CELL
proofread_singles = proofread.filter(
    pl.col("unitary").fill_null(False), pl.col("n_synapses_class") == 1
).to_pandas()
fig, joints = joint_axes(len(CELL_TYPES))
for (ax, top, right), cell_type in zip(joints, CELL_TYPES):
    data = singles.query("cell_type == @cell_type")
    for marginal, column, vertical in ((top, "volume", False), (right, "size", True)):
        edges = x_edges if column == "volume" else y_edges
        line_hist(
            marginal, data[column], edges, vertical=vertical, color="grey", fill=True,
            alpha=0.4, label="All",
        )
    for pre_type in ["excitatory", "inhibitory"]:
        pre = proofread_singles.query("cell_type == @cell_type and pre_type == @pre_type")
        ax.scatter(
            pre["volume"], pre["size"], s=1.5, color=EI_PALETTE[pre_type], alpha=0.4,
            linewidth=0, rasterized=True,
        )
        line_hist(top, pre["volume"], x_edges, color=EI_PALETTE[pre_type], linewidth=1.5)
        line_hist(
            right, pre["size"], y_edges, vertical=True, color=EI_PALETTE[pre_type],
            linewidth=1.5, label=f"{pre_type.capitalize()} pre (proofread)",
        )
        print(f"2 {cell_type} {pre_type}: n={len(pre):,}")
    ax.set(xscale="log", yscale="log", xlim=x_edges[[0, -1]], ylim=y_edges[[0, -1]])
    top.set_title(TYPE_NAMES[cell_type], pad=4)
hide_marginal_labels(joints)
joints[0][0].set_ylabel(SIZE_LABEL)
fig.supxlabel(VOLUME_LABEL, y=-0.09)
fig.legend(
    *right.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.17),
    ncol=3,
)
save_matplotlib_figure(fig, "single_spine_joint_pre", out_path, formats=["png"])

# %% 3: SYNAPSE SIZE BY PRE TYPE, ON ONE-SYNAPSE AND ON MULTI-INPUT SPINES
spine_kind = (
    pl.when(pl.col("is_multi").fill_null(False))
    .then(pl.lit("multi-input"))
    .when(pl.col("unitary").fill_null(False) & (pl.col("n_synapses_class") == 1))
    .then(pl.lit("one synapse"))
)
kinds = proofread.with_columns(kind=spine_kind).drop_nulls("kind").to_pandas()
fig, axs = plt.subplots(
    1, len(CELL_TYPES), figsize=(4 * len(CELL_TYPES), 3.5), layout="constrained",
    sharex=True, sharey=True,
)
for ax, cell_type in zip(axs, CELL_TYPES):
    line_hist(
        ax, singles.query("cell_type == @cell_type")["size"], y_edges, color="grey",
        fill=True, alpha=0.4, label="All, one synapse",
    )
    for (pre_type, kind), linestyle in zip(
        [("excitatory", "one synapse"), ("inhibitory", "one synapse"),
         ("excitatory", "multi-input"), ("inhibitory", "multi-input")],
        ["-", "-", "--", "--"],
    ):
        data = kinds.query(
            "cell_type == @cell_type and pre_type == @pre_type and kind == @kind"
        )
        line_hist(
            ax, data["size"], y_edges, color=EI_PALETTE[pre_type], linestyle=linestyle,
            linewidth=1.5, label=f"{pre_type.capitalize()} pre, {kind} spine",
        )
        print(
            f"3 {cell_type} {pre_type} {kind}: n={len(data):,}, "
            f"median size {data['size'].median():.3g}"
        )
    ax.set(title=TYPE_NAMES[cell_type])
fig.supxlabel(SIZE_LABEL)
fig.legend(*ax.get_legend_handles_labels(), loc="outside right center")
save_matplotlib_figure(fig, "multi_input_pre_sizes", out_path, formats=["png"])

# %% 4: PER-CELL MEANS UNDER EACH FILTER
fig, axs = plt.subplots(
    len(CELL_TYPES), len(MEASURES), figsize=(2.6 * len(MEASURES), 2.2 * len(CELL_TYPES)),
    layout="constrained",
)
# The volume columns share one x axis, and the size columns another.
kind_edges, kind_axes = {}, {}
for name, (_, _, _, value) in MEASURES.items():
    kind_edges.setdefault(value, []).append(cells[name].dropna())
kind_edges = {value: log_edges(pd.concat(v), n=50) for value, v in kind_edges.items()}
for j, (name, (label, _, _, value)) in enumerate(MEASURES.items()):
    for i, cell_type in enumerate(CELL_TYPES):
        ax = axs[i, j]
        if value in kind_axes:
            ax.sharex(kind_axes[value])
        else:
            kind_axes[value] = ax
        line_hist(
            ax, cells.query("cell_type == @cell_type")[name].dropna(), kind_edges[value],
            stat="count", color="black",
        )
        ax.tick_params(which="both", labelsize="x-small")
        if i < len(CELL_TYPES) - 1:
            ax.tick_params(which="both", labelbottom=False)
        if i == 0:
            ax.set_title(label, fontsize="small")
        if j == 0:
            ax.text(
                -0.08, 0.5, TYPE_NAMES[cell_type], transform=ax.transAxes, rotation=90,
                ha="right", va="center",
            )
# Set after the scales, because a new scale resets the tickers.
size_axis = kind_axes["size"].xaxis
size_axis.set_major_locator(ticker.LogLocator(subs=(2, 4, 6)))
size_axis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))
size_axis.set_minor_formatter(ticker.NullFormatter())
fig.supxlabel("Mean per cell (nm³)")
save_matplotlib_figure(fig, "cell_means", out_path, formats=["png"])

# %% 5: PER CELL, MEAN SPINE VOLUME VS MEAN SYNAPSE SIZE ON THE SAME SPINES
fig, joints = joint_axes(len(CELL_TYPES))
x_edges_cell = log_edges(cells["volume_one_synapse"], n=50)
y_edges_cell = log_edges(cells["size_one_synapse"], n=50)
for (ax, top, right), cell_type in zip(joints, CELL_TYPES):
    data = cells.query("cell_type == @cell_type")
    ax.scatter(
        data["volume_one_synapse"], data["size_one_synapse"], s=2, color="black",
        alpha=0.3, linewidth=0, rasterized=True,
    )
    line_hist(top, data["volume_one_synapse"], x_edges_cell, color="black")
    line_hist(right, data["size_one_synapse"], y_edges_cell, vertical=True, color="black")
    r = log_r(data["volume_one_synapse"], data["size_one_synapse"])
    ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes, va="top")
    ax.set(xscale="log", yscale="log")
    top.set_title(TYPE_NAMES[cell_type], pad=4)
hide_marginal_labels(joints)
joints[0][0].set_ylabel("Mean " + SYNAPSE_SIZE_LABEL)
fig.supxlabel("Mean spine volume (nm³)", y=-0.09)
fig.suptitle("Unitary spines with one synapse", y=1.32)
save_matplotlib_figure(fig, "cell_single_volume_vs_size", out_path, formats=["png"])


# %% 6: 4P, MEAN SPINE VOLUME VS DEPTH AND VISUAL AREA
def ridges(subfig, data, by, order, value, edges, labels=None, share=None):
    """One density histogram of `value` per group of `by`, stacked, on shared x and
    y, with the label and count of each group if `labels` is given. The x axis is
    also shared with the axes `share`, if given."""
    axs = subfig.subplots(len(order), 1)
    for k, (ax, group) in enumerate(zip(axs, order)):
        ax.sharex(share if share is not None else axs[0])
        if ax is not axs[0]:
            ax.sharey(axs[0])
        values = data.loc[data[by] == group, value]
        line_hist(ax, values, edges, color="black")
        if labels is not None:
            ax.text(
                1.02, 0.5, f"{labels[k]}\nn={len(values)}", transform=ax.transAxes,
                va="center",
            )
    for ax in axs[:-1]:
        ax.tick_params(which="both", labelbottom=False)
    axs[-1].set_xscale("log")
    return axs


four = cells.query("cell_type == @FOCUS_TYPE").copy()
value = "volume_one_synapse"
edges = log_edges(four[value], n=50)
four["depth_bin"] = pd.qcut(four["pt_position_um_y"], 6)
four["visual_area"] = four["visual_area"].fillna("None")
areas = four["visual_area"].value_counts()
areas = areas[areas >= 50].index.tolist()

fig = plt.figure(figsize=(14, 5), layout="constrained")
left, middle, right = fig.subfigures(1, 3, width_ratios=[1.2, 1, 1])
ax = left.subplots()
ax.scatter(
    four[value], four["pt_position_um_y"], s=3, color="black", alpha=0.3, linewidth=0,
    rasterized=True,
)
ax.set(xscale="log", ylabel="Soma depth (µm)")
ax.invert_yaxis()
depth_order = four["depth_bin"].cat.categories
for depth_bin in depth_order[:-1]:
    ax.axhline(depth_bin.right, color="grey", linestyle="--", linewidth=1)
print(
    f"6 4P depth vs log {value}: r={pearsonr(four['pt_position_um_y'], np.log(four[value]))[0]:.2f}"
)
left.suptitle(FOCUS)
ridges(middle, four, "depth_bin", depth_order, value, edges, share=ax)
middle.suptitle("By depth (sixths)")
ridges(right, four, "visual_area", areas, value, edges, areas, share=ax)
right.suptitle("By visual area")
fig.supxlabel("Mean spine volume (nm³), unitary spines with one synapse")
save_matplotlib_figure(fig, f"{FOCUS_TYPE}_depth_area", out_path, formats=["png"])

# %% 7: WITHIN-CELL VS BETWEEN-CELL SPREAD, AND THE TWO 4P GROUPS
# On the unitary spines with one synapse, in natural log. The within-cell SD is the
# median over cells of the SD over all the spines of the cell. The between-cell SD is the SD of
# the cell means from the rollup.
log_means = (
    synapse_rollup.filter(on_spine & unitary & one_synapse)
    .group_by("post_pt_root_id")
    .agg(
        log_size=pl.col("log_size_sum").sum() / pl.col("n").sum(),
        log_volume=pl.col("log_volume_sum").sum() / pl.col("n").sum(),
    )
    .to_pandas()
    .set_index("post_pt_root_id")
)
cells[["mean_log_size", "mean_log_volume"]] = log_means[["log_size", "log_volume"]]
cells["mean_log_ratio"] = cells["mean_log_volume"] - cells["mean_log_size"]
within = (
    singles.assign(log_size=np.log(singles["size"]), log_volume=np.log(singles["volume"]))
    .groupby(["cell_type", "post_pt_root_id"])[["log_size", "log_volume"]]
    .std()
    .groupby("cell_type")
    .median()
)
between = cells.groupby("cell_type")[["mean_log_size", "mean_log_volume"]].std()
spread = pd.concat(
    {
        "within-cell SD": within.rename(columns=lambda c: c.replace("log_", "")),
        "between-cell SD": between.rename(columns=lambda c: c.replace("mean_log_", "")),
    },
    axis=1,
)
print("7 SD of log values, unitary spines with one synapse:")
print(spread.round(3))

# The 4P cells split at the valley of the histogram of the log volume / size ratio,
# between its quartiles. The split only describes the two groups.
four = four.join(cells[["mean_log_size", "mean_log_volume", "mean_log_ratio"]])
counts, ratio_bins = np.histogram(four["mean_log_ratio"], bins=60)
centers = (ratio_bins[1:] + ratio_bins[:-1]) / 2
middle = (centers > four["mean_log_ratio"].quantile(0.25)) & (
    centers < four["mean_log_ratio"].quantile(0.75)
)
valley = centers[middle][np.argmin(counts[middle])]
four["ratio_group"] = np.where(four["mean_log_ratio"] < valley, "low", "high")
print(f"7 4P split at log ratio {valley:.2f} (ratio {np.exp(valley):.1f}):")
print(
    four.groupby("ratio_group")[["mean_log_size", "mean_log_volume", "mean_log_ratio"]]
    .agg(["count", "mean", "std"])
    .round(3)
    .T
)

# %% 7b: CELL MEANS AGAINST THE SHARED-POOL NULL
# The null: each cell draws its `n_one_synapse` spines independently from one pool per
# type, all the spines of its cells. Simulated: per cell, the mean of that many
# draws from the pool. Theory: by the central limit theorem, the mean of a cell is about
# Normal(pool mean, pool variance / n), so the cell means are a mixture of normals with
# one center. The overdispersion is the variance of the observed cell means over the
# mean of pool variance / n.
rng = np.random.default_rng(0)
null_edges = log_edges(cells["volume_one_synapse"].dropna(), n=120)
fig, axs = plt.subplots(
    len(CELL_TYPES), 1, figsize=(6, 1.9 * len(CELL_TYPES)), sharex=True,
    layout="constrained",
)
for ax, cell_type in zip(axs, CELL_TYPES):
    data = cells.query("cell_type == @cell_type and n_one_synapse > 0")
    observed = data["volume_one_synapse"].to_numpy()
    n = data["n_one_synapse"].to_numpy()
    pool = singles.loc[singles["cell_type"] == cell_type, "volume"].dropna().to_numpy()
    simulated = np.array([rng.choice(pool, k).mean() for k in n])
    sd = np.sqrt(pool.var() / n)
    cdf = norm.cdf((null_edges[:, None] - pool.mean()) / sd)
    theory = (cdf[1:] - cdf[:-1]).sum(axis=1)
    counts = np.histogram(observed, null_edges)[0]
    ax.stairs(counts, null_edges, fill=True, color="grey", alpha=0.4, label="Observed")
    ax.stairs(
        np.histogram(simulated, null_edges)[0], null_edges, color="black", linewidth=1,
        label="Null, simulated",
    )
    ax.stairs(theory, null_edges, color="tab:red", linestyle="--", linewidth=1, label="Null, theory")
    # The null peak is far above the observed one, so the y axis is cut.
    ax.set(xscale="log", ylim=(0, 1.3 * counts.max()), ylabel="Cells")
    ax.text(
        0.99, 0.95, f"{TYPE_NAMES[cell_type]}, n={len(data):,}\nnull peak: "
        f"{theory.max():.0f} cells per bin (cut)", transform=ax.transAxes, ha="right",
        va="top", fontsize="small",
    )
    print(
        f"7b {cell_type}: n={len(data):,} cells, median {np.median(n):.0f} spines, "
        f"pool of {len(pool):,}; SD of log10 cell means: observed "
        f"{np.log10(observed).std():.3f}, simulated {np.log10(simulated).std():.3f}; "
        f"overdispersion {observed.var() / np.mean(sd**2):.0f}"
    )
fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
fig.supxlabel("Mean spine volume per cell (nm³),\nunitary spines with one synapse")
save_matplotlib_figure(fig, "cell_means_null", out_path, formats=["png"])

# %% 8: GAUSSIAN MIXTURE PER TYPE ON SPINE VOLUME, SYNAPSE SIZE AND MULTI-INPUT RATE
# The fit is on log10 values, with full covariances. The clusters are numbered by
# their mean spine volume. The BIC of 1 to 4 components is printed for reference.
N_COMPONENTS = 2
multi_rate = (
    spine_rollup.filter(unitary & synaptic)
    .group_by("post_pt_root_id")
    .agg(
        rate=pl.col("n").filter(pl.col("is_multi").fill_null(False)).sum()
        / pl.col("n").sum()
    )
    .to_pandas()
    .set_index("post_pt_root_id")["rate"]
)
cells["multi_rate"] = multi_rate
GMM_VARS = {
    "volume_one_synapse": "Mean spine\nvolume (nm³)",
    "size_one_synapse": "Mean synaptic junction\nsize (nm³)",
    "multi_rate": "Proportion of spines\nwith multiple inputs",
}
# Set2 colors 0 and 1 are I and E.
CLUSTER_PALETTE = dict(enumerate(sns.color_palette("Set2")[2:4]))


def sci_label(x: float, _=None) -> str:
    """`x` as m×10^e, or 10^e when m is 1."""
    exponent = int(np.floor(np.log10(x)))
    mantissa = round(x / 10**exponent, 2)
    power = f"10$^{{{exponent}}}$"
    return power if mantissa == 1 else f"{mantissa:g}×{power}"


# Fixed ticks, because the ranges are under a decade, where the log locator gives
# too few ticks.
GMM_TICKS = {
    "volume_one_synapse": ([6e7, 1e8, 1.5e8, 2e8], ticker.FuncFormatter(sci_label)),
    "size_one_synapse": ([3e6, 4e6, 6e6, 1e7], ticker.FuncFormatter(sci_label)),
    "multi_rate": ([0.02, 0.05, 0.1, 0.2], ticker.FormatStrFormatter("%g")),
}


def fit_gmm(X: np.ndarray, n_components: int) -> GaussianMixture:
    return GaussianMixture(
        n_components, covariance_type="full", n_init=10, random_state=0
    ).fit(X)


def ellipse(mean: np.ndarray, cov: np.ndarray, n_sd: float = 2) -> np.ndarray:
    """The points of the `n_sd` contour of a 2D Gaussian."""
    values, vectors = np.linalg.eigh(cov)
    angles = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    return mean[:, None] + vectors @ (n_sd * np.sqrt(values)[:, None] * circle)


gmm_labels = {}
for cell_type in CELL_TYPES:
    data = cells.query("cell_type == @cell_type")[list(GMM_VARS)].dropna()
    dropped = (data <= 0).any(axis=1)
    data = data[~dropped].copy()
    X = np.log10(data.to_numpy())
    bic = {k: round(fit_gmm(X, k).bic(X)) for k in range(1, 5)}
    gmm = fit_gmm(X, N_COMPONENTS)
    order = np.argsort(gmm.means_[:, 0])
    rank = np.empty_like(order)
    rank[order] = np.arange(N_COMPONENTS)
    data["cluster"] = rank[gmm.predict(X)]
    data["posterior"] = gmm.predict_proba(X).max(axis=1)
    gmm_labels[cell_type] = data[["cluster", "posterior"]]
    print(f"8 {cell_type}: n={len(data)} ({dropped.sum()} dropped at 0), BIC {bic}")
    for component in order:
        means = ", ".join(
            f"{name} {value:.3g}" for name, value in zip(GMM_VARS, 10 ** gmm.means_[component])
        )
        print(
            f"   cluster {rank[component]}: weight {gmm.weights_[component]:.2f}, "
            f"n={(data['cluster'] == rank[component]).sum()}, means {means}"
        )

    data = data.rename(columns=GMM_VARS)
    labels = list(GMM_VARS.values())
    g = sns.PairGrid(
        data, vars=labels, hue="cluster", palette=CLUSTER_PALETTE, corner=True,
        height=3.2, diag_sharey=False,
    )
    g.map_lower(sns.scatterplot, s=3, alpha=0.3, linewidth=0, rasterized=True)
    # The diagonal: each cluster scaled by its share of the cells, and all cells.
    g.map_diag(
        sns.histplot, bins=40, log_scale=True, stat="density", common_norm=True,
        element="poly", fill=False,
    )
    for label, ax in zip(labels, g.diag_axes):
        line_hist(
            ax, data[label], log_edges(data[label], n=40), color="grey", fill=True,
            alpha=0.3, zorder=0,
        )
    for i, j in zip(*np.tril_indices(len(labels), k=-1)):
        ax = g.axes[i, j]
        for component in range(N_COMPONENTS):
            index = [j, i]
            points = ellipse(
                gmm.means_[component][index], gmm.covariances_[component][np.ix_(index, index)]
            )
            ax.plot(
                10 ** points[0], 10 ** points[1], color=CLUSTER_PALETTE[rank[component]],
                linewidth=1.5,
            )
        ax.set(xscale="log", yscale="log")
    # Set after the scales. The axes of a column share x, and of a row share y.
    for k, name in enumerate(GMM_VARS):
        axes = [g.axes[-1, k].xaxis] + ([g.axes[k, 0].yaxis] if k > 0 else [])
        for axis in axes:
            ticks, formatter = GMM_TICKS[name]
            axis.set_major_locator(ticker.FixedLocator(ticks))
            axis.set_major_formatter(formatter)
            axis.set_minor_formatter(ticker.NullFormatter())
    g.add_legend(title="Cluster", markerscale=4)
    g.figure.suptitle(
        f"{TYPE_NAMES[cell_type]}: Gaussian mixture, {N_COMPONENTS} components "
        f"(2 SD ellipses), n={len(data):,}",
        y=1.02,
    )
    save_matplotlib_figure(g.figure, f"gmm_pairs_{cell_type}", out_path, formats=["png"])

# %% 8b: ONE OR TWO GROUPS OF CELLS, A BOOTSTRAP TEST PER TYPE
# On the log10 mean spine volume per cell (unitary, one synapse). The alternative is a
# 2-component Gaussian mixture. The nulls are one Gaussian, and one skew-normal, which
# has one mode but can skew. The statistic is 2 × (the log-likelihood of the mixture -
# that of the null). The chi-square test does not hold for a mixture, so each null is
# a parametric bootstrap (McLachlan 1987): `N_BOOT` sets of cells drawn from the
# fitted null, with both models fit again to each. The p-value is (1 + the replicates
# at or above the observed value) / (1 + `N_BOOT`). The skew-normal is not nested in
# the mixture, so its statistic can be negative, and the bootstrap still calibrates
# it. Ashman's D is the distance between the two means of the mixture over their
# pooled SD, and D > 2 is a clean split. The modes are those of the fitted mixture
# curve.
N_BOOT = 500
BOOT_CHUNK = 50
NULL_STYLES = {
    "gaussian": ("1 Gaussian", "tab:blue"),
    "skewnorm": ("Skew-normal", "tab:orange"),
}


def fit_null(x: np.ndarray, null: str):
    """The fitted null of `x`, as a frozen scipy distribution."""
    if null == "gaussian":
        return norm(x.mean(), x.std())
    return skewnorm(*skewnorm.fit(x))


def fit_mixture(x: np.ndarray) -> GaussianMixture:
    return GaussianMixture(2, n_init=5, random_state=0).fit(x[:, None])


def lr_statistic(x: np.ndarray, null: str) -> float:
    # One Gaussian is a mixture with two equal components, so it bounds the mixture
    # from below.
    mixture_ll = max(
        fit_mixture(x).score(x[:, None]) * len(x), fit_null(x, "gaussian").logpdf(x).sum()
    )
    return 2 * (mixture_ll - fit_null(x, null).logpdf(x).sum())


def bootstrap_chunk(x: np.ndarray, null: str, seed) -> list[float]:
    rng = np.random.default_rng(seed)
    fitted = fit_null(x, null)
    return [lr_statistic(fitted.rvs(len(x), random_state=rng), null) for _ in range(BOOT_CHUNK)]


mixture_x = {
    cell_type: np.log10(
        cells.query("cell_type == @cell_type and volume_one_synapse > 0")[
            "volume_one_synapse"
        ].to_numpy()
    )
    for cell_type in CELL_TYPES
}
boot_stash = Stash(
    "r2_11_4P_bimodality/mixture_bootstrap",
    stash_inputs,
    root_ids=sorted(cells.index.tolist()),
    cell_types=CELL_TYPES,
    value="volume_one_synapse",
    n_boot=N_BOOT,
)
if not boot_stash.exists():
    tasks = [
        (cell_type, null, chunk)
        for cell_type in CELL_TYPES
        for null in NULL_STYLES
        for chunk in range(N_BOOT // BOOT_CHUNK)
    ]
    seeds = np.random.SeedSequence(0).spawn(len(tasks))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=4, verbose=5)(
            delayed(bootstrap_chunk)(mixture_x[cell_type], null, seed)
            for (cell_type, null, _), seed in zip(tasks, seeds)
        )
    boot_stash.save(
        pl.DataFrame(
            [
                {"cell_type": cell_type, "null": null, "statistic": value}
                for (cell_type, null, _), values in zip(tasks, results)
                for value in values
            ]
        )
    )
boot = boot_stash.load().to_pandas()

fig, axs = plt.subplots(
    2, len(CELL_TYPES), figsize=(3.4 * len(CELL_TYPES), 6), layout="constrained",
    sharex="row", sharey="row", height_ratios=[2, 1],
)
jitter = np.random.default_rng(0)
grid = np.linspace(*np.quantile(np.concatenate(list(mixture_x.values())), [0.001, 0.999]), 400)
mixture_rows = []
for j, cell_type in enumerate(CELL_TYPES):
    x = mixture_x[cell_type]
    mixture = fit_mixture(x)
    means = mixture.means_.ravel()
    sds = np.sqrt(mixture.covariances_.ravel())
    ashman_d = np.sqrt(2) * abs(means[0] - means[1]) / np.sqrt((sds**2).sum())
    mixture_pdf = np.exp(mixture.score_samples(grid[:, None]))
    n_modes = int(((mixture_pdf[1:-1] > mixture_pdf[:-2]) & (mixture_pdf[1:-1] > mixture_pdf[2:])).sum())
    row = {
        "cell_type": cell_type, "n": len(x), "ashman_d": ashman_d, "modes": n_modes,
        "weights": np.round(np.sort(mixture.weights_), 2),
    }

    # Top: the cell means and the fits. The densities are per unit of log10 volume.
    ax = axs[0, j]
    sns.histplot(
        x=10**x, log_scale=True, stat="density", bins=60, element="step", color="grey",
        alpha=0.4, linewidth=0, ax=ax,
    )
    for null, (label, color) in NULL_STYLES.items():
        ax.plot(10**grid, fit_null(x, null).pdf(grid), color=color, linewidth=1.5, label=label)
    ax.plot(10**grid, mixture_pdf, color="black", linestyle="--", linewidth=1.5, label="2-Gaussian mixture")
    for weight, mean, sd in zip(mixture.weights_, means, sds):
        ax.plot(
            10**grid, weight * norm.pdf(grid, mean, sd), color="black", linestyle=":",
            linewidth=0.8,
        )
    ax.set_title(
        f"{TYPE_NAMES[cell_type]}, n={len(x):,}\nD={ashman_d:.2f}, {n_modes} "
        f"mode{'s' * (n_modes > 1)}",
        fontsize="medium",
    )
    ax.set(xlabel="", ylabel="Density" if j == 0 else "")

    # Bottom: per null, the bootstrap replicates as a jittered strip, and the observed
    # statistic as a diamond with its p-value.
    ax = axs[1, j]
    for k, (null, (label, color)) in enumerate(NULL_STYLES.items()):
        values = boot.query("cell_type == @cell_type and null == @null")["statistic"].to_numpy()
        observed = lr_statistic(x, null)
        p = (1 + (values >= observed).sum()) / (1 + len(values))
        p_text = f"p<{1 / (1 + len(values)):.3f}" if p == 1 / (1 + len(values)) else f"p={p:.3f}"
        ax.scatter(
            values, k + jitter.uniform(-0.2, 0.2, len(values)), s=4, color=color, alpha=0.3,
            linewidth=0, rasterized=True, label="Bootstrap replicate" if k == 0 else None,
        )
        ax.scatter(
            observed, k, marker="D", s=30, color=color, edgecolor="black", zorder=3,
            label="Observed" if k == 0 else None,
        )
        ax.text(observed, k + 0.3, p_text, ha="center", va="bottom", fontsize="x-small")
        row |= {f"lr_{null}": observed, f"p_{null}": p, f"boot_max_{null}": values.max()}
    mixture_rows.append(row)
axs[1, 0].set(
    yticks=range(len(NULL_STYLES)), yticklabels=[label for label, _ in NULL_STYLES.values()],
    ylim=(-0.5, len(NULL_STYLES) - 0.1),
)
axs[1, 0].set_xscale("symlog", linthresh=10)
axs[1, 0].set_xlim(-200, 5000)
axs[1, 0].xaxis.set_major_locator(ticker.FixedLocator([-100, 0, 1000]))
axs[1, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
axs[1, 0].xaxis.set_minor_locator(ticker.FixedLocator([-10, 10, 100]))
axs[0, 0].xaxis.set_major_locator(ticker.FixedLocator([7e7, 1e8, 1.5e8]))
axs[0, 0].xaxis.set_major_formatter(ticker.FixedFormatter(["0.7", "1", "1.5"]))
axs[0, 0].xaxis.set_minor_formatter(ticker.NullFormatter())
axs[0, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
axs[1, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", markerscale=2)
axs[0, len(CELL_TYPES) // 2].set_xlabel(
    "Mean spine volume per cell (10⁸ nm³), unitary spines with one synapse"
)
axs[1, len(CELL_TYPES) // 2].set_xlabel(
    "Likelihood ratio, 2 × (log-likelihood of the mixture - of the null)"
)
print(f"8b mixture tests, {N_BOOT} bootstrap replicates per null:")
print(pd.DataFrame(mixture_rows).set_index("cell_type").round(3).to_string())
save_matplotlib_figure(fig, "mixture_test", out_path, formats=["png"])

# %% 8c: PER CELL, THE WEIGHT OF THE LARGE MODE OF SPINE VOLUME
# The model: in a cell, the log10 volume of a unitary spine with one synapse is a
# mixture of a small and a large Gaussian. The two Gaussians are shared by the cells of
# a type, and the one parameter of a cell is w, the weight of the large Gaussian. The
# Gaussians and the weights are fit together, by EM over all the cells of the type on
# the bin centers of `volume_counts`, the counts of each cell per bin of log10 volume:
# the E-step splits the spines of a bin between the Gaussians by the w of their cell,
# and the M-step fits each w to its cell and the Gaussians to all the cells. EM starts
# from two points, the fit to the pooled spines of the type and two Gaussians at their
# quartiles, and the start with the higher likelihood is kept. EM stops when the total
# log-likelihood gains less than `JOINT_TOL`. The fits are stashed. With the Gaussians
# fixed, w of each cell is then fit again by maximum likelihood on the exact bin
# probabilities. The end bins are open, so they hold the spines past the range. The
# log-likelihood is concave in w, so EM on w alone reaches its one maximum. The SE of
# w is from the Fisher information. The overdispersion is the variance of w over the
# mean of SE². The shared weight of B in 8d is the one w that fits the pooled spines
# best.
SD_FLOOR = 0.05
centers = (volume_edges[1:] + volume_edges[:-1]) / 2


def fit_free(counts, components, w_start, free_sd: bool, n_iter=500):
    """Per row of `counts`, EM for a mixture of two Gaussians on the bin centers,
    from `components`. With `free_sd`, the SDs are fit too, each at least `SD_FLOOR`.
    Returns w and the (mean, SD) of the small and the large Gaussian, per row."""
    n_cells = len(counts)
    w = np.full(n_cells, w_start, dtype=float)
    (mean_s, sd_s), (mean_l, sd_l) = [
        (np.full(n_cells, mean, dtype=float), np.full(n_cells, sd, dtype=float))
        for mean, sd in components
    ]
    for _ in range(n_iter):
        dense_s = norm.pdf(centers, mean_s[:, None], sd_s[:, None])
        dense_l = norm.pdf(centers, mean_l[:, None], sd_l[:, None])
        p = np.maximum((1 - w)[:, None] * dense_s + w[:, None] * dense_l, 1e-300)
        large = counts * w[:, None] * dense_l / p
        small = counts - large
        n_large = large.sum(axis=1) + 1e-9
        n_small = small.sum(axis=1) + 1e-9
        w = n_large / counts.sum(axis=1)
        mean_l = (large * centers).sum(axis=1) / n_large
        mean_s = (small * centers).sum(axis=1) / n_small
        if free_sd:
            sd_l = np.sqrt((large * (centers - mean_l[:, None]) ** 2).sum(axis=1) / n_large)
            sd_s = np.sqrt((small * (centers - mean_s[:, None]) ** 2).sum(axis=1) / n_small)
            sd_l, sd_s = np.maximum(sd_l, SD_FLOOR), np.maximum(sd_s, SD_FLOOR)
    return w, (mean_s, sd_s), (mean_l, sd_l)


def bin_probabilities(mean: float, sd: float) -> np.ndarray:
    """The probability of each bin under a Gaussian, with open end bins."""
    cdf = norm.cdf(volume_edges, mean, sd)
    cdf[0], cdf[-1] = 0, 1
    return np.diff(cdf)


def fit_weights(counts: np.ndarray, p_small: np.ndarray, p_large: np.ndarray, n_iter=1000):
    """Per row of `counts`, the weight of the large Gaussian and its SE."""
    n = counts.sum(axis=1)
    w = np.full(len(counts), 0.5)
    for _ in range(n_iter):
        p = (1 - w)[:, None] * p_small + w[:, None] * p_large
        w_next = (counts * w[:, None] * p_large / p).sum(axis=1) / n
        change, w = np.abs(w_next - w).max(), w_next
    p = (1 - w)[:, None] * p_small + w[:, None] * p_large
    information = (counts * ((p_large - p_small) / p) ** 2).sum(axis=1)
    return w, 1 / np.sqrt(information), change


def mixture_moments(w, components):
    """Per weight `w`, the mean volume and the SD of log10 volume of the mixture."""
    ln10 = np.log(10)
    weights = [1 - w, w]
    # Per component, the mean and the second moment of volume (lognormal), and of
    # log10 volume.
    volume_mean = sum(
        k * 10**mean * np.exp((sd * ln10) ** 2 / 2) for k, (mean, sd) in zip(weights, components)
    )
    volume_square = sum(
        k * 10 ** (2 * mean) * np.exp(2 * (sd * ln10) ** 2)
        for k, (mean, sd) in zip(weights, components)
    )
    log_mean = sum(k * mean for k, (mean, _) in zip(weights, components))
    log_square = sum(k * (sd**2 + mean**2) for k, (mean, sd) in zip(weights, components))
    return volume_mean, volume_square - volume_mean**2, np.sqrt(log_square - log_mean**2)


JOINT_TOL = 1e-4
JOINT_MAX_ITER = 20_000


def fit_joint(counts: np.ndarray, components) -> dict:
    """EM for two Gaussians shared by the rows of `counts` and a weight per row, on
    the bin centers, from `components`."""
    counts = counts.astype(float)
    n = counts.sum(axis=1)
    per_bin = counts.sum(axis=0)
    (mean_s, sd_s), (mean_l, sd_l) = components
    w = np.full(len(counts), 0.5)
    previous = -np.inf
    for n_iter in range(JOINT_MAX_ITER):
        dense_s = norm.pdf(centers, mean_s, sd_s)
        dense_l = norm.pdf(centers, mean_l, sd_l)
        p = np.maximum((1 - w)[:, None] * dense_s + w[:, None] * dense_l, 1e-300)
        loglik = (counts * np.log(p)).sum()
        if loglik - previous < JOINT_TOL:
            break
        previous = loglik
        large = counts * w[:, None] * dense_l / p
        w = large.sum(axis=1) / n
        large = large.sum(axis=0)
        small = per_bin - large
        mean_l = (large * centers).sum() / large.sum()
        mean_s = (small * centers).sum() / small.sum()
        sd_l = np.sqrt((large * (centers - mean_l) ** 2).sum() / large.sum())
        sd_s = np.sqrt((small * (centers - mean_s) ** 2).sum() / small.sum())
    if mean_s > mean_l:
        (mean_s, sd_s), (mean_l, sd_l) = (mean_l, sd_l), (mean_s, sd_s)
    return dict(
        mean_s=mean_s, sd_s=sd_s, mean_l=mean_l, sd_l=sd_l, loglik=loglik, n_iter=n_iter
    )


def joint_fits(cell_type: str) -> list[dict]:
    """The joint fit of `cell_type` from each start."""
    ids = cells.index[(cells["cell_type"] == cell_type)].intersection(volume_counts.index)
    counts = volume_counts.loc[ids].to_numpy()
    pooled = counts.sum(axis=0, keepdims=True)
    quartiles = np.interp([0.25, 0.75], np.cumsum(pooled[0]) / pooled.sum(), centers)
    at_quartiles = [(quartiles[0], 0.2), (quartiles[1], 0.2)]
    _, *pooled_fit = fit_free(pooled, at_quartiles, 0.5, free_sd=True, n_iter=5000)
    starts = {
        "pooled": [(mean[0], sd[0]) for mean, sd in pooled_fit],
        "quartiles": at_quartiles,
    }
    return [
        {"cell_type": cell_type, "start": start}
        | {f"start_{k}": v for k, v in zip(["mean_s", "sd_s", "mean_l", "sd_l"], np.ravel(components))}
        | fit_joint(counts, components)
        for start, components in starts.items()
    ]


def per_cell_loglik(counts: np.ndarray, components) -> float:
    """The total log-likelihood of `counts` on the exact bin probabilities, with
    `components` fixed and w fit per row."""
    p_small, p_large = (bin_probabilities(*c) for c in components)
    w = fit_weights(counts, p_small, p_large)[0]
    p = (1 - w)[:, None] * p_small + w[:, None] * p_large
    return (counts * np.log(np.maximum(p, 1e-300))).sum()


joint_stash = Stash(
    "r2_11_4P_bimodality/joint_components",
    [singles_stash],
    root_ids=sorted(volume_counts.index.tolist()),
    cell_types=CELL_TYPES,
    volume_range=VOLUME_RANGE,
    n_bins=N_VOLUME_BINS,
    tol=JOINT_TOL,
    max_iter=JOINT_MAX_ITER,
)
if not joint_stash.exists():
    joint_stash.save(pd.DataFrame([fit for t in CELL_TYPES for fit in joint_fits(t)]))
joint = joint_stash.load().to_pandas()
print("8c joint fits, per start (log10 means and SDs):")
print(joint.round(4).to_string(index=False))

weights = []
type_components, type_pooled_weight = {}, {}
for cell_type in CELL_TYPES:
    ids = cells.index[(cells["cell_type"] == cell_type)].intersection(volume_counts.index)
    counts = volume_counts.loc[ids].to_numpy()
    best = joint.query("cell_type == @cell_type").sort_values("loglik").iloc[-1]
    components = [(best["mean_s"], best["sd_s"]), (best["mean_l"], best["sd_l"])]
    type_components[cell_type] = components
    p_small, p_large = (bin_probabilities(*c) for c in components)
    pooled_w = fit_weights(counts.sum(axis=0, keepdims=True), p_small, p_large)[0]
    type_pooled_weight[cell_type] = pooled_w[0]
    start = joint.query("cell_type == @cell_type and start == 'pooled'").iloc[0]
    pooled_components = [
        (start["start_mean_s"], start["start_sd_s"]), (start["start_mean_l"], start["start_sd_l"])
    ]
    gain = per_cell_loglik(counts, components) - per_cell_loglik(counts, pooled_components)
    print(
        f"8c {cell_type}: joint fit from the {best['start']} start; with w per cell, the "
        f"joint Gaussians beat the pooled ones by {gain:.0f} log-likelihood "
        f"({1000 * gain / counts.sum():.2f} per spine × 1000)"
    )
    w, se, change = fit_weights(counts, p_small, p_large)
    n = counts.sum(axis=1)
    volume_mean, volume_var, log_sd = mixture_moments(w, components)
    observed_log_mean = (counts * centers).sum(axis=1) / n
    observed_log_sd = np.sqrt((counts * (centers - observed_log_mean[:, None]) ** 2).sum(axis=1) / n)
    frame = pd.DataFrame(
        {
            "cell_type": cell_type, "w": w, "se": se, "n": n,
            "observed_mean": cells.loc[ids, "volume_one_synapse"].to_numpy(),
            "predicted_mean": volume_mean,
            "mean_z": (cells.loc[ids, "volume_one_synapse"].to_numpy() - volume_mean)
            / np.sqrt(volume_var / n),
            "observed_log_sd": observed_log_sd,
            "predicted_log_sd": log_sd,
        },
        index=ids,
    )
    weights.append(frame)
    print(
        f"8c {cell_type}: small {10 ** components[0][0]:.3g} nm³ (SD {components[0][1]:.3f} "
        f"log10), large {10 ** components[1][0]:.3g} nm³ (SD {components[1][1]:.3f}), "
        f"pooled weight {pooled_w[0]:.2f}; {len(ids):,} cells, EM change "
        f"{change:.1e}; w median {np.median(w):.2f}, SD {w.std():.3f}, median SE "
        f"{np.median(se):.3f}, overdispersion {w.var() / np.mean(se**2):.0f}; SD of mean z "
        f"{frame['mean_z'].std():.1f}; bins / rollup n "
        f"{np.median(n / cells.loc[ids, 'n_one_synapse'].to_numpy()):.4f}"
    )
weights = pd.concat(weights)

# The weights per type.
fig, axs = plt.subplots(
    1, len(CELL_TYPES), figsize=(3.2 * len(CELL_TYPES), 3), layout="constrained",
    sharex=True, sharey=True,
)
for ax, cell_type in zip(axs, CELL_TYPES):
    data = weights.query("cell_type == @cell_type")
    ax.hist(data["w"], bins=np.linspace(0, 1, 61), color="grey", alpha=0.6)
    ax.set_title(
        f"{TYPE_NAMES[cell_type]}, n={len(data):,}\nmedian SE {data['se'].median():.3f}",
        fontsize="medium",
    )
axs[0].set_ylabel("Cells")
axs[len(CELL_TYPES) // 2].set_xlabel("Weight of the large Gaussian, w")
save_matplotlib_figure(fig, "weights_by_type", out_path, formats=["png"])

# Example cells of the focus type, at even quantiles of w: the histogram of the cell and
# its fitted mixture, as densities per unit of log10 volume.
N_EXAMPLE_ROWS, N_EXAMPLE_COLUMNS = 4, 6
focus = weights.query("cell_type == @FOCUS_TYPE").sort_values("w")
quantiles = np.linspace(0.02, 0.98, N_EXAMPLE_ROWS * N_EXAMPLE_COLUMNS)
examples = focus.iloc[(quantiles * (len(focus) - 1)).round().astype(int)]
step = volume_edges[1] - volume_edges[0]
fig, axs = plt.subplots(
    N_EXAMPLE_ROWS, N_EXAMPLE_COLUMNS, figsize=(2.4 * N_EXAMPLE_COLUMNS, 1.9 * N_EXAMPLE_ROWS),
    layout="constrained", sharex=True, sharey=True,
)
grid = np.linspace(6.5, 9, 300)
components = type_components[FOCUS_TYPE]
for ax, (root_id, row) in zip(axs.flat, examples.iterrows()):
    counts = volume_counts.loc[root_id].to_numpy()
    ax.stairs(counts / counts.sum() / step, 10**volume_edges, fill=True, color="grey", alpha=0.5)
    parts = [
        k * norm.pdf(grid, mean, sd) for k, (mean, sd) in zip([1 - row["w"], row["w"]], components)
    ]
    ax.plot(10**grid, parts[0] + parts[1], color="black", linewidth=1.2)
    ax.text(
        0.03, 0.95, f"w={row['w']:.2f}\nn={row['n']:.0f}", transform=ax.transAxes, va="top",
        fontsize="x-small",
    )
    ax.set(xscale="log", xlim=(10**6.5, 10**9))
fig.supxlabel(VOLUME_LABEL + ", unitary spines with one synapse")
fig.supylabel("Density per log10 unit")
fig.suptitle(
    f"{FOCUS}: cells at even quantiles of w, with the fitted mixture (fixed Gaussians)"
)
save_matplotlib_figure(fig, f"weights_cell_fits_{FOCUS_TYPE}", out_path, formats=["png"])

# Per cell, the observed mean volume against the mean predicted from w.
fig, axs = plt.subplots(
    1, len(CELL_TYPES), figsize=(3.2 * len(CELL_TYPES), 3.4), layout="constrained",
)
for ax, cell_type in zip(axs, CELL_TYPES):
    data = weights.query("cell_type == @cell_type")
    ax.scatter(
        data["predicted_mean"], data["observed_mean"], s=2, color="black", alpha=0.2,
        linewidth=0, rasterized=True,
    )
    limits = np.quantile(
        np.concatenate([data["predicted_mean"], data["observed_mean"]]), [0.001, 0.999]
    )
    ax.plot(limits, limits, color="tab:red", linewidth=1)
    r = pearsonr(data["predicted_mean"], data["observed_mean"])[0]
    ax.text(0.03, 0.97, f"r={r:.2f}", transform=ax.transAxes, va="top", fontsize="small")
    ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits, title=TYPE_NAMES[cell_type])
axs[0].set_ylabel("Observed mean\nspine volume (nm³)")
axs[len(CELL_TYPES) // 2].set_xlabel("Predicted from w (nm³)")
save_matplotlib_figure(fig, "weights_predicted_moments", out_path, formats=["png"])

# One cell of the focus type, to show what the SD of log10 volume of a cell is. The
# cell has w near 0.5 and at least 1000 spines, at the 90th percentile of observed -
# predicted SD among those cells. The bars are the mean ± 1 SD of log10 volume, of the
# spines of the cell and of the fitted mixture.
candidates = focus.query("w.between(0.45, 0.55) and n >= 1000")
gap = candidates["observed_log_sd"] - candidates["predicted_log_sd"]
root_id = (gap - gap.quantile(0.9)).abs().idxmin()
row = focus.loc[root_id]
counts = volume_counts.loc[root_id].to_numpy()
log_mean = (counts * centers).sum() / counts.sum()
predicted_log_mean = (1 - row["w"]) * components[0][0] + row["w"] * components[1][0]
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.stairs(counts / counts.sum() / step, volume_edges, fill=True, color="grey", alpha=0.5, label="Spines of the cell")
parts = [k * norm.pdf(grid, mean, sd) for k, (mean, sd) in zip([1 - row["w"], row["w"]], components)]
ax.plot(grid, parts[0] + parts[1], color="tab:red", linewidth=1.5, label=f"Fitted mixture, w={row['w']:.2f}")
for part in parts:
    ax.plot(grid, part, color="tab:red", linestyle=":", linewidth=1)
top = 1.1 * max((counts / counts.sum() / step).max(), (parts[0] + parts[1]).max())
for height, center, sd, color, label in [
    (top, log_mean, row["observed_log_sd"], "black", "observed"),
    (top * 1.12, predicted_log_mean, row["predicted_log_sd"], "tab:red", "predicted"),
]:
    ax.errorbar(center, height, xerr=sd, color=color, capsize=4, marker="o", markersize=4)
    ax.text(
        center + sd + 0.03, height, f"{label}: SD {sd:.3f} (×{10 ** sd:.2f} per SD)",
        color=color, va="center", fontsize="small",
    )
ax.set(
    xlim=(6.6, 9.2), ylim=(0, top * 1.3), ylabel="Density per log10 unit",
    xlabel="log10 spine volume (nm³), unitary spines with one synapse",
    title=f"One {FOCUS} cell, n={row['n']:.0f} spines: mean ± 1 SD of log10 volume",
)
ax.legend(loc="center left", fontsize="small")
print(
    f"8c SD example {root_id}: w={row['w']:.3f}, observed SD {row['observed_log_sd']:.3f}, "
    f"predicted {row['predicted_log_sd']:.3f}, highest predicted SD for any w "
    f"{weights.query('cell_type == @FOCUS_TYPE')['predicted_log_sd'].max():.3f}"
)
save_matplotlib_figure(fig, f"weights_sd_example_{FOCUS_TYPE}", out_path, formats=["png"])

# %% 8d: A LADDER OF PER-CELL MODELS, SCORED BY HELD-OUT LIKELIHOOD WITHIN THE CELL
# Each model is a mixture of a small and a large Gaussian of log10 volume, fit to the
# counts per bin of one cell. They differ in what is free per cell; the rest is the
# fit of 8c for the type:
# - A, weight: w.
# - B, offset: c, a shift of both Gaussians, with the pooled weight of the type.
# - C, weight + offset: w and c.
# - D, free means: w and the two means. The SDs are those of the type.
# - E, full: w, the two means and the two SDs, each SD at least `SD_FLOOR`.
# A, B and C are fit by a grid search over w and c, D and E by EM on the bin centers.
# The score is held out within the cell: binomial thinning splits the spines of each
# cell into two halves, each model is fit to one half and scored on the other with the
# exact bin probabilities, and then the halves swap. A cell's score of a model is the
# sum over the two halves, so each spine counts once. The scores are stashed.
W_GRID = np.linspace(0, 1, 201)
C_GRID = np.linspace(-0.4, 0.4, 201)
MODELS = {
    "A": "A: weight",
    "B": "B: offset",
    "C": "C: weight + offset",
    "D": "D: free means",
    "E": "E: full",
}


def grid_best(counts: np.ndarray, log_p: np.ndarray) -> np.ndarray:
    """Per row of `counts`, the column of `log_p` (bins × parameters) that fits it
    best."""
    return np.concatenate(
        [
            (counts[start : start + 500] @ log_p).argmax(axis=1)
            for start in range(0, len(counts), 500)
        ]
    )


def cell_bin_probabilities(mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per row, the probability of each bin under a Gaussian, with open end bins."""
    cdf = norm.cdf(volume_edges[None, :], mean[:, None], sd[:, None])
    cdf[:, 0], cdf[:, -1] = 0, 1
    return np.diff(cdf, axis=1)


def free_log_p(counts, components, w_start, free_sd: bool) -> np.ndarray:
    """Per row of `counts`, the log probability of each bin under the mixture that
    `fit_free` fits to the row."""
    w, (mean_s, sd_s), (mean_l, sd_l) = fit_free(counts, components, w_start, free_sd)
    p = (1 - w)[:, None] * cell_bin_probabilities(mean_s, sd_s) + w[
        :, None
    ] * cell_bin_probabilities(mean_l, sd_l)
    return np.log(np.maximum(p, 1e-300))


def grid_log_p(cell_type: str) -> dict[str, np.ndarray]:
    """The log probabilities of the bins (bins × parameters) over the grids of A, B
    and C, for the Gaussians of `cell_type`."""
    components = type_components[cell_type]
    pooled_w = type_pooled_weight[cell_type]
    # Bins × c, per Gaussian.
    shifted = [
        np.stack([bin_probabilities(mean + c, sd) for c in C_GRID], axis=1)
        for mean, sd in components
    ]
    zero = np.argmin(np.abs(C_GRID))
    p_grid = (1 - W_GRID)[None, :, None] * shifted[0][:, None, :] + W_GRID[
        None, :, None
    ] * shifted[1][:, None, :]
    # Far in a tail, a bin of a shifted Gaussian can underflow to 0, and 0 spines ×
    # log 0 is NaN, so the probabilities have a floor.
    log_p = {
        "A": p_grid[:, :, zero],
        "B": (1 - pooled_w) * shifted[0] + pooled_w * shifted[1],
        "C": p_grid.reshape(N_VOLUME_BINS, -1),
    }
    return {model: np.log(np.maximum(p, 1e-300)) for model, p in log_p.items()}


def fitted_log_p(counts: np.ndarray, cell_type: str, grids: dict) -> tuple[dict, int]:
    """Per model, the log probabilities of the bins (rows × bins) of the model fit to
    each row of `counts`, and the number of C fits at the edge of the c grid."""
    fitted = {}
    for model, log_p in grids.items():
        best = grid_best(counts, log_p)
        fitted[model] = log_p[:, best].T
        if model == "C":
            edge_hits = np.isin(best % len(C_GRID), [0, len(C_GRID) - 1]).sum()
    for model, free_sd in [("D", False), ("E", True)]:
        fitted[model] = free_log_p(
            counts, type_components[cell_type], type_pooled_weight[cell_type], free_sd
        )
    return fitted, edge_hits


def ladder_scores(cell_type: str) -> pd.DataFrame:
    """Per cell of `cell_type`, the held-out log-likelihood of each model."""
    ids = weights.query("cell_type == @cell_type").index
    counts = volume_counts.loc[ids].to_numpy().astype(np.int64)
    grids = grid_log_p(cell_type)
    half = np.random.default_rng(0).binomial(counts, 0.5)
    scores = {model: np.zeros(len(ids)) for model in MODELS}
    edge_hits = 0
    for train, test in [(half, counts - half), (counts - half, half)]:
        fitted, hits = fitted_log_p(train, cell_type, grids)
        edge_hits += hits
        for model in MODELS:
            scores[model] += (test * fitted[model]).sum(axis=1)
    print(f"8d {cell_type}: {edge_hits} C fits at the edge of the c grid, over both halves")
    return pd.DataFrame(scores, index=ids).assign(cell_type=cell_type, n=counts.sum(axis=1))


ladder_stash = Stash(
    "r2_11_4P_bimodality/model_ladder",
    [singles_stash],
    root_ids=sorted(weights.index.tolist()),
    cell_types=CELL_TYPES,
    w_grid=len(W_GRID),
    c_grid=(C_GRID[0], C_GRID[-1], len(C_GRID)),
    sd_floor=SD_FLOOR,
    seed=0,
    components="joint",
)
if not ladder_stash.exists():
    ladder_stash.save(
        pd.concat([ladder_scores(t) for t in CELL_TYPES]).rename_axis("root_id").reset_index()
    )
ladder = ladder_stash.load().to_pandas().set_index("root_id")
ladder["w"] = weights.loc[ladder.index, "w"]

# Per type, the held-out gain of each model over A, in log-likelihood per spine, and
# the share of cells where the model beats A.
gains = {
    model: (ladder[model] - ladder["A"]) for model in MODELS if model != "A"
}
summary = pd.DataFrame(
    {
        f"{model} per spine (x1000)": 1000
        * gain.groupby(ladder["cell_type"]).sum()
        / ladder.groupby("cell_type")["n"].sum()
        for model, gain in gains.items()
    }
    | {
        f"{model} share of cells better": (gain > 0).groupby(ladder["cell_type"]).mean()
        for model, gain in gains.items()
    }
).loc[CELL_TYPES]
print("8d held-out gain over A:")
print(summary.round(3).to_string())

# Top: the summed gain per spine of each model over A. Bottom: per cell, the gain of
# B, C and D over A against w, as the median and quartiles in bins of w.
LADDER_COLORS = {"B": "tab:blue", "C": "tab:purple", "D": "tab:green", "E": "tab:orange"}
fig, axs = plt.subplots(
    2, len(CELL_TYPES), figsize=(3.2 * len(CELL_TYPES), 6), layout="constrained",
    sharey="row",
)
w_bins = np.linspace(0, 1, 26)
for j, cell_type in enumerate(CELL_TYPES):
    ax = axs[0, j]
    values = [summary.loc[cell_type, f"{model} per spine (x1000)"] for model in LADDER_COLORS]
    ax.bar(range(len(values)), values, color=list(LADDER_COLORS.values()))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xticks=range(len(values)), xticklabels=list(LADDER_COLORS),
        title=TYPE_NAMES[cell_type],
    )
    ax = axs[1, j]
    data = ladder.query("cell_type == @cell_type")
    which = np.clip(np.digitize(data["w"], w_bins), 1, len(w_bins) - 1)
    w_centers = (w_bins[1:] + w_bins[:-1]) / 2
    for model in ["B", "C", "D"]:
        gain = (data[model] - data["A"]).groupby(which)
        enough = gain.size() >= 30
        quartiles = gain.quantile([0.25, 0.5, 0.75]).unstack()[enough]
        x = w_centers[quartiles.index - 1]
        ax.plot(x, quartiles[0.5], color=LADDER_COLORS[model], label=MODELS[model])
        ax.fill_between(x, quartiles[0.25], quartiles[0.75], color=LADDER_COLORS[model], alpha=0.2, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlim=(0, 1))
axs[0, 0].set_ylabel("Held-out gain over A\nper spine (×1000)")
axs[1, 0].set_ylabel("Held-out gain over A\nper cell (log-likelihood)")
axs[1, len(CELL_TYPES) // 2].set_xlabel("w of the cell (model A, all spines)")
axs[1, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
save_matplotlib_figure(fig, "model_ladder", out_path, formats=["png"])

# %% 8e: THE MODELS OF 8d ON EXAMPLE CELLS
# Cells of the focus type at even quantiles of w, one per row, and the models, one per
# column. Each panel is the histogram of the cell and the model fit to all its spines,
# both as densities per log10 unit at the resolution of the bins, with A dashed for
# reference. The number is the held-out gain of the model over A from 8d.
EXAMPLE_QUANTILES = np.linspace(0.05, 0.95, 6)
MODEL_COLORS = {"A": "black"} | LADDER_COLORS
focus = weights.query("cell_type == @FOCUS_TYPE").sort_values("w")
example_ids = focus.index[(EXAMPLE_QUANTILES * (len(focus) - 1)).round().astype(int)]
example_counts = volume_counts.loc[example_ids].to_numpy()
example_fits, _ = fitted_log_p(example_counts, FOCUS_TYPE, grid_log_p(FOCUS_TYPE))
fig, axs = plt.subplots(
    len(example_ids), len(MODELS), figsize=(2.6 * len(MODELS), 1.9 * len(example_ids)),
    layout="constrained", sharex=True, sharey="row",
)
for i, root_id in enumerate(example_ids):
    counts = example_counts[i]
    for j, model in enumerate(MODELS):
        ax = axs[i, j]
        ax.stairs(counts / counts.sum() / step, 10**volume_edges, fill=True, color="grey", alpha=0.4)
        if model != "A":
            ax.stairs(
                np.exp(example_fits["A"][i]) / step, 10**volume_edges, color="black",
                linestyle="--", linewidth=0.8,
            )
        ax.stairs(
            np.exp(example_fits[model][i]) / step, 10**volume_edges, color=MODEL_COLORS[model],
            linewidth=1.5,
        )
        gain = ladder.loc[root_id, model] - ladder.loc[root_id, "A"]
        label = f"w={focus.loc[root_id, 'w']:.2f}\nn={counts.sum()}" if model == "A" else f"{gain:+.0f}"
        ax.text(0.03, 0.95, label, transform=ax.transAxes, va="top", fontsize="x-small")
        ax.set(xscale="log", xlim=(10**6.7, 10**9))
        if i == 0:
            ax.set_title(MODELS[model], fontsize="medium", color=MODEL_COLORS[model])
fig.supxlabel(VOLUME_LABEL + ", unitary spines with one synapse")
fig.supylabel("Density per log10 unit")
fig.suptitle(
    f"{FOCUS}: example cells at even quantiles of w. Number: held-out gain over A "
    "(log-likelihood, both halves)"
)
save_matplotlib_figure(fig, f"model_fits_cells_{FOCUS_TYPE}", out_path, formats=["png"])

# %% 9: 4P, MATCHED PAIRS ACROSS THE TWO CLUSTERS, AND PROOFREAD THALAMIC INPUT
# The cells with a cluster posterior of 0.9 or more. A pair is one small-spine
# (cluster 0) and one large-spine (cluster 1) cell. The pairs are the 1:1 assignment
# that minimizes the total soma distance, and pairs more than 50 µm apart drop. The
# outcomes per cell are the synapses from the proofread thalamic axons on the unitary
# spines with one synapse, as a count and as a share of all the synapses on those
# spines. The test is the Wilcoxon signed-rank
# test on the paired differences, which drops the pairs with no difference.
MIN_POSTERIOR = 0.9
CALIPER_UM = 50
POSITION = ["pt_position_um_x", "pt_position_um_y", "pt_position_um_z"]
labels = gmm_labels[FOCUS_TYPE].query("posterior >= @MIN_POSTERIOR")
print(f"9 4P cells with posterior >= {MIN_POSTERIOR}: {labels['cluster'].value_counts().to_dict()}")
small = cells.loc[labels.index[labels["cluster"] == 0], POSITION]
large = cells.loc[labels.index[labels["cluster"] == 1], POSITION]
distance = cdist(small.to_numpy(), large.to_numpy())
# A distance past the caliper costs more than any set of allowed pairs, so the
# assignment avoids it where it can.
rows, columns = linear_sum_assignment(np.where(distance <= CALIPER_UM, distance, 1e9))
matched = distance[rows, columns] <= CALIPER_UM
pairs = pd.DataFrame(
    {
        "small": small.index[rows[matched]],
        "large": large.index[columns[matched]],
        "distance_um": distance[rows, columns][matched],
    }
)
print(
    f"9 {len(pairs)} pairs within {CALIPER_UM} µm, "
    f"median distance {pairs['distance_um'].median():.1f} µm"
)
for name, column in [("small", "small"), ("large", "large")]:
    means = cells.loc[pairs[column], POSITION].mean().round(1).to_dict()
    print(f"   mean soma position, {name}-spine cells in pairs: {means}")

thalamic = (
    Tables["proofread_pre_synapses"]
    .lazy()
    .filter(where, pl.col("pre_broad_type") == "thalamic", pl.col("tag") == "spine")
    .with_columns(spine_flags("spine_"))
    .group_by("post_pt_root_id")
    # A signed type, so that the paired differences can be negative.
    .agg(
        n_thalamic=(pl.col("unitary").fill_null(False) & (pl.col("n_synapses_class") == 1))
        .sum()
        .cast(pl.Int64),
        n_thalamic_all_spines=pl.len().cast(pl.Int64),
    )
    .collect()
    .to_pandas()
    .set_index("post_pt_root_id")
)
cells[thalamic.columns] = thalamic.reindex(cells.index, fill_value=0)
cells["p_thalamic"] = cells["n_thalamic"] / cells["n_one_synapse"]
cells["p_thalamic_all_spines"] = cells["n_thalamic_all_spines"] / cells["post_spine_synapses"]
OUTCOMES = {
    "n_thalamic": "Synapses from proofread\nthalamic axons\n(unitary, one synapse)",
    "p_thalamic": "Share of synapses from\nproofread thalamic axons\n(unitary, one synapse)",
}
# The same over all the spine synapses, with no unitary or synapse-count filter.
ALL_SPINE_OUTCOMES = {
    "n_thalamic_all_spines": "Synapses from proofread\nthalamic axons\n(all spine synapses)",
    "p_thalamic_all_spines": "Share of synapses from\nproofread thalamic axons\n(all spine synapses)",
}


def compare_pairs(pairs: pd.DataFrame, outcomes: dict, filename: str) -> pd.DataFrame:
    """Per outcome (a column of `cells`), the values of both cells of each pair,
    the summary of the paired differences and the Wilcoxon signed-rank test, and a
    figure of the pair distances and the paired differences."""
    pairs = pairs.copy()
    fig, axs = plt.subplots(
        1, len(outcomes) + 1, figsize=(4.3 * (len(outcomes) + 1), 3.8),
        layout="constrained",
    )
    axs[0].set_xlabel("Soma distance in pair (µm)")
    line_hist(axs[0], pairs["distance_um"], 40, log=False, stat="count", color="black")
    for ax, (outcome, label) in zip(axs[1:], outcomes.items()):
        small_values = cells.loc[pairs["small"], outcome].to_numpy(dtype=float)
        large_values = cells.loc[pairs["large"], outcome].to_numpy(dtype=float)
        pairs[f"{outcome}_small"], pairs[f"{outcome}_large"] = small_values, large_values
        difference = large_values - small_values
        result = wilcoxon(large_values, small_values)
        print(
            f"{filename} {outcome}: mean small {small_values.mean():.4g}, "
            f"mean large {large_values.mean():.4g}, "
            f"median difference (large - small) {np.median(difference):.4g}, "
            f"large > small in {(difference > 0).mean():.2f}, "
            f"large < small in {(difference < 0).mean():.2f}, "
            f"no difference in {(difference == 0).sum()} pairs, "
            f"Wilcoxon statistic {result.statistic:.4g}, p={result.pvalue:.3g}"
        )
        is_count = np.all(np.mod(difference, 1) == 0)
        bins = np.arange(difference.min(), difference.max() + 2) - 0.5 if is_count else 60
        ax.set_xlabel(label)
        line_hist(ax, difference, bins, log=False, stat="count", color="black")
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    fig.suptitle(
        f"{FOCUS}: {len(pairs)} matched pairs, large-spine minus small-spine cell"
    )
    save_matplotlib_figure(fig, filename, out_path, formats=["png"])
    return pairs


pairs = compare_pairs(pairs, OUTCOMES, f"{FOCUS_TYPE}_matched_thalamic")

# %% 10: 4P, THE SAME PAIRS, AND THE SOMA AND SHAFT SYNAPSES
# Most inhibitory input onto these cells lands on the soma and the dendritic shafts.
# The shares are of the known synapses (spine, shaft and soma), as in `post_p_*`.
cells["post_soma_shaft_synapses"] = cells["post_soma_synapses"] + cells["post_shaft_synapses"]
cells["post_p_soma_shaft_synapse"] = (
    cells["post_soma_shaft_synapses"] / cells["post_total_known_synapses"]
)
SOMA_SHAFT_OUTCOMES = {
    "post_soma_synapses": "Soma synapses",
    "post_shaft_synapses": "Shaft synapses",
    "post_soma_shaft_synapses": "Soma + shaft synapses",
    "post_p_soma_synapse": "Share of known synapses\non the soma",
    "post_p_shaft_synapse": "Share of known synapses\non shafts",
    "post_p_soma_shaft_synapse": "Share of known synapses\non soma + shafts",
}
pairs = compare_pairs(pairs, SOMA_SHAFT_OUTCOMES, f"{FOCUS_TYPE}_matched_soma_shaft")

# %% 11: 4P, THE OUTCOMES PER CLUSTER, BEFORE ANY MATCHING
# All the 4P cells with a cluster posterior of 0.9 or more.
CLASS_NAMES = {0: "Small spine", 1: "Large spine"}
classes = cells.loc[labels.index].assign(
    group=labels["cluster"].map(CLASS_NAMES)
)

# %% 11a: 4P, SPINE VOLUME HISTOGRAMS OF 30 COLUMN CELLS PER CLUSTER
# Per cell, the volumes of its unitary spines with one synapse, from
# `spine_census_features`. The cells are random column cells of each cluster, sorted
# by their mean spine volume. The dashed line is the mean of the cell.
N_HIST_CELLS = 30
HIST_COLUMNS = 6
column_pool = classes[classes["is_column"]]
hist_ids = {}
for name in CLASS_NAMES.values():
    group = column_pool.query("group == @name")
    print(f"11a: {len(group)} {name} column cells to sample from")
    hist_ids[name] = (
        group.sample(min(N_HIST_CELLS, len(group)), random_state=0)
        .sort_values("volume_one_synapse")
        .index.tolist()
    )
hist_spines = (
    Tables["spine_census_features"]
    .lazy()
    .filter(pl.col("post_pt_root_id").is_in(pl.Series(sum(hist_ids.values(), [])).implode()))
    .with_columns(spine_flags(""))
    .filter(pl.col("unitary").fill_null(False), pl.col("n_synapses_class") == 1)
    .select("post_pt_root_id", pl.col("size_nm3").alias("volume"))
    .collect()
    .to_pandas()
)
volumes = dict(tuple(hist_spines.groupby("post_pt_root_id")["volume"]))
# Sanity check: the spine table gives the cell means of the synapse rollup.
ratio = pd.Series({r: v.mean() for r, v in volumes.items()}) / cells["volume_one_synapse"]
print(f"11a mean volume / volume_one_synapse: {ratio.min():.4f} to {ratio.max():.4f}")
hist_edges = log_edges(hist_spines["volume"], n=40)
rows_per_class = int(np.ceil(N_HIST_CELLS / HIST_COLUMNS))
fig = plt.figure(figsize=(2.2 * HIST_COLUMNS, 2.4 * rows_per_class), layout="constrained")
hist_axes = []
for subfig, (cluster, name) in zip(fig.subfigures(2, 1), CLASS_NAMES.items()):
    axs = subfig.subplots(rows_per_class, HIST_COLUMNS, sharex=True, sharey=True)
    for ax in axs.flat:
        ax.axis("off")
    for ax, root_id in zip(axs.flat, hist_ids[name]):
        ax.axis("on")
        values = volumes[root_id]
        line_hist(ax, values, hist_edges, color=CLUSTER_PALETTE[cluster], fill=True, alpha=0.5)
        ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1)
        ax.text(
            0.97, 0.95, f"n={len(values)}", transform=ax.transAxes, ha="right", va="top",
            fontsize="x-small",
        )
        ax.set(xlabel="", xlim=hist_edges[[0, -1]])
        ax.tick_params(which="both", labelsize="x-small")
        hist_axes.append(ax)
    for ax in axs[:-1].flat:
        ax.tick_params(which="both", labelbottom=False)
    subfig.suptitle(f"{name} cells", color=CLUSTER_PALETTE[cluster])
# One y scale for both clusters.
top = max(ax.get_ylim()[1] for ax in hist_axes)
for ax in hist_axes:
    ax.set_ylim(0, top)
fig.supxlabel(f"{VOLUME_LABEL}, unitary spines with one synapse")
fig.suptitle(f"{FOCUS}: {N_HIST_CELLS} random column cells per cluster, by mean volume")
save_matplotlib_figure(fig, f"{FOCUS_TYPE}_class_cell_volume_hists_column", out_path, formats=["png"])

# %% 11b: 4P, SPINE VOLUME VS SYNAPSE SIZE OF THE SAME CELLS
# Per synapse on the unitary spines with one synapse, for the cells of 11a, in the
# same order. The marginals are density histograms, and r is the log-log Pearson r.
hist_where = pl.col("post_pt_root_id").is_in(pl.Series(sum(hist_ids.values(), [])).implode())
hist_synapses = (
    census_synapses(hist_where)
    .with_columns(spine_flags("spine_"))
    .filter(
        pl.col("tag") == "spine",
        pl.col("unitary").fill_null(False),
        pl.col("n_synapses_class") == 1,
    )
    .select(
        "post_pt_root_id",
        pl.col("spine_size_nm3").alias("volume"),
        size=synapse_size_nm3(pl.col("size")),
    )
    .to_pandas()
)
cell_synapses = dict(tuple(hist_synapses.groupby("post_pt_root_id")))
size_edges = log_edges(hist_synapses["size"], n=40)
fig = plt.figure(figsize=(1.9 * HIST_COLUMNS, 2 * 1.9 * rows_per_class + 1))
joints = []
for subfig, (cluster, name) in zip(fig.subfigures(2, 1), CLASS_NAMES.items()):
    color = CLUSTER_PALETTE[cluster]
    axs = subfig.subplots(
        rows_per_class, HIST_COLUMNS, sharex=True, sharey=True,
        gridspec_kw=dict(left=0.08, right=0.94, bottom=0.04, top=0.92, wspace=0.4, hspace=0.4),
    )
    for ax in axs.flat:
        ax.axis("off")
    for ax, root_id in zip(axs.flat, hist_ids[name]):
        ax.axis("on")
        data = cell_synapses[root_id]
        top = ax.inset_axes([0, 1.04, 1, 0.25], sharex=ax)
        right = ax.inset_axes([1.04, 0, 0.25, 1], sharey=ax)
        for marginal in (top, right):
            marginal.spines[["top", "right"]].set_visible(False)
        ax.scatter(
            data["volume"], data["size"], s=1, alpha=0.3, color=color, linewidth=0,
            rasterized=True,
        )
        line_hist(top, data["volume"], hist_edges, color=color, fill=True, alpha=0.5)
        line_hist(
            right, data["size"], size_edges, vertical=True, color=color, fill=True,
            alpha=0.5,
        )
        ax.text(
            0.04, 0.96, f"r={log_r(data['volume'], data['size']):.2f}",
            transform=ax.transAxes, va="top", fontsize="xx-small",
        )
        ax.set(
            xscale="log", yscale="log", xlim=hist_edges[[0, -1]], ylim=size_edges[[0, -1]],
            xlabel="", ylabel="",
        )
        ax.tick_params(
            which="both", direction="in", labelsize="xx-small", width=0.8, color="0.4",
            labelcolor="0.4",
        )
        ax.tick_params(which="major", length=4)
        ax.tick_params(which="minor", length=2)
        joints.append((ax, top, right))
    for ax in axs[:-1].flat:
        ax.tick_params(which="both", labelbottom=False)
    for ax in axs[:, 1:].flat:
        ax.tick_params(which="both", labelleft=False)
    subfig.suptitle(f"{name} cells", color=color, y=0.995)
hide_marginal_labels(joints)
for _, top, right in joints:
    for marginal in (top, right):
        marginal.tick_params(which="both", length=0)
fig.supxlabel(f"{VOLUME_LABEL}, unitary spines with one synapse", y=-0.01)
fig.supylabel(SIZE_LABEL, x=0.01)
fig.suptitle(
    f"{FOCUS}: {N_HIST_CELLS} random column cells per cluster, by mean volume", y=1.02
)
save_matplotlib_figure(fig, f"{FOCUS_TYPE}_class_cell_joints_column", out_path, formats=["png"])


# %%
def class_marginals(
    outcomes: dict,
    filename: str,
    pool: pd.DataFrame | None = None,
    cells_label: str = "cells",
    n_bins: int | None = None,
) -> None:
    """Per outcome, the density of each cluster, and the per-cluster summary, over the
    rows of `pool` (all of `classes` by default). With `n_bins`, every outcome has
    that many bins, also a count."""
    pool = classes if pool is None else pool
    fig, axs = plt.subplots(
        1, len(outcomes), figsize=(4 * len(outcomes), 3.5), layout="constrained"
    )
    for ax, (outcome, label) in zip(axs, outcomes.items()):
        values = pool[outcome]
        # One bin per value only for a count with a small range.
        is_count = np.all(np.mod(values, 1) == 0) and values.max() <= 100 and n_bins is None
        sns.histplot(
            pool, x=outcome, hue="group", hue_order=list(CLASS_NAMES.values()),
            palette=dict(zip(CLASS_NAMES.values(), CLUSTER_PALETTE.values())),
            stat="density", common_norm=False, element="poly", fill=False,
            discrete=is_count, bins="auto" if is_count else n_bins or 50, ax=ax,
            legend=ax is axs[-1],
        )
        ax.set_xlabel(label)
        hide_count_axis(ax)
        if ax.get_legend() is not None:
            ax.get_legend().set_title(None)
        summary = pool.groupby("group")[outcome].agg(["count", "mean", "median"])
        print(f"{filename} {outcome}:\n{summary.round(4)}")
    fig.suptitle(f"{FOCUS}: {cells_label} with cluster posterior >= {MIN_POSTERIOR}")
    save_matplotlib_figure(fig, filename, out_path, formats=["png"])


class_marginals(OUTCOMES, f"{FOCUS_TYPE}_class_thalamic")
class_marginals(
    OUTCOMES, f"{FOCUS_TYPE}_class_thalamic_column", classes[classes["is_column"]],
    "column cells", n_bins=15,
)
class_marginals(ALL_SPINE_OUTCOMES, f"{FOCUS_TYPE}_class_thalamic_all_spines")
class_marginals(
    ALL_SPINE_OUTCOMES, f"{FOCUS_TYPE}_class_thalamic_all_spines_column",
    classes[classes["is_column"]], "column cells", n_bins=15,
)
class_marginals(SOMA_SHAFT_OUTCOMES, f"{FOCUS_TYPE}_class_soma_shaft")

# %% 12: 4P, CELL FEATURES PER CLUSTER
# `nuc_volume` is from `cell_info`, in µm³. The shares are of the known synapses.
class_marginals(
    {
        "pt_position_um_y": "Soma depth (µm)",
        "post_total_synapses": "Input synapses",
        "n_one_synapse": "Synapses on unitary\nspines with one synapse",
        "post_soma_synapses": "Soma synapses",
        "p_one_synapse": "Share of known synapses on\nunitary spines with one synapse",
        "post_p_soma_synapse": "Share of known synapses\non the soma",
        "nuc_volume": "Nucleus volume (µm³)",
    },
    f"{FOCUS_TYPE}_class_cell_features",
)

# %% 13: 4P, SOMA POSITIONS FROM ABOVE (X-Z PLANE), BY CLUSTER
# The cells under the posterior cutoff are grey.
uncertain = gmm_labels[FOCUS_TYPE].query("posterior < @MIN_POSTERIOR").index
fig, ax = plt.subplots(1, 1, figsize=(8, 5), layout="constrained")
ax.scatter(
    cells.loc[uncertain, "pt_position_um_x"], cells.loc[uncertain, "pt_position_um_z"],
    s=4, color="lightgrey", linewidth=0, label=f"Posterior < {MIN_POSTERIOR}",
    rasterized=True,
)
for cluster, name in CLASS_NAMES.items():
    group = classes.query("group == @name")
    ax.scatter(
        group["pt_position_um_x"], group["pt_position_um_z"], s=4, alpha=0.6,
        color=CLUSTER_PALETTE[cluster], linewidth=0, label=name, rasterized=True,
    )
ax.set(xlabel="Soma x (µm)", ylabel="Soma z (µm)", aspect="equal", title=FOCUS)
ax.legend(markerscale=3)
save_matplotlib_figure(fig, f"{FOCUS_TYPE}_class_top_down", out_path, formats=["png"])

# %% 14: 4P, SPINE VOLUME VS SYNAPSE SIZE PER SYNAPSE, BY THE CLUSTER OF THE CELL
# All the synapses on the unitary spines with one synapse. The scatter shows 20,000
# synapses per cluster, and the marginals all of them. The trend line per cluster is
# the least-squares line of log10 size on log10 volume, over all its synapses.
N_SHOWN = 20_000
class_singles = singles.merge(
    classes[["group"]], left_on="post_pt_root_id", right_index=True, how="inner"
)
fig, joints = joint_axes(1, width=5)
ax, top, right = joints[0]
for cluster, name in CLASS_NAMES.items():
    data = class_singles.query("group == @name")
    shown = data.sample(min(N_SHOWN, len(data)), random_state=0)
    color = CLUSTER_PALETTE[cluster]
    ax.scatter(
        shown["volume"], shown["size"], s=1, alpha=0.2, color=color, linewidth=0,
        rasterized=True,
    )
    line_hist(top, data["volume"], x_edges, color=color, linewidth=1.5)
    line_hist(
        right, data["size"], y_edges, vertical=True, color=color, linewidth=1.5,
        label=f"{name} cells",
    )
    # In float64: `volume` is float32, and a float32 fit is wrong at this n.
    slope, intercept = np.polyfit(
        np.log10(data["volume"].to_numpy(dtype=float)),
        np.log10(data["size"].to_numpy(dtype=float)),
        1,
    )
    line_x = np.geomspace(*data["volume"].quantile([0.01, 0.99]), 50)
    ax.plot(line_x, 10**intercept * line_x**slope, color=color, linewidth=2)
    print(f"14 {name}: log10 size = {slope:.3f} log10 volume + {intercept:.3f}")
    ratio = np.log(data["volume"]) - np.log(data["size"])
    print(
        f"14 {name}: {len(data):,} synapses on {data['post_pt_root_id'].nunique():,} "
        f"cells, log-log r={log_r(data['volume'], data['size']):.2f}, "
        f"median volume / size {np.exp(ratio.median()):.1f}"
    )
ax.set(
    xscale="log", yscale="log", xlim=x_edges[[0, -1]], ylim=y_edges[[0, -1]],
    xlabel=VOLUME_LABEL, ylabel=SIZE_LABEL,
)
hide_marginal_labels(joints)
top.set_title(f"{FOCUS}: unitary spines with one synapse", pad=4)
ax.legend(*right.get_legend_handles_labels(), loc="lower right", markerscale=3)
save_matplotlib_figure(fig, f"{FOCUS_TYPE}_class_single_spine_joint", out_path, formats=["png"])

# %% 14b: THE SAME, WITH A LEAST-SQUARES LINE PER CELL
# Per cell, the line of log10 size on log10 volume over all its synapses on the
# unitary spines with one synapse, from the sums stashed per cell. A cell line spans
# the 5th to 95th percentile of the log volume of the cell. The thick, darker line per
# cluster is the fit over all the synapses of its cells, pooled, and spans the median
# of those percentiles. The scatter and the marginals are those of 14.
fit_stash = Stash("r2_11_4P_bimodality/cell_fit_sums", stash_inputs, **stash_params)


def fit_sums(part: pl.LazyFrame) -> pl.LazyFrame:
    x = pl.col("spine_size_nm3").cast(pl.Float64).log10()
    y = synapse_size_nm3(pl.col("size").cast(pl.Float64)).log10()
    return (
        part.with_columns(spine_flags("spine_"))
        .filter(
            pl.col("tag") == "spine",
            pl.col("unitary").fill_null(False),
            pl.col("n_synapses_class") == 1,
        )
        .group_by("post_pt_root_id")
        .agg(
            n=pl.len(),
            x_sum=x.sum(),
            y_sum=y.sum(),
            xx_sum=(x * x).sum(),
            xy_sum=(x * y).sum(),
            x_low=x.quantile(0.05),
            x_high=x.quantile(0.95),
        )
    )


def fit_line(sums) -> tuple[float, float]:
    """The slope and intercept of the least-squares line, from the sums of a fit."""
    n, x, y, xx, xy = (sums[k] for k in ("n", "x_sum", "y_sum", "xx_sum", "xy_sum"))
    slope = (n * xy - x * y) / (n * xx - x * x)
    return slope, (y - slope * x) / n


if not fit_stash.exists():
    fit_stash.save(census_synapses(where, then=fit_sums))
cell_fits = (
    fit_stash.load()
    .to_pandas()
    .set_index("post_pt_root_id")
    .join(classes[["group"]], how="inner")
)
cell_fits["slope"], cell_fits["intercept"] = fit_line(cell_fits)
fig, joints = joint_axes(1, width=5)
ax, top, right = joints[0]
for cluster, name in CLASS_NAMES.items():
    color = CLUSTER_PALETTE[cluster]
    data = class_singles.query("group == @name")
    shown = data.sample(min(N_SHOWN, len(data)), random_state=0)
    ax.scatter(
        shown["volume"], shown["size"], s=1, alpha=0.1, color=color, linewidth=0,
        rasterized=True,
    )
    line_hist(top, data["volume"], x_edges, color=color, linewidth=1.5)
    line_hist(
        right, data["size"], y_edges, vertical=True, color=color, linewidth=1.5,
        label=f"{name} cells",
    )
    fits = cell_fits.query("group == @name")
    for fit in fits.itertuples():
        line_x = np.array([fit.x_low, fit.x_high])
        ax.plot(
            10**line_x, 10 ** (fit.intercept + fit.slope * line_x), color=color,
            linewidth=0.3, alpha=0.03,
        )
    slope, intercept = fit_line(fits[["n", "x_sum", "y_sum", "xx_sum", "xy_sum"]].sum())
    line_x = np.array([fits["x_low"].median(), fits["x_high"].median()])
    ax.plot(
        10**line_x, 10 ** (intercept + slope * line_x), color=np.array(color) * 0.6,
        linewidth=2.5,
    )
    print(
        f"14b {name}: {len(fits)} cells, pooled log10 size = {slope:.3f} log10 volume "
        f"+ {intercept:.3f}, cell slopes median {fits['slope'].median():.3f} "
        f"(IQR {fits['slope'].quantile(0.25):.3f} to {fits['slope'].quantile(0.75):.3f})"
    )
ax.set(
    xscale="log", yscale="log", xlim=x_edges[[0, -1]], ylim=y_edges[[0, -1]],
    xlabel=VOLUME_LABEL, ylabel=SIZE_LABEL,
)
hide_marginal_labels(joints)
top.set_title(f"{FOCUS}: unitary spines with one synapse, a line per cell", pad=4)
ax.legend(*right.get_legend_handles_labels(), loc="lower right", markerscale=3)
save_matplotlib_figure(
    fig, f"{FOCUS_TYPE}_class_single_spine_joint_cell_fits", out_path, formats=["png"]
)

# %% 15: 4P, DENDRITE MUGSHOTS OF 30 CELLS PER CLUSTER
# As in `scripts/spine_density/mugshots.py`: the dendrite and soma skeleton in the
# streamline frame. The horizontal axis is the first principal axis in x and z, and
# the vertical axis is the height above the soma. All panels share one scale. The
# skeleton files are cached by `cached_skeleton`, and the 2D points are stashed.
N_MUGSHOTS = 30
MUGSHOT_COLUMNS = 10


def mugshot_points(root_id: int) -> pl.DataFrame:
    """Per positional vertex of the dendrite skeleton of `root_id`: its parent, its
    2D position in µm with the soma at (0, 0), and the soma depth."""
    skel = dendrite_skeleton(root_id)
    geometry = vertex_geometry(skel, "streamline")
    xyz = geometry.select("x_um", "y_um", "z_um").to_numpy()
    root = skel.root_positional
    horizontal = PCA(n_components=1).fit_transform(xyz[:, [0, 2]] - xyz[root, [0, 2]])[:, 0]
    # The PCA sign is arbitrary. Put the bigger half of the arbor on the right.
    if np.median(horizontal) < 0:
        horizontal = -horizontal
    return pl.DataFrame(
        {
            "root_id": root_id,
            "vertex": np.arange(len(xyz)),
            "parent": np.asarray(skel.parent_node_array),
            "x_um": horizontal - horizontal[root],
            # Down is deeper, as the frame y.
            "y_um": -geometry["height_above_soma_um"].to_numpy(),
            "soma_depth_um": xyz[root, 1],
        }
    )


def mugshot_gallery(pool: pd.DataFrame, filename: str, title: str) -> None:
    """The mugshots of `N_MUGSHOTS` random cells per cluster from `pool` (rows of
    `classes`), one class after the other, each sorted by soma depth."""
    mugshot_ids = {}
    for name in CLASS_NAMES.values():
        group = pool.query("group == @name")
        print(f"15 {filename}: {len(group)} {name} cells to sample from")
        mugshot_ids[name] = group.sample(min(N_MUGSHOTS, len(group)), random_state=0).index.tolist()
    stash = Stash(
        "r2_11_4P_bimodality/mugshots",
        root_ids=sorted(sum(mugshot_ids.values(), [])),
        frame="streamline",
    )
    if not stash.exists():
        with ThreadPoolExecutor(max_workers=8) as executor:
            stash.save(pl.concat(executor.map(mugshot_points, stash.params["root_ids"])))
    shots = {
        root_id: frame.to_pandas()
        for (root_id,), frame in stash.load().partition_by("root_id", as_dict=True).items()
    }
    # The panel fits the 95th percentile of the cell extents. Larger cells draw past it.
    half_width = np.quantile([shot["x_um"].abs().max() for shot in shots.values()], 0.95)
    up = np.quantile([-shot["y_um"].min() for shot in shots.values()], 0.95)
    down = np.quantile([shot["y_um"].max() for shot in shots.values()], 0.95)
    rows_per_class = int(np.ceil(N_MUGSHOTS / MUGSHOT_COLUMNS))
    n_rows = 2 * rows_per_class
    panel_width = 16 / MUGSHOT_COLUMNS
    panel_height = panel_width * (up + down) / (2 * half_width)
    fig, axs = plt.subplots(
        n_rows, MUGSHOT_COLUMNS, figsize=(16, n_rows * panel_height), squeeze=False
    )
    fig.subplots_adjust(left=0.03, right=1, bottom=0, top=0.95, wspace=0, hspace=0)
    for ax in axs.flat:
        ax.axis("off")
    for k, (cluster, name) in enumerate(CLASS_NAMES.items()):
        ids = sorted(mugshot_ids[name], key=lambda r: shots[r]["soma_depth_um"].iloc[0])
        class_axes = axs[k * rows_per_class : (k + 1) * rows_per_class].flat
        for ax, root_id in zip(class_axes, ids):
            shot = shots[root_id]
            has_parent = shot["parent"].to_numpy() >= 0
            edges = np.column_stack(
                [shot["vertex"].to_numpy()[has_parent], shot["parent"].to_numpy()[has_parent]]
            )
            add_skeleton(
                ax, edges, shot[["x_um", "y_um"]].to_numpy(),
                color=CLUSTER_PALETTE[cluster], linewidths=1.5, clip_on=False,
            )
            ax.set(xlim=(-half_width, half_width), ylim=(down, -up), aspect="equal")
            ax.text(
                0, min(shot["y_um"].max(), down), str(root_id)[-6:], ha="center",
                va="top", fontsize=5, color="0.5", clip_on=False,
            )
        middle = axs[k * rows_per_class + rows_per_class // 2, 0]
        middle.text(
            -0.05, 0.5, f"{name} cells", transform=middle.transAxes, rotation=90,
            ha="right", va="center", color=CLUSTER_PALETTE[cluster],
        )
    fig.suptitle(title)
    save_matplotlib_figure(fig, filename, out_path, formats=["png"], dpi=300)


mugshot_gallery(
    classes,
    f"{FOCUS_TYPE}_class_mugshots",
    f"{FOCUS}: dendrites of {N_MUGSHOTS} random cells per cluster "
    "(posterior >= 0.9), sorted by soma depth",
)
mugshot_gallery(
    classes[classes["is_column"]],
    f"{FOCUS_TYPE}_class_mugshots_column",
    f"{FOCUS}: dendrites of {N_MUGSHOTS} random column cells per cluster "
    "(posterior >= 0.9), sorted by soma depth",
)

# %% 16: 4P, SPINE DENSITY AGAINST PATH DISTANCE, BY CLUSTER
# The cells of `classes` with a local skeleton file: the column cells, and the other
# cells that a mugshot gallery (here or in an earlier run) downloaded. The spines are
# the unitary spines with one synapse, and a spine sits on the vertex of its synapse,
# as in `scripts/spine_density/spine_density.py`. A shaft synapse sits on its vertex
# too. Only the synapses on the dendrite (`dendrite_mask`) count, and the soma vertex
# drops. Per cell, the density of a bin is its count (or sum of spine volume or synapse
# size) over its cable, over the whole dendrite. Bins with less than half a bin of
# cable drop. The thick lines are the cable-weighted means of the cells per cluster, for
# bins with `MIN_BIN_CELLS` cells. Path lengths are native.
DENSITY_BIN_UM = 20
MIN_BIN_CELLS = 3
local_ids = {
    int(path.stem)
    for cache in (SKELETON_CACHE, L4IT_SKELETON_CACHE)
    for path in cache.glob("*.osy")
}
density_cells = classes[classes.index.isin(local_ids)]
print(
    f"16 cells with a local skeleton: "
    f"{density_cells.groupby(['group', 'is_column']).size().to_dict()}"
)
density_where = pl.col("post_pt_root_id").is_in(
    pl.Series(density_cells.index.to_numpy()).implode()
)
# The synapses on the unitary spines with one synapse, and the shaft synapses.
density_synapses = (
    census_synapses(density_where)
    .with_columns(spine_flags("spine_"))
    .with_columns(
        kind=pl.when(pl.col("unitary").fill_null(False) & (pl.col("n_synapses_class") == 1))
        .then(pl.lit("spine"))
        .when(pl.col("tag") == "shaft")
        .then(pl.lit("shaft")),
        size=synapse_size_nm3(pl.col("size")),
    )
    .drop_nulls("kind")
    .select(
        "post_pt_root_id", "synapse_id", "kind", "size",
        pl.col("spine_size_nm3").alias("volume"),
        pl.col("spine_area_nm2").alias("area"),
    )
    .partition_by("post_pt_root_id", as_dict=True)
)
# column: (kind of synapse, weight or None for a count, y label, figure suffix). A
# synapse size per µm is the sum of the synaptic junction sizes.
DENSITY_MEASURES = {
    "n_spines": ("spine", None, "Spines per µm\n(unitary, one synapse)", "spine_density"),
    "n_shaft": ("shaft", None, "Shaft synapses per µm", "shaft_density"),
    "spine_volume": (
        "spine", "volume", "Spine volume per µm\n(nm³/µm, unitary, one synapse)",
        "spine_volume_density",
    ),
    "spine_area": (
        "spine", "area", "Spine surface area per µm\n(nm²/µm, unitary, one synapse)",
        "spine_area_density",
    ),
    "spine_size": (
        "spine", "size",
        "Spine synapse size per µm\n(nm³/µm, unitary, one synapse)",
        "spine_size_density",
    ),
    "shaft_size": (
        "shaft", "size", "Shaft synapse size per µm\n(nm³/µm)", "shaft_size_density",
    ),
}


def binned_dendrite_density(root_id: int) -> pl.DataFrame:
    """Per `DENSITY_BIN_UM` bin of path distance to the soma, the dendrite cable of
    `root_id` and the sums of `DENSITY_MEASURES` on it."""
    cell = cached_skeleton(root_id)
    skel = cell.skeleton
    post = cell.annotations["post_syn"]
    synapses = density_synapses[(root_id,)].join(
        pl.DataFrame(
            {
                "synapse_id": np.asarray(post.vertex_df.index, dtype=np.int64),
                "vertex": post.map_index_to_layer("skeleton", as_positional=True).astype(
                    np.int64
                ),
            }
        ),
        on="synapse_id",
        how="inner",
    )
    mask = dendrite_mask(skel)
    synapses = synapses.filter(mask[synapses["vertex"].to_numpy()])
    # The masked skeleton keeps the vertex ids of `skel`, and its half-edge lengths
    # count only dendrite edges.
    dendrite = cell.apply_mask("skeleton", mask).skeleton
    ids = np.asarray(dendrite.vertex_index)
    not_soma = ids != skel.root_positional
    sums = {}
    for column, (kind, weight, _, _) in DENSITY_MEASURES.items():
        part = synapses.filter(pl.col("kind") == kind)
        weights = None if weight is None else part[weight].to_numpy().astype(float)
        sums[column] = np.bincount(
            part["vertex"].to_numpy(), weights=weights, minlength=skel.n_vertices
        )[ids[not_soma]]
    return (
        pl.DataFrame(
            {
                "distance_um": dendrite.distance_to_root(as_positional=True)[not_soma] / 1000,
                "cable_um": np.asarray(dendrite.half_edge_length)[not_soma] / 1000,
                **sums,
            }
        )
        .with_columns(
            bin_um=pl.col("distance_um") // DENSITY_BIN_UM * DENSITY_BIN_UM + DENSITY_BIN_UM / 2
        )
        .group_by("bin_um")
        .agg(pl.col("cable_um", *DENSITY_MEASURES).sum())
        .with_columns(root_id=pl.lit(root_id, pl.Int64))
    )


density_stash = Stash(
    "r2_11_4P_bimodality/dendrite_density",
    [
        Tables["microns_synapses"],
        Tables["synapse_census_features"],
        Tables["spine_census_features"],
    ],
    root_ids=sorted(density_cells.index.tolist()),
    bin_um=DENSITY_BIN_UM,
    spines="unitary_one_synapse",
    measures=list(DENSITY_MEASURES),
)
if not density_stash.exists():
    with ThreadPoolExecutor(max_workers=8) as executor:
        density_stash.save(
            pl.concat(executor.map(binned_dendrite_density, density_stash.params["root_ids"]))
        )
cell_groups = pl.DataFrame(
    {"root_id": density_cells.index.to_numpy(), "group": density_cells["group"].values}
)
density = density_stash.load().join(cell_groups, on="root_id").sort("root_id", "bin_um")
cell_density = (
    density.group_by("root_id", "group")
    .agg(pl.col(column).sum() / pl.col("cable_um").sum() for column in DENSITY_MEASURES)
    .group_by("group")
    .agg(pl.len().alias("n"), pl.col(*DENSITY_MEASURES).median())
)
print(f"16 per µm of dendrite, median over cells:\n{cell_density}")
# The thin lines are smoothed. Per cell, the values and the cable of the bins are
# summed with Gaussian weights of SD `SMOOTH_SD_UM` in path distance, and then
# divided. A bin with no dendrite adds 0 to both sums. The thick lines are not
# smoothed.
SMOOTH_SD_UM = 30
kept = density.filter(pl.col("cable_um") >= DENSITY_BIN_UM / 2)
offset = pl.col("bin_um") - pl.col("center_um")
smoothed = (
    kept.select("root_id", "group", center_um="bin_um")
    .join(density.drop("group"), on="root_id")
    .filter(offset.abs() <= 3 * SMOOTH_SD_UM)
    .with_columns(weight=(-(offset**2) / (2 * SMOOTH_SD_UM**2)).exp())
    .group_by("root_id", "group", "center_um")
    .agg((pl.col("cable_um", *DENSITY_MEASURES) * pl.col("weight")).sum())
    .rename({"center_um": "bin_um"})
    .sort("root_id", "bin_um")
)
for column, (_, _, ylabel, suffix) in DENSITY_MEASURES.items():
    lines = smoothed.with_columns(per_um=pl.col(column) / pl.col("cable_um"))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), layout="constrained")
    for cluster, name in CLASS_NAMES.items():
        color = CLUSTER_PALETTE[cluster]
        group = kept.filter(pl.col("group") == name)
        cell_lines = lines.filter(pl.col("group") == name)
        for (_,), line in cell_lines.partition_by("root_id", as_dict=True).items():
            ax.plot(line["bin_um"], line["per_um"], color=color, lw=0.5, alpha=0.3)
        mean = (
            group.group_by("bin_um")
            .agg((pl.col(column).sum() / pl.col("cable_um").sum()).alias("per_um"), n=pl.len())
            .filter(pl.col("n") >= MIN_BIN_CELLS)
            .sort("bin_um")
        )
        ax.plot(
            mean["bin_um"], mean["per_um"], color=color, lw=2.5,
            label=f"{name} (n = {group['root_id'].n_unique()})",
        )
    ax.set(
        xlabel="Path distance to soma (µm)", ylabel=ylabel, xlim=(0, None),
        ylim=(0, lines["per_um"].quantile(0.995)),
        title=f"{FOCUS}: cells with a local skeleton, {DENSITY_BIN_UM} µm bins,\n"
        f"cell lines smoothed (SD {SMOOTH_SD_UM} µm)",
    )
    ax.legend()
    save_matplotlib_figure(fig, f"{FOCUS_TYPE}_class_{suffix}", out_path, formats=["png"])

# %% 17: 4P, PUTATIVE THALAMIC INPUT OF THE COLUMN CELLS, BY CLUSTER
# `per_cell_counts.parquet` holds, per cell (root ids at v1718) and per predicted type
# of the pre cell, the synapse count and its share of all the input synapses of the
# cell. The types are nested in levels. "thalamocortical" has the same counts in levels
# 2 to 4, and every cell has rows in level 2, so level 2 is used. A cell with no
# thalamocortical row has 0. The cells match by nucleus id.
input_pred_path = "/Users/ben.pedigo/code/meshrep/neuron-hks-analysis/scratch/per_cell_counts.parquet"
input_pred = pl.read_parquet(input_pred_path)
root_ids_1718 = input_pred["cell_root_id"].unique().to_list()

client_1718 = CAVEclient(DATASTACK, version=1718)
cell_lookup = client_1718.materialize.views.nucleus_detection_lookup_v1(
    pt_root_id=root_ids_1718
).query()
root_to_nuc_lookup = cell_lookup.set_index("pt_root_id")["id"].to_dict()

is_thalamic = pl.col("pre_type") == "thalamocortical"
thalamic_input_props = (
    input_pred.filter(pl.col("level") == 2)
    .group_by("cell_root_id")
    .agg(
        n_putative_thalamic=pl.col("subtype_synapse_count").filter(is_thalamic).sum(),
        p_putative_thalamic=pl.col("ratio").filter(is_thalamic).sum(),
    )
    .with_columns(
        nucleus_id=pl.col("cell_root_id").replace_strict(
            root_to_nuc_lookup, default=None, return_dtype=pl.Int64
        )
    )
    .drop_nulls("nucleus_id")
    .to_pandas()
)
print(
    f"17 {len(thalamic_input_props)} of {len(root_ids_1718)} cells of the file with a "
    f"nucleus id, {thalamic_input_props['nucleus_id'].duplicated().sum()} duplicate ids"
)
column_classes = classes[classes["is_column"]].merge(
    thalamic_input_props.drop(columns="cell_root_id"), on="nucleus_id", how="left"
)
print(
    f"17 column cells with no putative input counts: "
    f"{column_classes['n_putative_thalamic'].isna().sum()} of {len(column_classes)}"
)
class_marginals(
    {
        "n_putative_thalamic": "Synapses from putative\nthalamic axons",
        "p_putative_thalamic": "Share of input synapses from\nputative thalamic axons",
    },
    f"{FOCUS_TYPE}_class_putative_thalamic_column",
    column_classes.dropna(subset=["n_putative_thalamic"]),
    "column cells",
    n_bins=15,
)

# %% 18: L3 AND L4, A NEUROGLANCER LINK WITH SEGMENT PROPERTIES
# Per cell of the Gaussian mixture of 8: the type and the cluster as tags, and the two
# means that the mixture uses (unitary spines with one synapse) as numbers. The random
# columns hold integers from 0 to 2^RANDOM_BITS - 1, to select random cells in the
# viewer. They are int16, because nglui 4.9.0 fails to cast its int64 list to uint16,
# so 15 bits is the most.
# No segments are selected: a query on the properties selects them.
LINK_TYPES = ["L3IT", "L4IT"] if USE_MESO_TYPE else ["23P", "4P"]
N_RANDOM = 4
RANDOM_BITS = 15
RANDOM_COLUMNS = [f"random_{i}" for i in range(N_RANDOM)]
rng = np.random.default_rng(0)
link_cells = pd.concat([gmm_labels[cell_type] for cell_type in LINK_TYPES]).join(
    cells[["cell_type", "volume_one_synapse", "size_one_synapse"]]
)
link_cells = (
    link_cells.assign(
        cell_type=link_cells["cell_type"].map(TYPE_NAMES),
        spine_class=link_cells["cluster"].map(CLASS_NAMES),
        **{
            column: rng.integers(0, 2**RANDOM_BITS, len(link_cells), dtype=np.int16)
            for column in RANDOM_COLUMNS
        },
    )
    .rename_axis("pt_root_id")
    .reset_index()
)
print(link_cells.groupby(["cell_type", "spine_class"]).size())
client = CAVEclient(DATASTACK, version=VERSION)
vs = ViewerState(client=client, capabilities="main").add_layers_from_client()
vs = vs.add_segment_properties(
    link_cells,
    id_column="pt_root_id",
    label_column="cell_type",
    tag_value_columns=["cell_type", "spine_class"],
    number_columns=["volume_one_synapse", "size_one_synapse", *RANDOM_COLUMNS],
)
print(f"16 {len(link_cells)} cells: {vs.to_url(shorten=True)}")

# %% 19: L3 AND L4, OUTPUT OF THE PROOFREAD COLUMN AND THALAMIC CELLS ONTO THE CLUSTERS
# Per proofread column or thalamic cell, its synapses onto the small-spine and onto the
# large-spine cells of L3 and L4 together, on any compartment, and the share onto the
# small-spine cells. The clusters are those of 8, with a posterior of `MIN_POSTERIOR`
# or more, as in 11. The two lines are the share of an axon that picks its targets at
# random: dashed, the share of the small-spine cells among all the cells of the
# clusters, and dotted, the share of their input synapses (`post_total_synapses`).
# The cell types are from `cell_info`, and the cells with no cell type drop.
OUTPUT_POST_TYPES = ["L3IT", "L4IT"] if USE_MESO_TYPE else ["23P", "4P"]
column_roots = (
    Tables["cell_selection"]
    .lazy()
    .filter(pl.col("is_column") | pl.col("is_thalamic"))
    .select("pt_root_id")
)
post_classes = pd.concat(
    [gmm_labels[t].query("posterior >= @MIN_POSTERIOR") for t in OUTPUT_POST_TYPES]
)
post_groups = post_classes["cluster"].map(CLASS_NAMES)
class_frame = pl.DataFrame(
    {"post_pt_root_id": post_classes.index.to_numpy(), "group": post_groups.to_numpy()}
)
pre_types = Tables["cell_info"].lazy().select(
    pl.col("pt_root_id").alias("pre_pt_root_id"), pl.col("cell_type").cast(pl.String)
)
column_outputs = (
    Tables["proofread_pre_synapses"]
    .lazy()
    .join(column_roots, left_on="pre_pt_root_id", right_on="pt_root_id")
    .join(class_frame.lazy(), on="post_pt_root_id")
    .group_by("pre_pt_root_id", "group")
    .len()
    .collect()
    .pivot(on="group", index="pre_pt_root_id", values="len")
    .fill_null(0)
    .join(pre_types.collect(), on="pre_pt_root_id", how="left")
    .to_pandas()
)
column_outputs["p_small"] = column_outputs["Small spine"] / (
    column_outputs["Small spine"] + column_outputs["Large spine"]
)
shown = column_outputs[column_outputs["cell_type"].notna()]
pre_order = [
    t
    for t in Tables["cell_info"].lazy().collect_schema()["cell_type"].categories
    if t in set(shown["cell_type"])
]
is_small = post_groups == CLASS_NAMES[0]
expected = is_small.mean()
post_synapses = cells.loc[post_classes.index, "post_total_synapses"]
expected_synapses = post_synapses[is_small].sum() / post_synapses.sum()
print(
    f"19 {len(column_outputs)} pre cells with output onto the clusters of "
    f"{[TYPE_NAMES[t] for t in OUTPUT_POST_TYPES]} ({post_groups.value_counts().to_dict()} "
    f"cells), {len(shown)} shown, expected share {expected:.3f} by cells, "
    f"{expected_synapses:.3f} by input synapses"
)
print(
    shown.groupby("cell_type")["p_small"]
    .agg(["count", "median", "mean"])
    .reindex(pre_order)
    .round(3)
)
# A scatter with its own jitter, because `sns.stripplot` takes one size for all the
# dots. The area of a dot goes with the square root of its number of synapses onto
# the clusters, which run from 1 to about 10^4.
shown = shown.assign(n_onto=shown["Small spine"] + shown["Large spine"])
print(f"19 synapses onto the clusters per pre cell: {shown['n_onto'].describe().round(1).to_dict()}")


def dot_size(n):
    return 2 * np.sqrt(n)


fig, ax = plt.subplots(figsize=(6, 0.4 * len(pre_order) + 1), layout="constrained")
jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(shown))
ax.scatter(
    shown["p_small"], shown["cell_type"].map(pre_order.index) + jitter,
    s=dot_size(shown["n_onto"]), c=shown["cell_type"].map(CELL_TYPE_PALETTE),
    alpha=0.6, linewidth=0,
)
ax.invert_yaxis()
ax.axvline(expected, color="grey", linestyle="--", linewidth=1, label="Random, by cells")
ax.axvline(
    expected_synapses, color="grey", linestyle=":", linewidth=1.5,
    label="Random, by input synapses",
)
for n in [10, 100, 1000]:
    ax.scatter([], [], s=dot_size(n), color="grey", linewidth=0, label=f"{n} synapses")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize="small")
ax.set_yticks(range(len(pre_order)), [remapper(t) for t in pre_order])
ax.set(
    xlim=(0, 1),
    xlabel="Share of synapses onto\nsmall-spine cells",
    ylabel="",
    title=f"Output of the proofread column and thalamic cells\nonto "
    f"{' and '.join(TYPE_NAMES[t] for t in OUTPUT_POST_TYPES)}",
)
save_matplotlib_figure(fig, "class_column_output_share", out_path, formats=["png"])
