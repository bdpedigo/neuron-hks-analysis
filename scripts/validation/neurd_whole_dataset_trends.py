# %%
"""Spike (ssa-backlog TASK-67): the within-type trends of the input spine
proportion under HKS, HKS* and NEURD, over the whole dataset.

The column-cell version is `composite_trend_rho` in
scripts/validation/neurd_downstream.py. Here the cells are every root of
`cell_info` with an excitatory `meso_type`, and the NEURD label of each input
comes from `neurd_export`, not `census_neurd_labels`, which holds only the column
cells. HKS* is HKS on the synapses that NEURD labels.

The first figure is the panel for L2/3 IT (L2IT and L3IT) and L4 IT (L4IT). The
second section looks at the excitatory cells per meso type and pooled.

It writes figures only, no glued variables.

    uv run python scripts/validation/neurd_whole_dataset_trends.py
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from analysis import Stash, Tables, save_matplotlib_figure, set_matplotlib_theme
from analysis.products.validation import NEURD_DETAILED_LABELS, NEURD_LABELS

set_matplotlib_theme()

SUBFOLDER = "neurd_downstream"
COMPARTMENTS = ["spine", "shaft", "soma"]
EXC_TYPES = ["L2IT", "L3IT", "L4IT", "L5IT", "L5ET", "L5NP", "L6IT", "L6CT"]
# The label counts are cached per batch of meso types, so that adding types does
# not redo the join for the types already counted.
STASH_BATCHES = [["L2IT", "L3IT", "L4IT", "L5IT", "L6IT"], ["L5ET", "L5NP", "L6CT"]]
MESO_GROUPS = {"L2IT": "L2/3 IT", "L3IT": "L2/3 IT", "L4IT": "L4 IT"}
# A cell needs this many labeled inputs for a labeling, as in the column panels.
MIN_CELL_SYNAPSES = 100
COVARIATES = {
    "n_inputs": "input synapses",
    "nuc_volume": "nucleus volume",
    "depth_um": "soma depth",
}
VERSIONS = ["HKS", "HKS*", "NEURD"]
VERSION_PALETTE = {"HKS": "black", "HKS*": "#5a8fc4", "NEURD": "#b0b0b0"}

cells = (
    Tables["cell_info"]
    .lazy()
    .filter(pl.col("meso_type").is_in(EXC_TYPES))
    .select(
        "pt_root_id",
        "nuc_volume",
        "meso_type",
        depth_um=pl.col("pt_position_um_y"),
        meso_group=pl.col("meso_type").replace_strict(MESO_GROUPS, default=None),
    )
    .collect()
)
print(cells.group_by("meso_type").len().sort("meso_type"))

# %% THE HKS AND NEURD LABEL COUNTS OF EACH CELL


def label_counts(meso_types: list[str]) -> pl.DataFrame:
    """One row per cell of `meso_types` and pair of HKS and NEURD labels (null for
    none), with the count of its input synapses. One lazy query, run streaming.

    The synapses are the rows of `synapse_census_features` (no autapses, at the
    `census` cutoff). A synapse with no mesh mapping has no row there, so it is not
    counted (about 0.01% of the column-cell inputs).
    """
    roots = cells.filter(pl.col("meso_type").is_in(meso_types)).select(
        post_pt_root_id="pt_root_id"
    )
    hks = (
        Tables["synapse_census_features"]
        .lazy()
        .join(roots.lazy(), on="post_pt_root_id", how="semi")
        .select("synapse_id", "post_pt_root_id", hks_label="tag")
    )
    export = Tables["neurd_export"].lazy()
    # NOTE: a synapse with two valid NEURD rows (one at v1822) has an ambiguous
    # label, so it has none here.
    ambiguous = export.group_by("target_id").agg(pl.len()).filter(pl.col("len") > 1)
    neurd_classes = {code: NEURD_LABELS.get(label) for code, label in NEURD_DETAILED_LABELS.items()}
    neurd = export.join(ambiguous, on="target_id", how="anti").select(
        pl.col("target_id").alias("synapse_id"),
        neurd_label=pl.col("value").replace_strict(neurd_classes, return_dtype=pl.String),
        has_neurd_row=pl.lit(True),
    )
    counts = (
        hks.join(neurd, on="synapse_id", how="left")
        .group_by("post_pt_root_id", "hks_label", "neurd_label", "has_neurd_row")
        .agg(n_synapses=pl.len())
        .collect(engine="streaming")
    )
    # A synapse with two NEURD rows would count twice.
    assert counts["n_synapses"].sum() == hks.select(pl.len()).collect().item()
    return counts


batches = []
for meso_types in STASH_BATCHES:
    stash = Stash(
        "neurd_whole_dataset_trends/label_counts",
        inputs=[
            Tables["cell_info"],
            Tables["microns_synapses"],
            Tables["synapse_census_features"],
            Tables["spine_census_features"],
            Tables["neurd_export"],
        ],
        meso_types=meso_types,
    )
    if not stash.exists():
        stash.save(label_counts(meso_types))
    batches.append(stash.load())
counts = pl.concat(batches)
print(f"{counts['n_synapses'].sum():,} input synapses onto {counts['post_pt_root_id'].n_unique():,} cells")

# %% PER-CELL SPINE PROPORTION PER LABELING


def spine_proportion(label: str, keep: pl.Expr) -> pl.Expr:
    labeled = keep & pl.col(label).is_not_null()
    n_labeled = pl.col("n_synapses").filter(labeled).sum()
    n_spine = pl.col("n_synapses").filter(labeled & (pl.col(label) == "spine")).sum()
    return pl.when(n_labeled >= MIN_CELL_SYNAPSES).then(n_spine / n_labeled)


per_cell = (
    counts.group_by("post_pt_root_id")
    .agg(
        n_inputs=pl.col("n_synapses").sum(),
        neurd_fraction=pl.col("n_synapses").filter(pl.col("neurd_label").is_not_null()).sum()
        / pl.col("n_synapses").sum(),
        **{
            "HKS": spine_proportion("hks_label", pl.lit(True)),
            "HKS*": spine_proportion("hks_label", pl.col("neurd_label").is_not_null()),
            "NEURD": spine_proportion("neurd_label", pl.lit(True)),
        },
    )
    .join(cells.rename({"pt_root_id": "post_pt_root_id"}), on="post_pt_root_id")
)
print(
    per_cell.group_by("meso_type")
    .agg(
        n_cells=pl.len(),
        no_neurd=(pl.col("neurd_fraction") == 0).sum(),
        median_neurd_fraction=pl.col("neurd_fraction").median(),
        **{f"n_{v}": pl.col(v).is_not_null().sum() for v in VERSIONS},
        **{f"mean_{v}": pl.col(v).mean() for v in VERSIONS},
    )
    .sort("meso_type")
)

# %% SPEARMAN RHO WITH EACH COVARIATE, WITH 95% INTERVALS
# The same statistic as `composite_trend_rho`, which uses a bootstrap interval.
# TODO: the interval here is a handwritten Fisher z interval with the standard
# error sqrt(1.06 / (n - 3)) for Spearman (Fieller, Hartley and Pearson 1957), a
# placeholder for speed. If a figure from this script is kept, replace it with a
# library method chosen with the user (for example `scipy.stats.bootstrap`).


def spearman_ci(x, y):
    """The Spearman rho of `x` and `y` over the pairs with no NaN, its 95%
    interval and the number of pairs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    n = int(mask.sum())
    rho = stats.spearmanr(x[mask], y[mask])[0]
    half_width = 1.96 * np.sqrt(1.06 / (n - 3))
    lo, hi = np.tanh(np.arctanh(rho) + np.array([-half_width, half_width]))
    return rho, lo, hi, n


rho_rows = []
for group in ["L2/3 IT", "L4 IT"]:
    group_cells = per_cell.filter(pl.col("meso_group") == group)
    for covariate, covariate_label in COVARIATES.items():
        for version in VERSIONS:
            rho, lo, hi, n = spearman_ci(
                group_cells[covariate].to_numpy().astype(float),
                group_cells[version].to_numpy().astype(float),
            )
            rho_rows.append(
                dict(row=f"{group}, {covariate_label}", version=version, rho=rho, lo=lo, hi=hi, n_cells=n)
            )
rho_table = pl.DataFrame(rho_rows)
with pl.Config(tbl_rows=-1):
    print(rho_table)

rows = list(dict.fromkeys(rho_table["row"]))
offsets = dict(zip(VERSIONS, [-0.22, 0.0, 0.22]))
fig, ax = plt.subplots(figsize=(6, 6))
for version in VERSIONS:
    sub = rho_table.filter(pl.col("version") == version)
    y = np.array([rows.index(r) for r in sub["row"]]) + offsets[version]
    ax.errorbar(
        sub["rho"],
        y,
        xerr=[sub["rho"] - sub["lo"], sub["hi"] - sub["rho"]],
        fmt="o",
        color=VERSION_PALETTE[version],
        markersize=6,
        capsize=0,
        label=version,
    )
ax.axvline(0, color="grey", linewidth=1, zorder=-1)
ax.set_yticks(range(len(rows)), rows)
ax.invert_yaxis()
ax.set(xlabel="Spearman ρ with P(spine input)", xlim=(-0.6, 0.8), title="Whole dataset, by meso_type")
ax.legend(frameon=False, loc="lower right", fontsize="small")
save_matplotlib_figure(fig, "spike_whole_dataset_trend_rho", subfolder=SUBFOLDER)

# %% EXCITATORY CELLS PER MESO TYPE AND POOLED
# Input count and nucleus volume only: depth differs by layer by construction.
# "All excitatory" pools the raw values, so it mixes the differences between the
# types into the trend. "All excitatory, within type" pools the percentile ranks
# within each meso type, so it is the trend inside the types, with one rho for all
# of them.

EXC_COVARIATES = {"n_inputs": "Input synapses", "nuc_volume": "Nucleus volume"}
EXC_GROUPS = [*EXC_TYPES, "All excitatory", "All excitatory, within type"]


def within_type_ranks(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """`columns` as percentile ranks within each meso type. A null stays null."""
    return frame.with_columns(
        (pl.col(c).rank() / pl.col(c).count()).over("meso_type").alias(c) for c in columns
    )


def group_cells(group: str) -> pl.DataFrame:
    if group == "All excitatory":
        return per_cell
    if group == "All excitatory, within type":
        return within_type_ranks(per_cell, [*EXC_COVARIATES, *VERSIONS])
    return per_cell.filter(pl.col("meso_type") == group)


exc_rows = []
for group in EXC_GROUPS:
    frame = group_cells(group)
    for covariate in EXC_COVARIATES:
        for version in VERSIONS:
            rho, lo, hi, n = spearman_ci(
                frame[covariate].to_numpy().astype(float), frame[version].to_numpy().astype(float)
            )
            exc_rows.append(dict(group=group, covariate=covariate, version=version, rho=rho, lo=lo, hi=hi, n_cells=n))
exc_rho = pl.DataFrame(exc_rows)
with pl.Config(tbl_rows=-1, tbl_width_chars=200):
    print(
        exc_rho.pivot(on="version", index=["covariate", "group"], values="rho").join(
            exc_rho.filter(pl.col("version") == "NEURD").select("covariate", "group", n_neurd_cells="n_cells"),
            on=["covariate", "group"],
        )
    )

fig, axs = plt.subplots(1, len(EXC_COVARIATES), figsize=(11, 7), sharey=True, layout="constrained")
for ax, (covariate, title) in zip(axs, EXC_COVARIATES.items()):
    for version in VERSIONS:
        sub = exc_rho.filter(pl.col("covariate") == covariate, pl.col("version") == version)
        y = np.array([EXC_GROUPS.index(g) for g in sub["group"]]) + offsets[version]
        ax.errorbar(
            sub["rho"],
            y,
            xerr=[sub["rho"] - sub["lo"], sub["hi"] - sub["rho"]],
            fmt="o",
            color=VERSION_PALETTE[version],
            markersize=6,
            capsize=0,
            label=version,
        )
    ax.axvline(0, color="grey", linewidth=1, zorder=-1)
    ax.axhline(len(EXC_TYPES) - 0.5, color="black", linewidth=1, linestyle="--")
    ax.set(title=title, xlim=(-0.6, 0.8))
fig.supxlabel("Spearman ρ with P(spine input)")
axs[0].set_yticks(range(len(EXC_GROUPS)), EXC_GROUPS)
axs[0].invert_yaxis()
axs[-1].legend(frameon=False, loc="lower right", fontsize="small")
save_matplotlib_figure(fig, "spike_exc_trend_rho", subfolder=SUBFOLDER)

# %% THE SHAPE OF THE EXCITATORY TRENDS
# The mean spine proportion of each labeling in deciles of the covariate, within
# each meso type. A cell counts for a labeling only if it has a value for it.

N_BINS = 10
binned = (
    per_cell.unpivot(
        on=VERSIONS,
        index=["post_pt_root_id", "meso_type", *EXC_COVARIATES],
        variable_name="version",
        value_name="spine",
    )
    .drop_nulls("spine")
    .unpivot(
        on=list(EXC_COVARIATES),
        index=["post_pt_root_id", "meso_type", "version", "spine"],
        variable_name="covariate",
        value_name="value",
    )
    .drop_nulls("value")
    .with_columns(
        bin=(pl.col("value").rank() * N_BINS / (pl.col("value").count() + 1))
        .floor()
        .over("meso_type", "version", "covariate")
    )
    .group_by("meso_type", "version", "covariate", "bin")
    .agg(
        value=pl.col("value").median(),
        mean=pl.col("spine").mean(),
        sem=pl.col("spine").std() / pl.col("spine").count().sqrt(),
    )
    .sort("meso_type", "version", "covariate", "bin")
)

fig, axs = plt.subplots(
    len(EXC_COVARIATES), len(EXC_TYPES), figsize=(26, 7), sharey=True, layout="constrained"
)
for row, (covariate, xlabel) in enumerate(EXC_COVARIATES.items()):
    for col, meso_type in enumerate(EXC_TYPES):
        ax = axs[row, col]
        for version in VERSIONS:
            sub = binned.filter(
                pl.col("meso_type") == meso_type, pl.col("version") == version, pl.col("covariate") == covariate
            )
            ax.errorbar(
                sub["value"], sub["mean"], yerr=sub["sem"], color=VERSION_PALETTE[version], marker="o",
                markersize=3, linewidth=1.5, capsize=0, label=version,
            )
        ax.set(xlabel=xlabel, title=meso_type if row == 0 else "")
    axs[row, 0].set_ylabel("Mean P(spine input)\nper covariate decile")
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside right center", frameon=False)
save_matplotlib_figure(fig, "spike_exc_binned_trends", subfolder=SUBFOLDER)

# %%
