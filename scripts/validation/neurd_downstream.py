# %%
"""Reviewer R2.1: do the NEURD and HKS compartment labels give different downstream
results?

The input panels read the synapses onto the column cells (ssa-backlog TASK-38).
The output panels read the synapses from the proofread cells, which need the full
run (ssa-backlog TASK-40). They do not run while `proofread_pre_synapses` is a
planned table.

It writes figures only, no glued variables.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

from analysis import (
    CELL_TYPE_CATEGORIES,
    CELL_TYPE_PALETTE,
    Tables,
    cell_type_mapper,
    save_matplotlib_figure,
    set_matplotlib_theme,
)
from analysis.products.census import column_root_ids

set_matplotlib_theme()

SUBFOLDER = "neurd_downstream"
COMPARTMENTS = ["spine", "shaft", "soma"]
CELL_TYPES = [c for c in CELL_TYPE_CATEGORIES if c not in ("TH", "Unk")]
NEURD_DETAILED = ["spine_head", "spine_neck", "spine_no_neck", "shaft", "soma", "axon_bouton", "axon_other"]
# A cell or a cell type pair needs this many labeled synapses to be shown, for
# each labeling. 50 is the cutoff of the paper's cell type pair bar plots.
MIN_CELL_SYNAPSES = 100
MIN_PAIR_SYNAPSES = 50

STATUS_ORDER = ["compartment", "unknown", "axon", "no prediction", "no NEURD row"]
STATUS_COLORS = {
    "compartment": "#4d4d4d",
    "unknown": "#a6a6a6",
    "axon": "#d9a441",
    "no prediction": "#e0e0e0",
    "no NEURD row": "#e0e0e0",
}
# The third version uses the HKS labels on only the synapses that NEURD labels,
# so a difference from NEURD on it is label disagreement, not coverage.
VERSIONS = {
    "HKS": ("hks_label", pl.lit(True)),
    "NEURD": ("neurd_label", pl.lit(True)),
    "HKS on NEURD-labeled": ("hks_label", pl.col("neurd_label").is_not_null()),
}
VERSION_PALETTE = dict(zip(VERSIONS, ["black", "#b0b0b0", "#5a8fc4"]))


def label_synapses(table_name: str) -> pl.DataFrame:
    """The synapses of a census table with their HKS and NEURD labels, and the
    status of each label: `compartment`, or why the synapse has no compartment."""
    neurd = Tables["census_neurd_labels"].lazy().select(
        "synapse_id", "neurd_detailed_label", "neurd_label"
    )
    synapses = (
        Tables[table_name]
        .lazy()
        .select(
            "synapse_id",
            "pre_pt_root_id",
            "post_pt_root_id",
            pl.col("tag").cast(pl.String),
            "pre_cell_type",
            "post_cell_type",
            "post_broad_type",
        )
        .join(neurd, on="synapse_id", how="left")
        .with_columns(
            hks_label=pl.col("tag").replace("unknown", None),
            hks_status=pl.when(pl.col("tag").is_null())
            .then(pl.lit("no prediction"))
            .when(pl.col("tag") == "unknown")
            .then(pl.lit("unknown"))
            .otherwise(pl.lit("compartment")),
            neurd_status=pl.when(pl.col("neurd_detailed_label").is_null())
            .then(pl.lit("no NEURD row"))
            .when(pl.col("neurd_label").is_null())
            .then(pl.lit("axon"))
            .otherwise(pl.lit("compartment")),
        )
        .collect()
    )
    assert synapses["synapse_id"].is_unique().all()
    return synapses


def compartment_proportions(df: pl.DataFrame, label: str, by: list[str], min_synapses: int):
    """Per group in `by`, the count of synapses with a `label` and the proportion of
    them in each compartment. A group with fewer than `min_synapses` is dropped."""
    return (
        df.filter(pl.col(label).is_not_null())
        .group_by(by)
        .agg(
            n_synapses=pl.len(),
            **{c: (pl.col(label) == c).mean() for c in COMPARTMENTS},
        )
        .filter(pl.col("n_synapses") >= min_synapses)
    )


def version_proportions(df: pl.DataFrame, by: list[str], min_synapses: int) -> pl.DataFrame:
    """`compartment_proportions` for each labeling in `VERSIONS`, with a `version`
    column."""
    return pl.concat(
        [
            compartment_proportions(df.filter(keep), label, by, min_synapses).with_columns(
                version=pl.lit(version)
            )
            for version, (label, keep) in VERSIONS.items()
        ]
    )


def set_cell_type_ticks(ax, axis="y"):
    ticks = np.arange(len(CELL_TYPES))
    labels = [cell_type_mapper(c) for c in CELL_TYPES]
    if axis == "y":
        ax.set_yticks(ticks, labels)
    else:
        ax.set_xticks(ticks, labels)


def plot_coverage(synapses: pl.DataFrame, by: str, xlabel: str):
    """Stacked bars per cell type in `by`: the fraction of synapses with each HKS
    and each NEURD label status."""
    coverage = (
        synapses.unpivot(
            on=["hks_status", "neurd_status"], index=by, variable_name="method", value_name="status"
        )
        .group_by(by, "method", "status")
        .len()
        .with_columns(fraction=pl.col("len") / pl.col("len").sum().over(by, "method"))
    )
    print(
        coverage.group_by("method", "status")
        .agg(pl.col("len").sum())
        .with_columns(fraction=pl.col("len") / pl.col("len").sum().over("method"))
        .sort("method", "status")
    )
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (method, title) in zip(axs, [("hks_status", "HKS"), ("neurd_status", "NEURD")]):
        wide = (
            coverage.filter(pl.col("method") == method)
            .to_pandas()
            .pivot(index=by, columns="status", values="fraction")
            .reindex(CELL_TYPES)
            .fillna(0)
        )
        statuses = [s for s in STATUS_ORDER if s in wide.columns]
        wide[statuses].plot(
            kind="barh",
            stacked=True,
            ax=ax,
            width=0.8,
            color=[STATUS_COLORS[s] for s in statuses],
            legend=False,
        )
        ax.set(title=title, xlim=(0, 1), xlabel=xlabel, ylabel="")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    set_cell_type_ticks(axs[0])
    axs[0].invert_yaxis()
    return fig


def plot_cell_proportions(props: pl.DataFrame, y: str, xlabel: str):
    """Per cell, the proportion of synapses in each compartment for each labeling,
    with the cell type mean as a bar."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    plot_data = props.filter(pl.col(y).is_in(CELL_TYPES)).to_pandas()
    kws = dict(
        data=plot_data,
        y=y,
        hue="version",
        hue_order=list(VERSIONS),
        palette=VERSION_PALETTE,
        order=CELL_TYPES,
    )
    for ax, compartment in zip(axs, COMPARTMENTS):
        sns.stripplot(x=compartment, dodge=True, size=2, alpha=0.5, legend=False, ax=ax, **kws)
        sns.pointplot(
            x=compartment,
            dodge=0.53,
            estimator="mean",
            errorbar=None,
            markers="|",
            markersize=15,
            linestyle="none",
            legend=compartment == "soma",
            ax=ax,
            **kws,
        )
        ax.set(xlim=(-0.02, 1.02), xlabel=f"{xlabel} on {compartment}", ylabel="")
    set_cell_type_ticks(axs[0])
    sns.move_legend(axs[-1], "upper left", bbox_to_anchor=(1, 1), title=None, frameon=False)
    return fig


def print_cell_type_summary(props: pl.DataFrame, by: str):
    """The mean and spread across cells of each compartment proportion, per cell
    type and labeling."""
    print(
        props.filter(pl.col(by).is_in(CELL_TYPES))
        .group_by(by, "version")
        .agg(
            n_cells=pl.len(),
            **{f"{c}_mean": pl.col(c).mean() for c in COMPARTMENTS},
            **{f"{c}_std": pl.col(c).std() for c in COMPARTMENTS},
        )
        .sort(pl.col(by).cast(pl.Enum(CELL_TYPES)), "version")
    )


# %% INPUTS OF THE COLUMN CELLS

inputs = label_synapses("column_post_synapses")
print(f"{len(inputs):,} synapses onto {inputs['post_pt_root_id'].n_unique():,} column cells")

fig = plot_coverage(inputs, "post_cell_type", "Fraction of input synapses")
save_matplotlib_figure(fig, "input_coverage_by_cell_type", subfolder=SUBFOLDER)

# %% NEURD COVERAGE PER COLUMN CELL

cell_coverage = (
    inputs.group_by("post_pt_root_id", "post_cell_type")
    .agg(neurd_fraction=(pl.col("neurd_status") == "compartment").mean())
    .filter(pl.col("post_cell_type").is_in(CELL_TYPES))
)
n_unlabeled = cell_coverage.filter(pl.col("neurd_fraction") == 0).height
print(f"{n_unlabeled} of {cell_coverage.height} column cells have no NEURD compartment label")

fig, ax = plt.subplots(figsize=(6, 5))
sns.stripplot(
    data=cell_coverage.to_pandas(),
    y="post_cell_type",
    x="neurd_fraction",
    hue="post_cell_type",
    palette=CELL_TYPE_PALETTE,
    order=CELL_TYPES,
    size=3,
    jitter=0.3,
    legend=False,
    ax=ax,
)
ax.set(xlim=(-0.02, 1.02), xlabel="Fraction of inputs with\na NEURD compartment", ylabel="")
set_cell_type_ticks(ax)
save_matplotlib_figure(fig, "input_neurd_coverage_per_cell", subfolder=SUBFOLDER)

# %% AGREEMENT ON THE SYNAPSES THAT BOTH LABEL

both = inputs.filter(pl.col("hks_label").is_not_null(), pl.col("neurd_detailed_label").is_not_null())

fig, axs = plt.subplots(1, 2, figsize=(13, 3.5), sharey=True)
for ax, broad_type in zip(axs, ["excitatory", "inhibitory"]):
    counts = (
        both.filter(pl.col("post_broad_type") == broad_type)
        .to_pandas()
        .groupby(["hks_label", "neurd_detailed_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=COMPARTMENTS, columns=NEURD_DETAILED, fill_value=0)
    )
    sns.heatmap(
        counts.div(counts.sum(axis=1), axis=0),
        annot=True,
        fmt=".2f",
        cmap="Greys",
        vmin=0,
        vmax=1,
        cbar=False,
        ax=ax,
    )
    ax.set(title=f"Post {broad_type} ({counts.values.sum():,} synapses)", xlabel="NEURD", ylabel="")
    ax.set_xticks(ax.get_xticks(), NEURD_DETAILED, rotation=45, ha="right")
axs[0].set_ylabel("HKS")
save_matplotlib_figure(fig, "input_confusion_by_broad_type", subfolder=SUBFOLDER)

# %% PER-CELL INPUT COMPARTMENT PROPORTIONS

input_props = version_proportions(inputs, ["post_pt_root_id", "post_cell_type"], MIN_CELL_SYNAPSES)
print_cell_type_summary(input_props, "post_cell_type")
fig = plot_cell_proportions(input_props, "post_cell_type", "Proportion of inputs")
save_matplotlib_figure(fig, "input_cell_proportions", subfolder=SUBFOLDER)

# %% PER-CELL INPUT SPINE PROPORTION, HKS AGAINST NEURD

paired = input_props.pivot(on="version", index=["post_pt_root_id", "post_cell_type"], values="spine")

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
for ax, version in zip(axs, ["HKS", "HKS on NEURD-labeled"]):
    sns.scatterplot(
        data=paired.to_pandas(),
        x=version,
        y="NEURD",
        hue="post_cell_type",
        palette=CELL_TYPE_PALETTE,
        s=8,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, zorder=-1)
    ax.set(xlabel=f"Spine proportion,\n{version}", ylabel="Spine proportion, NEURD")
save_matplotlib_figure(fig, "input_spine_proportion_hks_vs_neurd", subfolder=SUBFOLDER)

# %% PER-CELL INPUT SPINE PROPORTION OF E AND I CELLS, PER LABELING
# The lower panel of Fig 4A left (scripts/census/input_compartments.py), once per
# labeling.

broad_types = inputs.select("post_pt_root_id", "post_broad_type").unique()
assert broad_types["post_pt_root_id"].is_unique().all()
broad_props = input_props.join(broad_types, on="post_pt_root_id").to_pandas()

fig, axs = plt.subplots(
    len(VERSIONS), 1, figsize=(5.31, 8.5), dpi=300, sharex=True, layout="constrained"
)
for ax, version in zip(axs, VERSIONS):
    sns.histplot(
        data=broad_props.query("version == @version"),
        x="spine",
        hue="post_broad_type",
        hue_order=["inhibitory", "excitatory"],
        palette=CELL_TYPE_PALETTE,
        bins=np.linspace(0, 1, 21),
        stat="density",
        element="step",
        common_norm=False,
        legend=False,
        linewidth=0,
        kde=True,
        ax=ax,
    )
    ax.set(title=version, ylabel="Normalized\ncell density", yticks=[], xlabel="")
    ax.spines[["top", "right", "left"]].set_visible(False)
for xpos, label, ha in [(0.1, "Inhibitory", "left"), (0.9, "Excitatory", "right")]:
    axs[0].text(
        xpos, 0.8, label, transform=axs[0].transAxes, color=CELL_TYPE_PALETTE[label.lower()], ha=ha
    )
axs[-1].set_xlabel("Proportion of synapses on spines")
save_matplotlib_figure(fig, "input_spine_proportion_by_broad_type", subfolder=SUBFOLDER)

# %% WITHIN-TYPE PER-CELL AGREEMENT AND TRENDS, 23P AND 4P
# HKS on the NEURD-labeled synapses against NEURD, so a difference is label
# disagreement, not coverage.

TREND_TYPES = ["23P", "4P"]
MATCHED = "HKS on NEURD-labeled"
COVARIATES = {
    "n_inputs": "Input synapses",
    "nuc_volume": "Nucleus volume (µm³)",
    "depth_um": "Soma depth (µm)",
}

cell_covariates = (
    inputs.group_by("post_pt_root_id")
    .agg(n_inputs=pl.len())
    .join(
        Tables["cell_info"]
        .lazy()
        .select(
            pl.col("pt_root_id").alias("post_pt_root_id"),
            "nuc_volume",
            pl.col("pt_position_um_y").alias("depth_um"),
        )
        .collect(),
        on="post_pt_root_id",
        how="left",
    )
)
trend_cells = (
    paired.filter(pl.col("post_cell_type").cast(pl.String).is_in(TREND_TYPES))
    .drop_nulls([MATCHED, "NEURD"])
    .join(cell_covariates, on="post_pt_root_id", how="left")
    .to_pandas()
)


def spearman_label(x, y) -> str:
    rho, p = stats.spearmanr(x, y, nan_policy="omit")
    return f"ρ = {rho:.2f}, p = {p:.1g}"


fig, axs = plt.subplots(1, len(TREND_TYPES), figsize=(10, 5), sharex=True, sharey=True)
for ax, cell_type in zip(axs, TREND_TYPES):
    cells = trend_cells.query("post_cell_type == @cell_type")
    sns.scatterplot(
        data=cells, x=MATCHED, y="NEURD", color=CELL_TYPE_PALETTE[cell_type], s=10, linewidth=0, ax=ax
    )
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, zorder=-1)
    ax.set(
        title=f"{cell_type_mapper(cell_type)} ({len(cells)} cells)\n{spearman_label(cells[MATCHED], cells['NEURD'])}",
        xlabel="Spine proportion,\nHKS on NEURD-labeled",
        ylabel="Spine proportion, NEURD",
        xlim=(0.25, 0.9),
        ylim=(0.25, 0.9),
        aspect="equal",
    )
save_matplotlib_figure(fig, "input_spine_proportion_within_type", subfolder=SUBFOLDER)

# HKS on all synapses keeps the cells that NEURD does not label. Each labeling
# uses the cells that have a value for it.
TREND_VERSIONS = ["HKS", MATCHED, "NEURD"]
TREND_NAMES = {"HKS": "HKS", MATCHED: "HKS*", "NEURD": "NEURD"}
trend_all = (
    paired.filter(pl.col("post_cell_type").cast(pl.String).is_in(TREND_TYPES))
    .join(cell_covariates, on="post_pt_root_id", how="left")
    .to_pandas()
)

fig, axs = plt.subplots(
    len(TREND_TYPES), len(COVARIATES), figsize=(15, 10.5), sharey=True, layout="constrained"
)
for row, cell_type in enumerate(TREND_TYPES):
    cells = trend_all.query("post_cell_type == @cell_type")
    for col, (covariate, xlabel) in enumerate(COVARIATES.items()):
        ax = axs[row, col]
        for version in TREND_VERSIONS:
            sns.regplot(
                data=cells.dropna(subset=[version]),
                x=covariate,
                y=version,
                logx=covariate == "n_inputs",
                ci=None,
                color=VERSION_PALETTE[version],
                scatter_kws=dict(s=6, edgecolor="none", alpha=0.4),
                line_kws=dict(linewidth=2),
                ax=ax,
            )
        if covariate == "n_inputs":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.set_xticks([1000, 2000, 4000, 8000])
        stats_lines = [
            f"{TREND_NAMES[version]} ({cells[version].notna().sum()}): "
            f"{spearman_label(cells[covariate], cells[version])}"
            for version in TREND_VERSIONS
        ]
        ax.set_title("\n".join(stats_lines), fontsize="small")
        ax.set(xlabel=xlabel, ylabel=f"{cell_type_mapper(cell_type)}\nspine proportion" if col == 0 else "")
handles = [
    mpl.lines.Line2D([], [], color=VERSION_PALETTE[version], linewidth=2, label=label)
    for version, label in [
        ("HKS", "HKS, all synapses"),
        (MATCHED, "HKS on NEURD-labeled (HKS*)"),
        ("NEURD", "NEURD"),
    ]
]
fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False)
save_matplotlib_figure(fig, "input_spine_proportion_trends_within_type", subfolder=SUBFOLDER)

# %% COLUMN-TO-COLUMN CELL TYPE PAIR PROPORTIONS

column_column = inputs.filter(pl.col("pre_pt_root_id").is_in(column_root_ids()))
pair_props = {
    version: compartment_proportions(
        column_column, VERSIONS[version][0], ["pre_cell_type", "post_cell_type"], MIN_PAIR_SYNAPSES
    )
    for version in ["HKS", "NEURD"]
}


def pair_matrix(props: pl.DataFrame, compartment: str):
    return (
        props.to_pandas()
        .pivot(index="pre_cell_type", columns="post_cell_type", values=compartment)
        .reindex(index=CELL_TYPES, columns=CELL_TYPES)
    )


fig, axs = plt.subplots(3, 3, figsize=(16, 15), sharex=True, sharey=True)
for row, compartment in enumerate(COMPARTMENTS):
    hks = pair_matrix(pair_props["HKS"], compartment)
    neurd = pair_matrix(pair_props["NEURD"], compartment)
    diff = neurd - hks
    limit = max(np.nanmax(np.abs(diff.values)), 0.05)
    for ax, matrix, kws in [
        (axs[row, 0], hks, dict(cmap="Greys", vmin=0, vmax=1)),
        (axs[row, 1], neurd, dict(cmap="Greys", vmin=0, vmax=1)),
        (axs[row, 2], diff, dict(cmap="RdBu_r", vmin=-limit, vmax=limit)),
    ]:
        sns.heatmap(matrix, square=True, cbar=True, cbar_kws=dict(shrink=0.6), ax=ax, **kws)
        # A pair below `MIN_PAIR_SYNAPSES` has no value, so it shows this color.
        ax.set_facecolor("#f3e3c3")
        ax.set(xlabel="", ylabel="")
    axs[row, 0].set_ylabel(f"{compartment}\nPre cell type")
for ax, title in zip(axs[0], ["HKS", "NEURD", "NEURD - HKS"]):
    ax.set_title(title)
for ax in axs[-1]:
    ax.set_xlabel("Post cell type")
    ax.set_xticks(np.arange(len(CELL_TYPES)) + 0.5)
    ax.set_xticklabels([cell_type_mapper(c) for c in CELL_TYPES], rotation=90)
for ax in axs[:, 0]:
    ax.set_yticks(np.arange(len(CELL_TYPES)) + 0.5)
    ax.set_yticklabels([cell_type_mapper(c) for c in CELL_TYPES], rotation=0)
save_matplotlib_figure(fig, "column_pair_proportions", subfolder=SUBFOLDER)

# %% COMPOSITE PANELS
# Small versions for the NEURD composite figure. Panel A is the macro F1 bar plot
# of scripts/validation/compare_to_neurd.py, so these follow its style: NEURD in
# light grey, HKS in black, and the E and I cell types on shaded backgrounds.

METHOD_COLORS = {"NEURD": "lightgrey", "HKS": "black"}
BAR_CELL_TYPES = [c for c in CELL_TYPE_CATEGORIES[1:-1]]


def shade_broad_types(ax, n_excitatory: int, n_types: int):
    ax.autoscale(False)
    ax.fill_between(
        [-1, n_excitatory - 0.5], 0, 1, color=CELL_TYPE_PALETTE["Excitatory"], alpha=0.2, zorder=-1
    )
    ax.fill_between(
        [n_excitatory - 0.5, n_types], 0, 1, color=CELL_TYPE_PALETTE["Inhibitory"], alpha=0.2, zorder=-1
    )
    ax.text(0.0, 1.0, "Excitatory", color=CELL_TYPE_PALETTE["Excitatory"], transform=ax.transAxes, va="bottom")
    ax.text(
        n_excitatory / n_types, 1.0, "Inhibitory", color=CELL_TYPE_PALETTE["Inhibitory"], transform=ax.transAxes, va="bottom"
    )


# B: the share of the column cells of each type with at least one labeled input.
cell_labeled = (
    inputs.group_by("post_pt_root_id", "post_cell_type")
    .agg(
        HKS=pl.col("hks_label").is_not_null().any(),
        NEURD=pl.col("neurd_label").is_not_null().any(),
    )
    .unpivot(index=["post_pt_root_id", "post_cell_type"], variable_name="method", value_name="labeled")
    .group_by("post_cell_type", "method")
    .agg(n_cells=pl.len(), fraction=pl.col("labeled").mean())
    .with_columns(pl.col("post_cell_type").cast(pl.String))
)
print(cell_labeled.sort("post_cell_type", "method"))

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=cell_labeled.to_pandas(),
    x="post_cell_type",
    order=BAR_CELL_TYPES,
    y="fraction",
    hue="method",
    hue_order=["NEURD", "HKS"],
    palette=METHOD_COLORS,
    legend=False,
    ax=ax,
)
shade_broad_types(ax, n_excitatory=7, n_types=len(BAR_CELL_TYPES))
ax.set_xticks(range(len(BAR_CELL_TYPES)), [cell_type_mapper(c) for c in BAR_CELL_TYPES], rotation=45, ha="right")
ax.set(ylim=(0, 1), xlabel="Cell Type", ylabel="Proportion of cells\nwith labeled inputs")
save_matplotlib_figure(fig, "composite_cell_coverage", subfolder=SUBFOLDER)

# C: the per-cell input spine proportion, HKS on the NEURD-labeled synapses
# against NEURD.
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    data=paired.to_pandas(),
    x=MATCHED,
    y="NEURD",
    hue="post_cell_type",
    palette=CELL_TYPE_PALETTE,
    s=8,
    linewidth=0,
    legend=False,
    ax=ax,
)
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, zorder=-1)
ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
    xlabel="P(spine input), HKS*",
    ylabel="P(spine input), NEURD",
)
save_matplotlib_figure(fig, "composite_spine_proportion_scatter", subfolder=SUBFOLDER)

# D (draft): Spearman rho of the per-cell spine proportion against each covariate,
# with 95% bootstrap intervals, for 23P and 4P.


def bootstrap_rho(x, y, n_boot: int = 1000, seed: int = 0):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    rng = np.random.default_rng(seed)
    draws = [
        stats.spearmanr(x[idx], y[idx])[0]
        for idx in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))
    ]
    return stats.spearmanr(x, y)[0], *np.quantile(draws, [0.025, 0.975])


rho_rows = []
for cell_type in TREND_TYPES:
    cells = trend_all.query("post_cell_type == @cell_type")
    for covariate, covariate_label in COVARIATES.items():
        for version in TREND_VERSIONS:
            rho, lo, hi = bootstrap_rho(cells[covariate].to_numpy(float), cells[version].to_numpy(float))
            rho_rows.append(
                dict(
                    row=f"{cell_type_mapper(cell_type)}, {covariate_label.split(' (')[0].lower()}",
                    version=version,
                    rho=rho,
                    lo=lo,
                    hi=hi,
                )
            )
rho_table = pl.DataFrame(rho_rows)
rows = list(dict.fromkeys(rho_table["row"]))
offsets = dict(zip(TREND_VERSIONS, [-0.22, 0.0, 0.22]))
fig, ax = plt.subplots(figsize=(6, 6))
for version in TREND_VERSIONS:
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
        label=TREND_NAMES[version],
    )
ax.axvline(0, color="grey", linewidth=1, zorder=-1)
ax.set_yticks(range(len(rows)), rows)
ax.invert_yaxis()
ax.set(xlabel="Spearman ρ with P(spine input)", xlim=(-0.6, 0.8))
ax.legend(frameon=False, loc="lower right", fontsize="small")
save_matplotlib_figure(fig, "composite_trend_rho", subfolder=SUBFOLDER)

# D, wide: the same rho for every cell type, in the order of panels A and B, one
# row per covariate. Depth is left out.
WIDE_COVARIATES = ["n_inputs", "nuc_volume"]
type_cells = paired.join(cell_covariates, on="post_pt_root_id", how="left").to_pandas()
wide_rows = []
for cell_type in BAR_CELL_TYPES:
    cells = type_cells.query("post_cell_type == @cell_type")
    for covariate in WIDE_COVARIATES:
        for version in TREND_VERSIONS:
            rho, lo, hi = bootstrap_rho(cells[covariate].to_numpy(float), cells[version].to_numpy(float))
            wide_rows.append(dict(cell_type=cell_type, covariate=covariate, version=version, rho=rho, lo=lo, hi=hi))
wide_rho = pl.DataFrame(wide_rows)

wide_offsets = dict(zip(TREND_VERSIONS, [-0.25, 0.0, 0.25]))
ROW_LABELS = {"n_inputs": "Inputs", "nuc_volume": "Nucleus\nvolume"}
# The size of panel D in scripts/composites/neurd_comparison.py (PanelMosaic
# get_axis_sizes), so the panel is not scaled.
fig, axs = plt.subplots(
    len(WIDE_COVARIATES), 1, figsize=(13.25, 6.12), sharex=True, sharey=True, layout="constrained"
)
for ax, covariate in zip(axs, WIDE_COVARIATES):
    ax.axvspan(-0.5, 6.5, color=CELL_TYPE_PALETTE["Excitatory"], alpha=0.2, zorder=-2, linewidth=0)
    ax.axvspan(6.5, len(BAR_CELL_TYPES) - 0.5, color=CELL_TYPE_PALETTE["Inhibitory"], alpha=0.2, zorder=-2, linewidth=0)
    ax.axhline(0, color="grey", linewidth=1, zorder=-1)
    for version in TREND_VERSIONS:
        sub = wide_rho.filter(pl.col("covariate") == covariate, pl.col("version") == version)
        x = np.array([BAR_CELL_TYPES.index(c) for c in sub["cell_type"]]) + wide_offsets[version]
        ax.errorbar(
            x,
            sub["rho"],
            yerr=[sub["rho"] - sub["lo"], sub["hi"] - sub["rho"]],
            fmt="o",
            color=VERSION_PALETTE[version],
            markersize=4,
            capsize=0,
            linewidth=1.5,
            label=TREND_NAMES[version],
        )
    ax.set(ylim=(-1, 1), yticks=[-1, 0, 1], ylabel=ROW_LABELS[covariate])
axs[0].text(0.0, 1.0, "Excitatory", color=CELL_TYPE_PALETTE["Excitatory"], transform=axs[0].transAxes, va="bottom")
axs[0].text(
    7 / len(BAR_CELL_TYPES), 1.0, "Inhibitory", color=CELL_TYPE_PALETTE["Inhibitory"], transform=axs[0].transAxes, va="bottom"
)
fig.supylabel("Spearman ρ with P(spine input)", fontsize="medium")
axs[-1].set_xlim(-0.5, len(BAR_CELL_TYPES) - 0.5)
axs[-1].set_xticks(range(len(BAR_CELL_TYPES)), [cell_type_mapper(c) for c in BAR_CELL_TYPES], rotation=45, ha="right")
axs[-1].set_xlabel("Cell Type")
save_matplotlib_figure(fig, "composite_trend_rho_by_cell_type", subfolder=SUBFOLDER, bbox_inches=None)

# %% OUTPUTS OF THE PROOFREAD CELLS

# NOTE: these panels need `proofread_pre_synapses`, which the full run builds
# (ssa-backlog TASK-18). Until then the table is planned, and they do not run.
if Tables["proofread_pre_synapses"].kind == "planned":
    print("proofread_pre_synapses is planned, so the output panels do not run")
else:
    outputs = label_synapses("proofread_pre_synapses")
    print(f"{len(outputs):,} synapses from {outputs['pre_pt_root_id'].n_unique():,} proofread cells")

    fig = plot_coverage(outputs, "pre_cell_type", "Fraction of output synapses")
    save_matplotlib_figure(fig, "output_coverage_by_cell_type", subfolder=SUBFOLDER)

    # The paper splits the outputs of a cell by the broad type of the target.
    for target_type in ["excitatory", "inhibitory"]:
        output_props = version_proportions(
            outputs.filter(pl.col("post_broad_type") == target_type),
            ["pre_pt_root_id", "pre_cell_type"],
            MIN_CELL_SYNAPSES,
        )
        print(f"Outputs to {target_type} targets")
        print_cell_type_summary(output_props, "pre_cell_type")
        fig = plot_cell_proportions(
            output_props, "pre_cell_type", f"Proportion of outputs to {target_type}"
        )
        save_matplotlib_figure(fig, f"output_to_{target_type}_cell_proportions", subfolder=SUBFOLDER)

# %%
