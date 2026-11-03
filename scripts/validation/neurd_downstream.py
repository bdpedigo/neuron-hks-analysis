# %%
"""Reviewer R2.1: do the NEURD and HKS compartment labels give different downstream
results?

The input panels read the synapses onto the column cells (ssa-backlog TASK-38).
The output panels read the synapses from the proofread cells onto the neurons of
the full run (ssa-backlog TASK-40).

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
    LABEL_CATEGORIES,
    Tables,
    cell_type_mapper,
    save_matplotlib_figure,
    set_matplotlib_theme,
)
from analysis.plots import (
    OUTPUT_CELL_TYPE_CATEGORIES,
    add_cell_type_y,
    cell_type_dotplot,
    sorted_stripplot_with_means,
    weighted_means,
)
from analysis.products.census import PRE_TARGETS, aggregate_pre, column_root_ids

set_matplotlib_theme()

SUBFOLDER = "neurd_downstream"
COMPARTMENTS = ["spine", "shaft", "soma"]
CELL_TYPES = [c for c in CELL_TYPE_CATEGORIES if c not in ("TH", "Unk")]
# The proofread cells include the thalamic cells, so the output panels keep TH.
OUTPUT_CELL_TYPES = OUTPUT_CELL_TYPE_CATEGORIES
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
MATCHED = "HKS on NEURD-labeled"
VERSIONS = {
    "HKS": ("hks_label", pl.lit(True)),
    "NEURD": ("neurd_label", pl.lit(True)),
    MATCHED: ("hks_label", pl.col("neurd_label").is_not_null()),
}
VERSION_PALETTE = dict(zip(VERSIONS, ["black", "#b0b0b0", "#5a8fc4"]))
TREND_VERSIONS = ["HKS", MATCHED, "NEURD"]
TREND_NAMES = {"HKS": "HKS", MATCHED: "HKS*", "NEURD": "NEURD"}


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
            "size",
            "ctr_pt_position_um_y",
            "pre_cell_type",
            "pre_broad_type",
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


def set_cell_type_ticks(ax, axis="y", order=CELL_TYPES):
    ticks = np.arange(len(order))
    labels = [cell_type_mapper(c) for c in order]
    if axis == "y":
        ax.set_yticks(ticks, labels)
    else:
        ax.set_xticks(ticks, labels)


def plot_coverage(synapses: pl.DataFrame, by: str, xlabel: str, order=CELL_TYPES):
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
            .reindex(order)
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
    set_cell_type_ticks(axs[0], order=order)
    axs[0].invert_yaxis()
    return fig


def plot_cell_proportions(props: pl.DataFrame, y: str, xlabel: str, order=CELL_TYPES):
    """Per cell, the proportion of synapses in each compartment for each labeling,
    with the cell type mean as a bar."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    plot_data = props.filter(pl.col(y).is_in(order)).to_pandas()
    kws = dict(
        data=plot_data,
        y=y,
        hue="version",
        hue_order=list(VERSIONS),
        palette=VERSION_PALETTE,
        order=order,
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
        ax.set(xlim=(-0.02, 1.02), xlabel=f"{xlabel}\non {compartment}", ylabel="")
    set_cell_type_ticks(axs[0], order=order)
    sns.move_legend(axs[-1], "upper left", bbox_to_anchor=(1, 1), title=None, frameon=False)
    return fig


def print_cell_type_summary(props: pl.DataFrame, by: str, order=CELL_TYPES):
    """The mean and spread across cells of each compartment proportion, per cell
    type and labeling."""
    print(
        props.filter(pl.col(by).is_in(order))
        .group_by(by, "version")
        .agg(
            n_cells=pl.len(),
            **{f"{c}_mean": pl.col(c).mean() for c in COMPARTMENTS},
            **{f"{c}_std": pl.col(c).std() for c in COMPARTMENTS},
        )
        .sort(pl.col(by).cast(pl.String).cast(pl.Enum(order)), "version")
    )


def plot_cell_coverage(synapses: pl.DataFrame, cell: str, by: str, xlabel: str, order=CELL_TYPES):
    """Per cell in `cell`, the fraction of its synapses with a NEURD compartment,
    one row per cell type in `by`."""
    cell_coverage = (
        synapses.group_by(cell, by)
        .agg(neurd_fraction=(pl.col("neurd_status") == "compartment").mean())
        .filter(pl.col(by).is_in(order))
    )
    n_unlabeled = cell_coverage.filter(pl.col("neurd_fraction") == 0).height
    print(f"{n_unlabeled} of {cell_coverage.height} cells have no NEURD compartment label")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.stripplot(
        data=cell_coverage.to_pandas(),
        y=by,
        x="neurd_fraction",
        hue=by,
        palette=CELL_TYPE_PALETTE,
        order=order,
        size=3,
        jitter=0.3,
        legend=False,
        ax=ax,
    )
    ax.set(xlim=(-0.02, 1.02), xlabel=xlabel, ylabel="")
    set_cell_type_ticks(ax, order=order)
    return fig


def plot_confusions(panels: dict[str, pl.DataFrame], ncols: int):
    """One HKS by NEURD heatmap per panel, on the synapses that both label. Each
    row is normalized."""
    nrows = -(-len(panels) // ncols)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(6.5 * ncols, 3.5 * nrows), sharey=True, squeeze=False, layout="constrained"
    )
    for ax, (title, synapses) in zip(axs.flat, panels.items()):
        counts = (
            synapses.filter(pl.col("hks_label").is_not_null(), pl.col("neurd_detailed_label").is_not_null())
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
        ax.set(title=f"{title} ({counts.values.sum():,} synapses)", xlabel="NEURD", ylabel="")
        ax.set_xticks(ax.get_xticks(), NEURD_DETAILED, rotation=45, ha="right")
    for ax in axs[:, 0]:
        ax.set_ylabel("HKS")
        ax.set_yticks(ax.get_yticks(), COMPARTMENTS, rotation=0)
    return fig


def plot_spine_scatter(paired: pl.DataFrame, hue: str, label: str, order=CELL_TYPES):
    """Per cell, the spine proportion of HKS and of HKS* against NEURD."""
    data = paired.filter(pl.col(hue).is_in(order)).to_pandas()
    data[hue] = data[hue].astype(str)
    present = [c for c in order if c in set(data[hue])]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, layout="constrained")
    for ax, version in zip(axs, ["HKS", MATCHED]):
        sns.scatterplot(
            data=data,
            x=version,
            y="NEURD",
            hue=hue,
            hue_order=present,
            palette=CELL_TYPE_PALETTE,
            s=8,
            linewidth=0,
            legend=ax is axs[-1],
            ax=ax,
        )
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, zorder=-1)
        ax.set(xlabel=f"{label},\n{version}", ylabel=f"{label}, NEURD")
    sns.move_legend(
        axs[-1], "upper left", bbox_to_anchor=(1, 1), title=None, frameon=False, markerscale=2
    )
    return fig


def plot_spine_joint(cells, broad_types: list[str], quantity: str):
    """Per cell, the spine proportion of HKS* against NEURD, colored by the broad
    type in `broad_type`. The margins hold the distribution of each broad type,
    for HKS* on top and NEURD on the right. `quantity` is the axis label before
    the labeling. It has two lines in each use, so the panels have the same size."""
    grid = sns.JointGrid(
        data=cells, x=MATCHED, y="NEURD", height=6, ratio=4, space=0.1, xlim=(0, 1), ylim=(0, 1)
    )
    hue_kws = dict(
        data=cells, hue="broad_type", hue_order=broad_types, palette=CELL_TYPE_PALETTE, legend=False
    )
    sns.scatterplot(x=MATCHED, y="NEURD", s=8, linewidth=0, ax=grid.ax_joint, **hue_kws)
    marginal_kws = dict(
        bins=np.linspace(0, 1, 21),
        stat="density",
        element="step",
        common_norm=False,
        linewidth=0,
        kde=True,
        **hue_kws,
    )
    sns.histplot(x=MATCHED, ax=grid.ax_marg_x, **marginal_kws)
    sns.histplot(y="NEURD", ax=grid.ax_marg_y, **marginal_kws)
    ax = grid.ax_joint
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, zorder=-1)
    # The wording of the Fig 4A axis (scripts/census/input_compartments.py).
    ax.set(
        xlabel=f"{quantity}\n(HKS, NEURD-labeled synapses)",
        ylabel=f"{quantity}\n(NEURD)",
    )
    for i, broad_type in enumerate(reversed(broad_types)):
        ax.text(
            0.03,
            0.95 - 0.07 * i,
            broad_type.capitalize(),
            transform=ax.transAxes,
            color=CELL_TYPE_PALETTE[broad_type],
            va="top",
        )
    return grid.figure


def plot_broad_type_distributions(broad_props, broad_types: list[str], xlabel: str, labels):
    """Per labeling, the density of the per-cell spine proportion of each broad
    type in `broad_type`. `labels` holds (x, y, name, ha) of the colored type names."""
    fig, axs = plt.subplots(
        len(VERSIONS), 1, figsize=(5.31, 8.5), dpi=300, sharex=True, layout="constrained"
    )
    for ax, version in zip(axs, VERSIONS):
        sns.histplot(
            data=broad_props.query("version == @version"),
            x="spine",
            hue="broad_type",
            hue_order=broad_types,
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
    for xpos, ypos, label, ha in labels:
        axs[0].text(
            xpos, ypos, label, transform=axs[0].transAxes, color=CELL_TYPE_PALETTE[label.lower()], ha=ha
        )
    axs[-1].set_xlabel(xlabel)
    return fig


def spearman_label(x, y) -> str:
    rho, p = stats.spearmanr(x, y, nan_policy="omit")
    return f"ρ = {rho:.2f}, p = {p:.1g}"


def bootstrap_rho(x, y, n_boot: int = 1000, seed: int = 0):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    rng = np.random.default_rng(seed)
    draws = [
        stats.spearmanr(x[idx], y[idx])[0]
        for idx in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))
    ]
    return stats.spearmanr(x, y)[0], *np.quantile(draws, [0.025, 0.975])


# A cell type needs this many cells with both values for a rho.
MIN_RHO_CELLS = 10


def cell_type_rho(cells, by: str, cell_types: list[str], covariates: list[str]) -> pl.DataFrame:
    """Per cell type, covariate and labeling, the Spearman rho of the per-cell
    spine proportion with the covariate, with a 95% bootstrap interval."""
    rows = []
    for cell_type in cell_types:
        type_cells = cells[cells[by] == cell_type]
        for covariate in covariates:
            for version in TREND_VERSIONS:
                x = type_cells[covariate].to_numpy(float)
                y = type_cells[version].to_numpy(float)
                if (~(np.isnan(x) | np.isnan(y))).sum() < MIN_RHO_CELLS:
                    continue
                rho, lo, hi = bootstrap_rho(x, y)
                rows.append(
                    dict(cell_type=cell_type, covariate=covariate, version=version, rho=rho, lo=lo, hi=hi)
                )
    return pl.DataFrame(rows)


def plot_rho_by_cell_type(
    rho: pl.DataFrame, cell_types, row_labels: dict, n_excitatory: int, ylabel: str, figsize, legend=False
):
    """One row per covariate: the rho of each cell type and labeling, with the E
    and I cell types on shaded backgrounds."""
    offsets = dict(zip(TREND_VERSIONS, [-0.25, 0.0, 0.25]))
    fig, axs = plt.subplots(
        len(row_labels), 1, figsize=figsize, sharex=True, sharey=True, layout="constrained"
    )
    for ax, covariate in zip(axs, row_labels):
        ax.axvspan(
            -0.5, n_excitatory - 0.5, color=CELL_TYPE_PALETTE["Excitatory"], alpha=0.2, zorder=-2, linewidth=0
        )
        ax.axvspan(
            n_excitatory - 0.5,
            len(cell_types) - 0.5,
            color=CELL_TYPE_PALETTE["Inhibitory"],
            alpha=0.2,
            zorder=-2,
            linewidth=0,
        )
        ax.axhline(0, color="grey", linewidth=1, zorder=-1)
        for version in TREND_VERSIONS:
            sub = rho.filter(pl.col("covariate") == covariate, pl.col("version") == version)
            x = np.array([cell_types.index(c) for c in sub["cell_type"]]) + offsets[version]
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
        ax.set(ylim=(-1, 1), yticks=[-1, 0, 1], ylabel=row_labels[covariate])
    axs[0].text(0.0, 1.0, "Excitatory", color=CELL_TYPE_PALETTE["Excitatory"], transform=axs[0].transAxes, va="bottom")
    axs[0].text(
        n_excitatory / len(cell_types),
        1.0,
        "Inhibitory",
        color=CELL_TYPE_PALETTE["Inhibitory"],
        transform=axs[0].transAxes,
        va="bottom",
    )
    fig.supylabel(ylabel, fontsize="medium")
    axs[-1].set_xlim(-0.5, len(cell_types) - 0.5)
    axs[-1].set_xticks(range(len(cell_types)), [cell_type_mapper(c) for c in cell_types], rotation=45, ha="right")
    axs[-1].set_xlabel("Cell Type")
    if legend:
        axs[0].legend(frameon=False, loc="lower right", ncols=3, fontsize="small")
    return fig


def pair_matrix(props: pl.DataFrame, compartment: str, pre_order=CELL_TYPES, post_order=CELL_TYPES):
    return (
        props.to_pandas()
        .pivot(index="pre_cell_type", columns="post_cell_type", values=compartment)
        .reindex(index=pre_order, columns=post_order)
    )


def plot_pair_proportions(pair_props: dict, pre_order=CELL_TYPES, post_order=CELL_TYPES):
    """Per compartment, the cell type pair proportions of HKS and NEURD and their
    difference."""
    fig, axs = plt.subplots(3, 3, figsize=(16, 15), sharex=True, sharey=True)
    for row, compartment in enumerate(COMPARTMENTS):
        hks = pair_matrix(pair_props["HKS"], compartment, pre_order, post_order)
        neurd = pair_matrix(pair_props["NEURD"], compartment, pre_order, post_order)
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
        ax.set_xticks(np.arange(len(post_order)) + 0.5)
        ax.set_xticklabels([cell_type_mapper(c) for c in post_order], rotation=90)
    for ax in axs[:, 0]:
        ax.set_yticks(np.arange(len(pre_order)) + 0.5)
        ax.set_yticklabels([cell_type_mapper(c) for c in pre_order], rotation=0)
    return fig


# %% INPUTS OF THE COLUMN CELLS

inputs = label_synapses("column_post_synapses")
print(f"{len(inputs):,} synapses onto {inputs['post_pt_root_id'].n_unique():,} column cells")

fig = plot_coverage(inputs, "post_cell_type", "Fraction of input synapses")
save_matplotlib_figure(fig, "input_coverage_by_cell_type", subfolder=SUBFOLDER)

# %% NEURD COVERAGE PER COLUMN CELL

fig = plot_cell_coverage(
    inputs, "post_pt_root_id", "post_cell_type", "Fraction of inputs with\na NEURD compartment"
)
save_matplotlib_figure(fig, "input_neurd_coverage_per_cell", subfolder=SUBFOLDER)

# %% AGREEMENT ON THE SYNAPSES THAT BOTH LABEL

fig = plot_confusions(
    {
        f"Post {broad_type}": inputs.filter(pl.col("post_broad_type") == broad_type)
        for broad_type in ["excitatory", "inhibitory"]
    },
    ncols=2,
)
save_matplotlib_figure(fig, "input_confusion_by_broad_type", subfolder=SUBFOLDER)

# %% PER-CELL INPUT COMPARTMENT PROPORTIONS

input_props = version_proportions(inputs, ["post_pt_root_id", "post_cell_type"], MIN_CELL_SYNAPSES)
print_cell_type_summary(input_props, "post_cell_type")
fig = plot_cell_proportions(input_props, "post_cell_type", "Proportion of inputs")
save_matplotlib_figure(fig, "input_cell_proportions", subfolder=SUBFOLDER)

# %% PER-CELL INPUT SPINE PROPORTION, HKS AGAINST NEURD

paired = input_props.pivot(on="version", index=["post_pt_root_id", "post_cell_type"], values="spine")

fig = plot_spine_scatter(paired, "post_cell_type", "Input spine proportion")
save_matplotlib_figure(fig, "input_spine_proportion_hks_vs_neurd", subfolder=SUBFOLDER)

# %% PER-CELL INPUT SPINE PROPORTION OF E AND I CELLS, PER LABELING
# The lower panel of Fig 4A left (scripts/census/input_compartments.py), once per
# labeling.

broad_types = inputs.select("post_pt_root_id", broad_type="post_broad_type").unique()
assert broad_types["post_pt_root_id"].is_unique().all()
broad_props = input_props.join(broad_types, on="post_pt_root_id").to_pandas()
fig = plot_broad_type_distributions(
    broad_props,
    ["inhibitory", "excitatory"],
    "Proportion of synapses on spines",
    [(0.1, 0.8, "Inhibitory", "left"), (0.9, 0.8, "Excitatory", "right")],
)
save_matplotlib_figure(fig, "input_spine_proportion_by_broad_type", subfolder=SUBFOLDER)

# %% WITHIN-TYPE PER-CELL AGREEMENT AND TRENDS, 23P AND 4P
# HKS on the NEURD-labeled synapses against NEURD, so a difference is label
# disagreement, not coverage.

TREND_TYPES = ["23P", "4P"]
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
fig = plot_pair_proportions(pair_props)
save_matplotlib_figure(fig, "column_pair_proportions", subfolder=SUBFOLDER)

# %% COMPOSITE PANELS
# Small versions for the NEURD composite figure. Panel A is the macro F1 bar plot
# of scripts/validation/compare_to_neurd.py, so these follow its style: NEURD in
# light grey, HKS in black, and the E and I cell types on shaded backgrounds.

METHOD_COLORS = {"NEURD": "lightgrey", "HKS": "black"}
BAR_CELL_TYPES = [c for c in CELL_TYPE_CATEGORIES[1:-1]]


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

fig, axs = cell_type_dotplot(
    cell_labeled.rename({"post_cell_type": "cell_type"}).to_pandas(),
    x="fraction",
    hue="method",
    palette=METHOD_COLORS,
    side="post",
    xlabel="Proportion of\ncells with any\nlabeled inputs",
)
save_matplotlib_figure(fig, "composite_cell_coverage", subfolder=SUBFOLDER)

# D: the per-cell input spine proportion, HKS on the NEURD-labeled synapses
# against NEURD. The margins hold the E and I distributions of
# input_spine_proportion_by_broad_type.
fig = plot_spine_joint(
    paired.join(broad_types, on="post_pt_root_id").to_pandas(),
    ["inhibitory", "excitatory"],
    "Proportion of input\nsynapses on spines",
)
save_matplotlib_figure(fig, "composite_spine_proportion_scatter", subfolder=SUBFOLDER)

# D (draft): Spearman rho of the per-cell spine proportion against each covariate,
# with 95% bootstrap intervals, for 23P and 4P.


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
wide_rho = cell_type_rho(type_cells, "post_cell_type", BAR_CELL_TYPES, WIDE_COVARIATES)
ROW_LABELS = {"n_inputs": "Inputs", "nuc_volume": "Nucleus\nvolume"}
# The size it had as a panel of scripts/composites/neurd_comparison.py (PanelMosaic
# get_axis_sizes), so the panel is not scaled.
fig = plot_rho_by_cell_type(
    wide_rho, BAR_CELL_TYPES, ROW_LABELS, 7, "Spearman ρ with P(spine input)", (13.25, 6.12)
)
save_matplotlib_figure(fig, "composite_trend_rho_by_cell_type", subfolder=SUBFOLDER, bbox_inches=None)

# %% OUTPUTS OF THE PROOFREAD CELLS
# The paper splits the outputs of a cell by the broad type of the target, so the
# panels keep the excitatory and inhibitory targets. The thalamic cells are
# proofread cells, so the pre side keeps TH.

TARGET_TYPES = ["excitatory", "inhibitory"]
PRE_BROAD_TYPES = ["excitatory", "inhibitory", "thalamic"]

all_outputs = label_synapses("proofread_pre_synapses")
print(f"{len(all_outputs):,} synapses from {all_outputs['pre_pt_root_id'].n_unique():,} proofread cells")
print(
    all_outputs.with_columns(neuron_target=pl.col("post_broad_type").is_in(TARGET_TYPES))
    .unpivot(on=["hks_status", "neurd_status"], index="neuron_target", variable_name="method", value_name="status")
    .group_by("neuron_target", "method", "status")
    .len()
    .with_columns(fraction=pl.col("len") / pl.col("len").sum().over("neuron_target", "method"))
    .sort("neuron_target", "method", "status")
)
outputs = all_outputs.filter(pl.col("post_broad_type").is_in(TARGET_TYPES))
print(f"{len(outputs):,} of them onto excitatory or inhibitory neurons")
print(
    outputs.group_by("pre_broad_type", "post_broad_type")
    .agg(n_synapses=pl.len(), neurd_fraction=(pl.col("neurd_status") == "compartment").mean())
    .sort("pre_broad_type", "post_broad_type")
)

fig = plot_coverage(outputs, "pre_cell_type", "Fraction of output synapses", order=OUTPUT_CELL_TYPES)
save_matplotlib_figure(fig, "output_coverage_by_cell_type", subfolder=SUBFOLDER)

# Composite panel C: the share of the output synapses of each pre cell type with a
# label. Every proofread cell has some NEURD-labeled outputs, so the share of cells
# (panel B for the inputs) would be 1 for both methods.
output_labeled = (
    outputs.group_by("pre_cell_type")
    .agg(
        HKS=pl.col("hks_label").is_not_null().mean(),
        NEURD=pl.col("neurd_label").is_not_null().mean(),
    )
    .unpivot(index="pre_cell_type", variable_name="method", value_name="fraction")
    .rename({"pre_cell_type": "cell_type"})
    .with_columns(pl.col("cell_type").cast(pl.String))
)
print(output_labeled.sort("cell_type", "method"))
fig, axs = cell_type_dotplot(
    output_labeled.to_pandas(),
    x="fraction",
    hue="method",
    palette=METHOD_COLORS,
    side="pre",
    xlabel="Proportion of\noutput synapses\nwith a label",
)
save_matplotlib_figure(fig, "composite_output_coverage", subfolder=SUBFOLDER)

# %% NEURD COVERAGE PER PROOFREAD CELL

fig = plot_cell_coverage(
    outputs,
    "pre_pt_root_id",
    "pre_cell_type",
    "Fraction of outputs with\na NEURD compartment",
    order=OUTPUT_CELL_TYPES,
)
save_matplotlib_figure(fig, "output_neurd_coverage_per_cell", subfolder=SUBFOLDER)

# %% OUTPUT AGREEMENT BY PRE AND TARGET BROAD TYPE

fig = plot_confusions(
    {
        f"{pre_type.capitalize()} to {target_type}": outputs.filter(
            pl.col("pre_broad_type") == pre_type, pl.col("post_broad_type") == target_type
        )
        for target_type in TARGET_TYPES
        for pre_type in PRE_BROAD_TYPES
    },
    ncols=len(PRE_BROAD_TYPES),
)
save_matplotlib_figure(fig, "output_confusion_by_broad_type", subfolder=SUBFOLDER)

# %% PER-CELL OUTPUT COMPARTMENT PROPORTIONS, BY TARGET BROAD TYPE

output_props = {}
for target_type in TARGET_TYPES:
    output_props[target_type] = version_proportions(
        outputs.filter(pl.col("post_broad_type") == target_type),
        ["pre_pt_root_id", "pre_cell_type"],
        MIN_CELL_SYNAPSES,
    )
    print(f"Outputs to {target_type} targets, cells with {MIN_CELL_SYNAPSES} or more labeled outputs")
    print(
        output_props[target_type]
        .group_by("pre_cell_type")
        .agg(**{version: (pl.col("version") == version).sum() for version in VERSIONS})
        .sort("pre_cell_type")
    )
    print_cell_type_summary(output_props[target_type], "pre_cell_type", order=OUTPUT_CELL_TYPES)
    fig = plot_cell_proportions(
        output_props[target_type],
        "pre_cell_type",
        f"Proportion of outputs\nto {target_type}",
        order=OUTPUT_CELL_TYPES,
    )
    save_matplotlib_figure(fig, f"output_to_{target_type}_cell_proportions", subfolder=SUBFOLDER)

# %% FIG 4C AND 4D PER LABELING
# The paper panels of scripts/census/outputs.py, with each labeling in place of
# `tag` in `aggregate_pre`. The HKS version gives the paper values.


def labeled_rollup(version: str) -> pl.DataFrame:
    """`aggregate_pre` on the outputs, with the label of `version` as `tag`. Only
    the spine, shaft and soma counts are meaningful: there is no structure or
    detailed tag."""
    label, keep = VERSIONS[version]
    synapses = outputs.filter(keep).select(
        "pre_pt_root_id",
        "post_pt_root_id",
        "post_broad_type",
        "size",
        tag=pl.col(label).cast(pl.Enum(LABEL_CATEGORIES)),
        tag_detailed=pl.lit(None, pl.String),
        structure_id=pl.lit(None, pl.Int32),
    )
    return aggregate_pre(synapses.lazy()).collect()


# The thalamic cells have no soma in the volume, so their depth is the mean depth
# of their output synapses. It only orders the cells within a row.
output_depth = (
    outputs.filter(pl.col("hks_label").is_not_null())
    .group_by("pre_pt_root_id")
    .agg(output_depth=pl.col("ctr_pt_position_um_y").mean())
)
proofread_cells = (
    Tables["proofread_cell_info"]
    .lazy()
    .select("cell_type", "broad_type", "is_thalamic", "pt_position_um_y", pre_pt_root_id="pt_root_id")
    .collect()
    .join(output_depth, on="pre_pt_root_id", how="left")
    .with_columns(
        depth=pl.when(pl.col("is_thalamic")).then("output_depth").otherwise("pt_position_um_y")
    )
)
PAPER_LEGEND_KWS = {
    "excitatory": dict(loc="upper left", bbox_to_anchor=(0.02, 0.985)),
    "inhibitory": dict(loc="upper right", bbox_to_anchor=(0.98, 0.98)),
}
VERSION_SLUGS = {"HKS": "hks", MATCHED: "hks_star", "NEURD": "neurd"}
paper_means = []
for version in VERSIONS:
    cells = add_cell_type_y(
        proofread_cells.join(labeled_rollup(version), on="pre_pt_root_id", how="left").to_pandas(),
        depth="depth",
    )
    for target_type, prefix in PRE_TARGETS.items():
        x = f"{prefix}p_spine_synapse"
        weight_x = f"{prefix}total_known_synapses"
        fig, axs, means = sorted_stripplot_with_means(
            data=cells,
            x=x,
            weight_x=weight_x,
            size=weight_x,
            size_title=f"Output\nsynapses\nto {target_type}",
            xlabel="Proportion of synapses on spines",
            title=TREND_NAMES[version],
            **PAPER_LEGEND_KWS[target_type],
        )
        save_matplotlib_figure(
            fig, f"output_to_{target_type}_paper_{VERSION_SLUGS[version]}", subfolder=SUBFOLDER
        )
        broad_means = weighted_means(cells, "broad_type", x, weight_x)
        for group, value in [*means.items(), *broad_means.items()]:
            paper_means.append(dict(target=target_type, group=str(group), version=version, mean=value))
print("Weighted mean output spine proportion, as in Fig 4C and 4D")
with pl.Config(tbl_rows=-1):
    print(
        pl.DataFrame(paper_means)
        .pivot(on="version", index=["target", "group"], values="mean")
        .with_columns(neurd_minus_hks=pl.col("NEURD") - pl.col("HKS"))
    )

# %% PER-CELL OUTPUT SPINE PROPORTION TO EXCITATORY TARGETS, HKS AGAINST NEURD

paired_out = output_props["excitatory"].pivot(
    on="version", index=["pre_pt_root_id", "pre_cell_type"], values="spine"
)
fig = plot_spine_scatter(
    paired_out, "pre_cell_type", "Output spine proportion\nto excitatory", order=OUTPUT_CELL_TYPES
)
save_matplotlib_figure(fig, "output_spine_proportion_hks_vs_neurd", subfolder=SUBFOLDER)
print("Spearman rho of the per-cell output spine proportion to excitatory, against NEURD")
for version in ["HKS", MATCHED]:
    both_values = paired_out.drop_nulls([version, "NEURD"])
    print(f"  {version}: {spearman_label(both_values[version], both_values['NEURD'])}, {len(both_values)} cells")

# Composite panel E: panel D for the outputs to excitatory targets.
pre_broad_types = outputs.select("pre_pt_root_id", broad_type="pre_broad_type").unique()
assert pre_broad_types["pre_pt_root_id"].is_unique().all()
fig = plot_spine_joint(
    paired_out.join(pre_broad_types, on="pre_pt_root_id").to_pandas(),
    PRE_BROAD_TYPES[::-1],
    "Proportion of outputs to\nexcitatory on spines",
)
save_matplotlib_figure(fig, "composite_output_spine_proportion_scatter", subfolder=SUBFOLDER)

# %% PER-CELL OUTPUT SPINE PROPORTION BY PRE BROAD TYPE, PER LABELING

for target_type in TARGET_TYPES:
    fig = plot_broad_type_distributions(
        output_props[target_type].join(pre_broad_types, on="pre_pt_root_id").to_pandas(),
        PRE_BROAD_TYPES,
        f"Proportion of outputs to\n{target_type} on spines",
        [(0.05, 0.8, "Inhibitory", "left"), (0.5, 0.8, "Excitatory", "center"), (0.86, 0.95, "Thalamic", "right")],
    )
    save_matplotlib_figure(
        fig, f"output_to_{target_type}_spine_proportion_by_broad_type", subfolder=SUBFOLDER
    )

# %% WITHIN-TYPE TRENDS OF THE OUTPUT SPINE PROPORTION TO EXCITATORY TARGETS
# TH has no nucleus in the volume, so it has no nucleus volume rho.

output_covariates = (
    outputs.group_by("pre_pt_root_id")
    .agg(n_outputs=pl.len())
    .join(
        Tables["cell_info"].lazy().select("nuc_volume", pre_pt_root_id="pt_root_id").collect(),
        on="pre_pt_root_id",
        how="left",
    )
)
output_rho = cell_type_rho(
    paired_out.join(output_covariates, on="pre_pt_root_id", how="left").to_pandas(),
    "pre_cell_type",
    OUTPUT_CELL_TYPES,
    ["n_outputs", "nuc_volume"],
)
with pl.Config(tbl_rows=-1):
    print(output_rho.pivot(on="version", index=["cell_type", "covariate"], values="rho"))
fig = plot_rho_by_cell_type(
    output_rho,
    OUTPUT_CELL_TYPES,
    {"n_outputs": "Outputs", "nuc_volume": "Nucleus\nvolume"},
    8,
    "Spearman ρ with P(spine output\nto excitatory)",
    (13.25, 6.12),
    legend=True,
)
save_matplotlib_figure(fig, "output_trend_rho_by_cell_type", subfolder=SUBFOLDER, bbox_inches=None)

# %% PRE TO POST CELL TYPE PAIR PROPORTIONS, ALL NEURON TARGETS

output_pair_props = {
    version: compartment_proportions(
        outputs, VERSIONS[version][0], ["pre_cell_type", "post_cell_type"], MIN_PAIR_SYNAPSES
    )
    for version in ["HKS", "NEURD"]
}
fig = plot_pair_proportions(output_pair_props, pre_order=OUTPUT_CELL_TYPES)
save_matplotlib_figure(fig, "output_pair_proportions", subfolder=SUBFOLDER)

# %% MEAN SIZE OF THE OUTPUTS ONTO EXCITATORY CELLS, BY PRE BROAD TYPE AND COMPARTMENT
# The analog of the size values of scripts/census/outputs.py, per labeling.

sizes = pl.concat(
    [
        outputs.filter(pl.col("post_broad_type") == "excitatory", keep, pl.col(label).is_not_null())
        .group_by("pre_broad_type", compartment=pl.col(label))
        .agg(mean_size=pl.col("size").mean(), n_synapses=pl.len())
        .with_columns(version=pl.lit(version))
        for version, (label, keep) in VERSIONS.items()
    ]
)
with pl.Config(tbl_rows=-1):
    print(
        sizes.pivot(on="version", index=["pre_broad_type", "compartment"], values="mean_size").sort(
            "pre_broad_type", "compartment"
        )
    )

# %%
