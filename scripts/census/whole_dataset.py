"""Figure 4B, 5B, S5 and S6, the LDA check of the input features, and the totals of
the whole dataset.

Scope: full run. It reads the post side of every neuron (`post_aggregation`) and the
outputs of the proofread cells. Run it with `--scratch` until these tables exist
(see `analysis.census_inputs`).

    uv run python scripts/census/whole_dataset.py [--scratch]
"""

# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import ticker
from scipy.stats import lognorm, pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.preprocessing import QuantileTransformer

from analysis import (
    CELL_TYPE_PALETTE,
    Tables,
    save_matplotlib_figure,
    set_matplotlib_theme,
)
from analysis.census_inputs import glue, out_path, post_cells, pre_synapses
from analysis.io import CELL_TYPE_CATEGORIES
from analysis.plots import remapper
from analysis.products.census import aggregate_pre

SCRATCH = "--scratch" in sys.argv
set_matplotlib_theme()
out_path = out_path(SCRATCH)
cells = post_cells(SCRATCH).to_pandas().set_index("pt_root_id")

# %% TOTALS OF THE WHOLE DATASET
count_columns = [
    "post_spine_synapses",
    "post_shaft_synapses",
    "post_soma_synapses",
    "post_total_synapses",
    "post_total_known_synapses",
    "post_single_spine_synapses",
    "post_multi_spine_synapses",
    "post_spine_sites",
    "post_multi_spine_sites",
]


def totals_in_millions(table):
    totals = table[count_columns].sum() / 1e6
    return {name.removeprefix("post_"): value for name, value in totals.items()}


glue(
    SCRATCH,
    prefix="summary_n_",
    **totals_in_millions(cells),
    format="{:.1f} million",
)
glue(
    SCRATCH,
    summary_n_neurons_ran=len(cells.query("post_total_known_synapses > 0")),
    format="{:,d}",
)
for broad_type in ["excitatory", "inhibitory"]:
    table = cells.query("broad_type == @broad_type")
    glue(
        SCRATCH,
        **{
            f"summary_n_{broad_type}_neurons": len(
                table.query("post_total_known_synapses > 0")
            )
        },
        format="{:,d}",
    )
    glue(
        SCRATCH,
        prefix=f"summary_n_{broad_type}_",
        **totals_in_millions(table),
        format="{:.1f} million",
    )

# The input and output synapse counts of the proofread cells, by cell type.
pre_totals = aggregate_pre(pre_synapses(SCRATCH)).select(
    pl.col("pre_pt_root_id").alias("pt_root_id"), "pre_total_known_synapses"
)
proofread = (
    Tables["proofread_cell_info"]
    .lazy()
    .join(pre_totals, on="pt_root_id", how="left")
    .group_by("cell_type")
    .agg(
        pl.len().alias("n_cells"),
        pl.col("post_total_known_synapses").sum(),
        pl.col("pre_total_known_synapses").sum(),
    )
    .drop_nulls("cell_type")
    .collect()
)
for col in ["n_cells", "post_total_known_synapses", "pre_total_known_synapses"]:
    glue(
        SCRATCH,
        prefix=f"summary_{col}_by_cell_type_",
        **dict(zip(proofread["cell_type"].cast(pl.String), proofread[col])),
        format="{:,d}",
    )

# %% LDA CHECK: DO THE INPUT FEATURES SEPARATE THE CURATED E AND I CELLS?
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

transformer = QuantileTransformer(output_distribution="normal")
transformer.fit(X[mask])
lda = LinearDiscriminantAnalysis()
lda.fit(transformer.transform(train[features]), train["broad_type"].astype(str))
print(
    classification_report(
        train["broad_type"].astype(str),
        lda.predict(transformer.transform(train[features])),
    )
)
cells.loc[mask, "broad_type_lda_prediction"] = lda.predict(
    transformer.transform(X[mask])
)
disagree = cells["broad_type"].astype(str) != cells["broad_type_lda_prediction"]
print(
    f"LDA disagrees with the curated broad type for {disagree.mean():.1%} of the cells"
)

# Cells that both the label and the LDA call excitatory, with enough inputs.
excitatory = cells.query(
    "broad_type == 'excitatory' and broad_type_lda_prediction == 'excitatory'"
)

# %% 4B: SOMA DEPTH AND THE PROPORTION OF INPUTS ONTO SPINES
x = "post_p_spine_synapse"
data = cells.query("post_total_synapses > 500")
fig, ax = plt.subplots(1, 1, figsize=(8.29, 7.17), dpi=300)
sns.scatterplot(
    data=data,
    x=x,
    y="pt_position_um_y",
    color="grey",
    s=2,
    linewidth=0,
    alpha=0.6,
    ax=ax,
)
sns.scatterplot(
    data=data.query("is_column"),
    x=x,
    y="pt_position_um_y",
    hue="broad_type",
    hue_order=["excitatory", "inhibitory"],
    palette=CELL_TYPE_PALETTE,
    zorder=2,
    linewidth=0.5,
    alpha=0.9,
    s=30,
    legend=False,
    ax=ax,
)
ax.invert_yaxis()
ax.set(
    xlabel="Proportion of input synapses on spines",
    ylabel="Soma depth (um)",
    xlim=(0, 0.9),
)
ax.text(0.1, 0.9, "Curated\ninhibitory", color=CELL_TYPE_PALETTE["inhibitory"])
ax.text(0.38, 0.9, "Uncurated", color="grey")
ax.text(0.65, 0.9, "Curated\nexcitatory", color=CELL_TYPE_PALETTE["excitatory"])
sns.despine(fig, ax, trim=True)
save_matplotlib_figure(fig, f"whole_dataset_soma_depth_vs_{x}", out_path)


# %% 5B: DISTRIBUTION OF THE MULTI-INPUT SHARE OF EXCITATORY CELLS, AND P SPINE
def histplot_with_lognorm_fit(values, ax, bins, color):
    sns.histplot(x=values, stat="density", bins=bins, ax=ax, color=color)
    fit = lognorm.fit(values)
    x_fit = np.geomspace(values.min(), values.max(), 100)
    ax.plot(x_fit, lognorm.pdf(x_fit, *fit), color="dimgrey", linestyle="--")
    ax.set_xscale("log")
    ax.text(0.0, 0.8, "Lognormal\nfit", transform=ax.transAxes, color="dimgrey")
    ax.spines["left"].set_visible(False)
    ax.set(yticks=[])


def log_ticks(ax, ticks):
    ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(ticks))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


x = "post_p_spine_site_is_multi"
y = "post_p_spine_synapse"
ticks = [0.02, 0.05, 0.1, 0.2]
data = excitatory.query("post_total_synapses > 500").dropna(subset=[x, y])
data = data[data[x] > 0]

fig, axs = plt.subplots(2, 1, figsize=(4.56, 7.17), layout="tight", sharex=True)
histplot_with_lognorm_fit(
    data[x].values,
    axs[0],
    bins=np.geomspace(0.005, 0.4, 50),
    color=CELL_TYPE_PALETTE["excitatory"],
)
axs[0].set(ylabel="Density\n(cells)")
ax = axs[1]
sns.scatterplot(data=data, x=x, y=y, ax=ax, color="black", s=1, linewidth=0, alpha=0.5)
ax.set(
    yscale="log",
    ylim=(0.45, 0.88),
    xlim=(0.01, 0.25),
    xlabel="Proportion of spines\nwith multiple inputs",
    ylabel="Proportion of\nsynapses on spines",
)
log_ticks(ax, ticks)
ax.yaxis.set_major_locator(plt.FixedLocator([0.5, 0.6, 0.7, 0.8]))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
save_matplotlib_figure(fig, f"excitatory_{x}_distribution_with_p_spine", out_path)

r, _ = pearsonr(np.log(data[x]), np.log(data[y]))
glue(
    SCRATCH,
    prefix="column_summary_",
    multi_spine_spine_prop_pearson_r=r,
    format="{:.2f}",
)

# %% S5: THE SAME DISTRIBUTION BY EXCITATORY CELL TYPE
cell_types = CELL_TYPE_CATEGORIES[1:8]
fig, axs = plt.subplots(len(cell_types), 1, figsize=(8, 12), layout="constrained")
for i, (ax, cell_type) in enumerate(zip(axs, cell_types)):
    values = excitatory.query("cell_type == @cell_type and post_total_synapses > 500")[
        x
    ].dropna()
    values = values[values > 0].values
    if len(values) < 2:
        ax.set_axis_off()
        continue
    sns.histplot(
        x=values,
        stat="density",
        bins=np.geomspace(0.005, 0.3, 50),
        ax=ax,
        color="black",
    )
    fit = lognorm.fit(values)
    x_fit = np.geomspace(values.min(), values.max(), 100)
    ax.plot(
        x_fit,
        lognorm.pdf(x_fit, *fit),
        color="red",
        linestyle="--",
        label="Lognormal fit",
    )
    ax.set_xscale("log")
    ax.spines["left"].set_visible(False)
    ax.text(0, 0.7, cell_type, transform=ax.transAxes)
    ax.set(xlim=(0.01, 0.2), yticks=[], xlabel="", ylabel="")
    log_ticks(ax, ticks)
    if i != len(cell_types) - 1:
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
    else:
        ax.set(xlabel="Proportion of spines with multiple inputs")
    if i == 0:
        ax.legend()
save_matplotlib_figure(fig, f"excitatory_{x}_distribution_by_cell_type", out_path)

# %% S6: CORRELATION OF THE MULTI-INPUT SHARE WITH OTHER FEATURES OF THE CELL
x = "post_p_spine_synapse_is_multi"
y_columns = {
    "pt_position_um_y": {"scale": "linear", "label": "Depth (um)"},
    "post_shaft_synapses": {"scale": "log", "label": "Number of\nshaft synapses"},
    "post_p_shaft_synapse": {
        "scale": "log",
        "label": "Proportion of\nsynapses on shaft",
    },
    "post_mean_synapse_size": {"scale": "log", "label": "Mean synapse\nsize (voxels)"},
}
cell_types = ["23P", "4P", "5P-IT", "6P-IT"]
fig, axs = plt.subplots(
    len(cell_types),
    len(y_columns),
    figsize=(4 * len(y_columns), 4 * len(cell_types)),
    layout="constrained",
    squeeze=False,
    sharex=True,
)
for j, cell_type in enumerate(cell_types):
    data = cells.query(
        "cell_type == @cell_type and broad_type_lda_prediction == 'excitatory'"
        " and post_total_synapses >= 2000"
    )
    for i, (y_name, y_props) in enumerate(y_columns.items()):
        ax = axs[j, i]
        sns.scatterplot(
            data=data, x=x, y=y_name, alpha=0.7, linewidth=0, ax=ax, s=3, color="black"
        )
        sns.scatterplot(
            data=data.query("is_column"),
            x=x,
            y=y_name,
            linewidth=0.2,
            ax=ax,
            s=12,
            color="red",
            zorder=10,
        )
        ylim = np.quantile(data[y_name], [0.005, 0.995])
        if y_name == "pt_position_um_y":
            ylim = ylim[::-1]
        ax.set(yscale=y_props["scale"], xscale="log", ylim=ylim)

        ys = data[y_name].values
        if y_props["scale"] == "log":
            ys = np.log(ys)
        r, _ = pearsonr(np.log(data[x].values), ys)
        glue(
            SCRATCH,
            prefix=f"column_summary_{cell_type}_",
            **{f"p_multi_spine_{y_name}_pearson_r": r},
            format="{:.2f}",
        )
        ax.set_title(f"r={r:.2f}")
        ax.set_ylabel(y_props["label"])
        ax.set_xlabel("Proportion of spines\nwith multiple inputs")
        if i == 0:
            ax.text(
                -0.5,
                0.5,
                remapper(cell_type),
                transform=ax.transAxes,
                va="center",
                ha="right",
            )
save_matplotlib_figure(fig, "p_multi_correlations", out_path)
