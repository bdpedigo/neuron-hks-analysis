"""Figure 4A and S4: the compartments of the input synapses of the column cells.

Scope: test cohort. It reads only the synapses onto the column cells.

    uv run python scripts/census/input_compartments.py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle

from analysis import (
    CELL_TYPE_PALETTE,
    COMPARTMENT_PALETTE_MUTED_HEX,
    FIG_PATH,
    Tables,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)
from analysis.io import CELL_TYPE_CATEGORIES
from analysis.plots import (
    TAG_CATEGORIES,
    add_cell_type_y,
    draw_bracket,
    remapper,
    sorted_stripplot_with_means,
    weighted_means,
)

set_matplotlib_theme()
out_path = FIG_PATH / "census"

proofread_cells = (
    Tables["proofread_cell_info"].lazy().collect().to_pandas().set_index("pt_root_id")
)
column_cells = add_cell_type_y(proofread_cells.query("is_column"))

# %% COUNTS OF PROOFREAD CELLS BY TYPE (the column cells and the thalamic cells)
for col in ["broad_type", "cell_type"]:
    counts = proofread_cells[col].value_counts()
    save_variables(prefix="column_summary_n_", **counts[counts > 0].to_dict())

# %% 4A, LEFT: HISTOGRAMS OF THE SPINE INPUTS OF E AND I CELLS
histplot_kws = dict(
    hue="broad_type",
    hue_order=["inhibitory", "excitatory"],
    palette=CELL_TYPE_PALETTE,
    stat="density",
    element="step",
    common_norm=False,
    legend=False,
    linewidth=0,
    kde=True,
)
fig, axs = plt.subplots(2, 1, figsize=(5.31, 7.17), dpi=300, layout="constrained")

ax = axs[0]
x = "post_spine_synapses"
data = column_cells.dropna(subset=[x]).query(f"{x} > 0")
sns.histplot(data=data, x=x, ax=ax, log_scale=True, bins=21, **histplot_kws)
for xpos, label, ha in [(0.1, "Inhibitory", "left"), (0.9, "Excitatory", "right")]:
    ax.text(
        xpos, 0.8, label, transform=ax.transAxes, color=CELL_TYPE_PALETTE[label.lower()], ha=ha
    )
ax.set(ylabel="Normalized\ncell density", yticks=[], xlabel="Number of synapses on spines")
ax.spines[["top", "right", "left"]].set_visible(False)
save_variables(
    prefix="column_summary_n_spine_",
    format="{:.0f}",
    **data.groupby("broad_type", observed=True)[x].mean().to_dict(),
)

ax = axs[1]
x = "post_p_spine_synapse"
data = column_cells.dropna(subset=[x])
sns.histplot(data=data, x=x, ax=ax, bins=np.linspace(0, 1, 21), **histplot_kws)
ax.set(ylabel="Normalized\ncell density", yticks=[], xlabel="Proportion of synapses on spines")
ax.spines[["top", "right", "left"]].set_visible(False)
save_matplotlib_figure(fig, "spine_histograms", out_path)
save_variables(
    prefix="column_summary_p_spine_",
    format="{:.3g}%",
    **data.groupby("broad_type", observed=True)[x].mean().to_dict(),
)

# %% 4A, RIGHT: PROPORTION OF SPINE INPUTS BY CELL TYPE
x = "post_p_spine_synapse"
weight_x = "post_total_synapses"
fig, axs, means = sorted_stripplot_with_means(
    data=column_cells,
    x=x,
    weight_x=weight_x,
    size="post_total_synapses",
    size_title="Input\nsynapses",
    size_norm=(2500, 25000),
    legend_sizes=(2500, 10000, 25000),
    xlabel="Proportion of synapses on spines",
)
save_matplotlib_figure(fig, "cell_type_input_spine_proportion", out_path)
save_variables(format="{:.1f}%", prefix=f"column_summary_{x}_", **means.to_dict())
save_variables(
    format="{:.1f}%",
    prefix=f"column_summary_{x}_",
    **weighted_means(column_cells, "broad_type", x, weight_x).to_dict(),
)

# %% S4: COMPARTMENTS OF THE SYNAPSES BETWEEN CELL TYPE PAIRS OF THE COLUMN
column_ids = column_cells.index.to_list()
pair_counts = (
    Tables["column_post_synapses"]
    .lazy()
    .filter(pl.col("pre_pt_root_id").is_in(column_ids), pl.col("tag") != "unknown")
    .group_by("pre_cell_type", "post_cell_type", "tag")
    .len("synapse_count")
    .with_columns(
        total_count=pl.col("synapse_count").sum().over("pre_cell_type", "post_cell_type")
    )
    .with_columns(synapse_proportion=pl.col("synapse_count") / pl.col("total_count"))
    .collect()
)

plot_categories = [c for c in CELL_TYPE_CATEGORIES if c not in ("TH", "Unk")]
n_exc, n_inh = 7, 4
label_fontsize = "small"
tag_colors = [COMPARTMENT_PALETTE_MUTED_HEX[tag] for tag in TAG_CATEGORIES]

fig = plt.figure(figsize=(7.17, 7.17), dpi=300)
outer_gs = GridSpec(
    3,
    3,
    width_ratios=[0.5, n_exc, n_inh],
    height_ratios=[0.5, n_exc, n_inh],
    wspace=0.07,
    hspace=0.07,
    figure=fig,
)
bracket_kws = dict(axis_off=True, normalize_limits=True, text_kws={"fontsize": label_fontsize})
x_shift = -2.7
ax = fig.add_subplot(outer_gs[1, 0])
draw_bracket(ax, 0, 1, axis="y", label="Excitatory", shift=x_shift, **bracket_kws)
ax.text(
    x_shift - 1.7,
    0.2,
    "Presynaptic cell type",
    rotation=90,
    rotation_mode="anchor",
    va="center",
    ha="center",
)
ax = fig.add_subplot(outer_gs[2, 0])
draw_bracket(ax, 0, 1, axis="y", label="Inhibitory", shift=x_shift, **bracket_kws)
y_shift = -2.2
ax = fig.add_subplot(outer_gs[0, 1])
draw_bracket(ax, 0, 1, axis="x", label="Excitatory", shift=y_shift, **bracket_kws)
ax.text(0.8, y_shift - 1.7, "Postsynaptic cell type", va="center", ha="center")
ax = fig.add_subplot(outer_gs[0, 2])
draw_bracket(ax, 0, 1, axis="x", label="Inhibitory", shift=y_shift, **bracket_kws)

space = 0.06
blocks = {
    (0, 0): (n_exc, n_exc, outer_gs[1, 1]),
    (0, n_exc): (n_exc, n_inh, outer_gs[1, 2]),
    (n_exc, 0): (n_inh, n_exc, outer_gs[2, 1]),
    (n_exc, n_exc): (n_inh, n_inh, outer_gs[2, 2]),
}
axs = np.empty((n_exc + n_inh, n_exc + n_inh), dtype=object)
for (i0, j0), (n_rows, n_cols, spec) in blocks.items():
    gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=spec, wspace=space, hspace=space)
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i0 + i, j0 + j] = fig.add_subplot(gs[i, j])

for i, pre_type in enumerate(plot_categories):
    for j, post_type in enumerate(plot_categories):
        ax = axs[i, j]
        ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1))
        pair = pair_counts.filter(
            pl.col("pre_cell_type") == pre_type, pl.col("post_cell_type") == post_type
        )
        if pair.height == 0 or pair["total_count"][0] < 10:
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.set_facecolor("#d0d0d0")
            continue
        props = dict(zip(pair["tag"].cast(pl.String), pair["synapse_proportion"]))
        start = 0.0
        for tag, color in zip(TAG_CATEGORIES, tag_colors):
            amount = props.get(tag, 0.0)
            ax.add_patch(Rectangle(xy=(start, 0.0), width=amount, height=1.0, color=color))
            start += amount
        ax.spines[["left", "right", "top", "bottom"]].set_visible(True)
        ax.spines[["left", "right", "top", "bottom"]].set_linewidth(1.0)

for i, pre_type in enumerate(plot_categories):
    axs[i, 0].set_ylabel(
        remapper(pre_type), fontsize=label_fontsize, rotation=0, ha="right", va="center"
    )
for j, post_type in enumerate(plot_categories):
    axs[0, j].set_title(
        remapper(post_type),
        fontsize=label_fontsize,
        rotation=45,
        ha="left",
        va="bottom",
        rotation_mode="anchor",
        pad=1,
    )
save_matplotlib_figure(fig, "cell_type_pair_waffles", out_path)
