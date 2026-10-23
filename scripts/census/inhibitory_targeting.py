"""Figure 5E right and 5F: the share of multi-input spines among the excitatory spines
that an axon type targets, by the distance to the postsynaptic soma.

Scope: full run. It reads the synapses from the proofread cells onto every target.
Run it with `--scratch` until `proofread_pre_synapses` exists (see
`analysis.census_inputs`).

    uv run python scripts/census/inhibitory_targeting.py [--scratch]
"""

# %%
import sys

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

from analysis import CELL_TYPE_PALETTE, save_matplotlib_figure, set_matplotlib_theme
from analysis.census_inputs import glue, out_path, pre_synapses

SCRATCH = "--scratch" in sys.argv
set_matplotlib_theme()
out_path = out_path(SCRATCH)
synapses = pre_synapses(SCRATCH)

y = "post_euc_distance_to_nuc_um"


def spine_synapses(
    group: str, groups: list[str], max_distance_um: float
) -> pd.DataFrame:
    """The spine synapses onto excitatory cells from the proofread axons whose
    `group` is in `groups`, with `is_multi` and a distance bin per group."""
    data = (
        synapses.filter(
            pl.col("post_broad_type") == "excitatory",
            pl.col("tag_detailed").is_in(["single_spine", "multi_spine"]),
            pl.col("pre_axon_cleaned"),
            pl.col(group).is_in(groups),
        )
        .select(
            pl.col(group).cast(pl.String),
            (pl.col("post_euc_distance_to_nuc") / 1000).alias(y),
            is_multi=pl.col("tag_detailed") == "multi_spine",
        )
        .filter(pl.col(y) < max_distance_um)
        .collect()
        .to_pandas()
    )
    return data


def add_bins(data: pd.DataFrame, group: str, n_bins: int) -> pd.DataFrame:
    """Add `{y}_bin_rep`, the middle of the quantile bin of `y` within each group."""
    data[f"{y}_bin_rep"] = data.groupby(group)[y].transform(
        lambda d: pd.qcut(d, q=n_bins).map(lambda b: b.mid).astype(float)
    )
    return data


# %% 5F: BASKET AND MARTINOTTI AXONS
hue_order = ["BC", "MC"]
data = add_bins(spine_synapses("pre_cell_type", hue_order, 300), "pre_cell_type", 25)
color = CELL_TYPE_PALETTE["inhibitory"]

fig, axs = plt.subplots(1, 2, figsize=(4.56, 7.17), sharey=True, layout="constrained")
ax = axs[0]
for cell_type, linestyle in zip(hue_order, ["-", "--"]):
    sns.histplot(
        data=data.query("pre_cell_type == @cell_type"),
        y=y,
        ax=ax,
        element="poly",
        stat="density",
        color=color,
        linestyle=linestyle,
        alpha=0.1,
        linewidth=2,
    )
ax.spines["bottom"].set_visible(False)
ax.set(
    xlabel="Normalized\ndensity\n(spines)",
    ylabel="Distance to postsynaptic soma (um)",
    xticks=[],
)
ax.text(0.2, 0.65, "Martinotti\n(MC)", transform=ax.transAxes, color=color)
ax.text(0.4, 0.04, "Basket\n(BC)", transform=ax.transAxes, color=color).set_in_layout(
    False
)
sns.lineplot(
    data=data,
    y=f"{y}_bin_rep",
    x="is_multi",
    orient="y",
    hue="pre_cell_type",
    hue_order=hue_order,
    style="pre_cell_type",
    style_order=hue_order,
    palette=dict.fromkeys(hue_order, color),
    ax=axs[1],
    legend=False,
)
axs[1].set(
    xlabel="Proportion of\nspines with\nmultiple inputs",
    xlim=(0, None),
    xticks=[0.0, 0.4, 0.8],
    ylim=(None, 200),
)
save_matplotlib_figure(
    fig, "bc_vs_mc_multi_spine_prop_by_euc_distance_to_nuc", out_path, pad_inches=0.2
)

# %% 5E, RIGHT: LOCAL EXCITATORY AND THALAMIC AXONS
hue_order = ["excitatory", "thalamic"]
data = add_bins(spine_synapses("pre_broad_type", hue_order, 200), "pre_broad_type", 20)

fig, axs = plt.subplots(1, 2, figsize=(4.56, 7.17), sharey=True, layout="constrained")
ax = axs[0]
sns.histplot(
    data=data,
    y=y,
    hue="pre_broad_type",
    hue_order=hue_order,
    ax=ax,
    palette=CELL_TYPE_PALETTE,
    element="poly",
    stat="density",
    common_norm=False,
    legend=False,
)
ax.spines["bottom"].set_visible(False)
ax.set(
    xlabel="Normalized\ndensity\n(spines)",
    ylabel="Distance to postsynaptic soma (um)",
    xticks=[],
)
sns.lineplot(
    data=data,
    y=f"{y}_bin_rep",
    x="is_multi",
    hue="pre_broad_type",
    hue_order=hue_order,
    orient="y",
    palette=CELL_TYPE_PALETTE,
    ax=axs[1],
    legend=False,
)
axs[1].set(xlabel="Proportion of\nspines with\nmultiple inputs", xlim=(0, None))
save_matplotlib_figure(
    fig,
    "multi_spine_prop_by_euc_distance_to_nuc_by_exc_broad_type",
    out_path,
    pad_inches=0.2,
)

# %% THE DISTANCE TO THE SOMA THAT 95% OF THE BASKET CELL OUTPUTS FALL WITHIN
q95 = (
    synapses.filter(pl.col("pre_cell_type") == "BC")
    .select(pl.col("post_euc_distance_to_nuc").quantile(0.95))
    .collect()
    .item()
)
glue(
    SCRATCH,
    prefix="column_summary_",
    bc_post_euc_distance_to_nuc_q95=q95 / 1000,
    format="{:.0f}",
)
