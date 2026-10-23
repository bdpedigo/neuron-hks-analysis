"""Figure 5A, 5C, 5E left and S7: the multi-input spines of the column cells.

Scope: test cohort. It reads only the synapses onto the column cells and their spines.

    uv run python scripts/census/input_multi_spines.py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from analysis import (
    CELL_TYPE_PALETTE,
    COMPARTMENT_PALETTE_MUTED_HEX,
    FIG_PATH,
    Stash,
    Tables,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)
from analysis.plots import (
    add_cell_type_y,
    filter_and_bin,
    multi_density_histplot,
    multi_proportion_lineplot,
    sorted_stripplot_with_means,
    weighted_means,
)

set_matplotlib_theme()
out_path = FIG_PATH / "census"
rng = np.random.default_rng(8888)

column_cells = add_cell_type_y(
    Tables["proofread_cell_info"]
    .lazy()
    .filter(pl.col("is_column"))
    .collect()
    .to_pandas()
    .set_index("pt_root_id")
)

# %% THE SPINES OF THE EXCITATORY COLUMN CELLS
# One row per spine, with its volume, `is_multi`, the mean distance of its synapses
# to the soma, and whether a proofread excitatory or thalamic axon targets it.
spine_inputs = [Tables["column_post_synapses"], Tables["column_spines"], Tables["cell_selection"]]
spines_stash = Stash("census/excitatory_column_spines", inputs=spine_inputs)
if not spines_stash.exists():
    proofread_ids = (
        Tables["cell_selection"].lazy().filter(pl.col("is_proofread")).select("pt_root_id")
    )
    pre_type = pl.col("pre_broad_type")
    from_proofread = pl.col("pre_pt_root_id").is_in(proofread_ids.collect()["pt_root_id"])
    spine_synapses = (
        Tables["column_post_synapses"]
        .lazy()
        .filter(
            pl.col("post_broad_type") == "excitatory",
            pl.col("tag") == "spine",
            pl.col("structure_id").is_not_null(),
        )
        .group_by(root_id="post_pt_root_id", structure_id="structure_id")
        .agg(
            pl.col("post_cell_type").first(),
            (pl.col("post_euc_distance_to_nuc").mean() / 1000).alias(
                "post_euc_distance_to_nuc_um"
            ),
            (from_proofread & (pre_type == "excitatory")).any().alias("from_excitatory"),
            (from_proofread & (pre_type == "thalamic")).any().alias("from_thalamic"),
        )
    )
    spines_stash.save(
        Tables["column_spines"]
        .lazy()
        .select("root_id", "structure_id", "size_nm3", "is_multi")
        .join(spine_synapses, on=["root_id", "structure_id"], how="inner")
        .with_columns(
            tag_detailed=pl.when("is_multi")
            .then(pl.lit("multi_spine"))
            .otherwise(pl.lit("single_spine"))
        )
        .collect()
    )
spines = spines_stash.load().to_pandas()
print(f"{len(spines):,} spines on excitatory column cells")

# %% 5A: PROPORTION OF MULTI-INPUT SPINES BY CELL TYPE
x = "post_p_spine_site_is_multi"
weight_x = "post_spine_synapses"
fig, axs, means = sorted_stripplot_with_means(
    data=column_cells,
    x=x,
    weight_x=weight_x,
    size=weight_x,
    size_title="Input\nsynapses\nto spine",
    size_norm=(2000, 20000),
    legend_sizes=(2000, 10000, 20000),
    loc="upper right",
    bbox_to_anchor=(0.98, 0.985),
    xlim=(0, 0.6),
    xlabel="Proportion of spines with multiple inputs",
)
save_matplotlib_figure(fig, "cell_type_input_proportion_multi_spine", out_path)
save_variables(format="{:.1f}%", prefix=f"column_summary_{x}_", **means.to_dict())
save_variables(
    format="{:.1f}%",
    prefix=f"column_summary_{x}_",
    **weighted_means(column_cells.dropna(subset=[x]), "broad_type", x, weight_x).to_dict(),
)

p_23P = column_cells.query("cell_type == '23P'")[x]
save_variables(
    prefix=f"column_summary_{x}_",
    format="{:.1f}%",
    **{
        "23P_min": p_23P.min(),
        "23P_max": p_23P.max(),
        "23P_q90": p_23P.quantile(0.9),
        "23P_q10": p_23P.quantile(0.1),
    },
)

# %% 5C, LEFT: SPINE VOLUME AND THE PROPORTION OF MULTI-INPUT SPINES
low, high = spines["size_nm3"].quantile([0.001, 0.999])
size_bins = np.geomspace(low, high, 50)
spines["spine_size_bin_mid"] = (
    pd.qcut(spines["size_nm3"], q=50).map(lambda b: b.mid).astype(float)
)

fig, axs = plt.subplots(1, 2, figsize=(4.58, 7.17), sharey=True)
ax = axs[0]
sns.histplot(
    data=spines,
    y="size_nm3",
    hue="tag_detailed",
    hue_order=["single_spine", "multi_spine"],
    ax=ax,
    bins=size_bins,
    common_norm=False,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    element="poly",
    stat="proportion",
    legend=False,
)
ax.set(
    ylim=(5e6, 1e9),
    yscale="log",
    xticks=[],
    ylabel="Spine volume (nm³)",
    xlabel="Normalized\ndensity\n(spines)",
)
ax.spines[["bottom"]].set_visible(False)
for ypos, label, tag in [
    (0.1, "Single-\ninput\nspines", "single_spine"),
    (0.85, "Multi-\ninput\nspines", "multi_spine"),
]:
    ax.text(0.3, ypos, label, transform=ax.transAxes, color=COMPARTMENT_PALETTE_MUTED_HEX[tag])
ax = axs[1]
sns.lineplot(
    data=spines,
    y="spine_size_bin_mid",
    x="is_multi",
    orient="y",
    color="black",
    errorbar=None,
    ax=ax,
)
ax.set(xlabel="Proportion of\nspines with\nmultiple inputs")
save_matplotlib_figure(fig, "spine_size_vs_p_multi_rotated", out_path, pad_inches=0.2)

# %% 5C, RIGHT: DISTANCE TO THE SOMA AND THE PROPORTION OF MULTI-INPUT SPINES
y = "post_euc_distance_to_nuc_um"
data, bins = filter_and_bin(spines.dropna(subset=[y]), y, trim_quantiles=(0.0, 0.999))
bins = np.concatenate(([0], bins))
fig, axs = plt.subplots(1, 2, figsize=(4.51, 7.17), sharey=True)
multi_density_histplot(data, y, bins, axs[0])
axs[0].set(ylabel="Distance to postsynaptic soma (um)")
multi_proportion_lineplot(data, y, axs[1])
save_matplotlib_figure(
    fig, "multi_spine_prop_by_euc_distance_to_nuc", out_path, pad_inches=0.2
)

# %% 5E, LEFT: SPINES TARGETED BY LOCAL EXCITATORY AND BY THALAMIC AXONS
size_bins = np.geomspace(low, high, 20)
spines["spine_size_bin_mid"] = (
    pd.cut(spines["size_nm3"], bins=size_bins)
    .map(lambda b: b.mid, na_action="ignore")
    .astype(float)
)
groups = {
    "excitatory": ("Local\nexcitatory", "-", spines.query("from_excitatory")),
    "thalamic": ("Thalamic", "--", spines.query("from_thalamic")),
}
fig, axs = plt.subplots(1, 2, figsize=(4.58, 7.17), sharey=True)
for broad_type, (label, linestyle, group) in groups.items():
    color = CELL_TYPE_PALETTE[broad_type]
    sns.histplot(
        data=group,
        y="size_nm3",
        ax=axs[0],
        color=color,
        alpha=0.3,
        bins=25,
        log_scale=True,
        stat="proportion",
        element="poly",
        linestyle=linestyle,
    )
    sns.lineplot(
        data=group,
        y="spine_size_bin_mid",
        x="is_multi",
        orient="y",
        color=color,
        ax=axs[1],
        linestyle=linestyle,
    )
ax = axs[0]
ax.set(
    ylim=(low, high),
    yscale="log",
    ylabel="Spine volume (nm³)",
    xticks=[],
    xlabel="Normalized\ndensity\n(spines)",
)
ax.spines["bottom"].set_visible(False)
ax.text(0.2, 0.8, "Thalamic", transform=ax.transAxes, color=CELL_TYPE_PALETTE["thalamic"])
ax.text(
    0.2, 0.15, "Local\nexcitatory", transform=ax.transAxes, color=CELL_TYPE_PALETTE["excitatory"]
)
axs[1].set(xlabel="Proportion of\nspines with\nmultiple inputs", ylim=(5e6, 1e9))
save_matplotlib_figure(fig, "exc_vs_th_size_by_multi_rotated", out_path, pad_inches=0.2)
for broad_type, (_, _, group) in groups.items():
    print(f"{broad_type}: {len(group):,} targeted spines, p_multi {group['is_multi'].mean():.3f}")

# %% S7: OBSERVED AND NULL PROPORTION OF MULTI-INPUT SPINE SYNAPSES OF 23P CELLS
# The null shuffles `tag_detailed` among the spine synapses of the same size bin.
synapses_23P = (
    Tables["column_post_synapses"]
    .lazy()
    .filter(pl.col("post_cell_type") == "23P", pl.col("tag") == "spine")
    .select("post_pt_root_id", "size", pl.col("tag_detailed").cast(pl.String))
    .collect()
    .to_pandas()
)
size_bin = pd.cut(np.log(synapses_23P["size"]), bins=100)
synapses_23P["shuffled_tag_detailed"] = synapses_23P.groupby(size_bin, observed=True)[
    "tag_detailed"
].transform(lambda tags: rng.permutation(tags.values))


def p_multi(tag_col: str) -> pd.Series:
    return (
        synapses_23P[tag_col].eq("multi_spine").groupby(synapses_23P["post_pt_root_id"]).mean()
    )


bins = np.geomspace(0.03, 0.3, 40)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.histplot(x=p_multi("tag_detailed"), ax=ax, bins=bins, label="Observed", color="black")
sns.histplot(
    x=p_multi("shuffled_tag_detailed"),
    ax=ax,
    color="lightgrey",
    alpha=0.5,
    bins=bins,
    label="Null",
)
ticks = [0.1, 0.2, 0.3]
ax.set(
    xlabel="Proportion of spines with multiple inputs",
    xlim=(0.05, 0.3),
    xscale="log",
    ylabel="Cell count",
)
ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
ax.xaxis.set_major_formatter(plt.FixedFormatter(ticks))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.legend()
save_matplotlib_figure(fig, "multi_spine_proportion_null_23P", out_path)
