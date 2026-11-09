"""Figure 5G, 5H and S8: the hand-checked multi-input spines with two proofread inputs.

Scope: full run. The spines are those of `multi_spine_labels` (every excitatory cell
with a spine whose inputs are all proofread), tagged `singlehead` by hand. The S8
background is the putative single-input spines of the proofread cells.

    uv run python scripts/census/multi_synapse_spines.py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.stats import pearsonr

from analysis import (
    CELL_TYPE_PALETTE,
    FIG_PATH,
    Tables,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)
from analysis.sources import (
    SYNAPSE_SIZE_LABEL,
    SYNAPSE_SIZE_LABEL_VX,
    SYNAPSE_VOXEL_VOLUME_NM3,
    synapse_size_nm3,
)

set_matplotlib_theme()
out_path = FIG_PATH / "census"

# %% THE SYNAPSES OF THE HAND-CHECKED DOUBLE-INPUT SPINES
# A double-input spine has 2 presynaptic roots. `singlehead` marks the spines where
# both axons synapse onto the same single head.
labels = Tables["multi_spine_labels"].lazy().filter(pl.col("singlehead")).collect()
synapses = (
    Tables["proofread_pre_synapses"]
    .lazy()
    .filter(pl.col("synapse_id").is_in(labels["synapse_ids"].explode().implode()))
    .select(
        "synapse_id",
        "pre_pt_root_id",
        "post_pt_root_id",
        "structure_id",
        "pre_broad_type",
        "size",
        "spine_size_nm3",
        "spine_n_pre_root_ids",
    )
    .collect()
)
assert len(synapses) == labels["synapse_ids"].list.len().sum()
double_synapses = synapses.filter(pl.col("spine_n_pre_root_ids") == 2)

# One row per spine: the number of presynaptic roots of each broad type.
input_counts = (
    double_synapses.unique(["post_pt_root_id", "structure_id", "pre_pt_root_id"])
    .with_columns(pl.col("pre_broad_type").cast(pl.String))
    .pivot(
        on="pre_broad_type",
        index=["post_pt_root_id", "structure_id"],
        values="pre_pt_root_id",
        aggregate_function="len",
    )
    .fill_null(0)
    .to_pandas()
)
for broad_type in ["excitatory", "inhibitory", "thalamic"]:
    if broad_type not in input_counts:
        input_counts[broad_type] = 0
input_counts["all_excitatory"] = input_counts["excitatory"] + input_counts["thalamic"]
print(f"{len(labels)} singlehead spines, {len(input_counts)} with 2 inputs")

# %% 5G: THE PAIRING OF THE TWO INPUTS
pairing_props = (
    input_counts[["all_excitatory", "inhibitory"]]
    .value_counts(normalize=True)
    .reset_index()
)
print(input_counts[["all_excitatory", "inhibitory"]].value_counts())

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
shift = 0.2
right = -0.05
for i, row in pairing_props.iterrows():
    x = row["proportion"]
    first = "inhibitory" if row["inhibitory"] >= 1 else "excitatory"
    second = "excitatory" if row["all_excitatory"] > 0 else "inhibitory"
    ax.barh(i + shift, x, height=0.4, color=CELL_TYPE_PALETTE[first])
    ax.barh(i - shift, x, height=0.4, color=CELL_TYPE_PALETTE[second])
    ax.plot([0, x], [i, i], color="black", linewidth=2)
    ax.text(right, i + shift, first.capitalize(), ha="right", va="center")
    ax.text(right, i - shift, second.capitalize(), ha="right", va="center")
    ax.text(x + 0.02, i, f"{x:.2f}", ha="left", va="center")
ax.set(yticks=[], xlabel="Proportion of \ndouble-input spines")
ax.set_ylabel("Input synapse pairing", labelpad=115)
ax.invert_yaxis()
save_matplotlib_figure(fig, "multi_spine_pairing", out_path, pad_inches=0.2)

# %% 5H: SYNAPSE SIZE AND SPINE VOLUME ON THE SPINES WITH ONE E AND ONE I INPUT
ei_spines = input_counts.query("excitatory == 1 & inhibitory == 1 & thalamic == 0")[
    ["post_pt_root_id", "structure_id"]
]
ei_synapses = (
    double_synapses.join(
        pl.from_pandas(ei_spines), on=["post_pt_root_id", "structure_id"], how="semi"
    )
    .with_columns(pl.col("pre_broad_type").cast(pl.String))
    .to_pandas()
)
print(f"{len(ei_spines)} spines with one excitatory and one inhibitory input")

# The unit of a synapse `size` is defined once, in `analysis.sources`. nm³ is what the
# paper reports, so it takes the plain filename; the voxel version keeps a `_vx`
# suffix, for checking against the older figures.
ei_synapses["size_nm3"] = synapse_size_nm3(ei_synapses["size"])
UNITS = {
    "nm3": ("size_nm3", SYNAPSE_SIZE_LABEL, "", SYNAPSE_VOXEL_VOLUME_NM3),
    "vx": ("size", SYNAPSE_SIZE_LABEL_VX, "_vx", 1),
}
y = "spine_size_nm3"

x = "size"
for broad_type in ["excitatory", "inhibitory"]:
    data = ei_synapses.query("pre_broad_type == @broad_type")
    pearson_r, pearson_p = pearsonr(np.log(data[x]), np.log(data[y]))
    print(f"{broad_type}: n={len(data)}, r={pearson_r:.2f}, p={pearson_p:.2e}")
    save_variables(
        prefix="column_summary_",
        format="{:.2f}",
        **{f"multi_synapse_{broad_type}_pearson_r": pearson_r},
    )
    save_variables(
        prefix="column_summary_",
        format="{:.2e}" if broad_type == "excitatory" else "{:.2f}",
        **{f"multi_synapse_{broad_type}_pearson_p": pearson_p},
    )
    save_variables(
        prefix="column_summary_",
        format="{:.0f}",
        **{f"multi_synapse_{broad_type}_mean_cleft_size": data[x].mean()},
    )

for x, xlabel, suffix, _ in UNITS.values():
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 6), layout="constrained", sharex=True, sharey=True
    )
    for ax, broad_type in zip(axs, ["excitatory", "inhibitory"]):
        data = ei_synapses.query("pre_broad_type == @broad_type")
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            color=CELL_TYPE_PALETTE[broad_type],
            s=25,
            ax=ax,
            legend=False,
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=f"{broad_type.capitalize()} {xlabel}",
            ylabel="Spine volume (nm³)",
        )
        pearson_r, _ = pearsonr(np.log(data[x]), np.log(data[y]))
        ax.text(
            0.05,
            0.95,
            f"Pearson r: {pearson_r:.2f}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize="medium",
        )
    save_matplotlib_figure(fig, f"multi_synapse_size_vs_spine_volume{suffix}", out_path)

# %% S8: SYNAPSE SIZE AND SPINE VOLUME, SINGLE- AND MULTI-INPUT SPINES
# The background is the putative single-input spines of the proofread cells, from
# proofread excitatory axons: one presynaptic root, on a structure that the unitary
# classifier calls one spine.
proofread_ids = (
    Tables["cell_selection"]
    .lazy()
    .filter(pl.col("is_proofread"))
    .select("pt_root_id")
    .collect()
)
single = (
    Tables["proofread_pre_synapses"]
    .lazy()
    .filter(
        pl.col("post_pt_root_id").is_in(proofread_ids["pt_root_id"].implode()),
        pl.col("post_broad_type") == "excitatory",
        pl.col("post_compartment") == "dendrite",
        pl.col("pre_compartment") == "axon",
        pl.col("pre_broad_type") == "excitatory",
        pl.col("tag") == "spine",
        pl.col("spine_size_nm3").is_not_null(),
        pl.col("spine_n_pre_root_ids") == 1,
        pl.col("spine_unitary_pred_label") == "unitary",
    )
    .select("size", y)
    .collect()
    .to_pandas()
)
single["size_nm3"] = synapse_size_nm3(single["size"])
print(f"{len(single):,} putative single-input spine synapses")

# The correlation on the log-log axes of the plot. The nm³ conversion is a constant
# factor, so it does not change the log-log r. The p value underflows to 0, so it is not
# glued.
pearson_r, _ = pearsonr(np.log(single["size"]), np.log(single[y]))
print(f"single: n={len(single):,}, r={pearson_r:.2f}")
save_variables(
    prefix="column_summary_",
    format="{:.2f}",
    single_synapse_excitatory_pearson_r=pearson_r,
)
save_variables(
    prefix="column_summary_",
    format="{:,}",
    single_synapse_excitatory_n=len(single),
)

# The legend gives the log-log Pearson r of each set, the same r as the glued variables.
# Its box is semi-transparent, so the few points under it still show.
for x, xlabel, suffix, scale in UNITS.values():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for data, s, alpha, linewidth, label, color in [
        (single, 2, 0.7, 0, "Putative single-input", "dimgrey"),
        (
            ei_synapses.query("pre_broad_type == 'excitatory'"),
            10,
            1,
            0.2,
            "Verified multi-input",
            "firebrick",
        ),
    ]:
        pearson_r, _ = pearsonr(np.log(data[x]), np.log(data[y]))
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            s=s,
            alpha=alpha,
            ax=ax,
            linewidth=linewidth,
            label=f"{label} ($r$={pearson_r:.2f})",
            color=color,
        )
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(0.5e7, 1e9),
        xlim=(0.15e3 * scale, 1e5 * scale),
    )
    ax.set(xlabel=f"Excitatory {xlabel}", ylabel="Spine volume (nm³)")
    ax.legend(
        markerscale=2,
        loc="upper left",
        bbox_to_anchor=(0.01, 1.03),
        handlelength=1,
        handletextpad=0.4,
        framealpha=0.7,
    )
    save_matplotlib_figure(
        fig, f"spine_synapse_area_corr_by_multi{suffix}", out_path, pad_inches=0.1
    )
