"""Figure 4C, 4D and 5D: the compartments of the output synapses of the proofread
cells, onto excitatory and inhibitory targets.

Scope: full run. It reads the synapses from the proofread cells onto every target.
Run it with `--scratch` until `proofread_pre_synapses` exists (see
`analysis.census_inputs`).

    uv run python scripts/census/outputs.py [--scratch]
"""

# %%
import sys

import polars as pl

from analysis import Tables, save_matplotlib_figure, set_matplotlib_theme
from analysis.census_inputs import glue, out_path, pre_synapses
from analysis.plots import add_cell_type_y, sorted_stripplot_with_means, weighted_means
from analysis.products.census import aggregate_pre

SCRATCH = "--scratch" in sys.argv
set_matplotlib_theme()
out_path = out_path(SCRATCH)
synapses = pre_synapses(SCRATCH).filter(pl.col("tag") != "unknown")

# %% COUNTS OF OUTPUT SYNAPSES PER CELL, BY TARGET BROAD TYPE AND COMPARTMENT
tag_counts = []
for target in ["excitatory", "inhibitory"]:
    counts = (
        synapses.filter(pl.col("post_broad_type") == target)
        .group_by("pre_pt_root_id")
        .agg(
            (pl.col("tag") == tag).sum().alias(f"pre_{tag}_synapse_to_{target}")
            for tag in ["spine", "shaft", "soma"]
        )
        .with_columns(
            pl.sum_horizontal(
                f"pre_{tag}_synapse_to_{target}" for tag in ["spine", "shaft", "soma"]
            ).alias(f"pre_total_synapse_to_{target}")
        )
        .with_columns(
            (
                pl.col(f"pre_{tag}_synapse_to_{target}")
                / pl.col(f"pre_total_synapse_to_{target}")
            ).alias(f"pre_p_{tag}_synapse_to_{target}")
            for tag in ["spine", "shaft", "soma"]
        )
    )
    tag_counts.append(counts)

# The thalamic cells have no soma in the volume, so their depth is the mean depth of
# their output synapses. It only orders the cells within a row.
output_depth = synapses.group_by("pre_pt_root_id").agg(
    output_depth=pl.col("ctr_pt_position_um_y").mean()
)
cells = Tables["proofread_cell_info"].lazy().rename({"pt_root_id": "pre_pt_root_id"})
for counts in tag_counts:
    cells = cells.join(counts, on="pre_pt_root_id", how="left")
cells = (
    cells.join(aggregate_pre(synapses), on="pre_pt_root_id", how="left")
    .join(output_depth, on="pre_pt_root_id", how="left")
    .with_columns(
        depth=pl.when(pl.col("is_thalamic"))
        .then("output_depth")
        .otherwise("pt_position_um_y")
    )
    .collect()
    .to_pandas()
)
cells = add_cell_type_y(cells, depth="depth")

# %% 4C, 4D: PROPORTION OF OUTPUTS ONTO SPINES, BY TARGET BROAD TYPE
legend_kws = {
    "excitatory": dict(loc="upper left", bbox_to_anchor=(0.02, 0.985)),
    "inhibitory": dict(loc="upper right", bbox_to_anchor=(0.98, 0.98)),
}
for target, kws in legend_kws.items():
    x = f"pre_p_spine_synapse_to_{target}"
    weight_x = f"pre_total_synapse_to_{target}"
    fig, axs, means = sorted_stripplot_with_means(
        data=cells,
        x=x,
        weight_x=weight_x,
        size=weight_x,
        size_title=f"Output\nsynapses\nto {target}",
        xlabel="Proportion of synapses on spines",
        **kws,
    )
    save_matplotlib_figure(
        fig, f"cell_type_output_proportion_to_{target}_scatter", out_path
    )
    glue(SCRATCH, format="{:.1f}%", prefix=f"column_summary_{x}_", **means.to_dict())
    glue(
        SCRATCH,
        format="{:.1f}%",
        prefix=f"column_summary_{x}_",
        **weighted_means(cells, "broad_type", x, weight_x).to_dict(),
    )

# %% MEAN SIZE OF THE SYNAPSES ONTO EXCITATORY CELLS, BY PRESYNAPTIC TYPE AND COMPARTMENT
sizes = (
    synapses.filter(pl.col("post_broad_type") == "excitatory")
    .group_by("pre_broad_type", "tag")
    .agg(pl.col("size").mean())
    .collect()
)


def mean_size(pre_broad_type: str, tag: str) -> float:
    row = sizes.filter(pl.col("pre_broad_type") == pre_broad_type, pl.col("tag") == tag)
    return row["size"][0] if row.height else float("nan")


glue(
    SCRATCH,
    prefix="column_summary_target_size_",
    th_spine=mean_size("thalamic", "spine"),
    th_shaft=mean_size("thalamic", "shaft"),
    exc_spine=mean_size("excitatory", "spine"),
    exc_shaft=mean_size("excitatory", "shaft"),
    format="{:,.0f}",
)

# %% 5D: PROPORTION OF MULTI-INPUT SPINES AMONG THE EXCITATORY SPINES A CELL TARGETS
x = "pre_to_exc_p_spine_site_is_multi"
weight_x = "pre_to_exc_spine_synapses"
fig, axs, means = sorted_stripplot_with_means(
    data=cells,
    x=x,
    weight_x=weight_x,
    size=weight_x,
    size_title="Output\nsynapses\nto excitatory\nspines",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.985),
    xlabel="Proportion of spines with multiple inputs",
)
save_matplotlib_figure(
    fig, "cell_type_output_proportion_multi_spine_to_excitatory", out_path
)
glue(SCRATCH, format="{:.1f}%", prefix=f"column_summary_{x}_", **means.to_dict())
glue(
    SCRATCH,
    format="{:.1f}%",
    prefix=f"column_summary_{x}_",
    **weighted_means(cells.dropna(subset=[x]), "broad_type", x, weight_x).to_dict(),
)
