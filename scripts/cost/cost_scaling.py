"""The compute time per neuron against its mesh size and its input synapses, with an
OLS fit of each, and the distribution of the time (and of its cost) at the sides.

- Time: the summed `elapsed_s` of the attempts that wrote the census tables, per
  root, from the deployment log table. Each worker holds one vCPU and runs one
  thread, so this is single-vCPU wall time.
- Mesh size: the `n_vertices` metric of the mesh loader in the same log.
- Cost: the time times the dollars per step-hour of the full-run session (see
  `analysis.products.deployment_logs`).

Scope: full run. Run it with `--scratch` until `post_aggregation` exists (see
`analysis.census_inputs`).

    uv run python scripts/cost/cost_scaling.py [--scratch]
"""

# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

from analysis import FIG_PATH, Tables, save_matplotlib_figure, set_matplotlib_theme
from analysis.census_inputs import post_cells
from analysis.products.deployment_logs import dollars_per_step_hour

SCRATCH = "--scratch" in sys.argv
OUT_PATH = FIG_PATH / "cost" / ("_scratch" if SCRATCH else "")
FIT_COLOR = "#d62728"
MARGIN_COLOR = "#8c8c8c"
LABEL_BOX = dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9)

set_matplotlib_theme()

# %% TIME, MESH SIZE AND SYNAPSES PER NEURON
seconds = (
    Tables["census_step_hours"]
    .lazy()
    .group_by("root_id")
    .agg(pl.col("elapsed_s").sum())
    .collect()
)
neurons = (
    post_cells(SCRATCH)
    .select(pl.col("pt_root_id").alias("root_id"), "post_total_synapses")
    .join(seconds, on="root_id", how="inner")
    .join(Tables["root_mesh_vertices"].collect(), on="root_id", how="inner")
)
dollars_per_hour = dollars_per_step_hour()
print(f"{neurons.height:,} neurons with a time and a vertex count")
print(f"${dollars_per_hour:.4f} per step-hour")

# %% FIGURE
sizes = [
    ("n_vertices", "Mesh vertices (millions)", 1e6, "M vertices"),
    ("post_total_synapses", "Input synapses (thousands)", 1e3, "k synapses"),
]
minutes = neurons["elapsed_s"].to_numpy() / 60
# The axes stop at the 99.5th percentile, so that the bulk of the neurons is visible.
ymax = np.quantile(minutes, 0.995)
t = stats.t.ppf(0.975, len(minutes) - 2)


def to_dollars(m):
    return m / 60 * dollars_per_hour


def to_minutes(d):
    return d / dollars_per_hour * 60


fig, axs = plt.subplots(
    1,
    4,
    figsize=(18, 7),
    sharey=True,
    layout="constrained",
    gridspec_kw={"width_ratios": [1, 4, 4, 1]},
)
kde_left, *scatters, kde_right = axs
for ax, (column, xlabel, scale, unit) in zip(scatters, sizes):
    x = neurons[column].to_numpy().astype(float) / scale
    xmax = np.quantile(x, 0.995)
    fit = stats.linregress(x, minutes)
    sns.scatterplot(
        x=x,
        y=minutes,
        ax=ax,
        color="black",
        s=4,
        alpha=0.08,
        linewidth=0,
        rasterized=True,
    )
    grid = np.linspace(0, xmax, 200)
    ax.plot(
        grid, fit.intercept + fit.slope * grid, color=FIT_COLOR, lw=3, label="OLS fit"
    )
    ax.text(
        0.03,
        0.97,
        f"{fit.intercept * 60:.0f} ± {t * fit.intercept_stderr * 60:.0f} s\n"
        f"+ {fit.slope * 60:.0f} ± {t * fit.stderr * 60:.0f} s per {unit}\n"
        f"R² {fit.rvalue**2:.2f}",
        transform=ax.transAxes,
        va="top",
    )
    ax.set(xlim=(0, xmax), xlabel=xlabel)
    ax.tick_params(labelleft=False)
    print(f"{column}: {fit}")
scatters[1].legend(frameon=False, loc="lower right")

# The time of every neuron on both sides, so the two KDEs are one distribution. The
# mean is over all neurons, with the 0.5% above the axis.
mean = minutes.mean()
for ax in (kde_left, kde_right):
    # The KDE is fit on every neuron and drawn on the visible range only.
    sns.kdeplot(
        y=minutes,
        ax=ax,
        fill=True,
        color=MARGIN_COLOR,
        alpha=1,
        linewidth=0,
        clip=(0, ymax),
        gridsize=500,
    )
    ax.axhline(mean, color="black", lw=1.5, ls="--")
    ax.set(xticks=[], xlabel="Neurons")
kde_left.invert_xaxis()
kde_right.spines[["left"]].set_visible(False)
kde_left.set(ylim=(0, ymax), ylabel="Single-vCPU time per neuron (minutes)")
kde_left.text(
    0.95,
    mean + 0.6,
    f"mean\n{mean:.1f} min",
    transform=kde_left.get_yaxis_transform(),
    ha="right",
    va="bottom",
    bbox=LABEL_BOX,
    fontsize="small",
)
kde_right.text(
    0.05,
    mean + 0.6,
    rf"mean" "\n" rf"\${to_dollars(mean):.4f}",
    transform=kde_right.get_yaxis_transform(),
    ha="left",
    va="bottom",
    bbox=LABEL_BOX,
    fontsize="small",
)
kde_right.tick_params(labelleft=False)
dollar_axis = kde_right.secondary_yaxis("right", functions=(to_dollars, to_minutes))
dollar_axis.set_ylabel(r"Cost per neuron (\$USD)")
dollar_axis.yaxis.set_major_formatter(lambda d, _: rf"\${d:.4f}")
print(f"mean {mean:.2f} min, ${to_dollars(mean):.5f} per neuron")
save_matplotlib_figure(fig, "time_vs_mesh_vertices_and_synapses", OUT_PATH)
