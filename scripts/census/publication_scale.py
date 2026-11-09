"""The spines shown in the publication of each compared method, against ours.

The compared methods come from `publication_spine_counts`, summed per method. Ours
is the number of spine structures with at least one synapse in
`spine_census_features`. The labels round the other methods up, and ours to the
nearest unit.

    uv run python scripts/census/publication_scale.py
"""

# %%
import math

import humanize
import matplotlib.pyplot as plt
import polars as pl

from analysis import Tables, save_matplotlib_figure

OURS = "Ours"

totals = dict(
    Tables["publication_spine_counts"]
    .lazy()
    .group_by("method")
    .agg(pl.col("value").sum())
    .collect()
    .iter_rows()
)
totals[OURS] = (
    Tables["spine_census_features"]
    .lazy()
    .select((pl.col("n_synapses") >= 1).sum())
    .collect()
    .item()
)
print(totals)


# %%
def approx(value, up=False):
    # up: round up to the shown unit, so the other methods are never understated
    unit = 100 if value < 1e3 else 1e3 if value < 1e6 else 1e6
    value = (math.ceil if up else round)(value / unit) * unit
    if value < 1e3:
        return f"~{value:.0f}"
    return "~" + humanize.intword(int(value), "%.0f")


order = sorted(totals, key=totals.get)
y = range(len(order))
values = [totals[m] for m in order]
ink = "#0b0b0b"

fig, ax = plt.subplots(figsize=(6, 3.3))
ax.hlines(y, 1e2, values, color="#e4e3df", lw=2, zorder=1)
ax.scatter(values, y, s=60, color=ink, zorder=3)
for i, method in enumerate(order):
    ax.annotate(
        approx(totals[method], up=method != OURS),
        (totals[method], i),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=ink,
    )

ax.set_xscale("log")
ax.set_xlim(1e2, 1e9)
ax.set_yticks(list(y), order)
ax.set_ylim(-0.9, len(order) - 0.5)
ax.set_xlabel("Spines shown in publication (approximate)")
ax.grid(axis="x", color="#ecebe8", lw=0.8)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)

save_matplotlib_figure(fig, "spines_per_publication", subfolder="publication_scale", dpi=300)
