"""The compute time of the census by pipeline step, with the large steps split into
their timed parts.

- Step time: the summed `elapsed_s` of the attempts that wrote the census tables,
  per root and step, from `census_step_hours`. Each worker holds one vCPU and runs
  one thread, so this is single-vCPU wall time.
- Parts: some steps log the time of their parts in `metrics` (`SUBSTEP_KEYS`). A
  part is the sum of its metric over the same attempts. The rest of the step time
  (writes, process start) is "other". The attempts with no part metrics (older
  runs) are "not split".
- Groups: the steps in plain terms (`GROUPS`). The figure is one bar of the share
  of the time in each group.
- Cost: the time times the dollars per step-hour of the full-run session (see
  `analysis.products.deployment_logs`).

Scope: full run. Run it with `--scratch` until `post_aggregation` exists (see
`analysis.census_inputs`).

    uv run python scripts/cost/cost_breakdown.py [--scratch]
"""

# %%
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import optimize

from analysis import FIG_PATH, Tables, save_matplotlib_figure, set_matplotlib_theme
from analysis.census_inputs import post_cells
from analysis.products.deployment_logs import (
    FINAL_RUN,
    dollars_per_step_hour,
    read_log,
)
from analysis.sources import DATASTACK

SCRATCH = "--scratch" in sys.argv
OUT_PATH = FIG_PATH / "cost" / ("_scratch" if SCRATCH else "")
# The metrics that time the parts of a step, as {step: {metric: part name}}.
SUBSTEP_KEYS = {
    "condensed_spectral": {
        "conditioning_s": "conditioning",
        "spectral_s": "spectral",
        "condense_s": "condense",
    },
    "structure_anchors": {"anchor_s": "anchoring"},
    "structure_morphometry": {"morphometry_s": "morphometry"},
}
# The steps in plain terms, as {group: steps}.
GROUPS = {
    "Data loading": ["mesh", "cave", "synapses", "nuclei"],
    "Feature computation": ["condensed_spectral"],
    "Structure extraction": ["structure_anchors", "structure_morphometry"],
    "Model predictions": [
        "domain_predictions",
        "structure_predictions",
        "synapse_predictions",
    ],
    "Synapse mapping": ["synapse_mapping"],
}
# Categorical colors, in a fixed order over the groups by share.
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# The parts in plain terms, as {(step, part): name}. A step with no parts is its own
# part.
PART_NAMES = {
    ("condensed_spectral", "spectral"): "Spectral features",
    ("condensed_spectral", "conditioning"): "Mesh conditioning",
    ("condensed_spectral", "condense"): "Feature agglomeration",
    ("structure_morphometry", "morphometry"): "Morphometry",
    ("structure_anchors", "anchoring"): "Supervoxel anchoring",
    ("mesh", "mesh"): "Mesh loading",
    ("synapses", "synapses"): "Synapse loading",
}
# In the second bar, the parts with no name in `PART_NAMES` are one "Other" segment per
# group, with no label. These groups, and a group with one part, are one segment
# there, with no label.
UNSPLIT_GROUPS = ["Model predictions"]
# The shape of the line from a label to its segment: "straight", or "elbow" (out from
# the bar, across, then out to the label).
LEADER_STYLE = "elbow"
# The space between a bar and its near row of labels, in bar heights, by leader style.
LABEL_GAP = {"straight": 0.3, "elbow": 0.8}
# The heights an elbow can turn at, as fractions of the gap, in order of preference.
# An elbow takes the first one where its leader is clear of the leaders before it.
ELBOW_LEVELS = (0.5, 0.25, 0.75)
# The space between two labels, in percent of the bar.
LABEL_PAD = 1.0
# The space between the two rows of labels, and above and below the outermost labels,
# in bar heights.
ROW_PAD = 0.05
# The color of the "Other" parts.
OTHER_COLOR = "#d4d4d4"

set_matplotlib_theme()

# %% STEP TIME PER NEURON
roots = post_cells(SCRATCH).select(pl.col("pt_root_id").alias("root_id"))
steps = Tables["census_step_hours"].collect().join(roots, on="root_id", how="semi")
step_hour_dollars = dollars_per_step_hour()
print(f"{roots.height:,} neurons, {steps['root_id'].n_unique():,} with a step time")
print(f"${step_hour_dollars:.4f} per step-hour")

# %% PART TIMES FROM THE LOG METRICS
# The same attempts as `census_step_hours`: the ok rows at the step hashes of
# `FINAL_RUN`, in any run. The loader rows have no part metrics, so they are left out.
rows = read_log(
    DATASTACK, columns=["run_id", "root_id", "step", "step_hash", "outcome", "metrics"]
).filter(pl.col("step").is_in(list(SUBSTEP_KEYS)))
final_hashes = (
    rows.filter(pl.col("run_id") == FINAL_RUN)
    .select("step", "step_hash")
    .unique()
    .drop_nulls()
)
part_metrics = pl.DataFrame(
    [(s, k, part) for s, keys in SUBSTEP_KEYS.items() for k, part in keys.items()],
    schema=["step", "key", "part"],
    orient="row",
)
parts = (
    rows.join(final_hashes, on=["step", "step_hash"], how="semi")
    .filter(pl.col("outcome") == "ok")
    .join(roots, on="root_id", how="semi")
    .select("root_id", "step", "metrics")
    .explode("metrics")
    .unnest("metrics")
    .join(part_metrics, on=["step", "key"], how="inner")
    .group_by("root_id", "step", "part")
    .agg(pl.col("value").sum().alias("elapsed_s"))
)

# The time of each step that its parts do not cover. A root and step with no part
# metrics is "not split".
split = parts.group_by("root_id", "step").agg(
    pl.col("elapsed_s").sum().alias("parts_s")
)
rest = (
    steps.join(split, on=["root_id", "step"], how="left")
    .with_columns(
        part=pl.when(pl.col("step").is_in(list(SUBSTEP_KEYS)).not_())
        .then(pl.col("step"))
        .when(pl.col("parts_s").is_null())
        .then(pl.lit("not split"))
        .otherwise(pl.lit("other")),
        elapsed_s=pl.col("elapsed_s") - pl.col("parts_s").fill_null(0),
    )
    .select("root_id", "step", "part", "elapsed_s")
)
pieces = pl.concat([parts, rest])
assert np.isclose(pieces["elapsed_s"].sum(), steps["elapsed_s"].sum())
print(f"{(rest['elapsed_s'] < 0).sum():,} root steps whose parts exceed the step time")

# %% BREAKDOWN TABLE
n_neurons = steps["root_id"].n_unique()
total_s = steps["elapsed_s"].sum()


def summarize(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    # The median is over the roots, so each root's time in a group is summed first.
    return (
        frame.group_by(*by, "root_id")
        .agg(pl.col("elapsed_s").sum())
        .group_by(by)
        .agg(
            (pl.col("elapsed_s").sum() / 3600).alias("step_hours"),
            pl.col("elapsed_s").median().alias("median_s"),
            pl.len().alias("n_roots"),
        )
        .with_columns(
            (pl.col("step_hours") * 3600 / total_s).alias("share"),
            (pl.col("step_hours") * 3600 / n_neurons).alias("mean_s_per_neuron"),
            (pl.col("step_hours") * step_hour_dollars).alias("dollars"),
        )
        .sort("step_hours", descending=True)
    )


step_groups = pl.DataFrame(
    [(s, g) for g, group_steps in GROUPS.items() for s in group_steps],
    schema=["step", "group"],
    orient="row",
)
assert set(steps["step"]) <= set(step_groups["step"])
steps = steps.join(step_groups, on="step")
pieces = pieces.join(step_groups, on="step")

by_group = summarize(steps, ["group"])
by_step = summarize(steps, ["group", "step"])
by_part = summarize(pieces, ["group", "step", "part"])
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
    print(by_group)
    print(by_step)
    print(by_part)
print(
    f"total {total_s / 3600:,.0f} step-hours, ${total_s / 3600 * step_hour_dollars:,.0f}, "
    f"{total_s / n_neurons / 60:.2f} min per neuron"
)
OUT_PATH.mkdir(parents=True, exist_ok=True)
by_part.write_csv(OUT_PATH / "cost_breakdown.csv")

# %% FIGURE
# Two bars of the share of the time: the groups on top, largest first, and their parts
# below, each under its group. The group labels sit above, and the part labels below.
# A label sits over its segment, or as close to it as the other labels and the ends of
# the bar allow, and it moves to a second row further out only if it must.
group_order = by_group["group"].to_list()
part_names = pl.DataFrame(
    [(s, p, name) for (s, p), name in PART_NAMES.items()],
    schema=["step", "part", "name"],
    orient="row",
)
parts_plotted = (
    by_part.join(part_names, on=["step", "part"], how="left")
    .with_columns(
        name=pl.when(
            pl.col("group").is_in(UNSPLIT_GROUPS) | (pl.len().over("group") == 1)
        )
        .then(pl.col("group"))
        .otherwise(pl.col("name").fill_null(pl.lit("Other")))
    )
    .group_by("group", "name")
    .agg(pl.col("share").sum())
    .with_columns(
        n_parts=pl.len().over("group"),
        group_rank=pl.col("group").cast(pl.Enum(group_order)),
        is_other=pl.col("name") == "Other",
    )
    .sort("group_rank", "is_other", "share", descending=[False, False, True])
)


def tint(color: str, amount: float) -> tuple:
    """The color mixed with white, or with black for a negative `amount`."""
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    return tuple(rgb + ((1 if amount > 0 else 0) - rgb) * abs(amount))


def spans(shares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lefts = np.concatenate([[0], np.cumsum(shares)[:-1]])
    return lefts, lefts + shares / 2


def place_on_rows(widths, centers, rows):
    """The x of each label, as close to its center as it can be (least squares), with
    the labels in order, each label clear of the next one on its row, and each label's
    center over the bar. None if no such x exists."""
    n = len(centers)
    constraints = [
        {"type": "ineq", "fun": lambda x, i=i: x[i + 1] - x[i]} for i in range(n - 1)
    ]
    for row in (0, 1):
        on_row = np.flatnonzero(rows == row)
        for i, j in itertools.pairwise(on_row):
            gap = (widths[i] + widths[j]) / 2 + LABEL_PAD
            constraints.append(
                {"type": "ineq", "fun": lambda x, i=i, j=j, gap=gap: x[j] - x[i] - gap}
            )
    bounds = [(0, 100)] * n
    x0 = centers
    result = optimize.minimize(
        lambda x: ((x - centers) ** 2).sum(),
        x0,
        jac=lambda x: 2 * (x - centers),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )
    feasible = result.success and all(c["fun"](result.x) > -1e-6 for c in constraints)
    return result.x if feasible else None


def leader_crosses(xs, rows, widths, heights, centers, tip_y, text_y, row_step):
    """Whether the leader line of a far-row label passes through a near-row label."""
    far_y = text_y + row_step
    for j in np.flatnonzero(rows == 1):
        for i in np.flatnonzero(rows == 0):
            # The leader's x where it passes the near-row label's band of y. An elbow
            # leader is vertical there, at the label's x.
            band = np.array([text_y, text_y + np.sign(row_step) * heights[i]])
            if LEADER_STYLE == "elbow":
                x_band = np.array([xs[j], xs[j]])
            else:
                x_band = centers[j] + (xs[j] - centers[j]) * (band - tip_y) / (
                    far_y - tip_y
                )
            half = widths[i] / 2 + LABEL_PAD / 2
            if x_band.max() > xs[i] - half and x_band.min() < xs[i] + half:
                return True
    return False


def label_segments(texts, lefts, shares, tip_y, direction):
    """Labels the segments above (`direction` 1) or below (-1) the bar. Of every
    choice of row for each label, it takes the one with the fewest labels on the far
    row, then the least movement, where no label overlaps another and no leader line
    passes through a label.

    An elbow leader starts at the point of its segment nearest its label. If that point
    is under the label, the leader is one vertical line instead."""
    lefts, shares = np.asarray(lefts), np.asarray(shares)
    centers = lefts + shares / 2
    text_y = tip_y + direction * LABEL_GAP[LEADER_STYLE]
    straight = dict(arrowstyle="-", color="black", lw=1, shrinkA=2, shrinkB=0)
    labels = [
        ax.annotate(
            text,
            xy=(center, tip_y),
            xytext=(center, text_y),
            ha="center",
            va="bottom" if direction > 0 else "top",
            fontsize="small",
            arrowprops=straight if LEADER_STYLE == "straight" else None,
        )
        for text, center in zip(texts, centers)
    ]
    # The label sizes in data units, measured on the drawn figure.
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    boxes = [label.get_window_extent().transformed(to_data) for label in labels]
    widths = np.array([box.width for box in boxes])
    heights = np.abs([box.height for box in boxes])
    row_step = direction * (heights.max() + ROW_PAD)
    n = len(labels)
    best = None
    for n_far in range(n + 1):
        for far in itertools.combinations(range(n), n_far):
            rows = np.isin(np.arange(n), far).astype(int)
            xs = place_on_rows(widths, centers, rows)
            if xs is None or leader_crosses(
                xs, rows, widths, heights, centers, tip_y, text_y, row_step
            ):
                continue
            moved = ((xs - centers) ** 2).sum()
            if best is None or moved < best[0]:
                best = (moved, xs, rows)
        if best is not None:
            break
    assert best is not None, "no placement of the labels clears them all"
    _, xs, rows = best
    leaders = []
    margin = LABEL_PAD / 2
    for label, left, share, center, x, width, row in zip(
        labels, lefts, shares, centers, xs, widths, rows
    ):
        label_y = text_y + row * row_step
        label.xyann = (x, label_y)
        if LEADER_STYLE != "elbow":
            continue
        end_y = label_y - direction * ROW_PAD
        tip_x = (
            np.clip(x, left + margin, left + share - margin)
            if share > 2 * margin
            else center
        )
        if abs(tip_x - x) < width / 2 - margin:
            leader = [(tip_x, tip_y), (tip_x, end_y)]
        else:
            for level in ELBOW_LEVELS:
                elbow_y = tip_y + direction * LABEL_GAP[LEADER_STYLE] * level
                leader = list(
                    zip([tip_x, tip_x, x, x], [tip_y, elbow_y, elbow_y, end_y])
                )
                if not any(leaders_touch(leader, other) for other in leaders):
                    break
            else:
                # No level is clear, so the elbow turns at the first one.
                elbow_y = tip_y + direction * LABEL_GAP[LEADER_STYLE] * ELBOW_LEVELS[0]
                leader = list(
                    zip([tip_x, tip_x, x, x], [tip_y, elbow_y, elbow_y, end_y])
                )
        leaders.append(leader)
        ax.plot(*zip(*leader), color="black", lw=1, solid_joinstyle="miter")


def leaders_touch(a, b, pad_y=0.05):
    """Whether two elbow leaders cross, or run along each other within `LABEL_PAD` in
    x or `pad_y` in y. Each leader is a list of points with axis-aligned segments."""
    pad_x = LABEL_PAD / 2
    for (ax0, ay0), (ax1, ay1) in itertools.pairwise(a):
        for (bx0, by0), (bx1, by1) in itertools.pairwise(b):
            # Each segment as its box, grown by the pads.
            a_box = (min(ax0, ax1), max(ax0, ax1), min(ay0, ay1), max(ay0, ay1))
            b_box = (min(bx0, bx1), max(bx0, bx1), min(by0, by1), max(by0, by1))
            if a_box[0] == a_box[1] and a_box[2] == a_box[3]:
                continue
            if b_box[0] == b_box[1] and b_box[2] == b_box[3]:
                continue
            if (
                a_box[0] - pad_x < b_box[1]
                and b_box[0] - pad_x < a_box[1]
                and a_box[2] - pad_y < b_box[3]
                and b_box[2] - pad_y < a_box[3]
            ):
                return True
    return False


def stacked(name: str, share: float) -> str:
    """The label of a segment: each word of its name on its own line, then the share."""
    return "\n".join([*name.split(), f"{share:.1f}%"])


def finish_axes():
    """Styles the x axis and crops y to the bars and labels. The figure shrinks by the
    same share as the axes, so that a data unit keeps its height and the labels keep
    their places."""
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Share of total compute time")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    extents = [
        a.get_window_extent().transformed(to_data) for a in ax.texts + ax.patches
    ]
    low = min(min(box.y0, box.y1) for box in extents) - ROW_PAD
    high = max(max(box.y0, box.y1) for box in extents) + ROW_PAD
    axes_inches = ax.get_window_extent().height / fig.dpi
    old_low, old_high = ax.get_ylim()
    ax.set_ylim(low, high)
    fig.set_figheight(
        fig.get_figheight() - axes_inches * (1 - (high - low) / (old_high - old_low))
    )


fig, ax = plt.subplots(figsize=(16, 8), layout="constrained")
ax.set(xlim=(0, 100), ylim=(-5.6, 5.2), yticks=[])
bar_kws = dict(height=1, edgecolor="white", linewidth=2)

# The groups, on y from 0 to 1.
group_shares = by_group["share"].to_numpy() * 100
group_colors = dict(zip(group_order, GROUP_COLORS))
lefts, centers = spans(group_shares)
ax.barh(
    0.5,
    group_shares,
    left=lefts,
    color=[group_colors[g] for g in group_order],
    **bar_kws,
)
label_segments(
    [stacked(g, s) for g, s in zip(group_order, group_shares)],
    lefts,
    group_shares,
    tip_y=1,
    direction=1,
)

# The parts, on y from -1.25 to -0.25. Within a group, the largest part is a darker
# shade of the group color and the next ones are lighter tints, so that no part has
# the group color itself. The "Other" parts are gray.
PART_SHADES = [-0.3, 0.35, 0.6]
part_shares = parts_plotted["share"].to_numpy() * 100
part_rank = parts_plotted.select(pl.int_range(pl.len()).over("group")).to_series()
part_colors = [
    OTHER_COLOR if other else tint(group_colors[g], PART_SHADES[r])
    for g, r, other in zip(parts_plotted["group"], part_rank, parts_plotted["is_other"])
]
lefts, centers = spans(part_shares)
ax.barh(-0.75, part_shares, left=lefts, color=part_colors, **bar_kws)
labeled = (~parts_plotted["is_other"] & (parts_plotted["n_parts"] > 1)).to_numpy()
label_segments(
    [
        stacked(n, s)
        for n, s in zip(parts_plotted.filter(labeled)["name"], part_shares[labeled])
    ],
    lefts[labeled],
    part_shares[labeled],
    tip_y=-1.25,
    direction=-1,
)

finish_axes()
save_matplotlib_figure(fig, "time_share_by_group", OUT_PATH)

# %% FIGURE, ONE LEVEL
# The parts of the second bar as one bar, in the order of `SINGLE_ORDER`, with no
# hierarchy: each part has its own categorical color. Model predictions and Synapse
# mapping are parts here, and every other part is one gray "Other" at the end.
SINGLE_ORDER = [
    "Mesh loading",
    "Synapse loading",
    "Synapse mapping",
    "Mesh conditioning",
    "Spectral features",
    "Feature agglomeration",
    "Model predictions",
    "Morphometry",
    "Supervoxel anchoring",
]
# Seaborn's Set3, in its order. Its ninth color is a gray, which "Other" takes, and the
# parts take the others.
SET3 = sns.color_palette("Set3")
SINGLE_OTHER_COLOR = SET3[8]
SINGLE_COLORS = [c for c in SET3 if c != SINGLE_OTHER_COLOR][: len(SINGLE_ORDER)]
single = (
    parts_plotted.with_columns(
        name=pl.when(pl.col("name").is_in(SINGLE_ORDER))
        .then(pl.col("name"))
        .otherwise(pl.lit("Other"))
    )
    .group_by("name")
    .agg(pl.col("share").sum())
    .sort(pl.col("name").cast(pl.Enum([*SINGLE_ORDER, "Other"])))
)
assert single["name"].to_list() == [*SINGLE_ORDER, "Other"]
single_shares = single["share"].to_numpy() * 100
fig, ax = plt.subplots(figsize=(16, 8), layout="constrained")
ax.set(xlim=(0, 100), ylim=(-5.6, 5.2), yticks=[])
lefts, _ = spans(single_shares)
ax.barh(
    0.5,
    single_shares,
    left=lefts,
    color=[*SINGLE_COLORS, SINGLE_OTHER_COLOR],
    **bar_kws,
)
label_segments(
    [stacked(n, s) for n, s in zip(single["name"], single_shares)],
    lefts,
    single_shares,
    tip_y=1,
    direction=1,
)
finish_axes()
save_matplotlib_figure(fig, "time_share_by_part", OUT_PATH)
