"""Matplotlib helpers for the census figures in `scripts/census/`. Only plot code
lives here: the scripts make the tables that these functions draw."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .colors import CELL_TYPE_PALETTE, COMPARTMENT_PALETTE_MUTED_HEX
from .io import CELL_TYPE_CATEGORIES

# The cell types on the y axis of the stripplots. Inputs leave out TH, because the
# thalamic cells have no dendrites in the volume.
OUTPUT_CELL_TYPE_CATEGORIES = [c for c in CELL_TYPE_CATEGORIES if c != "Unk"]
INPUT_CELL_TYPE_CATEGORIES = [c for c in OUTPUT_CELL_TYPE_CATEGORIES if c != "TH"]

TAG_CATEGORIES = ["spine", "shaft", "soma"]

# The line between the excitatory and the inhibitory rows of the stripplots.
EI_BORDER = 7.5
JITTER = 0.35

_SCATTER_KWS = dict(
    linewidth=0.2,
    alpha=0.8,
    sizes=(10, 70),
    palette=CELL_TYPE_PALETTE,
    hue="broad_type",
)


def remapper(cat: str) -> str:
    """The paper names of the IT cell types."""
    match cat:
        case "23P":
            return "2/3P-IT"
        case "4P":
            return "4P-IT"
        case _:
            return cat


def add_cell_type_y(cells: pd.DataFrame, depth: str = "pt_position_y") -> pd.DataFrame:
    """Add `cell_type_y`: the row of the cell type, jittered by the soma depth rank
    within the type, so that a row reads from superficial to deep."""
    cells = cells.copy()
    groups = cells.groupby("cell_type", observed=True)[depth]
    rank = groups.rank() / groups.transform("count")
    cells["cell_type_y"] = cells["cell_type"].cat.codes.astype(float)
    cells["cell_type_y"] += JITTER * (2 * rank - 1)
    return cells


def draw_bracket(
    ax,
    start,
    end,
    shift=0,
    axis="x",
    color="black",
    label=None,
    text_kws=None,
    normalize_limits=False,
    axis_off=True,
):
    """A curly bracket from `start` to `end` along `axis`, with a label."""
    if text_kws is None:
        text_kws = {}
    lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 500)
    tan = np.tan(lx)
    curve = (np.hstack((tan[::-1], tan)) / 20 + 1) / 2
    x = np.linspace(start, end, 1000)
    labelpad = 0.02
    if axis == "x":
        ax.plot(x, curve + shift, color=color, clip_on=False)
        if label is not None:
            ax.text(
                (start + end) / 2,
                shift - labelpad,
                label,
                ha="center",
                va="bottom",
                color=color,
                **text_kws,
            )
        if normalize_limits:
            ax.set(xlim=(0, 1), ylim=(1, 0))
    elif axis == "y":
        ax.plot(curve + shift, x, color=color, clip_on=False)
        if label is not None:
            ax.text(
                shift - labelpad,
                (start + end) / 2,
                label,
                ha="center",
                va="bottom",
                color=color,
                rotation=90,
                rotation_mode="anchor",
                **text_kws,
            )
        if normalize_limits:
            ax.set(xlim=(0, 1), ylim=(0, 1))
    if axis_off:
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax.set(xticks=[], yticks=[])


def _marker_size(value, size_norm, sizes):
    min_size, max_size = sizes
    min_norm, max_norm = size_norm
    scale = np.clip((value - min_norm) / (max_norm - min_norm), 0, 1)
    return min_size + scale * (max_size - min_size)


def _size_legend(ax, title, loc, bbox_to_anchor, legend_sizes, size_norm):
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=np.sqrt(_marker_size(s, size_norm, _SCATTER_KWS["sizes"])),
            label=f"{s:,}" if s != size_norm[-1] else f"{s:,}+",
        )
        for s in legend_sizes
    ]
    ax.legend(
        handles=handles, title=title, loc=loc, frameon=True, bbox_to_anchor=bbox_to_anchor
    )


def weighted_means(data: pd.DataFrame, by: str, x: str, weight: str) -> pd.Series:
    """The mean of `x` per group of `by`, weighted by `weight`."""
    values = data[x].fillna(0)
    weights = data[weight].fillna(0)
    groups = data[by]
    sums = (values * weights).groupby(groups, observed=True).sum()
    return (sums / weights.groupby(groups, observed=True).sum()).rename(x)


def _format_cell_type_axis(ax, categories, start):
    ax.set_yticks(np.arange(len(categories)) + start)
    ax.set_yticklabels([remapper(c) for c in categories])
    ax.invert_yaxis()
    ax.set(ylabel="")
    ax.spines["left"].set_bounds(-0.5 + start, len(categories) + start - 0.5)


def _label_ei_border(axs, start, side):
    ax = axs[0]
    ax.set(xticks=[], yticks=[], xlabel="")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    draw_bracket(ax, -0.5 + start, EI_BORDER, axis="y", label="Excitatory")
    draw_bracket(ax, EI_BORDER, 11.5, axis="y", label="Inhibitory")
    ax.set(ylim=axs[1].get_ylim())
    label = "Output cell" if side == "pre" else "Input cell"
    ax.set_ylabel(label, labelpad=35, fontsize="large")


def sorted_stripplot_with_means(
    data: pd.DataFrame,
    x: str,
    weight_x: str,
    size: str,
    size_title: str,
    y: str = "cell_type_y",
    size_norm=(200, 1000),
    loc="upper left",
    bbox_to_anchor=(0.02, 0.985),
    xlabel="",
    title="",
    legend_sizes=(200, 500, 1000),
    xlim=(0, 1),
    **kws,
):
    """One dot per cell at `x`, one row per cell type, and a bar at the weighted
    mean of each type. `data` needs `cell_type`, `broad_type` and `cell_type_y`
    (`add_cell_type_y`). Returns the figure, the axes and the weighted means."""
    if x.startswith("pre_"):
        categories, start, side = OUTPUT_CELL_TYPE_CATEGORIES, 0, "pre"
    else:
        categories, start, side = INPUT_CELL_TYPE_CATEGORIES, 1, "post"
        data = data.query("cell_type != 'TH'")

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(8.29, 7.17),
        dpi=300,
        layout="constrained",
        gridspec_kw=dict(width_ratios=[0.8, 10]),
    )
    ax = axs[1]
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        size=size,
        size_norm=size_norm,
        **{**_SCATTER_KWS, **kws},
    )
    _size_legend(ax, size_title, loc, bbox_to_anchor, legend_sizes, size_norm)

    means = weighted_means(data, "cell_type", x, weight_x)
    mean_y = data.groupby("cell_type", observed=True)[y].mean()
    height = JITTER * 3.0
    width = 0.005 * (xlim[1] - xlim[0])
    for cell_type, mean_x in means.items():
        ax.add_patch(
            plt.Rectangle(
                (mean_x - width / 2, mean_y[cell_type] - height / 2),
                width,
                height,
                facecolor="black",
                edgecolor="black",
                zorder=1,
            )
        )
    ax.axhline(EI_BORDER, color="black", linewidth=2, linestyle="--", zorder=3)

    _format_cell_type_axis(ax, categories, start)
    if "_p_" in x:
        if xlim == (0, 1):
            ax.set(xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(left=xlim[0])
        ax.spines["bottom"].set_bounds(*xlim)
    _label_ei_border(axs, start, side)
    ax.set(xlabel=xlabel, title=title)
    return fig, axs, means


def filter_and_bin(data, y, use_quantiles=True, trim_quantiles=(0.01, 0.99), n_bins=25):
    """Drop the rows outside the `trim_quantiles` of `y`, and add `{y}_bin` and
    `{y}_bin_mid`. Returns the frame and the bin edges."""
    if trim_quantiles is None:
        trim_quantiles = (0.0, 1.0)
    low, high = data[y].quantile(trim_quantiles)
    data = data[(data[y] >= low) & (data[y] <= high)].copy()
    if use_quantiles:
        data[f"{y}_bin"], bins = pd.qcut(data[y], q=n_bins, retbins=True)
    else:
        bins = np.linspace(low, high, n_bins)
        data[f"{y}_bin"] = pd.cut(data[y], bins=bins)
    data[f"{y}_bin_mid"] = data[f"{y}_bin"].apply(lambda b: b.mid).astype(float)
    return data, bins


def multi_density_histplot(data, y, bins, ax):
    """The density of single- and multi-input spines along `y`."""
    sns.histplot(
        data=data,
        y=y,
        hue="tag_detailed",
        ax=ax,
        stat="density",
        bins=bins,
        fill=True,
        hue_order=["single_spine", "multi_spine"],
        element="poly",
        kde=False,
        common_norm=False,
        palette=COMPARTMENT_PALETTE_MUTED_HEX,
        legend=False,
    )
    ax.spines["bottom"].set_visible(False)
    ax.set(xticks=[], xlabel="Normalized\ndensity\n(spines)")


def multi_proportion_lineplot(data, y, ax):
    """The proportion of multi-input spines per bin of `y` (`{y}_bin_mid`)."""
    sns.lineplot(
        data=data, y=y + "_bin_mid", x="is_multi", ax=ax, orient="y", color="black"
    )
    ax.set_xlabel("Proportion of \nspines with\nmultiple inputs")
    ax.set_xlim(0, ax.get_xlim()[1])
