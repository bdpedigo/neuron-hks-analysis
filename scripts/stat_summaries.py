# %% IMPORTS
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from caveclient import CAVEclient
from matplotlib import cm, rcdefaults, ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter
from nglui.parser import StateParser
from nglui.statebuilder import ViewerState
from pywaffle import Waffle
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import (
    lognorm,
    norm,
    pearsonr,
    powernorm,
    probplot,
    spearmanr,
    weibull_min,
)
from sklearn.covariance import EllipticEnvelope
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import classification_report, pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from analysis import (
    CELL_TYPE_CATEGORIES,
    CELL_TYPE_PALETTE,
    COMPARTMENT_PALETTE_MUTED_HEX,
    DATA_PATH,
    FIG_PATH,
    load_neuron_info,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)


def load_mtype_palette():
    palette_file = DATA_PATH / "ctype_hues.pkl"

    with open(palette_file, "rb") as f:
        ctype_hues = pickle.load(f)

    ctype_hues = {ctype: tuple(ctype_hues[ctype]) for ctype in ctype_hues.keys()}
    return ctype_hues


def remapper(cat):
    match cat:
        case "23P":
            return "2/3P-IT"
        case "4P":
            return "4P-IT"
        case _:
            return cat


rcdefaults()

sns.set_context("talk")

set_matplotlib_theme()

VERSION = 1412

figure_out_path = FIG_PATH / "stat_summaries"

cell_info = load_neuron_info(version=VERSION)
cell_info = (
    cell_info.reset_index().drop_duplicates("pt_root_id").set_index("pt_root_id")
)

# %% LOAD SYNAPSE TABLES

synapses_pl = pl.read_parquet(DATA_PATH / "filtered_synapses.parquet")
synapses = synapses_pl.to_pandas().set_index("synapse_id")

all_pre_synapses = synapses.query("pre_in_selection").copy()
all_post_synapses = synapses.query("post_in_selection").copy()

pre_target_info = (
    pl.read_parquet(DATA_PATH / "pre_synapse_aggregation.parquet")
    .to_pandas()
    .set_index("pre_pt_root_id")
)
pre_exc_target_info = (
    pl.read_parquet(DATA_PATH / "pre_synapse_to_exc_aggregation.parquet")
    .to_pandas()
    .set_index("pre_pt_root_id")
)
pre_inh_target_info = (
    pl.read_parquet(DATA_PATH / "pre_synapse_to_inh_aggregation.parquet")
    .to_pandas()
    .set_index("pre_pt_root_id")
)
post_target_info = (
    pl.read_parquet(DATA_PATH / "post_synapse_aggregation.parquet")
    .to_pandas()
    .set_index("post_pt_root_id")
)


cell_info = (
    cell_info.drop(columns=pre_target_info.columns, errors="ignore")
    .join(pre_target_info, how="left")
    .join(pre_exc_target_info, how="left")
    .join(pre_inh_target_info, how="left")
    .drop(columns=post_target_info.columns, errors="ignore")
    .join(post_target_info, how="left")
)


proofread_info = cell_info.query(
    "(cell_type_source == 'allen_v1_column_types_slanted_ref' or cell_type == 'TH') and axon_cleaned"
)
root_ids = proofread_info.index.unique()
thalamic_roots = proofread_info.query("broad_type == 'thalamic'").index.unique()
column_info = proofread_info.query(
    "cell_type_source == 'allen_v1_column_types_slanted_ref'"
).copy()

# %%

top_line_numbers = post_target_info[
    [
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
].sum()
top_line_numbers = top_line_numbers / 1e6
top_line_numbers.index = [name.replace("post_", "") for name in top_line_numbers.index]

save_variables(
    prefix="summary_n_", **top_line_numbers.to_dict(), format="{:.1f} million"
)
save_variables(
    summary_n_neurons_ran=len(post_target_info.query("post_total_known_synapses > 0")),
    format="{:,d}",
)

# %%

counts_table = proofread_info.groupby("cell_type").agg(
    {
        "cell_type": "count",
        "post_total_known_synapses": "sum",
        "pre_total_known_synapses": "sum",
    }
)
counts_table.rename(columns={"cell_type": "n_cells"}, inplace=True)

for col in counts_table.columns:
    save_variables(
        prefix=f"summary_{col}_by_cell_type_",
        **counts_table[col].astype(int).to_dict(),
        format="{:,d}",
    )

# %%
query_post_target_info = post_target_info.copy()
query_post_target_info["broad_type"] = query_post_target_info.index.map(
    cell_info["broad_type"]
)
excitatory_post_target_info = query_post_target_info.query("broad_type == 'excitatory'")
inhibitory_post_target_info = query_post_target_info.query("broad_type == 'inhibitory'")
n_excitatory_neurons = len(
    excitatory_post_target_info.query("post_total_known_synapses > 0")
)
n_inhibitory_neurons = len(
    inhibitory_post_target_info.query("post_total_known_synapses > 0")
)

save_variables(
    summary_n_excitatory_neurons=n_excitatory_neurons,
    summary_n_inhibitory_neurons=n_inhibitory_neurons,
    format="{:,d}",
)

for broad_type, broad_type_table in zip(
    ["excitatory", "inhibitory"],
    [excitatory_post_target_info, inhibitory_post_target_info],
):
    line_numbers = broad_type_table[
        [
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
    ].sum()
    line_numbers = line_numbers / 1e6
    line_numbers.index = [name.replace("post_", "") for name in line_numbers.index]
    save_variables(
        prefix=f"summary_n_{broad_type}_",
        **line_numbers.to_dict(),
        format="{:.1f} million",
    )

# %% GET COUNTS OF BROAD AND CELL TYPES
broad_type_counts = proofread_info["broad_type"].value_counts()
save_variables(prefix="column_summary_n_", **broad_type_counts.to_dict())

cell_type_counts = proofread_info["cell_type"].value_counts()
save_variables(prefix="column_summary_n_", **cell_type_counts.to_dict())


# %% CALCULATE MEAN OUTPUT SYNAPSE DEPTHS FOR THALAMIC CELLS
thalamic_pre_synapses = all_pre_synapses.query("pre_pt_root_id in @thalamic_roots")
mean_output_synapse_depths = (
    thalamic_pre_synapses.groupby("pre_pt_root_id")["ctr_pt_position_y"].mean() * 4
)
proofread_info.loc[mean_output_synapse_depths.index, "pt_position_y"] = (
    mean_output_synapse_depths
)


# %% IN COLUMN FLAG
cell_info["in_column"] = cell_info.index.isin(column_info.index)


# %% BUILD LDA CLASSIFIER ON SPINE FEATURES


features = [
    "post_p_spine_synapse",
    "post_p_shaft_synapse",
    "post_p_soma_synapse",
    "post_mean_spine_inputs",
    "post_p_spine_site_is_multi",
    "post_p_spine_synapse_is_multi",
    "nuc_volume",
]

column_cell_info = cell_info.query("in_column")
X = cell_info[features]
mask = ~X.isnull().any(axis=1)

X_train = column_cell_info[features]
y_train = column_cell_info["broad_type"]

transformer = QuantileTransformer(output_distribution="normal")
transformer.fit(X[mask])
X_train_transformed = transformer.transform(X_train)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_transformed, y_train)
y_pred = lda.predict(X_train_transformed)

print(classification_report(y_train, y_pred))

X_transformed = transformer.transform(X[mask])

out = lda.predict(X_transformed)
out = pd.Series(out.reshape(-1), index=X.index[mask])
cell_info["broad_type_lda_prediction"] = out

out = lda.transform(X_transformed)
out = pd.Series(out.reshape(-1), index=X.index[mask])
cell_info["broad_type_lda_transform"] = out


posterior_excitatory = lda.predict_proba(X_transformed)[
    :, np.where(lda.classes_ == "excitatory")[0][0]
]
posterior_excitatory = pd.Series(posterior_excitatory.reshape(-1), index=X.index[mask])
cell_info["p_excitatory_lda"] = posterior_excitatory
cell_info["uncertainty"] = np.abs(cell_info["p_excitatory_lda"] - 0.5) * 2


# %% COMPARE LDA SCORES FOR COLUMN VS WHOLE DATASET
set_matplotlib_theme()

fig, ax = plt.subplots(1, 1, figsize=(8, 3), sharex=True)

sns.histplot(
    data=cell_info.query("in_column"),
    x="broad_type_lda_transform",
    hue="broad_type",
    binwidth=0.25,
    kde=False,
    stat="density",
    palette=CELL_TYPE_PALETTE,
    hue_order=["excitatory", "inhibitory"],
    legend=False,
    linestyle="-",
    fill=False,
    element="poly",
    ax=ax,
)

sns.histplot(
    data=cell_info.query("post_total_synapses > 1000"),
    x="broad_type_lda_transform",
    hue="broad_type_lda_prediction",
    binwidth=0.25,
    kde=False,
    stat="density",
    palette=CELL_TYPE_PALETTE,
    hue_order=["excitatory", "inhibitory"],
    legend=False,
    linestyle=":",
    fill=False,
    element="poly",
    ax=ax,
)
ax.set(xlim=(-5, 5))

# %% PLOT OF 4P CELLS LDA SPLIT
fig, ax = plt.subplots(figsize=(6, 6))

cell_type = "4P"
hue = "broad_type_lda_prediction"
x = "post_p_spine_synapse"
y = "post_p_spine_site_is_multi"
# y = "post_mean_spine_inputs"
size = "post_total_synapses"
sns.scatterplot(
    data=cell_info.query("post_total_synapses > 500 and cell_type == @cell_type"),
    x=x,
    y=y,
    size=size,
    hue=hue,
    sizes=(2, 30),
    alpha=0.5,
    linewidth=0,
    ax=ax,
)

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# %% PLOT OF EXCITATORY CELLS ONLY, p_multi vs total synapses cloud
data = cell_info.query(
    "cell_type == @cell_type and broad_type_lda_prediction == 'excitatory'"
)
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=data,
    x="post_total_synapses",
    y="post_p_spine_site_is_multi",
    # size=size,
    # hue=hue,
    # sizes=(2, 30),
    s=5,
    alpha=0.5,
    linewidth=0,
    legend=False,
)
ax.set(yscale="log", xscale="log")


# %% NUMBER OF CELLS WHERE LDA PREDICTION DISAGREES WITH CURATED LABEL
len(cell_info.query("broad_type != broad_type_lda_prediction")) / len(cell_info)

# %% EXAMINE CELLS WHERE LDA PREDICTION DISAGREES WITH CURATED LABEL
cell_info.query(
    "broad_type == 'excitatory' and broad_type_lda_prediction == 'inhibitory'"
).sample(100).query("post_total_synapses > 500").index[:20]

# %% SOMA DEPTH VS P SPINE IS MULTI AND P SPINE SYNAPSE

name_map = {
    "post_p_spine_site_is_multi": "Proportion of spines with multiple inputs",
    "post_p_spine_synapse": "Proportion of input synapses on spines",
}
legend = False
xs = ["post_p_spine_site_is_multi", "post_p_spine_synapse"]
# x = 'post_spine_shaft_ratio'
for x in xs:
    fig, ax = plt.subplots(1, 1, figsize=(8.29, 7.17), dpi=300)
    sns.scatterplot(
        data=cell_info.query("post_total_synapses > 500"),
        x=x,
        y="pt_position_um_y",
        color="grey",
        s=2,
        linewidth=0,
        ax=ax,
        alpha=0.6,
        label="unlabeled",
    )

    sns.scatterplot(
        data=cell_info.query("post_total_synapses > 500").query("in_column"),
        hue="broad_type",
        palette=CELL_TYPE_PALETTE,
        hue_order=["excitatory", "inhibitory"],
        x=x,
        y="pt_position_um_y",
        color="grey",
        zorder=2,
        linewidth=0.5,
        alpha=0.9,
        s=30,
        ax=ax,
    )
    ax.invert_yaxis()
    ax.set(xlabel=name_map[x], ylabel="Soma depth (um)")
    ax.set(xlim=(0, 0.9))

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = []
        for label in labels:
            new_labels.append(label.capitalize())

        ax.legend(handles, new_labels)
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(0.9, 1), title="Manual labels"
        )
        legend = ax.get_legend()
        legend.legend_handles[0].set_sizes([30])
    else:
        ax.get_legend().remove()

    y = 0.9
    text = ax.text(
        0.1,
        y,
        "Curated\ninhibitory",
        color=CELL_TYPE_PALETTE["inhibitory"],
    )
    text = ax.text(0.38, y, "Uncurated", color="grey")
    text = ax.text(
        0.65,
        y,
        "Curated\nexcitatory",
        color=CELL_TYPE_PALETTE["excitatory"],
    )

    sns.despine(fig, ax, top=True, bottom=False, right=True, left=False, trim=True)
    save_matplotlib_figure(fig, f"whole_dataset_soma_depth_vs_{x}", figure_out_path)


# %% PLOT OF WHERE COLUMN CELLS ARE
cell_info["curated_broad_type"] = cell_info["broad_type"].astype(str)
CELL_TYPE_PALETTE["unlabeled"] = "#5d5d5d"
# where not in column, replace with unknown
cell_info.loc[cell_info.query("~in_column").index, "curated_broad_type"] = "unlabeled"


def clean_up_ax(ax):
    ax.invert_yaxis()
    ax.set(ylabel="Depth (um)", xlabel="X position (um)")


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(
    data=cell_info,
    x="pt_position_um_x",
    y="pt_position_um_y",
    color="grey",
    s=2,
)
sns.scatterplot(
    data=cell_info.query("in_column"),
    x="pt_position_um_x",
    y="pt_position_um_y",
    hue="broad_type",
    palette=CELL_TYPE_PALETTE,
    s=30,
)
clean_up_ax(ax)

# %% DEFINE COLUMN-COLUMN SYNAPSES

column_root_ids = column_info.index.unique()
column_column_synapses = all_post_synapses.query(
    "pre_pt_root_id in @column_root_ids and post_pt_root_id in @column_root_ids"
).copy()


# %% DEFINE SHOW COLUMNS
def show_columns():
    print("\n".join([c for c in proofread_info.columns if c.startswith("post_")]))


# %% COMPUTE GROUP TO GROUP CROSSTAB
# NOTE dropping unknown here
group_category_counts = (
    column_column_synapses.query('tag != "unknown"')
    .groupby(["pre_cell_type", "post_cell_type", "tag"], observed=True)
    .size()
    .rename("synapse_count")
)
group_category_proportions = (
    group_category_counts.groupby(level=[0, 1])
    .transform(lambda x: x / x.sum())
    .rename("synapse_proportion")
    .to_frame()
)
group_group_total_counts = group_category_counts.unstack().sum(axis=1)

group_category_proportions_square = group_category_proportions[
    "synapse_proportion"
].unstack()
group_category_proportions_square["total_count"] = group_category_counts.unstack().sum(
    axis=1
)
group_category_proportions_square


group_category_proportions["total_count"] = (
    group_category_proportions.reset_index()
    .set_index(["pre_cell_type", "post_cell_type"])
    .index.map(group_group_total_counts)
)

# %% BARPLOTS SHOWING TYPE TO TYPE PROPORTIONS

# plot_categories = CELL_TYPE_CATEGORIES[1:-1]  # dropping thalamic and unk here for now
drop_categories = ["TH", "Unk"]
plot_categories = [c for c in CELL_TYPE_CATEGORIES if c not in drop_categories]
n_groups = len(plot_categories)
scale = 2
fig, axs = plt.subplots(
    n_groups,
    n_groups,
    figsize=(n_groups * scale, n_groups * scale),
    dpi=300,
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.0, "hspace": 0.0},
)
plot_props = group_category_proportions.reset_index().set_index(
    ["pre_cell_type", "post_cell_type"]
)
for i, pre_type in enumerate(plot_categories):
    for j, post_type in enumerate(plot_categories):
        ax = axs[i, j]
        data = (
            plot_props.loc[(pre_type, post_type)]
            .reset_index()
            .query('tag != "unknown"')
            .copy()
        )
        data["tag"] = data["tag"].astype(str)
        if data["total_count"].values[0] < 50:
            ax.axis("off")
            continue
        # if len(data) == 0:
        #     ax.axis("off")
        #     continue
        sns.barplot(
            data=data,
            x="tag",
            y="synapse_proportion",
            ax=ax,
            hue="tag",
            palette=COMPARTMENT_PALETTE_MUTED_HEX,
            order=["spine", "shaft", "soma"],
            hue_order=["spine", "shaft", "soma"],
            # width=1,
        )
        ax.set(ylabel="")
for i, pre_type in enumerate(plot_categories):
    ax = axs[i, 0]
    ax.set_ylabel(pre_type, fontsize="large", rotation=0, labelpad=20, ha="right")
for j, post_type in enumerate(plot_categories):
    ax = axs[0, j]
    ax.set_title(post_type, fontsize="large")
ax.set(xticks=[], ylabel="", xlabel="", ylim=(0, 1))

save_matplotlib_figure(fig, "cell_type_pair_barplots", figure_out_path)


# %% CELL TYPE WAFFLES

set_matplotlib_theme()


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
                0 + shift - labelpad,
                label,
                ha="center",
                va="bottom",
                color=color,
                # transform=ax.get_xaxis_transform(),
                **text_kws,
            )
        if normalize_limits:
            ax.set(xlim=(0, 1), ylim=(1, 0))
    elif axis == "y":
        ax.plot(curve + shift, x, color=color, clip_on=False)
        if label is not None:
            ax.text(
                0 + shift - labelpad,
                (start + end) / 2,
                label,
                ha="center",
                va="bottom",
                color=color,
                rotation=90,
                rotation_mode="anchor",
                # transform=ax.get_yaxis_transform(),
                **text_kws,
            )
        if normalize_limits:
            ax.set(xlim=(0, 1), ylim=(0, 1))
    if axis_off:
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax.set(xticks=[], yticks=[])


scale = 1

e_groups = []
figsize = (7.17, 7.17)
(n_groups * scale, n_groups * scale)
fig = plt.figure(figsize=figsize, dpi=300)


n_exc = 7
n_inh = 4

label_fontsize = "small"


# outer gridspec to space out excitatory vs inhibitory a bit
outer_gs = GridSpec(
    3,
    3,
    width_ratios=[0.5, n_exc, n_inh],
    height_ratios=[0.5, n_exc, n_inh],
    wspace=0.07,
    hspace=0.07,
    figure=fig,
)

x_shift = -2.7
ax = fig.add_subplot(outer_gs[1, 0])
draw_bracket(
    ax,
    0,
    1,
    axis="y",
    label="Excitatory",
    shift=x_shift,
    axis_off=True,
    normalize_limits=True,
    text_kws={"fontsize": label_fontsize},
)
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
draw_bracket(
    ax,
    0,
    1,
    axis="y",
    label="Inhibitory",
    shift=x_shift,
    normalize_limits=True,
    axis_off=True,
    text_kws={"fontsize": label_fontsize},
)

y_shift = -2.2
ax = fig.add_subplot(outer_gs[0, 1])
draw_bracket(
    ax,
    0,
    1,
    axis="x",
    label="Excitatory",
    shift=y_shift,
    axis_off=True,
    normalize_limits=True,
    text_kws={"fontsize": label_fontsize},
)
ax.text(
    0.8,
    y_shift - 1.7,
    "Postsynaptic cell type",
    va="center",
    ha="center",
)

ax = fig.add_subplot(outer_gs[0, 2])
draw_bracket(
    ax,
    0,
    1,
    axis="x",
    label="Inhibitory",
    shift=y_shift,
    axis_off=True,
    normalize_limits=True,
    text_kws={"fontsize": label_fontsize},
)

space = 0.06
exc_exc_gs = GridSpecFromSubplotSpec(
    n_exc, n_exc, subplot_spec=outer_gs[1, 1], wspace=space, hspace=space
)
exc_inh_gs = GridSpecFromSubplotSpec(
    n_exc, n_inh, subplot_spec=outer_gs[1, 2], wspace=space, hspace=space
)
inh_exc_gs = GridSpecFromSubplotSpec(
    n_inh, n_exc, subplot_spec=outer_gs[2, 1], wspace=space, hspace=space
)
inh_inh_gs = GridSpecFromSubplotSpec(
    n_inh, n_inh, subplot_spec=outer_gs[2, 2], wspace=space, hspace=space
)

axs = np.empty((n_exc + n_inh, n_exc + n_inh), dtype=object)
for i in range(n_exc):
    for j in range(n_exc):
        axs[i, j] = fig.add_subplot(exc_exc_gs[i, j])
for i in range(n_exc):
    for j in range(n_inh):
        axs[i, j + n_exc] = fig.add_subplot(exc_inh_gs[i, j])
for i in range(n_inh):
    for j in range(n_exc):
        axs[i + n_exc, j] = fig.add_subplot(inh_exc_gs[i, j])
for i in range(n_inh):
    for j in range(n_inh):
        axs[i + n_exc, j + n_exc] = fig.add_subplot(inh_inh_gs[i, j])

for ax in axs.flat:
    ax.spines[["left", "top", "right", "bottom"]].set_linewidth(1.0)

COMPARTMENT_PALETTE_MUTED_HEX["unknown"] = "#7F7F7F"
tag_categories = ["spine", "shaft", "soma"]
tag_colors = [COMPARTMENT_PALETTE_MUTED_HEX[tag] for tag in tag_categories]

interval = 0.7
n_waffle_grid = 10
waffle_kws = dict(
    rows=n_waffle_grid,
    columns=n_waffle_grid,
    interval_ratio_x=interval,
    interval_ratio_y=interval,
    colors=tag_colors,
    rounding_rule="ceil",
    tight=False,
)
plot_type = "square_bar"


count_norm = mpl.colors.LogNorm(vmin=10, vmax=30000)
prop_norm = mpl.colors.Normalize(vmin=0, vmax=1)
count_cmap = mpl.colormaps["Blues"]
prop_cmap = mpl.colormaps["RdPu"]

plot_props = group_category_proportions.reset_index().set_index(
    ["pre_cell_type", "post_cell_type"]
)
for i, pre_type in enumerate(plot_categories):
    for j, post_type in enumerate(plot_categories):
        ax = axs[i, j]
        data = plot_props.loc[(pre_type, post_type)].reset_index()
        if data["total_count"].values[0] < 10:
            # ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.set_facecolor("#d0d0d0")
            continue
        values = (
            data.set_index("tag")
            .reindex(tag_categories)["synapse_proportion"]
            .fillna(0)
            .values
        )
        if plot_type == "waffle":
            Waffle.make_waffle(
                ax=ax,
                values=values,
                **waffle_kws,
            )
        elif plot_type == "square_bar":
            start = 0
            for k, amount in enumerate(values):
                rect = Rectangle(
                    xy=(start, 0.0), width=amount, height=1.0, color=tag_colors[k]
                )
                ax.add_patch(rect)
                start += amount
        elif plot_type == "squaremap_area_by_spine":
            p_spine = values[0]
            width = height = np.sqrt(p_spine)
            rect = Rectangle(
                xy=(0.5 - width / 2, 0.5 - height / 2),
                width=width,
                height=height,
                color=count_cmap(count_norm(data["total_count"].values[0])),
            )
            ax.add_patch(rect)
        elif plot_type == "squaremap_area_by_count":
            p_spine = values[0]
            total_count = data["total_count"].values[0]
            width = height = np.sqrt(total_count / 10000)
            rect = Rectangle(
                xy=(0.5 - width / 2, 0.5 - height / 2),
                width=width,
                height=height,
                color=prop_cmap(prop_norm(p_spine)),
            )
            ax.add_patch(rect)
        elif plot_type == "single_bar":
            rect = Rectangle(
                xy=(0, 0),
                width=values[0],
                height=1,
                color=tag_colors[0],
            )
            ax.add_patch(rect)

        ax.set(xlim=(0, 1), ylim=(0, 1))

        ax.axis("on")
        ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
        ax.spines[["left", "right", "top", "bottom"]].set_visible(True)

for i, pre_type in enumerate(plot_categories):
    pre_type = remapper(pre_type)
    ax = axs[i, 0]
    ax.set_ylabel(
        pre_type, fontsize=label_fontsize, rotation=0, ha="right", va="center"
    )
    # turn y label visibility back on
    ax.yaxis.label.set_visible(True)
    ax.set(xlim=(0, 1), ylim=(0, 1))

for j, post_type in enumerate(plot_categories):
    post_type = remapper(post_type)
    ax = axs[0, j]
    ax.set_title(
        post_type,
        fontsize=label_fontsize,
        rotation=45,
        ha="left",
        va="bottom",
        rotation_mode="anchor",
        pad=1,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1))

save_matplotlib_figure(fig, "cell_type_pair_waffles", figure_out_path)

# %% BROAD TYPE WAFFLES

broad_category_counts = (
    column_column_synapses.groupby(
        ["pre_broad_type", "post_broad_type", "tag"], observed=True
    )
    .size()
    .rename("synapse_count")
)
broad_category_proportions = (
    broad_category_counts.groupby(level=[0, 1])
    .transform(lambda x: x / x.sum())
    .rename("synapse_proportion")
    .to_frame()
)
broad_total_counts = broad_category_counts.unstack().sum(axis=1)

broad_category_proportions_square = broad_category_proportions[
    "synapse_proportion"
].unstack()
broad_category_proportions_square["total_count"] = broad_category_counts.unstack().sum(
    axis=1
)
broad_category_proportions_square

broad_category_proportions["total_count"] = (
    broad_category_proportions.reset_index()
    .set_index(["pre_broad_type", "post_broad_type"])
    .index.map(broad_total_counts)
)

fig, axs = plt.subplots(
    2, 2, figsize=(8, 8), dpi=300, gridspec_kw={"wspace": 0.1, "hspace": 0.1}
)
waffle_kws["interval_ratio_x"] = 0.15
waffle_kws["interval_ratio_y"] = 0.15
for i, pre_type in enumerate(["excitatory", "inhibitory"]):
    for j, post_type in enumerate(["excitatory", "inhibitory"]):
        ax = axs[i, j]
        data = broad_category_proportions.reset_index().query(
            "pre_broad_type == @pre_type and post_broad_type == @post_type"
        )
        if data["total_count"].values[0] < 50:
            ax.axis("off")
            continue
        Waffle.make_waffle(
            ax=ax,
            values=data.set_index("tag")
            .reindex(tag_categories)["synapse_proportion"]
            .fillna(0)
            .values,
            **waffle_kws,
        )
        ax.axis("on")
        ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(True)

for i, pre_type in enumerate(["excitatory", "inhibitory"]):
    ax = axs[i, 0]
    ax.set_ylabel(
        pre_type.capitalize(), fontsize="x-large", rotation=0, ha="right", va="center"
    )
    # turn y label visibility back on
    ax.yaxis.label.set_visible(True)

for j, post_type in enumerate(["excitatory", "inhibitory"]):
    ax = axs[0, j]
    ax.set_title(post_type.capitalize(), fontsize="x-large")

save_matplotlib_figure(fig, "broad_type_pair_waffles", figure_out_path)

# %% COLOR MAPPING
plot_proofread_info = proofread_info.sort_values(
    ["broad_type", "cell_type", "pt_position_y"]
)

plot_data = plot_proofread_info[
    [
        "post_p_soma_synapse",
        "post_p_shaft_synapse",
        "post_p_spine_synapse",
        "post_p_spine_site_is_multi",
    ]
].fillna(0)

ei_colors = plot_proofread_info["broad_type"].map(
    {"excitatory": "lightpink", "inhibitory": "lightblue"}
)
cell_colors = plot_proofread_info["cell_type"].map(CELL_TYPE_PALETTE)
column_colors = pd.DataFrame(
    {
        "E/I": ei_colors,
        "cell_type": cell_colors,
    },
    index=plot_data.index,
)

palette = COMPARTMENT_PALETTE_MUTED_HEX.copy()
spine_color = palette.pop("spine")


def hex_to_rgb(hex):
    hex = hex.lstrip("#")
    return np.array(tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4)))


spine_rgb = hex_to_rgb(spine_color)
single_spine_rgb = spine_rgb - 20
single_spine_rgb = np.clip(single_spine_rgb, 0, 255)
multi_spine_rgb = spine_rgb + 20
multi_spine_rgb = np.clip(multi_spine_rgb, 0, 255)


def rgb_to_hex(rgb):
    return "#" + "".join(f"{int(x):02x}" for x in rgb)


palette["single_spine"] = rgb_to_hex(single_spine_rgb)
palette["multi_spine"] = rgb_to_hex(multi_spine_rgb)


def color_mapper(value):
    if "multi" in value and "spine" in value:
        return palette["multi_spine"]
    elif "spine" in value:
        return palette["single_spine"]
    elif "soma" in value:
        return palette["soma"]
    elif "shaft" in value:
        return palette["shaft"]
    else:
        return palette["unknown"]


# %% HISTOGRAMS OF E/I COUNTS AND PROPORTIONS OF SPINE INPUTS

plot_proofread_info["broad_type_cap"] = plot_proofread_info[
    "broad_type"
].str.capitalize()

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
hist_size = (8.31, 3.0)
fig, ax = plt.subplots(1, 1, figsize=hist_size, dpi=300)

x = "post_p_spine_synapse"
sns.histplot(
    data=plot_proofread_info.dropna(subset=[x]),
    x=x,
    ax=ax,
    bins=np.linspace(0, 1, 21),
    **histplot_kws,
)

ax.set(
    ylabel="Normalized\ncell density",
    yticks=[],
    xlabel="Proportion of synapses on spines",
)
ax.spines[["top", "right", "left"]].set_visible(False)

save_matplotlib_figure(fig, "spine_synapse_proportion_histograms", figure_out_path)


fig, ax = plt.subplots(1, 1, figsize=hist_size, dpi=300)

x = "post_spine_synapses"
sns.histplot(
    data=plot_proofread_info.dropna(subset=[x]).query(f"{x}>0"),
    x=x,
    ax=ax,
    log_scale=True,
    bins=21,
    **histplot_kws,
)
ax.text(
    0.1,
    0.9,
    "Inhibitory",
    transform=ax.transAxes,
    color=CELL_TYPE_PALETTE["inhibitory"],
)
ax.text(
    0.9,
    0.9,
    "Excitatory",
    transform=ax.transAxes,
    color=CELL_TYPE_PALETTE["excitatory"],
    ha="right",
)
ax.set(
    ylabel="Normalized\ncell density",
    yticks=[],
    xlabel="Number of synapses on spines",
)
ax.spines[["top", "right", "left"]].set_visible(False)
save_matplotlib_figure(fig, "spine_synapse_count_histograms", figure_out_path)

# %% COMBINED EI COUNT AND PROPORTION SPINE HISTOGRAMS
double_hist_size = (5.31, 7.17)
fig, axs = plt.subplots(2, 1, figsize=double_hist_size, dpi=300, layout="constrained")

ax = axs[0]
x = "post_spine_synapses"
data = plot_proofread_info.dropna(subset=[x]).query(f"{x}>0")
sns.histplot(
    data=data,
    x=x,
    ax=ax,
    log_scale=True,
    bins=21,
    **histplot_kws,
)
ax.text(
    0.1,
    0.8,
    "Inhibitory",
    transform=ax.transAxes,
    color=CELL_TYPE_PALETTE["inhibitory"],
)
ax.text(
    0.9,
    0.8,
    "Excitatory",
    transform=ax.transAxes,
    color=CELL_TYPE_PALETTE["excitatory"],
    ha="right",
)
ax.set(
    ylabel="Normalized\ncell density",
    yticks=[],
    # xlabel="Number of synapses on spines",
    xlabel="Number of synapses on spines",
)
ax.spines[["top", "right", "left"]].set_visible(False)

broad_type_n_spines = data.groupby("broad_type", observed=True)[x].mean()

save_variables(
    prefix="column_summary_n_spine_", format="{:.0f}", **broad_type_n_spines.to_dict()
)

ax = axs[1]
x = "post_p_spine_synapse"
data = plot_proofread_info.dropna(subset=[x])
sns.histplot(
    data=data,
    x=x,
    ax=ax,
    bins=np.linspace(0, 1, 21),
    **histplot_kws,
)
ax.set(
    ylabel="Normalized\ncell density",
    yticks=[],
    xlabel="Proportion of synapses on spines",
)
ax.spines[["top", "right", "left"]].set_visible(False)
save_matplotlib_figure(fig, "spine_histograms", figure_out_path)
broad_type_p_spines = data.groupby("broad_type", observed=True)[x].mean()
save_variables(
    prefix="column_summary_p_spine_", format="{:.3g}%", **broad_type_p_spines.to_dict()
)


# %% SUMMARY PLOTS OF CELLS BY CELL TYPE SHOWING P SPINE

# TODO didn't I replace all this in the aggregation polars code?
props = []
for broad_type in ["excitatory", "inhibitory"]:
    column_info_select = (
        all_pre_synapses.query(f"post_broad_type == '{broad_type}'")
        .groupby(["pre_pt_root_id", "tag"])
        .size()
        .unstack()
        .fillna(0)
    )
    column_info_select["total"] = column_info_select[["spine", "shaft", "soma"]].sum(
        axis=1
    )
    column_info_select["total_dendrite"] = column_info_select[["spine", "shaft"]].sum(
        axis=1
    )
    column_props_select = (
        column_info_select.div(column_info_select["total"], axis=0)
        .fillna(0)
        .drop("total", axis=1)
    )
    column_props_select = column_props_select.rename(
        columns=lambda x: f"pre_p_{x}_synapse_to_{broad_type}",
    )
    column_props_select[f"pre_p_dendrite_spine_synapse_to_{broad_type}"] = (
        column_info_select["spine"] / column_info_select["total_dendrite"]
    ).fillna(0)
    # column_props_select["pre_total_synapse_to_" + broad_type] = column_info_select[
    #     "total"
    # ]
    column_props_select = column_props_select.join(
        column_info_select.rename(columns=lambda x: f"pre_{x}_synapse_to_{broad_type}")
    )
    props.append(column_props_select)

column_props = pd.concat(props, axis=1)


jitter = 0.35
mod_proofread_info = proofread_info.join(column_props).copy()
mod_proofread_info["group_depth_rank"] = mod_proofread_info.groupby("cell_type")[
    "pt_position_y"
].rank() / mod_proofread_info.groupby("cell_type")["pt_position_y"].transform("count")

mod_proofread_info["cell_type_y"] = mod_proofread_info["cell_type"].cat.codes.astype(
    float
)
mod_proofread_info["cell_type_y"] += jitter * (
    2 * (mod_proofread_info["group_depth_rank"]) - 1
)


PLOT_CELL_TYPE_CATEGORIES = CELL_TYPE_CATEGORIES.copy()
PLOT_CELL_TYPE_CATEGORIES.remove("Unk")


data = mod_proofread_info
size_norm = (200, 1000)
sizes = (10, 70)
scatter_kws = dict(
    linewidth=0.2,
    alpha=0.8,
    sizes=sizes,
    palette=CELL_TYPE_PALETTE,
    hue="broad_type",
)
mean_scatter_kws = dict(
    hue="cell_type",
    s=1500,
    marker="|",
    palette=CELL_TYPE_PALETTE,
    linewidth=3,
    legend=False,
    edgecolor="black",
    zorder=1,
)
histplot_kws = dict(
    stat="density",
    common_norm=False,
    element="step",
    linewidth=1.5,
    hue="broad_type",
    palette=CELL_TYPE_PALETTE,
    legend=False,
)
ei_border = 7.5

OUTPUT_CELL_TYPE_CATEGORIES = CELL_TYPE_CATEGORIES.copy()
OUTPUT_CELL_TYPE_CATEGORIES.remove("Unk")

INPUT_CELL_TYPE_CATEGORIES = OUTPUT_CELL_TYPE_CATEGORIES.copy()
INPUT_CELL_TYPE_CATEGORIES.remove("TH")


def get_marker_size(value, size_norm, sizes):
    min_size, max_size = sizes
    min_norm, max_norm = size_norm
    if value < min_norm:
        return min_size
    elif value > max_norm:
        return max_size
    else:
        scale = (value - min_norm) / (max_norm - min_norm)
        return min_size + scale * (max_size - min_size)


def fix_legend(
    ax,
    title,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.985),
    legend_sizes=(200, 500, 1000),
    size_norm=(200, 1000),
):
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=np.sqrt(get_marker_size(s, size_norm, sizes)),
            label=f"{s:,}" if s != size_norm[-1] else f"{s:,}+",
        )
        for s in legend_sizes
    ]
    ax.legend(
        handles=handles,
        title=title,
        loc=loc,
        frameon=True,
        bbox_to_anchor=bbox_to_anchor,
    )


def compute_weighted_means(data, by, x, weight_x):
    means = (
        data.groupby(by, observed=True)
        .agg(
            x=(
                x,
                lambda x: np.average(
                    x.fillna(0), weights=data.loc[x.index, weight_x].fillna(0)
                ),
            ),
        )["x"]
        .rename(x)
    )
    return means


def sorted_stripplot_with_means(
    data,
    x,
    y,
    weight_x,
    size,
    size_title,
    size_norm=(200, 1000),
    loc="upper left",
    bbox_to_anchor=(0.02, 0.985),
    xlabel="",
    title="",
    legend_sizes=(200, 500, 1000),
    xlim=(0, 1),
    **kws,
):
    if x.startswith("pre_"):
        categories = OUTPUT_CELL_TYPE_CATEGORIES
        start = 0
        side = "pre"
    elif x.startswith("post_"):
        categories = INPUT_CELL_TYPE_CATEGORIES
        start = 1
        side = "post"
        data = data.query("cell_type != 'TH'")
    else:
        categories = CELL_TYPE_CATEGORIES
        start = 0

    fig, axs = plt.subplots(
        1,
        2,
        # figsize=(8, 8),
        figsize=(8.29, 7.17),
        dpi=300,
        layout="constrained",
        gridspec_kw=dict(width_ratios=[0.8, 10]),
    )
    ax = axs[1]
    all_kws = dict(scatter_kws)
    all_kws.update(kws)
    sns.scatterplot(
        data=data, x=x, y=y, ax=ax, size=size, size_norm=size_norm, **all_kws
    )
    fix_legend(
        ax,
        size_title,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        size_norm=size_norm,
        legend_sizes=legend_sizes,
    )
    # means = data.groupby("cell_type")[[mean_x, y]].mean().reset_index()
    means = data.groupby("cell_type", observed=True).agg(
        x=(
            x,
            lambda x: np.average(
                x.fillna(0), weights=data.loc[x.index, weight_x].fillna(0)
            ),
        ),
        y=(y, "mean"),
    )

    means = means.reset_index()
    rects = []
    for i, row in means.iterrows():
        center = (row["x"], row["y"])
        height = jitter * 3.0
        width = 0.005
        xy = (center[0] - width / 2, center[1] - height / 2)
        rect = plt.Rectangle(
            xy,
            width,
            height,
            facecolor="black",
            zorder=1,
            alpha=1,
            edgecolor="black",
        )
        rects.append(rect)
        ax.add_patch(rect)

    ax.axhline(ei_border, color="black", linewidth=2, linestyle="--", zorder=3)

    format_cell_type_axis(ax, categories=categories, start=start)
    if "_p_" in x:
        format_x_axis(ax, xlim=xlim)
    label_ei_border(axs, start=start, side=side)

    ax.set(xlabel=xlabel, title=title)

    means = means.set_index("cell_type")["x"].rename(x)

    return fig, axs, means


def remapper(cat):
    match cat:
        case "23P":
            return "2/3P-IT"
        case "4P":
            return "4P-IT"
        case _:
            return cat


def remap_categories(categories):
    new_categories = [remapper(cat) for cat in categories]
    return new_categories


def format_cell_type_axis(ax, categories, start=0):
    ax.set_yticks(np.arange(len(categories)) + start)
    ax.set_yticklabels(remap_categories(categories))
    ax.invert_yaxis()
    ax.set(ylabel="")
    ax.spines["left"].set_bounds(-0.5 + start, len(categories) + start - 0.5)


def format_x_axis(ax, xlim):
    if xlim == (0, 1):
        ax.set(xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["bottom"].set_bounds(xlim[0], xlim[1])


def label_ei_border(axs, start=0, side="pre"):
    ax = axs[0]
    ax.set(xticks=[], yticks=[], xlabel="")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    draw_bracket(ax, -0.5 + start, ei_border, axis="y", label="Excitatory")
    draw_bracket(ax, ei_border, 11.5, axis="y", label="Inhibitory")
    ax.set(ylim=axs[1].get_ylim())
    # ax.set_ylabel(
    #     f"{side.capitalize()}synaptic cell type", labelpad=35, fontsize="medium"
    # )
    if side == "pre":
        ax.set_ylabel("Output cell", labelpad=35, fontsize="large")
    else:
        ax.set_ylabel("Input cell", labelpad=35, fontsize="large")


target_type = "excitatory"
target_compartment = "spine"
y = "cell_type_y"
x = f"pre_p_{target_compartment}_synapse_to_{target_type}"
weight_x = f"pre_total_synapse_to_{target_type}"
fig, axs, mean_data = sorted_stripplot_with_means(
    data=data,
    y=y,
    x=x,
    weight_x=weight_x,
    size=f"pre_total_synapse_to_{target_type}",
    size_title=f"Output\nsynapses\nto {target_type}",
    xlabel=f"Proportion of synapses on {target_compartment}s",
)
save_matplotlib_figure(
    fig, "cell_type_output_proportion_to_excitatory_scatter", figure_out_path
)
save_variables(format="{:.1f}%", prefix=f"column_summary_{x}_", **mean_data.to_dict())
broad_mean_data = compute_weighted_means(data, x=x, by="broad_type", weight_x=weight_x)
save_variables(
    format="{:.1f}%", prefix=f"column_summary_{x}_", **broad_mean_data.to_dict()
)


target_type = "inhibitory"
x = f"pre_p_{target_compartment}_synapse_to_{target_type}"
fig, axs, mean_data = sorted_stripplot_with_means(
    data=data,
    y=y,
    x=x,
    weight_x=f"pre_total_synapse_to_{target_type}",
    size=f"pre_total_synapse_to_{target_type}",
    size_title=f"Output\nsynapses\nto {target_type}",
    xlabel=f"Proportion of synapses on {target_compartment}s",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
)
save_matplotlib_figure(
    fig, "cell_type_output_proportion_to_inhibitory_scatter", figure_out_path
)
save_variables(format="{:.1f}%", prefix=f"column_summary_{x}_", **mean_data.to_dict())
broad_mean_data = compute_weighted_means(data, x=x, by="broad_type", weight_x=weight_x)
save_variables(
    format="{:.1f}%", prefix=f"column_summary_{x}_", **broad_mean_data.to_dict()
)


size_norm = (2500, 25000)
x = f"post_p_{target_compartment}_synapse"
fig, axs, mean_data = sorted_stripplot_with_means(
    data=data,
    y=y,
    x=x,
    weight_x="post_total_synapses",
    size="post_total_synapses",
    size_title="Input\nsynapses",
    size_norm=size_norm,
    xlabel=f"Proportion of synapses on {target_compartment}s",
    legend_sizes=(2500, 10000, 25000),
)
save_matplotlib_figure(fig, "cell_type_input_spine_proportion", figure_out_path)
save_variables(format="{:.1f}%", prefix=f"column_summary_{x}_", **mean_data.to_dict())
broad_mean_data = compute_weighted_means(data, x=x, by="broad_type", weight_x=weight_x)
save_variables(
    format="{:.1f}%", prefix=f"column_summary_{x}_", **broad_mean_data.to_dict()
)

# %% get median size of synapses by pre synaptic type and postsynaptic target
medians = (
    all_pre_synapses.query("post_broad_type == 'excitatory'")
    .groupby(["pre_broad_type", "tag"])["size"]
    .mean()
)

th_spine = medians[("thalamic", "spine")]
th_shaft = medians[("thalamic", "shaft")]

exc_spine = medians[("excitatory", "spine")]
exc_shaft = medians[("excitatory", "shaft")]

save_variables(
    prefix="column_summary_target_size_",
    th_spine=th_spine,
    th_shaft=th_shaft,
    exc_spine=exc_spine,
    exc_shaft=exc_shaft,
    format="{:,.0f}",
)


# %% COUNT OF SPINE INPUTS BY CELL
fig, axs, _ = sorted_stripplot_with_means(
    data=data,
    y="cell_type_y",
    x="post_spine_synapses",
    weight_x="post_total_synapses",
    size="post_total_synapses",
    size_title="Input\nsynapses",
    loc="upper left",
    size_norm=size_norm,
    legend_sizes=(2500, 10000, 25000),
    bbox_to_anchor=(0.02, 0.985),
)
ax = axs[1]
ax.autoscale()
ax.set(xlim=(200, 20000), xscale="log")
save_matplotlib_figure(fig, "cell_type_input_spine_count", figure_out_path)

# %% COUNT OF MULTI SPINE INPUTS BY CELL
fig, axs, _ = sorted_stripplot_with_means(
    data=mod_proofread_info,
    y="cell_type_y",
    x="post_multi_spine_synapses",
    weight_x="post_total_synapses",
    size="post_total_synapses",
    size_title=f"Synapses\nto {target_type}",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.985),
)
save_matplotlib_figure(fig, "cell_type_input_multi_spine_count", figure_out_path)

# %% PROPORTION OF MULTI SPINE INPUTS BY CELL

x = "post_p_spine_site_is_multi"
weight_x = "post_spine_synapses"
size = "post_spine_synapses"
fig, axs, mean_data = sorted_stripplot_with_means(
    data=mod_proofread_info,
    y="cell_type_y",
    x=x,
    weight_x=weight_x,
    size=size,
    size_title="Input\nsynapses\nto spine",
    size_norm=(2000, 20000),
    legend_sizes=(2000, 10000, 20000),
    loc="upper right",
    bbox_to_anchor=(0.98, 0.985),
    xlim=(0, 0.6),
    xlabel="Proportion of spines with multiple inputs",
    # title="All inputs to spines",
)

save_matplotlib_figure(fig, "cell_type_input_proportion_multi_spine", figure_out_path)
save_variables(format="{:.1f}%", prefix=f"column_summary_{x}_", **mean_data.to_dict())
broad_mean_data = compute_weighted_means(
    mod_proofread_info.query(f"{x}.notna()"), x=x, by="broad_type", weight_x=weight_x
)
save_variables(
    format="{:.1f}%", prefix=f"column_summary_{x}_", **broad_mean_data.to_dict()
)

# %% STATS ON 23P MULTI SPINE VARIABILITY
print(mod_proofread_info.query("cell_type == '23P'")[x].describe())
stats = {}
stats["23P_min"] = mod_proofread_info.query("cell_type == '23P'")[x].min()
stats["23P_max"] = mod_proofread_info.query("cell_type == '23P'")[x].max()
stats["23P_q90"] = mod_proofread_info.query("cell_type == '23P'")[x].quantile(0.9)
stats["23P_q10"] = mod_proofread_info.query("cell_type == '23P'")[x].quantile(0.1)
save_variables(prefix=f"column_summary_{x}_", **stats, format="{:.1f}%")

# %% EXCITATORY DEPTH VS MULTI SPINE PROPORTION
data = cell_info.query(
    "post_total_synapses > 500 and broad_type_lda_prediction == 'excitatory' and broad_type == 'excitatory'"
)
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    data=data,
    y="pt_position_um_y",
    x="post_p_spine_site_is_multi",
    size="post_total_synapses",
    color=CELL_TYPE_PALETTE["excitatory"],
    sizes=(0.1, 15),
    linewidth=0,
    alpha=0.5,
)
ax.invert_yaxis()
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set(xlabel="Proportion of spines with multiple inputs", ylabel="Depth (um)")

save_matplotlib_figure(
    fig, "excitatory_depth_vs_multi_spine_proportion", figure_out_path
)
# %% PROPORTION OF MULTI SPINE OUTPUTS TO EXCITATORY SPINES BY CELL
target_type = "spine"
x = "pre_to_exc_p_spine_site_is_multi"
weight_x = "pre_to_exc_spine_synapses"
fig, axs, mean_data = sorted_stripplot_with_means(
    data=mod_proofread_info,
    y="cell_type_y",
    x=x,
    weight_x=weight_x,
    size="pre_to_exc_spine_synapses",
    size_title="Output\nsynapses\nto excitatory\nspines",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.985),
    xlabel="Proportion of spines with multiple inputs",
)
save_matplotlib_figure(
    fig, "cell_type_output_proportion_multi_spine_to_excitatory", figure_out_path
)

save_variables(format="{:.1f}%", prefix=f"column_summary_{x}_", **mean_data.to_dict())
broad_mean_data = compute_weighted_means(
    mod_proofread_info.query(f"{x}.notna()"), x=x, by="broad_type", weight_x=weight_x
)
save_variables(
    format="{:.1f}%", prefix=f"column_summary_{x}_", **broad_mean_data.to_dict()
)

# %% SPINE SHAFT RELATIONSHIPS COMPARE COLUMN AND DATASET

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
data = cell_info.query(
    "broad_type == 'inhibitory' and broad_type_lda_prediction == 'inhibitory' and post_total_synapses >= 350"
).copy()
data = cell_info.query(
    "post_total_synapses >= 500 and (post_spine_synapses > 1) and (post_shaft_synapses > 1) and (broad_type == broad_type_lda_prediction)"
)

print(CELL_TYPE_CATEGORIES)
exc_categories = CELL_TYPE_CATEGORIES[1:8]
inh_categories = CELL_TYPE_CATEGORIES[8:12]

data = data.query(
    "((broad_type == 'inhibitory') and (cell_type in @inh_categories)) or ((broad_type == 'excitatory') and (cell_type in @exc_categories))"
)

data = data.copy()

data["post_non_spine_synapses"] = (
    data["post_shaft_synapses"] + data["post_soma_synapses"]
)
data["is_manual"] = data["cell_type_source"] == "allen_v1_column_types_slanted_ref"
# data["is_IT"] = data["cell_type"].str.contains("IT")
x = "post_spine_synapses"
y = "post_shaft_synapses"
sns.scatterplot(
    data,
    x=x,
    y=y,
    ax=ax,
    hue="cell_type",
    palette=CELL_TYPE_PALETTE,
    # style="is_IT",
    # size="post_total_synapses",
    # sizes=(5, 50),
    s=5,
    legend=False,
)
ax.set(xscale="log", yscale="log")

fig, axs = plt.subplots(2, 7, figsize=(20, 20 / 7 * 2), sharex=True, sharey=True)

for i, (name, group) in enumerate(data.groupby("cell_type", observed=True)):
    ax = axs.flat[i]
    sns.scatterplot(
        group.query("is_manual"),
        x=x,
        y=y,
        ax=ax,
        hue="cell_type",
        palette=CELL_TYPE_PALETTE,
        s=10,
        legend=False,
        zorder=2,
        linewidth=0.2,
    )
    sns.scatterplot(
        group.query("~is_manual"),
        x=x,
        y=y,
        ax=ax,
        s=1,
        color="gray",
        legend=False,
        zorder=1,
        linewidth=0,
    )
    ax.set(xscale="log", yscale="log")

    stat, pvalue = pearsonr(group[x], group[y])
    ax.text(0.05, 0.9, f"r={stat:.2f}", transform=ax.transAxes)

# %% EVALUATION OF SPINE SHAFT RELATIONSHIP OUTLIER DETECTION

fig, axs = plt.subplots(2, 7, figsize=(20, 20 / 7 * 2), sharex=True, sharey=True)

for i, (name, group) in enumerate(data.groupby("cell_type", observed=True)):
    X = group[[x, y]].dropna().to_numpy()
    X = np.log(X)
    # n_components = 1

    # gmm = GaussianMixture(
    #     n_components=n_components,
    #     covariance_type="full",
    #     reg_covar=1e-6,
    #     init_params="k-means++",
    #     n_init=50,
    #     max_iter=200,
    #     # tol=1e-4,
    #     # random_state=42,
    # )
    # gmm.fit(X)

    model = EllipticEnvelope(contamination=0.01)
    model.fit(X)

    X = group[[x, y]].dropna().to_numpy()
    X = np.log(X)
    n_components = 1
    posteriors = model.score_samples(X)

    ax = axs.flat[i]
    sns.scatterplot(
        group,
        x=x,
        y=y,
        ax=ax,
        hue=posteriors[:],
        palette="RdBu_r",
        # hue_norm=(0, 1),
        legend=False,
        s=1,
        linewidth=0,
    )
    ax.set(xscale="log", yscale="log")

# %% HISTOGRAMS OF SYNAPSE CLEFT SIZE BY PRE-SYNAPTIC BROAD TYPE TO EXCITATORY POSTSYNAPTIC CELLS


def broad_type_remapper(broad_type: str) -> str:
    match broad_type:
        case "excitatory":
            return "Local excitatory"
        case "inhibitory":
            return "Inhibitory"
        case "thalamic":
            return "Thalamic"
        case _:
            return broad_type


fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True, layout="constrained")
for i, pre_broad_type in enumerate(["thalamic", "excitatory", "inhibitory"]):
    data = all_pre_synapses.query(
        "pre_broad_type == @pre_broad_type and post_broad_type == 'excitatory' and pre_compartment != 'dendrite'"
    )

    if pre_broad_type == "inhibitory":
        hue_order = ["spine", "shaft", "soma"]
    else:
        hue_order = ["spine", "shaft"]
    ax = axs[i]
    sns.histplot(
        data=data,
        x="size",
        hue="tag",
        hue_order=hue_order,
        ax=ax,
        log_scale=True,
        bins=40,
        element="poly",
        stat="density",
        palette=COMPARTMENT_PALETTE_MUTED_HEX,
        common_norm=False,
        legend=False,
    )
    if i == 0:
        ax.text(
            0.05,
            0.45,
            "Shaft",
            color=COMPARTMENT_PALETTE_MUTED_HEX["shaft"],
            transform=ax.transAxes,
        )
        ax.text(
            0.95,
            0.45,
            "Spine",
            ha="right",
            color=COMPARTMENT_PALETTE_MUTED_HEX["spine"],
            transform=ax.transAxes,
        )
    if pre_broad_type == "inhibitory":
        ax.text(
            0.75,
            0.45,
            "Soma",
            ha="center",
            color=COMPARTMENT_PALETTE_MUTED_HEX["soma"],
            transform=ax.transAxes,
        )

    # sns.move_legend(ax, "upper left", title="Target")
    text = rf"{broad_type_remapper(pre_broad_type)}"
    if i == 0:
        text = "Outputs from: " + text
    ax.text(
        0.05,
        0.8,
        text,
        transform=ax.transAxes,
    )
    ax.set(
        xlabel="Synapse cleft size (voxels)",
        yticks=[],
        ylabel="",
    )
    if i == 0:
        ax.set(
            ylabel="Normalized\ndensity",
        )
    ax.spines["left"].set_visible(False)

save_matplotlib_figure(fig, "broad_type_to_excitatory_size_histogram", figure_out_path)

# %% HISTOGRAMS OF SYNAPSE CLEFT SIZE BY PRE-SYNAPTIC BROAD TYPE TO INHIBITORY POSTSYNAPTIC CELLS

fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True, layout="constrained")
for i, pre_broad_type in enumerate(["thalamic", "excitatory", "inhibitory"]):
    data = all_pre_synapses.query(
        "pre_broad_type == @pre_broad_type and post_broad_type == 'inhibitory' and pre_compartment != 'dendrite'"
    )

    if pre_broad_type == "inhibitory":
        hue_order = ["spine", "shaft", "soma"]
    else:
        hue_order = ["spine", "shaft"]
    ax = axs[i]
    sns.histplot(
        data=data,
        x="size",
        hue="tag",
        hue_order=hue_order,
        ax=ax,
        log_scale=True,
        bins=40,
        element="poly",
        stat="density",
        palette=COMPARTMENT_PALETTE_MUTED_HEX,
        common_norm=False,
        legend=False,
    )
    if i == 0:
        ax.text(
            0.75,
            0.45,
            "Shaft",
            color=COMPARTMENT_PALETTE_MUTED_HEX["shaft"],
            transform=ax.transAxes,
        )
        ax.text(
            0.35,
            0.45,
            "Spine",
            ha="right",
            color=COMPARTMENT_PALETTE_MUTED_HEX["spine"],
            transform=ax.transAxes,
        )
    if pre_broad_type == "inhibitory":
        ax.text(
            0.8,
            0.45,
            "Soma",
            ha="center",
            color=COMPARTMENT_PALETTE_MUTED_HEX["soma"],
            transform=ax.transAxes,
        )

    # sns.move_legend(ax, "upper left", title="Target")
    text = rf"{broad_type_remapper(pre_broad_type)}"
    if i == 0:
        text = "Outputs from: " + text
    ax.text(
        0.05,
        0.8,
        text,
        transform=ax.transAxes,
    )
    ax.set(
        xlabel="Synapse cleft size (voxels)",
        yticks=[],
        ylabel="",
    )
    if i == 0:
        ax.set(
            ylabel="Normalized\ndensity",
        )
    ax.spines["left"].set_visible(False)

save_matplotlib_figure(fig, "broad_type_to_inhibitory_size_histogram", figure_out_path)


# %% CELL TYPE TO EXCITATORY HISTOGRAMS OF SIZES BY TARGET
fig, axs = plt.subplots(4, 3, figsize=(15, 7), sharex=True, layout="constrained")

for i, (pre_cell_type, data) in enumerate(
    all_pre_synapses.query(
        "post_broad_type == 'excitatory' and pre_compartment != 'dendrite'"
    ).groupby("pre_cell_type", observed=True)
):
    ax = axs.T.flat[i]
    if pre_cell_type == "BC":
        hue_order = ["spine", "shaft", "soma"]
        ax.text(
            0.8,
            0.45,
            "Soma",
            ha="right",
            color=COMPARTMENT_PALETTE_MUTED_HEX["soma"],
            transform=ax.transAxes,
        )
    else:
        hue_order = ["spine", "shaft"]
    if len(data) > 0:
        sns.histplot(
            data=data,
            x="size",
            hue="tag",
            hue_order=hue_order,
            ax=ax,
            log_scale=True,
            bins=40,
            element="step",
            stat="density",
            palette=COMPARTMENT_PALETTE_MUTED_HEX,
            common_norm=False,
            legend=False,
        )
        # if i == 0:
        #     sns.move_legend(
        #         ax,
        #         "upper left",
        #         title="",
        #         ncol=2,
        #         bbox_to_anchor=(0.0, 0.7),
        #         handletextpad=0,
        #     )
        # else:
        #     ax.get_legend().remove()
        ax.set(
            # title=rf"{pre_cell_type} $\rightarrow$ excitatory synapses",
            xlabel="Synapse cleft size (voxels)",
            ylabel="",
            # ylabel="Normalized density",
            yticks=[],
        )
        if i == 0:
            ax.text(
                0.05,
                0.45,
                "Shaft",
                color=COMPARTMENT_PALETTE_MUTED_HEX["shaft"],
                transform=ax.transAxes,
            )
            ax.text(
                0.95,
                0.45,
                "Spine",
                ha="right",
                color=COMPARTMENT_PALETTE_MUTED_HEX["spine"],
                transform=ax.transAxes,
            )
            ax.set(ylabel="Normalized\ndensity")
            ax.text(
                0.05,
                0.8,
                f"Outputs from: {remapper(pre_cell_type)}",
                transform=ax.transAxes,
            )
        else:
            ax.text(0.05, 0.8, remapper(pre_cell_type), transform=ax.transAxes)

        ax.spines["left"].set_visible(False)

save_matplotlib_figure(fig, "cell_type_to_excitatory_size_histograms", figure_out_path)
# %% HISTOGRAMS OF SYNAPSES SIZE ONTO EXCITATORY OR INHIBITORY BY COMPARTMENT

fig, axs = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True, layout="constrained")
for i, post_broad_type in enumerate(["excitatory", "inhibitory"]):
    data = all_post_synapses.query("post_broad_type == @post_broad_type")

    hue_order = ["spine", "shaft", "soma"]

    ax = axs[i]
    sns.histplot(
        data=data,
        x="size",
        hue="tag",
        hue_order=hue_order,
        ax=ax,
        log_scale=True,
        bins=40,
        element="poly",
        stat="density",
        palette=COMPARTMENT_PALETTE_MUTED_HEX,
        common_norm=False,
        legend=False,
    )
    if i == 0:
        ax.text(
            0.05,
            0.45,
            "Shaft",
            color=COMPARTMENT_PALETTE_MUTED_HEX["shaft"],
            transform=ax.transAxes,
        )
        ax.text(
            0.85,
            0.45,
            "Spine",
            ha="right",
            color=COMPARTMENT_PALETTE_MUTED_HEX["spine"],
            transform=ax.transAxes,
        )
        ax.text(
            0.25,
            0.45,
            "Soma",
            ha="center",
            color=COMPARTMENT_PALETTE_MUTED_HEX["soma"],
            transform=ax.transAxes,
        )

    # sns.move_legend(ax, "upper left", title="Target")
    text = rf"{post_broad_type.capitalize()}"
    if i == 0:
        text = "Inputs to: " + text
    ax.text(
        0.05,
        0.8,
        text,
        transform=ax.transAxes,
    )

    ax.set(
        #
        xlabel="Synapse cleft size (voxels)",
        yticks=[],
        ylabel="",
    )
    if i == 0:
        ax.set(
            ylabel="Normalized\ndensity",
        )
    ax.spines["left"].set_visible(False)

save_matplotlib_figure(fig, "broad_type_input_size_histogram", figure_out_path)


# %% INPUT SIZE HISTOGRAMS BY COMPARTMENT AND CELL TYPE
fig, axs = plt.subplots(4, 3, figsize=(15, 7), sharex=True, layout="constrained")
axs[0, 0].set_visible(False)
for i, (post_cell_type, data) in enumerate(
    all_post_synapses.groupby("post_cell_type", observed=True)
):
    ax = axs.T.flat[i + 1]
    hue_order = ["spine", "shaft", "soma"]
    if len(data) > 0:
        sns.histplot(
            data=data,
            x="size",
            hue="tag",
            hue_order=hue_order,
            ax=ax,
            log_scale=True,
            bins=40,
            element="step",
            stat="density",
            palette=COMPARTMENT_PALETTE_MUTED_HEX,
            common_norm=False,
            legend=False,
        )
        # if i == 0:
        #     sns.move_legend(
        #         ax,
        #         "upper left",
        #         title="",
        #         ncol=2,
        #         bbox_to_anchor=(0.0, 0.7),
        #         handletextpad=0,
        #     )
        # else:
        #     ax.get_legend().remove()
        ax.set(
            # title=rf"{pre_cell_type} $\rightarrow$ excitatory synapses",
            xlabel="Synapse cleft size (voxels)",
            ylabel="",
            # ylabel="Normalized density",
            yticks=[],
        )
        if i == 0:
            ax.text(
                0.05,
                0.5,
                "Shaft",
                color=COMPARTMENT_PALETTE_MUTED_HEX["shaft"],
                transform=ax.transAxes,
            )
            ax.text(
                0.35,
                0.5,
                "Soma",
                ha="right",
                color=COMPARTMENT_PALETTE_MUTED_HEX["soma"],
                transform=ax.transAxes,
            )
            ax.text(
                0.9,
                0.5,
                "Spine",
                ha="right",
                color=COMPARTMENT_PALETTE_MUTED_HEX["spine"],
                transform=ax.transAxes,
            )
            ax.set(ylabel="Normalized\ndensity")
            ax.text(
                0.05,
                0.8,
                f"Inputs to: {remapper(post_cell_type)}",
                transform=ax.transAxes,
            )
        else:
            ax.text(0.05, 0.8, remapper(post_cell_type), transform=ax.transAxes)
        ax.spines["left"].set_visible(False)

save_matplotlib_figure(fig, "input_cell_type_size_histograms", figure_out_path)

# %% GET NUMBER OF CONNECTIONS IN TABLE
all_pre_synapses["spine_multi_count_in_table"] = (
    all_pre_synapses.query("spine_is_multi")
    .groupby(["post_pt_root_id", "component_id"])
    .transform("size")
)

all_pre_synapses.groupby(["post_broad_type", "spine_is_multi"]).size()

# %% GET FILTERED SYNAPSE COUNTS
filtered = all_pre_synapses
n_total_synapses = filtered.shape[0]
print(n_total_synapses)
filtered = filtered.query("post_broad_type == 'excitatory'")
n_to_e_synapses = filtered.shape[0]
print(n_to_e_synapses)
filtered = filtered.query("spine_is_multi")
n_multi_synapses = filtered.shape[0]
print(n_multi_synapses)


# %% INITIAL TABULATION FOR MULTI SYNAPSES
multi_synapses = all_pre_synapses.query(
    "spine_is_multi and post_broad_type == 'excitatory'"
).copy()

pre_counts = (
    multi_synapses.reset_index()
    .groupby(
        ["spine_group_id", "pre_broad_type"],
        observed=True,
    )["pre_pt_root_id"]
    .nunique()
    .unstack()
    .sum(axis=0)
)
pre_props = pre_counts / pre_counts.sum()
pre_props

# %% GET ANNOTATIONS OF TRUE MULTI SPINES


def get_multi_annotations():
    returned_links = [
        "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4811205330862080",  # EE on E
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6484872683061248",  # EI on E
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5527458345385984",  # IT on E
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5391743175360512",  # II on E
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5173236714176512",  # ET on E
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6098825721675776",  # EE on I
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5201463239245824",  # EI on I
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5047802907328512",  # ET on I
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4719068014706688",  # II on I
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6041454261567488",  # IT on I
    ]

    client = CAVEclient("minnie65_phase3_v1", version=1412)

    dfs = []
    for link in returned_links:
        state_id = int(link.split("/")[-1])
        state = client.state.get_state_json(state_id)
        parser = StateParser(state)
        df = parser.annotation_dataframe(expand_tags=True)
        dfs.append(df)

    annotations = pd.concat(dfs, ignore_index=True)
    annotations["post_pt_root_id"] = (
        annotations["description"].str.split("_").str[0].astype(int)
    )
    annotations["component_id"] = (
        annotations["description"].str.split("_").str[1].astype(int)
    )
    annotations = annotations.set_index(["post_pt_root_id", "component_id"])
    return annotations


annotations = get_multi_annotations()


# %% TABULATION OF KNOWN INPUTS PER MULTI SPINE
grouping = ["post_pt_root_id", "component_id", "spine_group_id"]
multi_synapses["component_id"] = multi_synapses["component_id"].astype("Int64")
multi_synapses["spine_group_id"] = multi_synapses["spine_group_id"].astype("Int64")
total_inputs = multi_synapses.groupby(grouping)["spine_n_pre_pt_root_ids"].first()
known_input_counts = (
    multi_synapses.reset_index()
    .groupby(
        grouping + ["pre_broad_type"],
        observed=True,
    )["pre_pt_root_id"]
    .nunique()
    .unstack()
    .fillna(0)
)

# %% FILTRATION OF KNOWN DOUBLE INPUT SPINES

input_counts = known_input_counts.copy()
input_counts["total"] = total_inputs
input_counts = input_counts[input_counts["total"] == 2]
input_counts["unknown"] = input_counts["total"] - input_counts[
    ["thalamic", "excitatory", "inhibitory"]
].sum(axis=1)
input_counts = input_counts.astype(int)
input_counts


type_counts = (
    known_input_counts[["thalamic", "excitatory", "inhibitory"]]
    .sum(axis=0)
    .transform(lambda x: x / x.sum())
    .rename("prop")
)
type_counts.index = type_counts.index.astype(str)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.barplot(
    type_counts.reset_index(), y="pre_broad_type", x="prop", color="black", ax=ax
)
ax.set(xlabel="Proportion of synapses", ylabel="")
for i, v in enumerate(type_counts):
    ax.text(
        v,
        i,
        f" {v:.2f}",
        color="black",
        va="center",
        ha="left",
        fontsize="medium",
    )


n_spines = input_counts.shape[0]
n_both_known_spines = input_counts[input_counts["unknown"] == 0].shape[0]

print(f"n_spines: {n_spines}")

n_both_known_synapses = n_both_known_spines * 2
print(f"n_both_known_synapses: {n_both_known_synapses}")


filter_summary = pd.Series(
    {
        "Proofread synapses": n_total_synapses,
        "^ to excitatory": n_to_e_synapses,
        "^ to multi-spine": n_multi_synapses,
        "^ with 2 proofread inputs": n_both_known_synapses,
    },
    name="count",
)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.barplot(filter_summary.reset_index(), y="index", x="count", color="black")
ax.set(xscale="log", ylabel="", xlabel="Number of synapses")
for i, v in enumerate(filter_summary):
    ax.text(
        v,
        i,
        f" {v:,}",
        color="black",
        va="center",
        ha="left",
        fontsize="medium",
    )
save_matplotlib_figure(fig, "multi_spine_filter_summary", figure_out_path)


input_counts = input_counts[input_counts["unknown"] == 0]

keep_index = annotations[annotations["singlehead"]].index

input_counts = input_counts.reset_index().set_index(["post_pt_root_id", "component_id"])

keep_index = keep_index.intersection(input_counts.index)

input_counts = input_counts.loc[keep_index]

input_counts = input_counts.reset_index(drop=True).set_index("spine_group_id")

# %% MULTI COINNERVATION WITH INHIBITORY
th_pairing = input_counts.query("thalamic >= 1")["inhibitory"].mean()
exc_pairing = input_counts.query("excitatory >= 1")["inhibitory"].mean()
x = input_counts.query("inhibitory >= 0").copy()
inh_pairing = (x["inhibitory"] == 2).mean()

pairing_data = pd.Series(
    {
        "Thalamic": th_pairing,
        "Intracortical\nExcitatory": exc_pairing,
        "Inhibitory": inh_pairing,
    }
)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.barplot(pairing_data.reset_index(), y="index", x=0, color="black", ax=ax)
ax.set(xlabel="Proportion double-input\nspines w/ inhibitory partner", ylabel="")

save_matplotlib_figure(
    fig, "multi_spine_inhibitory_pairing", figure_out_path, pad_inches=0.1
)

# %%
input_counts["all_excitatory"] = input_counts["excitatory"] + input_counts["thalamic"]

# %%
pairing_props = (
    input_counts[["all_excitatory", "inhibitory"]]
    .value_counts(normalize=True)
    .reset_index()
)
pairing_props["name"] = (
    pairing_props["all_excitatory"].astype(str)
    + " Excitatory\n"
    + pairing_props["inhibitory"].astype(str)
    + " Inhibitory"
)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# sns.barplot(pairing_props.reset_index(), x="proportion", y="name", ax=ax)

shift = 0.2
right = -0.05
for i, (_, row) in enumerate(pairing_props.iterrows()):
    x = row["proportion"]
    thing1 = "Inhibitory" if row["inhibitory"] >= 1 else "Excitatory"
    thing2 = "Excitatory" if row["all_excitatory"] > 0 else "Inhibitory"
    color1 = (
        CELL_TYPE_PALETTE["inhibitory"]
        if row["inhibitory"] >= 1
        else CELL_TYPE_PALETTE["excitatory"]
    )
    color2 = (
        CELL_TYPE_PALETTE["excitatory"]
        if row["all_excitatory"] > 0
        else CELL_TYPE_PALETTE["inhibitory"]
    )
    ax.barh(i + shift, x, height=0.4, color=color1)
    ax.barh(i - shift, x, height=0.4, color=color2)
    ax.plot([0, x], [i, i], color="black", linewidth=2)
    ax.text(
        right,
        i - shift,
        thing1,
        ha="right",
        va="center",
    )
    ax.text(
        right,
        i + shift,
        thing2,
        ha="right",
        va="center",
    )
    ax.text(x + 0.02, i, f"{x:.2f}", ha="left", va="center")

ax.set(
    yticks=[],
    # yticklabels=pairing_props["name"],
    xlabel="Proportion of \ndouble-input spines",
)
ax.set_ylabel("Input synapse pairing", labelpad=115)

ax.invert_yaxis()

save_matplotlib_figure(fig, "multi_spine_pairing", figure_out_path, pad_inches=0.2)

# %% NULL DISTRIBUTION OF MULTI SPINE PROPORTIONS FOR 23P CELLS
select_synapses = all_post_synapses.query(
    "post_cell_type == '23P' and tag == 'spine'"
).copy()
select_synapses["logsize"] = np.log(select_synapses["size"])
select_synapses["bin"] = pd.cut(select_synapses["logsize"], bins=100)
select_synapses["bin"].value_counts()

shuffle_synapses = select_synapses.copy()
new_tags = []
for bin, group in shuffle_synapses.groupby("bin"):
    shuffled_tags = group["tag_detailed"].sample(frac=1.0, replace=False).values
    shuffled_tags = pd.Series(shuffled_tags, index=group.index)
    new_tags.append(shuffled_tags)

new_tags = pd.concat(new_tags).sort_index()
shuffle_synapses["shuffled_tag_detailed"] = new_tags
counts = shuffle_synapses.groupby("post_pt_root_id")[
    "shuffled_tag_detailed"
].value_counts()
p_spine_synapse_is_multi = counts.unstack().fillna(0)
p_spine_synapse_is_multi_null = p_spine_synapse_is_multi.div(
    p_spine_synapse_is_multi.sum(axis=1), axis=0
)["multi_spine"]

counts = shuffle_synapses.groupby("post_pt_root_id")["tag_detailed"].value_counts()
p_spine_synapse_is_multi = counts.unstack().fillna(0)
p_spine_synapse_is_multi = p_spine_synapse_is_multi.div(
    p_spine_synapse_is_multi.sum(axis=1), axis=0
)["multi_spine"]

bins = np.geomspace(0.03, 0.3, 40)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.histplot(
    x=p_spine_synapse_is_multi,
    ax=ax,
    bins=bins,
    label="Observed",
    color="black",
)
sns.histplot(
    data=p_spine_synapse_is_multi_null,
    ax=ax,
    color="lightgrey",
    alpha=0.5,
    bins=bins,
    label="Null",
    # linewidth=1,
)
ax.set(xlabel="Proportion of spines with multiple inputs", xlim=(0.05, 0.3))
ax.set(xscale="log", ylabel="Cell count")
ticks = [0.1, 0.2, 0.3]
ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
ax.xaxis.set_major_formatter(plt.FixedFormatter(ticks))  # ty: ignore
ax.xaxis.set_minor_formatter(plt.NullFormatter())

ax.legend()

save_matplotlib_figure(fig, "multi_spine_proportion_null_23P", figure_out_path)

# %% LOGNORMAL FIT OF MULTI SPINE PROPORTION FOR EXCITATORY CELLS

x = "post_p_spine_site_is_multi"
cell_type = "excitatory"
query_cell_info = (
    cell_info.query(
        "broad_type == @cell_type and post_total_synapses > 500 and broad_type_lda_prediction == @cell_type"
    )
    .sort_values(x, ascending=False)
    .copy()
)

x_vals = query_cell_info[x].dropna().values

BINS = np.geomspace(0.005, 0.4, 50)


def histplot_with_lognorm_fit(x, ax=None, bins=None, label=None, color=None):
    if bins is None:
        bins = BINS
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    sns.histplot(
        data=query_cell_info,
        x=x,
        stat="density",
        bins=bins,
        ax=ax,
        # color="black",
        color=color,
        label=label,
        # element="step",
        # fill=False,
    )

    gaussian_fit = lognorm.fit(x)
    x_fit = np.geomspace(x.min(), x.max(), 100)
    y_fit = lognorm.pdf(x_fit, *gaussian_fit)
    ax.plot(x_fit, y_fit, color="dimgrey", linestyle="--", label="Lognormal")
    ax.set_xscale("log")
    # ax.legend(loc='lower left')
    ax.text(0.0, 0.8, "Lognormal\nfit", transform=ax.transAxes, color="dimgrey")
    ax.spines["left"].set_visible(False)
    ax.set(
        # ylabel="Density",
        yticks=[],
        xticks=[0.05, 0.1, 0.2, 0.3],
    )
    # ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    return fig, ax, gaussian_fit


ticks = [0.02, 0.05, 0.1, 0.2]
xlim = (0.01, 0.25)

fig, ax, fit = histplot_with_lognorm_fit(
    x_vals,
    color=CELL_TYPE_PALETTE["excitatory"],
)

ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
ax.xaxis.set_major_formatter(plt.FixedFormatter(ticks))  # ty: ignore
ax.set(xlabel="Proportion of spines with multiple inputs")
ax.set(
    xlabel="Proportion of spines\nwith multiple inputs",
)
ax.set(xlim=xlim)
save_matplotlib_figure(fig, f"excitatory_{x}_distribution", figure_out_path)

# %%


fig, axs = plt.subplots(2, 1, figsize=(4.56, 7.17), layout="tight", sharex=True)
ax = axs[0]
fig, ax, fit = histplot_with_lognorm_fit(
    x_vals,
    ax=ax,
    color=CELL_TYPE_PALETTE["excitatory"],
)
ax.set(ylabel="Density\n(cells)")

# query_cell_info = (
#     cell_info.query(
#         "broad_type == @cell_type and post_total_synapses > 1000 and broad_type_lda_prediction == @cell_type and in_column"
#     )
#     .sort_values(x, ascending=False)
#     .copy()
# )

# x_vals = query_cell_info[x].dropna().values
# fig, ax, fit = histplot_with_lognorm_fit(x_vals, color="red", ax=axs[0])


y = "post_p_spine_synapse"
ax = axs[1]
sns.scatterplot(
    data=query_cell_info,
    x=x,
    y=y,
    ax=ax,
    color="black",
    s=1,
    linewidth=0,
    alpha=0.5,
)
ax.set(yscale="log", ylim=(0.45, 0.88))

ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
ax.xaxis.set_major_formatter(plt.FixedFormatter(ticks))  # ty: ignore

ax.set(xlabel="Proportion of spines with multiple inputs")
ax.set(
    xlabel="Proportion of spines\nwith multiple inputs",
    ylabel="Proportion of\nsynapses on spines",
)
ax.set(xlim=xlim)
ax.yaxis.set_major_locator(plt.FixedLocator([0.5, 0.6, 0.7, 0.8]))

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

save_matplotlib_figure(
    fig, f"excitatory_{x}_distribution_with_p_spine", figure_out_path
)

# %%

xs = np.log(query_cell_info[x])
ys = np.log(query_cell_info[y])

stat, pvalue = pearsonr(xs, ys)

save_variables(
    prefix="column_summary_", multi_spine_spine_prop_pearson_r=stat, format="{:.2f}"
)


# %%
cell_type = "inhibitory"
x = "post_p_spine_site_is_multi"
sns.histplot(
    data=cell_info.query(
        "broad_type == @cell_type and post_total_synapses > 1000 and broad_type_lda_prediction == @cell_type and post_p_spine_site_is_multi > 0"
    )
    .sort_values(x, ascending=False)
    .copy(),
    x=x,
    stat="density",
    # bins=bins,
    log_scale=True,
    color="black",
)

# %%
x = "post_p_spine_site_is_multi"

cell_type = "MC"
query_cell_info = (
    cell_info.query(
        "cell_type == @cell_type and broad_type == 'inhibitory' and broad_type_lda_prediction == 'inhibitory' and post_total_synapses > 500"
    )
    .sort_values(x, ascending=False)
    .copy()
)
x_vals = query_cell_info[x].dropna().values
bins = np.geomspace(x_vals.min(), x_vals.max(), 20)
fig, ax, fit = histplot_with_lognorm_fit(x_vals, bins=bins)
ax.text(0.1, 0.8, f"{cell_type}", transform=ax.transAxes)

# %% LOGNORMAL FIT OF MULTI SPINE PROPORTION BY EXCITATORY CELL TYPE
cell_types = CELL_TYPE_CATEGORIES[1:8]
cell_types

# major_ticks = [0.02, 0.2]

fig, axs = plt.subplots(len(cell_types), 1, figsize=(8, 12), layout="constrained")
fig2, axs2 = plt.subplots(1, 1, figsize=(8, 8))
for i, cell_type in enumerate(cell_types):
    ax = axs[i]
    query_cell_info = (
        cell_info.query(
            "cell_type == @cell_type and broad_type_lda_prediction == 'excitatory' and post_total_synapses > 500 and broad_type == 'excitatory'"
        )
        .sort_values(x, ascending=False)
        .copy()
    )
    x_vals = query_cell_info[x].dropna().values

    bins = np.geomspace(0.005, 0.3, 50)
    sns.histplot(
        x=x_vals,
        stat="density",
        bins=bins,
        ax=ax,
        color="black",
    )

    gaussian_fit = lognorm.fit(x_vals)
    x_fit = np.geomspace(x_vals.min(), x_vals.max(), 100)
    y_fit = lognorm.pdf(x_fit, *gaussian_fit)
    ax.plot(x_fit, y_fit, color="red", linestyle="--", label="Lognormal fit")
    ax.set_xscale("log")
    ax.spines["left"].set_visible(False)
    # ax.set_title()
    ax.text(0, 0.7, f"{cell_type}", transform=ax.transAxes)
    ax.set(xlim=(0.01, 0.2), yticks=[], xticks=[], xticklabels=[], xlabel="", ylabel="")
    # ax.set_xticklabels([])
    if i != len(cell_types) - 1:
        ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(plt.FixedFormatter(ticks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set(xlabel="Proportion of spines with multiple inputs")
    if i == 0:
        ax.legend()

    # make a Q-Q plot on the other axis

    plot_data, stat_data = probplot(x_vals, sparams=gaussian_fit, dist=lognorm)
    ax2 = axs2
    ax2.plot(
        plot_data[0],
        plot_data[1],
        color=CELL_TYPE_PALETTE[cell_type],
        label=cell_type,
    )

save_matplotlib_figure(
    fig, f"excitatory_{x}_distribution_by_cell_type", figure_out_path
)

# %% CELL TYPE FOCUS COMPARISON TO MC INPUT COUNTS
# cell_type = "23P"

# x = "post_p_spine_site_is_multi"
# query_cell_info = (
#     cell_info.query(
#         "cell_type == @cell_type and broad_type == 'excitatory' and broad_type_lda_prediction == 'excitatory' and post_total_synapses > 1000"
#     )
#     .sort_values(x, ascending=False)
#     .copy()
# )

# query_synapses = column_column_synapses.query("post_cell_type == @cell_type")

# pre_type_counts = (
#     query_synapses.groupby(["post_pt_root_id", "pre_cell_type"]).size().unstack()
# )
# query_cell_info["bc_mc_ratio"] = pre_type_counts["BC"] / pre_type_counts["MC"]
# query_cell_info["mc_count"] = pre_type_counts["MC"]
# query_cell_info["bc_count"] = pre_type_counts["BC"]

# query_cell_info["mc_total_prop"] = pre_type_counts["MC"] / pre_type_counts.sum(axis=1)
# query_cell_info["bc_total_prop"] = pre_type_counts["BC"] / pre_type_counts.sum(axis=1)

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# sns.scatterplot(
#     data=query_cell_info,
#     x="mc_count",
#     y="post_p_spine_site_is_multi",
#     hue="mtype",
#     # hue_order=['L2a', "L2b", "L2c","L3a", "L3b"],
#     hue_order=["L4a", "L4b", "L4c"],
#     legend=True,
# )
# # ax.set_xscale("log")
# ax.set(
#     xlabel="MC input count",
#     ylabel="Proportion spines with multiple inputs",
#     title="4P cells in column",
# )
# ax.set_xscale("log")
# ax.set_yscale("log")

# %%
# cell_type = "5P-IT"
# query_cell_info = cell_info.query(
#     "cell_type == @cell_type and broad_type == 'excitatory' and broad_type_lda_prediction == 'excitatory' and post_total_synapses > 1000"
# )

# skel_info = (
#     pd.read_csv("neuron_skel_info.csv")
#     .set_index("root_id")
#     .drop(columns=["broad_type", "cell_type"])
# )

# y = "length_max_dist_to_tip_below_40um"
# x = "post_p_spine_site_is_multi"
# query_cell_info = (
#     query_cell_info.drop(columns=skel_info.columns, errors="ignore")
#     .join(skel_info, how="left")
#     .dropna(subset=[x, y])
# )

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# sns.scatterplot(
#     data=query_cell_info,
#     x=x,
#     y=y,
#     # hue="mtype",
#     # hue_order=['L2a', "L2b", "L2c","L3a", "L3b"],
#     # hue_order=["L4a", "L4b", "L4c"],
#     legend=True,
#     s=10,
# )
# ax.set(xscale="log", yscale="log")

# xs = np.log(query_cell_info[x].values)
# ys = np.log(query_cell_info[y].values)
# pearsonr(xs, ys)

# %% MULTI SPINE RATE CORRELATION WITH TIP LENGTH DISTRIBUTION

# sns.histplot(data=skel_info, x=y, log_scale=True, bins=40)

# %% TIP LENGTH DISTRIBUTION


# %% INDEX WRITING (FOR INTERACTION WITH TIP STATS FILE)
# use numpy savetxt to write to a csv
# np.savetxt(
#     "index.csv", query_cell_info.index.to_numpy().astype(int), delimiter=",", fmt="%d"
# )


# %% P MULTI DISTRIBUTION BY MTYPE WITHIN CELL TYPE
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# sns.histplot(
#     data=query_cell_info,
#     x="post_p_spine_site_is_multi",
#     hue="mtype",
#     # hue_order=['L2a', "L2b", "L2c","L3a", "L3b"],
#     hue_order=["L4a", "L4b", "L4c"],
#     legend=True,
#     log_scale=True,
#     element="step",
#     common_norm=False,
#     stat="density",
# )
# ax.set(xlabel="Proportion of spines\nwith multiple inputs")

# %% GET SOME EXAMPLE NEURONS WITH MANY MULTIS
cell_type = "23P"
query_cell_info = cell_info.query(
    "cell_type == @cell_type and broad_type == 'excitatory' and broad_type_lda_prediction == 'excitatory' and post_total_synapses > 1000 and in_column"
)
query_cell_info.sort_values("post_p_spine_site_is_multi", ascending=True).index[:15]

# %% DISTRIBUTION OF P SPINE FOR 23P CELLS
cell_type = "23P"

x = "post_spine_synapses"
x = "post_p_spine_site_is_multi"
data = cell_info.query("cell_type == @cell_type and in_column")
data = cell_info.query(
    "cell_type == @cell_type and broad_type == 'excitatory' and broad_type_lda_prediction == 'excitatory' and post_total_synapses > 1000"
)
# data = data.query("in_column")
pad = 0.001
quantiles = np.quantile(data[x], [pad, 1])
print(len(data))
data = data.query(f"{x} >= @quantiles[0]")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.histplot(
    data=data,
    # hue="in_column",
    x=x,
    log_scale=True,
    common_norm=False,
    stat="proportion",
    bins=40,
    # element='poly',
    kde=True,
)
ax.set(xlim=quantiles)

# %% OUTPUT MULTI SPINE RATES
cell_type = "4P"

data = cell_info.query("cell_type == @cell_type and in_column")

x = "pre_p_spine_site_is_multi"

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.histplot(
    data=data,
    x=x,
    log_scale=True,
    common_norm=False,
    stat="proportion",
    # bins=30,
    # element="step",
)
# %% OUTPUT SPINE RATES
# cell_types = ["MC", "BC"]

# data = cell_info.query("cell_type.isin(@cell_types) and in_column")

# x = "pre_p_spine_synapse"

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# sns.histplot(
#     data=data,
#     x=x,
#     log_scale=True,
#     common_norm=False,
#     stat="proportion",
#     # bins=30,
#     # element="step",
# )
# %%
# cell_types = ["MC"]
# data = cell_info.query("cell_type.isin(@cell_types) and in_column")

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# sns.scatterplot(
#     data=data,
#     x=x,
#     y="pt_position_um_y",
#     # log_scale=True,
#     # common_norm=False,
#     # stat="proportion",
#     # bins=30,
#     # element="step",
# )
# ax.set_xscale("log")


# %% CELL P MULTI CORRELATION WITH OTHER FACTORS
import numpy as np

x = "post_p_spine_synapse_is_multi"
y = "post_p_spine_synapse"
y_columns = {
    "pt_position_um_y": {"scale": "linear", "label": "Depth (um)"},
    "post_shaft_synapses": {"scale": "log", "label": "Number of\nshaft synapses"},
    "post_p_shaft_synapse": {
        "scale": "log",
        "label": "Proportion of\nsynapses on shaft",
    },
    "post_mean_synapse_size": {"scale": "log", "label": "Mean synapse\nsize (voxels)"},
}
highlight = True
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
    for i, (y_name, y_props) in enumerate(y_columns.items()):
        ax = axs[j, i]
        data = cell_info.query(
            'cell_type == @cell_type and broad_type_lda_prediction == "excitatory" and post_total_synapses >= 2000'
        )

        sns.scatterplot(
            data=data,
            x=x,
            y=y_name,
            alpha=0.7,
            linewidth=0,
            ax=ax,
            s=3,
            color="black",
        )
        if highlight:
            sns.scatterplot(
                data=data.query("in_column"),
                x=x,
                y=y_name,
                alpha=1,
                linewidth=0.2,
                ax=ax,
                s=12,
                color="red",
                zorder=10,
            )
        ax.set(yscale=y_props["scale"], xscale="log")

        ys = data[y_name].values
        ylim = np.quantile(ys, [0.005, 0.995])
        if y_name == "pt_position_um_y":
            ylim = ylim[::-1]
        ax.set(ylim=ylim)

        xs = data[x].values
        xs = np.log(xs)

        if y_props["scale"] == "log":
            ys = np.log(ys)
        stat, pvalue = pearsonr(xs, ys)

        save_variables(
            prefix=f"column_summary_{cell_type}_",
            **{f"p_multi_spine_{y_name}_pearson_r": stat},
            format="{:.2f}",
        )

        ax.set_title(f"r={stat:.2f}")
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

save_matplotlib_figure(fig, "p_multi_correlations", figure_out_path)


# %% CELL P MULTI EFFECT BY VISUAL AREA

# from scipy.stats import ks_2samp

# hues = ["visual_area"]
# for hue in hues:
#     fig, ax = plt.subplots(1, 1, figsize=(6, 4))
#     sns.histplot(
#         data=data,
#         x=x,
#         hue=hue,
#         hue_order=["V1", "RL", "AL"],
#         stat="density",
#         common_norm=False,
#         log_scale=True,
#         bins=40,
#         element="step",
#         fill=False,
#     )

#     for i, (group1, data1) in enumerate(data.groupby(hue)):
#         x1 = data1[x].values
#         for j, (group2, data2) in enumerate(data.groupby(hue)):
#             if j <= i:
#                 continue
#             x2 = data2[x].values

#             result = ks_2samp(x1, x2)
#             print(group1, group2, result.statistic, result.pvalue)


# %% COMPUTE DELTA DEPTH (NEED TO REFACTOR)

# TODO refactor into the table creation
all_post_synapses["post_delta_depth"] = (
    column_cell_info.loc[
        all_post_synapses["post_pt_root_id"], "pt_position_um_y"
    ].values
    - all_post_synapses["transformed_um_y"].values
)
# %% EXCITATORY P MULTI HISTOGRAMS BY DEPTH AND DISTANCES

# def filter_and_bin(data, y):
#     percentiles = data[y].quantile([0.01, 0.99])
#     data = data.query(f"{y} >= @percentiles[0.01] and {y} <= @percentiles[0.99]").copy()
#     bins = np.linspace(percentiles[0.01], percentiles[0.99], 25)
#     data[f"{y}_bin"] = pd.cut(data[y], bins=bins)
#     data[f"{y}_bin_mid"] = data[f"{y}_bin"].apply(lambda x: x.mid).astype(float)
#     return data, bins


def filter_and_bin(data, y, use_quantiles=True, trim_quantiles=(0.01, 0.99), n_bins=25):
    if trim_quantiles is None:
        trim_quantiles = (0.0, 1.0)
    percentiles = data[y].quantile(trim_quantiles)
    data = data.query(
        f"{y} >= @percentiles[{trim_quantiles[0]}] and {y} <= @percentiles[{trim_quantiles[1]}]"
    ).copy()
    # bins =
    if use_quantiles:
        data[f"{y}_bin"], bins = pd.qcut(data[y], q=n_bins, retbins=True)
    else:
        bins = np.linspace(
            percentiles[trim_quantiles[0]], percentiles[trim_quantiles[1]], 25
        )
        data[f"{y}_bin"] = pd.cut(data[y], bins=bins)
    data[f"{y}_bin_mid"] = data[f"{y}_bin"].apply(lambda x: x.mid).astype(float)
    return data, bins


# def get_ratios(data, bin_col):
#     counts = (
#         data.query("spine_group_id.notna()")
#         .groupby([f"{bin_col}_bin", "tag_detailed"], observed=True)["spine_group_id"]
#         .nunique()
#         .unstack()
#         .fillna(0)
#     )
#     return counts["multi_spine"] / counts.sum(axis=1)


def histplot(data, y, bins, ax):
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
        # linewidth=0,
        # kde=True,
        kde=False,
        common_norm=False,
        palette=COMPARTMENT_PALETTE_MUTED_HEX,
        legend=False,
    )
    ax.spines["bottom"].set_visible(False)
    ax.set(xticks=[], xlabel="Normalized\ndensity\n(spines)")


def lineplot(data, y, ax):
    sns.lineplot(
        data=data,
        y=y + "_bin_mid",
        x="is_multi",
        ax=ax,
        orient="y",
        color="black",
    )
    ax.set_xlabel("Proportion of \nspines with\nmultiple inputs")
    ax.set_xlim(0, ax.get_xlim()[1])


figsize = (4.51, 7.17)

base_data = (
    all_post_synapses.query('post_broad_type == "excitatory"')
    .query("tag == 'spine'")
    .copy()
)
# base_data = base_data.join(
#     skeleton_extras.select("synapse_id", "max_dist_to_tip")
#     .to_pandas()
#     .set_index("synapse_id")
# )


base_data["post_max_dist_to_tip_um"] = base_data["max_dist_to_tip"] / 1000.0
base_data["post_euc_distance_to_nuc_um"] = (
    base_data["post_euc_distance_to_nuc"] / 1000.0
)
base_data["post_path_distance_to_nuc_um"] = (
    base_data["post_path_distance_to_nuc"] / 1000.0
)

base_data["spine_group_id"] = base_data["spine_group_id"].astype("Int64")
cols = [
    "post_euc_distance_to_nuc_um",
    "post_path_distance_to_nuc_um",
    "transformed_um_y",
    "post_delta_depth",
    "post_max_dist_to_tip_um",
]
spine_base_data = base_data.groupby("spine_group_id")[cols].mean()

assert (
    base_data.groupby("spine_group_id")["spine_n_pre_pt_root_ids"].nunique().max() == 1
)

is_multi = base_data.groupby("spine_group_id")["spine_n_pre_pt_root_ids"].first() > 1
is_multi = is_multi.rename("is_multi")
tag_detailed = base_data.groupby("spine_group_id")["tag_detailed"].first()
post_cell_type = base_data.groupby("spine_group_id")["post_cell_type"].first()
base_data = pd.concat([spine_base_data, is_multi, tag_detailed, post_cell_type], axis=1)


bin_col = "is_multi"

y = "transformed_um_y"
data, bins = filter_and_bin(base_data, y)
fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
ax = axs[0]
histplot(data, y, bins, ax)
ax.invert_yaxis()
ax.set(ylabel="Synapse depth (um)")
ax = axs[1]
lineplot(data, y, ax)
save_matplotlib_figure(
    fig, "multi_spine_prop_by_depth", figure_out_path, pad_inches=0.2
)


y = "post_delta_depth"
data, bins = filter_and_bin(base_data, y, trim_quantiles=(0.001, 0.999))
fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
ax = axs[0]
histplot(data, y, bins, ax)
ax.set(
    ylabel="Height above postsynaptic soma (um)",
)
ax = axs[1]
lineplot(data, y, ax)
save_matplotlib_figure(
    fig, "multi_spine_prop_by_delta_depth", figure_out_path, pad_inches=0.2
)


y = "post_euc_distance_to_nuc_um"
data, bins = filter_and_bin(base_data, y, trim_quantiles=(0.000, 0.999))
bins = np.concatenate(([0], bins))
fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
ax = axs[0]
histplot(data, y, bins, ax)
ax.set(
    ylabel="Distance to postsynaptic soma (um)",
)
ax = axs[1]
lineplot(data, y, ax)
save_matplotlib_figure(
    fig, "multi_spine_prop_by_euc_distance_to_nuc", figure_out_path, pad_inches=0.2
)

y = "post_max_dist_to_tip_um"
data, bins = filter_and_bin(base_data, y, use_quantiles=True)
fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
ax = axs[0]
histplot(data, y, bins, ax)
ax.set(
    ylabel="Distance to main tip arbor tip (um)",
)
ax = axs[1]
lineplot(data, y, ax)
save_matplotlib_figure(
    fig, "multi_spine_prop_by_distance_to_tip", figure_out_path, pad_inches=0.2
)


# base_data = (
#     all_post_synapses.query('post_broad_type == "excitatory"')
#     .query("tag == 'spine'")
#     .copy()
# )
# base_data["post_euc_distance_to_nuc_um"] = (
#     base_data["post_euc_distance_to_nuc"] / 1000.0
# )
# base_data.query("post_euc_distance_to_nuc_um < 10")[["post_pt_root_id", "tag_detailed"]]

# base_data['post_delta_depth']

# skeleton_extras.filter(pl.col("compartment").is_in(["dendrite"])).select(
#     pl.col("segment_id").is_null().mean()
# )


groups = ["23P", "4P", "5P-IT", "5P-ET", "6P-IT", "6P-CT"]
base_data = base_data.query("post_cell_type.isin(@groups)")
# base_data["is_multi"] = base_data["spine_n_pre_pt_root_ids"] > 1
n_groups = len(groups)
# n_groups = base_data["post_cell_type"].nunique()

ys = [
    "transformed_um_y",
    "post_delta_depth",
    "post_euc_distance_to_nuc_um",
    "post_path_distance_to_nuc_um",
]
names = {
    "transformed_um_y": "Synapse depth (um)",
    "post_delta_depth": "Height above postsynaptic soma (um)",
    "post_euc_distance_to_nuc_um": "Euc. distance to postsynaptic soma (um)",
    "post_path_distance_to_nuc_um": "Path distance to postsynaptic soma (um)",
}

layer_bounds = [91.80615154, 261.21908419, 391.8631847, 537.04973966, 753.58049474]

for y in ys:
    fig, axs = plt.subplots(
        1,
        n_groups + 1,
        figsize=(14, 6),
        dpi=300,
        layout="constrained",
        sharey=True,
        sharex=False,
    )

    ax = axs[0]
    data, bins = filter_and_bin(base_data, y)
    # histplot(data, y, bins, ax)

    lineplot(data, y, ax)
    ax.set_title("All excitatory")

    for i, (cell_type, group) in enumerate(
        base_data.groupby("post_cell_type", observed=True)
    ):
        ax = axs[i + 1]
        data, bins = filter_and_bin(group, y)

        # histplot(data, y, bins, ax)
        lineplot(data, y, ax)
        ax.set(title=remapper(cell_type))

    for ax in axs.flat:
        # ax.spines["bottom"].set_visible(False)
        ax.set(xlabel="", xlim=(0, 0.15))
        if y == "transformed_um_y":
            for bound in layer_bounds:
                ax.axhline(
                    bound,
                    color="gray",
                    linestyle="-",
                    linewidth=0.8,
                    zorder=-1,
                    alpha=0.5,
                )

    ax = axs[0]
    ax.set(ylabel=names[y])

    ax = axs[3]
    ax.set(xlabel="Proportion of \nspines with\nmultiple inputs")

    if y == "transformed_um_y":
        ax.invert_yaxis()

    save_matplotlib_figure(
        fig,
        f"multi_spine_prop_by_{y}_by_post_cell_type",
        figure_out_path,
    )

# %%

base_data = (
    all_pre_synapses.query('post_broad_type == "excitatory" and pre_cell_type == "BC"')
    .query("tag == 'spine'")
    .copy()
)

base_data["post_euc_distance_to_nuc_um"] = (
    base_data["post_euc_distance_to_nuc"] / 1000.0
)
base_data["post_path_distance_to_nuc_um"] = (
    base_data["post_path_distance_to_nuc"] / 1000.0
)

base_data["spine_group_id"] = base_data["spine_group_id"].astype("Int64")
cols = [
    "post_euc_distance_to_nuc_um",
    "post_path_distance_to_nuc_um",
    "transformed_um_y",
    # "post_delta_depth",
    # "post_max_dist_to_tip_um",
]
spine_base_data = base_data.groupby("spine_group_id")[cols].mean()

assert (
    base_data.groupby("spine_group_id")["spine_n_pre_pt_root_ids"].nunique().max() == 1
)

is_multi = base_data.groupby("spine_group_id")["spine_n_pre_pt_root_ids"].first() > 1
is_multi = is_multi.rename("is_multi")
tag_detailed = base_data.groupby("spine_group_id")["tag_detailed"].first()
post_cell_type = base_data.groupby("spine_group_id")["post_cell_type"].first()
base_data = pd.concat([spine_base_data, is_multi, tag_detailed, post_cell_type], axis=1)
base_data = base_data.query("post_euc_distance_to_nuc_um <= 100")
bin_col = "is_multi"
y = "post_euc_distance_to_nuc_um"

data, bins = filter_and_bin(
    base_data, y, use_quantiles=True, trim_quantiles=None, n_bins=50
)
bins = np.concatenate(([0], bins))
fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
ax = axs[0]
histplot(data, y, bins, ax)
ax.set(ylabel="Euclidean distance to postsynaptic nucleus (um)")


ax = axs[1]
lineplot(data, y, ax)


save_matplotlib_figure(
    fig, "bc_multi_spine_prop_by_euc_distance_to_nuc", figure_out_path, pad_inches=0.2
)

# %%
hue_order = ["BC", "MC"]
pre_broad_type = ["inhibitory"]

# hue_order = ["TH", "23P", "4P"]
# pre_broad_type = ["excitatory", "thalamic"]

y = "post_euc_distance_to_nuc"
base_data = (
    all_pre_synapses.query('post_broad_type == "excitatory"')
    .query("tag == 'spine'")
    .query("tag_detailed == 'single_spine' or tag_detailed == 'multi_spine'")
    .query("pre_axon_cleaned")
    .query("pre_broad_type.isin(@pre_broad_type) and pre_cell_type.isin(@hue_order)")
    .query(f"{y} < 300_000")
    .copy()
)
base_data[f"{y}_um"] = base_data[y] / 1000

bin_data = []


for cell_type, cell_type_data in base_data.groupby("pre_cell_type", observed=True):
    cell_type_binned_data = pd.qcut(cell_type_data[f"{y}_um"], q=25)
    cell_type_binned_data_rep = cell_type_binned_data.apply(lambda x: x.mid).astype(
        float
    )
    bin_data.append(cell_type_binned_data_rep)

binned_data_rep = pd.concat(bin_data, axis=0)

base_data[f"{y}_um_bin_rep"] = binned_data_rep

base_data["is_multi"] = base_data["tag_detailed"] == "multi_spine"


fig, axs = plt.subplots(1, 2, figsize=(4.56, 7.17), sharey=True, layout="constrained")

ax = axs[0]
sns.histplot(
    data=base_data.query("pre_cell_type == 'BC'"),
    y=f"{y}_um",
    # hue="pre_broad_type",
    # hue_order=hue_order,
    ax=ax,
    element="poly",
    stat="density",
    # common_norm=False,
    legend=False,
    color=CELL_TYPE_PALETTE["inhibitory"],
    alpha=0.1,
)
sns.histplot(
    data=base_data.query("pre_cell_type == 'MC'"),
    y=f"{y}_um",
    # hue="pre_broad_type",
    # hue_order=hue_order,
    ax=ax,
    element="poly",
    stat="density",
    # common_norm=False,
    legend=False,
    color=CELL_TYPE_PALETTE["inhibitory"],
    linestyle="--",
    # fill=False,
    alpha=0.1,
    linewidth=2,
)
ax.spines["bottom"].set_visible(False)
ax.set(
    xlabel="Normalized\ndensity\n(spines)",
    ylabel="Distance to postsynaptic soma (um)",
    xticks=[],
)
text = ax.text(
    0.2,
    0.65,
    "Martinotti\n(MC)",
    transform=ax.transAxes,
    color=CELL_TYPE_PALETTE["inhibitory"],
)
text = ax.text(
    0.4,
    0.04,
    "Basket\n(BC)",
    transform=ax.transAxes,
    color=CELL_TYPE_PALETTE["inhibitory"],
)
text.set_in_layout(False)

minipal = dict(zip(hue_order, len(hue_order) * [CELL_TYPE_PALETTE["inhibitory"]]))
ax = axs[1]
sns.lineplot(
    data=base_data,
    hue_order=hue_order,
    style="pre_cell_type",
    style_order=hue_order,
    y=f"{y}_um_bin_rep",
    x="is_multi",
    orient="y",
    hue="pre_cell_type",
    ax=ax,
    palette=minipal,
    legend=False,
)
ax.set(
    xlabel="Proportion of\nspines with\nmultiple inputs",
    xlim=(0, None),
    xticks=[0.0, 0.4, 0.8],
    ylim=(None, 200),
)

save_matplotlib_figure(
    fig,
    "bc_vs_mc_multi_spine_prop_by_euc_distance_to_nuc",
    figure_out_path,
    pad_inches=0.2,
)


# %%

pre_broad_type = ["excitatory", "thalamic"]
hue_order = ["excitatory", "thalamic"]

y = "post_euc_distance_to_nuc"
base_data = (
    all_pre_synapses.query('post_broad_type == "excitatory"')
    .query("tag == 'spine'")
    .query("tag_detailed == 'single_spine' or tag_detailed == 'multi_spine'")
    .query("pre_axon_cleaned")
    .query("pre_broad_type.isin(@pre_broad_type)")
    .query(f"{y} < 200_000")
    .copy()
)
base_data[f"{y}_um"] = base_data[y] / 1000

bin_data = []


for cell_type, cell_type_data in base_data.groupby("pre_broad_type", observed=True):
    cell_type_binned_data = pd.qcut(cell_type_data[f"{y}_um"], q=20)
    cell_type_binned_data_rep = cell_type_binned_data.apply(lambda x: x.mid).astype(
        float
    )
    bin_data.append(cell_type_binned_data_rep)

binned_data_rep = pd.concat(bin_data, axis=0)

base_data[f"{y}_um_bin_rep"] = binned_data_rep

base_data["is_multi"] = base_data["tag_detailed"] == "multi_spine"


fig, axs = plt.subplots(1, 2, figsize=(4.56, 7.17), sharey=True, layout="constrained")
ax = axs[0]
sns.histplot(
    data=base_data,
    y=f"{y}_um",
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

ax = axs[1]
sns.lineplot(
    data=base_data,
    hue_order=hue_order,
    y=f"{y}_um_bin_rep",
    x="is_multi",
    hue="pre_broad_type",
    orient="y",
    ax=ax,
    palette=CELL_TYPE_PALETTE,
    legend=False,
)
ax.set(xlabel="Proportion of\nspines with\nmultiple inputs", xlim=(0, None))

save_matplotlib_figure(
    fig,
    "multi_spine_prop_by_euc_distance_to_nuc_by_exc_broad_type",
    figure_out_path,
    pad_inches=0.2,
)


# %%
# import statsmodels.api as sm
# from sklearn.preprocessing import QuantileTransformer, StandardScaler  # noqa

# factors = ["log_spine_size_nm3", "post_euc_distance_to_nuc", "is_thalamic"]
# base_data["is_thalamic"] = (base_data["pre_broad_type"] == "thalamic").astype(int)
# base_data["log_spine_size_nm3"] = np.log(base_data["spine_size_nm3"])
# fit_data = base_data[factors + ["is_multi"]].dropna()
# fit_data["post_euc_distance_to_nuc"] = fit_data["post_euc_distance_to_nuc"] / 1000
# endog = fit_data["is_multi"].astype(int)
# exog = fit_data[factors]

# normalizer = QuantileTransformer()
# exog = normalizer.fit_transform(exog)

# model = sm.GLM(endog, exog, family=sm.families.Gaussian())
# # model = sm.Logit(endog, exog)
# res = model.fit()
# print(res.summary())


# %%

fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, layout="constrained")
ax = axs[0]
sns.lineplot(
    data=base_data,
    hue_order=hue_order,
    x=f"{y}_um_bin_rep",
    y="spine_size_nm3",
    hue="pre_cell_type",
    ax=ax,
    palette=CELL_TYPE_PALETTE,
)

ax = axs[1]
sns.histplot(
    data=base_data,
    x=f"{y}_um",
    # y="is_multi",
    hue="pre_cell_type",
    hue_order=hue_order,
    ax=ax,
    palette=CELL_TYPE_PALETTE,
    element="poly",
    stat="density",
    common_norm=False,
)


# %% GET PRE CONNECTIONS TABLE

all_pre_connections = (
    all_pre_synapses.groupby(["pre_pt_root_id", "post_pt_root_id"])
    .size()
    .rename("n_synapses_in_connection")
)
all_pre_connections

all_pre_synapses = all_pre_synapses.drop(
    columns="n_synapses_in_connection", errors="ignore"
).join(all_pre_connections, on=["pre_pt_root_id", "post_pt_root_id"])


# %% PLOT P MULTI AS A FUNCTION OF NUMBER OF SYNAPSES IN CONNECTION
pre_cell_type = "MC"
post_broad_type = "excitatory"
data = (
    all_pre_synapses.query(
        "pre_cell_type == @pre_cell_type and post_broad_type == @post_broad_type"
    )
    # .query("tag == 'spine'")
    .groupby("n_synapses_in_connection")["tag_detailed"]
    .value_counts(normalize=True)
    .reset_index()
)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(
    data=data,
    x="n_synapses_in_connection",
    y="proportion",
    hue="tag_detailed",
    hue_order=["soma", "shaft", "single_spine", "multi_spine"],
    # hue_order=["single_spine", "multi_spine"],
    marker="o",
    ax=ax,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
)
ax.set(
    xlim=(0.99, 10.1),
    xlabel="Number of synapses in connection",
    ylabel="Proportion",
    title=f"{pre_cell_type} to {post_broad_type}",
)

# %%
cell_types = ["23P", "4P", "5P-IT", "6P-IT", "5P-ET", "6P-CT"]

layer_23P_cells = (
    column_cell_info.query("cell_type.isin(@cell_types)")
    .sort_values("post_p_spine_site_is_multi", ascending=False)
    .index
)

query_synapses = all_pre_synapses.query("post_pt_root_id.isin(@layer_23P_cells)").query(
    "pre_broad_type == 'inhibitory'"
)

pre_conn_vectors = (
    query_synapses.groupby(["pre_pt_root_id", "post_pt_root_id"])
    .size()
    .reset_index(name="n_synapses_in_connection")
    .pivot(
        index="post_pt_root_id",
        columns="pre_pt_root_id",
        values="n_synapses_in_connection",
    )
    .fillna(0)
    .copy()
)
print(pre_conn_vectors.shape)

layer_23P_cells = pre_conn_vectors.index.copy()
print(layer_23P_cells.shape)

X = pre_conn_vectors.values
print(X.shape)


conn_distances = pairwise_distances(X, metric="cosine")
print(conn_distances.shape)

Z = linkage(squareform(conn_distances), method="average")

feature_name = "cell_type"

feature = column_cell_info.loc[layer_23P_cells, feature_name].values

if feature_name == "mtype":
    color_items = sns.color_palette("tab10")
    palette = dict(zip(np.unique(feature), color_items))
    colors = [palette[x] for x in feature]
elif feature_name == "post_p_spine_site_is_multi":
    cmap = cm.get_cmap("Reds")
    colors = [cmap(x / 0.25) for x in feature]
elif feature_name == "cell_type":
    color_items = sns.color_palette("tab10")
    palette = dict(zip(np.unique(feature), color_items))
    colors = [palette[x] for x in feature]
    # colors = [CELL_TYPE_PALETTE[x] for x in feature]


sns.clustermap(
    conn_distances,
    row_linkage=Z,
    col_linkage=Z,
    row_colors=colors,
    col_colors=colors,
    xticklabels=False,
    yticklabels=False,
)

print(np.unique(feature))
color_items


# %%
# from sklearn.manifold import MDS, ClassicalMDS  # noqa

# projector = MDS(
#     dissimilarity="precomputed",
#     max_iter=1000,
#     init="classical_mds",
#     eps=1e-7,
#     n_components=2,
#     n_init=1,
# )

# # projector = ClassicalMDS(metric='precomputed')
# # projector = UMAP(metric='precomputed')

# data = column_cell_info.loc[layer_23P_cells].copy()

# X = projector.fit_transform(conn_distances)
# print(X.shape)
# print(projector.n_components)

# data["MDS0"] = X[:, 0]
# data["MDS1"] = X[:, 1]

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# sns.scatterplot(
#     data=data,
#     x="MDS0",
#     y="MDS1",
#     hue="cell_type",
#     ax=ax,
#     palette="tab10",
# )

# sns.move_legend(ax, bbox_to_anchor=(1, 1), loc="upper left")


# %%

root_id = (
    column_cell_info.query("cell_type == '23P'")
    .sort_values("post_p_spine_site_is_multi", ascending=False)
    .index[0]
)

syns = all_post_synapses.query("post_pt_root_id == @root_id").query(
    "tag_detailed=='multi_spine'"
)

multis = syns.groupby("spine_group_id").agg(
    {
        "ctr_pt_position_x": "mean",
        "ctr_pt_position_y": "mean",
        "ctr_pt_position_z": "mean",
        "pre_pt_root_id": "unique",
    }
)
multis = multis.sort_values("ctr_pt_position_y")

# from caveclient import CAVEclient
# from nglui.statebuilder import ViewerState

# client = CAVEclient("minnie65_phase3_v1")
# vs = ViewerState(client=client)
# vs.add_layers_from_client()
# vs.add_segments([root_id])
# vs.add_segmentation_layer(vs.layers[-1].source, name="pre")
# vs.add_points(
#     data=multis,
#     point_column="ctr_pt_position",
#     segment_column="pre_pt_root_id",
#     linked_segmentation="pre",
# )
# vs.to_browser(browser="firefox")

# %% LOOK AT P MULTI AS A FUNCTION OF SPATIAL FACTORS BUT FOR INDIVIDUAL CELLS
# data = all_pre_synapses.query('post_broad_type == "excitatory"').query("tag == 'spine'")
# post_cell_type = "23P"
data = all_post_synapses.query(
    "post_broad_type == 'excitatory' and tag == 'spine' and post_compartment != 'axon'"
).copy()
data["post_delta_depth"] = (
    cell_info.loc[data["post_pt_root_id"], "pt_position_um_y"].values
    - data["transformed_um_y"].values
)
data["post_radial_distance_to_nuc"] = np.linalg.norm(
    data[["transformed_um_x", "transformed_um_z"]].values
    - cell_info.loc[
        data["post_pt_root_id"], ["pt_position_um_x", "pt_position_um_z"]
    ].values,
    axis=1,
)
data["is_multi"] = data["tag_detailed"] == "multi_spine"


bins = np.geomspace(1e6, 1e9, 100)

# data["spine_size_bin"] = pd.cut(data["spine_size_nm3"], bins=bins)
data["spine_size_bin"] = pd.qcut(data["spine_size_nm3"], q=50)
data["spine_size_bin_mid"] = data["spine_size_bin"].apply(lambda x: x.mid)
data["spine_size_bin_lower"] = data["spine_size_bin"].apply(lambda x: x.left)
data["spine_size_bin_upper"] = data["spine_size_bin"].apply(lambda x: x.right)


fig, axs = plt.subplots(
    4, 20, figsize=(20, 10), sharex=False, sharey=True, layout="constrained"
)


y = "post_path_distance_to_nuc"
y = "post_delta_depth"
# y = 'post_radial_distance_to_nuc'
for i, (root_id, subdata) in enumerate(data.groupby("post_pt_root_id")):
    if i >= len(axs.flat):
        break
    ax = axs.flat[i]
    sns.histplot(
        data=subdata,
        y=y,
        ax=ax,
        # log_scale=True,
        # bins=30,
        bins=30,
        hue="tag_detailed",
        element="step",
        common_norm=False,
        stat="proportion",
        palette=COMPARTMENT_PALETTE_MUTED_HEX,
        hue_order=["single_spine", "multi_spine"],
        legend=False,
    )
    ax.set(
        xlabel="",
        xticks=[],
        title=f"{cell_info.loc[root_id, 'post_p_spine_site_is_multi'].item():.2f}",
    )

# %% EXTRACT SPATIAL FEATURES FOR 4P CELLS
feature_names = [
    "post_path_distance_to_nuc",
    "post_radial_distance_to_nuc",
    "post_delta_depth",
    "post_euc_distance_to_nuc",
]
feature_datas = []
for i, (root_id, subdata) in enumerate(
    data.query("post_cell_type == '4P'").groupby("post_pt_root_id")
):
    features = {}
    for feature_name in feature_names:
        # get quantiles for that feature for that neuron
        quantiles = np.quantile(
            subdata[feature_name],
            [
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
            ],
        )
        for q, val in zip(
            [10, 25, 50, 75, 90],
            quantiles,
        ):
            features[f"{feature_name}_q{q}"] = val
    features["post_pt_root_id"] = root_id
    outcome = np.log(subdata["is_multi"].mean())
    features["p_multi"] = outcome
    feature_datas.append(features)

features = pd.DataFrame(feature_datas).set_index("post_pt_root_id")

y = features["p_multi"]
features = features.drop(columns="p_multi")

# %% BUILD LINEAR REGRESSION MODEL FOR P_MULTI PREDICTION

# model = LinearRegression()

pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", RidgeCV())])

X = features.sample(frac=1.0)
scores = cross_val_score(pipeline, X, y, cv=10)

print(scores)

pipeline.fit(X, y)
y_pred = pipeline.predict(X)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x=y, y=y_pred, ax=ax)

# %% ANALYZE SPINE SIZE BIN STATISTICS
out = data.groupby("spine_size_bin_mid")["is_multi"].agg(["mean", "std", "count"])
out

# %% ANALYZE SIZE OF 23P TO 23P EXCITATORY CONNECTIONS
data = all_pre_synapses.query(
    'pre_broad_type == "excitatory" and post_broad_type == "excitatory"'
).query("tag == 'spine'")
pre_cell_type = "23P"
post_cell_type = "23P"
data = all_pre_synapses.query(
    "pre_cell_type == @pre_cell_type and post_cell_type == @post_cell_type"
)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(
    data=data,
    x="spine_size_nm3",
    # x="size",
    hue="tag_detailed",
    hue_order=["single_spine", "multi_spine"],
    ax=ax,
    log_scale=True,
    bins=50,
    common_norm=False,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    element="step",
    stat="density",
)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=data,
    y="spine_size_nm3",
    x="size",
    hue="tag_detailed",
    hue_order=["single_spine", "multi_spine"],
    ax=ax,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    s=3,
    linewidth=0,
    alpha=0.2,
)
ax.set(yscale="log", xscale="log")

# %% ANALYZE DOUBLE SYNAPSE CONNECTIONS
pre_broad_type = "excitatory"
post_broad_type = "excitatory"
pre_cell_type = None
post_cell_type = None
data = all_pre_synapses
data = data.query("tag == 'spine' and spine_size_nm3.notna()")
data = data.query(
    "(pre_broad_type == @pre_broad_type) and (post_broad_type == @post_broad_type)"
)
if (pre_cell_type is not None) and (post_cell_type is not None):
    data = data.query(
        "(pre_cell_type == @pre_cell_type) and (post_cell_type == @post_cell_type)"
    )
data = data.query("n_synapses_in_connection == 2").query("tag_detailed.notna()").copy()
data["tag_detailed"] = data["tag_detailed"].astype("str")

data.groupby(["pre_pt_root_id", "post_pt_root_id"])[
    "tag_detailed"
].value_counts().unstack(fill_value=0).value_counts()


connection_codes = (
    data.groupby(["pre_pt_root_id", "post_pt_root_id"])["tag_detailed"]
    .value_counts()
    .unstack(fill_value=0)
)


double_multis = connection_codes.query("multi_spine == 2 and single_spine == 0").index
one_ones = connection_codes.query("single_spine == 1 and multi_spine == 1").index
double_singles = connection_codes.query("single_spine == 2 and multi_spine == 0").index


double_multi_synapses = (
    data.reset_index()
    .set_index(["pre_pt_root_id", "post_pt_root_id"])
    .loc[double_multis]
    .reset_index()
).query("spine_size_nm3.notna()")
one_one_synapses = (
    data.reset_index()
    .set_index(["pre_pt_root_id", "post_pt_root_id"])
    .loc[one_ones]
    .reset_index()
).query("spine_size_nm3.notna()")
double_single_synapses = (
    data.reset_index()
    .set_index(["pre_pt_root_id", "post_pt_root_id"])
    .loc[double_singles]
    .reset_index()
).query("spine_size_nm3.notna()")

one_one_synapses.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()


def compare_values(syns, x):
    x1 = syns.iloc[::2][x].values
    x2 = syns.iloc[1::2][x].values
    return np.block([[x1, x2], [x2, x1]]).T


fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

x = "spine_size_nm3"
# x = 'size'
if x == "size":
    xlabel = "Cleft 1 size (vx)"
    ylabel = "Cleft 2 size (vx)"
elif x == "spine_size_nm3":
    xlabel = "Cleft 1 spine volume (nm³)"
    ylabel = "Cleft 2 spine volume (nm³)"

scatter_kws = dict(alpha=0.3, s=5, linewidth=0.1)

PLOT_TYPE = "scatter"


def comparison_plot(synapses, x, ax, title, color):
    sizes = compare_values(synapses, x)
    sizes = sizes[sizes[:, 0] != sizes[:, 1]]  # remove equal sizes
    if PLOT_TYPE == "scatter":
        sns.scatterplot(
            x=sizes[:, 0],
            y=sizes[:, 1],
            ax=ax,
            color=color,
            zorder=1,
            **scatter_kws,
        )
        ax.autoscale(False)
        sns.kdeplot(
            x=sizes[:, 0],
            y=sizes[:, 1],
            ax=ax,
            color="black",
            levels=4,
            linewidths=0.5,
            log_scale=True,
            zorder=2,
        )
        ax.autoscale(True)
    elif PLOT_TYPE == "hist":
        sns.histplot(
            x=sizes[:, 0],
            y=sizes[:, 1],
            log_scale=True,
            ax=ax,
            color=color,
            bins=25,
            # bins=50,
            # pthresh=0.01,
            cmap="Greys",
        )
    elif PLOT_TYPE == "both":
        sns.histplot(
            x=sizes[:, 0],
            y=sizes[:, 1],
            log_scale=True,
            ax=ax,
            color=color,
            # bins=20,
            # bins=50,
            pthresh=None,
            cmap="Greys",
            zorder=2,
            alpha=0.9,
        )
        sns.scatterplot(
            x=sizes[:, 0],
            y=sizes[:, 1],
            ax=ax,
            color="black",
            zorder=1,
            alpha=0.7,
            s=1,
        )

    n_sizes = sizes.shape[0]
    n_pairs = n_sizes // 2
    stat, _ = pearsonr(np.log(sizes[:n_pairs, 0]), np.log(sizes[:n_pairs, 1]))
    ax.text(0.1, 0.9, f"r={stat:.2f}", transform=ax.transAxes)
    title += f" (n={n_pairs})"
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set(xscale="log", yscale="log")
    ax.set(xlim=ax.get_ylim())
    ax.set(xlim=(1e7, 6e8), ylim=(1e7, 6e8))
    return sizes


ax = axs[0]
comparison_plot(
    double_single_synapses,
    x,
    ax,
    "Single-single",
    COMPARTMENT_PALETTE_MUTED_HEX["single_spine"],
)

ax = axs[1]
comparison_plot(
    one_one_synapses,
    x,
    ax,
    "Single-multi",
    "tab:blue",
)

comparison_plot(
    double_multi_synapses,
    x,
    axs[2],
    "Multi-multi",
    COMPARTMENT_PALETTE_MUTED_HEX["multi_spine"],
)
suptitle = ""
if (pre_cell_type is not None) and (post_cell_type is not None):
    suptitle = (
        f"{pre_cell_type}" + r"$\rightarrow$" + f"{post_cell_type} with 2 synapses"
    )
else:
    suptitle = r"E$\rightarrow$E connections with 2 synapses"

fig.suptitle(suptitle, y=1.05)

# %% LOAD AND PREPARE SPINE TABLE DATA

spine_table = pl.scan_parquet(str(DATA_PATH / "filtered_spine_info.parquet"))
# drop all columns that have "hks" in the name
spine_table = spine_table.drop(
    pl.selectors.contains("min_"), pl.selectors.contains("max_")
)
spine_table = spine_table.collect().to_pandas()

spine_table["post_broad_type"] = spine_table["post_pt_root_id"].map(
    column_cell_info["broad_type"]
)
spine_table["post_cell_type"] = spine_table["post_pt_root_id"].map(
    column_cell_info["cell_type"]
)
spine_table["tag_detailed"] = "single_spine"
spine_table.loc[spine_table[spine_table["is_multi"]].index, "tag_detailed"] = (
    "multi_spine"
)

if True:
    # import joblib

    # model = joblib.load(
    #     "/Users/ben.pedigo/code/meshrep/meshrep/models/true_multi_rf.joblib"
    # )
    # predictions = model.predict(spine_table[model.feature_names_in_])
    # spine_table = spine_table.drop(
    #     [col for col in spine_table.columns if "hks" in col], axis=1, errors="ignore"
    # )

    # spine_table["is_multi_prediction"] = predictions
    spine_table["tag_detailed_original"] = "single_spine"
    spine_table.loc[
        spine_table[spine_table["n_pre_pt_root_ids"] > 1].index, "tag_detailed_original"
    ] = "multi_spine"

    spine_table["tag_detailed_corrected"] = "single_spine"
    spine_table.loc[
        spine_table[spine_table["is_multi"]].index, "tag_detailed_corrected"
    ] = "multi_spine"

    p_multi_before = (
        spine_table.groupby("post_pt_root_id")[["tag_detailed_original"]]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)["multi_spine"]
    )

    p_multi_after = (
        spine_table.groupby("post_pt_root_id")[["tag_detailed_corrected"]]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)["multi_spine"]
    )

    data = column_cell_info.copy()
    data["p_multi_before"] = p_multi_before
    data["p_multi_after"] = p_multi_after

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.scatterplot(
        data=data.query("broad_type == 'excitatory'"),
        x="p_multi_before",
        y="p_multi_after",
        ax=ax,
        s=2,
    )
    ax.plot([0, 0.2], [0, 0.2], color="black", linestyle="--")

# %%
post_spines = spine_table.query(
    "~size_nm3.isna() and post_broad_type == 'excitatory'"
).copy()
post_spines["spine_size_bin"] = pd.qcut(post_spines["size_nm3"], q=50)
post_spines["spine_size_bin_mid"] = (
    post_spines["spine_size_bin"].map(lambda x: x.mid).astype(float)
)

# %% SPINE SIZE VS PROBABILITY OF MULTI, FOR ALL EXCITATORY

# data["spine_size_bin"] = pd.cut(data["spine_size_nm3"], bins=bins)
# post_spines["spine_size_bin"] = pd.qcut(data["spine_size_nm3"], q=50)

common_norm = False
if not common_norm:
    ylabel = "Normalized\ndensity"
else:
    ylabel = "Density"
x = "size_nm3"
fig, axs = plt.subplots(2, 1, figsize=(9.15, 7.17), sharex=True)
ax = axs[0]
sns.histplot(
    data=post_spines,
    x=x,
    hue="tag_detailed",
    hue_order=["single_spine", "multi_spine"],
    ax=ax,
    log_scale=False,
    bins=bins,
    common_norm=common_norm,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    element="poly",
    stat="proportion",
    legend=False,
)
ax.set(xlim=(5e6, 1e9))
ax.set(xscale="log", ylabel=ylabel, yticks=[])
ax.spines[["left"]].set_visible(False)
ax.text(
    0.05,
    0.5,
    "Single-input\nspines",
    transform=ax.transAxes,
    color=COMPARTMENT_PALETTE_MUTED_HEX["single_spine"],
)
ax.text(
    0.8,
    0.5,
    "Multi-input\nspines",
    transform=ax.transAxes,
    color=COMPARTMENT_PALETTE_MUTED_HEX["multi_spine"],
)

ax = axs[1]
sns.lineplot(
    data=post_spines,
    x="spine_size_bin_mid",
    y="is_multi",
    color="black",
)
ax.set(ylabel="Proportion of spines\nwith multiple inputs", xlabel="Spine volume (nm³)")

save_matplotlib_figure(fig, "spine_size_vs_p_multi", figure_out_path)

# %% SPINE SIZE VS PROBABILITY OF MULTI, FOR ALL EXCITATORY, ROTATED

common_norm = False
# if not common_norm:
#     ylabel = "Normalized\ndensity"
# else:
#     ylabel = "Density"
x = "size_nm3"
fig, axs = plt.subplots(1, 2, figsize=(4.58, 7.17), sharey=True)
ax = axs[0]
sns.histplot(
    data=post_spines,
    y=x,
    hue="tag_detailed",
    hue_order=["single_spine", "multi_spine"],
    ax=ax,
    log_scale=False,
    bins=bins,
    common_norm=common_norm,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    element="poly",
    stat="proportion",
    legend=False,
)
ax.set(ylim=(5e6, 1e9))
ax.set(
    yscale="log",
    xticks=[],
    ylabel="Spine volume (nm³)",
    xlabel="Normalized\ndensity\n(spines)",
)
ax.spines[["bottom"]].set_visible(False)
ax.text(
    0.3,
    0.1,
    "Single-\ninput\nspines",
    transform=ax.transAxes,
    color=COMPARTMENT_PALETTE_MUTED_HEX["single_spine"],
)
ax.text(
    0.3,
    0.85,
    "Multi-\ninput\nspines",
    transform=ax.transAxes,
    color=COMPARTMENT_PALETTE_MUTED_HEX["multi_spine"],
)

ax = axs[1]
sns.lineplot(
    data=post_spines,
    y="spine_size_bin_mid",
    x="is_multi",
    orient="y",
    color="black",
    errorbar=None,
)
ax.set(xlabel="Proportion of\nspines with\nmultiple inputs")

save_matplotlib_figure(
    fig, "spine_size_vs_p_multi_rotated", figure_out_path, pad_inches=0.2
)


# %% SPINE SIZE VS PROBABILITY OF MULTI, FOR INDIVIDUAL CELLS

cell_type = "23P"


new_data = post_spines.query("post_cell_type == '23P'").copy()
new_data["spine_size_bin"] = pd.qcut(new_data["size_nm3"], q=4)
new_data["spine_size_bin_mid"] = new_data["spine_size_bin"].apply(lambda x: x.mid)


temp_palette = dict(
    zip(
        new_data["post_pt_root_id"].unique(),
        new_data["post_pt_root_id"].nunique() * ["#000000"],
    )
)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.lineplot(
    data=new_data,
    hue="post_pt_root_id",
    x="spine_size_bin_mid",
    y="is_multi",
    color="black",
    palette=temp_palette,
    legend=False,
    alpha=0.2,
    linewidth=0.8,
    errorbar=None,
)
ax.set(xscale="log")
ax.set(ylabel="Proportion of spines\nwith multiple inputs", xlabel="Spine volume (nm³)")

save_matplotlib_figure(
    fig, f"p_multi_by_size_by_neuron_cell_type={cell_type}", figure_out_path
)


fig, ax = plt.subplots(1, 1, figsize=(4, 6))
sns.lineplot(
    data=new_data,
    hue="post_pt_root_id",
    y="spine_size_bin_mid",
    x="is_multi",
    color="black",
    palette=temp_palette,
    legend=False,
    alpha=0.2,
    linewidth=0.8,
    errorbar=None,
    orient="y",
)
ax.set(yscale="log")
ax.set(xlabel="Proportion of spines\nwith multiple inputs", ylabel="Spine volume (nm³)")

save_matplotlib_figure(
    fig, f"p_multi_by_size_by_neuron_cell_type={cell_type}_rotated", figure_out_path
)


# %% ANALYZE SPINE SIZE DISTRIBUTION BY INPUT TYPE

all_pre_synapses.groupby(["pre_broad_type"])["spine_size_nm3"].mean()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.histplot(
    data=all_pre_synapses,
    hue="pre_broad_type",
    log_scale=True,
    x="spine_size_nm3",
    hue_order=["thalamic", "excitatory", "inhibitory"],
    # x = 'size',
    # hue_order=['thalamic', 'excitatory', 'inhibitory'],
    bins=50,
    stat="density",
    ax=ax,
    common_norm=False,
    palette=CELL_TYPE_PALETTE,
    element="poly",
)
ax.set(xlabel="Spine volume (nm³)", xlim=(3e6, 1e9))
sns.move_legend(ax, "upper left", title="Input type")
save_matplotlib_figure(fig, "spine_size_by_input", figure_out_path)


# %% MULTI-SPINE PROBABILITY BY SIZE AND PRESYNAPTIC TYPE

exc_synapses = all_pre_synapses.query(
    "pre_broad_type == 'excitatory' and tag == 'spine'"
).copy()
exc_spines_index = exc_synapses["spine_group_id"].dropna().unique().astype(int)

th_synapses = all_pre_synapses.query(
    "pre_broad_type == 'thalamic' and tag == 'spine'"
).copy()
th_spines_index = th_synapses["spine_group_id"].dropna().unique().astype(int)

x = "size_nm3"
low = 0.001
high = 1.0 - low
quantiles = post_spines[x].quantile([low, high])
bins = np.geomspace(quantiles[low], quantiles[high], 20)


def get_p_multi(index=None):
    post_spines = (
        spine_table.query("~size_nm3.isna() and post_broad_type == 'excitatory'")
        .copy()
        .set_index("group_id")
    )
    if index is not None:
        index = pd.Index(index)
        print(len(index))
        index = index.intersection(post_spines.index)
        print(len(index))
        print("here")
        post_spines = post_spines.loc[index]
        print(len(post_spines))

    post_spines["spine_size_bin"] = pd.cut(post_spines["size_nm3"], bins=bins)
    print(post_spines["size_nm3"].mean())

    p_multi = post_spines.groupby("spine_size_bin")["is_multi"].mean()
    p_multi = p_multi.reset_index()
    p_multi["spine_size_bin_mid"] = p_multi["spine_size_bin"].map(lambda x: x.mid)

    return p_multi, post_spines


def get_spine_subset(index):
    post_spines = (
        spine_table.query(f"~{x}.isna() and post_broad_type == 'excitatory'")
        .copy()
        .set_index("group_id")
    )
    # post_spines["spine_size_bin"] = pd.qcut(post_spines[x], 25)
    post_spines["spine_size_bin"] = pd.cut(post_spines[x], bins=bins)
    post_spines["spine_size_bin_mid"] = (
        post_spines["spine_size_bin"].apply(lambda x: x.mid).astype(float)
    )

    if index is not None:
        index = pd.Index(index)
        index = index.intersection(post_spines.index)
        post_spines = post_spines.loc[index]

    return post_spines


exc_spines = get_spine_subset(exc_spines_index)
th_spines = get_spine_subset(th_spines_index)
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax = axs[0]
sns.lineplot(
    data=exc_spines,
    x="spine_size_bin_mid",
    y="is_multi",
    color=CELL_TYPE_PALETTE["excitatory"],
    ax=ax,
    label="Local excitatory",
)
sns.lineplot(
    data=th_spines,
    x="spine_size_bin_mid",
    y="is_multi",
    color=CELL_TYPE_PALETTE["thalamic"],
    ax=ax,
    linestyle="--",
    label="Thalamic",
)
ax.set(ylabel="Proportion\nmulti-spine")
sns.move_legend(ax, "upper left", title="Presynaptic axon")
# bins = np.geomspace(quantiles[low], quantiles[high], 20)
bins = 25
ax = axs[1]
sns.histplot(
    data=exc_spines,
    x=x,
    ax=ax,
    color=CELL_TYPE_PALETTE["excitatory"],
    alpha=0.3,
    bins=bins,
    log_scale=True,
    stat="proportion",
    label="Excitatory",
    element="step",
)
sns.histplot(
    data=th_spines,
    x=x,
    ax=ax,
    color=CELL_TYPE_PALETTE["thalamic"],
    alpha=0.3,
    bins=bins,
    log_scale=True,
    stat="proportion",
    label="Thalamic",
    element="step",
    linestyle="--",
)
ax.set(
    xlim=(quantiles[low], quantiles[high]),
    xlabel="Spine volume (nm³)",
    ylabel="Proportion of\ntargeted spines",
)
ax.set(xscale="log")

save_matplotlib_figure(fig, "exc_vs_th_size_by_multi", figure_out_path)

# %%

fig, axs = plt.subplots(1, 2, figsize=(4.58, 7.17), sharey=True)
ax = axs[0]
sns.histplot(
    data=exc_spines,
    y=x,
    ax=ax,
    color=CELL_TYPE_PALETTE["excitatory"],
    alpha=0.3,
    bins=bins,
    log_scale=True,
    stat="proportion",
    label="Excitatory",
    element="poly",
)
sns.histplot(
    data=th_spines,
    y=x,
    ax=ax,
    color=CELL_TYPE_PALETTE["thalamic"],
    alpha=0.3,
    bins=bins,
    log_scale=True,
    stat="proportion",
    label="Thalamic",
    element="poly",
    linestyle="--",
)
ax.set(
    ylim=(quantiles[low], quantiles[high]),
    ylabel="Spine volume (nm³)",
    xlabel="",
    xticks=[],
)
ax.set(yscale="log", xlabel="Normalized\ndensity\n(spines)")
ax.spines["bottom"].set_visible(False)
ax.text(
    0.2, 0.8, "Thalamic", transform=ax.transAxes, color=CELL_TYPE_PALETTE["thalamic"]
)
ax.text(
    0.2,
    0.15,
    "Local\nexcitatory",
    transform=ax.transAxes,
    color=CELL_TYPE_PALETTE["excitatory"],
)


ax = axs[1]
sns.lineplot(
    data=exc_spines,
    y="spine_size_bin_mid",
    x="is_multi",
    orient="y",
    color=CELL_TYPE_PALETTE["excitatory"],
    ax=ax,
    label="Local excitatory",
)
sns.lineplot(
    data=th_spines,
    y="spine_size_bin_mid",
    x="is_multi",
    orient="y",
    color=CELL_TYPE_PALETTE["thalamic"],
    ax=ax,
    linestyle="--",
    label="Thalamic",
)
ax.set(xlabel="Proportion of\nspines with\nmultiple inputs")
ax.get_legend().remove()
ax.set(ylim=(5e6, 1e9))


save_matplotlib_figure(
    fig, "exc_vs_th_size_by_multi_rotated", figure_out_path, pad_inches=0.2
)

# %%
# x = "post_p_spine_synapse"
x = "post_p_spine_site_is_multi"
data = cell_info.query(
    "cell_type == '23P' and broad_type == 'excitatory' and broad_type_lda_prediction == 'excitatory' and post_total_synapses > 1000"
)
x_vals = data[x]
# sns.histplot(
#     data=data,
#     x=x,
#     # hue="in_column",
#     stat="proportion",
#     # bins=50,
#     common_norm=False,
#     # log_scale=True,
# )

# compare normal and log normal fits


# fit
distributions = {
    "Normal": norm,
    "Lognormal": lognorm,
    "Weibull": weibull_min,
    "Powernorm": powernorm,
}
distribution_params = {}
for name, dist in distributions.items():
    if name == "Lognormal":
        fit_params = dict()
    else:
        fit_params = dict()
    params = dist.fit(x_vals, **fit_params)
    distribution_params[name] = params
    loglik = np.sum(dist.logpdf(x_vals, *params))
    print(f"{name} loglik: {loglik}")


# mu_log, std_log = norm.fit(np.log(x_vals))

#  PLOT THE FITS
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.histplot(
    data=data,
    x=x,
    # hue="in_column",
    stat="density",
    # bins=50,
    common_norm=False,
    # log_scale=True,
    color="black",
    ax=ax,
)

xmin, xmax = ax.get_xlim()
x_fit = np.linspace(xmin, xmax, 100)
# plot fits
for dist_name, params in distribution_params.items():
    p_norm = distributions[dist_name].pdf(x_fit, *params)
    ax.plot(x_fit, p_norm, "--", label=f"{dist_name}", linewidth=1.5, alpha=0.9)

ax.legend()

# plot log normal fit
# p_lognorm = norm.pdf(np.log(x_fit), mu_log, std_log) / x_fit
# p_lognorm = lognorm.pdf(x_fit, shape, loc=loc, scale=scale)
# ax.plot(x_fit, p_lognorm, "g--", label="Log-normal fit")


# %%
cell_type = "5P-IT"
x = "post_p_spine_site_is_multi"
y = "pre_total_synapses"
data = cell_info.query(f"cell_type == '{cell_type}' and in_column")

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=data,
    x=x,
    y=y,
    ax=ax,
    s=30,
)
ax.set(xscale="log", yscale="log")

xs = data[x]
ys = data[y]

stat, pvalue = pearsonr(xs, ys)

ax.set_title(f"{cell_type}: r = {stat:.2f}, p = {pvalue:.0e}")


# %% COMPARE WITH LAYER 6 CT SPINE TARGETING CURVE

if False:
    l6ct_synapses = all_pre_synapses.query(
        "pre_broad_type == 'inhibitory' and tag == 'spine'"
    ).copy()
    l6ct_spines_index = l6ct_synapses["spine_group_id"].dropna().unique().astype(int)

    exc_spines = get_spine_subset(exc_spines_index)
    l6ct_spines = get_spine_subset(l6ct_spines_index)
    th_spines = get_spine_subset(th_spines_index)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax = axs[0]
    sns.lineplot(
        data=exc_spines,
        x="spine_size_bin_mid",
        y="is_multi",
        color=CELL_TYPE_PALETTE["excitatory"],
        ax=ax,
        label="Local excitatory",
    )
    sns.lineplot(
        data=l6ct_spines,
        x="spine_size_bin_mid",
        y="is_multi",
        color=CELL_TYPE_PALETTE["inhibitory"],
        ax=ax,
        linestyle="--",
        label="Inhibitory",
    )
    sns.lineplot(
        data=th_spines,
        x="spine_size_bin_mid",
        y="is_multi",
        color=CELL_TYPE_PALETTE["thalamic"],
        ax=ax,
        linestyle="--",
        label="Thalamic",
    )
    ax.set(ylabel="Proportion\nmulti-spine")
    sns.move_legend(ax, "upper left", title="Presynaptic axon")
    ax = axs[1]
    sns.histplot(
        data=exc_spines,
        x=x,
        ax=ax,
        color=CELL_TYPE_PALETTE["excitatory"],
        alpha=0.3,
        bins=bins,
        # log_scale=True,
        stat="proportion",
        label="Excitatory",
        element="poly",
    )
    sns.histplot(
        data=l6ct_spines,
        x=x,
        ax=ax,
        color=CELL_TYPE_PALETTE["inhibitory"],
        alpha=0.3,
        bins=bins,
        # log_scale=True,
        stat="proportion",
        # label="6P-CT",
        label="Inhibitory",
        element="poly",
        linestyle="--",
    )
    sns.histplot(
        data=th_spines,
        x=x,
        ax=ax,
        color=CELL_TYPE_PALETTE["thalamic"],
        alpha=0.3,
        bins=bins,
        # log_scale=True,
        stat="proportion",
        label="Thalamic",
        element="poly",
        linestyle="--",
    )
    ax.set(
        xlim=(quantiles[low], quantiles[high]),
        xlabel="Spine volume (nm³)",
        ylabel="Proportion of\ntargeted spines",
    )
    ax.set(xscale="log")

    # save_matplotlib_figure(fig, "exc_vs_th_size_by_multi", figure_out_path)


# %% FIT ISOTONIC REGRESSION FOR SPINE SIZE

if False:
    from sklearn.isotonic import IsotonicRegression

    iso_reg = IsotonicRegression(
        y_min=quantiles[low], y_max=quantiles[high], increasing=True
    )
    x = "size_nm3"
    y = "is_multi"
    iso_reg.fit(np.log(post_spines[x]), post_spines[y])

    iso_reg.predict(bins)

# %% ANALYZE SPINE SIZE BY PRESYNAPTIC CELL TYPE
syns = all_pre_synapses.query(
    "post_broad_type == 'excitatory' and tag == 'spine'"
).copy()
syns["spine_size_bin"] = pd.cut(syns["spine_size_nm3"], bins=bins)
syns["spine_size_bin_mid"] = syns["spine_size_bin"].apply(lambda x: x.mid)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(
    data=syns,
    x="spine_size_bin_mid",
    y="spine_is_multi",
    hue="pre_cell_type",
    palette=CELL_TYPE_PALETTE,
    ax=ax,
)
sns.move_legend(ax, "upper left", title="Presynaptic cell type", bbox_to_anchor=(1, 1))


# %% VALIDATE SPINE GROUP MAPPING
# synapse_table_group_map = all_pre_synapses.groupby(["post_pt_root_id", "component_id"])[
#     "spine_group_id"
# ].unique().explode()

# spine_table_group_map = spine_table.groupby(['post_pt_root_id', 'component_id'])[
#     'group_id'
# ].unique().explode()

all_pre_synapses_pl = (
    pl.DataFrame(all_pre_synapses)
    .with_columns(
        pl.col("component_id").cast(pl.Int64),
        pl.col("spine_group_id").cast(pl.Int64),
    )
    .lazy()
)
spine_table_pl = pl.DataFrame(spine_table).lazy()

result = (
    all_pre_synapses_pl.join(spine_table_pl, on=["post_pt_root_id", "component_id"])
    .select(["post_pt_root_id", "component_id", "group_id", "spine_group_id"])
    .collect()
)
assert (
    result.select((pl.col("group_id") == pl.col("spine_group_id")).alias("equals"))
    .min()
    .item()
)

# %%
result.filter(pl.col("group_id") != pl.col("spine_group_id"))

# %% REFINE P_MULTI ANALYSIS WITH BETTER BINNING

low = 0.001
high = 0.999
quantiles = all_pre_synapses["spine_size_nm3"].quantile([low, high])
# bins = np.geomspace(quantiles[low], quantiles[high], 16)

exc_synapses["spine_size_bin"] = pd.cut(exc_synapses["spine_size_nm3"], bins=bins)
th_synapses["spine_size_bin"] = pd.cut(th_synapses["spine_size_nm3"], bins=bins)

p_multi_exc = exc_synapses.groupby("spine_size_bin")["spine_is_multi"].mean() / 2
p_multi_th = th_synapses.groupby("spine_size_bin")["spine_is_multi"].mean() / 2

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax = axs[0]
sns.lineplot(
    x=p_multi_exc.index.categories.mid.values,
    y=p_multi_exc.values,
    color=CELL_TYPE_PALETTE["excitatory"],
    ax=ax,
    # marker='o'
)
sns.lineplot(
    x=p_multi_th.index.categories.mid.values,
    y=p_multi_th.values,
    color=CELL_TYPE_PALETTE["thalamic"],
    ax=ax,
    # marker='o'
)

ax.set(ylabel="Proportion\nmulti-spine")
ax.set(xscale="log")


ax = axs[1]
sns.histplot(
    data=exc_synapses,
    x="spine_size_nm3",
    ax=ax,
    color=CELL_TYPE_PALETTE["excitatory"],
    alpha=0.3,
    # bins=bins,
    # log_scale=True,
    stat="proportion",
    label="Excitatory",
    element="poly",
)
sns.histplot(
    data=th_synapses,
    x="spine_size_nm3",
    ax=ax,
    color=CELL_TYPE_PALETTE["thalamic"],
    alpha=0.3,
    # bins=bins,
    # log_scale=True,
    stat="proportion",
    label="Thalamic",
    element="poly",
)
ax.set(
    xscale="log",
    xlim=(quantiles[low], quantiles[high]),
    xlabel="Spine volume (nm³)",
    ylabel="Proportion\nof spines",
)
# %% ANALYZE 23P CELLS WITH HIGH MULTI-SPINE RATES

query_cell_info = (
    cell_info.query(
        "cell_type == '23P' and broad_type_lda_prediction == 'excitatory' and post_total_synapses > 1000"
    )
    .sort_values("post_p_spine_synapse_is_multi", ascending=False)
    .copy()
)
query_synapses = column_column_synapses.query('post_cell_type == "23P"')
pre_type_counts = (
    query_synapses.groupby(["post_pt_root_id", "pre_cell_type"]).size().unstack()
)
query_cell_info["bc_mc_ratio"] = pre_type_counts["BC"] / pre_type_counts["MC"]
query_cell_info["mc_count"] = pre_type_counts["MC"]
query_cell_info["bc_count"] = pre_type_counts["BC"]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

sns.scatterplot(data=query_cell_info, x="mc_count", y="post_p_spine_site_is_multi")
ax.set(
    xlabel="Number of MC inputs",
    ylabel="Proportion spines with multiple inputs",
    title="23P cells in column",
)

# %% PLOT SPINE COUNT HISTOGRAM

sns.histplot(data=query_cell_info, x="post_spine_synapses", log_scale=True)

# %% ANALYZE TARGETED SPINES BY CELL SELECTION
spine_ids = query_synapses["spine_group_id"].unique().astype(int)

all_post_spines = spine_table.query(
    "~size_nm3.isna() and post_broad_type == 'excitatory' and group_id.isin(@spine_ids)"
).copy()
select_cells = query_cell_info.query("mc_count > 75").index

all_post_spines["is_targeted"] = all_post_spines["post_pt_root_id"].isin(select_cells)

all_post_spines["spine_size_bin"] = pd.qcut(all_post_spines["size_nm3"], q=8)
fig, ax = plt.subplots(1, 1, figsize=(8, 3))

for is_targeted, post_spines in all_post_spines.groupby("is_targeted"):
    p_multi = post_spines.groupby("spine_size_bin")["is_multi"].mean()
    p_multi = p_multi.reset_index()
    p_multi["spine_size_bin_mid"] = p_multi["spine_size_bin"].map(lambda x: x.mid)

    sns.lineplot(
        data=p_multi,
        x="spine_size_bin_mid",
        y="is_multi",
        # color="black",
        ax=ax,
        label=is_targeted,
    )
    ax.set(ylabel="Proportion\nmulti-spine")

# %% ANALYZE PRESYNAPTIC TYPES FOR EACH SPINE

pre_types = {}
for i, row in all_post_spines.iterrows():
    pre_ids = row["pre_pt_root_ids"]
    cell_types = cell_info.reindex(pre_ids)["cell_type"]
    pre_types[i] = cell_types.value_counts(dropna=False).to_dict()

pre_types = pd.DataFrame(pre_types)

# %% ANALYZE INHIBITORY MULTI-SPINE CONNECTIONS
data = (
    all_pre_synapses.query(
        'pre_broad_type == "inhibitory" and post_broad_type == "excitatory"'
    )
    .query("tag == 'spine'")
    .query("tag_detailed == 'multi_spine'")
)

x = "spine_size_nm3"
y = "size"
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.scatterplot(data=data, x=x, y=y, alpha=0.5, s=5, ax=ax)
ax.set(xscale="log", yscale="log", xlabel="Spine volume (nm³)", ylabel="Synapse size")

sub_data = data.dropna(subset=[x, y])
pearsonr(sub_data[x], sub_data[y])


# %% REGRESSION MODEL OF SPINE SIZE AS FUNCTION OF INPUTS
data = all_pre_synapses.query(
    "post_broad_type == 'excitatory' and tag == 'spine' and spine_size_nm3.notna()"
).copy()
data["pre_broad_type"] = data["pre_broad_type"].map(
    lambda x: "excitatory" if x == "thalamic" else x
)
rows = []
for spine_group_id, spine_data in data.groupby("spine_group_id"):
    n = spine_data["spine_n_pre_synapses"].iloc[0]
    if n == len(spine_data):
        row = spine_data.groupby("pre_broad_type")["size"].sum().to_dict()
        count_row = (
            spine_data.groupby("pre_pt_root_id")["pre_broad_type"]
            .first()
            .value_counts()
            .fillna(0)
            .to_dict()
        )
        for key, count in count_row.items():
            row[f"{key}_count"] = count
        row["spine_group_id"] = int(spine_group_id)
        row["spine_size_nm3"] = spine_data["spine_size_nm3"].iloc[0]
        row["is_multi"] = spine_data["spine_is_multi"].iloc[0]
        rows.append(row)

reg_df = pd.DataFrame(rows).set_index("spine_group_id").fillna(0)

# reg_df = reg_df.query("unknown == 0")

reg_df["both"] = reg_df["excitatory"] + reg_df["inhibitory"]
reg_df["product"] = reg_df["excitatory"] * reg_df["inhibitory"]
# reg_df = reg_df.drop(columns=["unknown"])


# X = reg_df[["excitatory", "inhibitory"]]
X = reg_df[["both"]]
y = reg_df["spine_size_nm3"]

train_indices, test_indices = train_test_split(reg_df.index, test_size=0.2)

X_train = X.loc[train_indices]
y_train = y.loc[train_indices]
X_test = X.loc[test_indices]
y_test = y.loc[test_indices]

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
model.coef_, model.intercept_

score = model.score(X_test, y_test)

y_pred_test = model.predict(X_test)

pred_df = pd.DataFrame(
    {
        "true_spine_size_nm3": y_test,
        "pred_spine_size_nm3": y_pred_test,
    }
)
pred_df["is_multi"] = reg_df.loc[pred_df.index, "is_multi"]
pred_df["residual"] = pred_df["true_spine_size_nm3"] - pred_df["pred_spine_size_nm3"]
pred_df["count_map"] = (
    reg_df.loc[pred_df.index, "excitatory_count"].astype(int).astype(str)
    + "E-"
    + reg_df.loc[pred_df.index, "inhibitory_count"].astype(int).astype(str)
    + "I"
)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=pred_df,
    x="true_spine_size_nm3",
    y="pred_spine_size_nm3",
    hue="count_map",
    size="is_multi",
    size_norm=(0, 1),
    sizes=(1, 50),
    alpha=1,
    ax=ax,
    color="black",
)
# sns.scatterplot(
#     data=pred_df.query("is_multi == True"),
#     x="true_spine_size_nm3",
#     y="pred_spine_size_nm3",
#     hue="count_map",
#     # alpha=0.5,
#     s=20,
#     ax=ax,
#     color="black",
# )
ax.set(
    xscale="log",
    yscale="log",
    xlabel="True spine size (nm³)",
    ylabel="Predicted spine size (nm³)",
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.autoscale(False)
ax.plot([1e6, 1e9], [1e6, 1e9], color="black", linestyle="--")
ax.text(0.05, 0.9, f"R² = {score:.2f}", transform=ax.transAxes)
# %% ANALYZE PREDICTION RESIDUALS
sns.stripplot(
    data=pred_df,
    x="is_multi",
    y="residual",
    hue="is_multi",
)

# %% ANALYZE BC TO 23P CONNECTIONS

data = all_pre_synapses.query(
    "pre_cell_type == 'BC' and post_cell_type == '23P' and tag == 'spine'"
).copy()
data["of_interest"] = (data["tag_detailed"] == "single_spine") & (
    data["pre_cell_type"] == "BC"
)
sns.histplot(
    data=data,
    x="post_euc_distance_to_nuc",
    hue="of_interest",
    bins=100,
    log_scale=True,
    common_norm=False,
    stat="density",
)

# %% PLOT SPINE VS SYNAPSE SIZE SCATTER
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=data,
    x="spine_size_nm3",
    y="size",
    ax=ax,
    s=2,
    linewidth=0,
    hue="of_interest",
)
ax.set(xscale="log", yscale="log", xlabel="Spine volume (nm³)", ylabel="Synapse size")


# %% ANALYZE 4P CELL MTYPE DISTRIBUTIONS
cell_type = "4P"
x = "post_p_spine_site_is_multi"
weights = "post_spine_sites"

fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(
    cell_info.query(
        "cell_type == @cell_type and broad_type_lda_prediction == 'excitatory'"
    ),
    x=x,
    weights=weights,
    hue="mtype",
    # hue='visual_area',
    bins=50,
    log_scale=False,
    hue_order=["L4a", "L4b", "L4c"],
    # hue_order=["L2a", "L2b", "L2c"],
    element="step",
    stat="density",
    common_norm=False,
    ax=ax,
)
ax.set_title(f"Cell type = {cell_type}")


# %% ANALYZE L6 SYNAPSE LOCATION PATTERNS

target_tag = "shaft"
target_type = "excitatory"
select_syns = all_pre_synapses.query(
    "pre_cell_type.isin(['6P-IT', '6P-CT']) and post_broad_type == @target_type and tag == @target_tag"
)

fig, axs = plt.subplots(1, 2, figsize=(8, 8), dpi=300, sharey=True)

scatter_kws = dict(
    s=2,
    linewidth=0.0,
    alpha=0.5,
    palette=CELL_TYPE_PALETTE,
    hue="pre_cell_type",
    y="ctr_pt_position_y",
    x="ctr_pt_position_x",
    # hue_order=["6P-IT", "6P-CT"],
    legend=False,
    ax=axs[0],
)
ax = axs[0]
sns.scatterplot(data=select_syns, **scatter_kws)

ax = axs[1]
histplot_kws = dict(
    # bins=40,
    stat="density",
    common_norm=True,
    hue="pre_cell_type",
    palette=CELL_TYPE_PALETTE,
    hue_order=["6P-IT", "6P-CT"],
    element="step",
    fill=False,
    linewidth=0,
    kde=True,
)
sns.histplot(data=select_syns, y="ctr_pt_position_y", ax=ax, **histplot_kws)
ax.invert_yaxis()

# fig.suptitle(f"Outputs to excitatory neurons on {tag}")


# %% COMPARE DEPTH DISTRIBUTIONS BY COMPARTMENT
fig, axs = plt.subplots(
    1, 2, figsize=(8, 6), dpi=300, sharey=True, layout="constrained"
)
y = "transformed_um_y"
histplot_kws = dict(
    stat="density",
    common_norm=False,
    hue="pre_cell_type",
    palette=CELL_TYPE_PALETTE,
    hue_order=["6P-IT", "6P-CT"],
    element="step",
    fill=True,
    linewidth=2,
    alpha=0.1,
    kde=True,
)

target_tag = "spine"
target_type = "excitatory"
select_syns = all_pre_synapses.query(
    "pre_cell_type.isin(['6P-IT', '6P-CT']) and post_broad_type == @target_type and tag == @target_tag"
)
ax = axs[0]
sns.histplot(data=select_syns, y=y, ax=ax, **histplot_kws)
sns.move_legend(ax, "upper right", title="Presynaptic\ncell type", markerscale=10)
ax.set(ylabel="Depth (um)", title="Synapses on\nexcitatory spines")

target_tag = "shaft"
target_type = "excitatory"
select_syns = all_pre_synapses.query(
    "pre_cell_type.isin(['6P-IT', '6P-CT']) and post_broad_type == @target_type and tag == @target_tag"
)
ax = axs[1]
sns.histplot(data=select_syns, y=y, ax=ax, legend=False, **histplot_kws)
ax.set(ylabel="", title="Synapses on\nexcitatory shafts")

ax.invert_yaxis()

# %% COMPARE L6 INPUTS TO EXCITATORY NEURONS BY COMPARTMENT AND SPACE
fig, axs = plt.subplots(
    1, 2, figsize=(8, 6), dpi=300, sharey=True, sharex=True, layout="constrained"
)
y = "transformed_um_y"

histplot_kws = dict(
    stat="density",
    common_norm=True,
    hue="tag",
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    hue_order=["spine", "shaft"],
    element="step",
    fill=True,
    linewidth=2,
    alpha=0.1,
    kde=True,
)

target_tags = ["spine", "shaft"]
target_type = "excitatory"
source_types = ["6P-IT"]
select_syns = all_pre_synapses.query(
    "pre_cell_type.isin(@source_types) and post_broad_type == @target_type and tag.isin(@target_tags)"
)
ax = axs[0]
sns.histplot(data=select_syns, y=y, ax=ax, **histplot_kws)
sns.move_legend(ax, "upper right", title="Target", markerscale=10)
ax.set(ylabel="Depth (um)", title=r"6P-IT$\rightarrow$excitatory")

target_tags = ["spine", "shaft"]
target_type = "excitatory"
source_types = ["6P-CT"]
select_syns = all_pre_synapses.query(
    "pre_cell_type.isin(@source_types) and post_broad_type == @target_type and tag.isin(@target_tags)"
)
ax = axs[1]
sns.histplot(data=select_syns, y=y, ax=ax, legend=False, **histplot_kws)
ax.set(ylabel="", title=r"6P-CT$\rightarrow$excitatory")

ax.invert_yaxis()

# %% ANALYZE L6 CONNECTION DEPTHS BY TARGET TYPE

fig, axs = plt.subplots(
    2, 7, figsize=(24, 10), dpi=300, layout="constrained", sharex=True
)

histplot_kws = dict(
    y="transformed_um_y",
    hue="tag",
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    hue_order=["spine", "shaft", "soma"],
    element="step",
    fill=False,
    linewidth=1,
    alpha=0.7,
    kde=True,
    stat="count",
    common_norm=True,
)
for i, source in enumerate(["6P-IT", "6P-CT"]):
    for j, target in enumerate(
        ["23P", "4P", "5P-IT", "5P-ET", "5P-NP", "6P-IT", "6P-CT"]
    ):
        ax = axs[i, j]
        target_query = f"post_cell_type == '{target}'"
        title = target
        select_syns = all_pre_synapses.query(
            "pre_cell_type == @source and " + target_query
        ).query("tag in ['spine', 'shaft']")
        sns.histplot(data=select_syns, ax=ax, **histplot_kws)
        ax.set(title=title)
        if j == 0:
            ax.set(ylabel="Depth (um)")
        else:
            ax.set(ylabel="")
        ax.invert_yaxis()
        if i == 0 and j == 7:
            sns.move_legend(
                ax, "upper right", title="Target compartment", markerscale=5
            )
        else:
            ax.legend_.remove()
axs[0, 0].text(
    -0.5,
    0.5,
    "6P-IT\npresynaptic",
    ha="right",
    va="center",
    clip_on=False,
    transform=axs[0, 0].transAxes,
    fontsize="large",
)
axs[1, 0].text(
    -0.5,
    0.5,
    "6P-CT\npresynaptic",
    ha="right",
    va="center",
    clip_on=False,
    transform=axs[1, 0].transAxes,
    fontsize="large",
)

# %% ANALYZE EXCITATORY TARGET DISTRIBUTIONS

post_on_excitatory = all_post_synapses.query("post_broad_type == 'excitatory'")
n_groups = post_on_excitatory["post_cell_type"].nunique()

fig, axs = plt.subplots(
    1, n_groups + 1, figsize=(14, 6), dpi=300, layout="constrained", sharey=True
)

histplot_kws = dict(
    y="transformed_um_y",
    hue="tag",
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    hue_order=["spine", "shaft", "soma"],
    element="step",
    fill=False,
    linewidth=0.0,
    alpha=0.7,
    kde=True,
    stat="count",
    common_norm=True,
    bins=10,
    legend=False,
)
ax = axs[0]
sns.histplot(
    data=post_on_excitatory,
    ax=ax,
    **histplot_kws,
)
ax.set_title("All excitatory")

for i, (cell_type, group) in enumerate(
    post_on_excitatory.groupby("post_cell_type", observed=True)
):
    ax = axs[i + 1]
    sns.histplot(
        data=group,
        ax=ax,
        **histplot_kws,
    )
    ax.set(title=cell_type)

for ax in axs.flat:
    ax.spines["bottom"].set_visible(False)
    ax.set(xticks=[], xlabel="")

ax = axs[0]
ax.set(ylim=(800, 0), ylabel="Depth (um)")

save_matplotlib_figure(fig, "excitatory_target_distributions_by_depth", figure_out_path)


# %% ANALYZE OUTPUTS BY COMPARTMENT AND TARGET TYPE

fig, axs = plt.subplots(2, 3, figsize=(16, 12), dpi=300, layout="constrained")

for i, target in enumerate(["excitatory", "inhibitory"]):
    for j, compartment in enumerate(["spine", "shaft", "soma"]):
        ax = axs[i, j]
        if target == "all":
            data = mod_proofread_info
            title = "All outputs"
        else:
            data = mod_proofread_info.query(f"pre_total_synapse_to_{target} > 0")
            title = f"Outputs to {target}"

        if target == "all":
            x = f"pre_p_{compartment}_synapse"
            size = "pre_total_synapses"
        else:
            x = f"pre_p_{compartment}_synapse_to_{target}"
            size = f"pre_total_synapse_to_{target}"
        sns.scatterplot(
            data=data,
            y="cell_type_y",
            x=x,
            hue="cell_type",
            palette=CELL_TYPE_PALETTE,
            ax=ax,
            legend=False,
            size=size,
            size_norm=(0, 1000),
            sizes=(1, 50),
            linewidth=0.2,
            alpha=0.8,
        )
        ax.invert_yaxis()

        ax.set_yticks(np.arange(len(mod_proofread_info["cell_type"].cat.categories)))
        ax.set_yticklabels(mod_proofread_info["cell_type"].cat.categories)

        ax.set(xlabel=f"Proportion of outputs \nto {target}on {compartment}", ylabel="")
        if j == 0:
            ax.set_ylabel(
                f"Outputs to\n{target}",
                rotation=0,
                rotation_mode="anchor",
                ha="right",
                labelpad=20,
            )

# %% ANALYZE BASKET CELL OUTPUT DISTANCES
basket_cell_outputs = all_pre_synapses.query("pre_cell_type == 'BC'").copy()


histplot_kws = dict(
    stat="density",
    common_norm=False,
    element="step",
    bins=100,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
)
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
sns.histplot(
    data=basket_cell_outputs.query("post_broad_type == 'inhibitory'"),
    x="pre_path_distance_to_nuc",
    hue="tag",
    hue_order=["spine", "shaft", "soma"],
    ax=ax,
    log_scale=True,
    element="step",
    bins=100,
    stat="density",
    common_norm=False,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    linestyle="--",
)
sns.histplot(
    data=basket_cell_outputs.query("post_broad_type == 'excitatory'"),
    x="pre_path_distance_to_nuc",
    hue="tag",
    hue_order=["spine", "shaft", "soma"],
    ax=ax,
    log_scale=True,
    element="step",
    bins=100,
    stat="density",
    common_norm=False,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
)

# %% ANALYZE POST SYNAPTIC DISTANCES
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
sns.histplot(
    data=basket_cell_outputs,
    x="post_path_distance_to_nuc",
    hue="tag",
    ax=ax,
    log_scale=True,
    element="step",
    bins=50,
    stat="density",
    common_norm=True,
    # fill=False,
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
)

# %% VISUALIZE INDIVIDUAL BASKET CELL OUTPUT PATTERNS

n_cells = 60
n_cols = 6
fig, axs = plt.subplots(
    n_cells // n_cols,
    n_cols,
    figsize=(15, 15),
    sharex=True,
    sharey=False,
    dpi=300,
    layout="constrained",
)
for i, (pre_pt_root_id, group) in enumerate(
    basket_cell_outputs.groupby("pre_pt_root_id")
):
    ax = axs[i // n_cols, i % n_cols]
    sns.histplot(
        # data=group.query("post_broad_type == 'inhibitory'"),
        data=group,
        x="pre_path_distance_to_nuc",
        # hue="post_broad_type",
        hue="tag",
        hue_order=["spine", "shaft", "soma"],
        # hue_order=["excitatory", "inhibitory"],
        ax=ax,
        log_scale=False,
        element="step",
        stat="density",
        common_norm=False,
        palette=COMPARTMENT_PALETTE_MUTED_HEX,
        linestyle="-",
        legend=False,
        bins=np.geomspace(100_000, 1_000_000, 50),
    )
    ax.set(ylabel="", yticks=[])
    # sns.histplot(
    #     data=group.query("post_broad_type == 'excitatory'"),
    #     x="pre_distance_to_root_um",
    #     hue="tag",
    #     hue_order=["spine", "shaft", "soma"],
    #     ax=ax,
    #     log_scale=True,
    #     element="step",
    #     stat="density",
    #     common_norm=False,
    #     palette=COMPARTMENT_PALETTE_MUTED_HEX,
    #     legend=False,
    # )
    if i >= n_cells - 1:
        break

# %% ANALYZE MULTI SYNAPSE COMPARTMENTS
multi_synapses = all_pre_synapses.query("spine_is_multi").copy()
multi_synapses["component_id"] = multi_synapses["component_id"].astype("Int64")

multi_synapses.sort_values(["post_pt_root_id", "component_id"], inplace=True)

multi_synapses["pre_compartment"] = multi_synapses["pre_compartment"].fillna("unknown")
multi_synapses["post_compartment"] = multi_synapses["post_compartment"].fillna(
    "unknown"
)

# %% TABULATE COMPARTMENT COMBINATIONS

multi_synapses[["pre_compartment", "post_compartment"]].value_counts(
    dropna=False
).reset_index().fillna("unknown")

# %% FILTER MULTI SYNAPSES BY COMPARTMENT
multi_synapses.query(
    "pre_compartment.isin(['axon', 'unknown']) and post_compartment.isin(['dendrite', 'unknown', 'perisoma'])",
    inplace=True,
)
multi_synapses[["pre_compartment", "post_compartment"]].value_counts()


# %% CREATE COUNT TABLE FOR MULTI SYNAPSES

count_table = (
    multi_synapses.groupby(
        ["post_pt_root_id", "component_id", "pre_broad_type", "post_broad_type"],
        dropna=False,
        observed=True,
    )["pre_pt_root_id"]
    .nunique()
    .rename("count")
    .to_frame()
    .reset_index()
    # .rename(columns={np.nan: "unknown"})
    .pivot(
        index=["post_pt_root_id", "component_id", "post_broad_type"],
        columns=["pre_broad_type"],
        values="count",
    )
    .fillna(0)
    .astype(int)
    .reset_index(level="post_broad_type")
    # .query("unproofread == 0")
)
# %%
multi_synapses.groupby(
    ["post_pt_root_id", "component_id", "pre_broad_type", "post_broad_type"],
    dropna=False,
    observed=True,
)["pre_pt_root_id"].nunique().rename("count").to_frame().reset_index().pivot(
    index=["post_pt_root_id", "component_id", "post_broad_type"],
    columns=["pre_broad_type"],
    values="count",
)

# %% ANALYZE MULTI SPINE ANNOTATIONS

annotations = get_multi_annotations()
subset_count_table = count_table.loc[
    count_table.index.intersection(annotations.query("singlehead").index)
]

# subset_count_table = count_table


# query_roots = cell_info.query('cell_type.isin(["23P", "4P"])').index
# subset_count_table = (
#     subset_count_table.reset_index()
#     .query("post_pt_root_id.isin(@query_roots)")
#     .set_index(["post_pt_root_id", "component_id"])
# )

# %% DEFINE COUNT SUMMARIZATION FUNCTION
pre_codes = ["thalamic", "excitatory", "inhibitory"]
post_codes = ["excitatory", "inhibitory"]


def summarize_counts(count_table):
    # post_codes = ["excitatory", "inhibitory"]
    count_table["total"] = count_table[pre_codes].sum(axis=1)
    count_summary = (
        count_table.query("total > 1")
        .groupby(["post_broad_type"] + pre_codes, observed=True)
        .size()
        .rename("count")
        .to_frame()
        .reset_index()
        .sort_values(["post_broad_type", "count"], ascending=[True, False])
    )
    count_summary["x"] = np.arange(len(count_summary))
    return count_summary


full_count_summary = summarize_counts(count_table)
count_summary = summarize_counts(subset_count_table)
code_counts = count_summary.set_index(pre_codes)["count"]

comparison = full_count_summary.set_index(
    ["post_broad_type", "thalamic", "excitatory", "inhibitory"]
)[["count"]].join(
    count_summary.set_index(
        ["post_broad_type", "thalamic", "excitatory", "inhibitory"]
    )[["count"]],
    rsuffix="_subset",
)
(comparison["count_subset"] / comparison["count"]).fillna(0)

# %% COMPUTE EXCITATORY SUBSET STATISTICS
comparison.query("post_broad_type == 'excitatory'")[["count", "count_subset"]].sum()


# %% VISUALIZE MULTI SPINE COUNT DISTRIBUTIONS

plt.rcParams["figure.dpi"] = 300

# fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# code_counts = codes.groupby(codes.columns.tolist()).size().sort_values(ascending=False)

x = np.arange(len(code_counts))
y = code_counts.values

fig, axs = plt.subplots(
    2,
    1,
    figsize=(7, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 0.2], "hspace": 0.04},
)

ax = axs[0]
# ax.bar(x, y, color="black", alpha=0.5)
sns.barplot(
    data=count_summary,
    y="count",
    x="x",
    hue="post_broad_type",
    hue_order=post_codes,
    palette=CELL_TYPE_PALETTE,
    ax=ax,
    # color="black",
    alpha=0.5,
)
ax.set_ylabel("Multi-input spine count")
sns.move_legend(ax, "upper right", title="Postsynaptic cell")
handles, labels = ax.get_legend_handles_labels()
labels = [label.capitalize() for label in labels]
ax.legend(handles, labels, title="Postsynaptic cell")
for idx, row in count_summary.iterrows():
    ax.text(
        row["x"],
        row["count"] + 0.5,
        f"{row['count']}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
    )


ax = axs[1]
indicator_matrix = code_counts.index.to_frame(index=False).T
# indicator_matrix = indicator_matrix.loc[marginals.index]

point_y, point_x = np.nonzero(indicator_matrix.values)
ax.scatter(
    point_x,
    point_y,
    color="black",
    alpha=1,
    s=200,
)
for x_i, y_i in zip(point_x, point_y):
    ax.text(
        x_i,
        y_i,
        indicator_matrix.iloc[y_i, x_i],
        color="white",
        ha="center",
        va="center",
        fontsize=14,
    )

ax.set_yticks(np.arange(len(indicator_matrix)))
ax.set_yticklabels(indicator_matrix.index.str.capitalize(), rotation=0)
ax.spines["bottom"].set_visible(False)
ax.set_xticks([])
for i in range(indicator_matrix.shape[1]):
    ax.axvline(i, color="black", alpha=0.5, lw=0.5, zorder=-4, ymax=2, clip_on=True)
for j in range(indicator_matrix.shape[0]):
    ax.axhline(j, color="black", alpha=0.5, lw=0.5, zorder=-2)
ax.set_ylim(-0.5, len(indicator_matrix) - 0.5)
ax.invert_yaxis()
ax.set_ylabel("Input type")
ax.set_xlabel("Input type count")


# %% ANALYZE PERISOMATIC SPINE INPUTS
put_soma_spines = (
    all_post_synapses.query("post_path_distance_to_nuc <= 10000 and tag == 'spine'")
    .query("post_broad_type == 'excitatory'")
    .copy()
)

put_soma_spines["segments"] = list(
    zip(put_soma_spines["pre_pt_root_id"], put_soma_spines["post_pt_root_id"])
)
put_soma_spines[["pre_axon_cleaned", "pre_cell_type"]].value_counts().sort_index()


# %% ANALYZE CLEAN EXCITATORY SPINE INPUTS
clean_all_post_synapses = all_post_synapses.query(
    "pre_axon_cleaned and tag == 'spine' and post_broad_type == 'excitatory'"
).copy()
clean_all_post_synapses.query("post_euc_distance_to_nuc < 10000")[
    "spine_n_pre_pt_root_ids"
].value_counts(normalize=True)

# %% BIN SYNAPSES BY DISTANCE TO NUCLEUS

bins = np.linspace(5000, 100_000, 51)
mids = (bins[:-1] + bins[1:]) / 2 / 1000
clean_all_post_synapses["post_euc_distance_to_nuc_bin"] = pd.cut(
    clean_all_post_synapses["post_euc_distance_to_nuc"],
    bins=bins,
    labels=mids,
    include_lowest=True,
)

props_by_distance = clean_all_post_synapses.groupby("post_euc_distance_to_nuc_bin")[
    "pre_broad_type"
].value_counts(normalize=True, dropna=False)
props_by_distance = props_by_distance.reset_index(name="proportion")
props_by_distance["pre_broad_type"] = (
    props_by_distance["pre_broad_type"].astype(str).replace("nan", "unknown")
)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

CELL_TYPE_PALETTE["unknown"] = "#373737"
sns.lineplot(
    data=props_by_distance,
    x="post_euc_distance_to_nuc_bin",
    y="proportion",
    hue="pre_broad_type",
    hue_order=["excitatory", "inhibitory", "thalamic"],
    palette=CELL_TYPE_PALETTE,
    ax=ax,
)
sns.move_legend(ax, "upper right", title="Presynaptic cell", bbox_to_anchor=(1.0, 1))
ax.set(
    ylabel="Proportion of spine synapses",
    xlabel="Euclidean distance to postsynaptic nucleus (um)",
)
save_matplotlib_figure(fig, "spine_input_type_by_distance_to_nucleus", figure_out_path)

# %% ANALYZE SYNAPSE COUNT PROPORTIONS BY DISTANCE
count_props_by_distance = clean_all_post_synapses.groupby(
    "post_euc_distance_to_nuc_bin"
)["spine_n_pre_pt_root_ids"].value_counts(normalize=True)
count_props_by_distance = count_props_by_distance.reset_index(name="proportion")
count_props_by_distance["spine_n_pre_pt_root_ids"] = (
    count_props_by_distance["spine_n_pre_pt_root_ids"].astype("int").astype("str")
)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(
    data=count_props_by_distance,
    x="post_euc_distance_to_nuc_bin",
    y="proportion",
    hue="spine_n_pre_pt_root_ids",
    hue_order=["1", "2", "3"],
    palette="Blues",
    # palette="viridis",
    ax=ax,
)
sns.move_legend(
    ax, "upper left", title="Number of\nsynapses on\n spine", bbox_to_anchor=(1.0, 1)
)
ax.set(
    ylabel="Proportion of spine synapses",
    xlabel="Euclidean distance to postsynaptic nucleus (um)",
)


# %% SETUP DISTANCE ANALYSIS PARAMETERS

# parameters
distance_type = "euc"
distance_column = f"post_{distance_type}_distance_to_nuc"
distance_bin_column = f"post_{distance_type}_distance_to_nuc_bin"

pre_broad_type = "inhibitory"  # "excitatory"
pre_cell_type = "BC"
post_broad_type = "excitatory"
post_cell_type = None

agg_column = "tag_detailed"
tags = ["single_spine", "multi_spine"]
normalize = True
bins = np.linspace(0, 100_000, 25)
include_total = False

palette = COMPARTMENT_PALETTE_MUTED_HEX.copy()
palette["total"] = "#616161"
palette["multi_spine"] = "#cb0eff"
palette["single_spine"] = palette["spine"]

if include_total:
    plot_tags = tags + ["total"]
else:
    plot_tags = tags

if normalize:
    ylabel = "Proportion of spine synapses"
else:
    ylabel = "Synapse count"

if distance_type == "euc":
    xlabel = "Euc. distance to post nucleus (um)"
elif distance_type == "path":
    xlabel = "Path distance to post nucleus (um)"

pre_title = "Proofread "
if pre_cell_type is not None:
    pre_title += f"{pre_cell_type}" + r"$\rightarrow$"
elif pre_broad_type is not None:
    pre_title += f"{pre_broad_type}" + r"$\rightarrow$"
if post_cell_type is not None:
    pre_title += f"{post_cell_type}"
elif post_broad_type is not None:
    pre_title += f"{post_broad_type}"

post_title = r"Any$\rightarrow$"
if post_cell_type is not None:
    post_title += f"{post_cell_type}"
elif post_broad_type is not None:
    post_title += f"{post_broad_type}"
post_title += " in column"

# bin the data
mids = (bins[:-1] + bins[1:]) / 2 / 1000
all_pre_synapses[distance_bin_column] = pd.cut(
    all_pre_synapses[distance_column],
    bins=bins,
    labels=mids,
    include_lowest=True,
)
all_post_synapses[distance_bin_column] = pd.cut(
    all_post_synapses[distance_column],
    bins=bins,
    labels=mids,
    include_lowest=True,
)

# pre side
query_str = ""
if pre_broad_type is not None:
    query_str += f"pre_broad_type == '{pre_broad_type}'"
if pre_cell_type is not None:
    if query_str != "":
        query_str += " and "
    query_str += f"pre_cell_type == '{pre_cell_type}'"
if post_broad_type is not None:
    if query_str != "":
        query_str += " and "
    query_str += f"post_broad_type == '{post_broad_type}'"
if post_cell_type is not None:
    if query_str != "":
        query_str += " and "
    query_str += f"post_cell_type == '{post_cell_type}'"
if tags is not None:
    if query_str != "":
        query_str += " and "
    query_str += f"{agg_column} in {tags}"

query_pre_synapses = all_pre_synapses.query(query_str)
pre_agg_by_distance = (
    query_pre_synapses.groupby(distance_bin_column)[agg_column]
    .value_counts(dropna=False, normalize=normalize)
    .rename("val")
    .reset_index()
)
if include_total:
    pre_agg_by_distance_all = (
        pre_agg_by_distance.groupby(distance_bin_column)["val"].sum().reset_index()
    )
    pre_agg_by_distance_all[agg_column] = "total"
    pre_agg_by_distance = pd.concat(
        [pre_agg_by_distance, pre_agg_by_distance_all], ignore_index=True
    )

fig, axs = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
ax = axs[0]
fig, ax = plt.subplots(1, 1, figsize=(8.28, 7.17))
sns.lineplot(
    data=pre_agg_by_distance,
    x=distance_bin_column,
    y="val",
    hue=agg_column,
    hue_order=["single_spine", "multi_spine"],
    palette=palette,
    style=agg_column,
    style_order=["single_spine", "multi_spine"],
    ax=ax,
)
ax.set(ylabel=ylabel, xlabel=xlabel, title=pre_title)
legend = ax.get_legend()
# legend.set_title("Target")
handles, labels = ax.get_legend_handles_labels()

new_labels = ["Single-input spine", "Multi-input spine"]
ax.legend(handles, new_labels)
sns.move_legend(ax, "upper right", bbox_to_anchor=(1.0, 1.00), title="Target")
ax.set(title="")

save_matplotlib_figure(
    fig, "basket_cell_multispine_targeting_by_space", figure_out_path
)

# %%


# %% COMPUTE BC POST DISTANCE 95TH PERCENTILE
q = all_pre_synapses.query('pre_cell_type=="BC"')["post_euc_distance_to_nuc"].quantile(
    0.95
)

save_variables(
    prefix="column_summary_", bc_post_euc_distance_to_nuc_q95=q / 1000, format="{:.0f}"
)


# %% VISUALIZE BC CONNECTIONS IN NEUROGLANCER


query_pre_synapses = all_pre_synapses.query(
    "pre_cell_type == 'BC' and post_broad_type == 'excitatory'"
)
spine_connections = (
    query_pre_synapses.groupby(["pre_pt_root_id", "post_pt_root_id"])["tag_detailed"]
    .value_counts()
    .unstack()
    .sort_values(["single_spine"], ascending=False)
)

connection = spine_connections.index[0]

pre_syns = query_pre_synapses.query(
    f"(pre_pt_root_id == {connection[0]}) and (post_pt_root_id == {connection[1]})"
)

client = CAVEclient("minnie65_phase3_v1")
vs = ViewerState(client=client)
vs.add_layers_from_client().add_segments(connection).add_points(
    pre_syns,
    name="synapses",
    point_column=["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"],
    data_resolution=[4, 4, 40],
    tag_column="tag_detailed",
    swap_visible_segments_on_move=False,
)
vs.to_browser(shorten=True)

# %% ANALYZE BC CELL TARGETING PATTERNS
x = "pre_to_exc_spine_synapses"
x = "pre_to_exc_single_spine_synapses"
x = "pre_to_exc_multi_spine_synapses"
# x = "pre_to_exc_p_multi_spine_synapse"
x = "pre_to_exc_p_single_spine_synapse"
x = "pre_to_exc_p_spine_synapse"
x = "pre_to_exc_p_spine_synapse_is_multi"
y = "post_p_spine_synapse"
# y = 'post_p_spine_site_is_multi'
# y = "pt_position_um_y"


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=cell_info.query("cell_type == 'BC' and axon_cleaned and in_column"),
    x=x,
    y=y,
    ax=ax,
    size="pre_to_exc_total_synapses",
)
sns.move_legend(ax, "upper left", title="Synapse to E", bbox_to_anchor=(1.0, 1))
ax.invert_yaxis()

# %% CONSTRUCT QUERY STRING FOR ANALYSIS
# post side

query_str = ""
if post_broad_type is not None:
    if query_str != "":
        query_str += " and "
    query_str += f"post_broad_type == '{post_broad_type}'"
if post_cell_type is not None:
    if query_str != "":
        query_str += " and "
    query_str += f"post_cell_type == '{post_cell_type}'"
if tags is not None:
    if query_str != "":
        query_str += " and "
    query_str += f"{agg_column} in {tags}"

query_post_synapses = all_post_synapses.query(query_str)

post_agg_by_distance = (
    query_post_synapses.groupby(distance_bin_column)[agg_column]
    .value_counts(dropna=False, normalize=normalize)
    .rename("val")
    .reset_index()
)
if include_total:
    post_agg_by_distance_all = (
        post_agg_by_distance.groupby(distance_bin_column)["val"].sum().reset_index()
    )
    post_agg_by_distance_all[agg_column] = "total"
    post_agg_by_distance = pd.concat(
        [post_agg_by_distance, post_agg_by_distance_all], ignore_index=True
    )

ax = axs[1]
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.lineplot(
    data=post_agg_by_distance,
    x=distance_bin_column,
    y="val",
    hue=agg_column,
    palette=palette,
    hue_order=plot_tags,
    ax=ax,
    legend=False,
)
ax.set(ylabel=ylabel, xlabel=xlabel, title=post_title)

# %% SETUP SPINE MORPHOMETRY ANALYSIS
# cloud_morphometry_path = (
#     "gs://bdp-ssa/minnie65_phase3_v1/absolute-solo-yak/spine_morphometry_deltalake"
# )

# spine_size_table = pl.scan_delta(cloud_morphometry_path).select(
#     [
#         "post_pt_root_id",
#         "component_id",
#         "size_nm3",
#     ]
# )
# spine_size_table.collect_schema()


# %% ANALYZE EXCITATORY SPINE SIZE DISTRIBUTIONS, GAUSSIAN MIXTURE MODEL FITTING
query_synapses = synapses.query(
    "tag == 'spine' and post_broad_type == 'excitatory' and pre_axon_cleaned and pre_compartment == 'axon' and post_compartment == 'dendrite' and pre_broad_type == 'excitatory'"
)


def fit_and_plot_gmm(x, ax, color="red"):
    x = x[~np.isnan(x)]
    x = x.reshape(-1, 1)
    bics = []
    gmms = []
    train_indices, test_indices = np.split(
        np.random.permutation(x.shape[0]), [int(0.6 * x.shape[0])]
    )
    x_train = x[train_indices]
    x_test = x[test_indices]
    for n_components in range(1, 4):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=0,
            init_params="k-means++",
            n_init=25,
            max_iter=200,
            tol=1e-4,
            # random_state=42,
        )
        gmm.fit(np.log(x_train))
        bic = gmm.bic(np.log(x_test))
        bics.append(bic)
        gmms.append(gmm)

    n_components = np.argmin(bics) + 1
    gmm = gmms[n_components - 1]

    x_range = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
    y_gmm = gmm.score_samples(np.log(x_range))
    y_gmm = np.exp(y_gmm)  # Convert log probabilities to probabilities
    ax.plot(
        x_range,
        y_gmm * 2.25,  # Scale to match histogram
        # color="red",
        linewidth=2,
        color=color,
        # label="GMM Fit",
    )

    # show each of the components of the mixture model
    for i in range(n_components):
        mean = gmm.means_[i][0]
        std = np.sqrt(gmm.covariances_[i][0])
        y_component = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((np.log(x_range) - mean) / std) ** 2
        )
        y_component = y_component * gmm.weights_[i]
        ax.plot(
            x_range,
            y_component * 2.25,  # Scale to match histogram
            linestyle="--",
            linewidth=1,
            color=color,
        )


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(
    data=query_synapses.query("tag_detailed == 'single_spine'"),
    x="spine_size_nm3",
    ax=ax,
    log_scale=True,
    element="poly",
    stat="density",
    common_norm=False,
    color="pink",
)

fit_and_plot_gmm(
    query_synapses.query("tag_detailed == 'single_spine'")["spine_size_nm3"].values,
    ax,
    color="pink",
)

sns.histplot(
    data=query_synapses.query("tag_detailed == 'multi_spine'"),
    x="spine_size_nm3",
    ax=ax,
    log_scale=True,
    element="poly",
    stat="density",
    common_norm=False,
    color="purple",
)
fit_and_plot_gmm(
    query_synapses.query("tag_detailed == 'multi_spine'")["spine_size_nm3"].values,
    ax,
    color="purple",
)


# %% SETUP NEUROGLANCER VISUALIZATION

if False:
    vs = ViewerState(client=client)
    state = (
        vs.add_layers_from_client(skeleton_source=False, alpha_3d=0.9)
        .add_points(
            put_soma_spines.query("pre_proofread"),
            name="putative soma spines",
            point_column=[
                "ctr_pt_position_x",
                "ctr_pt_position_y",
                "ctr_pt_position_z",
            ],
            description_column="pre_cell_type",
            segment_column="segments",
        )
        .set_viewer_properties(layout="3d")
        .to_browser(browser="firefox")
    )


# %% ANALYZE THALAMIC-INHIBITORY DOUBLE INPUTS

if False:
    indices = count_table.query(
        "excitatory == 0 & inhibitory == 1 & thalamic == 1 & post_broad_type == 'inhibitory'"
    ).index

    double_df = (
        multi_synapses.set_index(["post_pt_root_id", "component_id"])
        .loc[indices]
        .reset_index()
    )
    double_df = double_df.groupby(["post_pt_root_id", "component_id"]).agg(
        {
            "ctr_pt_position_x": "mean",
            "ctr_pt_position_y": "mean",
            "ctr_pt_position_z": "mean",
            "pre_pt_root_id": lambda x: list(x.unique()),
            # "post_pt_root_id": lambda x: list(x.unique()),
        }
    )
    double_df["pt_a"] = (
        multi_synapses.groupby(["post_pt_root_id", "component_id"])[
            ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
        ]
        .first(0)
        .apply(list, axis=1)
    )
    double_df["pt_b"] = (
        multi_synapses.groupby(["post_pt_root_id", "component_id"])[
            ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
        ]
        .last(1)
        .apply(list, axis=1)
    )

    double_df = double_df.reset_index()
    double_df["root_ids"] = double_df.apply(
        lambda row: np.unique(row["pre_pt_root_id"] + [row["post_pt_root_id"]]), axis=1
    )
    double_df["name"] = [
        # f"{row.pre_pt_root_id[0]}_{row.pre_pt_root_id[1]}"
        f"{row.post_pt_root_id}_{row.component_id}"
        for _, row in double_df.iterrows()
    ]

    target_url = (
        "https://cj-mesh-bounds-dot-neuroglancer-dot-seung-lab.ue.r.appspot.com/"
    )
    vs = ViewerState(client=client, target_url=target_url)
    state = (
        vs.add_layers_from_client(skeleton_source=False, alpha_3d=0.9)
        .add_lines(
            double_df,
            name="lines",
            point_a_column="pt_a",
            point_b_column="pt_b",
            segment_column="root_ids",
            description_column="name",
            tags=["singlehead", "y", "merge", "noncanonical", "nonspine"],
        )
        .set_viewer_properties(layout="3d")
        .to_dict()
    )
    state["layers"][1]["source"][0]["state"] = {
        "focusMeshCulling": True,
        "focusBoundingBoxSize": 12,
    }
    ViewerState(base_state=state, target_url=target_url).to_browser(browser="firefox")

# %% ANALYZE EXCITATORY-INHIBITORY DUAL INPUTS

query_count_table = subset_count_table.query(
    'post_broad_type == "excitatory" & excitatory == 1 & inhibitory == 1 & thalamic == 0'
)
query_components = query_count_table.index

query_multi_synapses = (
    multi_synapses.reset_index()
    .set_index(["post_pt_root_id", "component_id"])
    .loc[query_components]
)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

data = query_multi_synapses.pivot(
    columns="pre_broad_type",
    values="size",
)

sns.scatterplot(data=data, x="excitatory", y="inhibitory", s=15, ax=ax)
ax.set(xscale="log", yscale="log")


# %% ANALYZE SYNAPSE SIZE CORRELATIONS
fig, axs = plt.subplots(
    1,
    2,
    figsize=(12, 6),
    layout="constrained",
    sharex=True,
    sharey=True,
)

ax = axs[0]
data = query_multi_synapses.query('pre_broad_type == "excitatory"')
x = "size"
y = "spine_size_nm3"
sns.scatterplot(
    data=data,
    x=x,
    y=y,
    hue="pre_broad_type",
    palette=CELL_TYPE_PALETTE,
    hue_order=["excitatory", "inhibitory"],
    # hue_order=["23P", "4P", "5P-IT", "5P-ET", "5P-NP", "6P-IT", "6P-CT"],
    s=25,
    ax=ax,
    legend=False,
)
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Excitatory cleft size (vx)",
    ylabel="Spine volume (nm³)",
)
pearson_r, pearson_p = pearsonr(np.log(data[x]), np.log(data[y]))
spearman_rho, spearman_p = spearmanr(data[x], data[y])
save_variables(
    prefix="column_summary_",
    multi_synapse_excitatory_pearson_r=pearson_r,
    format="{:.2f}",
)
save_variables(
    prefix="column_summary_",
    multi_synapse_excitatory_pearson_p=pearson_p,
    format="{:.2e}",
)
mean_e_cleft_size = np.mean(data[x])
save_variables(
    prefix="column_summary_",
    multi_synapse_excitatory_mean_cleft_size=mean_e_cleft_size,
    format="{:.0f}",
)

text = f"Pearson r: {pearson_r:.2f}\nSpearman ρ: {spearman_rho:.2f}"
text = f"Pearson r: {pearson_r:.2f}"
ax.text(
    0.05,
    0.95,
    text,
    ha="left",
    va="top",
    transform=ax.transAxes,
    fontsize="medium",
)
# sns.move_legend(ax, loc="upper left", title="Pre cell type", bbox_to_anchor=(1, 1))

ax = axs[1]
data = query_multi_synapses.query('pre_broad_type == "inhibitory"')
sns.scatterplot(
    data=data,
    x=x,
    y=y,
    hue="pre_broad_type",
    hue_order=["excitatory", "inhibitory"],
    # hue_order=["BC", "MC", "BPC", "NGC"],
    palette=CELL_TYPE_PALETTE,
    s=25,
    ax=ax,
    legend=False,
)
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Inhibitory cleft size (vx)",
    ylabel="Spine volume (nm³)",
)
pearson_r, pearson_p = pearsonr(np.log(data[x]), np.log(data[y]))
spearman_rho, spearman_p = spearmanr(data[x], data[y])
save_variables(
    prefix="column_summary_",
    multi_synapse_inhibitory_pearson_r=pearson_r,
    format="{:.2f}",
)
save_variables(
    prefix="column_summary_",
    multi_synapse_inhibitory_pearson_p=pearson_p,
    format="{:.2f}",
)
mean_i_cleft_size = np.mean(data[x])
save_variables(
    prefix="column_summary_",
    multi_synapse_inhibitory_mean_cleft_size=mean_i_cleft_size,
    format="{:.0f}",
)

text = f"Pearson r: {pearson_r:.2f}\nSpearman ρ: {spearman_rho:.2f}"
text = f"Pearson r: {pearson_r:.2f}"

ax.text(
    0.05,
    0.95,
    text,
    ha="left",
    va="top",
    transform=ax.transAxes,
    fontsize="medium",
)
# sns.move_legend(ax, loc="upper left", title="Pre\ncell type", bbox_to_anchor=(1, 1))
save_matplotlib_figure(fig, "multi_synapse_size_vs_spine_volume", figure_out_path)


# %% ANALYZE PAIRED SYNAPSE SIZES
paired_size_data = query_multi_synapses.reset_index().pivot(
    index=["post_pt_root_id", "component_id"], columns="pre_broad_type", values="size"
)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=paired_size_data, x="inhibitory", y="excitatory", ax=ax, color="black", s=15
)
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Inhibitory cleft size (vx)",
    ylabel="Excitatory cleft size (vx)",
)
ax.plot([1e2, 1e5], [1e2, 1e5], color="dimgrey", linestyle="--", zorder=-1)

(paired_size_data["excitatory"] > paired_size_data["inhibitory"]).mean()

pearson_r, pearson_p = pearsonr(
    np.log(paired_size_data["excitatory"]), np.log(paired_size_data["inhibitory"])
)

save_variables(
    prefix="column_summary_",
    multi_synapse_inhibitory_excitatory_size_pearson_r=pearson_r,
    format="{:.2f}",
)
save_variables(
    prefix="column_summary_",
    multi_synapse_inhibitory_excitatory_size_pearson_p=pearson_p,
    format="{:.2f}",
)

save_matplotlib_figure(fig, "inh_vs_exc_multi_synapse_size", figure_out_path)

# %% ANALYZE EXCITATORY SPINE DATA WITH MULTI STATUS

data = (
    synapses.query("post_broad_type == 'excitatory'")
    .query("post_compartment == 'dendrite'")
    .query("pre_axon_cleaned")
    # .query("pre_in_selection")
    .query("post_in_selection")
    .query("pre_compartment == 'axon'")
    .query("pre_broad_type == 'excitatory'")
)
print(len(data))

y = "size"
x = "spine_size_nm3"
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(
    data=data,
    x=x,
    y=y,
    hue="spine_is_multi",
    s=2,
    alpha=0.5,
    ax=ax,
    linewidth=0,
)
ax.set(xscale="log", yscale="log")


# %% ANALYZE EXCITATORY SPINE SIZE CORRELATIONS


data = (
    synapses.query("post_broad_type == 'excitatory'")
    .query("post_compartment == 'dendrite'")
    .query("pre_axon_cleaned")
    # .query("pre_in_selection")
    .query("post_in_selection")
    .query("pre_compartment == 'axon'")
    .query("pre_broad_type == 'excitatory'")
)
data = data.join(
    spine_table.set_index(["post_pt_root_id", "component_id"])[["is_unitary_spine"]],
    how="inner",
    on=["post_pt_root_id", "component_id"],
)

print(len(data))

x = "size"
y = "spine_size_nm3"
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(
    data=data.query("~spine_is_multi"),
    x=x,
    y=y,
    s=2,
    alpha=0.7,
    ax=ax,
    linewidth=0,
    label="Putative single-input",
    color="dimgrey",
)
sns.scatterplot(
    data=query_multi_synapses.query('pre_broad_type == "excitatory"'),
    x=x,
    y=y,
    s=10,
    alpha=1,
    ax=ax,
    linewidth=0.2,
    label="Verified multi-input",
    color="firebrick",
)
ax.set(xscale="log", yscale="log", ylim=(0.5e7, 1e9), xlim=(0.15e3, 1e5))
ax.set(xlabel="Excitatory cleft size (vx)", ylabel="Spine volume (nm³)")
ax.legend(markerscale=2)
save_matplotlib_figure(fig, "spine_synapse_area_corr_by_multi", figure_out_path)


# %% PLOTS OF SYNAPSE LOCATION DISTRIBUTIONS

frac = 0.1

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))

fig, axs = plt.subplots(
    2,
    3,
    figsize=(12 * 1.2, 5 * 1.2),
    # layout="tight",
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [1, 0.5], "hspace": 0.05, "wspace": 0.05},
)
background_kws = dict(
    x="pt_position_um_x", y="pt_position_um_y", color="grey", s=0.5, alpha=0.25
)
foreground_kws = dict(
    x="transformed_um_x",
    y="transformed_um_y",
    s=0.2,
    # alpha=0.2,
    linewidth=0,
    hue="tag",
    palette=COMPARTMENT_PALETTE_MUTED_HEX,
    legend=False,
)
ax = axs[0, 0]
sns.scatterplot(data=cell_info, ax=ax, **background_kws)
sns.scatterplot(data=all_post_synapses.sample(frac=frac), ax=ax, **foreground_kws)
ax.set_aspect("equal", adjustable="box")
ax.set(ylabel="Depth (um)", title="Inputs to column neurons")

ax = axs[0, 1]
sns.scatterplot(data=cell_info, ax=ax, **background_kws)
sns.scatterplot(data=all_pre_synapses.sample(frac=frac), ax=ax, **foreground_kws)
ax.set_aspect("equal", adjustable="box")
ax.set(title="Outputs from column neurons")

ax = axs[0, 2]
sns.scatterplot(data=cell_info, ax=ax, **background_kws)
sns.scatterplot(data=column_column_synapses.sample(frac=frac), ax=ax, **foreground_kws)
ax.set_aspect("equal", adjustable="box")
ax.set(title="Within column connectivity")
ax.invert_yaxis()

background_kws["y"] = "pt_position_um_z"
foreground_kws["y"] = "transformed_um_z"
ax = axs[1, 0]
sns.scatterplot(data=cell_info, ax=ax, **background_kws)
sns.scatterplot(data=all_post_synapses.sample(frac=frac), ax=ax, **foreground_kws)
ax.set_aspect("equal", adjustable="box")
ax.set(ylabel="Z (um)", xlabel="X (um)")

ax = axs[1, 1]
sns.scatterplot(data=cell_info, ax=ax, **background_kws)
sns.scatterplot(data=all_pre_synapses.sample(frac=frac), ax=ax, **foreground_kws)
ax.set_aspect("equal", adjustable="box")
ax.set(xlabel="X (um)")


ax = axs[1, 2]
sns.scatterplot(data=cell_info, ax=ax, **background_kws)
sns.scatterplot(data=column_column_synapses.sample(frac=frac), ax=ax, **foreground_kws)
ax.set(xlabel="X (um)")

ax.set_aspect("equal", adjustable="box")

save_matplotlib_figure(fig, "synapse_summary_spatial", figure_out_path)
