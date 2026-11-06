# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from analysis import FIG_PATH, Stash, Tables, save_matplotlib_figure, set_matplotlib_theme
from analysis.models import MODEL_SPECS, RF_PARAMS, feature_columns, make_model
from analysis.products.features import add_distance_to_nucleus

OUT_PATH = FIG_PATH / "ablation"
ARMS = ["proposed", "no_cluster", "no_chunk", "no_simplify", "full_res_vertex"]
CATEGORIES = ["soma", "shaft", "spine"]
N_SPLITS = 10
SPEC = MODEL_SPECS["ssa_model_full"]
# Part of the stash key of the split counts. Bump it after a change to `root_splits`.
SPLITS_VERSION = 1

set_matplotlib_theme()
pd.set_option("display.width", 200)

# %% TIME, MEMORY AND STORAGE PER ROOT
runs = (
    Tables["ablation_runs"]
    .collect()
    .join(Tables["root_mesh_vertices"].collect(), on="root_id", how="left")
    .with_columns(
        peak_gib=pl.col("rss_peak_bytes") / 2**30,
        features_mb=pl.col("domain_features_bytes") / 1e6,
        storage_mb=(pl.col("domain_features_bytes") + pl.col("vertex_domains_bytes")) / 1e6,
    )
    .to_pandas()
)
print(
    runs.groupby("arm")[
        ["wall_s", "peak_gib", "storage_mb", "features_mb", "domain_features_rows"]
    ]
    .median()
    .reindex(ARMS)
    .round(1)
)
print(runs.groupby("arm").size().reindex(ARMS))


# %% SCORING
def arm_rows(arm: str) -> pd.DataFrame:
    """The labeled synapses of `arm`, with its domain features."""
    rows = (
        Tables["ablation_labeled_features"]
        .lazy()
        .filter(pl.col("arm") == arm)
        .sort("root_id", "synapse_id")
    )
    rows = add_distance_to_nucleus(rows, Tables["root_nuclei"].lazy())
    return rows.collect().to_pandas()


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, columns, seed: int):
    model = make_model(random_state=seed)
    model.fit(train[columns], train["tag"], sample_weight=train["sample_weight"])
    return model.predict(test[columns])


def root_splits(rows: pd.DataFrame, columns: list[str], fit_predict=fit_and_predict) -> pd.DataFrame:
    """Per split of the roots: the confusion counts of its test labels, one row per
    true and predicted label. The splits of the original Fig 2D."""
    roots = np.sort(rows["root_id"].unique())
    counts = []
    for fold in range(N_SPLITS):
        train_roots, test_roots = train_test_split(roots, test_size=0.2, random_state=fold*8888)
        train = rows[rows["root_id"].isin(train_roots)]
        test = rows[rows["root_id"].isin(test_roots)]
        predicted = fit_predict(train, test, columns, seed=fold*8888)
        matrix = pd.DataFrame(
            confusion_matrix(test["tag"], predicted, labels=CATEGORIES),
            index=pd.Index(CATEGORIES, name="true"),
            columns=pd.Index(CATEGORIES, name="predicted"),
        )
        counts.append(matrix.stack().rename("count").reset_index().assign(fold=fold))
    return pd.concat(counts, ignore_index=True)


counts_by_arm = {}
for arm in ARMS:
    splits = Stash(
        f"ablation/root_splits/{arm}",
        inputs=[Tables["ablation_labeled_features"], Tables["root_nuclei"]],
        spec=SPEC,
        forest=RF_PARAMS,
        n_splits=N_SPLITS,
        categories=CATEGORIES,
        version=SPLITS_VERSION,
    )
    if not splits.exists():
        rows = arm_rows(arm)
        columns = feature_columns(SPEC, rows.columns)
        # NOTE: a synapse whose vertex is in no domain has null features; the forest
        # takes missing values, as in the main model.
        print(arm, len(columns), "columns", f"{rows['domain_x'].isna().mean():.4f} of rows with no domain")
        splits.save(root_splits(rows, columns))
    counts_by_arm[arm] = splits.load().to_pandas().assign(arm=arm)
counts = pd.concat(counts_by_arm.values(), ignore_index=True)
scores = (
    counts.assign(correct=counts["count"].where(counts["true"] == counts["predicted"], 0))
    .groupby(["arm", "fold"])[["correct", "count"]]
    .sum()
    .eval("accuracy = correct / count")
    .reset_index()
)

# %% EACH ARM AGAINST PROPOSED, PAIRED BY SPLIT, AS IN THE ORIGINAL FIG 2D
paired = scores.pivot(index="fold", columns="arm", values="accuracy")[ARMS]
print(paired.round(4))
summary = pd.DataFrame(
    {
        "mean_test_accuracy": paired.mean(),
        "minus_proposed": paired.sub(paired["proposed"], axis=0).mean(),
    }
)
for arm in ARMS[1:]:
    result = wilcoxon(paired[arm], paired["proposed"])
    summary.loc[arm, ["wilcoxon_statistic", "wilcoxon_pvalue"]] = [
        result.statistic,
        result.pvalue,
    ]
print(summary.round(4))

# %% FIGURES
palette = dict(zip(ARMS, sns.color_palette("Set2", len(ARMS))))

fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(
    data=runs, x="n_vertices", y="wall_s", hue="arm", hue_order=ARMS, palette=palette, ax=ax
)
ax.set(xlabel="Input mesh vertices", ylabel="Wall time per neuron (s)", yscale="log")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Arm")
save_matplotlib_figure(fig, "time_by_size", out_path=OUT_PATH)

fig, ax = plt.subplots(figsize=(5, 5))
sns.stripplot(data=runs, x="arm", y="storage_mb", order=ARMS, hue="arm", palette=palette, ax=ax)
ax.set(xlabel="", ylabel="Storage per neuron (MB)", yscale="log")
ax.tick_params(axis="x", rotation=45)
save_matplotlib_figure(fig, "storage", out_path=OUT_PATH)

long = paired.reset_index().melt(id_vars="fold", var_name="arm", value_name="accuracy")
for name, ylim in [("accuracy", (0, 1)), ("accuracy_zoom", None)]:
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(
        data=long, x="arm", y="accuracy", units="fold", estimator=None,
        color="lightgrey", ax=ax, sort=False,
    )
    sns.stripplot(
        data=long, x="arm", y="accuracy", order=ARMS, hue="arm", palette=palette,
        jitter=False, ax=ax,
    )
    sns.pointplot(
        data=long, x="arm", y="accuracy", order=ARMS, errorbar=None, markers="_",
        markersize=30, linestyles="none", color="black", ax=ax,
    )
    ax.set(xlabel="", ylabel="Test accuracy", ylim=ylim)
    ax.tick_params(axis="x", rotation=45)
    save_matplotlib_figure(fig, name, out_path=OUT_PATH)

matrix = (
    counts[counts["arm"] == "proposed"]
    .pivot_table(index="true", columns="predicted", values="count", aggfunc="sum")
    .reindex(index=CATEGORIES, columns=CATEGORIES)
)
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(
    matrix.div(matrix.sum(axis=1), axis=0), annot=True, fmt=".2f", cmap="Greys",
    cbar=False, square=True, ax=ax,
)
ax.set(
    xlabel="Predicted label", ylabel="True label", title=f"Proposed, summed over {N_SPLITS} splits"
)
save_matplotlib_figure(fig, "confusion_proposed", out_path=OUT_PATH)


# %% FIGURES IN THE STYLE OF THE ORIGINAL FIG 2B-C
# NOTE: full_res_vertex (simplify and cluster off, chunking on) matches the original
# "naive", which stored one HKS vector per vertex of the full mesh.
LABELS = {
    "proposed": "Proposed",
    "no_cluster": "No cluster",
    "no_chunk": "No chunk",
    "no_simplify": "No simplify",
    "full_res_vertex": "No simplify,\nno cluster",
}
# The order of the arms, left to right, in every panel: the ablations by mean wall time,
# longest first, then proposed.
ORDER = ["no_chunk", "no_simplify", "full_res_vertex", "no_cluster", "proposed"]
# One color per arm in every panel. GREYS puts proposed in black and the others in grey.
GREYS = {
    "proposed": "#000000",
    "no_chunk": "#4D4D4D",
    "full_res_vertex": "#7A7A7A",
    "no_simplify": "#A3A3A3",
    "no_cluster": "#C4C4C4",
}
COLORS = dict(zip(ORDER[:-1], sns.color_palette("Set2"))) | {"proposed": "#000000"}


def method_strip(
    data: pd.DataFrame, y: str, arms: list[str], ylabel: str, name: str, yscales=("linear",)
) -> None:
    """One dot per row of `data` and a line at the mean of each arm, as in the original
    Fig 2B-D. The arms are drawn left to right in the order given. A log panel is saved
    as `{name}_log`."""
    order = [LABELS[arm] for arm in arms]
    colors = {LABELS[arm]: COLORS[arm] for arm in arms}
    data = data[data["arm"].isin(arms)].assign(method=lambda df: df["arm"].map(LABELS))
    means = data.groupby("method", as_index=False)[y].mean()

    for yscale in yscales:
        fig, ax = plt.subplots(figsize=(0.75 * len(arms) + 1.5, 6))
        sns.stripplot(
            data=data, x="method", y=y, hue="method", order=order, palette=colors,
            linewidth=0, size=5, jitter=0.25, ax=ax,
        )
        sns.stripplot(
            data=means, x="method", y=y, hue="method", order=order, palette=colors,
            marker="_", linewidth=3, size=50, zorder=-1, jitter=False, ax=ax,
        )
        ax.set(xlabel="Method", ylabel=ylabel, yscale=yscale)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        suffix = "" if yscale == "linear" else f"_{yscale}"
        save_matplotlib_figure(fig, name + suffix, out_path=OUT_PATH)

    arm_means = data.groupby("arm")[y].mean().reindex(arms)
    print(
        pd.DataFrame(
            {"mean": arm_means, "ratio_of_means_to_proposed": arm_means / arm_means["proposed"]}
        )
    )


method_strip(
    runs, "wall_s", ORDER, "Total time per neuron (s)", "time_comparison",
    yscales=("linear", "log"),
)
method_strip(
    runs, "storage_mb", ORDER, "Storage size per neuron (MB)", "storage_comparison",
    yscales=("linear", "log"),
)
# One dot per split.
method_strip(long, "accuracy", ORDER, "Test accuracy", "accuracy_comparison")

#%%
# MINI COMPOSITE OF FIG 2B-D, NOT SAVED
# NOTE: cairosvg needs the Homebrew cairo library, as in scripts/composites.
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:" + os.environ.get(
    "DYLD_LIBRARY_PATH", ""
)
from panel_mosaic import PanelMosaic  # noqa: E402

for suffix in ["", "_log"]:
    PanelMosaic(
        "BBCCDD",
        panel_mapping={
            "B": OUT_PATH / f"time_comparison{suffix}.svg",
            "C": OUT_PATH / f"storage_comparison{suffix}.svg",
            "D": OUT_PATH / "accuracy_comparison.svg",
        },
        figsize=(15, 6),
        layout="constrained",
        label_fontsize=20,
        panel_borders=False,
    ).show()


# %% MLP ON THE SAME SPLITS, AGAINST THE FOREST
# NOTE: the MLP does not take missing values, so a null feature is set to the median
# of the training rows, with an indicator column per feature that has nulls. The forest
# has class_weight="balanced"; the MLP has no class weights, so the same balanced
# weights multiply its sample weights.
MLP_PARAMS = dict(hidden_layer_sizes=(256, 128), alpha=1e-4, early_stopping=True, max_iter=500)


def fit_and_predict_mlp(train: pd.DataFrame, test: pd.DataFrame, columns, seed: int):
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        MLPClassifier(**MLP_PARAMS, random_state=seed),
    )
    weight = compute_sample_weight("balanced", train["tag"]) * train["sample_weight"]
    model.fit(train[columns], train["tag"], mlpclassifier__sample_weight=weight)
    return model.predict(test[columns])


mlp_counts_by_arm = {}
for arm in ARMS:
    splits = Stash(
        f"ablation/mlp_root_splits/{arm}",
        inputs=[Tables["ablation_labeled_features"], Tables["root_nuclei"]],
        spec=SPEC,
        mlp=MLP_PARAMS,
        n_splits=N_SPLITS,
        categories=CATEGORIES,
        version=SPLITS_VERSION,
    )
    if not splits.exists():
        rows = arm_rows(arm)
        columns = feature_columns(SPEC, rows.columns)
        splits.save(root_splits(rows, columns, fit_predict=fit_and_predict_mlp))
    mlp_counts_by_arm[arm] = splits.load().to_pandas().assign(arm=arm)
mlp_counts = pd.concat(mlp_counts_by_arm.values(), ignore_index=True)
mlp_scores = (
    mlp_counts.assign(
        correct=mlp_counts["count"].where(mlp_counts["true"] == mlp_counts["predicted"], 0)
    )
    .groupby(["arm", "fold"])[["correct", "count"]]
    .sum()
    .eval("accuracy = correct / count")
    .reset_index()
)

# Each arm's MLP against its forest, paired by split.
mlp_paired = mlp_scores.pivot(index="fold", columns="arm", values="accuracy")[ARMS]
print(mlp_paired.round(4))
mlp_summary = pd.DataFrame(
    {
        "mean_test_accuracy_forest": paired.mean(),
        "mean_test_accuracy_mlp": mlp_paired.mean(),
        "mlp_minus_forest": mlp_paired.sub(paired).mean(),
    }
)
for arm in ARMS:
    result = wilcoxon(mlp_paired[arm], paired[arm])
    mlp_summary.loc[arm, ["wilcoxon_statistic", "wilcoxon_pvalue"]] = [
        result.statistic,
        result.pvalue,
    ]
print(mlp_summary.round(4))
