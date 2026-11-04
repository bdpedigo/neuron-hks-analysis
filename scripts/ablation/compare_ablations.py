# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from analysis import FIG_PATH, Tables, save_matplotlib_figure, set_matplotlib_theme
from analysis.models import MODEL_SPECS, feature_columns, make_model
from analysis.products.features import add_distance_to_nucleus

OUT_PATH = FIG_PATH / "ablation"
ARMS = ["proposed", "no_cluster", "no_chunk", "no_simplify", "full_res_vertex"]
CATEGORIES = ["soma", "shaft", "spine"]
N_SPLITS = 10
SPEC = MODEL_SPECS["ssa_model_full"]

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


def root_splits(rows: pd.DataFrame, columns: list[str]):
    """Per split of the roots: the accuracy pooled over its test labels, and the
    confusion counts summed over splits. The splits of the original Fig 2D."""
    roots = np.sort(rows["root_id"].unique())
    scores = []
    matrix = np.zeros((3, 3), dtype=int)
    for fold in range(N_SPLITS):
        train_roots, test_roots = train_test_split(roots, test_size=0.2, random_state=fold)
        train = rows[rows["root_id"].isin(train_roots)]
        test = rows[rows["root_id"].isin(test_roots)]
        predicted = fit_and_predict(train, test, columns, seed=fold)
        scores.append({"fold": fold, "accuracy": (predicted == test["tag"].to_numpy()).mean()})
        matrix += confusion_matrix(test["tag"], predicted, labels=CATEGORIES)
    return pd.DataFrame(scores), matrix


def leave_one_cell_out(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per held-out cell: its accuracy."""
    scores = []
    for fold, held_out in enumerate(sorted(rows["root_id"].unique())):
        train = rows[rows["root_id"] != held_out]
        test = rows[rows["root_id"] == held_out]
        predicted = fit_and_predict(train, test, columns, seed=fold)
        scores.append(
            {
                "root_id": held_out,
                "accuracy": (predicted == test["tag"].to_numpy()).mean(),
                "n_labels": len(test),
            }
        )
    return pd.DataFrame(scores)


scores_by_arm = {}
cell_scores_by_arm = {}
matrices = {}
for arm in ARMS:
    rows = arm_rows(arm)
    columns = feature_columns(SPEC, rows.columns)
    # NOTE: a synapse whose vertex is in no domain has null features; the forest
    # takes missing values, as in the main model.
    print(arm, f"{rows['domain_x'].isna().mean():.4f} of rows with no domain")
    scores, matrices[arm] = root_splits(rows, columns)
    scores_by_arm[arm] = scores.assign(arm=arm)
    cell_scores_by_arm[arm] = leave_one_cell_out(rows, columns).assign(arm=arm)
    print(arm, len(columns), "columns", f"mean test accuracy {scores['accuracy'].mean():.4f}")
scores = pd.concat(scores_by_arm.values(), ignore_index=True)
cell_scores = pd.concat(cell_scores_by_arm.values(), ignore_index=True)

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

# %% SIDE TABLE: PER HELD-OUT CELL, THE MEAN OVER CELLS AND THE ACCURACY POOLED OVER LABELS
cell_scores["correct"] = cell_scores["accuracy"] * cell_scores["n_labels"]
by_arm = cell_scores.groupby("arm")
print(
    pd.DataFrame(
        {
            "mean_over_cells": by_arm["accuracy"].mean(),
            "pooled_over_labels": by_arm["correct"].sum() / by_arm["n_labels"].sum(),
        }
    )
    .reindex(ARMS)
    .round(4)
)

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

matrix = pd.DataFrame(matrices["proposed"], index=CATEGORIES, columns=CATEGORIES)
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(
    matrix.div(matrix.sum(axis=1), axis=0), annot=True, fmt=".2f", cmap="Greys",
    cbar=False, square=True, ax=ax,
)
ax.set(
    xlabel="Predicted label", ylabel="True label", title=f"Proposed, summed over {N_SPLITS} splits"
)
save_matplotlib_figure(fig, "confusion_proposed", out_path=OUT_PATH)
