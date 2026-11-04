# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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


def root_splits(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per split of the roots: the confusion counts of its test labels, one row per
    true and predicted label. The splits of the original Fig 2D."""
    roots = np.sort(rows["root_id"].unique())
    counts = []
    for fold in range(N_SPLITS):
        train_roots, test_roots = train_test_split(roots, test_size=0.2, random_state=fold*8888)
        train = rows[rows["root_id"].isin(train_roots)]
        test = rows[rows["root_id"].isin(test_roots)]
        predicted = fit_and_predict(train, test, columns, seed=fold*8888)
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
