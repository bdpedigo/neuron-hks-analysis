# %%
"""Does training on a few H01 labels help on held-out H01 data?

Ported from meshrep/clean_paper_code/retrain_h01.py. Each fold trains on the
MICrONS training rows, with or without the H01 retrain rows of the other folds,
and scores the held-out H01 retrain rows. Two splits: by neuron (experiment 1)
and by synapse, stratified on the label (experiment 2).

Reads only `model_inputs` (MICrONS) and `h01_retrain_model_inputs` (H01), so the
H01 validation cohort stays held out. The features, the nucleus mask and the
forest are those of the main model (`analysis.models`).
"""

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm

from analysis import (
    Tables,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)
from analysis.models import (
    MAIN_MODEL,
    MODEL_SPECS,
    make_model,
    nucleus_masked_roots,
    select_spectral_columns,
)

SUBFOLDER = "h01_retrain"
LABELS = ["soma", "shaft", "spine"]
H01_SAMPLE_WEIGHT = 1000
N_SPLITS = 10
SPLIT_SEED = 42

# Which rows each mode trains on, and whether it applies frustratingly easy domain
# adaptation (feature augmentation, Daumé 2007). The test rows are always H01.
TRAIN_MODES = {
    "microns_only": dict(datastacks=("microns",), feda=False),
    "both_upweighted": dict(datastacks=("h01", "microns"), feda=False),
    # "h01_only": dict(datastacks=("h01",), feda=False),
    # "both_feda": dict(datastacks=("h01", "microns"), feda=True),
}
MODE_NAMES = {
    "microns_only": "Train on\nMICrONS only",
    "h01_only": "Train on\nH01 only",
    "both_upweighted": "Train on both\n(H01 upweighted)",
    "both_feda": "Train on both\n(FEDA)",
}
BASELINE_MODE, COMPARE_MODE = "microns_only", "both_upweighted"

set_matplotlib_theme()

# %%
microns = Tables["model_inputs"].collect()
spectral_columns = select_spectral_columns(microns.columns)
feature_columns = spectral_columns + MODEL_SPECS[MAIN_MODEL]["extra_features"]

# The MICrONS rows as train.py builds them: rows with domain features, and the
# nucleus mask by root.
microns = (
    microns.drop_nulls(spectral_columns)
    .with_columns(
        distance_to_nucleus=pl.when(
            pl.col("root_id").is_in(
                nucleus_masked_roots(microns.drop_nulls(spectral_columns)["root_id"])
            )
        )
        .then(None)
        .otherwise(pl.col("distance_to_nucleus")),
        datastack=pl.lit("microns"),
        sample_weight=pl.col("sample_weight").cast(pl.Float64),
    )
    .select(
        "root_id",
        "synapse_id",
        pl.col("tag").alias("label"),
        "datastack",
        "sample_weight",
        *feature_columns,
    )
)
h01 = (
    Tables["h01_retrain_model_inputs"]
    .collect()
    .drop_nulls(spectral_columns)
    .with_columns(
        datastack=pl.lit("h01"), sample_weight=pl.lit(H01_SAMPLE_WEIGHT, pl.Float64)
    )
    .select(
        "root_id", "synapse_id", "label", "datastack", "sample_weight", *feature_columns
    )
)
assert set(microns["label"].unique()) == set(h01["label"].unique()) == set(LABELS)
h01_roots = np.unique(h01["root_id"].to_numpy())
print(
    f"MICrONS: {len(microns):,} synapses. H01 retrain: {len(h01):,} synapses on {len(h01_roots)} roots"
)

save_variables(
    h01_retrain_n_roots=len(h01_roots),
    h01_retrain_n_synapses=len(h01),
    h01_retrain_sample_weight=H01_SAMPLE_WEIGHT,
    format="{:,}",
)


# %%
def feda_augment(X, domains, source="microns", target="h01"):
    # A general copy and one copy per domain: a target row becomes [X, 0, X].
    is_source = (domains == source)[:, None]
    is_target = (domains == target)[:, None]
    return np.hstack([X, np.where(is_source, X, 0.0), np.where(is_target, X, 0.0)])


def get_xyw(df: pl.DataFrame, feda: bool):
    X = df.select(feature_columns).to_numpy()
    if feda:
        X = feda_augment(X, df["datastack"].to_numpy())
    return X, df["label"].to_numpy(), df["sample_weight"].to_numpy()


def run_folds(splits, train_h01_of, test_h01_of) -> pd.DataFrame:
    """Score every mode on every fold. `train_h01_of(fold)` and `test_h01_of(fold)`
    give the H01 rows the fold trains and tests on."""
    rows = []
    for fold, split in enumerate(tqdm(splits)):
        test_df = test_h01_of(split)
        X_test, y_test, _ = get_xyw(test_df, feda=False)
        for mode, config in TRAIN_MODES.items():
            parts = []
            if "h01" in config["datastacks"]:
                parts.append(train_h01_of(split))
            if "microns" in config["datastacks"]:
                parts.append(microns)
            X_train, y_train, w_train = get_xyw(pl.concat(parts), feda=config["feda"])
            if config["feda"]:
                X_test_mode, _, _ = get_xyw(test_df, feda=True)
            else:
                X_test_mode = X_test
            model = make_model(random_state=13294587 * fold)
            model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = model.predict(X_test_mode)
            rows.append(
                {
                    "fold": fold,
                    "train_mode": mode,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
                    "macro_f1": f1_score(y_test, y_pred, average="macro"),
                    "conf_mat": confusion_matrix(y_test, y_pred, labels=LABELS),
                }
            )
    return pd.DataFrame(rows)


# %%
# Experiment 1: split by neuron, so no test neuron is in the training rows.
neuron_splits = list(
    KFold(N_SPLITS, shuffle=True, random_state=SPLIT_SEED).split(h01_roots)
)
neuron_results = run_folds(
    neuron_splits,
    train_h01_of=lambda split: h01.filter(pl.col("root_id").is_in(h01_roots[split[0]])),
    test_h01_of=lambda split: h01.filter(pl.col("root_id").is_in(h01_roots[split[1]])),
)

# %%
# Experiment 2: split by synapse, stratified on the label.
synapse_splits = list(
    StratifiedKFold(N_SPLITS, shuffle=True, random_state=SPLIT_SEED).split(
        np.zeros(len(h01)), h01["label"].to_numpy()
    )
)
synapse_results = run_folds(
    synapse_splits,
    train_h01_of=lambda split: h01[split[0].tolist()],
    test_h01_of=lambda split: h01[split[1].tolist()],
)

# %%
EXPERIMENTS = {
    "neuron_split": (neuron_results, "neurons"),
    "synapse_split": (synapse_results, "synapses"),
}
METRIC = "macro_f1"


def plot_folds(results: pd.DataFrame, held_out: str):
    modes = list(TRAIN_MODES)
    x_base = {mode: i for i, mode in enumerate(modes)}
    jitter = 0.1
    rng = np.random.default_rng(42)
    results = results.assign(
        x=results["train_mode"].map(x_base) + rng.uniform(-jitter, jitter, len(results))
    )
    fig, ax = plt.subplots(figsize=(2 * len(modes), 6))
    for _, fold_rows in results.groupby("fold"):
        fold_rows = fold_rows.sort_values("x")
        ax.plot(fold_rows["x"], fold_rows[METRIC], color="gray", alpha=0.7, zorder=0)
    for mode in modes:
        mode_rows = results.query("train_mode == @mode")
        ax.scatter(mode_rows["x"], mode_rows[METRIC], s=30, zorder=1)
        mean = mode_rows[METRIC].mean()
        ax.plot(
            [x_base[mode] - jitter - 0.05, x_base[mode] + jitter + 0.05],
            [mean, mean],
            color="black",
            zorder=2,
        )
    ax.set_xticks(list(x_base.values()), [MODE_NAMES[mode] for mode in modes])
    ax.set_ylabel(f"Macro F1 on held-out H01 {held_out}")
    return fig


for experiment, (results, held_out) in EXPERIMENTS.items():
    summary = results.groupby("train_mode")[METRIC].agg(["mean", "std"])
    print(experiment)
    print(
        results.groupby("train_mode")[["accuracy", "weighted_f1", "macro_f1"]]
        .agg(["mean", "std"])
        .round(3)
    )

    baseline = results.query("train_mode == @BASELINE_MODE").sort_values("fold")[METRIC]
    compare = results.query("train_mode == @COMPARE_MODE").sort_values("fold")[METRIC]
    stat, p_value = wilcoxon(baseline.to_numpy(), compare.to_numpy())
    print(
        f"Wilcoxon ({BASELINE_MODE} vs {COMPARE_MODE}): statistic={stat}, p={p_value:.3g}"
    )

    save_variables(
        prefix=f"h01_retrain_{experiment}_",
        **{
            f"{mode}_{METRIC}_{stat_name}": value
            for mode, row in summary.iterrows()
            for stat_name, value in row.items()
        },
        format="{:.3f}",
    )
    save_variables(
        prefix=f"h01_retrain_{experiment}_", wilcoxon_p=p_value, format="{:.2g}"
    )

    save_matplotlib_figure(
        plot_folds(results, held_out),
        f"retrain_{experiment}_{METRIC}",
        subfolder=SUBFOLDER,
    )

    # The held-out confusion matrices, summed over folds and row normalized.
    fig, axs = plt.subplots(
        1, len(TRAIN_MODES), figsize=(4 * len(TRAIN_MODES), 4), layout="constrained"
    )
    for ax, mode in zip(np.atleast_1d(axs), TRAIN_MODES):
        summed = np.sum(
            np.stack(results.query("train_mode == @mode")["conf_mat"]), axis=0
        )
        confusion = pd.DataFrame(summed, index=LABELS, columns=LABELS)
        confusion = confusion.div(confusion.sum(axis=1), axis=0)
        sns.heatmap(
            confusion,
            ax=ax,
            square=True,
            cmap="Greys",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar=False,
        )
        ax.set(xlabel="Predicted label", ylabel="Manual label", title=MODE_NAMES[mode])
    save_matplotlib_figure(
        fig, f"retrain_{experiment}_confusion_matrices", subfolder=SUBFOLDER
    )
