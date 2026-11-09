# %%
"""Does training on a few H01 labels help on held-out H01 data?

Ported from meshrep/clean_paper_code/retrain_h01.py. Each fold trains on the
MICrONS training rows, with or without the H01 retrain rows of the other folds,
and scores the held-out H01 retrain rows. Two splits: by neuron (experiment 1)
and by synapse, stratified on the label (experiment 2). Every model variant in
`MODEL_SPECS` runs, with its own features and labels. The glued variables and
the per-experiment figures are for the main model.

Reads only `model_inputs` (MICrONS) and `h01_retrain_model_inputs` (H01), so the
H01 validation cohort stays held out. The nucleus mask and the forest are those
of `train.py` (`analysis.models`).

The fold results of each variant are stashed (`analysis.Stash`) under a key of
the input tables and the fold settings, and a stashed result is read instead of
fit again. After a change to `run_folds`, bump `FOLDS_VERSION`.

The mixing curves (R3.2) fit the main model on a grid of H01 size, MICrONS size
and H01 weight, on the folds of the synapse split. Every fit is scored on the
held-out H01 fold and on a held-out set of MICrONS synapses. The fold scores and
their summary (the plot table) are stashed. After a change to
`run_mixing_fold`, bump `CURVE_VERSION`.

The diet curves are the one part that reads the H01 validation cohort, and only
to score. Each fit trains on a mix of H01 retrain and MICrONS synapses, at a fixed
H01 fraction and a total size, with all weights 1. The repeat scores and their
summary are stashed. After a change to `run_diet_repeat`, bump `DIET_VERSION`.
"""

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm.auto import tqdm

from analysis import (
    Stash,
    Tables,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)
from analysis.io import MODEL_PATH
from analysis.models import (
    MAIN_MODEL,
    MODEL_SPECS,
    MODEL_VERSION,
    RF_PARAMS,
    feature_columns,
    make_model,
    nucleus_masked_roots,
    save_model,
    select_spectral_columns,
)

SUBFOLDER = "h01_retrain"
# Part of the stash key of the fold results. Bump it after a change to `run_folds`.
FOLDS_VERSION = 1
# Part of the stash key of the mixing curves. Bump it after a change to
# `run_mixing_fold`.
CURVE_VERSION = 1
# Part of the stash key of the diet curves. Bump it after a change to
# `run_diet_repeat`.
DIET_VERSION = 1
LABELS = ["soma", "shaft", "spine"]
BINARY_LABELS = ["not_spine", "spine"]
H01_SAMPLE_WEIGHT = 1000
N_SPLITS = 10
SPLIT_SEED = 42
# The neuron split is off for now, to save time: the synapse split runs alone.
RUN_NEURON_SPLIT = False

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
METRIC = "macro_f1"

# The mixing grid. A size of None is all rows of the pool, and 0 is no rows.
MIXING_H01_SIZES = [250, 500, 1000, None]
MIXING_MICRONS_SIZES = [250, 500, 1000, 1780, 4000, 16000, 64000, None]
MIXING_H01_WEIGHTS = [1, 10, 100, 1000]
# (arm, H01 size, MICrONS size, H01 per-sample weight). One dataset alone needs
# no weight.
MIXING_CONFIGS = (
    [("h01_only", size, 0, None) for size in MIXING_H01_SIZES]
    + [("microns_only", 0, size, None) for size in MIXING_MICRONS_SIZES]
    + [
        ("both", None, size, weight)
        for weight in MIXING_H01_WEIGHTS
        for size in MIXING_MICRONS_SIZES
    ]
)
# The MICrONS synapses that every mixing fit is scored on, and none trains on.
MICRONS_TEST_SIZE = 2000
MICRONS_TEST_SEED = 7

# The diet grid: the H01 fraction of the training synapses, and their total. A
# total of None is all MICrONS rows, at fraction 0 only.
DIET_H01_FRACTIONS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
DIET_TOTAL_SIZES = [100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, None]
DIET_REPEATS = 5

set_matplotlib_theme()

# %%
microns = Tables["model_inputs"].collect()
spectral_columns = select_spectral_columns(microns.columns)
all_feature_columns = spectral_columns + ["distance_to_nucleus"]

# The MICrONS rows as train.py builds them: rows with domain features, and the
# nucleus mask by root.
microns = microns.drop_nulls(spectral_columns)
microns = microns.with_columns(
    distance_to_nucleus=pl.when(
        pl.col("root_id").is_in(nucleus_masked_roots(microns["root_id"]))
    )
    .then(None)
    .otherwise(pl.col("distance_to_nucleus")),
    datastack=pl.lit("microns"),
    sample_weight=pl.col("sample_weight").cast(pl.Float64),
).select(
    "root_id",
    "synapse_id",
    pl.col("tag").alias("label"),
    "datastack",
    "sample_weight",
    *all_feature_columns,
)
h01 = (
    Tables["h01_retrain_model_inputs"]
    .collect()
    .drop_nulls(spectral_columns)
    .with_columns(
        datastack=pl.lit("h01"), sample_weight=pl.lit(H01_SAMPLE_WEIGHT, pl.Float64)
    )
    .select(
        "root_id",
        "synapse_id",
        "label",
        "datastack",
        "sample_weight",
        *all_feature_columns,
    )
)
assert set(microns["label"].unique()) == set(h01["label"].unique()) == set(LABELS)
h01_roots = np.unique(h01["root_id"].to_numpy())
print(
    f"MICrONS: {len(microns):,} synapses. "
    f"H01 retrain: {len(h01):,} synapses on {len(h01_roots)} roots"
)

save_variables(
    h01_retrain_n_roots=len(h01_roots),
    h01_retrain_n_synapses=len(h01),
    h01_retrain_sample_weight=H01_SAMPLE_WEIGHT,
    format="{:,}",
)


# %%
def spec_labels(spec: dict) -> list[str]:
    return BINARY_LABELS if spec["binary"] else LABELS


def feda_augment(X, domains, source="microns", target="h01"):
    # A general copy and one copy per domain: a target row becomes [X, 0, X].
    is_source = (domains == source)[:, None]
    is_target = (domains == target)[:, None]
    return np.hstack([X, np.where(is_source, X, 0.0), np.where(is_target, X, 0.0)])


def get_xyw(df: pl.DataFrame, spec: dict, feda: bool):
    X = df.select(feature_columns(spec, spectral_columns)).to_numpy()
    if feda:
        X = feda_augment(X, df["datastack"].to_numpy())
    y = df["label"].to_numpy()
    if spec["binary"]:
        # As train.py does: soma and shaft merge into not_spine.
        y = np.where(y == "spine", "spine", "not_spine")
    return X, y, df["sample_weight"].to_numpy()


def run_folds(spec: dict, splits, train_h01_of, test_h01_of) -> pl.DataFrame:
    """Score every mode on every fold. `train_h01_of(split)` and `test_h01_of(split)`
    give the H01 rows the fold trains and tests on."""
    labels = spec_labels(spec)
    rows = []
    for fold, split in enumerate(tqdm(splits)):
        test_df = test_h01_of(split)
        for mode, config in TRAIN_MODES.items():
            parts = []
            if "h01" in config["datastacks"]:
                parts.append(train_h01_of(split))
            if "microns" in config["datastacks"]:
                parts.append(microns)
            X_train, y_train, w_train = get_xyw(pl.concat(parts), spec, config["feda"])
            X_test, y_test, _ = get_xyw(test_df, spec, config["feda"])
            model = make_model(random_state=13294587 * fold)
            model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = model.predict(X_test)
            rows.append(
                {
                    "fold": fold,
                    "train_mode": mode,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
                    "macro_f1": f1_score(y_test, y_pred, average="macro"),
                    # Row-major, in the order of `spec_labels(spec)`.
                    "conf_mat": confusion_matrix(y_test, y_pred, labels=labels)
                    .ravel()
                    .tolist(),
                }
            )
    return pl.DataFrame(rows)


# %%
neuron_splits = list(
    KFold(N_SPLITS, shuffle=True, random_state=SPLIT_SEED).split(h01_roots)
)
synapse_splits = list(
    StratifiedKFold(N_SPLITS, shuffle=True, random_state=SPLIT_SEED).split(
        np.zeros(len(h01)), h01["label"].to_numpy()
    )
)
# name: (splits, H01 train rows of a split, H01 test rows of a split, held-out unit)
EXPERIMENTS = {
    # Experiment 1: split by neuron, so no test neuron is in the training rows.
    "neuron_split": (
        neuron_splits,
        lambda split: h01.filter(pl.col("root_id").is_in(h01_roots[split[0]])),
        lambda split: h01.filter(pl.col("root_id").is_in(h01_roots[split[1]])),
        "neurons",
    ),
    # Experiment 2: split by synapse, stratified on the label.
    "synapse_split": (
        synapse_splits,
        lambda split: h01[split[0].tolist()],
        lambda split: h01[split[1].tolist()],
        "synapses",
    ),
}
if not RUN_NEURON_SPLIT:
    del EXPERIMENTS["neuron_split"]

# The cached tables under `model_inputs` and `h01_retrain_model_inputs`.
RETRAIN_INPUTS = [
    Tables["labeled_synapse_features"],
    Tables["labeled_point_features"],
    Tables["root_nuclei"],
    Tables["h01_labeled_synapse_features"],
    Tables["h01_root_nuclei"],
]

results = {}
for experiment, (splits, train_h01_of, test_h01_of, _) in EXPERIMENTS.items():
    for name, spec in MODEL_SPECS.items():
        folds = Stash(
            f"h01_retrain/{experiment}/{name}",
            inputs=RETRAIN_INPUTS,
            spec=spec,
            modes=TRAIN_MODES,
            n_splits=N_SPLITS,
            split_seed=SPLIT_SEED,
            h01_sample_weight=H01_SAMPLE_WEIGHT,
            forest=RF_PARAMS,
            version=FOLDS_VERSION,
        )
        if not folds.exists():
            print(f"{experiment}: {name}")
            folds.save(run_folds(spec, splits, train_h01_of, test_h01_of))
        results[experiment, name] = folds.load().to_pandas()

all_results = pd.concat(
    [
        df.assign(experiment=experiment, model=name)
        for (experiment, name), df in results.items()
    ],
    ignore_index=True,
)

# %%
# Every variant: the mean and spread over folds, and the paired test of the modes.
print(
    all_results.groupby(["experiment", "model", "train_mode"], sort=False)[
        ["weighted_f1", "macro_f1"]
    ]
    .agg(["mean", "std"])
    .round(3)
    .to_string()
)


def paired_test(results: pd.DataFrame) -> float:
    baseline = results.query("train_mode == @BASELINE_MODE").sort_values("fold")[METRIC]
    compare = results.query("train_mode == @COMPARE_MODE").sort_values("fold")[METRIC]
    return wilcoxon(baseline.to_numpy(), compare.to_numpy()).pvalue


for (experiment, name), df in results.items():
    print(
        f"{experiment} {name}: Wilcoxon ({BASELINE_MODE} vs {COMPARE_MODE}) p={paired_test(df):.3g}"
    )

# %%
# The mixing curves (R3.2): the main model on the grid of `MIXING_CONFIGS`.

# The MICrONS test synapses, stratified on the label. They leave the training pool.
synapse_rows = np.flatnonzero(microns["synapse_id"].is_not_null().to_numpy())
_, microns_test_rows = train_test_split(
    synapse_rows,
    test_size=MICRONS_TEST_SIZE,
    stratify=microns["label"].to_numpy()[synapse_rows],
    random_state=MICRONS_TEST_SEED,
)
is_microns_test = np.isin(np.arange(len(microns)), microns_test_rows)
microns_test = microns.filter(pl.Series(is_microns_test))
microns_pool = microns.filter(pl.Series(~is_microns_test))


def stratified_prefix(df: pl.DataFrame, size: int | None, seed) -> pl.DataFrame:
    """`size` rows of `df` in the label proportions of `df`: the first rows of each
    label in a seeded order. With one seed, a smaller size is a subset of a larger
    one. A size of None is all rows."""
    if size is None or size >= len(df):
        return df
    rng = np.random.default_rng(seed)
    labels = df["label"].to_numpy()
    keep = []
    for label in np.unique(labels):
        rows = rng.permutation(np.flatnonzero(labels == label))
        keep.append(rows[: round(size * len(rows) / len(df))])
    return df[np.sort(np.concatenate(keep)).tolist()]


def mixing_scores(model, test_df: pl.DataFrame, spec: dict) -> dict:
    X_test, y_test, _ = get_xyw(test_df, spec, feda=False)
    y_pred = model.predict(X_test)
    labels = spec_labels(spec)
    class_f1 = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        **{f"{label}_f1": value for label, value in zip(labels, class_f1)},
        # Row-major, in the order of `spec_labels(spec)`.
        "conf_mat": confusion_matrix(y_test, y_pred, labels=labels).ravel().tolist(),
    }


def run_mixing_fold(fold: int, split, configs=MIXING_CONFIGS) -> pl.DataFrame:
    """Fit the main model on each config of one fold of the synapse split, and score
    it on the held-out H01 rows of the fold and on the MICrONS test rows."""
    spec = MODEL_SPECS[MAIN_MODEL]
    h01_train, h01_test = h01[split[0].tolist()], h01[split[1].tolist()]
    rows = []
    for arm, h01_size, microns_size, h01_weight in tqdm(configs, desc=f"fold {fold}"):
        parts = []
        if h01_size != 0:
            parts.append(
                stratified_prefix(h01_train, h01_size, seed=(fold, 0)).with_columns(
                    sample_weight=pl.lit(h01_weight or 1, pl.Float64)
                )
            )
        if microns_size != 0:
            parts.append(stratified_prefix(microns_pool, microns_size, seed=(fold, 1)))
        train_df = pl.concat(parts)
        X_train, y_train, w_train = get_xyw(train_df, spec, feda=False)
        is_h01 = train_df["datastack"].to_numpy() == "h01"
        # The same forest seed as `run_folds`, so the full-size fits are comparable.
        model = make_model(random_state=13294587 * fold)
        model.fit(X_train, y_train, sample_weight=w_train)
        for test_domain, test_df in [("h01", h01_test), ("microns", microns_test)]:
            rows.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "h01_size": h01_size,
                    "microns_size": microns_size,
                    "h01_weight": h01_weight,
                    "test_domain": test_domain,
                    "n_h01": int(is_h01.sum()),
                    "n_microns": int((~is_h01).sum()),
                    "h01_weight_share": w_train[is_h01].sum() / w_train.sum(),
                    **mixing_scores(model, test_df, spec),
                }
            )
    return pl.DataFrame(rows)


MIXING_KEYS = ["arm", "h01_size", "microns_size", "h01_weight", "test_domain"]
MIXING_METRICS = ["accuracy", "weighted_f1", "macro_f1"] + [
    f"{label}_f1" for label in LABELS
]


def pooled_f1(conf_mats: pd.Series) -> dict:
    """The F1 of each label from the confusion matrices summed over folds, and their
    mean. Soma has 2 or 3 test synapses per fold, so a fold F1 is noisy."""
    summed = np.sum(np.stack(conf_mats.to_list()), axis=0).reshape(
        len(LABELS), len(LABELS)
    )
    f1 = 2 * np.diag(summed) / (summed.sum(axis=0) + summed.sum(axis=1))
    return {f"{label}_pooled_f1": value for label, value in zip(LABELS, f1)} | {
        "macro_pooled_f1": f1.mean()
    }


def summarize_mixing(folds: pd.DataFrame, keys=MIXING_KEYS) -> pd.DataFrame:
    """One row per config (`keys`): the mean and std over folds, and the pooled
    F1."""
    grouped = folds.groupby(keys, dropna=False, sort=False)
    summary = grouped[MIXING_METRICS + ["n_h01", "n_microns", "h01_weight_share"]].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary["n_folds"] = grouped.size()
    pooled = pd.DataFrame(
        [pooled_f1(conf_mats) for _, conf_mats in grouped["conf_mat"]],
        index=summary.index,
    )
    return pd.concat([summary, pooled], axis=1).reset_index()


# %%
mixing_stashes = [
    Stash(
        f"h01_retrain/mixing/fold{fold}",
        inputs=RETRAIN_INPUTS,
        spec=MODEL_SPECS[MAIN_MODEL],
        configs=MIXING_CONFIGS,
        n_splits=N_SPLITS,
        split_seed=SPLIT_SEED,
        microns_test=(MICRONS_TEST_SIZE, MICRONS_TEST_SEED),
        forest=RF_PARAMS,
        version=CURVE_VERSION,
    )
    for fold in range(N_SPLITS)
]
for fold, (stash, split) in enumerate(zip(mixing_stashes, synapse_splits)):
    if not stash.exists():
        stash.save(run_mixing_fold(fold, split))

# The plot table.
mixing_summary = Stash(
    "h01_retrain/mixing/summary", inputs=mixing_stashes, version=CURVE_VERSION
)
if not mixing_summary.exists():
    mixing_folds = pl.concat([stash.load() for stash in mixing_stashes])
    mixing_summary.save(summarize_mixing(mixing_folds.to_pandas()))
mixing = mixing_summary.load().to_pandas()


# %%
def plot_band(ax, rows: pd.DataFrame, x: str, metric: str, **kwargs):
    """`metric` against `x`: the fold mean with a band of one std, or a pooled value
    with no band."""
    rows = rows.sort_values(x)
    mean = rows[f"{metric}_mean"] if f"{metric}_mean" in rows else rows[metric]
    (line,) = ax.plot(rows[x], mean, marker="o", **kwargs)
    if f"{metric}_std" in rows:
        ax.fill_between(
            rows[x],
            mean - rows[f"{metric}_std"],
            mean + rows[f"{metric}_std"],
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )


h01_test = mixing.query("test_domain == 'h01'")
h01_only_full = h01_test.query("arm == 'h01_only' and h01_size.isna()")
weight_colors = dict(
    zip(MIXING_H01_WEIGHTS, sns.color_palette("viridis", len(MIXING_H01_WEIGHTS)))
)
# The H01 weights that the mixing figure shows. The table has all of them.
PLOT_H01_WEIGHTS = [10]
# Where the mixing figure puts 0 MICrONS synapses (H01 only) on its log axis, left
# of a break.
ZERO_X = 100


def zero_break_axis(ax, zero_x=ZERO_X, first_x=min(MIXING_MICRONS_SIZES[:-1])):
    """A log x axis with a "0" tick at `zero_x` and a break before `first_x`."""
    ax.set_xscale("log")
    ax.set_xlim(left=zero_x / 1.4)
    ax.set_xticks([zero_x, 1e3, 1e4, 1e5], ["0", "$10^3$", "$10^4$", "$10^5$"])
    ax.xaxis.set_minor_locator(
        FixedLocator(
            [
                m * 10**e
                for e in range(2, 6)
                for m in range(2, 10)
                if first_x <= m * 10**e <= 2e5
            ]
        )
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    x_break = np.sqrt(zero_x * first_x)
    for x in [x_break / 1.08, x_break * 1.08]:
        ax.plot(
            x,
            0,
            transform=ax.get_xaxis_transform(),
            marker=[(-1, -2), (1, 2)],
            markersize=10,
            color="black",
            clip_on=False,
        )


# The left panels in each F1 metric: macro F1, and weighted F1 (the metric of the
# zero-shot H01 validation, Fig 6A).
for fold_metric, fold_ylabel, suffix in [
    ("macro_f1", "Macro F1 (fold mean)", ""),
    ("weighted_f1", "Weighted F1 (fold mean)", "_weighted_f1"),
]:
    # Mixing: all H01 training rows, plus a varying number of MICrONS rows.
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric, ylabel in [
        (axs[0], fold_metric, fold_ylabel),
        (axs[1], "soma_pooled_f1", "Soma F1 (folds pooled)"),
    ]:
        for weight in PLOT_H01_WEIGHTS:
            # The H01 only fit is the point at 0 MICrONS synapses.
            rows = pd.concat(
                [
                    h01_only_full.assign(n_microns_mean=ZERO_X),
                    h01_test.query("arm == 'both' and h01_weight == @weight"),
                ]
            )
            plot_band(
                ax,
                rows,
                "n_microns_mean",
                metric,
                color=weight_colors[weight],
                label=f"H01 weight {weight:,}",
            )
        plot_band(
            ax,
            h01_test.query("arm == 'microns_only'"),
            "n_microns_mean",
            metric,
            color="gray",
            label="MICrONS only",
        )
        zero_break_axis(ax)
        ax.set(xlabel="MICrONS training synapses", ylabel=ylabel)
    axs[0].set_title("Held-out H01 synapses", loc="left")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    save_matplotlib_figure(fig, f"mixing_microns_size{suffix}", subfolder=SUBFOLDER)

    # Equal n: each dataset alone on H01, and MICrONS on MICrONS.
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric, ylabel in [
        (axs[0], fold_metric, fold_ylabel),
        (axs[1], "soma_pooled_f1", "Soma F1 (folds pooled)"),
    ]:
        for arm, test_domain, x, color, linestyle, label in [
            ("h01_only", "h01", "n_h01_mean", "C0", "-", "Train H01, test H01"),
            (
                "microns_only",
                "h01",
                "n_microns_mean",
                "C1",
                "-",
                "Train MICrONS, test H01",
            ),
            (
                "microns_only",
                "microns",
                "n_microns_mean",
                "C1",
                ":",
                "Train MICrONS, test MICrONS",
            ),
        ]:
            rows = mixing.query("arm == @arm and test_domain == @test_domain")
            plot_band(
                ax, rows, x, metric, color=color, linestyle=linestyle, label=label
            )
        ax.set(xscale="log", xlabel="Training synapses", ylabel=ylabel)
    axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    save_matplotlib_figure(fig, f"mixing_equal_size{suffix}", subfolder=SUBFOLDER)

# %%
# The diet curves: scored on the H01 validation cohort, which never trains.
validation = (
    Tables["h01_model_inputs"]
    .lazy()
    .filter(pl.col("cohort") == "validation")
    .collect()
    .drop_nulls(spectral_columns)
    .with_columns(datastack=pl.lit("h01_validation"), sample_weight=pl.lit(1.0))
    .select(h01.columns)
)
assert not set(validation["root_id"]) & set(h01["root_id"])
print(f"H01 validation: {len(validation):,} synapses")


def diet_configs(n_h01_pool: int, n_microns_pool: int) -> list[tuple]:
    """(H01 fraction, total size, H01 rows, MICrONS rows) of each diet fit. Each
    line with H01 ends at all H01 rows."""
    configs = []
    for fraction in DIET_H01_FRACTIONS:
        totals = [
            n_microns_pool if total is None else total for total in DIET_TOTAL_SIZES
        ]
        if fraction > 0:
            totals = [t for t in totals if t < n_h01_pool / fraction]
            totals.append(int(n_h01_pool / fraction))
        for total in sorted(set(totals)):
            n_h01 = round(fraction * total)
            if n_h01 <= n_h01_pool and total - n_h01 <= n_microns_pool:
                configs.append((fraction, total, n_h01, total - n_h01))
    return configs


DIET_CONFIGS = diet_configs(len(h01), len(microns))
DIET_KEYS = ["h01_fraction", "total_size"]


def run_diet_repeat(repeat: int, configs=DIET_CONFIGS) -> pl.DataFrame:
    """Fit the main model on each diet with one subsample seed, and score it on
    the H01 validation cohort."""
    spec = MODEL_SPECS[MAIN_MODEL]
    rows = []
    for fraction, total, n_h01, n_microns in tqdm(configs, desc=f"repeat {repeat}"):
        train_df = pl.concat(
            [
                stratified_prefix(pool, n, seed=(repeat, seed)).head(n)
                for pool, n, seed in [(h01, n_h01, 0), (microns, n_microns, 1)]
                if n > 0
            ]
        )
        X_train, y_train, _ = get_xyw(train_df, spec, feda=False)
        model = make_model(random_state=repeat)
        model.fit(X_train, y_train)
        is_h01 = train_df["datastack"].to_numpy() == "h01"
        rows.append(
            {
                "repeat": repeat,
                "h01_fraction": fraction,
                "total_size": total,
                "n_h01": int(is_h01.sum()),
                "n_microns": int((~is_h01).sum()),
                "h01_weight_share": is_h01.mean(),
                **mixing_scores(model, validation, spec),
            }
        )
    return pl.DataFrame(rows)


diet_stashes = [
    Stash(
        f"h01_retrain/diet/repeat{repeat}",
        inputs=RETRAIN_INPUTS,
        spec=MODEL_SPECS[MAIN_MODEL],
        configs=DIET_CONFIGS,
        forest=RF_PARAMS,
        version=DIET_VERSION,
    )
    for repeat in range(DIET_REPEATS)
]
for repeat, stash in enumerate(diet_stashes):
    if not stash.exists():
        stash.save(run_diet_repeat(repeat))

# The plot table.
diet_summary = Stash(
    "h01_retrain/diet/summary", inputs=diet_stashes, version=DIET_VERSION
)
if not diet_summary.exists():
    diet_repeats = pl.concat([stash.load() for stash in diet_stashes])
    diet_summary.save(summarize_mixing(diet_repeats.to_pandas(), keys=DIET_KEYS))
diet = diet_summary.load().to_pandas()

# %%
fraction_colors = dict(
    zip(DIET_H01_FRACTIONS, sns.color_palette("viridis", len(DIET_H01_FRACTIONS)))
)
fig, axs = plt.subplots(2, 3, figsize=(13, 8), layout="constrained", sharex=True)
for ax, metric, ylabel in [
    (axs[0, 0], "weighted_f1", "Weighted F1"),
    (axs[0, 1], "macro_f1", "Macro F1"),
    (axs[1, 0], "soma_f1", "Soma F1"),
    (axs[1, 1], "shaft_f1", "Shaft F1"),
    (axs[1, 2], "spine_f1", "Spine F1"),
]:
    for fraction in DIET_H01_FRACTIONS:
        plot_band(
            ax,
            diet.query("h01_fraction == @fraction"),
            "total_size",
            metric,
            color=fraction_colors[fraction],
            label=f"{fraction:.0%} H01",
        )
    ax.set(xscale="log", ylabel=ylabel)
for ax in axs[1]:
    ax.set_xlabel("Training synapses")
axs[0, 2].axis("off")
axs[0, 2].legend(
    *axs[0, 0].get_legend_handles_labels(),
    loc="center left",
    frameon=False,
    title="Training synapses from H01",
)
fig.suptitle(
    f"H01 validation cohort ({len(validation):,} synapses, mean of {DIET_REPEATS} repeats)"
)
save_matplotlib_figure(fig, "validation_diet", subfolder=SUBFOLDER)

# %%
# The main model: the figures and glued variables for the paper.


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


for experiment, (*_, held_out) in EXPERIMENTS.items():
    main = results[experiment, MAIN_MODEL]
    # Weighted F1 too, the metric of the zero-shot H01 validation (Fig 6A).
    for metric in [METRIC, "weighted_f1"]:
        summary = main.groupby("train_mode")[metric].agg(["mean", "std"])
        save_variables(
            prefix=f"h01_retrain_{experiment}_",
            **{
                f"{mode}_{metric}_{stat_name}": value
                for mode, row in summary.iterrows()
                for stat_name, value in row.items()
            },
            format="{:.3f}",
        )
    save_variables(
        prefix=f"h01_retrain_{experiment}_",
        wilcoxon_p=paired_test(main),
        format="{:.2g}",
    )
    save_matplotlib_figure(
        plot_folds(main, held_out),
        f"retrain_{experiment}_{METRIC}",
        subfolder=SUBFOLDER,
    )

    # The held-out confusion matrices, summed over folds and row normalized.
    fig, axs = plt.subplots(
        1, len(TRAIN_MODES), figsize=(4 * len(TRAIN_MODES), 4), layout="constrained"
    )
    for ax, mode in zip(np.atleast_1d(axs), TRAIN_MODES):
        summed = np.sum(np.stack(main.query("train_mode == @mode")["conf_mat"]), axis=0)
        confusion = pd.DataFrame(
            summed.reshape(len(LABELS), len(LABELS)), index=LABELS, columns=LABELS
        )
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

# %%
# Every variant on one figure. The binary variants get their own panel, because
# their macro F1 is over 2 classes, not 3.
multiclass = [name for name, spec in MODEL_SPECS.items() if not spec["binary"]]
binary = [name for name, spec in MODEL_SPECS.items() if spec["binary"]]
for experiment, (*_, held_out) in EXPERIMENTS.items():
    df = all_results.query("experiment == @experiment").assign(
        train_mode=lambda d: d["train_mode"].map(MODE_NAMES)
    )
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        layout="constrained",
        gridspec_kw=dict(width_ratios=[len(multiclass), len(binary)]),
    )
    for ax, names, title in [
        (axs[0], multiclass, "soma, shaft, spine"),
        (axs[1], binary, "spine, not spine"),
    ]:
        panel = df[df["model"].isin(names)]
        sns.stripplot(
            data=panel,
            x="model",
            y=METRIC,
            hue="train_mode",
            order=names,
            dodge=True,
            alpha=0.6,
            ax=ax,
            legend=ax is axs[1],
        )
        sns.pointplot(
            data=panel,
            x="model",
            y=METRIC,
            hue="train_mode",
            order=names,
            dodge=0.4,
            linestyle="none",
            markers="_",
            markersize=20,
            palette="dark:black",
            errorbar=None,
            ax=ax,
            legend=False,
        )
        ax.set(xlabel="", ylabel="", title=title)
        ax.set_xticks(
            range(len(names)),
            [n.removeprefix("ssa_model_") for n in names],
            rotation=30,
        )
    axs[0].set_ylabel(f"Macro F1\n(held-out H01 {held_out})")
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1), title="")
    save_matplotlib_figure(fig, f"retrain_all_models_{experiment}", subfolder=SUBFOLDER)

# %%
# The retrained models for viewing: each variant fit once on all MICrONS rows and
# all H01 retrain rows (the `both_upweighted` mode), with no held-out fold. They
# are saved as `data/models/{variant}_h01_{MODEL_VERSION}`, next to the MICrONS
# only models, so `load_ssa_model` reads either kind. A saved model is read
# instead of fit again, as the fold results are.
# NOTE: the labeled synapses of the retrain roots are in the training rows of these
# models, so their predictions on those synapses are not held out.
retrain_frame = pl.concat([h01, microns]).to_pandas()
for name, spec in MODEL_SPECS.items():
    model_name = f"{name}_h01"
    if (MODEL_PATH / f"{model_name}_{MODEL_VERSION}").exists():
        continue
    print(f"final fit: {model_name}")
    X = retrain_frame[feature_columns(spec, spectral_columns)]
    y = retrain_frame["label"].to_numpy()
    if spec["binary"]:
        y = np.where(y == "spine", "spine", "not_spine")
    model = make_model(random_state=0)
    model.fit(X, y, sample_weight=retrain_frame["sample_weight"].to_numpy())
    save_model(model, X, model_name)
