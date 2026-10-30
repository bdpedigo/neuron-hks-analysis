# %%

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from analysis import Stash, Tables, save_matplotlib_figure, set_matplotlib_theme
from analysis.models import (
    RF_PARAMS,
    feature_columns,
    make_model,
    select_spectral_columns,
)

SUBFOLDER = "simple_features"
# Part of the stash key of the out-of-fold results. Bump it after a change to `out_of_fold`.
FOLDS_VERSION = 1
N_SPLITS = 5
SEEDS = (0, 1, 2)
LABELS = ["shaft", "soma", "spine"]
BDP_TABLE = "bdp_synapse_compartment_labels"
N_BOOT = 4000
BASELINE = "full"


def arm(*prefixes):
    return dict(prefixes=prefixes, extra_features=[], binary=False)


ARMS = {
    "full": arm("hks_", "curvature_", "normal_"),
    "hks_curv": arm("hks_", "curvature_"),
    "hks_only": arm("hks_"),
    "curv_only": arm("curvature_"),
    "normal_only": arm("normal_"),
    # a prefix that is a whole column name selects that column
    "raw_curv": arm("curvature_mean_raw", "curvature_gauss_raw"),
}
# The arm names in the figure.
ARM_LABELS = {
    "full": "All features",
    "hks_curv": "HKS + diff. curvature",
    "hks_only": "HKS",
    "curv_only": "Diffused curvature",
    "normal_only": "Diffused normals",
    "raw_curv": "Raw curvature",
}

set_matplotlib_theme()

# %%
# The synapse rows, as train.py builds them: rows with domain features only.
table = Tables["model_inputs"].collect()
spectral_columns = select_spectral_columns(table.columns)
synapses = (
    table.filter(pl.col("synapse_id").is_not_null())
    .drop_nulls(spectral_columns)
    .to_pandas()
    .set_index("label_key")
)
y = synapses["tag"]
w = synapses["sample_weight"].astype(float)
groups = synapses["root_id"]
is_bdp = (synapses["table_name"] == BDP_TABLE).to_numpy()
assert set(y.unique()) == set(LABELS)
print(f"{len(synapses):,} synapse rows on {groups.nunique():,} roots ({is_bdp.sum():,} BDP)")


def training_x(spec: dict) -> pd.DataFrame:
    return synapses[feature_columns(spec, spectral_columns)]


def out_of_fold(X: pd.DataFrame, y: pd.Series, w: pd.Series, groups: pd.Series, seeds) -> pl.DataFrame:
    """One row per (row, seed): the prediction of the fold model that did not see
    the row's root, and the class probabilities."""
    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    out = []
    for seed in seeds:
        for fold, (train, test) in enumerate(tqdm(folds, desc=f"seed {seed}")):
            assert not np.isin(groups.iloc[test], groups.iloc[train]).any()
            model = make_model(random_state=fold + 100 * seed)
            model.fit(X.iloc[train], y.iloc[train], sample_weight=w.iloc[train])
            proba = model.predict_proba(X.iloc[test])
            out.append(
                pl.DataFrame(
                    {
                        "label_key": X.index[test].to_numpy(),
                        "seed": seed,
                        "fold": fold,
                        "pred": model.classes_[proba.argmax(1)],
                        **{f"p_{c}": proba[:, k].astype(np.float32) for k, c in enumerate(model.classes_)},
                    }
                )
            )
    return pl.concat(out)


def group_bootstrap(correct: np.ndarray, baseline: np.ndarray, groups: np.ndarray, seed: int = 0):
    """The difference in mean correctness against `baseline`, and its 95% CI from
    a paired bootstrap over `groups`."""
    codes, inverse = np.unique(groups, return_inverse=True)
    n = np.bincount(inverse, minlength=len(codes))
    diff = np.bincount(inverse, weights=correct - baseline, minlength=len(codes))
    draws = np.random.default_rng(seed).integers(0, len(codes), size=(N_BOOT, len(codes)))
    boot = diff[draws].sum(1) / n[draws].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diff.sum() / n.sum(), lo, hi


# %%
results = {}
fold_accuracy = {}
truth = pl.DataFrame({"label_key": synapses.index.to_numpy(), "tag": y.to_numpy()})
for name, spec in ARMS.items():
    X = training_x(spec)
    stash = Stash(
        f"{SUBFOLDER}/{name}",
        # The cached tables under `model_inputs`.
        inputs=[Tables["labeled_synapse_features"], Tables["labeled_point_features"], Tables["root_nuclei"]],
        spec=spec,
        columns=list(X.columns),
        seeds=list(SEEDS),
        n_splits=N_SPLITS,
        forest=RF_PARAMS,
        version=FOLDS_VERSION,
    )
    if not stash.exists():
        print(f"{name}: {X.shape[1]} columns")
        stash.save(out_of_fold(X, y, w, groups, SEEDS))
    oof = stash.load()
    assert oof.height == len(SEEDS) * len(X)
    # rows in the order of `synapses`, one column of predictions per seed
    results[name] = (
        oof.pivot(on="seed", index="label_key", values="pred")
        .to_pandas()
        .set_index("label_key")
        .loc[synapses.index, [str(s) for s in SEEDS]]
    )
    # the accuracy on each held-out fold, per seed
    fold_accuracy[name] = (
        oof.join(truth, on="label_key")
        .group_by("seed", "fold")
        .agg(accuracy=(pl.col("pred") == pl.col("tag")).mean())["accuracy"]
        .to_numpy()
    )

# %%
y_true = y.to_numpy()


def scores(pred: np.ndarray) -> dict:
    row = dict(accuracy=(pred == y_true).mean(), macro_f1=f1_score(y_true, pred, average="macro"))
    for c in LABELS:
        row[f"{c}_recall"] = (pred[y_true == c] == c).mean()
    for c in ["spine", "shaft"]:
        row[f"bdp_{c}_recall"] = (pred[(y_true == c) & is_bdp] == c).mean()
    return row


# the per-row correctness, averaged over the seeds
correct = {
    name: np.mean([preds[s].to_numpy() == y_true for s in preds.columns], axis=0)
    for name, preds in results.items()
}
rows = []
for name, preds in results.items():
    per_seed = pd.DataFrame([scores(preds[s].to_numpy()) for s in preds.columns])
    row = dict(arm=name, n_columns=len(training_x(ARMS[name]).columns))
    for metric in per_seed.columns:
        row[metric] = per_seed[metric].mean()
        row[f"{metric}_range"] = per_seed[metric].max() - per_seed[metric].min()
    diff, lo, hi = group_bootstrap(correct[name], correct[BASELINE], groups.to_numpy())
    row.update(accuracy_vs_full=diff, accuracy_vs_full_lo95=lo, accuracy_vs_full_hi95=hi)
    # against a baseline of 0, the bootstrap gives the CI of the accuracy itself
    _, lo, hi = group_bootstrap(correct[name], np.zeros_like(correct[name]), groups.to_numpy())
    row.update(accuracy_lo95=lo, accuracy_hi95=hi)
    rows.append(row)
summary = pd.DataFrame(rows).set_index("arm")
metrics = ["accuracy", "macro_f1", "spine_recall", "shaft_recall", "soma_recall", "bdp_spine_recall", "bdp_shaft_recall"]
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print((100 * summary[metrics]).round(2).assign(n_columns=summary["n_columns"]))
    print((100 * summary[[f"{m}_range" for m in metrics]]).round(2))
    print((100 * summary[["accuracy_vs_full", "accuracy_vs_full_lo95", "accuracy_vs_full_hi95"]]).round(2))

# %%
# Out-of-fold accuracy per arm (seed mean) as a bar, over the accuracy on each
# held-out fold of each seed.
fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
xx = np.arange(len(summary))
rng = np.random.default_rng(0)
for x, name in zip(xx, summary.index):
    acc = fold_accuracy[name]
    ax.scatter(x + rng.uniform(-0.15, 0.15, len(acc)), acc, s=10, color="gray", alpha=0.6, zorder=0)
ax.hlines(summary["accuracy"], xx - 0.25, xx + 0.25, color="black", linewidth=2)
ax.set(ylabel="CV accuracy\n(held out cells)")
ax.set_xticks(xx, summary.index.map(ARM_LABELS), rotation=45, ha="right")
save_matplotlib_figure(fig, "feature_families_accuracy", subfolder=SUBFOLDER, pad_inches=0.1)
