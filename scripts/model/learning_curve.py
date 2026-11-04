"""The learning curve of the main model (`hks_synapse_learning_curve`) and the
`model_sample_curve_*` values: how much training data do we actually need?

It uses only the main model's features and the synapse labels, so it stays
comparable across model versions. The rows and the nucleus mask are those of
train.py. The fold scores are stashed (`analysis.Stash`) under a key of the input
tables and the curve settings, and a stashed result is read instead of fit again.
After a change to `run_curve`, bump `CURVE_VERSION`.

    uv run python scripts/model/learning_curve.py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from analysis import (
    FIG_PATH,
    Stash,
    Tables,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)
from analysis.models import (
    MAIN_MODEL,
    MODEL_SPECS,
    NUCLEUS_MASK_FRACTION,
    NUCLEUS_MASK_SEED,
    RF_PARAMS,
    feature_columns,
    make_model,
    nucleus_masked_roots,
    select_spectral_columns,
)

# Part of the stash key of the fold scores. Bump it after a change to `run_curve`.
CURVE_VERSION = 1
N_FOLDS = 10
DATA_FRACTIONS = np.geomspace(0.001, 1.0, num=13)

set_matplotlib_theme()
figure_out_path = FIG_PATH / "model_training"


# %%
def load_xy():
    """The main model's features, labels and sample weights on the labeled
    synapses, as train.py builds them."""
    result = Tables["model_inputs"].collect()
    spectral_columns = select_spectral_columns(result.columns)
    table = (
        result.drop_nulls(subset=spectral_columns).to_pandas().set_index("label_key")
    )
    spec = MODEL_SPECS[MAIN_MODEL]
    X = table[feature_columns(spec, spectral_columns)]
    if "distance_to_nucleus" in X:
        # The mask is drawn over all training roots, the point-label roots too.
        masked = table["root_id"].isin(nucleus_masked_roots(table["root_id"]))
        X = X.assign(distance_to_nucleus=X["distance_to_nucleus"].mask(masked))
    y = table["tag"]
    if spec["binary"]:
        y = y.where(y == "spine", "not_spine")
    is_synapse = table["synapse_id"].notna()
    weight = table["sample_weight"].astype(float)
    return X[is_synapse], y[is_synapse], weight[is_synapse]


def score(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "weighted_f1": f1_score(y, y_pred, average="weighted"),
        "macro_f1": f1_score(y, y_pred, average="macro"),
    }


def run_curve(X_train, y_train, curve_weight) -> pd.DataFrame:
    rows = []
    for fold in tqdm(range(N_FOLDS)):
        train_index, test_index = train_test_split(
            np.arange(len(X_train)), test_size=0.2, random_state=fold * 10345
        )
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        weights_train_fold = curve_weight.iloc[train_index]

        rng = np.random.default_rng(fold)
        for frac in DATA_FRACTIONS:
            n_samples = int(len(X_train_fold) * frac)
            sample_indices = rng.choice(len(X_train_fold), size=n_samples, replace=False)
            fold_model = make_model(random_state=fold)
            fold_model.fit(
                X_train_fold.iloc[sample_indices],
                y_train_fold.iloc[sample_indices],
                sample_weight=weights_train_fold.iloc[sample_indices],
            )

            fold_scores = score(fold_model, X_test_fold, y_test_fold)
            fold_scores["data_fraction"] = frac
            fold_scores["fold"] = fold
            fold_scores["n_samples"] = n_samples
            fold_scores["n_test"] = len(test_index)
            rows.append(fold_scores)
    return pd.DataFrame(rows)


curve = Stash(
    "model_training/learning_curve",
    # The cached tables under `model_inputs`.
    inputs=[
        Tables["labeled_synapse_features"],
        Tables["labeled_point_features"],
        Tables["root_nuclei"],
    ],
    model=MAIN_MODEL,
    spec=MODEL_SPECS[MAIN_MODEL],
    forest=RF_PARAMS,
    nucleus_mask=(NUCLEUS_MASK_FRACTION, NUCLEUS_MASK_SEED),
    n_folds=N_FOLDS,
    data_fractions=DATA_FRACTIONS.tolist(),
    version=CURVE_VERSION,
)
if not curve.exists():
    curve.save(run_curve(*load_xy()))
score_df = curve.load().to_pandas()

save_variables(
    model_sample_curve_n_test_synapses=int(score_df["n_test"].iloc[0]), format="{:,}"
)


# %%

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
sns.lineplot(
    data=score_df, x="n_samples", y="weighted_f1", marker="o", ax=axs[0], color="black"
)
axs[0].set(xlabel="Number of training samples", ylabel="Weighted F1 score")
sns.lineplot(
    data=score_df, x="n_samples", y="weighted_f1", marker="o", ax=axs[1], color="black"
)
axs[1].set_xscale("log")
axs[1].set(xlabel="Number of training samples")

save_matplotlib_figure(fig, "hks_synapse_learning_curve", figure_out_path)

# %%

mean_scores = (
    score_df.query("data_fraction == 0.01").mean().drop(["fold", "n_samples", "n_test"])
)
save_variables(
    prefix="model_sample_curve_fraction_0.01_", **mean_scores.to_dict(), format="{:.3f}"
)

mean_scores = (
    score_df.query("data_fraction == 0.1").mean().drop(["fold", "n_samples", "n_test"])
)
save_variables(
    prefix="model_sample_curve_fraction_0.1_", **mean_scores.to_dict(), format="{:.3f}"
)

fraction_to_sample_size = score_df.groupby("data_fraction")["n_samples"].mean()
save_variables(
    prefix="model_sample_curve_",
    **{
        f"fraction_{frac:.2f}_n_samples": int(n)
        for frac, n in fraction_to_sample_size.items()
    },
    format="{:,}",
)
