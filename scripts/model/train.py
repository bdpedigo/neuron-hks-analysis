# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from upath import UPath

from analysis import (
    FIG_PATH,
    Tables,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)
from analysis.models import (
    MAIN_MODEL,
    MODEL_SPECS,
    NUCLEUS_MASK_FRACTION,
    RF_PARAMS,
    make_model,
    nucleus_masked_roots,
    save_model,
    select_spectral_columns,
)

# %%

datastack = "minnie65_phase3_v1"
base = "gs://bdp-ssa/meshmash-deployment"
out = "scratch_data/synapse_hks_features.parquet"
base = UPath(base)


# %%

joined = Tables["model_inputs"].lazy()

joined.collect_schema()
# %%

start_time = time.time()
result = Tables["model_inputs"].collect()
end_time = time.time()
print("Collection time:", end_time - start_time)

# %%

set_matplotlib_theme()
figure_out_path = FIG_PATH / "model_training"

spectral_columns = select_spectral_columns(result.columns)

# rows without domain features (not yet processed by the meshmash deployment) can't
# be used for training or evaluation. Every variant trains on these same rows.
training_table = result.drop_nulls(subset=spectral_columns).to_pandas().set_index(
    "synapse_id"
)
sample_weight = training_table["sample_weight"].astype(float)

save_variables(model_n_training_synapses=len(training_table), format="{:,}")

# The nucleus mask: see `analysis.models`.
save_variables(params_nucleus_mask_fraction=NUCLEUS_MASK_FRACTION)
nucleus_masked = training_table["root_id"].isin(
    nucleus_masked_roots(training_table["root_id"])
)


def make_xy(spec):
    X = training_table[spectral_columns + spec["extra_features"]]
    if "distance_to_nucleus" in X:
        X = X.assign(distance_to_nucleus=X["distance_to_nucleus"].mask(nucleus_masked))
    y = training_table["tag"]
    if spec["binary"]:
        # `soma_spine` is already `spine` in the label table, so this merges only
        # soma and shaft.
        y = y.where(y == "spine", "not_spine")
    return X, y


save_variables(
    params_rf_max_depth=RF_PARAMS["max_depth"],
    params_rf_n_estimators=RF_PARAMS["n_estimators"],
)


# %% train and save each variant on all available synapses

models = {}
for name, spec in MODEL_SPECS.items():
    X, y = make_xy(spec)
    currtime = time.time()
    model = make_model(random_state=0)
    model.fit(X, y, sample_weight=sample_weight)
    model_dir = save_model(model, X, name)
    print(
        f"{name}: {time.time() - currtime:.1f} s, {X.shape[1]} features, "
        f"classes {list(model.classes_)}, saved to {model_dir}"
    )
    models[name] = model

# %% the learning curve below uses only the main model's features and labels

X_train, y_train = make_xy(MODEL_SPECS[MAIN_MODEL])
model = models[MAIN_MODEL]


# %% learning curve: how much training data do we actually need?


def score(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "weighted_f1": f1_score(y, y_pred, average="weighted"),
        "macro_f1": f1_score(y, y_pred, average="macro"),
    }


rows = []
data_fractions = np.geomspace(0.001, 1.0, num=13)
for fold in tqdm(range(10)):
    train_index, test_index = train_test_split(
        np.arange(len(X_train)), test_size=0.2, random_state=fold * 10345
    )
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold = sample_weight.iloc[train_index]

    rng = np.random.default_rng(fold)
    for frac in data_fractions:
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
        rows.append(fold_scores)

score_df = pd.DataFrame(rows)

save_variables(model_sample_curve_n_test_synapses=len(test_index), format="{:,}")


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

mean_scores = score_df.query("data_fraction == 0.01").mean().drop(["fold", "n_samples"])
save_variables(
    prefix="model_sample_curve_fraction_0.01_", **mean_scores.to_dict(), format="{:.3f}"
)

mean_scores = score_df.query("data_fraction == 0.1").mean().drop(["fold", "n_samples"])
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
