# %%
import shutil
import time

import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from caveclient import CAVEclient
from joblib import dump
from mlflow.models import infer_signature
from nglui.parser import StateParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from tqdm.auto import tqdm
from upath import UPath

from analysis import (
    FIG_PATH,
    MODEL_PATH,
    VERSION,
    Tables,
    load_neuron_info,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)
from analysis.products.features import add_distance_to_nucleus, synapse_features
from analysis.products.sources import (
    DOMAIN_FEATURES_PARAMS,
    DOMAIN_FEATURES_VERSION,
    LABELED_AS_OF,
)

# %%

datastack = "minnie65_phase3_v1"
base = "gs://bdp-ssa/meshmash-deployment"
out = "scratch_data/synapse_hks_features.parquet"
base = UPath(base)


# %%

joined = Tables["training_features"].lazy()

joined.collect_schema()
# %%

start_time = time.time()
result = Tables["training_features"].collect()
end_time = time.time()
print("Collection time:", end_time - start_time)

# %%

set_matplotlib_theme()
figure_out_path = FIG_PATH / "model_training"

FEATURE_PREFIXES = ("hks_", "curvature_", "normal_")
spectral_columns = [c for c in result.columns if c.startswith(FEATURE_PREFIXES)]

# NOTE: `distance_to_nucleus` is null for a root with no nucleus point. The
# random forest accepts missing values, so those rows stay in. See also the mask
# below.
MODEL_SPECS = {
    "ssa_model_full": dict(extra_features=["distance_to_nucleus"], binary=False),
    "ssa_model_no_nuc": dict(extra_features=[], binary=False),
    "ssa_model_binary": dict(extra_features=[], binary=True),
}
# The learning curve, the glued variables and every downstream figure use this one.
MAIN_MODEL = "ssa_model_full"
# NOTE: a published model name is write-once in meshmash-deployment, so bump this
# before publishing a retrained model.
MODEL_VERSION = "v1"

# rows without domain features (not yet processed by the meshmash deployment) can't
# be used for training or evaluation. Every variant trains on these same rows.
training_table = result.drop_nulls(subset=spectral_columns).to_pandas().set_index(
    "synapse_id"
)
sample_weight = training_table["sample_weight"].astype(float)

save_variables(model_n_training_synapses=len(training_table), format="{:,}")

# NOTE: almost every training root has a nucleus, but many deployment roots do not
# (for example, orphan fragments). Without masking, the forest sees almost no missing
# distances and sends them down whichever branch is larger. So the distance is set
# to null on a fixed share of the training roots. The mask is by root, the same way
# a missing nucleus looks in the deployment.
NUCLEUS_MASK_FRACTION = 0.2
NUCLEUS_MASK_SEED = 0
save_variables(params_nucleus_mask_fraction=NUCLEUS_MASK_FRACTION)

_roots = np.sort(training_table["root_id"].unique())
nucleus_masked_roots = np.random.default_rng(NUCLEUS_MASK_SEED).permutation(_roots)[
    : round(NUCLEUS_MASK_FRACTION * len(_roots))
]
nucleus_masked = training_table["root_id"].isin(nucleus_masked_roots)


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


RF_PARAMS = dict(class_weight="balanced", max_depth=15, n_estimators=500)
save_variables(
    params_rf_max_depth=RF_PARAMS["max_depth"],
    params_rf_n_estimators=RF_PARAMS["n_estimators"],
)


def make_model(random_state):
    return RandomForestClassifier(**RF_PARAMS, n_jobs=-1, random_state=random_state)


def save_model(model, X, name):
    # Saved in the MLflow model format, which records this environment and the
    # feature signature. meshmash-deployment publishes it from this directory with
    # `just publish-model`; it refuses a model without `classes` and `trained_on`.
    # The folder name is also the published name. A rerun at the same MODEL_VERSION
    # replaces the local folder.
    model_dir = MODEL_PATH / f"{name}_{MODEL_VERSION}"
    shutil.rmtree(model_dir, ignore_errors=True)  # save_model refuses a non-empty path
    mlflow.sklearn.save_model(
        model,
        str(model_dir),
        signature=infer_signature(X.head(5), model.predict_proba(X.head(5))),
        metadata={
            "classes": [str(label) for label in model.classes_],
            # The domain_features table the training rows were joined from.
            "trained_on": {
                "product": "domain_features",
                "param_hash": DOMAIN_FEATURES_PARAMS,
                "step_version": DOMAIN_FEATURES_VERSION,
            },
        },
        # skops refuses tree internals unless named; this model is our own.
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    return model_dir


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

# the below was mostly superseeded by validation.py
quit()

# %% 

validation_state_urls = """
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5402780981264384
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6645276717613056
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5613072445079552
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4901917330243584
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6063346229968896
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6382416582148096
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6373304087609344
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4537006792114176
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6451260747153408
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5925623993204736
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4985050583007232
https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5328690731810816
""".strip().split("\n")

client = CAVEclient(datastack)

annotation_dfs = []
for url in validation_state_urls:
    state_id = int(url.split("/")[-1])
    state_dict = client.state.get_state_json(state_id)
    annotation_df = StateParser(state_dict).annotation_dataframe(expand_tags=True)
    annotation_df["state_id"] = state_id
    annotation_dfs.append(annotation_df)
annotation_df = pd.concat(annotation_dfs, ignore_index=True)

annotation_df["annotation_count"] = annotation_df[["soma", "shaft", "spine"]].sum(
    axis=1
)
manual_labels = annotation_df.query("annotation_count == 1").copy()
manual_labels["tag"] = manual_labels[["soma", "shaft", "spine"]].idxmax(axis=1)
manual_labels = manual_labels.dropna(subset="description")
manual_labels["synapse_id"] = manual_labels["description"].astype(int)
manual_labels = manual_labels.set_index("synapse_id")

validation_synapses = client.materialize.query_table(
    "synapses_pni_2", filter_in_dict=dict(id=manual_labels.index.tolist())
).set_index("id")

# %% compute features for the validation synapses using the same table pipeline

validation_synapse_ids = pl.DataFrame(
    {
        "synapse_id": validation_synapses.index.to_numpy(),
        "root_id": validation_synapses["post_pt_root_id"].to_numpy(),
    }
)

# NOTE ALL OF THIS NEEDS AN UPDATE FOR THE NEW SCHEME OF TABLES
validation_features = (
    add_distance_to_nucleus(
        synapse_features(validation_synapse_ids, as_of=LABELED_AS_OF).lazy(),
        Tables["root_nuclei"].lazy(),
    )
    .collect()
    .drop_nulls(subset=spectral_columns)
    .to_pandas()
    .set_index("synapse_id")
)

val_index = validation_features.index.intersection(manual_labels.index)
X_validation = validation_features.loc[val_index, X_train.columns]

# %% score the model against the validation set

pred_label = pd.Series(model.predict(X_validation), index=val_index, name="pred_label")
posteriors = pd.DataFrame(
    model.predict_proba(X_validation), index=val_index, columns=model.classes_
)
posterior_max = posteriors.max(axis=1).rename("posterior_max")

qt = QuantileTransformer(
    n_quantiles=min(1000, len(val_index)), output_distribution="uniform"
)
posterior_max_rank = pd.Series(
    qt.fit_transform(posterior_max.to_numpy().reshape(-1, 1)).reshape(-1),
    index=val_index,
    name="posterior_max_rank",
)
dump(qt, MODEL_PATH / "synapse_hks_posterior_max_qt.joblib")

val_df = pd.concat([pred_label, posteriors, posterior_max, posterior_max_rank], axis=1)
val_df["manual_label"] = manual_labels.loc[val_index, "tag"]
val_df["post_pt_root_id"] = validation_synapses.loc[val_index, "post_pt_root_id"]
val_df["correct"] = val_df["manual_label"] == val_df["pred_label"]

save_variables(model_validation_n_synapses=len(val_df), format="{:,}")
save_variables(
    model_validation_accuracy=accuracy_score(val_df["manual_label"], val_df["pred_label"]),
    model_validation_weighted_f1=f1_score(
        val_df["manual_label"], val_df["pred_label"], average="weighted"
    ),
    format="{:.3f}",
)

# %% reliability (calibration) plot: for each class, does posterior probability
# track the empirical frequency of that class being the correct manual label?

fig, ax = plt.subplots(figsize=(6, 6))
for label_class in model.classes_:
    bins = pd.cut(val_df[label_class], bins=np.linspace(0, 1, 21), include_lowest=True)
    is_class = val_df["manual_label"] == label_class
    bin_props = is_class.groupby(bins, observed=True).mean()
    bin_mids = [interval.mid for interval in bin_props.index]
    sns.lineplot(x=bin_mids, y=bin_props.to_numpy(), marker="o", ax=ax, label=label_class)
ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=2, zorder=-1)
ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    xlabel="Predicted posterior",
    ylabel="Proportion of true label",
)
ax.legend(title="Class")

save_matplotlib_figure(fig, "hks_synapse_calibration", figure_out_path)

# %% confusion matrix on the validation set

mat = confusion_matrix(val_df["manual_label"], val_df["pred_label"], labels=model.classes_)
mat = pd.DataFrame(mat, index=model.classes_, columns=model.classes_)
mat.index.name = "manual_label"
mat.columns.name = "pred_label"

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", square=True, cbar=False, ax=ax)
ax.set_title("Validation confusion matrix")

save_matplotlib_figure(fig, "hks_synapse_validation_confusion", figure_out_path)

# %% validation accuracy broken down by post-synaptic cell type

cell_info = load_neuron_info(version=VERSION)
val_df["post_cell_type"] = val_df["post_pt_root_id"].map(cell_info["cell_type"])
accuracy_by_type = (
    val_df.groupby("post_cell_type", observed=True)["correct"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 4))
sns.stripplot(data=accuracy_by_type, x="post_cell_type", y="correct", ax=ax)
ax.set(xlabel="Post-synaptic cell type", ylabel="Validation accuracy")
plt.xticks(rotation=45)

save_matplotlib_figure(
    fig, "hks_synapse_validation_accuracy_by_cell_type", figure_out_path
)