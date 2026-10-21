# NOTE: Historical record of the H01 paper/deployment comparison. It will probably
# not run: it reads pins and cutoffs that no longer exist (see analysis.sources).
# %%
"""Score saved models on the H01 retrain cohort, to compare models locally.

Reads only `h01_retrain_model_inputs` and the retrain rows of `h01_predictions`,
so the validation cohort stays held out. A model here is any MLflow sklearn
folder whose signature columns are in `h01_model_inputs` (see `analysis.models`).
"""

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from analysis import Tables, save_matplotlib_figure, set_matplotlib_theme
from analysis.models import load_ssa_model, predict

SUBFOLDER = "h01_retrain"
MODELS = [
    "ssa_model_full_v1",
    "ssa_model_no_nuc_v1",
    "ssa_model_binary_v1",
    "ssa_model_hks_v1",
    "ssa_model_hks_nuc_v1",
    "ssa_model_hks_binary_v1",
]

set_matplotlib_theme()

inputs = Tables["h01_retrain_model_inputs"].collect()
deployment = (
    Tables["h01_predictions"].lazy().filter(pl.col("cohort") == "retrain").collect()
)
print(f"{len(inputs)} retrain label synapses on {inputs['root_id'].n_unique()} roots")

# %%
# Parity: the local path must reproduce the deployment for the deployed model, so
# that a difference between models comes from the model alone.
local = predict(load_ssa_model("ssa_model_full_v1"), inputs)
posterior_columns = [c for c in local.columns if c.startswith("posterior_")]
parity = local.join(deployment, on="synapse_id", how="full", suffix="_deployment")
assert len(parity) == len(inputs) == len(deployment)
label_mismatch = parity.filter(
    pl.col("pred_label").ne_missing(pl.col("pred_label_deployment"))
)
max_posterior_difference = max(
    (parity[c] - parity[f"{c}_deployment"]).abs().max() or 0.0
    for c in posterior_columns
)
print(f"{len(label_mismatch)} pred_label mismatches")
print(f"largest posterior difference: {max_posterior_difference:.2e}")
if len(label_mismatch):
    print(label_mismatch.select("synapse_id", "pred_label", "pred_label_deployment"))
assert len(label_mismatch) == 0 and max_posterior_difference <= 1e-6

# %%


def to_binary(labels: pl.Series) -> pl.Series:
    return (
        pl.when(labels == "spine").then(pl.lit("spine")).otherwise(pl.lit("not_spine"))
    )


scores = []
confusions = {}
for name in MODELS:
    model = load_ssa_model(name)
    predictions = predict(
        model, inputs, keys=("synapse_id", "root_id", "label")
    ).drop_nulls("pred_label")
    if set(model.classes) == {"spine", "not_spine"}:
        predictions = predictions.with_columns(label=to_binary(pl.col("label")))
    y_true, y_pred = predictions["label"], predictions["pred_label"]
    scores.append(
        {
            "model": name,
            "n_scored": len(predictions),
            "n_unscored": len(inputs) - len(predictions),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "accuracy": accuracy_score(y_true, y_pred),
        }
    )
    order = [c for c in ["soma", "shaft", "spine", "not_spine"] if c in model.classes]
    confusions[name] = pd.crosstab(
        y_true.to_pandas(), y_pred.to_pandas(), normalize="index"
    ).reindex(index=order, columns=order, fill_value=0.0)

scores = pd.DataFrame(scores).set_index("model")
print(scores.round(3))

# %%
n_columns = 3
n_rows = -(-len(MODELS) // n_columns)
fig, axs = plt.subplots(
    n_rows, n_columns, figsize=(4 * n_columns, 4 * n_rows), layout="constrained"
)
for ax in np.ravel(axs)[len(MODELS) :]:
    ax.set_axis_off()
for ax, (name, confusion) in zip(np.ravel(axs), confusions.items()):
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
    ax.set(xlabel="Predicted label", ylabel="Manual label", title=name)
save_matplotlib_figure(fig, "retrain_model_confusion_matrices", subfolder=SUBFOLDER)
