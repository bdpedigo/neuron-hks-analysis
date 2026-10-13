# %%
"""Fig 6A-C: the MICrONS-trained model applied to the hand-labeled H01 synapses.

Ported from meshrep/sandbox/h01_validation.py. The predictions come from the
deployment `h01` job (P5 `h01_predictions`), not from a local run of an older
model. The mesh panel (6D) and the composite are not ported yet.
"""

import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import accuracy_score, classification_report, f1_score

from analysis import (
    Tables,
    read_variables,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)

SUBFOLDER = "h01_validation"
LABEL_ORDER = ["soma", "shaft", "spine"]

set_matplotlib_theme()

# %%
labels = Tables["h01_predictions"].lazy().collect()
n_missing = labels["pred_label"].null_count()
print(f"{n_missing} of {len(labels)} label synapses have no prediction")
print(labels.filter(pl.col("pred_label").is_null()).group_by("cohort", "label").len())

predictions = labels.drop_nulls("pred_label")

# %%
# The model never saw H01 data, so both cohorts are held out. The Fig 6 cohort
# is `validation`; `retrain` comes from the later labeling round.
scores = []
for cohort, subset in [
    *predictions.group_by("cohort", maintain_order=True),
    (("pooled",), predictions),
]:
    scores.append(
        {
            "cohort": cohort[0],
            "n_synapses": len(subset),
            "n_roots": subset["root_id"].n_unique(),
            "weighted_f1": f1_score(subset["label"], subset["pred_label"], average="weighted"),
            "accuracy": accuracy_score(subset["label"], subset["pred_label"]),
        }
    )
scores = pd.DataFrame(scores).set_index("cohort")
print(scores)
print(classification_report(predictions["label"], predictions["pred_label"]))

# %%
fig6 = predictions.filter(pl.col("cohort") == "validation")
h01_f1 = scores.loc["validation", "weighted_f1"]
save_variables(
    h01_validation_exc_f1=h01_f1,
    h01_validation_accuracy=scores.loc["validation", "accuracy"],
    format="{:.3f}",
)
save_variables(
    h01_validation_n_synapses=int(scores.loc["validation", "n_synapses"]),
    h01_validation_n_roots=int(scores.loc["validation", "n_roots"]),
    format="{:,}",
)

# %%
# 6A: the MICrONS bar is the excitatory validation F1 that validation.py writes.
microns_f1 = float(read_variables()["validation_exc_f1"])
results = pd.DataFrame(
    {
        "dataset": ["MICrONS\n(mouse)", "H01\n(human)"],
        "value": [microns_f1, h01_f1],
    }
)
palette = {"MICrONS\n(mouse)": "#000000", "H01\n(human)": "#ffffff"}


def score_comparison(height: float, title_y: float):
    fig, ax = plt.subplots(figsize=(3, height))
    sns.barplot(
        data=results, x="dataset", y="value", hue="dataset", palette=palette, dodge=False, ax=ax
    )
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(2)
    ax.set(xlabel="Dataset", ylabel="Validation weighted F1 score", ylim=(0, 1))
    for row in results.itertuples():
        ax.text(x=row.dataset, y=row.value + 0.01, s=f"{row.value:.2f}", ha="center", va="bottom")
    ax.annotate(
        "",
        xy=(1, 0.5),
        xytext=(0, 0.5),
        arrowprops=dict(facecolor="white", width=15, headwidth=30),
        ha="center",
        va="center",
    )
    ax.set_title("MICrONS-trained model\n applied to H01", y=title_y)
    return fig


save_matplotlib_figure(score_comparison(6, 1.05), "h01_microns_score_comparison", subfolder=SUBFOLDER)
save_matplotlib_figure(
    score_comparison(4, 1.1), "h01_microns_score_comparison_short", subfolder=SUBFOLDER
)

# %%
# 6B: the row-normalized confusion matrix.
confusion_matrix = pd.crosstab(
    fig6["label"].to_pandas(), fig6["pred_label"].to_pandas(), normalize="index"
).reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0.0)

fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(
    confusion_matrix,
    ax=ax,
    square=True,
    cmap="Greys",
    vmin=0,
    vmax=1,
    annot=True,
    fmt=".2f",
    cbar=False,
)
ax.spines[["top", "right", "left", "bottom"]].set_visible(True)
ax.set_xticklabels([label.get_text().capitalize() for label in ax.get_xticklabels()])
ax.set_yticklabels(
    [label.get_text().capitalize() for label in ax.get_yticklabels()], rotation=0
)
ax.set(xlabel="Predicted label", ylabel="Manual label", title="H01 confusion matrix\n(row normalized)")
save_matplotlib_figure(fig, "h01_confusion_matrix", subfolder=SUBFOLDER)

# %%
# 6C: calibration of P(spine), spine against shaft or soma.
fig, ax = plt.subplots(figsize=(4, 4))
CalibrationDisplay.from_predictions(
    (fig6["label"] == "spine").to_numpy().astype(int),
    fig6["posterior_spine"].to_numpy(),
    n_bins=10,
    strategy="uniform",
    ax=ax,
    color="black",
)
ax.set(xlabel="Predicted spine probability ", ylabel="Fraction correct")
ax.set_title("Calibration curve\n(spine vs. shaft or soma)")
ax.get_legend().remove()
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
save_matplotlib_figure(fig, "h01_calibration_curve_binary", subfolder=SUBFOLDER)

# %%
# The calibration of each class (not in the paper).
fig, ax = plt.subplots(figsize=(5, 5))
for label in ["spine", "shaft", "soma"]:
    CalibrationDisplay.from_predictions(
        (fig6["label"] == label).to_numpy().astype(int),
        fig6[f"posterior_{label}"].to_numpy(),
        n_bins=10,
        ax=ax,
        name=label.capitalize(),
    )
ax.set_title("Calibration curve (H01 validation)")
save_matplotlib_figure(fig, "h01_calibration_curve", subfolder=SUBFOLDER)
