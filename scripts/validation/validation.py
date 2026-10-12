# %%
from pathlib import Path

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score

from meshrep import (
    CELL_TYPE_CATEGORIES,
    CELL_TYPE_PALETTE,
    TABLE_PATHS,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)


def cell_type_mapper(cell_type):
    match cell_type:
        case "23P":
            return "2/3P-IT"
        case "4P":
            return "4P-IT"
        case "5P-PT":
            return "5P-ET"
        case _:
            return cell_type


data_path = Path("/Users/ben.pedigo/code/meshrep/meshrep/data/validation")
cols = [
    "synapse_id",
    "pre_pt_root_id",
    "post_pt_root_id",
    "post_cell_type",
    "post_nucleus_id",
    "tag",
]
exc_labels = pl.read_csv(data_path / "exc_sample_synapses_labeled.csv").select(cols)
inh_labels = pl.read_csv(data_path / "inh_sample_synapses_labeled.csv").select(cols)
labels = pl.concat([exc_labels, inh_labels])
labels = labels.rename({"tag": "manual_label"})

predictions = pl.scan_delta(TABLE_PATHS["prediction"]).filter(
    pl.col("synapse_id").is_in(labels["synapse_id"].to_list())
)
predictions = predictions.rename({"label": "predicted_label"})
predictions = predictions.collect()

# %%
predictions = predictions.join(labels, on="synapse_id", how="full")
predictions = predictions.filter(
    pl.col("manual_label").is_not_null(),
    pl.col("predicted_label").is_not_null(),
    pl.col("predicted_label") != "unknown",
    pl.col("manual_label") != "other",
)

# %%
predictions = predictions.with_columns(
    (predictions["predicted_label"] == predictions["manual_label"]).alias("is_correct"),
    (
        pl.col("post_cell_type")
        .replace(
            {
                "23P": "Excitatory",
                "4P": "Excitatory",
                "5P-IT": "Excitatory",
                "5P-PT": "Excitatory",
                "5P-ET": "Excitatory",
                "5P-NP": "Excitatory",
                "6P-CT": "Excitatory",
                "6P-IT": "Excitatory",
                "BC": "Inhibitory",
                "BPC": "Inhibitory",
                "MC": "Inhibitory",
                "NGC": "Inhibitory",
            }
        )
        .alias("post_broad_type")
    ),
)
# %%
predictions["is_correct"].mean()

# %%
acc_data = (
    predictions.group_by("post_cell_type")
    .agg(
        pl.col("is_correct").mean().alias("accuracy"),
        pl.len().alias("n_synapses"),
    )
    .sort("accuracy")
)

set_matplotlib_theme()

CELL_TYPE_PALETTE["5P-PT"] = CELL_TYPE_PALETTE["5P-ET"]

fig, ax = plt.subplots(figsize=(8.67, 5.85))

acc_data = acc_data.with_columns(
    acc_data["post_cell_type"]
    .replace(
        {
            "23P": "Excitatory",
            "4P": "Excitatory",
            "5P-IT": "Excitatory",
            "5P-PT": "Excitatory",
            "5P-ET": "Excitatory",
            "5P-NP": "Excitatory",
            "6P-CT": "Excitatory",
            "6P-IT": "Excitatory",
            "BC": "Inhibitory",
            "BPC": "Inhibitory",
            "MC": "Inhibitory",
            "NGC": "Inhibitory",
        }
    )
    .alias("broad_type")
).rename({"post_cell_type": "cell_type"})

sns.stripplot(
    data=acc_data,
    x="cell_type",
    y="accuracy",
    hue="broad_type",
    order=CELL_TYPE_CATEGORIES[1:-1],
    ax=ax,
    size=10,
    jitter=False,
    palette=CELL_TYPE_PALETTE,
    legend=False,
)
ax.set_xlabel("Cell Type")
ax.set_ylabel("Accuracy")

ax.autoscale(False)
ax.fill_between([-1, 6.5], 0, 1, color=CELL_TYPE_PALETTE["Excitatory"], alpha=0.2)
ax.fill_between([6.5, 11], 0, 1, color=CELL_TYPE_PALETTE["Inhibitory"], alpha=0.05)
ax.text(
    0.05,
    0.98,
    "Excitatory",
    color=CELL_TYPE_PALETTE["Excitatory"],
    transform=ax.transAxes,
    va="top",
)
ax.text(
    0.67,
    0.98,
    "Inhibitory",
    color=CELL_TYPE_PALETTE["Inhibitory"],
    transform=ax.transAxes,
    va="top",
)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels([cell_type_mapper(ct.get_text()) for ct in ax.get_xticklabels()])

# rotate x-axis labels 45
plt.xticks(rotation=45, ha="right")
ax.set(ylim=(0.9, 1))

save_matplotlib_figure(
    fig,
    "validation_accuracy_by_cell_type",
    subfolder="validation",
)
#%%
total_predictions = predictions
print(
    classification_report(
        total_predictions["manual_label"],
        total_predictions["predicted_label"],
    )
)
total_accuracy = accuracy_score(
    total_predictions["manual_label"],
    total_predictions["predicted_label"],
)
total_macro_f1 = f1_score(
    total_predictions["manual_label"],
    total_predictions["predicted_label"],
    average="macro",
)
total_weighted_f1 = f1_score(
    total_predictions["manual_label"],
    total_predictions["predicted_label"],
    average="weighted",
) 

save_variables(
    total_accuracy=total_accuracy,
    total_macro_f1=total_macro_f1,
    total_weighted_f1=total_weighted_f1,
    prefix="validation_",
    format="{:.3f}",
)

# %%

excitatory_predictions = predictions.filter(pl.col("post_broad_type") == "Excitatory")

print(
    classification_report(
        excitatory_predictions["manual_label"],
        excitatory_predictions["predicted_label"],
    )
)
exc_accuracy = accuracy_score(
    excitatory_predictions["manual_label"],
    excitatory_predictions["predicted_label"],
)
exc_macro_f1 = f1_score(
    excitatory_predictions["manual_label"],
    excitatory_predictions["predicted_label"],
    average="macro",
)
exc_weighted_f1 = f1_score(
    excitatory_predictions["manual_label"],
    excitatory_predictions["predicted_label"],
    average="weighted",
)

save_variables(
    exc_accuracy=exc_accuracy,
    exc_macro_f1=exc_macro_f1,
    exc_weighted_f1=exc_weighted_f1,
    prefix="validation_",
    format="{:.3f}",
)

# %%

inhibitory_predictions = predictions.filter(pl.col("post_broad_type") == "Inhibitory")

inh_accuracy = accuracy_score(
    inhibitory_predictions["manual_label"],
    inhibitory_predictions["predicted_label"],
)
inh_macro_f1 = f1_score(
    inhibitory_predictions["manual_label"],
    inhibitory_predictions["predicted_label"],
    average="macro",
)
inh_weighted_f1 = f1_score(
    inhibitory_predictions["manual_label"],
    inhibitory_predictions["predicted_label"],
    average="weighted",
)

save_variables(
    inh_accuracy=inh_accuracy,
    inh_macro_f1=inh_macro_f1,
    inh_weighted_f1=inh_weighted_f1,
    prefix="validation_",
    format="{:.3f}",
)

# %%

# compute a cross tabulation of predicted vs manual labels, grouping by post_cell_type
# normalize by the number of synapses in each manual label
cell_type_crosstab = (
    predictions.group_by(["post_cell_type", "manual_label", "predicted_label"])
    .agg(pl.len().alias("n_synapses"))
    .join(
        predictions.group_by(["post_cell_type", "manual_label"]).agg(
            pl.len().alias("total_synapses")
        ),
        on=["post_cell_type", "manual_label"],
        how="left",
    )
    .with_columns((pl.col("n_synapses") / pl.col("total_synapses")).alias("proportion"))
    .sort(["post_cell_type", "predicted_label"])
)

broad_type_crosstab = (
    predictions.group_by(["post_broad_type", "manual_label", "predicted_label"])
    .agg(pl.len().alias("n_synapses"))
    .join(
        predictions.group_by(["post_broad_type", "manual_label"]).agg(
            pl.len().alias("total_synapses")
        ),
        on=["post_broad_type", "manual_label"],
        how="left",
    )
    .with_columns((pl.col("n_synapses") / pl.col("total_synapses")).alias("proportion"))
    .sort(["post_broad_type", "predicted_label"])
)


def make_confusion_matrix_for_subset(crosstab, filter_key, filter_value):
    subset = crosstab.filter(pl.col(filter_key) == filter_value)
    confusion_matrix = subset.pivot(
        index="manual_label", on="predicted_label", values="proportion"
    )
    confusion_matrix = confusion_matrix.to_pandas().set_index("manual_label")
    confusion_matrix = confusion_matrix.reindex(
        index=["soma", "shaft", "spine"],
        columns=["soma", "shaft", "spine"],
        fill_value=0.0,
    )
    return confusion_matrix


scale = 3.5
n_cols = 8
fig, axs = plt.subplots(
    2,
    n_cols,
    figsize=(scale * 0.9 * n_cols, scale * 2),
    # layout="constrained",
    gridspec_kw=dict(wspace=0.00, hspace=0.6),
)
categories = CELL_TYPE_CATEGORIES[1:-1]
categories = ["excitatory"] + [
    "23P",
    "4P",
    "5P-IT",
    "5P-ET",
    "5P-NP",
    "6P-IT",
    "6P-CT",
]
categories += ["inhibitory"] + ["BC", "BPC", "MC", "NGC"]


for i in range(len(categories), len(axs.flat)):
    # axs.flat[i].remove()
    ax = axs.flat[i]
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_yticklabels([])

for i, cell_type in enumerate(categories):
    if cell_type == "excitatory" or cell_type == "inhibitory":
        broad_type = cell_type.capitalize()
        confusion_matrix = make_confusion_matrix_for_subset(
            broad_type_crosstab, "post_broad_type", broad_type
        )

    else:
        confusion_matrix = make_confusion_matrix_for_subset(
            cell_type_crosstab, "post_cell_type", cell_type
        )

    ax = axs.flat[i]
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
    title = cell_type_mapper(cell_type)
    if cell_type == "excitatory" or cell_type == "inhibitory":
        title = "All " + title
    ax.set(title=title, xlabel="", ylabel="")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(True)

for i in range(2):
    for j in range(n_cols):
        ax = axs[i, j]
        if not ax.has_data():
            continue
        # ax.set_xticklabels(
        #     [label.get_text().capitalize() for label in ax.get_xticklabels()],
        # )

        ax.set_xticklabels(
            [label.get_text().capitalize() for label in ax.get_xticklabels()],
        )
        ax.set_xlabel("Predicted")

        if j == 0:
            ax.set_yticklabels(
                [label.get_text().capitalize() for label in ax.get_yticklabels()],
                rotation=0,
            )
            ax.set_ylabel("Manual")
        else:
            ax.set_yticks([])

# draw a vertical line between the first column and second column
ax = axs[0, 0]
x = (ax.get_position().x1 + axs[0, 1].get_position().x0) / 2
ax.plot(
    [x, x],
    [0.05, 0.93],
    color="black",
    lw=3,
    transform=fig.transFigure,
    clip_on=False,
    linestyle="-",
    solid_capstyle="round",
)
fig.suptitle("Confusion matrices (row normalized)")


save_matplotlib_figure(
    fig,
    "validation_confusion_matrices",
    subfolder="validation",
)

# %%

fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 4), layout="constrained")
broad_types = ["Excitatory", "Inhibitory"]
for i, broad_type in enumerate(broad_types):
    confusion_matrix = make_confusion_matrix_for_subset(
        broad_type_crosstab, "post_broad_type", broad_type
    )

    ax = axs.flat[i]
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
    ax.set_title(f"{broad_type}")

# %%
predictions = predictions.with_columns(
    pl.col("predicted_label")
    .replace({"soma": "not_spine", "shaft": "not_spine"})
    .alias("predicted_is_spine_label"),
    pl.col("manual_label")
    .replace({"soma": "not_spine", "shaft": "not_spine"})
    .alias("manual_is_spine_label"),
)
import numpy as np
from sklearn.metrics import precision_score, recall_score

rows = []
for threshold in np.linspace(0, 1, 101):
    subset = predictions.filter(
        pl.col("p_spine") >= threshold, pl.col("manual_label").is_not_null()
    )
    y_true = subset["manual_is_spine_label"].to_list()
    y_pred = subset["predicted_is_spine_label"].to_list()
    precision = precision_score(y_true, y_pred, pos_label="spine", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="spine", zero_division=0)

    rows.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
        }
    )
pr_data = pl.DataFrame(rows)


fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(pr_data["recall"], pr_data["precision"], marker=".")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve for Spine Synapse Detection")

# %%
from sklearn.metrics import roc_curve

y_true = predictions["manual_is_spine_label"].to_list()
y_scores = predictions["p_spine"].to_list()
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label="spine")
# %%
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(fpr, tpr, color="black")
idx = np.argmin(np.abs(thresholds - 0.5))
ax.plot(fpr[idx], tpr[idx], marker="o", color="red")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve (spine vs. not spine)")

save_matplotlib_figure(
    fig,
    "validation_roc_curve_spine_detection",
    subfolder="validation",
)
