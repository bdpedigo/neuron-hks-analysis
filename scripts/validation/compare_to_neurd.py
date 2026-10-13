# %%
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score

from analysis import (
    CELL_TYPE_CATEGORIES,
    CELL_TYPE_PALETTE,
    Tables,
    cell_type_mapper,
    save_matplotlib_figure,
    save_variables,
    set_matplotlib_theme,
)

# %%
# NOTE: NEURD labels about two thirds of the validation synapses, and its axon
# labels have no class (null `neurd_label`). The comparison uses only the
# synapses that both methods label.
label_table_with_both = (
    Tables["validation_predictions"]
    .lazy()
    .filter(
        pl.col("tag") != "other",
        pl.col("pred_label").is_not_null(),
        pl.col("neurd_label").is_not_null(),
    )
    .rename({"tag": "manual_label", "pred_label": "hks_label"})
    .collect()
)
save_variables(compare_n_synapses=len(label_table_with_both), format="{:,}")

print("NEURD")
print(
    classification_report(
        label_table_with_both["manual_label"],
        label_table_with_both["neurd_label"],
    )
)

print("HKS")
print(
    classification_report(
        label_table_with_both["manual_label"],
        label_table_with_both["hks_label"],
    )
)


# %%


def score(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


for method in ["neurd", "hks"]:
    row = score(
        label_table_with_both["manual_label"], label_table_with_both[f"{method}_label"]
    )
    save_variables(prefix=f"compare_{method}_", **row, format="{:.3f}")

# %%
rows = []
for (cell_type,), group in label_table_with_both.group_by("post_cell_type"):
    for method in ["neurd", "hks"]:
        row = score(group["manual_label"], group[f"{method}_label"])
        row["cell_type"] = cell_type
        row["method"] = method
        rows.append(row)
score_results = pl.DataFrame(rows)

# %%

set_matplotlib_theme()

fig, ax = plt.subplots(figsize=(10, 6))

y = "macro_f1"
sns.barplot(
    data=score_results,
    x="cell_type",
    order=CELL_TYPE_CATEGORIES[1:-1],
    y=y,
    hue="method",
    hue_order=["neurd", "hks"],
    palette=["lightgrey", "black"],
    ax=ax,
)

ax.autoscale(False)
ax.fill_between(
    [-1, 6.5], 0, 1, color=CELL_TYPE_PALETTE["Excitatory"], alpha=0.2, zorder=-1
)
ax.fill_between(
    [6.5, 11], 0, 1, color=CELL_TYPE_PALETTE["Inhibitory"], alpha=0.2, zorder=-1
)
ax.text(
    0.0,
    1.0,
    "Excitatory",
    color=CELL_TYPE_PALETTE["Excitatory"],
    transform=ax.transAxes,
    va="bottom",
)
ax.text(
    0.65,
    1.0,
    "Inhibitory",
    color=CELL_TYPE_PALETTE["Inhibitory"],
    transform=ax.transAxes,
    va="bottom",
)
ax.set_xticks(ax.get_xticks())
ax.set_xlabel("Cell Type")
ax.set_xticklabels([cell_type_mapper(ct.get_text()) for ct in ax.get_xticklabels()])
ax.set(ylim=(0, 1), ylabel="Macro F1 Score")
plt.xticks(rotation=45, ha="right")
ax.text(
    -0.2, 0.62, "NEURD", rotation=90, color="black", fontsize=15, va="bottom", ha="center"
)
ax.text(
    +0.2, 0.88, "HKS", rotation=90, color="white", fontsize=15, va="bottom", ha="center"
)
ax.get_legend().remove()

save_matplotlib_figure(fig, "compare_neurd_hks_by_cell_type", subfolder="compare_to_neurd")
