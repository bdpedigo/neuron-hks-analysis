# %%
"""Train the unitary-spine model (P13) on the V2 training spines.

Two feature sets, both with the structure geometry: the HKS means, and every
spectral mean (HKS, curvature, normal). Cross-validation is grouped by cell, and
also ungrouped, which is how the paper's 0.951 was computed. The glued CV F1 is
that of `MAIN_UNITARY_MODEL`, grouped by cell.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate

from analysis import Tables, save_variables
from analysis.models import (
    MAIN_UNITARY_MODEL,
    MODEL_VERSION,
    UNITARY_MODEL_SPECS,
    UNITARY_RF_PARAMS,
    feature_columns,
    save_model,
)
from analysis.products.unitary_spines import UNITARY_CLASSES

N_FOLDS = 10
POSITIVE = UNITARY_CLASSES[1]

# %%

spines = Tables["unitary_training_spines"].collect().to_pandas()
y = spines["label"]
groups = spines["root_id"]
print(
    f"{len(spines)} spines on {groups.nunique()} cells, "
    f"{(y == POSITIVE).mean():.1%} {POSITIVE}"
)


def make_unitary_model(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(**UNITARY_RF_PARAMS, n_jobs=-1, random_state=random_state)


scoring = {
    "f1": make_scorer(f1_score, pos_label=POSITIVE),
    "precision": make_scorer(precision_score, pos_label=POSITIVE),
    "recall": make_scorer(recall_score, pos_label=POSITIVE),
}
splits = {
    "grouped by cell": (
        StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0),
        groups,
    ),
    "ungrouped": (StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0), None),
}

# %% cross-validation

rows = []
for name, spec in UNITARY_MODEL_SPECS.items():
    X = spines[feature_columns(spec, spines.columns)]
    for split_name, (cv, cv_groups) in splits.items():
        scores = cross_validate(
            make_unitary_model(0), X, y, groups=cv_groups, cv=cv, scoring=scoring
        )
        for metric in scoring:
            values = scores[f"test_{metric}"]
            rows.append(
                {
                    "variant": name,
                    "n_features": X.shape[1],
                    "cv": split_name,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
cv_results = pd.DataFrame(rows)
with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
    print(cv_results.to_string(index=False))

is_main_f1 = (cv_results["variant"] == MAIN_UNITARY_MODEL) & (cv_results["metric"] == "f1")
main_f1 = cv_results[is_main_f1].set_index("cv")
save_variables(model_n_training_spines=len(spines), format="{:,}", prefix="unitary_spines_")
save_variables(
    model_mean_cv_f1_score=main_f1.loc["grouped by cell", "mean"],
    model_mean_cv_f1_score_ungrouped=main_f1.loc["ungrouped", "mean"],
    format="{:.3f}",
    prefix="unitary_spines_",
)
save_variables(
    model_n_estimators=UNITARY_RF_PARAMS["n_estimators"],
    model_max_depth=UNITARY_RF_PARAMS["max_depth"],
    prefix="unitary_spines_",
)

# %% fit each variant on all spines, and save it

importances = {}
for name, spec in UNITARY_MODEL_SPECS.items():
    X = spines[feature_columns(spec, spines.columns)]
    currtime = time.time()
    model = make_unitary_model(0).fit(X, y)
    model_dir = save_model(
        model, X, f"{name}", version=MODEL_VERSION, features_product="structure_features"
    )
    print(f"{name}: {time.time() - currtime:.1f} s, saved to {model_dir}")
    importances[name] = pd.Series(model.feature_importances_, index=X.columns)

# %% feature importances (mean decrease in impurity)

for name, importance in importances.items():
    top = importance.sort_values(ascending=False)
    print(f"\n{name}: top 15 of {len(top)}")
    print(top.head(15).round(4).to_string())
    is_geometry = np.isin(top.index, UNITARY_MODEL_SPECS[name]["extra_features"])
    print(f"geometry share {top[is_geometry].sum():.3f}, spectral share {top[~is_geometry].sum():.3f}")
