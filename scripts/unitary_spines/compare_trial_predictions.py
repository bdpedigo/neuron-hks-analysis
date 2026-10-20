# %%
"""Compare the deployment's unitary predictions (jobs/unitary_trial) with a local
scoring of the same model on the same structures.

The local model is the folder that was published as `unitary_trial_s3`. Both
sides read the structures of the column run, so the posteriors match up to the
float32 feature cast.
"""

import numpy as np
import pandas as pd
import polars as pl

from analysis import Tables
from analysis.models import load_ssa_model
from analysis.products.sources import COLUMN_JOB_AS_OF, UNITARY_TRIAL_AS_OF
from analysis.products.structures import (
    read_structure_predictions,
    read_structures,
    structure_pre_roots,
)

LOCAL_MODEL = "unitary_hks_trial_s3"
TRIAL_ROOTS = [
    864691134918597642,
    864691134942468707,
    864691135308095046,
    864691135477839784,
    864691135492595551,
    864691135493079135,
    864691135499765267,
    864691135658819202,
    864691136900235630,
    864691136952769247,
]

pl.Config.set_tbl_rows(40)

# %%

deployed = read_structure_predictions(TRIAL_ROOTS, as_of=UNITARY_TRIAL_AS_OF)
print(f"{deployed['root_id'].n_unique()} of {len(TRIAL_ROOTS)} roots have predictions")
print(f"missing: {sorted(set(TRIAL_ROOTS) - set(deployed['root_id'].to_list()))}")

structures = read_structures(TRIAL_ROOTS, as_of=COLUMN_JOB_AS_OF)
counts = (
    structures.group_by("root_id")
    .len("n_structures")
    .join(deployed.group_by("root_id").len("n_predictions"), on="root_id", how="full", coalesce=True)
    .sort("root_id")
)
print(counts)

# %% local scoring

model = load_ssa_model(LOCAL_MODEL)
matrix = (
    structures.select(pl.col(c).cast(pl.Float32) for c in model.feature_columns)
    .to_numpy()
    .astype(np.float32)
)
posteriors = model.estimator.predict_proba(
    pd.DataFrame(matrix, columns=list(model.feature_columns))
)
local = structures.select("root_id", "structure_id").with_columns(
    *[
        pl.Series(f"local_posterior_{name}", posteriors[:, i])
        for i, name in enumerate(model.classes)
    ],
    pl.Series("local_pred_label", np.asarray(model.classes)[posteriors.argmax(axis=1)]),
)
compared = deployed.join(local, on=["root_id", "structure_id"], how="inner")
assert len(compared) == len(deployed), "a deployed structure is not in the local tables"
difference = (compared["posterior_unitary"] - compared["local_posterior_unitary"]).abs()
print(
    f"{len(compared)} structures: max |posterior difference| {difference.max():.2e}, "
    f"label agreement {(compared['pred_label'] == compared['local_pred_label']).mean():.4f}"
)

# %% on the training spines of the trial roots

training = Tables["unitary_training_spines"].collect().select("root_id", "structure_id", "label")
on_training = compared.join(training, on=["root_id", "structure_id"], how="inner")
print(
    f"{len(on_training)} training spines: deployed label = local label "
    f"{(on_training['pred_label'] == on_training['local_pred_label']).mean():.4f}, "
    f"= hand label {(on_training['pred_label'] == on_training['label']).mean():.4f} "
    "(in-sample)"
)

# %% the share of putative multi-input spines called unitary, per root

multi = (
    compared.join(
        structure_pre_roots(compared["root_id"].unique().to_list(), as_of=COLUMN_JOB_AS_OF),
        on=["root_id", "structure_id"],
        how="left",
    )
    .filter(pl.col("n_pre_root_ids") >= 2)
    .with_columns(is_label_root=pl.col("root_id").is_in(training["root_id"].unique().implode()))
)
print(
    multi.group_by("root_id", "is_label_root")
    .agg(
        pl.len().alias("n_putative_multi"),
        (pl.col("pred_label") == "unitary").mean().alias("p_unitary"),
    )
    .sort("is_label_root", "root_id")
)
