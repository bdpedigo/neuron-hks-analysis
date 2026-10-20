# %%
"""How the unitary-spine labels (S11) carry over to the new spine structures.

Each label is a spine of the old pipeline. Its synapses map it to the new
structures: kept, split, merged or lost (see `products/unitary_spines.py`). The
cells with no structures yet are listed apart.
"""

import polars as pl

from analysis import Tables
from analysis.products.unitary_spines import OUTCOMES, TAG_CATEGORIES

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)

outcomes = Tables["unitary_label_outcomes"].collect()
training = Tables["unitary_training_spines"].collect()

# %% the outcome of each label, by its hand tag (a label can have several tags)

by_tag = pl.concat(
    [
        outcomes.filter(pl.col(tag))
        .group_by("outcome")
        .len()
        .with_columns(tag=pl.lit(tag))
        for tag in TAG_CATEGORIES
    ]
    + [outcomes.group_by("outcome").len().with_columns(tag=pl.lit("all"))]
)
print(
    by_tag.pivot("outcome", index="tag", values="len")
    .fill_null(0)
    .select("tag", *[o for o in OUTCOMES if o in by_tag["outcome"].cast(pl.String)])
)

# %% the counts behind each outcome, on the cells that have structures

print(
    outcomes.filter(pl.col("outcome") != "root_not_done")
    .group_by("outcome")
    .agg(
        pl.len().alias("n_labels"),
        pl.col("n_synapses").sum(),
        pl.col("n_on_structure").sum(),
        (pl.col("n_on_structure") < pl.col("n_synapses")).sum().alias("n_partly_on"),
    )
    .sort("outcome")
)

# %% the label cells with no structures at the pin

not_done = (
    outcomes.filter(pl.col("outcome") == "root_not_done")
    .group_by("root_id", "label_root_id")
    .len("n_labels")
    .sort("root_id")
)
print(f"{len(not_done)} label cells have no structures:")
print(not_done)

# %% the training spines

print(f"{len(training)} training spines on {training['root_id'].n_unique()} cells")
print(training.group_by("label").len().sort("label"))
print(training.group_by("n_pre_root_ids").len().sort("n_pre_root_ids"))
