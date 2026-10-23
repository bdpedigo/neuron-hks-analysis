"""The tables of the full-run census scripts (`scripts/census/outputs.py`,
`inhibitory_targeting.py`, `whole_dataset.py`).

The full-run tables are `proofread_pre_synapses` and `post_aggregation` (TASK-18).
Until they exist, the scripts run with `--scratch`, and each function here then
gives a test-cohort stand-in. A scratch run writes to `figures/census/_scratch/` and
writes no glued values. Its figures only show that the code runs: they are not
results.
"""

from pathlib import Path

import polars as pl

from .io import FIG_PATH
from .products import Tables
from .utils import save_variables


def out_path(scratch: bool) -> Path:
    if scratch:
        return FIG_PATH / "census" / "_scratch"
    return FIG_PATH / "census"


def glue(scratch: bool, **kwargs) -> None:
    """`save_variables`, except in a scratch run."""
    if not scratch:
        save_variables(**kwargs)


def pre_synapses(scratch: bool) -> pl.LazyFrame:
    """The synapses from the proofread cells onto every target. The scratch stand-in
    holds only the targets that are column cells."""
    if not scratch:
        return Tables["proofread_pre_synapses"].lazy()
    proofread_ids = (
        Tables["cell_selection"]
        .lazy()
        .filter(pl.col("is_proofread"))
        .select("pt_root_id")
    )
    return (
        Tables["column_post_synapses"]
        .lazy()
        .join(
            proofread_ids, left_on="pre_pt_root_id", right_on="pt_root_id", how="semi"
        )
    )


def post_cells(scratch: bool) -> pl.DataFrame:
    """One row per neuron with its `post_*` aggregation, the columns of `cell_info`
    and `is_column`. The scratch stand-in holds only the column cells."""
    if scratch:
        return (
            Tables["proofread_cell_info"].lazy().filter(pl.col("is_column")).collect()
        )
    flags = Tables["cell_selection"].lazy().select("pt_root_id", "is_column")
    return (
        Tables["post_aggregation"]
        .lazy()
        .rename({"post_pt_root_id": "pt_root_id"})
        .join(Tables["cell_info"].lazy(), on="pt_root_id", how="inner")
        .join(flags, on="pt_root_id", how="left")
        .with_columns(pl.col("is_column").fill_null(False))
        .collect()
    )
