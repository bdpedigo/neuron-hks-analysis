"""The census tables on the column cells."""

import polars as pl

from ..sources import VERSION, cutoff, pin_params
from .catalog import Tables
from .structures import read_spines

SPINE_PRODUCTS = (
    "synapse_mapping",
    "vertex_structures",
    "structure_morphometry",
    "structure_features",
    "structure_predictions",
)


def column_root_ids() -> list[int]:
    """The roots of the column cells in `cell_selection`."""
    return (
        Tables["cell_selection"]
        .lazy()
        .filter(pl.col("is_column"))
        .select("pt_root_id")
        .collect()["pt_root_id"]
        .to_list()
    )


# NOTE: the old `filtered_spine_info` also held the spines on other roots that a
# proofread axon reaches, but 5C left and 5E left kept only the column cells
# (`scripts/stat_summaries.py`, the map of `post_broad_type`). Revisit this scope
# with the full-run tables: a wider table is another call of `read_spines`.
Tables.register(
    "column_spines",
    description=(
        "One row per spine structure on a column cell, keyed by (`root_id`, "
        "`structure_id`). It holds the morphometry and spectral features, the "
        "synapse and presynaptic root counts, the unitary-spine posteriors and "
        "`is_multi`. Read at the `test_census` cutoff. It replaces the old "
        "`filtered_spine_info`."
    ),
    build=lambda: read_spines(column_root_ids(), as_of=cutoff("test_census")).lazy(),
    cache_params=lambda: {
        "version": VERSION,
        "cell_selection": Tables["cell_selection"].cache_key,
        **pin_params(*SPINE_PRODUCTS, as_of=cutoff("test_census")),
    },
)
