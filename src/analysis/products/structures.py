"""Spine structures (S4), from the meshmash-deployment tables.

A `structure_id` is valid only inside one pinned S4 snapshot, so these reads
key it by `root_id` and the pins in `sources.py`.
"""

from datetime import datetime

import polars as pl

from .deployment import delta_table, latest_attempts, scan_roots
from .features import synapse_vertex_join
from .sources import DATASTACK, MICRONS_SYNAPSES_PATH, deployment_path


def synapse_structure_join(
    synapses: pl.DataFrame, *, as_of: datetime, datastack: str = DATASTACK
) -> pl.DataFrame:
    """`synapses` (with `root_id` and `synapse_id`) with the `structure_id` of each
    synapse, through `synapse_mapping` and `vertex_structures` at `as_of`.

    `structure_id` is null for a synapse in no structure, and for a synapse whose
    root the deployment has not processed.
    """
    return synapses.join(
        synapse_vertex_join(
            synapses, "vertex_structures", "structure_id", as_of=as_of, datastack=datastack
        ),
        on=["root_id", "synapse_id"],
        how="left",
    )


def read_structures(
    root_ids: list[int], *, as_of: datetime, datastack: str = DATASTACK
) -> pl.DataFrame:
    """`structure_morphometry` joined to `structure_features`, one row per structure
    of `root_ids` at `as_of`."""
    morphometry, features = latest_attempts(
        [
            scan_roots(deployment_path(product, datastack), root_ids, as_of=as_of)
            for product in ("structure_morphometry", "structure_features")
        ]
    )
    morphometry, features = pl.collect_all([morphometry, features], engine="streaming")
    joined = morphometry.join(features, on=["root_id", "structure_id"], how="full", coalesce=True)
    assert len(joined) == len(morphometry) == len(features), (
        "structure_morphometry and structure_features hold different structures"
    )
    return joined.sort("root_id", "structure_id")


def read_structure_predictions(
    root_ids: list[int], *, as_of: datetime, datastack: str = DATASTACK
) -> pl.DataFrame:
    """`structure_predictions`, one row per structure of `root_ids` at `as_of`."""
    (predictions,) = latest_attempts(
        [scan_roots(deployment_path("structure_predictions", datastack), root_ids, as_of=as_of)]
    )
    return predictions.collect(engine="streaming").sort("root_id", "structure_id")


def structure_pre_roots(
    root_ids: list[int], *, as_of: datetime
) -> pl.DataFrame:
    """`root_id`, `structure_id` and `n_pre_root_ids`, the number of distinct
    presynaptic roots (S0, v1822) among the synapses on each structure of `root_ids`.

    A structure with no mapped synapse has no row.
    """
    synapses = (
        pl.scan_delta(delta_table(str(MICRONS_SYNAPSES_PATH)))
        .filter(pl.col("post_pt_root_id").is_in(root_ids))
        .select(
            pl.col("post_pt_root_id").alias("root_id"),
            pl.col("id").alias("synapse_id"),
            pl.col("pre_pt_root_id"),
        )
        .collect()
    )
    return (
        synapse_structure_join(synapses, as_of=as_of)
        .drop_nulls("structure_id")
        .group_by("root_id", "structure_id")
        .agg(pl.col("pre_pt_root_id").n_unique().alias("n_pre_root_ids"))
        .sort("root_id", "structure_id")
    )
