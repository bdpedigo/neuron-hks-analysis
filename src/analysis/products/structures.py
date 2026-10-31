"""Spine structures (S4), from the meshmash-deployment tables.

A `structure_id` is valid only inside one pinned S4 snapshot, so these reads
key it by `root_id` and the pins in `sources.py`.
"""

from datetime import datetime

import polars as pl

from ..sources import DATASTACK, deployment_path
from .deployment import query_latest_attempts, scan_roots
from .features import synapse_vertex_join
from .synapses import scan_post_synapses


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
    morphometry, features = query_latest_attempts(
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


def _root_synapses(root_ids: list[int], *, as_of: datetime) -> pl.DataFrame:
    """`synapse_id`, `root_id`, `pre_pt_root_id` and `structure_id` of every synapse
    in `microns_synapses` (v1822) onto the post roots `root_ids`.

    `structure_id` is null for a synapse in no structure, and for a synapse whose
    root the deployment has not processed.
    """
    synapses = (
        scan_post_synapses(root_ids)
        .select(
            pl.col("id").alias("synapse_id"),
            pl.col("post_pt_root_id").alias("root_id"),
            pl.col("pre_pt_root_id"),
        )
        .collect()
    )
    return synapse_structure_join(synapses, as_of=as_of)


def _count_inputs(synapses: pl.DataFrame) -> pl.DataFrame:
    """`n_synapses` and `n_pre_root_ids` (distinct presynaptic roots) per
    (`root_id`, `structure_id`), from synapses with a `structure_id`.

    A structure with no synapse has no row.
    """
    return (
        synapses.drop_nulls("structure_id")
        .group_by("root_id", "structure_id")
        .agg(
            pl.len().alias("n_synapses"),
            pl.col("pre_pt_root_id").n_unique().alias("n_pre_root_ids"),
        )
        .sort("root_id", "structure_id")
    )


def _add_is_multi(structures: pl.DataFrame) -> pl.DataFrame:
    """Add `is_multi`: more than one presynaptic root, and the unitary-spine model
    calls the structure one spine. A structure with no prediction is not multi-input.
    """
    return structures.with_columns(
        is_multi=(
            (pl.col("n_pre_root_ids") > 1) & (pl.col("unitary_pred_label") == "unitary")
        ).fill_null(False)
    )
