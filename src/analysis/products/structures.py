"""Spine structures (S4), from the meshmash-deployment tables.

A `structure_id` is valid only inside one pinned S4 snapshot, so these reads
key it by `root_id` and the pins in `sources.py`.
"""

from datetime import datetime

import polars as pl

from ..sources import DATASTACK, deployment_path
from .deployment import latest_attempts, scan_roots
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


# NOTE: the domain predictions also have `pred_label` and `posterior_entropy`, so
# the unitary-spine columns take a prefix. The posterior class names differ.
UNITARY_RENAMES = {
    "pred_label": "unitary_pred_label",
    "posterior_entropy": "unitary_posterior_entropy",
}


def read_structure_predictions(
    root_ids: list[int], *, as_of: datetime, datastack: str = DATASTACK
) -> pl.DataFrame:
    """`structure_predictions` (the unitary-spine model), one row per structure of
    `root_ids` at `as_of`, with `pred_label` and `posterior_entropy` renamed as in
    `UNITARY_RENAMES`."""
    (predictions,) = latest_attempts(
        [scan_roots(deployment_path("structure_predictions", datastack), root_ids, as_of=as_of)]
    )
    return (
        predictions.collect(engine="streaming")
        .rename(UNITARY_RENAMES)
        .sort("root_id", "structure_id")
    )


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


def _read_synapse_structure(
    root_ids: list[int], *, as_of: datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    synapses = _root_synapses(root_ids, as_of=as_of)
    # NOTE: the full join keeps a structure with synapses but no prediction (its
    # root is not predicted yet) and a predicted structure with no synapse.
    structures = (
        _count_inputs(synapses)
        .join(
            read_structure_predictions(root_ids, as_of=as_of),
            on=["root_id", "structure_id"],
            how="full",
            coalesce=True,
        )
        .with_columns(pl.col("n_synapses", "n_pre_root_ids").fill_null(0))
        .pipe(_add_is_multi)
        .sort("root_id", "structure_id")
    )
    return synapses, structures


def spine_structures(root_ids: list[int], *, as_of: datetime) -> pl.DataFrame:
    """One row per spine structure of the post roots `root_ids` at `as_of`, keyed
    by (`root_id`, `structure_id`).

    - `n_synapses` and `n_pre_root_ids`: the synapses on the structure and their
      distinct presynaptic roots, from every synapse in the synapse table (v1822)
      on the root, not from a selection.
    - `posterior_unitary`, `posterior_not_unitary`, `unitary_pred_label` and
      `unitary_posterior_entropy`: the unitary-spine model, null for a structure
      with no prediction.
    - `is_multi`: see `_add_is_multi`.

    It reads every table at `as_of`. To scan many roots, call it on the roots of
    one hash bucket at a time.
    """
    return _read_synapse_structure(root_ids, as_of=as_of)[1]


def _join_spines(structures: pl.DataFrame, spines: pl.DataFrame) -> pl.DataFrame:
    """`structures` (from `read_structures`) joined to `spines` (from
    `spine_structures`) on (`root_id`, `structure_id`).

    Both have `n_synapses`. The one from `spines` stays, because it comes from the
    same synapse read as `n_pre_root_ids`.
    """
    key = ["root_id", "structure_id"]
    joined = structures.join(spines, on=key, how="full", coalesce=True, suffix="_spines")
    assert len(joined) == len(structures), (
        "spine_structures holds a structure with no morphometry"
    )
    both = joined.drop_nulls(["n_synapses", "n_synapses_spines"])
    assert (both["n_synapses"] == both["n_synapses_spines"]).all(), (
        "structure_morphometry and the synapse table count different synapses"
    )
    return (
        joined.drop("n_synapses")
        .rename({"n_synapses_spines": "n_synapses"})
        .sort(key)
    )


def read_spines(root_ids: list[int], *, as_of: datetime) -> pl.DataFrame:
    """One row per spine structure of the post roots `root_ids` at `as_of`, keyed
    by (`root_id`, `structure_id`): the morphometry and spectral features of
    `read_structures`, and the columns of `spine_structures`.

    A table of spines on other roots is another call on those roots.
    """
    return _join_spines(
        read_structures(root_ids, as_of=as_of), spine_structures(root_ids, as_of=as_of)
    )


def synapse_structure(root_ids: list[int], *, as_of: datetime) -> pl.DataFrame:
    """One row per synapse in the synapse table (v1822) onto the post roots
    `root_ids`, with `synapse_id`, `root_id`, `pre_pt_root_id`, the `structure_id`
    of the spine structure it is on (from `synapse_mapping` and
    `vertex_structures`), and the columns of `spine_structures` for that structure.

    The structure columns are null for a synapse in no structure. This replaces the
    old synapse component table. It is a cloud product: a lazy read per root set,
    with no cache. To scan many roots, call it on the roots of one hash bucket at a
    time.
    """
    synapses, structures = _read_synapse_structure(root_ids, as_of=as_of)
    return synapses.join(structures, on=["root_id", "structure_id"], how="left")
