"""P4: the validation labels with the HKS posteriors and the NEURD labels."""


import polars as pl
from caveclient import CAVEclient

from ..sources import DATASTACK, NEURD_LABELS_PATH, VERSION, cutoff, pin_params
from . import labels  # noqa: F401 -- registers "validation_labels", whose cache key P4 reads
from .catalog import Tables, write_parquet_atomic
from .predictions import synapse_posteriors
from .synapses import scan_post_synapses

# The `value` codes of the NEURD table `synapse_target_structure`.
NEURD_DETAILED_LABELS = {
    0: "spine_head",
    1: "spine_neck",
    2: "shaft",
    3: "spine_no_neck",  # a spine with no clear head and neck
    4: "axon_bouton",
    5: "axon_other",
    6: "soma",
}
# The detailed labels in the classes of the validation labels. The axon labels
# have no class, so they map to null.
NEURD_LABELS = {
    "spine_head": "spine",
    "spine_neck": "spine",
    "spine_no_neck": "spine",
    "shaft": "shaft",
    "soma": "soma",
}


def _neurd_classes(targets: pl.DataFrame) -> pl.DataFrame:
    """The rows `id`, `target_id`, `value` of `synapse_target_structure`, with the
    detailed label and the class of each value."""
    return (
        targets.select("id", "target_id", "value")
        .rename({"id": "neurd_id", "target_id": "synapse_id", "value": "neurd_value"})
        .with_columns(
            neurd_detailed_label=pl.col("neurd_value").replace_strict(
                NEURD_DETAILED_LABELS, return_dtype=pl.String
            )
        )
        .with_columns(
            neurd_label=pl.col("neurd_detailed_label").replace_strict(
                NEURD_LABELS, default=None, return_dtype=pl.String
            )
        )
    )


def fetch_neurd_labels(client, synapse_ids: list[int], version=VERSION) -> pl.DataFrame:
    """The NEURD (S7) label of each synapse in `synapse_ids` that NEURD labeled."""
    targets = client.materialize.query_table(
        "synapse_target_structure",
        filter_in_dict=dict(target_id=synapse_ids),
        materialization_version=version,
        log_warning=False,
    )
    return _neurd_classes(pl.from_pandas(targets))


def fetch_census_neurd_labels() -> pl.DataFrame:
    """The NEURD label of each synapse onto a column cell or from a proofread cell,
    one row per synapse, with the flags `on_column` and `from_proofread`.

    The NEURD export has no root IDs, so it is joined to `microns_synapses`. The
    pre side scans every bucket, because the table is bucketed on the post root.
    """
    selection = Tables["cell_selection"].collect()
    column = selection.filter(pl.col("is_column"))["pt_root_id"].to_list()
    proofread = selection.filter(pl.col("is_proofread"))["pt_root_id"].to_list()
    columns = ["id", "pre_pt_root_id", "post_pt_root_id"]
    synapses = pl.concat(
        [
            scan_post_synapses(column).select(columns),
            Tables["microns_synapses"]
            .lazy()
            .filter(pl.col("pre_pt_root_id").is_in(proofread))
            .select(columns),
        ]
    ).unique("id")
    targets = (
        pl.scan_delta(str(NEURD_LABELS_PATH))
        .filter(pl.col("valid"))
        .select("id", "target_id", "value")
    )
    # A synapse from a proofread cell onto a column cell is in both pulls.
    joined = synapses.join(targets, left_on="id", right_on="target_id", how="inner", suffix="_neurd")
    joined = joined.select(
        pl.col("id_neurd").alias("id"),
        pl.col("id").alias("target_id"),
        "value",
        "pre_pt_root_id",
        "post_pt_root_id",
    ).collect()
    return (
        pl.concat(
            [_neurd_classes(joined), joined.select("pre_pt_root_id", "post_pt_root_id")],
            how="horizontal",
        )
        .with_columns(
            on_column=pl.col("post_pt_root_id").is_in(column),
            from_proofread=pl.col("pre_pt_root_id").is_in(proofread),
        )
        .sort("synapse_id")
    )


Tables.register(
    "neurd_labels",
    description=(
        "The NEURD (S7) label of each validation synapse that NEURD labeled, from the "
        "CAVE table `synapse_target_structure`. `neurd_label` is null for the axon "
        "labels, which have no class in the validation labels."
    ),
    materialize=lambda path: write_parquet_atomic(
        fetch_neurd_labels(
            CAVEclient(DATASTACK),
            Tables["validation_labels"].collect()["synapse_id"].to_list(),
        ),
        path,
    ),
    cache_params=lambda: {
        "version": VERSION,
        "validation_labels": Tables["validation_labels"].cache_key,
    },
)


# NOTE: the census tables read the deployment at a cutoff, but NEURD is an export
# at `VERSION`, so this table reads it at `VERSION`.
Tables.register(
    "census_neurd_labels",
    description=(
        "The NEURD label of each synapse onto a column cell or from a proofread "
        "cell that NEURD labeled, one row per synapse, from the export of the "
        "NEURD table `synapse_target_structure`. These are the synapses of `column_post_synapses` "
        "and `proofread_pre_synapses`, with autapses. Join on `synapse_id`. "
        "`on_column` and `from_proofread` say which side holds the synapse. "
        "`neurd_label` is null for the axon labels."
    ),
    materialize=lambda path: write_parquet_atomic(
        fetch_census_neurd_labels(), path
    ),
    cache_params=lambda: {
        "version": VERSION,
        "cell_selection": Tables["cell_selection"].cache_key,
    },
)


def build_validation_predictions(
    validation_labels: pl.DataFrame, neurd_labels: pl.DataFrame
) -> pl.DataFrame:
    """`validation_labels` with the HKS posteriors and `pred_label` of each synapse,
    and its NEURD label.

    The posteriors are joined on `root_id` (the cell), so a synapse that is not on
    its cell (`on_cell` false) has null posteriors.
    """
    predictions = synapse_posteriors(validation_labels, as_of=cutoff("validation"))
    return predictions.join(neurd_labels, on="synapse_id", how="left")


Tables.register(
    "validation_predictions",
    description=(
        "`validation_labels` with the deployment posteriors and `pred_label` of each "
        "synapse, joined to `neurd_labels`. Read by the validation and NEURD "
        "comparison figures."
    ),
    materialize=lambda path: write_parquet_atomic(
        build_validation_predictions(
            Tables["validation_labels"].collect(), Tables["neurd_labels"].collect()
        ),
        path,
    ),
    cache_params=lambda: {
        "version": VERSION,
        "validation_labels": Tables["validation_labels"].cache_key,
        "neurd_labels": Tables["neurd_labels"].cache_key,
        **pin_params(
            "synapse_mapping",
            "vertex_domains",
            "domain_predictions",
            as_of=cutoff("validation"),
        ),
    },
)
