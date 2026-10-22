"""P4: the validation labels with the HKS posteriors and the NEURD labels."""

import polars as pl
from caveclient import CAVEclient

from ..sources import DATASTACK, VERSION, cutoff, pin_params
from . import labels  # noqa: F401 -- registers "validation_labels", whose cache key P4 reads
from .catalog import Tables, write_parquet_atomic
from .predictions import synapse_posteriors

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


def fetch_neurd_labels(client, synapse_ids: list[int], version=VERSION) -> pl.DataFrame:
    """The NEURD (S7) label of each synapse in `synapse_ids` that NEURD labeled."""
    targets = client.materialize.query_table(
        "synapse_target_structure",
        filter_in_dict=dict(target_id=synapse_ids),
        materialization_version=version,
        log_warning=False,
    )
    return (
        pl.from_pandas(targets[["id", "target_id", "value"]])
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


def build_validation_predictions(client, validation_labels: pl.DataFrame) -> pl.DataFrame:
    """`validation_labels` with the HKS posteriors and `pred_label` of each synapse,
    and its NEURD label.

    The posteriors are joined on `root_id` (the cell), so a synapse that is not on
    its cell (`on_cell` false) has null posteriors.
    """
    predictions = synapse_posteriors(validation_labels, as_of=cutoff("validation"))
    neurd = fetch_neurd_labels(client, validation_labels["synapse_id"].to_list())
    return predictions.join(neurd, on="synapse_id", how="left")


Tables.register(
    "validation_predictions",
    description=(
        "`validation_labels` with the deployment posteriors and `pred_label` of each "
        "synapse, and its NEURD label from the CAVE table `synapse_target_structure`. "
        "Read by the validation and NEURD comparison figures."
    ),
    materialize=lambda path: write_parquet_atomic(
        build_validation_predictions(
            CAVEclient(DATASTACK), Tables["validation_labels"].collect()
        ),
        path,
    ),
    cache_params=lambda: {
        "version": VERSION,
        "validation_labels": Tables["validation_labels"].cache_key,
        **pin_params(
            "synapse_mapping",
            "vertex_domains",
            "domain_predictions",
            as_of=cutoff("validation"),
        ),
    },
)
