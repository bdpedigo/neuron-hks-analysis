"""P4: the validation labels with the HKS posteriors and the NEURD labels."""

import logging

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


NEURD_ROOTS_PER_QUERY = 25


class _CutWarnings(logging.Handler):
    """Records the server warnings that say a query result was cut."""

    def __init__(self):
        super().__init__()
        self.cut = False

    def emit(self, record):
        if "Limited query" in record.getMessage():
            self.cut = True


def _query_neurd_batch(client, side: str, root_ids: list[int], version: int) -> pl.DataFrame:
    """The NEURD rows of the synapses whose `{side}_pt_root_id` is in `root_ids`,
    with both root IDs.

    A CAVE query returns at most a server-set number of rows. The server says so
    in a warning, and the result is cut with no error. So a cut batch is split in
    two and each half is queried again.
    """
    handler = _CutWarnings()
    logger = logging.getLogger("caveclient.base")
    logger.addHandler(handler)
    try:
        targets = client.materialize.tables.synapse_target_structure(
            **{f"{side}_pt_root_id": root_ids}
        ).query(
            select_columns={
                "synapse_target_structure": ["id", "target_id", "value"],
                "synapses_pni_2": ["pre_pt_root_id", "post_pt_root_id"],
            },
            materialization_version=version,
            log_warning=True,
        )
    finally:
        logger.removeHandler(handler)
    if handler.cut:
        if len(root_ids) == 1:
            raise RuntimeError(f"the NEURD query of root {root_ids[0]} was cut")
        half = len(root_ids) // 2
        return pl.concat(
            [
                _query_neurd_batch(client, side, root_ids[:half], version),
                _query_neurd_batch(client, side, root_ids[half:], version),
            ]
        )
    targets = pl.from_pandas(targets)
    return pl.concat(
        [_neurd_classes(targets), targets.select("pre_pt_root_id", "post_pt_root_id")],
        how="horizontal",
    )


def fetch_root_neurd_labels(
    client, root_ids: list[int], side: str, version=VERSION
) -> pl.DataFrame:
    """The NEURD label of each synapse that NEURD labeled whose `{side}_pt_root_id`
    is in `root_ids`, with `pre_pt_root_id` and `post_pt_root_id`.

    `synapse_target_structure` is a reference table on `synapses_pni_2`, so the
    filter on a root goes through the table interface, which joins the two. A root
    that NEURD did not label has no rows.
    """
    return pl.concat(
        [
            _query_neurd_batch(client, side, root_ids[start : start + NEURD_ROOTS_PER_QUERY], version)
            for start in range(0, len(root_ids), NEURD_ROOTS_PER_QUERY)
        ]
    )


def fetch_census_neurd_labels(client, version=VERSION) -> pl.DataFrame:
    """The NEURD label of each synapse onto a column cell or from a proofread cell,
    one row per synapse, with the flags `on_column` and `from_proofread`."""
    selection = Tables["cell_selection"].collect()
    column = selection.filter(pl.col("is_column"))["pt_root_id"].to_list()
    proofread = selection.filter(pl.col("is_proofread"))["pt_root_id"].to_list()
    neurd = pl.concat(
        [
            fetch_root_neurd_labels(client, column, "post", version),
            fetch_root_neurd_labels(client, proofread, "pre", version),
        ]
    )
    # A synapse from a proofread cell onto a column cell is in both pulls.
    return (
        neurd.unique("synapse_id")
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


# NOTE: the census tables read the deployment at a cutoff, but NEURD is a CAVE
# table, so this table reads it at `VERSION`. It pulls from CAVE until the NEURD
# delta export exists (ssa-backlog TASK-39).
Tables.register(
    "census_neurd_labels",
    description=(
        "The NEURD label of each synapse onto a column cell or from a proofread "
        "cell that NEURD labeled, one row per synapse, from the CAVE table "
        "`synapse_target_structure`. These are the synapses of `column_post_synapses` "
        "and `proofread_pre_synapses`, with autapses. Join on `synapse_id`. "
        "`on_column` and `from_proofread` say which side holds the synapse. "
        "`neurd_label` is null for the axon labels."
    ),
    materialize=lambda path: write_parquet_atomic(
        fetch_census_neurd_labels(CAVEclient(DATASTACK)), path
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
