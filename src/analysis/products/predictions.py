"""P15: domain posteriors for synapses, from the meshmash-deployment tables."""

from datetime import datetime

import polars as pl

from . import features  # noqa: F401 -- registers "synapse_to_domain", which the view reads
from .catalog import Tables
from .deployment import latest_attempt
from .features import synapse_domain_join
from .sources import DATASTACK, DOMAIN_PREDICTIONS_PATH, deployment_path

# NOTE: these views read the latest attempts, so they change as the deployment
# writes. They are for exploration only. Products pin a cutoff.
Tables.register(
    "domain_predictions",
    lambda: latest_attempt(pl.scan_delta(DOMAIN_PREDICTIONS_PATH), as_of=None),
)
Tables.register(
    "synapse_posteriors",
    lambda: Tables["synapse_to_domain"]
    .lazy()
    .join(
        Tables["domain_predictions"].lazy().drop("attempt_id", "root_id_bucket"),
        on=["root_id", "domain_id"],
        how="left",
    ),
)


def synapse_posteriors(
    synapses: pl.DataFrame, *, as_of: datetime, datastack: str = DATASTACK
) -> pl.DataFrame:
    """`synapses` left-joined to the posterior and `pred_label` of its domain
    (see `synapse_domain_join`).

    Unlike the deployment's `synapse_predictions`, this keeps a synapse with no
    prediction, with null values.
    """
    return synapse_domain_join(
        synapses,
        deployment_path("domain_predictions", datastack),
        as_of=as_of,
        datastack=datastack,
    )
