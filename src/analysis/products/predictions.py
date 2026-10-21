"""P15: domain posteriors for synapses, from the meshmash-deployment tables."""

from datetime import datetime

import polars as pl

from .features import synapse_domain_join
from .sources import DATASTACK


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
        "domain_predictions",
        as_of=as_of,
        datastack=datastack,
    )
