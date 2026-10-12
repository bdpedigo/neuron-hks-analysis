"""The nucleus points of each root, for the `distance_to_nucleus` feature."""

import polars as pl
from caveclient import CAVEclient

from ..io import VERSION
from .catalog import Tables, write_parquet_atomic
from .sources import DATASTACK

# NOTE: a root can have more than one nucleus (for example, a merge), so this
# table has one row per nucleus, not per root. The feature is the distance to
# the nearest nucleus of the root. The deployment must use the same rule when it
# computes the feature (meshmash-deployment TASK-99).
NUCLEUS_VIEW = "nucleus_detection_lookup_v1"


def fetch_root_nuclei(version=VERSION) -> pl.DataFrame:
    """One row per nucleus on a root, with the nucleus point in nm."""
    client = CAVEclient(DATASTACK, version=version)
    nuclei = pl.from_pandas(
        client.materialize.query_view(
            NUCLEUS_VIEW,
            split_positions=True,
            desired_resolution=[1, 1, 1],
        )
    )
    return nuclei.filter(pl.col("pt_root_id") != 0).select(
        pl.col("pt_root_id").alias("root_id"),
        pl.col("id").alias("nucleus_id"),
        pl.col("pt_position_x").cast(pl.Float64).alias("nucleus_x"),
        pl.col("pt_position_y").cast(pl.Float64).alias("nucleus_y"),
        pl.col("pt_position_z").cast(pl.Float64).alias("nucleus_z"),
    )


Tables.register(
    "root_nuclei",
    materialize=lambda path: write_parquet_atomic(
        fetch_root_nuclei(version=VERSION), path
    ),
    cache_params={"version": VERSION},
)
