"""Reads of the meshmash-deployment Delta tables, with the latest attempt per root."""

from datetime import datetime
from functools import lru_cache

import polars as pl
from deltalake import DeltaTable
from upath import UPath

from .partitions import bucket_count, hash_partition


def attempt_cutoff(as_of: datetime) -> int:
    # attempt_id is milliseconds since the Unix epoch.
    return int(as_of.timestamp() * 1000)


@lru_cache
def delta_table(table_root: str) -> DeltaTable:
    # NOTE: reading the Delta log takes seconds, so each table reads it once per
    # process. The snapshot does not see commits made after it is opened.
    return DeltaTable(table_root)


def scan_roots(
    table_root: UPath, root_ids: list[int], *, as_of: datetime
) -> pl.LazyFrame:
    n_buckets = bucket_count(str(table_root))
    buckets = sorted({hash_partition(root_id, n_buckets) for root_id in root_ids})
    # NOTE: Polars cannot derive `root_id_bucket` from `root_id`, so without the
    # bucket filter it opens every file in every bucket.
    # NOTE: the as_of cutoff is part of the scan, so both the query that picks
    # each root's winning attempt and the rows joined to it respect it.
    return pl.scan_delta(delta_table(str(table_root))).filter(
        pl.col("root_id_bucket").is_in(buckets),
        pl.col("root_id").is_in(root_ids),
        pl.col("attempt_id") <= attempt_cutoff(as_of),
    )


def latest_attempts(scans: list[pl.LazyFrame]) -> list[pl.LazyFrame]:
    """Each scan reduced to the rows of its latest attempt per root."""
    # NOTE: the winners come from their own query, not a window. The streaming
    # engine runs `over()` in memory, and a filter on any column other than
    # root_id, placed before the window, could change which attempt wins.
    winners = pl.collect_all(
        [lf.group_by("root_id").agg(pl.col("attempt_id").max()) for lf in scans],
        engine="streaming",
    )
    return [
        lf.join(w.lazy(), on=["root_id", "attempt_id"], how="semi").drop(
            "attempt_id", "root_id_bucket"
        )
        for lf, w in zip(scans, winners)
    ]
