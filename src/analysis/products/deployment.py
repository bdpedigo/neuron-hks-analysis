"""Reads of the meshmash-deployment Delta tables, with the latest attempt per root.

Two helpers keep the latest attempt per root, with the same rule:

- `query_latest_attempts` finds the winners ahead of time, with one query over the
  whole scan, and then reads their rows. A later filter cannot change the winner.
- `drop_older_attempts` finds them in memory, in a frame that holds every attempt.
  It reads each table once, but the max sees only the rows that the read kept. So
  filter such a read on whole roots only (the bucket, the root, the cutoff).
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import polars as pl
from deltalake import DeltaTable
from upath import UPath

from ..sources import deployment_path
from .partitions import bucket_count, bucket_layout, hash_partition, nested_buckets


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


def query_latest_attempts(scans: list[pl.LazyFrame]) -> list[pl.LazyFrame]:
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


def bucket_filter(table_root: str | UPath | Path, chunk: int, n_chunks: int) -> pl.Expr:
    """The partition filter for the roots of `chunk`, a bucket of `bucket[n_chunks]`,
    in any bucketed table. The bucket column and its N come from the table's own
    sidecar (see `bucket_layout`)."""
    column, n_buckets = bucket_layout(str(table_root))
    return pl.col(column).is_in(nested_buckets(chunk, n_chunks, n_buckets))


def scan_chunk(
    table_root: UPath, chunk: int, n_chunks: int, *, as_of: datetime
) -> pl.LazyFrame:
    """Every attempt at or before `as_of` of the roots of `chunk` (a bucket of
    `bucket[n_chunks]`) in the deployment table at `table_root`. Pass it to
    `drop_older_attempts`."""
    return pl.scan_delta(delta_table(str(table_root))).filter(
        bucket_filter(table_root, chunk, n_chunks),
        pl.col("attempt_id") <= attempt_cutoff(as_of),
    )


def drop_older_attempts(frame: pl.DataFrame) -> pl.DataFrame:
    """The rows of the latest attempt per root, without `attempt_id`.

    `root_id` is the attempt grain of every deployment table (meshmash-deployment
    `docs/schemas.md`, "The dedupe key is the attempt grain"). Call it on one table,
    before any join: a max over a joined frame can mix the attempts of two tables.
    """
    return frame.filter(
        pl.col("attempt_id") == pl.col("attempt_id").max().over("root_id")
    ).drop("attempt_id")


def estimate_reads(products) -> str:
    """The rows and bytes of the pinned deployment tables `products`, every attempt
    and every column, from their Delta logs. Use it as the `estimate` of a table that
    a build reads from the cloud."""
    lines = ["The build reads these deployment tables (every attempt, every column):"]
    for product in products:
        actions = pl.from_arrow(delta_table(str(deployment_path(product))).get_add_actions(flatten=True))
        lines.append(
            f"  {product:>22}: {actions['num_records'].sum() / 1e9:.2f}B rows, "
            f"{actions['size_bytes'].sum() / 1e9:.1f} GB"
        )
    return "\n".join(lines)
