"""The ``bucket[N]`` partition hash that meshmash-deployment tables use.

This is Apache Iceberg's bucket transform: 32-bit MurmurHash3 (x86, seed 0) over
the id's 8 little-endian bytes, sign bit cleared, reduced mod N. It is a copy of
``meshmash_deployment.partitions.hash_partition``. For power-of-two N, bucket
counts nest: ``hash_partition(k, Nb) % Na == hash_partition(k, Na)`` when Na
divides Nb.
"""

import json
import re
from functools import lru_cache

import mmh3
from upath import UPath

PARTITIONS_PATH = "_meshmash/partitions.json"
# The sidecar of a local bucketed table (cistern_layout_v1).
LAYOUT_PATH = "_info/layout.json"


def hash_partition(key: int, n: int) -> int:
    """Iceberg's ``bucket[n]`` transform applied to ``key``."""
    key_bytes = key.to_bytes(8, "little", signed=True)
    return (mmh3.hash(key_bytes, seed=0, signed=False) & 0x7FFFFFFF) % n


@lru_cache
def bucket_count(table_root: str, column: str = "root_id_bucket") -> int:
    """N for a table's ``bucket[N]`` partition column, from its partitions sidecar."""
    specs = json.loads((UPath(table_root) / PARTITIONS_PATH).read_text())
    transform = specs[column]["transform"]
    match = re.fullmatch(r"bucket\[(\d+)\]", transform)
    if match is None:
        raise ValueError(f"{column!r} in {table_root} is {transform!r}, not bucket[N]")
    return int(match.group(1))


@lru_cache
def bucket_layout(table_root: str) -> tuple[str, int]:
    """The bucket column of a table and its N, from the table's sidecar.

    A meshmash-deployment table has `_meshmash/partitions.json`, and a local table
    has `_info/layout.json`. Each must name exactly one bucket column.
    """
    root = UPath(table_root)
    if (root / PARTITIONS_PATH).exists():
        specs = json.loads((root / PARTITIONS_PATH).read_text())
        columns = [
            column for column, spec in specs.items()
            if re.fullmatch(r"bucket\[\d+\]", spec["transform"])
        ]
        if len(columns) != 1:
            raise ValueError(f"{table_root} has the bucket columns {columns}, not one")
        return columns[0], bucket_count(table_root, columns[0])
    layout = json.loads((root / LAYOUT_PATH).read_text())
    transforms = [
        t for t in layout["transforms"] if t["transform"]["@type"] == "iceberg_bucket"
    ]
    if len(transforms) != 1:
        raise ValueError(f"{table_root} has {len(transforms)} bucket transforms, not one")
    return transforms[0]["name"], transforms[0]["transform"]["n_buckets"]


def nested_buckets(chunk: int, n_chunks: int, n_buckets: int) -> list[int]:
    """The buckets of a `bucket[n_buckets]` table that hold the roots of `chunk`, a
    bucket of `bucket[n_chunks]`. They are `chunk + n_chunks * k`, because bucket
    counts nest when `n_chunks` divides `n_buckets`."""
    if n_buckets % n_chunks:
        raise ValueError(f"bucket[{n_chunks}] does not nest in bucket[{n_buckets}]")
    return list(range(chunk, n_buckets, n_chunks))
