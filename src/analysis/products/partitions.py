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
