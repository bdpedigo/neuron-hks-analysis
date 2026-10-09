from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import polars as pl
from deltalake import DeltaTable
from tqdm.auto import tqdm
from upath import UPath

from .catalog import Tables
from .io import VERSION
from .partitions import bucket_count, hash_partition
from .wrangle import get_label_table

DATASTACK = "minnie65_phase3_v1"
BASE_CLOUD_PATH = "gs://bdp-ssa/meshmash-deployment"
base = UPath(BASE_CLOUD_PATH)

synapse_mapping_version = "s2"
synapse_mapping_params = "p-72b901b7386541b7"

vertex_domains_version = "s2"
# vertex_domains_params = "p-2e737b09a92af64d"
vertex_domains_params = "p-0f6493d8b4cf1870"

domain_features_version = "s2"
# domain_features_params = "p-4b0b62980242847a"
domain_features_params = "p-90e8423092581ed2"

domain_edges_version = "s2"
domain_edges_params = "p-b60a113f02ed8102" # old

synapse_mapping_path = (
    base
    / DATASTACK
    / "synapse_mapping"
    / synapse_mapping_version
    / synapse_mapping_params
)
vertex_domains_path = (
    base / DATASTACK / "vertex_domains" / vertex_domains_version / vertex_domains_params
)
domain_features_path = (
    base
    / DATASTACK
    / "domain_features"
    / domain_features_version
    / domain_features_params
)
domain_edges_path = (
    base / DATASTACK / "domain_edges" / domain_edges_version / domain_edges_params
)


def _latest_attempt(lf: pl.LazyFrame) -> pl.LazyFrame:
    # An attempt replaces all of a root's rows at once (dedupe_key="root_id"),
    # so the newest attempt_id for a root wins, whole-root.
    return lf.filter(pl.col("attempt_id") == pl.col("attempt_id").max().over("root_id"))


Tables.register(
    "synapse_mapping",
    lambda: _latest_attempt(pl.scan_delta(synapse_mapping_path)).select(
        "root_id", "synapse_id", "mesh_index"
    ),
)
Tables.register(
    "vertex_domains",
    lambda: _latest_attempt(pl.scan_delta(vertex_domains_path)).select(
        "root_id", "mesh_index", "domain_id"
    ),
)
Tables.register(
    "synapse_to_domain",
    lambda: Tables["synapse_mapping"]
    .lazy()
    .join(Tables["vertex_domains"].lazy(), on=["root_id", "mesh_index"], how="left")
    .drop("mesh_index"),
)
Tables.register(
    "domain_features",
    lambda: _latest_attempt(pl.scan_delta(domain_features_path)),
)
Tables.register("labels", lambda: get_label_table(version=VERSION).lazy())
Tables.register(
    "synapse_labels",
    lambda: Tables["labels"]
    .lazy()
    .filter(pl.col("target_id") != -1)
    .rename({"target_id": "synapse_id", f"pt_root_id_{VERSION}": "root_id"}),
)


@lru_cache
def _delta_table(table_root: str) -> DeltaTable:
    # NOTE: reading the Delta log takes seconds, so each table reads it once per
    # process. The snapshot does not see commits made after it is opened.
    return DeltaTable(table_root)


def _scan_roots(table_root: UPath, root_ids: list[int]) -> pl.LazyFrame:
    n_buckets = bucket_count(str(table_root))
    buckets = sorted({hash_partition(root_id, n_buckets) for root_id in root_ids})
    # NOTE: Polars cannot derive `root_id_bucket` from `root_id`, so without the
    # bucket filter it opens every file in every bucket.
    return pl.scan_delta(_delta_table(str(table_root))).filter(
        pl.col("root_id_bucket").is_in(buckets), pl.col("root_id").is_in(root_ids)
    )


def _latest_attempts(scans: list[pl.LazyFrame]) -> list[pl.LazyFrame]:
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


def synapse_features(synapses: pl.DataFrame) -> pl.DataFrame:
    """`synapses` (with `root_id` and `synapse_id`) left-joined to its domain features.

    Features are null for a synapse whose root the deployment has not processed.
    """
    root_ids = synapses["root_id"].unique().to_list()
    mapping_lf, domains_lf, features_lf = _latest_attempts(
        [
            _scan_roots(synapse_mapping_path, root_ids),
            _scan_roots(vertex_domains_path, root_ids),
            _scan_roots(domain_features_path, root_ids),
        ]
    )
    mapping = (
        mapping_lf.filter(pl.col("synapse_id").is_in(synapses["synapse_id"].unique().to_list()))
        .select("root_id", "synapse_id", "mesh_index")
        .collect(engine="streaming")
    )
    # NOTE: vertex_domains has a row per mesh vertex. The semi-join keeps only
    # synapse vertices as the scan streams, so they are never all in memory.
    domains = (
        domains_lf.join(
            mapping.lazy().select("root_id", "mesh_index").unique(),
            on=["root_id", "mesh_index"],
            how="semi",
        )
        .select("root_id", "mesh_index", "domain_id")
        .collect(engine="streaming")
    )
    synapse_domains = mapping.join(
        domains, on=["root_id", "mesh_index"], how="left"
    ).drop("mesh_index")
    features = (
        features_lf.join(
            synapse_domains.lazy().select("root_id", "domain_id").drop_nulls().unique(),
            on=["root_id", "domain_id"],
            how="semi",
        )
        .collect(engine="streaming")
    )
    return (
        synapses.join(synapse_domains, on=["root_id", "synapse_id"], how="left")
        .join(features, on=["root_id", "domain_id"], how="left")
        .drop("domain_id")
    )


def _write_part(synapses: pl.DataFrame, part: Path) -> None:
    tmp = part.with_suffix(".tmp")
    synapse_features(synapses).write_parquet(tmp, compression="snappy")
    tmp.rename(part)


def write_synapse_features(
    synapses: pl.DataFrame, out: Path, n_chunks: int = 64, max_workers: int = 4
) -> None:
    """Write `synapse_features(synapses)` to `out` as a directory of parquet parts.

    Up to `max_workers` chunks of roots run at once, and peak memory is about that
    many chunks. An interrupted run resumes from the parts it finished.
    """
    # NOTE: chunks group roots by bucket[n_chunks]. Bucket counts nest, so when
    # n_chunks divides a table's bucket count, each chunk reads only
    # 1 / n_chunks of that table's bucket directories.
    chunk_of_root = {
        root_id: hash_partition(root_id, n_chunks)
        for root_id in synapses["root_id"].unique().to_list()
    }
    chunks = synapses.with_columns(
        _chunk=pl.col("root_id").replace_strict(chunk_of_root, return_dtype=pl.Int32)
    ).partition_by("_chunk", as_dict=True, include_key=False)
    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    todo = [
        (chunk_synapses, partial / f"part-{chunk:04d}.parquet")
        for (chunk,), chunk_synapses in sorted(chunks.items())
    ]
    todo = [(chunk_synapses, part) for chunk_synapses, part in todo if not part.exists()]
    # NOTE: threads are enough here, because Polars releases the GIL in collect.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_write_part, *item) for item in todo]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
    partial.rename(out)


Tables.register(
    "labeled_synapse_features",
    materialize=lambda path: write_synapse_features(
        Tables["synapse_labels"].collect(), path
    ),
    cache_params={
        "synapse_mapping_version": synapse_mapping_version,
        "synapse_mapping_params": synapse_mapping_params,
        "vertex_domains_version": vertex_domains_version,
        "vertex_domains_params": vertex_domains_params,
        "domain_features_version": domain_features_version,
        "domain_features_params": domain_features_params,
    },
)
