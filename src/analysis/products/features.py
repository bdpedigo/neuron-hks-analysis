"""P2: domain features for synapses, from the meshmash-deployment tables."""

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from . import labels  # noqa: F401 -- registers "labels", whose cache key P2 reads
from . import nuclei  # noqa: F401 -- registers "root_nuclei", which the training view reads
from .catalog import Tables
from .deployment import latest_attempt, latest_attempts, scan_roots
from .partitions import hash_partition
from .sources import (
    DOMAIN_FEATURES_PARAMS,
    DOMAIN_FEATURES_PATH,
    DOMAIN_FEATURES_VERSION,
    LABELED_AS_OF,
    SYNAPSE_MAPPING_PARAMS,
    SYNAPSE_MAPPING_PATH,
    SYNAPSE_MAPPING_VERSION,
    VERTEX_DOMAINS_PARAMS,
    VERTEX_DOMAINS_PATH,
    VERTEX_DOMAINS_VERSION,
)

# NOTE: these views read the latest attempts, so they change as the deployment
# writes. They are for exploration only. Products pin a cutoff.
Tables.register(
    "synapse_mapping",
    lambda: latest_attempt(pl.scan_delta(SYNAPSE_MAPPING_PATH), as_of=None).select(
        "root_id", "synapse_id", "mesh_index"
    ),
)
Tables.register(
    "vertex_domains",
    lambda: latest_attempt(pl.scan_delta(VERTEX_DOMAINS_PATH), as_of=None).select(
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
    lambda: latest_attempt(pl.scan_delta(DOMAIN_FEATURES_PATH), as_of=None),
)


def synapse_features(synapses: pl.DataFrame, *, as_of: datetime) -> pl.DataFrame:
    """`synapses` (with `root_id` and `synapse_id`) left-joined to its domain features,
    as the deployment tables were at `as_of`.

    Features are null for a synapse whose root the deployment has not processed.
    """
    root_ids = synapses["root_id"].unique().to_list()
    mapping_lf, domains_lf, features_lf = latest_attempts(
        [
            scan_roots(SYNAPSE_MAPPING_PATH, root_ids, as_of=as_of),
            scan_roots(VERTEX_DOMAINS_PATH, root_ids, as_of=as_of),
            scan_roots(DOMAIN_FEATURES_PATH, root_ids, as_of=as_of),
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


def _write_part(synapses: pl.DataFrame, part: Path, as_of: datetime) -> None:
    tmp = part.with_suffix(".tmp")
    synapse_features(synapses, as_of=as_of).write_parquet(tmp, compression="snappy")
    tmp.rename(part)


def write_synapse_features(
    synapses: pl.DataFrame,
    out: Path,
    *,
    as_of: datetime,
    n_chunks: int = 64,
    max_workers: int = 4,
) -> None:
    """Write `synapse_features(synapses, as_of=as_of)` to `out` as a single parquet file.

    Up to `max_workers` chunks of roots run at once, and peak memory is about that
    many chunks. Each chunk is written as a part, and the parts are combined into
    `out` at the end. An interrupted run resumes from the parts it finished.
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
        futures = [pool.submit(_write_part, *item, as_of) for item in todo]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
    # NOTE: the parts are combined in a streaming pass, so the whole table is
    # never in memory at once.
    tmp = out.with_name(out.name + ".tmp")
    pl.scan_parquet(partial / "*.parquet").sink_parquet(tmp, compression="snappy")
    tmp.rename(out)
    shutil.rmtree(partial)


Tables.register(
    "labeled_synapse_features",
    materialize=lambda path: write_synapse_features(
        Tables["synapse_labels"].collect(), path, as_of=LABELED_AS_OF
    ),
    cache_params={
        "labels": Tables["labels"].cache_key,
        "as_of": LABELED_AS_OF.isoformat(),
        "synapse_mapping_version": SYNAPSE_MAPPING_VERSION,
        "synapse_mapping_params": SYNAPSE_MAPPING_PARAMS,
        "vertex_domains_version": VERTEX_DOMAINS_VERSION,
        "vertex_domains_params": VERTEX_DOMAINS_PARAMS,
        "domain_features_version": DOMAIN_FEATURES_VERSION,
        "domain_features_params": DOMAIN_FEATURES_PARAMS,
    },
)


def add_distance_to_nucleus(lf: pl.LazyFrame, nuclei: pl.LazyFrame) -> pl.LazyFrame:
    """Add `distance_to_nucleus`: the Euclidean distance (nm) from each row's domain
    centroid (`domain_x/y/z`) to the nearest nucleus point on its root.

    `nuclei` has one row per nucleus, so a root can have several. The value is null
    for a root with no nucleus, and for a row with no domain centroid.
    """
    distances = (
        lf.select("root_id", "domain_x", "domain_y", "domain_z")
        .with_row_index("_row")
        .join(
            nuclei.select("root_id", "nucleus_x", "nucleus_y", "nucleus_z"),
            on="root_id",
            how="inner",
        )
        .group_by("_row")
        .agg(
            distance_to_nucleus=(
                (pl.col("domain_x") - pl.col("nucleus_x")) ** 2
                + (pl.col("domain_y") - pl.col("nucleus_y")) ** 2
                + (pl.col("domain_z") - pl.col("nucleus_z")) ** 2
            )
            .sqrt()
            .min()
        )
    )
    # NOTE: the left join keeps the input row order, because the random forest's
    # bootstrap samples depend on it.
    return (
        lf.with_row_index("_row")
        .join(distances, on="_row", how="left", maintain_order="left")
        .drop("_row")
    )


# NOTE: meshmash-deployment computes `distance_to_nucleus` in its prediction step,
# not in `domain_features`, so the training rows get the feature here
Tables.register(
    "training_features",
    lambda: add_distance_to_nucleus(
        Tables["labeled_synapse_features"].lazy(), Tables["root_nuclei"].lazy()
    ),
)
