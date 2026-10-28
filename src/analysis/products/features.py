"""P2: domain features for synapses, from the meshmash-deployment tables."""

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from ..sources import DATASTACK, cutoff, deployment_path, pin_params
from . import labels  # noqa: F401 -- registers "labels", whose cache key P2 reads
from . import nuclei  # noqa: F401 -- registers "root_nuclei", which the training view reads
from .catalog import Tables
from .deployment import latest_attempts, scan_roots
from .partitions import hash_partition


def synapse_vertex_join(
    synapses: pl.DataFrame,
    vertex_product: str,
    column: str,
    *,
    as_of: datetime,
    datastack: str = DATASTACK,
) -> pl.DataFrame:
    """`root_id`, `synapse_id` and `column` of the vertex each synapse maps to, from
    the per-vertex deployment table `vertex_product` (for example `vertex_domains`),
    as the deployment tables were at `as_of`.

    A synapse that `synapse_mapping` does not hold has no row. A mapped synapse on
    a root that `vertex_product` does not hold has a null `column`.
    """
    root_ids = synapses["root_id"].unique().to_list()
    mapping_lf, vertices_lf = latest_attempts(
        [
            scan_roots(deployment_path("synapse_mapping", datastack), root_ids, as_of=as_of),
            scan_roots(deployment_path(vertex_product, datastack), root_ids, as_of=as_of),
        ]
    )
    mapping = (
        mapping_lf.filter(pl.col("synapse_id").is_in(synapses["synapse_id"].unique().to_list()))
        .select("root_id", "synapse_id", "mesh_index")
        .collect(engine="streaming")
    )
    # NOTE: a per-vertex table has a row per mesh vertex. The semi-join keeps only
    # synapse vertices as the scan streams, so they are never all in memory.
    vertices = (
        vertices_lf.join(
            mapping.lazy().select("root_id", "mesh_index").unique(),
            on=["root_id", "mesh_index"],
            how="semi",
        )
        .select("root_id", "mesh_index", column)
        .collect(engine="streaming")
    )
    return mapping.join(vertices, on=["root_id", "mesh_index"], how="left").drop(
        "mesh_index"
    )


def synapse_domain_join(
    synapses: pl.DataFrame,
    domain_product: str,
    *,
    as_of: datetime,
    datastack: str = DATASTACK,
) -> pl.DataFrame:
    """`synapses` (with `root_id` and `synapse_id`) left-joined to the row of
    `domain_product` (a per-domain deployment table) for the domain of each synapse,
    as the deployment tables were at `as_of`.

    The domain columns are null for a synapse whose root the deployment has not
    processed.
    """
    root_ids = synapses["root_id"].unique().to_list()
    synapse_domains = synapse_vertex_join(
        synapses, "vertex_domains", "domain_id", as_of=as_of, datastack=datastack
    )
    (domain_rows_lf,) = latest_attempts(
        [scan_roots(deployment_path(domain_product, datastack), root_ids, as_of=as_of)]
    )
    domain_rows = (
        domain_rows_lf.join(
            synapse_domains.lazy().select("root_id", "domain_id").drop_nulls().unique(),
            on=["root_id", "domain_id"],
            how="semi",
        )
        .collect(engine="streaming")
    )
    return (
        synapses.join(synapse_domains, on=["root_id", "synapse_id"], how="left")
        .join(domain_rows, on=["root_id", "domain_id"], how="left")
        .drop("domain_id")
    )


def synapse_features(synapses: pl.DataFrame, *, as_of: datetime) -> pl.DataFrame:
    """`synapses` left-joined to its domain features (see `synapse_domain_join`)."""
    return synapse_domain_join(synapses, "domain_features", as_of=as_of)


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
    description=(
        "One row per labeled synapse (from `synapse_labels`), with the spectral "
        "features of the mesh domain that it maps to and the domain and mesh "
        "component sizes. Read from the deployment tables at the `training` cutoff. "
        "These are the training rows of the synapse models."
    ),
    materialize=lambda path: write_synapse_features(
        Tables["synapse_labels"].collect(), path, as_of=cutoff("training")
    ),
    cache_params=lambda: {
        "labels": Tables["labels"].cache_key,
        **pin_params(
            "synapse_mapping", "vertex_domains", "domain_features", as_of=cutoff("training")
        ),
    },
)


def add_distance_to_nucleus(
    lf: pl.LazyFrame,
    nuclei: pl.LazyFrame,
    position: tuple[str, str, str] = ("domain_x", "domain_y", "domain_z"),
    name: str = "distance_to_nucleus",
) -> pl.LazyFrame:
    """Add `name`: the Euclidean distance (nm) from each row's `position` columns
    (by default the domain centroid) to the nearest nucleus point on its root.

    `nuclei` has one row per nucleus, so a root can have several. The value is null
    for a root with no nucleus, and for a row with no position.
    """
    x, y, z = position
    distances = (
        lf.select("root_id", x, y, z)
        .with_row_index("_row")
        .join(
            nuclei.select("root_id", "nucleus_x", "nucleus_y", "nucleus_z"),
            on="root_id",
            how="inner",
        )
        .group_by("_row")
        .agg(
            (
                (pl.col(x) - pl.col("nucleus_x")) ** 2
                + (pl.col(y) - pl.col("nucleus_y")) ** 2
                + (pl.col(z) - pl.col("nucleus_z")) ** 2
            )
            .sqrt()
            .min()
            .alias(name)
        )
    )
    # NOTE: the left join keeps the input row order, because the random forest's
    # bootstrap samples depend on it.
    return (
        lf.with_row_index("_row")
        .join(distances, on="_row", how="left", maintain_order="left")
        .drop("_row")
    )


# The models train only on roots with at least this many synapse labels, the same
# rule as the labeled roots of the deployment test cohort
# (meshmash-deployment jobs/test_cohort/select.py). The count is per root, over
# all of its labels, with or without features.
MIN_ROOT_LABELS = 5


def model_inputs() -> pl.LazyFrame:
    synapse_rows = (
        Tables["labeled_synapse_features"]
        .lazy()
        .filter(pl.len().over("root_id") >= MIN_ROOT_LABELS)
        .with_columns(label_key=pl.format("synapse_{}", "synapse_id"))
    )
    # NOTE: `MIN_ROOT_LABELS` is a rule for synapse labels, so it skips the points.
    point_rows = (
        Tables["labeled_point_features"]
        .lazy()
        .drop("mesh_index", "distance_nm", "domain_id")
        .rename({"x": "pt_position_x", "y": "pt_position_y", "z": "pt_position_z"})
        .with_columns(label_key=pl.format("{}_{}", "table_name", "id"))
    )
    return add_distance_to_nucleus(
        pl.concat([synapse_rows, point_rows], how="diagonal_relaxed"),
        Tables["root_nuclei"].lazy(),
    )


# NOTE: meshmash-deployment computes `distance_to_nucleus` in its prediction step,
# not in `domain_features`, so the training rows get the feature here
Tables.register(
    "model_inputs",
    model_inputs,
    description=(
        "The training rows: the labeled synapses on roots with at least "
        "`MIN_ROOT_LABELS` labels, and every point label. `label_key` is the row "
        "key (`synapse_id` is null on a point row, `point_id` on a synapse row)."
    ),
)
