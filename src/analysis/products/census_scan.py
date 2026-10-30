"""The two grain tables of the census, from one scan of the full-run tables.

`synapse_census_features` has one row per mapped synapse, and `spine_census_features`
one row per spine structure. The census tables derive from them, `microns_synapses`
and the fixed lookups.

The scan does not choose roots. One chunk is one bucket of `microns_synapses`
(`bucket[256]` of the post root), and it holds every root with data in the nested
deployment buckets at the cutoff. Each cloud table is read once per chunk, and
`drop_older_attempts` keeps the latest attempt per root in memory.
"""

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import polars as pl
import polars.selectors as cs
import pyarrow.parquet as pq
from deltalake import DeltaTable, Field, Schema, convert_to_deltalake
from tqdm.auto import tqdm

from ..io import LABEL_CATEGORIES, TABLE_PATH
from ..sources import VERSION, cutoff, deployment_path, pin_params
from .catalog import Tables
from .deployment import bucket_filter, drop_older_attempts, estimate_reads, scan_chunk
from .partitions import LAYOUT_PATH, hash_partition
from .structures import _add_is_multi, _count_inputs
from .synapses import BUCKET_COLUMN, N_BUCKETS, ROW_GROUP_ROWS, _write_layout

SYNAPSES = "synapse_census_features"
SPINES = "spine_census_features"
ATTEMPTS = "census_scan_attempts"

# The columns that the scan reads from each cloud table, besides `root_id` and
# `attempt_id`.
CLOUD_COLUMNS = {
    "synapse_mapping": ["synapse_id", "mesh_index"],
    "synapse_predictions": ["synapse_id", "pred_label", "posterior_max"],
    "vertex_structures": ["mesh_index", "structure_id"],
    "structure_morphometry": ["structure_id", "size_nm3", "area_nm2", "n_synapses"],
    "structure_predictions": ["structure_id", "posterior_unitary", "pred_label"],
}
UNITARY_LABELS = ["unitary", "not_unitary"]

# NOTE: bump WRITE_OPTIONS_VERSION when SORT or ENCODINGS change, so the cache key
# moves.
WRITE_OPTIONS_VERSION = 1
COMPRESSION_LEVEL = 9
SORT = {
    SYNAPSES: ["post_pt_root_id", "synapse_id"],
    SPINES: ["post_pt_root_id", "structure_id"],
}
# "dictionary", or a Parquet encoding without a dictionary. The smallest encoding
# of each column at zstd 9, on chunk 0 at the `census` cutoff (2026-10-30: 740,080
# synapse rows, 644,558 spine rows, 300 roots), in bytes per row:
# - synapses: `synapse_id` 1.76, `post_pt_root_id` 0.01, `tag` 0.15,
#   `tag_posterior` 2.70, `structure_id` 0.87, so 5.5 in all (about 1.0 GB).
#   Sorting on `structure_id` before `synapse_id` gave 5.6.
# - spines: `structure_id` 0.14, `size_nm3` 3.44, `area_nm2` 3.33, the counts 0.21
#   each, `posterior_unitary` 1.97 (a random forest gives few distinct values),
#   `unitary_pred_label` 0.09, `is_multi` 0.05, so 9.5 in all (about 1.6 GB).
ENCODINGS = {
    SYNAPSES: {
        "synapse_id": "BYTE_STREAM_SPLIT",
        "post_pt_root_id": "DELTA_BINARY_PACKED",
        "tag": "dictionary",
        "tag_posterior": "BYTE_STREAM_SPLIT",
        "structure_id": "BYTE_STREAM_SPLIT",
    },
    SPINES: {
        "post_pt_root_id": "DELTA_BINARY_PACKED",
        "structure_id": "DELTA_BINARY_PACKED",
        "size_nm3": "BYTE_STREAM_SPLIT",
        "area_nm2": "BYTE_STREAM_SPLIT",
        "n_synapses": "dictionary",
        "n_pre_root_ids": "dictionary",
        "posterior_unitary": "dictionary",
        "unitary_pred_label": "dictionary",
        "is_multi": "PLAIN",
    },
}
MAX_WORKERS = 4


def read_chunk(chunk: int, *, as_of: datetime) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """The inputs of `join_chunk` for `chunk`, a bucket of `microns_synapses`, with
    every table at `as_of`, and the kept attempt per (`root_id`, cloud table).

    Each cloud table is read once, filtered only on whole roots (the bucket and the
    cutoff), and then `drop_older_attempts` keeps its latest attempt per root. The
    one exception is `vertex_structures`: see `join_chunk`.
    """
    synapse_path = Tables["microns_synapses"].ensure_cached()
    synapses = (
        Tables["microns_synapses"]
        .lazy()
        .filter(bucket_filter(synapse_path, chunk, N_BUCKETS))
        .select(pl.col("id").alias("synapse_id"), "pre_pt_root_id", "post_pt_root_id")
    )

    def scan(product: str) -> pl.LazyFrame:
        return scan_chunk(deployment_path(product), chunk, N_BUCKETS, as_of=as_of).select(
            "root_id", "attempt_id", *CLOUD_COLUMNS[product]
        )

    whole_root_reads = [p for p in CLOUD_COLUMNS if p != "vertex_structures"]
    synapses, *frames = pl.collect_all(
        [synapses, *[scan(p) for p in whole_root_reads]], engine="streaming"
    )
    raw = dict(zip(whole_root_reads, frames))
    mapped_vertices = (
        drop_older_attempts(raw["synapse_mapping"]).select("root_id", "mesh_index").unique()
    )
    # NOTE: a chunk of `vertex_structures` is about 500M rows, one per vertex, so
    # the read keeps only the mapped vertices before `drop_older_attempts`. Every
    # attempt has a row per vertex, so the latest attempt keeps its rows.
    # `join_chunk` checks it.
    raw["vertex_structures"] = (
        scan("vertex_structures")
        .join(mapped_vertices.lazy(), on=["root_id", "mesh_index"], how="semi")
        .collect(engine="streaming")
    )
    attempts = pl.concat(
        [
            frame.group_by("root_id")
            .agg(pl.col("attempt_id").max())
            .with_columns(pl.lit(product).alias("product"))
            for product, frame in raw.items()
        ]
    ).sort("root_id", "product")
    inputs = {product: drop_older_attempts(frame) for product, frame in raw.items()}
    return {"synapses": synapses, **inputs}, attempts


def join_chunk(
    synapses: pl.DataFrame,
    synapse_mapping: pl.DataFrame,
    synapse_predictions: pl.DataFrame,
    vertex_structures: pl.DataFrame,
    structure_morphometry: pl.DataFrame,
    structure_predictions: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The rows of `synapse_census_features` and `spine_census_features` from one
    chunk of inputs. Each cloud table holds one attempt per root, and its rows carry
    `root_id`.

    - `synapses` (`synapse_id`, `pre_pt_root_id`, `post_pt_root_id`): autapses drop
      here, before any count.
    - A synapse with no `synapse_mapping` row has no row. `tag` and `tag_posterior`
      (`pred_label` and `posterior_max` of `synapse_predictions`) are null for a
      synapse with no prediction, and `structure_id` for a synapse on no structure.
    - Every structure of `structure_morphometry` has a spine row. `n_synapses` and
      `n_pre_root_ids` come from the synapses of the chunk (see `_count_inputs`),
      so they leave out the autapses, and they are 0 on a structure with no synapse.
      `is_multi`: see `_add_is_multi`.
    """
    # NOTE: the two tables of one root can come from different attempts. The joins
    # are on canonical keys (meshmash-deployment `docs/schemas.md`), so they agree.
    mapping = synapse_mapping.rename({"root_id": "post_pt_root_id"})
    vertices = vertex_structures.rename({"root_id": "post_pt_root_id"})
    # NOTE: a root with `vertex_structures` rows must have one at every mapped
    # vertex. A miss means its latest attempt is partial.
    covered = mapping.join(vertices.select("post_pt_root_id").unique(), on="post_pt_root_id")
    missing = covered.join(vertices, on=["post_pt_root_id", "mesh_index"], how="anti")
    assert missing.is_empty(), f"{len(missing)} mapped vertices have no vertex_structures row"

    synapse_rows = (
        synapses.filter(pl.col("pre_pt_root_id") != pl.col("post_pt_root_id"))
        .join(mapping, on=["post_pt_root_id", "synapse_id"], how="inner", validate="1:1")
        .join(
            synapse_predictions.select(
                pl.col("root_id").alias("post_pt_root_id"),
                "synapse_id",
                pl.col("pred_label").cast(pl.Enum(LABEL_CATEGORIES)).alias("tag"),
                pl.col("posterior_max").cast(pl.Float32).alias("tag_posterior"),
            ),
            on=["post_pt_root_id", "synapse_id"],
            how="left",
            validate="1:1",
        )
        .join(vertices, on=["post_pt_root_id", "mesh_index"], how="left", validate="m:1")
        .with_columns(pl.col("structure_id").cast(pl.Int32))
    )

    key = ["root_id", "structure_id"]
    counts = _count_inputs(
        synapse_rows.select(
            pl.col("post_pt_root_id").alias("root_id"), "structure_id", "pre_pt_root_id"
        )
    )
    morphometry = structure_morphometry.rename({"n_synapses": "n_synapses_deployment"})
    orphans = counts.join(morphometry, on=key, how="anti")
    assert orphans.is_empty(), f"{len(orphans)} structures with synapses have no morphometry"
    spines = (
        morphometry.join(
            structure_predictions.rename({"pred_label": "unitary_pred_label"}),
            on=key,
            how="left",
            validate="1:1",
        )
        .join(counts, on=key, how="left", validate="1:1")
        .with_columns(pl.col("n_synapses", "n_pre_root_ids").fill_null(0).cast(pl.Int32))
    )
    # NOTE: the deployment count includes the autapses, so it can only be larger.
    assert (spines["n_synapses_deployment"] >= spines["n_synapses"]).all(), (
        "a structure has more synapses than structure_morphometry counts"
    )
    spines = (
        spines.pipe(_add_is_multi)
        .select(
            pl.col("root_id").alias("post_pt_root_id"),
            pl.col("structure_id").cast(pl.Int32),
            pl.col("size_nm3", "area_nm2").cast(pl.Float32),
            "n_synapses",
            "n_pre_root_ids",
            pl.col("posterior_unitary").cast(pl.Float32),
            pl.col("unitary_pred_label").cast(pl.Enum(UNITARY_LABELS)),
            "is_multi",
        )
        .sort(SORT[SPINES])
    )
    synapse_rows = synapse_rows.select(
        "synapse_id", "post_pt_root_id", "tag", "tag_posterior", "structure_id"
    ).sort(SORT[SYNAPSES])
    return synapse_rows, spines


def census_chunk(chunk: int, *, as_of: datetime) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """`join_chunk` on `read_chunk`: the synapse rows, the spine rows and the kept
    attempts of `chunk`."""
    inputs, attempts = read_chunk(chunk, as_of=as_of)
    return *join_chunk(**inputs), attempts


def write_part(frame: pl.DataFrame, path: Path, encodings: dict[str, str]) -> None:
    """Write `frame` to `path` with zstd and one encoding per column, so that an
    interrupted write never leaves a file there.

    Delta has no Enum type, so an Enum column is written as a string. Its
    dictionary encoding costs the same bytes.
    """
    frame = frame.with_columns(cs.enum().cast(pl.String))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(
        frame.to_arrow(),
        tmp,
        compression="zstd",
        compression_level=COMPRESSION_LEVEL,
        row_group_size=ROW_GROUP_ROWS,
        use_dictionary=[c for c, e in encodings.items() if e == "dictionary"],
        column_encoding={c: e for c, e in encodings.items() if e != "dictionary"},
    )
    tmp.rename(path)


def _parts(work: Path, chunk: int) -> dict[str, Path]:
    bucket = f"{BUCKET_COLUMN}={chunk}"
    return {
        SYNAPSES: work / SYNAPSES / bucket / "part-0.parquet",
        SPINES: work / SPINES / bucket / "part-0.parquet",
        ATTEMPTS: work / ATTEMPTS / f"chunk-{chunk:03d}.parquet",
    }


def _write_chunk(chunk: int, work: Path, as_of: datetime) -> None:
    synapse_rows, spines, attempts = census_chunk(chunk, as_of=as_of)
    parts = _parts(work, chunk)
    write_part(synapse_rows, parts[SYNAPSES], ENCODINGS[SYNAPSES])
    write_part(spines, parts[SPINES], ENCODINGS[SPINES])
    # NOTE: the attempts part is written last, so a chunk with all three parts is
    # done.
    parts[ATTEMPTS].parent.mkdir(parents=True, exist_ok=True)
    tmp = parts[ATTEMPTS].with_name(parts[ATTEMPTS].name + ".tmp")
    attempts.write_parquet(tmp)
    tmp.rename(parts[ATTEMPTS])


def _to_delta(folder: Path) -> None:
    """Make the hive folder of parts at `folder` a Delta table with one commit and a
    layout file."""
    for tmp in folder.rglob("*.tmp"):
        tmp.unlink()
    if (folder / "_delta_log").exists():
        shutil.rmtree(folder / "_delta_log")
    convert_to_deltalake(
        str(folder),
        partition_by=Schema([Field(BUCKET_COLUMN, "integer")]),
        partition_strategy="hive",
    )
    _write_layout(folder, N_BUCKETS)


def _check(folder: Path, key: list[str]) -> None:
    """Assert that the Delta table at `folder` has one version and one file per
    bucket, and that each file is sorted on `post_pt_root_id`, holds only the roots
    of its bucket and has unique keys."""
    table = DeltaTable(str(folder))
    assert len(table.history()) == 1, "the Delta log holds more than one version"
    actions = pl.from_arrow(table.get_add_actions(flatten=True))
    assert actions[f"partition.{BUCKET_COLUMN}"].is_unique().all(), "a bucket has more than one file"
    assert len(actions) == N_BUCKETS, f"{len(actions)} of {N_BUCKETS} buckets have a file"
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    expected = {*actions["path"], "_delta_log/00000000000000000000.json", LAYOUT_PATH}
    assert files == expected, f"unexpected files: {sorted(files ^ expected)}"
    for file in actions["path"]:
        frame = pl.read_parquet(folder / file, columns=key, hive_partitioning=False)
        bucket = int(file.split("/")[0].split("=")[1])
        assert frame["post_pt_root_id"].is_sorted(), f"{file} is not sorted on post_pt_root_id"
        roots = frame["post_pt_root_id"].unique()
        assert all(hash_partition(root, N_BUCKETS) == bucket for root in roots), (
            f"{file} has a root of another bucket"
        )
        assert not frame.is_duplicated().any(), f"{file} has a duplicate key"


def _work_path() -> Path:
    return TABLE_PATH / f"v{VERSION}" / "census_scan" / f"{Tables[SYNAPSES].cache_key}.partial"


def build_census_scan(max_workers: int = MAX_WORKERS) -> None:
    """Write `synapse_census_features`, `spine_census_features` and
    `census_scan_attempts` at their cache paths, from one scan at the `census`
    cutoff.

    Each chunk writes its parts to a work folder, so an interrupted scan resumes
    from the chunks it finished. When every chunk is done, the parts become the two
    Delta tables, and the work folder is removed after both pass `_check`.

    The three share one cache key, and one scan writes all of them. If only some are
    missing (for example, one was deleted), the scan runs again and writes only the
    missing ones. The others stay, because the same pins and cutoff give the same
    rows.
    """
    as_of = cutoff("census")
    paths = {name: Tables[name].cache_path for name in (SYNAPSES, SPINES, ATTEMPTS)}
    if all(path.exists() for path in paths.values()):
        return
    work = _work_path()
    todo = [
        chunk
        for chunk in range(N_BUCKETS)
        if not all(part.exists() for part in _parts(work, chunk).values())
    ]
    # NOTE: threads are enough here, because Polars releases the GIL in collect.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_write_chunk, chunk, work, as_of) for chunk in todo]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

    keys = {SYNAPSES: ["post_pt_root_id", "synapse_id"], SPINES: ["post_pt_root_id", "structure_id"]}
    for name, key in keys.items():
        _to_delta(work / name)
        _check(work / name, key)
    for name in keys:
        if not paths[name].exists():
            paths[name].parent.mkdir(parents=True, exist_ok=True)
            (work / name).rename(paths[name])
    if not paths[ATTEMPTS].exists():
        paths[ATTEMPTS].parent.mkdir(parents=True, exist_ok=True)
        tmp = paths[ATTEMPTS].with_name(paths[ATTEMPTS].name + ".tmp")
        pl.scan_parquet(work / ATTEMPTS / "*.parquet").sort("root_id", "product").sink_parquet(tmp)
        tmp.rename(paths[ATTEMPTS])
    shutil.rmtree(work)


def _materialize(path: Path) -> None:
    build_census_scan()
    assert path.exists(), f"the census scan did not write {path}"


def _cache_params() -> dict:
    return {
        "version": VERSION,
        "microns_synapses": Tables["microns_synapses"].cache_key,
        **pin_params(*CLOUD_COLUMNS, as_of=cutoff("census")),
        "write_options": WRITE_OPTIONS_VERSION,
    }


Tables.register(
    SYNAPSES,
    description=(
        "One row per synapse of `microns_synapses` that the deployment mapped, at the "
        "`census` cutoff, with no autapses: `synapse_id`, `post_pt_root_id`, the "
        "domain label as `tag` and its posterior as `tag_posterior` (from "
        "`synapse_predictions`, null with no prediction), and the `structure_id` of "
        "its spine (null if it is on no spine). An unmapped synapse has no row. It "
        "holds every root with deployment data, not only the neuron roots. A local "
        "Delta table on `post_pt_root_id_bucket`, the buckets of `microns_synapses`. "
        "`tag` is a string on disk: cast it to `pl.Enum(LABEL_CATEGORIES)`. One scan "
        "writes it with `spine_census_features`."
    ),
    materialize=_materialize,
    estimate=lambda: estimate_reads(CLOUD_COLUMNS),
    cache_params=_cache_params,
    format="delta",
)
Tables.register(
    SPINES,
    description=(
        "One row per spine structure of `structure_morphometry` at the `census` "
        "cutoff, with or without a synapse, keyed by (`post_pt_root_id`, "
        "`structure_id`): `size_nm3`, `area_nm2`, `n_synapses` and `n_pre_root_ids` "
        "(from `microns_synapses`, no autapses, 0 with no synapse), "
        "`posterior_unitary`, `unitary_pred_label` (a string on disk) and "
        "`is_multi`. No census table reads the other morphometry or the spectral "
        "features, so the table leaves them out. A local Delta table on "
        "`post_pt_root_id_bucket`. The same scan writes it with "
        "`synapse_census_features`."
    ),
    materialize=_materialize,
    estimate=lambda: estimate_reads(CLOUD_COLUMNS),
    cache_params=_cache_params,
    format="delta",
)
Tables.register(
    ATTEMPTS,
    description=(
        "One row per (`root_id`, `product`): the `attempt_id` that the census scan "
        "kept for the root in each cloud table it read. Two tables of one root can "
        "come from different attempts. The same scan writes it with "
        "`synapse_census_features`."
    ),
    materialize=_materialize,
    estimate=lambda: estimate_reads(CLOUD_COLUMNS),
    cache_params=_cache_params,
)
