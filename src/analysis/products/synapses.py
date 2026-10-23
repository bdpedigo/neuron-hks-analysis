"""A local copy of the MICrONS synapse table (v1822), on hash buckets of the post root.

The public export is range-partitioned on `post_pt_root_id`, so a chunk of roots
from one deployment bucket reads most of its 310 files. The copy is partitioned
on `bucket[256]` of `post_pt_root_id` (Iceberg's bucket transform, as in
`partitions.py`), which nests with the deployment's `bucket[2048]`: deployment
bucket `b` is in copy bucket `b % 256`. It keeps only the columns the analysis
reads, and its Delta log holds one version.

The layout is in `_info/layout.json`, in the draft cistern format
(`cistern/docs/specs/layout.md`).
"""

import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from deltalake import ColumnProperties, DeltaTable, WriterProperties, write_deltalake
from deltalake.transaction import AddAction, create_table_with_add_actions
from upath import UPath

from ..io import TABLE_PATH
from ..sources import MICRONS_SYNAPSES_PATH, VERSION
from .catalog import Tables
from .partitions import hash_partition

COLUMNS = [
    "id",
    "pre_pt_root_id",
    "post_pt_root_id",
    "ctr_pt_position_x",
    "ctr_pt_position_y",
    "ctr_pt_position_z",
    "size",
]
N_BUCKETS = 256
BUCKET_COLUMN = "post_pt_root_id_bucket"
LAYOUT_PATH = "_info/layout.json"
# Source files read and written as one batch, about 1M rows each.
BATCH_FILES = 16

# The first writes are rewritten by the sort, so they take the cheap codec.
SCATTER_WRITER = WriterProperties(compression="ZSTD", compression_level=1)
# NOTE: bump WRITE_OPTIONS_VERSION when FINAL_COLUMNS or FINAL_LEVEL change, so the
# cache key moves.
WRITE_OPTIONS_VERSION = 1
FINAL_LEVEL = 9
_PLAIN = ColumnProperties(dictionary_enabled=False, encoding="PLAIN")
_DELTA = ColumnProperties(dictionary_enabled=False, encoding="DELTA_BINARY_PACKED")
_SPLIT = ColumnProperties(dictionary_enabled=False, encoding="BYTE_STREAM_SPLIT")
# The smallest encoding of each column in the estimate, on one sorted source file
# at zstd 9 (2026-10-23). The built table takes, in bytes per row: `id` 3.54,
# `pre_pt_root_id` 4.04, `post_pt_root_id` 0.38, `ctr_pt_position_{x,y,z}` 2.18,
# 2.16 and 1.75, and `size` 1.72, so 15.7 in all. The source export takes 17.
# NOTE: the estimate gives `post_pt_root_id` 0.03, because a source file holds a
# narrow range of roots (about 12k per 1M rows). A bucket holds roots from the
# whole range (about 210k per 1.2M rows), so the steps between roots are larger.
# On a built file, DELTA_BINARY_PACKED is still the smallest: 0.40 against 0.45
# for PLAIN and 0.81 for dictionary.
FINAL_COLUMNS = {
    "id": _SPLIT,
    "pre_pt_root_id": _SPLIT,
    "post_pt_root_id": _DELTA,
    "ctr_pt_position_x": _SPLIT,
    "ctr_pt_position_y": _SPLIT,
    "ctr_pt_position_z": _DELTA,
    "size": _SPLIT,
}
# A read of a few roots skips most row groups of a file.
ROW_GROUP_ROWS = 128 * 1024
# Above the size of one bucket, so the sort writes one file per bucket.
TARGET_FILE_BYTES = 1 << 30
# The first writes and the sorted files are both on disk until the vacuum. Both
# are about the size of the kept columns at the source encoding.
PEAK_DISK_FACTOR = 2.5

# Each encoding the estimate tries on every column.
CANDIDATE_ENCODINGS = {
    "dictionary": ColumnProperties(dictionary_enabled=True),
    "PLAIN": _PLAIN,
    "DELTA_BINARY_PACKED": _DELTA,
    "BYTE_STREAM_SPLIT": _SPLIT,
}


def _writer(columns: dict[str, ColumnProperties], level: int = FINAL_LEVEL) -> WriterProperties:
    return WriterProperties(
        compression="ZSTD",
        compression_level=level,
        max_row_group_size=ROW_GROUP_ROWS,
        column_properties=columns,
    )


def _source_files(source: UPath) -> tuple[pafs.FileSystem, list[str], pl.DataFrame]:
    """The filesystem, file paths and add actions of the Delta table at `source`."""
    actions = pl.from_arrow(DeltaTable(str(source)).get_add_actions(flatten=True))
    filesystem, root = pafs.FileSystem.from_uri(str(source))
    return filesystem, [f"{root}/{path}" for path in actions["path"]], actions


def _read(filesystem: pafs.FileSystem, paths: list[str]) -> pl.DataFrame:
    table = ds.dataset(paths, filesystem=filesystem, format="parquet").to_table(columns=COLUMNS)
    return pl.from_arrow(table)


def _with_bucket(synapses: pl.DataFrame, n_buckets: int) -> pl.DataFrame:
    """`synapses` with `size` as Int32 and the bucket of each post root."""
    size = synapses["size"]
    assert (size == size.round()).all() and size.max() < 2**31, "`size` is not a whole Int32"
    roots = synapses["post_pt_root_id"].unique()
    buckets = pl.Series([hash_partition(root, n_buckets) for root in roots], dtype=pl.Int32)
    return synapses.with_columns(
        pl.col("size").cast(pl.Int32),
        pl.col("post_pt_root_id")
        .replace_strict(roots, buckets, return_dtype=pl.Int32)
        .alias(BUCKET_COLUMN),
    )


def _column_bytes(files: list) -> dict[str, int]:
    """The compressed bytes of each column, summed over the parquet `files` (paths
    or open files)."""
    total: dict[str, int] = {}
    for file in files:
        metadata = pq.ParquetFile(file).metadata
        for group in range(metadata.num_row_groups):
            row_group = metadata.row_group(group)
            for i in range(row_group.num_columns):
                column = row_group.column(i)
                total[column.path_in_schema] = total.get(column.path_in_schema, 0) + column.total_compressed_size
    return total


def _download_bytes(filesystem: pafs.FileSystem, paths: list[str]) -> int:
    """The compressed bytes of the kept columns over the source files, from their
    footers. A read of the kept columns transfers only these bytes."""
    with ThreadPoolExecutor(16) as pool:
        footers = pool.map(lambda path: _column_bytes([filesystem.open_input_file(path)]), paths)
        return sum(footer[column] for footer in footers for column in COLUMNS)


def _sample_bytes(sample: pl.DataFrame, columns: dict[str, ColumnProperties]) -> dict[str, int]:
    """The compressed bytes of each column of `sample`, written as a Delta table."""
    with tempfile.TemporaryDirectory() as folder:
        write_deltalake(folder, sample.to_arrow(), writer_properties=_writer(columns))
        return _column_bytes(list(Path(folder).glob("*.parquet")))


def _free_bytes(path: Path) -> int:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.2f} GB"


def estimate_bucketed_synapses(source: UPath, n_buckets: int = N_BUCKETS) -> str:
    """A report of the download, the output size and the disk the build needs, with
    the bytes per row of each column under each candidate encoding."""
    filesystem, paths, actions = _source_files(source)
    n_rows = actions["num_records"].sum()
    download = _download_bytes(filesystem, paths)

    sample = _with_bucket(_read(filesystem, paths[:1]), n_buckets).drop(BUCKET_COLUMN).sort("post_pt_root_id")
    by_encoding = {
        name: _sample_bytes(sample, dict.fromkeys(COLUMNS, properties))
        for name, properties in CANDIDATE_ENCODINGS.items()
    }
    final = _sample_bytes(sample, FINAL_COLUMNS)
    output = sum(final.values()) / len(sample) * n_rows
    peak = PEAK_DISK_FACTOR * download
    free = _free_bytes(TABLE_PATH)

    lines = [
        f"Source: {source}",
        f"  {len(paths)} files, {n_rows:,} rows, {_gb(actions['size_bytes'].sum())}",
        f"  download of the kept columns: {_gb(download)}",
        f"Output: about {_gb(output)} in {n_buckets} files of about {_gb(output / n_buckets)}",
        f"Disk: peak about {_gb(peak)}, free {_gb(free)}" + ("" if peak < free else "  NOT ENOUGH"),
        f"Bytes per row on one sorted source file ({len(sample):,} rows, zstd {FINAL_LEVEL}):",
        "  " + "".join(f"{name:>22}" for name in ["column", *CANDIDATE_ENCODINGS, "chosen"]),
    ]
    for column in COLUMNS:
        values = [by_encoding[name][column] / len(sample) for name in CANDIDATE_ENCODINGS]
        values.append(final[column] / len(sample))
        lines.append("  " + f"{column:>22}" + "".join(f"{value:>22.3f}" for value in values))
    lines.append("  " + f"{'total':>22}" + " " * 22 * len(CANDIDATE_ENCODINGS) + f"{sum(final.values()) / len(sample):>22.3f}")
    return "\n".join(lines)


def _checksum(ids: np.ndarray) -> tuple[int, int, int]:
    """The count, sum and sum of squares of `ids`, the sums mod 2**64."""
    ids = ids.astype(np.uint64)
    return len(ids), int(ids.sum(dtype=np.uint64)), int((ids * ids).sum(dtype=np.uint64))


def _add(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return a[0] + b[0], (a[1] + b[1]) % 2**64, (a[2] + b[2]) % 2**64


def squash_history(path: Path) -> None:
    """Replace the Delta log at `path` with one commit that adds the current files."""
    table = DeltaTable(str(path))
    schema = table.schema()
    partition_by = table.metadata().partition_columns
    actions = [
        AddAction(
            path=row["path"],
            size=row["size_bytes"],
            partition_values={key: str(value) for key, value in row["partition"].items()},
            modification_time=row["modification_time"],
            data_change=True,
            stats=json.dumps(
                {
                    "numRecords": row["num_records"],
                    "minValues": row["min"],
                    "maxValues": row["max"],
                    "nullCount": row["null_count"],
                }
            ),
        )
        for row in pa.table(table.get_add_actions(flatten=False)).to_pylist()
    ]
    old_log = path / "_delta_log.old"
    (path / "_delta_log").rename(old_log)
    create_table_with_add_actions(str(path), schema, actions, mode="error", partition_by=partition_by)
    shutil.rmtree(old_log)


def _write_layout(path: Path, n_buckets: int) -> None:
    layout = {
        "@type": "cistern_layout_v1",
        "transforms": [
            {
                "name": BUCKET_COLUMN,
                "source": "post_pt_root_id",
                "transform": {"@type": "iceberg_bucket", "n_buckets": n_buckets},
            }
        ],
    }
    (path / LAYOUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (path / LAYOUT_PATH).write_text(json.dumps(layout, indent=2) + "\n")


def _check(path: Path, n_buckets: int, source_checksum: tuple[int, int, int]) -> None:
    """Assert that the table at `path` holds the source rows, one sorted file per
    bucket, and nothing else."""
    table = DeltaTable(str(path))
    assert len(table.history()) == 1, "the Delta log holds more than one version"
    actions = pl.from_arrow(table.get_add_actions(flatten=True))
    assert actions[f"partition.{BUCKET_COLUMN}"].is_unique().all(), "a bucket has more than one file"
    assert actions["min.post_pt_root_id"].null_count() == 0, "a file has no stats"
    files = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    expected = {*actions["path"], "_delta_log/00000000000000000000.json", LAYOUT_PATH}
    assert files == expected, f"unexpected files: {sorted(files ^ expected)}"

    checksum = (0, 0, 0)
    for file in actions["path"]:
        frame = pl.read_parquet(path / file, columns=["id", "post_pt_root_id"], hive_partitioning=False)
        assert frame["post_pt_root_id"].is_sorted(), f"{file} is not sorted on post_pt_root_id"
        bucket = int(file.split("/")[0].split("=")[1])
        roots = frame["post_pt_root_id"].unique()
        assert all(hash_partition(root, n_buckets) == bucket for root in roots), f"{file} has a root of another bucket"
        checksum = _add(checksum, _checksum(frame["id"].to_numpy()))
    assert checksum == source_checksum, f"the ids differ from the source: {checksum} != {source_checksum}"


def build_bucketed_synapses(
    source: UPath, path: Path, n_buckets: int = N_BUCKETS, batch_files: int = BATCH_FILES
) -> None:
    """Write the kept columns of the Delta table at `source` to `path`, as a Delta
    table on `bucket[n_buckets]` of `post_pt_root_id`, with one version.

    The build runs in `{path}.partial` and renames it to `path` at the end.
    """
    start = time.perf_counter()
    filesystem, paths, actions = _source_files(source)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    peak = PEAK_DISK_FACTOR * _download_bytes(filesystem, paths)
    free = _free_bytes(partial)
    if peak > free:
        raise OSError(f"the build needs about {_gb(peak)} of disk, and {_gb(free)} is free")

    checksum = (0, 0, 0)
    batches = [paths[i : i + batch_files] for i in range(0, len(paths), batch_files)]
    for i, batch in enumerate(batches):
        synapses = _with_bucket(_read(filesystem, batch), n_buckets)
        checksum = _add(checksum, _checksum(synapses["id"].to_numpy()))
        write_deltalake(
            partial,
            synapses.to_arrow(),
            partition_by=[BUCKET_COLUMN],
            mode="error" if i == 0 else "append",
            writer_properties=SCATTER_WRITER,
        )
        print(f"  batch {i + 1}/{len(batches)}: {len(synapses):,} rows ({time.perf_counter() - start:.0f} s)")
    assert checksum[0] == actions["num_records"].sum(), "the read rows differ from the source row count"

    table = DeltaTable(str(partial))
    table.optimize.z_order(
        ["post_pt_root_id"], target_size=TARGET_FILE_BYTES, writer_properties=_writer(FINAL_COLUMNS)
    )
    print(f"  sorted ({time.perf_counter() - start:.0f} s)")
    table.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False, full=True)
    squash_history(partial)
    _write_layout(partial, n_buckets)
    _check(partial, n_buckets, checksum)
    partial.rename(path)
    print(f"  checked ({time.perf_counter() - start:.0f} s)")


def scan_post_synapses(root_ids: list[int]) -> pl.LazyFrame:
    """The rows of `microns_synapses` onto the post roots `root_ids`, without the
    bucket column. The scan opens only the buckets of `root_ids`."""
    buckets = sorted({hash_partition(root_id, N_BUCKETS) for root_id in root_ids})
    # NOTE: Polars cannot derive the bucket from `post_pt_root_id`, so without the
    # bucket filter it opens every file.
    return (
        Tables["microns_synapses"]
        .lazy()
        .filter(pl.col(BUCKET_COLUMN).is_in(buckets), pl.col("post_pt_root_id").is_in(root_ids))
        .drop(BUCKET_COLUMN)
    )


Tables.register(
    "microns_synapses",
    description=(
        "Every synapse of `synapses_pni_2` at v1822: `id`, `pre_pt_root_id`, "
        "`post_pt_root_id`, `ctr_pt_position_{x,y,z}` (voxels, 4 x 4 x 40 nm) and "
        "`size`. A local Delta table on `post_pt_root_id_bucket`, which is "
        "`bucket[256]` of `post_pt_root_id`, so deployment bucket `b` is in bucket "
        "`b % 256`. Each file is sorted on `post_pt_root_id`. Build it with "
        "`just materialize microns_synapses`, which prints an estimate first."
    ),
    materialize=lambda path: build_bucketed_synapses(MICRONS_SYNAPSES_PATH, path),
    estimate=lambda: estimate_bucketed_synapses(MICRONS_SYNAPSES_PATH),
    cache_params=lambda: {
        "version": VERSION,
        "source": str(MICRONS_SYNAPSES_PATH),
        "columns": COLUMNS,
        "n_buckets": N_BUCKETS,
        "write_options": WRITE_OPTIONS_VERSION,
    },
    format="delta",
)
