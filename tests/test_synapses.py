import json

import numpy as np
import polars as pl
from deltalake import DeltaTable, write_deltalake
from upath import UPath

from analysis.products.partitions import hash_partition
from analysis.products.synapses import (
    BUCKET_COLUMN,
    COLUMNS,
    LAYOUT_PATH,
    build_bucketed_synapses,
    estimate_bucketed_synapses,
)


def _source(path, n_files=3, rows_per_file=200):
    """A synapse table like the public export: extra columns, a float `size`, and
    range partitions on `post_pt_root_id`."""
    rng = np.random.default_rng(0)
    for i in range(n_files):
        post = np.sort(rng.integers(864691135000000000 + i * 1000, 864691135000000000 + (i + 1) * 1000, rows_per_file))
        frame = pl.DataFrame(
            {
                "id": rng.permutation(rows_per_file) + i * rows_per_file,
                "valid": True,
                "pre_pt_root_id": rng.integers(864691135000000000, 864691136000000000, rows_per_file),
                "post_pt_root_id": post,
                "ctr_pt_position_x": rng.integers(0, 400_000, rows_per_file).astype(np.int32),
                "ctr_pt_position_y": rng.integers(0, 300_000, rows_per_file).astype(np.int32),
                "ctr_pt_position_z": rng.integers(0, 30_000, rows_per_file).astype(np.int32),
                "size": rng.integers(100, 100_000, rows_per_file).astype(np.float64),
                "post_pt_root_id_partition": np.full(rows_per_file, i, dtype=np.int32),
            }
        )
        write_deltalake(str(path), frame.to_arrow(), partition_by=["post_pt_root_id_partition"], mode="append")
    return pl.read_delta(str(path))


def test_bucket_matches_the_cistern_vector():
    assert hash_partition(34, 16) == 3


def test_build_keeps_the_rows_on_sorted_buckets(tmp_path):
    source = _source(tmp_path / "source")
    out = tmp_path / "out.delta"
    build_bucketed_synapses(UPath(tmp_path / "source"), out, n_buckets=4, batch_files=2)

    table = DeltaTable(str(out))
    assert len(table.history()) == 1
    copy = pl.read_delta(str(out))
    assert copy.columns == [*COLUMNS, BUCKET_COLUMN]
    assert copy.schema["size"] == pl.Int32
    expected = source.select(COLUMNS).with_columns(pl.col("size").cast(pl.Int32)).sort("id")
    assert copy.drop(BUCKET_COLUMN).sort("id").equals(expected)
    buckets = [hash_partition(root, 4) for root in copy["post_pt_root_id"]]
    assert copy[BUCKET_COLUMN].to_list() == buckets

    layout = json.loads((out / LAYOUT_PATH).read_text())
    assert layout == {
        "@type": "cistern_layout_v1",
        "transforms": [
            {
                "name": BUCKET_COLUMN,
                "source": "post_pt_root_id",
                "transform": {"@type": "iceberg_bucket", "n_buckets": 4},
            }
        ],
    }
    assert not (tmp_path / "out.delta.partial").exists()


def test_estimate_reports_every_column(tmp_path):
    _source(tmp_path / "source")
    report = estimate_bucketed_synapses(UPath(tmp_path / "source"), n_buckets=4)
    assert "600 rows" in report
    for column in COLUMNS:
        assert column in report
