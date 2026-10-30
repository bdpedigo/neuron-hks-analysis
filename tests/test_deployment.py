import json

import polars as pl
import pytest

from analysis.products.deployment import bucket_filter, drop_older_attempts
from analysis.products.partitions import (
    LAYOUT_PATH,
    PARTITIONS_PATH,
    bucket_layout,
    hash_partition,
    nested_buckets,
)


def test_nested_buckets_hold_the_roots_of_the_chunk():
    roots = [864691134323204809 + 7919 * i for i in range(2000)]
    for root in roots:
        chunk = hash_partition(root, 256)
        assert hash_partition(root, 2048) in nested_buckets(chunk, 256, 2048)
    assert nested_buckets(3, 256, 2048) == [3 + 256 * k for k in range(8)]
    assert nested_buckets(3, 256, 256) == [3]


def test_nested_buckets_raise_when_the_counts_do_not_nest():
    with pytest.raises(ValueError):
        nested_buckets(3, 256, 1000)


def test_bucket_layout_reads_both_sidecars(tmp_path):
    cloud = tmp_path / "cloud"
    (cloud / PARTITIONS_PATH).parent.mkdir(parents=True)
    (cloud / PARTITIONS_PATH).write_text(
        json.dumps({"root_id_bucket": {"source": "root_id", "transform": "bucket[2048]"}})
    )
    local = tmp_path / "local"
    (local / LAYOUT_PATH).parent.mkdir(parents=True)
    (local / LAYOUT_PATH).write_text(
        json.dumps(
            {
                "transforms": [
                    {
                        "name": "post_pt_root_id_bucket",
                        "source": "post_pt_root_id",
                        "transform": {"@type": "iceberg_bucket", "n_buckets": 256},
                    }
                ]
            }
        )
    )
    assert bucket_layout(str(cloud)) == ("root_id_bucket", 2048)
    assert bucket_layout(str(local)) == ("post_pt_root_id_bucket", 256)
    frame = pl.DataFrame({"root_id_bucket": [3, 259, 4]})
    assert frame.filter(bucket_filter(cloud, 3, 256))["root_id_bucket"].to_list() == [3, 259]


def test_drop_older_attempts_keeps_one_whole_attempt_per_root():
    frame = pl.DataFrame(
        {
            "root_id": [1, 1, 1, 1, 2],
            "attempt_id": [10, 10, 20, 20, 5],
            "mesh_index": [0, 1, 0, 1, 0],
        }
    )
    kept = drop_older_attempts(frame)
    assert kept.columns == ["root_id", "mesh_index"]
    assert kept.sort("root_id", "mesh_index").rows() == [(1, 0), (1, 1), (2, 0)]
    assert len(drop_older_attempts(frame.filter(pl.col("root_id") == 1))) == 2
