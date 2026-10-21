import polars as pl
import pytest

from analysis.products.catalog import Table, condense_schema


def test_condense_collapses_indices_and_axes():
    schema = pl.Schema(
        {
            "synapse_id": pl.Int64,
            **{f"hks_{i}": pl.Float32 for i in range(32)},
            "curvature_mean_raw": pl.Float32,
            **{f"curvature_mean_{i}": pl.Float32 for i in range(8)},
            "domain_x": pl.Float64,
            "domain_y": pl.Float64,
            "domain_z": pl.Float64,
            **{f"rep_pt_{i}_{axis}": pl.Float64 for i in range(4) for axis in "xyz"},
        }
    )
    assert condense_schema(schema) == [
        ("synapse_id", "Int64", 1),
        ("hks_*", "Float32", 32),
        ("curvature_mean_raw", "Float32", 1),
        ("curvature_mean_*", "Float32", 8),
        ("domain_{x,y,z}", "Float64", 3),
        ("rep_pt_*_{x,y,z}", "Float64", 12),
    ]


def test_condense_keeps_lone_columns_and_reports_mixed_dtypes():
    schema = pl.Schema(
        {
            "x": pl.Int64,
            "pos_x": pl.Int64,
            "pt_root_id_1822": pl.Int64,
            "a_0": pl.Int64,
            "a_1": pl.Float32,
        }
    )
    assert condense_schema(schema) == [
        ("x", "Int64", 1),
        ("pos_x", "Int64", 1),
        ("pt_root_id_1822", "Int64", 1),
        ("a_*", "mixed", 2),
    ]


def test_planned_table_has_no_data():
    table = Table("future", description="A table that is not built yet.")
    assert table.kind == "planned"
    assert table.cache_path is None
    assert table.peek_schema() is None
    with pytest.raises(NotImplementedError):
        table.lazy()


def test_planned_table_needs_a_description():
    with pytest.raises(ValueError):
        Table("future")
