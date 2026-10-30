import polars as pl
import pytest

from analysis.products.census_scan import join_chunk

# Two roots, 10 and 11, that both have a structure 7.
SYNAPSES = pl.DataFrame(
    {
        "synapse_id": [1, 2, 3, 4, 5, 6],
        "pre_pt_root_id": [20, 30, 10, 20, 40, 50],
        "post_pt_root_id": [10, 10, 10, 11, 10, 10],
    }
)
MAPPING = pl.DataFrame(
    {
        # Synapse 5 is unmapped. Synapse 3 is an autapse.
        "root_id": [10, 10, 10, 11, 10],
        "synapse_id": [1, 2, 3, 4, 6],
        "mesh_index": [100, 101, 102, 200, 103],
    }
)
PREDICTIONS = pl.DataFrame(
    {
        # Synapse 6 has no prediction.
        "root_id": [10, 10, 10, 11],
        "synapse_id": [1, 2, 3, 4],
        "pred_label": ["spine", "spine", "spine", "spine"],
        "posterior_max": [0.9, 0.8, 0.7, 0.6],
    }
)
VERTICES = pl.DataFrame(
    {
        "root_id": [10, 10, 10, 10, 11],
        "mesh_index": [100, 101, 102, 103, 200],
        "structure_id": [7, 7, 7, None, 7],
    }
)
MORPHOMETRY = pl.DataFrame(
    {
        # Structure 8 of root 10 has no synapse.
        "root_id": [10, 10, 11],
        "structure_id": [7, 8, 7],
        "size_nm3": [1.0, 2.0, 3.0],
        "area_nm2": [4.0, 5.0, 6.0],
        "n_synapses": [3, 0, 1],
    }
)
UNITARY = pl.DataFrame(
    {
        "root_id": [10, 10, 11],
        "structure_id": [7, 8, 7],
        "posterior_unitary": [0.9, 0.2, 0.9],
        "pred_label": ["unitary", "not_unitary", "unitary"],
    }
)


def _join(**overrides):
    inputs = dict(
        synapses=SYNAPSES,
        synapse_mapping=MAPPING,
        synapse_predictions=PREDICTIONS,
        vertex_structures=VERTICES,
        structure_morphometry=MORPHOMETRY,
        structure_predictions=UNITARY,
    )
    return join_chunk(**{**inputs, **overrides})


def test_synapse_rows():
    synapses, _ = _join()
    # No autapse (3) and no unmapped synapse (5). Synapse 6 has no prediction and no
    # structure.
    assert synapses.rows() == [
        (1, 10, "spine", pytest.approx(0.9), 7),
        (2, 10, "spine", pytest.approx(0.8), 7),
        (6, 10, None, None, None),
        (4, 11, "spine", pytest.approx(0.6), 7),
    ]


def test_spine_rows():
    _, spines = _join()
    rows = {
        (r["post_pt_root_id"], r["structure_id"]): r for r in spines.iter_rows(named=True)
    }
    assert sorted(rows) == [(10, 7), (10, 8), (11, 7)]
    # Structure 7 of root 10 has two synapses from two roots once the autapse drops.
    assert (rows[10, 7]["n_synapses"], rows[10, 7]["n_pre_root_ids"]) == (2, 2)
    assert rows[10, 7]["is_multi"]
    # A structure with no synapse keeps its row, with zero counts.
    assert (rows[10, 8]["n_synapses"], rows[10, 8]["is_multi"]) == (0, False)
    # The structure 7 of root 11 is not the one of root 10.
    assert (rows[11, 7]["n_synapses"], rows[11, 7]["is_multi"]) == (1, False)


def test_a_partial_vertex_attempt_raises():
    with pytest.raises(AssertionError, match="vertex_structures"):
        _join(vertex_structures=VERTICES.filter(pl.col("mesh_index") != 101))


def test_a_structure_without_morphometry_raises():
    with pytest.raises(AssertionError, match="morphometry"):
        _join(structure_morphometry=MORPHOMETRY.filter(pl.col("root_id") == 10))


def test_write_part_round_trips_with_the_encodings(tmp_path):
    from analysis.products.census_scan import ENCODINGS, SPINES, SYNAPSES, write_part

    for name, frame in zip((SYNAPSES, SPINES), _join()):
        path = tmp_path / name / "part-0.parquet"
        write_part(frame, path, ENCODINGS[name])
        read = pl.read_parquet(path)
        assert read.columns == frame.columns
        assert read.equals(frame.with_columns(pl.col(pl.Enum).cast(pl.String)))
        assert not list(path.parent.glob("*.tmp"))
