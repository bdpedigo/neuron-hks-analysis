import polars as pl

from analysis.products.structures import _add_is_multi, _count_inputs, _join_spines


def test_count_inputs_counts_synapses_and_distinct_pre_roots():
    synapses = pl.DataFrame(
        {
            "synapse_id": [1, 2, 3, 4, 5],
            "root_id": [10, 10, 10, 20, 20],
            "pre_pt_root_id": [100, 100, 200, 100, 300],
            "structure_id": [1, 1, 1, 1, None],
        },
        schema_overrides={"structure_id": pl.Int32},
    )
    counts = _count_inputs(synapses)
    # The synapse in no structure has no row, and root 20's structure 1 is not
    # root 10's structure 1.
    assert counts.to_dicts() == [
        {"root_id": 10, "structure_id": 1, "n_synapses": 3, "n_pre_root_ids": 2},
        {"root_id": 20, "structure_id": 1, "n_synapses": 1, "n_pre_root_ids": 1},
    ]


def test_is_multi_needs_two_pre_roots_and_a_unitary_prediction():
    structures = pl.DataFrame(
        {
            "n_pre_root_ids": [1, 2, 2, 2, 0],
            "unitary_pred_label": ["unitary", "unitary", "not_unitary", None, None],
        }
    )
    assert _add_is_multi(structures)["is_multi"].to_list() == [
        False,
        True,
        False,
        False,
        False,
    ]


def test_join_spines_keeps_every_structure_and_one_synapse_count():
    structures = pl.DataFrame(
        {
            "root_id": [10, 20, 20],
            "structure_id": [1, 1, 2],
            "size_nm3": [1.0, 2.0, 3.0],
            "n_synapses": [2, 1, 0],
        }
    )
    # Root 20's structure 2 has no synapse, so its counts are 0.
    spines = pl.DataFrame(
        {
            "root_id": [10, 20, 20],
            "structure_id": [1, 1, 2],
            "n_synapses": [2, 1, 0],
            "n_pre_root_ids": [2, 1, 0],
        }
    )
    joined = _join_spines(structures, spines)
    assert joined.columns.count("n_synapses") == 1
    assert "n_synapses_spines" not in joined.columns
    assert joined.to_dicts() == [
        {"root_id": 10, "structure_id": 1, "size_nm3": 1.0, "n_synapses": 2, "n_pre_root_ids": 2},
        {"root_id": 20, "structure_id": 1, "size_nm3": 2.0, "n_synapses": 1, "n_pre_root_ids": 1},
        {"root_id": 20, "structure_id": 2, "size_nm3": 3.0, "n_synapses": 0, "n_pre_root_ids": 0},
    ]
