import polars as pl

from analysis.products.structures import _add_is_multi, _count_inputs


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
