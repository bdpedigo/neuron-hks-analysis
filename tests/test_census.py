import polars as pl

from analysis.io import LABEL_CATEGORIES
from analysis.products.census import (
    _add_skeleton_sides,
    _add_tag_detailed,
    _drop_autapses,
    _spine_columns,
)


def test_tag_detailed_splits_only_spine_synapses():
    synapses = pl.DataFrame(
        {
            "tag": ["spine", "spine", "spine", "shaft", "shaft", None],
            "spine_is_multi": [True, False, None, True, None, True],
        },
        schema_overrides={"tag": pl.Enum(LABEL_CATEGORIES)},
    )
    # A spine with no multi-input call is single, a multi-input shaft synapse stays
    # shaft, and a synapse with no tag stays null.
    assert _add_tag_detailed(synapses)["tag_detailed"].cast(pl.String).to_list() == [
        "multi_spine",
        "single_spine",
        "single_spine",
        "shaft",
        "shaft",
        None,
    ]


def test_drop_autapses():
    synapses = pl.DataFrame(
        {"synapse_id": [1, 2], "pre_pt_root_id": [10, 20], "post_pt_root_id": [10, 10]}
    )
    assert _drop_autapses(synapses)["synapse_id"].to_list() == [2]


def test_spine_columns_prefix_all_but_the_keys():
    structure = pl.DataFrame(
        {
            "synapse_id": [1],
            "root_id": [10],
            "pre_pt_root_id": [20],
            "structure_id": [3],
            "n_pre_root_ids": [2],
            "is_multi": [True],
        }
    )
    assert _spine_columns(structure).columns == [
        "synapse_id",
        "structure_id",
        "spine_n_pre_root_ids",
        "spine_is_multi",
    ]


def test_skeleton_sides_match_the_root_of_each_side():
    synapses = pl.DataFrame(
        {
            "synapse_id": [1, 2, 3],
            "pre_pt_root_id": [100, 200, 100],
            "post_pt_root_id": [10, 10, 30],
        }
    )
    features = pl.DataFrame(
        {
            "synapse_id": [1, 1, 2, 3],
            "root_id": [10, 100, 10, 999],
            "side": ["post", "pre", "post", "post"],
            "compartment": ["dendrite", "axon", "soma", "dendrite"],
            "path_distance_to_nuc": [1.0, 2.0, 3.0, 4.0],
            "max_dist_to_tip": [5.0, None, 6.0, 7.0],
        }
    )
    joined = _add_skeleton_sides(synapses, features.lazy())
    # Synapse 2's pre root has no skeleton. Synapse 3's feature row is on another
    # root, so it does not match: root 30 has no skeleton.
    assert joined.select(
        "synapse_id", "pre_compartment", "post_compartment", "post_max_dist_to_tip"
    ).to_dicts() == [
        {"synapse_id": 1, "pre_compartment": "axon", "post_compartment": "dendrite", "post_max_dist_to_tip": 5.0},
        {"synapse_id": 2, "pre_compartment": None, "post_compartment": "soma", "post_max_dist_to_tip": 6.0},
        {"synapse_id": 3, "pre_compartment": None, "post_compartment": None, "post_max_dist_to_tip": None},
    ]
    assert "pre_max_dist_to_tip" not in joined.columns
