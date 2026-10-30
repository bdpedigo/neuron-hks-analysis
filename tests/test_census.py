import polars as pl

import pytest

from analysis.io import LABEL_CATEGORIES, LABEL_DETAILED_CATEGORIES
from analysis.products.census import (
    _add_skeleton_sides,
    _add_tag_detailed,
    _drop_autapses,
    _spine_columns,
    aggregate_post,
    aggregate_pre,
    join_census_features,
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


def test_join_census_features():
    synapses = pl.LazyFrame(
        {
            "id": [1, 2, 3, 4],
            "pre_pt_root_id": [20, 30, 10, 20],
            "post_pt_root_id": [10, 10, 10, 11],
            "size": [5, 6, 7, 8],
        }
    )
    # Synapse 2 is unmapped, and synapse 3 is an autapse. Synapse 4 is on root 11,
    # and its structure ID 7 is a spine of root 10 only.
    features = pl.LazyFrame(
        {"synapse_id": [1, 3, 4], "tag": ["spine", "spine", "spine"], "structure_id": [7, 7, 7]}
    )
    spines = pl.LazyFrame(
        {
            "root_id": [10],
            "structure_id": [7],
            "n_synapses": [2],
            "n_pre_root_ids": [2],
            "unitary_pred_label": ["unitary"],
            "is_multi": [True],
            "size_nm3": [1.0],
            "area_nm2": [2.0],
            "neck_length_nm": [3.0],
        }
    )
    joined = join_census_features(synapses, features, spines).collect().sort("synapse_id")
    assert joined["synapse_id"].to_list() == [1, 2, 4]
    assert joined["tag"].cast(pl.String).to_list() == ["spine", None, "spine"]
    assert joined["spine_is_multi"].to_list() == [True, None, None]
    assert joined["tag_detailed"].cast(pl.String).to_list() == [
        "multi_spine",
        None,
        "single_spine",
    ]
    assert "spine_neck_length_nm" not in joined.columns


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


def _synapses(rows: list[tuple]) -> pl.LazyFrame:
    """Rows of (pre root, post root, post broad type, tag, tag_detailed,
    structure_id, size)."""
    columns = [
        "pre_pt_root_id",
        "post_pt_root_id",
        "post_broad_type",
        "tag",
        "tag_detailed",
        "structure_id",
        "size",
    ]
    return pl.DataFrame(
        rows,
        schema=columns,
        orient="row",
        schema_overrides={
            "tag": pl.Enum(LABEL_CATEGORIES),
            "tag_detailed": pl.Enum(LABEL_DETAILED_CATEGORIES),
        },
    ).lazy()


def test_spine_sites_are_keyed_by_post_root_and_structure():
    # Pre root 1 reaches structure 7 on two post roots. These are two sites.
    synapses = _synapses(
        [
            (1, 10, "excitatory", "spine", "single_spine", 7, 1.0),
            (1, 20, "excitatory", "spine", "single_spine", 7, 1.0),
        ]
    )
    pre = aggregate_pre(synapses).collect()
    assert pre["pre_spine_sites"].to_list() == [2]
    post = aggregate_post(synapses).collect()
    assert post["post_spine_sites"].to_list() == [1, 1]


def test_aggregate_post_counts_and_proportions():
    synapses = _synapses(
        [
            # Two inputs on multi-input structure 1, one on single structure 2.
            (1, 10, "excitatory", "spine", "multi_spine", 1, 2.0),
            (2, 10, "excitatory", "spine", "multi_spine", 1, 4.0),
            (3, 10, "excitatory", "spine", "single_spine", 2, 6.0),
            # A spine synapse with no structure counts as a spine synapse, not a site.
            (4, 10, "excitatory", "spine", "single_spine", None, 8.0),
            (5, 10, "excitatory", "shaft", "shaft", None, 10.0),
            (6, 10, "excitatory", "soma", "soma", None, 12.0),
            (7, 10, "excitatory", None, None, None, 14.0),
        ]
    )
    row = aggregate_post(synapses).collect().row(0, named=True)
    assert row["post_pt_root_id"] == 10
    assert {
        name: row[f"post_{name}"]
        for name in [
            "total_synapses",
            "total_known_synapses",
            "unknown_synapses",
            "spine_synapses",
            "shaft_synapses",
            "soma_synapses",
            "single_spine_synapses",
            "multi_spine_synapses",
            "spine_sites",
            "multi_spine_sites",
        ]
    } == {
        "total_synapses": 7,
        "total_known_synapses": 6,
        "unknown_synapses": 1,
        "spine_synapses": 4,
        "shaft_synapses": 1,
        "soma_synapses": 1,
        "single_spine_synapses": 2,
        "multi_spine_synapses": 2,
        "spine_sites": 2,
        "multi_spine_sites": 1,
    }
    assert row["post_p_spine_synapse"] == pytest.approx(4 / 6)
    assert row["post_p_unknown_synapse"] == pytest.approx(1 / 7)
    assert row["post_p_spine_site_is_multi"] == pytest.approx(1 / 2)
    assert row["post_mean_spine_inputs"] == pytest.approx(4 / 2)
    assert row["post_mean_multi_spine_inputs"] == pytest.approx(2 / 1)
    assert row["post_mean_synapse_size"] == pytest.approx(8.0)
    assert row["post_mean_multi_spine_size"] == pytest.approx(3.0)


def test_aggregate_pre_splits_the_excitatory_targets():
    synapses = _synapses(
        [
            (1, 10, "excitatory", "spine", "single_spine", 1, 1.0),
            (1, 20, "inhibitory", "shaft", "shaft", None, 1.0),
            (1, 20, "inhibitory", "shaft", "shaft", None, 1.0),
            (2, 20, "inhibitory", "shaft", "shaft", None, 1.0),
        ]
    )
    pre = aggregate_pre(synapses).collect()
    assert pre.select(
        "pre_pt_root_id", "pre_total_synapses", "pre_shaft_synapses", "pre_to_exc_total_synapses"
    ).rows() == [(1, 3, 2, 1), (2, 1, 1, None)]
