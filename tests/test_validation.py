from types import SimpleNamespace

import polars as pl

from analysis.products import validation


def _synapses():
    # `id` is the synapse ID, which is `target_id` in the NEURD export.
    return pl.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "pre_pt_root_id": [500, 100, 500, 600],
            "post_pt_root_id": [100, 100, 200, 300],
        }
    )


def _export():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "valid": [True, True, True, True, False],
            "target_id": [10, 11, 12, 13, 10],
            "value": [0, 4, 2, 6, 1],
        }
    )


def test_census_neurd_labels_join_on_synapse_id(monkeypatch):
    selection = pl.DataFrame(
        {
            "pt_root_id": [100, 200, 500],
            "is_column": [True, True, False],
            "is_proofread": [True, False, True],
        }
    )
    synapses = _synapses()
    monkeypatch.setattr(
        validation,
        "Tables",
        {
            "cell_selection": SimpleNamespace(collect=lambda: selection),
            "microns_synapses": SimpleNamespace(lazy=lambda: synapses.lazy()),
        },
    )
    monkeypatch.setattr(
        validation,
        "scan_post_synapses",
        lambda roots: synapses.filter(pl.col("post_pt_root_id").is_in(roots)).lazy(),
    )
    monkeypatch.setattr(validation.pl, "scan_delta", lambda path: _export().lazy())
    labels = validation.fetch_census_neurd_labels()
    # Synapses 10 and 12 are from a proofread cell onto a column cell, so both
    # pulls hold them. An invalid export row is dropped.
    assert labels["synapse_id"].to_list() == [10, 11, 12]
    assert labels["neurd_detailed_label"].to_list() == ["spine_head", "axon_bouton", "shaft"]
    assert labels["neurd_label"].to_list() == ["spine", None, "shaft"]
    assert labels["on_column"].to_list() == [True, True, True]
    assert labels["from_proofread"].to_list() == [True, True, True]
