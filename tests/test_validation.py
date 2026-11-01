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
            "neurd_export": SimpleNamespace(
                lazy=lambda: _export().filter(pl.col("valid")).drop("valid").lazy()
            ),
        },
    )
    monkeypatch.setattr(
        validation,
        "scan_post_synapses",
        lambda roots: synapses.filter(pl.col("post_pt_root_id").is_in(roots)).lazy(),
    )
    labels = validation.fetch_census_neurd_labels()
    # Synapses 10 and 12 are from a proofread cell onto a column cell, so both
    # pulls hold them.
    assert labels["synapse_id"].to_list() == [10, 11, 12]
    assert labels["neurd_detailed_label"].to_list() == ["spine_head", "axon_bouton", "shaft"]
    assert labels["neurd_label"].to_list() == ["spine", None, "shaft"]
    assert labels["on_column"].to_list() == [True, True, True]
    assert labels["from_proofread"].to_list() == [True, True, True]


def test_neurd_export_keeps_the_valid_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(validation.pl, "scan_delta", lambda path: _export().lazy())
    path = tmp_path / "neurd_export.parquet"
    validation.download_neurd_export(path)
    export = pl.read_parquet(path)
    assert export.columns == ["id", "target_id", "value"]
    # The invalid row (id 5) is dropped, and the rows are sorted on target_id.
    assert export["id"].to_list() == [1, 2, 3, 4]
    assert export["target_id"].to_list() == [10, 11, 12, 13]
