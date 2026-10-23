import logging
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from analysis.products import validation
from analysis.products.validation import fetch_root_neurd_labels


def _rows():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "target_id": [10, 11, 12, 13],
            "value": [0, 4, 2, 6],
            "pre_pt_root_id": [500, 100, 500, 600],
            "post_pt_root_id": [100, 100, 200, 300],
        }
    )


def _client(rows: pd.DataFrame, calls: list, max_rows: int = 1000):
    """A fake CAVE client. A query with more than `max_rows` rows logs the
    server's cut warning and returns the first `max_rows` rows, as CAVE does."""

    def synapse_target_structure(**filters):
        ((column, root_ids),) = filters.items()
        calls.append((column, root_ids))
        found = rows[rows[column].isin(root_ids)]

        def query(**kwargs):
            if len(found) > max_rows:
                logging.getLogger("caveclient.base").warning(
                    f'201 - "Limited query to {max_rows} rows'
                )
                return found.head(max_rows)
            return found

        return SimpleNamespace(query=query)

    tables = SimpleNamespace(synapse_target_structure=synapse_target_structure)
    return SimpleNamespace(materialize=SimpleNamespace(tables=tables))


def test_root_neurd_labels_batch_the_roots(monkeypatch):
    monkeypatch.setattr(validation, "NEURD_ROOTS_PER_QUERY", 2)
    calls = []
    # Root 400 has no NEURD rows, so it gives no rows.
    labels = fetch_root_neurd_labels(_client(_rows(), calls), [100, 200, 300, 400], "post")
    assert calls == [("post_pt_root_id", [100, 200]), ("post_pt_root_id", [300, 400])]
    assert labels.columns == [
        "neurd_id",
        "synapse_id",
        "neurd_value",
        "neurd_detailed_label",
        "neurd_label",
        "pre_pt_root_id",
        "post_pt_root_id",
    ]
    assert labels["synapse_id"].to_list() == [10, 11, 12, 13]
    # An axon label has no class.
    assert labels["neurd_label"].to_list() == ["spine", None, "shaft", "soma"]


def test_root_neurd_labels_filter_on_the_pre_side():
    calls = []
    labels = fetch_root_neurd_labels(_client(_rows(), calls), [500], "pre")
    assert calls == [("pre_pt_root_id", [500])]
    assert labels["synapse_id"].to_list() == [10, 12]


def test_a_cut_batch_is_split_and_queried_again():
    calls = []
    labels = fetch_root_neurd_labels(_client(_rows(), calls, max_rows=2), [100, 200, 300], "post")
    assert [roots for _, roots in calls] == [[100, 200, 300], [100], [200, 300]]
    assert labels["synapse_id"].to_list() == [10, 11, 12, 13]


def test_a_cut_single_root_raises():
    with pytest.raises(RuntimeError, match="was cut"):
        fetch_root_neurd_labels(_client(_rows(), [], max_rows=1), [100], "post")


def test_census_neurd_labels_have_one_row_per_synapse(monkeypatch):
    selection = pl.DataFrame(
        {
            "pt_root_id": [100, 200, 500],
            "is_column": [True, True, False],
            "is_proofread": [True, False, True],
        }
    )
    monkeypatch.setattr(
        validation, "Tables", {"cell_selection": SimpleNamespace(collect=lambda: selection)}
    )
    labels = validation.fetch_census_neurd_labels(_client(_rows(), []))
    # Synapses 10 and 12 are from a proofread cell onto a column cell, so both
    # pulls hold them.
    assert labels["synapse_id"].to_list() == [10, 11, 12]
    assert labels["on_column"].to_list() == [True, True, True]
    assert labels["from_proofread"].to_list() == [True, True, True]
