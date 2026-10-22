from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from analysis import stash as stash_module
from analysis.stash import Stash


@pytest.fixture(autouse=True)
def stash_path(tmp_path, monkeypatch):
    monkeypatch.setattr(stash_module, "STASH_PATH", tmp_path)


def test_save_then_load():
    s = Stash("results/folds", n_splits=10)
    assert not s.exists()
    s.save(pd.DataFrame({"fold": [0, 1]}))
    assert s.exists()
    assert s.load().equals(pl.DataFrame({"fold": [0, 1]}))


def test_new_input_key_or_param_gives_new_path():
    table = SimpleNamespace(name="labels", cache_key="a")
    s = Stash("results/folds", inputs=[table], n_splits=10)
    s.save(pl.DataFrame({"fold": [0]}))
    assert Stash("results/folds", inputs=[table], n_splits=10).exists()
    assert not Stash("results/folds", inputs=[table], n_splits=5).exists()
    table.cache_key = "b"
    assert not Stash("results/folds", inputs=[table], n_splits=10).exists()


def test_stash_is_an_input():
    upstream = Stash("results/folds", n_splits=10)
    assert Stash("results/summary", inputs=[upstream]).input_keys == {
        "results/folds": upstream.cache_key
    }


def test_input_without_key_raises():
    with pytest.raises(ValueError, match="no `cache_key`"):
        Stash("results/folds", inputs=[SimpleNamespace(name="model_inputs", cache_key=None)])
