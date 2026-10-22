from pathlib import Path
from typing import Iterable, Protocol, Union

import pandas as pd
import polars as pl

from .io import DATA_PATH, param_hash
from .products.catalog import write_parquet_atomic

STASH_PATH = DATA_PATH / "stash"


class Keyed(Protocol):
    name: str
    cache_key: str


class Stash:
    """A small table that a script computes, saved under a key of its inputs.

    The key is a hash of the `cache_key` of each input and of `params`, and it is
    part of the path, `data/stash/{name}/{key}.parquet`. So a new input or param
    gives a new path, and a script never reads a result of old inputs. An input is
    a registry `Table` or another `Stash`.

    >>> s = Stash("h01_retrain/folds", inputs=[Tables["labels"]], n_splits=10)
    >>> if not s.exists():
    ...     s.save(compute())
    >>> df = s.load()
    """

    def __init__(self, name: str, inputs: Iterable[Keyed] = (), **params):
        self.name = name
        self.params = params
        self.input_keys = {}
        for table in inputs:
            if table.cache_key is None:
                raise ValueError(
                    f"{table.name!r} has no `cache_key`, so it cannot key {name!r}. "
                    "Pass the cached tables that it reads."
                )
            self.input_keys[table.name] = table.cache_key

    @property
    def cache_key(self) -> str:
        return param_hash(**self.input_keys, **self.params)

    @property
    def path(self) -> Path:
        return STASH_PATH / self.name / f"{self.cache_key}.parquet"

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, df: Union[pl.DataFrame, pd.DataFrame]) -> None:
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(df, self.path)

    def load(self) -> pl.DataFrame:
        return pl.read_parquet(self.path)
