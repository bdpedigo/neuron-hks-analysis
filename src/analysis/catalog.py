from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import polars as pl

from .io import TABLE_PATH, VERSION, param_hash


class Table:
    """A named table whose query plan is built lazily on first access, with optional
    opt-in parquet caching (keyed by `cache_params`) independent of its ingredients.

    A table is defined either by `build`, a query plan, or by `materialize`, which
    writes the table as parquet files under the path it is given. A materialized
    table needs `cache_params`."""

    def __init__(
        self,
        name: str,
        build: Optional[Callable[[], pl.LazyFrame]] = None,
        cache_params: Optional[dict] = None,
        materialize: Optional[Callable[[Path], None]] = None,
    ):
        if (build is None) == (materialize is None):
            raise ValueError(f"{name!r}: pass exactly one of `build` or `materialize`")
        if materialize is not None and cache_params is None:
            raise ValueError(f"{name!r}: a materialized table needs `cache_params`")
        self.name = name
        self._build = build
        self._materialize = materialize
        self.cache_params = cache_params

    @cached_property
    def _plan(self) -> pl.LazyFrame:
        return self._build()

    @property
    def cache_path(self) -> Optional[Path]:
        if self.cache_params is None:
            return None
        key = param_hash(**self.cache_params)
        return TABLE_PATH / f"v{VERSION}" / f"{self.name}-{key}.parquet"

    def lazy(self) -> pl.LazyFrame:
        path = self.cache_path
        if path is not None and path.exists():
            return pl.scan_parquet(path)
        if self._materialize is not None:
            self._materialize(path)
            return pl.scan_parquet(path)
        return self._plan

    def collect(self, **kwargs) -> pl.DataFrame:
        path = self.cache_path
        if self._build is not None and path is not None and not path.exists():
            df = self._plan.collect(**kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path, compression="snappy")
            return df
        return self.lazy().collect(**kwargs)

    @cached_property
    def schema(self) -> pl.Schema:
        return self.lazy().collect_schema()

    def describe(self) -> str:
        path = self.cache_path
        return f"{self.name} (not cached)" if path is None else f"{self.name} (cached at {path})"


class TableRegistry:
    def __init__(self):
        self._tables: dict[str, Table] = {}

    def register(
        self,
        name: str,
        build: Optional[Callable[[], pl.LazyFrame]] = None,
        cache_params: Optional[dict] = None,
        materialize: Optional[Callable[[Path], None]] = None,
    ) -> Table:
        table = Table(name, build, cache_params=cache_params, materialize=materialize)
        self._tables[name] = table
        return table

    def __getitem__(self, name: str) -> Table:
        if name not in self._tables:
            available = ", ".join(sorted(self._tables)) or "(none registered)"
            raise KeyError(f"No table named {name!r}. Available tables: {available}")
        return self._tables[name]

    def __iter__(self):
        return iter(self._tables)

    def __contains__(self, name: str) -> bool:
        return name in self._tables


Tables = TableRegistry()
