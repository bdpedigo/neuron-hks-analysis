import shutil
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import polars as pl

from ..io import TABLE_PATH, VERSION, param_hash


class Table:
    """A named table whose query plan is built lazily on first access, with optional
    opt-in parquet caching (keyed by `cache_params`) independent of its ingredients.

    A table is defined either by `build`, a query plan, or by `materialize`, which
    writes the table as a parquet file at the path it is given. A materialized
    table needs `cache_params`.

    A table with `cache_params` is always read from its cache, which is built on
    first access. The cache is at `data/tables/{folder}/{name}/{cache_key}.parquet`,
    where `folder` defaults to `v{VERSION}`.
    To make a product's key change when an input changes, put the input's
    `cache_key` in the product's `cache_params`."""

    def __init__(
        self,
        name: str,
        build: Optional[Callable[[], pl.LazyFrame]] = None,
        cache_params: Optional[dict] = None,
        materialize: Optional[Callable[[Path], None]] = None,
        folder: Optional[str] = None,
    ):
        if (build is None) == (materialize is None):
            raise ValueError(f"{name!r}: pass exactly one of `build` or `materialize`")
        if materialize is not None and cache_params is None:
            raise ValueError(f"{name!r}: a materialized table needs `cache_params`")
        self.name = name
        self._build = build
        self._materialize = materialize
        self.cache_params = cache_params
        self.folder = folder or f"v{VERSION}"

    @cached_property
    def _plan(self) -> pl.LazyFrame:
        return self._build()

    @property
    def kind(self) -> str:
        if self._materialize is not None:
            return "materialized"
        if self.cache_params is not None:
            return "cached build"
        return "uncached"

    @property
    def cache_key(self) -> Optional[str]:
        if self.cache_params is None:
            return None
        return param_hash(**self.cache_params)

    @property
    def cache_path(self) -> Optional[Path]:
        key = self.cache_key
        if key is None:
            return None
        return TABLE_PATH / self.folder / self.name / f"{key}.parquet"

    def ensure_cached(self, force: bool = False) -> Path:
        """Build the cache if it is missing, and return its path.

        With `force`, delete the cache and any interrupted build first.
        """
        path = self.cache_path
        if path is None:
            raise ValueError(f"{self.name!r} has no `cache_params`, so it has no cache")
        if force:
            for leftover in (path, _partial_path(path), _tmp_path(path)):
                _remove(leftover)
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._materialize is not None:
            self._materialize(path)
        else:
            write_parquet_atomic(self._plan.collect(), path)
        return path

    def lazy(self) -> pl.LazyFrame:
        if self.cache_path is None:
            return self._plan
        return pl.scan_parquet(self.ensure_cached())

    def collect(self, **kwargs) -> pl.DataFrame:
        return self.lazy().collect(**kwargs)

    @cached_property
    def schema(self) -> pl.Schema:
        return self.lazy().collect_schema()

    def describe(self) -> str:
        path = self.cache_path
        return f"{self.name} (not cached)" if path is None else f"{self.name} (cached at {path})"


def _partial_path(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path`, so that an interrupted write never leaves a file there."""
    tmp = _tmp_path(path)
    df.write_parquet(tmp, compression="snappy")
    tmp.rename(path)


class TableRegistry:
    def __init__(self):
        self._tables: dict[str, Table] = {}

    def register(
        self,
        name: str,
        build: Optional[Callable[[], pl.LazyFrame]] = None,
        cache_params: Optional[dict] = None,
        materialize: Optional[Callable[[Path], None]] = None,
        folder: Optional[str] = None,
    ) -> Table:
        table = Table(
            name, build, cache_params=cache_params, materialize=materialize, folder=folder
        )
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
