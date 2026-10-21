import shutil
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import polars as pl

from ..io import TABLE_PATH, param_hash
from ..sources import VERSION, UnpinnedError


class Table:
    """A named table whose query plan is built lazily on first access, with optional
    opt-in parquet caching (keyed by `cache_params`) independent of its ingredients.

    `cache_params` is a function that returns the dict of the cache key, so the pins
    it reads are looked up only when the key is needed, not at import.

    A table is defined either by `build`, a query plan, or by `materialize`, which
    writes the table as a parquet file at the path it is given. A materialized
    table needs `cache_params`.

    A table with neither is a planned table: a name and a `description` only, so
    the catalog can list it before it is built.

    A table with `cache_params` is always read from its cache, which is built on
    first access. The cache is at `data/tables/{folder}/{name}/{cache_key}.parquet`,
    where `folder` defaults to `v{VERSION}`.
    To make a product's key change when an input changes, put the input's
    `cache_key` in the product's `cache_params`. A table whose pins do not exist
    yet raises `UnpinnedError` from `cache_key`."""

    def __init__(
        self,
        name: str,
        build: Optional[Callable[[], pl.LazyFrame]] = None,
        cache_params: Optional[Callable[[], dict]] = None,
        materialize: Optional[Callable[[Path], None]] = None,
        folder: Optional[str] = None,
        description: Optional[str] = None,
    ):
        if build is not None and materialize is not None:
            raise ValueError(f"{name!r}: pass at most one of `build` or `materialize`")
        if build is None and materialize is None and (cache_params is not None or not description):
            raise ValueError(f"{name!r}: a planned table takes only a `description`")
        if materialize is not None and cache_params is None:
            raise ValueError(f"{name!r}: a materialized table needs `cache_params`")
        self.name = name
        self.description = description
        self._build = build
        self._materialize = materialize
        self.cache_params = cache_params
        self.folder = folder or f"v{VERSION}"

    @cached_property
    def _plan(self) -> pl.LazyFrame:
        if self._build is None:
            raise NotImplementedError(f"{self.name!r} is planned, so it has no data yet")
        return self._build()

    @property
    def kind(self) -> str:
        if self._build is None and self._materialize is None:
            return "planned"
        if self._materialize is not None:
            return "materialized"
        if self.cache_params is not None:
            return "cached build"
        return "uncached"

    @property
    def cache_key(self) -> Optional[str]:
        if self.cache_params is None:
            return None
        return param_hash(**self.cache_params())

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

    def peek_schema(self) -> Optional[pl.Schema]:
        """The schema of the current cache, or None if it does not exist. This never
        builds anything."""
        try:
            path = self.cache_path
        except UnpinnedError:
            return None
        if path is None or not path.exists():
            return None
        return pl.scan_parquet(path).collect_schema()

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
        cache_params: Optional[Callable[[], dict]] = None,
        materialize: Optional[Callable[[Path], None]] = None,
        folder: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Table:
        table = Table(
            name,
            build,
            cache_params=cache_params,
            materialize=materialize,
            folder=folder,
            description=description,
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


def condense_schema(schema: pl.Schema) -> list[tuple[str, str, int]]:
    """`(pattern, dtype, count)` for each group of columns that differ only in an
    index or a final x/y/z, in first-seen order.

    In each name, an all-digit token becomes `*`, so `hks_0` ... `hks_31` is one
    group, `hks_*`. A final `x`, `y` or `z` token becomes a set of the ones that
    are present, so `domain_x`, `domain_y`, `domain_z` is `domain_{x,y,z}`. A
    group of one column keeps its name. The dtype is `mixed` when the columns of a
    group differ.
    """
    groups: dict[tuple[str, ...], dict] = {}
    for name, dtype in schema.items():
        tokens = tuple("*" if token.isdigit() else token for token in name.split("_"))
        axis = None
        if len(tokens) > 1 and tokens[-1] in AXES:
            tokens, axis = tokens[:-1] + ("{axis}",), tokens[-1]
        group = groups.setdefault(
            tokens, {"name": name, "dtypes": set(), "axes": set(), "count": 0}
        )
        group["dtypes"].add(str(dtype))
        group["count"] += 1
        if axis is not None:
            group["axes"].add(axis)
    condensed = []
    for tokens, group in groups.items():
        axes = [axis for axis in AXES if axis in group["axes"]]
        axis = axes[0] if len(axes) == 1 else "{" + ",".join(axes) + "}"
        pattern = "_".join(axis if token == "{axis}" else token for token in tokens)
        if group["count"] == 1:
            pattern = group["name"]
        dtype = group["dtypes"].pop() if len(group["dtypes"]) == 1 else "mixed"
        condensed.append((pattern, dtype, group["count"]))
    return condensed


AXES = ("x", "y", "z")

Tables = TableRegistry()
