"""Build and inspect the caches of registered tables.

    uv run python scripts/tables.py materialize NAME [--force]
    uv run python scripts/tables.py status

The justfile wraps both commands (`just materialize NAME`, `just status`).
"""

import argparse
import time
from pathlib import Path

import humanize
import polars as pl

from analysis import Tables
from analysis.io import TABLE_PATH, VERSION
from analysis.products.sources import UnpinnedError


def _size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def materialize(name: str, force: bool) -> None:
    try:
        table = Tables[name]
    except KeyError as error:
        raise SystemExit(error.args[0])
    try:
        path = table.cache_path
    except UnpinnedError as error:
        raise SystemExit(f"{name!r} is unpinned: {error.args[0]}")
    if path is None:
        raise SystemExit(f"{name!r} is {table.kind}, so it has no cache to build.")
    if path.exists() and not force:
        print(f"{name}: already cached")
    else:
        print(f"{name}: building {path}")
        start = time.perf_counter()
        table.ensure_cached(force=force)
        elapsed = humanize.precisedelta(time.perf_counter() - start)
        print(f"{name}: built in {elapsed}")
    n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    print(f"  path: {path}")
    print(f"  rows: {n_rows:,}")
    print(f"  size: {humanize.naturalsize(_size(path))}")


def status() -> None:
    version_dir = TABLE_PATH / f"v{VERSION}"
    rows = []
    current = set()
    for name in Tables:
        table = Tables[name]
        try:
            path = table.cache_path
        except UnpinnedError:
            rows.append((name, table.kind, "unpinned", ""))
            continue
        if path is None:
            state, size = "-", ""
        elif path.exists():
            state, size = "cached", humanize.naturalsize(_size(path))
            current.add(path)
        elif path.with_name(path.name + ".partial").exists():
            state, size = "partial", ""
        else:
            state, size = "missing", ""
        rows.append((name, table.kind, state, size))

    widths = [max(len(row[i]) for row in rows + [("table", "kind", "state", "size")]) for i in range(4)]
    print(f"Registered tables (v{VERSION}):")
    for row in [("table", "kind", "state", "size")] + rows:
        print("  " + "  ".join(value.ljust(width) for value, width in zip(row, widths)))

    # Files in the version folder that are not a current cache: older cache keys
    # of a registered table, interrupted builds, and files from other mechanisms.
    others = []
    if version_dir.exists():
        for entry in sorted(version_dir.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and entry.name in Tables:
                for child in sorted(entry.iterdir()):
                    if child in current:
                        continue
                    label = "partial" if child.name.endswith(".partial") else "stale"
                    others.append((label, child))
            else:
                others.append(("not in registry", entry))
    if others:
        print(f"\nOther files in {version_dir}:")
        for label, path in others:
            print(f"  {label:<16}{path.relative_to(version_dir)}  ({humanize.naturalsize(_size(path))})")

    other_versions = sorted(
        p.name for p in TABLE_PATH.iterdir() if p.is_dir() and p != version_dir
    )
    if other_versions:
        print(f"\nOther folders in {TABLE_PATH}: {', '.join(other_versions)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("materialize", help="build a table's cache if it is missing")
    build.add_argument("name", help="the registered table name")
    build.add_argument("--force", action="store_true", help="delete the cache and rebuild it")
    commands.add_parser("status", help="list registered tables and their cache state")
    args = parser.parse_args()
    if args.command == "materialize":
        materialize(args.name, args.force)
    else:
        status()


if __name__ == "__main__":
    main()
