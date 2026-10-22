"""Build and inspect the caches of registered tables.

    uv run python scripts/tables.py materialize NAME [--force]
    uv run python scripts/tables.py status
    uv run python scripts/tables.py catalog [--out TABLES.html]

The justfile wraps these commands (`just materialize NAME`, `just status`,
`just catalog`).
"""

import argparse
import time
from html import escape
from pathlib import Path

import humanize
import polars as pl

from analysis import Tables
from analysis.io import TABLE_PATH
from analysis.products.catalog import condense_schema
from analysis.sources import VERSION, UnpinnedError


def _size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def _frozen_state(table) -> tuple[str, str]:
    """Whether the file of a frozen table is present, and its size."""
    if not table.frozen.exists():
        return "missing", ""
    return "present", humanize.naturalsize(_size(table.frozen))


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
        if table.frozen is not None:
            rows.append((name, table.kind, *_frozen_state(table)))
            continue
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

    # The folders of frozen tables are not caches, so they are not listed here.
    frozen = {Tables[name].frozen for name in Tables if Tables[name].frozen is not None}
    other_versions = sorted(
        p.name
        for p in TABLE_PATH.iterdir()
        if p.is_dir() and p != version_dir and p not in frozen
    )
    if other_versions:
        print(f"\nOther folders in {TABLE_PATH}: {', '.join(other_versions)}")


CATALOG_STYLE = """
:root { --bg: #fff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d9d9de; --chip: #f0f0f3; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161618; --fg: #ececef; --muted: #9a9aa2; --line: #34343a; --chip: #26262b; }
}
body { background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 900px;
  padding: 24px 16px; font: 15px/1.5 -apple-system, system-ui, sans-serif; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 8px; border-bottom: 1px solid var(--line); }
.meta { color: var(--muted); font-size: 13px; }
.table { margin: 20px 0; }
.table h3 { font: 600 16px ui-monospace, Menlo, monospace; margin: 0; }
.chip { background: var(--chip); border-radius: 4px; padding: 1px 6px; font-size: 12px; }
.table p { margin: 4px 0 8px; }
table.schema { border-collapse: collapse; font: 13px ui-monospace, Menlo, monospace; }
table.schema td { padding: 1px 16px 1px 0; vertical-align: top; }
table.schema td.dtype, table.schema td.count { color: var(--muted); }
nav a { color: inherit; margin-right: 12px; font: 13px ui-monospace, Menlo, monospace; }
"""


def _cache_state(table) -> tuple[str, str]:
    """The cache state of `table` and its row count, without building anything."""
    if table.frozen is not None:
        return _frozen_state(table)
    try:
        path = table.cache_path
    except UnpinnedError:
        return "unpinned", ""
    if path is None:
        return "-", ""
    if not path.exists():
        return "missing", ""
    n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    return "cached", f"{n_rows:,} rows, {humanize.naturalsize(_size(path))}"


def _catalog_entry(table) -> str:
    state, stats = _cache_state(table)
    chips = [table.kind] + ([state] if state != "-" else []) + ([stats] if stats else [])
    html = [f'<div class="table" id="{table.name}"><h3>{escape(table.name)}</h3>']
    html.append('<div class="meta">' + " ".join(f'<span class="chip">{escape(c)}</span>' for c in chips) + "</div>")
    html.append(f"<p>{_code(table.description or '(no description)')}</p>")
    schema = table.peek_schema()
    if schema is not None:
        html.append('<table class="schema">')
        for pattern, dtype, count in condense_schema(schema):
            count_text = f"×{count}" if count > 1 else ""
            html.append(
                f'<tr><td>{escape(pattern)}</td><td class="dtype">{escape(dtype)}</td>'
                f'<td class="count">{count_text}</td></tr>'
            )
        html.append("</table>")
    html.append("</div>")
    return "\n".join(html)


def _code(text: str) -> str:
    """Escape `text`, and show its `backticked` parts as code."""
    parts = escape(text).split("`")
    return "".join(f"<code>{part}</code>" if i % 2 else part for i, part in enumerate(parts))


def catalog(out: Path) -> None:
    tables = [Tables[name] for name in Tables if Tables[name].kind != "uncached"]
    sections = {
        "Built tables": [t for t in tables if t.kind not in ("planned", "frozen")],
        "Frozen tables": [t for t in tables if t.kind == "frozen"],
        "Planned tables": [t for t in tables if t.kind == "planned"],
    }
    missing = [t.name for t in tables if not t.description]
    if missing:
        print(f"No description: {', '.join(missing)}")
    body = [
        "<h1>Table catalog</h1>",
        f'<div class="meta">Generated {time.strftime("%Y-%m-%d %H:%M")} by '
        "<code>just catalog</code>. A schema shows the current cache, or the file of a "
        "frozen table. Columns that "
        "differ only in an index or in x/y/z are one row.</div>",
    ]
    for title, section in sections.items():
        body.append(f"<h2>{title}</h2><nav>")
        body.extend(f'<a href="#{t.name}">{t.name}</a>' for t in section)
        body.append("</nav>")
        body.extend(_catalog_entry(t) for t in section)
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Table catalog</title><style>{CATALOG_STYLE}</style></head>"
        f"<body>{''.join(body)}</body></html>"
    )
    out.write_text(page)
    print(f"Wrote {out} ({len(tables)} tables)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("materialize", help="build a table's cache if it is missing")
    build.add_argument("name", help="the registered table name")
    build.add_argument("--force", action="store_true", help="delete the cache and rebuild it")
    commands.add_parser("status", help="list registered tables and their cache state")
    page = commands.add_parser("catalog", help="write an HTML page that lists the tables")
    page.add_argument("--out", type=Path, default=Path("TABLES.html"), help="the page to write")
    args = parser.parse_args()
    if args.command == "materialize":
        materialize(args.name, args.force)
    elif args.command == "catalog":
        catalog(args.out)
    else:
        status()


if __name__ == "__main__":
    main()
