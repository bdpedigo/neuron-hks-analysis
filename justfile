# Build a registered table's cache if it is missing. Pass --force to rebuild it.
materialize name *args:
    uv run python scripts/tables.py materialize {{name}} {{args}}

# List the registered tables and the state of their caches.
status:
    uv run python scripts/tables.py status
