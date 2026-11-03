# Build a registered table's cache if it is missing. Pass --force to rebuild it.
materialize name *args:
    uv run python scripts/tables.py materialize {{name}} {{args}}

# List the registered tables and the state of their caches.
status:
    uv run python scripts/tables.py status

# Write TABLES.html, a page with the description and schema of each table.
catalog:
    uv run python scripts/tables.py catalog

# Write the manifest's figures into the paper repo, re-encoded for pdfTeX. Commits nothing.
publish-figures *args:
    uv run python scripts/publish.py figures {{args}}

# Copy glued_variables.tex into the paper repo. Commits nothing.
publish-variables *args:
    uv run python scripts/publish.py variables {{args}}

# Report what publishing would change, without writing anything.
publish-status *args:
    uv run python scripts/publish.py status {{args}}
