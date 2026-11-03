"""Counts copied by hand from the publications of the compared methods."""

import polars as pl

from ..io import FROZEN_PATH
from .catalog import Tables

# The rows of meshmash-benchmarks `docs/publication_scale.csv` that the
# spines-per-publication plot sums, copied as text.
PUBLICATION_SPINE_COUNTS_PATH = FROZEN_PATH / "publication_spine_counts.csv"

Tables.register(
    "publication_spine_counts",
    description=(
        "One row per spine count of a compared method (FreSeg, VSOT, Curvature, NEURD), "
        "with the paper, dataset, use, value, derivation, the place in the paper "
        "(`location`) and the quote. The rows of meshmash-benchmarks "
        "`docs/publication_scale.csv` that `scripts/census/publication_scale.py` sums. "
        "The NEURD row is our own upper bound from `neurd_export`, not a published number."
    ),
    build=lambda: pl.scan_csv(PUBLICATION_SPINE_COUNTS_PATH),
    frozen=PUBLICATION_SPINE_COUNTS_PATH,
)
