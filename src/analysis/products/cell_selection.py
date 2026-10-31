"""P6: which cells are in the census (proofread, column, thalamic)."""

import numpy as np
import pandas as pd
import polars as pl

from .cell_info import load_neuron_info
from ..sources import VERSION
from .catalog import Tables, write_parquet_atomic

COLUMN_SOURCE = "allen_v1_column_types_slanted_ref"
THALAMIC_TYPE = "TH"
THALAMIC_BROAD_TYPE = "thalamic"
FULLY_EXTENDED = "fully_extended"
# NOTE: bump SELECTION_VERSION when a rule of `select_cells` changes, so the cache
# key of `cell_selection` and of the tables that read it moves. 2: a thalamic cell
# needs a fully extended axon (2026-10-30).
SELECTION_VERSION = 2


def select_cells(cell_info: pd.DataFrame) -> pd.DataFrame:
    """Flag each root of `cell_info` (indexed by `pt_root_id`) for the census.

    A root is proofread if its axon is cleaned, and it is a column cell or a
    thalamic cell with a fully extended axon. Roots that appear twice are dropped.
    """
    # NOTE: the old census (meshrep `tools/make_mega_tables.py`) kept every
    # thalamic cell with a cleaned axon. At v1822, 38 of the 73 are partially
    # extended, and 20 of those are small pieces (median 144 outputs) whose output
    # rates are noisy. So a thalamic cell needs `axon_strategy` fully_extended
    # (user decision, 2026-10-30). The column cells keep every strategy.
    cell_info = cell_info[~cell_info.index.duplicated(keep=False)]
    is_column = cell_info["cell_type_source"] == COLUMN_SOURCE
    is_full_thalamic = (cell_info["cell_type"] == THALAMIC_TYPE) & (
        cell_info["axon_strategy"] == FULLY_EXTENDED
    )
    is_proofread = (is_column | is_full_thalamic) & cell_info["axon_cleaned"].astype(bool)
    return pd.DataFrame(
        {
            "is_proofread": is_proofread,
            "is_column": is_proofread & is_column,
            "is_thalamic": is_proofread & (cell_info["broad_type"] == THALAMIC_BROAD_TYPE),
        },
        index=cell_info.index,
    )


def _roots(selection: pd.DataFrame, flag: str) -> np.ndarray:
    return selection.index[selection[flag]].unique().to_numpy()


def proofread_roots(selection: pd.DataFrame) -> np.ndarray:
    return _roots(selection, "is_proofread")


def column_roots(selection: pd.DataFrame) -> np.ndarray:
    return _roots(selection, "is_column")


def thalamic_roots(selection: pd.DataFrame) -> np.ndarray:
    return _roots(selection, "is_thalamic")


def build_cell_selection(version=VERSION) -> pl.DataFrame:
    selection = select_cells(load_neuron_info(version=version))
    return pl.from_pandas(selection.reset_index())


Tables.register(
    "cell_selection",
    description=(
        "One row per root in the CAVE view `aibs_cell_info`, with the census flags "
        "`is_proofread`, `is_column` and `is_thalamic`. A root is proofread if its "
        "axon is cleaned, and it is a column cell or a thalamic cell with a fully "
        "extended axon (`axon_strategy`)."
    ),
    materialize=lambda path: write_parquet_atomic(build_cell_selection(), path),
    cache_params=lambda: {"version": VERSION, "rules": SELECTION_VERSION},
)
