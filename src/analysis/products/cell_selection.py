"""P6: which cells are in the census (proofread, column, thalamic)."""

import numpy as np
import pandas as pd
import polars as pl

from ..io import VERSION, load_neuron_info
from .catalog import Tables, write_parquet_atomic

COLUMN_SOURCE = "allen_v1_column_types_slanted_ref"
THALAMIC_TYPE = "TH"
THALAMIC_BROAD_TYPE = "thalamic"


def select_cells(cell_info: pd.DataFrame) -> pd.DataFrame:
    """Flag each root of `cell_info` (indexed by `pt_root_id`) for the census.

    A root is proofread if it is a column cell or a thalamic cell, and its axon
    is cleaned. Roots that appear twice are dropped.
    """
    cell_info = cell_info[~cell_info.index.duplicated(keep=False)]
    is_column = cell_info["cell_type_source"] == COLUMN_SOURCE
    is_proofread = (is_column | (cell_info["cell_type"] == THALAMIC_TYPE)) & cell_info[
        "axon_cleaned"
    ].astype(bool)
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
    materialize=lambda path: write_parquet_atomic(build_cell_selection(), path),
    cache_params=lambda: {"version": VERSION},
)
