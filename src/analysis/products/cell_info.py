"""P1: the AIBS cell info per root (S1), from CAVE at `VERSION`."""

import numpy as np
import pandas as pd
import polars as pl
from caveclient import CAVEclient
from standard_transform.datasets import minnie_transform_nm

from ..io import (
    BROAD_TYPE_CATEGORIES,
    CELL_TYPE_CATEGORIES,
    MTYPE_CATEGORIES,
    cache_table,
)
from ..sources import VERSION
from .catalog import Tables, write_parquet_atomic


def _get_client(version=VERSION):
    return CAVEclient("minnie65_public", version=version)


@cache_table("aibs_cell_info")
def _fetch_aibs_cell_info(version=VERSION, **query_args):
    client = _get_client(version=version)
    return client.materialize.query_view("aibs_cell_info", **query_args)


@cache_table("proofreading_status_and_strategy")
def _fetch_proofreading_table(version=VERSION, **query_args):
    client = _get_client(version=version)
    return client.materialize.tables.proofreading_status_and_strategy().query(
        **query_args
    )


def _as_bool(values: pd.Series) -> pd.Series:
    """Parse a CAVE boolean column, which is `'t'`/`'f'` in old versions and
    `True`/`False` from v1822 on. A missing value is False."""
    return values.map({True: True, False: False, "t": True, "f": False}).eq(True)


def load_neuron_info(version=VERSION, transform_positions=True, add_thalamic=True):
    query_args = dict(desired_resolution=[1, 1, 1], split_positions=True)

    cell_info = _fetch_aibs_cell_info(version=version, **query_args).to_pandas()

    cell_info.drop_duplicates("pt_root_id", keep=False, inplace=True)
    cell_info.set_index("pt_root_id", inplace=True)
    cell_info["axon_cleaned"] = _as_bool(cell_info["axon_cleaned"])

    if transform_positions:
        tform = minnie_transform_nm()
        cell_info[["pt_position_um_x", "pt_position_um_y", "pt_position_um_z"]] = (
            tform.apply_dataframe("pt_position", cell_info)
        )

    if add_thalamic:
        proofreading_table = _fetch_proofreading_table(
            version=version, **query_args
        ).to_pandas()

        thalamic_table = proofreading_table[
            ~_as_bool(proofreading_table["status_dendrite"])
            & (proofreading_table["strategy_dendrite"] == "none")
        ]
        thalamic_roots = thalamic_table["pt_root_id"].unique()

        root_ids = cell_info.index.unique()
        root_ids = np.concatenate([thalamic_roots, root_ids])
        root_ids = np.unique(root_ids)

        # NOTE: the thalamic roots have no AIBS row, so the reindex gives them
        # missing IDs. A float column cannot hold a supervoxel ID exactly.
        cell_info = cell_info.astype({"id": "Int64", "pt_supervoxel_id": "Int64"})
        cell_info = cell_info.reindex(root_ids)
        cell_info.loc[thalamic_roots, "broad_type"] = "thalamic"
        cell_info.loc[thalamic_roots, "mtype"] = "TH"
        cell_info.loc[thalamic_roots, "cell_type"] = "TH"
        cell_info.loc[thalamic_roots, "pt_position_y"] = 0
        cell_info.loc[thalamic_roots, "axon_cleaned"] = True

    cell_info["axon_cleaned"] = cell_info["axon_cleaned"].astype(bool)

    cell_info["broad_type"] = pd.Categorical(
        cell_info["broad_type"], categories=BROAD_TYPE_CATEGORIES, ordered=True
    )

    cell_info["cell_type"] = pd.Categorical(
        cell_info["cell_type"], categories=CELL_TYPE_CATEGORIES, ordered=True
    )

    cell_info["mtype"] = pd.Categorical(
        cell_info["mtype"], categories=MTYPE_CATEGORIES, ordered=True
    )

    cell_info.query("broad_type.isin(@BROAD_TYPE_CATEGORIES)", inplace=True)

    return cell_info


def build_cell_info(version=VERSION) -> pl.DataFrame:
    """`load_neuron_info` as polars, with the type columns as Enums.

    The thalamic roots have no AIBS row, so their nucleus ID is null.
    """
    cell_info = pl.from_pandas(load_neuron_info(version=version).reset_index())
    return cell_info.rename({"id": "nucleus_id"}).with_columns(
        pl.col("broad_type").cast(pl.Enum(BROAD_TYPE_CATEGORIES)),
        pl.col("cell_type").cast(pl.Enum(CELL_TYPE_CATEGORIES)),
        pl.col("mtype").cast(pl.Enum(MTYPE_CATEGORIES)),
    )


Tables.register(
    "cell_info",
    description=(
        "One row per root in the CAVE view `aibs_cell_info`, plus the thalamic roots "
        "from `proofreading_status_and_strategy`. It holds the `nucleus_id`, the broad "
        "type, cell type "
        "and mtype, `axon_cleaned`, and the soma point in nm (`pt_position_*`) and "
        "in transformed µm (`pt_position_um_*`). Roots that appear twice are dropped."
    ),
    materialize=lambda path: write_parquet_atomic(build_cell_info(), path),
    cache_params=lambda: {"version": VERSION},
)
