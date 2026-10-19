"""P13 inputs: the unitary-spine labels (S11) and the cells they are on.

The labels are points in 7 neuroglancer states, exported once to
`data/tables/unitary_spines/states/` by
`scripts/unitary_spines/export_unitary_states.py`. Each state holds the putative
multi-input spines of 10 cells of one excitatory type, one point per spine,
tagged by hand (`true_multi`, `y`, `merge`, ...).

A point's description is `{root}_{component_id}`: the v1412 root and the spine
id of the old pipeline. Neither carries over to the new pipeline, so the cells
map forward by nucleus ID, and the spines map forward through their synapse IDs.
"""

import json

import pandas as pd
import polars as pl
from caveclient import CAVEclient
from nglui.parser import StateParser

from ..io import TABLE_PATH, VERSION
from .catalog import Tables, write_parquet_atomic
from .labels import fetch_nucleus_roots
from .sources import DATASTACK

UNITARY_TABLE_PATH = TABLE_PATH / "unitary_spines"
LABEL_VERSION = 1412
LABEL_CELL_INFO_PATH = TABLE_PATH / f"v{LABEL_VERSION}" / "aibs_cell_info.csv.gz"
# The old pipeline's synapse -> spine map (meshrep/tools/pull_morphometry.py), a
# frozen local copy. `index` is the synapse ID.
OLD_COMPONENT_PATH = (
    TABLE_PATH.parents[2] / "meshrep" / "data" / "synapse_component_deltalake"
)

TAG_CATEGORIES = ["true_multi", "y", "not_spine", "merge", "tip", "overseg", "other", "review"]


def read_unitary_annotations() -> pl.DataFrame:
    """One row per labeled spine: the state, the old spine key, and its tags."""
    states = pd.read_csv(UNITARY_TABLE_PATH / "states.csv")
    frames = []
    for state in states.itertuples():
        state_json = json.loads(
            (UNITARY_TABLE_PATH / "states" / f"{state.state_id}.json").read_text()
        )
        df = StateParser(state_json).annotation_dataframe(expand_tags=True)
        df = df.query("layer == 'synapse_points'").copy()
        df["state_id"] = state.state_id
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    key = df["description"].str.split("_", expand=True).astype("int64")
    return pl.DataFrame(
        {
            "state_id": df["state_id"].astype("int64"),
            "anno_id": df["anno_id"],
            "label_root_id": key[0],
            "component_id": key[1],
            **{tag: df[tag].astype(bool) for tag in TAG_CATEGORIES},
        }
    )


def fetch_unitary_label_cells(client, version=VERSION) -> pl.DataFrame:
    """The label cells: v1412 root -> nucleus ID -> root ID at `version`."""
    label_roots = read_unitary_annotations()["label_root_id"].unique().to_list()
    cell_info = pd.read_csv(LABEL_CELL_INFO_PATH, usecols=["id", "pt_root_id", "cell_type"])
    cell_info = cell_info[cell_info["pt_root_id"].isin(label_roots)]
    assert cell_info["pt_root_id"].is_unique, "a label root has more than one nucleus"
    assert len(cell_info) == len(label_roots), "a label root has no nucleus at v1412"
    cells = pl.from_pandas(
        cell_info.rename(
            columns={"id": "nucleus_id", "pt_root_id": "label_root_id"}
        )
    )
    roots = fetch_nucleus_roots(client, cells["nucleus_id"].to_list(), version=version)
    return cells.join(roots, on="nucleus_id", how="left").sort("label_root_id")


Tables.register(
    "unitary_label_cells",
    materialize=lambda path: write_parquet_atomic(
        fetch_unitary_label_cells(CAVEclient(DATASTACK), version=VERSION), path
    ),
    cache_params={"version": VERSION, "label_version": LABEL_VERSION},
)


def build_unitary_label_synapses() -> pl.DataFrame:
    """The synapse IDs on each labeled spine, from the old synapse -> spine map."""
    annotations = read_unitary_annotations()
    keys = annotations.select("label_root_id", "component_id").unique()
    return (
        pl.scan_delta(str(OLD_COMPONENT_PATH))
        .filter(pl.col("post_pt_root_id").is_in(keys["label_root_id"].implode()))
        .select(
            pl.col("index").alias("synapse_id"),
            pl.col("post_pt_root_id").alias("label_root_id"),
            "component_id",
        )
        .collect()
        .join(keys, on=["label_root_id", "component_id"], how="semi")
        .sort("label_root_id", "component_id", "synapse_id")
    )


Tables.register(
    "unitary_label_synapses",
    materialize=lambda path: write_parquet_atomic(build_unitary_label_synapses(), path),
    cache_params={"label_version": LABEL_VERSION},
)
