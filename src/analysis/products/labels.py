"""P1: the training label table, and the validation cells it excludes."""

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
from caveclient import CAVEclient
from pytz import utc

from ..io import TABLE_PATH
from ..sources import DATASTACK, VERSION
from .catalog import Tables, write_parquet_atomic

TABLE_WEIGHTS = {
    "vortex_compartment_targets": 1,
    "bdp_synapse_compartment_labels": 25,
    "bdp_point_compartment_labels": 10,
}


def fetch_nucleus_roots(client, nucleus_ids, version=VERSION) -> pl.DataFrame:
    """The root ID of each nucleus at `version`, as `nucleus_id`, `root_id`."""
    nuc_table = client.materialize.query_view(
        "nucleus_detection_lookup_v1",
        filter_in_dict=dict(id=list(nucleus_ids)),
        materialization_version=version,
    )
    return pl.from_pandas(
        nuc_table[["id", "pt_root_id"]].rename(
            columns={"id": "nucleus_id", "pt_root_id": "root_id"}
        )
    )


def fetch_validation_cells(client, version=VERSION) -> pl.DataFrame:
    """The validation cells (S9 nucleus IDs) with their root IDs at `version`."""
    validation_path = TABLE_PATH / "validation" / "validation_ids_used.csv"
    nucleus_ids = pd.read_csv(validation_path)["nucleus_id"].unique()
    return fetch_nucleus_roots(client, nucleus_ids, version=version)


Tables.register(
    "validation_cells",
    description=(
        "One row per held-out validation cell: the nucleus ID from "
        "`validation_ids_used.csv` and its root ID at the current version."
    ),
    materialize=lambda path: write_parquet_atomic(
        fetch_validation_cells(CAVEclient(DATASTACK), version=VERSION), path
    ),
    cache_params=lambda: {"version": VERSION},
)


VALIDATION_LABEL_FILES = {
    "exc": "exc_sample_synapses_labeled.csv",
    "inh": "inh_sample_synapses_labeled.csv",
}
VALIDATION_LABEL_COLUMNS = [
    "synapse_id",
    "post_nucleus_id",
    "post_cell_type",
    "post_mtype",
    "post_cell_type_depth",
    "visual_area",
    "tag",
]


def read_validation_label_files() -> pl.DataFrame:
    """The S9 validation labels, without the root IDs stored in the files."""
    frames = []
    for label_set, file_name in VALIDATION_LABEL_FILES.items():
        frame = pl.read_csv(TABLE_PATH / "validation" / file_name)
        frames.append(
            frame.select(
                pl.lit(label_set).alias("label_set"),
                *[col for col in VALIDATION_LABEL_COLUMNS if col in frame.columns],
            )
        )
    return pl.concat(frames, how="diagonal")


def fetch_validation_labels(
    client, validation_cells: pl.DataFrame, version=VERSION
) -> pl.DataFrame:
    """The S9 validation labels, with the synapse and cell root IDs at `version`.

    `root_id` is the cell (from the label's nucleus), and `post_pt_root_id` is
    the post root of the synapse. They differ when an edit after the labeling
    moved the synapse off the cell, and `on_cell` is false for those rows.
    """
    labels = read_validation_label_files()
    synapses = client.materialize.query_table(
        "synapses_pni_2",
        filter_in_dict=dict(id=labels["synapse_id"].to_list()),
        materialization_version=version,
        desired_resolution=[1, 1, 1],
        split_positions=True,
    )
    synapses = pl.from_pandas(
        synapses[
            [
                "id",
                "pre_pt_root_id",
                "post_pt_root_id",
                "size",
                "ctr_pt_position_x",
                "ctr_pt_position_y",
                "ctr_pt_position_z",
            ]
        ]
    ).rename({"id": "synapse_id"})
    cells = validation_cells.select(
        pl.col("nucleus_id").alias("post_nucleus_id"), "root_id"
    )
    return (
        labels.join(cells, on="post_nucleus_id", how="left")
        .join(synapses, on="synapse_id", how="left")
        .with_columns(
            (pl.col("post_pt_root_id") == pl.col("root_id"))
            .fill_null(False)
            .alias("on_cell")
        )
    )


Tables.register(
    "validation_labels",
    description=(
        "One row per hand-labeled validation synapse, from the exported `exc` and "
        "`inh` label files, joined to the CAVE synapse table. `root_id` is the "
        "labeled cell. `on_cell` is false when the synapse is no longer on that cell."
    ),
    materialize=lambda path: write_parquet_atomic(
        fetch_validation_labels(
            CAVEclient(DATASTACK),
            Tables["validation_cells"].collect(),
            version=VERSION,
        ),
        path,
    ),
    cache_params=lambda: {
        "version": VERSION,
        "validation_cells": Tables["validation_cells"].cache_key,
    },
)


def get_validation_ids(client, version=VERSION) -> np.ndarray:
    """The root IDs of the validation cells at `version`.

    The default (`VERSION`) comes from the registry cache. Another version is
    fetched from CAVE and not cached.
    """
    if version == VERSION:
        validation_cells = Tables["validation_cells"].collect()
    else:
        validation_cells = fetch_validation_cells(client, version=version)
    return validation_cells["root_id"].unique().to_numpy()


def fetch_label_table(version=VERSION, live=False) -> pl.DataFrame:
    client = CAVEclient("minnie65_phase3_v1")

    # drop_cols = ["valid", "superceded_id"]
    drop_cols = []
    table_names = [
        "vortex_compartment_targets",
        "bdp_synapse_compartment_labels",
        "bdp_point_compartment_labels",
    ]
    root_timestamp = client.materialize.get_timestamp(version)
    annotation_timestamp = datetime.now(tz=utc) if live else root_timestamp
    keep_cols = [
        "id",
        "target_id",
        "tag",
        "pt_supervoxel_id",
        # "pt_root_id_now",
        f"pt_root_id_{version}",
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
        "table_name",
        "created",
    ]

    tables = []
    for table_name in table_names:
        print(f"Loading {table_name}...")
        table = client.materialize.live_live_query(
            table_name,
            timestamp=annotation_timestamp,
            desired_resolution=[1, 1, 1],
            split_positions=True,
            log_warning=False,
        )
        table.drop(columns=drop_cols, errors="ignore", inplace=True)
        if "target_id" in table.columns:
            base_table = client.materialize.live_live_query(
                "synapses_pni_2",
                desired_resolution=[1, 1, 1],
                split_positions=True,
                filter_in_dict=dict(
                    synapses_pni_2=dict(
                        id=table["target_id"].unique().astype(int).tolist()
                    )
                ),
                timestamp=root_timestamp,
            ).drop(columns=drop_cols + ["created", "deleted"], errors="ignore")
            table = (
                table.set_index("target_id")
                .join(base_table.set_index("id"), how="left")
                .reset_index()
            )
            table = table.drop_duplicates(subset=["target_id", "tag"], keep="last")
        table.query("tag.isin(['soma', 'shaft', 'spine', 'soma_spine'])", inplace=True)
        table["tag"] = table["tag"].replace({"soma_spine": "spine"})
        table["table_name"] = table_name

        if "post_pt_supervoxel_id" in table.columns:
            table = table.rename(
                columns={
                    "post_pt_root_id": f"pt_root_id_{version}",
                    "post_pt_supervoxel_id": "pt_supervoxel_id",
                    "ctr_pt_position_x": "pt_position_x",
                    "ctr_pt_position_y": "pt_position_y",
                    "ctr_pt_position_z": "pt_position_z",
                }
            )

        # print(f"Getting supervoxel ids for {table_name}...")
        # table[f"pt_root_id_{version}"] = client.chunkedgraph.get_roots(
        #     table["pt_supervoxel_id"], timestamp=root_timestamp
        # )
        # print()

        table["table_name"] = table_name
        tables.append(table[table.columns.intersection(keep_cols)])

    label_table = pd.concat(tables, ignore_index=True)
    label_table.rename_axis(index="label_id", inplace=True)
    label_table["target_id"] = label_table["target_id"].fillna(-1).astype(int)
    # label_table["pt_root_id_now"] = label_table["pt_root_id_now"].fillna(-1).astype(int)
    label_table["sample_weight"] = (
        label_table["table_name"].map(TABLE_WEIGHTS).fillna(1.0)
    )

    validation_root_ids = get_validation_ids(client, version)
    validation_root_ids
    label_table = label_table.query(f"~pt_root_id_{version}.isin(@validation_root_ids)")

    synapse_label_table = label_table.query("target_id != -1")
    synapse_label_table = synapse_label_table.drop_duplicates(
        ["target_id", "tag"], keep="last"
    )
    synapse_label_table = synapse_label_table.sort_values(
        "table_name",
        key=lambda x: x.map(
            {
                "bdp_synapse_compartment_labels": 0,
                "bdp_point_compartment_labels": 1,
                "vortex_compartment_targets": 2,
            }
        ),
    ).drop_duplicates("target_id", keep="first")

    non_synapse_label_table = label_table.query("target_id == -1")

    label_table = pd.concat(
        [synapse_label_table, non_synapse_label_table], ignore_index=True
    )

    label_counts_by_root = label_table.groupby(f"pt_root_id_{version}").size()
    label_table[f"root_id_{version}_count"] = label_table[f"pt_root_id_{version}"].map(
        label_counts_by_root
    )

    return pl.from_pandas(label_table)


Tables.register(
    "labels",
    description=(
        "One row per manual compartment label from the three CAVE label tables, with "
        "a sample weight per table. `target_id` is the labeled synapse, or -1 for a "
        "point label. The validation cells are excluded."
    ),
    materialize=lambda path: write_parquet_atomic(
        fetch_label_table(version=VERSION, live=False), path
    ),
    cache_params=lambda: {"version": VERSION},
)
Tables.register(
    "synapse_labels",
    lambda: Tables["labels"]
    .lazy()
    .filter(pl.col("target_id") != -1)
    .rename({"target_id": "synapse_id", f"pt_root_id_{VERSION}": "root_id"}),
)


def get_label_table(version=VERSION, live=False) -> pl.DataFrame:
    """The label table at `version`.

    The default (`VERSION`, not live) comes from the registry cache. A live table,
    or a table at another version, is fetched from CAVE and not cached.
    """
    if not live and version == VERSION:
        return Tables["labels"].collect()
    return fetch_label_table(version=version, live=live)
