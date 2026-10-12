"""P1: the training label table, and the validation cells it excludes."""

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
from caveclient import CAVEclient
from pytz import utc

from ..io import TABLE_PATH, VERSION
from .catalog import Tables, write_parquet_atomic
from .sources import DATASTACK

TABLE_WEIGHTS = {
    "vortex_compartment_targets": 1,
    "bdp_synapse_compartment_labels": 25,
    "bdp_point_compartment_labels": 10,
}


def fetch_validation_cells(client, version=VERSION) -> pl.DataFrame:
    """The validation cells (S9 nucleus IDs) with their root IDs at `version`."""
    validation_path = TABLE_PATH / "validation" / "validation_ids_used.csv"
    nucleus_ids = pd.read_csv(validation_path)["nucleus_id"].unique()
    nuc_table = client.materialize.query_view(
        "nucleus_detection_lookup_v1",
        filter_in_dict=dict(id=nucleus_ids),
        materialization_version=version,
    )
    return pl.from_pandas(
        nuc_table[["id", "pt_root_id"]].rename(
            columns={"id": "nucleus_id", "pt_root_id": "root_id"}
        )
    )


Tables.register(
    "validation_cells",
    materialize=lambda path: write_parquet_atomic(
        fetch_validation_cells(CAVEclient(DATASTACK), version=VERSION), path
    ),
    cache_params={"version": VERSION},
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
    materialize=lambda path: write_parquet_atomic(
        fetch_label_table(version=VERSION, live=False), path
    ),
    cache_params={"version": VERSION},
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
