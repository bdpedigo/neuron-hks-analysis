from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from caveclient import CAVEclient
from pytz import utc

from .io import VERSION, cache_table

TABLE_WEIGHTS = {
    "vortex_compartment_targets": 1,
    "bdp_synapse_compartment_labels": 25,
    "bdp_point_compartment_labels": 10,
}


def get_validation_ids(client, version=VERSION) -> np.ndarray:
    from .io import TABLE_PATH

    validation_path = TABLE_PATH / "general" / "validation_ids_used.csv"
    validation_ids = pd.read_csv(validation_path)["nucleus_id"].unique()
    validation_nuc_table = client.materialize.query_view(
        "nucleus_detection_lookup_v1",
        filter_in_dict=dict(id=validation_ids),
        materialization_version=version,
    )
    validation_root_ids = validation_nuc_table["pt_root_id"].unique()
    return validation_root_ids


@cache_table("label_table_threshold={threshold}")
def _fetch_label_table(version=VERSION, threshold=None, live=False) -> pl.DataFrame:
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

        if table_name == "vortex_compartment_targets" and threshold is not None:
            counts = table.groupby(f"pt_root_id_{version}").size()
            keep_index = counts[counts >= threshold].index
            table = table[table[f"pt_root_id_{version}"].isin(keep_index)]

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


def get_label_table(
    version=VERSION,
    threshold=None,
    live=False,
) -> pl.DataFrame:
    if live:
        return _fetch_label_table.__wrapped__(
            version=version, threshold=threshold, live=True
        )
    return _fetch_label_table(version=version, threshold=threshold, live=False)


def get_synapse_sizes(
    synapse_ids, client: Union[str, CAVEclient] = "minnie65_phase3_v1"
):
    if isinstance(client, str):
        client = CAVEclient(client)
    syn_table_name = client.info.get_datastack_info()["synapse_table"]
    syn_table = client.materialize.query_table(
        syn_table_name,
        filter_in_dict=dict(id=synapse_ids),
        split_positions=True,
    )
    syn_sizes = syn_table[["size", "id"]].set_index("id")["size"].astype(float)

    table_info = client.materialize.get_table_metadata(syn_table_name)
    voxel_res = np.array(table_info["voxel_resolution"])
    voxel_vol = np.prod(voxel_res)

    syn_sizes = syn_sizes * voxel_vol

    return syn_sizes
