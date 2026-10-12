from typing import Union

import numpy as np
from caveclient import CAVEclient


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
