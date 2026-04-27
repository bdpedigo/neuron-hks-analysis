import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from caveclient import CAVEclient
from joblib import Parallel, delayed, load
from standard_transform.datasets import minnie_transform_nm

from .utils import project_points_to_mesh

# src/analysis/ -> src/ -> neuron-hks-analysis/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data"
FIG_PATH = PROJECT_ROOT / "figures"
MESH_DATA_PATH = DATA_PATH / "computed_mesh_data"


def param_name(**kwargs):
    return "_".join([f"{k}={v}" for k, v in kwargs.items()])


def param_hash(**kwargs):
    return hash_name(param_name(**kwargs))


def hash_name(name):
    return hashlib.md5(bytes(name.encode("utf-8")), usedforsecurity=False).hexdigest()


def load_signatures(
    # root_id,
    # smooth_n_iter,
    # target_reduction,
    # agg,
    # max_eval,
    # n_scales,
    # t_min,
    # t_max,
    # low_memory,
    **params,
) -> tuple:
    # full_name = f"root_id={root_id}_"
    # full_name += f"smooth_n_iter={smooth_n_iter}_"
    # full_name += f"target_reduction={target_reduction}_"
    # full_name += f"agg={agg}_"
    # full_name += f"max_eval={max_eval}_"
    # full_name += f"n_scales={n_scales}_"
    # full_name += f"t_min={t_min}_"
    # full_name += f"t_max={t_max}_"
    # full_name += f"low_memory={low_memory}"
    # full_id = hash_name(full_name)

    full_id = param_hash(**params)
    signature_out_path = MESH_DATA_PATH / full_id

    signature_out_file = signature_out_path / "signature.npz"

    data = np.load(signature_out_file)
    hks = data["hks"]
    # evals = data["evals"]
    return hks


def decode_neurd(value):
    """
    0 : Synapse onto spine head
    1 : Synapse onto spine neck
    2 : Synapse onto dendritic shaft
    3 : Synapse onto spine but no clear head/neck separation for spine.
    4 : Synapse onto axonal bouton
    5 : Synapse onto axonal structure other than a bouton
    6 : Synapse onto soma
    """
    if value == 0:  # map spine head to spine
        return "spine_head"
    elif value == 1:  # map spine neck to spine
        return "spine_neck"
    elif value == 2:  # map dendritic shaft to shaft
        return "shaft"
    elif value == 3:  # map spine with no clear neck to spine
        return "spine_no_neck"
    elif value == 4:  # map axonal bouton to axon
        return "axon_bouton"
    elif value == 5:  # map axonal structure other than bouton to axon
        return "axon_other"
    elif value == 6:  # map soma to soma
        return "soma"


def simplify_neurd(tag):
    if tag in ["spine_head", "spine_neck", "spine_no_neck"]:
        return "spine"
    elif tag == "soma":
        return "soma"
    elif tag == "shaft":
        return "shaft"
    else:
        return None


def load_synapses(
    root_id,
    mesh,
    client,
    labeled=True,
    distance_threshold=300,
    mapping_column="ctr_pt_position",
):
    ts = client.chunkedgraph.get_root_timestamps(root_id, latest=True)[0]
    post_synapses = client.materialize.query_table(
        "synapses_pni_2",
        filter_equal_dict={"post_pt_root_id": root_id},
        timestamp=ts,
        log_warning=False,
        split_positions=True,
        desired_resolution=[1, 1, 1],
    )
    post_synapses.set_index("id", inplace=True)

    if labeled:
        # client.timestamp = datetime.datetime.now(datetime.timezone.utc)
        # compartment_targets = (
        #     client.materialize.tables.vortex_compartment_targets(
        #         post_pt_root_id=[root_id],
        #         timestamp=ts,
        #     )
        #     .query()
        #     .set_index("target_id")
        #     .rename(columns={"tag": "label"})
        # )
        compartment_targets = (
            client.query_table(
                "vortex_compartment_targets",
                filter_in_dict={"target_id": post_synapses.index},
                timestamp=ts,
            )
            .rename(columns={"tag": "label"})
            .set_index("target_id")
        )
        post_synapses = post_synapses.join(compartment_targets["label"])

        def remap_labels(label):
            if label in ["soma", "shaft", "spine"]:
                return label
            else:
                return np.nan

        post_synapses["label"] = post_synapses["label"].map(remap_labels)

        post_synapses["spine"] = post_synapses["label"].str.contains("spine")
        post_synapses["soma"] = post_synapses["label"].str.contains("soma")
        post_synapses["shaft"] = post_synapses["label"].str.contains("shaft")

        neurd_targets = client.materialize.query_table(
            "synapse_target_structure",
            filter_in_dict={"target_id": post_synapses.index},
            timestamp=ts,
            log_warning=False,
        ).set_index("target_id")
        neurd_targets["tag"] = neurd_targets["value"].map(decode_neurd)
        neurd_targets["label"] = neurd_targets["tag"].map(simplify_neurd)

        post_synapses = post_synapses.join(
            neurd_targets[["tag", "label"]], rsuffix="_neurd"
        )

    synapse_locs = post_synapses[
        [f"{mapping_column}_x", f"{mapping_column}_y", f"{mapping_column}_z"]
    ].values

    indices, distances = project_points_to_mesh(
        synapse_locs, mesh, distance_threshold=distance_threshold, return_distances=True
    )

    post_synapses["mesh_index"] = indices
    post_synapses["distance_to_mesh"] = distances

    post_synapses.query("mesh_index != -1", inplace=True)

    if isinstance(mesh, tuple):
        vertices = mesh[0]
    else:
        vertices = mesh.vertices

    mesh_pts = vertices[post_synapses["mesh_index"]]
    post_synapses["mesh_pt_position_x"] = mesh_pts[:, 0]
    post_synapses["mesh_pt_position_y"] = mesh_pts[:, 1]
    post_synapses["mesh_pt_position_z"] = mesh_pts[:, 2]

    if labeled:
        post_synapses = post_synapses.query("label.notnull()")
    return post_synapses


def save_pyvista_figure(
    plotter: pv.Plotter,
    filename: str,
    out_path: Optional[Path] = FIG_PATH,
    subfolder="",
    formats="common",
    doc_save=False,
    slide_save=False,
    slide_subfolder="",
    scale=None,
    show=False,
):
    if formats == "all":
        formats = ["png", "pdf", "svg", "html"]
    elif formats == "docs":
        formats = ["svg", "html"]
    elif formats == "common":
        formats = ["png", "svg", "html"]

    if not (out_path / subfolder).exists():
        (out_path / subfolder).mkdir(parents=True, exist_ok=True)

    if "png" in formats:
        plotter.screenshot(
            out_path / subfolder / f"{filename}.png",
            return_img=False,
            transparent_background=True,
            scale=scale,
        )
    if "pdf" in formats:
        plotter.save_graphic(out_path / subfolder / f"{filename}.pdf", raster=False)
    if "svg" in formats:
        plotter.save_graphic(out_path / subfolder / f"{filename}.svg", raster=False)
    if "html" in formats:
        plotter.export_html(out_path / subfolder / f"{filename}.html")

    if doc_save:
        out = "::: {.panel-tabset group='format'}\n\n"
        out += "## SVG (static)\n\n"
        out += "![](./images/"
        if subfolder != "":
            out += f"{subfolder}/"
        out += f"{filename}.svg)\n\n"
        out += "## HTML (dynamic)\n\n"
        out += '<iframe width="800" height="600" src="./images/'
        if subfolder != "":
            out += f"{subfolder}/"
        out += f'{filename}.html"></iframe>\n\n'
        out += ":::"
        print(out)

        # copy files to docs images directory

        for fmt in formats:
            shutil.copy(
                out_path / subfolder / f"{filename}.{fmt}",
                out_path.parent.parent / "docs" / "images" / subfolder,
            )

    if slide_save:
        # Template looks like this
        """
        <div>
        <embed src="./images/vortex_neurd/vortex_neurd_sample_meshes_3.svg" width="96%" height="600px" name="vortex_neurd_sample_meshes_3"></embed>

        <a href="./images/vortex_neurd/vortex_neurd_sample_meshes_3.html" target="vortex_neurd_sample_meshes_3">
        <img src="./../../images/icons/search.svg"></img>
        </a>
        </div>
        """
        out = f"""
        <div>
        <embed src="./images/{subfolder}/{filename}.svg" width="96%" height="600px" name="{filename}"></embed>

        <a href="./images/{subfolder}/{filename}.html" target="{filename}">
        <img src="./../../images/icons/search.svg"></img>
        </a>
        </div>
        """
        # remove the leading whitespace
        out = "\n".join([line.lstrip() for line in out.split("\n")])
        print(out)

        SLIDE_PATH = Path(
            f"/Users/ben.pedigo/code/talks/docs/slides/{slide_subfolder}/images"
        )

        for fmt in formats:
            shutil.copy(
                out_path / subfolder / f"{filename}.{fmt}",
                SLIDE_PATH / subfolder,
            )

    if show and "png" in formats:
        from IPython.display import Image, display

        img_path = out_path / subfolder / f"{filename}.png"
        display(Image(filename=img_path))


def save_matplotlib_figure(
    fig: plt.Figure,
    filename: str,
    out_path: Optional[Path] = FIG_PATH,
    subfolder: str = "",
    formats="all",
    doc_save=False,
    bbox_inches="tight",
    slide_save=False,
    slide_subfolder="images",
    pad_inches=0,
    **kwargs,
):
    if formats == "all":
        formats = ["png", "pdf", "svg"]
    elif formats == "common":
        formats = ["png", "svg"]
    elif formats == "docs":
        formats = ["svg"]

    if not (out_path / subfolder).exists():
        (out_path / subfolder).mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(
            out_path / subfolder / f"{filename}.{fmt}",
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            **kwargs,
        )

    if doc_save:
        out = ""
        for fmt in formats:
            out += "![](./images/"
            if subfolder != "":
                out += f"{subfolder}/"
            out += f"{filename}.{fmt})\n\n"
        print(out)

        # copy files to docs images directory

        for fmt in formats:
            shutil.copy(
                out_path / subfolder / f"{filename}.{fmt}",
                out_path.parent.parent / "docs" / "images" / subfolder,
            )

    slide_path = Path("/Users/ben.pedigo/code/talks/docs/slides")
    if slide_subfolder is not None:
        SLIDE_PATH = slide_path / slide_subfolder / "images"
    # slide_path = slide_path / "images"
    if slide_save:
        for fmt in formats:
            shutil.copy(
                out_path / subfolder / f"{filename}.{fmt}",
                SLIDE_PATH / subfolder,
            )


def load_model(model_name):
    path = Path(__file__).parent.parent.parent / "models" / f"{model_name}.pkl"
    return load(path)


def _read_root_synapses(root_id: int, side="post", version=1412) -> pd.DataFrame:
    if side == "post":
        file_path = (
            DATA_PATH / f"column_labeled_post_synapses_{version}" / f"{root_id}.csv.gz"
        )
    elif side == "pre":
        file_path = (
            DATA_PATH / f"column_labeled_pre_synapses_{version}" / f"{root_id}.csv.gz"
        )
    else:
        raise ValueError("side must be 'post' or 'pre'")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path, index_col=0)


def read_synapses(root_ids: int, side="post", version=1412) -> pd.DataFrame:
    if isinstance(root_ids, int):
        root_ids = [root_ids]

    all_synapses = Parallel(n_jobs=-1)(
        delayed(_read_root_synapses)(root_id, side=side, version=version)
        for root_id in root_ids
    )
    all_synapses = pd.concat(all_synapses, axis=0)

    return all_synapses


CELL_TYPE_CATEGORIES = [
    "TH",
    "23P",
    "4P",
    "5P-IT",
    "5P-ET",
    "5P-NP",
    "6P-IT",
    "6P-CT",
    "BC",
    "BPC",
    "MC",
    "NGC",
    "Unk",
]

MTYPE_CATEGORIES = [
    "TH",
    "L2a",
    "L2b",
    "L2c",
    "L3a",
    "L3b",
    "L4a",
    "L4b",
    "L4c",
    "L5ET",
    "L5NP",
    "L5a",
    "L5b",
    "L6short-a",
    "L6short-b",
    "L6tall-a",
    "L6tall-b",
    "L6tall-c",
    "L6wm",
    "DTC",
    "ITC",
    "PTC",
    "STC",
    "Unk",
]

BROAD_TYPE_CATEGORIES = ["thalamic", "excitatory", "inhibitory", "unknown"]

LABEL_CATEGORIES = ["spine", "shaft", "soma", "unknown"]
LABEL_DETAILED_CATEGORIES = [
    "spine",
    "single_spine",
    "multi_spine",
    "shaft",
    "soma",
    "unknown",
]

COMPARTMENT_CATEGORIES = ["axon", "dendrite", "perisoma", "unknown"]

TABLE_CACHE_PATH = DATA_PATH / "table_cache"


def _get_client(version=1412):
    return CAVEclient("minnie65_public", version=version)


def load_neuron_info(version=1412, transform_positions=True, add_thalamic=True):
    query_args = dict(desired_resolution=[1, 1, 1], split_positions=True)

    table_path = DATA_PATH / f"v{version}-aibs_cell_info.csv.gz"
    if table_path.exists():
        cell_info = pd.read_csv(table_path, index_col=0)
    else:
        client = _get_client(version=version)
        cell_info = client.materialize.query_view("aibs_cell_info", **query_args)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        cell_info.to_csv(table_path)

    cell_info.drop_duplicates("pt_root_id", keep=False, inplace=True)
    cell_info.set_index("pt_root_id", inplace=True)
    cell_info["axon_cleaned"] = cell_info["axon_cleaned"] == "t"

    if transform_positions:
        tform = minnie_transform_nm()
        cell_info[["pt_position_um_x", "pt_position_um_y", "pt_position_um_z"]] = (
            tform.apply_dataframe("pt_position", cell_info)
        )

    if add_thalamic:
        proofreading_path = (
            DATA_PATH / f"v{version}-proofreading_status_and_strategy.csv.gz"
        )
        if proofreading_path.exists():
            proofreading_table = pd.read_csv(proofreading_path, index_col=0)
        else:
            client = _get_client(version=version)
            proofreading_table = (
                client.materialize.tables.proofreading_status_and_strategy().query(
                    **query_args
                )
            )
            proofreading_path.parent.mkdir(parents=True, exist_ok=True)
            proofreading_table.to_csv(proofreading_path)

        thalamic_table = proofreading_table.query("status_dendrite=='f'").query(
            "strategy_dendrite=='none'"
        )
        thalamic_roots = thalamic_table["pt_root_id"].unique()

        root_ids = cell_info.index.unique()
        root_ids = np.concatenate([thalamic_roots, root_ids])
        root_ids = np.unique(root_ids)

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
