"""The H01 outputs of the paper run, for comparison with the deployment.

The paper's Fig 6 used the cloud-mesh run `absolute-solo-yak` (meshmash about
`initial-submission`) and the model `auburn-elk-detour`. The run wrote one file per
root and per object to `gs://bdp-ssa/h01_c3_flat/absolute-solo-yak/`:

- `features/{root}.npz`: the per-domain `hks_*` columns (log, float16), indexed
  -1, 0, 1, ..., and the per-vertex domain labels (-1 for no domain).
- `graphs/{root}.npz`: the per-domain node table (centroid, area, component size)
  and the domain edges.
- `post-synapse-mappings/{root}.npz`: synapse id to mesh vertex.

All 160 H01 label roots are in the run. The paper's H01 `distance_to_nucleus` is
all null in the graphs, because the run looked for a `nucleus_detection` table. The
Fig 6 script recomputed it from the domain centroids and the v1054 `nucleus` table,
as `paper_distance_to_nucleus` does here.

Kind: cloud, read through a local copy at `data/paper_h01/{object}/{root}.npz`. The
run will not be repeated, so the local copy is frozen once it is complete.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from joblib import load
from upath import UPath

from meshmash.io import read_condensed_features, read_condensed_graph, read_id_to_mesh_map

from analysis.io import DATA_PATH, PROJECT_ROOT

PAPER_H01_CLOUD_PATH = UPath("gs://bdp-ssa/h01_c3_flat/absolute-solo-yak")
PAPER_H01_PATH = DATA_PATH / "paper_h01"
PAPER_OBJECTS = ("features", "graphs", "post-synapse-mappings")
# The header files that the meshmash readers look for next to each object.
PAPER_HEADERS = {
    "features": ("header.txt",),
    "graphs": ("nodes_header.txt", "edges_header.txt"),
}

MESHREP_PATH = PROJECT_ROOT.parent / "meshrep"
PAPER_MODEL_PATH = MESHREP_PATH / "experiments" / "auburn-elk-detour" / "synapse_hks_model.joblib"
# NOTE: `live_live_query` of the H01 `nucleus` table at the v1054 timestamp, written
# by `meshrep/clean_paper_code/retrain_h01.py`. The Fig 6 script queried the same
# table at version 1054. Positions are in nm.
PAPER_NUCLEI_PATH = PROJECT_ROOT.parent / "cloud-mesh" / "temp_data" / "h01_nucleus_table.parquet"


def _local(obj: str, root_id: int) -> Path:
    """The local copy of `obj` for `root_id`, copied from the cloud on first use."""
    folder = PAPER_H01_PATH / obj
    folder.mkdir(parents=True, exist_ok=True)
    for header in PAPER_HEADERS.get(obj, ()):
        if not (folder / header).exists():
            (folder / header).write_bytes((PAPER_H01_CLOUD_PATH / obj / header).read_bytes())
    path = folder / f"{root_id}.npz"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes((PAPER_H01_CLOUD_PATH / obj / f"{root_id}.npz").read_bytes())
        tmp.rename(path)
    return path


@dataclass
class PaperRoot:
    """The paper run's outputs for one root."""

    root_id: int
    # Per-domain `hks_*` columns, indexed -1 (no domain), 0, 1, ...
    features: pd.DataFrame
    # Per-vertex domain label of the raw mesh, -1 for no domain.
    labels: np.ndarray
    # Per-domain node table (`x`, `y`, `z` in nm, `area`, `n_vertices`, ...),
    # indexed 0, 1, ...
    nodes: pd.DataFrame
    # Synapse id to mesh vertex.
    synapse_vertex: pd.Series


def load_paper_root(root_id: int) -> PaperRoot:
    features, labels = read_condensed_features(str(_local("features", root_id)))
    nodes, _ = read_condensed_graph(str(_local("graphs", root_id)))
    mapping = read_id_to_mesh_map(str(_local("post-synapse-mappings", root_id)))
    return PaperRoot(
        root_id=root_id,
        features=features,
        labels=labels,
        nodes=nodes,
        synapse_vertex=pd.Series(mapping[:, 1], index=pd.Index(mapping[:, 0], name="synapse_id")),
    )


@lru_cache
def paper_nuclei() -> pd.DataFrame:
    """The v1054 H01 `nucleus` rows: `root_id`, `nucleus_id`, `nucleus_x/y/z` (nm).
    A root can have several rows."""
    nuclei = pd.read_parquet(PAPER_NUCLEI_PATH)
    return nuclei.rename(
        columns={
            "pt_root_id": "root_id",
            "id": "nucleus_id",
            "pt_position_x": "nucleus_x",
            "pt_position_y": "nucleus_y",
            "pt_position_z": "nucleus_z",
        }
    )[["root_id", "nucleus_id", "nucleus_x", "nucleus_y", "nucleus_z"]]


def paper_distance_to_nucleus(root: PaperRoot) -> pd.Series:
    """`distance_to_nucleus` (nm) of each domain, by the rule of the Fig 6 script
    (`meshrep/sandbox/h01_validation.py`): the distance from the domain centroid to
    the root's one nucleus point. Null for a root with no or several nuclei, which
    the script skipped."""
    nuc = paper_nuclei().query("root_id == @root.root_id")
    if len(nuc) != 1:
        return pd.Series(np.nan, index=root.nodes.index, name="distance_to_nucleus")
    point = nuc[["nucleus_x", "nucleus_y", "nucleus_z"]].to_numpy()[0]
    dist = np.linalg.norm(root.nodes[["x", "y", "z"]].to_numpy() - point, axis=1)
    return pd.Series(dist, index=root.nodes.index, name="distance_to_nucleus")


@lru_cache
def load_paper_model():
    """The paper's random forest: `hks_0`..`hks_31` + `distance_to_nucleus`."""
    return load(PAPER_MODEL_PATH)


def paper_domain_posteriors(root: PaperRoot, distance: bool = True) -> pd.DataFrame:
    """The paper model's posteriors for each domain of the paper run:
    `posterior_<class>` and `pred_label`, indexed by domain.

    With `distance`, as the Fig 6 script scored them. Without, with a null
    `distance_to_nucleus`, as the cloud run scored them: its saved predictions
    (`auburn-elk-detour-synapse_hks_model/post-synapse-predictions`) match these.
    """
    model = load_paper_model()
    distances = paper_distance_to_nucleus(root)
    if not distance:
        distances[:] = np.nan
    features = root.features.loc[root.nodes.index].join(distances)
    probabilities = model.predict_proba(features[list(model.feature_names_in_)])
    out = pd.DataFrame(
        probabilities, index=root.nodes.index, columns=[f"posterior_{c}" for c in model.classes_]
    )
    out["pred_label"] = model.classes_[probabilities.argmax(axis=1)]
    return out


def paper_vertex_posteriors(root: PaperRoot, distance: bool = True) -> pl.DataFrame:
    """`paper_domain_posteriors` on each vertex of the raw mesh, null for a vertex
    in no domain of the paper run."""
    domains = paper_domain_posteriors(root, distance=distance)
    return pl.from_pandas(domains.reindex(root.labels).reset_index(drop=True))
