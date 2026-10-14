"""Whole-root data: one root's mesh, domains, features, deployment predictions and
synapses, as an ossify `Cell`.

The other products hold only labeled synapses. A view of a whole root needs every
domain and every vertex, so this module reads the deployment tables for one root
at a time. It has no cache; `root_cache.cached` is an optional wrapper.

Layers of the cell:

- `mesh`: the raw CloudVolume mesh, the one that `mesh_index` in the deployment
  tables indexes. Each vertex links to its domain in `domains`.
- `domains`: one point per domain at its centroid. Features: the `domain_features`
  columns, `distance_to_nucleus`, and the deployment `posterior_<class>`,
  `posterior_entropy` and `pred_label` (null where the deployment did not run
  the prediction step on the root). A vertex in no domain links to the
  `NO_DOMAIN` row, whose features are null, because an ossify link cannot be
  negative.
- `synapses`: the synapses that map to the mesh, each linked to its vertex.
  Feature `label`, the manual label, null for an unlabeled synapse.
- `unmapped_synapses`: the synapses that map to no vertex (beyond the mapping's
  distance threshold), with no link.
- `nuclei`: the nucleus points of the root.
"""

import json
from functools import lru_cache
from typing import Optional

import numpy as np
import polars as pl
from caveclient import CAVEclient
from ossify import Cell, Link

from .catalog import Tables
from .deployment import delta_table, latest_attempts, scan_roots
from .features import add_distance_to_nucleus
from .sources import (
    DATASTACK,
    H01_AS_OF,
    H01_DATASTACK,
    H01_SYNAPSES_PATH,
    MICRONS_AS_OF,
    MICRONS_SYNAPSE_RESOLUTION,
    MICRONS_SYNAPSES_PATH,
    STEP_PINS,
    deployment_path,
)

# The pins and local tables of each datastack.
# - `synapse_resolution`: nm per unit of the synapse positions in `synapses_path`.
# - `label_tables`: the local tables with manual labels, each with its label
#   column. A table's `root_id` is the cell the label belongs to.
# - `held_out_cohort`: the `cohort` of `cells_table` that model work must not look
#   at, or None.
ROOT_SOURCES = {
    H01_DATASTACK: dict(
        as_of=H01_AS_OF,
        synapses_path=H01_SYNAPSES_PATH,
        # NOTE: the snapshot stores positions in nm.
        synapse_resolution=(1.0, 1.0, 1.0),
        nuclei_table="h01_root_nuclei",
        label_tables={"h01_labels": "label"},
        cells_table="h01_cells",
        held_out_cohort="validation",
    ),
    DATASTACK: dict(
        as_of=MICRONS_AS_OF,
        synapses_path=MICRONS_SYNAPSES_PATH,
        synapse_resolution=MICRONS_SYNAPSE_RESOLUTION,
        nuclei_table="root_nuclei",
        # The training labels and the labels of the 132 validation cells.
        label_tables={"synapse_labels": "tag", "validation_labels": "tag"},
        held_out_cohort=None,
    ),
}
LABELS = ("spine", "shaft", "soma")
ROOT_PRODUCTS = ("vertex_domains", "domain_features", "domain_predictions", "synapse_mapping")

# The `domains` row for a vertex in no domain. Above any real domain id.
NO_DOMAIN = np.iinfo(np.int32).max
DOMAIN_POSITION = ["domain_x", "domain_y", "domain_z"]
POSITION = ["x", "y", "z"]


class HeldOutRootError(ValueError):
    pass


def check_not_held_out(root_id: int, datastack: str, allow_held_out: bool = False) -> None:
    """Raise if `root_id` is in the held-out cohort of `datastack`, unless allowed."""
    source = ROOT_SOURCES[datastack]
    if allow_held_out or source.get("held_out_cohort") is None:
        return
    cohort = (
        Tables[source["cells_table"]]
        .lazy()
        .filter(pl.col("root_id") == root_id)
        .select("cohort")
        .collect()["cohort"]
    )
    if source["held_out_cohort"] in cohort.to_list():
        raise HeldOutRootError(
            f"root {root_id} is in the held-out {source['held_out_cohort']!r} cohort of "
            f"{datastack}; pass allow_held_out=True to load it anyway"
        )


@lru_cache
def mesh_get_kwargs(datastack: str) -> dict:
    """The `cv.mesh.get` kwargs that the deployment used for the pinned tables.

    NOTE: read from the deployment's own record of the run, so the mesh here has
    the vertices that `mesh_index` refers to.
    """
    identity = deployment_path("vertex_domains", datastack) / "_meshmash" / "identity.json"
    return json.loads(identity.read_text())["inputs"]["mesh"]["mesh_get_kwargs"]


def root_pins(datastack: str) -> dict:
    """Everything that decides the cell of a root, for a cache key."""
    source = ROOT_SOURCES[datastack]
    return {
        "datastack": datastack,
        "as_of": source["as_of"].isoformat(),
        **{product: "/".join(STEP_PINS[product]) for product in ROOT_PRODUCTS},
        "mesh_get_kwargs": mesh_get_kwargs(datastack),
        "nuclei": _cache_key(source["nuclei_table"]),
        "labels": {table: _cache_key(table) for table in source["label_tables"]},
    }


def _cache_key(name: str) -> str:
    # A view has no cache key of its own; its inputs are pinned elsewhere.
    return Tables[name].cache_key or name


@lru_cache
def _cloudvolume(datastack: str):
    return CAVEclient(datastack).info.segmentation_cloudvolume(progress=False)


def fetch_mesh(root_id: int, datastack: str, n_tries: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The raw mesh of `root_id`, as the deployment fetched it."""
    kwargs = mesh_get_kwargs(datastack)
    for attempt in range(n_tries):
        try:
            mesh = _cloudvolume(datastack).mesh.get(root_id, **kwargs)[root_id]
            break
        except Exception as error:
            if attempt == n_tries - 1:
                raise RuntimeError(
                    f"mesh fetch failed for root {root_id} from "
                    f"{_cloudvolume(datastack).cloudpath}"
                ) from error
            print(f"mesh fetch for root {root_id} failed ({error!r}), trying again")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def fetch_root_tables(root_id: int, datastack: str) -> dict[str, pl.DataFrame]:
    """The rows of `root_id` in each of `ROOT_PRODUCTS`, at the pinned `as_of`."""
    as_of = ROOT_SOURCES[datastack]["as_of"]
    scans = latest_attempts(
        [
            scan_roots(deployment_path(product, datastack), [root_id], as_of=as_of)
            for product in ROOT_PRODUCTS
        ]
    )
    return dict(zip(ROOT_PRODUCTS, pl.collect_all([lf.drop("root_id") for lf in scans])))


def fetch_root_synapses(root_id: int, datastack: str) -> pl.DataFrame:
    """`synapse_id` and the synapse position (nm) of each post synapse on the root,
    the rows that the deployment's `synapse_mapping` mapped."""
    source = ROOT_SOURCES[datastack]
    return (
        pl.scan_delta(delta_table(str(source["synapses_path"])))
        .filter(pl.col("post_pt_root_id") == root_id)
        .select(
            pl.col("id").alias("synapse_id"),
            *[
                (pl.col(f"ctr_pt_position_{c}").cast(pl.Float64) * resolution).alias(c)
                for c, resolution in zip("xyz", source["synapse_resolution"])
            ],
        )
        .collect()
    )


def fetch_root_labels(root_id: int, datastack: str) -> pl.DataFrame:
    """`synapse_id` and `label` (spine, shaft or soma) of the labeled synapses of
    the root, from every label table of `datastack`."""
    frames = [
        Tables[table]
        .lazy()
        .filter(pl.col("root_id") == root_id)
        .select("synapse_id", pl.col(column).alias("label"))
        .collect()
        for table, column in ROOT_SOURCES[datastack]["label_tables"].items()
    ]
    return (
        pl.concat(frames)
        .filter(pl.col("label").is_in(LABELS))
        .unique("synapse_id", keep="first", maintain_order=True)
    )


def _vertex_domain(vertex_domains: pl.DataFrame, n_vertices: int) -> np.ndarray:
    """Each vertex's domain id, `NO_DOMAIN` for none."""
    mesh_index = vertex_domains["mesh_index"].to_numpy()
    # The deployment writes one row per mesh vertex. Another count means another
    # mesh, for example after a re-mesh of the root.
    if len(mesh_index) != n_vertices or np.unique(mesh_index).size != n_vertices:
        raise ValueError(
            f"vertex_domains has {len(mesh_index)} rows for a mesh of {n_vertices} "
            "vertices; the mesh is not the one the deployment ran on"
        )
    domains = np.full(n_vertices, NO_DOMAIN, dtype=np.int64)
    domains[mesh_index] = vertex_domains["domain_id"].fill_null(NO_DOMAIN).to_numpy()
    return domains


def build_root_cell(
    root_id: int,
    mesh: tuple[np.ndarray, np.ndarray],
    tables: dict[str, pl.DataFrame],
    synapses: pl.DataFrame,
    nuclei: pl.DataFrame,
    labels: pl.DataFrame,
    meta: Optional[dict] = None,
) -> Cell:
    """The cell of `root_id` from already fetched parts (see the module docstring)."""
    vertices, faces = mesh
    vertex_domain = _vertex_domain(tables["vertex_domains"], len(vertices))

    # NOTE: as in `h01_model_inputs`: the deployment computes `distance_to_nucleus`
    # in its prediction step, not in `domain_features`.
    domains = (
        add_distance_to_nucleus(
            tables["domain_features"].lazy().with_columns(root_id=pl.lit(root_id, pl.Int64)),
            nuclei.lazy(),
        )
        .drop("root_id")
        .collect()
        .join(tables["domain_predictions"], on="domain_id", how="left")
    )
    sentinel = pl.DataFrame({"domain_id": [NO_DOMAIN]}).cast({"domain_id": domains["domain_id"].dtype})
    domains = pl.concat([domains, sentinel], how="diagonal_relaxed")
    domains_pd = domains.to_pandas().set_index("domain_id")
    domains_pd[DOMAIN_POSITION] = domains_pd[DOMAIN_POSITION].fillna(0.0)

    mapping = tables["synapse_mapping"].select("synapse_id", "mesh_index")
    synapses = synapses.join(labels.select("synapse_id", "label"), on="synapse_id", how="left").join(
        mapping, on="synapse_id", how="left"
    )
    mapped = synapses.filter(pl.col("mesh_index").is_not_null())
    unmapped = synapses.filter(pl.col("mesh_index").is_null())

    cell = Cell(name=root_id, meta={"root_id": root_id, **(meta or {})})
    cell.add_point_annotations(
        "domains",
        vertices=domains_pd[DOMAIN_POSITION],
        spatial_columns=DOMAIN_POSITION,
        features=domains_pd.drop(columns=DOMAIN_POSITION),
    )
    cell.add_mesh(vertices, faces, linkage=Link(mapping=vertex_domain, target="domains"))
    for name, frame in [("synapses", mapped), ("unmapped_synapses", unmapped)]:
        frame_pd = frame.to_pandas().set_index("synapse_id")
        cell.add_point_annotations(
            name,
            vertices=frame_pd[POSITION],
            spatial_columns=POSITION,
            features=frame_pd[["label"]],
            linkage=Link(
                mapping=frame_pd["mesh_index"].to_numpy().astype(np.int64),
                target="mesh",
                map_value_is_index=False,
            )
            if name == "synapses"
            else None,
        )
    nuclei_pd = nuclei.select("nucleus_id", "nucleus_x", "nucleus_y", "nucleus_z").to_pandas()
    cell.add_point_annotations(
        "nuclei",
        vertices=nuclei_pd.set_index("nucleus_id"),
        spatial_columns=["nucleus_x", "nucleus_y", "nucleus_z"],
    )
    return cell


def load_root_cell(root_id: int, datastack: str, allow_held_out: bool = False) -> Cell:
    """The cell of `root_id` (see the module docstring), read from the cloud.

    A root of the held-out cohort raises `HeldOutRootError`, unless `allow_held_out`.
    """
    check_not_held_out(root_id, datastack, allow_held_out)
    source = ROOT_SOURCES[datastack]
    tables = fetch_root_tables(root_id, datastack)
    mesh = fetch_mesh(root_id, datastack)
    nuclei = Tables[source["nuclei_table"]].lazy().filter(pl.col("root_id") == root_id).collect()
    labels = fetch_root_labels(root_id, datastack)
    return build_root_cell(
        root_id,
        mesh,
        tables,
        fetch_root_synapses(root_id, datastack),
        nuclei,
        labels,
        # The cell metadata goes into the .osy file as JSON.
        meta=json.loads(json.dumps(root_pins(datastack))),
    )


@lru_cache
def feature_columns(datastack: str) -> list[str]:
    """The model feature columns of the pinned `domain_features` table, in table
    order, then `distance_to_nucleus`."""
    from ..models import select_spectral_columns

    path = deployment_path("domain_features", datastack)
    names = pl.scan_delta(delta_table(str(path))).collect_schema().names()
    return select_spectral_columns(names) + ["distance_to_nucleus"]


def domain_frame(cell: Cell) -> pl.DataFrame:
    """The `domains` features of `cell` with `domain_id`, without the `NO_DOMAIN` row.

    `analysis.models.predict(model, domain_frame(cell), keys=("domain_id",))` scores
    every domain of the root.
    """
    features = cell.annotations["domains"].features
    features = features[features.index != NO_DOMAIN]
    return pl.from_pandas(features.rename_axis("domain_id").reset_index())


def deployment_posteriors(cell: Cell) -> pl.DataFrame:
    """The deployment's predictions for each domain, in the form of `predict`."""
    frame = domain_frame(cell)
    return frame.select(
        "domain_id",
        *[c for c in frame.columns if c.startswith("posterior_") and c != "posterior_entropy"],
        "pred_label",
    )
