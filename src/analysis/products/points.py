"""Domain features for point labels, mapped to the meshes locally.

The deployment maps only synapses (`synapse_mapping`). A point label goes the same
way, here on the laptop: the point (nm) to its mesh vertex with
`meshmash.utils.project_points_to_mesh` (the deployment's method and distance
threshold), the vertex to its domain in `vertex_domains`, and the domain to its row
of `domain_features`. The mesh is the one the deployment fetched (`fetch_mesh`).
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from meshmash.utils import project_points_to_mesh
from tqdm.auto import tqdm

from ..io import param_hash
from ..sources import DATASTACK, cutoff, deployment_path, pin_params
from .catalog import Tables
from .deployment import query_latest_attempts, scan_roots
from .roots import NO_DOMAIN, _vertex_domain, fetch_mesh, mesh_get_kwargs

# The parameters of the deployment's `synapse_mapping` step (job spec params).
MAPPING_METHOD = "surface"
DISTANCE_THRESHOLD_NM = 500.0


def _load_mesh(root_id: int, datastack: str, mesh_cache: Optional[Path]) -> tuple[np.ndarray, np.ndarray]:
    if mesh_cache is None:
        return fetch_mesh(root_id, datastack)
    path = mesh_cache / datastack / param_hash(**mesh_get_kwargs(datastack)) / f"{root_id}.npz"
    if path.exists():
        with np.load(path) as f:
            return f["vertices"], f["faces"]
    vertices, faces = fetch_mesh(root_id, datastack)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{root_id}.tmp.npz")
    np.savez(tmp, vertices=vertices, faces=faces)
    tmp.rename(path)
    return vertices, faces


def map_root_points(
    points: pl.DataFrame, mesh: tuple[np.ndarray, np.ndarray], vertex_domains: pl.DataFrame
) -> pl.DataFrame:
    """`point_id`, `mesh_index`, `distance_nm` and `domain_id` of the `points` of one
    root, on its `mesh` and its `vertex_domains` rows (`mesh_index`, `domain_id`)."""
    vertices, faces = mesh
    vertex_domain = _vertex_domain(vertex_domains, len(vertices))
    indices, distances = project_points_to_mesh(
        points.select("x", "y", "z").to_numpy(),
        (vertices, faces),
        distance_threshold=DISTANCE_THRESHOLD_NM,
        return_distances=True,
        method=MAPPING_METHOD,
    )
    domain = np.where(indices >= 0, vertex_domain[np.maximum(indices, 0)], NO_DOMAIN)
    return pl.DataFrame({
        "point_id": points["point_id"],
        "mesh_index": pl.Series(indices, dtype=pl.Int64),
        "distance_nm": pl.Series(distances, dtype=pl.Float64),
        "domain_id": pl.Series(domain, dtype=pl.Int64),
    }).with_columns(
        mesh_index=pl.when(pl.col("mesh_index") >= 0).then("mesh_index"),
        domain_id=pl.when(pl.col("domain_id") != NO_DOMAIN).then("domain_id"),
    )


def map_points_to_domains(
    points: pl.DataFrame,
    *,
    as_of: datetime,
    datastack: str = DATASTACK,
    mesh_cache: Optional[Path] = None,
) -> pl.DataFrame:
    """`points` (`point_id`, `root_id`, `x`, `y`, `z` in nm) with the `mesh_index`,
    `distance_nm` and `domain_id` of each point, from `vertex_domains` as the
    deployment tables were at `as_of`.

    `mesh_index` and `domain_id` are null beyond `DISTANCE_THRESHOLD_NM`, and
    `domain_id` is null for a vertex in no domain. With a `mesh_cache` folder, the
    meshes are read from and written to it.
    """
    root_ids = points["root_id"].unique().sort().to_list()
    (vertex_lf,) = query_latest_attempts(
        [scan_roots(deployment_path("vertex_domains", datastack), root_ids, as_of=as_of)]
    )
    vertex_domains = vertex_lf.select("root_id", "mesh_index", "domain_id").collect(engine="streaming")
    by_root = vertex_domains.partition_by("root_id", as_dict=True, include_key=False)
    out = []
    for (root_id,), frame in tqdm(
        points.partition_by("root_id", as_dict=True, maintain_order=True).items(), total=len(root_ids)
    ):
        if (root_id,) not in by_root:
            # NOTE: a root the deployment did not process has no domains.
            out.append(frame.select("point_id").with_columns(
                mesh_index=pl.lit(None, pl.Int64), distance_nm=pl.lit(None, pl.Float64),
                domain_id=pl.lit(None, pl.Int64),
            ))
            continue
        mesh = _load_mesh(root_id, datastack, mesh_cache)
        out.append(map_root_points(frame, mesh, by_root[(root_id,)]))
    return points.join(pl.concat(out), on="point_id", how="left", maintain_order="left")


def point_features(points: pl.DataFrame, *, as_of: datetime, mesh_cache: Optional[Path] = None) -> pl.DataFrame:
    """`points` mapped to domains (see `map_points_to_domains`), left-joined to the
    `domain_features` row of each domain."""
    mapped = map_points_to_domains(points, as_of=as_of, mesh_cache=mesh_cache)
    root_ids = mapped["root_id"].unique().to_list()
    (domain_lf,) = query_latest_attempts(
        [scan_roots(deployment_path("domain_features", DATASTACK), root_ids, as_of=as_of)]
    )
    domain_rows = domain_lf.join(
        mapped.lazy().select("root_id", "domain_id").drop_nulls().unique(),
        on=["root_id", "domain_id"],
        how="semi",
    ).collect(engine="streaming")
    return mapped.join(
        domain_rows.with_columns(pl.col("domain_id").cast(pl.Int64)),
        on=["root_id", "domain_id"],
        how="left",
        maintain_order="left",
    )


Tables.register(
    "labeled_point_features",
    description=(
        "One row per point label (from `point_labels`), with the mesh vertex and "
        "domain it maps to (mapped locally, as the deployment maps synapses) and that "
        "domain's spectral features. Read from the deployment tables at the "
        "`training_points` cutoff. The point rows of the training set."
    ),
    materialize=lambda path: point_features(
        Tables["point_labels"].collect(), as_of=cutoff("training_points")
    ).write_parquet(path),
    cache_params=lambda: {
        "labels": Tables["labels"].cache_key,
        # NOTE: a frozen table has no cache key, so its content stands in.
        "hand_point_labels": hashlib.md5(Tables["hand_point_labels"].frozen.read_bytes()).hexdigest(),
        "mapping": {"method": MAPPING_METHOD, "distance_threshold": DISTANCE_THRESHOLD_NM},
        "mesh_get_kwargs": mesh_get_kwargs(DATASTACK),
        **pin_params("vertex_domains", "domain_features", as_of=cutoff("training_points")),
    },
)
