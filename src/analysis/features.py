import polars as pl
from upath import UPath

DATASTACK = "minnie65_phase3_v1"
BASE_CLOUD_PATH = "gs://bdp-ssa/meshmash-deployment"
base = UPath(BASE_CLOUD_PATH)

synapse_mapping_version = "s1"
synapse_mapping_params = "p-72b901b7386541b7"

vertex_domains_version = "s1"
vertex_domains_params = "p-0f6493d8b4cf1870"

domain_features_version = "s1"
domain_features_params = "p-90e8423092581ed2"

domain_edges_version = "s1"
domain_edges_params = "p-213a115ea1c269b6"

synapse_mapping_path = (
    base
    / DATASTACK
    / "synapse_mapping"
    / synapse_mapping_version
    / synapse_mapping_params
)
vertex_domains_path = (
    base / DATASTACK / "vertex_domains" / vertex_domains_version / vertex_domains_params
)
domain_features_path = (
    base
    / DATASTACK
    / "domain_features"
    / domain_features_version
    / domain_features_params
)
domain_edges_path = (
    base / DATASTACK / "domain_edges" / domain_edges_version / domain_edges_params
)


def _latest_attempt(lf: pl.LazyFrame) -> pl.LazyFrame:
    # An attempt replaces all of a root's rows at once (dedupe_key="root_id"),
    # so the newest attempt_id for a root wins, whole-root.
    return lf.filter(pl.col("attempt_id") == pl.col("attempt_id").max().over("root_id"))


synapse_mapping_lf = _latest_attempt(pl.scan_delta(synapse_mapping_path)).select(
    "root_id", "synapse_id", "mesh_index"
)
vertex_domains_lf = _latest_attempt(pl.scan_delta(vertex_domains_path)).select(
    "root_id", "mesh_index", "domain_id"
)
synapse_to_domain_lf = synapse_mapping_lf.join(
    vertex_domains_lf, on=["root_id", "mesh_index"], how="left"
).drop("mesh_index")
domain_features_lf = _latest_attempt(pl.scan_delta(domain_features_path))
