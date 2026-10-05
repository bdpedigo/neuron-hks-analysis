import polars as pl
import polars_io_tools  # noqa: F401 -- registers the .piot namespace
from upath import UPath

from .catalog import Tables
from .io import VERSION
from .wrangle import get_label_table

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


Tables.register(
    "synapse_mapping",
    lambda: _latest_attempt(pl.scan_delta(synapse_mapping_path)).select(
        "root_id", "synapse_id", "mesh_index"
    ),
)
Tables.register(
    "vertex_domains",
    lambda: _latest_attempt(pl.scan_delta(vertex_domains_path)).select(
        "root_id", "mesh_index", "domain_id"
    ),
)
Tables.register(
    "synapse_to_domain",
    lambda: Tables["synapse_mapping"]
    .lazy()
    .join(Tables["vertex_domains"].lazy(), on=["root_id", "mesh_index"], how="left")
    .drop("mesh_index"),
)
Tables.register(
    "domain_features",
    lambda: _latest_attempt(pl.scan_delta(domain_features_path)),
)
Tables.register("labels", lambda: get_label_table(version=VERSION).lazy())
Tables.register(
    "synapse_labels",
    lambda: Tables["labels"]
    .lazy()
    .filter(pl.col("target_id") != -1)
    .rename({"target_id": "synapse_id", f"pt_root_id_{VERSION}": "root_id"}),
)


# TODO performance here will degrade once we have more cloud data
# need to either accept that or include the hash in the join or something
# that might not even help based on root ID coverage
# TODO also doubt that the filtered_join is very effective here for the same reason
def synapse_features_lf(synapses: pl.LazyFrame) -> pl.LazyFrame:
    return (
        synapses.piot.filtered_join(
            Tables["synapse_to_domain"].lazy(),
            on=["root_id", "synapse_id"],
            how="left",
            coalesce=False,
        )
        .join(Tables["domain_features"].lazy(), on=["root_id", "domain_id"], how="left")
        .drop("domain_id")
    )


Tables.register(
    "labeled_synapse_features",
    lambda: synapse_features_lf(Tables["synapse_labels"].lazy()),
    cache_params={
        "synapse_mapping_version": synapse_mapping_version,
        "synapse_mapping_params": synapse_mapping_params,
        "vertex_domains_version": vertex_domains_version,
        "vertex_domains_params": vertex_domains_params,
        "domain_features_version": domain_features_version,
        "domain_features_params": domain_features_params,
    },
)
