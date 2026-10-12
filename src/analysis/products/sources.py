"""Cloud paths and pins for the sources that products read."""

from datetime import datetime, timezone

from upath import UPath

DATASTACK = "minnie65_phase3_v1"
BASE_CLOUD_PATH = UPath("gs://bdp-ssa/meshmash-deployment")

SYNAPSE_MAPPING_VERSION = "s2"
SYNAPSE_MAPPING_PARAMS = "p-663726ec6deb9adf"

VERTEX_DOMAINS_VERSION = "s3"
VERTEX_DOMAINS_PARAMS = "p-2e737b09a92af64d"

DOMAIN_FEATURES_VERSION = "s3"
DOMAIN_FEATURES_PARAMS = "p-e758527c9d981378"

DOMAIN_EDGES_VERSION = "s3"
DOMAIN_EDGES_PARAMS = "p-b60a113f02ed8102"

SYNAPSE_MAPPING_PATH = (
    BASE_CLOUD_PATH
    / DATASTACK
    / "synapse_mapping"
    / SYNAPSE_MAPPING_VERSION
    / SYNAPSE_MAPPING_PARAMS
)
VERTEX_DOMAINS_PATH = (
    BASE_CLOUD_PATH
    / DATASTACK
    / "vertex_domains"
    / VERTEX_DOMAINS_VERSION
    / VERTEX_DOMAINS_PARAMS
)
DOMAIN_FEATURES_PATH = (
    BASE_CLOUD_PATH
    / DATASTACK
    / "domain_features"
    / DOMAIN_FEATURES_VERSION
    / DOMAIN_FEATURES_PARAMS
)
DOMAIN_EDGES_PATH = (
    BASE_CLOUD_PATH
    / DATASTACK
    / "domain_edges"
    / DOMAIN_EDGES_VERSION
    / DOMAIN_EDGES_PARAMS
)

# NOTE: the deployment tables are append-only, and the latest attempt per root
# wins at read time. A read with an `as_of` cutoff ignores the attempts after
# it, so a table that grows later (for example, the full run) reads the same.
# Each product chooses its own cutoff, so the read functions take it as an
# argument rather than reading a module value.

# The microns_top_labeled job was complete at this time (all roots but one,
# which cannot be processed).
LABELED_AS_OF = datetime(2026, 10, 12, 4, 13, tzinfo=timezone.utc)
