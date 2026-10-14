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

# NOTE: the domain_predictions hash includes the model, here
# `ssa_model_full_v1`. A new model gives a new path.
DOMAIN_PREDICTIONS_VERSION = "s1"
DOMAIN_PREDICTIONS_PARAMS = "p-96cea889064d4d41"

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
DOMAIN_PREDICTIONS_PATH = (
    BASE_CLOUD_PATH
    / DATASTACK
    / "domain_predictions"
    / DOMAIN_PREDICTIONS_VERSION
    / DOMAIN_PREDICTIONS_PARAMS
)

STEP_PINS = {
    "synapse_mapping": (SYNAPSE_MAPPING_VERSION, SYNAPSE_MAPPING_PARAMS),
    "vertex_domains": (VERTEX_DOMAINS_VERSION, VERTEX_DOMAINS_PARAMS),
    "domain_features": (DOMAIN_FEATURES_VERSION, DOMAIN_FEATURES_PARAMS),
    "domain_edges": (DOMAIN_EDGES_VERSION, DOMAIN_EDGES_PARAMS),
    "domain_predictions": (DOMAIN_PREDICTIONS_VERSION, DOMAIN_PREDICTIONS_PARAMS),
}


def deployment_path(product: str, datastack: str = DATASTACK) -> UPath:
    """The deployment table of `product` for `datastack`, at the pinned step
    version and parameter hash. The datastack is not part of the hash, so every
    datastack shares the pins."""
    version, params = STEP_PINS[product]
    return BASE_CLOUD_PATH / datastack / product / version / params


# NOTE: H01 is read at the time the labels were made (v1054), which CAVE no
# longer serves as a version. The deployment reads synapses and nuclei from a
# one-time snapshot of the job roots (meshmash-deployment
# scripts/snapshot_inputs.py), so the products read the same rows from it.
H01_DATASTACK = "h01_c3_flat"
H01_VERSION = 1054
H01_INPUTS_PATH = BASE_CLOUD_PATH / "_inputs" / H01_DATASTACK / f"v{H01_VERSION}"
H01_SYNAPSES_PATH = (
    H01_INPUTS_PATH / "synapses_using_sample_voxel_coords" / "post_pt_root_id"
)
H01_NUCLEI_PATH = H01_INPUTS_PATH / "nucleus" / "flat"

# NOTE: the deployment tables are append-only, and the latest attempt per root
# wins at read time. A read with an `as_of` cutoff ignores the attempts after
# it, so a table that grows later (for example, the full run) reads the same.
# Each product chooses its own cutoff, so the read functions take it as an
# argument rather than reading a module value.

# The microns_top_labeled job was complete at this time (all roots but one,
# which cannot be processed).
LABELED_AS_OF = datetime(2026, 10, 12, 4, 13, tzinfo=timezone.utc)

# The microns_validation job (run-20261012-235436-628a) was complete at this
# time: all 132 roots, last commit 2026-10-13 04:57 UTC.
VALIDATION_AS_OF = datetime(2026, 10, 13, 5, 0, tzinfo=timezone.utc)

# The MICrONS deployment tables hold two jobs: microns_top_labeled (features for
# the 6,010 labeled roots) and microns_validation (features and predictions for the
# 132 validation roots). This cutoff is after both, for the whole-root viewers.
MICRONS_AS_OF = VALIDATION_AS_OF

# S0: the public CAVE export of `synapses_pni_2` at v1822, the rows that the
# deployment mapped to the meshes. Positions are in voxels.
MICRONS_SYNAPSES_PATH = UPath(
    "gs://mat_dbs/public/deltalake_exports/minnie65_phase3_v1/v1822/synapses_pni_2/post_pt_root_id"
)
MICRONS_SYNAPSE_RESOLUTION = (4.0, 4.0, 40.0)

# NOTE: the ossify skeletons (S5) carry the synapses at the time of the
# skeleton, which is the v1822 timestamp. `skeleton.py` asserts this per file,
# because a file that is overwritten in place keeps its path.
SKELETON_PATH = UPath("gs://csm-skel/osy/minnie65/skel")

# The h01 job (run-20261013-224416-dcc1) was complete at this time: all 160
# roots, last commit 2026-10-14 14:47 UTC.
H01_AS_OF = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
