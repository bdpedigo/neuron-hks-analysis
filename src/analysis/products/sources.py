"""Cloud paths and pins for the sources that products read."""

from datetime import datetime, timezone

from upath import UPath

DATASTACK = "minnie65_phase3_v1"
BASE_CLOUD_PATH = UPath("gs://bdp-ssa/meshmash-deployment")

# The cloud tables that the analysis reads, at one step version and parameter
# hash each. Every datastack shares them, because the datastack is not part of
# the hash.
# NOTE: a downstream hash includes the identities of its upstream tables, so a
# rerun of an upstream step gives every table below it a new path.
STEP_PINS = {
    "synapse_mapping": ("s2", "p-663726ec6deb9adf"),
    "vertex_domains": ("s4", "p-2e737b09a92af64d"),
    "domain_features": ("s4", "p-e758527c9d981378"),
    "domain_edges": ("s4", "p-b60a113f02ed8102"),
}


class UnpinnedError(KeyError):
    pass


def step_pin(product: str) -> tuple[str, str]:
    """The pinned step version and parameter hash of `product`."""
    if product not in STEP_PINS:
        raise UnpinnedError(f"{product!r} has no pin in STEP_PINS")
    return STEP_PINS[product]


def deployment_path(product: str, datastack: str = DATASTACK) -> UPath:
    """The deployment table of `product` for `datastack`, at its pin."""
    version, params = step_pin(product)
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
# it, so a table that grows later (for example, in the full run) reads the same.
# One table can have more than one cutoff: the training features keep theirs
# when the full run appends rows. So a product chooses its cutoff, and the read
# functions take it as an argument. A product reads all of its tables at the
# cutoff of its newest input.
CUTOFFS = {
    # The features of the MICrONS and H01 test cohorts were complete at this
    # time (last data commits 2026-10-21 16:58 UTC and 01:55 UTC).
    "features": datetime(2026, 10, 21, 18, 45, tzinfo=timezone.utc),
}


def cutoff(name: str) -> datetime:
    if name not in CUTOFFS:
        raise UnpinnedError(f"no cutoff named {name!r} in CUTOFFS")
    return CUTOFFS[name]


def pin_params(*products: str, as_of: datetime) -> dict:
    """The pins of a read of `products` at `as_of`, for a cache key."""
    return {
        "as_of": as_of.isoformat(),
        **{product: "/".join(step_pin(product)) for product in products},
    }


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
