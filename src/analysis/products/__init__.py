"""Data products: each module builds and registers its tables in `Tables`."""

from . import cell_info, cell_selection, census, census_scan, features, h01, labels, nuclei, planned, points, predictions, skeleton, synapses, unitary_spines, validation  # noqa: F401 -- populate the Tables registry
from .catalog import Tables

__all__ = ["Tables"]
