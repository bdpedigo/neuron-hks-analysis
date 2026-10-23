"""Data products: each module builds and registers its tables in `Tables`."""

from . import cell_selection, census, features, h01, labels, nuclei, planned, predictions, skeleton, unitary_spines, validation  # noqa: F401 -- populate the Tables registry
from .catalog import Tables

__all__ = ["Tables"]
