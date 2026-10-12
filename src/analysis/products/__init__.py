"""Data products: each module builds and registers its tables in `Tables`."""

from . import features, labels, nuclei  # noqa: F401 -- populate the Tables registry
from .catalog import Tables

__all__ = ["Tables"]
