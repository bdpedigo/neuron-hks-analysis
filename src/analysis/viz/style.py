"""The conventions of every pyvista view: the label palette, the categorical and
scalar color settings, and the plotter defaults.

A viewer or figure imports these and does not set them itself, so that two views
of the same labels always look the same.
"""

import warnings

import numpy as np
import pyvista as pv

from ..colors import COMPARTMENT_PALETTE_MUTED_HEX

# NOTE: a fixed order, so a class has the same code (and color) in every view,
# whichever classes a given model or cell has.
CLASS_ORDER = ("soma", "shaft", "spine", "not_spine")
LABEL_PALETTE = {label: COMPARTMENT_PALETTE_MUTED_HEX[label] for label in CLASS_ORDER}

MESH_COLOR = "lightgray"
# Just enough to see synapses through the mesh.
MESH_OPACITY = 0.85
# Darker than the mesh, so a point with no value still shows against it.
NO_DATA_COLOR = "dimgray"
POINT_SIZE = 10
SCALAR_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

BACKGROUND = "white"
INK = "#18181b"
FONT_SIZE = 12
WINDOW = (1500, 1000)


def label_codes(labels, categories=CLASS_ORDER) -> np.ndarray:
    """Each label's index in `categories` as float32, NaN for a null label.

    An unknown label is NaN too, with a warning.
    """
    labels = np.asarray(labels, dtype=object)
    lookup = {label: code for code, label in enumerate(categories)}
    codes = np.array([lookup.get(label, np.nan) for label in labels], dtype=np.float32)
    unknown = {
        label
        for label in labels
        if label is not None and label == label and label not in lookup
    }
    if unknown:
        warnings.warn(f"no color for labels {sorted(map(str, unknown))}", stacklevel=2)
    return codes


def categorical_kwargs(categories=CLASS_ORDER) -> dict:
    """`add_mesh` kwargs that color the codes of `label_codes` by `LABEL_PALETTE`."""
    return {
        "cmap": [LABEL_PALETTE[c] for c in categories],
        "n_colors": len(categories),
        "clim": (-0.5, len(categories) - 0.5),
        # NOTE: interpolating codes across a face passes through the codes between
        # them (soma to spine through shaft), so the colors interpolate instead.
        "interpolate_before_map": False,
        "show_scalar_bar": False,
        "nan_color": NO_DATA_COLOR,
    }


def scalar_clim(values: np.ndarray, symmetric: bool = False, percentiles=(1, 99)) -> tuple:
    """The color range of a scalar: the `percentiles` of its finite values, or
    `(-m, m)` for the larger absolute percentile `m` when `symmetric`."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        return (0.0, 1.0)
    low, high = np.percentile(finite, percentiles)
    if symmetric:
        extent = max(abs(low), abs(high)) or 1.0
        return (-float(extent), float(extent))
    if high <= low:
        high = low + 1.0
    return (float(low), float(high))


def scalar_kwargs(values: np.ndarray, symmetric: bool = False, percentiles=(1, 99)) -> dict:
    """`add_mesh` kwargs for a continuous value (see `scalar_clim`)."""
    return {
        "cmap": DIVERGING_CMAP if symmetric else SCALAR_CMAP,
        "clim": scalar_clim(values, symmetric, percentiles),
        "nan_color": NO_DATA_COLOR,
    }


def class_rgb(classes) -> np.ndarray:
    """The palette colors of `classes`, as an (n, 3) array in [0, 1]."""
    return np.array([pv.Color(LABEL_PALETTE[c]).float_rgb for c in classes])


def posterior_rgb(posteriors: np.ndarray, classes) -> np.ndarray:
    """The class colors weighted by the posterior of each row, (n, 3) in [0, 1].

    A row with a missing posterior is NaN.
    """
    return np.asarray(posteriors, dtype=np.float64) @ class_rgb(classes)


def rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """(n, 3) floats in [0, 1] as uint8 colors, with NaN rows as `NO_DATA_COLOR`."""
    out = np.empty(rgb.shape, dtype=np.uint8)
    missing = ~np.isfinite(rgb).all(axis=1)
    out[~missing] = np.clip(np.round(rgb[~missing] * 255), 0, 255)
    out[missing] = pv.Color(NO_DATA_COLOR).int_rgb
    return out


def new_plotter(**kwargs) -> pv.Plotter:
    """A plotter with the shared defaults. Right-click flies the camera to the
    clicked point."""
    plotter = pv.Plotter(window_size=kwargs.pop("window_size", WINDOW), **kwargs)
    plotter.set_background(BACKGROUND)
    # Sorts the translucent mesh against itself and the points.
    plotter.enable_depth_peeling()
    plotter.enable_fly_to_right_click()
    return plotter
