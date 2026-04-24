import numpy as np

xgfs_normal6 = [
    (64, 83, 211),
    (221, 179, 16),
    (181, 29, 20),
    (0, 190, 255),
    (251, 73, 176),
    (0, 178, 93),
    (202, 202, 202),
]
xgfs_normal12 = [
    (235, 172, 35),
    (184, 0, 88),
    (0, 140, 249),
    (0, 110, 0),
    (0, 187, 173),
    (209, 99, 230),
    (178, 69, 2),
    (255, 146, 135),
    (89, 84, 214),
    (0, 198, 248),
    (135, 133, 0),
    (0, 167, 108),
    (189, 189, 189),
]
xgfs_bright6 = [
    (239, 230, 69),
    (233, 53, 161),
    (0, 227, 255),
    (225, 86, 44),
    (83, 126, 255),
    (0, 203, 133),
    (238, 238, 238),
]
xgfs_dark6 = [
    (0, 89, 0),
    (0, 0, 120),
    (73, 13, 0),
    (138, 3, 79),
    (0, 90, 138),
    (68, 53, 0),
    (88, 88, 88),
]
xgfs_fancy6 = [
    (86, 100, 26),
    (192, 175, 251),
    (230, 161, 118),
    (0, 103, 138),
    (152, 68, 100),
    (94, 204, 171),
    (205, 205, 205),
]
xgfs_tarnish6 = [
    (39, 77, 82),
    (199, 162, 166),
    (129, 139, 112),
    (96, 78, 60),
    (140, 159, 183),
    (121, 104, 128),
    (192, 192, 192),
]

COMPARTMENT_PALETTE = {
    "soma": xgfs_bright6[2],
    # "shaft": xgfs_bright6[0],
    # "shaft": xgfs_normal6[1],
    "shaft": (221, 205, 37),
    "not_spine": (221, 205, 37),
    "spine": xgfs_bright6[1],
    "unknown": (20, 20, 20),
}

COMPARTMENT_PALETTE_MUTED = {
    "soma": xgfs_normal6[3],
    "shaft": xgfs_normal6[1],
    "not_spine": xgfs_normal6[1],
    "spine": xgfs_normal6[4],
    "single_spine": xgfs_normal6[4],
    "multi_spine": (145, 19, 204),
    "unknown": (100, 100, 100),
}


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


COMPARTMENT_PALETTE_HEX = {k: rgb_to_hex(v) for k, v in COMPARTMENT_PALETTE.items()}

COMPARTMENT_PALETTE_MUTED_HEX = {
    k: rgb_to_hex(v) for k, v in COMPARTMENT_PALETTE_MUTED.items()
}

# COMPARTMENT_PALETTE_BREWER2_HEX = {
#     "soma": "#8DA0CB",
#     "shaft": "#FFD92F",
#     "not_spine": "#FFD92F",
#     "spine": "#E78AC3",
#     "single_spine": "#E78AC3",
#     "multi_spine": "#87118D",
#     "unknown": "#B3B3B3",
# }
COMPARTMENT_PALETTE_BREWER2_HEX = {
    "soma": "#7570B3",
    "shaft": "#E6AB02",
    "not_spine": "#E6AB02",
    "spine": "#E7298A",
    "single_spine": "#E7298A",
    "multi_spine": "#FF0066",
    "unknown": "#B3B3B3",
}

COMPARTMENT_PALETTE_BREWER2 = {
    k: hex_to_rgb(v) for k, v in COMPARTMENT_PALETTE_BREWER2_HEX.items()
}


def color_weights(model, posteriors):
    rgb_colors = np.array([COMPARTMENT_PALETTE[label] for label in model.classes_])
    rgb_colors = rgb_colors / 255
    return np.dot(posteriors, rgb_colors)


def predict_proba_colors(model, X):
    posteriors = model.predict_proba(X)
    rgb_colors = np.array([COMPARTMENT_PALETTE[label] for label in model.classes_])
    rgb_colors = rgb_colors / 255
    return np.dot(posteriors, rgb_colors)


# REF: https://scottplot.net/cookbook/5.0/palettes/
# CELL_TYPE_PALETTE = {
#     "E": "#ffa3c7",
#     "23P": "#FFC8C3",
#     "4P": "#FEAE7C",
#     "5P-IT": "#FF9287",
#     "5P-ET": "#B80058",
#     "5P-PT": "#B80058",
#     "5P-NP": "#FF5CAA",
#     "6P-IT": "#EDC1F5",
#     "6P-CT": "#D163E6",
#     "I": "#488f9cff",
#     "BC": "#008CF9",
#     "BPC": "#00A76C",
#     "MC": "#BDF2FF",
#     "NGC": "#AAAAAA",
#     "Thalamic": "#FFFC43",
# }
# CELL_TYPE_PALETTE["Excitatory"] = CELL_TYPE_PALETTE["E"]
# CELL_TYPE_PALETTE["Inhibitory"] = CELL_TYPE_PALETTE["I"]


CELL_TYPE_PALETTE = {
    "E": "#FC8D62",
    "23P": "#FFAFA8",
    "4P": "#FEA36B",
    "5P-IT": "#FD8679",
    "5P-ET": "#B80058",
    "5P-PT": "#B80058",
    "5P-NP": "#FF5CAA",
    "6P-IT": "#EDC1F5",
    "6P-CT": "#D163E6",
    "I": "#66C2A5",
    "BC": "#008CF9",
    "BPC": "#00A76C",
    "MC": "#88E6FD",
    "NGC": "#AAAAAA",
    "TH": "#E5C494",
}
CELL_TYPE_PALETTE["Excitatory"] = CELL_TYPE_PALETTE["E"]
CELL_TYPE_PALETTE["Inhibitory"] = CELL_TYPE_PALETTE["I"]
CELL_TYPE_PALETTE["excitatory"] = CELL_TYPE_PALETTE["E"]
CELL_TYPE_PALETTE["inhibitory"] = CELL_TYPE_PALETTE["I"]
CELL_TYPE_PALETTE["Th"] = CELL_TYPE_PALETTE["TH"]
CELL_TYPE_PALETTE["thalamic"] = CELL_TYPE_PALETTE["TH"]
CELL_TYPE_PALETTE["Thalamic"] = CELL_TYPE_PALETTE["TH"]
CELL_TYPE_PALETTE["Unknown"] = "#272727"
CELL_TYPE_PALETTE["unknown"] = "#272727"
CELL_TYPE_PALETTE["Unk"] = "#272727"


import colorsys

import matplotlib.colors as mcolors
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Core mapping functions
# ---------------------------------------------------------------------------


def _normalize(X: np.ndarray) -> np.ndarray:
    """Robustly normalize each column of X to [0, 1] using interquartile range."""
    scaler = RobustScaler()
    return scaler.fit_transform(X)


def pca2d_to_colors(
    X: np.ndarray, lightness: float = 0.6, saturation: float = 0.85
) -> np.ndarray:
    """
    Map 2D PCA coordinates to RGB colors via polar HSL.

    The angle in the 2D plane → hue (full 360° range).
    The radius from the centroid → saturation weight (outer = more vivid).

    Parameters
    ----------
    X           : (N, 2) array of 2D PCA coordinates.
    lightness   : HSL lightness, 0–1. 0.5–0.65 works well.
    saturation  : HSL saturation, 0–1.

    Returns
    -------
    colors : (N, 3) float32 array of RGB values in [0, 1].
    """
    assert X.shape[1] == 2, "X must be (N, 2)"

    # Center the cloud
    centered = X - np.nanmean(X, axis=0)

    # Polar coordinates → hue from angle, value from radius
    angles = np.arctan2(centered[:, 1], centered[:, 0])  # [-π, π]
    hue = (angles / (2 * np.pi)) % 1.0  # [0, 1)
    print(hue)

    # Optionally modulate saturation by distance from center
    radii = np.linalg.norm(centered, axis=1)
    radii_norm = radii / (np.nanmax(radii) + 1e-8)  # [0, 1]
    sat = saturation * (0.4 + 0.6 * radii_norm)  # inner pts less vivid

    # HSL → RGB via matplotlib
    hsl = np.stack([hue, sat, np.full_like(hue, lightness)], axis=1)
    print(hsl)
    rgb = np.array([mcolors.hsv_to_rgb([h, s, l]) for h, s, l in hsl])
    # Note: matplotlib's hls_to_rgb takes (h, l, s)
    rgb = np.array(
        [colorsys.hls_to_rgb(h, lightness, s) for h, s in zip(hue, sat)],
        dtype=np.float32,
    )
    return np.clip(rgb, 0, 1)


def pca2d_to_colors_lightness(
    X: np.ndarray,
    L_range: tuple = (0.35, 0.75),
    saturation: float = 0.85,
) -> np.ndarray:
    """
    Map 2D PCA coordinates to RGB colors via polar HLS, varying both hue and lightness.

    The angle in the 2D plane → hue (full 360° range).
    The radius from the centroid → lightness (outer = lighter, inner = darker).
    Saturation is held fixed.

    Parameters
    ----------
    X           : (N, 2) array of 2D PCA coordinates.
    L_range     : (min, max) lightness as fractions in [0, 1]. Controls the
                  range of lightness mapped from the center to the periphery.
    saturation  : HLS saturation, 0–1.

    Returns
    -------
    colors : (N, 3) float32 array of RGB values in [0, 1].
    """
    assert X.shape[1] == 2, "X must be (N, 2)"

    centered = X - np.nanmean(X, axis=0)

    angles = np.arctan2(centered[:, 1], centered[:, 0])  # [-π, π]
    hue = (angles / (2 * np.pi)) % 1.0  # [0, 1)

    radii = np.linalg.norm(centered, axis=1)
    radii_norm = radii / (np.nanmax(radii) + 1e-8)  # [0, 1]

    L_min, L_max = L_range
    lightness = L_min + (L_max - L_min) * radii_norm
    print(lightness)
    rgb = np.array(
        [colorsys.hls_to_rgb(h, l, saturation) for h, l in zip(hue, lightness)],
        dtype=np.float32,
    )
    return np.clip(rgb, 0, 1)


def pca2d_to_colors_lch(
    X: np.ndarray,
    L_range: tuple = (0.40, 0.65),
    chroma: float = 35,
) -> np.ndarray:
    """
    Map 2D PCA coordinates to RGB colors via CIELAB LCh color space.

    The angle in the 2D plane → LCh hue angle (converted to a* / b*).
    The radius from the centroid → lightness (outer = lighter).

    Using LCh rather than HLS avoids perceptual non-uniformity: all hues
    appear equally vivid at the same chroma value.  Lower chroma (~30–40)
    gives muted/natural tones; higher chroma (~60–80) is more vivid.

    Parameters
    ----------
    X        : (N, 2) array of 2D PCA coordinates.
    L_range  : (min, max) lightness as fractions in [0, 1].
    chroma   : LCh chroma (≈ colorfulness). 35 = muted, 60 = moderate, 80+ = vivid.

    Returns
    -------
    colors : (N, 3) float32 array of sRGB values in [0, 1].
    """
    assert X.shape[1] == 2, "X must be (N, 2)"

    from colorspacious import cspace_convert

    centered = X - np.nanmean(X, axis=0)

    angles_rad = np.arctan2(centered[:, 1], centered[:, 0])

    radii = np.linalg.norm(centered, axis=1)
    radii_norm = radii / (np.nanmax(radii) + 1e-8)

    L = (L_range[0] + (L_range[1] - L_range[0]) * radii_norm) * 100  # [0, 100]
    a = chroma * np.cos(angles_rad)
    b = chroma * np.sin(angles_rad)

    lab = np.stack([L, a, b], axis=1)
    rgb = cspace_convert(lab, "CIELab", "sRGB1")
    return np.clip(rgb, 0, 1).astype(np.float32)


def pca3d_to_colors(
    X: np.ndarray, L_range: tuple = (30, 85), ab_range: tuple = (-80, 80)
) -> np.ndarray:
    """
    Map 3D PCA coordinates to RGB colors via CIELAB color space.

    PC1 → L* (lightness)
    PC2 → a* (green↔red axis)
    PC3 → b* (blue↔yellow axis)

    CIELAB is perceptually uniform: equal Euclidean distance ≈ equal perceived
    color difference, so the color spread mirrors the feature spread.

    Parameters
    ----------
    X        : (N, 3) array of 3D PCA coordinates.
    L_range  : (min, max) lightness range within [0, 100].
    ab_range : (min, max) for a* and b* axes (symmetric recommended).

    Returns
    -------
    colors : (N, 3) float32 array of sRGB values in [0, 1].
    """
    assert X.shape[1] == 3, "X must be (N, 3)"

    try:
        from colorspacious import cspace_convert

        _has_colorspacious = True
    except ImportError:
        _has_colorspacious = False

    # Normalize PCs to target ranges
    norm = _normalize(X)  # (N, 3) in [0, 1]

    L = norm[:, 0] * (L_range[1] - L_range[0]) + L_range[0]
    a = norm[:, 1] * (ab_range[1] - ab_range[0]) + ab_range[0]
    b = norm[:, 2] * (ab_range[1] - ab_range[0]) + ab_range[0]

    lab = np.stack([L, a, b], axis=1)  # (N, 3)

    if _has_colorspacious:
        # High-quality conversion via colorspacious
        rgb = cspace_convert(lab, "CIELab", "sRGB1")
    else:
        # Fallback: use matplotlib's built-in Lab→XYZ→sRGB pipeline
        # matplotlib expects Lab normalized differently; use manual route
        rgb = _lab_to_rgb_fallback(lab)

    return np.clip(rgb, 0, 1).astype(np.float32)


def _lab_to_rgb_fallback(lab: np.ndarray) -> np.ndarray:
    """
    Pure-numpy CIELab → sRGB conversion (no external deps beyond numpy).
    Illuminant D65, 2° observer.
    """
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    # Lab → XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def f_inv(t):
        delta = 6 / 29
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))

    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)

    # XYZ → linear sRGB
    xyz = np.stack([X, Y, Z], axis=1)
    M = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    rgb_lin = xyz @ M.T

    # Linear sRGB → gamma-corrected sRGB
    def gamma(u):
        return np.where(
            u <= 0.0031308,
            12.92 * u,
            1.055 * np.power(np.maximum(u, 0), 1 / 2.4) - 0.055,
        )

    return gamma(rgb_lin)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def features_to_colors(X: np.ndarray, **kwargs) -> np.ndarray:
    """
    Auto-dispatch based on dimensionality.

    Parameters
    ----------
    X      : (N, 2) or (N, 3) PCA array.
    kwargs : Passed through to pca2d_to_colors or pca3d_to_colors.

    Returns
    -------
    colors : (N, 3) float32 RGB array in [0, 1].
    """
    if X.shape[1] == 2:
        return pca2d_to_colors(X, **kwargs)
    elif X.shape[1] == 3:
        return pca3d_to_colors(X, **kwargs)
    else:
        raise ValueError(f"X must have 2 or 3 columns, got {X.shape[1]}")
