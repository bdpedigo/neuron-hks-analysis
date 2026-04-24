from pathlib import Path
from typing import Union

import humanize
import numpy as np
import pyvista as pv
import toml
from sklearn.neighbors import NearestNeighbors

from grotto.client import GrottoClient

from .colors import COMPARTMENT_PALETTE

# Global variable for default variables file
DEFAULT_VARIABLES_FILE = Path(__file__).parent.parent.parent / "glued_variables.tex"


def start_client():
    client = GrottoClient("minnie65_phase3_v1", version=1181)
    return client


def project_points_to_mesh(
    points, mesh, distance_threshold=None, return_distances=False
):
    if isinstance(mesh, tuple):
        vertices = mesh[0]
    else:
        vertices = mesh.vertices
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(vertices)

    distances, indices = nn.kneighbors(points)
    indices = indices.reshape(-1)
    distances = distances.reshape(-1)
    if distance_threshold is not None:
        indices[distances > distance_threshold] = -1

    if return_distances:
        return indices, distances
    else:
        return indices


def mesh_to_graph_tables(vertices, faces):
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    nodes = np.asarray(vertices)
    return nodes, edges


# def mesh_to_csgraph(vertices, faces):
#     nodes, edges = mesh_to_graph_tables(vertices, faces)

#     nodes = pd.DataFrame(nodes, columns=["x", "y", "z"])
#     edges = pd.DataFrame(edges, columns=["source", "target"])

#     nf = NetworkFrame(nodes, edges)
#     nf.apply_node_features(["x", "y", "z"], inplace=True)
#     X = nf.edges[["source_x", "source_y", "source_z"]]
#     Y = nf.edges[["target_x", "target_y", "target_z"]]
#     distances = paired_euclidean_distances(X, Y)
#     nf.edges["distance"] = distances

#     adjacency = nf.to_sparse_adjacency(weight_col="distance")
#     return adjacency


# def mask_neighborhoods(adjacency, indices, distance=500):
#     from scipy.sparse.csgraph import dijkstra

#     adjacency = self.to_csgraph()
#     dists = dijkstra(
#         adjacency, directed=False, limit=distance, unweighted=False, indices=indices
#     )
#     mask = np.isfinite(dists)
#     return mask


def nan_predict(X, model, method="predict"):
    index = np.arange(len(X))
    mask = np.isnan(X).any(axis=1)
    non_na_indices = index[~mask]
    if method == "predict":
        scalars = model.predict(X[non_na_indices])
        new_scalars = np.array([] * len(X))
    elif method == "predict_proba":
        scalars = model.predict_proba(X[non_na_indices])
        new_scalars = np.zeros((len(X), len(model.classes_)))
        new_scalars[non_na_indices] = scalars
    return new_scalars


def set_pyvista_theme():
    import pyvista as pv

    pv.set_plot_theme("default")
    pv.global_theme.lighting_params.ambient = 0.35
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_scalar_bar = False


def set_matplotlib_theme(font_scale=1.0):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk", font_scale=font_scale)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["figure.dpi"] = 300


def _replace_none(parameters):
    for key, value in parameters.items():
        if isinstance(value, dict):
            parameters[key] = _replace_none(value)
        elif value == "None":
            parameters[key] = None
    return parameters


def get_experiment_info(file):
    if file.endswith(".py") and "/" in file:
        experiment_name = file.split("/")[-2]
    else:
        experiment_name = file

    from .io import PROJECT_ROOT

    path = PROJECT_ROOT / "experiments" / experiment_name
    parameters = toml.load(path / "parameters.toml")

    # replace and "None" strings with None. keep in mind that some parameters may be
    # dicts themselves

    parameters = _replace_none(parameters)

    return experiment_name, path, parameters


def make_spheres_from_points(
    points, labels, theta_resolution=20, phi_resolution=20, radius=800
):
    spheres = []
    for i, point in enumerate(points):
        sphere = pv.Sphere(
            radius=radius,
            center=point,
            theta_resolution=theta_resolution,
            phi_resolution=phi_resolution,
        )
        sphere.point_data["tag"] = np.full(
            sphere.n_points,
            labels[i],
        )
        sphere.point_data["rgb"] = (
            np.array([COMPARTMENT_PALETTE[c] for c in sphere.point_data["tag"]]) / 255
        )
        spheres.append(sphere)
    spheres = pv.MultiBlock(spheres).combine().triangulate().extract_surface()
    return spheres


# def encode(name, value):
#     return f"{name}={value}"

# def decode(line):
#     name, value = line.split("=", 1)
#     return name, value


# def encode(name, value):
#     return r"\def\glued@var@" + f"{name}" + "{" + f"{value}" + "}"


# def decode(line):
#     prefix = r"\def\glued@var@"
#     if line.startswith(prefix):
#         rest = line[len(prefix) :]
#         name, value = rest.split("}", 1)
#         value = value.lstrip("{").rstrip("}")
#         return name, value
#     else:
#         raise ValueError(f"Line does not start with expected prefix: {line}")


def encode(name, value):
    return r"\expandafter\def\csname " + f"{name}" + r"\endcsname{" + f"{value}" + "}"


def decode(line):
    prefix = r"\expandafter\def\csname "
    suffix = r"\endcsname"
    if line.startswith(prefix) and suffix in line:
        rest = line[len(prefix) :]
        name, value = rest.split(suffix, 1)
        value = value.lstrip("{").rstrip("}")
        return name, value
    else:
        raise ValueError(f"Line does not start with expected prefix or suffix: {line}")


def format_string(value, format_str: str) -> str:
    if format_str.endswith("%"):
        value = float(value) * 100
        format_str = format_str.rstrip("%") + r"\%"
    elif format_str.startswith("sci"):
        rest = format_str[len("sci") :].strip()
        if len(rest) > 0:
            format_str = rest
        else:
            format_str = "{:.1e}"
        formatted_value = format_str.format(value)
        mantissa, exp = formatted_value.split("e")
        formatted_value = mantissa
        formatted_value += r" \times 10^{" + str(int(exp)) + "}"
        formatted_value = "$" + formatted_value + "$"
        return formatted_value
    elif format_str.startswith("intword"):
        rest = format_str[len("intword") :].strip()
        if len(rest) > 0:
            format_str = rest
        else: 
            format_str = "%.1f"
        return humanize.intword(int(value), format=format_str)
    formatted_value = format_str.format(value)
    return formatted_value


def save_variables(**kwargs) -> None:
    """
    Write or update named variables in a key=value text file.

    Each variable is stored as 'name=value' on a separate line.
    Reads existing file as a dict, updates with new variables, and rewrites entire file.

    Parameters
    ----------
    **kwargs
        Variable names and values to write/update
    format : str, optional
        Python format string to apply to all values (e.g., "{:.2f}", "{:.4e}")

    Examples
    --------
    >>> save_variables(accuracy=0.95, n_samples=1000)
    >>> save_variables(loss=0.023456, epochs=100, format="{:.2f}")
    """
    if not kwargs:
        return

    # Extract format string if provided
    format_str = kwargs.pop("format", None)

    prefix = kwargs.pop("prefix", "")

    if not kwargs:  # If only format was provided
        return

    file_path = DEFAULT_VARIABLES_FILE

    # Read existing variables using read_variables function
    existing_vars = read_variables(file_path)

    # Update with new variables, applying format if specified
    if format_str:
        formatted_vars = {
            prefix + str(k): format_string(v, format_str) for k, v in kwargs.items()
        }
    else:
        formatted_vars = {prefix + str(k): str(v) for k, v in kwargs.items()}
    for name, value in formatted_vars.items():
        print(f"Saving variable: {name} = {value}")
    existing_vars.update(formatted_vars)

    # Write back to file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for name, value in existing_vars.items():
                f.write(f"{encode(name, value)}\n")
    except IOError as e:
        raise IOError(f"Could not write to file {file_path}: {e}")


def read_variables(file_path: Union[str, Path, None] = None) -> dict:
    """
    Read variables from a key=value text file into a dictionary.

    Parameters
    ----------
    file_path : str, Path, or None, default None
        Path to the key=value text file. If None, uses DEFAULT_VARIABLES_FILE.

    Returns
    -------
    dict
        Dictionary of variable names and values

    Examples
    --------
    >>> vars_dict = read_variables()  # Uses default file
    >>> print(vars_dict["accuracy"])
    """
    if file_path is None:
        file_path = DEFAULT_VARIABLES_FILE

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Variables file not found: {file_path}")

    variables = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                name, value = decode(line)
                variables[name] = value
    except IOError as e:
        raise IOError(f"Could not read file {file_path}: {e}")

    return variables

def cell_type_mapper(cell_type):
    match cell_type:
        case "23P":
            return "2/3P-IT"
        case "4P":
            return "4P-IT"
        case "5P-PT":
            return "5P-ET"
        case _:
            return cell_type
