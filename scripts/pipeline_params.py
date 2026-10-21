# %%
import json

from analysis import save_variables
from analysis.sources import deployment_path, step_pin

# The deployment writes the resolved parameters of the step that made the table
# next to it, so these are the values the pinned run used, defaults included.
with (deployment_path("domain_features") / "_meshmash" / "identity.json").open("r") as f:
    identity = json.load(f)

assert identity["step"] == "condensed_spectral"
assert (identity["step_version"], identity["param_hash"]) == step_pin("domain_features")
params = identity["params"]

# %%
save_variables(
    params_max_eigenvalue=params["max_eigenvalue"],
    params_t_max=params["t_max"],
    params_t_min=params["t_min"],
    params_mollify_factor=params["mollify_factor"],
    format="sci{:.0e}",
)
# The meshes are in nm, so this is vertices per nm^2.
save_variables(
    params_simplify_target_density=params["simplify_target_density"],
    format="sci{:.2e}",
)
save_variables(
    params_max_overlap_neighbors=params["max_overlap_neighbors"],
    params_max_vertex_threshold=params["max_vertex_threshold"],
    params_min_vertex_threshold=params["min_vertex_threshold"],
    params_target_vertices=params["target_vertices"],
    # nm to microns
    params_overlap_distance=params["overlap_distance"] // 1000,
    format="{:,}",
)
save_variables(
    params_n_components=params["n_components"],
    params_n_scales=params["n_scales"],
    params_simplify_agg=params["simplify_agg"],
)
save_variables(
    params_distance_threshold=params["distance_threshold"],
    format="{:.0f}",
)

# The old pipeline simplified by a fixed reduction fraction
# (`params_simplify_target_reduction`). This run targets a vertex density, so
# that variable has no value here.
assert params["simplify_target_reduction"] is None
