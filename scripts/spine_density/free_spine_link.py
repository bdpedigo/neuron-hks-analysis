"""A neuroglancer link with the synapse-free spine structures of one cell.

Picks an excitatory column cell with many synapse-free spine structures on its
dendrite. Then it puts a sample of them, and a smaller sample of its synaptic spines
for comparison, as point annotations at their representative points.

    uv run python scripts/spine_density/free_spine_link.py
"""

# %%
import polars as pl
from caveclient import CAVEclient
from nglui.statebuilder import ViewerState

from analysis import Tables
from analysis.sources import DATASTACK, VERSION
from analysis.dendrites import EXCITATORY_TYPES, column_cells, spine_density_stash

N_FREE = 300
N_SYNAPTIC = 100
# The cell is at this quantile of the per-cell count of synapse-free dendrite spines,
# so that it has many of them but is not the most extreme cell.
CELL_QUANTILE = 0.9
SEED = 0

# %% PICK A CELL
cells = column_cells(EXCITATORY_TYPES)
placed = (
    spine_density_stash("spines")
    .load()
    .select("root_id", "structure_id", "method", "is_axon_vertex")
    .join(cells, on="root_id")
)
on_dendrite = placed.filter(~pl.col("is_axon_vertex"))
counts = (
    on_dendrite.group_by("root_id", "cell_type")
    .agg(
        n_free=(pl.col("method") == "nearest_l2").sum(),
        n_synaptic=(pl.col("method") == "synapse").sum(),
    )
    .sort("n_free")
)
root_id, cell_type = counts.row(int(CELL_QUANTILE * (counts.height - 1)))[:2]
print(counts.filter(pl.col("root_id") == root_id))

# %% THE POINTS
spines = (
    Tables["column_spines"]
    .lazy()
    .filter(pl.col("root_id") == root_id)
    .select("root_id", "structure_id", "rep_x", "rep_y", "rep_z", "size_nm3", "max_dt_nm")
    .collect()
    .join(on_dendrite.select("root_id", "structure_id", "method"), on=["root_id", "structure_id"])
    .with_columns(
        description=pl.format(
            "structure {}, {} µm³, max_dt {} nm",
            "structure_id",
            (pl.col("size_nm3") / 1e9).round(4),
            pl.col("max_dt_nm").round(0),
        )
    )
)
samples = {
    "no synapse": spines.filter(pl.col("method") == "nearest_l2").sample(N_FREE, seed=SEED),
    "synaptic": spines.filter(pl.col("method") == "synapse").sample(N_SYNAPTIC, seed=SEED),
}

# %% THE LINK
client = CAVEclient(DATASTACK, version=VERSION)
vs = ViewerState(client=client).add_layers_from_client().add_segments([root_id])
for (name, sample), color in zip(samples.items(), ["#ff3d00", "#00b0ff"]):
    vs = vs.add_points(
        sample.to_pandas(),
        name=name,
        point_column=["rep_x", "rep_y", "rep_z"],
        segment_column="root_id",
        description_column="description",
        data_resolution=[1, 1, 1],
        color=color,
    )
url = vs.to_url(shorten=True)
print(f"{root_id} ({cell_type}): {url}")
