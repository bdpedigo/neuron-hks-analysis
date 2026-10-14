#%%
# Checks of P8 `skeleton_synapse_features` on one cell. The product itself is
# built with `just materialize skeleton_synapse_features`.
import matplotlib.pyplot as plt
import polars as pl

from analysis.products.skeleton import load_skeleton, skeleton_synapse_features

root_id = 864691134885292282

cell = load_skeleton(root_id)
features = skeleton_synapse_features(cell)

print(features.group_by("side", "compartment").len().sort("side", "compartment"))

#%%
post = features.filter(pl.col("side") == "post", pl.col("compartment") != "axon")

fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, column in zip(axs, ["path_distance_to_nuc", "max_dist_to_tip"]):
    ax.hist(post[column].drop_nulls() / 1000, bins=50)
    ax.set_xlabel(f"{column} (µm)")
axs[0].set_ylabel("Post synapses")
fig.suptitle(f"{root_id}, dendrite and soma post synapses")
