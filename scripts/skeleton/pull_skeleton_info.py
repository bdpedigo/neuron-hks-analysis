#%%
import humanize
import ossify as osy
from upath import UPath

test_skeleton_path = "gs://csm-skel/osy/minnie65/skel/864691134885292282.osy"

print(humanize.naturalsize(UPath(test_skeleton_path).stat().st_size))

cell = osy.load_cell(
    test_skeleton_path
)

#%%

cell.skeleton.nodes

#%%
cell.annotations['post_syn'].nodes

# for each pre and post synapse we want to save out the
# - geodesic distance to root, 
# - the compartment prediction,
# - the distance to nucleus? or that happens elsewhere
# - the distance to tips? or did I even use this for anything
# - the primary branch ID? again did I even use this for anything
# - the soma relative centroid x,y,z (did this need the standard transform)