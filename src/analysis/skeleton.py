import numpy as np
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

from meshmash import graph_to_adjacency, subset_mesh_by_indices


def mask_dendrite_by_client_skeleton(mesh, root_id, client, distance_threshold=5):
    skel = client.skeleton.get_skeleton(root_id, output_format="dict")
    skeleton_vertices = skel["vertices"]
    skeleton_edges = skel["edges"]

    nbrs = NearestNeighbors(n_neighbors=1).fit(skeleton_vertices)
    _, indices = nbrs.kneighbors(mesh[0])

    skel_compartment = skel["compartment"].copy()
    skel_compartment = np.vectorize({1: "soma", 2: "axon", 3: "dendrite"}.get)(
        skel_compartment
    )

    adj = graph_to_adjacency((skeleton_vertices, skeleton_edges))

    dendrosoma_indices = np.where(
        (skel_compartment == "soma") | (skel_compartment == "dendrite")
    )[0]

    hops_to_dendrosoma = dijkstra(
        adj, indices=dendrosoma_indices, min_only=True, directed=False
    )

    is_dendrosoma = hops_to_dendrosoma <= distance_threshold

    mesh_is_dendrosoma = is_dendrosoma[indices.flatten()]

    mesh = subset_mesh_by_indices(mesh, mesh_is_dendrosoma)

    return mesh
