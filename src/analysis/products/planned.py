"""Tables that are planned but not built yet: a name and a description only, so
that the catalog can list them. Move a table to its own module when it is built."""

from .catalog import Tables

Tables.register(
    "proofread_pre_synapses",
    description=(
        "One row per synapse from a proofread cell, onto any root, with the columns of "
        "`column_post_synapses`. The distance to the post nucleus comes from the "
        "nucleus position, because most targets have no skeleton. The full synapse "
        "scan keeps these rows as it builds `post_aggregation`. It replaces the "
        "proofread half of the old `filtered_synapses`."
    ),
)
Tables.register(
    "post_aggregation",
    description=(
        "One row per postsynaptic root, for all neurons: the counts and proportions "
        "of spine, shaft and soma synapses and of single- and multi-input spines, the "
        "mean synapse size and the mean number of inputs per spine. It comes from one "
        "scan of all synapses. It replaces the old `post_synapse_aggregation`."
    ),
)
Tables.register(
    "multi_spine_annotations",
    description=(
        "One row per hand-checked multi-input spine annotation on the column cells, "
        "keyed by the synapse IDs on the spine. The old export is keyed by v1412 roots "
        "and old spine IDs, which do not carry over."
    ),
)
