"""Tables that are planned but not built yet: a name and a description only, so
that the catalog can list them. Move a table to its own module when it is built."""

from .catalog import Tables

Tables.register(
    "synapse_census_features",
    description=(
        "One row per mapped synapse onto a neuron root of the full run, with no "
        "autapses: `synapse_id`, `post_pt_root_id`, the domain label as `tag`, and "
        "the `structure_id` of its spine (null if it is on no spine). An unmapped "
        "synapse has no row. It holds only the run features: join it to "
        "`microns_synapses` on `synapse_id`. It is in the same 256 buckets as "
        "`microns_synapses`, and one scan of the deployment tables writes it with "
        "`spine_census_features`."
    ),
)
Tables.register(
    "spine_census_features",
    description=(
        "One row per spine structure with at least one synapse (no autapses) on a "
        "neuron root of the full run, keyed by (`root_id`, `structure_id`): "
        "`size_nm3`, `area_nm2`, `n_synapses`, `n_pre_root_ids`, "
        "`posterior_unitary`, `unitary_pred_label` and `is_multi`. No census table "
        "reads the other morphometry or the spectral features, so the table leaves "
        "them out, and the scan does not read `structure_features`. The same scan "
        "writes it with `synapse_census_features`."
    ),
)
Tables.register(
    "proofread_pre_synapses",
    description=(
        "One row per synapse from a proofread cell onto any root, with no "
        "autapses, and with the columns of `column_post_synapses`. As in the old "
        "`filtered_synapses`, the targets are not filtered. A target outside the "
        "full run has null run features (`tag`, `structure_id`, `spine_*`), and a "
        "target outside `cell_info` has null cell info. The distance to the post "
        "nucleus comes from the nucleus position, because most targets have no "
        "skeleton. It derives from `microns_synapses`, `synapse_census_features` and "
        "`spine_census_features`. It replaces the proofread half of the old "
        "`filtered_synapses`."
    ),
)
Tables.register(
    "post_aggregation",
    description=(
        "One row per neuron root: the counts and proportions of spine, shaft and "
        "soma synapses and of single- and multi-input spines, the mean synapse size "
        "and the mean number of inputs per spine. It derives from "
        "`synapse_census_features` and `spine_census_features`. It replaces the old "
        "`post_synapse_aggregation`."
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
