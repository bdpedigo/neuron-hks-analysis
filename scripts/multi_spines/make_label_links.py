# %%
"""Neuroglancer links to label the multi-input spines whose inputs are all proofread.

A candidate is a spine on an excitatory cell (`post_broad_type`, any source) with 2 or
more presynaptic roots, every one of them proofread. The unitary classifier is not
used, because each candidate is checked by hand. A spine with a presynaptic synapse
off the skeleton axon is not a candidate: a dendrite or soma synapse, or a synapse with
no compartment on a root with a skeleton. The thalamic roots have no skeleton, so
their synapses stay. In the first link, the user tagged exactly these spines
`noncanonical` (user decision, 2026-10-30).

The old labels (10 states at v1412, `lines` layer) carry over to a candidate only if
the old synapses are exactly the synapses of the candidate. An old annotation with no
tag is not carried. Every other candidate goes into the links. There is no spatial
fallback: it attaches labels to a different object.

The tags are those of the old states. `singlehead` marks a true multi-input spine.

`data/tables/multi_spines/` holds `spines.parquet` (every candidate, its synapses and
the carried tags) and `links.csv`. They are not written again if they exist. A link in
`LABELED_LINKS` was made and labeled before this run: its saved state is in
`data/tables/multi_spines/states/`, and its spines go into no new link.
`scripts/multi_spines/freeze_labels.py` reads the labeled states back.

    uv run python scripts/multi_spines/make_label_links.py
"""

import json

import numpy as np
import pandas as pd
import polars as pl
from caveclient import CAVEclient
from nglui.statebuilder import ViewerState

from analysis import Tables
from analysis.colors import COMPARTMENT_PALETTE_MUTED_HEX
from analysis.io import TABLE_PATH
from analysis.sources import DATASTACK, MICRONS_SYNAPSE_RESOLUTION, VERSION

OUT_PATH = TABLE_PATH / "multi_spines"
STATES_PATH = OUT_PATH / "states"
OLD_ANNOTATIONS_PATH = TABLE_PATH / "v1412" / "multi_spine_annotations.parquet"
TAGS = ["singlehead", "y", "merge", "noncanonical", "nonspine"]
SPINES_PER_LINK = 100
# Make one link only and write nothing, to check the link before the rest.
FIRST_LINK_ONLY = False
# The check link of the first run (EE), labeled by the user before the axon filter.
LABELED_LINKS = [
    {
        "link": 0,
        "input_types": "EE",
        "url": "https://spelunker.cave-explorer.org/#!middleauth+https://globalv1.daf-apis.com/nglstate/api/v1/6749621439692800",
        "state_id": 5637135655239680,
    }
]
# Mesh opacity, so that points behind a mesh stay visible.
MESH_ALPHA = 0.9
BROAD_TYPE_CODES = {"excitatory": "E", "inhibitory": "I", "thalamic": "T"}

# The shader colors: the figure color of a multi-input spine for `singlehead`, one
# color per other tag, and a warning color for a point with more than one tag.
TAG_COLORS = {
    "singlehead": COMPARTMENT_PALETTE_MUTED_HEX["multi_spine"],
    "y": "#1f78b4",
    "merge": "#ff7f00",
    "noncanonical": "#a6d854",
    "nonspine": COMPARTMENT_PALETTE_MUTED_HEX["shaft"],
}
UNLABELED_COLOR = "#ffffff"
CONFLICT_COLOR = "#ff0000"
SYNAPSE_COLOR = "#00e5ff"


def label_shader(tags: list[str]) -> str:
    """A point shader for bool tags that colors each point by its tag.

    A copy of the shader in `scripts/h01/make_soma_links.py`, with one color per tag.
    """
    lines = [
        f'#uicontrol vec3 {tag}Color color(default="{TAG_COLORS[tag]}")' for tag in tags
    ]
    lines += [
        f'#uicontrol vec3 unlabeledColor color(default="{UNLABELED_COLOR}")',
        f'#uicontrol vec3 conflictColor color(default="{CONFLICT_COLOR}")',
        "#uicontrol float markerSize slider(min=0, max=20, default=8)",
        "void main() {",
        "  setPointMarkerSize(markerSize);",
        "  int n = 0;",
        "  vec3 color = unlabeledColor;",
    ]
    for tag in tags:
        # NOTE: float() reads a bool or a number property the same way.
        lines.append(
            f"  if (float(prop_{tag}()) > 0.5) {{ n += 1; color = {tag}Color; }}"
        )
    lines += ["  if (n > 1) { color = conflictColor; }", "  setColor(color);", "}"]
    return "\n".join(lines)


def synapse_key(synapse_ids: pl.Expr) -> pl.Expr:
    """The sorted synapse IDs of a spine, joined by `_`."""
    return (
        synapse_ids.list.sort().list.eval(pl.element().cast(pl.String)).list.join("_")
    )


# %% THE CANDIDATES
proofread_ids = (
    Tables["cell_selection"]
    .lazy()
    .filter(pl.col("is_proofread"))
    .select("pt_root_id")
    .collect()
)
spine_synapses = (
    Tables["proofread_pre_synapses"]
    .lazy()
    .filter(
        pl.col("post_broad_type") == "excitatory",
        pl.col("tag") == "spine",
        pl.col("structure_id").is_not_null(),
        pl.col("spine_n_pre_root_ids") >= 2,
    )
    # NOTE: not named x, y, z, because `y` is also a tag.
    .with_columns(
        (pl.col(f"ctr_pt_position_{c}") * resolution).alias(f"position_{c}")
        for c, resolution in zip("xyz", MICRONS_SYNAPSE_RESOLUTION)
    )
    .collect()
)
assert (
    spine_synapses["pre_pt_root_id"].is_in(proofread_ids["pt_root_id"].implode()).all()
)
spines = (
    spine_synapses.sort("synapse_id")
    .group_by("post_pt_root_id", "structure_id")
    .agg(
        pl.col("post_cell_type").first(),
        pl.col("spine_n_synapses").first(),
        pl.col("spine_n_pre_root_ids").first(),
        pl.col("pre_pt_root_id").n_unique().alias("n_proofread_pre_root_ids"),
        pl.col("synapse_id").alias("synapse_ids"),
        pl.col("pre_pt_root_id").alias("pre_pt_root_ids"),
        pl.col("pre_broad_type").cast(pl.String).alias("pre_broad_types"),
        pl.col("pre_cell_type").cast(pl.String).alias("pre_cell_types"),
        # One code per presynaptic root, such as "EI".
        pl.col("pre_broad_type")
        .filter(pl.col("pre_pt_root_id").is_first_distinct())
        .cast(pl.String)
        .replace_strict(BROAD_TYPE_CODES)
        .sort()
        .str.join("")
        .alias("input_types"),
        *[pl.col(f"position_{c}").mean() for c in "xyz"],
        off_axon=(
            pl.col("pre_compartment").is_in(["dendrite", "soma"])
            | (
                pl.col("pre_compartment").is_null()
                & (pl.col("pre_broad_type") != "thalamic")
            )
        ).any(),
    )
    # Every presynaptic root of the spine is proofread.
    .filter(pl.col("n_proofread_pre_root_ids") == pl.col("spine_n_pre_root_ids"))
    .with_columns(synapse_key=synapse_key(pl.col("synapse_ids")))
    .sort("input_types", "post_pt_root_id", "structure_id")
)
# A spine whose inputs are all proofread has only synapses in this table.
assert (spines["synapse_ids"].list.len() == spines["spine_n_synapses"]).all()
assert spines["synapse_key"].is_unique().all()
print(
    f"{spines['off_axon'].sum()} spines with a presynaptic synapse off the skeleton axon dropped"
)
spines = spines.filter(~pl.col("off_axon")).drop("off_axon")
print(
    f"{len(spines):,} candidate spines on {spines['post_pt_root_id'].n_unique():,} cells"
)
print(spines.group_by("input_types").len().sort("len", descending=True))

# %% THE OLD LABELS
old = pl.read_parquet(OLD_ANNOTATIONS_PATH).select(
    pl.col("post_pt_root_id").alias("old_post_pt_root_id"),
    "component_id",
    *[pl.col(tag).fill_null(False).cast(pl.Boolean) for tag in TAGS],
)
old_synapses = (
    Tables["old_synapse_components"]
    .lazy()
    .filter(
        pl.col("post_pt_root_id").is_in(old["old_post_pt_root_id"].unique().implode())
    )
    .select(
        pl.col("index").alias("synapse_id"),
        pl.col("post_pt_root_id").alias("old_post_pt_root_id"),
        "component_id",
    )
    .collect()
    .group_by("old_post_pt_root_id", "component_id")
    .agg(synapse_ids=pl.col("synapse_id"))
)
old = old.join(old_synapses, on=["old_post_pt_root_id", "component_id"], how="left")
assert old["synapse_ids"].is_not_null().all(), "an old annotation has no synapses"
old = old.with_columns(
    synapse_key=synapse_key(pl.col("synapse_ids")), tagged=pl.any_horizontal(TAGS)
)
assert old["synapse_key"].is_unique().all()

matched = old.join(spines.select("synapse_key"), on="synapse_key", how="semi")
print(
    f"{len(old)} old annotations: {len(matched)} match a candidate "
    f"({matched['tagged'].sum()} tagged), {len(old) - len(matched)} do not"
)

# %% THE CANDIDATES TO LABEL
carried = matched.filter("tagged").select(
    "synapse_key", "old_post_pt_root_id", "component_id", *TAGS
)
spines = spines.join(carried, on="synapse_key", how="left").with_columns(
    to_label=pl.col("singlehead").is_null()
)
print(
    f"{(~spines['to_label']).sum()} candidates carry an old label, "
    f"{spines['to_label'].sum()} are to label"
)
to_label = spines.filter("to_label")

# The spines of a labeled link stay in that link.
labeled = []
for labeled_link in LABELED_LINKS:
    state = json.loads((STATES_PATH / f"{labeled_link['state_id']}.json").read_text())
    (layer,) = [layer for layer in state["layers"] if layer["name"] == "spines"]
    keys = [annotation["description"] for annotation in layer["annotations"]]
    labeled.append(
        pl.DataFrame({"synapse_key": keys}).with_columns(
            link=pl.lit(labeled_link["link"], pl.Int64)
        )
    )
labeled = pl.concat(labeled).join(
    to_label.select("synapse_key"), on="synapse_key", how="semi"
)
to_label = to_label.join(labeled, on="synapse_key", how="anti")
print(
    f"{len(labeled)} of them are in a labeled link, {len(to_label)} go into new links"
)

# New links per input types, at most `SPINES_PER_LINK` spines each.
link_ids = []
link_start = max(labeled_link["link"] for labeled_link in LABELED_LINKS) + 1
for _, group in to_label.group_by("input_types", maintain_order=True):
    n_links = int(np.ceil(len(group) / SPINES_PER_LINK))
    link_ids.append(link_start + np.arange(len(group)) * n_links // len(group))
    link_start += n_links
to_label = to_label.with_columns(
    link=pl.Series(np.concatenate(link_ids), dtype=pl.Int64)
)
link_of_spine = pl.concat([labeled, to_label.select("synapse_key", "link")])
spines = spines.join(link_of_spine, on="synapse_key", how="left")

# %% THE LINKS
client = CAVEclient(DATASTACK, version=VERSION)
for name in ["spines.parquet", "links.csv"]:
    if (OUT_PATH / name).exists() and not FIRST_LINK_ONLY:
        raise FileExistsError(f"{OUT_PATH / name} exists: delete it to make new links")

links = [
    {**labeled_link, "n_spines": int((labeled["link"] == labeled_link["link"]).sum())}
    for labeled_link in LABELED_LINKS
]
for (link,), group in to_label.group_by("link", maintain_order=True):
    points = group.with_columns(
        segments=pl.concat_list(pl.col("post_pt_root_id"), pl.col("pre_pt_root_ids")),
        position=pl.concat_list("position_x", "position_y", "position_z"),
    ).to_pandas()
    synapses = (
        group.select("synapse_ids", "pre_cell_types")
        .explode("synapse_ids", "pre_cell_types")
        .rename({"synapse_ids": "synapse_id", "pre_cell_types": "pre_cell_type"})
        .join(
            spine_synapses.select(
                "synapse_id", "position_x", "position_y", "position_z"
            ),
            on="synapse_id",
        )
        .with_columns(position=pl.concat_list("position_x", "position_y", "position_z"))
        .to_pandas()
    )
    vs = ViewerState(client=client, capabilities="main")
    vs = vs.add_layers_from_client(alpha_3d=MESH_ALPHA)
    # The link opens on the segments of the first spine. A move to a point shows the
    # segments of that point (the annotation layer swaps the visible segments).
    first = points["segments"].iloc[0]
    all_segments = np.unique(np.concatenate(points["segments"].to_list()))
    vs = vs.add_segments(all_segments, visible=np.isin(all_segments, first))
    vs = vs.add_points(
        points,
        name="spines",
        data_resolution=[1, 1, 1],
        point_column="position",
        segment_column="segments",
        description_column="synapse_key",
        tags=TAGS,
        shader=label_shader(TAGS),
    )
    vs = vs.add_points(
        synapses,
        name="synapses",
        data_resolution=[1, 1, 1],
        point_column="position",
        description_column="pre_cell_type",
        linked_segmentation=False,
        color=SYNAPSE_COLOR,
        swap_visible_segments_on_move=False,
    )
    links.append(
        {
            "link": link,
            "input_types": group["input_types"][0],
            "url": vs.to_url(shorten=True),
            "n_spines": len(group),
        }
    )
    print(
        f"link {link} ({links[-1]['input_types']}, {len(group)} spines): {links[-1]['url']}"
    )
    if FIRST_LINK_ONLY:
        break

# %%
if not FIRST_LINK_ONLY:
    OUT_PATH.mkdir(exist_ok=True)
    spines.write_parquet(OUT_PATH / "spines.parquet")
    pd.DataFrame(links).to_csv(OUT_PATH / "links.csv", index=False)
    print(f"{len(links)} links, {spines['to_label'].sum()} spines to label: {OUT_PATH}")
