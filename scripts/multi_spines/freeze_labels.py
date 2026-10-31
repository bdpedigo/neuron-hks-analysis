# %%
"""One-time freeze of the multi-input spine labels into `multi_spine_labels`.

`scripts/multi_spines/make_label_links.py` wrote the candidates and the links to
`data/tables/multi_spines/`. After the labeling, put the IDs of the saved states in
`STATE_IDS`. This script copies the states to `data/tables/multi_spines/states/` and
never overwrites a file there. It then joins the new tags (by the description, the
`synapse_key` of the spine) to the carried old tags, and writes
`multi_spine_labels.parquet` once.

    uv run python scripts/multi_spines/freeze_labels.py
"""

import json

import pandas as pd
import polars as pl
from caveclient import CAVEclient
from nglui.parser import StateParser

from analysis.io import TABLE_PATH

OUT_PATH = TABLE_PATH / "multi_spines"
STATES_PATH = OUT_PATH / "states"
LABELS_PATH = OUT_PATH / "multi_spine_labels.parquet"
TAGS = ["singlehead", "y", "merge", "noncanonical", "nonspine"]

# The saved states of the labeled links, one per link (`links.csv`).
STATE_IDS: list[int] = [
    5637135655239680,  # link 0, EE
    6467322500349952,  # link 1, EI
    6424444868755456,  # link 2, EI
    5389801105129472,  # link 3, ET
    6438691040395264,  # link 4, II
    6100150229925888,  # link 5, III
    6126785616412672,  # link 6, IT
    6047702954016768,  # link 7, TT
]

# %%
if LABELS_PATH.exists():
    raise FileExistsError(f"{LABELS_PATH} exists")
assert STATE_IDS, "put the saved state IDs in STATE_IDS"

STATES_PATH.mkdir(exist_ok=True)
state_client = CAVEclient()
for state_id in STATE_IDS:
    out = STATES_PATH / f"{state_id}.json"
    if not out.exists():
        out.write_text(json.dumps(state_client.state.get_state_json(state_id)))
states_csv = OUT_PATH / "states.csv"
if not states_csv.exists():
    pd.DataFrame({"state_id": STATE_IDS}).to_csv(states_csv, index=False)

# %%
frames = []
for state_id in STATE_IDS:
    state_json = json.loads((STATES_PATH / f"{state_id}.json").read_text())
    df = StateParser(state_json).annotation_dataframe(expand_tags=True)
    df = df.query("layer == 'spines'")
    # NOTE: nglui adds a column only for a tag that some point carries.
    frames.append(
        pl.DataFrame(
            {
                "synapse_key": df["description"].astype(str).to_list(),
                **{
                    tag: [value is True for value in df[tag]] if tag in df else False
                    for tag in TAGS
                },
            }
        ).with_columns(pl.lit(state_id).alias("state_id"))
    )
new = pl.concat(frames)
assert new["synapse_key"].is_unique().all(), "a spine is in more than one state"

spines = pl.read_parquet(OUT_PATH / "spines.parquet")
to_label = spines.filter("to_label").select("synapse_key")
# NOTE: the first link was labeled before the axon filter, so some of its points are
# no longer candidates.
unknown = new.join(to_label, on="synapse_key", how="anti")
print(f"{len(unknown)} labeled points are not candidates, and are dropped")
new = new.join(to_label, on="synapse_key", how="semi")
missing = to_label.join(new, on="synapse_key", how="anti")
assert missing.is_empty(), f"{len(missing)} candidates are in no state"

labels = (
    spines.join(new, on="synapse_key", how="left", suffix="_new")
    .with_columns(
        *[
            pl.when("to_label")
            .then(pl.col(f"{tag}_new"))
            .otherwise(pl.col(tag))
            .alias(tag)
            for tag in TAGS
        ],
        label_source=pl.when("to_label").then(pl.lit("new")).otherwise(pl.lit("old")),
    )
    .select(
        "post_pt_root_id",
        "structure_id",
        "synapse_key",
        "synapse_ids",
        "pre_pt_root_ids",
        "input_types",
        *TAGS,
        "label_source",
        "state_id",
        "old_post_pt_root_id",
        "component_id",
    )
)
untagged = labels.filter(~pl.any_horizontal(TAGS))
print(f"{len(labels)} spines, {len(untagged)} with no tag")
print(labels.group_by("label_source").agg(pl.len(), pl.col("singlehead").sum()))
labels.write_parquet(LABELS_PATH)
print(f"wrote {LABELS_PATH}")
