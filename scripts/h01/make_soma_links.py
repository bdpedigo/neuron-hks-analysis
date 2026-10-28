# %%
"""Neuroglancer links for a second H01 labeling round, near the soma.

The first round (meshrep/sandbox/h01_relabeling.py) sampled 25 random synapses per
cell, so few labels are soma. This round offers synapses near the nucleus of the
same cells. The link code is that of the first round; only the choice of synapses
differs. The synapses are chosen by distance alone, not by a model prediction:
- on a synapse that the deployment mapped, so it has features and predictions
- not yet labeled in the first round (`h01_labels`)
- within `RADIUS_NM` of the nucleus of its cell

Each cohort gets its own links and its own folder, so that the retrain and
validation rounds stay apart: `data/tables/h01_soma_{cohort}/` holds
`points.parquet` (the offered points) and `links.csv` (the cells of each link).
A folder that exists is not written again.

`TAG_SCHEME` picks how the links store tags. "bool" is the scheme of current
Spelunker: one `bool` property per tag, with a hotkey that toggles it, and a
shader that colors each point by its tag. "legacy" is the scheme of the first
round: `uint8` properties with a `tag` key. nglui (4.8.0 and later) parses both
into the same tag columns.
"""

import numpy as np
import pandas as pd
import polars as pl
from caveclient import CAVEclient
from nglui.statebuilder import ViewerState

from analysis import Tables
from analysis.colors import COMPARTMENT_PALETTE_MUTED_HEX
from analysis.io import TABLE_PATH
from analysis.products.deployment import scan_roots
from analysis.products.h01 import TAG_CATEGORIES
from analysis.sources import H01_DATASTACK, H01_SYNAPSES_PATH, cutoff, deployment_path

# The 26 soma labels of the retrain cohort are within 17.4 um of the nucleus.
RADIUS_NM = 20_000
N_PER_CELL = 25
CELLS_PER_LINK = 10
SEED = 8888
COHORTS = ["retrain", "validation"]
# Make one link of the first cohort, to check it before the rest.
FIRST_LINK_ONLY = False
TAG_SCHEME = "bool"
# Mesh opacity, so that points behind a mesh stay visible.
MESH_ALPHA = 0.95

# The shader colors: the figure colors for the three labels, one color for the
# other tags, and a warning color for a point with more than one tag.
LABEL_COLORS = {label: COMPARTMENT_PALETTE_MUTED_HEX[label] for label in ["spine", "shaft", "soma"]}
OTHER_COLOR = COMPARTMENT_PALETTE_MUTED_HEX["unknown"]
UNLABELED_COLOR = "#ffffff"
CONFLICT_COLOR = "#ff0000"


def label_shader(tags: list[str]) -> str:
    """A point shader for the "bool" scheme that colors each point by its tag."""
    lines = [f'#uicontrol vec3 {label}Color color(default="{c}")' for label, c in LABEL_COLORS.items()]
    lines += [
        f'#uicontrol vec3 otherColor color(default="{OTHER_COLOR}")',
        f'#uicontrol vec3 unlabeledColor color(default="{UNLABELED_COLOR}")',
        f'#uicontrol vec3 conflictColor color(default="{CONFLICT_COLOR}")',
        "#uicontrol float markerSize slider(min=0, max=20, default=6)",
        "void main() {",
        "  setPointMarkerSize(markerSize);",
        "  int n = 0;",
        "  vec3 color = unlabeledColor;",
    ]
    for tag in tags:
        tag_color = f"{tag}Color" if tag in LABEL_COLORS else "otherColor"
        # NOTE: float() reads a bool or a number property the same way.
        lines.append(f"  if (float(prop_{tag}()) > 0.5) {{ n += 1; color = {tag_color}; }}")
    lines += ["  if (n > 1) { color = conflictColor; }", "  setColor(color);", "}"]
    return "\n".join(lines)


# %%
cells = Tables["h01_cells"].collect().select("root_id", "cohort")
root_ids = cells["root_id"].to_list()
nuclei = Tables["h01_root_nuclei"].collect().filter(pl.col("root_id").is_in(root_ids))
assert nuclei["root_id"].is_unique().all() and len(nuclei) == len(cells)

synapses = (
    pl.scan_delta(str(H01_SYNAPSES_PATH))
    .filter(pl.col("post_pt_root_id").is_in(root_ids))
    .select(
        pl.col("id").alias("synapse_id"),
        pl.col("post_pt_root_id").alias("root_id"),
        *[pl.col(f"ctr_pt_position_{c}").alias(c) for c in "xyz"],
    )
    .collect()
)
mapped = (
    scan_roots(
        deployment_path("synapse_mapping", H01_DATASTACK), root_ids, as_of=cutoff("validation")
    )
    .select("root_id", "synapse_id")
    .unique()
    .collect()
)
labeled = Tables["h01_labels"].collect().select("synapse_id")

candidates = (
    synapses.join(mapped, on=["root_id", "synapse_id"])
    .join(labeled, on="synapse_id", how="anti")
    .join(nuclei, on="root_id")
    .join(cells, on="root_id")
    .with_columns(
        distance_to_nucleus=(
            (pl.col("x") - pl.col("nucleus_x")) ** 2
            + (pl.col("y") - pl.col("nucleus_y")) ** 2
            + (pl.col("z") - pl.col("nucleus_z")) ** 2
        ).sqrt()
    )
    .filter(pl.col("distance_to_nucleus") < RADIUS_NM)
    .sort("root_id", "synapse_id")
)
print(candidates.group_by("cohort").agg(n=pl.len(), n_roots=pl.col("root_id").n_unique()))

# %%
# Up to `N_PER_CELL` per cell, or all of a cell's candidates if it has fewer.
offered = (
    candidates.to_pandas()
    .sample(frac=1, random_state=SEED)
    .groupby("root_id")
    .head(N_PER_CELL)
    .sort_values(["root_id", "synapse_id"])
)
offered["ctr_pt_position"] = offered[["x", "y", "z"]].values.tolist()
offered = offered.rename(columns={"root_id": "post_pt_root_id"})
print(offered.groupby("cohort").size())

# %%
client = CAVEclient(H01_DATASTACK)

for cohort in COHORTS:
    out_path = TABLE_PATH / f"h01_soma_{cohort}"
    if out_path.exists():
        print(f"{out_path} exists: skipping {cohort}")
        continue
    syns = offered.query("cohort == @cohort")
    cohort_roots = np.sort(syns["post_pt_root_id"].unique())
    batches = np.array_split(cohort_roots, int(np.ceil(len(cohort_roots) / CELLS_PER_LINK)))
    links = []
    for i, batch in enumerate(batches):
        vs = ViewerState(client=client, capabilities="main" if TAG_SCHEME == "bool" else "legacy")
        vs = vs.add_layers_from_client(alpha_3d=MESH_ALPHA)
        # One cell shows at a time: the link opens on the first, and moving to a
        # point shows its cell (the annotation layer swaps the visible segments).
        vs = vs.add_segments(batch, visible=np.arange(len(batch)) == 0)
        vs = vs.add_points(
            syns.query("post_pt_root_id.isin(@batch)"),
            data_resolution=[1, 1, 1],
            point_column="ctr_pt_position",
            segment_column="post_pt_root_id",
            tags=TAG_CATEGORIES,
            shader=label_shader(TAG_CATEGORIES) if TAG_SCHEME == "bool" else None,
        )
        links.append(
            {
                "link": i,
                "url": vs.to_url(shorten=True),
                "root_ids": " ".join(map(str, batch)),
                "n_points": int(syns["post_pt_root_id"].isin(batch).sum()),
            }
        )
        print(f"{cohort} link {i}: {links[-1]['url']}")
        if FIRST_LINK_ONLY:
            break
    if FIRST_LINK_ONLY:
        break
    out_path.mkdir(parents=True)
    pl.from_pandas(
        syns[["synapse_id", "post_pt_root_id", "x", "y", "z", "distance_to_nucleus"]].rename(
            columns={"post_pt_root_id": "root_id"}
        )
    ).write_parquet(out_path / "points.parquet")
    pd.DataFrame(links).to_csv(out_path / "links.csv", index=False)
    print(f"{len(links)} links, {len(syns)} points: {out_path}")
