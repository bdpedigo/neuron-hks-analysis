# %%
"""One-time export of the old multi-input spine label states (v1412) into the repo.

The 10 neuroglancer states are the original record of the old labels, and a state
link can stop working. This copies them to `data/tables/multi_spines/old_states/` and
never overwrites a file there. `make_label_links.py` carried these labels from the
cached copy `data/tables/v1412/multi_spine_annotations.parquet`, which was parsed from
the same states (meshrep `clean_paper_code/column_io_summary_from_mega.py`,
`get_multi_annotations`).

    uv run python scripts/multi_spines/export_old_states.py
"""

import json

import pandas as pd
from caveclient import CAVEclient

from analysis.io import TABLE_PATH

OLD_STATES_PATH = TABLE_PATH / "multi_spines" / "old_states"

# The input types and the target type of each state, as in `get_multi_annotations`.
STATES = {
    4811205330862080: "EE on E",
    6484872683061248: "EI on E",
    5527458345385984: "IT on E",
    5391743175360512: "II on E",
    5173236714176512: "ET on E",
    6098825721675776: "EE on I",
    5201463239245824: "EI on I",
    5047802907328512: "ET on I",
    4719068014706688: "II on I",
    6041454261567488: "IT on I",
}

OLD_STATES_PATH.mkdir(parents=True, exist_ok=True)
state_client = CAVEclient()
for state_id in STATES:
    out = OLD_STATES_PATH / f"{state_id}.json"
    if not out.exists():
        out.write_text(json.dumps(state_client.state.get_state_json(state_id)))

states_csv = OLD_STATES_PATH.parent / "old_states.csv"
if not states_csv.exists():
    pd.DataFrame({"state_id": list(STATES), "pairing": list(STATES.values())}).to_csv(
        states_csv, index=False
    )
print(f"{len(list(OLD_STATES_PATH.glob('*.json')))} states in {OLD_STATES_PATH}")
