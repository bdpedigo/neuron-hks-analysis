# %%
"""One-time export of the unitary-spine label states (S11) into the repo.

The neuroglancer states are the original record of the labels, and a state link
can stop working. This copies them to data/tables/unitary_spines/states/ and
never overwrites a file there, so the exported copies are the frozen record.

Each state holds the putative multi-input spines of 10 cells of one excitatory
type (meshrep/sandbox/sample_cells_for_multi_validation.py), one point per spine
in the `synapse_points` layer, tagged by hand.
"""

import json

import pandas as pd
from caveclient import CAVEclient

from analysis.io import TABLE_PATH

UNITARY_TABLE_PATH = TABLE_PATH / "unitary_spines"
STATES_PATH = UNITARY_TABLE_PATH / "states"

# meshrep/sandbox/sample_cells_for_multi_validation.py, lines 371-379.
STATE_IDS = [
    6127783213268992,
    4650826252419072,
    5717498879016960,
    4856400298639360,
    5551394391064576,
    5020338042699776,
    5659063231709184,
]

STATES_PATH.mkdir(parents=True, exist_ok=True)
state_client = CAVEclient()
for state_id in STATE_IDS:
    out = STATES_PATH / f"{state_id}.json"
    if not out.exists():
        out.write_text(json.dumps(state_client.state.get_state_json(state_id)))

states_csv = UNITARY_TABLE_PATH / "states.csv"
if not states_csv.exists():
    pd.DataFrame({"state_id": STATE_IDS}).to_csv(states_csv, index=False)
print(f"{len(list(STATES_PATH.glob('*.json')))} states in {STATES_PATH}")
