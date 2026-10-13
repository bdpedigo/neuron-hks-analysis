# %%
"""One-time export of the H01 label states (S12) into the repo.

The neuroglancer states are the original record of the H01 labels, and a state
link can stop working. This copies them to data/tables/h01/states/ and never
overwrites a file there, so the exported copies are the frozen record.
"""

import json
import shutil
from pathlib import Path

import pandas as pd
from caveclient import CAVEclient

from analysis.io import TABLE_PATH

H01_TABLE_PATH = TABLE_PATH / "h01"
STATES_PATH = H01_TABLE_PATH / "states"

# The Fig 6 validation labels: meshrep/sandbox/h01_validation.py, lines 96-110.
# Copies of these states are in ../h01-states (the ninth file there is unused).
VALIDATION_STATE_IDS = [
    6689773140836352,
    6514818184904704,
    4576276340801536,
    6559538470715392,
    4953136346169344,
    5184974507474944,
    4909523222396928,
    6729292711985152,
]
LOCAL_STATES_PATH = Path(__file__).resolve().parents[3] / "h01-states"

# The retraining labels: meshrep/clean_paper_code/retrain_h01.py, lines 12-24.
# These exist only on the global.daf-apis.com state server.
RETRAIN_STATE_IDS = [
    5543298467364864,
    6161862375243776,
    4781557964341248,
    5033287509082112,
    6269452279087104,
    5706877933977600,
    4582915726901248,
    5067407534587904,
]

STATES_PATH.mkdir(parents=True, exist_ok=True)
for state_id in VALIDATION_STATE_IDS:
    out = STATES_PATH / f"{state_id}.json"
    if not out.exists():
        shutil.copyfile(LOCAL_STATES_PATH / f"{state_id}.json", out)

state_client = CAVEclient()
for state_id in RETRAIN_STATE_IDS:
    out = STATES_PATH / f"{state_id}.json"
    if not out.exists():
        out.write_text(json.dumps(state_client.state.get_state_json(state_id)))

states = pd.DataFrame(
    [{"state_id": s, "cohort": "validation"} for s in VALIDATION_STATE_IDS]
    + [{"state_id": s, "cohort": "retrain"} for s in RETRAIN_STATE_IDS]
)
states_csv = H01_TABLE_PATH / "states.csv"
if not states_csv.exists():
    states.to_csv(states_csv, index=False)
print(f"{len(list(STATES_PATH.glob('*.json')))} states in {STATES_PATH}")
