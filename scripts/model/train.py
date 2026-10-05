# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from caveclient import CAVEclient
from joblib import Parallel, delayed, dump, load
from nglui.parser import StateParser
from nglui.statebuilder import ViewerState
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from upath import UPath
from analysis import Tables

from analysis import save_variables, set_matplotlib_theme

# %%

datastack = "minnie65_phase3_v1"
base = "gs://bdp-ssa/meshmash-deployment"
out = "scratch_data/synapse_hks_features.parquet"
base = UPath(base)


# %%

joined = Tables["labeled_synapse_features"].lazy()

joined.collect_schema()
# %%

start_time = time.time()
result = Tables["labeled_synapse_features"].collect(engine="streaming")
end_time = time.time()
print("Collection time:", end_time - start_time)

# #%%
# roots = Tables["synapse_labels"].lazy().filter(
#     (pl.col("table_name") == "vortex_compartment_targets")
# ).group_by("root_id").agg(pl.len()).filter(pl.col("len") > 30).select(pl.col("root_id")).collect()['root_id'].to_list()

# result.filter(pl.col("root_id").is_in(roots))['hks_0'].is_not_null().mean()

#%%