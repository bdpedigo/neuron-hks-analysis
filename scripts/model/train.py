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
from analysis import domain_features_lf, synapse_to_domain_lf

from analysis import VERSION, get_label_table, save_variables, set_matplotlib_theme

label_table = get_label_table(version=VERSION)

# %%
synapse_label_table = label_table.filter(pl.col("target_id") != -1)
# assert synapse_label_table.index.is_unique

synapse_label_lf = synapse_label_table.lazy().rename(
    {"target_id": "synapse_id", f"pt_root_id_{VERSION}": "root_id"}
)
# %%

datastack = "minnie65_phase3_v1"
base = "gs://bdp-ssa/meshmash-deployment"
out = "scratch_data/synapse_hks_features.parquet"
base = UPath(base)


print(synapse_to_domain_lf.collect_schema())

# %%

import polars_io_tools  # noqa

# TODO performance here will degrade once we have more cloud data
# need to either accept that or include the hash in the join or something
# that might not even help based on root ID coverage
# TODO also doubt that the filtered_join is very effective here for the same reason

joined = (
    (
        synapse_label_lf.piot.filtered_join(
            synapse_to_domain_lf,
            on=["root_id", "synapse_id"],
            how="left",
            coalesce=False,
        )
    )
    .join(domain_features_lf, on=["root_id", "domain_id"], how="left")
    .drop("domain_id")
)

joined.collect_schema()
# %%

start_time = time.time()
result = joined.collect(engine="streaming")
end_time = time.time()
print("Collection time:", end_time - start_time)

#%%
roots = synapse_label_lf.filter(
    (pl.col("table_name") == "vortex_compartment_targets")
).group_by("root_id").agg(pl.len()).filter(pl.col("len") > 30).select(pl.col("root_id")).collect()['root_id'].to_list()

result.filter(pl.col("root_id").is_in(roots))['hks_0'].is_not_null().mean()
