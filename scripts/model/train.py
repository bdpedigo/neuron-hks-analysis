# %%
import time

from analysis import MODEL_PATH, Tables, save_variables
from analysis.models import (
    MODEL_SPECS,
    MODEL_VERSION,
    NUCLEUS_MASK_FRACTION,
    RF_PARAMS,
    feature_columns,
    make_model,
    nucleus_masked_roots,
    save_model,
    select_spectral_columns,
)
from analysis.products.features import MIN_ROOT_LABELS

# NOTE: with RETRAIN off, a variant whose model folder exists is skipped, not
# trained again. The folders at MODEL_VERSION are what the deployment publishes.
# The learning curve is in learning_curve.py.
RETRAIN = False

# %%

joined = Tables["model_inputs"].lazy()

joined.collect_schema()
# %%

start_time = time.time()
result = Tables["model_inputs"].collect()
end_time = time.time()
print("Collection time:", end_time - start_time)

# %%

spectral_columns = select_spectral_columns(result.columns)

# rows without domain features (not yet processed by the meshmash deployment) can't
# be used for training or evaluation. Every variant trains on these same rows.
training_table = result.drop_nulls(subset=spectral_columns).to_pandas().set_index(
    "label_key"
)
sample_weight = training_table["sample_weight"].astype(float)

# Synapse rows have a `synapse_id`; point rows (`point_labels`) have a `point_id`.
is_synapse = training_table["synapse_id"].notna()
save_variables(model_n_training_synapses=int(is_synapse.sum()), format="{:,}")
save_variables(model_n_training_points=int((~is_synapse).sum()), format="{:,}")
save_variables(params_min_root_labels=MIN_ROOT_LABELS)

# The nucleus mask: see `analysis.models`.
save_variables(params_nucleus_mask_fraction=NUCLEUS_MASK_FRACTION)
nucleus_masked = training_table["root_id"].isin(
    nucleus_masked_roots(training_table["root_id"])
)


def make_xy(spec):
    X = training_table[feature_columns(spec, spectral_columns)]
    if "distance_to_nucleus" in X:
        X = X.assign(distance_to_nucleus=X["distance_to_nucleus"].mask(nucleus_masked))
    y = training_table["tag"]
    if spec["binary"]:
        # `soma_spine` is already `spine` in the label table, so this merges only
        # soma and shaft.
        y = y.where(y == "spine", "not_spine")
    return X, y


save_variables(
    params_rf_max_depth=RF_PARAMS["max_depth"],
    params_rf_n_estimators=RF_PARAMS["n_estimators"],
)


# %% train and save each variant on all available synapses

for name, spec in MODEL_SPECS.items():
    if not RETRAIN and (MODEL_PATH / f"{name}_{MODEL_VERSION}").exists():
        print(f"{name}: the {MODEL_VERSION} model exists")
        continue
    X, y = make_xy(spec)
    currtime = time.time()
    model = make_model(random_state=0)
    model.fit(X, y, sample_weight=sample_weight)
    model_dir = save_model(model, X, name)
    print(
        f"{name}: {time.time() - currtime:.1f} s, {X.shape[1]} features, "
        f"classes {list(model.classes_)}, saved to {model_dir}"
    )
