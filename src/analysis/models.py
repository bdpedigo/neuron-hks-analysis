"""The SSA synapse models: the training configuration that scripts share, and a
local predict that follows the meshmash-deployment domain prediction step.

A model is an MLflow sklearn folder, as `save_model` writes it. Any such folder
whose signature columns are in a features table can be scored with `predict`.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from .io import MODEL_PATH
from .products.sources import step_pin

FEATURE_PREFIXES = ("hks_", "curvature_", "normal_")

# NOTE: `distance_to_nucleus` is null for a root with no nucleus point. The
# random forest accepts missing values, so those rows stay in. See also the
# nucleus mask below.
# `prefixes` names the spectral columns a variant reads, and `extra_features` the
# columns after them.
HKS_PREFIXES = ("hks_",)
MODEL_SPECS = {
    "ssa_model_full": dict(
        prefixes=FEATURE_PREFIXES, extra_features=["distance_to_nucleus"], binary=False
    ),
    "ssa_model_no_nuc": dict(prefixes=FEATURE_PREFIXES, extra_features=[], binary=False),
    "ssa_model_binary": dict(prefixes=FEATURE_PREFIXES, extra_features=[], binary=True),
    "ssa_model_hks": dict(prefixes=HKS_PREFIXES, extra_features=[], binary=False),
    "ssa_model_hks_nuc": dict(
        prefixes=HKS_PREFIXES, extra_features=["distance_to_nucleus"], binary=False
    ),
    "ssa_model_hks_binary": dict(prefixes=HKS_PREFIXES, extra_features=[], binary=True),
}
# The learning curve, the glued variables and every downstream figure use this one.
MAIN_MODEL = "ssa_model_full"
# NOTE: a published model name is write-once in meshmash-deployment, so bump this
# before publishing a retrained model.
MODEL_VERSION = "v2"

RF_PARAMS = dict(class_weight="balanced", max_depth=15, n_estimators=500)


def trained_on(product: str) -> dict:
    """The pinned table of `product` (for example `domain_features`), as the
    deployment names it. A model records the table it is trained on, and the
    deployment's bind compares it with the table a run reads."""
    step_version, param_hash = step_pin(product)
    return {"product": product, "param_hash": param_hash, "step_version": step_version}


# The unitary-spine model (P13): is a putative multi-input spine one spine with a
# single head? The geometry columns come from `structure_morphometry`, the spectral
# means from `structure_features`.
UNITARY_GEOMETRY = [
    "fiedler_eval",
    "size_nm3",
    "area_nm2",
    "sphericity",
    "pca_val_1",
    "pca_val_2",
    "pca_val_3",
    "max_dt_nm",
    "mean_dt_nm",
    "bound_volume_um3",
]
UNITARY_MODEL_SPECS = {
    "unitary_hks": dict(prefixes=HKS_PREFIXES, extra_features=UNITARY_GEOMETRY),
    "unitary_spectral": dict(prefixes=FEATURE_PREFIXES, extra_features=UNITARY_GEOMETRY),
}
# NOTE: unlike RF_PARAMS, no class weights, as in the paper's model.
UNITARY_RF_PARAMS = dict(max_depth=15, n_estimators=500)

# NOTE: almost every training root has a nucleus, but many deployment roots do not
# (for example, orphan fragments). Without masking, the forest sees almost no missing
# distances and sends them down whichever branch is larger. So the distance is set
# to null on a fixed share of the training roots. The mask is by root, the same way
# a missing nucleus looks in the deployment.
NUCLEUS_MASK_FRACTION = 0.2
NUCLEUS_MASK_SEED = 0

# A row is a domain when it has this column: it is never null in `domain_features`,
# so it is null only for a synapse that maps to no domain.
DOMAIN_MARKER = "domain_n_vertices"


def select_spectral_columns(columns) -> list[str]:
    """The spectral feature columns among `columns`, in their table order."""
    return [c for c in columns if c.startswith(FEATURE_PREFIXES)]


def feature_columns(spec: dict, columns) -> list[str]:
    """The columns a variant reads: its spectral columns among `columns`, in table
    order, then its extra features."""
    return [c for c in columns if c.startswith(spec["prefixes"])] + list(
        spec["extra_features"]
    )


def nucleus_masked_roots(root_ids) -> np.ndarray:
    """The roots whose `distance_to_nucleus` is set to null for training."""
    roots = np.sort(pd.unique(np.asarray(root_ids)))
    return np.random.default_rng(NUCLEUS_MASK_SEED).permutation(roots)[
        : round(NUCLEUS_MASK_FRACTION * len(roots))
    ]


def make_model(random_state) -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS, n_jobs=-1, random_state=random_state)


def save_model(
    model, X, name, version=MODEL_VERSION, features_product="domain_features"
) -> Path:
    # Saved in the MLflow model format, which records this environment and the
    # feature signature. meshmash-deployment publishes it from this directory with
    # `just publish-model`; it refuses a model without `classes` and `trained_on`.
    # The folder name is also the published name. A rerun at the same MODEL_VERSION
    # replaces the local folder.
    import mlflow.sklearn
    from mlflow.models import infer_signature

    model_dir = MODEL_PATH / f"{name}_{version}"
    shutil.rmtree(model_dir, ignore_errors=True)  # save_model refuses a non-empty path
    mlflow.sklearn.save_model(
        model,
        str(model_dir),
        signature=infer_signature(X.head(5), model.predict_proba(X.head(5))),
        metadata={
            "classes": [str(label) for label in model.classes_],
            # The deployment table the training rows were joined from.
            "trained_on": trained_on(features_product),
        },
        # skops refuses tree internals unless named; this model is our own.
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    return model_dir


@dataclass(frozen=True)
class SSAModel:
    name: str
    estimator: RandomForestClassifier
    feature_columns: tuple[str, ...]
    classes: tuple[str, ...]
    accepts_missing: bool


def load_ssa_model(name_or_path: str | Path) -> SSAModel:
    """Load an MLflow sklearn model folder: a name under `MODEL_PATH`, or a path."""
    import mlflow.sklearn
    from mlflow.models import Model

    path = Path(name_or_path)
    if not path.exists():
        path = MODEL_PATH / str(name_or_path)
    estimator = mlflow.sklearn.load_model(str(path))
    feature_columns = tuple(Model.load(str(path)).signature.inputs.input_names())
    # The deployment's test: a model accepts missing values when it scores an
    # all-NaN row.
    try:
        estimator.predict_proba(
            pd.DataFrame(
                [[np.nan] * len(feature_columns)], columns=list(feature_columns)
            )
        )
        accepts_missing = True
    except ValueError:
        accepts_missing = False
    return SSAModel(
        name=path.name,
        estimator=estimator,
        feature_columns=feature_columns,
        classes=tuple(str(c) for c in estimator.classes_),
        accepts_missing=accepts_missing,
    )


def predict(
    model: SSAModel, features: pl.DataFrame, keys=("synapse_id", "root_id")
) -> pl.DataFrame:
    """`keys` of `features` with `posterior_<class>` and `pred_label`, by the
    rules of the deployment domain prediction step.

    The features are cast to float32, with nulls as NaN. A row is scored when it
    is a domain, and when it has no missing feature or the model accepts missing
    values. A row that is not scored has null predictions.
    """
    matrix = (
        features.select(pl.col(c).cast(pl.Float32) for c in model.feature_columns)
        .to_numpy()
        .astype(np.float32)
    )
    scored = features[DOMAIN_MARKER].is_not_null().to_numpy()
    if not model.accepts_missing:
        scored &= ~np.isnan(matrix).any(axis=1)
    posteriors = np.full((len(features), len(model.classes)), np.nan, dtype=np.float32)
    if scored.any():
        frame = pd.DataFrame(matrix[scored], columns=list(model.feature_columns))
        posteriors[scored] = model.estimator.predict_proba(frame)
    labels = np.asarray(model.classes, dtype=object)[
        np.argmax(np.nan_to_num(posteriors, nan=-1), axis=1)
    ]
    labels[~scored] = None
    return features.select(keys).with_columns(
        *[
            pl.Series(f"posterior_{name}", posteriors[:, i]).fill_nan(None)
            for i, name in enumerate(model.classes)
        ],
        pl.Series("pred_label", labels, dtype=pl.String),
    )
