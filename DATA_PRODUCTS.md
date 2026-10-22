# Data products for the paper refactor

Goal: rebuild every data product behind a **data panel** in the paper (`../ssa/main.tex`) on the new sources. The new sources are the meshmash-deployment Delta tables, CAVE v1822 and the `csm-skel` skeletons. Explanatory and one-off schematic panels are out of scope (they are listed at the end).

This file is organized **by product**, because a product is the stable unit: scripts change, but a figure always reads the same table. Each source has an ID (`S#`) and each product has an ID (`P#`). The figure index at the bottom maps panels to IDs.

Scope labels used below:

- **all**: every synapse or neuron in the dataset (about 208M synapses onto about 75k neurons in the old run).
- **selection**: synapses with `pre ∈ proofread` or `post ∈ column` (5.3M rows in the old run).
- **column**: synapses or spines onto the about 1.3k column cells.
- **labeled**: the training label set (about 137k synapses on about 6k roots).
- **validation**: 3,300 synapses on the 132 held-out cells.

## Fields for each source and product

- **Origin**: where the data comes from. For a source, this is a CAVE table or view, a deployment table, a neuroglancer state or a manual file. For a product, this is the list of `S#` / `P#` inputs.
- **Pin**: what makes a read repeatable. This is a materialization version, a timestamp, an `as_of` cutoff or a parameter hash. A product's pin is the pins of its inputs plus its own parameters.
- **Cache**: the local path. The registry derives it from the origin and the pin. Do not write cache paths by hand.
- **Kind**: one of three values.
  - **cloud**: the data stays in GCS and is read with its pin each time. No local copy of the whole table exists.
  - **rebuildable**: the local cache is safe to delete. The registry rebuilds it from the origin and the pin.
  - **frozen**: the local file is the record, because the origin cannot give the same data again. Keep it in a versioned location and never delete it.
- **The local caches are the source of truth.** Each CAVE pull runs once at v1822. Its cache is then the record for later readers. At submission, every local cache becomes frozen. The rebuildable kind matters only during development.

---

## Conventions

- [ ] **One configuration file** holds the origin, pin and kind of every source. The `Tables` registry is the only way that scripts load data. Scripts move over one at a time.
  - Partly done: `products/sources.py` holds the deployment origins and pins. It has no kind field yet, and only `train.py` reads through the registry.
- **Product scripts and figure scripts are separate.** Product scripts read the cloud and write local caches. Figure scripts read only local caches.
- **Glued variables come only from scripts.** The scripts write `glued_variables.tex` in this repo (`save_variables`). The paper reads that whole file. Do not edit or copy single values by hand.
  - [ ] Point the paper at the generated file. Do this when every key that `ssa/main.tex` uses has a generator here. Now 45 keys of `ssa/glued_variables.tex` have no generator, from scripts that are not ported yet (for example `profile_*`, `unitary_spines_*`, `column_summary_*`).
  - `save_variables` never removes a key. Before the switch, delete the keys that no script writes any more, for example `params_simplify_target_reduction`.
- [ ] **No loads at import time.** Remove the module-level `load_neuron_info` call in `src/analysis/tables.py` (still at line 62).
- **Join on stable IDs.**
  - `synapse_id` is stable. Use it to join synapse-level data across versions.
  - Use the nucleus ID (from the soma table) to identify a cell across versions. Take `root_id` from the v1822 synapse or nucleus table. Only 582 of 1,333 column cells keep the same root ID between v1412 and v1822.
  - `structure_id` is valid only inside one pinned S4 snapshot. Never store it as a key that must survive a re-run.
- **Keep all columns.** Do not drop columns from products or deployment tables. The plan only notes columns that no figure uses.
- **Pin deployment tables with `as_of`, not with a Delta version.**
  - `as_of` is a cutoff on `attempt_id`.
  - The deployment sets `delta.logRetentionDuration` to 24 hours and `delta.deletedFileRetentionDuration` to 6 hours. It runs OPTIMIZE and VACUUM on a schedule. So a Delta version number stops working after about a day.
  - The tables are append-only, and the latest attempt per root wins at read time. OPTIMIZE rewrites files but keeps every row. So `attempt_id <= as_of`, applied before the latest-attempt filter, gives the same rows at any later time.
  - [x] Put each `as_of` in the configuration file and in the cache key. Then a cache rebuilds only after a deliberate change to `as_of`.
    - `STEP_PINS` in `products/sources.py` holds the path of each deployment table (step version and `p-…` hash). Every datastack shares it.
    - `CUTOFFS` in `products/sources.py` holds the named cutoffs. A product reads all of its tables at the cutoff of its newest input. `pin_params` puts the cutoff and the paths in the cache key.
    - One table can have more than one cutoff. P2 keeps the `features` cutoff when the full run appends rows, so P2 and the models trained on it keep their keys.
    - A product whose table or cutoff is not in the configuration yet is `unpinned` in `just status`, and a read of it raises `UnpinnedError`.
  - Set `as_of` after a job finishes. An attempt that starts before `as_of` and commits after it appears only in later reads.
  - Two events break the pin: a deletion of old attempts, or a move of the table to a new `p-…` path. A new path means new parameters, so that is a deliberate change.
  - A product built from deployment tables without an `as_of` is frozen, not rebuildable. For example, the full run can process the 6,010 labeled roots again. Their new attempts then win, and an unpinned P2 changes.
- [ ] **Store frozen files in a versioned location** (git LFS, or a write-once prefix in the bucket). Now they exist only on the laptop. `git ls-files data` shows only `ctype_hues.pkl`, and the LFS rules in `data/.gitattributes` name old file paths that no longer exist.
- **Stash small script tables with `Stash`** (`src/analysis/stash.py`), for example fold scores. Its path is `data/stash/{name}/{key}.parquet`. The key is a hash of the `cache_key` of each input table and of the params that the script passes. So a new input or param gives a new path, and a script never reads a result of old inputs. The compute code stays in the script: `if not s.exists(): s.save(...)`, then `s.load()`.
  - Pass cached tables as inputs. A view such as `model_inputs` has no `cache_key`, so pass the cached tables that it reads.
  - The key does not see code. After a change to the compute, bump a version param.
  - The stash is rebuildable. Delete `data/stash/` to clear it.
- [x] **Tooling.** If the cache of a registered table is missing, `just materialize NAME` builds it. `--force` rebuilds it. `just status` lists each registered table with its kind and cache state. It also lists the other files in the version folder: stale cache keys, interrupted builds, and files not in the registry. Product code is in `src/analysis/products/`, and the CLI is `scripts/tables.py`.
  - [ ] Add the frozen kind. The registry kinds are now `materialized`, `cached build` and `uncached`. After the frozen kind exists, `status` must never mark a frozen file as an orphan.

---

## Sources (not built here, only pinned)

### Cloud and CAVE sources

| ID | Origin | Pin | Cache | Kind | Scope | Status |
|---|---|---|---|---|---|---|
| S0 | CAVE `synapses_pni_2`, as the Delta export at `gs://mat_dbs/public/deltalake_exports/minnie65_phase3_v1/v1822/synapses_pni_2/post_pt_root_id` | v1822 (in the path) | none (selection rows go to P9) | cloud | all | exists |
| S1 | CAVE `minnie65_public`: view `aibs_cell_info`, table `proofreading_status_and_strategy` | v1822 | `data/tables/v1822/{aibs_cell_info,proofreading_status_and_strategy}.parquet` | rebuildable | all cells | cached |
| S2 | Deployment tables `synapse_mapping`, `vertex_domains`, `domain_features` at `gs://bdp-ssa/meshmash-deployment/minnie65_phase3_v1/…` | step version + `p-…` hash (in the path) + `as_of`. V2: `synapse_mapping` s2, the spectral tables s4, cutoff `features` | none (labeled rows go to P2) | cloud | the test cohort now (job `test_cohort`, 6,378 of 6,379 roots), then **all** after the full run | partial |
| S3 | Deployment tables `domain_predictions` (full `posterior_<class>`) + `synapse_predictions` (argmax summary only) | as S2, and the `p-…` hash includes the P3 model URI | none | cloud | validation roots now (job `microns_validation`, 132 roots), then **all** after the full run | partial. The model is P3 `gs://bdp-ssa/meshmash-deployment/models/ssa_model_full_v1` |
| S4 | Deployment tables `structure_morphometry` + `vertex_structures` (spines) | as S2 | none | cloud | all | not run (50-root `spine_trial` only) |
| S5 | Skeletons at `gs://csm-skel/osy/minnie65/skel/{root}.osy` | root ID. Make sure that files are not overwritten in place. | none (fields go to P8) | cloud | proofread + column cells | exists |
| S6 | CAVE annotation tables `vortex_compartment_targets`, `bdp_synapse_compartment_labels`, `bdp_point_compartment_labels` | v1822 timestamp, for both annotations and segmentation (`live=False`, the default, in `products/labels.py`) | P1 | rebuildable | labeled | exists |
| S7 | CAVE `synapse_target_structure` (NEURD) | v1822 | P4 | rebuildable | validation | exists |
| S8 | Strawman comparison outputs at `gs://bdp-ssa/comparisons/{shiny-wolf-message,…-strawman}` | run name (in the path) | none | cloud | strawman neurons | exists, re-run likely |

### Manual label sources

The original records of these labels are neuroglancer states or notebooks. A state link can stop working, so the exported table is the record. All of them are frozen.

| ID | Origin | Pin | Cache | Scope | Status |
|---|---|---|---|---|---|
| S9 | Validation labels from the labeling notebook `BD_250519_validation_set.ipynb`. The files carry `synapse_id` and `post_nucleus_id`. | none (one-time export) | `data/tables/validation/{exc,inh}_sample_synapses_labeled.csv`, `validation_ids_used.csv` (nucleus IDs) | validation | cached |
| S10 | Checked multi-input spines from 10 neuroglancer states (`get_multi_annotations`) | none (one-time export) | `data/tables/v1412/multi_spine_annotations.parquet` | column | cached, keyed by v1412 IDs |
| S11 | Single-spine labels from 7 neuroglancer states + `cell_type_selection_for_multis.csv` (`meshrep/sandbox/sample_cells_for_multi_validation.py`) | none (one-time export) | not exported yet | labeled spines | todo |
| S12 | H01 labels from 16 neuroglancer states (`scripts/h01/export_h01_states.py`): 8 `validation` states (Fig 6) and 8 `retrain` states | H01 v1054 timestamp | `data/tables/h01/states/{state_id}.json`, `data/tables/h01/states.csv` | H01 labeled | cached |
| S13 | Deployment tables for H01 (`synapse_mapping`, `vertex_domains`, `domain_features`, `domain_predictions`) on the `h01_c3_flat` datastack, job `h01_test_cohort` | as S2 (the same `sN/p-…` segments and the same cutoff `features`) | none (rows go to P5) | cloud | the 160 H01 label roots | exists |

---

## Products

### Model and validation track

**P1 `labels`** (labeled, 5 MB)
- Origin: S6. Pin: v1822 timestamp (annotations and segmentation). Cache: `data/tables/v1822/labels/{hash}.parquet`. Kind: rebuildable.
- [x] Registered as `labels` in `products/labels.py` (cache key: `version`). `get_label_table(live=True)` queries annotations at the current time and skips the cache.
- [x] Validation cells are excluded. The exclusion reads the root IDs from P16. The cached table had 0 rows on the 132 validation roots and none of the 3,300 validation `synapse_id`s. The validation roots come from the S9 nucleus IDs, mapped to v1822 through S1.
- [x] Fixed the path in `get_validation_ids` (`data/tables/general/` → `data/tables/validation/`).

**P16 `validation_cells`** (validation, 132 rows)
- Origin: S9 nucleus IDs ⋈ CAVE `nucleus_detection_lookup_v1`. Pin: v1822. Cache: `data/tables/v1822/validation_cells/{hash}.parquet`. Kind: rebuildable.
- Columns: `nucleus_id`, `root_id` (v1822).
- [x] Registered as `validation_cells` in `products/labels.py` (cache key: `version`). `get_validation_ids` reads it.

**P2 `labeled_synapse_features`** (labeled, about 60 MB)
- Origin: P1 ⋈ S2. Pin: P1 + S2 (`as_of`). Cache: `data/tables/v1822/labeled_synapse_features/{hash}.parquet`. Kind: rebuildable. The cache key includes the `labels` cache key.
- [x] Built for s2 features (now stale).
- [x] `products/sources.py` points at the s3 paths, with `as_of` = 2026-10-12 04:13 UTC. The labeled job is complete, except for 1 root (a large merge that cannot be processed).
- [x] Rebuild: `just materialize labeled_synapse_features`.
- [x] Keep the three old s2 `labeled_synapse_features-*` files in `data/tables/v1822/` until P3 is retrained. The s2 paths are gone, so these files are the only copy of the features that `synapse_hks_model_on_bulk` was trained on.
- [x] P3 is retrained on s3. Delete or archive the three old s2 files.
- [x] V2: `products/sources.py` points at the s4 spectral tables, with the `features` cutoff (2026-10-21 18:45 UTC). The s3 cache stays as a stale key until the V1 caches are cleared.
- [x] After the build: check the share of label synapses with non-null features, and check the failed merge root for partial rows from a crashed attempt.
  - Result: 111,640 of 133,686 synapse labels (83.5%) have all features, on 6,009 roots. The other synapse labels are on 12,683 roots with fewer than 5 labels, which the job did not include. The failed root `864691135858491438` (17 labels) has no rows, so no partial attempt is in the table. 10 roots have features for only some of their synapses.

**P3 HKS synapse model**
- Origin: P2, by `scripts/model/train.py`. Pin: P2 + training parameters + random seed. Cache: `data/models/`. Kind: frozen, because S3 depends on this exact model. Keep it in MLflow.
- [x] Retrain on the s3 feature columns: 111,640 training synapses. `train.py` selects features by prefix (`hks_`, `curvature_`, `normal_`), so it uses the s3 columns without changes.
  - `distance_to_nucleus` is added locally, after P2. A new product `root_nuclei` (`products/nuclei.py`) holds every nucleus point on each root in nm. It comes from CAVE `nucleus_detection_lookup_v1` at v1822, and a root can have several rows. The uncached view `model_inputs` = P2 ⋈ `root_nuclei` adds the distance from `domain_x/y/z` to the **nearest** nucleus on the root. It is null for a root with no nucleus, and the random forest accepts the missing values.
  - [x] Build `root_nuclei` (`just materialize root_nuclei`, a CAVE query), then re-run the training.
- [x] Train six model variants. `MODEL_SPECS` in `src/analysis/models.py` defines them: `prefixes` picks the spectral columns, and `extra_features` the columns after them. All six use the same 111,640 rows and the same `RF_PARAMS`.

  | Variant | Features | Classes |
  |---|---|---|
  | `ssa_model_full` | 74 spectral + `distance_to_nucleus` | shaft, soma, spine |
  | `ssa_model_no_nuc` | 74 spectral | shaft, soma, spine |
  | `ssa_model_binary` | 74 spectral | not_spine, spine (soma and shaft merged) |
  | `ssa_model_hks` | 32 HKS | shaft, soma, spine |
  | `ssa_model_hks_nuc` | 32 HKS + `distance_to_nucleus` | shaft, soma, spine |
  | `ssa_model_hks_binary` | 32 HKS | not_spine, spine |

  - The three HKS variants were added on 2026-10-14, to measure what the curvature and normal features add. `train.py` has `RETRAIN = False`, so a rerun loads a variant whose folder exists and trains only the missing ones. The rerun that trained them left `glued_variables.tex` unchanged.

  - [x] Each variant saves to `data/models/{variant}_{MODEL_VERSION}`, for example `data/models/ssa_model_full_v1`. The folder name is also the published name. The `trained_on` metadata in the model records the feature table (`DOMAIN_FEATURES_VERSION` + `DOMAIN_FEATURES_PARAMS`). The random forest and the learning-curve sampling are seeded.
  - Published names are write-once. Bump `MODEL_VERSION` in `train.py` (`v1` → `v2`) for each retrained model that you publish.
  - [x] `ssa_model_full` is the main model (`MAIN_MODEL`). The learning curve, the glued variables, and every downstream figure use only `ssa_model_full`.
  - Result: learning curve weighted F1 for `ssa_model_full`: 0.959 at 1% of the data and 0.966 at 10%. The model without `distance_to_nucleus` had 0.958 and 0.965. The paper had 0.935 and 0.948.
  - [x] Run `train.py` once more to write the three `ssa_model_*_v1` folders. Then delete the folders from earlier runs in `data/models/`:
    - `synapse_hks_model-s3-p-e758527c9d981378` and `synapse_hks_model-s3-p-e758527c9d981378-distance_to_nucleus`
    - `synapse_hks_model-s3-p-e758527c9d981378-{full,no_nucleus,spine_binary}`
    - `ssa_model_{full,no_nuc,binary}-s3-p-e758527c9d981378`
- [x] Register the model for the deployment, so that S3 can run. In meshmash-deployment, run `just publish-model ../meshrep/neuron-hks-analysis/data/models/ssa_model_full_v1`. The published name defaults to the folder name, and the upload is write-once.
  - Published to `gs://bdp-ssa/meshmash-deployment/models/ssa_model_full_v1`. Content hash `m-d0631a8065374702`. A job spec gives this URI as `domain_predictions.model`. Only `ssa_model_full_v1` is published.
  - The deployment can run all three variants. Its domain prediction step computes `distance_to_nucleus` with the same rule as `root_nuclei` (meshmash-deployment `TASK-99`). The column is not in `domain_features`.
- [x] Panel outputs: learning curve (Fig S2), glued `model_n_training_synapses`, `model_sample_curve_*`, `params_rf_max_depth`, `params_rf_n_estimators`.
  - The last run of `train.py` wrote all of these from `ssa_model_full`.
  - The old `params_*` came from `meshrep/clean_paper_code/count_things.py`. `train.py` now writes the two `params_rf_*` variables from `RF_PARAMS`, with the same keys.
  - [x] The other `params_*` variables describe the `condensed_spectral` step, not the model. `scripts/pipeline_params.py` writes them.
    - The script reads the resolved parameters of the S2 run from `_meshmash/identity.json` in the `domain_features` table folder. It asserts that the step version and the parameter hash match `products/sources.py`.
    - Same keys as before: `params_max_eigenvalue`, `params_t_min`, `params_t_max`, `params_n_components`, `params_simplify_agg`, `params_max_overlap_neighbors`, `params_overlap_distance`, `params_max_vertex_threshold`, `params_min_vertex_threshold`, `params_mollify_factor`, `params_distance_threshold`.
    - New keys: `params_simplify_target_density`, `params_target_vertices`, `params_n_scales`.
    - Removed key: `params_simplify_target_reduction`. The S2 run simplifies to a vertex density, not a reduction fraction.
    - Changed values: `params_max_overlap_neighbors` 60,000 → 50,000, `params_overlap_distance` 20 → 10 microns, `params_min_vertex_threshold` 200 → 10.
    - The methods text in `ssa/main.tex` has `% TODO` notes where the description is out of date: simplification, mesh splitting, overlap, and the curvature and normal features.
    - [ ] Rewrite the simplification text in `ssa/main.tex`. It still uses `params_simplify_target_reduction`, which no script writes.

**P4 `validation_predictions`** (validation, 3,300 rows)
- Origin: S9 ⋈ P16 ⋈ P15 for those synapses ⋈ S7. Pin: S9 + S2 + S3 (`as_of`) + v1822. Cache: `data/tables/v1822/validation_predictions/{hash}.parquet`. Kind: rebuildable.
- [x] Registered as `validation_predictions` in `products/validation.py`. The cache key includes the `validation_labels` cache key, `VALIDATION_AS_OF` and the S2 and S3 versions and parameter hashes.
  - It is `validation_labels` ⋈ `synapse_posteriors(as_of=VALIDATION_AS_OF)` ⋈ the S7 NEURD labels. The NEURD columns are `neurd_id`, `neurd_value`, `neurd_detailed_label` and `neurd_label`. `neurd_label` is null for the two NEURD axon classes.
  - The posteriors join on `root_id` (the cell). So the one synapse that is not on its cell (`244492646`, tag `other`) has null posteriors. All other 3,299 synapses have a prediction.
  - NEURD labels 2,229 of the 3,300 synapses. 8 of them are axon classes.
- Map S9 forward on `synapse_id` (labels) and `post_nucleus_id` (cells). Do not use the root IDs stored in the files.
- The predictions come from the deployment (S2 + S3), not from a local run.
- [x] Registered the label side as `validation_labels` in `products/labels.py` (cache key: `version` + the `validation_cells` cache key). It holds S9 ⋈ P16 ⋈ the v1822 synapse table, 3,300 rows.
  - `root_id` is the cell (from `post_nucleus_id` through P16), and `post_pt_root_id` is the v1822 post root of the synapse. `on_cell` is true when the two are the same.
  - Result: all 3,300 `synapse_id`s exist at v1822. 3,299 are on their cell. The one synapse that is not (`244492646`) has the tag `other`, which the validation drops. 625 rows have a different post root ID than the one stored in the files.
  - The `point` column in the files is the synapse `ctr_pt_position` in voxels (4, 4, 40 nm), so the table keeps only `ctr_pt_position_{x,y,z}` in nm.
- [x] Run a deployment job for the 132 validation roots. Take the root list from P16. None of them was in `microns_top_labeled`. The full run also covers them.
  - The job is `microns_validation` in meshmash-deployment. Its `validation_cells.parquet` is a copy of the P16 cache.
  - Run `run-20261012-235436-628a`: all 132 roots passed all four steps. The last commit was at 2026-10-13 04:57 UTC. `VALIDATION_AS_OF` in `products/sources.py` is 05:00 UTC.
  - The output paths are the paths in `products/sources.py`: `domain_predictions/s1/p-96cea889064d4d41`, with S2 at the same paths as P2. Each root has one attempt.
- [x] Port `scripts/validation/validation.py`. It reads only `validation_predictions`.
  - The broad type comes from `label_set` (`exc` or `inh`), which matches the cell types exactly.
  - The precision-recall plot (not in the paper) now uses `precision_recall_curve`. The old loop failed on an empty threshold, and it did not threshold the predictions.
- [x] Port `meshrep/clean_paper_code/compare_to_neurd.py` to `scripts/validation/compare_to_neurd.py`. It reads only `validation_predictions`.
- [x] Delete the old validation draft after `quit()` in `scripts/model/train.py`, and the imports that only the draft used.
- [x] Panel outputs:
  - Fig 3B (`figures/validation/validation_accuracy_by_cell_type`), Fig S9 (`validation_confusion_matrices`), and the ROC curve.
  - Fig S3 (`figures/compare_to_neurd/compare_neurd_hks_by_cell_type`).
  - Glued `validation_*` and `compare_*`. New keys: `validation_n_synapses` (3,294) and `compare_n_synapses` (2,219).
- [x] `validation_exc_f1` is now written by `validation.py`. It is the excitatory weighted F1, the same value as `validation_exc_weighted_f1`.
- Result, old paper value → new value:

  | Variable | Old | New |
  |---|---|---|
  | `validation_total_accuracy` | 0.961 | 0.966 |
  | `validation_total_weighted_f1` | 0.961 | 0.966 |
  | `validation_total_macro_f1` | 0.969 | 0.973 |
  | `validation_exc_f1` / `validation_exc_weighted_f1` | 0.977 | 0.980 |
  | `validation_inh_weighted_f1` | 0.935 | 0.943 |
  | `validation_inh_macro_f1` | 0.925 | 0.932 |
  | `compare_hks_weighted_f1` | 0.963 | 0.969 |
  | `compare_neurd_weighted_f1` | 0.842 | 0.842 |

  The NEURD scores did not change, so the label side and the NEURD join agree with the old run.

**P5 `h01_predictions`** (H01 labeled, 3,961 rows)
- Origin: S12 ⋈ S13. Pin: S12 + S13 (the `features` cutoff for the features, the `predictions` cutoff for the predictions). Kind: rebuildable. Code: `products/h01.py`. Cache folder: `data/tables/h01_c3_flat/v1054/`.
- Two cohorts: `validation` (the Fig 6 states, 80 roots, 1,980 label synapses) and `retrain` (a later labeling round, 80 other roots, 1,981 label synapses). `meshrep/sandbox/h01_relabeling.py` drew the retrain cells, and `meshrep/clean_paper_code/retrain_h01.py` read the states back.
- [x] Run a deployment job on the `h01_c3_flat` datastack for the annotated roots, in the same format as the MICrONS jobs.
  - Job `h01` in meshmash-deployment, run `run-20261013-224416-dcc1`: all 160 roots passed all four steps. The last commit was at 2026-10-14 14:47 UTC. `H01_AS_OF` in `products/sources.py` is 15:00 UTC.
  - H01 is read at the labeling time (the v1054 timestamp, which CAVE no longer serves as a version). The workers read synapses and nuclei from a one-time snapshot, `gs://bdp-ssa/meshmash-deployment/_inputs/h01_c3_flat/v1054/`, because the H01 materialize service is unstable. The snapshot is the only Delta copy of these rows. Treat it as frozen.
- The tables follow the MICrONS chain with an `h01_` prefix:

  | MICrONS | H01 | Kind |
  |---|---|---|
  | `synapse_labels` | `h01_labels` | rebuildable (from S12 and the snapshot) |
  | `root_nuclei` | `h01_root_nuclei` (the snapshot nucleus rows, 1 per root) | rebuildable |
  | `labeled_synapse_features` | `h01_labeled_synapse_features` | rebuildable |
  | `model_inputs` | `h01_model_inputs` | view |
  | `validation_predictions` | `h01_predictions` (the deployment posteriors) | rebuildable |

  - `h01_retrain_model_inputs` is `h01_model_inputs` for the `retrain` cohort only. **Model work reads only this table**, so the `validation` cohort stays held out until the final evaluation.
  - 3 label synapses in each cohort map to no domain, so they have no features and no prediction.
- [x] Port from `meshrep/sandbox/h01_validation.py` to `scripts/validation/h01_validation.py`. It reads only `h01_predictions`. **Not run yet**, so that the validation cohort is not looked at during model work.
  - [x] The MICrONS bar in 6A reads `validation_exc_f1` (written by `validation.py` from P4), not the hard-coded 0.977.
  - [x] The calibration panel (6C) takes `posterior_spine` from S13 `domain_predictions`.
  - [ ] The script reports each cohort and the pooled set. Decide which value goes into Fig 6.
  - [ ] Panel outputs: Fig 6A-C, glued `h01_validation_exc_f1`.
  - [ ] The 4-panel `_with_calibration` composite (the paper layout) is not ported. Port it with the composites.
- **Local models** (`src/analysis/models.py`): `load_ssa_model(name_or_path)` loads any MLflow sklearn folder, and `predict(model, features)` scores it by the rules of the deployment prediction step. `train.py` imports its configuration (`MODEL_SPECS`, `RF_PARAMS`, the nucleus mask) from this module.
  - [x] `scratch/retrain_models.py` scores the models in its `MODELS` list on the retrain cohort. Its parity check reproduces the deployment exactly for `ssa_model_full_v1` (0 label mismatches, posterior difference 0).
  - Retrain cohort, zero-shot (no H01 training data), 1,978 scored synapses:

    | Variant | Weighted F1 | Macro F1 |
    |---|---|---|
    | `ssa_model_full_v1` | 0.958 | 0.913 |
    | `ssa_model_no_nuc_v1` | 0.953 | 0.870 |
    | `ssa_model_hks_v1` | 0.948 | 0.873 |
    | `ssa_model_hks_nuc_v1` | 0.950 | 0.864 |
    | `ssa_model_binary_v1` (2 classes) | 0.962 | 0.953 |
    | `ssa_model_hks_binary_v1` (2 classes) | 0.956 | 0.946 |
- [x] Port `meshrep/clean_paper_code/retrain_h01.py` to `scripts/h01/retrain_h01.py` (paper revision). It reads only `model_inputs` and `h01_retrain_model_inputs`.
  - Train modes `microns_only` and `both_upweighted` (H01 rows weighted 1,000). Experiment 1 splits the retrain roots into 10 folds, experiment 2 splits the synapses (stratified). The features, the nucleus mask and the forest are those of the main model, so the old script's values (32 HKS columns, 250 trees) do not carry over.
  - The experiment 2 Wilcoxon test now uses the experiment 2 results. The old script used the experiment 1 results.
  - Panel outputs: `figures/h01_retrain/retrain_{neuron,synapse}_split_{macro_f1,confusion_matrices}`, glued `h01_retrain_*`.
  - Result (1,978 retrain synapses with features, 80 roots), mean ± standard deviation over 10 folds:

    | Split | Mode | Weighted F1 | Macro F1 | Wilcoxon p (macro F1) |
    |---|---|---|---|---|
    | neuron | `microns_only` | 0.956 ± 0.015 | 0.864 ± 0.127 | 0.074 |
    | neuron | `both_upweighted` | 0.965 ± 0.011 | 0.911 ± 0.096 | |
    | synapse | `microns_only` | 0.956 ± 0.016 | 0.899 ± 0.076 | 0.002 |
    | synapse | `both_upweighted` | 0.964 ± 0.011 | 0.936 ± 0.058 | |

    The H01 rows help most on shaft (0.89 → 0.94 of shaft labels correct, neuron split). Macro F1 varies much between folds, because the retrain cohort has only 26 soma labels.
  - The script now runs every variant in `MODEL_SPECS`, and stashes the fold results of each (`Stash`, see Conventions). A stashed result is read instead of fit again. After a change to `run_folds`, bump `FOLDS_VERSION`. The neuron split is off (`RUN_NEURON_SPLIT = False`), so the HKS variants have synapse split results only. The glued variables and the per-experiment figures are for the main model. `retrain_all_models_synapse_split` shows every variant.
  - Result, all variants, synapse split (mean ± standard deviation over 10 folds):

    | Variant | Weighted F1, MICrONS only | Weighted F1, both | Macro F1, MICrONS only | Macro F1, both | Wilcoxon p (macro F1) |
    |---|---|---|---|---|---|
    | `ssa_model_full` | 0.956 ± 0.016 | 0.964 ± 0.011 | 0.899 ± 0.076 | 0.936 ± 0.058 | 0.002 |
    | `ssa_model_no_nuc` | 0.954 ± 0.014 | 0.961 ± 0.010 | 0.883 ± 0.066 | 0.906 ± 0.051 | 0.006 |
    | `ssa_model_hks` | 0.948 ± 0.011 | 0.950 ± 0.012 | 0.870 ± 0.054 | 0.892 ± 0.062 | 0.027 |
    | `ssa_model_hks_nuc` | 0.948 ± 0.011 | 0.956 ± 0.017 | 0.858 ± 0.068 | 0.926 ± 0.073 | 0.037 |
    | `ssa_model_binary` (2 classes) | 0.963 ± 0.012 | 0.971 ± 0.009 | 0.955 ± 0.014 | 0.964 ± 0.011 | 0.004 |
    | `ssa_model_hks_binary` (2 classes) | 0.956 ± 0.009 | 0.956 ± 0.012 | 0.947 ± 0.011 | 0.947 ± 0.014 | 0.55 |
- [x] Retrained models for viewing. The last cell of `retrain_h01.py` fits each variant once on all MICrONS rows and all H01 retrain rows (`both_upweighted`, no held-out fold). It saves them as `data/models/{variant}_h01_v1`. A saved model is read instead of fit again. These models trained on the labels of the retrain cells, so their predictions on those labels are not held out. Kind: rebuildable (the fit is seeded). They are not published.
- [x] Whole-root viewers: `analysis.model_viewer` is the app, and two scripts run it. `scripts/model/view_microns_models.py` runs it on MICrONS cells (`--pool validation`, the default, or `--pool labeled`). `scratch/view_retrain_models.py` shows the 6 variants x {no retrain, retrain} on one H01 retrain cell at a time (see "Whole-root data" below).

### Whole-root data (viewers)

- `load_root_cell(root_id, datastack)` in `products/roots.py` gives an ossify `Cell` for one root: the raw mesh, its domains (with `domain_features`, `distance_to_nucleus` and the deployment predictions), the synapses with their manual labels, and the nuclei. The module docstring lists the layers.
  - `ROOT_SOURCES` names, for each datastack, the cutoff, the synapse source and its resolution, the nuclei table and the label tables.
  - H01: S13 (`vertex_domains`, `domain_features`, `domain_predictions`, `synapse_mapping`, and the synapse snapshot), `h01_root_nuclei`, `h01_labels`. Pin: `STEP_PINS` + the `predictions` cutoff (`ROOT_CUTOFF`).
  - MICrONS: S2 + S3 at the `predictions` cutoff, S0 for the synapses (voxels x (4, 4, 40)), `root_nuclei`, and the labels of `synapse_labels` and `validation_labels`. Only the 132 validation roots have deployment predictions; the other roots have null deployment columns.
  - The mesh comes from CloudVolume through the CAVE segmentation source of the datastack.
  - The mesh kwargs come from `_meshmash/identity.json` of the pinned `vertex_domains` table, so the mesh is the one that `mesh_index` indexes. A different vertex count raises.
  - Without `allow_held_out=True`, a root of the H01 `validation` cohort raises.
  - One H01 root takes about 50 s from the cloud (the mesh and `vertex_domains` take most of it).
- Optional cache: `products/root_cache.cached(load_root_cell, max_roots=5)`. One `.osy` file per root (about 70 MB, mostly the mesh) at `data/root_cache/{datastack}/{pin hash}/{root_id}.osy`. The folder keeps the 5 files read last. Kind: rebuildable, for development only. No product reads it.
- `analysis.viz` has the conventions that every pyvista view uses (`style.py`: the label palette in a fixed class order, the categorical settings with `interpolate_before_map=False`, the posterior blend), the `Scene` contract between data and viewer (`scene.py`), and the `Viewer` (`viewer.py`). The viewer builds the next root in a worker process while one root shows.

### Census track (the stat_summaries inputs)

**P6 `cell_selection`** (tiny)
- Origin: S1. Pin: v1822. Kind: rebuildable.
- [x] Moved the `proofread` / `column` / thalamic rules into `select_cells` in `products/cell_selection.py`, registered as `cell_selection`.
  - `tables.py` and `stat_summaries.py` both call `select_cells`. `tests/test_cell_selection.py` tests the rules.
- Glued counts (1,352 cells: 1,182 E, 150 I, 20 TH) will change at v1822.
- [x] Fixed the boolean parsing in `load_neuron_info`. At v1822, CAVE gives `axon_cleaned` and `status_dendrite` as booleans, not as `'t'`/`'f'`. So the old cache had 0 proofread cells. `_as_bool` in `io.py` reads both encodings.
  - Result at v1822: 1,405 proofread cells (1,182 E, 150 I, 73 TH) and 1,332 column cells.
  - The thalamic count changed from 20 to 73. At v1822, 73 roots have `status_dendrite` False and `strategy_dendrite` `none`.

**P7 `synapse_structure`** (all)
- Origin: S2 `synapse_mapping` ⋈ S4 `vertex_structures`. Pin: S2 + S4. Kind: cloud (a lazy join, no cache of the whole table).
- Gives `structure_id` per synapse. Per structure, this gives `n_synapses`, `n_pre_root_ids` and `is_multi`. This replaces the old `synapse_component_deltalake`.
- [ ] Needed for **all** synapses, because P10 needs multi-input spine rates for every neuron (Fig 4B, S5, S6).

**P8 `skeleton_synapse_features`** (proofread cells, 5.6M rows, 199 MB)
- Origin: S5 ⋈ `root_nuclei`. The synapses and their positions come from the `pre_syn` and `post_syn` layers of each skeleton file, not from S0. Pin: S5 + v1822. Cache: `data/tables/v1822/skeleton_synapse_features/{hash}.parquet`. Kind: rebuildable.
- [x] Registered as `skeleton_synapse_features` in `products/skeleton.py`. Build: `just materialize skeleton_synapse_features` (about 2 minutes). The cache key includes the `cell_selection` and `root_nuclei` cache keys, the skeleton path and the tip parameters.
  - `load_skeleton` asserts that the timestamp of each skeleton file is the v1822 timestamp.
  - `scripts/skeleton/pull_skeleton_info.py` checks the features on one cell.
- Long format: one row per synapse and side, for every proofread root. P9 pivots it to `{pre,post}_*` columns on `synapse_id`.
- Columns: `synapse_id`, `root_id`, `side`, `compartment`, `path_distance_to_nuc`, `max_dist_to_tip`, `ctr_pt_position_{x,y,z}`, `euc_distance_to_nuc`. All distances are in nm.
- Compartment rule, from the skeleton vertex that the synapse maps to:
  - `is_axon` True → `axon`. Axon wins over soma.
  - `is_axon` False and the vertex is the root → `soma`. The old name was `perisoma`.
  - Otherwise → `dendrite`.
- `path_distance_to_nuc` is the path distance along the skeleton to the root. `euc_distance_to_nuc` is the straight distance to the nearest nucleus on the root (the same rule as `distance_to_nucleus` in P3).
- `max_dist_to_tip`: the path distance to the farthest real tip below the synapse's vertex. A real tip is a tip that is not axon, with no branch point within 3 hops above it. Vertices with no real tip below them take the parent's value, for up to 4 rounds. The value is null on the pre side and on axon synapses.
  - The old code removed the axon before it found the tips. The new code keeps the topology and excludes only the axon tips.
- Dropped: `primary_branch_id`, `segment_id` and `soma_relative_centroid_*`. No figure uses them.
- Check against the old v1412 tables, on 5 column cells with the same root ID at v1412 and v1822: the synapse sets are the same, and the compartments agree on 99.8% to 100% of the synapses. `path_distance_to_nuc` and `max_dist_to_tip` have a correlation of 1.0 with the old values.
- Result:
  - Pre side: 1.06M rows (98.3% axon, 1.6% dendrite, 0.1% soma).
  - Post side: 4.55M rows (97.3% dendrite, 2.4% soma, 0.5% axon).
  - 8,839 of 4.53M dendrite and soma post synapses (0.2%) have no `max_dist_to_tip`. `euc_distance_to_nuc` has no nulls.
- [ ] 74 proofread roots have no skeleton file: all 73 thalamic cells and the column cell `864691136674080135`. The thalamic cells matter only for the pre side, and the old tables did not include them either. In P9, their `pre_compartment` is null.
- The figures use pre-side fields for proofread axons and post-side fields for column cells.

**P9 `selection_synapses`** (replaces `filtered_synapses`, selection, about 300-500 MB)
- Origin: S0 (selection) ⋈ P15 ⋈ P7 ⋈ P8 ⋈ P6 cell types ⋈ spatial transform. Pin: the pins of its inputs. Kind: rebuildable.
- Keep all columns. About 30 of the 58 old columns are used by figures (listed below).
- [ ] Panel outputs: most of Figs 4 and 5, Figs S4 and S7.

**P10 `post_aggregation`** (all, about 9 MB)
- Origin: S0 ⋈ P15 ⋈ P7, about 200M rows. Pin: the pins of its inputs. Kind: rebuildable, but expensive.
- Per post root counts and proportions of spine/shaft/soma and single/multi, plus `post_spine_sites`, `post_mean_synapse_size`, `post_mean_spine_inputs`.
- [ ] This is the **only product that needs a scan of all synapses**. Do it as a streaming job over `root_id_bucket` (one chunk at a time), or as a deployment step that writes per-root aggregates.
- [ ] Panel outputs: Fig 4B, Fig 5B, Figs S5 and S6, `summary_*` glue. Also `counts_n_synapses_with_known_prediction` and `counts_n_neurons_ran`: take these from the same scan.

**P11 `column_spines`** (replaces `filtered_spine_info`, column, about 700 MB with all columns)
- Origin: S4 `structure_morphometry` for structures on column cells ⋈ P7 ⋈ P13. Pin: the pins of its inputs. Kind: rebuildable.
- Keep all columns. Figures use only `post_pt_root_id`, `structure_id`, `size_nm3`, `is_multi`, `n_pre_root_ids`, `is_unitary_spine`. No figure uses the old 32 `mean_hks_*` columns.
- [ ] Panel outputs: Fig 5C-left and 5E-left, Fig S8.

**P12 `multi_spine_annotations`** (column, 926 rows)
- Origin: S10, re-mapped through S0 and P7. Pin: S10 + v1822 + S4. Kind: rebuildable (S10 itself is frozen).
- [ ] Re-map each annotation point to the `synapse_id`s on that spine, and store those as the key. Get `structure_id` from P7 at read time. S10 is keyed by v1412 root and the **old** `component_id`, and neither carries over.
- [ ] Panel outputs: Fig 5G-left and 5H, Fig S8, `column_summary_multi_synapse_*` glue.

**P13 single-spine (unitary) model and predictions**
- Origin: S11 ⋈ S4 morphometry (training), then the model applied to all of S4. Pin: S11 + S4 + training parameters + random seed.
- Kind: the model is frozen, like P3. The predictions are rebuildable from the model and S4.
  - [ ] Export S11 as a local parquet keyed by point / `structure_id`.
  - [ ] Train the random forest on the new morphometry columns.
  - [ ] Apply the model to **every spine** (all scope). Run it as a streaming job over S4 `structure_morphometry`, like P10. P11 then joins the result.
  - [ ] Decide where the predictions for all spines go. A local cache is about 1-2 GB (194M rows in the old run). The other option is a Delta table in the bucket.
- [ ] Panel outputs: Fig S8, glued `unitary_spines_*`.

**P15 `synapse_posteriors`** (all)
- Origin: S2 `synapse_mapping` ⋈ `vertex_domains` ⋈ S3 `domain_predictions`. Pin: S2 + S3. Kind: cloud (a lazy join).
- [x] Register `synapse_posteriors`. The code is in `products/predictions.py`. P4 is the first product that reads it, on the 132 validation roots (68 seconds).
  - `synapse_posteriors(synapses, as_of=...)` is the pinned read for products such as P4.
  - It shares `synapse_domain_join` with P2 (`products/features.py`). The refactored P2 read matched the P2 cache on 52 synapses from 5 roots.
  - `products/sources.py` has the `domain_predictions` path `s1/p-96cea889064d4d41`. The `microns_validation` run wrote to this path. A change to the model or to the deployment step code moves this hash.
- Gives the full posterior (`posterior_spine`, `posterior_shaft`, `posterior_soma`) and `pred_label` per synapse. No change to the deployment is needed.
- `synapse_predictions` takes its values from the domain that each synapse maps to, so it is a subset of this join. `synapse_predictions` drops synapses that have no prediction, but the join keeps them with null values.
- Figures use only `pred_label` (`tag`). Keep the posteriors for the validation ROC and the public tables.
- Cost: the join reads all of `vertex_domains` for each root, about 1.7 MB per root. That is 9.95 GiB for the 6,010 labeled roots, or about 125 GB for the full run. P7 reads `vertex_structures` in the same way. Together, that is about 6 hours on the laptop, and the chunked build resumes after a stop. For comparison, P2 also reads `domain_features` (33.7 GiB for the labeled roots) and took about an hour.
- Decision: accept this cost and do not change the pipeline. If the all-synapse products need frequent rebuilds, add a deployment step like `synapse_predictions`. The new step writes one row per synapse: `synapse_id`, `domain_id`, `structure_id`. Do not z-order on a "has synapse" flag.

**Pre-side aggregations** (`pre_synapse_aggregation`, `pre_synapse_to_exc`, `pre_synapse_to_inh`)
- [ ] Recommendation: **drop these as stored products** and compute them in the figure script from P9 (1,352 rows each).
- `pre_to_inh` is never read. Most `pre_*` columns that the figures use are already recomputed in-script.

### Strawman comparison track

**P14 `profile_strawman`** (Fig 2B-E, `profile_*` glue)
- Origin: S8. Pin: the S8 run names. Kind: rebuildable (only the figures are local).
- A re-run on the new pipeline is likely. The fair way to compare the two approaches is still open (question 1).
- [ ] Re-run the strawman comparison on the new pipeline, after question 1 has an answer.
- [ ] The save call for 2D (`test_accuracy_comparison_zoom`) is commented out in `profile_approaches.py`. Turn it back on.

---

## Columns that P9 / P10 / P11 must keep

- **P9:**
  - `synapse_id`, `pre_pt_root_id`, `post_pt_root_id`, `pre_in_selection`, `post_in_selection`, `size`, `tag`, `tag_detailed`
  - `structure_id` (old `component_id` / `spine_group_id`), `spine_is_multi`, `spine_n_pre_pt_root_ids`, `spine_n_pre_synapses`, `spine_size_nm3`
  - `{pre,post}_{broad_type,cell_type}`, `pre_axon_cleaned`
  - `{pre,post}_compartment`, `{pre,post}_path_distance_to_nuc`, `max_dist_to_tip` (from the post side of P8)
  - `post_euc_distance_to_nuc`, `transformed_um_{x,y,z}`, `ctr_pt_position_{x,y,z}`
- **P10:** the `post_*` counts and proportions listed under P10.
- **P11:** the columns listed under P11.

---

## Figure index (data panels → products)

| Figure | Panels | Products | Script now |
|---|---|---|---|
| 2 | B-E | P14 | `meshrep/clean_paper_code/profile_approaches.py` (not ported) |
| 3 | B | P4 | `scripts/validation/validation.py` |
| 4 | A-mid, A-right, C-right, D-right | P9, P6 | `stat_summaries.py` |
| 4 | B | P10, P6 | `stat_summaries.py` |
| 5 | A-right, D-right | P9 (+ P10) | `stat_summaries.py` |
| 5 | B | P10 | `stat_summaries.py` |
| 5 | C-left, E-left | P11, P9 | `stat_summaries.py` |
| 5 | C-right, E-right, F | P9 (P8 distances) | `stat_summaries.py` |
| 5 | G-left, H | P12, P9, P11 | `stat_summaries.py` |
| 6 | A-C | P5 | `scripts/validation/h01_validation.py` |
| H01 retraining (revision) | - | P5 (`h01_retrain_model_inputs`), `model_inputs` | `scripts/h01/retrain_h01.py` |
| S2 | - | P2, P3 | `scripts/model/train.py` |
| S3 | - | P4 | `meshrep/clean_paper_code/compare_to_neurd.py` (not ported) |
| S4 | - | P9 | `stat_summaries.py` |
| S5, S6 | - | P10 | `stat_summaries.py` |
| S7 | - | P9 | `stat_summaries.py` |
| S8 | - | P11, P12, P13 | `stat_summaries.py` |
| S9 | - | P4 | `scripts/validation/validation.py` |
| Methods table | `tab:cell-type-summary` | P6, P9, P10 | `stat_summaries.py` |

The figure numbers S2-S9 in this table are supplementary figures, not the `S#` source IDs.

Composite scripts (they read figure files only, no data): `diagram_approach_figure.py`, `stat_summary_figure.py`, `multi_spine_figure.py`, `show_pipeline_figure.py`. Port them last.

Out of scope:
- Explanatory or single-cell panels: Fig 1 (all panels), Fig 2A, Fig S1 (across datasets).
- Rendered mesh panels: Fig 3A gallery (`show_posterior_on_mesh.py`), the `diagram_cells.py` composites in Figs 4 and 5, Fig 6D.

---

## Suggested order

1. [ ] Add the configuration file with origins, pins and kinds. Point `products/sources.py` at the s3 tables.
   - Partly done: `products/sources.py` points at the s3 tables. The configuration file does not exist yet.
2. [ ] Move the frozen files (S9, S10) to a versioned location.
3. [ ] P1 → P2 → P3. Register P3, then run S3 on the labeled and validation roots, then build P4. This gives the training and validation figures.
   - Partly done: P1, P2, P3 and P4 are built. S3 ran on the validation roots only.
4. [ ] Start the full deployment run (S2 + S3 + S4 for all roots) while you work on P6, P8 and the S11 export.
5. [ ] P15 → P7 → P9 → P11, then P12 re-map, then P13 apply.
6. [ ] P10 (the full scan).
7. [ ] Point `stat_summaries.py` at the new products.
8. [ ] P5 (H01), P14, composites.

## Open questions

1. Fig 2B-E strawman comparison: what is a fair comparison between the naive and the proposed approach on the new pipeline?
2. The `S#` source IDs clash with the supplementary figure numbers (Fig S2, S3 and so on). Rename the sources to `D#` or `SRC#`?

## Decisions

- Rendered mesh panels are out of scope for this pass.
- Keep the full posterior (P15). Do not change deployment or product tables without a request. Note the unused columns only.
- Apply the single-spine model to every spine.

---

## Publication

The hashed cache is for development. The published tables have clean names. A separate export step makes them, so both exist at the same time.

- **Cache layout (development):** `data/tables/v{VERSION}/{name}/{hash}.parquet`. The hash comes from the table's `cache_params`, which include the cache keys of its inputs. A change to any pin or input gives a new hash, so the cache rebuilds. The folder names are the clean table names, and stale entries sit next to the current one.
- [ ] **`just publish` (planned):** copies the current cache file of each product to `release/{name}.parquet`.
  - If any product has no current cache, it fails. A release then cannot mix old and new files.
  - It writes the pins of each table (origin paths, `as_of`, versions, parameter hashes) into the parquet footer metadata of that file. The pins are then part of the file and cannot drift from it.
- [ ] **Release mode (planned):** with release mode on, `Tables[name].lazy()` reads `release/{name}.parquet` and builds nothing. A reader with only the release files can run the figure scripts, with no cloud access and no hashes. During development, release mode is off, and the registry uses the hashed cache.
- **After publication:** a new pin makes a new hashed cache. Only a new run of `just publish` changes the `release/` folder.
