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
- **The local caches are the source of truth.** Each CAVE pull runs once at v1822, and its cache is then the record for anyone who comes later. At submission, every local cache becomes frozen. The rebuildable kind matters only during development.

---

## Conventions

- **One configuration file** holds the origin, pin and kind of every source. The `Tables` registry is the only way that scripts load data. Scripts move over one at a time.
- **Product scripts and figure scripts are separate.** Product scripts read the cloud and write local caches. Figure scripts read only local caches.
- **No loads at import time.** Remove the module-level `load_neuron_info` call in `src/analysis/tables.py`.
- **Join on stable IDs.**
  - `synapse_id` is stable. Use it to join synapse-level data across versions.
  - Use the nucleus ID (from the soma table) to identify a cell across versions. Take `root_id` from the v1822 synapse or nucleus table. Only 582 of 1,333 column cells keep the same root ID between v1412 and v1822.
  - `structure_id` is valid only inside one pinned S4 snapshot. Never store it as a key that must survive a re-run.
- **Keep all columns.** Do not drop columns from products or deployment tables. The plan only notes columns that no figure uses.
- **Pin deployment tables with an `as_of` cutoff on `attempt_id`, not with a Delta version.**
  - The deployment sets `delta.logRetentionDuration` to 24 hours and `delta.deletedFileRetentionDuration` to 6 hours. It runs OPTIMIZE and VACUUM on a schedule. So a Delta version number stops working after about a day.
  - The tables are append-only, and the latest attempt per root wins at read time. OPTIMIZE rewrites files but keeps every row. So `attempt_id <= as_of`, applied before the latest-attempt filter, gives the same rows at any later time.
  - Put each table's `as_of` in the configuration file and in the cache key. Then a cache rebuilds only after a deliberate change to `as_of`.
  - Set `as_of` after a job finishes. An attempt that starts before `as_of` and commits after it appears only in later reads.
  - Two events break the pin: a deletion of old attempts, or a move of the table to a new `p-…` path. A new path means new parameters, so that is a deliberate change.
  - A product built from deployment tables without an `as_of` is frozen, not rebuildable. For example, the full run can process the 6,010 labeled roots again. Their new attempts then win, and an unpinned P2 changes.
- **Store frozen files in a versioned location** (git LFS, or a write-once prefix in the bucket). Now they exist only on the laptop. `git ls-files data` shows only `ctype_hues.pkl`, and the LFS rules in `data/.gitattributes` name old file paths that no longer exist.
- **The `status` command** lists each registered table with its kind and cache state (current, stale or orphan). It never marks a frozen file as an orphan.

---

## Sources (not built here, only pinned)

### Cloud and CAVE sources

| ID | Origin | Pin | Cache | Kind | Scope | Status |
|---|---|---|---|---|---|---|
| S0 | CAVE `synapses_pni_2`, as the Delta export at `gs://mat_dbs/public/deltalake_exports/minnie65_phase3_v1/v1822/synapses_pni_2/post_pt_root_id` | v1822 (in the path) | none (selection rows go to P9) | cloud | all | exists |
| S1 | CAVE `minnie65_public`: view `aibs_cell_info`, table `proofreading_status_and_strategy` | v1822 | `data/tables/v1822/{aibs_cell_info,proofreading_status_and_strategy}.parquet` | rebuildable | all cells | cached |
| S2 | Deployment tables `synapse_mapping`, `vertex_domains`, `domain_features` at `gs://bdp-ssa/meshmash-deployment/minnie65_phase3_v1/…` | step version + `p-…` hash (in the path) + `as_of` | none (labeled rows go to P2) | cloud | labeled roots now (job `microns_top_labeled`, 6,010 roots), then **all** after the full run | partial |
| S3 | Deployment tables `domain_predictions` (full `posterior_<class>`) + `synapse_predictions` (argmax summary only) | as S2, and the `p-…` hash includes the P3 model URI | none | cloud | all | not run, needs P3 registered as an MLflow URI |
| S4 | Deployment tables `structure_morphometry` + `vertex_structures` (spines) | as S2 | none | cloud | all | not run (50-root `spine_trial` only) |
| S5 | Skeletons at `gs://csm-skel/osy/minnie65/skel/{root}.osy` | root ID. Make sure that files are not overwritten in place. | none (fields go to P8) | cloud | proofread + column cells | exists |
| S6 | CAVE annotation tables `vortex_compartment_targets`, `bdp_synapse_compartment_labels`, `bdp_point_compartment_labels` | v1822 timestamp, for both annotations and segmentation (`live=False`, the default, `wrangle.py:44-45`) | P1 | rebuildable | labeled | exists |
| S7 | CAVE `synapse_target_structure` (NEURD) | v1822 | P4 | rebuildable | validation | exists |
| S8 | Benchmark outputs at `gs://bdp-ssa/comparisons/{shiny-wolf-message,…-strawman}` | run name (in the path) | none | cloud | benchmark neurons | exists, re-run likely |

### Manual label sources

The original records of these labels are neuroglancer states or notebooks. A state link can stop working, so the exported table is the record. All of them are frozen.

| ID | Origin | Pin | Cache | Scope | Status |
|---|---|---|---|---|---|
| S9 | Validation labels from the labeling notebook `BD_250519_validation_set.ipynb`. The files carry `synapse_id` and `post_nucleus_id`. | none (one-time export) | `data/tables/validation/{exc,inh}_sample_synapses_labeled.csv`, `validation_ids_used.csv` (nucleus IDs) | validation | cached |
| S10 | Checked multi-input spines from 10 neuroglancer states (`get_multi_annotations`) | none (one-time export) | `data/tables/v1412/multi_spine_annotations.parquet` | column | cached, keyed by v1412 IDs |
| S11 | Single-spine labels from 7 neuroglancer states + `cell_type_selection_for_multis.csv` (`meshrep/sandbox/sample_cells_for_multi_validation.py`) | none (one-time export) | not exported yet | labeled spines | todo |
| S12 | H01 labels from neuroglancer states + H01 CAVE `h01_c3_flat` | H01 v1054 | not exported yet | H01 labeled | todo |
| S13 | Deployment tables for H01 (`synapse_mapping`, `vertex_domains`, `domain_features`, `domain_predictions`) on the `h01_c3_flat` datastack | as S2, and the `p-…` hash includes the P3 model URI | none (rows go to P5) | cloud | H01 annotated roots | not run |

---

## Products

### Model and validation track

**P1 `label_table`** (labeled, 5 MB)
- Origin: S6. Pin: v1822 timestamp (annotations and segmentation). Cache: `data/tables/v1822/label_table_threshold=None.parquet`. Kind: rebuildable.
- [x] Built by `wrangle.get_label_table`. `live=True` queries annotations at the current time and skips the cache.
- [x] Validation cells are excluded. On 2026-10-11, the cached table had 0 rows on the 132 validation roots (nucleus ID → v1822 root through S1), and none of the 3,300 validation `synapse_id`s.
- [x] Fixed the path in `get_validation_ids` (`data/tables/general/` → `data/tables/validation/`).

**P2 `labeled_synapse_features`** (labeled, about 60 MB)
- Origin: P1 ⋈ S2. Pin: P1 + S2 (`as_of`). Cache: `data/tables/v1822/labeled_synapse_features-{hash}.parquet/`. Kind: rebuildable.
- [x] Built for s2 features (now stale).
- [ ] Rebuild on the s3 paths. Delete the three stale `labeled_synapse_features-*` files in `data/tables/v1822/`.

**P3 HKS synapse model**
- Origin: P2, by `scripts/model/train.py`. Pin: P2 + training parameters + random seed. Cache: `data/models/`. Kind: frozen, because S3 depends on this exact model. Keep it in MLflow.
- [ ] Retrain on the s3 feature columns.
- [ ] Register the model for the deployment, so that S3 can run.
- [ ] Panel outputs: learning curve (Fig S2), glued `model_n_training_synapses`, `model_sample_curve_*`, `params_*` (the old `params_*` came from `count_things.py`).

**P4 `validation_predictions`** (validation, about 3.3k rows)
- Origin: S9 ⋈ P15 for those synapses ⋈ S7. Pin: S9 + S2 + S3 + v1822. Kind: rebuildable.
- Map S9 forward on `synapse_id` (labels) and `post_nucleus_id` (cells). Do not use the root IDs stored in the files.
- The predictions come from the deployment (S2 + S3), not from a local run.
- [ ] Run a deployment job for the 132 validation roots. On 2026-10-11, none of them was in `microns_top_labeled`. The full run also covers them.
- [ ] Port `scripts/validation/validation.py`. It still imports `meshrep` and reads the old prediction lake.
- [ ] Panel outputs:
  - Fig 3B (accuracy by cell type), Fig S9 (confusion matrices).
  - Fig S3 (NEURD vs HKS), ported from `meshrep/clean_paper_code/compare_to_neurd.py`.
  - Glued `validation_*` and `compare_*`.
- [ ] `validation_exc_f1` (0.977) has no generator in any script. Make the validation script write it.

**P5 `h01_validation`** (H01 labeled, small)
- Origin: S12 ⋈ S13, joined the same way as P15. Pin: S12 + S13. Kind: rebuildable.
- [ ] Run a deployment job on the `h01_c3_flat` datastack for the annotated roots, in the same format as the MICrONS jobs.
- [ ] Port from `meshrep/sandbox/h01_validation.py` (not `clean_paper_code`).
- [ ] Panel outputs: Fig 6A-C, glued `h01_validation_exc_f1`.
- [ ] The MICrONS bar in 6A is hard-coded (0.977). Read it from P4 instead.
- [ ] The current sandbox script writes the 3-panel layout to `h01_validation_composite`. The paper uses the 4-panel `_with_calibration` layout.
- [ ] The calibration panel (6C) needs `p_spine`. Take it from the S13 `domain_predictions` posteriors.

### Census track (the stat_summaries inputs)

**P6 `cell_selection`** (tiny)
- Origin: S1. Pin: v1822. Kind: rebuildable.
- [ ] Move the `proofread` / `column` / thalamic rules into one function or table. They are now copied in `tables.py` and `stat_summaries.py`.
- Glued counts (1,352 cells: 1,182 E, 150 I, 20 TH) will change at v1822.

**P7 `synapse_structure`** (all)
- Origin: S2 `synapse_mapping` ⋈ S4 `vertex_structures`. Pin: S2 + S4. Kind: cloud (a lazy join, no cache of the whole table).
- Gives `structure_id` per synapse. Per structure, this gives `n_synapses`, `n_pre_root_ids` and `is_multi`. This replaces the old `synapse_component_deltalake`.
- [ ] Needed for **all** synapses, because P10 needs multi-input spine rates for every neuron (Fig 4B, S5, S6).

**P8 `skeleton_synapse_features`** (selection, about 100 MB)
- Origin: S5 + S0 synapse positions. Pin: S5 + v1822. Kind: rebuildable.
- Fields: `compartment`, `path_distance_to_nuc`, `max_dist_to_tip`, and maybe `soma_relative_centroid`.
- [ ] Finish `scripts/skeleton/pull_skeleton_info.py`.
- The figures use pre-side fields for proofread axons and post-side fields for column cells. `primary_branch_id` and `segment_id` are not used by any figure.

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
- Gives the full posterior (`posterior_spine`, `posterior_shaft`, `posterior_soma`) and `pred_label` per synapse. No change to the deployment is needed.
- `synapse_predictions` takes its values from the domain that each synapse maps to, so it is a subset of this join. `synapse_predictions` drops synapses that have no prediction, but the join keeps them with null values.
- Figures use only `pred_label` (`tag`). Keep the posteriors for the validation ROC and the public tables.

**Pre-side aggregations** (`pre_synapse_aggregation`, `pre_synapse_to_exc`, `pre_synapse_to_inh`)
- Recommendation: **drop these as stored products** and compute them in the figure script from P9 (1,352 rows each).
- `pre_to_inh` is never read. Most `pre_*` columns that the figures use are already recomputed in-script.

### Method benchmark track

**P14 `profile_benchmark`** (Fig 2B-E, `profile_*` glue)
- Origin: S8. Pin: the S8 run names. Kind: rebuildable (only the figures are local).
- A re-run on the new pipeline is likely. The fair way to compare the two approaches is still open (question 1).
- The save call for 2D (`test_accuracy_comparison_zoom`) is commented out in `profile_approaches.py`.

---

## Columns that P9 / P10 / P11 must keep

- **P9:**
  - `synapse_id`, `pre_pt_root_id`, `post_pt_root_id`, `pre_in_selection`, `post_in_selection`, `size`, `tag`, `tag_detailed`
  - `structure_id` (old `component_id` / `spine_group_id`), `spine_is_multi`, `spine_n_pre_pt_root_ids`, `spine_n_pre_synapses`, `spine_size_nm3`
  - `{pre,post}_{broad_type,cell_type}`, `pre_axon_cleaned`
  - `{pre,post}_compartment`, `{pre,post}_path_distance_to_nuc`, `max_dist_to_tip`
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
| 6 | A-C | P5 | `meshrep/sandbox/h01_validation.py` (not ported) |
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

1. Add the configuration file with origins, pins and kinds. Point `features.py` at the s3 tables.
2. Move the frozen files (S9, S10) to a versioned location.
3. P1 → P2 → P3. Register P3, then run S3 on the labeled and validation roots, then build P4. This gives the training and validation figures.
4. Start the full deployment run (S2 + S3 + S4 for all roots) while you work on P6, P8 and the S11 export.
5. P15 → P7 → P9 → P11, then P12 re-map, then P13 apply.
6. P10 (the full scan).
7. Point `stat_summaries.py` at the new products.
8. P5 (H01), P14, composites.

## Open questions

1. Fig 2B-E benchmarks: what is a fair comparison between the naive and the proposed approach on the new pipeline?
2. The `S#` source IDs clash with the supplementary figure numbers (Fig S2, S3 and so on). Rename the sources to `D#` or `SRC#`?

## Decisions

- Rendered mesh panels are out of scope for this pass.
- Keep the full posterior (P15). Do not change deployment or product tables without a request. Note the unused columns only.
- Apply the single-spine model to every spine.
