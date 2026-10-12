# Data products for the paper refactor

Goal: rebuild every data product behind a **data panel** in the paper (`../ssa/main.tex`) on the new sources. The new sources are the meshmash-deployment Delta tables, CAVE v1822 and the `csm-skel` skeletons. Explanatory and one-off schematic panels are out of scope (they are listed at the end).

This file is organized **by product**, because a product is the stable unit: scripts change, but a figure always reads the same table. Each product has an ID (`P#`). The figure index at the bottom maps panels to IDs.

Scope labels used below:

- **all**: every synapse or neuron in the dataset (about 208M synapses onto about 75k neurons in the old run).
- **selection**: synapses with `pre ∈ proofread` or `post ∈ column` (5.3M rows in the old run).
- **column**: synapses or spines onto the about 1.3k column cells.
- **labeled**: the training label set (about 137k synapses on about 6k roots).
- **validation**: 3,300 synapses on the 132 held-out cells.

Storage labels: **cloud** = stays in GCS and is never cached whole. **local** = cached parquet in `data/tables/v1822/`, and figure scripts read only these.

---

## Conventions

- **One configuration file** holds every cloud path and pin. The `Tables` registry is the only way that scripts load data. Scripts move over one at a time.
- **Product scripts and figure scripts are separate.** Product scripts read the cloud and write local caches. Figure scripts read only local caches.
- **No loads at import time.** Remove the module-level `load_neuron_info` call in `src/analysis/tables.py`.
- **Join on stable IDs.** Use `synapse_id` / `structure_id`, and take `root_id` from the v1822 synapse table. Only 582 of 1,333 column cells keep the same root ID between v1412 and v1822.
- **Keep all columns.** Do not drop columns from products or deployment tables. The plan only notes columns that no figure uses.
- **Pin cloud tables with an `as_of` cutoff on `attempt_id`, not with a Delta version.**
  - The deployment sets `delta.logRetentionDuration` to 24 hours and `delta.deletedFileRetentionDuration` to 6 hours. It runs OPTIMIZE and VACUUM on a schedule. So a Delta version number stops working after about a day.
  - The tables are append-only, and the latest attempt per root wins at read time. OPTIMIZE rewrites files but keeps every row. So `attempt_id <= as_of`, applied before the latest-attempt filter, gives the same rows at any later time.
  - Put each table's `as_of` in the configuration file and in the cache key. A cache rebuilds only when you change `as_of` on purpose.
  - Set `as_of` after a job finishes. An attempt that starts before `as_of` and commits after it appears only in later reads.
  - The pin fails if old attempts are ever deleted, or if a table moves to a new `p-…` path. A new path means new parameters, so that is a deliberate change.
- **The local training cache is the frozen copy of the training data.** If the full run processes the 6,010 labeled roots again, their new attempts win. A training cache that has no `as_of` would then change without notice.

---

## Sources (not built here, only pinned)

| ID | Source | Location | Scope | Status |
|---|---|---|---|---|
| S0 | `synapses_pni_2` Delta export, v1822 | `gs://mat_dbs/public/deltalake_exports/minnie65_phase3_v1/v1822/synapses_pni_2/post_pt_root_id` | all | exists |
| S1 | `aibs_cell_info` + `proofreading_status_and_strategy`, v1822 | `data/tables/v1822/*.parquet` (via `load_neuron_info`) | all cells | cached |
| S2 | `synapse_mapping`, `vertex_domains`, `domain_features` | `gs://bdp-ssa/meshmash-deployment/minnie65_phase3_v1/…` | labeled roots now (job `microns_top_labeled`, 6,010 roots), then **all** after the full run | partial |
| S3 | `domain_predictions` (full `posterior_<class>`) + `synapse_predictions` (argmax summary only) | deployment, needs P3 registered as an MLflow URI | all | not run |
| S4 | `structure_morphometry` + `vertex_structures` (spines) | deployment | all | not run (50-root `spine_trial` only) |
| S5 | Skeletons (`.osy`) | `gs://csm-skel/osy/minnie65/skel/{root}.osy` | proofread + column cells | exists |
| S6 | Training labels: CAVE `vortex_compartment_targets`, `bdp_synapse_compartment_labels`, `bdp_point_compartment_labels` | CAVE | labeled | exists |

---

## Products

### Model and validation track

**P1 `label_table`** (labeled, local, 5 MB)
- [x] Built by `wrangle.get_label_table` → `data/tables/v1822/label_table_threshold=None.parquet`.
- [ ] Make sure that the validation cells are excluded. This depends on the `get_validation_ids` path fix.

**P2 `labeled_synapse_features`** (labeled, local, about 60 MB) = P1 ⋈ S2
- [x] Built for s2 features (now stale).
- [ ] Rebuild on the s3 paths. Delete the three stale `labeled_synapse_features-*` files in `data/tables/v1822/`.

**P3 HKS synapse model** (local, `data/models/`) ← P2, by `scripts/model/train.py`
- [ ] Retrain on the s3 feature columns.
- [ ] Register the model for the deployment, so that S3 can run.
- [ ] Panel outputs: learning curve (Fig S2), glued `model_n_training_synapses`, `model_sample_curve_*`, `params_*` (the old `params_*` came from `count_things.py`).

**P4 `validation_predictions`** (validation, local, about 3.3k rows)
- = validation CSVs (`data/tables/validation/`) ⋈ predictions for those synapses ⋈ NEURD `synapse_target_structure` v1822.
- Cheapest build: run `synapse_features()` + P3 locally on the 132 cells. This does not wait for S3.
- [ ] Port `scripts/validation/validation.py`. It still imports `meshrep` and reads the old prediction lake.
- [ ] Panel outputs:
  - Fig 3B (accuracy by cell type), Fig S9 (confusion matrices).
  - Fig S3 (NEURD vs HKS), ported from `meshrep/clean_paper_code/compare_to_neurd.py`.
  - Glued `validation_*` and `compare_*`.
- [ ] `validation_exc_f1` (0.977) has no generator in any script. Make the validation script write it.

**P5 `h01_validation`** (H01 labeled, local, small)
- = H01 annotated synapses + features + P3 predictions.
- [ ] Decide on the feature source: run the deployment on the `h01_c3_flat` datastack for the annotated roots, or compute locally with meshmash.
- [ ] Port from `meshrep/sandbox/h01_validation.py` (not `clean_paper_code`).
- [ ] Panel outputs: Fig 6A-C, glued `h01_validation_exc_f1`.
- [ ] The MICrONS bar in 6A is hard-coded (0.977). Read it from P4 instead.
- [ ] The current sandbox script writes the 3-panel layout to `h01_validation_composite`. The paper uses the 4-panel `_with_calibration` layout.
- [ ] The calibration panel (6C) needs `p_spine`. Compute it locally from P3, because H01 is a local product.

### Census track (the stat_summaries inputs)

**P6 `cell_selection`** (local, tiny) ← S1
- [ ] Move the `proofread` / `column` / thalamic rules into one function or table. They are now copied in `tables.py` and `stat_summaries.py`.
- Glued counts (1,352 cells: 1,182 E, 150 I, 20 TH) will change at v1822.

**P7 `synapse_structure`** (all, cloud-derived) = S2 `synapse_mapping` ⋈ S4 `vertex_structures`
- Gives `structure_id` per synapse. Per structure, this gives `n_synapses`, `n_pre_root_ids` and `is_multi`. This replaces the old `synapse_component_deltalake`.
- [ ] Needed for **all** synapses, because P10 needs multi-input spine rates for every neuron (Fig 4B, S5, S6).

**P8 `skeleton_synapse_features`** (selection, local, about 100 MB) ← S5
- Fields: `compartment`, `path_distance_to_nuc`, `max_dist_to_tip`, and maybe `soma_relative_centroid`.
- [ ] Finish `scripts/skeleton/pull_skeleton_info.py`.
- The figures use pre-side fields for proofread axons and post-side fields for column cells. `primary_branch_id` and `segment_id` are not used by any figure. Skip them.

**P9 `selection_synapses`** (replaces `filtered_synapses`, selection, local, about 300-500 MB)
- = S0 (selection) ⋈ P15 ⋈ P7 ⋈ P8 ⋈ P6 cell types ⋈ spatial transform.
- Keep all columns. About 30 of the 58 old columns are used by figures (listed below).
- [ ] Panel outputs: most of Figs 4 and 5, Figs S4 and S7.

**P10 `post_aggregation`** (all, local, about 9 MB)
- Per post root counts and proportions of spine/shaft/soma and single/multi, plus `post_spine_sites`, `post_mean_synapse_size`, `post_mean_spine_inputs`.
- [ ] This is the **only product that needs a scan of all synapses**: S0 ⋈ P15 ⋈ P7, about 200M rows. Do it as a streaming job over `root_id_bucket` (one chunk at a time), or as a deployment step that writes per-root aggregates.
- [ ] Panel outputs: Fig 4B, Fig 5B, Figs S5 and S6, `summary_*` glue. Also `counts_n_synapses_with_known_prediction` and `counts_n_neurons_ran`: take these from the same scan.

**P11 `column_spines`** (replaces `filtered_spine_info`, column, local, about 700 MB with all columns)
- = S4 `structure_morphometry` for structures on column cells ⋈ P7 ⋈ P13.
- Keep all columns. Figures use only `post_pt_root_id`, `structure_id`, `size_nm3`, `is_multi`, `n_pre_root_ids`, `is_unitary_spine`. No figure uses the old 32 `mean_hks_*` columns.
- [ ] Panel outputs: Fig 5C-left and 5E-left, Fig S8.

**P12 `multi_spine_annotations`** (manually checked double-input spines, local, 926 rows)
- [x] Exists in `data/tables/v1412/`, but it is keyed by v1412 root and the **old** `component_id`.
- [ ] Re-map each annotation point to a v1822 synapse and then to a new `structure_id`. The old component IDs do not carry over to S4.
- [ ] Panel outputs: Fig 5G-left and 5H, Fig S8, `column_summary_multi_synapse_*` glue.

**P13 single-spine (unitary) labels and model** (labeled spines, local, small)
- = manual labels from the neuroglancer states in `meshrep/sandbox/sample_cells_for_multi_validation.py` (plus `cell_type_selection_for_multis.csv`) ⋈ S4 morphometry.
  - [ ] Save the labels as a local parquet keyed by point / `structure_id`.
  - [ ] Train the random forest on the new morphometry columns.
  - [ ] Apply the model to **every spine** (all scope). Run it as a streaming job over S4 `structure_morphometry`, like P10. P11 then joins the result.
- [ ] Panel outputs: Fig S8, glued `unitary_spines_*`.

**P15 `synapse_posteriors`** (all, cloud-derived) = S2 `synapse_mapping` ⋈ `vertex_domains` ⋈ S3 `domain_predictions`
- Gives the full posterior (`posterior_spine`, `posterior_shaft`, `posterior_soma`) and `pred_label` per synapse. No change to the deployment is needed.
- `synapse_predictions` takes its values from the domain that each synapse maps to, so it is a subset of this join. `synapse_predictions` drops synapses that have no prediction, but the join keeps them with null values.
- Figures use only `pred_label` (`tag`). Keep the posteriors for the validation ROC and the public tables.

**Pre-side aggregations** (`pre_synapse_aggregation`, `pre_synapse_to_exc`, `pre_synapse_to_inh`)
- Recommendation: **drop these as stored products** and compute them in the figure script from P9 (1,352 rows each).
- `pre_to_inh` is never read. Most `pre_*` columns that the figures use are already recomputed in-script.

### Method benchmark track

**P14 `profile_benchmark`** (Fig 2B-E, `profile_*` glue)
- Reads `gs://bdp-ssa/comparisons/{shiny-wolf-message,…-strawman}`.
- These do not depend on the materialization version. A re-run on the new pipeline is likely. The fair way to compare the two approaches is still open (question 1).
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

Composite scripts (they read figure files only, no data): `diagram_approach_figure.py`, `stat_summary_figure.py`, `multi_spine_figure.py`, `show_pipeline_figure.py`. Port them last.

Out of scope:
- Explanatory or single-cell panels: Fig 1 (all panels), Fig 2A, Fig S1 (across datasets).
- Rendered mesh panels: Fig 3A gallery (`show_posterior_on_mesh.py`), the `diagram_cells.py` composites in Figs 4 and 5, Fig 6D.

---

## Suggested order

1. Add the configuration file with `as_of` pins. Point `features.py` at the s3 tables.
2. P1 → P2 → P3 → P4. This gives the validation and training figures, and it gives the deployment a model for S3.
3. Start the full deployment run (S2 + S3 + S4 for all roots) while you work on P6, P8 and P13 labels.
4. P15 → P7 → P9 → P11, then P12 re-map, then P13 apply.
5. P10 (the full scan).
6. Point `stat_summaries.py` at the new products.
7. P5 (H01), P14, composites.

## Open questions

1. Fig 2B-E benchmarks: what is a fair comparison between the naive and the proposed approach on the new pipeline?

## Decisions

- Rendered mesh panels are out of scope for this pass.
- Keep the full posterior (P15). Do not change deployment or product tables without a request. Note the unused columns only.
- Apply the single-spine model to every spine.
