# Data and result guide

This repository provides event-role assignments, derived predictions, statistical outputs and execution diagnostics for respiratory-sound classification. Files use technical experiment identifiers; directory names distinguish processing stages and versions.

## What is included

- 15,987 annotated respiratory events across ICBHI 2017 and SPRSound.
- Database-prefixed, coded participant identifiers for 414 participants, with event labels and fixed train/validation/calibration/test roles.
- Per-seed and ensemble probabilities, labels and classifications for the reference models and the masking, consistency-loss and matched-budget adaptation comparisons.
- Classification statistics, calibration metrics, confidence intervals and saved bootstrap arrays where generated.
- Coverage, numerical-stability, timing and resource measurements, including failed-run records.
- Configurations and execution records needed to interpret the saved outputs.

Source audio, original annotation files, upstream model weights, trained checkpoints and recovery states are not redistributed. Source recordings remain available from their original providers; see [Data sources and licences](DATA_SOURCES_AND_LICENCES.md). This collection is derived analysis data, not a replacement distribution of either source database.

## File index

| Location | Contents |
| --- | --- |
| `data/manifests/cross_domain_binary.csv` | Event identities, source recording references, labels, timing information and role assignments |
| `data/manifests/*audit.json` | Counts, partition checks and source integrity summaries |
| `artifacts/post_gate11a/locked_predictions/` | Reference Full-FT and practical-schedule LoRA predictions on fixed test partitions |
| `artifacts/post_gate11a/calibration_predictions/` | Reference calibration and model-selection predictions |
| `artifacts/post_gate11a/locked_final.json` | Reference classification estimates, intervals and probability summaries |
| `artifacts/post_gate11a/calibration_analysis/` | Temperature-scaling and selective-prediction analyses |
| `runs/gate11a_exactmask_fullft_seed*/` | Reference-model validation predictions and execution records |
| `artifacts/revision_2026_09_03/predictions/` | Length-based masking, no-consistency-loss and matched-budget LoRA predictions |
| `artifacts/revision_2026_09_03/analysis_a_corrected_2026_09_05/` | Canonical exact-versus-length-based masking comparison |
| `artifacts/revision_2026_09_03/analysis_b_path_corrected_2026_09_05/` | Canonical consistency-loss comparison |
| `artifacts/revision_2026_09_03/analysis_c_2026_09_06/` | Canonical Full-FT versus matched-budget LoRA comparison |
| `artifacts/revision_2026_09_03/entropy_analysis/` | Class-conditional entropy and coverage results |
| `artifacts/revision_2026_09_03/mask_mapping/` | Event-placement geometry checks and replay records |
| `artifacts/revision_2026_09_03/stability_panel_2026_09_06/` | Forward/backward numerical-stability diagnostics |
| `artifacts/revision_2026_09_03/resource_profile_2026_09_06/` | Timing samples, latency, throughput and resource summaries |
| `artifacts/revision_2026_09_03/failures/` | Failed-run evidence and diagnostics |
| `runs/revision_2026_09_03/` | Ablation run configurations, training histories and validation predictions |
| `DATA_MANIFEST.json` | File sizes, source-copy hashes and distributed-file hashes |

Other retained stage outputs document intermediate analyses and provenance. For final comparison estimates, use the canonical directories above and their independent audit companions, rather than superseded intermediate outputs.

## Event and prediction schema

`sample_id` is a database-prefixed event key. `patient_id` is the coded participant key used for partitioning and patient-cluster resampling. `recording_id`, `source_sample_id` and related fields identify source files/events; they are not direct personal identifiers. `wav_path` is a relative reference to externally obtained audio, not a bundled file.

The manifest contains event start/end times and duration, database and acquisition metadata, fine labels, binary labels and `protocol_role`. Available age/sex metadata are source-database fields and can be missing. Do not interpret blank values as zero.

For the binary task, label `0` denotes normal and label `1` denotes adventitious sounds. Prediction CSVs use `target` for the true binary label, `probability_1` for the adventitious probability, and `prediction` for the classification at threshold 0.5. Some files also contain individual-seed probability columns and `probability_0`. Ensemble probabilities are the mean of the specified members, not a selected best member. Consult each file header for its exact columns.

| Role | Use |
| --- | --- |
| `train_fit` | Model fitting |
| `validation_select` | Checkpoint selection and development decisions |
| `calibration` | Probability-calibration fitting |
| `locked_test` | ICBHI primary test partition |
| `locked_inter_test` | SPRSound primary test partition |
| `locked_intra_test` | Descriptive SPRSound partition with participant overlap; not an independent test partition |

The first five roles are participant-disjoint within each database as applicable. Database prefixes are retained to avoid conflating identifiers from different providers.

## Comparison definitions

The reference Full-FT ensemble contains seeds 20260729, 20260730 and 20260731. The original practical-schedule LoRA comparison uses its own optimisation schedule; it is distinct from the matched-budget LoRA comparison.

- **Mapping (A):** Exact versus length-based masking uses the two completed seed pairs, 20260729 and 20260731. The length-based 20260730 run failed numerically and is retained as a failed run, not substituted by an earlier checkpoint.
- **Consistency (B):** JS-enabled versus no-JS training uses all three seeds.
- **Adaptation (C):** Full-FT versus matched-budget LoRA uses all three seeds with matched update counts and learning-rate schedules.

Reference decision thresholds, data roles and saved predictions are fixed. Negative, null and failed outcomes are retained. Do not use the saved test outputs to select new thresholds, seeds or model variants.

## Read-only verification

Run from the repository root:

```sh
python -B scripts/verify_reference_results.py
python -B scripts/verify_ablation_results.py
```

The reference verifier uses the Python standard library. It verifies manifest and prediction-file integrity, participant separation, reference classification point estimates and selected calibration, coverage and resource records. Its printed original confidence interval is read from the saved summary; this helper does not recompute that interval.

The ablation verifier requires NumPy, pandas and SciPy. It checks every file in `DATA_MANIFEST.json` and independently reconstructs:

- 154 classification rows and 924 classification metric cells;
- 252 probability metric cells and 84 joint calibration intercept/slope fits;
- 36,000 paired classification bootstrap draws and derived confidence intervals;
- 535 saved placement predictions and 60 forward/backward diagnostic records;
- the latency and throughput summaries for 20 recorded resource settings.

These are checks of saved data and calculations. They do not rerun model inference, repeat training, reproduce machine timings, or recompute every historical analysis. Full model replication additionally requires the external assets, an explicitly recorded environment and applicable execution-plan/freeze inputs.

## Integrity and provenance

The manifest records the original input copy's `source_sha256` and the distributed file's `sha256`. All CSV and NPZ files retain their original bytes. Where `metadata_normalized` is true, only administrative stage labels or machine-specific planning paths were normalised; numerical JSON values were checked for equality. Historical freeze/checkpoint/source hashes remain historical identities and need not equal the current source code's bytes. Use `DATA_MANIFEST.json` for distributed-data integrity.

Some audit files retain earlier pending/corrected labels. Read them with the canonical independent audit and final analysis files listed above. The read-only verification scripts do not bypass execution guards or write new scientific outputs.

For a fixed reference to this collection, use a GitHub commit URL. A GitHub URL is not a DOI. Licensing and third-party terms are described in [LICENSING.md](LICENSING.md).
