# Audit scripts (run order)

Each script writes JSON (and sometimes PNG) into `../results/`.
Set `OOC_DATA` if the dataset is not at `../../data/OOC_image_dataset`.

| order | script | produces | finding |
|---|---|---|---|
| 1 | `leakage_experiment.py` | `leakage_results.json` | published split has 57/59 sessions in train∩test |
| 1b | `leakage_controlled.py` | `leakage_controlled.json` | controlled A/B, 8 seeds: +7.9 pp accuracy, +8.9 pp AUC |
| 1c | `01b_leakage_ci.py` | `leakage_ci.json` | paired t-interval over the 8 seeds: accuracy [4.2, 11.5], AUC [6.7, 11.0]; 8/8 seeds positive |
| 2 | `structure_probe.py` | `structure_probe.json` | runs test: failures are contiguous; metadata model (in-sample 0.562) and per-cell-line bad rates |
| 2a | `02c_celltype_control.py` | `celltype_control.json` | runs test survives a cell-type control |
| 2b | `02d_redundancy.py` | `redundancy.json` | consecutive fields are correlated but not duplicates |
| 2b | `block_structure.py` | `block_structure.json` | ICC 0.321, design effect 17, effective N ≈ 181 |
| 2c | `label_sufficiency.py` | `label_sufficiency.json` | m-field majority vs session majority |
| 3 | `adaptive_sampling.py` | `adaptive_sampling.json` | label-only: random = best for the call, adaptive = best for localisation (does not survive 3c) |
| 3b | `stopping_rule.py` | `stopping_rule.json` | label-only stopping simulation (optimistic) |
| 3c | `03c_policy_model_in_loop.py` | `policy_model_in_loop.json` | policy ranking does NOT survive the model in the loop |
| 3d | `03d_policy_bootstrap.py` | `policy_bootstrap.json` | paired bootstrap CIs over chips — none survives a Bonferroni correction for 6 comparisons |
| 3c | `aggregation_experiment.py` | `aggregation_results.json` | mean/max/top2 aggregation trade-off |
| 3e | `smoothing_test.py` | `smoothing_test.json` | **negative result**: moving-average smoothing along acquisition order does not help (±0.007) |
| 3f | `label_vs_model_same_rule.py` | `label_vs_model.json` | deployed rule on the same 25 chips, ground-truth labels vs model calls (REPORT §6.7) |
| 4 | `recovery_test.py` | `recovery_test.json` | **negative result**: re-image/discard is not predictable |
| 4a | `04b_run_image_stats.py` | `run_image_groups.json` | focus and darkness of isolated vs long bad runs vs good fields (motivation in REPORT §4.5) |
| 4b | `06_oct_leakage.py` | `oct_leakage.json` | second benchmark: 40% of test files share a (well, day) with training |
| 5 | `perfield_model.py` | `perfield_*.pt/json` | trains the deployed per-field model (`--arch large`, `--size 512` for the capacity/resolution checks) |
| 5b | `05_lolo_cellline.py` | `lolo_cellline.json` | leave-one-cell-line-out: unseen cell line costs ~7 AUC points |
| 5b | `qc_map.py` | `qc_maps_384.png` | per-chip QC maps |
| 6 | `../evaluate.py` | `tool_evaluation.json` | deployed-tool numbers (section 3 of README) |
| 6a | `09_inner_cv_minfields.py` | `inner_cv_oof.json`, `inner_cv_minfields.json` | re-selects the minimum-fields guard by 5-fold session-grouped CV on the 34 non-test sessions (test chips untouched) |
| 6b | `10_ood_check.py` | `ood_check.json` | image-statistics and feature-space Mahalanobis alarms (training p99) on the 25 unseen chips: 0/25, 21/25 (median field) or 4/25 (mean feature); none flags 230405 |
| 6c | `08_full_sessions.py` | `full_sessions.json` | same tool on all fields of the 25 held-out sessions: 0.76 at 9.0 fields vs 0.76 reading all 55.1 |
| 6e | `07_efficiency.py` | `efficiency.json` | field efficiency: all fields, fixed k, cap k, sequential rule by minimum (Table 6) |
| 6d | `11_per_chip_calls.py` | `per_chip_calls.json` | per-chip calls of the shipped rule (Table 7: four wrong, two inconclusive) and the first-*k* failure on 230405 (*pass* at 0.94 after 6 fields; spread order: *fail* at 0.91) |

Note: `stopping_rule.py` simulates the sequential rule on **ground-truth labels**, which overstates
the deployed tool. `evaluate.py` re-measures it with the model's predictions (9.5 fields, 82.6%),
and `03c_policy_model_in_loop.py` shows that the *policy ranking* from the label simulation does not
survive the model in the loop — the report therefore claims no ranking.
