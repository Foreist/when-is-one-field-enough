# Audit scripts (run order)

Each script writes JSON (and sometimes PNG) into `../results/`.
Set `OOC_DATA` if the dataset is not at `../../data/OOC_image_dataset`.

| order | script | produces | finding |
|---|---|---|---|
| 1 | `leakage_experiment.py` | `leakage_results.json` | published split has 57/59 sessions in train∩test |
| 1b | `leakage_controlled.py` | `leakage_controlled.json` | controlled A/B, 8 seeds: +8.2 pp accuracy, +9.6 pp AUC |
| 2 | `structure_probe.py` | `structure_probe.json` | runs test: failures are contiguous |
| 2a | `02c_celltype_control.py` | `celltype_control.json` | runs test survives a cell-type control |
| 2b | `02d_redundancy.py` | `redundancy.json` | consecutive fields are correlated but not duplicates |
| 2b | `block_structure.py` | `block_structure.json` | ICC 0.321, design effect 17, effective N ≈ 181 |
| 2c | `label_sufficiency.py` | `label_sufficiency.json` | m-field majority vs session majority |
| 3 | `adaptive_sampling.py` | `adaptive_sampling.json` | random = best for the call, adaptive = best for localisation |
| 3b | `stopping_rule.py` | `stopping_rule.json` | label-only stopping simulation (optimistic) |
| 3c | `03c_policy_model_in_loop.py` | `policy_model_in_loop.json` | policy ranking does NOT survive the model in the loop |
| 3d | (see 3c + bootstrap) | `policy_bootstrap.json` | paired CIs over chips (all include zero) |
| 3c | `aggregation_experiment.py` | `aggregation_results.json` | mean/max/top2 aggregation trade-off |
| 3e | `smoothing_test.py` | `smoothing_test.json` | **negative result**: moving-average smoothing along acquisition order does not help (±0.007) |
| 3f | `label_vs_model_same_rule.py` | `label_vs_model.json` | deployed rule on the same 25 chips, ground-truth labels vs model calls (REPORT §6.7) |
| 4 | `recovery_test.py` | `recovery_test.json` | **negative result**: re-image/discard is not predictable |
| 4b | `06_oct_leakage.py` | `oct_leakage.json` | second benchmark: 40% of test files share a (well, day) with training |
| 5 | `perfield_model.py` | `perfield_*.pt/json` | trains the deployed per-field model (`--arch large`, `--size 512` for the capacity/resolution checks) |
| 5b | `05_lolo_cellline.py` | `lolo_cellline.json` | leave-one-cell-line-out: unseen cell line costs ~7 AUC points |
| 5b | `qc_map.py` | `qc_maps_384.png` | per-chip QC maps |
| 6 | `../evaluate.py` | `tool_evaluation.json` | deployed-tool numbers (section 3 of README) |

Note: `stopping_rule.py` simulates the sequential rule on **ground-truth labels**, which overstates
the deployed tool. `evaluate.py` re-measures it with the model's predictions (9.5 fields, 82.6%),
and `03c_policy_model_in_loop.py` shows that the *policy ranking* from the label simulation does not
survive the model in the loop — the report therefore claims no ranking.
