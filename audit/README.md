# Audit scripts (run order)

Each script writes JSON (and sometimes PNG) into `../results/`.
Set `OOC_DATA` if the dataset is not at `../../data/OOC_image_dataset`.

| order | script | produces | finding |
|---|---|---|---|
| 1 | `leakage_experiment.py` | `leakage_results.json` | published split has 57/59 sessions in train∩test |
| 1b | `leakage_controlled.py` | `leakage_controlled.json` | controlled A/B: +6.1 pp accuracy, +10.2 pp AUC |
| 2 | `structure_probe.py` | `structure_probe.json` | runs test: failures are contiguous |
| 2b | `block_structure.py` | `block_structure.json` | ICC 0.321, design effect 17, effective N ≈ 181 |
| 2c | `label_sufficiency.py` | `label_sufficiency.json` | m-field majority vs session majority |
| 3 | `adaptive_sampling.py` | `adaptive_sampling.json` | random = best for the call, adaptive = best for localisation |
| 3b | `stopping_rule.py` | `stopping_rule.json` | label-only stopping simulation (optimistic) |
| 3c | `aggregation_experiment.py` | `aggregation_results.json` | mean/max/top2 aggregation trade-off |
| 4 | `recovery_test.py` | `recovery_test.json` | **negative result**: re-image/discard is not predictable |
| 5 | `perfield_model.py` | `perfield_*.pt/json` | trains the deployed per-field model |
| 5b | `qc_map.py` | `qc_maps_384.png` | per-chip QC maps |
| 6 | `../evaluate.py` | `tool_evaluation.json` | deployed-tool numbers (section 3 of README) |

Note: `stopping_rule.py` simulates the sequential rule on **ground-truth labels**, which overstates
the deployed tool. `evaluate.py` re-measures it with the model's predictions (9.5 fields, 82.6%).
