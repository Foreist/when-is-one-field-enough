# Organ-on-a-chip QC: audit, corrected protocol, and a field-sampling tool

This repository contains (1) an **audit** of a public organ-on-a-chip (OoC) quality-control
benchmark, (2) a **corrected evaluation protocol**, and (3) a **chip-level QC tool** that decides
whether a chip passes or fails from a handful of brightfield fields, with a calibrated confidence
and an explicit "inconclusive" outcome.

**Deliverables**

| | |
|---|---|
| Technical report (20 pages, PDF) | [`report.pdf`](report.pdf) · source: [`REPORT.md`](REPORT.md) |
| Demo video (4:41, narrated) | [`demo_video.mp4`](demo_video.mp4) |
| **Interactive demo** (permanent, no server, no login) | **https://taewoong23-ooc-chip-qc-demo.static.hf.space/index.html** |
| Reproducible results | [`results/`](results/) — one JSON per claim, written by the script that made it |

Everything runs from one public dataset. Every number in this README is reproduced by the scripts
in `audit/` and `evaluate.py`; the raw outputs are in `results/`.

---

## 0. Quickstart

```bash
pip install -r requirements.txt
python3 inference.py --plate demo/examples --out out/   # 3 bundled chips — no dataset needed
python3 demo/app.py                                     # interactive demo (Gradio) at :7861
```

The tool, the demo, `figures.py` and `make_report_pdf.py` run without the dataset (figures and the
PDF are rebuilt from the committed `results/*.json`; the PDF additionally needs
`playwright install chromium`). `evaluate.py` and the `audit/` scripts need the full dataset from §1.

---

## 1. Data

**OOC Image Dataset** — https://doi.org/10.5281/zenodo.10203721 (CC-BY-4.0; the MDPI data
descriptor lists CC-BY-SA — we therefore do **not** redistribute the full dataset, only the 44 demo
example fields, with attribution; download the full dataset from Zenodo).
3,072 brightfield images (2056x1542) from an automated microscope on an OoC setup, 6 cell lines,
**59 acquisition sessions** (file names `YYMMDD_N.png`), labels `good`/`bad` assigned by four
cell-biology experts (majority vote), plus protocol metadata (seeding density, flow rate, day).
Reference paper: Movčana et al., *Data* 2024, 9, 28 (`10.3390/data9020028`).

Download and extract to `../data/OOC_image_dataset/` (or set `OOC_DATA`):

```bash
mkdir -p ../data
curl -L -o ooc.zip "https://zenodo.org/api/records/10203721/files/OOC_image_dataset.zip/content"
python3 -c "import zipfile; zipfile.ZipFile('ooc.zip').extractall('../data')"   # unzip(1) fails on this zip64
rm ooc.zip
```

The zip contains a top-level `OOC_image_dataset/` folder, so this lands at `../data/OOC_image_dataset/`.

## 2. What we found (audit)

| # | Finding | Measurement | Script |
|---|---|---|---|
| 1 | **The published split leaks sessions** | train∩test = **57/59 sessions**. Controlled A/B (same test images, same training size, 8 seeds): **+8.2 pp accuracy (95% CI 6.0–10.4) / +9.6 pp AUC (8.4–10.8)** inflation | `audit/leakage_controlled.py` |
| 2 | **The 3,072 labels are not 3,072 independent observations** | session ICC 0.321 → design effect 17 → **effective N ≈ 181**; lag-1 autocorrelation 0.32; run length 6.08 vs 2.03 under i.i.d. | `audit/block_structure.py` |
| 3 | **Failures occupy contiguous stretches of the chip** | runs test: 23/45 sessions p<0.05, 36/45 clustered; survives cell-type control (38/72); not duplicates (98% distinct views) | `audit/structure_probe.py` |
| 4 | **Which fields are read matters — but no policy ranking is claimed** | label-based simulation suggests random > scan > adaptive (0.907 at k=8); with the model in the loop the ordering changes and **all paired differences include zero** (13–15 chips). The tool uses spread sampling because reading the *first* k fields called a 100%-bad chip *pass* with 0.94 confidence | `audit/adaptive_sampling.py`, `audit/03c_policy_model_in_loop.py` |
| 4b | **The leakage is not unique to this benchmark** | a second organoid benchmark (OCT organoid tracking, zenodo.15783866) has **40.0% of test files in a (well, day) group that also appears in training** | `audit/06_oct_leakage.py` |
| 5 | **A re-image/discard rule is NOT supported** | recovery target CV AUC 0.613 vs permutation null p95 0.564; simple rules worse than the base rate → reported as an open problem | `audit/recovery_test.py` |

Known prior art we build on (leakage in benchmarks is a known class of problem):
*Data Leakage in Visual Datasets* (ICCV 2025 W / arXiv 2508.17416); *Auditing Data Leakage in
Whole-Slide Image Benchmarks* (arXiv 2607.12278); Tampu et al., *Sci Data* 2022 (OCT split leakage);
`AutoQC-Bench` (2025) for microscopy QC benchmarks.

## 3. The tool

```bash
python3 inference.py --images /path/to/one_chip_fields --out out/
```

Outputs `out/chip_report.json` and `out/qc_map.png`:

* **per-field P(bad)** (continuous, not a hard label)
* **chip call**: `pass` / `fail` / **`inconclusive`** with a posterior confidence
* **how many fields were used** (fields are sampled *spread across the chip*, not the first k —
  reading the first k can land inside a good region and stop early with a wrong confident call)
* **model card** with the measured performance, so the number is never read out of context

### Measured performance (25 unseen chips, 684 fields, session-disjoint)

| metric | value |
|---|---|
| per-field accuracy / AUC | 0.734 / 0.791 |
| chip accuracy, all fields, mean | 0.80 |
| chip accuracy **among confident calls** (sequential, spread fields, min 8) | **0.826** (95% Wilson CI 0.63–0.93, 23 calls) |
| **false-confident calls** (confident and wrong) | **13%** |
| inconclusive (budget exhausted near P=0.5) | 8% |
| **fields used** | **9.5 per chip vs 27.4 for the read-everything baseline — same accuracy (0.800), 2.9× fewer fields** |
| held-out cell line (leave-one-cell-line-out, 6 folds) | accuracy **0.670**, AUC **0.719** (vs 0.734 / 0.791 in-distribution) |
| larger backbone / higher resolution | no gain (AUC 0.788 with MobileNetV3-large; 0.797 at 512 px) |

`evaluate.py` reproduces this table on the session-disjoint split.

### Demo

```bash
python3 demo/app.py          # http://127.0.0.1:7861
```

**Plate triage** — `python3 inference.py --plate /path/to/plate --out out/`, where the folder contains
one subfolder per chip. Chips are ranked by how much attention they need, and the run reports how
many fields were spent. On the 25 test chips: **238 of 684 fields used (65% saved)**, with
6 chips called *fail*, 2 *inconclusive* and 17 *pass*. The demo app has the same mode in its
"plate triage" tab.

Pick a bundled example chip or upload the fields of your own chip; the app shows the chip call,
the posterior confidence, the number of fields used and the per-field QC map. Three example chips
are bundled (12/12/20 fields, with attribution in `demo/examples/README.md`): a mostly-good chip, a
100%-bad chip and a borderline chip.

### Input / output formats

* **Input** — a folder of field images (`.png`, `.jpg`, `.tif`), one folder per chip; the file names
  are read in natural order (`230425_7.png` before `230425_10.png`) and that order is treated as the
  acquisition order. With `--plate`, the given folder must contain one subfolder per chip.
* **Output** — `chip_report.json` (per-field P(bad), chip call, posterior confidence, fields used,
  model card) and `qc_map.png`; in plate mode also `plate_summary.csv` with one row per chip
  (`chip, call, p_bad, confidence, fields_used, fields_available, attention_rank`).

## 4. Limitations (read this before using the tool)

* **13% confident-but-wrong** on unseen chips (4 of 23 calls wrong, 3 of them with ≥0.9
  confidence). This is a research prototype, not a
  validated instrument; do not discard a chip on its output alone.
* **No reliable out-of-distribution detector.** We tried image-statistics and feature-space
  Mahalanobis distances; both missed the worst failure (a 100%-bad chip called `pass` with 0.94
  confidence). The `image_statistics_distance` field is diagnostic only.
* **The re-image vs discard recommendation is not validated** — the recovery target does not support
  it (finding 5). The tool reports `pass`/`fail`/`inconclusive`, nothing more.
* **Single dataset, 25 test chips.** No wet-lab validation; we make no claim about biology or
  clinical validity. The claim is about *measurement validity*.
* Simulating decision rules on ground-truth labels **overstates** real performance (label-only
  simulation suggested 6.6 fields at 94%; the deployed model gives 9.5 fields at 82.6%).

## 5. Layout

```
inference.py            chip-level tool (per-field P(bad) -> call + confidence + fields used)
evaluate.py             session-disjoint evaluation of the tool (reproduces section 3)
model/                  MobileNetV3-small checkpoint + OOD reference statistics
audit/                  audit scripts (findings 1-5), each writing JSON into results/
results/                raw outputs: leakage, block structure, sampling, tool evaluation, QC maps
```

## 6. Licence and attribution

Code: MIT (see `LICENSE`). Derived artefacts (predictions, figures): CC-BY-SA, matching the more
restrictive of the dataset's two licence statements. The full dataset is **not** redistributed; only
the 44 demo example fields in `demo/examples/` are included, with attribution. Download the full
dataset from Zenodo and cite:

> Movčana, V.; Strods, A.; et al. Organ-On-A-Chip (OOC) Image Dataset for Machine Learning and
> Tissue Model Evaluation. *Data* 2024, 9, 28. DOI 10.3390/data9020028. Dataset: 10.5281/zenodo.10203721.
