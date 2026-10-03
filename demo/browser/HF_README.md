---
title: Organ-on-a-chip QC reference replay
emoji: 🧫
sdk: static
app_file: demo/browser/index.html
license: mit
---

# Organ-on-a-chip QC — reference replay and exploratory field scores

This corrected Space source separates **cached Python reference replay** from **live field scores**.

Bundled examples replay Python probabilities for the 44 stored 1,024 × 768 images. They do not reproduce every full-resolution report result exactly. The borderline reference replay is a wrong, low-confidence PASS; the original full-resolution Python case is inconclusive.

Live uploads accept only 8-bit grayscale/RGB non-interlaced still PNGs without transparency, EXIF or color-profile chunks; other formats and animation are rejected. The page displays exploratory per-field probabilities but suppresses operational PASS/FAIL and chip confidence. The browser input path is not a validated instrument. Raw resizing was checked on the bundled PNGs in Chromium, but this is not evidence for every image format, browser or physical chip. Use the Python reference for reproducing the experiment.

The reported experiment uses **25 withheld half-session proxies**, not identified physical chips: 19/23 called proxies correct, four wrong calls (three at confidence ≥0.9); the minimum of eight was test-selected. Without those test proxies the selected minimum is one: 0.708 on 24 calls, 0.680 forced. A decision budget of 9.5 versus 27.4 fields matched 0.800 on this sample. Validation's 12 vs 5 discordant halves are clustered descriptive counts, with no iid McNemar p reported; the one-proxy-per-session test comparison gives p = 0.22. All fields are still scored for the QC view; no imaging, inference or expert-review time saving was measured.

Code and report: https://github.com/Foreist/when-is-one-field-enough

Data: OOC Image Dataset, https://doi.org/10.5281/zenodo.10203721 (Zenodo CC-BY-4.0; descriptor CC-BY-SA), 44 attributed example fields only. Code is MIT; Pillow-style resampling retains MIT-CMU notices. ONNX Runtime Web 1.20.1 and the pinned model artifact are external runtime dependencies; local Python execution is the fallback.

Deployment note: upload `demo/browser/` and `demo/examples/` with the relative paths preserved, set this card as the Space README, and test the deployed URL. These files are prepared locally; they have not replaced the existing public Space.
