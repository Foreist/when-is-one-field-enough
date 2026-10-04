# Browser reference replay and exploratory preview

This tree is the source for the corrected static demo deployed to the existing HF Space (input/output guard update on 2026-10-04: `5429258a96151aca94267ec2fc805199473f0d55`; UI update: `645382d0697bd2df6c01fba8c8530ddba7ee3add`; initial corrected release: `852a6c1624268779a9a66d662a64f6001131ee12`). Controls have visible labels and permanent input-format help; narrow screens use a named, keyboard-focusable table scroll region instead of overflowing the page.

- Bundled buttons replay Python probabilities on `demo/examples/` images, not the full-resolution report inputs. The borderline example is a wrong, low-confidence reference call.
- Live uploads display **field scores only**. PASS/FAIL and chip confidence are suppressed until the actual decoding/preprocessing/model path is validated. Only 8-bit grayscale/RGB non-interlaced still PNGs without EXIF/color-profile chunks are accepted; JPEG/BMP, animation, other bit depths and transparent/profiled PNGs fail with an unsupported-input message. This does not turn a research prototype into a validated instrument.
- Invalid reference manifests/probabilities or model outputs fail without a decision or score table. Cached probabilities must be finite numeric values in [0,1]; pinned model logits must have shape [1,2] and finite values. This input/output contract does not establish clinical validity.
- All fields are scored for the view. Decision-rule consumption is not measured inference, microscope or operator time saving.
- `preprocess.js` provides an experimental Pillow-style bilinear resampler, with MIT-CMU attribution in `LICENSE-PILLOW.txt`. Do not infer production equivalence from its implementation or identical-tensor ONNX parity alone.
- `chips.json` pins the existing HF ONNX revision and SHA-256. `audit/export_onnx.py` recreates a model from the shipped PyTorch checkpoint; export metadata records provenance.

Serve the public repo root locally, then open `/demo/browser/index.html`. The runtime is pinned to ONNX Runtime Web 1.20.1 via jsDelivr; live inference requires that CDN and the pinned HF model download. The cached reference replay does not require a model.

For an HF deployment, preserve the relative `../examples/` paths by uploading this page under `demo/browser/` and the examples under `demo/examples/`, or explicitly adjust the manifest paths. Do not overwrite the existing Space without reviewing the deployment diff and obtaining publishing approval.
