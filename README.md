# sat_ext – SAHI satellite detection, Kaggle-ready (v3.5 - 13 split into 13A export + 13B re-score; v3.4: 13 fix test labels & re-score all saved detections (no retraining); + 12 paper figures (48); reads Kaggle-unzipped detections; 09 covers native (incl. ensembles) + SkySeaLand; 05 native release; input links in every notebook; 08D TTA, 11 label check, T15 efficiency)

Same delivery style as `rmhd_ext`: upload this folder as the Kaggle Dataset `sat-ext`, import the notebooks, attach inputs, Run All.
**Read `plan/00-START-HERE.md` first.**

Notebooks: `00` Roboflow download (optional) · `01` audit + leakage-free split · `01b` same for the second dataset (SkySeaLand) ·
`02A–02F` training per account (nothing to edit), `02G` second dataset · `03` re-score the old paper weights + cross-dataset test · `04` paper tables and figures · `05` Roboflow upload package + description.

Package: `config.py` (every fixed setting) · `utils.py` · `dataset.py` (load, clean with log, splits, YOLO folders) ·
`audit.py` (SHA-256, pHash, ORB overlap, grouping) · `split.py` · `train.py` · `infer.py` (test metrics, SAHI, latency) ·
`metrics.py` (COCO AP via pycocotools or the internal copy, TP/FP matching, bootstrap) · `report.py` (tables, figures) ·
`plans.py` (runs per account) · `roboflow_io.py` · `tools/` (synthetic data, fakes, smoke test, notebook builder).

Tests: `python tools/smoke_test.py` and `python tools/test_metrics.py` (both pass on CPU, with fake Ultralytics/SAHI).

