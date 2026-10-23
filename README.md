# sat_ext: When Does Slicing Help? SAHI vs. Real-Time Detectors for Small Objects in Satellite Images

Code for a leakage-aware, multi-seed and statistically tested benchmark of **SAHI** (slicing aided hyper inference) against **19 real-time detector configurations** (YOLOv8 to YOLO26 and RT-DETR) for small objects (planes, ships and vehicles) in Google Earth satellite images.

The pipeline runs as Kaggle notebooks on one T4 GPU. Training, testing, the SAHI ablations, the statistics and every paper table and figure are produced by code, not by hand.

> **Main finding.** On images resized to the network input, SAHI significantly raises recall but does not improve COCO AP for any model, and it costs 5–27× more time. Higher-resolution training or a newer architecture is the better use of compute. Slicing helps only when it brings objects back to the size the model saw during training.

---

## Contents
1. [Pipeline overview](#1-pipeline-overview)
2. [Repository layout](#2-repository-layout)
3. [Python package (core modules)](#3-python-package-core-modules)
4. [Notebooks](#4-notebooks)
5. [Planning documents](#5-planning-documents)
6. [Tools and tests](#6-tools-and-tests)
7. [How to run](#7-how-to-run)
8. [Outputs](#8-outputs)
9. [Datasets](#9-datasets)
10. [Requirements](#10-requirements)
11. [Citation and licence](#11-citation-and-licence)

---

## 1. Pipeline overview

```
 Google Earth images (COCO export)
        │
        ▼
 01  Audit: SHA-256 · perceptual hash · ORB/RANSAC overlap ──► ground areas ──► grouped (leakage-free) split
        │
        ▼
 02 / 07  Training: 19 configurations × 3 seeds (Ultralytics, fixed recipe)
        │
        ▼
 Inference: standard · SAHI (slice 480/320/160) · TTA · inference at 1024 · WBF ensembles
        │
        ▼
 Evaluation: COCO AP, AP50, AP_small · P / R / F1 at conf 0.25 · TP/FP · latency
             McNemar + image bootstrap (Holm) · TIDE-style errors
        │
        ▼
 Generalisation: SkySeaLand · cross-dataset · DOTA-v1.0 zero-shot · native 1920×1080 frames
        │
        ▼
 04 / 09 / 12  Paper tables (CSV + LaTeX) and figures
```

---

## 2. Repository layout

```
sat_ext/
├── README.md               ← this file
├── requirements.txt        ← Python dependencies (pinned Ultralytics)
├── config.py               ← every fixed setting (paths, models, recipe, SAHI, metrics)
├── utils.py                ← shared helpers (I/O, seeds, GPU checks, installs)
├── dataset.py              ← load / clean COCO, split manifests, YOLO folders
├── audit.py                ← duplicate and overlap audit, ground-area grouping
├── split.py                ← original and grouped (leakage-free) splits
├── extra_splits.py         ← 1-copy, learning-curve and native-resolution splits
├── train.py                ← one training run + test evaluation
├── infer.py                ← standard / SAHI / TTA inference, latency, metrics
├── metrics.py              ← COCO AP, TP/FP matching, bootstrap, McNemar, Holm
├── analysis.py             ← error types, McNemar tables, WBF ensembles, dataset stats
├── report.py               ← collects all runs → paper tables and figures
├── figures.py              ← 40+ paper figures from saved results
├── dota.py                 ← DOTA-v1.0 external test set
├── relabel.py              ← optional: corrected test labels → re-score
├── roboflow_io.py          ← Roboflow download and release package
├── plans.py                ← which runs each Kaggle account trains
├── notebooks/              ← Kaggle notebooks (run in numeric order)
├── plan/                   ← step-by-step guides
└── tools/                  ← notebook builder, smoke test, fakes, synthetic data
```

---

## 3. Python package (core modules)

### Configuration and helpers

| File | What it does |
|---|---|
| **`config.py`** | The **single source of truth** for every fixed setting. It defines: the Kaggle input and working paths; the class list (plane, ship, vehicle) and class aliases for other datasets (`airplane→plane`, `boat/ship→ship`, `car→vehicle`) plus the DOTA class map; the **model registry** (19 configurations with weights, batch size, input size and group); the **training recipe** (50 epochs, patience 20, AdamW lr 0.001429, linear decay, mosaic/MixUp/copy-paste, rotation 45°, flips, HSV, AMP, deterministic); **evaluation settings** (conf 0.01 for AP, 0.25 operating point, NMS IoU 0.7, max 1,000 detections, match IoU 0.5); **SAHI settings** (slices 480/320/160, overlap 0.2, NMS/IoU 0.5 merge, full-image pass; 640/480/320 on native frames); audit thresholds (pHash ≤ 8 bits, ORB ≥ 25 inliers); seeds; bootstrap iterations (2,000); and the pinned Ultralytics version. Change a setting here and every notebook follows. |
| **`utils.py`** | Shared helpers: `show_inputs` (prints what is attached under `/kaggle/input`), `ensure_dirs`, `set_seed` (Python/NumPy/PyTorch), `timer`, JSON save/load including gzip, `dets_path` / `load_any_json` (reads detection files whether Kaggle kept them gzipped or unzipped them), `sha256_file`, `find_files`, `gpu_info`, `warn_if_gpu_wasted` / `require_gpu` (stop a notebook running on the wrong accelerator), `pip_install` / `install_detection_stack` (pinned Ultralytics + SAHI), and `versions` (logs library versions for reproducibility). |

### Data, audit and splits

| File | What it does |
|---|---|
| **`dataset.py`** | Loads the Roboflow COCO export: finds the COCO roots, detects the export type (640 px or native), and maps category names to the three classes. **Cleans boxes with a full log**: clips boxes to the image, removes boxes under 1 px and exact duplicates, and writes every change to `clean_log.csv`. Builds the image table with the source id (which links augmented copies to their original). Writes split manifests and fingerprints. `materialize()` builds the YOLO folder structure and the per-split COCO ground truth used for evaluation. `find_split_dir()` locates a split in attached Datasets, unpacks zipped splits, re-cuts native tiles on the fly and prefers corrected test labels when they exist. |
| **`audit.py`** | **Leakage audit.** Counts images and boxes, checks box validity and object sizes (micro < 16², tiny 16²–32², medium, large). Finds **exact duplicates** (SHA-256), **near-duplicates** (perceptual hash, minimum over all 8 flips and rotations), and **partial overlap of the same ground area** (ORB features + RANSAC homography, multi-process). `build_groups()` joins all links into connected components called *ground areas*. `leakage_summary()` reports how many areas and images cross the original splits. It also saves montages of suspicious pairs, ground-truth grids and size figures. |
| **`split.py`** | Builds the splits. **`orig`**: the original Roboflow train/valid/test assignment with cleaned labels, which reproduces the earlier paper. **`grouped`**: a leakage-free split in which whole ground areas go to train, val or test with balanced class counts; val/test keep one image per source (the least-rotated copy) and train keeps the augmented copies. `check_no_group_leak()` asserts that no ground area appears in two splits. |
| **`extra_splits.py`** | Splits derived from `grouped`, with val/test **identical** so that all comparisons are paired. `one_copy` keeps one augmented copy per training source (augmentation ablation). `learning_curve` keeps 25% or 50% of the training ground areas. `native_manifest` covers full 1920×1080 frames, and `make_tiles` cuts 640 px tiles at native resolution with overlap 0.2. `build_all()` creates them all. |

### Training and inference

| File | What it does |
|---|---|
| **`train.py`** | Trains one *(split, model, seed)* run with the fixed recipe from `config.py`, keeps `best.pt` (chosen on validation), and immediately calls the test evaluation in `infer.py`. Finished runs found in the attached Datasets are **skipped**, so a notebook can be restarted after a Kaggle timeout. `run_plan()` runs the list of jobs for one account; `run_ablation()` runs the data-ablation jobs. `expected_h()` estimates GPU hours per job. |
| **`infer.py`** | Test-set evaluation of one trained model. `ultralytics_test_metrics()` runs the Ultralytics validator on the test split (not validation). `predict_standard()` does one forward pass, optionally with TTA (`augment=True`) or a different input size (`res1024`). `sahi_model()` / `predict_sahi()` run SAHI with slice size, overlap and merge rule. `n_slices()` / `_passes()` count forward passes per image. `evaluate_run()` scores standard + SAHI modes and writes `eval_test.json`, the detections and per-image CSVs. `sahi_ablation()` varies merge rule (NMS/NMM/GREEDYNMM), match metric (IoU/IoS), confidence floor, overlap and full-image pass. `extra_modes()` adds TTA and inference at 1024. **Latency** is wall-clock per image (file → final boxes), with CUDA synchronisation and warm-up. `model_cost()` reports parameters and GFLOPs. |
| **`dota.py`** | External test set **DOTA-v1.0** (Xia et al., CVPR 2018). Finds DOTA in the inputs, parses the label files, converts oriented polygons to axis-aligned boxes, maps `plane`, `ship`, `small-vehicle` and `large-vehicle` to our classes, and builds a COCO ground truth for zero-shot testing (standard, SAHI-640, SAHI-1024). |

### Metrics, statistics and analysis

| File | What it does |
|---|---|
| **`metrics.py`** | `coco_eval()` computes COCO AP / AP50 / AP75 / recall per size range (micro, tiny, small, medium, large) and per class with **pycocotools** (an identical internal implementation is used if pycocotools is missing). `match_detail()` / `match_per_image()` match detections to objects one-to-one at IoU ≥ 0.5 and conf ≥ 0.25, giving TP, FP, FN, duplicates and recall per size bucket. `gt_hits()` marks each object as found or missed for McNemar. `bootstrap_delta()` runs a **paired image-level bootstrap** (same resampled images for both methods; 95% percentile CI; two-sided p). `mcnemar()` is the **exact McNemar test** (binomial). `holm()` applies the **Holm** multiple-comparison correction. |
| **`analysis.py`** | CPU-only analyses on saved detections. `error_breakdown()` gives **TIDE-style error types** (duplicate, classification, localisation, cls+loc, background, missed by size). `table_mcnemar()` compares all model pairs with object-level McNemar + image-level bootstrap, Holm-corrected; `table_mcnemar_modes()` compares SAHI vs. standard per model. `table_ensembles()` builds **Weighted Box Fusion** ensembles (seeds, architectures, standard + SAHI). `dataset_stats()` reports dataset statistics, and `agreement()` an optional annotator-agreement check. |
| **`report.py`** | Collects every `eval_test.json` from all attached result Datasets (newest or corrected run wins) into one long table (`all_runs_long.csv`) and builds the **paper tables** as CSV + LaTeX: main benchmark, standard vs. SAHI, bootstrap, per-class, original vs. grouped split, cross-dataset, native resolution, SAHI ablation, DOTA, data ablations, efficiency. It also draws the Pareto front, size-wise recall, extra detections and qualitative figures. |
| **`figures.py`** | Notebook 12: **40+ independent paper figures** from saved results, with no new training. They cover class counts, size histograms, objects per image, centre heat maps, aspect ratios, split composition, sample images, benchmark bars, AP vs. parameters / GFLOPs, per-class heat map, seed spread, SAHI curves, TP/FP, recall by size, latency, ablations, training resolution, inference strategies, split comparison, native resolution, transfer, DOTA, error types, confidence histograms and qualitative comparisons. A figure whose data is missing is skipped and reported, never faked. |

### Release and planning

| File | What it does |
|---|---|
| **`relabel.py`** | *Optional* test-label correction. `export_for_relabel()` packs the 82 test images with their current boxes for Roboflow. `load_relabeled()` matches the re-exported images by file-name hash or thumbnail. `corrected_gt()` / `change_report()` build the corrected ground truth and a before/after table. `rescore_all()` **re-scores every saved detection file** on the corrected labels without retraining. |
| **`roboflow_io.py`** | Optional Roboflow download (needs an API key), comparison of two exports, and `build_upload_package()`, which builds a cleaned dataset release with the grouped split and a dataset card (`datacard`). |
| **`plans.py`** | Which runs each Kaggle account trains, so that no account exceeds the weekly GPU quota. Planning hours per model come from real training logs. `all_runs()` lists every planned run, `expected_run_ids()` is used to check that nothing is missing, and `show()` prints the plan. |
| **`requirements.txt`** | `ultralytics==8.4.37`, `sahi`, `pycocotools`, `opencv-python-headless`, `scipy`, `pandas`, `matplotlib`, `roboflow` (download only). |

---

## 4. Notebooks

Every notebook first prints the attached inputs, then imports `sat_ext` from the attached code Dataset. All settings come from `config.py`, so nothing needs editing.

### Stage 1: data audit and splits

| Notebook | GPU | What it does | Output Dataset |
|---|---|---|---|
| **`01-audit-and-splits`** | no | Cleans the COCO export, runs the full leakage audit (SHA-256, pHash, ORB/RANSAC), builds ground areas and the `orig` + `grouped` splits, and saves audit figures and tables. | `sat-splits` |
| **`01b-audit-and-splits-skysealand`** | no | The same audit and grouped split for the second dataset (SkySeaLand), with class mapping. | `sat-splits-sky` |
| **`06-make-extra-splits`** | no | Builds the 1-copy, learning-curve (25/50%) and native-resolution splits (optional; notebooks 07I–07L can also build them themselves). | `sat-splits-extra` |

### Stage 2: training (one notebook per Kaggle account / time budget)

| Notebook | Runs |
|---|---|
| **`02a-train-rtdetr-l-grouped-s42-s43`** | RT-DETR-L, grouped split, seeds 42, 43 |
| **`02b-train-rtdetr-l-grouped-s44-and-orig-s42`** | RT-DETR-L seed 44 + original split seed 42 |
| **`02c-train-yolo11m-grouped-3seeds`** | YOLO11m, 3 seeds |
| **`02d-train-yolov8m-grouped-3seeds`** | YOLOv8m, 3 seeds |
| **`02e-train-yolo11n-yolo11s-grouped-3seeds`** | YOLO11n and YOLO11s, 3 seeds each |
| **`02f-train-4yolo-orig-split-s42`** | Four YOLO models on the original split (split comparison) |
| **`02g-train-skysealand-5models-s42`** | The five original models on SkySeaLand + cross-dataset tests |
| **`07a-train-hires-yolo11s-1024-1280-3seeds`** | YOLO11s at 1024 and 1280 px |
| **`07b-train-hires-yolo11m-1024-3seeds`** | YOLO11m at 1024 px |
| **`07c-train-hires-yolo11m-1280-s42-s43`** | YOLO11m at 1280 px, seeds 42, 43 |
| **`07d-train-hires-yolo11m-1280-s44-and-yolov8s-p2`** | YOLO11m-1280 seed 44 + YOLOv8s-P2 |
| **`07e-train-arch-yolov8s-v9s-v10s-3seeds`** | YOLOv8s, YOLOv9s, YOLOv10s |
| **`07f-train-arch-yolo12s-yolo26s-and-yolov8m-p2`** | YOLO12s, YOLO26s, YOLOv8m-P2 |
| **`07g-train-arch-yolov9m-v10m-3seeds`** | YOLOv9m, YOLOv10m |
| **`07h-train-arch-yolo12m-yolo26m-3seeds`** | YOLO12m, YOLO26m |
| **`07i-train-ablation-1copy-learning-curve`** | Data ablations: 1 augmented copy; 25% / 50% of ground areas |
| **`07j-train-native-yolo11s-and-yolo11m-640`** | Full native frames resized to 640 |
| **`07k-train-native-tiles-yolo11m`** | 640 px tiles cut at native resolution |
| **`07l-train-native-yolo11m-1280`** | Full native frames at 1280 |

Every training run also evaluates itself on the test split (standard + SAHI-480/320/160, latency) and saves `eval_test.json`, the detections and per-image tables. Save each notebook's output as a Kaggle Dataset named `sat-results-<id>`.

### Stage 3: extra inference experiments (GPU, no training)

| Notebook | What it does |
|---|---|
| **`03-rescore-legacy-weights`** | Re-scores the weights of the earlier manuscript correctly on the test split, and runs cross-dataset tests. |
| **`08a-sahi-ablation-main`** / **`08a1-sahi-ablation-main`** | SAHI ablation on the main dataset: merge rule, IoU vs. IoS, confidence floor 0.01 vs. 0.25, overlap 0.1/0.3, no full-image pass. 08a1 runs the remaining models, so that each part fits the Kaggle time limit. |
| **`08b-sahi-ablation-skysealand`** / **`08b1-sahi-ablation-skysealand`** | The same ablation on SkySeaLand (two parts). |
| **`08c-external-test-dota`** | Zero-shot test on 300 DOTA-v1.0 validation images (standard, SAHI-640, SAHI-1024). |
| **`08d-tta-and-inference-resolution`** | Test-time augmentation and inference at 1024 px with 640-trained weights. |

### Stage 4: analysis, tables and figures (CPU)

| Notebook | What it does |
|---|---|
| **`04-paper-tables-figures`** | Collects all `sat-results-*` Datasets and writes every paper table (CSV + LaTeX) and the main figures to `results/paper/`. |
| **`09-errors-mcnemar-ensembles`** | TIDE-style error analysis, McNemar + bootstrap tests (all model pairs; SAHI vs. standard), WBF ensembles, for the main dataset, native frames and SkySeaLand. |
| **`10-dataset-statistics-and-agreement`** | Dataset statistics (per class, sizes, co-occurrence) and the dataset comparison template. |
| **`11-label-quality-check`** | Shows the highest-confidence false positives as crop sheets (`fp_review.zip`) so that missing labels can be checked by eye. |
| **`12-paper-figures`** | Generates the 40+ figures in `figures.py` from the saved results. |
| **`05-roboflow-export`** | Builds the cleaned dataset release (640 px or native) with the grouped split and a dataset card. |

---

## 5. Planning documents

| File | Contents |
|---|---|
| **`plan/00-START-HERE.md`** | Step-by-step run order, which Kaggle Dataset to attach to each notebook, account plan, and how to save outputs. **Read this first.** |
| **`plan/01-PAPER-FIXES-and-REVIEWER-MAP.md`** | What was wrong in the earlier pipeline (e.g. metrics computed on validation instead of test, conf 0.25 used for mAP) and which code change fixes each problem, mapped to the reviewer comments. |
| **`plan/03-ROBOFLOW-GUIDE.md`** | How to check and fix the public Roboflow dataset and prepare a corrected release. |

---

## 6. Tools and tests

| File | What it does |
|---|---|
| **`tools/build_notebooks.py`** | Generates all notebooks in `notebooks/` from one script, so their setup cells stay identical. Edit this file, not the `.ipynb` files. |
| **`tools/smoke_test.py`** | **End-to-end test on CPU** in a few minutes: builds a synthetic dataset, then runs every notebook in order with fake Ultralytics/SAHI, imitating Kaggle behaviour such as Datasets being un-gzipped. It checks the outputs of each stage. |
| **`tools/test_metrics.py`** | Unit tests for the internal COCO evaluator (same algorithm as pycocotools) and the TP/FP matcher in `metrics.py`. |
| **`tools/make_synthetic.py`** | Creates a fake Roboflow-style COCO export for the smoke test, with **planted problems** that the audit must find: augmented copies with rotations, a test tile overlapping a training tile (ORB), a mirrored copy under a new name (pHash), a shared source name (source id), and broken boxes (negative coordinates, boxes beyond the border, exact duplicates). |
| **`tools/fakes/ultralytics/`**, **`tools/fakes/sahi/`** | Lightweight stand-ins for Ultralytics and SAHI with the same interfaces, so the pipeline can be tested without a GPU or downloads. |

```bash
python tools/test_metrics.py     # unit tests
python tools/smoke_test.py       # full pipeline on CPU with fakes
```

---

## 7. How to run

1. **Code:** upload this folder as a Kaggle Dataset named `sat-ext`.
2. **Data:** attach the COCO export of the satellite dataset (640 px; the native version for 07J–07L) and, for the second dataset, SkySeaLand in COCO format.
3. **Order:** `01` → `01b` → (`06`) → `02*` / `07*` → `03`, `08*` → `09`, `10`, `11` → `04`, `12` (→ `05`).
4. In each notebook: attach the inputs listed in its first cell, choose the accelerator (GPU T4 for training and inference, *None* for analysis), and **Run All**.
5. Save each notebook's output as a Kaggle Dataset (`sat-splits`, `sat-results-<id>`, …) and attach it to the later notebooks.
6. Runs that are already finished are skipped automatically, so an interrupted notebook can simply be run again.

---

## 8. Outputs

| Location | Contents |
|---|---|
| `splits/<name>/` | Split manifest (`manifest.csv` with ground-area id), COCO ground truth per split, fingerprint |
| `audit/` | Audit tables, duplicate/overlap pair montages, size figures, `clean_log.csv` |
| `results/<run_id>/` | `eval_test.json` (all metrics per mode), detections (`dets_<mode>.json.gz`), per-image CSVs, training log |
| `results/paper/` | Paper tables `T1`–`T15`, `D1`–`D4` (CSV + LaTeX), `all_runs_long.csv` (every run × mode), figures |

Key metrics in `eval_test.json`: COCO AP, AP50, AP75, AP/AR per size bucket and class; TP, FP, FN, duplicates, precision and recall at conf 0.25; micro/tiny recall; latency (ms per image) and passes; parameters and GFLOPs.

---

## 9. Datasets

| Dataset | Use | Link |
|---|---|---|
| Dataset version | Purpose | Link |
|---|---|---|
| Satellite dataset (this study), 640 px COCO export | Main experiments | [Roboflow Dataset v2](https://app.roboflow.com/rifat4018/aerial-object-detection-u3f5u/2) |
| Satellite dataset (this study), native resolution | Native-frame experiments (07J–07L) | [Roboflow Dataset v3](https://app.roboflow.com/rifat4018/aerial-object-detection-u3f5u/3) |
| **SkySeaLand** (airplane, boat, car, ship → plane, ship, vehicle; CC BY 4.0) | second dataset | Mendeley Data: https://data.mendeley.com/datasets/d42n3cp86p/3 · Roboflow Universe: `aquantic-research/skysealand-tf38k` |
| **DOTA-v1.0** (oriented boxes → axis-aligned) | zero-shot external test (300 val images) | https://captain-whu.github.io/DOTA/ |

The satellite dataset contains 544 Google Earth frames with 18,872 boxes (plane, ship, vehicle). Google Earth imagery is subject to Google's terms of use.

---

## 10. Requirements

- Python ≥ 3.10, Kaggle GPU (T4) for training/inference; CPU is enough for notebooks 04, 09, 10, 11, 12 and the tests
- `ultralytics==8.4.37`, `sahi`, `pycocotools`, `opencv-python-headless`, `scipy`, `pandas`, `matplotlib`

```bash
pip install -r requirements.txt
```

---

## 11. Citation and licence

Code: https://github.com/Rifat4018/Satelite-coco-Native

If you use this code or dataset, please cite the accompanying paper (citation will be added after publication).

Code: add your licence here (e.g. MIT). Google Earth imagery is subject to Google's terms of use and is not redistributed here.
