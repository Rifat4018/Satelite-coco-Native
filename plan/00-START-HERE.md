# SAT project – START HERE (v1, 20 Oct 2026)

Revision of *SAHI-Augmented Multi-Architecture Satellite Image Object Detection* (paper 355 / 524).
Everything is retrained **from scratch** with a fixed, explicit recipe, then scored correctly on the TEST split.

## Which dataset to use
**Use your Kaggle Dataset `rifat4018/satelite-coco`.** It is the exact Roboflow COCO export your paper used,
it is frozen (cannot change under you), and every account can attach it without an API key.
* Roboflow `aerial-object-detection-u3f5u` is the same data *if* it is the same version. Notebook **00** downloads it
  (secret `ROBOFLOW_API_KEY`, Internet ON) and prints whether it is identical to the Kaggle copy. Do this once; never
  download per account (a different version number = different files = runs that cannot be compared).
* The other accounts need access: Kaggle -> the dataset -> Settings -> Sharing -> add each username
  (or make it public only if the image licence allows redistribution).

## Second dataset: SkySeaLand v14 (the files you sent on 20 Oct)
Those annotation files are **not** the paper's dataset. They are *SkySeaLand v14* (Roboflow Universe
`aquantic-research/skysealand-tf38k`, CC BY 4.0): 1048/127/132 native screenshots (~1918×850), classes airplane/boat/car/ship,
no augmentation and almost no box errors (39 sub-pixel clips, 1 degenerate box). They answer reviewer 524-R4 ("single dataset").
* Download it from Roboflow in **COCO** format and upload it as the Kaggle Dataset `skysealand-v14`.
* Classes are mapped onto ours: airplane→plane, boat+ship→ship, car→vehicle (`cfg.CLASS_ALIASES`). Say so in the paper.
* It is much less small-object-heavy (about 13% of boxes under 32² px² vs 69% in yours), so expect a smaller SAHI gain. That is a finding, not a failure.
* Cite it (CC BY 4.0 requires attribution).

## Kaggle setup (once per account)
1. Upload this `sat_ext/` folder as a Kaggle Dataset named `sat-ext` (the `notebooks/` folder is not needed inside it).
2. File -> Import Notebook -> choose the `.ipynb`.
3. Add Input: the Datasets listed at the top of each notebook. Set Accelerator / Internet as stated there.
4. Every notebook starts with `show_inputs()`. Read it before trusting the rest.

## Run order
| Step | Notebook | Account | HW | Est. time | Attach | Make Dataset from Output |
|---|---|---|---|---|---|---|
| 1 | `01-audit-and-splits` | A | CPU | 10-25 min | `satelite-coco`, `sat-ext` | **`sat-splits`** (share with B-F) |
| 2 | `02E-train-account-E` with `PILOT = True` | E | GPU+Net | ~10 min | `satelite-coco`, `sat-splits`, `sat-ext` | – (pipeline check) |
| 3 | `02A` … `02F` with `PILOT = False`, **Save & Run All** | A…F in parallel | GPU+Net | 2.5-4.5 h each | same three | `sat-results-a` … `sat-results-f` |
| 3b | `01b-audit-and-splits-skysealand` (optional) | any | CPU | 10-30 min | `skysealand-v14`, `sat-ext` | **`sat-splits-sky`** |
| 3c | `02G-train-account-G` = SkySeaLand, 5 models s42 (optional) | 2nd session on C/D/E/F | GPU+Net | ~5.5 h | `skysealand-v14`, `sat-splits-sky`, `sat-ext` | `sat-results-g` |
| 4 | `03-rescore-legacy-weights` (+ cross-dataset cell) | any | GPU+Net | 30-60 min | + `resultsyolo11`, `resultsrfdetr` (+ sky inputs, results A-G) | `sat-results-legacy` |
| 5 | `04-paper-tables-figures` | A | CPU | 5-15 min | everything above | – |
| 6 | `05-roboflow-export` | A | CPU | 5 min | `satelite-coco`, `sat-splits`, `sat-ext` | – (download the zip) |

**What each account trains** (`plans.py`; hours = planning estimates from your own logs, not measurements):

| Account | Runs | ~GPU h |
|---|---|---|
| A | grouped RT-DETR-L s42, s43 | 4.2 |
| B | grouped RT-DETR-L s44 + orig RT-DETR-L s42 | 4.2 |
| C | grouped YOLO11m s42, s43, s44 | 3.0 |
| D | grouped YOLOv8m s42, s43, s44 | 3.0 |
| E | grouped YOLO11n ×3 + YOLO11s ×3 | 3.0 |
| F | orig YOLO11n, YOLO11s, YOLO11m, YOLOv8m (s42) | 3.0 |

Optional G: SkySeaLand grouped split, YOLO11n/s/m, YOLOv8m, RT-DETR-L, seed 42 (about 5.5 h). Notebook 03 then tests every
model on the other dataset without fine-tuning, which produces T6 (cross-dataset generalisation).

Total ≈ 20 GPU hours (+5.5 h for G). That fits in the weekly quota (30 h) of **two** accounts; six only make it finish in one afternoon.
Note: Kaggle's terms allow one account per person. Use your co-authors' accounts, not extra accounts of your own,
or the accounts (and their results) can be banned.

If a session stops early, run the same notebook again: finished runs are skipped, and weights without an evaluation
are evaluated.

## The two splits
* `orig`: your original Roboflow train/valid/test, with cleaned boxes. It reproduces the paper's data. Accounts B and F train on it, to measure how much the split matters.
* `grouped`: images that show the same ground area are kept together (augmented copies, duplicates, pHash near-duplicates, ORB-overlapping tiles). There is no ground area in two splits. **This split gives the main results (3 seeds).**

## Verified vs not verified
* `python tools/smoke_test.py` passes. It builds a synthetic Roboflow-style export with planted leaks (an overlapping tile, a mirrored copy, a reused source name). It then runs every notebook end to end, in separate processes, moving outputs between them the way Kaggle Datasets do. All three planted leaks were found and kept in one split. `python tools/test_metrics.py` checks the COCO-AP code against hand-computed cases.
* **NOT verified:** real Ultralytics training, Ultralytics `val`, and SAHI. The smoke test replaces them with fakes in `tools/fakes/`, because pip was blocked where this was built. All real timings are also unverified. **That is why step 2 (PILOT) comes first.**
* The first evaluation in each session prints `pycocotools vs internal AP`. The two numbers must be equal.

## Phase 2 (v2, 20 Oct): Tier 1 + 2 for the journal version
See `SAT_Phase2_Run_Guide.docx` for the full schedule. New notebooks:
`06` extra splits (1-copy ablation, learning curve, native resolution, tiles) · `07A-07L` training
(more architectures, P2 heads, 1024/1280 inputs, ablations, native) · `08A/08B` SAHI post-processing ablation ·
`08C` DOTA external test · `09` error breakdown, McNemar + bootstrap, WBF ensembles · `10` dataset statistics + annotator agreement.
`04` now also writes T6-T10. The smoke test covers every notebook (fakes for Ultralytics/SAHI).
