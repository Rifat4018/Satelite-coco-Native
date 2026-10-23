"""Generates every notebook in notebooks/ from the cell texts below (keeps them consistent).
Usage: python tools/build_notebooks.py"""
import os, json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plans import NAMES, PLANS
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "notebooks")

SHOW = '''import os

def show_inputs(depth=4, max_files_per_dir=6):
    """Print what is attached under /kaggle/input - check it before trusting the rest."""
    root0 = os.environ.get("SAT_INPUT", "/kaggle/input")
    for root, dirs, files in os.walk(root0):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        level = root.replace(root0, "").count(os.sep)
        if level > depth:
            dirs[:] = []
            continue
        print("  " * level + root + "/")
        for fn in sorted(files)[:max_files_per_dir]:
            print("  " * level + "  " + fn)
        if len(files) > max_files_per_dir:
            print("  " * level + f"  ... +{len(files) - max_files_per_dir} more files")

show_inputs()'''

SETUP = '''import os, sys, warnings
warnings.filterwarnings("ignore")
INPUT_ROOT = os.environ.get("SAT_INPUT", "/kaggle/input")

def _find_code_root():
    for root, dirs, files in os.walk(INPUT_ROOT):
        if {"config.py", "audit.py", "metrics.py", "infer.py"} <= set(files):
            return root
    return None

CODE_ROOT = os.environ.get("SAT_CODE") or _find_code_root()
assert CODE_ROOT, "sat_ext code folder not found - Add Input -> your 'sat-ext' code Dataset"
sys.path.insert(0, CODE_ROOT)
print("CODE_ROOT:", CODE_ROOT)

import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 70)
import config as cfg, utils
utils.ensure_dirs(); utils.set_seed(cfg.SEED)
print("work dir:", cfg.WORK_DIR, "| smoke test:", cfg.SMOKE)'''


def md(t):
    return dict(cell_type="markdown", metadata={}, source=t.strip("\n").splitlines(True))


def code(t):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=t.strip("\n").splitlines(True))


def nb(cells, gpu=False, internet=False):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isGpuEnabled": gpu,
                       "isInternetEnabled": internet, "language": "python", "sourceType": "notebook"}}
    return dict(cells=cells, metadata=meta, nbformat=4, nbformat_minor=5)


# =============================================================================================== 00
NB00 = [md('''
# SAT-00 · Get the dataset from Roboflow (OPTIONAL)  (CPU, Internet ON)

**You only need this if** you want to (a) prove that your Kaggle copy `rifat4018/satelite-coco` is the same as the
Roboflow version, or (b) build the Kaggle copy again from Roboflow.

**Before running:**
1. Roboflow -> Settings -> API Keys -> copy the *Private API Key*.
2. Kaggle notebook -> Add-ons -> Secrets -> add a secret named `ROBOFLOW_API_KEY`, tick it for this notebook.
3. Open your project page on Roboflow. The URL looks like `app.roboflow.com/<WORKSPACE>/aerial-object-detection-u3f5u/<VERSION>`.
   Put those two values in the settings cell below.

**Attach:** the `sat-ext` code Dataset (+ optionally `satelite-coco` to compare). **Accelerator:** None. **Internet:** ON.
**Produces:** `/kaggle/working/rf_<project>_v<version>/` (train/ valid/ test/ COCO) -> Output tab -> *New Dataset* if you want to use it.
'''), code(SHOW), code(SETUP), code('''
RF_WORKSPACE = cfg.RF_WORKSPACE      # CHECK: the part after app.roboflow.com/ in your project URL
RF_PROJECT   = cfg.RF_PROJECT        # "aerial-object-detection-u3f5u"
RF_VERSION   = cfg.RF_VERSION        # CHECK: the version number you generated (Versions tab)
print(RF_WORKSPACE, RF_PROJECT, RF_VERSION)
'''), code('''
utils.warn_if_gpu_wasted()
import roboflow_io as rio, dataset as ds
dest = rio.download(RF_WORKSPACE, RF_PROJECT, RF_VERSION)
'''), code('''
# compare with the Kaggle copy if it is attached (the copy your paper used)
others = [r for r in ds.find_coco_roots() if not r.startswith(cfg.WORK_DIR)]
if others:
    table, identical = rio.compare_exports(others[0], dest)
    table.to_csv(os.path.join(cfg.WORK_DIR, "roboflow_vs_kaggle.csv"), index=False)
else:
    print("no Kaggle copy attached -> nothing to compare. The download is in", dest)
''')]

# =============================================================================================== 01
def nb01(kind):
    return [md('''
# SAT-01 · Dataset audit + leakage-free split  (CPU)

**Attach:** your dataset `rifat4018/satelite-coco` (the Roboflow COCO export used in the paper) + the `sat-ext` code Dataset.
**Accelerator:** None (CPU). **Internet:** not needed. **Planning estimate:** 10-25 min (most of it is the ORB overlap check).

What it does (reviewer points R1-1 and R3-4):
1. Counts images / boxes / classes per split and compares them with the numbers printed in the paper.
2. Cleans the boxes (clip to the image corners, drop boxes < 1 px and exact duplicates) - **every change is logged in `clean_log.csv`; the raw export is never touched.**
3. Finds images that show the same ground area: Roboflow augmented copies (same source name), byte-identical files (SHA-256),
   near-duplicates under flips/rotations (pHash) and partially overlapping tiles (ORB + RANSAC).
4. Measures how much the ORIGINAL random split leaks, then builds two splits:
   * `orig` - your original train/valid/test with cleaned boxes (reproduces the paper's data);
   * `grouped` - whole ground-area groups assigned to train/val/test (no ground area in two splits).

**Produces:** `audit/` (tables, montages, figures) and `splits/orig`, `splits/grouped`.
**After it finishes:** Save Version -> Output tab -> **New Dataset** named `sat-splits` (all 6 accounts attach THIS one version).
'''), code(SHOW), code(SETUP), code('''
DATASET = "@@KIND@@"          # preset: which attached export this notebook audits
PREFIX = cfg.DATASETS[DATASET]["prefix"]      # split names: PREFIX + "orig" / PREFIX + "grouped"
USE_ORB = True        # partial-overlap check between tiles; set False only if you are short of time
utils.warn_if_gpu_wasted()
import dataset as ds, audit, split
AUDIT = os.path.join(cfg.AUDIT_OUT, DATASET)
os.makedirs(AUDIT, exist_ok=True)
RAW_ROOT = ds.find_coco_root(DATASET)
print("raw export:", RAW_ROOT)
raw = ds.load_raw(RAW_ROOT)
cmap = ds.category_map(raw)
for s, c in raw.items():
    print(f"{s:<5} categories {[(x['id'], x['name']) for x in c['categories']]}  ->  used as YOLO classes {cmap[s]}")
imgs = ds.image_table(raw)
print("images whose file is missing:", int((~imgs.exists).sum()))
'''), md('''
## 1. Counts per split (raw export, before any cleaning)
`copies_per_source` = how many images share one Roboflow source name, e.g. `{3: 435}` means 435 sources with 3 copies each.
'''), code('''
counts, cats = audit.split_counts(raw, imgs, cmap)
print(counts.set_index("split").T.to_string())
counts.to_csv(f"{AUDIT}/01_split_counts.csv", index=False)
'''), code('''
# reconcile with the numbers printed in the rejected manuscript (paper dataset only)
PAPER = {"train images": 1305, "val images": 54, "test images": 55,
         "train boxes": 44909, "val boxes": 2032, "test boxes": 1826,
         "train negative x": 358, "train negative y": 366}
c = counts.set_index("split")
MEAS = {"train images": c.loc["train", "images"], "val images": c.loc["val", "images"], "test images": c.loc["test", "images"],
        "train boxes": c.loc["train", "annotations"], "val boxes": c.loc["val", "annotations"], "test boxes": c.loc["test", "annotations"],
        "train negative x": c.loc["train", "neg_x"], "train negative y": c.loc["train", "neg_y"]}
rec = pd.DataFrame({"paper": PAPER, "measured": MEAS})
rec["match"] = rec.paper == rec.measured
if DATASET == "main":
    print(rec.to_string())
n_src = imgs.source_id.nunique()
print(f"\\nunique source images over all splits: {n_src}  (the paper's '1,414 base images' counts augmented copies)")
rec.to_csv(f"{AUDIT}/01_paper_reconciliation.csv")
'''), md('''
## 2. Box cleaning (logged)
'''), code('''
boxes, clean_log = ds.clean_annotations(raw, cmap)
print(clean_log.groupby(["split", "action"]).size().unstack(fill_value=0).to_string())
clean_log.to_csv(f"{AUDIT}/02_clean_log.csv", index=False)
print("boxes kept:", sum(len(v) for v in boxes.values()))
'''), md('''
## 3. Exact and near duplicates (SHA-256, pHash under the 8 flips/rotations)
'''), code('''
with utils.timer("hashes"):
    feat, PH = audit.image_features(imgs)
sha_dup = feat[feat.duplicated("sha256", keep=False)].sort_values("sha256")
print("byte-identical images:", len(sha_dup))
ph = audit.phash_pairs(feat, PH)
ph.to_csv(f"{AUDIT}/03_phash_pairs.csv", index=False)
print(f"pHash pairs (distance <= {cfg.PHASH_MAX_HAMMING}): {len(ph)}; across splits: {int(ph.cross_split.sum()) if len(ph) else 0}; "
      f"across DIFFERENT source names: {int((~ph.same_source).sum()) if len(ph) else 0}")
if len(ph):
    print(ph.sort_values("hamming").head(20)[["key_i", "key_j", "hamming", "cross_split", "same_source"]].to_string(index=False))
    cross = ph[ph.cross_split | ~ph.same_source].sort_values("hamming")
    p = audit.pair_montage(feat, cross, f"{AUDIT}/03_phash_pairs_crosssplit.png",
                           "pHash near-duplicates across splits / across source names - CHECK BY EYE")
    print("montage:", p)
'''), md('''
## 4. Overlapping tiles (ORB features + RANSAC homography)
One representative per source image (the least-rotated copy). Pairs with at least `ORB_MIN_INLIERS` geometric inliers are
treated as the same ground area. **Look at the montage**: if unrelated scenes appear (repetitive parking lots can fool
the matcher), raise `cfg.ORB_MIN_INLIERS` in this cell and rerun this cell and the ones below.
'''), code('''
# cfg.ORB_MIN_INLIERS = 40     # <- raise only if the montage shows unrelated scenes
rep = audit.representatives(feat)
orb = None
if USE_ORB:
    with utils.timer("ORB overlap"):
        orb = audit.orb_pairs(feat.path.values[rep])
    orb["key_i"], orb["key_j"] = feat.key.values[rep[orb.i]], feat.key.values[rep[orb.j]]
    orb["split_i"], orb["split_j"] = feat.src_split.values[rep[orb.i]], feat.src_split.values[rep[orb.j]]
    orb.to_csv(f"{AUDIT}/04_orb_pairs.csv", index=False)
    strong = orb[orb.inliers >= cfg.ORB_MIN_INLIERS]
    print(f"overlapping pairs: {len(strong)} (across splits: {int((strong.split_i != strong.split_j).sum())})")
    print("inlier distribution of the 30 strongest pairs:", orb.inliers.head(30).tolist())
    mont = orb.assign(i=rep[orb.i], j=rep[orb.j])
    audit.pair_montage(feat, mont[mont.inliers >= cfg.ORB_MIN_INLIERS], f"{AUDIT}/04_orb_overlap_top.png",
                       "ORB overlap - strongest pairs", score_col="inliers")
    border = mont[(mont.inliers >= cfg.ORB_MIN_INLIERS * 0.6) & (mont.inliers < cfg.ORB_MIN_INLIERS * 1.6)]
    audit.pair_montage(feat, border, f"{AUDIT}/04_orb_overlap_borderline.png",
                       f"ORB pairs near the threshold ({cfg.ORB_MIN_INLIERS}) - are these the same place?", score_col="inliers")
'''), md('''
## 5. Groups and leakage of the original split
'''), code('''
feat_g, evidence = audit.build_groups(feat, ph, orb, rep, use_orb=USE_ORB)
leak = audit.leakage_summary(feat_g, ph, orb, rep)
leak["evidence_links"] = evidence
utils.save_json(leak, f"{AUDIT}/05_leakage_summary.json")
feat_g.drop(columns=["path"]).to_csv(f"{AUDIT}/05_images_groups.csv", index=False)
for k, v in leak.items():
    print(f"  {k:<45} {v}")
if leak["largest_group_share_of_sources"] > 0.15:
    print("\\nWARNING: one group holds more than 15% of all source images. Check 04_orb_overlap_borderline.png;"
          " if those pairs are NOT the same place, raise cfg.ORB_MIN_INLIERS (cell above) and rerun from there.")
'''), md('''
## 6. Object sizes (areas in px² on the 640×640 images) and ground-truth samples
'''), code('''
sizes = audit.box_sizes(boxes, imgs)
tab = sizes.groupby(["split", "bucket"]).size().unstack(fill_value=0).reindex(columns=[b for b, _, _ in cfg.BUCKETS], fill_value=0)
print(tab.to_string()); print((tab.T / tab.sum(1)).T.round(3).to_string())
tab.to_csv(f"{AUDIT}/06_size_buckets.csv")
print("COCO convention (paper): small = area < 32^2 px^2, medium = 32^2..96^2, large >= 96^2; micro (ours) = area < 16^2")
print(audit.size_figure(sizes, f"{AUDIT}/06_size_distribution.png"))
for s in ("train", "val", "test"):
    print(audit.gt_grid(feat, boxes, f"{AUDIT}/06_gt_samples_{s}.png", s))
'''), md('''
## 7. Write the two splits
'''), code('''
man_o = split.orig_manifest(feat_g)
meta_o = ds.write_split(PREFIX + "orig", man_o, imgs, boxes, log=clean_log, extra=dict(dataset=DATASET, leakage=leak))
man_g, ginfo = split.grouped_manifest(feat_g, boxes)
split.check_no_group_leak(man_g)
meta_g = ds.write_split(PREFIX + "grouped", man_g, imgs, boxes, log=clean_log, extra=dict(dataset=DATASET, leakage=leak, grouping=ginfo))
import shutil; shutil.copytree(AUDIT, os.path.join(cfg.SPLITS_OUT, "audit_" + DATASET), dirs_exist_ok=True)
'''), md('''
## 8. Read this before the GPU notebooks
* `orig` fingerprint and `grouped` fingerprint are printed above. **Every training notebook prints the fingerprint it used;
  they must all be the same** (the aggregation notebook checks it).
* Save Version -> Output -> New Dataset `sat-splits`. Share it with the other accounts (Dataset -> Settings -> Sharing -> add
  the usernames, or make it public if the image licence allows).
'''), code('''
print("orig    :", meta_o["fingerprint"], meta_o["summary"])
print("grouped :", meta_g["fingerprint"], meta_g["summary"])
print("files:", sorted(os.listdir(cfg.SPLITS_OUT)))
''')]


def nb01k(kind):
    cells = nb01(kind)
    for c in cells:
        c['source'] = [l.replace('@@KIND@@', kind) for l in c['source']]
    if kind != 'main':
        c0 = cells[0]['source']
        cells[0] = md(SKY_HEADER)
    return cells


SKY_HEADER = '''
# SAT-01b · Second dataset (SkySeaLand v14): audit + leakage-free split  (CPU)

**Attach:** the SkySeaLand COCO export (Roboflow Universe `aquantic-research/skysealand-tf38k`, version 14, licence CC BY 4.0 -
download it in COCO format and upload it as a Kaggle Dataset, e.g. `skysealand-v14`) + `sat-ext`.
It can stay attached next to `satelite-coco`: the notebooks tell the two exports apart by their class names.

Classes are mapped onto the paper's three: airplane -> plane, boat + ship -> ship, car -> vehicle (state this in the paper).
Same audit as notebook 01 (duplicates, pHash, ORB overlap of the screenshots), splits `sky_orig` and `sky_grouped`.
Images keep their native size (about 1918x850); box areas below are in native pixels.
**After it finishes:** Output -> New Dataset `sat-splits-sky`.
'''


# =============================================================================================== 02X
ATTACH = {
    "G": "`skysealand-v14` (raw export) + `sat-splits-sky` (notebook 01b output) + `sat-ext` (code).",
    "07I": "`satelite-coco` (raw export) + `sat-splits` (notebook 01 output) + `sat-ext` (code). The split is built at the start (no notebook 06 needed).",
    "07J": "`satelite-coco` + `satelite-native` (native-resolution Roboflow export) + `sat-splits` (notebook 01 output) + `sat-ext` (code). The split is built at the start (no notebook 06 needed).",
}
ATTACH["07K"] = ATTACH["07L"] = ATTACH["07J"]


def nb02(acc):
    import train as _t
    hours = sum(_t.expected_h(sp, m) for sp, m, _ in PLANS[acc])
    attach = ATTACH.get(acc, "`satelite-coco` (raw export) + `sat-splits` (notebook 01 output) + `sat-ext` (code).")
    attach += f" **Planning estimate:** ~{hours:.1f} GPU h (from your T4 logs, not measured for this notebook)."
    return [md(f'''
# SAT-{NAMES[acc]}  (GPU)

**Attach:** {attach}
**Accelerator:** GPU T4 (x2 is fine, one GPU is used) · **Internet:** ON (downloads ultralytics + pretrained weights).
**Nothing to edit.** The runs of this notebook are fixed in `plans.py` (printed below). Each Kaggle account can run 2 GPU notebooks at once.

**How to run:** first time on a new account, set `PILOT = True` and run interactively (1-epoch YOLO11n, ~10 min) to see that
everything works on the GPU. Then set `PILOT = False` and use **Save Version -> Save & Run All (Commit)** so it runs in the
background (up to 12 h). Finished runs are skipped if you run the notebook again.

**Per run it produces** `results/<split>__<model>__s<seed>/` (training curves, `train_meta.json`, `eval_test.json`,
detections, per-image TP/FP) and `weights/<run_id>_best.pt`.
**After it finishes:** Output tab -> **New Dataset** `sat-results-{acc.lower()}` (or download `results_{acc}.zip`).
'''), code(SHOW), code(SETUP), code(f'''
KEY = "{acc}"            # preset - do not change
PILOT = False         # True = 1-epoch YOLO11n check of the whole GPU pipeline (tagged __pilot, ignored by the tables)
import plans, train
PLAN = plans.PILOT if PILOT else plans.PLANS[KEY]
EPOCHS = 1 if (PILOT or cfg.SMOKE) else None
TAG = "pilot" if PILOT else ""
if PILOT:
    print("PILOT: grouped / YOLO11n / seed 42, 1 epoch")
else:
    plans.show(KEY)
print("epochs:", EPOCHS or cfg.TRAIN_PARAMS["epochs"])
'''), code('''
utils.require_gpu()
assert utils.install_detection_stack(), "pip install failed - is Internet ON? (Settings -> Internet)"
print(utils.versions())
'''), code('''
done = train.run_plan(PLAN, epochs=EPOCHS, tag=TAG)
'''), code(f'''
import shutil, glob
for rid, status in done:
    ev = os.path.join(cfg.RESULTS_DIR, rid, "eval_test.json")
    if os.path.isfile(ev):
        d = utils.load_json(ev)
        u = d["ultralytics_test"]
        print(f"{{rid:<30}} test mAP50={{u.get('mAP50', float('nan')):.4f}} mAP50-95={{u.get('mAP50_95', float('nan')):.4f}} | " +
              " ".join(f"{{m}}:AP={{v['coco'].get('AP_all', float('nan')):.3f}}" for m, v in d["modes"].items()))
z = shutil.make_archive(os.path.join(cfg.WORK_DIR, f"results_{{KEY}}"), "zip", cfg.RESULTS_DIR)
print("\\nsmall results zip (no weights):", z, f"{{os.path.getsize(z) / 1e6:.1f}} MB")
for f in glob.glob(os.path.join(cfg.WORK_DIR, "runs")) + glob.glob(os.path.join(cfg.WORK_DIR, "data_*")):
    shutil.rmtree(f, ignore_errors=True)       # Ultralytics run folders are duplicated in results/ + weights/
''')]


# =============================================================================================== 03
NB03 = [md('''
# SAT-03 · Re-score the weights from the rejected manuscript  (GPU, optional, ~30 min)

**Attach:** `satelite-coco` + `sat-splits` + `sat-ext` + your two old output Datasets (`resultsyolo11`, `resultsrfdetr`, the ones
with `YOLO11n_best.pt` ... `RT-DETR-L_best.pt`). For the cross-dataset cell also: `skysealand-v14`, `sat-splits-sky`,
the results Datasets of 02B-02F/02A (grouped s42 weights) and 02G. **Accelerator:** GPU · **Internet:** ON.

Your old tables were computed on the VALIDATION split at conf 0.25 and never scored SAHI. This notebook scores the SAME old
weights correctly on the ORIGINAL test split (they were trained on the original train split, so the grouped test split
would be unfair). Run ids: `orig__<model>__slegacy`. It also evaluates any `<run_id>_best.pt` attached without an evaluation.
'''), code(SHOW), code(SETUP), code('''
import infer, dataset as ds, train
utils.require_gpu()
assert utils.install_detection_stack()
LEGACY = {"YOLO11n": "YOLO11n_best.pt", "YOLO11s": "YOLO11s_best.pt", "YOLO11m": "YOLO11m_best.pt",
          "YOLOv8m": "YOLOv8m_best.pt", "RT-DETR-L": "RT-DETR-L_best.pt"}
info = ds.materialize("orig")
for model, fn in LEGACY.items():
    found = utils.find_files(fn)
    rid = f"orig__{model}__slegacy"
    if not found:
        print("not attached:", fn); continue
    if train.find_existing(rid)[0]:
        print("already done:", rid); continue
    infer.evaluate_run(rid, model, found[0], info, meta=dict(note="weights of the rejected manuscript", path=found[0]))
'''), code('''
# any trained weights whose evaluation is missing (e.g. a session ended during evaluation)
import re
for root, dirs, files in os.walk(cfg.INPUT_ROOT):
    for f in files:
        m = re.match(r"^(orig|grouped)__(.+)__s(\\d+)_best\\.pt$", f)
        if not m: continue
        rid = f[:-len("_best.pt")]
        if train.find_existing(rid)[0]: continue
        split_name, model = m.group(1), m.group(2)
        print("evaluating missing:", rid)
        infer.evaluate_run(rid, model, os.path.join(root, f), ds.materialize(split_name))
'''), md('''
## Cross-dataset test (optional, needs the SkySeaLand runs of 02G and both split Datasets attached)
Models trained on one dataset are tested on the OTHER dataset's grouped test split without fine-tuning.
Run ids `xfer-main2sky__<model>__s42` and `xfer-sky2main__<model>__s42` -> table T6 in notebook 04.
'''), code('''
XFER = [("grouped", "sky_grouped", "xfer-main2sky"), ("sky_grouped", "grouped", "xfer-sky2main")]
for src_split, dst_split, tag in XFER:
    try:
        dst = ds.materialize(dst_split)
    except FileNotFoundError as e:
        print("skip", tag, "-", e); continue
    for model in cfg.MODELS:
        w = utils.find_files(f"{src_split}__{model}__s42_best.pt")
        rid = f"{tag}__{model}__s42"
        if not w:
            print("no weights for", src_split, model); continue
        if train.find_existing(rid)[0]:
            print("already done:", rid); continue
        infer.evaluate_run(rid, model, w[0], dst, meta=dict(trained_on=src_split, tested_on=dst_split))
''')]

# =============================================================================================== 04
NB04 = [md('''
# SAT-04 · Paper tables and figures  (CPU)

**Attach:** `sat-ext` + `sat-splits` + `satelite-coco` + every results Dataset (`sat-results-a` ... `sat-results-g`, `sat-results-07a` ...
`sat-results-07l`, `sat-results-08a/08b/08c`, 03's output) + `sat-splits-extra`, `sat-splits-sky`, `skysealand-v14`
(+ `satelite-native` if you ran 07J-07L). Missing optional parts are skipped.
**Accelerator:** None. **Planning estimate:** 5-15 min.

**Produces** `results/paper/`: T1 main table (mean ± std over 3 seeds), T2 standard vs SAHI with TP/FP, T3 bootstrap CIs of the
SAHI recall gain, T4 per-class AP, T5 original vs grouped split; Pareto figures, size-stratified recall, extra-detection
decomposition, qualitative TP/FP/FN figure. Each table is saved as CSV and LaTeX.
'''), code(SHOW), code(SETUP), code('''
utils.warn_if_gpu_wasted()
import report, plans, dataset as ds
runs = report.collect_runs()
df = report.long_table(runs)
os.makedirs(report.PAPER_DIR, exist_ok=True)
df.drop(columns=["folder"]).to_csv(os.path.join(report.PAPER_DIR, "all_runs_long.csv"), index=False)
expected = set(plans.expected_run_ids())
optional = set(plans.expected_run_ids(include_optional=True)) - expected
print("phase-2 / optional runs found:", len(optional & set(runs)), "of", len(optional),
      "| not found:", sorted(optional - set(runs))[:40])
missing = sorted(expected - set(runs))
print("expected runs:", len(expected), "| found:", len(expected & set(runs)), "| MISSING:", missing or "none")
print("extra runs (legacy etc.):", sorted(set(runs) - expected))
for rid, (d, _) in runs.items():
    if "engine_check" in d:
        print("pycocotools vs internal AP (should be equal):", rid, d["engine_check"])
'''), code('''
t1 = report.table_main(df, "grouped")
t2 = report.table_sahi(df, "grouped")
t3 = report.table_bootstrap(df, "grouped")
t4 = report.table_per_class(df, "grouped")
t5 = report.table_orig_vs_grouped(df)
if (df.split == "orig").any():
    report.table_sahi(df, "orig")
if (df.split == "sky_grouped").any():           # second dataset (one seed)
    report.table_main(df, "sky_grouped"); report.table_sahi(df, "sky_grouped"); report.table_per_class(df, "sky_grouped")
    report.fig_pareto(df, "sky_grouped"); report.fig_size_recall(df, "sky_grouped")
t6 = report.table_transfer(df)
report.table_splits_compare(df, ["grouped_1copy", "grouped"], ["YOLO11s", "YOLO11m"], "T7a_offline_augmentation",
    "Offline augmentation ablation: train on 1 vs all 3 Roboflow augmented copies per source (same test images).")
report.table_splits_compare(df, [f"grouped_lc{int(f * 100)}" for f in cfg.LEARNING_CURVE] + ["grouped"], ["YOLO11s", "YOLO11m"],
    "T7b_learning_curve", "Learning curve: share of the training ground-area groups used (same test images).")
print(report.fig_learning_curve(df))
report.table_native(df)
abl = report.collect_ablation()
if len(abl):
    report.table_sahi_ablation(abl, "grouped"); report.table_sahi_ablation(abl, "sky_grouped")
report.table_external(df)
report.table_efficiency(df, "grouped")
'''), code('''
for f in [report.fig_pareto(df, "grouped"), report.fig_size_recall(df, "grouped"), report.fig_extra_dets(df, "grouped")]:
    print(f)
info = ds.materialize("grouped")
print(report.fig_qualitative(df, info, split="grouped"))
'''), code('''
try:
    from IPython.display import Image as _I, display
    for f in sorted(os.listdir(report.PAPER_DIR)):
        if f.endswith(".png"):
            print(f); display(_I(os.path.join(report.PAPER_DIR, f), width=900))
except ImportError:
    print(sorted(os.listdir(report.PAPER_DIR)))
''')]

# =============================================================================================== 05
NB05 = [md('''
# SAT-05 · Roboflow upload package + project description  (CPU)

**Attach:** `sat-ext` + `sat-splits` + `satelite-coco` + `satelite-native`. **Accelerator:** None.

Builds two release packages (COCO folders `train/ valid/ test/` + `README.dataset.md` built ONLY from measured numbers):
* `roboflow_upload_native_grouped` - **main release**: the 536 original screenshots at native size, no augmentation,
  same leakage-free split assignment as the paper.
* `roboflow_upload_grouped` - the exact 640 x 640 images used in the experiments (for reproducing the paper).
Upload steps: `plan/03-ROBOFLOW-GUIDE.md`.
'''), code(SHOW), code(SETUP), code('''
utils.warn_if_gpu_wasted()
import roboflow_io as rio
LICENSE = ("Annotations: CC BY 4.0. Imagery: Google Earth, (c) Google and its data providers - research use only, "
           "attribution required; not covered by the CC licence.")
SOURCE = "Screenshots of Google Earth (satellite / aerial imagery), collected and annotated by the authors."
CITATION = "<FILL IN after acceptance>"
for SPLIT in ["native_grouped", "grouped"]:
    try:
        card = rio.datacard(SPLIT, license_text=LICENSE, citation=CITATION, source_text=SOURCE)
    except (FileNotFoundError, AssertionError) as e:
        print("skip", SPLIT, "-", e); continue
    print(card)
    out, z, counts = rio.build_upload_package(SPLIT, readme_text=card)
''')]


NB06 = [md('''
# SAT-06 · Extra splits (CPU) - OPTIONAL since v2.6

**You do not need this notebook any more:** 07I-07L build their split themselves at the start (same result, ~1-4 min).
Run it only if you want the split files for the paper / supplement.

**Attach:** `satelite-coco` + `sat-splits` + `sat-ext` (+ `satelite-native`). **Accelerator:** None. **Internet:** off.
Run the cells interactively (no "Save Version" needed): the last cell shows a **download link** for `sat_splits_extra.zip`.
'''), code(SHOW), code(SETUP), code('''
import extra_splits as xs, shutil
made = xs.build_all(cfg.SPLITS_OUT)
print("splits written:", made)
z = shutil.make_archive(os.path.join(cfg.WORK_DIR, "sat_splits_extra"), "zip", cfg.WORK_DIR, "splits")
for f in os.listdir(cfg.WORK_DIR):          # keep only the zip
    p = os.path.join(cfg.WORK_DIR, f)
    if os.path.abspath(p) != os.path.abspath(z):
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
print(os.listdir(cfg.WORK_DIR))
try:
    from IPython.display import FileLink, display
    display(FileLink("sat_splits_extra.zip"))   # click it to download (works in an interactive session)
except ImportError:
    print("download:", z)
''')]


def nb08(key):
    abl = key in ("08A", "08B")
    attach = {"08A": "`satelite-coco` + `sat-splits` + `sat-ext` + the results Datasets with the grouped weights: "
                     "`sat-results-a`, `sat-results-b`, `sat-results-c`, `sat-results-d`, `sat-results-e`",
              "08B": "`skysealand-v14` + `sat-splits-sky` + `sat-ext` + `sat-results-g`",
              "08C": "a **DOTA-v1.0** dataset (validation images + labelTxt) + `sat-ext` + `sat-results-a`, `sat-results-c`, "
                     "`sat-results-d`, `sat-results-e` (grouped seed-42 weights)"}[key]
    what = ("SAHI post-processing ablation, inference only (no training): merge rule (NMS / GREEDYNMM / NMM), match metric "
            "(IoU / IoS), overlap 0.1 / 0.2 / 0.3, with and without the full-image pass - one factor at a time from the paper "
            "setting. Produces `results/<run>/sahi_ablation.csv`." if abl else
            "Zero-shot external test on DOTA-v1.0: plane, ship, small-vehicle + large-vehicle -> vehicle, horizontal boxes from "
            "the polygons, 'difficult' objects ignored, fixed random sample of images. Standard (whole image at 640) and "
            "SAHI with 640 / 1024 slices at native DOTA resolution. Produces `results/xdota__<model>__s42/`.")
    body = (f'''
import plans, train
PLAN, SLICES = plans.ABLATION["{key}"], plans.ABLATION_SLICES["{key}"]
print(len(PLAN), "runs:", PLAN, "| slices:", SLICES or "all of the split")
utils.require_gpu()
assert utils.install_detection_stack(), "pip install failed - is Internet ON?"
done = train.run_ablation(PLAN, SLICES)
''' if abl else '''
import plans, train, infer, dota
utils.require_gpu()
assert utils.install_detection_stack(), "pip install failed - is Internet ON?"
info = dota.build()
for split, m, seed in plans.EXTERNAL:
    rid = f"xdota__{m}__s{seed}"
    if train.find_existing(rid)[0]:
        print("already done:", rid); continue
    _, wt = train.find_existing(f"{split}__{m}__s{seed}")
    if wt is None:
        print("weights not attached:", split, m, seed); continue
    infer.evaluate_run(rid, m, wt, info, modes=dota.modes(), meta=dict(trained_on=split, tested_on="DOTA-v1.0 val sample"))
''')
    return [md(f'''
# SAT-{NAMES[key]}  (GPU)

**Attach:** {attach}. **Accelerator:** GPU T4 · **Internet:** ON. **Planning estimate:** 1-3 h.

{what}

**After it finishes:** Output -> New Dataset `sat-results-{key.lower()}`.
'''), code(SHOW), code(SETUP), code(body), code('''
import shutil
z = shutil.make_archive(os.path.join(cfg.WORK_DIR, "results_" + "KEYX"), "zip", cfg.RESULTS_DIR)
print("results zip:", z)
shutil.rmtree(os.path.join(cfg.WORK_DIR, "runs"), ignore_errors=True)
'''.replace("KEYX", key))]


NB09 = [md('''
# SAT-09 · Error analysis, McNemar, ensembles  (CPU, Internet ON for one pip package)

**Attach:** `sat-ext` + `sat-splits` (+ `sat-splits-extra`, `sat-splits-sky`) + ALL results Datasets (`sat-results-a` ... `-g`,
`sat-results-07a` ... `-07l`, `sat-results-08a/08b/08c`, legacy). **Accelerator:** None. **Internet:** ON (installs `ensemble-boxes`).
**Planning estimate:** 10-30 min.

* **T11** error breakdown (TIDE definitions: duplicate / classification / localization / cls+loc / background, missed by size)
* **T12** pairwise model comparison: exact McNemar on objects + image-level paired bootstrap (primary), Holm-corrected
* **T12b** McNemar: each SAHI setting vs standard inference
* **T13** Weighted Box Fusion ensembles (seeds, architectures, standard + SAHI)
'''), code(SHOW), code(SETUP), code('''
utils.warn_if_gpu_wasted()
utils.pip_install("ensemble-boxes")
import report, analysis
runs = report.collect_runs()
df = report.long_table(runs)
os.makedirs(report.PAPER_DIR, exist_ok=True)
'''), code('''
err = analysis.table_errors(df, "grouped")
print(analysis.fig_errors(err, "grouped"))
if (df.split == "sky_grouped").any():
    analysis.fig_errors(analysis.table_errors(df, "sky_grouped"), "sky_grouped")
'''), code('''
analysis.table_mcnemar(df, "grouped", "standard")
analysis.table_mcnemar_modes(df, "grouped")
if "sky_grouped" in set(df.split):
    analysis.table_mcnemar(df, "sky_grouped", "standard"); analysis.table_mcnemar_modes(df, "sky_grouped")
# native resolution (07J-07L): native_tiles has the SAME val/test images and ids as native_grouped -> one paired table
nat = df[df.split.isin(["native_grouped", "native_tiles"])].copy()
if len(nat):
    nat["model"] = nat.model.astype(str) + np.where(nat.split == "native_tiles", " (tiles)", "")
    nat["split"] = "native_grouped"
    analysis.fig_errors(analysis.table_errors(nat, "native_grouped", models=sorted(set(nat.model))), "native_grouped")
    analysis.table_mcnemar(nat, "native_grouped", "standard", models=sorted(set(nat.model)))
    analysis.table_mcnemar_modes(nat, "native_grouped", models=sorted(set(nat.model)))
    analysis.table_ensembles(df, "native_grouped")     # WBF of seeds / architectures at native resolution
else:
    print("no native runs attached (07J-07L) -> native McNemar skipped")
'''), code('''
analysis.table_ensembles(df, "grouped")
import shutil
print(shutil.make_archive(os.path.join(cfg.WORK_DIR, "paper_analysis"), "zip", report.PAPER_DIR))
''')]


NB10 = [md('''
# SAT-10 · Dataset statistics (CPU)

**Attach:** `sat-ext` + `sat-splits` + `satelite-coco`. **Accelerator:** None. **Internet:** off. ~2 min.

Statistics of the 544 source images: objects per image, box sizes per class, object centres, class co-occurrence
(tables D1-D4 + figure in `paper/`), plus a comparison-table template for the related datasets (fill their numbers
from their papers). Nothing else to do.
'''), code(SHOW), code(SETUP), code('''
utils.warn_if_gpu_wasted()
import analysis, report
os.makedirs(report.PAPER_DIR, exist_ok=True)
summary, per_cls, fig = analysis.dataset_stats("orig")
print(per_cls.to_string(index=False)); print(fig)
''')]


NB08D = [md('''
# SAT-08D · TTA and inference-only resolution vs SAHI  (GPU, ~1-2 h)

**Attach:** `satelite-coco` + `sat-splits` + `sat-ext` + `sat-results-a`, `sat-results-b`, `sat-results-c`, `sat-results-d`,
`sat-results-e` (grouped weights, 3 seeds). **Accelerator:** GPU T4 · **Internet:** ON.

Same weights, no training. For every grouped run of the 5 paper models: **TTA** (Ultralytics flips + 3 scales; not available
for RT-DETR) and **inference at 1024** (640-trained weights, larger input only). Scored exactly like the main table
(COCO AP by size, TP/FP at conf 0.25, latency). Answers the reviewer question "SAHI vs TTA vs higher resolution".

**After it finishes:** Output -> New Dataset `sat-results-08d`. Notebooks 04/09 merge it automatically.
'''), code(SHOW), code(SETUP), code('''
import plans, train, infer, dataset as ds
utils.require_gpu()
assert utils.install_detection_stack(), "pip install failed - is Internet ON?"
info = ds.materialize("grouped")
done_ids = {utils.load_json(p)["run_id"] for p in utils.find_files("eval_extra.json", also_work=True)}
for split, m, seed in plans.EXTRA:
    rid = f"{split}__{m}__s{seed}"
    if rid in done_ids:
        print("already done:", rid); continue
    _, wt = train.find_existing(rid)
    if wt is None:
        print("weights not attached:", rid); continue
    infer.extra_modes(rid, m, wt, info)
'''), code('''
import shutil
z = shutil.make_archive(os.path.join(cfg.WORK_DIR, "results_08D"), "zip", cfg.RESULTS_DIR)
print("results zip:", z)
for f in os.listdir(cfg.WORK_DIR):
    if f.startswith("data_") or f == "runs":
        shutil.rmtree(os.path.join(cfg.WORK_DIR, f), ignore_errors=True)
''')]


NB11 = [md('''
# SAT-11 · Label-quality check of the top false positives  (CPU, ~10 min of your time)

**Attach:** `sat-ext` + `sat-splits` + `satelite-coco` + `sat-results-e` (YOLO11s; if `sat-results-07a` is attached its
YOLO11s-1024 is used instead). **Accelerator:** None. **Internet:** off. Run the cells **interactively** (no Save Version needed).

1. Takes the 50 highest-confidence **background** false positives (no ground-truth box of any class with IoU >= 0.1) on the
   grouped test split and shows them as numbered crops (red box = detection).
2. You look at the pages and type the numbers of the crops that are **real objects that were not labelled** into `MISSING`
   (and doubtful ones into `UNSURE`). Then run the last cell -> table **T16** for the paper.
'''), code(SHOW), code(SETUP), code('''
import numpy as np, cv2, analysis, report, metrics, dataset as ds
os.makedirs(report.PAPER_DIR, exist_ok=True)
runs = {}
for p in utils.find_files("eval_test.json"):     # only folders that also hold the saved detections
    if utils.dets_path(os.path.dirname(p), "standard"):
        runs[utils.load_json(p)["run_id"]] = os.path.dirname(p)
RID, dets = None, None
for r in ["grouped__YOLO11s-1024__s42", "grouped__YOLO11s__s42", "grouped__YOLO11m__s42"]:
    if r in runs and analysis._dets(runs[r], "standard") is not None:
        RID, dets = r, analysis._dets(runs[r], "standard"); break
    if r in runs:
        print("no saved detections for", r, "-> trying the next model")
assert RID, "attach sat-results-e (grouped YOLO11s seed 42, with its results/ folder)"
gt = analysis._gt_for("grouped")
gb = {}
for a in gt["annotations"]:
    gb.setdefault(a["image_id"], []).append(a["bbox"])
fps = []
for d in dets:
    if d["score"] < cfg.OP_CONF:
        continue
    g = np.array(gb.get(d["image_id"], []), float).reshape(-1, 4)
    iou = metrics._iou_xywh(np.array([d["bbox"]], float), g)[0] if len(g) else np.zeros(0)
    if not len(iou) or iou.max() < 0.1:
        fps.append(d)
fps = sorted(fps, key=lambda d: -d["score"])[:50]
names = {i["id"]: i["file_name"] for i in gt["images"]}
info = ds.materialize("grouped")
CELL, COLS, PER_PAGE = 320, 4, 12        # bigger crops, 12 per page -> easier to judge
tiles = []
for k, d in enumerate(fps, 1):
    im = cv2.imread(os.path.join(info["images"]["test"], names[d["image_id"]]))
    x, y, w, h = d["bbox"]
    r = max(48, 1.5 * max(w, h))
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1, y1 = int(min(im.shape[1], cx + r)), int(min(im.shape[0], cy + r))
    crop = im[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (int(x - x0), int(y - y0)), (int(x + w - x0), int(y + h - y0)), (0, 0, 255), 1)
    crop = cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (0, 0), (CELL, 26), (0, 0, 0), -1)
    cv2.putText(crop, f"#{k}  {cfg.CLASSES[d['category_id'] - 1]}  {d['score']:.2f}", (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    tiles.append(crop)
pages = []
for p in range(0, len(tiles), PER_PAGE):
    t = tiles[p:p + PER_PAGE] + [np.zeros((CELL, CELL, 3), np.uint8)] * (PER_PAGE - len(tiles[p:p + PER_PAGE]))
    page = np.vstack([np.hstack(t[r * COLS:(r + 1) * COLS]) for r in range(PER_PAGE // COLS)])
    fn = os.path.join(report.PAPER_DIR, f"fp_review_page{p // PER_PAGE + 1}.png")
    cv2.imwrite(fn, page); pages.append(fn)
import pandas as pd
pd.DataFrame([dict(id=k, file_name=names[d["image_id"]], cls=cfg.CLASSES[d["category_id"] - 1], score=round(d["score"], 3),
                   bbox=[round(v, 1) for v in d["bbox"]]) for k, d in enumerate(fps, 1)]).to_csv(
    os.path.join(report.PAPER_DIR, "fp_review_list.csv"), index=False)
print(RID, "|", len(fps), "background false positives at conf >=", cfg.OP_CONF)
import zipfile, shutil
zp = os.path.join(cfg.WORK_DIR, "fp_review.zip")      # ONE small file: the 5 pages + the list
with zipfile.ZipFile(zp, "w") as z:
    for fn in pages + [os.path.join(report.PAPER_DIR, "fp_review_list.csv")]:
        z.write(fn, os.path.basename(fn))
shutil.rmtree(info["root"], ignore_errors=True)       # copied images are not needed in the output
print("download:", zp, round(os.path.getsize(zp) / 1e6, 1), "MB")
try:
    from IPython.display import Image, display, FileLink
    display(FileLink("fp_review.zip"))                # click to download (interactive session)
    for fn in pages:
        display(Image(fn))
except ImportError:
    print(pages)
'''), code('''
# >>> Look at EVERY page above. Type the #numbers YOU judged, then run this cell. <<<
MISSING = []     # crops showing a REAL plane / ship / vehicle that has no label (annotation miss)
UNSURE = []      # crops you cannot decide
n = len(fps)
assert set(MISSING) | set(UNSURE) <= set(range(1, n + 1)), "numbers must be between 1 and %d" % n
assert not set(MISSING) & set(UNSURE), "a crop cannot be both MISSING and UNSURE"
print("reviewed", n, "| missing:", sorted(MISSING), "| unsure:", sorted(UNSURE))
lst = pd.read_csv(os.path.join(report.PAPER_DIR, "fp_review_list.csv"))
lst["verdict"] = ["missing_label" if i in MISSING else "unsure" if i in UNSURE else "true_false_positive" for i in lst.id]
lst.to_csv(os.path.join(report.PAPER_DIR, "fp_review_list.csv"), index=False)
print(lst.groupby(["cls", "verdict"]).size().unstack(fill_value=0))
t = pd.DataFrame([dict(Model=RID, Reviewed=n, Missing_label=len(MISSING), Unsure=len(UNSURE),
                       True_false_positive=n - len(MISSING) - len(UNSURE),
                       Missing_share=f"{100 * len(MISSING) / max(n, 1):.0f}%")])
report.save(t, "T16_label_quality", f"Manual review of the {n} highest-confidence background false positives of {RID} on the "
            "grouped test split: share that are real but unlabelled objects (annotation misses).")
''')]


# ---------------------------------------------------------------- input setup cell (v2.8)
OWN = "https://www.kaggle.com/datasets/rifat4018/"
LINKS = {
    "sat-ext": ("code (upload the sat_ext folder of the zip)", OWN + "sat-ext"),
    "satelite-coco": ("Roboflow COCO export, 640x640 (or download: universe.roboflow.com/rifat4018/aerial-object-detection-u3f5u)", OWN + "satelite-coco"),
    "satelite-native": ("Roboflow COCO export, native size, no augmentation", OWN + "satelite-native"),
    "sat-splits": ("output of notebook 01", OWN + "sat-splits"),
    "sat-splits-sky": ("output of notebook 01b", OWN + "sat-splits-sky"),
    "sat-splits-extra": ("output of notebook 06 (optional)", OWN + "sat-splits-extra"),
    "skysealand-v14": ("SkySeaLand (public, Mendeley Data)", "https://data.mendeley.com/datasets/d42n3cp86p"),
    "DOTA": ("DOTA-v1.0 with labelTxt (public)", "https://www.kaggle.com/datasets/shadow4ever/dota-dataset-complete-new"),
}


def inputs_cell(text):
    import re
    text = text.split("After it finishes")[0]
    names = [n for n in LINKS if n in text]
    res = sorted(set(re.findall(r"sat-results-[0-9a-z]+", text)))
    if "every results Dataset" in text or "ALL results Datasets" in text:
        res = ([f"sat-results-{a}" for a in "abcdefg"] + [f"sat-results-07{c}" for c in "abcdefghijkl"]
               + ["sat-results-08a", "sat-results-08a1", "sat-results-08b", "sat-results-08b1", "sat-results-08c",
                  "sat-results-08d", "the output of notebook 03 (legacy weights + cross-dataset)"])
    if not names and not res:
        return None
    lines = ["**Input setup** - right panel -> *Add Input* -> search the name -> *Add*. Links (your own Datasets are "
             "private: open them from the account that owns them, or share them with this account first):", ""]
    lines += [f"* `{n}` - {LINKS[n][0]}: {LINKS[n][1]}" for n in names]
    if res:
        lines.append(f"* {', '.join('`%s`' % r for r in res)} - outputs of earlier notebooks "
                     "(Output tab of that notebook -> *New Dataset*, same name); search them under *Your Datasets*.")
    lines.append("* Check the printed input tree in the next cell before running the rest.")
    return md("\n".join(lines))


NB12 = [md('''
# SAT-12 · Paper figures (40+) from the saved results  (CPU, ~5-10 min)

**Attach:** the same inputs as notebook 04 (`sat-ext`, `sat-splits`, `sat-splits-extra`, `sat-splits-sky`, `satelite-coco`,
every `sat-results-*`, the 03 output). **Accelerator:** None. **Internet:** off. No training, no GPU.

Dataset statistics, architecture benchmark, SAHI vs standard, SAHI ablation, resolution / TTA / tiles, leakage (original vs
grouped split), offline augmentation, learning curve, native resolution, cross-dataset, DOTA, error types, confidence, and
qualitative examples. Each figure is skipped (and listed) if its data is not attached.
**Output:** `paper_figures.zip` (300 dpi PNG) - download it from the Output tab.
'''), code(SHOW), code(SETUP), code('''
utils.warn_if_gpu_wasted()
import figures, shutil
done, skipped = figures.make_all()
z = shutil.make_archive(os.path.join(cfg.WORK_DIR, "paper_figures"), "zip", figures.FIG_DIR)
for f in os.listdir(cfg.WORK_DIR):
    if f.startswith("data_"):
        shutil.rmtree(os.path.join(cfg.WORK_DIR, f), ignore_errors=True)
print(z, round(os.path.getsize(z) / 1e6, 1), "MB")
''')]


NB13A = [md('''
# SAT-13A · Export the 82 test images for relabelling  (CPU, ~2 min)

**Attach:** `sat-ext` + `sat-splits` + `satelite-coco`. **Accelerator:** None.

Run all, then click the link and download **`test_relabel_upload.zip`** (82 test images + their current boxes, COCO).

**Then in Roboflow (your manual step):** *New Project* (Object Detection, classes plane / ship / vehicle) -> *Upload* the
unzipped folder (images **and** `_annotations.coco.json`, so the existing boxes come with them) -> open each of the 82
images and **add every missing plane / ship / vehicle** (most are small cars in car parks and on roads; Label Assist may
propose boxes - check each) -> *Generate Version* with **no preprocessing, no augmentation**, all images in one split ->
*Export* COCO -> upload the unzipped export to Kaggle as Dataset **`sat-test-relabeled`** (name must contain "relabel").
Then run **13B**.
'''), code(SHOW), code(SETUP), code('''
utils.warn_if_gpu_wasted()
import relabel
z = relabel.export_for_relabel()
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(z, cfg.WORK_DIR)))
except ImportError:
    pass
''')]

NB13B = [md('''
# SAT-13B · Re-score every saved prediction on the corrected test labels  (CPU, ~20-40 min, no retraining)

**Attach:** `sat-ext` + `sat-splits` + `satelite-coco` + **`sat-test-relabeled`** + every `sat-results-*` Dataset
(+ `sat-splits-extra`). **Accelerator:** None. Run all / Save Version.

Check the log says `matched test images: 82 of 82`. Output -> New Dataset **`sat-results-13`** (tables T17, T18).
Then re-run **04, 09, 12** with `sat-results-13` attached too (logs print "runs use the CORRECTED test labels").
'''), code(SHOW), code(SETUP), code('''
utils.warn_if_gpu_wasted()
import relabel, report, dataset as ds
sdir = ds.find_split_dir("grouped")
gt_old = utils.load_json(os.path.join(sdir, "coco", "test.json"))
info = ds.materialize("grouped")
new_boxes = relabel.load_relabeled(gt_old, info["images"]["test"])
gt_new = relabel.corrected_gt(gt_old, new_boxes)
chg = relabel.change_report(gt_old, gt_new)
os.makedirs(report.PAPER_DIR, exist_ok=True)
report.save(chg, "T17_test_relabel_changes", "Test-set annotation correction (notebook 13): boxes per class and size before and "
            "after adding the missing objects (same 82 images).")
relabel.write_corrected_splits(gt_new)
t = relabel.rescore_all(gt_new)
t.to_csv(os.path.join(report.PAPER_DIR, "T18_rescore_old_vs_new.csv"), index=False)
s = t[t["mode"].isin(["standard", "sahi480", "sahi320", "sahi160", "tta", "res1024"])]
print(s.groupby(["model", "mode"])[["AP_old", "AP_new", "P_old", "P_new", "R_old", "R_new", "FP_old", "FP_new"]].mean().round(3).to_string())
import shutil
shutil.rmtree(info["root"], ignore_errors=True)
'''), code('''
import shutil
z = shutil.make_archive(os.path.join(cfg.WORK_DIR, "results_13"), "zip", cfg.WORK_DIR, "results")
print(sorted(os.listdir(cfg.WORK_DIR)), z)
''')]


NOTEBOOKS = {
    "00-get-data-roboflow.ipynb": (NB00, False, True),
    "01-audit-and-splits.ipynb": (nb01k("main"), False, False),
    "01b-audit-and-splits-skysealand.ipynb": (nb01k("sky"), False, False),
    **{f"{NAMES[a]}.ipynb": (nb02(a), True, True) for a in "ABCDEFG"},
    "03-rescore-legacy-weights.ipynb": (NB03, True, True),
    "04-paper-tables-figures.ipynb": (NB04, False, False),
    "05-roboflow-export.ipynb": (NB05, False, False),
    "06-make-extra-splits.ipynb": (NB06, False, False),
    "08D-tta-and-inference-resolution.ipynb": (NB08D, True, True),
    "12-all-paper-results-04-plus-09.ipynb": ([md('''
# SAT-12 · ALL paper results in one run = notebook 04 + notebook 09  (CPU, ~30-45 min)

Use this instead of running 04 and 09 separately. **Attach:** everything listed for 04 (sat-ext, sat-splits, sat-splits-extra,
sat-splits-sky, satelite-coco, satelite-native, skysealand-v14, every `sat-results-*` incl. 03). **Accelerator:** None.
**Internet:** ON (09 part installs `ensemble-boxes`). Output: `results/paper/` with every table (T1-T16, D1-D4) and figure.
''')] + NB04[1:] + [c for c in NB09 if c is not NB09[0] and c["source"] not in (NB04[1]["source"], NB04[2]["source"])], False, True),
    "11-label-quality-check.ipynb": (NB11, False, False),
    "12-paper-figures.ipynb": (NB12, False, False),
    "13A-export-test-images-for-relabel.ipynb": (NB13A, False, False),
    "13B-rescore-on-corrected-labels.ipynb": (NB13B, False, False),
    **{f"{NAMES[k]}.ipynb": (nb02(k), True, True) for k in PLANS if k.startswith("07")},
    **{f"{NAMES[k]}.ipynb": (nb08(k), True, True) for k in ("08A", "08B", "08C")},
    "09-errors-mcnemar-ensembles.ipynb": (NB09, False, True),
    "10-dataset-statistics-and-agreement.ipynb": (NB10, False, False),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".ipynb"):
            os.remove(os.path.join(OUT, f))
    for name, (cells, gpu, net) in NOTEBOOKS.items():
        src = "".join(cells[0]["source"]) if cells and cells[0].get("cell_type") == "markdown" else ""
        if "DOTA" in src and "dota" not in name.lower():
            src = src.replace("DOTA", "")
        extra = inputs_cell(src)
        if extra is not None:
            cells = [cells[0], extra] + list(cells[1:])
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(nb(cells, gpu, net), f, indent=1)
    print("wrote", len(NOTEBOOKS), "notebooks to", OUT)


if __name__ == "__main__":
    main()
