"""
Central configuration for the SAHI satellite object-detection study (paper ID 355 / 524 revision).

Everything that must stay IDENTICAL across notebooks and Kaggle accounts lives here, so the
five detectors are compared fairly. Paths can be overridden with environment variables
(tools/smoke_test.py does this); on Kaggle the defaults are correct.
"""
import os

SEED = 42
SEEDS = [42, 43, 44]                    # final multi-seed runs on the grouped split

# ---- Paths ---------------------------------------------------------------
INPUT_ROOT = os.environ.get("SAT_INPUT", "/kaggle/input")
WORK_DIR = os.environ.get("SAT_WORK", "/kaggle/working")
RESULTS_DIR = os.path.join(WORK_DIR, "results")       # small files: JSON / CSV  (share these)
WEIGHTS_DIR = os.path.join(WORK_DIR, "weights")       # best.pt per run         (large)
FIG_DIR = os.path.join(WORK_DIR, "figures")
SPLITS_OUT = os.path.join(WORK_DIR, "splits")         # notebook 02 output
AUDIT_OUT = os.path.join(WORK_DIR, "audit")           # notebook 01 output
SMOKE = os.environ.get("SAT_SMOKE", "0") == "1"       # set by tools/smoke_test.py only

# ---- Data source -----------------------------------------------------------
# Recommended: the frozen Kaggle copy you already used for the paper (rifat4018/satelite-coco).
# Optional: download the same version from Roboflow in notebook 00 (needs an API key secret).
KAGGLE_DATASET = "rifat4018/satelite-coco"
RF_WORKSPACE = os.environ.get("RF_WORKSPACE", "rifat4018")            # CHECK: the slug in your Roboflow URL
RF_PROJECT = os.environ.get("RF_PROJECT", "aerial-object-detection-u3f5u")
RF_VERSION = int(os.environ.get("RF_VERSION", "1"))                    # CHECK: the version you exported
RF_FORMAT = "coco"

SPLIT_DIRS = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}
ANN_FILE = "_annotations.coco.json"

# ---- Classes ---------------------------------------------------------------
CLASSES = ["plane", "ship", "vehicle"]          # YOLO class index = position in this list
# other exports' class names mapped onto ours (SkySeaLand: airplane, boat, car, ship; boat+ship -> ship)
CLASS_ALIASES = {"airplane": "plane", "aircraft": "plane", "boat": "ship", "car": "vehicle"}
# how to tell the attached exports apart: a class name only that export uses
# "main" and "native" share class names; they are told apart by image size (main = all 640x640)
DATASETS = {"main": dict(signature="vehicle", prefix="", title="satelite-coco (paper dataset, 640x640 stretched)"),
            "native": dict(signature="vehicle", prefix="native_",
                           title="the SAME Roboflow project exported with NO resize and NO augmentation"),
            "sky": dict(signature="car", prefix="sky_", title="SkySeaLand v14 (aquantic-research, CC BY 4.0)")}

# ---- Extra splits (notebook 06) -----------------------------------------------
LEARNING_CURVE = [0.25, 0.50]   # fractions of the grouped TRAIN groups (1.0 = the grouped split itself)
TILE = 640                      # native-resolution tiles for tile training
TILE_OVERLAP = 0.2
TILE_MIN_VIS = 0.5              # a box is kept in a tile if >= 50% of its area is inside

# ---- Annotation cleaning (notebook 02) --------------------------------------
MIN_BOX_PX = 1.0          # boxes thinner than this after corner clipping are removed (logged)
DUP_ROUND = 1             # boxes equal after rounding to this many decimals = exact duplicates

# ---- Leakage audit (notebook 01) -------------------------------------------
PHASH_MAX_HAMMING = 8     # pHash distance (min over the 8 flips/rotations) counted as near-duplicate
ORB_FEATURES = 500
ORB_RATIO = 0.75
ORB_MIN_GOOD = 15         # good matches needed before RANSAC is attempted
ORB_MIN_INLIERS = 25      # RANSAC homography inliers that count as "same ground area" (overlap)
WATERMARK_LONG, WATERMARK_SHORT = 0.35, 0.16   # corner areas ignored by ORB (fraction of width / height)
ORB_MIN_SPREAD = 0.05     # inliers must cover >= 5% of the image (bbox area); rejects matches on the
                          # "Google Earth" watermark / UI text, which are clustered in a tiny corner
AUDIT_WORKERS = max(1, (os.cpu_count() or 2))

# ---- Grouped split (notebook 02) -------------------------------------------
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}   # of GROUPS (one group = one ground area)
SPLIT_TRIALS = 3000       # random group assignments tried; the most class-balanced one is kept

# ---- Models ---------------------------------------------------------------
# batch sizes = the ones in your paper (hardware-limited on a T4). Ultralytics accumulates gradients
# to a nominal batch of 64 (nbs=64), so the effective batch is ~64 for every model.
# Optional keys: imgsz (train + test input size), yaml (architecture file, weights = pretrained to load),
# group (which table block the model belongs to).
MODELS = {
    # --- paper models ---
    "YOLO11n":   dict(weights="yolo11n.pt",  family="yolo",   batch=32, group="paper"),
    "YOLO11s":   dict(weights="yolo11s.pt",  family="yolo",   batch=16, group="paper"),
    "YOLO11m":   dict(weights="yolo11m.pt",  family="yolo",   batch=4,  group="paper"),
    "YOLOv8m":   dict(weights="yolov8m.pt",  family="yolo",   batch=4,  group="paper"),
    "RT-DETR-L": dict(weights="rtdetr-l.pt", family="rtdetr", batch=4,  group="paper"),
    # --- more architectures (same recipe) ---
    "YOLOv8s":   dict(weights="yolov8s.pt",  family="yolo",   batch=16, group="arch"),
    "YOLOv9s":   dict(weights="yolov9s.pt",  family="yolo",   batch=16, group="arch"),
    "YOLOv10s":  dict(weights="yolov10s.pt", family="yolo",   batch=16, group="arch"),
    "YOLO12s":   dict(weights="yolo12s.pt",  family="yolo",   batch=16, group="arch"),
    "YOLO26s":   dict(weights="yolo26s.pt",  family="yolo",   batch=16, group="arch"),
    "YOLOv9m":   dict(weights="yolov9m.pt",  family="yolo",   batch=4,  group="arch"),
    "YOLOv10m":  dict(weights="yolov10m.pt", family="yolo",   batch=4,  group="arch"),
    "YOLO12m":   dict(weights="yolo12m.pt",  family="yolo",   batch=4,  group="arch"),
    "YOLO26m":   dict(weights="yolo26m.pt",  family="yolo",   batch=4,  group="arch"),
    # --- small-object head (extra stride-4 P2 detection layer), pretrained backbone from the base model ---
    "YOLOv8s-P2": dict(weights="yolov8s.pt", yaml="yolov8s-p2.yaml", family="yolo", batch=16, group="p2"),
    "YOLOv8m-P2": dict(weights="yolov8m.pt", yaml="yolov8m-p2.yaml", family="yolo", batch=4,  group="p2"),
    # --- higher input resolution (train AND test at this size) ---
    "YOLO11s-1024": dict(weights="yolo11s.pt", family="yolo", batch=8, imgsz=1024, group="hires"),
    "YOLO11s-1280": dict(weights="yolo11s.pt", family="yolo", batch=8, imgsz=1280, group="hires"),
    "YOLO11m-1024": dict(weights="yolo11m.pt", family="yolo", batch=4, imgsz=1024, group="hires"),
    "YOLO11m-1280": dict(weights="yolo11m.pt", family="yolo", batch=4, imgsz=1280, group="hires"),
}


def model_imgsz(key):
    return MODELS.get(key, {}).get("imgsz", IMGSZ)

# Same recipe as your original NB1/NB2 runs, but every value is now EXPLICIT.
# Your original logs show optimizer=auto -> AdamW(lr=0.001429, momentum=0.9) and cos_lr=False,
# so this reproduces them; the paper text must say "linear LR decay", not cosine.
TRAIN_PARAMS = dict(
    epochs=50, imgsz=640, patience=20, warmup_epochs=3,
    optimizer="AdamW", lr0=0.001429, momentum=0.9, weight_decay=0.0005, cos_lr=False,
    mosaic=1.0, mixup=0.1, copy_paste=0.1, degrees=45.0, translate=0.1, scale=0.5,
    flipud=0.5, fliplr=0.5, hsv_h=0.015, hsv_s=0.7,
    workers=2, deterministic=True, amp=True, plots=True, verbose=False,
)
ULTRALYTICS_PIN = "ultralytics==8.4.37"   # the version in your original logs

# ---- Evaluation (test split only; best.pt is chosen on val by Ultralytics) ----
IMGSZ = 640
EVAL_CONF = 0.001          # standard for mAP (your old tables used 0.25 -> mAP was understated)
PRED_CONF = 0.01           # confidence floor for the COCO-style AP of standard vs SAHI predictions
OP_CONF = 0.25             # operating point for counting TP / FP / FN (your old "raw detection" counts)
NMS_IOU = 0.7
MAX_DET = 1000             # images contain up to ~354 objects; COCO's default 100 would cap recall
MATCH_IOU = 0.5            # IoU for a detection to count as a true positive
SAHI_SLICES = [480, 320, 160]
SAHI_OVERLAP = 0.2
SAHI_POSTPROCESS = dict(postprocess_type="NMS", postprocess_match_metric="IOU",
                        postprocess_match_threshold=0.5, perform_standard_pred=True)
MODES = ["standard"] + [f"sahi{s}" for s in SAHI_SLICES]
# native-resolution images (~1900x850): slices at and below the training tile size
SAHI_SLICES_NATIVE = [640, 480, 320]


def modes_for(split):
    sl = SAHI_SLICES_NATIVE if split.startswith("native") else SAHI_SLICES
    return ["standard"] + [f"sahi{s}" for s in sl]


# SAHI ablation (notebook 08): one factor changed at a time from the paper setting (NMS, IoU, 0.2, full pass)
SAHI_ABLATION = {
    "paper_nms_iou":     dict(postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.5,
                              perform_standard_pred=True, overlap=0.2),
    "greedynmm_ios":     dict(postprocess_type="GREEDYNMM", postprocess_match_metric="IOS", postprocess_match_threshold=0.5,
                              perform_standard_pred=True, overlap=0.2),
    "nmm_ios":           dict(postprocess_type="NMM", postprocess_match_metric="IOS", postprocess_match_threshold=0.5,
                              perform_standard_pred=True, overlap=0.2),
    "nms_ios":           dict(postprocess_type="NMS", postprocess_match_metric="IOS", postprocess_match_threshold=0.5,
                              perform_standard_pred=True, overlap=0.2),
    "no_full_image":     dict(postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.5,
                              perform_standard_pred=False, overlap=0.2),
    "overlap_0.1":       dict(postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.5,
                              perform_standard_pred=True, overlap=0.1),
    "overlap_0.3":       dict(postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.5,
                              perform_standard_pred=True, overlap=0.3),
    # SAHI's default merge (GREEDYNMM / IoS) is meant for a model confidence >= 0.1; at 0.01 it chains low-score
    # boxes into huge merged boxes. These two rows compare the merge rules at SAHI's own operating confidence 0.25
    # (AP is then computed from detections >= 0.25 only -> compare TP / FP, and compare the two rows with each other).
    "paper_nms_iou_c25": dict(postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.5,
                              perform_standard_pred=True, overlap=0.2, model_conf=0.25),
    "greedynmm_ios_c25": dict(postprocess_type="GREEDYNMM", postprocess_match_metric="IOS", postprocess_match_threshold=0.5,
                              perform_standard_pred=True, overlap=0.2, model_conf=0.25),
}

# External test set (notebook 08C): DOTA-v1.0 (horizontal boxes from the oriented polygons)
DOTA_CLASS_MAP = {"plane": "plane", "ship": "ship", "small-vehicle": "vehicle", "large-vehicle": "vehicle"}
DOTA_MAX_IMAGES = 300        # images that contain at least one mapped class (random, fixed seed)
DOTA_SLICES = [640, 1024]    # native DOTA pixels; standard = whole image at 640
WARMUP_IMAGES = 3

# Object size buckets by box AREA in px^2 on the 640x640 images (COCO convention + a micro bucket).
# COCO 'small' (<32^2) INCLUDES 'micro' (<16^2); the disjoint buckets are used for counts.
AREA_RANGES = [("all", 0, 1e10), ("micro", 0, 16 ** 2), ("small", 0, 32 ** 2),
               ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, 1e10)]
BUCKETS = [("micro", 0, 16 ** 2), ("tiny16_32", 16 ** 2, 32 ** 2),
           ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, 1e10)]

BOOTSTRAP_ITERS = 2000
SESSION_HOURS = 11.5       # Kaggle stops a session at 12 h; runs are not started after this


# notebook 08D: inference-only baselines against SAHI (same weights, no retraining)
EXTRA_MODES = ["tta", "res1024"]
