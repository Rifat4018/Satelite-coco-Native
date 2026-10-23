"""Which runs each Kaggle account trains (notebooks 02A-02F). Planning hours come from YOUR logs
(T4, 50 epochs): YOLO11n 0.25 h, YOLO11s 0.37 h, YOLO11m 0.80 h, YOLOv8m 0.79 h, RT-DETR-L 1.77 h,
plus about 10 min of test evaluation per run. They are estimates, not measurements of these notebooks."""
import config as cfg

S = cfg.SEEDS   # [42, 43, 44]

PLANS = {
    # grouped (leakage-free) split, 3 seeds each = the paper's main results
    "A": [("grouped", "RT-DETR-L", 42), ("grouped", "RT-DETR-L", 43)],
    "B": [("grouped", "RT-DETR-L", 44), ("orig", "RT-DETR-L", 42)],
    "C": [("grouped", "YOLO11m", s) for s in S],
    "D": [("grouped", "YOLOv8m", s) for s in S],
    "E": [("grouped", "YOLO11n", s) for s in S] + [("grouped", "YOLO11s", s) for s in S],
    # original Roboflow split, seed 42: same recipe, shows how much the split matters
    "F": [("orig", m, 42) for m in ["YOLO11n", "YOLO11s", "YOLO11m", "YOLOv8m"]],
}

# second dataset (SkySeaLand), its own grouped split, seed 42 - run as a SECOND session on any account
PLANS["G"] = [("sky_grouped", m, 42) for m in ["YOLO11n", "YOLO11s", "YOLO11m", "YOLOv8m", "RT-DETR-L"]]

# ---------------- phase 2 (Tier 1 + 2). Any 07x notebook can start NOW (they do not need phase 1). ----------------
P5 = ["YOLO11n", "YOLO11s", "YOLO11m", "YOLOv8m", "RT-DETR-L"]
PLANS.update({
    "07A": [("grouped", "YOLO11s-1024", s) for s in S] + [("grouped", "YOLO11s-1280", s) for s in S],
    "07B": [("grouped", "YOLO11m-1024", s) for s in S],
    "07C": [("grouped", "YOLO11m-1280", s) for s in (42, 43)],
    "07D": [("grouped", "YOLO11m-1280", 44)] + [("grouped", "YOLOv8s-P2", s) for s in S],
    "07E": [("grouped", m, s) for m in ["YOLOv8s", "YOLOv9s", "YOLOv10s"] for s in S],
    "07F": [("grouped", m, s) for m in ["YOLO12s", "YOLO26s"] for s in S] + [("grouped", "YOLOv8m-P2", s) for s in (42, 43)],
    "07G": [("grouped", m, s) for m in ["YOLOv9m", "YOLOv10m"] for s in S],
    "07H": [("grouped", m, s) for m in ["YOLO12m", "YOLO26m"] for s in S],
    "07I": [("grouped", "YOLOv8m-P2", 44)] + [(sp, m, s) for sp in ["grouped_1copy", "grouped_lc25", "grouped_lc50"]
                                              for m in ["YOLO11s", "YOLO11m"] for s in S],
    # native resolution (needs the native Roboflow export + notebook 06)
    "07J": [("native_grouped", m, s) for m in ["YOLO11s", "YOLO11s-1280", "YOLO11m"] for s in S]
           + [("native_tiles", "YOLO11s", s) for s in S],
    "07K": [("native_tiles", "YOLO11m", s) for s in S],
    "07L": [("native_grouped", "YOLO11m-1280", s) for s in S],
})
# SAHI post-processing ablation (inference only; needs the weights of phase 1 attached)
ABLATION = {"08A": [("grouped", m, s) for m in P5 for s in S],
            "08B": [("sky_grouped", m, 42) for m in P5]}
ABLATION_SLICES = {"08A": None, "08B": [480, 320]}     # None = all slice sizes of the split
EXTERNAL = [("grouped", m, 42) for m in P5]
EXTRA = [("grouped", m, s) for m in P5 for s in S]          # 08D: TTA + inference at 1024 (no training)           # 08C: models tested on DOTA without fine-tuning

NAMES = {"A": "02A-train-rtdetr-l-grouped-s42-s43",
         "B": "02B-train-rtdetr-l-grouped-s44-and-orig-s42",
         "C": "02C-train-yolo11m-grouped-3seeds",
         "D": "02D-train-yolov8m-grouped-3seeds",
         "E": "02E-train-yolo11n-yolo11s-grouped-3seeds",
         "F": "02F-train-4yolo-orig-split-s42",
         "G": "02G-train-skysealand-5models-s42",
         "07A": "07A-train-hires-yolo11s-1024-1280-3seeds",
         "07B": "07B-train-hires-yolo11m-1024-3seeds",
         "07C": "07C-train-hires-yolo11m-1280-s42-s43",
         "07D": "07D-train-hires-yolo11m-1280-s44-and-yolov8s-p2",
         "07E": "07E-train-arch-yolov8s-v9s-v10s-3seeds",
         "07F": "07F-train-arch-yolo12s-yolo26s-and-yolov8m-p2",
         "07G": "07G-train-arch-yolov9m-v10m-3seeds",
         "07H": "07H-train-arch-yolo12m-yolo26m-3seeds",
         "07I": "07I-train-ablation-1copy-learning-curve",
         "07J": "07J-train-native-yolo11s-and-yolo11m-640",
         "07K": "07K-train-native-tiles-yolo11m",
         "07L": "07L-train-native-yolo11m-1280",
         "08A": "08A-sahi-ablation-main",
         "08B": "08B-sahi-ablation-skysealand",
         "08C": "08C-external-test-dota",
         "08D": "08D-tta-and-inference-resolution"}

PILOT = [("grouped", "YOLO11n", 42)]     # 1 epoch, tag 'pilot': checks the whole GPU pipeline in ~10 min


def all_runs():
    return [r for p in PLANS.values() for r in p]


REQUIRED = {"A", "B", "C", "D", "E", "F"}   # phase 1; everything else is reported if present
OPTIONAL = set(PLANS) - REQUIRED


def expected_run_ids(include_optional=False):
    return [f"{sp}__{m}__s{s}" for k, p in PLANS.items() if include_optional or k not in OPTIONAL for sp, m, s in p]


def show(key):
    """Print the runs of THIS notebook only."""
    import train
    p = PLANS[key]
    h = sum(train.EXPECTED_H[m] for _, m, _ in p)
    print(f"{NAMES[key]}: {len(p)} runs, ~{h:.1f} GPU h (planning estimate)")
    for sp, m, s in p:
        print(f"  split={sp:<12} model={m:<10} seed={s}")
