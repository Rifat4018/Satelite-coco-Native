"""Notebook 12: 40+ paper figures from the results that notebooks 02-08 already saved (CPU, no new training).
Every figure is independent: if its data is not attached it is skipped and listed at the end."""
import os, collections, traceback

import numpy as np
import pandas as pd

import config as cfg
import utils
import metrics
import report

FIG_DIR = os.path.join(report.PAPER_DIR, "figures")
DONE, SKIPPED = [], []
PAPER = [m for m, v in cfg.MODELS.items() if v.get("group") == "paper"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    return plt


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    p = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(p)
    _plt().close(fig)
    DONE.append(name)
    return p


def run(name, fn, *a, **k):
    try:
        r = fn(*a, **k)
        if r is False:
            SKIPPED.append((name, "no data"))
    except Exception as e:                  # one broken figure never stops the others
        SKIPPED.append((name, f"{type(e).__name__}: {e}"))
        traceback.print_exc(limit=1)


# ------------------------------------------------------------------ helpers
def _std(df, split="grouped", mode="standard"):
    d = report.indomain(df)
    d = d[(d.split == split) & (d["mode"] == mode)].copy()
    d["model"] = d.model.astype(str)
    return d


def _agg(d, col, by="model"):
    g = d.groupby(by)[col].agg(["mean", "std", "count"]).reset_index()
    return g.sort_values("mean", ascending=False)


def _bar(ax, g, label_col="model", color="#4C72B0"):
    x = np.arange(len(g))
    ax.bar(x, g["mean"], yerr=g["std"].fillna(0), color=color, capsize=2, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(g[label_col], rotation=60, ha="right", fontsize=7)


def _gt(split="grouped"):
    import dataset as ds
    sdir = ds.find_split_dir(split)
    out = {sp: utils.load_json(os.path.join(sdir, "coco", f"{sp}.json")) for sp in ("train", "val", "test")}
    man = pd.read_csv(os.path.join(sdir, "manifest.csv"), dtype={"source_id": str, "file_name": str, "key": str})
    return out, man


def _source_boxes():
    """One image per source (no augmented copies) of the grouped split -> boxes DataFrame."""
    gts, man = _gt("grouped")
    keep = set(man.sort_values(["split", "key"]).drop_duplicates("source_id").image_id)
    rows, imgs = [], {}
    for sp, g in gts.items():
        for im in g["images"]:
            if im["id"] in keep:
                imgs[im["id"]] = (sp, im)
        for a in g["annotations"]:
            if a["image_id"] in keep:
                x, y, w, h = a["bbox"]
                rows.append(dict(image_id=a["image_id"], split=sp, cls=cfg.CLASSES[a["category_id"] - 1],
                                 x=x, y=y, w=w, h=h, side=np.sqrt(max(w * h, 1e-6))))
    return pd.DataFrame(rows), imgs


# ------------------------------------------------------------------ A. dataset
def f_class_counts(b):
    plt = _plt(); fig, ax = plt.subplots(figsize=(4, 3))
    c = b.cls.value_counts().reindex(cfg.CLASSES)
    ax.bar(c.index, c.values, color=["#4C72B0", "#55A868", "#C44E52"])
    for i, v in enumerate(c.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("objects"); ax.set_title("Objects per class (one image per source)")
    _save(fig, "F01_class_distribution")


def f_size_hist(b):
    plt = _plt(); fig, ax = plt.subplots(figsize=(5, 3))
    bins = np.logspace(0, np.log10(max(b.side.max(), 10)), 40)
    for c in cfg.CLASSES:
        ax.hist(b[b.cls == c].side, bins=bins, histtype="step", linewidth=1.4, label=c)
    for t, l in [(16, "16 px"), (32, "32 px"), (96, "96 px")]:
        ax.axvline(t, color="gray", ls="--", lw=0.8); ax.text(t, ax.get_ylim()[1] * 0.9, l, fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("box side sqrt(w*h) [px, 640x640]"); ax.set_ylabel("objects")
    ax.legend(); ax.set_title("Object size distribution")
    _save(fig, "F02_object_size_histogram")


def f_size_buckets(b):
    plt = _plt(); fig, ax = plt.subplots(figsize=(5, 3))
    area = b.side ** 2
    lab = pd.cut(area, [0, 16 ** 2, 32 ** 2, 96 ** 2, 1e12], labels=["<16 px", "16-32 px", "32-96 px", ">96 px"])
    t = pd.crosstab(b.cls, lab, normalize="index").reindex(cfg.CLASSES) * 100
    t.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="black", linewidth=0.3)
    ax.set_ylabel("% of objects"); ax.set_xlabel(""); ax.legend(fontsize=7, title="box side")
    ax.set_title("Share of objects per size range"); ax.tick_params(axis="x", rotation=0)
    _save(fig, "F03_size_ranges_per_class")


def f_objects_per_image(b, imgs):
    plt = _plt(); fig, ax = plt.subplots(figsize=(4.5, 3))
    n = b.groupby("image_id").size().reindex(list(imgs), fill_value=0)
    ax.hist(n, bins=40, color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax.axvline(n.median(), color="red", ls="--", label=f"median {n.median():.0f}")
    ax.set_xlabel("objects per image"); ax.set_ylabel("images"); ax.legend(); ax.set_title("Object density")
    _save(fig, "F04_objects_per_image")


def f_centre_heatmap(b, imgs):
    plt = _plt(); fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, c in zip(axes, cfg.CLASSES):
        s = b[b.cls == c]
        W = np.array([imgs[i][1]["width"] for i in s.image_id]); H = np.array([imgs[i][1]["height"] for i in s.image_id])
        ax.hist2d((s.x + s.w / 2) / W, (s.y + s.h / 2) / H, bins=25, cmap="magma")
        ax.set_title(c); ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis()
    fig.suptitle("Object centre positions (normalised image coordinates)")
    _save(fig, "F05_object_centre_heatmaps")


def f_aspect(b):
    plt = _plt(); fig, ax = plt.subplots(figsize=(4.5, 3))
    ar = np.maximum(b.w, b.h) / np.maximum(np.minimum(b.w, b.h), 1e-6)
    data = [ar[b.cls == c].clip(upper=8) for c in cfg.CLASSES]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(cfg.CLASSES) + 1)); ax.set_xticklabels(cfg.CLASSES)
    ax.set_ylabel("long side / short side"); ax.set_title("Box aspect ratio")
    _save(fig, "F06_aspect_ratio")


def f_split_composition():
    plt = _plt(); gts, _ = _gt("grouped")
    rows = []
    for sp, g in gts.items():
        c = collections.Counter(cfg.CLASSES[a["category_id"] - 1] for a in g["annotations"])
        rows.append(dict(split=sp, **{k: c[k] for k in cfg.CLASSES}))
    t = pd.DataFrame(rows).set_index("split")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    (t.div(t.sum(1), axis=0) * 100).plot(kind="bar", stacked=True, ax=ax, edgecolor="black", linewidth=0.3)
    ax.set_ylabel("% of boxes"); ax.set_title("Class mix per split (grouped)"); ax.tick_params(axis="x", rotation=0)
    _save(fig, "F07_split_class_mix")


def _draw(ax, img, boxes, color, lw=0.8):
    import matplotlib.patches as P
    ax.imshow(img)
    for (x, y, w, h) in boxes:
        ax.add_patch(P.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw))
    ax.axis("off")


def f_sample_grid(info):
    import cv2
    plt = _plt(); gts, _ = _gt("grouped"); g = gts["test"]
    by = collections.defaultdict(list)
    for a in g["annotations"]:
        by[a["image_id"]].append(a)
    ims = sorted(g["images"], key=lambda i: -len(by[i["id"]]))
    pick = ims[::max(1, len(ims) // 9)][:9]
    cols = {1: "cyan", 2: "lime", 3: "yellow"}
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, im in zip(axes.ravel(), pick):
        img = cv2.cvtColor(cv2.imread(os.path.join(info["images"]["test"], im["file_name"])), cv2.COLOR_BGR2RGB)
        import matplotlib.patches as P
        ax.imshow(img)
        for a in by[im["id"]]:
            x, y, w, h = a["bbox"]
            ax.add_patch(P.Rectangle((x, y), w, h, fill=False, edgecolor=cols[a["category_id"]], linewidth=0.6))
        ax.set_title(f"{len(by[im['id']])} objects", fontsize=8); ax.axis("off")
    fig.suptitle("Test images with ground truth (cyan plane, green ship, yellow vehicle)")
    _save(fig, "F08_sample_images_ground_truth")


# ------------------------------------------------------------------ B. architecture benchmark
def f_metric_bars(df, col, name, ylabel):
    d = _std(df)
    if d.empty or col not in d:
        return False
    plt = _plt(); g = _agg(d, col)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = ["#C44E52" if m in PAPER else "#4C72B0" for m in g.model]
    x = np.arange(len(g))
    ax.bar(x, g["mean"], yerr=g["std"].fillna(0), color=colors, capsize=2, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(g.model, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} on the grouped test split (mean +/- s.d. over seeds; red = models of the original paper)")
    _save(fig, name)


def f_scatter_cost(df, xcol, name, xlabel, logx=False):
    d = _std(df)
    if d.empty:
        return False
    plt = _plt(); g = d.groupby("model").agg(x=(xcol, "mean"), ap=("AP_all", "mean"), sd=("AP_all", "std")).dropna(subset=["x"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(g.x, g.ap, yerr=g.sd.fillna(0), fmt="o", ms=4, capsize=2)
    for m, r in g.iterrows():
        ax.annotate(m, (r.x, r.ap), fontsize=6, xytext=(3, 2), textcoords="offset points")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("COCO AP@[.5:.95]"); ax.set_title(f"Accuracy vs {xlabel.split(' [')[0].lower()}")
    _save(fig, name)


def f_per_class_heatmap(df):
    d = _std(df)
    cols = [f"AP50_cls_{c}" for c in cfg.CLASSES]
    if d.empty or not all(c in d for c in cols):
        return False
    plt = _plt(); t = d.groupby("model")[cols].mean().sort_values(cols[0], ascending=False)
    fig, ax = plt.subplots(figsize=(4, 0.28 * len(t) + 1))
    im = ax.imshow(t.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cfg.CLASSES); ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=7)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f"{t.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")
    fig.colorbar(im, ax=ax); ax.set_title("Per-class AP50 (standard inference)")
    _save(fig, "F14_per_class_ap50_heatmap")


def f_seed_spread(df):
    d = _std(df)
    if d.empty:
        return False
    plt = _plt(); order = _agg(d, "AP_all").model.tolist()
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, m in enumerate(order):
        v = d[d.model == m].AP_all.values
        ax.scatter(np.full(len(v), i) + np.random.RandomState(0).uniform(-0.1, 0.1, len(v)), v, s=12)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("COCO AP"); ax.set_title("Seed-to-seed variation (each dot = one training seed)")
    _save(fig, "F15_seed_variation")


# ------------------------------------------------------------------ C. SAHI
def _paper_modes(df, split="grouped"):
    d = report.indomain(df)
    d = d[(d.split == split) & d.model.astype(str).isin(PAPER) & d["mode"].astype(str).str.match(r"^(standard|sahi\d+)$")].copy()
    d["model"] = d.model.astype(str)
    return d


def f_sahi_lines(df, col, name, ylabel, split="grouped"):
    d = _paper_modes(df, split)
    if d.empty or col not in d:
        return False
    plt = _plt(); modes = ["standard"] + [m for m in cfg.modes_for(split)[1:]]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m in PAPER:
        g = d[d.model == m].groupby("mode")[col].mean().reindex(modes)
        ax.plot(range(len(modes)), g.values, marker="o", label=m)
    ax.set_xticks(range(len(modes))); ax.set_xticklabels([report.mode_label(m) for m in modes])
    ax.set_ylabel(ylabel); ax.legend(fontsize=7); ax.set_title(f"{ylabel}: standard vs SAHI ({split})")
    _save(fig, name)


def f_sahi_tpfp(df):
    d = _paper_modes(df)
    if d.empty:
        return False
    plt = _plt(); modes = cfg.modes_for("grouped")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for ax, col, t in [(axes[0], "op_TP", "true positives"), (axes[1], "op_FP", "false positives")]:
        w = 0.8 / len(modes)
        for k, mo in enumerate(modes):
            g = d[d["mode"] == mo].groupby("model")[col].mean().reindex(PAPER)
            ax.bar(np.arange(len(PAPER)) + k * w, g.values, w, label=report.mode_label(mo))
        ax.set_xticks(np.arange(len(PAPER)) + 0.4 - w / 2); ax.set_xticklabels(PAPER, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{t} at conf 0.25")
    axes[0].legend(fontsize=7)
    _save(fig, "F19_sahi_tp_fp_counts")


def f_recall_by_size(df):
    d = _paper_modes(df)
    cols = [f"op_R_{b}" for b, _, _ in cfg.BUCKETS]
    if d.empty or not all(c in d for c in cols):
        return False
    plt = _plt(); fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for mo in cfg.modes_for("grouped"):
        v = d[d["mode"] == mo][cols].mean()
        ax.plot(range(len(cols)), v.values, marker="o", label=report.mode_label(mo))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(["<16 px", "16-32 px", "32-96 px", ">96 px"])
    ax.set_ylabel("recall (conf 0.25, IoU 0.5)"); ax.legend(fontsize=7)
    ax.set_title("Recall per object size, mean of the 5 paper models")
    _save(fig, "F20_recall_by_size_sahi")


def f_sahi_latency(df):
    d = _paper_modes(df)
    if d.empty:
        return False
    plt = _plt(); modes = cfg.modes_for("grouped")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m in PAPER:
        g = d[d.model == m].groupby("mode").ms.mean().reindex(modes)
        ax.plot(range(len(modes)), g.values, marker="o", label=m)
    ax.set_yscale("log"); ax.set_xticks(range(len(modes))); ax.set_xticklabels([report.mode_label(m) for m in modes])
    ax.set_ylabel("ms per image (T4, log)"); ax.legend(fontsize=7); ax.set_title("Inference cost of slicing")
    _save(fig, "F21_sahi_latency")


def f_ablation(abl, split, col, name, ylabel):
    a = abl[abl.split == split] if len(abl) else abl
    if a.empty or col not in a:
        return False
    plt = _plt(); slices = sorted(a.slice.unique(), reverse=True)
    configs = [c for c in cfg.SAHI_ABLATION if c in set(a.config)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / len(slices)
    for k, s in enumerate(slices):
        g = a[a.slice == s].groupby("config")[col].agg(["mean", "std"]).reindex(configs)
        ax.bar(np.arange(len(configs)) + k * w, g["mean"], w, yerr=g["std"].fillna(0), capsize=1.5, label=f"slice {s}")
    ax.set_xticks(np.arange(len(configs)) + 0.4 - w / 2); ax.set_xticklabels(configs, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(ylabel); ax.legend(fontsize=7)
    ax.set_title(f"SAHI post-processing ablation, {split} (mean +/- s.d. over models and seeds)")
    _save(fig, name)


def f_ablation_c25(abl):
    a = abl[(abl.split == "grouped") & abl.config.isin(["paper_nms_iou_c25", "greedynmm_ios_c25"])] if len(abl) else abl
    if a.empty:
        return False
    plt = _plt(); fig, ax = plt.subplots(figsize=(6, 3.2))
    for k, c in enumerate(["paper_nms_iou_c25", "greedynmm_ios_c25"]):
        g = a[a.config == c].groupby("model").AP_all.mean().reindex(PAPER)
        ax.bar(np.arange(len(PAPER)) + k * 0.4, g.values, 0.4, label=c.replace("_c25", " (conf 0.25)"))
    ax.set_xticks(np.arange(len(PAPER)) + 0.2); ax.set_xticklabels(PAPER, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("COCO AP"); ax.legend(fontsize=7); ax.set_title("SAHI merge rule at SAHI's own confidence 0.25: NMS vs GREEDYNMM")
    _save(fig, "F26_nms_vs_greedynmm_c25")


def f_ablation_overlap(abl):
    a = abl[(abl.split == "grouped") & abl.config.isin(["overlap_0.1", "paper_nms_iou", "overlap_0.3"])] if len(abl) else abl
    if a.empty:
        return False
    plt = _plt(); fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ov = {"overlap_0.1": 0.1, "paper_nms_iou": 0.2, "overlap_0.3": 0.3}
    a = a.assign(ov=a.config.map(ov))
    for s, g in a.groupby("slice"):
        v = g.groupby("ov").AP_all.mean()
        ax.plot(v.index, v.values, marker="o", label=f"slice {s}")
    ax.set_xlabel("slice overlap ratio"); ax.set_ylabel("COCO AP"); ax.legend(fontsize=7); ax.set_title("Effect of slice overlap")
    _save(fig, "F27_overlap_effect")


# ------------------------------------------------------------------ D. resolution / strategies
def f_train_resolution(df):
    d = _std(df)
    fam = {"YOLO11s": ["YOLO11s", "YOLO11s-1024", "YOLO11s-1280"], "YOLO11m": ["YOLO11m", "YOLO11m-1024", "YOLO11m-1280"]}
    if d.empty:
        return False
    plt = _plt(); fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for f, ms in fam.items():
        x, y, e = [], [], []
        for m in ms:
            g = d[d.model == m]
            if len(g):
                x.append(cfg.model_imgsz(m)); y.append(g.AP_all.mean()); e.append(g.AP_all.std())
        if x:
            ax.errorbar(x, y, yerr=np.nan_to_num(e), marker="o", capsize=2, label=f)
    ax.set_xlabel("training and inference input size [px]"); ax.set_ylabel("COCO AP"); ax.legend()
    ax.set_title("Higher input resolution (no slicing)")
    _save(fig, "F28_train_resolution")


def _strategies(df, base):
    d = report.indomain(df)
    d = d[d.split == "grouped"].copy(); d["model"] = d.model.astype(str); d["mode"] = d["mode"].astype(str)
    rows = []
    for (m, mo), g in d[d.model == base].groupby(["model", "mode"]):
        rows.append((report.mode_label(mo) + " (640 train)", g))
    for m in (f"{base}-1024", f"{base}-1280"):
        g = d[(d.model == m) & (d["mode"] == "standard")]
        if len(g):
            rows.append((f"train+infer {cfg.model_imgsz(m)}", g))
    return rows


def f_strategies(df, base, col, name):
    rows = _strategies(df, base)
    if not rows:
        return False
    plt = _plt(); rows = sorted(rows, key=lambda r: -r[1][col].mean())
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(rows))
    ax.bar(x, [r[1][col].mean() for r in rows], yerr=[np.nan_to_num(r[1][col].std()) for r in rows], capsize=2,
           color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(col.replace("_all", "")); ax.set_title(f"{base}: small-object strategies ({col.replace('_all', '')})")
    _save(fig, name)


def f_strategy_cost(df):
    plt = _plt(); fig, ax = plt.subplots(figsize=(5.5, 4)); n = 0
    for base, mk in [("YOLO11s", "o"), ("YOLO11m", "s")]:
        for lab, g in _strategies(df, base):
            if g.gflops_img.notna().any():
                ax.scatter(g.gflops_img.mean(), g.AP_all.mean(), marker=mk, s=25)
                ax.annotate(f"{base} {lab.replace(' (640 train)', '')}", (g.gflops_img.mean(), g.AP_all.mean()),
                            fontsize=5.5, xytext=(3, 2), textcoords="offset points"); n += 1
    if not n:
        return False
    ax.set_xscale("log"); ax.set_xlabel("GFLOPs per image (all passes, log)"); ax.set_ylabel("COCO AP")
    ax.set_title("Accuracy vs compute of every strategy")
    _save(fig, "F33_strategy_compute")


# ------------------------------------------------------------------ E. splits, ablations, generalisation
def f_orig_vs_grouped(df, col, name, ylabel):
    d = report.indomain(df); d = d[(d["mode"] == "standard") & d.split.isin(["orig", "grouped"])].copy()
    d["model"] = d.model.astype(str); d = d[d.model.isin(PAPER) & (d.seed.astype(str) != "legacy")]
    if d.split.nunique() < 2:
        return False
    plt = _plt(); fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for k, sp in enumerate(["orig", "grouped"]):
        g = d[d.split == sp].groupby("model")[col].mean().reindex(PAPER)
        ax.bar(np.arange(len(PAPER)) + k * 0.4, g.values, 0.4, label={"orig": "original Roboflow split", "grouped": "leakage-free grouped split"}[sp])
    ax.set_xticks(np.arange(len(PAPER)) + 0.2); ax.set_xticklabels(PAPER, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel); ax.legend(fontsize=7); ax.set_title(f"Effect of near-duplicate leakage on {ylabel}")
    _save(fig, name)


def f_split_compare(df, splits, labels, name, title):
    d = report.indomain(df); d = d[(d["mode"] == "standard") & d.split.isin(splits)].copy()
    d["model"] = d.model.astype(str); d = d[d.model.isin(["YOLO11s", "YOLO11m"])]
    if d.split.nunique() < 2:
        return False
    plt = _plt(); fig, ax = plt.subplots(figsize=(5, 3.2)); w = 0.8 / len(splits)
    for k, (sp, lab) in enumerate(zip(splits, labels)):
        g = d[d.split == sp].groupby("model").AP_all.agg(["mean", "std"]).reindex(["YOLO11s", "YOLO11m"])
        ax.bar(np.arange(2) + k * w, g["mean"], w, yerr=g["std"].fillna(0), capsize=2, label=lab)
    ax.set_xticks(np.arange(2) + 0.4 - w / 2); ax.set_xticklabels(["YOLO11s", "YOLO11m"])
    ax.set_ylabel("COCO AP"); ax.legend(fontsize=7); ax.set_title(title)
    _save(fig, name)


def f_native(df):
    d = report.indomain(df); d = d[d.split.isin(["native_grouped", "native_tiles"])].copy()
    if d.empty:
        return False
    d["label"] = d.model.astype(str) + np.where(d.split == "native_tiles", " (tiles)", "")
    plt = _plt(); modes = [m for m in ["standard"] + [f"sahi{s}" for s in cfg.SAHI_SLICES_NATIVE] if m in set(d["mode"])]
    labs = sorted(d.label.unique()); fig, ax = plt.subplots(figsize=(7, 3.4)); w = 0.8 / len(modes)
    for k, mo in enumerate(modes):
        g = d[d["mode"] == mo].groupby("label").AP_all.mean().reindex(labs)
        ax.bar(np.arange(len(labs)) + k * w, g.values, w, label=report.mode_label(mo))
    ax.set_xticks(np.arange(len(labs)) + 0.4 - w / 2); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("COCO AP"); ax.legend(fontsize=7); ax.set_title("Native-resolution screenshots: input size, tiles, SAHI")
    _save(fig, "F38_native_resolution")


def f_transfer(df):
    d = df[df.run_id.astype(str).str.startswith("xfer") & (df["mode"] == "standard")].copy()
    if d.empty:
        return False
    d["model"] = d.model.astype(str); ind = report.indomain(df)
    plt = _plt(); fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, tag, tgt, title in [(axes[0], "xfer-main2sky", "sky_grouped", "ours -> SkySeaLand"),
                                (axes[1], "xfer-sky2main", "grouped", "SkySeaLand -> ours")]:
        x = d[d.run_id.str.startswith(tag)].groupby("model").AP_all.mean().reindex(PAPER)
        i = ind[(ind.split == tgt) & (ind["mode"] == "standard") & (ind.seed.astype(str) == "42")]
        i = i.assign(model=i.model.astype(str)).groupby("model").AP_all.mean().reindex(PAPER)
        ax.bar(np.arange(len(PAPER)) - 0.2, i.values, 0.4, label="trained on target (in-domain)")
        ax.bar(np.arange(len(PAPER)) + 0.2, x.values, 0.4, label="trained on the other dataset")
        ax.set_xticks(range(len(PAPER))); ax.set_xticklabels(PAPER, rotation=30, ha="right", fontsize=7); ax.set_title(title)
    axes[0].set_ylabel("COCO AP"); axes[0].legend(fontsize=7)
    _save(fig, "F39_cross_dataset")


def f_dota(df):
    d = df[df.run_id.astype(str).str.startswith("xdota")].copy()
    if d.empty:
        return False
    d["model"] = d.model.astype(str); modes = list(dict.fromkeys(d["mode"].astype(str)))
    plt = _plt(); fig, ax = plt.subplots(figsize=(6, 3.2)); w = 0.8 / len(modes)
    for k, mo in enumerate(modes):
        g = d[d["mode"] == mo].groupby("model").AP50_all.mean().reindex(PAPER)
        ax.bar(np.arange(len(PAPER)) + k * w, g.values, w, label=report.mode_label(mo))
    ax.set_xticks(np.arange(len(PAPER)) + 0.4 - w / 2); ax.set_xticklabels(PAPER, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("AP50"); ax.legend(fontsize=7); ax.set_title("Zero-shot transfer to DOTA-v1.0 (no fine-tuning)")
    _save(fig, "F40_dota_zero_shot")


# ------------------------------------------------------------------ F. errors / qualitative
def _dets_of(df, model, mode="standard", seed="42", split="grouped"):
    import analysis
    d = report.indomain(df)
    g = d[(d.split == split) & (d.model.astype(str) == model) & (d["mode"] == mode) & (d.seed.astype(str) == seed)]
    return analysis._dets(g.folder.iloc[0], mode) if len(g) else None


def f_errors(df):
    import analysis
    gt = analysis._gt_for("grouped"); rows = []
    for m in PAPER:
        for mo in ["standard", "sahi320"]:
            dt = _dets_of(df, m, mo)
            if dt is not None:
                rows.append(dict(model=m, mode=report.mode_label(mo), **analysis.error_breakdown(gt, dt)))
    if not rows:
        return False
    t = pd.DataFrame(rows); err = ["duplicate", "classification", "localization", "cls+loc", "background"]
    plt = _plt(); fig, ax = plt.subplots(figsize=(7, 3.4)); bottom = np.zeros(len(t))
    for e in err:
        ax.bar(range(len(t)), t[e], bottom=bottom, label=e); bottom += t[e].values
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t.model + "\n" + t["mode"], fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("false positives (conf 0.25)"); ax.legend(fontsize=7); ax.set_title("False-positive types (TIDE), seed 42")
    _save(fig, "F41_false_positive_types")
    fig, ax = plt.subplots(figsize=(7, 3.2)); bottom = np.zeros(len(t))
    for b, _, _ in cfg.BUCKETS:
        ax.bar(range(len(t)), t[f"missed_{b}"], bottom=bottom, label=b); bottom += t[f"missed_{b}"].values
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t.model + "\n" + t["mode"], fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("missed objects"); ax.legend(fontsize=7); ax.set_title("Missed objects by size, seed 42")
    _save(fig, "F42_missed_by_size")


def f_conf_hist(df, model="YOLO11s"):
    import analysis
    dt = _dets_of(df, model)
    if dt is None:
        return False
    gt = analysis._gt_for("grouped"); by = collections.defaultdict(list)
    for a in gt["annotations"]:
        by[a["image_id"]].append(a)
    tp, fp = [], []
    D = collections.defaultdict(list)
    for x in dt:
        D[x["image_id"]].append(x)
    for iid, ds_ in D.items():
        g = by.get(iid, []); used = np.zeros(len(g), bool)
        gb = np.array([a["bbox"] for a in g], float).reshape(-1, 4); gc = np.array([a["category_id"] for a in g])
        for x in sorted(ds_, key=lambda z: -z["score"]):
            iou = metrics._iou_xywh(np.array([x["bbox"]], float), gb)[0] if len(g) else np.zeros(0)
            ok = np.where((gc == x["category_id"]) & ~used & (iou >= cfg.MATCH_IOU))[0] if len(g) else []
            if len(ok):
                used[ok[np.argmax(iou[ok])]] = True; tp.append(x["score"])
            else:
                fp.append(x["score"])
    plt = _plt(); fig, ax = plt.subplots(figsize=(5, 3.2)); bins = np.linspace(0, 1, 41)
    ax.hist(tp, bins=bins, alpha=0.6, label="true positive"); ax.hist(fp, bins=bins, alpha=0.6, label="false positive")
    ax.axvline(cfg.OP_CONF, color="k", ls="--", lw=0.8); ax.set_yscale("log")
    ax.set_xlabel("confidence"); ax.set_ylabel("detections (log)"); ax.legend(); ax.set_title(f"Confidence of TP vs FP ({model}, seed 42)")
    _save(fig, "F43_confidence_tp_fp")


def f_qualitative(df, info, model, modes, name, n=3):
    import cv2, analysis
    gt = analysis._gt_for("grouped"); dets = {mo: _dets_of(df, model, mo) for mo in modes}
    if any(v is None for v in dets.values()):
        return False
    by = collections.defaultdict(list)
    for a in gt["annotations"]:
        by[a["image_id"]].append(a["bbox"])
    ims = sorted(gt["images"], key=lambda i: -len(by[i["id"]]))[1:1 + 3 * n:3][:n]
    plt = _plt(); fig, axes = plt.subplots(n, len(modes) + 1, figsize=(3.2 * (len(modes) + 1), 3.2 * n))
    axes = np.atleast_2d(axes)
    for r, im in enumerate(ims):
        img = cv2.cvtColor(cv2.imread(os.path.join(info["images"]["test"], im["file_name"])), cv2.COLOR_BGR2RGB)
        _draw(axes[r, 0], img, by[im["id"]], "lime"); axes[r, 0].set_title(f"ground truth ({len(by[im['id']])})", fontsize=8)
        for c, mo in enumerate(modes, 1):
            b = [x["bbox"] for x in dets[mo] if x["image_id"] == im["id"] and x["score"] >= cfg.OP_CONF]
            _draw(axes[r, c], img, b, "red"); axes[r, c].set_title(f"{model} {report.mode_label(mo)} ({len(b)})", fontsize=8)
    _save(fig, name)


# ------------------------------------------------------------------ driver
def make_all():
    import dataset as ds
    df = report.long_table(report.collect_runs())
    abl = report.collect_ablation()
    try:
        b, imgs = _source_boxes()
    except Exception as e:
        b, imgs = pd.DataFrame(), {}
        SKIPPED.append(("dataset figures", str(e)))
    if len(b):
        run("F01", f_class_counts, b); run("F02", f_size_hist, b); run("F03", f_size_buckets, b)
        run("F04", f_objects_per_image, b, imgs); run("F05", f_centre_heatmap, b, imgs); run("F06", f_aspect, b)
    run("F07", f_split_composition)
    info = None
    try:
        info = ds.materialize("grouped")
    except Exception as e:
        SKIPPED.append(("images (attach satelite-coco)", str(e)))
    if info:
        run("F08", f_sample_grid, info)
    for col, nm, yl in [("AP_all", "F09_benchmark_ap", "COCO AP@[.5:.95]"), ("AP50_all", "F10_benchmark_ap50", "AP50"),
                        ("AP_small", "F11_benchmark_ap_small", "AP small (<32 px)"),
                        ("AR50_micro", "F12_benchmark_recall_micro", "recall micro (<16 px)"),
                        ("ms", "F13_benchmark_latency", "ms per image (T4)")]:
        run(nm, f_metric_bars, df, col, nm, yl)
    run("F14", f_per_class_heatmap, df); run("F15", f_seed_spread, df)
    run("F16", f_scatter_cost, df, "params_M", "F16_ap_vs_parameters", "parameters [M]")
    run("F17", f_scatter_cost, df, "gflops", "F17_ap_vs_gflops", "GFLOPs per forward pass", True)
    run("F18", f_sahi_lines, df, "AP_all", "F18_sahi_ap", "COCO AP")
    run("F18b", f_sahi_lines, df, "AP_small", "F18b_sahi_ap_small", "AP small")
    run("F18c", f_sahi_lines, df, "AP_all", "F18c_sahi_ap_skysealand", "COCO AP", "sky_grouped")
    run("F19", f_sahi_tpfp, df); run("F20", f_recall_by_size, df); run("F21", f_sahi_latency, df)
    run("F22", f_ablation, abl, "grouped", "AP_all", "F22_sahi_ablation_ap", "COCO AP")
    run("F23", f_ablation, abl, "grouped", "op_FP", "F23_sahi_ablation_fp", "false positives (conf 0.25)")
    run("F24", f_ablation, abl, "grouped", "op_FP_duplicate", "F24_sahi_ablation_duplicates", "duplicate FP")
    run("F25", f_ablation, abl, "sky_grouped", "AP_all", "F25_sahi_ablation_ap_skysealand", "COCO AP")
    run("F26", f_ablation_c25, abl); run("F27", f_ablation_overlap, abl)
    run("F28", f_train_resolution, df)
    run("F29", f_strategies, df, "YOLO11s", "AP_all", "F29_strategies_yolo11s_ap")
    run("F30", f_strategies, df, "YOLO11m", "AP_all", "F30_strategies_yolo11m_ap")
    run("F31", f_strategies, df, "YOLO11s", "AP_small", "F31_strategies_yolo11s_ap_small")
    run("F32", f_strategies, df, "YOLO11s", "AR50_micro", "F32_strategies_yolo11s_recall_micro")
    run("F33", f_strategy_cost, df)
    run("F34", f_orig_vs_grouped, df, "AP_all", "F34_leakage_ap", "COCO AP")
    run("F35", f_orig_vs_grouped, df, "AR50_micro", "F35_leakage_recall_micro", "recall micro")
    run("F36", f_split_compare, df, ["grouped_1copy", "grouped"], ["1 copy per source", "3 augmented copies"],
        "F36_offline_augmentation", "Offline augmentation (same test images)")
    run("F37", f_split_compare, df, ["grouped_lc25", "grouped_lc50", "grouped"], ["25 % of groups", "50 %", "100 %"],
        "F37_learning_curve_bars", "Learning curve (same test images)")
    run("F38", f_native, df); run("F39", f_transfer, df); run("F40", f_dota, df)
    run("F41", f_errors, df); run("F43", f_conf_hist, df)
    if info:
        best = _agg(_std(df), "AP_all").model.iloc[0] if len(_std(df)) else "YOLO11s"
        run("F44", f_qualitative, df, info, "YOLO11s", ["standard", "sahi320"], "F44_qualitative_yolo11s_standard_vs_sahi")
        run("F45", f_qualitative, df, info, "RT-DETR-L", ["standard", "sahi480"], "F45_qualitative_rtdetr_standard_vs_sahi")
        run("F46", f_qualitative, df, info, "YOLO11s", ["standard", "tta"], "F46_qualitative_yolo11s_tta")
        run("F47", f_qualitative, df, info, str(best), ["standard"], "F47_qualitative_best_model")
    print(f"\n{len(DONE)} figures written to {FIG_DIR}")
    for n, why in SKIPPED:
        print("  skipped", n, "-", why)
    return DONE, SKIPPED
