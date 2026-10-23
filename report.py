"""Collect every eval_test.json from the attached result Datasets and build the paper tables / figures.

Tables (CSV + LaTeX in results/paper/):
  T1_main_standard     per model on the grouped split, mean +/- std over seeds (standard inference)
  T2_sahi              per model x inference mode: COCO AP by size + TP/FP at the operating point + latency
  T3_sahi_bootstrap    recall gain of each SAHI mode vs standard with an image-level bootstrap 95% CI
  T4_per_class         per-class AP50 (standard)
  T5_orig_vs_grouped   same model / seed on the original Roboflow split vs the grouped split (+ legacy weights)
Figures (results/paper/fig_*.png, 300 dpi): Pareto fronts, size-stratified recall, extra-detection
decomposition, qualitative TP/FP/FN overlay.
"""
import os, glob, collections

import numpy as np
import pandas as pd

import config as cfg
import utils
import metrics

PAPER_DIR = os.path.join(cfg.RESULTS_DIR, "paper")
MODEL_ORDER = list(cfg.MODELS)
class _ModeLabel(dict):
    def __missing__(self, k):
        if k == "standard":
            return "Standard"
        if k.startswith("sahi"):
            return "SAHI-" + k[4:]
        if k == "tta":
            return "TTA (flip + 3 scales)"
        if k.startswith("res"):
            return f"Inference @{k[3:]}"
        return k


MODE_LABEL = _ModeLabel()


def mode_label(m):
    return MODE_LABEL[m]


def paper_models(present=None):
    m = [k for k, v in cfg.MODELS.items() if v.get("group") == "paper"]
    return [x for x in m if present is None or x in present]


# ---------------------------------------------------------------- loading
def collect_runs():
    """{run_id: (eval dict, folder)}; if a run exists twice the newest file wins (printed)."""
    found = collections.defaultdict(list)
    for p in utils.find_files("eval_test.json", also_work=True):
        d = utils.load_json(p)
        if "__pilot" in d["run_id"] or "__smoke" in d["run_id"]:
            continue
        has_dets = utils.dets_path(os.path.dirname(p), "standard") is not None
        found[d["run_id"]].append((int(d.get("label_version", 1)), has_dets, os.path.getmtime(p), p, d))
    runs = {}
    for rid, lst in sorted(found.items()):
        lst.sort(key=lambda t: t[:3])
        if len(lst) > 1:
            pass                     # same run in results/ and results_<X>/ of one Dataset: keep the copy with detections
        runs[rid] = (lst[-1][4], os.path.dirname(lst[-1][3]))
    n_extra = 0
    extras = [(utils.load_json(p), p) for p in utils.find_files("eval_extra.json", also_work=True)]
    extras.sort(key=lambda t: int(t[0].get("label_version", 1)))       # corrected labels (13) applied last
    for e, p in extras:                                                  # 08D: extra inference modes
        if e["run_id"] in runs:
            d = runs[e["run_id"]][0]
            for m, v in e["modes"].items():
                d["modes"][m] = dict(v, _folder=os.path.dirname(p))
                n_extra += 1
    nv2 = sum(int(d.get("label_version", 1)) > 1 for d, _ in runs.values())
    if nv2:
        print(f"{nv2} runs use the CORRECTED test labels (notebook 13)")
    print(f"{len(runs)} evaluated runs found" + (f" (+{n_extra} TTA / inference-resolution modes from 08D)" if n_extra else ""))
    return runs


def seed_of(rid):
    s = rid.split("__")[-1]
    s = s[1:] if s.startswith("s") else s
    return int(s) if s.isdigit() else s


def long_table(runs):
    rows = []
    for rid, (d, folder) in runs.items():
        u = d.get("ultralytics_test", {})
        for mode, m in d["modes"].items():
            c, o = m["coco"], m["op"]
            rows.append(dict(run_id=rid, split=d["split"], fingerprint=d["split_fingerprint"], model=d["model"],
                             transfer=rid.startswith(("xfer", "xdota")), source=(rid.split("__")[0] if rid.startswith(("xfer", "xdota")) else d["split"]),
                             group=cfg.MODELS.get(d["model"], {}).get("group", "other"), imgsz=d.get("imgsz", cfg.IMGSZ),
                             seed=seed_of(rid), mode=mode, params_M=d["params_M"], gflops=d["gflops"],
                             gflops_img=m["gflops_per_image"], passes=m["forward_passes_per_image"],
                             ms=m["ms_per_image"], fps=m["fps_end_to_end"],
                             u_mAP50=u.get("mAP50") if mode == "standard" else np.nan,
                             u_mAP50_95=u.get("mAP50_95") if mode == "standard" else np.nan,
                             u_P=u.get("P") if mode == "standard" else np.nan,
                             u_R=u.get("R") if mode == "standard" else np.nan,
                             **{k: v for k, v in c.items() if k.startswith(("AP", "AR"))},
                             **{f"op_{k}": v for k, v in o.items()}, folder=m.get("_folder", folder)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    fps = df.groupby("split").fingerprint.nunique()     # transfer runs share the TARGET split's fingerprint
    for sp, n in fps.items():
        if n > 1:
            print(f"WARNING: split '{sp}' has {n} different fingerprints across runs -> runs used DIFFERENT "
                  f"versions of the split. Use only one 'sat-splits' Dataset version everywhere.")
    df["model"] = pd.Categorical(df.model, [m for m in MODEL_ORDER if m in set(df.model)] +
                                 sorted(set(df.model) - set(MODEL_ORDER)))
    return df


# ---------------------------------------------------------------- formatting
def pm(series, digits=3):
    s = series.dropna()
    if s.empty:
        return "--"
    if len(s) == 1:
        return f"{s.iloc[0]:.{digits}f}"
    return f"{s.mean():.{digits}f} $\\pm$ {s.std(ddof=1):.{digits}f}"


def to_latex(df, path, caption, label):
    cols = list(df.columns)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}",
             "\\toprule", " & ".join(str(c).replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append("--" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append(str(v).replace("_", "\\_").replace("%", "\\%"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def save(df, name, caption):
    os.makedirs(PAPER_DIR, exist_ok=True)
    df.to_csv(os.path.join(PAPER_DIR, f"{name}.csv"), index=False)
    to_latex(df, os.path.join(PAPER_DIR, f"{name}.tex"), caption, f"tab:{name}")
    print(f"\n### {name}: {caption}")
    print(df.to_string(index=False))


def indomain(df):
    return df[~df.transfer] if "transfer" in df else df


# ---------------------------------------------------------------- tables
def table_main(df, split="grouped"):
    df = indomain(df)
    d = df[(df.split == split) & (df["mode"] == "standard")]
    rows = []
    for m, g in d.groupby("model", observed=True):
        rows.append(dict(Model=m, Group=g.group.iloc[0], Input=int(g.imgsz.iloc[0]), Seeds=len(g), Params_M=f"{g.params_M.iloc[0]:.2f}",
                         GFLOPs=("--" if g.gflops.isna().all() else f"{g.gflops.iloc[0]:.1f}"),
                         mAP50=pm(g.u_mAP50), mAP50_95=pm(g.u_mAP50_95), P=pm(g.u_P), R=pm(g.u_R),
                         AP_small=pm(g.AP_small), AR50_micro=pm(g.AR50_micro),
                         ms_img=pm(g.ms, 1), FPS=pm(g.fps, 1)))
    out = pd.DataFrame(rows)
    save(out, f"T1_main_standard_{split}",
         f"Test-set results ({split} split, standard (unsliced) inference at each model's input size, mean $\\pm$ std over seeds; "
         "GFLOPs at that input size). "
         "mAP/P/R from Ultralytics at conf 0.001; AP\\_small = COCO AP for area $<32^2$ px$^2$; "
         "AR50\\_micro = recall at IoU 0.5 for area $<16^2$ px$^2$; latency = end-to-end per image on a T4.")
    return out


def table_sahi(df, split="grouped"):
    df = indomain(df)
    d = df[(df.split == split) & df["mode"].astype(str).str.match(r"^(standard|sahi\d+)$")]   # TTA / res -> T15
    std = d[d["mode"] == "standard"].set_index("run_id")
    rows = []
    for (m, mode), g in d.groupby(["model", "mode"], observed=True):
        base = std.loc[g.run_id]
        rows.append(dict(Model=m, Inference=MODE_LABEL[mode], Passes=int(g.passes.iloc[0]),
                         AP50=pm(g.AP50_all), AP=pm(g.AP_all), AP50_small=pm(g.AP50_small),
                         AR50_micro=pm(g.AR50_micro), AR50_small=pm(g.AR50_small),
                         Dets=pm(g.op_dets, 0), TP=pm(g.op_TP, 0), FP=pm(g.op_FP, 0),
                         extra_TP=pm(pd.Series(g.op_TP.values - base.op_TP.values), 0),
                         extra_FP=pm(pd.Series(g.op_FP.values - base.op_FP.values), 0),
                         P=pm(g.op_P), R=pm(g.op_R), ms_img=pm(g.ms, 1)))
    out = pd.DataFrame(rows)
    order = {MODE_LABEL[m]: i for i, m in enumerate(cfg.modes_for(split))}
    morder = {m: i for i, m in enumerate(MODEL_ORDER)}
    if not out.empty:
        out = out.sort_values(["Model", "Inference"], key=lambda s: s.map(order if s.name == "Inference" else morder))
    save(out, f"T2_sahi_{split}",
         f"Standard vs SAHI on the {split} test split. AP columns: COCO-style (conf $\\geq$ {cfg.PRED_CONF}). "
         f"Dets/TP/FP/P/R: detections with conf $\\geq$ {cfg.OP_CONF} matched one-to-one to ground truth at IoU "
         f"$\\geq$ {cfg.MATCH_IOU}; extra\\_TP / extra\\_FP = change versus standard inference of the same model.")
    return out


def table_bootstrap(df, split="grouped"):
    df = indomain(df)
    d = df[df.split == split]
    rows = []
    for m, g in d.groupby("model", observed=True):
        folders = g[g["mode"] == "standard"].set_index("seed").folder
        for mode in cfg.modes_for(split)[1:]:
            a_list, b_list = [], []
            for seed, folder in folders.items():
                pa, pb = os.path.join(folder, "perimg_standard.csv"), os.path.join(folder, f"perimg_{mode}.csv")
                if os.path.isfile(pa) and os.path.isfile(pb):
                    a_list.append(pd.read_csv(pa))
                    b_list.append(pd.read_csv(pb))
            if not a_list:
                continue
            # average the per-image counts over seeds, then resample images
            A = pd.concat(a_list).groupby("image_id").mean(numeric_only=True).reset_index()
            B = pd.concat(b_list).groupby("image_id").mean(numeric_only=True).reset_index()
            for label, stat in [("recall_all", metrics.recall_stat()), ("recall_micro", metrics.recall_stat("micro")),
                                ("recall_16_32", metrics.recall_stat("tiny16_32")), ("precision", metrics.precision_stat)]:
                dlt, lo, hi, p = metrics.bootstrap_delta(A, B, stat)
                rows.append(dict(Model=m, Inference=MODE_LABEL[mode], Metric=label, Delta=round(dlt, 4),
                                 CI95=f"[{lo:+.4f}, {hi:+.4f}]", p_boot=round(p, 4), seeds=len(a_list)))
    out = pd.DataFrame(rows)
    if not out.empty:
        save(out, f"T3_sahi_bootstrap_{split}",
             f"Change of recall / precision (conf $\\geq$ {cfg.OP_CONF}, IoU $\\geq$ {cfg.MATCH_IOU}) with SAHI "
             f"versus standard inference; 95\\% CI from {cfg.BOOTSTRAP_ITERS} image-level bootstrap resamples.")
    return out


def table_per_class(df, split="grouped"):
    df = indomain(df)
    d = df[(df.split == split) & (df["mode"] == "standard")]
    rows = []
    for m, g in d.groupby("model", observed=True):
        rows.append(dict(Model=m, **{f"AP50_{c}": pm(g[f"AP50_cls_{c}"]) for c in cfg.CLASSES if f"AP50_cls_{c}" in g},
                         **{f"AP_{c}": pm(g[f"AP_cls_{c}"]) for c in cfg.CLASSES if f"AP_cls_{c}" in g}))
    out = pd.DataFrame(rows)
    save(out, f"T4_per_class_{split}", f"Per-class COCO AP50 / AP ({split} test split, standard inference).")
    return out


def table_orig_vs_grouped(df):
    df = indomain(df)
    d = df[(df["mode"] == "standard") & df.split.isin(["orig", "grouped"])]
    d = d[d.model.isin([m for m in set(d[d.split == "orig"].model)])]
    rows = []
    for (m, sp, seed), g in d.groupby(["model", "split", "seed"], observed=True):
        r = g.iloc[0]
        rows.append(dict(Model=m, Split=sp, Seed=seed, Test_images=None, mAP50=r.u_mAP50, mAP50_95=r.u_mAP50_95,
                         AP_small=r.AP_small, AR50_micro=r.AR50_micro))
    out = pd.DataFrame(rows).sort_values(["Model", "Split", "Seed"], key=lambda s: s.astype(str))
    out = out.drop(columns=["Test_images"])
    save(out, "T5_orig_vs_grouped",
         "Same recipe on the original Roboflow split vs the leakage-free grouped split "
         "(seed 'legacy' = the weights reported in the rejected manuscript, re-scored correctly).")
    return out


# ---------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    return plt


def pareto_front(x, y):
    """indices of points not dominated (lower x better, higher y better)."""
    order = np.argsort(x)
    best, keep = -np.inf, []
    for i in order:
        if y[i] > best:
            keep.append(i)
            best = y[i]
    return keep


def fig_pareto(df, split="grouped"):
    df = indomain(df)
    plt = _plt()
    d = df[df.split == split].groupby(["model", "mode"], observed=True).agg(
        ms=("ms", "mean"), gfl=("gflops_img", "mean"), AP=("AP_all", "mean"), AP50=("AP50_all", "mean"),
        ARm=("AR50_micro", "mean"), APs=("AP_small", "mean")).reset_index()
    if d.empty:
        return None
    markers = dict(zip(cfg.modes_for(split), ["o", "s", "^", "D", "v", "P"]))
    import matplotlib
    present = [m for m in MODEL_ORDER if m in set(d.model)]
    cmap = matplotlib.colormaps["tab20"]
    colors = {m: cmap(i % 20) for i, m in enumerate(present)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("ms", "AP", "End-to-end latency per image (ms, T4, log)", "COCO AP@[.5:.95]"),
              ("gfl", "AP", "GFLOPs per image (forward passes x model GFLOPs, log)", "COCO AP@[.5:.95]"),
              ("ms", "ARm", "End-to-end latency per image (ms, T4, log)", r"Recall@0.5, micro objects ($<16^2$ px$^2$)")]
    for ax, (xc, yc, xl, yl) in zip(axes, panels):
        dd = d.dropna(subset=[xc, yc]).reset_index(drop=True)
        for r in dd.itertuples():
            ax.scatter(getattr(r, xc), getattr(r, yc), marker=markers.get(r.mode, "x"),
                       color=colors.get(r.model, "gray"), s=55, edgecolor="k", lw=0.4, zorder=3)
        if len(dd):
            keep = pareto_front(dd[xc].to_numpy(), dd[yc].to_numpy())
            f = dd.iloc[keep].sort_values(xc)
            ax.step(f[xc], f[yc], where="post", color="k", lw=1, ls="--", zorder=2)
            for r in f.itertuples():
                ax.annotate(f"{r.model} {MODE_LABEL[r.mode]}", (getattr(r, xc), getattr(r, yc)),
                            fontsize=6.5, xytext=(4, -9), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, label=m) for m, c in colors.items() if m in set(d.model)]
    h += [Line2D([], [], marker=mk, ls="", color="gray", label=MODE_LABEL[m]) for m, mk in markers.items()]
    h += [Line2D([], [], ls="--", color="k", label="Pareto front")]
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 9), fontsize=7, frameon=False)
    plt.tight_layout(rect=(0, 0.06 + 0.035 * ((len(h) - 1) // 9), 1, 1))
    p = os.path.join(PAPER_DIR, f"fig_pareto_{split}.png")
    fig.savefig(p, dpi=300)
    fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    return p


def fig_size_recall(df, split="grouped", models=None):
    df = indomain(df)
    plt = _plt()
    d = df[df.split == split]
    buckets = [b for b, _, _ in cfg.BUCKETS]
    models = models or paper_models(set(d.model)) or [m for m in MODEL_ORDER if m in set(d.model)]
    MODES = cfg.modes_for(split)
    if not models:
        return None
    fig, axes = plt.subplots(1, len(buckets), figsize=(4 * len(buckets), 3.6), sharey=True)
    w = 0.8 / len(MODES)
    for ax, b in zip(np.atleast_1d(axes), buckets):
        for k, mode in enumerate(MODES):
            vals = [d[(d.model == m) & (d["mode"] == mode)][f"op_R_{b}"].mean() for m in models]
            errs = [d[(d.model == m) & (d["mode"] == mode)][f"op_R_{b}"].std(ddof=1) for m in models]
            ax.bar(np.arange(len(models)) + (k - (len(MODES) - 1) / 2) * w, vals, w,
                   yerr=np.nan_to_num(errs), capsize=2, label=MODE_LABEL[mode])
        n = int(d[f"op_n_gt_{b}"].iloc[0]) if f"op_n_gt_{b}" in d else 0
        lab = {"micro": r"micro ($<16^2$)", "tiny16_32": r"small ($16^2$–$32^2$)",
               "medium": r"medium ($32^2$–$96^2$)", "large": r"large ($\geq 96^2$)"}[b]
        ax.set_title(f"{lab} px$^2$, n={n}")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel(f"Recall (conf $\\geq$ {cfg.OP_CONF}, IoU $\\geq$ {cfg.MATCH_IOU})")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    plt.tight_layout()
    p = os.path.join(PAPER_DIR, f"fig_size_recall_{split}.png")
    fig.savefig(p, dpi=300)
    fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    return p


def fig_extra_dets(df, split="grouped", models=None):
    """Where do the extra SAHI boxes go? extra TP vs extra FP (split into duplicates / others)."""
    df = indomain(df)
    plt = _plt()
    d = df[df.split == split]
    std = d[d["mode"] == "standard"].set_index("run_id")
    models = models or paper_models(set(d.model)) or [m for m in MODEL_ORDER if m in set(d.model)]
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.8))
    labels, tp, fpd, fpo = [], [], [], []
    for m in models:
        for mode in cfg.modes_for(split)[1:]:
            g = d[(d.model == m) & (d["mode"] == mode)]
            if g.empty:
                continue
            b = std.loc[g.run_id]
            labels.append(f"{m}\n{MODE_LABEL[mode]}")
            tp.append(np.mean(g.op_TP.values - b.op_TP.values))
            dd = np.mean(g.op_FP_duplicate.values - b.op_FP_duplicate.values)
            fpd.append(dd)
            fpo.append(np.mean(g.op_FP.values - b.op_FP.values) - dd)
    x = np.arange(len(labels))
    ax.bar(x, tp, color="#55A868", label="extra true positives")
    ax.bar(x, fpd, bottom=np.maximum(tp, 0), color="#C44E52", label="extra duplicate boxes (FP)")
    ax.bar(x, fpo, bottom=np.maximum(tp, 0) + np.maximum(fpd, 0), color="#DD8452", label="extra other FP")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(f"change vs standard (boxes on the test set, conf $\\geq$ {cfg.OP_CONF})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(PAPER_DIR, f"fig_extra_detections_{split}.png")
    fig.savefig(p, dpi=300)
    fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    return p


def fig_qualitative(df, split_info, model=None, split="grouped", modes=("standard", "sahi160"), n_images=2):
    """TP (green) / FP (red) / missed GT (yellow) for the test images with the most micro objects."""
    import cv2
    plt = _plt()
    d = indomain(df)[lambda x: x.split == split]
    if d.empty:
        return None
    if model is None:   # model with the best standard AP
        model = d[d["mode"] == "standard"].groupby("model", observed=True).AP_all.mean().idxmax()
    row = d[(d.model == model) & (d["mode"] == modes[0])].sort_values("seed", key=lambda s: s.astype(str)).iloc[0]
    folder = row.folder
    gt = utils.load_json(split_info["gt"]["test"])
    per = pd.read_csv(os.path.join(folder, f"perimg_{modes[0]}.csv")).sort_values("gt_micro", ascending=False)
    pick = per.image_id.head(n_images).tolist()
    ims = {i["id"]: i for i in gt["images"]}
    fig, axes = plt.subplots(n_images, len(modes), figsize=(5.2 * len(modes), 5.2 * n_images))
    axes = np.atleast_2d(axes)
    for r, iid in enumerate(pick):
        g = [a for a in gt["annotations"] if a["image_id"] == iid]
        for c, mode in enumerate(modes):
            dets = [x for x in utils.load_any_json(utils.dets_path(folder, mode))
                    if x["image_id"] == iid and x["score"] >= cfg.OP_CONF]
            sub = dict(images=[ims[iid]], annotations=g, categories=gt["categories"])
            pim = metrics.match_per_image(sub, dets)
            img = cv2.cvtColor(cv2.imread(os.path.join(split_info["images"]["test"], ims[iid]["file_name"])),
                               cv2.COLOR_BGR2RGB)
            # recompute matches to colour boxes
            used, tpb = set(), []
            for cat in {x["category_id"] for x in g} | {x["category_id"] for x in dets}:
                gg = [a for a in g if a["category_id"] == cat]
                dd = sorted([x for x in dets if x["category_id"] == cat], key=lambda x: -x["score"])
                I = metrics._iou_xywh(np.array([x["bbox"] for x in dd], float).reshape(-1, 4),
                                      np.array([x["bbox"] for x in gg], float).reshape(-1, 4))
                u = np.zeros(len(gg), bool)
                for di, x in enumerate(dd):
                    cand = np.where(~u & (I[di] >= cfg.MATCH_IOU))[0] if len(gg) else []
                    if len(cand):
                        j = cand[np.argmax(I[di, cand])]
                        u[j] = True
                        tpb.append((x["bbox"], (40, 200, 40)))
                    else:
                        tpb.append((x["bbox"], (230, 40, 40)))
                for j, a in enumerate(gg):
                    if not u[j]:
                        tpb.append((a["bbox"], (255, 215, 0)))
            for (x, y, w, h), col in tpb:
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, 1)
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            t = pim.iloc[0]
            ax.set_title(f"{model} {MODE_LABEL[mode]}: TP {t.tp}  FP {t.fp}  missed {t.fn}", fontsize=9)
    fig.suptitle("green = correct detection, red = false positive, yellow = missed object", fontsize=9)
    plt.tight_layout()
    p = os.path.join(PAPER_DIR, f"fig_qualitative_{split}.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def table_transfer(df):
    """Cross-dataset test: models trained on one dataset, tested on the other (no fine-tuning),
    next to the in-domain result on the same target test set."""
    d = df[df["mode"] == "standard"]
    x = d[d.transfer & d.source.str.startswith("xfer")]
    if x.empty:
        return pd.DataFrame()
    rows = []
    for r in x.itertuples():
        same = d[(~d.transfer) & (d.split == r.split) & (d.model == r.model) & (d.seed == r.seed)]
        rows.append(dict(Model=r.model, Trained_on=r.source.replace("xfer-", "").split("2")[0], Tested_on=r.split,
                         AP50=r.AP50_all, AP=r.AP_all, AR50_small=r.AR50_small,
                         InDomain_AP50=(same.AP50_all.iloc[0] if len(same) else np.nan),
                         InDomain_AP=(same.AP_all.iloc[0] if len(same) else np.nan)))
    out = pd.DataFrame(rows)
    save(out, "T6_cross_dataset", "Cross-dataset generalisation: test-set COCO AP of a model trained on the other "
         "dataset (no fine-tuning) versus the model trained on the target dataset (InDomain).")
    return out


def table_splits_compare(df, splits, models, name, caption):
    """Same models on several training splits that share the SAME val/test images."""
    df = indomain(df)
    d = df[df.split.isin(splits) & (df["mode"] == "standard")]
    rows = []
    for m in models:
        for sp in splits:
            g = d[(d.model == m) & (d.split == sp)]
            if g.empty:
                continue
            rows.append(dict(Model=m, Train_split=sp, Seeds=len(g), mAP50=pm(g.u_mAP50), mAP50_95=pm(g.u_mAP50_95),
                             AP_small=pm(g.AP_small), AR50_micro=pm(g.AR50_micro), R=pm(g.op_R)))
    out = pd.DataFrame(rows)
    if not out.empty:
        save(out, name, caption)
    return out


def table_native(df):
    """Native-resolution experiments: best inference mode per (training setting, model) next to all modes."""
    df = indomain(df)
    d = df[df.split.str.startswith("native")]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for (sp, m, mode), g in d.groupby(["split", "model", "mode"], observed=True):
        rows.append(dict(Train=("640x640 tiles" if sp == "native_tiles" else f"full image, input {int(g.imgsz.iloc[0])}"),
                         Model=m, Inference=MODE_LABEL[mode], Seeds=len(g), AP=pm(g.AP_all), AP50=pm(g.AP50_all),
                         AP_small=pm(g.AP_small), AR50_small=pm(g.AR50_small), TP=pm(g.op_TP, 0), FP=pm(g.op_FP, 0),
                         ms_img=pm(g.ms, 1)))
    out = pd.DataFrame(rows)
    save(out, "T8_native_resolution",
         "Native-resolution screenshots (no resize): downscaled full-image input vs higher input size vs tile training, "
         "each with standard and SAHI inference (same test sources as the grouped split).")
    return out


def collect_ablation():
    files = [p for p in utils.find_files("sahi_ablation.csv", also_work=True)]
    if not files:
        return pd.DataFrame()
    a = pd.concat([pd.read_csv(p) for p in files])
    if "forced" not in a:
        a["forced"] = False
    a["forced"] = a.forced.fillna(False).astype(bool)
    # rows from runs before v2.1 (merge rule not forced) are replaced by forced rows of the same setting
    a = a.sort_values("forced").drop_duplicates(["run_id", "slice", "config"], keep="last")
    bad = a[~a.forced & a.postprocess.isin(["GREEDYNMM", "NMM"])]
    if len(bad):
        print(f"WARNING: {len(bad)} GREEDYNMM/NMM ablation rows were run without force_postprocess_type "
              "(SAHI silently used NMS-IoU) -> dropped. Re-run 08A/08B with sat-ext v2.1.")
        a = a.drop(bad.index)
    a["seed"] = a.run_id.map(seed_of)
    return a


def table_sahi_ablation(abl, split="grouped"):
    a = abl[abl.split == split] if len(abl) else abl
    if a.empty:
        return pd.DataFrame()
    rows = []
    for (s, c), g in a.groupby(["slice", "config"]):
        rows.append(dict(Slice=s, Setting=c, Models=g.model.nunique(), Runs=len(g), Passes=round(g.passes.mean(), 1),
                         AP=pm(g.AP_all), AP50=pm(g.AP50_all), AP_small=pm(g.AP_small), AR50_micro=pm(g.AR50_micro),
                         TP=pm(g.op_TP, 0), FP=pm(g.op_FP, 0), FP_duplicate=pm(g.op_FP_duplicate, 0), ms_img=pm(g.ms_per_image, 1)))
    out = pd.DataFrame(rows)
    save(out, f"T9_sahi_ablation_{split}",
         "SAHI post-processing ablation (one factor changed from the paper setting NMS / IoU 0.5 / overlap 0.2 / with full-image "
         f"pass, model confidence 0.01), mean $\\pm$ std over all models and seeds ({split} test split). Rows ending in \\_c25 use "
         "SAHI's own model confidence 0.25 (AP then truncated at 0.25; compare TP/FP and the two c25 rows with each other).")
    per_model = a.groupby(["model", "slice", "config"]).agg(AP=("AP_all", "mean"), FP=("op_FP", "mean"), TP=("op_TP", "mean")).reset_index()
    per_model.round(4).to_csv(os.path.join(PAPER_DIR, f"T9b_sahi_ablation_per_model_{split}.csv"), index=False)
    return out


def table_external(df):
    d = df[df.run_id.str.startswith("xdota")] if "run_id" in df else df.iloc[0:0]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for (m, mode), g in d.groupby(["model", "mode"], observed=True):
        rows.append(dict(Model=m, Inference=MODE_LABEL[mode], AP=g.AP_all.mean(), AP50=g.AP50_all.mean(),
                         AP_small=g.AP_small.mean(), AR50_small=g.AR50_small.mean(), TP=int(g.op_TP.mean()),
                         FP=int(g.op_FP.mean()), R=g.op_R.mean(), ms_img=g.ms.mean()))
    out = pd.DataFrame(rows).round(4)
    save(out, "T10_external_dota",
         "Zero-shot test on DOTA-v1.0 (plane, ship, small/large vehicle; horizontal boxes; difficult objects ignored): "
         "models trained on our grouped split (seed 42), no fine-tuning.")
    return out


def fig_learning_curve(df, models=("YOLO11s", "YOLO11m")):
    df = indomain(df)
    frac = {f"grouped_lc{int(f * 100)}": f for f in cfg.LEARNING_CURVE}
    frac["grouped"] = 1.0
    d = df[df.split.isin(frac) & (df["mode"] == "standard") & df.model.isin(models)].copy()
    if d.split.nunique() < 2:
        return None
    d["frac"] = d.split.map(frac)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for m in models:
        g = d[d.model == m].groupby("frac")
        for a, col, lab in [(ax[0], "AP_all", "COCO AP@[.5:.95]"), (ax[1], "AR50_micro", r"Recall@0.5, micro ($<16^2$)")]:
            mu, sd = g[col].mean(), g[col].std(ddof=1).fillna(0)
            a.errorbar(mu.index * 100, mu.values, yerr=sd.values, marker="o", capsize=3, label=m)
            a.set_xlabel("training groups used (%)")
            a.set_ylabel(lab)
    ax[0].legend()
    plt.tight_layout()
    p = os.path.join(PAPER_DIR, "fig_learning_curve.png")
    fig.savefig(p, dpi=300)
    fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    return p


def table_efficiency(df, split="grouped"):
    """T15: accuracy vs cost for every way of handling small objects: SAHI, TTA, inference-only resolution,
    and training at higher resolution (mean over seeds)."""
    d = indomain(df)
    d = d[d.split == split]
    if d.empty:
        return None
    rows = []
    for (m, mode), g in d.groupby(["model", "mode"], observed=True):
        if g.empty:
            continue
        rows.append(dict(Model=m, Train_size=int(g.imgsz.iloc[0]), Inference=mode_label(mode), Seeds=g.seed.nunique(),
                         AP=pm(g.AP_all), AP_small=pm(g.AP_small), AR50_micro=pm(g.AR50_micro),
                         Params_M=round(float(g.params_M.iloc[0]), 1) if pd.notna(g.params_M.iloc[0]) else None,
                         GFLOPs_img=round(float(g.gflops_img.mean()), 1) if g.gflops_img.notna().any() else None,
                         ms_img=pm(g.ms, 1), _ap=g.AP_all.mean()))
    t = pd.DataFrame(rows).sort_values(["Model", "_ap"], ascending=[True, False]).drop(columns="_ap")
    save(t, f"T15_efficiency_{split}", "Accuracy versus cost of the small-object strategies on the grouped test split "
         "(mean $\\pm$ s.d. over seeds; GFLOPs per image include every forward pass; latency end-to-end on a T4).")
    return t
