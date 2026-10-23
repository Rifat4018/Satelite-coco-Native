"""CPU analyses on saved detections (no GPU): TIDE-style error breakdown, object-level McNemar with
image-level bootstrap, Weighted Box Fusion ensembles, dataset statistics, annotation agreement."""
import os, collections, itertools

import numpy as np
import pandas as pd

import config as cfg
import utils
import metrics
import report

OUT = os.path.join(cfg.RESULTS_DIR, "paper")


def _gt_for(split):
    import dataset as ds
    return utils.load_json(os.path.join(ds.find_split_dir(split), "coco", "test.json"))


def _dets(folder, mode):
    p = utils.dets_path(folder, mode)
    return utils.load_any_json(p) if p else None


# ================================================================ TIDE-style error breakdown
ERR = ["TP", "duplicate", "classification", "localization", "cls+loc", "background"]


def error_breakdown(gt, dets, conf=cfg.OP_CONF, fg=cfg.MATCH_IOU, bg=0.1):
    """Each detection (conf >= conf, highest score first) gets ONE label, following the TIDE definitions
    (Bolya et al., ECCV 2020): TP; duplicate (IoU>=fg with an object of its class that is already
    detected); classification (IoU>=fg with an object of another class); localization (bg<=IoU<fg with
    an object of its class); cls+loc (bg<=IoU<fg with another class); background (IoU<bg with all).
    Objects never detected are 'missed' (by size bucket)."""
    by_img = collections.defaultdict(list)
    for a in gt["annotations"]:
        if not a.get("iscrowd", 0):
            by_img[a["image_id"]].append(a)
    D = collections.defaultdict(list)
    for d in dets:
        if d["score"] >= conf:
            D[d["image_id"]].append(d)
    cnt = collections.Counter()
    missed = collections.Counter()
    for im in gt["images"]:
        g = by_img.get(im["id"], [])
        d = sorted(D.get(im["id"], []), key=lambda x: -x["score"])
        gb = np.array([x["bbox"] for x in g], float).reshape(-1, 4)
        gc = np.array([x["category_id"] for x in g])
        used = np.zeros(len(g), bool)
        for x in d:
            iou = metrics._iou_xywh(np.array([x["bbox"]], float), gb)[0] if len(g) else np.zeros(0)
            same = gc == x["category_id"] if len(g) else np.zeros(0, bool)
            cand = np.where(same & ~used & (iou >= fg))[0]
            if len(cand):
                used[cand[np.argmax(iou[cand])]] = True
                cnt["TP"] += 1
            elif len(g) and (same & (iou >= fg)).any():
                cnt["duplicate"] += 1
            elif len(g) and (~same & (iou >= fg)).any():
                cnt["classification"] += 1
            elif len(g) and (same & (iou >= bg)).any():
                cnt["localization"] += 1
            elif len(g) and (~same & (iou >= bg)).any():
                cnt["cls+loc"] += 1
            else:
                cnt["background"] += 1
        for j in np.where(~used)[0]:
            missed[metrics._bucket(g[j]["area"])] += 1
    out = {k: cnt[k] for k in ERR}
    out["detections"] = sum(cnt.values())
    out["missed"] = sum(missed.values())
    for b, _, _ in cfg.BUCKETS:
        out[f"missed_{b}"] = missed[b]
    return out


def table_errors(df, split="grouped", models=None, modes=None):
    gt = _gt_for(split)
    d = df[(df.split == split) & ~df.transfer]
    models = models or [m for m in cfg.MODELS if cfg.MODELS[m].get("group") == "paper" and m in set(d.model)]
    modes = modes or cfg.modes_for(split)
    rows = []
    for m in models:
        for mode in modes:
            g = d[(d.model == m) & (d["mode"] == mode)]
            recs = []
            for folder in g.folder:
                dets = _dets(folder, mode)
                if dets is not None:
                    recs.append(error_breakdown(gt, dets))
            if not recs:
                continue
            avg = pd.DataFrame(recs).mean()
            row = dict(Model=m, Inference=report.mode_label(mode), Seeds=len(recs))
            row.update({k: round(float(avg[k]), 1) for k in ERR + ["missed"] + [f"missed_{b}" for b, _, _ in cfg.BUCKETS]})
            fp = sum(avg[k] for k in ERR[1:])
            for k in ERR[1:]:
                row[f"{k}_%FP"] = round(100 * avg[k] / max(fp, 1e-9), 1)
            rows.append(row)
    out = pd.DataFrame(rows)
    report.save(out, f"T11_errors_{split}",
                f"Error breakdown at conf $\\geq$ {cfg.OP_CONF} (TIDE definitions, fg IoU {cfg.MATCH_IOU}, bg IoU 0.1), "
                "mean over seeds. %FP = share of all false positives.")
    return out


def fig_errors(err, split="grouped"):
    plt = report._plt()
    if err.empty:
        return None
    lab = [f"{r.Model}\n{r.Inference}" for r in err.itertuples()]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(err)), 4))
    bottom = np.zeros(len(err))
    for k, c in zip(ERR[1:], ["#C44E52", "#8172B3", "#DD8452", "#937860", "#4C72B0"]):
        ax.bar(range(len(err)), err[k], bottom=bottom, label=k, color=c)
        bottom += err[k].to_numpy()
    ax.set_xticks(range(len(err)))
    ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_ylabel(f"false positives on the test set (conf $\\geq$ {cfg.OP_CONF})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(OUT, f"fig_errors_{split}.png")
    fig.savefig(p, dpi=300)
    fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    return p


# ================================================================ McNemar + paired bootstrap between models
def _seed_hits(gt, folders, mode):
    """Object hit = detected in the majority of seeds."""
    hs = [metrics.gt_hits(gt, _dets(f, mode)) for f in folders if _dets(f, mode) is not None]
    if not hs:
        return None, 0
    keys = hs[0].keys()
    return {k: sum(h[k] for h in hs) * 2 > len(hs) for k in keys}, len(hs)


def _seed_perimg(folders, mode):
    t = [pd.read_csv(os.path.join(f, f"perimg_{mode}.csv")) for f in folders
         if os.path.isfile(os.path.join(f, f"perimg_{mode}.csv"))]
    return pd.concat(t).groupby("image_id").mean(numeric_only=True).reset_index() if t else None


def table_mcnemar(df, split="grouped", mode="standard", models=None):
    gt = _gt_for(split)
    d = df[(df.split == split) & (df["mode"] == mode) & ~df.transfer]
    models = models or [m for m in cfg.MODELS if m in set(d.model)]
    H, P = {}, {}
    for m in models:
        folders = list(d[d.model == m].folder)
        H[m], n = _seed_hits(gt, folders, mode)
        P[m] = _seed_perimg(folders, mode)
    rows = []
    for a, b in itertools.combinations([m for m in models if H.get(m)], 2):
        r = metrics.mcnemar(H[a], H[b])
        dl, lo, hi, pb = metrics.bootstrap_delta(P[a], P[b], metrics.recall_stat())
        rows.append(dict(A=a, B=b, Objects=r["objects"], Only_A=r["only_a"], Only_B=r["only_b"],
                         p_McNemar=r["p_exact"], Recall_B_minus_A=round(dl, 4), CI95_image_bootstrap=f"[{lo:+.4f}, {hi:+.4f}]",
                         p_bootstrap=pb))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_McNemar_Holm"] = metrics.holm(out.p_McNemar)
        out["p_bootstrap_Holm"] = metrics.holm(out.p_bootstrap)
        for c in ["p_McNemar", "p_McNemar_Holm", "p_bootstrap", "p_bootstrap_Holm"]:
            out[c] = out[c].map(lambda v: float(f"{v:.3g}"))
    report.save(out, f"T12_mcnemar_{split}_{mode}",
                f"Pairwise comparison of models ({split} test split, {report.mode_label(mode)} inference). McNemar: exact test "
                f"on objects detected (conf $\\geq$ {cfg.OP_CONF}, IoU $\\geq$ {cfg.MATCH_IOU}) by the majority of seeds; "
                "objects in one image are correlated, so the image-level paired bootstrap of the recall difference is "
                "the primary test. Holm correction over all pairs.")
    return out


def table_mcnemar_modes(df, split="grouped", models=None):
    """Each SAHI mode vs standard inference of the same model."""
    gt = _gt_for(split)
    d = df[(df.split == split) & ~df.transfer]
    models = models or [m for m in cfg.MODELS if cfg.MODELS[m].get("group") == "paper" and m in set(d.model)]
    rows = []
    for m in models:
        folders = list(d[(d.model == m) & (d["mode"] == "standard")].folder)
        hs, _ = _seed_hits(gt, folders, "standard")
        for mode in cfg.modes_for(split)[1:]:
            hm, n = _seed_hits(gt, folders, mode)
            if not hs or not hm:
                continue
            r = metrics.mcnemar(hs, hm)
            rows.append(dict(Model=m, Inference=report.mode_label(mode), Seeds=n, Only_standard=r["only_a"],
                             Only_SAHI=r["only_b"], p_McNemar=r["p_exact"]))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_Holm"] = [float(f"{v:.3g}") for v in metrics.holm(out.p_McNemar)]
        out["p_McNemar"] = out.p_McNemar.map(lambda v: float(f"{v:.3g}"))
    report.save(out, f"T12b_mcnemar_sahi_vs_standard_{split}",
                "Objects found only by standard inference vs only by SAHI (majority of seeds), exact McNemar, Holm-corrected.")
    return out


# ================================================================ Weighted Box Fusion ensembles
def _wbf_simple(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0):
    """Minimal Weighted Box Fusion (same algorithm as ensemble_boxes, conf_type='avg'); used only if the
    ensemble-boxes package cannot be installed."""
    N = len(boxes_list)
    w = weights or [1] * N
    allb = [(l, s * w[k], b) for k in range(N) for b, s, l in zip(boxes_list[k], scores_list[k], labels_list[k]) if s >= skip_box_thr]
    out_b, out_s, out_l = [], [], []
    for lab in sorted({a[0] for a in allb}):
        bx = sorted([a for a in allb if a[0] == lab], key=lambda a: -a[1])
        clusters, fused = [], []
        for _, sc, b in bx:
            b = np.asarray(b, float)
            j = -1
            if fused:
                F = np.array([f[:4] for f in fused])
                xx1, yy1 = np.maximum(F[:, 0], b[0]), np.maximum(F[:, 1], b[1])
                xx2, yy2 = np.minimum(F[:, 2], b[2]), np.minimum(F[:, 3], b[3])
                inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
                iou = inter / ((F[:, 2] - F[:, 0]) * (F[:, 3] - F[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-12)
                if iou.max() > iou_thr:
                    j = int(iou.argmax())
            if j < 0:
                clusters.append([(sc, b)])
                fused.append(np.r_[b, sc])
            else:
                clusters[j].append((sc, b))
                ss = np.array([c[0] for c in clusters[j]])
                bb = np.array([c[1] for c in clusters[j]])
                fused[j] = np.r_[(bb * ss[:, None]).sum(0) / ss.sum(), ss.mean()]
        for c, f in zip(clusters, fused):
            out_b.append(f[:4])
            out_s.append(f[4] * min(len(c), N) / N)
            out_l.append(lab)
    return np.array(out_b).reshape(-1, 4), np.array(out_s), np.array(out_l)


def _wbf(dets_list, gt, weights=None, iou_thr=0.55, top=500):
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        print("ensemble-boxes not installed -> built-in WBF (same algorithm)")
        weighted_boxes_fusion = _wbf_simple
    size = {i["id"]: (i["width"], i["height"]) for i in gt["images"]}
    per = [collections.defaultdict(list) for _ in dets_list]
    for k, dl in enumerate(dets_list):
        for d in dl:
            per[k][d["image_id"]].append(d)
    out = []
    for iid, (W, H) in size.items():
        B, S, L = [], [], []
        for k in range(len(dets_list)):
            dd = sorted(per[k].get(iid, []), key=lambda x: -x["score"])[:top]
            B.append([[x["bbox"][0] / W, x["bbox"][1] / H, (x["bbox"][0] + x["bbox"][2]) / W, (x["bbox"][1] + x["bbox"][3]) / H]
                      for x in dd] or np.zeros((0, 4)))
            S.append([x["score"] for x in dd])
            L.append([x["category_id"] for x in dd])
        if not any(len(s) for s in S):
            continue
        b, s, l = weighted_boxes_fusion([np.clip(np.asarray(x, float).reshape(-1, 4), 0, 1) for x in B], S, L,
                                        weights=weights, iou_thr=iou_thr, skip_box_thr=cfg.PRED_CONF)
        for (x1, y1, x2, y2), sc, lab in zip(b, s, l):
            out.append(dict(image_id=iid, category_id=int(lab), bbox=[x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H],
                            score=float(sc)))
    return out


def _score(name, members, gt, dets):
    c = metrics.coco_eval(gt, dets)
    o = metrics.totals(metrics.match_per_image(gt, dets))
    return dict(Ensemble=name, Members=members, AP=c.get("AP_all"), AP50=c.get("AP50_all"), AP_small=c.get("AP_small"),
                AR50_micro=c.get("AR50_micro"), TP=o["TP"], FP=o["FP"], R=o["R"], P=o["P"])


def table_ensembles(df, split="grouped"):
    gt = _gt_for(split)
    d = df[(df.split == split) & ~df.transfer]
    std = d[d["mode"] == "standard"]
    rows = []
    # (1) seed ensembles of each paper model
    for m in [k for k in cfg.MODELS if cfg.MODELS[k].get("group") == "paper"]:
        g = std[std.model == m]
        dl = [_dets(f, "standard") for f in g.folder]
        dl = [x for x in dl if x is not None]
        if len(dl) < 2:
            continue
        singles = [_score(f"{m} seed", "1", gt, x) for x in dl]
        rows.append(dict(_score(f"{m} single (mean of seeds)", "1", gt, dl[0]),
                         **{k: float(np.mean([s[k] for s in singles])) for k in ["AP", "AP50", "AP_small", "AR50_micro", "TP", "FP", "R", "P"]}))
        rows.append(_score(f"{m} WBF of {len(dl)} seeds", f"{len(dl)} seeds", gt, _wbf(dl, gt)))
    # (2) cross-architecture, seed 42, top-3 and all paper models
    s42 = std[std.seed.astype(str) == "42"]
    ranked = s42.sort_values("AP_all", ascending=False)
    for name, mlist in [("top-3 architectures", list(ranked.model.astype(str).head(3))),
                        ("5 paper models", [m for m in cfg.MODELS if cfg.MODELS[m].get("group") == "paper"])]:
        dl = [_dets(s42[s42.model == m].folder.iloc[0], "standard") for m in mlist if (s42.model == m).any()]
        dl = [x for x in dl if x is not None]
        if len(dl) >= 2:
            rows.append(_score(f"WBF {name}", "+".join(mlist), gt, _wbf(dl, gt)))
    # (3) standard + SAHI fused by WBF instead of SAHI's own merge (seed 42, paper models)
    for m in [k for k in cfg.MODELS if cfg.MODELS[k].get("group") == "paper"]:
        g = d[(d.model == m) & (d.seed.astype(str) == "42")]
        if g.empty:
            continue
        f = g.folder.iloc[0]
        a, b = _dets(f, "standard"), _dets(f, "sahi320")
        if a is None or b is None:
            continue
        rows.append(_score(f"{m} SAHI-320 (paper merge)", "SAHI", gt, b))
        rows.append(_score(f"{m} WBF(standard, SAHI-320)", "std+SAHI", gt, _wbf([a, b], gt)))
    out = pd.DataFrame(rows)
    for c in ["AP", "AP50", "AP_small", "AR50_micro", "R", "P"]:
        if c in out:
            out[c] = out[c].astype(float).round(4)
    report.save(out, f"T13_ensembles_{split}",
                "Weighted Box Fusion (Solovyev et al., 2021; IoU 0.55) of seeds, of architectures, and of standard + "
                f"SAHI-320 predictions ({split} test split).")
    return out


# ================================================================ dataset statistics
def dataset_stats(split="orig"):
    """Statistics on the SOURCE images (one copy per source: val/test images + the least-rotated
    train copy), i.e. what the dataset contains, not how often the augmentation repeats it."""
    import dataset as ds
    plt = report._plt()
    sdir = ds.find_split_dir(split)
    man = pd.read_csv(os.path.join(sdir, "manifest.csv"), dtype={"source_id": str, "file_name": str, "key": str})
    aud = utils.find_files("05_images_groups.csv", os.path.dirname(sdir))
    bc = pd.read_csv(aud[0])[["key", "black_corners"]] if aud else man[["key"]].assign(black_corners=0.0)
    man = man[man.split.isin(["train", "val", "test"])].merge(bc, on="key", how="left")
    pick = man.sort_values(["source_id", "black_corners", "key"]).groupby("source_id").head(1)
    rows = []
    for sp in ("train", "val", "test"):
        gt = utils.load_json(os.path.join(sdir, "coco", f"{sp}.json"))
        ids = {i["id"]: i for i in gt["images"]}
        keep = set(pick[pick.split == sp].image_id)
        for a in gt["annotations"]:
            if a["image_id"] in keep:
                im = ids[a["image_id"]]
                x, y, w, h = a["bbox"]
                rows.append(dict(split=sp, image_id=a["image_id"], cls=cfg.CLASSES[a["category_id"] - 1],
                                 w=w, h=h, area=w * h, cx=(x + w / 2) / im["width"], cy=(y + h / 2) / im["height"]))
    b = pd.DataFrame(rows)
    b["side"] = np.sqrt(b.area)
    b["aspect"] = np.maximum(b.w, b.h) / np.maximum(np.minimum(b.w, b.h), 1e-6)
    b["bucket"] = b.area.map(metrics._bucket)
    n_img = len(pick)
    per_img = b.groupby("image_id").size().reindex(pick.image_id, fill_value=0)
    summary = dict(source_images=n_img, objects=len(b), objects_per_image_mean=round(per_img.mean(), 1),
                   objects_per_image_median=float(per_img.median()), objects_per_image_max=int(per_img.max()),
                   images_without_objects=int((per_img == 0).sum()),
                   **{f"objects_{c}": int((b.cls == c).sum()) for c in cfg.CLASSES},
                   **{f"share_{k}": round(float((b.bucket == k).mean()), 3) for k, _, _ in cfg.BUCKETS},
                   median_box_side_px=round(float(b.side.median()), 1))
    per_cls = b.groupby("cls").agg(objects=("area", "size"), median_side_px=("side", "median"),
                                   p10_side=("side", lambda s: s.quantile(.1)), p90_side=("side", lambda s: s.quantile(.9)),
                                   median_aspect=("aspect", "median")).round(1).reset_index()
    cooc = pd.DataFrame(0, index=cfg.CLASSES, columns=cfg.CLASSES)
    for _, g in b.groupby("image_id"):
        cs = sorted(set(g.cls))
        for c1 in cs:
            for c2 in cs:
                cooc.loc[c1, c2] += 1
    os.makedirs(OUT, exist_ok=True)
    pd.Series(summary).to_csv(os.path.join(OUT, "D1_dataset_summary.csv"))
    report.save(per_cls, "D2_dataset_per_class", "Object statistics per class on the source images (box side = sqrt(area), px on 640x640).")
    cooc.to_csv(os.path.join(OUT, "D3_class_cooccurrence_images.csv"))
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    ax[0].hist(per_img, bins=40, color="#4C72B0")
    ax[0].set_xlabel("objects per image")
    ax[0].set_ylabel("images")
    for c, col in zip(cfg.CLASSES, ["#C44E52", "#55A868", "#4C72B0"]):
        ax[1].hist(b[b.cls == c].side, bins=np.arange(0, 160, 3), alpha=.6, label=c, color=col)
    for t in (16, 32, 96):
        ax[1].axvline(t, ls="--", c="k", lw=.8)
    ax[1].set_xlabel(r"box side $\sqrt{area}$ (px)")
    ax[1].legend()
    h = ax[2].hist2d(b.cx, b.cy, bins=32, cmap="viridis")
    ax[2].invert_yaxis()
    ax[2].set_title("object centres (normalised)")
    fig.colorbar(h[3], ax=ax[2])
    ax[3].imshow(cooc.values, cmap="Blues")
    ax[3].set_xticks(range(3))
    ax[3].set_xticklabels(cfg.CLASSES)
    ax[3].set_yticks(range(3))
    ax[3].set_yticklabels(cfg.CLASSES)
    for i in range(3):
        for j in range(3):
            ax[3].text(j, i, int(cooc.values[i, j]), ha="center", va="center")
    ax[3].set_title("images containing both classes")
    plt.tight_layout()
    p = os.path.join(OUT, "fig_dataset_statistics.png")
    fig.savefig(p, dpi=300)
    fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    template = pd.DataFrame([
        dict(Dataset="Ours", Source="Google Earth screenshots", Images=n_img, Classes=3, Instances=len(b),
             Instances_per_image=summary["objects_per_image_mean"], Share_lt_32px=round(summary["share_micro"] + summary["share_tiny16_32"], 3),
             Image_size="640x640 (stretched)", Reference="this work"),
    ] + [dict(Dataset=n, Source="", Images="", Classes="", Instances="", Instances_per_image="", Share_lt_32px="",
              Image_size="", Reference="FILL IN from the cited paper") for n in
         ["DOTA-v1.0", "DIOR", "xView", "AI-TOD", "VisDrone2019-DET", "NWPU VHR-10"]])
    template.to_csv(os.path.join(OUT, "D4_dataset_comparison_TEMPLATE.csv"), index=False)
    print(pd.Series(summary).to_string())
    return summary, per_cls, p


# ================================================================ second-annotator agreement
def agreement(ref_gt, other, iou=cfg.MATCH_IOU):
    """ref_gt / other: COCO dicts of the same images (matched by Roboflow source id)."""
    import dataset as ds
    sid = lambda fn: ds.source_id(fn)
    ref_img = {sid(i["file_name"]): i for i in ref_gt["images"]}
    oth_img = {sid(i["file_name"]): i for i in other["images"]}
    common = sorted(set(ref_img) & set(oth_img))
    onames = {c["id"]: cfg.CLASS_ALIASES.get(c["name"].lower(), c["name"].lower()) for c in other["categories"]}
    rows, ious, conf_cls = [], [], collections.Counter()
    R = collections.defaultdict(list)
    for a in ref_gt["annotations"]:
        R[a["image_id"]].append(a)
    O = collections.defaultdict(list)
    for a in other["annotations"]:
        if onames.get(a["category_id"]) in cfg.CLASSES:
            O[a["image_id"]].append(a)
    stats = collections.Counter()
    for s in common:
        ri, oi = ref_img[s], oth_img[s]
        sx, sy = ri["width"] / oi["width"], ri["height"] / oi["height"]
        r = R[ri["id"]]
        o = [dict(cls=onames[a["category_id"]], bbox=[a["bbox"][0] * sx, a["bbox"][1] * sy, a["bbox"][2] * sx, a["bbox"][3] * sy])
             for a in O[oi["id"]]]
        rb = np.array([a["bbox"] for a in r], float).reshape(-1, 4)
        ob = np.array([a["bbox"] for a in o], float).reshape(-1, 4)
        I = metrics._iou_xywh(ob, rb)
        used = np.zeros(len(r), bool)
        for k in np.argsort(-I.max(1) if len(r) else np.zeros(len(o))):
            cand = np.where(~used & (I[k] >= iou))[0] if len(r) else []
            if len(cand):
                j = cand[np.argmax(I[k, cand])]
                used[j] = True
                rc = cfg.CLASSES[r[j]["category_id"] - 1]
                conf_cls[(rc, o[k]["cls"])] += 1
                ious.append(I[k, j])
                if rc == o[k]["cls"]:
                    stats[f"match_{rc}"] += 1
                    stats[f"match_{metrics._bucket(r[j]['area'])}"] += 1
                else:
                    stats["class_disagreement"] += 1
            else:
                stats[f"only_second_{o[k]['cls']}"] += 1
        for j in np.where(~used)[0]:
            stats[f"only_ref_{cfg.CLASSES[r[j]['category_id'] - 1]}"] += 1
        for a in r:
            stats[f"ref_{metrics._bucket(a['area'])}"] += 1
    for c in cfg.CLASSES:
        m, nr = stats[f"match_{c}"], stats[f"match_{c}"] + stats[f"only_ref_{c}"] + sum(v for (rc, oc), v in conf_cls.items() if rc == c and oc != c)
        no = stats[f"match_{c}"] + stats[f"only_second_{c}"] + sum(v for (rc, oc), v in conf_cls.items() if oc == c and rc != c)
        p_, r_ = m / max(no, 1), m / max(nr, 1)
        rows.append(dict(Class=c, Reference_boxes=nr, Second_boxes=no, Matched=m, Precision=round(p_, 3), Recall=round(r_, 3),
                         F1=round(2 * p_ * r_ / max(p_ + r_, 1e-9), 3)))
    tot = pd.DataFrame(rows)
    M, NR, NO = tot.Matched.sum(), tot.Reference_boxes.sum(), tot.Second_boxes.sum()
    P_, R_ = M / max(NO, 1), M / max(NR, 1)
    tot.loc[len(tot)] = dict(Class="all", Reference_boxes=NR, Second_boxes=NO, Matched=M, Precision=round(P_, 3),
                             Recall=round(R_, 3), F1=round(2 * P_ * R_ / max(P_ + R_, 1e-9), 3))
    size = pd.DataFrame([dict(Bucket=b, Reference=stats[f"ref_{b}"], Matched=stats[f"match_{b}"],
                              Agreement=round(stats[f"match_{b}"] / max(stats[f"ref_{b}"], 1), 3)) for b, _, _ in cfg.BUCKETS])
    extra = dict(images=len(common), mean_IoU_matched=round(float(np.mean(ious)), 3) if ious else None,
                 class_disagreements=stats["class_disagreement"])
    return tot, size, extra
