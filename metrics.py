"""Detection metrics.

coco_eval()        COCO-style AP / AP50 / recall per size range. Uses pycocotools when installed and
                   an internal re-implementation of the same algorithm otherwise (the notebooks
                   print both once so you can see they agree).
match_per_image()  greedy IoU>=0.5 matching at the operating confidence -> TP / FP / FN per image and
                   per size bucket. This answers "are the extra SAHI boxes real objects?".
bootstrap_delta()  image-level bootstrap CI for a difference between two inference modes.
"""
import io, contextlib, collections

import numpy as np
import pandas as pd

import config as cfg

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)


def _iou_xywh(d, g):
    """IoU matrix between det boxes d (n,4) and gt boxes g (m,4), xywh, no crowd."""
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.clip(np.minimum(dx2[:, None], gx2[None]) - np.maximum(d[:, None, 0], g[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(dy2[:, None], gy2[None]) - np.maximum(d[:, None, 1], g[None, :, 1]), 0, None)
    inter = iw * ih
    ua = (d[:, 2] * d[:, 3])[:, None] + (g[:, 2] * g[:, 3])[None] - inter
    return inter / np.maximum(ua, 1e-12)


def coco_eval_internal(gt, dets, area_ranges=None, max_dets=(10, 100, cfg.MAX_DET)):
    """Same algorithm as pycocotools COCOeval(bbox): evaluate() + accumulate().
    Returns precision[T,R,K,A,M] and recall[T,K,A,M] arrays."""
    area_ranges = area_ranges or cfg.AREA_RANGES
    cats = sorted(c["id"] for c in gt["categories"])
    img_ids = sorted(i["id"] for i in gt["images"])
    G, D = collections.defaultdict(list), collections.defaultdict(list)
    for a in gt["annotations"]:
        G[(a["image_id"], a["category_id"])].append(a)
    for d in dets:
        D[(d["image_id"], d["category_id"])].append(d)
    T, R, K, A, M = len(IOU_THRS), len(REC_THRS), len(cats), len(area_ranges), len(max_dets)
    precision = -np.ones((T, R, K, A, M))
    recall = -np.ones((T, K, A, M))
    maxDet = max_dets[-1]
    for k, c in enumerate(cats):
        # matching is done once per image (top maxDet detections), as in pycocotools
        per_img = []
        for i in img_ids:
            g, d = G.get((i, c), []), D.get((i, c), [])
            if not g and not d:
                continue
            d = sorted(d, key=lambda x: -x["score"])[:maxDet]      # stable sort == mergesort
            gb = np.array([x["bbox"] for x in g], float).reshape(-1, 4)
            db = np.array([x["bbox"] for x in d], float).reshape(-1, 4)
            per_img.append((g, d, gb, db, _iou_xywh(db, gb)))
        for a, (_, lo, hi) in enumerate(area_ranges):
            E = []
            for (g, d, gb, db, ious0) in per_img:
                g_ign = np.array([(x["area"] < lo or x["area"] > hi or bool(x.get("iscrowd", 0))) for x in g], bool)
                gorder = np.argsort(g_ign, kind="mergesort")
                g_ign = g_ign[gorder]
                ious = ious0[:, gorder] if len(g) else ious0
                nd, ng = len(d), len(g)
                gtm = np.zeros((T, ng), bool)
                dtm = np.zeros((T, nd), bool)
                dt_ign = np.zeros((T, nd), bool)
                for t, thr in enumerate(IOU_THRS):
                    thr = min(thr, 1 - 1e-10)
                    for di in range(nd if ng else 0):
                        # pycocotools: best unmatched non-ignored gt (ties -> last), else best ignored gt
                        m = -1
                        for pool in (~g_ign, g_ign):
                            ok = pool & ~gtm[t] & (ious[di] >= thr)
                            if ok.any():
                                v = np.where(ok, ious[di], -1.0)
                                m = len(v) - 1 - int(np.argmax(v[::-1]))
                                break
                        if m == -1:
                            continue
                        dt_ign[t, di] = g_ign[m]
                        dtm[t, di] = True
                        gtm[t, m] = True
                d_area = db[:, 2] * db[:, 3] if nd else np.zeros(0)
                out_rng = (d_area < lo) | (d_area > hi)
                dt_ign = dt_ign | ((~dtm) & out_rng[None, :])
                scores = np.array([x["score"] for x in d], float)
                E.append((scores, dtm, dt_ign, g_ign))
            if not E:
                continue
            for mi, md in enumerate(max_dets):
                sc = np.concatenate([e[0][:md] for e in E])
                order = np.argsort(-sc, kind="mergesort")
                dtm = np.concatenate([e[1][:, :md] for e in E], axis=1)[:, order]
                dti = np.concatenate([e[2][:, :md] for e in E], axis=1)[:, order]
                npig = int(sum((~e[3]).sum() for e in E))
                if npig == 0:
                    continue
                tps = np.cumsum(dtm & ~dti, axis=1).astype(float)
                fps = np.cumsum(~dtm & ~dti, axis=1).astype(float)
                for t in range(T):
                    tp, fp = tps[t], fps[t]
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    recall[t, k, a, mi] = rc[-1] if nd else 0
                    q = np.zeros(R)
                    pr = pr.tolist()
                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]
                    inds = np.searchsorted(rc, REC_THRS, side="left")
                    for ri, pi in enumerate(inds):
                        if pi < nd:
                            q[ri] = pr[pi]
                    precision[t, :, k, a, mi] = q
    return precision, recall


def coco_eval_pycocotools(gt, dets, area_ranges=None, max_dets=(10, 100, cfg.MAX_DET)):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    area_ranges = area_ranges or cfg.AREA_RANGES
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        cd = cg.loadRes(dets) if dets else None
        if cd is None:
            return None, None
        E = COCOeval(cg, cd, "bbox")
        E.params.maxDets = list(max_dets)
        E.params.areaRng = [[lo, hi] for _, lo, hi in area_ranges]
        E.params.areaRngLbl = [n for n, _, _ in area_ranges]
        E.evaluate()
        E.accumulate()
    return E.eval["precision"], E.eval["recall"]


def summarize(precision, recall, area_ranges=None, cats=None):
    area_ranges = area_ranges or cfg.AREA_RANGES
    out = {}

    def mean(v):
        v = v[v > -1]
        return float(v.mean()) if v.size else float("nan")

    if precision is None:
        return out
    for a, (name, _, _) in enumerate(area_ranges):
        out[f"AP_{name}"] = mean(precision[:, :, :, a, -1])
        out[f"AP50_{name}"] = mean(precision[0, :, :, a, -1])
        out[f"AP75_{name}"] = mean(precision[5, :, :, a, -1])
        out[f"AR_{name}"] = mean(recall[:, :, a, -1])
        out[f"AR50_{name}"] = mean(recall[0, :, a, -1])
    if cats:
        for k, cname in enumerate(cats):
            out[f"AP50_cls_{cname}"] = mean(precision[0, :, k, 0, -1])
            out[f"AP_cls_{cname}"] = mean(precision[:, :, k, 0, -1])
    return out


def coco_eval(gt, dets, prefer="pycocotools"):
    """dict of COCO-style metrics; `engine` says which implementation produced them."""
    cats = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    if not dets:
        return dict(engine="none", n_dets=0)
    if prefer == "pycocotools":
        try:
            P, R = coco_eval_pycocotools(gt, dets)
            return dict(engine="pycocotools", n_dets=len(dets), **summarize(P, R, cats=cats))
        except ImportError:
            pass
    P, R = coco_eval_internal(gt, dets)
    return dict(engine="internal", n_dets=len(dets), **summarize(P, R, cats=cats))


# ---------------------------------------------------------------- operating-point matching
def _bucket(area):
    for name, lo, hi in cfg.BUCKETS:
        if lo <= area < hi:
            return name
    return cfg.BUCKETS[-1][0]


def match_detail(gt, dets, conf=cfg.OP_CONF, thr=cfg.MATCH_IOU):
    """Class-aware greedy matching (highest score first).
    Returns (per-image DataFrame, {annotation id: hit bool}). Crowd / 'difficult' GT (iscrowd=1) are
    ignored: they are not counted as objects, and detections on them are neither TP nor FP."""
    names = [b for b, _, _ in cfg.BUCKETS]
    G, C = collections.defaultdict(list), collections.defaultdict(list)
    for a in gt["annotations"]:
        (C if a.get("iscrowd", 0) else G)[(a["image_id"], a["category_id"])].append(a)
    D = collections.defaultdict(list)
    for d in dets:
        if d["score"] >= conf:
            D[(d["image_id"], d["category_id"])].append(d)
    rows = {i["id"]: dict(image_id=i["id"], file_name=i["file_name"], n_gt=0, n_det=0, tp=0, fp=0,
                          dup_fp=0, **{f"gt_{b}": 0 for b in names}, **{f"tp_{b}": 0 for b in names})
            for i in gt["images"]}
    hits = {}
    for a in gt["annotations"]:
        if a.get("iscrowd", 0):
            continue
        r = rows[a["image_id"]]
        r["n_gt"] += 1
        r[f"gt_{_bucket(a['area'])}"] += 1
        hits[a["id"]] = False
    for key in set(G) | set(D):
        r = rows[key[0]]
        g, d = G.get(key, []), sorted(D.get(key, []), key=lambda x: -x["score"])
        if not d:
            continue
        db = np.array([x["bbox"] for x in d], float)
        ign = np.zeros(len(d), bool)
        if C.get(key):
            Ic = _iou_xywh(db, np.array([x["bbox"] for x in C[key]], float))
            ign = (Ic >= thr).any(1)
        gb = np.array([x["bbox"] for x in g], float).reshape(-1, 4)
        I = _iou_xywh(db, gb)
        used = np.zeros(len(g), bool)
        for di in range(len(d)):
            cand = np.where(~used & (I[di] >= thr))[0] if len(g) else []
            if len(cand):
                j = cand[np.argmax(I[di, cand])]
                used[j] = True
                r["n_det"] += 1
                r["tp"] += 1
                r[f"tp_{_bucket(g[j]['area'])}"] += 1
                hits[g[j]["id"]] = True
            elif ign[di]:
                continue                               # on an ignored (difficult) object
            else:
                r["n_det"] += 1
                r["fp"] += 1
                if len(g) and (I[di] >= thr).any():  # overlaps an object that was already matched
                    r["dup_fp"] += 1
    df = pd.DataFrame(list(rows.values()))
    df["fn"] = df.n_gt - df.tp
    return df, hits


def match_per_image(gt, dets, conf=cfg.OP_CONF, thr=cfg.MATCH_IOU):
    return match_detail(gt, dets, conf, thr)[0]


def gt_hits(gt, dets, conf=cfg.OP_CONF, thr=cfg.MATCH_IOU):
    return match_detail(gt, dets, conf, thr)[1]


def totals(perimg):
    t = perimg.drop(columns=["image_id", "file_name"]).sum()
    out = dict(dets=int(t.n_det), TP=int(t.tp), FP=int(t.fp), FP_duplicate=int(t.dup_fp), FN=int(t.fn),
               n_gt=int(t.n_gt), P=float(t.tp / max(t.n_det, 1)), R=float(t.tp / max(t.n_gt, 1)))
    for b, _, _ in cfg.BUCKETS:
        out[f"R_{b}"] = float(t[f"tp_{b}"] / t[f"gt_{b}"]) if t[f"gt_{b}"] else float("nan")
        out[f"n_gt_{b}"] = int(t[f"gt_{b}"])
    return out


def bootstrap_delta(a, b, stat, iters=cfg.BOOTSTRAP_ITERS, seed=cfg.SEED):
    """a, b: per-image DataFrames for the same images (two models or two modes).
    stat: (numerator column, denominator column) -> ratio of sums, or a callable(df)->float.
    Images are resampled (paired). Returns (delta, lo95, hi95, p_two_sided)."""
    a = a.set_index("image_id").sort_index()
    b = b.set_index("image_id").sort_index()
    assert (a.index == b.index).all(), "the two tables must contain the same images"
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, (iters, n))
    if isinstance(stat, tuple):
        num, den = stat
        f = lambda df, ix: df[num].to_numpy()[ix].sum(-1) / np.maximum(df[den].to_numpy()[ix].sum(-1), 1e-12)
        obs = float(f(b, np.arange(n)) - f(a, np.arange(n)))
        deltas = f(b, idx) - f(a, idx)
        if a[den].sum() == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
    else:
        obs = stat(b) - stat(a)
        deltas = np.array([stat(b.iloc[ix]) - stat(a.iloc[ix]) for ix in idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(p, 1.0))


def recall_stat(bucket=None):
    return ("tp", "n_gt") if bucket is None else (f"tp_{bucket}", f"gt_{bucket}")


precision_stat = ("tp", "n_det")


def mcnemar(hits_a, hits_b):
    """Paired test on the same objects (dict ann id -> bool). Exact two-sided binomial p-value.
    Objects in one image are not independent, so report it next to the image-level bootstrap."""
    from scipy.stats import binomtest
    keys = sorted(set(hits_a) & set(hits_b))
    a = np.array([hits_a[k] for k in keys])
    b = np.array([hits_b[k] for k in keys])
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(objects=len(keys), only_a=n10, only_b=n01, both=int((a & b).sum()), neither=int((~a & ~b).sum()),
                p_exact=float(p))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj
