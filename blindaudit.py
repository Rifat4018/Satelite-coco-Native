"""Notebook 14: blinded annotation audit across BOTH inference modes, then re-score on the corrected labels.

Why: standard inference and SAHI may be hurt differently by missing labels. A check of one detector's top
false positives (notebook 11) cannot show that. Here the apparent false positives of standard inference AND
of SAHI are pooled, de-duplicated, shuffled and shown WITHOUT any mode or score, so the reviewer cannot know
which mode produced a box.

Step A  build_candidates() + make_sheets()   -> numbered review sheets (annotator A: all, annotator B: a random subset)
                                               + a PRIVATE key (audit_key.csv: which mode(s) produced each box).
Step B  (people) each annotator edits verdicts_<A|B>.csv: M = real object that is not labelled, P = box on only a
                part of a real object, N = nothing there (true false positive, default), U = cannot decide.
Step C  consensus() -> kappa, per-mode rates, corrected_gt() -> re-score every saved detection file
        (reuses relabel.write_corrected_splits / relabel.rescore_all).  Train labels are NOT changed.

Limit (state it in the paper): only boxes detected by the audited model(s) can be added, so objects that no
audited detector found stay unlabelled. Notebooks 13A/13B (full manual relabel of the 82 images) remove that
limit; the two are alternatives, use one of them for the paper numbers."""
import os, copy, math, collections

import numpy as np
import pandas as pd

import config as cfg
import utils
import metrics

AUDIT_MODES = ["standard", "sahi320"]            # add "sahi480" / "sahi160" to audit more SAHI settings
CODES = {"N": "true_false_positive", "M": "missing_label", "P": "partial_box", "U": "unsure"}


# ------------------------------------------------------------------ A. candidates
def background_fps(gt, dets, conf=cfg.OP_CONF, bg=0.1):
    """Detections >= conf that overlap NO ground-truth box (any class) with IoU >= bg (same rule as notebook 11)."""
    gb = collections.defaultdict(list)
    for a in gt["annotations"]:
        gb[a["image_id"]].append(a["bbox"])
    out = []
    for d in dets:
        if d["score"] < conf:
            continue
        g = np.array(gb.get(d["image_id"], []), float).reshape(-1, 4)
        iou = metrics._iou_xywh(np.array([d["bbox"]], float), g)[0] if len(g) else np.zeros(0)
        if not len(iou) or iou.max() < bg:
            out.append(d)
    return out


def build_candidates(gt, dets_by_mode, seed=cfg.SEED, merge_iou=0.5, max_per_mode=None):
    """One row per distinct apparent false positive (boxes of different modes with IoU >= merge_iou on the same
    image are ONE candidate). Rows are shuffled with `seed`; the id is the position after shuffling.
    max_per_mode=None reviews every apparent false positive (unbiased for every mode); a number draws a RANDOM
    sample of that size per mode (never 'top-N by score', which would favour whichever mode has fewer boxes)."""
    rng = np.random.default_rng(seed)
    names = {i["id"]: i for i in gt["images"]}
    cand, by_img = [], collections.defaultdict(list)
    for mode, dets in dets_by_mode.items():
        fps = background_fps(gt, dets)
        if max_per_mode and len(fps) > max_per_mode:
            fps = [fps[i] for i in sorted(rng.choice(len(fps), max_per_mode, replace=False))]
        for d in fps:
            hit = None
            for k in by_img[d["image_id"]]:
                if metrics._iou_xywh(np.array([d["bbox"]], float), np.array([cand[k]["bbox"]], float))[0][0] >= merge_iou:
                    hit = k
                    break
            if hit is None:
                cand.append(dict(image_id=d["image_id"], bbox=list(map(float, d["bbox"])), category_id=int(d["category_id"]),
                                 score=float(d["score"]), modes={mode}))
                by_img[d["image_id"]].append(len(cand) - 1)
            else:
                cand[hit]["modes"].add(mode)
                if d["score"] > cand[hit]["score"]:
                    cand[hit].update(bbox=list(map(float, d["bbox"])), category_id=int(d["category_id"]), score=float(d["score"]))
    order = rng.permutation(len(cand))
    rows = []
    for new_id, k in enumerate(order, 1):
        c = cand[k]
        x, y, w, h = c["bbox"]
        im = names[c["image_id"]]
        rows.append(dict(id=new_id, image_id=c["image_id"], file_name=im["file_name"], x=x, y=y, w=w, h=h,
                         category_id=c["category_id"], cls=cfg.CLASSES[c["category_id"] - 1], score=c["score"],
                         **{f"in_{m}": (m in c["modes"]) for m in dets_by_mode}))
    return pd.DataFrame(rows)


def second_subset(cands, frac=0.25, seed=cfg.SEED + 1):
    n = max(1, int(round(frac * len(cands))))
    return sorted(np.random.default_rng(seed).choice(cands["id"].values, n, replace=False).tolist())


def render_tile(im, row, cell=240):
    """Crop around the box (red rectangle), label '#id class' only - no score, no mode."""
    import cv2
    x, y, w, h = row.x, row.y, row.w, row.h
    r = max(48, 1.5 * max(w, h))
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1, y1 = int(min(im.shape[1], cx + r)), int(min(im.shape[0], cy + r))
    crop = im[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (int(x - x0), int(y - y0)), (int(x + w - x0), int(y + h - y0)), (0, 0, 255), 1)
    crop = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (0, 0), (cell, 24), (0, 0, 0), -1)
    cv2.putText(crop, f"#{int(row.id)}  {row.cls}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return crop


def make_sheets(cands, img_dir, out_dir, ids=None, per_page=20, cols=5, cell=240, order_seed=None):
    """Numbered review pages (PNG). ids=None -> all candidates, in id order; order_seed shuffles the page order
    (annotator B sees the same subset in a different order)."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    sub = cands if ids is None else cands[cands["id"].isin(ids)]
    sub = sub.sort_values("id")
    if order_seed is not None:
        sub = sub.iloc[np.random.default_rng(order_seed).permutation(len(sub))]
    cache, tiles, paths = {}, [], []
    for row in sub.itertuples():
        if row.file_name not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[row.file_name] = cv2.imread(os.path.join(img_dir, row.file_name))
        tiles.append((int(row.id), render_tile(cache[row.file_name], row, cell)))
    blank = np.zeros((cell, cell, 3), np.uint8)
    for p in range(0, len(tiles), per_page):
        chunk = tiles[p:p + per_page]
        t = [c for _, c in chunk] + [blank] * (per_page - len(chunk))
        page = np.vstack([np.hstack(t[r * cols:(r + 1) * cols]) for r in range(per_page // cols)])
        fn = os.path.join(out_dir, f"page_{p // per_page + 1:03d}_ids_{chunk[0][0]}-{chunk[-1][0]}.png")
        cv2.imwrite(fn, page)
        paths.append(fn)
    return paths


def write_templates(cands, out_dir, ids_b):
    os.makedirs(out_dir, exist_ok=True)
    # the verdict column starts EMPTY: a row nobody filled in is detected by read_verdicts (a default "N" would look
    # exactly like a real "not an object" answer)
    pd.DataFrame(dict(id=cands["id"], verdict="")).to_csv(os.path.join(out_dir, "verdicts_A.csv"), index=False)
    pd.DataFrame(dict(id=sorted(ids_b), verdict="")).to_csv(os.path.join(out_dir, "verdicts_B.csv"), index=False)


# ------------------------------------------------------------------ B/C. verdicts, agreement, rates
def read_verdicts(path, valid_ids):
    v = pd.read_csv(path)
    v = v[[c for c in v.columns if not str(c).startswith("Unnamed")]]          # stray empty column from a spreadsheet
    blank = v["verdict"].isna() | v["verdict"].astype(str).str.strip().isin(["", "NAN"])
    assert not blank.any(), (f"{os.path.basename(path)}: {int(blank.sum())} ids have no verdict yet "
                             f"(first: {v.loc[blank, 'id'].head(5).tolist()}). Every id needs one of M, P, N, U.")
    v["verdict"] = v["verdict"].astype(str).str.strip().str.upper()
    bad = sorted(set(v["verdict"]) - set(CODES))
    assert not bad, f"{os.path.basename(path)}: unknown verdict codes {bad} (use M, P, N or U)"
    assert v["id"].is_unique, f"{os.path.basename(path)}: duplicate ids"
    unknown = set(v["id"]) - set(valid_ids)
    assert not unknown, f"{os.path.basename(path)}: ids not in the key: {sorted(unknown)[:5]}"
    return v.set_index("id")["verdict"]


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def consensus(cands, va, vb=None):
    """Final verdict per candidate id: annotator A's verdict; where B also reviewed the id and the two differ the
    verdict becomes U (unsure, not added). Returns (final Series, agreement dict)."""
    missing = set(cands["id"]) - set(va.index)
    assert not missing, f"annotator A has no verdict for {len(missing)} ids"
    final = va.reindex(cands["id"]).copy()
    ag = dict(n_both=0, agree=float("nan"), kappa=float("nan"), kappa_missing_vs_rest=float("nan"))
    if vb is not None and len(vb):
        both = sorted(set(vb.index) & set(va.index))
        a, b = va.loc[both], vb.loc[both]
        dis = [i for i in both if va[i] != vb[i]]
        final.loc[dis] = "U"
        ag = dict(n_both=len(both), agree=float((a == b).mean()), kappa=cohen_kappa(a, b),
                  kappa_missing_vs_rest=cohen_kappa((a == "M").map({True: "M", False: "x"}), (b == "M").map({True: "M", False: "x"})))
    return final, ag


def integrity_check(va, vb=None):
    """Stop when the two annotator files cannot be independent work: B copied from A (or A from B), or a file that
    still holds untouched default answers. Raises AssertionError with the reason."""
    if vb is None or len(vb) == 0:
        return "no second annotator"
    both = sorted(set(va.index) & set(vb.index))
    assert len(both) == len(vb), "annotator B has ids that annotator A does not"
    agree = float((va.loc[both] == vb.loc[both]).mean())
    outside = va.drop(index=both)
    non_n_outside = int((outside != "N").sum())
    # all of A's non-N answers sitting exactly on B's ids = B's answers pasted into A (or the reverse)
    nn_all = set(va.index[va != "N"])
    assert not (len(nn_all) >= 20 and non_n_outside == 0 and (len(both) > 0 and set(va.loc[both].index[va.loc[both] != "N"]) == nn_all)), (
        f"A's {len(nn_all)} non-N answers are all on B's {len(both)} ids and A answered N for the other {len(outside)}: "
        "A's file looks like B's answers pasted in (or untouched defaults). Ask A to redo her file from the blank template, "
        "without seeing B's file.")
    assert not (len(both) >= 30 and agree >= 0.99), (
        f"A and B agree on {agree:.0%} of {len(both)} boxes - not plausible for independent work (typical kappa 0.4-0.8). "
        "Check that B did not see A's file.")
    assert not (len(both) >= 30 and (vb == "N").mean() == 0 and (va.drop(index=both) == "N").mean() > 0.3), (
        "B never used N although A used it often: check that B had N available and did not skip 'not an object' boxes.")
    return f"ok: agreement {agree:.0%} on {len(both)} boxes"


def model_agreement(vc, va, vb=None):
    """Agreement of an extra, MODEL-ASSISTED reviewer C with the human files (reported only; C never changes the labels)."""
    rows = []
    for name, v in (("A (human)", va), ("B (human)", vb)):
        if v is None or len(v) == 0:
            continue
        ids = sorted(set(v.index) & set(vc.index))
        a, c = v.loc[ids], vc.loc[ids]
        rows.append(dict(vs=name, n=len(ids), percent_agreement=round(100 * float((a == c).mean()), 1),
                         kappa_4_codes=round(cohen_kappa(a, c), 2),
                         kappa_missing_vs_rest=round(cohen_kappa((a == "M").map({True: "M", False: "x"}), (c == "M").map({True: "M", False: "x"})), 2)))
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mode_table(cands, final, modes):
    rows = []
    for m in modes:
        ids = cands.loc[cands[f"in_{m}"], "id"]
        v = final.loc[ids]
        n, k = len(v), int((v == "M").sum())
        lo, hi = wilson(k, n)
        rows.append(dict(mode=m, apparent_FP=n, missing_label=k, partial_box=int((v == "P").sum()),
                         true_FP=int((v == "N").sum()), unsure=int((v == "U").sum()),
                         missing_share=f"{100 * k / max(n, 1):.0f}%", ci95=f"[{100 * lo:.0f}, {100 * hi:.0f}]%"))
    if len(modes) >= 2:                                   # candidates found by only one mode
        for m in modes:
            only = cands[cands[f"in_{m}"] & ~cands[[f"in_{o}" for o in modes if o != m]].any(axis=1)]
            v = final.loc[only["id"]]
            n, k = len(v), int((v == "M").sum())
            lo, hi = wilson(k, n)
            rows.append(dict(mode=f"{m} only", apparent_FP=n, missing_label=k, partial_box=int((v == "P").sum()),
                             true_FP=int((v == "N").sum()), unsure=int((v == "U").sum()),
                             missing_share=f"{100 * k / max(n, 1):.0f}%", ci95=f"[{100 * lo:.0f}, {100 * hi:.0f}]%"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ corrected labels and re-scoring
def corrected_gt(gt_old, cands, final, include=("M",)):
    """Old GT + the audited boxes whose final verdict is in `include` (same image ids -> paired tests stay valid)."""
    g = copy.deepcopy(gt_old)
    size = {i["id"]: (i["width"], i["height"]) for i in g["images"]}
    nid = max([a["id"] for a in g["annotations"]] + [0]) + 1
    added = 0
    for r in cands.itertuples():
        if final.get(r.id) not in include:
            continue
        W, H = size[r.image_id]
        x1, y1 = max(0.0, r.x), max(0.0, r.y)
        x2, y2 = min(float(W), r.x + r.w), min(float(H), r.y + r.h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        g["annotations"].append(dict(id=nid, image_id=r.image_id, category_id=int(r.category_id), bbox=[x1, y1, x2 - x1, y2 - y1],
                                     area=(x2 - x1) * (y2 - y1), iscrowd=0))
        nid += 1
        added += 1
    g["info"] = dict(description=f"grouped/test - corrected by blinded audit (notebook 14), {added} boxes added")
    return g


def score_versions(gts, dets_by_mode, modes):
    """AP / P / R / F1 / FP per image of every mode on every GT version."""
    rows = []
    for vname, gt in gts.items():
        for m in modes:
            dets = dets_by_mode[m]
            co = metrics.coco_eval(gt, dets)
            op = metrics.totals(metrics.match_per_image(gt, dets))
            P, R = op["P"], op["R"]
            rows.append(dict(labels=vname, mode=m, n_gt=op["n_gt"], AP=co.get("AP_all"), AP_small=co.get("AP_small"),
                             P=P, R=R, F1=(2 * P * R / (P + R)) if P + R else 0.0, TP=op["TP"], FP=op["FP"],
                             FP_per_img=op["FP"] / max(len(gt["images"]), 1)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures for the paper
def fig_examples(cands, final, img_dir, out_png, n_per_group=4, cell=300):
    """Gallery of audited boxes confirmed as real but unlabelled (shown AFTER unblinding), one row per group."""
    import cv2
    modes = [c[3:] for c in cands.columns if c.startswith("in_")]
    groups = []
    if len(modes) >= 2:
        both = cands[cands[[f"in_{m}" for m in modes]].all(axis=1)]
        groups.append(("found by both modes", both))
        for m in modes:
            others = [f"in_{o}" for o in modes if o != m]
            groups.append((f"{m} only", cands[cands[f"in_{m}"] & ~cands[others].any(axis=1)]))
    else:
        groups.append((modes[0], cands))
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for title, g in groups:
        g = g[final.reindex(g["id"]).values == "M"]
        if not len(g):
            continue
        pick = g.iloc[rng.permutation(len(g))[:n_per_group]]
        tiles = []
        for r in pick.itertuples():
            im = cv2.imread(os.path.join(img_dir, r.file_name))
            t = render_tile(im, r, cell)
            cv2.rectangle(t, (0, 0), (cell, 24), (0, 0, 0), -1)
            cv2.putText(t, f"{title}: {r.cls}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            tiles.append(t)
        tiles += [np.zeros((cell, cell, 3), np.uint8)] * (n_per_group - len(tiles))
        rows.append(np.hstack(tiles))
    if rows:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        cv2.imwrite(out_png, np.vstack(rows))
    return out_png if rows else None


def fig_ap(table, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    versions = list(dict.fromkeys(table["labels"]))
    modes = list(dict.fromkeys(table["mode"]))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for a, (col, title) in zip(ax, [("AP", "COCO AP"), ("P", "Precision (conf $\\geq$ 0.25)")]):
        w = 0.8 / len(modes)
        for j, m in enumerate(modes):
            v = [float(table[(table["labels"] == ver) & (table["mode"] == m)][col].iloc[0]) for ver in versions]
            b = a.bar(np.arange(len(versions)) + j * w, v, w, label=m)
            a.bar_label(b, fmt="%.3f", fontsize=6, padding=1)
        a.set_xticks(np.arange(len(versions)) + w * (len(modes) - 1) / 2)
        a.set_xticklabels(versions, fontsize=7)
        a.set_title(title, fontsize=8)
        a.tick_params(labelsize=7)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png
