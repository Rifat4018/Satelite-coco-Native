"""Dataset audit: counts, box validity, exact duplicates (SHA-256), near-duplicates (pHash under all
8 flips/rotations), partial ground overlap between images (ORB features + RANSAC homography) and the
resulting groups of images that show the same ground area.

Evidence types that join two images into one group:
  source_id  - same Roboflow source name (augmented copies of one original)
  sha256     - byte-identical files
  phash      - visually near-identical (also when flipped / rotated by 90 degrees)
  orb        - the two images share ground area (overlapping or neighbouring tiles)
"""
import os, collections, multiprocessing as mp

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.fft import dct

import config as cfg
import utils
import dataset as ds


# ---------------------------------------------------------------- per-image features
def _phash_bits(gray32):
    """pHash exactly as imagehash.phash: 32x32 DCT, top-left 8x8, > median."""
    d = dct(dct(gray32.astype(np.float64), axis=0, norm=None), axis=1, norm=None)
    low = d[:8, :8]
    return (low > np.median(low)).flatten()


def dihedral_phashes(path):
    im = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    a = np.asarray(im)
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [_phash_bits(r), _phash_bits(r[:, ::-1])]
    return np.stack(out)            # (8, 64) bool; row 0 = identity


def black_corner_score(img):
    """Share of near-black pixels in the four 32x32 corners (rotated copies have black wedges)."""
    h, w = img.shape[:2]
    c = 32
    patches = [img[:c, :c], img[:c, w - c:], img[h - c:, :c], img[h - c:, w - c:]]
    return float(np.mean([np.mean(p.max(axis=-1) < 8) if p.ndim == 3 else np.mean(p < 8) for p in patches]))


def image_features(imgs):
    """sha256, pHash (8 orientations), black-corner score for every image."""
    sha, ph, black = [], [], []
    for p in imgs.path:
        sha.append(utils.sha256_file(p))
        ph.append(dihedral_phashes(p))
        im = cv2.imread(p)
        black.append(black_corner_score(im) if im is not None else 1.0)
    out = imgs.copy()
    out["sha256"] = sha
    out["black_corners"] = black
    return out, np.stack(ph)


# ---------------------------------------------------------------- counts + box validity
def split_counts(raw, imgs, cmap):
    rows = []
    for s, c in raw.items():
        WH = {im["id"]: (im["width"], im["height"]) for im in c["images"]}
        a = c["annotations"]
        part = imgs[imgs.src_split == s]
        cps = collections.Counter(collections.Counter(part.source_id).values())
        bx = np.array([x["bbox"] for x in a], float).reshape(-1, 4)
        wh = np.array([WH.get(x["image_id"], (np.nan, np.nan)) for x in a], float).reshape(-1, 2)
        tgt = [x["category_id"] in cmap[s] for x in a]
        names = {cat["id"]: cat["name"] for cat in c["categories"]}
        rows.append(dict(
            split=s, images=len(c["images"]), missing_files=int((~part.exists).sum()),
            unique_source_ids=part.source_id.nunique(),
            copies_per_source=dict(sorted(cps.items())),
            annotations=len(a), non_target_annotations=int(len(a) - sum(tgt)),
            empty_images=len(c["images"]) - len({x["image_id"] for x in a}),
            neg_x=int((bx[:, 0] < 0).sum()), neg_y=int((bx[:, 1] < 0).sum()),
            beyond_right=int((bx[:, 0] + bx[:, 2] > wh[:, 0] + 0.5).sum()),
            beyond_bottom=int((bx[:, 1] + bx[:, 3] > wh[:, 1] + 0.5).sum()),
            w_or_h_le_0=int(((bx[:, 2] <= 0) | (bx[:, 3] <= 0)).sum()),
            w_or_h_lt_2px=int(((bx[:, 2] < 2) | (bx[:, 3] < 2)).sum()),
            exact_dup_boxes=len(a) - len({(x["image_id"], x["category_id"], *[round(v, 1) for v in x["bbox"]]) for x in a}),
            resolutions=dict(collections.Counter(f"{w}x{h}" for w, h in zip(part.width, part.height))),
            **{f"n_{names[k]}": v for k, v in collections.Counter(x["category_id"] for x in a).items()},
        ))
    cats = {s: [(c_["id"], c_["name"]) for c_ in c["categories"]] for s, c in raw.items()}
    return pd.DataFrame(rows), cats


def box_sizes(boxes, imgs):
    rows = []
    sp = dict(zip(imgs.key, imgs.src_split))
    for key, bl in boxes.items():
        for (k, x1, y1, x2, y2) in bl:
            w, h = x2 - x1, y2 - y1
            rows.append(dict(key=key, split=sp[key], cls=cfg.CLASSES[k], w=w, h=h, area=w * h,
                             bucket=size_bucket(w * h)))
    return pd.DataFrame(rows)


def size_bucket(area):
    for name, lo, hi in cfg.BUCKETS:
        if lo <= area < hi:
            return name
    return cfg.BUCKETS[-1][0]


# ---------------------------------------------------------------- pHash pairs
def phash_pairs(feat, PH, max_d=cfg.PHASH_MAX_HAMMING):
    """All image pairs whose pHash distance (min over orientations) is <= max_d."""
    base = PH[:, 0, :]                                      # (N, 64)
    rows = []
    for i in range(len(feat)):
        d = (PH[i][:, None, :] != base[None, :, :]).sum(-1).min(0)    # (N,)
        d[: i + 1] = 999
        for j in np.where(d <= max_d)[0]:
            rows.append((i, int(j), int(d[j])))
    df = pd.DataFrame(rows, columns=["i", "j", "hamming"])
    return _label_pairs(df, feat)


def _label_pairs(df, feat):
    if df.empty:
        return df.assign(key_i=[], key_j=[], split_i=[], split_j=[], cross_split=[], same_source=[])
    df["key_i"], df["key_j"] = feat.key.values[df.i], feat.key.values[df.j]
    df["split_i"], df["split_j"] = feat.src_split.values[df.i], feat.src_split.values[df.j]
    df["cross_split"] = df.split_i != df.split_j
    df["same_source"] = feat.source_id.values[df.i] == feat.source_id.values[df.j]
    return df


# ---------------------------------------------------------------- ORB overlap between representatives
_F, _FF, _BF = None, None, None


def corner_mask(h, w, long=cfg.WATERMARK_LONG, short=cfg.WATERMARK_SHORT):
    """0 in the four corners (both orientations) where the 'Google Earth' watermark can sit after
    flips / 90-degree rotations; features there are ignored."""
    m = np.full((h, w), 255, np.uint8)
    for (lh, lw) in ((short, long), (long, short)):
        a, b = int(lh * h), int(lw * w)
        m[:a, :b] = 0; m[:a, w - b:] = 0; m[h - a:, :b] = 0; m[h - a:, w - b:] = 0
    return m


def _orb(gray):
    orb = cv2.ORB_create(cfg.ORB_FEATURES)
    kp, des = orb.detectAndCompute(gray, corner_mask(*gray.shape[:2]))
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return pts, des, float(gray.shape[0] * gray.shape[1])


def _init_worker():
    global _BF
    _BF = cv2.BFMatcher(cv2.NORM_HAMMING)


def _inliers(i, j):
    best = 0
    pa, da, area_a = _F[i]
    if da is None or len(da) < 2:
        return 0
    for pb, db, area_b in (_F[j], _FF[j]):
        if db is None or len(db) < 2:
            continue
        good = [m[0] for m in _BF.knnMatch(da, db, k=2)
                if len(m) == 2 and m[0].distance < cfg.ORB_RATIO * m[1].distance]
        if len(good) < cfg.ORB_MIN_GOOD:
            continue
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            continue
        det = abs(np.linalg.det(H[:2, :2]))
        if not (0.2 < det < 5.0):              # reject degenerate / absurd scale changes
            continue
        keep = mask.ravel().astype(bool)
        sa, sb = np.ptp(src[keep], axis=0), np.ptp(dst[keep], axis=0)
        if sa[0] * sa[1] < cfg.ORB_MIN_SPREAD * area_a or sb[0] * sb[1] < cfg.ORB_MIN_SPREAD * area_b:
            continue                           # clustered inliers = watermark / logo, not shared ground
        best = max(best, int(keep.sum()))
    return best


def _chunk(pairs):
    return [(i, j, _inliers(i, j)) for i, j in pairs]


def orb_pairs(rep_paths, workers=cfg.AUDIT_WORKERS):
    """ORB inliers for every pair of representative images (one per source group).
    Mirror images are handled by also matching against the horizontally flipped image."""
    global _F, _FF
    _F, _FF = [], []
    for p in rep_paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        _F.append(_orb(g))
        _FF.append(_orb(np.ascontiguousarray(g[:, ::-1])))
    n = len(rep_paths)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"ORB: {n} representative images, {len(pairs):,} pairs, {workers} workers")
    chunks = [pairs[k:k + 2000] for k in range(0, len(pairs), 2000)]
    out = []
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_chunk, chunks):
                out += [r for r in res if r[2] > 0]
    except Exception as e:                       # no fork (Windows / restricted) -> serial
        print("ORB parallel failed, running serially:", e)
        _init_worker()
        for c in chunks:
            out += [r for r in _chunk(c) if r[2] > 0]
    return pd.DataFrame(out, columns=["i", "j", "inliers"]).sort_values("inliers", ascending=False)


# ---------------------------------------------------------------- grouping
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def build_groups(feat, ph_pairs, orb_df, rep_index, use_orb=True):
    """Union-find over all evidence. Returns feat with a 'group' column + evidence counts."""
    n = len(feat)
    uf = _UF(n)
    ev = collections.Counter()
    first = {}
    for i, s in enumerate(feat.source_id):
        if s in first:
            uf.union(first[s], i)
            ev["source_id"] += 1
        else:
            first[s] = i
    first = {}
    for i, h in enumerate(feat.sha256):
        if h in first:
            uf.union(first[h], i)
            ev["sha256"] += 1
        else:
            first[h] = i
    for r in ph_pairs.itertuples():
        uf.union(r.i, r.j)
        ev["phash"] += 1
    if use_orb and orb_df is not None:
        for r in orb_df[orb_df.inliers >= cfg.ORB_MIN_INLIERS].itertuples():
            uf.union(rep_index[r.i], rep_index[r.j])
            ev["orb"] += 1
    roots = [uf.find(i) for i in range(n)]
    order = {r: g for g, r in enumerate(sorted(set(roots)))}
    out = feat.copy()
    out["group"] = [f"g{order[r]:04d}" for r in roots]
    return out, dict(ev)


def representatives(feat):
    """One image per source group: the one with the fewest black corners (least rotated)."""
    reps = (feat.assign(_o=np.arange(len(feat)))
            .sort_values(["source_id", "black_corners", "key"])
            .groupby("source_id", sort=True).head(1))
    return reps._o.to_numpy()


def leakage_summary(feat_g, ph_pairs, orb_df, rep_idx):
    """How the ORIGINAL split leaks: groups whose images sit in more than one split."""
    g = feat_g.groupby("group").src_split.nunique()
    leaking = set(g[g > 1].index)
    res = dict(groups_total=int(feat_g.group.nunique()), groups_spanning_splits=len(leaking),
               images_in_spanning_groups=int(feat_g.group.isin(leaking).sum()))
    shared = feat_g.groupby("source_id").src_split.nunique()
    res["source_ids_in_more_than_one_split"] = int((shared > 1).sum())
    sizes = feat_g.groupby("group").source_id.nunique().sort_values(ascending=False)
    res["largest_groups_sources"] = [int(v) for v in sizes.head(5)]
    res["largest_group_share_of_sources"] = float(sizes.iloc[0] / max(feat_g.source_id.nunique(), 1))
    res["phash_pairs_cross_split"] = int(ph_pairs.cross_split.sum()) if len(ph_pairs) else 0
    if orb_df is not None and len(orb_df):
        sp = feat_g.src_split.values
        o = orb_df[orb_df.inliers >= cfg.ORB_MIN_INLIERS]
        cross = [sp[rep_idx[r.i]] != sp[rep_idx[r.j]] for r in o.itertuples()]
        res["orb_overlap_pairs"] = int(len(o))
        res["orb_overlap_pairs_cross_split"] = int(np.sum(cross))
    for s in ("val", "test"):
        part = feat_g[feat_g.src_split == s]
        res[f"{s}_images_whose_group_has_train_images"] = int(
            part.group.isin(set(feat_g[feat_g.src_split == "train"].group)).sum())
    return res


# ---------------------------------------------------------------- figures
def pair_montage(feat, pairs, path, title, k=8, score_col="hamming"):
    if pairs is None or len(pairs) == 0:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = pairs.head(k)
    fig, ax = plt.subplots(len(top), 2, figsize=(6, 3 * len(top)))
    ax = np.atleast_2d(ax)
    for r, p in enumerate(top.itertuples()):
        for c, idx in enumerate((p.i, p.j)):
            row = feat.iloc[idx]
            ax[r, c].imshow(Image.open(row.path).convert("RGB"))
            ax[r, c].axis("off")
            ax[r, c].set_title(f"{row.src_split} | {row.source_id[:22]} | {score_col}={getattr(p, score_col)}",
                               fontsize=7)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def gt_grid(feat, boxes, path, split, n=6, seed=cfg.SEED):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    part = feat[feat.src_split == split]
    if part.empty:
        return None
    rows = part.sample(min(n, len(part)), random_state=seed)
    colors = {0: (255, 60, 60), 1: (60, 200, 60), 2: (60, 120, 255)}
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for a, r in zip(ax.flat, rows.itertuples()):
        im = cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB)
        cnt = collections.Counter()
        for (k, x1, y1, x2, y2) in boxes.get(r.key, []):
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), colors[k], 1)
            cnt[cfg.CLASSES[k]] += 1
        a.imshow(im)
        a.axis("off")
        a.set_title(" ".join(f"{c}:{cnt[c]}" for c in cfg.CLASSES), fontsize=9)
    for a in ax.flat[len(rows):]:
        a.axis("off")
    fig.suptitle(f"{split}: cleaned ground truth (red plane, green ship, blue vehicle)")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def size_figure(sizes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    side = np.sqrt(sizes.area.clip(lower=1))
    ax[0].hist(side, bins=np.arange(0, 200, 4), color="#4C72B0")
    for t, lab in [(16, r"$16^2$"), (32, r"$32^2$"), (96, r"$96^2$")]:
        ax[0].axvline(t, ls="--", c="k", lw=1)
        ax[0].text(t + 1, ax[0].get_ylim()[1] * 0.9, lab, fontsize=9)
    ax[0].set_xlabel(r"$\sqrt{\mathrm{box\ area}}$ (px, 640$\times$640 images)")
    ax[0].set_ylabel("boxes")
    ax[0].set_title("Object size (area thresholds 16², 32², 96² px²)")
    tab = sizes.groupby(["split", "bucket"]).size().unstack(fill_value=0)
    tab = tab[[b for b, _, _ in cfg.BUCKETS if b in tab.columns]]
    (tab.T / tab.sum(1)).T.plot.bar(stacked=True, ax=ax[1], rot=0)
    ax[1].set_ylabel("share of boxes")
    ax[1].set_title("Size buckets per split")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
