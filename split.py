"""Split builders.

orig    : your original Roboflow train/valid/test assignment, cleaned labels (reproduces the paper's data).
grouped : a leakage-free split. The unit is a GROUP (all images that show the same ground area:
          augmented copies, exact/near duplicates, overlapping tiles). Groups are assigned to
          train/val/test as a whole, so no ground area appears in two splits.
          train keeps every image of its groups (incl. the offline-augmented copies, as in the paper);
          val/test keep ONE image per original source (the least-rotated copy, preferring an image
          that was already in your original valid/test folder, which Roboflow does not augment).
"""
import collections

import numpy as np
import pandas as pd

import config as cfg


def orig_manifest(feat_g):
    return pd.DataFrame(dict(key=feat_g.key, split=feat_g.src_split, group=feat_g.group))


def _eval_images(feat_g):
    """For every (group, source_id): the single image used if that group lands in val/test."""
    f = feat_g.assign(_pref=(feat_g.src_split == "train").astype(int))
    pick = f.sort_values(["group", "source_id", "_pref", "black_corners", "key"]).groupby(
        ["group", "source_id"], sort=False).head(1)
    return set(pick.key)


def grouped_manifest(feat_g, boxes, fractions=None, trials=cfg.SPLIT_TRIALS, seed=cfg.SEED):
    fractions = fractions or cfg.SPLIT_FRACTIONS
    eval_keys = _eval_images(feat_g)
    groups = sorted(feat_g.group.unique())
    G = len(groups)
    gidx = {g: i for i, g in enumerate(groups)}
    K = len(cfg.CLASSES)
    cnt = np.zeros((G, K))           # class counts of the eval images of each group
    nimg = np.zeros(G)               # number of eval images of each group
    for r in feat_g.itertuples():
        if r.key in eval_keys:
            i = gidx[r.group]
            nimg[i] += 1
            for (k, *_) in boxes.get(r.key, []):
                cnt[i, k] += 1
    tot_c, tot_n = cnt.sum(0) + 1e-9, nimg.sum()
    n_test = max(1, int(round(fractions["test"] * G)))
    n_val = max(1, int(round(fractions["val"] * G)))
    rng = np.random.default_rng(seed)
    best, best_s = None, np.inf
    for _ in range(trials):
        perm = rng.permutation(G)
        te, va = perm[:n_test], perm[n_test:n_test + n_val]
        s = 0.0
        for idx, f in ((te, fractions["test"]), (va, fractions["val"])):
            s += (((cnt[idx].sum(0) / tot_c) - f) ** 2).sum() / f ** 2
            s += ((nimg[idx].sum() / tot_n - f) ** 2) / f ** 2
        if s < best_s:
            best, best_s = perm.copy(), s
    assign = np.array(["train"] * G, dtype=object)
    assign[best[:n_test]] = "test"
    assign[best[n_test:n_test + n_val]] = "val"
    gsplit = dict(zip(groups, assign))
    rows = []
    for r in feat_g.itertuples():
        sp = gsplit[r.group]
        if sp != "train" and r.key not in eval_keys:
            sp = "dropped"            # extra augmented copies of a val/test source (kept out on purpose)
        rows.append(dict(key=r.key, split=sp, group=r.group))
    man = pd.DataFrame(rows)
    print(f"grouped split: {G} groups -> test {n_test}, val {n_val}, train {G - n_test - n_val} "
          f"(balance score {best_s:.4f}; dropped copies {int((man.split == 'dropped').sum())})")
    return man, dict(groups=G, balance_score=float(best_s), seed=seed, trials=trials, fractions=fractions)


def check_no_group_leak(man):
    m = man[man.split.isin(["train", "val", "test"])]
    g = m.groupby("group").split.nunique()
    bad = g[g > 1]
    assert bad.empty, f"{len(bad)} groups occur in more than one split: {list(bad.index[:10])}"
    return True
