"""Extra splits derived from the grouped split (notebook 06). Val/test stay IDENTICAL to `grouped`
(same images, same image ids), so every comparison is paired on the same test images.

grouped_1copy     train keeps ONE of the 3 Roboflow augmented copies per source (offline-augmentation ablation)
grouped_lc25/50   train keeps 25% / 50% of the train GROUPS (nested), all copies (learning curve)
native_grouped    same sources and split as `grouped`, but the native-resolution export (no resize, no augmentation)
native_tiles      native_grouped with train images cut into 640x640 tiles (tile training); val/test = full images
"""
import os, collections

import numpy as np
import pandas as pd
import cv2

import config as cfg
import utils
import dataset as ds


def load_base(name="grouped"):
    sdir = ds.find_split_dir(name)
    man = pd.read_csv(os.path.join(sdir, "manifest.csv"), dtype={"source_id": str, "file_name": str, "key": str})
    meta = utils.load_json(os.path.join(sdir, "split_meta.json"))
    aud = [p for p in utils.find_files("05_images_groups.csv", os.path.dirname(sdir))
           if os.path.basename(os.path.dirname(p)) in ("audit_main", "main")]
    bc = pd.read_csv(aud[0])[["key", "black_corners"]] if aud else None
    print(f"base '{name}' ({meta['fingerprint']}) from {sdir}; audit table: {aud[0] if aud else 'NOT FOUND'}")
    return sdir, man, meta, bc


def _keep_eval(man):
    return man[man.split.isin(["val", "test"])][["key", "split", "group"]]


def one_copy(man, bc):
    tr = man[man.split == "train"].merge(bc, on="key", how="left") if bc is not None else man[man.split == "train"].assign(black_corners=0.0)
    keep = set(tr.sort_values(["source_id", "black_corners", "key"]).groupby("source_id").head(1).key)
    out = man[["key", "split", "group"]].copy()
    out.loc[(man.split == "train") & ~man.key.isin(keep), "split"] = "dropped"
    return out


def learning_curve(man, frac, seed=cfg.SEED):
    groups = np.array(sorted(man[man.split == "train"].group.unique()))
    order = np.random.default_rng(seed).permutation(len(groups))       # nested: 25% is a subset of 50%
    keep = set(groups[order[:max(1, int(round(frac * len(groups))))]])
    out = man[["key", "split", "group"]].copy()
    out.loc[(man.split == "train") & ~man.group.isin(keep), "split"] = "dropped"
    return out


def native_manifest(base_man, nimgs):
    """Assign every native image the split / group of its source in the base split."""
    src = base_man[base_man.split.isin(["train", "val", "test"])].drop_duplicates("source_id")
    src = src.assign(source_id=src.source_id.astype(str)).set_index("source_id")[["split", "group"]]
    m = nimgs[["key", "source_id"]].assign(source_id=nimgs.source_id.astype(str)).join(src, on="source_id")
    missing = m.split.isna().sum()
    if missing:
        print(f"WARNING: {missing} native images have no source in the base split -> left out:",
              m[m.split.isna()].source_id.head(10).tolist())
    m = m.dropna(subset=["split"])
    dup = m[m.duplicated("source_id", keep=False)]
    if len(dup):
        print(f"note: {dup.source_id.nunique()} sources have several native images (kept all, same group)")
    return m[["key", "split", "group"]]


def tile_positions(L, T=cfg.TILE, overlap=cfg.TILE_OVERLAP):
    if L <= T:
        return [0]
    step = int(T * (1 - overlap))
    pos = list(range(0, L - T + 1, step))
    if pos[-1] != L - T:
        pos.append(L - T)
    return pos


def make_tiles(nimgs, nboxes, train_keys, out_dir, T=cfg.TILE):
    """Cut native train images into tiles. Returns (tile image rows, tile boxes)."""
    os.makedirs(out_dir, exist_ok=True)
    rows, tboxes, stats = [], {}, collections.Counter()
    for r in nimgs[nimgs.key.isin(train_keys)].itertuples():
        im = cv2.imread(r.path)
        H, W = im.shape[:2]
        for y0 in tile_positions(H, T):
            for x0 in tile_positions(W, T):
                tw, th = min(T, W - x0), min(T, H - y0)
                bl = []
                for (k, x1, y1, x2, y2) in nboxes.get(r.key, []):
                    cx1, cy1, cx2, cy2 = max(x1, x0), max(y1, y0), min(x2, x0 + tw), min(y2, y0 + th)
                    if cx2 - cx1 < 1 or cy2 - cy1 < 1:
                        continue
                    if (cx2 - cx1) * (cy2 - cy1) < cfg.TILE_MIN_VIS * (x2 - x1) * (y2 - y1):
                        stats["boxes_cut_below_min_visibility"] += 1
                        continue
                    bl.append((k, cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0))
                fn = f"{os.path.splitext(r.file_name)[0]}__x{x0}_y{y0}.jpg"
                cv2.imwrite(os.path.join(out_dir, fn), im[y0:y0 + th, x0:x0 + tw])
                key = f"tile/{fn}"
                rows.append(dict(key=key, src_split="tile", file_name=fn, path=os.path.join(out_dir, fn),
                                 coco_id=-1, width=tw, height=th, source_id=r.source_id, exists=True, parent=r.key))
                tboxes[key] = bl
                stats["tiles"] += 1
                stats["empty_tiles"] += int(not bl)
    print("tiles:", dict(stats))
    return pd.DataFrame(rows), tboxes


MAIN_EXTRA = ["grouped_1copy"] + [f"grouped_lc{int(f * 100)}" for f in cfg.LEARNING_CURVE]
NATIVE_EXTRA = ["native_grouped", "native_tiles"]


def build_all(out_root=None, which=("main", "native")):
    """Everything notebook 06 does, callable from any notebook (07I-07L build their split themselves since v2.6).
    Deterministic: same inputs -> same fingerprints as a 06 run."""
    out_root = out_root or cfg.SPLITS_OUT
    sdir, base, meta, bc = load_base("grouped")
    made = []
    if "main" in which:
        raw = ds.load_raw(ds.find_coco_root("main"))
        imgs = ds.image_table(raw)
        boxes, log = ds.clean_annotations(raw, ds.category_map(raw))
        id_map = dict(zip(base.key, base.image_id))
        ds.write_split("grouped_1copy", one_copy(base, bc), imgs, boxes, out_root=out_root, log=log, id_map=id_map,
                       extra=dict(dataset="main", base=meta["fingerprint"], variant="1 augmented copy per train source"))
        for f in cfg.LEARNING_CURVE:
            ds.write_split(f"grouped_lc{int(f * 100)}", learning_curve(base, f), imgs, boxes, out_root=out_root, log=log,
                           id_map=id_map, extra=dict(dataset="main", base=meta["fingerprint"], variant=f"{int(f * 100)}% of train groups"))
        for name in MAIN_EXTRA:      # val/test must be identical to the base split
            for sp in ("val", "test"):
                a = utils.load_json(os.path.join(sdir, "coco", f"{sp}.json"))
                b = utils.load_json(os.path.join(out_root, name, "coco", f"{sp}.json"))
                assert a["images"] == b["images"] and len(a["annotations"]) == len(b["annotations"]), (name, sp)
        print("val/test identical to 'grouped' in every ablation split: OK")
        made += MAIN_EXTRA
    if "native" in which:
        try:
            nat = ds.find_coco_root("native")
        except FileNotFoundError as e:
            print("native export (satelite-native) not attached -> native splits skipped.", e)
            return made
        nraw = ds.load_raw(nat)
        nimgs = ds.image_table(nraw)
        nboxes, nlog = ds.clean_annotations(nraw, ds.category_map(nraw))
        print("native images:", len(nimgs), "| sizes (top 5):", nimgs.groupby(["width", "height"]).size().nlargest(5).to_dict())
        nman = native_manifest(base, nimgs)
        ds.write_split("native_grouped", nman, nimgs, nboxes, out_root=out_root, log=nlog,
                       extra=dict(dataset="native", base=meta["fingerprint"], variant="native resolution"))
        nid = pd.read_csv(os.path.join(out_root, "native_grouped", "manifest.csv"), dtype={"key": str})
        tdir = os.path.join(ds.UNZIP_DIR, "_tiles_tmp")
        trows, tboxes = make_tiles(nimgs, nboxes, set(nman[nman.split == "train"].key), tdir)
        grp = dict(zip(nman.key, nman.group))
        tman = pd.concat([nman[nman.split != "train"],
                          pd.DataFrame(dict(key=trows.key, split="train", group=trows.parent.map(grp), parent=trows.parent))])
        # tile JPEGs are not stored: materialize() re-cuts them from satelite-native ('parent' column)
        ds.write_split("native_tiles", tman, pd.concat([nimgs, trows.drop(columns=["parent"])], ignore_index=True),
                       {**nboxes, **tboxes}, out_root=out_root, log=nlog, id_map=dict(zip(nid.key, nid.image_id)),
                       tiles_from=None, extra=dict(dataset="native", base=meta["fingerprint"], variant=f"{cfg.TILE}px train tiles"))
        import shutil; shutil.rmtree(tdir, ignore_errors=True)
        made += NATIVE_EXTRA
    return made
