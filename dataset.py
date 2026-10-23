"""Loading the Roboflow COCO export, cleaning boxes (with a full audit log), split manifests,
YOLO label files and the per-split COCO ground truth used for evaluation.

The raw export is NEVER modified: every cleaned file is written to the working folder and every
change to a box is written to clean_log.csv.
"""
import os, re, json, shutil, hashlib, collections

import numpy as np
import pandas as pd

import config as cfg
import utils


# ---------------------------------------------------------------- finding the raw export
def _split_dir(root, split):
    for name in cfg.SPLIT_DIRS[split]:
        p = os.path.join(root, name)
        if os.path.isfile(os.path.join(p, cfg.ANN_FILE)):
            return p
    return None


def find_coco_roots(root=None):
    root = root or cfg.INPUT_ROOT
    found = []
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if all(_split_dir(r, s) for s in ("train", "val", "test")):
            found.append(r)
            dirs[:] = []
    return found


def export_kind(root):
    """'main' / 'native' / 'sky' / 'unknown' from the category names (and image sizes) of the train split."""
    with open(os.path.join(_split_dir(root, "train"), cfg.ANN_FILE)) as f:
        c = json.load(f)
    names = {str(x["name"]).strip().lower() for x in c["categories"]}
    if "car" in names:
        return "sky"
    if "vehicle" in names:
        sizes = {(im["width"], im["height"]) for im in c["images"]}
        return "main" if sizes == {(640, 640)} else "native"
    return "unknown"


def find_coco_root(kind="main", root=None):
    env = os.environ.get("SAT_RAW_ROOT" if kind == "main" else f"SAT_RAW_ROOT_{kind.upper()}")
    if env:
        return env
    found = [r for r in find_coco_roots(root) if export_kind(r) == kind]
    if not found:
        raise FileNotFoundError(
            f"No '{kind}' Roboflow COCO export found under /kaggle/input (need train/, valid/, test/ each with "
            f"{cfg.ANN_FILE} and class '{cfg.DATASETS[kind]['signature']}'). Add Input -> "
            f"{cfg.DATASETS[kind]['title']}.")
    if len(found) > 1:
        print(f"several '{kind}' exports attached, using the first:", found)
    return found[0]


def load_raw(raw_root):
    """{split: coco dict} with the image folder recorded."""
    out = {}
    for s in ("train", "val", "test"):
        d = _split_dir(raw_root, s)
        with open(os.path.join(d, cfg.ANN_FILE)) as f:
            c = json.load(f)
        c["_dir"] = d
        out[s] = c
    return out


def category_map(raw):
    """COCO category id -> class index (by NAME, never by position) for every split."""
    maps = {}
    for s, c in raw.items():
        m = {}
        for cat in c["categories"]:
            name = str(cat["name"]).strip().lower()
            name = cfg.CLASS_ALIASES.get(name, name)
            if name in cfg.CLASSES:
                m[cat["id"]] = cfg.CLASSES.index(name)
        maps[s] = m
        missing = set(cfg.CLASSES) - {cfg.CLASSES[v] for v in m.values()}
        if missing:
            print(f"WARNING split {s}: classes not in the COCO categories: {missing}")
    return maps


# ---------------------------------------------------------------- source / augmentation id
_RF = re.compile(r"_(?:jpe?g|png|bmp|tiff?)\.rf\.[0-9a-f]+\.(?:jpe?g|png|bmp|tiff?)$", re.I)


def source_id(file_name):
    """Roboflow names augmented copies <original>_jpg.rf.<hash>.jpg -> '<original>'."""
    base = os.path.basename(file_name)
    m = _RF.search(base)
    return base[:m.start()] if m else os.path.splitext(base)[0]


# ---------------------------------------------------------------- image table
def image_table(raw):
    rows = []
    for s, c in raw.items():
        for im in c["images"]:
            rows.append(dict(key=f"{s}/{im['file_name']}", src_split=s, file_name=im["file_name"],
                             path=os.path.join(c["_dir"], im["file_name"]), coco_id=im["id"],
                             width=im["width"], height=im["height"], source_id=source_id(im["file_name"])))
    df = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
    df["exists"] = df.path.map(os.path.isfile)
    return df


# ---------------------------------------------------------------- cleaning
def clean_annotations(raw, cmap=None):
    """Corner clipping to the image, removal of degenerate / exact-duplicate / non-target boxes.

    Returns (boxes, log): boxes = {key: [(cls, x1, y1, x2, y2), ...]} in pixels; log = DataFrame with
    one row per CHANGED annotation (the raw JSON stays untouched).
    """
    cmap = cmap or category_map(raw)
    boxes, log = collections.defaultdict(list), []
    for s, c in raw.items():
        ims = {im["id"]: im for im in c["images"]}
        seen = set()
        for a in sorted(c["annotations"], key=lambda a: a["id"]):
            im = ims.get(a["image_id"])
            if im is None:
                log.append(dict(split=s, file_name=None, ann_id=a["id"], action="orphan_annotation"))
                continue
            key = f"{s}/{im['file_name']}"
            W, H = im["width"], im["height"]
            x, y, w, h = [float(v) for v in a["bbox"]]
            rec = dict(split=s, file_name=im["file_name"], ann_id=a["id"], category_id=a["category_id"],
                       x=x, y=y, w=w, h=h)
            if a["category_id"] not in cmap[s]:
                log.append(dict(rec, action="non_target_class"))
                continue
            k = cmap[s][a["category_id"]]
            x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
            if x2 - x1 < cfg.MIN_BOX_PX or y2 - y1 < cfg.MIN_BOX_PX:
                log.append(dict(rec, action="degenerate_removed"))
                continue
            sig = (key, k) + tuple(round(v, cfg.DUP_ROUND) for v in (x1, y1, x2, y2))
            if sig in seen:
                log.append(dict(rec, action="exact_duplicate_removed"))
                continue
            seen.add(sig)
            if (x1, y1, x2, y2) != (x, y, x + w, y + h):
                log.append(dict(rec, action="clipped", nx1=x1, ny1=y1, nx2=x2, ny2=y2))
            boxes[key].append((k, x1, y1, x2, y2))
    cols = ["split", "file_name", "ann_id", "category_id", "x", "y", "w", "h", "action", "nx1", "ny1", "nx2", "ny2"]
    return dict(boxes), pd.DataFrame(log, columns=cols)


# ---------------------------------------------------------------- writing a split
def split_fingerprint(manifest):
    s = "\n".join(f"{r.key}|{r.split}|{r.group}" for r in manifest.sort_values("key").itertuples())
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def write_split(name, manifest, imgs, boxes, out_root=None, log=None, extra=None, id_map=None, tiles_from=None):
    """manifest: DataFrame(key, split, group, [use]) -> splits/<name>/ with labels, COCO GT, manifest.
    id_map: {key: image_id} keeps the image ids of a base split (same val/test ids -> paired comparisons).
    tiles_from: folder with tile images (rows with src_split == 'tile') copied into splits/<name>/tiles/."""
    out = os.path.join(out_root or cfg.SPLITS_OUT, name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    meta = imgs.set_index("key")
    man = manifest.merge(imgs[["key", "src_split", "file_name", "width", "height", "source_id"]], on="key")
    dup = man[man.split.isin(["train", "val", "test"])].groupby("split").file_name.apply(lambda s: s.duplicated().sum())
    assert int(dup.sum()) == 0, f"file-name collision inside a split: {dup.to_dict()}"
    man = man.sort_values(["split", "key"]).reset_index(drop=True)
    if id_map:
        known = man.key.map(id_map)
        start = int(max(id_map.values())) + 1
        new = np.arange(start, start + int(known.isna().sum()))
        known[known.isna()] = new
        man["image_id"] = known.astype(int)
        assert man.image_id.is_unique
    else:
        man["image_id"] = np.arange(1, len(man) + 1)
    if tiles_from is not None:
        os.makedirs(os.path.join(out, "tiles"), exist_ok=True)
        for r in man[man.src_split == "tile"].itertuples():
            shutil.copyfile(os.path.join(tiles_from, r.file_name), os.path.join(out, "tiles", r.file_name))
    summary = {}
    for sp in ("train", "val", "test"):
        part = man[man.split == sp]
        lbl_dir = os.path.join(out, "labels", sp)
        os.makedirs(lbl_dir, exist_ok=True)
        images, anns = [], []
        for r in part.itertuples():
            W, H = r.width, r.height
            lines = []
            for (k, x1, y1, x2, y2) in boxes.get(r.key, []):
                lines.append(f"{k} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                anns.append(dict(id=len(anns) + 1, image_id=int(r.image_id), category_id=k + 1,
                                 bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), iscrowd=0))
            with open(os.path.join(lbl_dir, os.path.splitext(r.file_name)[0] + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            images.append(dict(id=int(r.image_id), file_name=r.file_name, width=int(W), height=int(H)))
        gt = dict(info=dict(description=f"{name}/{sp} cleaned GT"), images=images, annotations=anns,
                  categories=[dict(id=i + 1, name=n) for i, n in enumerate(cfg.CLASSES)])
        utils.save_json(gt, os.path.join(out, "coco", f"{sp}.json"))
        cls = collections.Counter(cfg.CLASSES[a["category_id"] - 1] for a in anns)
        summary[sp] = dict(images=len(images), groups=int(part.group.nunique()), annotations=len(anns),
                           empty_images=len(images) - len({a["image_id"] for a in anns}), **cls)
    man.to_csv(os.path.join(out, "manifest.csv"), index=False)
    if log is not None:
        log.to_csv(os.path.join(out, "clean_log.csv"), index=False)
    fp = split_fingerprint(man)
    meta = dict(name=name, fingerprint=fp, summary=summary, classes=cfg.CLASSES, **(extra or {}))
    utils.save_json(meta, os.path.join(out, "split_meta.json"))
    print(f"split '{name}' fingerprint {fp}")
    print(pd.DataFrame(summary).T.to_string())
    return meta


# ---------------------------------------------------------------- using a split in a GPU notebook
UNZIP_DIR = os.environ.get("SAT_UNZIP", "/tmp/sat_unzipped")


def _unpack_split_zips():
    """v2.5: notebook 06 saves ONE file sat_splits_extra.zip (thousands of small files broke Kaggle downloads).
    Any attached sat_splits*.zip is extracted once to /tmp (not into the notebook output)."""
    import zipfile
    for z in utils.find_files("sat_splits_extra.zip"):
        dst = os.path.join(UNZIP_DIR, hashlib.sha1(z.encode()).hexdigest()[:10])
        if not os.path.isfile(os.path.join(dst, ".done")):
            with zipfile.ZipFile(z) as f:
                f.extractall(dst)
            open(os.path.join(dst, ".done"), "w").close()
            print("unzipped", z, "->", dst)


def find_split_dir(name, root=None):
    hits = []
    _unpack_split_zips()
    metas = utils.find_files("split_meta.json", root, also_work=True)
    if root is None and os.path.isdir(UNZIP_DIR):
        metas += utils.find_files("split_meta.json", UNZIP_DIR)
    for p in metas:
        m = utils.load_json(p)
        if m.get("name") == name:
            hits.append((os.path.dirname(p), m.get("fingerprint"), int(m.get("label_version", 1))))
    hits.sort(key=lambda h: -h[2])          # relabelled test GT (notebook 13, label_version 2) wins
    if hits and hits[0][2] > 1:
        print(f"split '{name}': using corrected test labels (label_version {hits[0][2]}) from {hits[0][0]}")
    hits = [(d, f) for d, f, _ in hits]
    if len({f for _, f in hits}) > 1:
        print(f"WARNING: split '{name}' is attached {len(hits)} times with DIFFERENT fingerprints "
              f"{[f for _, f in hits]} -> using {hits[0]}. Remove the old Dataset version from the inputs.")
    if hits:
        return hits[0][0]
    import extra_splits as xs
    if root is None and name in xs.MAIN_EXTRA + xs.NATIVE_EXTRA and not getattr(find_split_dir, "_building", False):
        print(f"split '{name}' not attached -> building it now from sat-splits (+ satelite-native); same result as notebook 06")
        find_split_dir._building = True
        try:
            xs.build_all(os.path.join(UNZIP_DIR, "built"), which=("main",) if name in xs.MAIN_EXTRA else ("native",))
        finally:
            find_split_dir._building = False
        return find_split_dir(name)
    raise FileNotFoundError(f"split '{name}' not found - attach notebook 02's output (Dataset 'sat-splits').")


_TILE_CACHE = {}


def _cut_tile(r, raw_root, dst):
    """Re-create a 06 tile (<stem>__x<X>_y<Y>.jpg) from its parent native image - same crop, same cv2 JPEG encoding."""
    import cv2
    parent = getattr(r, "parent", None)
    assert isinstance(parent, str) and "/" in parent, \
        f"tile {r.file_name}: no parent in the manifest - re-run notebook 06 with sat-ext v2.4"
    m = re.search(r"__x(\d+)_y(\d+)\.jpg$", r.file_name)
    x0, y0 = int(m.group(1)), int(m.group(2))
    if _TILE_CACHE.get("key") != parent:
        ps, pf = parent.split("/", 1)
        im = cv2.imread(os.path.join(_split_dir(raw_root, ps), pf))
        assert im is not None, f"native image not found for tile parent {parent} - attach satelite-native"
        _TILE_CACHE.clear(); _TILE_CACHE.update(key=parent, im=im)
    im = _TILE_CACHE["im"]
    cv2.imwrite(dst, im[y0:y0 + int(r.height), x0:x0 + int(r.width)])


def materialize(name, raw_root=None, work=None):
    """Build a YOLO folder (copied images + cleaned labels) and data.yaml for split `name`."""
    sdir = find_split_dir(name)
    meta = utils.load_json(os.path.join(sdir, "split_meta.json"))
    raw_root = raw_root or find_coco_root(meta.get("dataset", "main"))
    man = pd.read_csv(os.path.join(sdir, "manifest.csv"), dtype={"source_id": str, "file_name": str, "key": str})
    root = os.path.join(work or cfg.WORK_DIR, f"data_{name}")
    for sp in ("train", "val", "test"):
        idir, ldir = os.path.join(root, sp, "images"), os.path.join(root, sp, "labels")
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        for r in man[man.split == sp].itertuples():
            dst = os.path.join(idir, r.file_name)
            if not os.path.exists(dst):
                if r.src_split == "tile" and not os.path.exists(os.path.join(sdir, "tiles", r.file_name)):
                    _cut_tile(r, raw_root, dst)    # v2.4: tiles are re-cut from the native image (small 06 output)
                else:
                    src = (os.path.join(sdir, "tiles", r.file_name) if r.src_split == "tile"
                           else os.path.join(_split_dir(raw_root, r.src_split), r.file_name))
                    shutil.copyfile(src, dst)      # real copies: Ultralytics finds labels via the image path
            stem = os.path.splitext(r.file_name)[0] + ".txt"
            shutil.copyfile(os.path.join(sdir, "labels", sp, stem), os.path.join(ldir, stem))
    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {root}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                f"nc: {len(cfg.CLASSES)}\nnames: {cfg.CLASSES}\n")
    info = dict(name=name, dataset=meta.get("dataset", "main"), fingerprint=meta["fingerprint"], yaml=yaml_path, root=root, split_dir=sdir,
                gt={sp: os.path.join(sdir, "coco", f"{sp}.json") for sp in ("train", "val", "test")},
                images={sp: os.path.join(root, sp, "images") for sp in ("train", "val", "test")})
    n = {sp: len(os.listdir(info["images"][sp])) for sp in ("train", "val", "test")}
    print(f"materialized '{name}' ({meta['fingerprint']}) -> {root}  images {n}")
    return info
