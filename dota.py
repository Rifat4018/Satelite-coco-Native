"""External test set: DOTA-v1.0 (Xia et al., CVPR 2018), horizontal boxes from the oriented polygons.
Classes are mapped with cfg.DOTA_CLASS_MAP; 'difficult' objects are kept as ignore regions (iscrowd=1).
Any DOTA folder layout works as long as label files (*.txt, DOTA format) and the images (*.png/*.jpg)
with the same file stem are somewhere under /kaggle/input."""
import os, random, hashlib

from PIL import Image

import config as cfg
import utils

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _is_dota_label(path):
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 9 and not line.startswith(("imagesource", "gsd")):
                    float(p[0]); float(p[7])
                    return True
    except Exception:
        return False
    return False


def find_dota(root=None):
    """{stem: (image path, label path)} for every DOTA label with a matching image."""
    root = root or cfg.INPUT_ROOT
    labels, images = {}, {}
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() == ".txt" and "label" in r.lower():
                labels[stem] = os.path.join(r, f)
            elif ext.lower() in IMG_EXT:
                images.setdefault(stem, os.path.join(r, f))
    pairs = {s: (images[s], l) for s, l in labels.items() if s in images}
    pairs = {s: v for s, v in pairs.items() if _is_dota_label(v[1])}
    print(f"DOTA: {len(labels)} label files, {len(pairs)} with an image")
    return pairs


def parse(label_path):
    out = []
    with open(label_path) as f:
        for line in f:
            if line.startswith(("imagesource", "gsd")):
                continue
            p = line.split()
            if len(p) < 9:
                continue
            xs, ys = [float(v) for v in p[0:8:2]], [float(v) for v in p[1:8:2]]
            out.append(dict(cls=p[8], difficult=int(p[9]) if len(p) > 9 else 0,
                            box=(min(xs), min(ys), max(xs), max(ys))))
    return out


def build(max_images=cfg.DOTA_MAX_IMAGES, seed=cfg.SEED, work=None):
    """COCO ground truth for a fixed random sample of DOTA images that contain our classes."""
    pairs = find_dota()
    if not pairs:
        raise FileNotFoundError("No DOTA images + labelTxt found under /kaggle/input. Add Input -> a DOTA-v1.0 "
                                "dataset (validation part with images/ and labelTxt/).")
    keep = []
    for stem in sorted(pairs):
        objs = [o for o in parse(pairs[stem][1]) if o["cls"] in cfg.DOTA_CLASS_MAP and not o["difficult"]]
        if objs:
            keep.append(stem)
    random.Random(seed).shuffle(keep)
    keep = sorted(keep[:max_images])
    work = work or os.path.join(cfg.WORK_DIR, "dota")
    idir = os.path.join(work, "images")
    os.makedirs(idir, exist_ok=True)
    images, anns, stats = [], [], {}
    for i, stem in enumerate(keep, 1):
        ip, lp = pairs[stem]
        fn = stem + os.path.splitext(ip)[1]
        dst = os.path.join(idir, fn)
        if not os.path.lexists(dst):
            os.symlink(ip, dst)
        W, H = Image.open(ip).size
        images.append(dict(id=i, file_name=fn, width=W, height=H))
        for o in parse(lp):
            name = cfg.DOTA_CLASS_MAP.get(o["cls"])
            if name is None:
                continue
            x1, y1, x2, y2 = max(0, o["box"][0]), max(0, o["box"][1]), min(W, o["box"][2]), min(H, o["box"][3])
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            anns.append(dict(id=len(anns) + 1, image_id=i, category_id=cfg.CLASSES.index(name) + 1,
                             bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), iscrowd=int(o["difficult"])))
            stats[(o["cls"], o["difficult"])] = stats.get((o["cls"], o["difficult"]), 0) + 1
    gt = dict(images=images, annotations=anns, categories=[dict(id=i + 1, name=n) for i, n in enumerate(cfg.CLASSES)])
    gpath = os.path.join(work, "dota_gt.json")
    utils.save_json(gt, gpath)
    fp = hashlib.sha256("\n".join(keep).encode()).hexdigest()[:12]
    print(f"DOTA test sample: {len(images)} images, {len(anns)} boxes ({sum(a['iscrowd'] for a in anns)} difficult = ignored),"
          f" fingerprint {fp}")
    print("  objects by DOTA class (difficult flag):", dict(sorted(stats.items())))
    return dict(name="dota_val", dataset="dota", fingerprint=fp, yaml=None, root=work, split_dir=work,
                gt={"test": gpath}, images={"test": idir})


def modes():
    return ["standard"] + [f"sahi{s}" for s in cfg.DOTA_SLICES]
