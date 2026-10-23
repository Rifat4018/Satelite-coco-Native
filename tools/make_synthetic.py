"""Builds a FAKE Roboflow-style COCO export (train/valid/test) for the smoke test, with planted problems:
  * augmented copies (3 per train source, incl. +/-15 deg rotations with black corners)
  * a test tile that overlaps a train tile (same ground area)        -> must be found by ORB
  * a valid image that is a mirrored copy of a train image, new name -> must be found by pHash
  * a test image sharing a source name with train                     -> must be found by source_id
  * negative coordinates, boxes beyond the border, exact duplicate boxes, a supercategory 'objects'
Usage: python make_synthetic.py OUT_DIR
"""
import os, sys, json, hashlib

import numpy as np
import cv2

RNG = np.random.default_rng(7)
CLS = {1: "plane", 2: "ship", 3: "vehicle"}


def world(size=2600):
    base = (RNG.random((size // 20, size // 20, 3)) * 255).astype(np.uint8)
    w = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
    w = cv2.GaussianBlur(w, (0, 0), 3)
    objs = []
    for _ in range(1400):
        k = int(RNG.choice([1, 2, 3], p=[0.15, 0.1, 0.75]))
        s = {1: RNG.integers(25, 70), 2: RNG.integers(12, 40), 3: RNG.integers(5, 16)}[k]
        x, y = RNG.integers(0, size - 80, 2)
        w2, h2 = int(s), int(s * RNG.uniform(0.5, 1.5))
        col = tuple(int(c) for c in RNG.integers(0, 255, 3))
        cv2.rectangle(w, (int(x), int(y)), (int(x + w2), int(y + h2)), col, -1)
        cv2.line(w, (int(x), int(y)), (int(x + w2), int(y + h2)), (255 - col[0], 0, col[2]), 1)
        objs.append((k, float(x), float(y), float(w2), float(h2)))
    return w, objs


def tile(W, objs, x0, y0, T=640):
    img = W[y0:y0 + T, x0:x0 + T].copy()
    boxes = []
    for k, x, y, w, h in objs:
        if x + w > x0 and x < x0 + T and y + h > y0 and y < y0 + T:
            boxes.append([k, x - x0, y - y0, w, h])      # NOT clipped -> negative / beyond-border boxes
    return img, boxes


def aug(img, boxes, kind):
    H, W = img.shape[:2]
    out = []
    if kind == "hflip":
        img = img[:, ::-1].copy()
        out = [[k, W - x - w, y, w, h] for k, x, y, w, h in boxes]
    elif kind == "rot90":
        img = np.ascontiguousarray(np.rot90(img, 1))
        out = [[k, y, W - x - w, h, w] for k, x, y, w, h in boxes]
    elif kind.startswith("rot"):
        ang = float(kind[3:])
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))
        for k, x, y, w, h in boxes:
            pts = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]]) @ M.T
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            out.append([k, float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    else:
        out = [list(b) for b in boxes]
    img = np.clip(img.astype(float) * RNG.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
    return img, out


def rfname(src, salt):
    return f"{src}_jpg.rf.{hashlib.md5((src + salt).encode()).hexdigest()}.jpg"


def main_sky(out):
    """SkySeaLand-like: classes airplane/boat/car/ship, non-square native images, no augmented copies."""
    W, objs = world()
    cats = [dict(id=0, name="objects", supercategory="none")] + [dict(id=k, name=n, supercategory="objects")
                                                                for k, n in enumerate(["airplane", "boat", "car", "ship"], 1)]
    remap = {1: 1, 2: 4, 3: 3}               # synthetic plane->airplane, ship->ship, vehicle->car
    data = {s: dict(images=[], annotations=[], categories=cats) for s in ("train", "valid", "test")}
    k = 0
    for y0 in range(0, 2000, 480):
        for x0 in range(0, 1600, 950):
            split = "test" if k % 5 == 4 else "valid" if k % 5 == 3 else "train"
            img = W[y0:y0 + 420, x0:x0 + 900].copy()
            d = data[split]
            fname = rfname(f"Screenshot-{k:03d}_png", "s").replace("_png_jpg", "_png")
            os.makedirs(os.path.join(out, split), exist_ok=True)
            cv2.imwrite(os.path.join(out, split, fname), img)
            iid = len(d["images"])
            d["images"].append(dict(id=iid, file_name=fname, height=420, width=900))
            for c, x, y, w, h in objs:
                if x >= x0 and y >= y0 and x + w <= x0 + 900 and y + h <= y0 + 420:
                    cid = 2 if (c == 2 and w < 20) else remap[c]          # small ships become 'boat'
                    d["annotations"].append(dict(id=len(d["annotations"]), image_id=iid, category_id=cid,
                                                 bbox=[x - x0, y - y0, w, h], area=w * h, iscrowd=0, segmentation=[]))
            k += 1
    for s, d in data.items():
        with open(os.path.join(out, s, "_annotations.coco.json"), "w") as f:
            json.dump(d, f)
        print("sky", s, len(d["images"]), "images", len(d["annotations"]), "annotations")


def main(out):
    W, objs = world()
    cats = [dict(id=0, name="objects", supercategory="none")] + [dict(id=k, name=n, supercategory="objects") for k, n in CLS.items()]
    data = {s: dict(images=[], annotations=[], categories=cats) for s in ("train", "valid", "test")}

    def add(split, fname, img, boxes, dup_first=False):
        d = data[split]
        os.makedirs(os.path.join(out, split), exist_ok=True)
        cv2.imwrite(os.path.join(out, split, fname), img)
        iid = len(d["images"])
        d["images"].append(dict(id=iid, file_name=fname, height=img.shape[0], width=img.shape[1]))
        for b in boxes + (boxes[:1] if dup_first and boxes else []):
            k, x, y, w, h = b
            d["annotations"].append(dict(id=len(d["annotations"]), image_id=iid, category_id=k,
                                         bbox=[round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                                         area=round(w * h, 1), iscrowd=0, segmentation=[]))

    grid = [(x, y) for y in range(0, 1900, 700) for x in range(0, 1900, 700)]   # 9 non-overlapping tiles
    srcs = [f"{i:03d}" for i in range(len(grid))]
    train_src, val_src, test_src = srcs[:5], srcs[5:7], srcs[7:9]
    first_train = None
    for s, (x0, y0) in zip(srcs, grid):
        img, bx = tile(W, objs, x0, y0)
        if s in train_src:
            for j, kind in enumerate(["hflip", "rot90", f"rot{RNG.choice([-15, 12])}"]):
                im2, b2 = aug(img, bx, kind)
                add("train", rfname(s, str(j)), im2, b2, dup_first=(j == 0))
                if first_train is None:
                    first_train = (img, bx)
        elif s in val_src:
            add("valid", rfname(s, "v"), img, bx)
        else:
            add("test", rfname(s, "t"), img, bx)
    # planted leaks
    x0, y0 = grid[0]
    img, bx = tile(W, objs, x0 + 300, y0 + 250)                         # overlaps train tile 000
    add("test", rfname("900", "t"), img, bx)
    im_m, b_m = aug(first_train[0], first_train[1], "hflip")            # mirror of train source 000
    add("valid", rfname("901", "v"), im_m, b_m)
    img, bx = tile(W, objs, grid[1][0], grid[1][1])                     # same source name as train '001'
    add("test", rfname("001", "dup"), img, bx)
    for s, d in data.items():
        with open(os.path.join(out, s, "_annotations.coco.json"), "w") as f:
            json.dump(d, f)
        print(s, len(d["images"]), "images", len(d["annotations"]), "annotations")


def _crop(W, objs, x0, y0, w, h):
    img = W[y0:y0 + h, x0:x0 + w].copy()
    bx = [[k, x - x0, y - y0, bw, bh] for k, x, y, bw, bh in objs
          if x >= x0 and y >= y0 and x + bw <= x0 + w and y + bh <= y0 + h]
    return img, bx


def main_native(out, w=960, h=600):
    """Same world + same source names as main(), one image per source at a non-square 'native' size."""
    W, objs = world()
    cats = [dict(id=0, name="objects", supercategory="none")] + [dict(id=k, name=n, supercategory="objects") for k, n in CLS.items()]
    data = {s: dict(images=[], annotations=[], categories=cats) for s in ("train", "valid", "test")}
    grid = [(x, y) for y in range(0, 1900, 700) for x in range(0, 1900, 700)]
    items = [(f"{i:03d}", x, y, False) for i, (x, y) in enumerate(grid)]
    items += [("900", grid[0][0] + 300, grid[0][1] + 250, False), ("901", grid[0][0], grid[0][1], True)]
    for n, (s, x0, y0, mirror) in enumerate(items):
        img, bx = _crop(W, objs, x0, y0, w, h)
        if mirror:
            img = img[:, ::-1].copy()
            bx = [[k, w - x - bw, y, bw, bh] for k, x, y, bw, bh in bx]
        split = ["train", "valid", "test"][n % 3]
        d = data[split]
        fn = rfname(s, "native")
        os.makedirs(os.path.join(out, split), exist_ok=True)
        cv2.imwrite(os.path.join(out, split, fn), img)
        iid = len(d["images"])
        d["images"].append(dict(id=iid, file_name=fn, height=h, width=w))
        for k, x, y, bw, bh in bx:
            d["annotations"].append(dict(id=len(d["annotations"]), image_id=iid, category_id=k, bbox=[x, y, bw, bh],
                                         area=bw * bh, iscrowd=0, segmentation=[]))
    for sp, d in data.items():
        with open(os.path.join(out, sp, "_annotations.coco.json"), "w") as f:
            json.dump(d, f)
        print("native", sp, len(d["images"]), "images", len(d["annotations"]), "annotations")


def main_dota(out):
    W, objs = world()
    names = {1: "plane", 2: "ship", 3: "small-vehicle"}
    os.makedirs(os.path.join(out, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "val", "labelTxt"), exist_ok=True)
    for i, (x0, y0) in enumerate([(100, 100), (1200, 300), (500, 1300), (1400, 1400)]):
        img, bx = _crop(W, objs, x0, y0, 1100, 900)
        stem = f"P{i:04d}"
        cv2.imwrite(os.path.join(out, "val", "images", stem + ".png"), img)
        with open(os.path.join(out, "val", "labelTxt", stem + ".txt"), "w") as f:
            f.write("imagesource:GoogleEarth\ngsd:0.5\n")
            for j, (k, x, y, bw, bh) in enumerate(bx):
                f.write(f"{x} {y} {x + bw} {y} {x + bw} {y + bh} {x} {y + bh} {names[k]} {int(j % 10 == 0)}\n")
            f.write("10 10 60 10 60 40 10 40 harbor 0\n")
    print("dota: 4 images")


def make_agreement(zip_path, ref_gt_path, out):
    """Fake second annotator: the reference boxes jittered, 10% dropped, Roboflow-style renamed files."""
    import zipfile
    ref = json.load(open(ref_gt_path))
    by = {}
    for a in ref["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    ids = {i["file_name"]: i for i in ref["images"]}
    d = dict(images=[], annotations=[], categories=[dict(id=0, name="objects"), dict(id=1, name="plane"),
                                                     dict(id=2, name="ship"), dict(id=3, name="vehicle")])
    os.makedirs(os.path.join(out, "train"), exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for n, fn in enumerate(z.namelist()):
            new = rfname(os.path.splitext(fn)[0].split("_jpg.rf.")[0], "agree")
            open(os.path.join(out, "train", new), "wb").write(z.read(fn))
            im = ids[fn]
            d["images"].append(dict(id=n, file_name=new, width=im["width"], height=im["height"]))
            for a in by.get(im["id"], []):
                if RNG.random() < 0.1:
                    continue
                x, y, w, h = a["bbox"]
                j = RNG.normal(0, 1.0, 4)
                d["annotations"].append(dict(id=len(d["annotations"]), image_id=n, category_id=a["category_id"],
                                             bbox=[x + j[0], y + j[1], max(1, w + j[2]), max(1, h + j[3])], area=w * h, iscrowd=0))
    json.dump(d, open(os.path.join(out, "train", "_annotations.coco.json"), "w"))
    print("agreement export:", len(d["images"]), "images", len(d["annotations"]), "boxes")


if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "main"
    if mode == "agree":
        make_agreement(sys.argv[3], sys.argv[4], sys.argv[1])
    else:
        dict(main=main, sky=main_sky, native=main_native, dota=main_dota)[mode](sys.argv[1])
