"""Roboflow: download a dataset version (optional) and build the corrected upload package.

Download needs the `roboflow` pip package, Internet ON and your API key stored as a Kaggle secret
called ROBOFLOW_API_KEY (Add-ons -> Secrets). The key is never printed or saved.
"""
import os, json, shutil, zipfile, collections

import pandas as pd

import config as cfg
import utils
import dataset as ds


def api_key():
    k = os.environ.get("ROBOFLOW_API_KEY")
    if k:
        return k
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
    except Exception:
        return None


def download(workspace=cfg.RF_WORKSPACE, project=cfg.RF_PROJECT, version=cfg.RF_VERSION, fmt=cfg.RF_FORMAT,
             dest=None):
    key = api_key()
    if not key:
        raise RuntimeError("No ROBOFLOW_API_KEY secret. Kaggle: Add-ons -> Secrets -> add ROBOFLOW_API_KEY "
                           "(Roboflow -> Settings -> API Keys -> Private API key), tick it for this notebook.")
    utils.pip_install("roboflow")
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    proj = rf.workspace(workspace).project(project)
    try:
        print("versions available:", [v.version for v in proj.versions()])
    except Exception:
        pass
    dest = dest or os.path.join(cfg.WORK_DIR, f"rf_{project}_v{version}")
    v = proj.version(version)
    v.download(fmt, location=dest, overwrite=True)
    print("downloaded to", dest)
    return dest


def compare_exports(a_root, b_root):
    """File names, annotation counts and categories of two COCO exports, split by split."""
    A, B = ds.load_raw(a_root), ds.load_raw(b_root)
    rows = []
    for s in ("train", "val", "test"):
        fa = {im["file_name"] for im in A[s]["images"]}
        fb = {im["file_name"] for im in B[s]["images"]}
        rows.append(dict(split=s, images_a=len(fa), images_b=len(fb), same_names=len(fa & fb),
                         only_a=len(fa - fb), only_b=len(fb - fa),
                         anns_a=len(A[s]["annotations"]), anns_b=len(B[s]["annotations"]),
                         cats_a=[c["name"] for c in A[s]["categories"]], cats_b=[c["name"] for c in B[s]["categories"]]))
    df = pd.DataFrame(rows)
    identical = bool((df.only_a == 0).all() and (df.only_b == 0).all() and (df.anns_a == df.anns_b).all())
    print(df.to_string(index=False))
    print("IDENTICAL exports" if identical else "EXPORTS DIFFER - see table")
    return df, identical


def build_upload_package(split_name="grouped", raw_root=None, out=None, readme_text=None):
    """COCO folders train/ valid/ test/ (cleaned boxes, only the images used in the split) ready for
    Roboflow 'Upload Data'. Roboflow's own COCO convention is kept (folder 'valid', file
    _annotations.coco.json, categories plane/ship/vehicle)."""
    sdir = ds.find_split_dir(split_name)
    raw_root = raw_root or ds.find_coco_root(utils.load_json(os.path.join(sdir, "split_meta.json")).get("dataset", "main"))
    man = pd.read_csv(os.path.join(sdir, "manifest.csv"), dtype={"source_id": str, "file_name": str, "key": str})
    out = out or os.path.join(cfg.WORK_DIR, f"roboflow_upload_{split_name}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    counts = {}
    for sp, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        gt = utils.load_json(os.path.join(sdir, "coco", f"{sp}.json"))
        dst = os.path.join(out, folder)
        os.makedirs(dst, exist_ok=True)
        src_of = dict(zip(man.file_name, man.src_split))
        for im in gt["images"]:
            shutil.copyfile(os.path.join(ds._split_dir(raw_root, src_of[im["file_name"]]), im["file_name"]),
                            os.path.join(dst, im["file_name"]))
        gt["info"] = dict(description=f"satellite plane/ship/vehicle - {split_name} split - {sp}")
        with open(os.path.join(dst, cfg.ANN_FILE), "w") as f:
            json.dump(gt, f)
        counts[folder] = dict(images=len(gt["images"]), annotations=len(gt["annotations"]),
                              **collections.Counter(c["name"] for a in gt["annotations"]
                                                    for c in gt["categories"] if c["id"] == a["category_id"]))
    shutil.copyfile(os.path.join(sdir, "manifest.csv"), os.path.join(out, "split_manifest.csv"))
    if readme_text:
        with open(os.path.join(out, "README.dataset.md"), "w") as f:
            f.write(readme_text)
    z = out + ".zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for r, _, files in os.walk(out):
            for fn in files:
                zf.write(os.path.join(r, fn), os.path.relpath(os.path.join(r, fn), out))
    print(pd.DataFrame(counts).T.to_string())
    print("upload folder:", out, "\nzip:", z, f"({os.path.getsize(z) / 1e6:.1f} MB)")
    return out, z, counts


def datacard(split_name="grouped", license_text="<LICENSE - check the licence of the source imagery first>",
             citation="<paper citation once accepted>", source_text="<where the original images come from>"):
    """Copy-paste project description built ONLY from verified numbers in split_meta.json."""
    sdir = ds.find_split_dir(split_name)
    meta = utils.load_json(os.path.join(sdir, "split_meta.json"))
    s = meta["summary"]
    leak = meta.get("leakage", {})
    tot_img = sum(v["images"] for v in s.values())
    tot_ann = sum(v["annotations"] for v in s.values())
    cls = {c: sum(v.get(c, 0) for v in s.values()) for c in cfg.CLASSES}
    native = meta.get("dataset") == "native"
    import collections
    sizes = collections.Counter()
    for sp in ("train", "val", "test"):
        for im in utils.load_json(os.path.join(sdir, "coco", f"{sp}.json"))["images"]:
            sizes[(im["width"], im["height"])] += 1
    size_txt = "640 x 640 px"
    if native:
        size_txt = "original screenshot size: " + ", ".join(f"{w} x {h} px ({n} images)" for (w, h), n in sizes.most_common(3))
    lines = [
        "# Satellite Plane / Ship / Vehicle Detection",
        "",
        "Object detection of **planes, ships and vehicles** in satellite imagery, dominated by small and densely "
        "packed objects. Companion dataset of the paper *SAHI-Augmented Multi-Architecture Satellite Image "
        "Object Detection*.",
        "",
        f"* Images: **{tot_img}** ({size_txt}), annotations: **{tot_ann}** axis-aligned boxes",
        "* Classes: " + ", ".join(f"`{c}` ({n})" for c, n in cls.items()),
        "* Format: COCO JSON (`bbox` = [x, y, width, height] in pixels); YOLO export available from Roboflow",
        f"* Split ({split_name}, fingerprint `{meta['fingerprint']}`): "
        + ", ".join(f"{k} {v['images']} images / {v['annotations']} boxes / {v['groups']} ground areas" for k, v in s.items()),
        "",
        "## How the split was made",
        "Images that show the same ground area were grouped (Roboflow augmented copies of one source image, "
        "byte-identical files, perceptual-hash near-duplicates under flips/rotations, and tiles that overlap "
        "according to ORB feature matching with a RANSAC homography). Whole groups were assigned to train, "
        "validation or test, so no ground area occurs in two splits. "
        + ("Every split holds exactly one image per source screenshot (no augmented copies)." if native else
           "Validation and test keep one image per source image; train keeps the offline-augmented copies."),
        "",
        "## Pre-processing and augmentation",
        ("Original resolution, auto-orient and contrast stretching only, no resizing and no augmentation (users apply "
         "their own). The split assignment is identical to the 640 x 640 version used in the paper." if native else
         "Auto-orient, stretch to 640 x 640, contrast stretching; training images include 3 augmented copies per "
         "source (horizontal/vertical flip, 90-degree rotation, +/-15 degree rotation, +/-10% brightness and "
         "exposure).")
        + " Boxes were clipped to the image, boxes under 1 px and exact duplicate boxes were removed "
        "(every change is listed in `clean_log.csv`).",
        "",
        "## Known limitations",
        "* Source/geographic metadata is not available, so spatial independence is approximated by the "
        "image-similarity grouping above, not proven.",
    ] + ([] if native else ["* +/-15 degree rotated training copies have slightly loose boxes (Roboflow enlarges rotated boxes)."]) + [
        "* Class imbalance: vehicles dominate.",
    ]
    if leak:
        lines += ["", "## Audit of the earlier random split",
                  f"Groups spanning more than one split in the earlier version: {leak.get('groups_spanning_splits')} "
                  f"of {leak.get('groups_total')}."]
    lines += ["", "## Source and licence", source_text, "", license_text, "", "## Citation", citation, ""]
    return "\n".join(lines)
