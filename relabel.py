"""Notebook 13: corrected test labels -> re-score every saved detection file (CPU, no retraining).

Step A  export_for_relabel(): the 82 grouped test images + their current boxes as a Roboflow-ready COCO zip.
Step B  (you) Roboflow: upload the zip WITH annotations, add the missing boxes, export COCO (no preprocessing,
        no augmentation), upload the unzipped export to Kaggle as 'sat-test-relabeled'.
Step C  apply(): match the exported images back to the test images (by the original Roboflow hash in the file
        name, else by image similarity), write corrected test GT for grouped / grouped_1copy / grouped_lc*
        (same image ids -> paired comparisons stay valid), and re-score every saved dets_<mode> file.
Train labels are NOT changed (that would need retraining); only the evaluation uses the corrected GT."""
import os, re, shutil, zipfile, collections, copy

import numpy as np
import pandas as pd

import config as cfg
import utils
import metrics
import dataset as ds

HEX = re.compile(r"[0-9a-f]{32}")
SPLITS_USING_TEST = ["grouped", "grouped_1copy", "grouped_lc25", "grouped_lc50"]


def export_for_relabel(out=None):
    out = out or os.path.join(cfg.WORK_DIR, "test_relabel_upload")
    sdir = ds.find_split_dir("grouped")
    gt = utils.load_json(os.path.join(sdir, "coco", "test.json"))
    info = ds.materialize("grouped")
    d = os.path.join(out, "test")
    os.makedirs(d, exist_ok=True)
    for im in gt["images"]:
        shutil.copyfile(os.path.join(info["images"]["test"], im["file_name"]), os.path.join(d, im["file_name"]))
    utils.save_json(gt, os.path.join(d, "_annotations.coco.json"))
    z = shutil.make_archive(out, "zip", out)
    shutil.rmtree(info["root"], ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    print(f"{len(gt['images'])} test images, {len(gt['annotations'])} current boxes -> {z}")
    return z


def _find_export():
    files = [p for p in utils.find_files(cfg.ANN_FILE) if "relabel" in p.lower()]
    if not files:
        raise FileNotFoundError("attach the Roboflow re-export as a Dataset whose name contains 'relabel' "
                                "(e.g. sat-test-relabeled)")
    return files


def _thumb(path):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def load_relabeled(gt_old, test_img_dir):
    """Corrected boxes per ORIGINAL image id, in the original pixel frame."""
    by_hash = {}
    for im in gt_old["images"]:
        h = HEX.findall(im["file_name"])
        if h:
            by_hash[h[0]] = im
    olds = {im["id"]: im for im in gt_old["images"]}
    thumbs = None
    new_boxes, matched, how = collections.defaultdict(list), {}, collections.Counter()
    for f in _find_export():
        c = utils.load_json(f)
        cmap = {}
        for cat in c["categories"]:
            n = cfg.CLASS_ALIASES.get(str(cat["name"]).strip().lower(), str(cat["name"]).strip().lower())
            if n in cfg.CLASSES:
                cmap[cat["id"]] = cfg.CLASSES.index(n) + 1
        anns = collections.defaultdict(list)
        for a in c["annotations"]:
            anns[a["image_id"]].append(a)
        for im in c["images"]:
            old = next((by_hash[h] for h in HEX.findall(im["file_name"]) if h in by_hash), None)
            if old is not None:
                how["file-name hash"] += 1
            else:                                        # fall back: most similar test image
                if thumbs is None:
                    thumbs = {i: _thumb(os.path.join(test_img_dir, o["file_name"])) for i, o in olds.items()}
                p = os.path.join(os.path.dirname(f), im["file_name"])
                if not os.path.isfile(p):
                    continue
                t = _thumb(p)
                i, err = min(((i, float(np.mean((t - v) ** 2))) for i, v in thumbs.items()), key=lambda x: x[1])
                if err > 0.01:
                    print("not matched (skipped):", im["file_name"], round(err, 4)); continue
                old = olds[i]; how["image similarity"] += 1
            sx, sy = old["width"] / im["width"], old["height"] / im["height"]
            matched[old["id"]] = im["file_name"]
            for a in anns[im["id"]]:
                if a["category_id"] not in cmap:
                    continue
                x, y, w, h = a["bbox"]
                x, y, w, h = x * sx, y * sy, w * sx, h * sy
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(old["width"], x + w), min(old["height"], y + h)
                if x2 - x1 >= 1 and y2 - y1 >= 1:
                    new_boxes[old["id"]].append((cmap[a["category_id"]], x1, y1, x2 - x1, y2 - y1))
    print("matched test images:", len(matched), "of", len(olds), dict(how))
    missing = set(olds) - set(matched)
    assert not missing, f"{len(missing)} test images were not found in the re-export: {[olds[i]['file_name'] for i in list(missing)[:5]]}"
    return new_boxes


def corrected_gt(gt_old, new_boxes):
    g = copy.deepcopy(gt_old)
    anns = []
    for im in g["images"]:
        seen = set()
        for (k, x, y, w, h) in new_boxes.get(im["id"], []):
            key = (k, round(x, 1), round(y, 1), round(w, 1), round(h, 1))
            if key in seen:
                continue
            seen.add(key)
            anns.append(dict(id=len(anns) + 1, image_id=im["id"], category_id=k, bbox=[x, y, w, h], area=w * h, iscrowd=0))
    g["annotations"] = anns
    g["info"] = dict(description="grouped/test - corrected labels (notebook 13)")
    return g


def change_report(gt_old, gt_new):
    def cnt(g):
        c = collections.Counter()
        for a in g["annotations"]:
            b = metrics._bucket(a["area"])
            c[(cfg.CLASSES[a["category_id"] - 1], b)] += 1
        return c
    o, n = cnt(gt_old), cnt(gt_new)
    rows = [dict(cls=k[0], size=k[1], before=o[k], after=n[k], added=n[k] - o[k]) for k in sorted(set(o) | set(n))]
    t = pd.DataFrame(rows)
    tot = dict(cls="all", size="all", before=len(gt_old["annotations"]), after=len(gt_new["annotations"]),
               added=len(gt_new["annotations"]) - len(gt_old["annotations"]))
    return pd.concat([t, pd.DataFrame([tot])], ignore_index=True)


def write_corrected_splits(gt_new, out_root=None):
    """Copies of the split folders that share the grouped test images, with corrected test GT + YOLO labels."""
    out_root = out_root or os.path.join(cfg.WORK_DIR, "splits_relabeled")
    done = []
    for name in SPLITS_USING_TEST:
        try:
            src = ds.find_split_dir(name)
        except FileNotFoundError:
            continue
        dst = os.path.join(out_root, name)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        utils.save_json(gt_new, os.path.join(dst, "coco", "test.json"))
        ldir = os.path.join(dst, "labels", "test")
        by = collections.defaultdict(list)
        for a in gt_new["annotations"]:
            by[a["image_id"]].append(a)
        for im in gt_new["images"]:
            W, H = im["width"], im["height"]
            with open(os.path.join(ldir, os.path.splitext(im["file_name"])[0] + ".txt"), "w") as f:
                for a in by[im["id"]]:
                    x, y, w, h = a["bbox"]
                    f.write(f"{a['category_id'] - 1} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
        meta = utils.load_json(os.path.join(dst, "split_meta.json"))
        meta["label_version"] = 2
        meta["summary"]["test"]["annotations"] = len(gt_new["annotations"])
        utils.save_json(meta, os.path.join(dst, "split_meta.json"))
        done.append(name)
    print("corrected split folders:", done, "->", out_root)
    return out_root


def _rescore_modes(modes, folder, gt, out_folder=None):
    out = {}
    for mode, m in modes.items():
        p = utils.dets_path(m.get("_folder", folder), mode)
        if not p:
            continue
        dets = utils.load_any_json(p)
        per = metrics.match_per_image(gt, dets)
        if out_folder:
            os.makedirs(out_folder, exist_ok=True)
            per.to_csv(os.path.join(out_folder, f"perimg_{mode}.csv"), index=False)
        out[mode] = dict(m, coco=metrics.coco_eval(gt, dets), op=metrics.totals(per))
        out[mode].pop("_folder", None)
    return out


def rescore_all(gt_new, out_dir=None):
    """Every run evaluated on the grouped test images -> results_relabeled/<run_id>/eval_test.json (+ dets copied)."""
    out_dir = out_dir or os.path.join(cfg.WORK_DIR, "results_relabeled")
    found = collections.defaultdict(list)
    for p in utils.find_files("eval_test.json"):
        d = utils.load_json(p)
        if d.get("split") in SPLITS_USING_TEST and not d["run_id"].startswith(("xfer", "xdota")) \
                and "__pilot" not in d["run_id"]:
            found[d["run_id"]].append((utils.dets_path(os.path.dirname(p), "standard") is not None, p, d))
    extra = collections.defaultdict(list)
    for p in utils.find_files("eval_extra.json"):
        e = utils.load_json(p)
        extra[e["run_id"]].append((os.path.dirname(p), e))
    rows = []
    for rid, lst in sorted(found.items()):
        lst.sort(key=lambda t: t[0])
        has, p, d = lst[-1]
        folder = os.path.dirname(p)
        new = copy.deepcopy(d)
        new["modes"] = _rescore_modes(d["modes"], folder, gt_new, os.path.join(out_dir, rid))
        if not new["modes"]:
            print("no saved detections, skipped:", rid); continue
        new["ultralytics_test"] = {}                 # Ultralytics numbers were computed with the old labels
        new["label_version"] = 2
        od = os.path.join(out_dir, rid)
        os.makedirs(od, exist_ok=True)
        for mode in new["modes"]:
            src = utils.dets_path(folder, mode)
            shutil.copyfile(src, os.path.join(od, f"dets_{mode}.json" + (".gz" if src.endswith(".gz") else "")))
        utils.save_json(new, os.path.join(od, "eval_test.json"))
        for efolder, e in extra.get(rid, []):         # 08D modes (TTA, inference at 1024)
            em = _rescore_modes({k: dict(v, _folder=efolder) for k, v in e["modes"].items()}, efolder, gt_new,
                                os.path.join(out_dir, rid + "__extra"))
            if em:
                ed = os.path.join(out_dir, rid + "__extra")
                os.makedirs(ed, exist_ok=True)
                for mode in em:
                    src = utils.dets_path(efolder, mode)
                    shutil.copyfile(src, os.path.join(ed, f"dets_{mode}.json" + (".gz" if src.endswith(".gz") else "")))
                utils.save_json(dict(e, modes=em, label_version=2), os.path.join(ed, "eval_extra.json"))
        for mode, m in new["modes"].items():
            om = d["modes"][mode]
            rows.append(dict(run_id=rid, split=d["split"], model=d["model"], mode=mode,
                             AP_old=om["coco"].get("AP_all"), AP_new=m["coco"].get("AP_all"),
                             P_old=om["op"]["P"], P_new=m["op"]["P"], R_old=om["op"]["R"], R_new=m["op"]["R"],
                             FP_old=om["op"]["FP"], FP_new=m["op"]["FP"]))
    t = pd.DataFrame(rows)
    print(f"re-scored {t.run_id.nunique() if len(t) else 0} runs / {len(t)} run-modes -> {out_dir}")
    return t
