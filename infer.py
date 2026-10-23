"""Test-set evaluation of one trained model: Ultralytics metrics, then standard vs SAHI predictions
scored with COCO-style AP per size range, TP/FP/FN at the operating point, and measured latency.

Everything is written to results/<run_id>/ :
  eval_test.json           all numbers for the paper
  dets_<mode>.json.gz      raw detections (conf >= PRED_CONF) for later re-analysis / figures
  perimg_<mode>.csv        per-image TP/FP/FN by size bucket (for bootstrap CIs)
"""
import os, time, math

import numpy as np
import pandas as pd

import config as cfg
import utils
import metrics


def load_model(model_key, weights):
    from ultralytics import YOLO, RTDETR
    fam = cfg.MODELS[model_key]["family"]
    return RTDETR(weights) if fam == "rtdetr" else YOLO(weights)


def model_cost(model):
    """(params in millions, GFLOPs at 640) from Ultralytics; None if unavailable."""
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    gflops = None
    try:
        info = model.info(verbose=False)
        if isinstance(info, (tuple, list)) and len(info) >= 4:
            gflops = float(info[3])
    except Exception:
        pass
    return float(params), gflops


def n_slices(h, w, s, overlap=cfg.SAHI_OVERLAP):
    try:
        from sahi.slicing import get_slice_bboxes
        return len(get_slice_bboxes(h, w, s, s, overlap_height_ratio=overlap, overlap_width_ratio=overlap))
    except Exception:
        step = int(s * (1 - overlap))
        return math.ceil(max(h - s, 0) / step + 1) * math.ceil(max(w - s, 0) / step + 1)


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def ultralytics_test_metrics(model, data_yaml, device, project, imgsz=cfg.IMGSZ):
    r = model.val(data=data_yaml, split="test", imgsz=imgsz, batch=8 if imgsz <= 640 else 4, conf=cfg.EVAL_CONF, iou=cfg.NMS_IOU,
                  max_det=cfg.MAX_DET, device=device, plots=False, verbose=False, project=project,
                  name="ultra_test", exist_ok=True)
    out = dict(mAP50=float(r.box.map50), mAP50_95=float(r.box.map), P=float(r.box.mp), R=float(r.box.mr))
    try:
        ap50 = dict(zip([int(i) for i in r.box.ap_class_index], [float(v) for v in r.box.ap50]))
        ap = dict(zip([int(i) for i in r.box.ap_class_index], [float(v) for v in r.box.ap]))
        for k, n in enumerate(cfg.CLASSES):
            out[f"AP50_{n}"] = ap50.get(k, float("nan"))
            out[f"AP50_95_{n}"] = ap.get(k, float("nan"))
    except Exception as e:
        print("per-class AP not available:", e)
    try:
        out["speed_ms"] = {k: float(v) for k, v in r.speed.items()}
    except Exception:
        pass
    return out


def predict_standard(model, path, image_id, device, imgsz=cfg.IMGSZ, augment=False):
    r = model.predict(path, imgsz=imgsz, conf=cfg.PRED_CONF, iou=cfg.NMS_IOU, max_det=cfg.MAX_DET,
                      device=device, verbose=False, augment=augment)[0]
    b = r.boxes
    xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
    conf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
    cls = b.cls.cpu().numpy().astype(int) if len(b) else np.zeros(0, int)
    names = r.names
    dets = [dict(image_id=int(image_id), category_id=cfg.CLASSES.index(names[int(c)]) + 1,
                 bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)], score=float(s))
            for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls)]
    return dets, getattr(r, "speed", None)


def sahi_model(model, weights, device, imgsz=cfg.IMGSZ, conf=cfg.PRED_CONF):
    from sahi import AutoDetectionModel
    kw = dict(model_type="ultralytics", confidence_threshold=conf, device=device, image_size=imgsz)
    try:
        return AutoDetectionModel.from_pretrained(model=model, **kw)     # reuse the loaded YOLO / RTDETR object
    except Exception as e:
        print("SAHI with a model object failed (", e, ") -> loading from the weights path")
        return AutoDetectionModel.from_pretrained(model_path=weights, **kw)


def _sahi_has_force():
    try:
        import inspect
        from sahi.predict import get_sliced_prediction
        return "force_postprocess_type" in inspect.signature(get_sliced_prediction).parameters
    except Exception:
        return False


_SAHI_HAS_FORCE = None


def predict_sahi(dm, path, image_id, s, post=None):
    """post: dict with postprocess_* keys, perform_standard_pred and overlap (default = paper setting)."""
    from sahi.predict import get_sliced_prediction
    global _SAHI_HAS_FORCE
    if _SAHI_HAS_FORCE is None:
        _SAHI_HAS_FORCE = _sahi_has_force()
        if not _SAHI_HAS_FORCE:
            print("WARNING: this SAHI version has no force_postprocess_type -> GREEDYNMM/NMM may be replaced by NMS")
    post = dict(post or dict(cfg.SAHI_POSTPROCESS, overlap=cfg.SAHI_OVERLAP))
    ov = post.pop("overlap", cfg.SAHI_OVERLAP)
    post.pop("model_conf", None)
    # SAHI silently replaces GREEDYNMM / NMM by NMS-IoU when the model confidence threshold is < 0.1
    # (ours is 0.01 for AP) unless force_postprocess_type=True -> always force the requested merge rule.
    if _SAHI_HAS_FORCE:
        post["force_postprocess_type"] = True
    r = get_sliced_prediction(path, dm, slice_height=s, slice_width=s, overlap_height_ratio=ov,
                              overlap_width_ratio=ov, verbose=0, **post)
    out = []
    for p in r.object_prediction_list:
        x1, y1, x2, y2 = p.bbox.to_xyxy()
        out.append(dict(image_id=int(image_id), category_id=cfg.CLASSES.index(p.category.name) + 1,
                        bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)], score=float(p.score.value)))
    return out


def _run_mode(model, dm_getter, mode, imgs, device, imgsz, post=None):
    """Predict every image in one mode; returns (dets, per-image seconds, slice size or None)."""
    if mode == "standard":
        run, s = (lambda p, i: predict_standard(model, p, i, device, imgsz)[0]), None
    elif mode == "tta":            # Ultralytics test-time augmentation: flips + 3 scales, merged by NMS
        run, s = (lambda p, i: predict_standard(model, p, i, device, imgsz, augment=True)[0]), None
    elif mode.startswith("res"):   # same weights, larger input at inference only
        r_ = int(mode[3:])
        run, s = (lambda p, i: predict_standard(model, p, i, device, r_)[0]), None
    else:
        s = int(mode.replace("sahi", ""))
        dm = dm_getter()
        run = lambda p, i: predict_sahi(dm, p, i, s, post)
    for (i, p, _, _) in imgs[:cfg.WARMUP_IMAGES]:
        run(p, i)
    dets, times = [], []
    for (i, p, _, _) in imgs:
        _sync()
        t0 = time.perf_counter()
        dets += run(p, i)
        _sync()
        times.append(time.perf_counter() - t0)
    return dets, times, s


def _passes(mode, imgs, s, post=None, imgsz=cfg.IMGSZ):
    """Forward passes per image in units of one pass at the training size (used for GFLOPs/image)."""
    if mode == "standard":
        return 1.0
    if mode == "tta":
        return 1.0 + 0.83 ** 2 + 0.67 ** 2       # Ultralytics TTA scales 1, 0.83, 0.67
    if mode.startswith("res"):
        return (int(mode[3:]) / imgsz) ** 2
    post = post or dict(cfg.SAHI_POSTPROCESS, overlap=cfg.SAHI_OVERLAP)
    return float(np.mean([n_slices(h, w, s, post.get("overlap", cfg.SAHI_OVERLAP)) for (_, _, h, w) in imgs])) + \
        int(post.get("perform_standard_pred", True))


def _test_images(split_info):
    gt = utils.load_json(split_info["gt"]["test"])
    img_dir = split_info["images"]["test"]
    return gt, [(im["id"], os.path.join(img_dir, im["file_name"]), im["height"], im["width"]) for im in gt["images"]]


def _free(model):
    del model
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def evaluate_run(run_id, model_key, weights, split_info, meta=None, modes=None, device=None, compare_engines=False):
    """Full test-set evaluation of one weights file. Returns the eval dict (also saved)."""
    modes = modes or cfg.modes_for(split_info["name"])
    device = device or utils.gpu_info()["device"]
    imgsz = cfg.model_imgsz(model_key)
    out_dir = os.path.join(cfg.RESULTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    gt, imgs = _test_images(split_info)
    model = load_model(model_key, weights)
    params, gflops640 = model_cost(model)
    gflops = gflops640 * (imgsz / 640) ** 2 if gflops640 else None      # every pass runs at imgsz
    res = dict(run_id=run_id, model=model_key, split=split_info["name"], split_fingerprint=split_info["fingerprint"],
               weights=os.path.basename(weights), n_test_images=len(imgs), n_test_boxes=len(gt["annotations"]),
               params_M=params, gflops=gflops, imgsz=imgsz, device=utils.gpu_info(), versions=utils.versions(),
               config=dict(EVAL_CONF=cfg.EVAL_CONF, PRED_CONF=cfg.PRED_CONF, OP_CONF=cfg.OP_CONF,
                           MATCH_IOU=cfg.MATCH_IOU, MAX_DET=cfg.MAX_DET, SAHI_OVERLAP=cfg.SAHI_OVERLAP,
                           SAHI_POSTPROCESS=cfg.SAHI_POSTPROCESS, AREA_RANGES=cfg.AREA_RANGES),
               train_meta=meta or {})
    if split_info.get("yaml"):
        with utils.timer(f"{run_id} ultralytics test metrics"):
            res["ultralytics_test"] = ultralytics_test_metrics(model, split_info["yaml"], device, out_dir, imgsz)
    else:
        res["ultralytics_test"] = {}
    res["modes"] = {}
    cache = {}
    get_dm = lambda: cache.setdefault("dm", sahi_model(model, weights, device, imgsz))
    for mode in modes:
        dets, times, s = _run_mode(model, get_dm, mode, imgs, device, imgsz)
        ms = 1000 * float(np.mean(times))
        tiles = _passes(mode, imgs, s, imgsz=imgsz)
        coco = metrics.coco_eval(gt, dets)
        if compare_engines and coco.get("engine") == "pycocotools":
            internal = metrics.coco_eval(gt, dets, prefer="internal")
            res.setdefault("engine_check", {})[mode] = {k: (coco[k], internal[k]) for k in ("AP_all", "AP50_all", "AP_small")}
            print(f"  pycocotools vs internal AP ({mode}):", res["engine_check"][mode])
        per = metrics.match_per_image(gt, dets)
        per.to_csv(os.path.join(out_dir, f"perimg_{mode}.csv"), index=False)
        utils.save_json(dets, os.path.join(out_dir, f"dets_{mode}.json.gz"))
        res["modes"][mode] = dict(ms_per_image=ms, ms_p50=1000 * float(np.median(times)), fps_end_to_end=1000 / ms,
                                  forward_passes_per_image=tiles,
                                  gflops_per_image=(gflops * tiles if gflops else None),
                                  coco=coco, op=metrics.totals(per))
        o, c = res["modes"][mode]["op"], coco
        print(f"  {run_id:<28} {mode:<9} AP50={c.get('AP50_all', float('nan')):.4f} AP={c.get('AP_all', float('nan')):.4f} "
              f"AR50_micro={c.get('AR50_micro', float('nan')):.3f} | conf>={cfg.OP_CONF}: dets={o['dets']} "
              f"TP={o['TP']} FP={o['FP']} R={o['R']:.3f} | {ms:.1f} ms/img")
    utils.save_json(res, os.path.join(out_dir, "eval_test.json"))
    _free(model)
    return res


def sahi_ablation(run_id, model_key, weights, split_info, configs=None, slices=None, device=None, previous=None):
    """Every SAHI post-processing variant in cfg.SAHI_ABLATION x slice size on the test split.
    Writes results/<run_id>/sahi_ablation.csv (+ per-image TP/FP for each row)."""
    configs = configs or cfg.SAHI_ABLATION
    slices = slices or [int(m[4:]) for m in cfg.modes_for(split_info["name"]) if m != "standard"]
    device = device or utils.gpu_info()["device"]
    imgsz = cfg.model_imgsz(model_key)
    out_dir = os.path.join(cfg.RESULTS_DIR, run_id)
    os.makedirs(os.path.join(out_dir, "ablation"), exist_ok=True)
    gt, imgs = _test_images(split_info)
    model = load_model(model_key, weights)
    dms = {}
    rows = []
    done = set() if previous is None else set(zip(previous["slice"], previous["config"]))
    for s in slices:
        for name, post in configs.items():
            if (s, name) in done:
                continue
            mc = post.get("model_conf", cfg.PRED_CONF)
            dm = dms.setdefault(mc, sahi_model(model, weights, device, imgsz, mc))
            dets, times, _ = _run_mode(model, lambda: dm, f"sahi{s}", imgs, device, imgsz, dict(post))
            per = metrics.match_per_image(gt, dets)
            per.to_csv(os.path.join(out_dir, "ablation", f"perimg_sahi{s}_{name}.csv"), index=False)
            c, o = metrics.coco_eval(gt, dets), metrics.totals(per)
            rows.append(dict(run_id=run_id, model=model_key, split=split_info["name"],
                             split_fingerprint=split_info["fingerprint"], slice=s, config=name,
                             postprocess=post["postprocess_type"], metric=post["postprocess_match_metric"],
                             forced=bool(_SAHI_HAS_FORCE), model_conf=post.get("model_conf", cfg.PRED_CONF),
                             overlap=post["overlap"], full_image_pass=post["perform_standard_pred"],
                             passes=_passes(f"sahi{s}", imgs, s, post), ms_per_image=1000 * float(np.mean(times)),
                             **{k: v for k, v in c.items() if k.startswith(("AP", "AR"))},
                             **{f"op_{k}": v for k, v in o.items()}))
            print(f"  {run_id} sahi{s:<4} {name:<14} AP={c.get('AP_all', float('nan')):.4f} AP50={c.get('AP50_all', float('nan')):.4f}"
                  f" TP={o['TP']} FP={o['FP']} dupFP={o['FP_duplicate']}")
    df = pd.DataFrame(rows)
    if previous is not None and len(previous):
        df = pd.concat([previous, df], ignore_index=True)
    df.to_csv(os.path.join(out_dir, "sahi_ablation.csv"), index=False)
    _free(model)
    return df


def extra_modes(run_id, model_key, weights, split_info, modes=cfg.EXTRA_MODES, device=None):
    """Notebook 08D: TTA and inference-only higher resolution for an existing run, same scoring as evaluate_run.
    Writes results/<run_id>/eval_extra.json (+ dets/perimg files); report.collect_runs merges it into the run."""
    device = device or utils.gpu_info()["device"]
    imgsz = cfg.model_imgsz(model_key)
    out_dir = os.path.join(cfg.RESULTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    gt, imgs = _test_images(split_info)
    model = load_model(model_key, weights)
    params, gflops640 = model_cost(model)
    gflops = gflops640 * (imgsz / 640) ** 2 if gflops640 else None
    res = dict(run_id=run_id, model=model_key, split=split_info["name"], split_fingerprint=split_info["fingerprint"],
               imgsz=imgsz, modes={})
    for mode in modes:
        if mode == "tta" and model_key.startswith("RT-DETR"):
            print(f"  {run_id}: TTA skipped (Ultralytics has no TTA for RT-DETR)")
            continue
        dets, times, s = _run_mode(model, None, mode, imgs, device, imgsz)
        ms = 1000 * float(np.mean(times))
        tiles = _passes(mode, imgs, s, imgsz=imgsz)
        coco = metrics.coco_eval(gt, dets)
        per = metrics.match_per_image(gt, dets)
        per.to_csv(os.path.join(out_dir, f"perimg_{mode}.csv"), index=False)
        utils.save_json(dets, os.path.join(out_dir, f"dets_{mode}.json.gz"))
        res["modes"][mode] = dict(ms_per_image=ms, ms_p50=1000 * float(np.median(times)), fps_end_to_end=1000 / ms,
                                  forward_passes_per_image=tiles, gflops_per_image=(gflops * tiles if gflops else None),
                                  coco=coco, op=metrics.totals(per))
        o, c = res["modes"][mode]["op"], coco
        print(f"  {run_id:<28} {mode:<9} AP50={c.get('AP50_all', float('nan')):.4f} AP={c.get('AP_all', float('nan')):.4f} "
              f"AP_small={c.get('AP_small', float('nan')):.4f} | TP={o['TP']} FP={o['FP']} R={o['R']:.3f} | {ms:.1f} ms/img")
    utils.save_json(res, os.path.join(out_dir, "eval_extra.json"))
    _free(model)
    return res
