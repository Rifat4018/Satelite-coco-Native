"""Training one (split, model, seed) run with the fixed recipe in config.TRAIN_PARAMS, then the test
evaluation from infer.py. Finished runs (found in the inputs or the working folder) are skipped,
so a notebook can simply be run again after a session limit."""
import os, glob, shutil, time

import pandas as pd

import config as cfg
import utils
import infer

# planning numbers from YOUR logs (T4, 50 epochs, ~1.3k train images) + ~10 min evaluation
EXPECTED_H = {"YOLO11n": 0.45, "YOLO11s": 0.55, "YOLO11m": 1.0, "YOLOv8m": 1.0, "RT-DETR-L": 2.1,
              "YOLOv8s": 0.55, "YOLOv9s": 0.7, "YOLOv10s": 0.6, "YOLO12s": 0.7, "YOLO26s": 0.6,
              "YOLOv9m": 1.2, "YOLOv10m": 1.1, "YOLO12m": 1.3, "YOLO26m": 1.1,
              "YOLOv8s-P2": 0.8, "YOLOv8m-P2": 1.4,
              "YOLO11s-1024": 1.0, "YOLO11s-1280": 1.4, "YOLO11m-1024": 2.1, "YOLO11m-1280": 3.3}
# multipliers by split (train-set size relative to `grouped`)
SPLIT_FACTOR = {"grouped_1copy": 0.4, "grouped_lc25": 0.35, "grouped_lc50": 0.6, "native_grouped": 0.5,
                "native_tiles": 2.5, "sky_grouped": 1.0}


def expected_h(split, model_key):
    return EXPECTED_H.get(model_key, 2.0) * SPLIT_FACTOR.get(split, 1.0)


def run_id(split, model_key, seed, tag=""):
    return f"{split}__{model_key}__s{seed}" + (f"__{tag}" if tag else "")


def find_existing(rid):
    """eval_test.json and best.pt of a finished run anywhere under inputs / working dir."""
    ev = [p for p in utils.find_files("eval_test.json", also_work=True) if os.path.basename(os.path.dirname(p)) == rid]
    wt = [p for p in utils.find_files(f"{rid}_best.pt", also_work=True)]
    return (ev[0] if ev else None), (wt[0] if wt else None)


def train_one(split_info, model_key, seed, epochs=None, device=None, tag=""):
    from ultralytics import YOLO, RTDETR
    spec = cfg.MODELS[model_key]
    rid = run_id(split_info["name"], model_key, seed, tag)
    project = os.path.join(cfg.WORK_DIR, "runs")
    args = dict(cfg.TRAIN_PARAMS)
    args.update(data=split_info["yaml"], batch=spec["batch"], imgsz=cfg.model_imgsz(model_key), seed=seed,
                project=project, name=rid,
                exist_ok=True, device=0 if utils.gpu_info()["n_gpu"] else "cpu")
    if epochs:
        args["epochs"] = epochs
    M = RTDETR if spec["family"] == "rtdetr" else YOLO
    model = M(spec["yaml"]).load(spec["weights"]) if spec.get("yaml") else M(spec["weights"])
    utils.set_seed(seed)
    t0 = time.time()
    try:
        model.train(**args)
    except Exception as e:                       # CUDA out of memory at high resolution -> half the batch once
        if "out of memory" not in str(e).lower() or args["batch"] <= 2:
            raise
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
        args["batch"] = max(2, args["batch"] // 2)
        print(f"CUDA OOM -> retrying {rid} with batch {args['batch']} (nominal batch 64 is kept by gradient accumulation)")
        model = RTDETR(spec["weights"]) if spec["family"] == "rtdetr" else (
            YOLO(spec["yaml"]).load(spec["weights"]) if spec.get("yaml") else YOLO(spec["weights"]))
        t0 = time.time()
        model.train(**args)
    train_s = time.time() - t0
    save_dir = os.path.join(project, rid)
    best = os.path.join(save_dir, "weights", "best.pt")
    assert os.path.isfile(best), f"training produced no best.pt in {save_dir}"
    os.makedirs(cfg.WEIGHTS_DIR, exist_ok=True)
    wpath = os.path.join(cfg.WEIGHTS_DIR, f"{rid}_best.pt")
    shutil.copyfile(best, wpath)
    rdir = os.path.join(cfg.RESULTS_DIR, rid)
    os.makedirs(rdir, exist_ok=True)
    for fn in ["results.csv", "args.yaml"]:
        if os.path.isfile(os.path.join(save_dir, fn)):
            shutil.copyfile(os.path.join(save_dir, fn), os.path.join(rdir, fn))
    for png in glob.glob(os.path.join(save_dir, "*.png")):
        shutil.copyfile(png, os.path.join(rdir, os.path.basename(png)))
    hist = pd.read_csv(os.path.join(save_dir, "results.csv")) if os.path.isfile(os.path.join(save_dir, "results.csv")) else None
    meta = dict(run_id=rid, model=model_key, seed=seed, split=split_info["name"],
                split_fingerprint=split_info["fingerprint"], train_seconds=train_s,
                epochs_run=int(len(hist)) if hist is not None else None,
                best_epoch_by_val_map50_95=(int(hist.loc[hist.filter(like="mAP50-95").iloc[:, 0].idxmax(), "epoch"])
                                            if hist is not None and hist.filter(like="mAP50-95").shape[1] else None),
                train_args={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in args.items()},
                gpu=utils.gpu_info(), versions=utils.versions())
    utils.save_json(meta, os.path.join(rdir, "train_meta.json"))
    print(f"trained {rid} in {train_s / 3600:.2f} h -> {wpath}")
    del model
    return wpath, meta


def run_plan(plan, eval_after=True, epochs=None, tag=""):
    """plan: list of (split_name, model_key, seed). Materializes each split once, trains, evaluates.
    tag: e.g. 'pilot' -> run ids end in __pilot and are ignored by the paper tables."""
    import dataset as ds
    clock = utils.SessionClock()
    splits, done = {}, []
    for split, model_key, seed in plan:
        rid = run_id(split, model_key, seed, tag)
        ev, wt = find_existing(rid)
        if ev:
            print(f"SKIP {rid}: already evaluated ({ev})")
            done.append((rid, "skipped"))
            continue
        if split not in splits:
            splits[split] = ds.materialize(split)
        info = splits[split]
        meta = {}
        if wt is None:
            if not clock.can_start(expected_h(split, model_key)):
                done.append((rid, "not started (session time)"))
                continue
            try:
                wt, meta = train_one(info, model_key, seed, epochs=epochs, tag=tag)
            except Exception as e:
                import traceback
                traceback.print_exc()
                done.append((rid, f"TRAIN FAILED: {e}"))
                continue
        else:
            print(f"{rid}: weights found ({wt}) -> evaluation only")
            mp = os.path.join(os.path.dirname(os.path.dirname(wt)), "results", rid, "train_meta.json")
            meta = utils.load_json(mp) if os.path.isfile(mp) else {}
        if eval_after:
            try:
                infer.evaluate_run(rid, model_key, wt, info, meta=meta, compare_engines=(not done))
                done.append((rid, "trained+evaluated"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                done.append((rid, f"EVAL FAILED: {e}"))
        else:
            done.append((rid, "trained"))
    print("\n=== session summary ===")
    for rid, st in done:
        print(f"  {rid:<30} {st}")
    print(f"session time used: {clock.elapsed_h():.2f} h")
    return done


def run_ablation(plan, slices=None):
    """SAHI post-processing ablation on finished runs (weights must be attached). No training."""
    import dataset as ds
    clock = utils.SessionClock()
    splits, done = {}, []
    for split, model_key, seed in plan:
        rid = run_id(split, model_key, seed)
        prev = [pd.read_csv(p) for p in utils.find_files("sahi_ablation.csv", also_work=True)
                if os.path.basename(os.path.dirname(p)) == rid]
        prev = [p for p in prev if "forced" in p.columns and p.forced.astype(bool).all()]
        previous = pd.concat(prev).drop_duplicates(["slice", "config"], keep="last") if prev else None
        need_slices = slices or [int(m[4:]) for m in cfg.modes_for(split) if m != "standard"]
        todo = {(s, c) for s in need_slices for c in cfg.SAHI_ABLATION}
        if previous is not None:
            todo -= set(zip(previous["slice"], previous["config"]))
        if not todo:
            print("SKIP (done):", rid)
            done.append((rid, "skipped"))
            continue
        if previous is not None:
            print(f"{rid}: {len(previous)} rows already done, running the {len(todo)} missing ones")
        _, wt = find_existing(rid)
        if wt is None:
            print("NO WEIGHTS attached for", rid, "-> attach the results Dataset of the notebook that trained it")
            done.append((rid, "no weights"))
            continue
        if not clock.can_start(0.4):
            done.append((rid, "not started (session time)"))
            continue
        if split not in splits:
            splits[split] = ds.materialize(split)
        try:
            import infer
            infer.sahi_ablation(rid, model_key, wt, splits[split], slices=slices, previous=previous)
            done.append((rid, "ablation done"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            done.append((rid, f"FAILED: {e}"))
    print("\n=== session summary ===")
    for rid, st in done:
        print(f"  {rid:<34} {st}")
    return done
