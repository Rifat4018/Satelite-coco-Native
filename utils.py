"""Small helpers shared by every notebook."""
import os, sys, json, time, gzip, random, hashlib, subprocess, importlib
from contextlib import contextmanager

import numpy as np

import config as cfg


def show_inputs(depth=4, max_files_per_dir=6, root=None):
    """Print what is mounted under /kaggle/input so you can confirm the inputs of THIS notebook."""
    root = root or cfg.INPUT_ROOT
    if not os.path.isdir(root):
        print("no input folder:", root)
        return
    for r, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        level = r.replace(root, "").count(os.sep)
        if level > depth:
            dirs[:] = []
            continue
        ind = "  " * level
        print(f"{ind}{r}/")
        for fn in sorted(files)[:max_files_per_dir]:
            print(f"{ind}  {fn}")
        if len(files) > max_files_per_dir:
            print(f"{ind}  ... +{len(files) - max_files_per_dir} more files")


def ensure_dirs(*dirs):
    for d in dirs or (cfg.RESULTS_DIR, cfg.WEIGHTS_DIR, cfg.FIG_DIR):
        os.makedirs(d, exist_ok=True)


def set_seed(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@contextmanager
def timer(label=""):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[timer] {label}: {dt / 60:.1f} min ({dt:.0f}s)")


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def save_json(obj, path, gz=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if gz or path.endswith(".gz"):
        with gzip.open(path, "wt") as f:
            json.dump(obj, f, default=_np_default)
    else:
        with open(path, "w") as f:
            json.dump(obj, f, indent=1, default=_np_default)


def load_json(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def dets_path(folder, mode):
    """Saved detections of one mode. Kaggle un-gzips *.gz when an output becomes a Dataset, so accept
    dets_<mode>.json.gz, dets_<mode>.json, or a folder of that name holding the extracted file."""
    for name in (f"dets_{mode}.json.gz", f"dets_{mode}.json"):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            inner = [os.path.join(p, f) for f in os.listdir(p) if not f.startswith(".")]
            if inner:
                return inner[0]
    return None


def load_any_json(path):
    """JSON that may or may not be gzip-compressed, whatever the file name says."""
    with open(path, "rb") as f:
        head = f.read(2)
    if head == b"\x1f\x8b":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_files(name, root=None, also_work=False):
    """All files called `name` under the inputs (and optionally the working folder)."""
    roots = [root or cfg.INPUT_ROOT] + ([cfg.WORK_DIR] if also_work else [])
    out = []
    for rt in roots:
        if not os.path.isdir(rt):
            continue
        for r, dirs, files in os.walk(rt):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if name in files:
                out.append(os.path.join(r, name))
    return sorted(set(out))


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return dict(device="cuda:0", gpu=torch.cuda.get_device_name(0),
                        n_gpu=torch.cuda.device_count(), torch=torch.__version__)
        return dict(device="cpu", gpu=None, n_gpu=0, torch=torch.__version__)
    except Exception:
        return dict(device="cpu", gpu=None, n_gpu=0, torch=None)


def warn_if_gpu_wasted():
    """CPU-only notebooks call this so you do not burn GPU quota by accident."""
    if gpu_info()["n_gpu"] > 0:
        print("WARNING: a GPU is attached but this notebook only needs CPU. "
              "Settings -> Accelerator -> None saves your weekly GPU quota.")


def require_gpu():
    info = gpu_info()
    if info["n_gpu"] == 0 and not cfg.SMOKE:
        raise RuntimeError("No GPU. Settings -> Accelerator -> GPU T4 x2 (or P100), then restart.")
    print("GPU:", info)
    return info


def pip_install(*specs, fallback=None):
    """pip install inside a notebook (no '!' magic so the same code runs in the smoke test)."""
    if cfg.SMOKE:
        print("[smoke] skip pip install", specs)
        return True
    cmd = [sys.executable, "-m", "pip", "install", "-q", *specs]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and fallback:
        print("pip failed for", specs, "-> trying", fallback)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *fallback],
                           capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return False
    importlib.invalidate_caches()
    return True


def install_detection_stack():
    """ultralytics (pinned to the version of your original runs) + sahi + pycocotools."""
    ok = True
    try:
        import ultralytics  # noqa
        print("ultralytics already installed:", ultralytics.__version__)
    except Exception:
        ok &= pip_install(cfg.ULTRALYTICS_PIN, fallback=["ultralytics"])
    for mod, spec in [("sahi", "sahi"), ("pycocotools", "pycocotools")]:
        try:
            importlib.import_module(mod)
        except Exception:
            ok &= pip_install(spec)
    return ok


def versions():
    out = {}
    for m in ["ultralytics", "sahi", "torch", "pycocotools", "numpy", "cv2"]:
        try:
            mod = importlib.import_module(m)
            out[m] = getattr(mod, "__version__", "installed")
        except Exception:
            out[m] = None
    return out


class SessionClock:
    """Stops starting new runs before Kaggle kills the session."""

    def __init__(self, hours=cfg.SESSION_HOURS):
        self.t0, self.limit = time.time(), hours * 3600

    def elapsed_h(self):
        return (time.time() - self.t0) / 3600

    def can_start(self, expected_h):
        ok = (time.time() - self.t0) + expected_h * 3600 < self.limit
        if not ok:
            print(f"SESSION GUARD: {self.elapsed_h():.1f} h used, next run needs ~{expected_h:.1f} h -> skipped. "
                  "Run this notebook again (new session) to continue; finished runs are skipped.")
        return ok
