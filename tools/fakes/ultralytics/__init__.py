"""FAKE ultralytics for the CPU smoke test only. Predictions are made from the ground-truth labels
(jittered, with misses that depend on object size, plus random false positives) so the whole
evaluation pipeline can run without torch. NEVER used on Kaggle (it lives in tools/fakes)."""
import os, csv, numpy as np

__version__ = "0.0-fake"
NAMES = {0: "plane", 1: "ship", 2: "vehicle"}


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Arr(np.reshape(xyxy, (-1, 4))), _Arr(conf), _Arr(cls)

    def __len__(self):
        return len(self.conf.a)


class _Result:
    def __init__(self, boxes):
        self.boxes, self.names = boxes, NAMES
        self.speed = {"preprocess": 0.5, "inference": 5.0, "postprocess": 1.0}


class _P:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


class _Net:
    def __init__(self, n):
        self._p = [_P(n)]

    def parameters(self):
        return iter(self._p)


def _labels_for(img_path):
    d, f = os.path.split(img_path)
    lbl = os.path.join(os.path.dirname(d), "labels", os.path.splitext(f)[0] + ".txt")
    out = []
    if os.path.isfile(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) == 5:
                out.append([int(p[0])] + [float(v) for v in p[1:]])
    return out


class YOLO:
    _size = {"n": 2.6e6, "s": 9.4e6, "m": 20e6, "l": 32e6}

    def __init__(self, weights):
        self.weights = str(weights)
        key = os.path.basename(self.weights)
        self.strength = 0.55 + 0.07 * (sum(map(ord, key)) % 5)          # model-dependent recall
        self.model = _Net(int(self._size.get(next((c for c in "nsml" if f"{c}." in key or f"{c}_" in key), "s"), 9e6)))
        self.names = NAMES

    def load(self, weights):
        self.weights = self.weights + "|" + str(weights)
        return self

    def info(self, verbose=False):
        return (100, sum(p.numel() for p in self.model.parameters()), 0, 8.0)

    def train(self, data, project, name, epochs=1, seed=0, **kw):
        d = os.path.join(project, name, "weights")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "best.pt"), "wb").write(b"fake")
        with open(os.path.join(project, name, "results.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"])
            for e in range(1, epochs + 1):
                w.writerow([e, 0.5 + 0.01 * e, 0.3 + 0.01 * e])
        open(os.path.join(project, name, "args.yaml"), "w").write(f"seed: {seed}\nepochs: {epochs}\n")

    def predict(self, path, imgsz=640, conf=0.25, recall_boost=0.0, **kw):
        import zlib
        rng = np.random.default_rng(zlib.crc32(f"{self.weights}|{os.path.basename(str(path))}|{recall_boost}".encode()))
        xyxy, sc, cl = [], [], []
        from PIL import Image as _Im
        W, H = _Im.open(str(path)).size
        for k, cx, cy, w, h in _labels_for(str(path)):
            side = np.sqrt(w * W * h * H)
            p_hit = min(0.97, self.strength * (0.5 + 0.5 * min(side / 32, 1)) + recall_boost * (1 - min(side / 32, 1)))
            if rng.random() < p_hit:
                j = rng.normal(0, 0.06 * max(side, 4), 4)
                x1, y1 = (cx - w / 2) * W + j[0], (cy - h / 2) * H + j[1]
                x2, y2 = (cx + w / 2) * W + j[2], (cy + h / 2) * H + j[3]
                xyxy.append([x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)])
                sc.append(float(rng.uniform(0.3, 0.99)))
                cl.append(k)
                if recall_boost and rng.random() < 0.08:          # duplicate box at a tile seam
                    xyxy.append([x1 + 1, y1 + 1, max(x2, x1 + 1) + 1, max(y2, y1 + 1) + 1])
                    sc.append(float(rng.uniform(0.25, 0.6)))
                    cl.append(k)
        for _ in range(rng.integers(2, 8) + int(10 * recall_boost)):      # false positives
            x, y = rng.uniform(0, max(W - 40, 1)), rng.uniform(0, max(H - 40, 1))
            s = rng.uniform(5, 30)
            xyxy.append([x, y, x + s, y + s])
            sc.append(float(rng.uniform(0.02, 0.5)))
            cl.append(int(rng.integers(0, 3)))
        keep = [i for i, s in enumerate(sc) if s >= conf]
        return [_Result(_Boxes([xyxy[i] for i in keep], [sc[i] for i in keep], [cl[i] for i in keep]))]

    def val(self, data, split="val", **kw):
        class _B:
            pass
        b = _B()
        b.map50, b.map, b.mp, b.mr = 0.6 * self.strength + 0.2, 0.35 * self.strength + 0.1, 0.8, 0.7
        b.ap_class_index = [0, 1, 2]
        b.ap50 = [0.9, 0.6, 0.65]
        b.ap = [0.6, 0.4, 0.35]

        class _R:
            pass
        r = _R()
        r.box, r.speed = b, {"preprocess": 0.5, "inference": 5.0, "postprocess": 1.0}
        return r


class RTDETR(YOLO):
    pass
