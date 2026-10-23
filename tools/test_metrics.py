"""Unit tests of the internal COCO evaluator (same algorithm as pycocotools) and the TP/FP matcher.
On Kaggle, notebooks 02x also print pycocotools vs internal AP for the first run of each session."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import metrics


def gt_of(boxes):
    anns = [dict(id=i + 1, image_id=1, category_id=c, bbox=b, area=b[2] * b[3], iscrowd=0) for i, (c, b) in enumerate(boxes)]
    return dict(images=[dict(id=1, file_name="a.jpg", width=640, height=640)], annotations=anns,
                categories=[dict(id=1, name="plane"), dict(id=2, name="ship"), dict(id=3, name="vehicle")])


def det(c, b, s):
    return dict(image_id=1, category_id=c, bbox=b, score=s)


def ap(gt, dets, key):
    return metrics.coco_eval(gt, dets, prefer="internal")[key]


def close(a, b, tol=1e-6):
    assert abs(a - b) < tol, (a, b)


g = gt_of([(1, [10, 10, 50, 50]), (3, [200, 200, 10, 10])])
close(ap(g, [det(1, [10, 10, 50, 50], .9), det(3, [200, 200, 10, 10], .8)], "AP_all"), 1.0)          # perfect
g1 = gt_of([(1, [0, 0, 100, 100])])
close(ap(g1, [det(1, [0, 0, 100, 60], .9)], "AP50_all"), 1.0)                                        # IoU 0.6
close(ap(g1, [det(1, [0, 0, 100, 60], .9)], "AP75_all"), 0.0)
close(ap(g1, [det(1, [0, 0, 100, 60], .9)], "AP_all"), 0.3)                                          # matches at .50 .55 .60
g2 = gt_of([(1, [0, 0, 100, 100]), (1, [300, 300, 100, 100])])
close(ap(g2, [det(1, [0, 0, 100, 100], .9)], "AP50_all"), 51 / 101)                                  # recall 0.5
close(ap(g1, [det(1, [400, 400, 50, 50], .9), det(1, [0, 0, 100, 100], .8)], "AP50_all"), 0.5)      # FP ranked first
close(ap(g1, [det(1, [0, 0, 100, 100], .9), det(1, [400, 400, 50, 50], .8)], "AP50_all"), 1.0)      # FP ranked last
r = metrics.coco_eval(g1, [det(1, [0, 0, 100, 100], .9)], prefer="internal")
assert np.isnan(r["AP_small"]) and np.isnan(r["AP_micro"]) and abs(r["AP_large"] - 1) < 1e-9                # area ranges
g3 = gt_of([(3, [0, 0, 10, 10]), (3, [100, 100, 60, 60])])
r = metrics.coco_eval(g3, [det(3, [0, 0, 10, 10], .9), det(3, [500, 500, 80, 80], .95)], prefer="internal")
close(r["AP_micro"], 1.0)        # big unmatched FP lies outside the micro range -> ignored
close(r["AR50_medium"], 0.0)
# duplicate detection of one object = 1 TP + 1 FP
close(ap(g1, [det(1, [0, 0, 100, 100], .9), det(1, [1, 1, 100, 100], .8)], "AP50_all"), 1.0)
pi = metrics.match_per_image(g1, [det(1, [0, 0, 100, 100], .9), det(1, [1, 1, 100, 100], .8), det(2, [0, 0, 100, 100], .7)])
t = metrics.totals(pi)
assert (t["TP"], t["FP"], t["FP_duplicate"], t["FN"]) == (1, 2, 1, 0), t     # wrong-class box: FP, not duplicate
pi = metrics.match_per_image(g3, [det(3, [0, 0, 10, 10], .1)])                # below operating conf
assert metrics.totals(pi)["TP"] == 0
print("metrics tests passed")
# crowd (DOTA 'difficult') objects are ignored
gc = gt_of([(1, [0, 0, 100, 100]), (1, [300, 300, 50, 50])])
gc["annotations"][1]["iscrowd"] = 1
t = metrics.totals(metrics.match_per_image(gc, [det(1, [0, 0, 100, 100], .9), det(1, [300, 300, 50, 50], .9)]))
assert (t["TP"], t["FP"], t["n_gt"]) == (1, 0, 1), t
close(ap(gc, [det(1, [0, 0, 100, 100], .9), det(1, [300, 300, 50, 50], .95)], "AP50_all"), 1.0)
# McNemar + Holm
h1, h2 = {i: i < 30 for i in range(50)}, {i: i < 10 for i in range(50)}
m = metrics.mcnemar(h1, h2)
assert m["only_a"] == 20 and m["only_b"] == 0 and m["p_exact"] < 1e-5, m
assert list(metrics.holm([0.01, 0.04, 0.03])) == [0.03, 0.06, 0.06]
# vectorized bootstrap == loop version
import pandas as pd
A = pd.DataFrame(dict(image_id=range(20), tp=np.arange(20) % 5, n_gt=5))
Bd = pd.DataFrame(dict(image_id=range(20), tp=(np.arange(20) % 5 + 1).clip(0, 5), n_gt=5))
d1 = metrics.bootstrap_delta(A, Bd, ("tp", "n_gt"))
d2 = metrics.bootstrap_delta(A, Bd, lambda df: df.tp.sum() / df.n_gt.sum())
assert abs(d1[0] - d2[0]) < 1e-12 and abs(d1[1] - d2[1]) < 1e-12, (d1, d2)
print("extra metrics tests passed")
