"""Unit test of the blinded-audit logic (notebook 14) on tiny synthetic data. CPU, a few seconds."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, cv2
import blindaudit as ba, metrics

tmp = tempfile.mkdtemp(prefix="audit_test_")
imgdir = os.path.join(tmp, "img"); os.makedirs(imgdir)
rng = np.random.default_rng(0)
images = []
for i in (1, 2, 3):
    cv2.imwrite(os.path.join(imgdir, f"im{i}.jpg"), rng.integers(0, 255, (640, 640, 3), dtype=np.uint8))
    images.append(dict(id=i, file_name=f"im{i}.jpg", width=640, height=640))
def A(i, im, c, b): return dict(id=i, image_id=im, category_id=c, bbox=b, area=b[2] * b[3], iscrowd=0)
gt = dict(images=images, categories=[dict(id=1, name="plane"), dict(id=2, name="ship"), dict(id=3, name="vehicle")],
          annotations=[A(1, 1, 3, [100, 100, 20, 20]), A(2, 2, 3, [300, 300, 20, 20])])
def D(im, c, b, s): return dict(image_id=im, category_id=c, bbox=b, score=s)
std = [D(1, 3, [100, 100, 20, 20], .9),               # TP
       D(1, 3, [400, 400, 20, 20], .8),               # FP #A (also in SAHI)
       D(2, 3, [50, 50, 20, 20], .7),                 # FP #B (standard only)
       D(3, 3, [200, 200, 20, 20], .1)]               # below conf -> ignored
sahi = [D(1, 3, [100, 100, 20, 20], .9), D(2, 3, [300, 300, 20, 20], .9),
        D(1, 3, [402, 401, 20, 20], .85),             # same object as FP #A
        D(3, 3, [500, 100, 20, 20], .6),              # SAHI only #C
        D(3, 3, [100, 500, 20, 20], .5)]              # SAHI only #D
dm = dict(standard=std, sahi320=sahi)

c = ba.build_candidates(gt, dm)
assert len(c) == 4, c                                  # A, B, C, D  (A merged across modes)
assert c["in_standard"].sum() == 2 and c["in_sahi320"].sum() == 3
assert sorted(c["id"]) == [1, 2, 3, 4]
assert (c[["in_standard", "in_sahi320"]].all(axis=1)).sum() == 1
c2 = ba.build_candidates(gt, dm)
assert (c.id.values == c2.id.values).all() and (c.x.values == c2.x.values).all()          # reproducible shuffle

paths = ba.make_sheets(c, imgdir, os.path.join(tmp, "sheetsA"), per_page=4, cols=2)
assert len(paths) == 1 and cv2.imread(paths[0]).shape[0] == 2 * 240
ids_b = ba.second_subset(c, 0.5)
pb = ba.make_sheets(c, imgdir, os.path.join(tmp, "sheetsB"), ids=ids_b, per_page=4, cols=2, order_seed=7)
assert len(ids_b) == 2 and len(pb) == 1
ba.write_templates(c, os.path.join(tmp, "tpl"), ids_b)

# annotators: A says the SAHI-only #C and the shared #A are missing labels, #B true FP, #D unsure
key = {tuple(np.round(r[["x", "y"]].values.astype(float))): r.id for _, r in c.iterrows()}
idA = key[(400.0, 400.0)] if (400.0, 400.0) in key else key[(402.0, 401.0)]
idB = key[(50.0, 50.0)]; idC = key[(500.0, 100.0)]; idD = key[(100.0, 500.0)]
va = pd.Series({idA: "M", idB: "N", idC: "M", idD: "U"}).sort_index(); va.index.name = "id"
vb = pd.Series({i: va[i] for i in ids_b}); vb.loc[ids_b[0]] = "N" if va[ids_b[0]] != "N" else "M"
final, ag = ba.consensus(c, va, vb)
assert ag["n_both"] == 2 and ag["agree"] == 0.5, ag
assert final[ids_b[0]] == "U"                                                              # disagreement -> unsure
t = ba.mode_table(c, final, ["standard", "sahi320"]); print(t.to_string(index=False))
final2 = va.reindex(c.id)
g1 = ba.corrected_gt(gt, c, final2)
assert len(g1["annotations"]) == 2 + 2
g2 = ba.corrected_gt(gt, c, final2, include=("M", "U"))
assert len(g2["annotations"]) == 2 + 3
sc = ba.score_versions({"original": gt, "corrected": g1}, dm, ["standard", "sahi320"]); print(sc.round(3).to_string(index=False))
o = sc[(sc.labels == "original") & (sc["mode"] == "sahi320")].iloc[0]; n = sc[(sc.labels == "corrected") & (sc["mode"] == "sahi320")].iloc[0]
assert n.FP < o.FP and n.P > o.P                                                            # corrected labels remove FPs
assert ba.cohen_kappa(list("aabb"), list("aabb")) == 1.0
lo, hi = ba.wilson(5, 10); assert 0.2 < lo < 0.5 < hi < 0.8
f1 = ba.fig_examples(c, final2, imgdir, os.path.join(tmp, "ex.png")); f2 = ba.fig_ap(sc, os.path.join(tmp, "ap.png"))
assert os.path.getsize(f1) > 0 and os.path.getsize(f2) > 0
# an empty template must be rejected (nobody has filled it in)
try:
    ba.read_verdicts(os.path.join(tmp, "tpl", "verdicts_A.csv"), c.id); raise SystemExit("empty template not caught")
except AssertionError as e:
    print("caught empty:", e)
pd.DataFrame(dict(id=[1], verdict=["Z"])).to_csv(os.path.join(tmp, "bad.csv"), index=False)
try:
    ba.read_verdicts(os.path.join(tmp, "bad.csv"), c.id); raise SystemExit("bad code not caught")
except AssertionError as e:
    print("caught:", e)
# integrity check: B pasted into A is stopped; independent-looking files pass
import numpy as _np
rng = _np.random.default_rng(0)
ids = list(range(1, 201)); bid = sorted(rng.choice(ids, 50, replace=False).tolist())
vb_ = pd.Series({i: rng.choice(list("MPU")) for i in bid}); vb_.index.name = "id"
va_bad = pd.Series({i: (vb_[i] if i in vb_.index else "N") for i in ids}); va_bad.index.name = "id"
try:
    ba.integrity_check(va_bad, vb_); raise SystemExit("copied B not caught")
except AssertionError as e:
    print("caught copy:", str(e)[:90])
va_ok = pd.Series({i: rng.choice(list("MPNU"), p=[.3, .2, .4, .1]) for i in ids}); va_ok.index.name = "id"
vb_ok = pd.Series({i: (va_ok[i] if rng.random() < 0.7 else rng.choice(list("MPNU"))) for i in bid}); vb_ok.index.name = "id"
print(ba.integrity_check(va_ok, vb_ok))
assert len(ba.model_agreement(va_ok, va_ok, vb_ok)) == 2
print("test_blindaudit OK")
