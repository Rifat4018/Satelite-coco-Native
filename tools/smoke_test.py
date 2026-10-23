"""CPU smoke test: builds a synthetic Roboflow-style export with planted leaks, then executes the code
cells of every notebook in order (each in a fresh process, like a Kaggle session), moving outputs to the
"inputs" between notebooks the way you will on Kaggle. Ultralytics / SAHI are replaced by tools/fakes,
so this checks the pipeline logic, file flow and tables - NOT real training.
Usage: python tools/smoke_test.py [TMP_DIR]
"""
import os, sys, json, shutil, subprocess, glob, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="sat_smoke_")
INP, WORK = os.path.join(TMP, "input"), os.path.join(TMP, "work")
NB = os.path.join(HERE, "notebooks")
sys.path.insert(0, HERE)
from plans import NAMES, PLANS


def reset_work():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)


def run_nb(name, extra_env=None, patch=None):
    cells = json.load(open(os.path.join(NB, name)))["cells"]
    src = "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    if patch:
        for a, b in patch:
            assert a in src, f"patch target not found in {name}: {a}"
            src = src.replace(a, b)
    script = os.path.join(TMP, "_run_" + name.replace(".ipynb", ".py"))
    open(script, "w").write(src)
    env = dict(os.environ, SAT_INPUT=INP, SAT_WORK=WORK, SAT_SMOKE="1", MPLBACKEND="Agg",
               PYTHONPATH=os.path.join(HERE, "tools", "fakes"), **(extra_env or {}))
    r = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, cwd=TMP)
    log = os.path.join(TMP, "_log_" + name.replace(".ipynb", ".txt"))
    open(log, "w").write(r.stdout + "\n--- STDERR ---\n" + r.stderr)
    status = "OK" if r.returncode == 0 else "FAILED"
    print(f"[{status}] {name}  (log: {log})")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-5000:])
        sys.exit(1)
    return r.stdout


def publish(src, dataset_name):
    """Kaggle 'Output -> New Dataset': copy a working-dir folder into the inputs."""
    dst = os.path.join(INP, dataset_name)
    shutil.copytree(src, os.path.join(dst, os.path.basename(src)), dirs_exist_ok=True)
    import gzip                  # Kaggle un-gzips *.gz when an output becomes a Dataset - simulate it
    for r, _, fs in os.walk(dst):
        for f in fs:
            if f.endswith(".gz"):
                p = os.path.join(r, f)
                with gzip.open(p, "rb") as a, open(p[:-3], "wb") as b:
                    b.write(a.read())
                os.remove(p)


def main():
    print("smoke dir:", TMP)
    shutil.rmtree(INP, ignore_errors=True)
    os.makedirs(INP)
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "make_synthetic.py"),
                    os.path.join(INP, "datasets", "rifat4018", "satelite-coco")], check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "make_synthetic.py"),
                    os.path.join(INP, "datasets", "someone", "skysealand-v14"), "sky"], check=True)
    for mode, dst in [("native", ["datasets", "rifat4018", "satelite-native"]), ("dota", ["dota-v1"])]:
        subprocess.run([sys.executable, os.path.join(HERE, "tools", "make_synthetic.py"), os.path.join(INP, *dst), mode], check=True)
    shutil.copytree(HERE, os.path.join(INP, "sat-ext", "sat_ext"),
                    ignore=shutil.ignore_patterns("notebooks", "tools", "__pycache__", "*.zip"))
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "build_notebooks.py")], check=True)
    for f in glob.glob(os.path.join(NB, "*.ipynb")):                     # every notebook must at least compile
        cells = json.load(open(f))["cells"]
        compile("\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code"), f, "exec")
    print("all notebooks compile")

    reset_work()
    out = run_nb("01-audit-and-splits.ipynb")
    assert "groups_spanning_splits" in out
    publish(os.path.join(WORK, "splits"), "sat-splits")
    reset_work()
    out = run_nb("01b-audit-and-splits-skysealand.ipynb")
    assert "sky_grouped" in out
    publish(os.path.join(WORK, "splits"), "sat-splits-sky")

    # pilot path once (account E), then every account
    reset_work()
    run_nb(NAMES["E"] + ".ipynb", patch=[("PILOT = False", "PILOT = True")])
    assert glob.glob(os.path.join(WORK, "results", "*__pilot", "eval_test.json")), "pilot produced no eval"
    for acc in "ABCDEFG":
        reset_work()
        run_nb(NAMES[acc] + ".ipynb")
        publish(os.path.join(WORK, "results"), f"sat-results-{acc.lower()}")
        publish(os.path.join(WORK, "weights"), f"sat-results-{acc.lower()}")
    # rerun account A: everything must be skipped
    reset_work()
    out = run_nb(NAMES["A"] + ".ipynb")
    assert out.count("SKIP ") == 2, "finished runs were not skipped"

    # legacy weights of the old paper
    for m in ["YOLO11n", "YOLO11s", "YOLO11m"]:
        p = os.path.join(INP, "resultsyolo11", "saved_models", f"{m}_best.pt")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b"fake")
    for m in ["RT-DETR-L", "YOLOv8m"]:
        p = os.path.join(INP, "resultsrfdetr", "saved_models", f"{m}_best.pt")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b"fake")
    reset_work()
    run_nb("03-rescore-legacy-weights.ipynb")
    publish(os.path.join(WORK, "results"), "sat-results-legacy")

    # ---------------- phase 2
    reset_work()
    out = run_nb("06-make-extra-splits.ipynb")
    assert "val/test identical" in out and "native_tiles" in out, out[-2000:]
    assert os.listdir(WORK) == ["sat_splits_extra.zip"], os.listdir(WORK)   # 06 output NOT attached: 07x build it
    for k in [k for k in PLANS if k.startswith("07")]:
        reset_work()
        o = run_nb(NAMES[k] + ".ipynb")
        if k == "07K":
            import re as _re
            assert _re.search(r"materialized 'native_tiles'.*'train': [1-9]", o), o[-1500:]
        publish(os.path.join(WORK, "results"), f"sat-results-{k.lower()}")
        publish(os.path.join(WORK, "weights"), f"sat-results-{k.lower()}")
    reset_work()
    out = run_nb("08D-tta-and-inference-resolution.ipynb")
    assert "res1024" in out, out[-1500:]
    publish(os.path.join(WORK, "results"), "sat-results-08d")
    reset_work()
    out = run_nb("11-label-quality-check.ipynb")
    assert "T16_label_quality" in out, out[-1500:]
    for k in ("08A", "08B", "08C"):
        reset_work()
        run_nb(NAMES[k] + ".ipynb")
        publish(os.path.join(WORK, "results"), f"sat-results-{k.lower()}")
    # resume: 08B again with its own output attached -> everything skipped; then drop the c25 rows -> only those rerun
    reset_work()
    out = run_nb(NAMES["08B"] + ".ipynb")
    assert out.count("SKIP (done)") == 5, out[-1500:]
    import pandas as pd
    for f in glob.glob(os.path.join(INP, "sat-results-08b", "**", "sahi_ablation.csv"), recursive=True):
        d = pd.read_csv(f)
        d[~d.config.str.endswith("_c25")].to_csv(f, index=False)
    reset_work()
    out = run_nb(NAMES["08B"] + ".ipynb")
    assert out.count("rows already done, running the 4 missing ones") == 5, out[-1500:]
    publish(os.path.join(WORK, "results"), "sat-results-08b")
    reset_work()
    run_nb("09-errors-mcnemar-ensembles.ipynb")
    for f in ["T11_errors_grouped.csv", "T12_mcnemar_grouped_standard.csv", "T12b_mcnemar_sahi_vs_standard_grouped.csv",
              "T13_ensembles_grouped.csv", "fig_errors_grouped.png"]:
        assert os.path.isfile(os.path.join(WORK, "results", "paper", f)), f
    reset_work()
    out = run_nb("10-dataset-statistics-and-agreement.ipynb")
    assert os.path.isfile(os.path.join(WORK, "results", "paper", "fig_dataset_statistics.png"))

    reset_work()
    out12 = run_nb("12-all-paper-results-04-plus-09.ipynb")
    assert "T13_ensembles_grouped" in out12 and "MISSING: none" in out12, out12[-1500:]
    # ---- 13: simulate a Roboflow re-export of the test images with one extra box per image
    import json as _json
    gdir = os.path.join(INP, "sat-splits", "splits", "grouped")
    g = _json.load(open(os.path.join(gdir, "coco", "test.json")))
    rel = os.path.join(INP, "sat-test-relabeled", "train"); os.makedirs(rel, exist_ok=True)
    ims, anns = [], []
    cats = [dict(id=0, name="objects"), dict(id=1, name="plane"), dict(id=2, name="ship"), dict(id=3, name="car")]
    for k, im in enumerate(g["images"], 1):
        stem = os.path.splitext(im["file_name"])[0].replace(".", "_")
        nn = f"{stem}_jpg.rf.{k:032x}.jpg"
        ims.append(dict(id=k, file_name=nn, width=im["width"], height=im["height"]))
        for a in g["annotations"]:
            if a["image_id"] == im["id"]:
                anns.append(dict(a, id=len(anns) + 1, image_id=k, category_id={1: 1, 2: 2, 3: 3}[a["category_id"]]))
        anns.append(dict(id=len(anns) + 1, image_id=k, category_id=3, bbox=[5, 5, 12, 10], area=120, iscrowd=0))
    _json.dump(dict(images=ims, annotations=anns, categories=cats), open(os.path.join(rel, "_annotations.coco.json"), "w"))
    reset_work()
    out = run_nb("13A-export-test-images-for-relabel.ipynb")
    assert "test_relabel_upload" in out, out[-2000:]
    reset_work()
    out = run_nb("13B-rescore-on-corrected-labels.ipynb")
    assert "T17_test_relabel_changes" in out and "re-scored" in out, out[-3000:]
    publish(os.path.join(WORK, "results_relabeled"), "sat-results-13")
    publish(os.path.join(WORK, "splits_relabeled"), "sat-results-13")
    reset_work()
    out = run_nb("12-paper-figures.ipynb")
    import re as _re
    nfig = int(_re.search(r"(\d+) figures written", out).group(1))
    print("figures:", nfig, "| skipped:", out.count("  skipped "))
    assert nfig >= 35, out[-3000:]
    reset_work()
    out = run_nb("04-paper-tables-figures.ipynb")
    assert "runs use the CORRECTED test labels" in out, out[-2000:]
    assert "MISSING: none" in out, "aggregation reports missing runs"
    paper = os.path.join(WORK, "results", "paper")
    need = ["T1_main_standard_grouped.csv", "T2_sahi_grouped.csv", "T3_sahi_bootstrap_grouped.csv",
            "T4_per_class_grouped.csv", "T5_orig_vs_grouped.csv", "fig_pareto_grouped.png",
            "fig_size_recall_grouped.png", "fig_extra_detections_grouped.png", "fig_qualitative_grouped.png"]
    miss = [f for f in need if not os.path.isfile(os.path.join(paper, f))]
    assert not miss, f"missing outputs: {miss}"
    import pandas as pd
    t1 = pd.read_csv(os.path.join(paper, "T1_main_standard_grouped.csv"))
    assert (t1.Seeds == 3).all(), t1
    t2 = pd.read_csv(os.path.join(paper, "T2_sahi_grouped.csv"))
    assert len(t2) == 80, len(t2)
    t6 = pd.read_csv(os.path.join(paper, "T6_cross_dataset.csv"))
    assert len(t6) == 10 and set(t6.Trained_on) == {"main", "sky"}, t6
    assert os.path.isfile(os.path.join(paper, "T1_main_standard_sky_grouped.csv"))
    t5 = pd.read_csv(os.path.join(paper, "T5_orig_vs_grouped.csv"))
    assert (t5.Seed.astype(str) == "legacy").sum() == 5, t5

    for f in ["T7a_offline_augmentation.csv", "T7b_learning_curve.csv", "T8_native_resolution.csv",
              "T9_sahi_ablation_grouped.csv", "T9_sahi_ablation_sky_grouped.csv", "T10_external_dota.csv", "fig_learning_curve.png"]:
        assert os.path.isfile(os.path.join(paper, f)), f
    t1 = pd.read_csv(os.path.join(paper, "T1_main_standard_grouped.csv"))
    assert len(t1) == 20, t1.Model.tolist()
    reset_work()
    run_nb("05-roboflow-export.ipynb")
    assert os.path.isfile(os.path.join(WORK, "roboflow_upload_grouped.zip"))
    print("\nSMOKE TEST PASSED -", TMP)


if __name__ == "__main__":
    main()
