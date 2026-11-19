# Blinded annotation audit (meta-review point 1) - how to run notebooks 14A and 14B

Goal: show whether missing labels hurt standard inference and SAHI equally, and re-score both on corrected labels.
Notebooks 13A/13B (full manual relabel of the 82 images) are an ALTERNATIVE; use one of them, not both, for the paper numbers.

## What you need on Kaggle
`sat-ext` (this folder, updated), `sat-splits`, `satelite-coco`, `sat-results-e` (grouped YOLO11s seed 42 with its saved
detections for `standard` and `sahi320`), and for 14B also every other `sat-results-*` Dataset (same list as 13B).

## Steps
1. **Update `sat-ext`**: upload the new `sat_ext` folder as a new version of the Dataset (it adds `blindaudit.py`).
2. **14A** (CPU, ~5 min): Run all. It prints how many apparent false positives each mode has and how many pages there are.
   Download `audit_for_annotators.zip`. Output tab -> New Dataset **`sat-audit-key`** from `results/audit/` (private key).
3. **Annotators** (two people who did NOT make the original labels; A reviews everything, B a random 25 %, independently):
   open the PNG pages; in `verdicts_A.csv` (B: `verdicts_B.csv`) change `N` to `M` (real object, no label), `P` (box on
   part of a real object) or `U` (cannot decide). They see only `#id class`, never the score or the mode.
4. Upload the two filled CSVs as Dataset **`sat-audit-verdicts`**.
5. **14B** (CPU, 20-40 min): attach the inputs listed in its first cell. It writes T19-T23 and two figures
   (`fig_label_audit_examples.png`, `fig_label_audit_ap.png`) to `results/paper/`. Output -> New Dataset `sat-results-14`.
6. Re-run 04, 09, 12 with `sat-results-14` attached (they pick up the corrected labels automatically).

## What to put in the paper
* T19: share of apparent false positives that were real objects, per mode (with 95 % CI) - answers "is the bias equal?".
* T21: AP / precision / recall / F1 of standard vs SAHI on original vs corrected labels. If SAHI still loses AP, say so
  (the conclusion now has the audit behind it); if the gap closes, change the abstract and Section IV-B to what T21 shows.
* T20: kappa between the annotators. State the limit: only boxes found by the audited model can be added, so objects that
  no audited detector found stay unlabelled.
* Do not call AP losses "upper bounds".

## Tests
`python tools/test_blindaudit.py` (logic, a few seconds) and `python tools/smoke_test.py` (whole pipeline incl. 14A/14B).


## v3.7 changes
* The templates `verdicts_A.csv` / `verdicts_B.csv` now start with an EMPTY `verdict` column. Every id needs one letter
  (M, P, N, U); 14B refuses a file with empty rows (a default "N" looked like a real answer).
* 14B runs `blindaudit.integrity_check`: it stops if B's answers were pasted into A (A's non-N answers only on B's ids),
  if A and B agree on >= 99 % of >= 30 boxes, or if B never used N although A did. A and B must never see each other's
  files or sheets.
* Optional `verdicts_C_LLM.csv` (a model-assisted reviewer) is reported as T20b only; it never changes the labels.
* Notebook `15-fusion-time.ipynb`: measures Weighted Box Fusion time per image for the 3-model ensemble (CPU).
