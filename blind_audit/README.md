# Blinded annotation audit (notebooks 14a / 14b)

Released here (not on Zenodo) so that the audit in Section IV-E of the paper can be inspected and re-run.

**Protocol.** The background false positives (confidence >= 0.25, IoU < 0.1 with every label) of YOLO11s (seed 42) under standard inference and SAHI-320 were pooled (931 boxes), shuffled and shown as crops without the inference mode. Annotator A judged all 931 boxes; annotator B judged a random 25 % (233). Codes: **M** real object missing from the labels, **P** partial box or several objects, **N** not an object, **U** unsure. Where A and B differ on a box, the final verdict is U (not added to the labels).

| File | Content |
|---|---|
| `verdicts_A.csv` | annotator A, all 931 ids (`id,verdict`) |
| `verdicts_B.csv` | annotator B, 233 ids |
| `verdicts_C_model_assisted.csv` | a third, AI-model-assisted view of all 931 ids, for comparison only; it is not used in any reported number |
| `audit_key.csv`, `audit_meta.json` | id -> image, box, inference mode (output of notebook 14a; copy it here from the Kaggle dataset `sat-audit-key`) |
| `tables/T19-T23` | outputs of notebook 14b (CSV and LaTeX) |
| `figures/` | label-audit examples and AP before / after correction |

**Results (14b).** Final verdicts: U 411, M 299, P 137, N 84. Missing-label share of the false positives: standard 22 % (95 % CI 18-26), SAHI-320 32 % (29-35), SAHI-only boxes 40 % (36-44). Agreement of A and B on the 233 shared boxes: 40.3 %, Cohen's kappa 0.13 (0.22 for missing vs. rest). With the 299 confirmed objects added (2,735 to 3,034 test boxes), AP of standard / SAHI-320 changes from 0.393 / 0.293 to 0.371 / 0.314; counting the unsure boxes as well gives 0.368 / 0.342. All 78 saved runs were re-scored (`tables/T23_...csv`).

**Re-run.** Put `verdicts_A.csv` and `verdicts_B.csv` in a Kaggle dataset `sat-audit-verdicts`, attach it to notebook `14b` together with `sat-ext`, `sat-splits`, `satelite-coco`, the 14a output and the `sat-results-*` datasets, and run all cells. `blindaudit.py` refuses empty or copied annotator files.
