# What was wrong in the old pipeline (verified from your NB1/NB2/NB3 code and logs)

| # | Problem | Fixed by |
|---|---|---|
| 1 | Table I was computed on the **validation** split (54 images, 2,001 instances in the logs). It was not the test set. | `infer.ultralytics_test_metrics(split="test")` |
| 2 | mAP was computed at `conf=0.25`. The standard is 0.001, so mAP was understated (YOLO11s 0.778 in the training log vs 0.7425 in the table). | `cfg.EVAL_CONF = 0.001` |
| 3 | mAP/P/R repeat across the SAHI rows because SAHI accuracy was **never measured**. Only raw box counts were. | COCO AP for every mode + TP/FP matching |
| 4 | "Cosine LR" in the paper, but the logs show `cos_lr=False`. | Recipe is now explicit; write "linear decay" |
| 5 | SAHI merge was GREEDYNMM/IOS (the SAHI default), not "NMS IoU 0.5". It also ran on only 50 of 55 test images. | `cfg.SAHI_POSTPROCESS`, all test images |
| 6 | FPS came from 30 passes on a zero tensor, with no warm-up or sync (the same model showed 113–131 FPS). SAHI speed was never measured. | End-to-end ms/image with warm-up + sync, per mode |
| 7 | Box clipping clipped the centre and width separately. Also, the 724 anomalies cover the train split only. | Corner clipping, everything logged in `clean_log.csv` |
| 8 | Size notation "< 322 px" lost its superscript. | Write "area < 32² px² on 640×640 images" |
| 9 | RT-DETR was loaded with `YOLO()`, and SAHI used `model_type="yolov8"`. | `RTDETR()` class, SAHI `ultralytics` type |
| 10 | Ultralytics silently removed duplicate labels (up to 65 per image). | Removed and logged before training |
| 11 | "1,414 base images" counts augmented copies. | Notebook 01 prints the real number of source images |

# Reviewer comment -> evidence produced
| Reviewer point | Where |
|---|---|
| R1-1 augmented copies across splits, R3-4 spatial leakage | 01: `05_leakage_summary.json`, `03_*`/`04_*` montages; grouped split; T5 (orig vs grouped) |
| R1-2 detection count vs mAP/P/R | T2 (COCO AP **and** TP/FP per mode), `fig_extra_detections` |
| R1-3 size notation, SAHI FPS | `06_size_distribution.png`, T2 `ms_img` + `Passes` |
| R1-4 Ref [14] | **You:** find the real authors, pages and DOI of the IDAP 2024 paper |
| R3-1 contribution | Write it: the first leakage-controlled, multi-seed, size-stratified, TP-verified SAHI benchmark |
| R3-2 extra SAHI boxes real? | T2 `extra_TP` / `extra_FP`, T3 bootstrap CI, qualitative TP/FP/FN figure |
| R3-3 stratify by size | T1/T2 `AP_small`, `AR50_micro`, `fig_size_recall` |
| R3-5 Pareto | `fig_pareto` (latency and GFLOPs × passes) |
| 524 R1/R2/R4 figure quality, abstract, single dataset | 300-dpi PDF figures. Shorten the abstract yourself; state the single dataset as a limitation |

Report only numbers that notebook 04 prints. Keep the old numbers out of the revision; if you mention them, use T5's `legacy` rows (the old weights re-scored correctly).
