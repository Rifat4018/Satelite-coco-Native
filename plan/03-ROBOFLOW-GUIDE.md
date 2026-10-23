# Roboflow: fixing the public dataset

I could not open your Roboflow account or find a public page for `aerial-object-detection-u3f5u`, so check these steps against your screen.

1. **Find the rejection reason.** Roboflow does not normally "reject" a project. If the rejection was the *paper*, there is nothing to fix on Roboflow itself. If it was a Roboflow Universe message, send me the text or a screenshot.
2. **Keep the old project** ("Translation check", 536). It is probably your source images (notebook 01 prints the unique-source count to compare). Do not delete it.
3. **Create a new project:** Projects -> New Project -> type *Object Detection*. Name it, for example, `Satellite Plane Ship Vehicle Detection`; the annotation group is `objects`.
4. **Upload:** run notebook 05, download `roboflow_upload_grouped.zip`, unzip it, and drag the folder in (`train/`, `valid/`, `test/` each contain `_annotations.coco.json`). After the upload, check Dataset -> split counts against `README.dataset.md`. If they differ, do **not** use "Rebalance"; remove and upload again.
5. **Classes & Tags:** exactly `plane`, `ship`, `vehicle`.
6. **Generate a version** with Preprocessing **none** (the images are already 640×640) and Augmentation **none**. The train folder already contains the 3 augmented copies used in the experiments; adding Roboflow augmentation again would change the data.
7. Name the version `v-grouped-<fingerprint>`, using the fingerprint printed by notebook 01.
8. **Export test:** Download -> COCO and YOLOv8 -> attach the export to notebook 01 instead of `satelite-coco` and check that the counts match.
9. **Description:** paste `README.dataset.md` from notebook 05 and fill in LICENSE, SOURCE and CITATION. Make the project public **only** if the source imagery allows it (Google Earth screenshots usually do not allow redistribution).
