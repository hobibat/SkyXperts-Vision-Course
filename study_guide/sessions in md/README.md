# Study Guide: Sessions 1 → 5

A cell-by-cell walkthrough of every notebook in Sessions 1, 2, 3, 4 and 5. Where the original order jumps around, the guide puts it in a more logical order.

| File | Covers | Notebooks |
|---|---|---|
| [Session1.md](Session1.md) | Background for beginners, reading/showing, resizing (incl. how interpolation works), cropping, rotating, OOP refresher | `Session1.ipynb` |
| [Session2.md](Session2.md) | Reading images/video/camera, color spaces, filtering, edges | `Session2.ipynb`, `Session2_cont.ipynb`, `Filters_only.ipynb` |
| [Session3.md](Session3.md) | Keypoints, descriptors, matching, homography | `Session3.ipynb` |
| [Session4.md](Session4.md) | Hand-made filters → feature vectors (ORB, HOG) → CNNs → YOLO | all 15 notebooks in `Session4/` |
| [Session5.md](Session5.md) | Thresholding, morphology, edges recap, watershed | `Session5.ipynb` |

## Recommended study order

**1 → 2 → 5 → 3 → 4**

- Session 5 is pure pixel operations, like Session 2, and it's the most important one for the UGV project, so study it right after 2.
- Sessions 3 and 4 build up to features and deep learning. They're more advanced and less needed for the project.

## Symbols used in the guides

- 🎯 **Project:** directly used in the UGV flag detector.
- ⚠️ **Gotcha:** a bug, a misleading comment, or something that will bite you.
- 💡 **Why:** the idea behind the code.

## How to study each notebook

1. Open the notebook and its guide side by side.
2. Read the guide's section, run the cell, and look at the output.
3. Change one number (kernel size, threshold, sigma) and run it again. Predict the result *before* you run it.
4. Do the "Try it" items at the end of each section.

Run each notebook from its own folder, because the image paths are relative (`./images/...`).
