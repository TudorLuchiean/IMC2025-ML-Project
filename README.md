# IMC2025 Modular SfM Pipeline: Approach and Status

## Task
Given folders of unordered images (one folder per dataset), estimate a camera pose for every image that can be registered, group images into scenes (clusters), and label the rest `outliers`. Output is `submission.csv` in the IMC format, keeping Kaggle's `image_id` values.

## Approach
Each step is one self-contained notebook cell. Swapping a step only requires keeping its declared outputs. All hyperparameters live in cell 0.2.

| Step | Module | Current implementation |
|------|--------|------------------------|
| 1 | Image discovery | `ImageFolder` file index + `sample_submission.csv` |
| 2 | Global embeddings | DINOv2 ViT-S/14, CLS token (384-d) |
| 3 | Pair shortlisting | k-NN (cosine) inside each dataset |
| 4 | Local features | SuperPoint |
| 5 | Matching | LightGlue |
| 6 | Geometric verification | OpenCV fundamental-matrix RANSAC |
| 7 | Pair filtering | Thresholds on matches / inliers / inlier ratio |
| 8-9 | Image graph / components | `networkx`, components for inspection only |
| 10 | SfM | `pycolmap` database + incremental mapping per dataset |
| 11 | Submission | Largest model wins per image; unregistered images become `outliers` |

## Experiment log (default notebook: 5.71 / 5.83)

| Change | Score (left / right) | Result |
|---|---|---|
| Filter 50 inliers / 0.5 ratio (stricter) | 2.17 / 3.29 | Clearly worse |
| Mapper init 15, abs-pose 10 | 7.44 / 6.58 | Small gain |
| MIN_INLIERS 15, ratio 0.10 | 7.36 / 6.51 | Small gain |
| MAGSAC + RANSAC threshold 2.0 | 9.31 / 6.84 | Medium gain |
| K_NEIGHBORS 40, embed size 518 | 6.17 / 7.04 | Small, mixed gain |
| **2048 keypoints @ 1024 resize** | **17.34 / 12.75** | **Biggest gain: best so far** |

**Main learning:** feature quality was the bottleneck. The other changes are refinements on top of it, and their gains probably overlap.

## Where we are
- Best tested configuration is the 2048 keypoint / 1024 resize run (17.34 / 12.75).
- The combined configuration (all positive changes together) is the next run. Its expected settings:
  `COARSE_MAX_KEYPOINTS=2048`, `COARSE_RESIZE=1024`, `K_NEIGHBORS=40`, `EMBED_IMAGE_SIZE=(518, 518)`, `RANSAC_THRESHOLD=2.0` with `cv2.USAC_MAGSAC`, `MIN_INLIERS=15`, `MIN_INLIER_RATIO=0.10`, `MAPPER_INIT_MIN_NUM_INLIERS=15`, `MAPPER_ABS_POSE_MIN_NUM_INLIERS=10`.
- If the combination scores below 17.34, add the other changes one at a time on top of the 2048 / 1024 run.
- Not yet tested:
  - LightGlue without pruning (`depth_confidence=-1`, `width_confidence=-1`)
  - Camera mode `PER_IMAGE` / `SINGLE` instead of `AUTO`
- Watch runtime: the notebook has no time budget and the hidden rerun is larger.

## Next sweeps (one axis at a time, stop when the score drops)
1. Keypoints / resolution: (2048, 1536), (4096, 1024), (4096, 1536)
2. Filter floor: (10 / 0.05), (15 / 0.10), (20 / 0.15)
3. RANSAC threshold: 1.5, 3, 4
4. Mapper: init 10 / abs-pose 8, plus `abs_pose_min_inlier_ratio` 0.25 to 0.15
5. K_NEIGHBORS: 60, or exhaustive pairs for small datasets

## Modules to improve (ranked by gain per effort)
1. **Evaluation:** add a train-split mAA check (the `TRAIN_*` paths are unused) and a runtime guard.
2. **Features / matcher (steps 4-5):** add ALIKED or DISK + LightGlue and merge matches with SuperPoint; or use a detector-free matcher (RoMa, LoFTR). Check for rotated images and match at 0/90/180/270 degrees if present.
3. **Pair selection (step 3):** exhaustive matching for small datasets, a larger embedding model (DINOv2-B), and a second retrieval round after verification.
4. **Verification (steps 6, 10a):** let COLMAP verify raw matches (`pycolmap.verify_matches`) so it uses focal priors instead of an `UNCALIBRATED` geometry.
5. **SfM backend (step 10b):** tune mapper options, try global SfM (GLOMAP) if the pycolmap version supports it, map per connected component, and run a relaxed second pass to register leftover outliers.
6. **Fallback for hard scenes:** MASt3R / VGGT-style reconstruction where COLMAP registers very few images (needs weights in the offline assets dataset).

## Diagnostics to check after each run
- Step 7 retention rate low: features or filter problem.
- Step 7 retention fine but few images registered in step 10c: mapper problem.
- Many tiny models: fragmentation; loosen filters or try global SfM.
- Mostly `outliers` in the final `value_counts()`: filters or features are too strict.
