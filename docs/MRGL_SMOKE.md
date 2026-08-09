# Minimal Single-CXR MRGL Training and Inference

Languages: **English** | [日本語](MRGL_SMOKE.ja.md) | [简体中文](MRGL_SMOKE.zh-CN.md)

## Status

The MRGL training and inference smoke tests each completed on one batch of real
images derived from NIH ChestX-ray14. Both `loss.backward()` and
`optimizer.step()` completed successfully.

This is a minimal, paper-aligned smoke pipeline. It is not intended to reproduce
the full experiment, metrics, 18/60-label dataset, or long-running training.

## Added files

- [`models/mrgl_classifier.py`](../models/mrgl_classifier.py)
  - Single-CXR ResNet-50 backbone.
  - 26 anatomical node features extracted with ROIAlign.
  - Spatial, Semantic, and Implicit graph branches.
  - Equation 6 graph convolution and Equation 7 weighted ensemble.
- [`adapters/nih_dataset.py`](../adapters/nih_dataset.py)
  - NIH ChestX-ray14 DataLoader with a 14-dimensional multi-label target.
- [`adapters/cxas_bbox_cache.py`](../adapters/cxas_bbox_cache.py)
  - Explicit mapping between the 26 regions in paper Table V and CXAS classes.
  - Segmentation mask to bounding rectangle to normalized XYXY cache.
- [`adapters/semantic_graph.py`](../adapters/semantic_graph.py)
  - Table V group fallback Semantic adapter.
  - Loader for an external precomputed adjacency matrix.
- [`scripts/precompute_cxas_bboxes.py`](../scripts/precompute_cxas_bboxes.py)
- [`scripts/smoke_train_mrgl.py`](../scripts/smoke_train_mrgl.py)
- [`scripts/smoke_infer_mrgl.py`](../scripts/smoke_infer_mrgl.py)
- [`requirements-mrgl-smoke.txt`](../requirements-mrgl-smoke.txt)
- `.gitignore`, adapter package initialization, and this documentation in
  English, Japanese, and Simplified Chinese.

No pre-existing upstream model or preprocessing file was modified.

## Relationship to the paper and original repository

- `ChangeDetector` and the ReGAT-related classes are not reused because they
  consume before/after images and a question embedding. That structure does not
  match the single-CXR classifier in paper Figure 3.
- The implementation follows the 26-node order, ResNet layer4 2048-dimensional
  ROI features, and ROIAlign(1x1) design found in
  `anatomical_feature_extract.py`.
- The final graph dimension is 256 with three graph-convolution layers. The
  Spatial, Semantic, and Implicit ensemble weights are `0.3 / 0.4 / 0.3`.
- `feature extraction/combine_dicts.py` combines anatomy and disease-location
  detector nodes. It is not treated as the paper's 26-anatomy-node Semantic
  graph.

References:

- [MRGL paper](https://doi.org/10.1109/TMI.2024.3441494)
- [CXAS repository and class definitions](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation)

## Pipeline

```text
NIH CXR [B,3,256,256]
  -> ResNet-50 layer4 feature map [B,2048,8,8]
  -> 26 cached anatomical XYXY boxes
  -> ROIAlign(1x1)
  -> node features [B,26,2048]
  -> Spatial / Semantic / Implicit graph branches
  -> three graph-convolution layers, final dimension 256
  -> node average pooling
  -> branch-specific 14-label predictions
  -> weighted average (0.3 / 0.4 / 0.3)
  -> output [B,14]
```

The model returns probabilities, so the smoke training script uses `BCELoss`.
Adam is configured with the paper learning rate of `0.01` by default.

## Environment setup

Install the smoke-test dependencies:

```bash
python -m pip install -r requirements-mrgl-smoke.txt
```

CXAS downloads the `UNet_ResNet50_default` pretrained checkpoint the first time
bbox precomputation is run. The default MRGL backbone downloads ImageNet
ResNet-50 weights the first time the model is created.

## NIH ChestX-ray14 data

Use the standard NIH metadata CSV, image directory, and optional official split
lists. A typical layout is:

```text
/path/nih/
  Data_Entry_2017.csv
  images/
    00000001_000.png
    ...
  train_val_list.txt
  test_list.txt
```

`Finding Labels` is converted to the following 14-dimensional target order:

```text
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
Pleural_Thickening, Hernia
```

`No Finding` becomes an all-zero target vector.

## Precompute CXAS anatomical bbox caches

Run CXAS once before training. Training does not run CXAS inference repeatedly.

Training cache:

```bash
python scripts/precompute_cxas_bboxes.py \
  --csv /path/nih/Data_Entry_2017.csv \
  --image-dir /path/nih/images \
  --split-file /path/nih/train_val_list.txt \
  --output /path/cache/cxas_train_boxes.npz \
  --device cpu
```

Test cache:

```bash
python scripts/precompute_cxas_bboxes.py \
  --csv /path/nih/Data_Entry_2017.csv \
  --image-dir /path/nih/images \
  --split-file /path/nih/test_list.txt \
  --output /path/cache/cxas_test_boxes.npz \
  --device cpu
```

For CUDA CXAS inference, pass a CUDA index such as `--device 0`. Use `--limit N`
to precompute only the first `N` matching images for a small smoke dataset.

The cache stores normalized boxes as `[x0, y0, x1, y1]`, a validity flag for
each of the 26 nodes, and mapping metadata.

## Training smoke test

The command below performs one forward pass, computes `BCELoss`, runs
`loss.backward()`, and executes `optimizer.step()`:

```bash
python scripts/smoke_train_mrgl.py \
  --csv /path/nih/Data_Entry_2017.csv \
  --image-dir /path/nih/images \
  --bbox-cache /path/cache/cxas_train_boxes.npz \
  --split-file /path/nih/train_val_list.txt \
  --batch-size 1 \
  --device cpu \
  --optimizer-step
```

Without `--optimizer-step`, the script still runs `loss.backward()` and leaves
the configured optimizer ready to step.

## Inference smoke test

The command below performs one test batch under `torch.no_grad()`:

```bash
python scripts/smoke_infer_mrgl.py \
  --csv /path/nih/Data_Entry_2017.csv \
  --image-dir /path/nih/images \
  --bbox-cache /path/cache/cxas_test_boxes.npz \
  --split-file /path/nih/test_list.txt \
  --batch-size 1 \
  --device cpu
```

## Important options

- `--image-size`: input size; default `256`, matching the paper setup.
- `--iou-threshold`: Spatial graph IoU threshold; default `0.3`.
- `--semantic-adj`: load an external `[26,26]` or `[B,26,26]` Semantic
  adjacency from `.npy` or `.npz`.
- `--no-pretrained-backbone`: keep the ResNet-50 architecture but do not
  download or load ImageNet weights.
- `--num-workers`: DataLoader workers; default `0` for a portable smoke test.
- `--seed`: PyTorch random seed; default `0`.

## Anatomical mapping limitations

The explicit CXAS mapping contains:

```text
15 exact
1 composed
1 terminology match
2 approximations
7 unavailable
```

The two lower-lung regions use the corresponding CXAS lung-base masks and are
marked as approximations. CXAS has no direct class for the right/left hilum,
right/left costophrenic sulcus, main bronchus, superior vena cava, or cavoatrial
region. These seven nodes are kept as invalid and their node features are
zeroed; they are not assigned guessed masks.

## Semantic graph limitation

The paper Semantic graph requires graph-Grad-CAM top-1/top-2 abnormality labels
for each anatomical node and the Figure 7 abnormality co-occurrence knowledge
graph. The public repository does not contain a complete machine-readable
artifact for that construction.

The default `TableVGroupSemanticAdapter` is therefore an explicitly separated
smoke-only fallback that connects nodes within the four Table V groups. It is
not claimed to reproduce the paper Semantic graph. Supply a reconstructed or
precomputed adjacency with `--semantic-adj` when one is available.

## Verified tensor shapes

```text
image.shape            (1, 3, 256, 256)
target.shape           (1, 14)
bbox.shape             (1, 26, 4)
bbox_valid_per_image   [19]
node_features.shape    (1, 26, 2048)
spatial_adj.shape      (1, 26, 26)
semantic_adj.shape     (1, 26, 26)
implicit_adj.shape     (1, 26, 26)
output.shape           (1, 14)
```

## Verified smoke-test results

Training:

```text
loss 0.8319577574729919
loss.backward() ok
optimizer.step() executed
```

Inference:

```text
output.shape (1, 14)
torch.no_grad() forward ok
```

The verification environment was PyTorch 2.5.1, torchvision 0.20.1, CXAS
0.0.18, and CPU. Smoke-test images and generated CXAS caches are stored below
`data/` and excluded from Git.
