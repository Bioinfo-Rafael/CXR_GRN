# Minimal single-CXR MRGL smoke pipeline

## Repository audit

- MRGL-related preprocessing: `anatomical_feature_extract.py`,
  `graph_mimic_new.py`, and `feature extraction/`. The first file establishes
  the repository's ResNet-50 layer4 feature shape (2048 channels) and
  ROIAlign(1x1) usage.
- VQA/ReGAT-derived code: `models/modules.py`, `models/relation_encoder.py`,
  `models/graph_att.py`, `models/graph_att_layer.py`, language/speaker modules,
  and `train.py`. `ChangeDetector` consumes before/after images and a question,
  so it is not used by the single-image classifier.
- Reused conventions: the 26-node order in `anatomical_feature_extract.py`,
  ImageNet normalization, ResNet-50 layer4 ROI features, and the paper's 256-D
  final graph feature.
- `feature extraction/combine_dicts.py` joins anatomy and disease-location
  detector nodes. This is not assumed to be the paper's 26-anatomy-node
  semantic graph.

## Data and bbox cache

Use a standard NIH `Data_Entry_2017.csv`, its image directory, and optional
official `train_val_list.txt` / `test_list.txt` files. Precompute CXAS boxes once:

```bash
python scripts/precompute_cxas_bboxes.py \
  --csv /path/Data_Entry_2017.csv \
  --image-dir /path/images \
  --split-file /path/train_val_list.txt \
  --output /path/cxas_train_boxes.npz
```

The explicit Table V-to-CXAS mapping is `ANATOMICAL_REGIONS` in
`adapters/cxas_bbox_cache.py`. Seven areas unavailable in CXAS remain invalid
nodes instead of receiving guessed masks. The two lower-lung zones use CXAS
lung-base masks and are marked `approximation` in cache metadata.

## Smoke commands

```bash
python scripts/smoke_train_mrgl.py \
  --csv /path/Data_Entry_2017.csv --image-dir /path/images \
  --bbox-cache /path/cxas_train_boxes.npz --split-file /path/train_val_list.txt

python scripts/smoke_infer_mrgl.py \
  --csv /path/Data_Entry_2017.csv --image-dir /path/images \
  --bbox-cache /path/cxas_test_boxes.npz --split-file /path/test_list.txt
```

Training uses `BCELoss` because the Eq. 7 ensemble returns 14 probabilities.
Pass `--optimizer-step` to execute the already-configured Adam step.

The default semantic adapter connects anatomy nodes within the four Table V
groups. It is a smoke-only substitute, not a claim of paper reproduction. The
paper graph requires graph-Grad-CAM top-1/top-2 labels plus the Figure 7
abnormality co-occurrence graph, for which the public repository does not ship
a complete machine-readable artifact. A supplied `[26,26]` adjacency can be
used with `--semantic-adj`.
