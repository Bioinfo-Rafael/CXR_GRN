# Uncertain-Label

環境構築・実画像・全ローカルテストの統合RUN_GUIDE:
[日本語](../CXR_LLM/docs/RUN_GUIDE.ja.md) | [English](../CXR_LLM/docs/RUN_GUIDE.en.md) | [简体中文](../CXR_LLM/docs/RUN_GUIDE.zh-CN.md)

MIMIC adapterの詳細・初回実行履歴: [MIMIC実画像smoke](docs/MIMIC_SMOKE.ja.md)

## MRGL Smoke Quick Start — Real Images

This repository now lives at `/Users/cls-lab/Git/LinGu/CXR_GRN`, alongside
`CURV`, `CURV_stage3_repro`, and `CXR_LLM`. Run the existing smoke pipeline
with real images; no separate real-image README or dummy-label training is needed.

| Input | Available images | Training | Inference |
|---|---:|---|---|
| NIH ChestX-ray14 | 2 (train 1 / test 1) | Real CSV labels; backward + optimizer step | Real-image, no_grad |
| Shared manual MIMIC-CXR-JPG | 12 | Not run: compatible labels are missing | All 12; no paired teacher JSON required |

The local environment is `scdiffusion` (Python 3.9 / torch 2.5.1 / CXAS 0.0.18).
The commands below assume the images and model weights already exist locally.

### NIH: one training batch and inference

```bash
cd /Users/cls-lab/Git/LinGu/CXR_GRN
conda run --no-capture-output -n scdiffusion python scripts/smoke_train_mrgl.py \
  --csv data/nih_smoke/Data_Entry_2017.csv --image-dir data/nih_smoke/images \
  --bbox-cache data/nih_smoke/cxas_boxes.npz --split-file data/nih_smoke/train_list.txt \
  --device cpu --optimizer-step
conda run --no-capture-output -n scdiffusion python scripts/smoke_infer_mrgl.py \
  --csv data/nih_smoke/Data_Entry_2017.csv --image-dir data/nih_smoke/images \
  --bbox-cache data/nih_smoke/cxas_boxes.npz --split-file data/nih_smoke/test_list.txt --device cpu
```

Success markers: `loss.backward() ok`, `optimizer.step() executed`, and
`torch.no_grad() forward ok`. To create the initial NIH bbox cache, follow
the existing adapter documentation linked below.

### MIMIC: share images, build CXAS cache, run inference

First organize downloaded JPGs through the integrated
[image preparation steps](../CXR_LLM/docs/RUN_GUIDE.en.md#images).
The source files remain in Downloads; these repositories share them through
symlinks without copying image bytes or downloading MIMIC again.

```bash
cd /Users/cls-lab/Git/LinGu/CXR_GRN
conda run --no-capture-output -n scdiffusion python scripts/prepare_mimic_manifest.py \
  --source-manifest ../CXR_LLM/data/processed/manual_mimic_images_manifest.jsonl
conda run --no-capture-output -n scdiffusion python scripts/precompute_mimic_cxas_bboxes.py \
  --manifest data/mimic_smoke/manifest.jsonl --output data/mimic_smoke/cxas_boxes.npz --device cpu
conda run --no-capture-output -n scdiffusion python scripts/smoke_infer_mimic_mrgl.py \
  --manifest data/mimic_smoke/manifest.jsonl --bbox-cache data/mimic_smoke/cxas_boxes.npz --device cpu
```

The existing cache may be reused when image IDs and bytes are unchanged;
rebuild it and `cxas_boxes.npz.images.json` after changing the image set.
Success markers: `MIMIC CXAS CACHE OK` and
`MIMIC REAL IMAGE MRGL INFERENCE OK 12`. Results are saved to
`outputs/mimic_mrgl_inference.json`. Every image must have
`placeholder_image=false`, finite image/node/adjacency/output tensors,
`output_shape=[1,14]`, and `no_grad=true`.

The model uses an ImageNet-pretrained ResNet-50 with randomly initialized
MRGL graph/head and the Table V Semantic fallback. This verifies execution,
not trained diagnostic performance. Unmapped CXAS regions remain invalid;
the two lung-base approximations are explicitly recorded in cache metadata.

MIMIC training requires legitimate CAD-Chest/CheXpert/preprocessing labels,
study/dicom linkage, an explicit class-order mapping, and a missing/uncertain
label policy. NIH and CheXpert are not interchangeable merely because both
have 14 dimensions. Never replace missing supervision with all-zero labels.

実画像smokeは上記の既存手順に統合済みです。NIHは正規教師で学習、MIMICは
12枚の推論までが対象です。MIMIC用教師がないため、training成功とは扱いません。

Adapter details (existing documentation):

- [English documentation](docs/MRGL_SMOKE.md)
- [日本語ドキュメント](docs/MRGL_SMOKE.ja.md)
- [简体中文文档](docs/MRGL_SMOKE.zh-CN.md)

This repository includes the introduction to uncertain labels in Chest X-ray diagnosis. We'd like to provide the introduction of our paper and dataset below.

The dataset introduction is updated to a new link: https://github.com/MengRes/MIMIC-CAD.



## Dataset Introduction

Our dataset is based on the MIMIC-CXR dataset. The MIMIC-CXR dataset provides Chest X-ray images and corresponding radiology reports. We extract labels from the reports and provide structured files to link our labels and CXR images.

You can download the dataset from: [Dataset Download](https://physionet.org/content/cad-chest/1.0/)

The dataset is stored in Physionet, before assessing the dataset, you may register the Physionet account and assign some related agreement.

Please cite the reference if you use the dataset:

[1] Zhang, M., Hu, X., Gu, L., Harada, T., Kobayashi, K., Summers, R., & Zhu, Y. (2023). CAD-Chest: Comprehensive Annotation of Diseases based on MIMIC-CXR Radiology Report (version 1.0). PhysioNet. https://doi.org/10.13026/44pd-vz36.

[2] M. Zhang et al., "A New Benchmark: Clinical Uncertainty and Severity Aware Labeled Chest X-Ray Images with Multi-Relationship Graph Learning," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2024.3441494.
