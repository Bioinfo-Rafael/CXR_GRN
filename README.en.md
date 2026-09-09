# CXR_GRN — Anatomical Graph Classification of Chest X-rays

[日本語](README.md) | [English](README.en.md) | [简体中文](README.zh-CN.md)

This repository represents chest X-rays as 26 anatomical regions and models their relationships with MRGL (Multi-Relationship Graph Learning). It adds **adapters and smoke entry points for single-CXR training/inference** to the original CAD-Chest/uncertain-label research code.

[Capabilities](#capabilities) · [Layout](#layout) · [Running](#run) · [Implementation changes](#implementation) · [Original text/paper information](#archive)

<a id="capabilities"></a>

## 1. What You Can Run

| Data | Capabilities | Supervision requirements |
|---|---|---|
| NIH ChestX-ray14 | One real-image training batch, backward, optimizer step, inference | Legitimate 14-label CSV targets |
| Manually downloaded MIMIC-CXR-JPG | Share CXR_LLM images, build CXAS bbox cache, run MRGL inference | No paired JSON for inference; no training until legitimate labels are available |

Locally verified: two NIH images (one train/one test) and 12 MIMIC images. Matching a Stage 1/2 annotation is not a prerequisite for GRN inference.

The pipeline is `image → CXAS bbox → ResNet-50 → 26 ROIAlign nodes → three graph types → 14-dimensional output`. VLM report generation belongs to [CXR_LLM](../CXR_LLM/README.en.md); GRPO belongs to [CURV_stage3_repro](../CURV_stage3_repro/README.en.md).

<a id="layout"></a>

## 2. Directory Layout

```text
CXR_GRN/
├── models/                     # Existing models and added MRGLClassifier
├── adapters/                   # NIH/MIMIC loaders, CXAS cache, Semantic graph
├── scripts/                    # Bboxes, NIH training/inference, MIMIC inference
├── data/
│   ├── nih_smoke/              # Local NIH images, CSV, splits, bbox cache
│   └── mimic_smoke/            # Local shared-image symlinks, manifest, cache
├── tests/                      # MIMIC integrity / no-fabricated-label tests
├── outputs/                    # MIMIC inference results
├── docs/                       # Existing MRGL adapter specifications/evidence
├── requirements-mrgl-smoke.txt  # Smoke dependencies
├── anatomical_feature_extract.py
├── disease_feature_extract.py
├── feature extraction/         # Original feature extraction/combination
├── graph_mimic_new.py
├── train.py                    # Original training entry, separate from this smoke
└── utils/
```

<a id="run"></a>

## 3. How to Run

Run from this repository root. The verified environment is `scdiffusion` (Python 3.9 / torch 2.5.1 / CXAS 0.0.18). Prepare a compatible environment on another machine.

Initial dependencies are in `requirements-mrgl-smoke.txt`. The existing RUN_GUIDE covers the complete workflow, including environment and image preparation.

[日本語](../CXR_LLM/docs/RUN_GUIDE.ja.md#grn) | [English](../CXR_LLM/docs/RUN_GUIDE.en.md#grn) | [简体中文](../CXR_LLM/docs/RUN_GUIDE.zh-CN.md#grn)

### NIH: One Real-Image Training Batch and Inference

These commands assume legitimate images, CSV, splits, and CXAS cache already exist. For initial cache creation, see the [existing adapter guide](docs/MRGL_SMOKE.md).

```bash
conda run --no-capture-output -n scdiffusion python scripts/smoke_train_mrgl.py \
  --csv data/nih_smoke/Data_Entry_2017.csv --image-dir data/nih_smoke/images \
  --bbox-cache data/nih_smoke/cxas_boxes.npz --split-file data/nih_smoke/train_list.txt \
  --device cpu --optimizer-step
conda run --no-capture-output -n scdiffusion python scripts/smoke_infer_mrgl.py \
  --csv data/nih_smoke/Data_Entry_2017.csv --image-dir data/nih_smoke/images \
  --bbox-cache data/nih_smoke/cxas_boxes.npz --split-file data/nih_smoke/test_list.txt --device cpu
```

Success markers: `loss.backward() ok`, `optimizer.step() executed`, and `torch.no_grad() forward ok`. Omitting `--optimizer-step` skips the update.

### MIMIC: Shared Images, Bbox Cache, and Inference

First create a manifest through [CXR_LLM image preparation](../CXR_LLM/docs/RUN_GUIDE.en.md#images). Images are shared through symlinks without copying or downloading MIMIC again.

```bash
conda run --no-capture-output -n scdiffusion python scripts/prepare_mimic_manifest.py \
  --source-manifest ../CXR_LLM/data/processed/manual_mimic_images_manifest.jsonl
conda run --no-capture-output -n scdiffusion python scripts/precompute_mimic_cxas_bboxes.py \
  --manifest data/mimic_smoke/manifest.jsonl --output data/mimic_smoke/cxas_boxes.npz --device cpu
conda run --no-capture-output -n scdiffusion python scripts/smoke_infer_mimic_mrgl.py \
  --manifest data/mimic_smoke/manifest.jsonl --bbox-cache data/mimic_smoke/cxas_boxes.npz --device cpu
conda run --no-capture-output -n scdiffusion python -m unittest discover -s tests -v
```

Reuse the cache when IDs and image bytes are unchanged. After adding images, regenerate the cache and `.npz.images.json`.

Success markers: `MIMIC CXAS CACHE OK` and `MIMIC REAL IMAGE MRGL INFERENCE OK 12`. Results go to `outputs/mimic_mrgl_inference.json`. Every image must have placeholder false, finite tensors, nodes `[1,26,2048]`, adjacency `[1,26,26]`, output `[1,14]`, and no_grad.

<a id="implementation"></a>

## 4. Implementation Additions, Changes, and Limits

| Area | Implementation policy |
|---|---|
| Single-CXR model | Added 26-node MRGL in `models/mrgl_classifier.py`, separate from existing paths requiring image pairs/question embeddings |
| Image/label adapters | NIH reads legitimate CSV labels; MIMIC verifies manifests/SHA-256 and returns no inference target |
| CXAS | Seven unmapped regions and empty masks stay invalid; two lung-base approximations are explicit in metadata |
| ROIAlign | Invalid-node features are zero; the 1×1 sentinel is computational, not anatomical |
| Semantic graph | Table V group fallback, not the paper-exact adjacency |
| Weights | ImageNet-pretrained ResNet; randomly initialized MRGL graph/head |
| Ensemble | Spatial/Semantic/Implicit combined with weights 0.3/0.4/0.3 |

This checks execution, not paper classification performance, 18/60-label settings, or all CAD-Chest experiments. MIMIC training requires legitimate labels, study/dicom linkage, class ordering, and a missing/uncertainty policy. NIH and CheXpert have different taxonomies despite both having 14 dimensions; dummy/all-zero labels are not substitutes.

Original model/preprocessing files are not rewritten; wrappers/adapters supplement them. Detailed specifications: [日本語](docs/MRGL_SMOKE.ja.md), [English](docs/MRGL_SMOKE.md), [简体中文](docs/MRGL_SMOKE.zh-CN.md). The [existing MIMIC notes](docs/MIMIC_SMOKE.ja.md) retain initial execution history.

<a id="archive"></a>

## Appendix: Original Text, Dataset Introduction, Citations, and Layout History

The complete previous README is preserved verbatim below in its original language. Paper/dataset descriptions, citations, English commands, and machine-specific layout information are not omitted.

<details>
<summary>整理前のREADME全文 / Previous README (verbatim) / 原README全文</summary>

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

</details>
