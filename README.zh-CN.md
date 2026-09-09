# CXR_GRN — 胸部 X 光解剖区域图分类

[日本語](README.md) | [English](README.en.md) | [简体中文](README.zh-CN.md)

本 repository 将胸部 X 光表示为 26 个解剖区域，用 MRGL（Multi-Relationship Graph Learning）处理区域间关系。在原 CAD-Chest／uncertain-label 研究代码基础上，增加了**用于验证单张 CXR 训练与推理的 adapter 和 smoke 入口**。

[功能](#capabilities) · [目录](#layout) · [运行](#run) · [实现修改](#implementation) · [原文与论文信息](#archive)

<a id="capabilities"></a>

## 1. 可以运行什么

| 数据 | 功能 | 教师条件 |
|---|---|---|
| NIH ChestX-ray14 | 真实图像 training 1 batch、backward、optimizer step、inference | 使用正规 CSV 的 14-label 教师 |
| 手动 MIMIC-CXR-JPG | 共享 CXR_LLM 图像，CXAS bbox cache→MRGL 推理 | 推理不需要 paired JSON；正规标签齐备前不训练 |

本地已验证 NIH 2 张（train 1／test 1）和 MIMIC 12 张。MIMIC 图像是否匹配 Stage 1/2 annotation，不是 GRN 推理的前提。

流程为 `图像 → CXAS bbox → ResNet-50 → ROIAlign 的 26 nodes → 三类 graph → 14 维输出`。VLM 文章生成由 [CXR_LLM](../CXR_LLM/README.zh-CN.md)负责，GRPO 由 [CURV_stage3_repro](../CURV_stage3_repro/README.zh-CN.md)负责。

<a id="layout"></a>

## 2. Directory 结构

```text
CXR_GRN/
├── models/                     # 既有 model 和新增 MRGLClassifier
├── adapters/                   # NIH/MIMIC loader、CXAS cache、Semantic graph
├── scripts/                    # bbox 创建、NIH 训练／推理、MIMIC 推理入口
├── data/
│   ├── nih_smoke/              # 本地 NIH 图像、CSV、split、bbox cache
│   └── mimic_smoke/            # 本地共享图像 symlink、manifest、cache
├── tests/                      # MIMIC 一致性／不伪造标签的测试
├── outputs/                    # MIMIC 推理结果
├── docs/                       # 既有 MRGL adapter 规格与验证资料
├── requirements-mrgl-smoke.txt  # Smoke 依赖
├── anatomical_feature_extract.py
├── disease_feature_extract.py
├── feature extraction/         # 原特征提取／合并处理
├── graph_mimic_new.py
├── train.py                    # 原训练入口，与本 smoke 分开
└── utils/
```

<a id="run"></a>

## 3. 运行方法

从本 repository root 执行。验证环境为 `scdiffusion`（Python 3.9／torch 2.5.1／CXAS 0.0.18）。其他机器需准备兼容环境。

首次依赖见 `requirements-mrgl-smoke.txt`。包含图像和环境准备的完整流程见既有 RUN_GUIDE。

[日本語](../CXR_LLM/docs/RUN_GUIDE.ja.md#grn) | [English](../CXR_LLM/docs/RUN_GUIDE.en.md#grn) | [简体中文](../CXR_LLM/docs/RUN_GUIDE.zh-CN.md#grn)

### NIH：真实图像单 Batch 训练与推理

以下假设正规图像、CSV、split、CXAS cache 已准备完成。首次创建 cache 见[既有 adapter 指南](docs/MRGL_SMOKE.zh-CN.md)。

```bash
conda run --no-capture-output -n scdiffusion python scripts/smoke_train_mrgl.py \
  --csv data/nih_smoke/Data_Entry_2017.csv --image-dir data/nih_smoke/images \
  --bbox-cache data/nih_smoke/cxas_boxes.npz --split-file data/nih_smoke/train_list.txt \
  --device cpu --optimizer-step
conda run --no-capture-output -n scdiffusion python scripts/smoke_infer_mrgl.py \
  --csv data/nih_smoke/Data_Entry_2017.csv --image-dir data/nih_smoke/images \
  --bbox-cache data/nih_smoke/cxas_boxes.npz --split-file data/nih_smoke/test_list.txt --device cpu
```

成功 marker 为 `loss.backward() ok`、`optimizer.step() executed`、`torch.no_grad() forward ok`。省略 `--optimizer-step` 将不执行更新。

### MIMIC：共享图像、创建 Bbox Cache、推理

先通过 [CXR_LLM 图像整理](../CXR_LLM/docs/RUN_GUIDE.zh-CN.md#images)生成 manifest。用 symlink 共享图像，不复制、不重新下载 MIMIC。

```bash
conda run --no-capture-output -n scdiffusion python scripts/prepare_mimic_manifest.py \
  --source-manifest ../CXR_LLM/data/processed/manual_mimic_images_manifest.jsonl
conda run --no-capture-output -n scdiffusion python scripts/precompute_mimic_cxas_bboxes.py \
  --manifest data/mimic_smoke/manifest.jsonl --output data/mimic_smoke/cxas_boxes.npz --device cpu
conda run --no-capture-output -n scdiffusion python scripts/smoke_infer_mimic_mrgl.py \
  --manifest data/mimic_smoke/manifest.jsonl --bbox-cache data/mimic_smoke/cxas_boxes.npz --device cpu
conda run --no-capture-output -n scdiffusion python -m unittest discover -s tests -v
```

ID 和图像 byte 不变时可复用 cache。增加图像后重新生成 cache 和 `.npz.images.json`。

成功 marker 为 `MIMIC CXAS CACHE OK`、`MIMIC REAL IMAGE MRGL INFERENCE OK 12`。结果保存到 `outputs/mimic_mrgl_inference.json`。每张图像都需检查 placeholder false、有限 tensor、node `[1,26,2048]`、adjacency `[1,26,26]`、output `[1,14]` 和 no_grad。

<a id="implementation"></a>

## 4. 实现补充、修改与限制

| 部分 | 实现策略 |
|---|---|
| 单张 CXR model | 在 `models/mrgl_classifier.py` 增加 26-node MRGL，与需要图像 pair／question embedding 的既有路径分开 |
| 图像／教师 adapter | NIH 使用正规 CSV；MIMIC 验证 manifest 和 SHA-256，推理 dataset 不返回 target |
| CXAS | 无直接映射的 7 个区域及空 mask 保持 invalid；metadata 明确记录两个 lung-base 近似 |
| ROIAlign | Invalid node 特征置 0；1×1 sentinel 仅用于计算，不是解剖 bbox |
| Semantic graph | Table V group fallback，与论文精确 adjacency 区分 |
| Weight | ResNet 使用 ImageNet 预训练，MRGL graph/head 随机初始化 |
| Ensemble | Spatial／Semantic／Implicit 以 0.3／0.4／0.3 合成 |

这是执行验证，不是论文分类性能、18/60-label 配置或完整 CAD-Chest 实验复现。MIMIC training 需要正规标签、study/dicom 对应、类别顺序及缺失／uncertainty 策略。NIH 与 CheXpert 即使同为 14 维，taxonomy 也不同，不使用 dummy／全 0 教师代替。

不改写原 model／preprocessing 文件，通过 wrapper／adapter 补充。详细规格见[日本語](docs/MRGL_SMOKE.ja.md)、[English](docs/MRGL_SMOKE.md)、[简体中文](docs/MRGL_SMOKE.zh-CN.md)。初期 MIMIC 运行记录保留在[既有资料](docs/MIMIC_SMOKE.ja.md)中。

<a id="archive"></a>

## 附录：既有原文、数据集介绍、引用与目录历史

以下按原语言完整保留整理前的 README，未省略论文／dataset 说明、引用、英文运行命令或机器相关目录信息。

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
