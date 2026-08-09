# 最小单张CXR MRGL训练与推理

语言: [English](MRGL_SMOKE.md) | [日本語](MRGL_SMOKE.ja.md) | **简体中文**

## 状态

已使用源自NIH ChestX-ray14的真实图像，分别完成一个batch的MRGL训练和推理smoke test。
`loss.backward()` 和 `optimizer.step()` 均成功完成。

这是一个优先与论文保持一致的最小smoke pipeline。其目标不是完整复现实验、评估指标、
18/60-label dataset或长时间训练。

## 新增文件

- [`models/mrgl_classifier.py`](../models/mrgl_classifier.py)
  - 单张CXR ResNet-50 backbone。
  - 使用ROIAlign提取26个anatomical node features。
  - Spatial、Semantic和Implicit三个graph branch。
  - 公式6的graph convolution和公式7的weighted ensemble。
- [`adapters/nih_dataset.py`](../adapters/nih_dataset.py)
  - 使用14维multi-label target的NIH ChestX-ray14 DataLoader。
- [`adapters/cxas_bbox_cache.py`](../adapters/cxas_bbox_cache.py)
  - 论文Table V的26个区域与CXAS class之间的显式mapping。
  - 从segmentation mask到bounding rectangle再到normalized XYXY cache的转换。
- [`adapters/semantic_graph.py`](../adapters/semantic_graph.py)
  - 基于Table V group的fallback Semantic adapter。
  - 外部precomputed adjacency matrix loader。
- [`scripts/precompute_cxas_bboxes.py`](../scripts/precompute_cxas_bboxes.py)
- [`scripts/smoke_train_mrgl.py`](../scripts/smoke_train_mrgl.py)
- [`scripts/smoke_infer_mrgl.py`](../scripts/smoke_infer_mrgl.py)
- [`requirements-mrgl-smoke.txt`](../requirements-mrgl-smoke.txt)
- `.gitignore`、adapter package初始化，以及英文、日文和简体中文版本的本文档。

未修改原upstream中已有的model或preprocessing file。

## 与论文和原repository的关系

- 未复用 `ChangeDetector` 和ReGAT相关class，因为它们接收before/after图像和question
  embedding。该结构与论文Figure 3的单张CXR classifier不一致。
- 本实现沿用 `anatomical_feature_extract.py` 中的26-node顺序、ResNet layer4的
  2048维ROI features以及ROIAlign(1x1)设计。
- Graph最终维度为256，并使用三层graph-convolution。Spatial、Semantic和Implicit的
  ensemble weight为 `0.3 / 0.4 / 0.3`。
- `feature extraction/combine_dicts.py` 合并anatomy node和disease-location detector
  node。本实现不将其视为论文的26-anatomy-node Semantic graph。

参考:

- [MRGL论文](https://doi.org/10.1109/TMI.2024.3441494)
- [CXAS repository与class定义](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation)

## Pipeline

```text
NIH CXR [B,3,256,256]
  -> ResNet-50 layer4 feature map [B,2048,8,8]
  -> 已cache的26个anatomical XYXY boxes
  -> ROIAlign(1x1)
  -> node features [B,26,2048]
  -> Spatial / Semantic / Implicit graph branches
  -> 3层graph-convolution，最终维度256
  -> node average pooling
  -> 各branch独立的14-label predictions
  -> weighted average (0.3 / 0.4 / 0.3)
  -> output [B,14]
```

Model返回probability，因此smoke training script使用 `BCELoss`。Adam的default learning
rate按照论文设为 `0.01`。

## 环境配置

安装smoke-test dependency：

```bash
python -m pip install -r requirements-mrgl-smoke.txt
```

第一次运行CXAS bbox预计算时，CXAS会下载pretrained `UNet_ResNet50_default`
checkpoint。第一次创建default MRGL backbone时，会下载ImageNet ResNet-50 weights。

## NIH ChestX-ray14数据

使用标准NIH metadata CSV、image directory，以及可选的官方split list。典型目录结构如下：

```text
/path/nih/
  Data_Entry_2017.csv
  images/
    00000001_000.png
    ...
  train_val_list.txt
  test_list.txt
```

`Finding Labels` 按以下顺序转换为14维target：

```text
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
Pleural_Thickening, Hernia
```

`No Finding` 转换为全零target vector。

## 预计算CXAS anatomical bbox cache

训练前运行一次CXAS。训练期间不会重复执行CXAS inference。

Training cache：

```bash
python scripts/precompute_cxas_bboxes.py \
  --csv /path/nih/Data_Entry_2017.csv \
  --image-dir /path/nih/images \
  --split-file /path/nih/train_val_list.txt \
  --output /path/cache/cxas_train_boxes.npz \
  --device cpu
```

Test cache：

```bash
python scripts/precompute_cxas_bboxes.py \
  --csv /path/nih/Data_Entry_2017.csv \
  --image-dir /path/nih/images \
  --split-file /path/nih/test_list.txt \
  --output /path/cache/cxas_test_boxes.npz \
  --device cpu
```

使用CUDA进行CXAS inference时，请传入CUDA index，例如 `--device 0`。使用 `--limit N`
可以仅预计算前 `N` 个匹配图像，以构建小型smoke dataset。

cache保存normalized `[x0, y0, x1, y1]` boxes、26个node各自的validity flag和mapping
metadata。

## Training smoke test

以下command执行一次forward、计算 `BCELoss`、运行 `loss.backward()`，并执行
`optimizer.step()`：

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

即使省略 `--optimizer-step`，script仍会运行 `loss.backward()`，且配置好的optimizer已处于
可执行step的状态。

## Inference smoke test

以下command在 `torch.no_grad()` 中执行一个test batch：

```bash
python scripts/smoke_infer_mrgl.py \
  --csv /path/nih/Data_Entry_2017.csv \
  --image-dir /path/nih/images \
  --bbox-cache /path/cache/cxas_test_boxes.npz \
  --split-file /path/nih/test_list.txt \
  --batch-size 1 \
  --device cpu
```

## 重要option

- `--image-size`：input size；default为与论文设置一致的 `256`。
- `--iou-threshold`：Spatial graph的IoU threshold；default为 `0.3`。
- `--semantic-adj`：从 `.npy` 或 `.npz` 加载外部 `[26,26]` / `[B,26,26]`
  Semantic adjacency。
- `--no-pretrained-backbone`：保留ResNet-50 architecture，但不下载或加载ImageNet
  weights。
- `--num-workers`：DataLoader worker数量；为保证smoke test可移植，default为 `0`。
- `--seed`：PyTorch random seed；default为 `0`。

## Anatomical mapping限制

显式CXAS mapping的组成如下：

```text
15 exact
1 composed
1 terminology match
2 approximations
7 unavailable
```

左右lower-lung region使用对应的CXAS lung-base mask，并明确标记为approximation。CXAS没有
直接对应左右hilum、左右costophrenic sulcus、main bronchus、superior vena cava或
cavoatrial region的class。这七个node保持invalid，其node feature被置零，不会为其分配
推测的mask。

## Semantic graph限制

论文的Semantic graph需要每个anatomical node的graph-Grad-CAM top-1/top-2 abnormality
label，以及Figure 7中的abnormality co-occurrence knowledge graph。公开repository未包含
完成该构建所需的完整machine-readable artifact。

因此，default `TableVGroupSemanticAdapter` 被明确分离为smoke专用fallback，仅连接Table V
四个group各自内部的node。本实现不声称它复现了论文的Semantic graph。如果已有重建或预计算
adjacency，可通过 `--semantic-adj` 提供。

## 已验证tensor shape

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

## 已验证smoke-test结果

Training：

```text
loss 0.8319577574729919
loss.backward() ok
optimizer.step() executed
```

Inference：

```text
output.shape (1, 14)
torch.no_grad() forward ok
```

验证环境为PyTorch 2.5.1、torchvision 0.20.1、CXAS 0.0.18和CPU。smoke-test图像及
生成的CXAS cache保存在 `data/` 下，并从Git中排除。
