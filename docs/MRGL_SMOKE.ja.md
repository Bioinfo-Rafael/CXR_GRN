# 最小single-CXR MRGL Training / Inference

言語: [English](MRGL_SMOKE.md) | **日本語** | [简体中文](MRGL_SMOKE.zh-CN.md)

## 最短手順

repository rootで、2つのNIH pathを置き換えて実行します。

```bash
python -m pip install -r requirements-mrgl-smoke.txt

python scripts/precompute_cxas_bboxes.py \
  --csv /path/nih/Data_Entry_2017.csv --image-dir /path/nih/images \
  --output /path/cache/cxas_boxes.npz --limit 2 --device cpu

python scripts/smoke_train_mrgl.py \
  --csv /path/nih/Data_Entry_2017.csv --image-dir /path/nih/images \
  --bbox-cache /path/cache/cxas_boxes.npz --device cpu --optimizer-step

python scripts/smoke_infer_mrgl.py \
  --csv /path/nih/Data_Entry_2017.csv --image-dir /path/nih/images \
  --bbox-cache /path/cache/cxas_boxes.npz --device cpu
```

## 状態

NIH ChestX-ray14由来の実画像に対し、MRGLのtrainingとinferenceのsmoke testを
それぞれ1 batch実行しました。`loss.backward()` と `optimizer.step()` は両方とも
正常に完了しています。

これは論文との整合性を重視した最小smoke pipelineです。完全な実験再現、評価指標、
18/60-label dataset、長時間trainingを目的としたものではありません。

## 追加ファイル

- [`models/mrgl_classifier.py`](../models/mrgl_classifier.py)
  - single-CXR ResNet-50 backbone。
  - ROIAlignで抽出する26 anatomical node features。
  - Spatial、Semantic、Implicitの3 graph branch。
  - Eq.6 graph convolutionとEq.7 weighted ensemble。
- [`adapters/nih_dataset.py`](../adapters/nih_dataset.py)
  - 14-dimensional multi-label targetを持つNIH ChestX-ray14 DataLoader。
- [`adapters/cxas_bbox_cache.py`](../adapters/cxas_bbox_cache.py)
  - 論文Table Vの26領域とCXAS classの明示的mapping。
  - segmentation maskからbounding rectangle、normalized XYXY cacheへの変換。
- [`adapters/semantic_graph.py`](../adapters/semantic_graph.py)
  - Table V groupによるfallback Semantic adapter。
  - 外部precomputed adjacency matrixのloader。
- [`scripts/precompute_cxas_bboxes.py`](../scripts/precompute_cxas_bboxes.py)
- [`scripts/smoke_train_mrgl.py`](../scripts/smoke_train_mrgl.py)
- [`scripts/smoke_infer_mrgl.py`](../scripts/smoke_infer_mrgl.py)
- [`requirements-mrgl-smoke.txt`](../requirements-mrgl-smoke.txt)
- `.gitignore`、adapter package初期化、英語・日本語・簡体字中国語の本document。

既存upstreamのmodelおよびpreprocessing fileは変更していません。

## 論文・元repositoryとの関係

- `ChangeDetector` とReGAT関連classはbefore/after画像とquestion embeddingを入力するため
  再利用していません。この構造は論文Figure 3のsingle-CXR classifierと一致しません。
- `anatomical_feature_extract.py` にある26-node順序、ResNet layer4の
  2048-dimensional ROI features、ROIAlign(1x1)設計を踏襲しています。
- Graphの最終次元は256、graph-convolutionは3層です。Spatial、Semantic、Implicitの
  ensemble weightは `0.3 / 0.4 / 0.3` です。
- `feature extraction/combine_dicts.py` はanatomy nodeとdisease-location detector nodeを
  結合します。これを論文の26-anatomy-node Semantic graphとはみなしていません。

参照:

- [MRGL論文](https://doi.org/10.1109/TMI.2024.3441494)
- [CXAS repositoryとclass定義](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation)

## Pipeline

```text
NIH CXR [B,3,256,256]
  -> ResNet-50 layer4 feature map [B,2048,8,8]
  -> cache済み26 anatomical XYXY boxes
  -> ROIAlign(1x1)
  -> node features [B,26,2048]
  -> Spatial / Semantic / Implicit graph branches
  -> 3 graph-convolution layers、最終次元256
  -> node average pooling
  -> branch固有の14-label predictions
  -> weighted average (0.3 / 0.4 / 0.3)
  -> output [B,14]
```

Modelはprobabilityを返すため、smoke training scriptは `BCELoss` を使います。
Adamのdefault learning rateは論文に合わせて `0.01` です。

## 環境構築

smoke-test用dependencyをinstallします。

```bash
python -m pip install -r requirements-mrgl-smoke.txt
```

CXAS bboxの事前計算を初めて実行すると、CXASはpretrained
`UNet_ResNet50_default` checkpointをdownloadします。defaultのMRGL backboneは、
modelの初回作成時にImageNet ResNet-50 weightsをdownloadします。

## NIH ChestX-ray14 data

標準NIH metadata CSV、image directory、任意で公式split listを使用します。
代表的な配置は次のとおりです。

```text
/path/nih/
  Data_Entry_2017.csv
  images/
    00000001_000.png
    ...
  train_val_list.txt
  test_list.txt
```

`Finding Labels` は次の順序の14-dimensional targetに変換されます。

```text
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
Pleural_Thickening, Hernia
```

`No Finding` は全要素0のtarget vectorになります。

## CXAS anatomical bbox cacheの事前計算

training前にCXASを1回実行します。training中にCXAS inferenceを繰り返しません。

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

CUDAでCXAS inferenceを行う場合は `--device 0` のようにCUDA indexを指定します。
小規模smoke datasetの先頭 `N` 件だけを事前計算するには `--limit N` を使います。

cacheにはnormalized `[x0, y0, x1, y1]` boxes、26 nodeそれぞれのvalidity flag、
mapping metadataが保存されます。

## Training smoke test

次のcommandは1回のforward、`BCELoss` 計算、`loss.backward()`、
`optimizer.step()` を実行します。

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

`--optimizer-step` を省略しても `loss.backward()` は実行され、設定済みoptimizerは
step可能な状態になります。

## Inference smoke test

次のcommandはtest 1 batchを `torch.no_grad()` 内で実行します。

```bash
python scripts/smoke_infer_mrgl.py \
  --csv /path/nih/Data_Entry_2017.csv \
  --image-dir /path/nih/images \
  --bbox-cache /path/cache/cxas_test_boxes.npz \
  --split-file /path/nih/test_list.txt \
  --batch-size 1 \
  --device cpu
```

## 主要option

- `--image-size`: input size。defaultは論文設定に合わせた `256`。
- `--iou-threshold`: Spatial graphのIoU threshold。defaultは `0.3`。
- `--semantic-adj`: `.npy` または `.npz` の外部 `[26,26]` / `[B,26,26]`
  Semantic adjacencyを読み込む。
- `--no-pretrained-backbone`: ResNet-50 architectureは維持し、ImageNet weightsを
  downloadおよびloadしない。
- `--num-workers`: DataLoader worker数。portableなsmoke test用defaultは `0`。
- `--seed`: PyTorch random seed。defaultは `0`。

## Anatomical mappingの制約

明示的CXAS mappingの内訳は次のとおりです。

```text
15 exact
1 composed
1 terminology match
2 approximations
7 unavailable
```

左右lower-lung regionには対応するCXAS lung-base maskを使用し、approximationと明記して
います。CXASには左右hilum、左右costophrenic sulcus、main bronchus、superior vena cava、
cavoatrial regionの直接対応classがありません。この7 nodeはinvalidのままとし、node
featureをゼロ化します。推測したmaskは割り当てません。

## Semantic graphの制約

論文のSemantic graphには、各anatomical nodeに対するgraph-Grad-CAM top-1/top-2
abnormality labelと、Figure 7のabnormality co-occurrence knowledge graphが必要です。
公開repositoryには、この構築に必要な完全なmachine-readable artifactがありません。

そのためdefaultの `TableVGroupSemanticAdapter` は、Table Vの4 group内でnodeを結ぶ
smoke専用fallbackとして明示的に分離しています。論文Semantic graphの再現を主張する
ものではありません。再構築または事前計算したadjacencyが利用可能な場合は
`--semantic-adj` で指定します。

## 確認済みtensor shape

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

## 確認済みsmoke-test結果

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

検証環境はPyTorch 2.5.1、torchvision 0.20.1、CXAS 0.0.18、CPUです。
smoke-test imageと生成したCXAS cacheは `data/` 以下に保存し、Git対象外にしています。
