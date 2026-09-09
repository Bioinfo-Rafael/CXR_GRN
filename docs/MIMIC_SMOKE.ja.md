# MIMIC実画像MRGL smoke（label不要・画像コピーなし）

**追加画像取込後の最新状態は12/12枚でinference成功、symlink 12本、画像コピー0 bytes。**
Stage 1/2・全ID・CXAS有効node数を含む最新結果は
[統合RUN_GUIDEの検証結果](../../CXR_LLM/docs/RUN_GUIDE.ja.md#results)。
下記commandは同じまま全12枚を処理する。以前の4枚・valid node数は履歴として残している。

全repository・全ローカルsmokeの入口は
[RUN_GUIDE.ja.md](../../CXR_LLM/docs/RUN_GUIDE.ja.md)。
絶対path: `/Users/cls-lab/Git/LinGu/CXR_LLM/docs/RUN_GUIDE.ja.md`。
NIH training/inference手順は同書[第6節](../../CXR_LLM/docs/RUN_GUIDE.ja.md#grn)、adapterの詳細は[MRGL_SMOKE.ja.md](MRGL_SMOKE.ja.md)にある。

## 再実行

このマシンのCXAS環境は`scdiffusion`（Python 3.9 / torch 2.5.1 / CXAS 0.0.18）。
`cxr-curv`やシステムPythonと混ぜない。

```bash
cd /Users/cls-lab/Git/LinGu/CXR_GRN
conda run --no-capture-output -n scdiffusion python scripts/prepare_mimic_manifest.py \
  --source-manifest /Users/cls-lab/Git/LinGu/CXR_LLM/data/processed/manual_mimic_images_manifest.jsonl
conda run --no-capture-output -n scdiffusion python scripts/precompute_mimic_cxas_bboxes.py \
  --manifest data/mimic_smoke/manifest.jsonl --output data/mimic_smoke/cxas_boxes.npz --device cpu
conda run --no-capture-output -n scdiffusion python scripts/smoke_infer_mimic_mrgl.py \
  --manifest data/mimic_smoke/manifest.jsonl --bbox-cache data/mimic_smoke/cxas_boxes.npz --device cpu
conda run --no-capture-output -n scdiffusion python -m unittest discover -s tests -v
```

成功marker: `MIMIC CXAS CACHE OK`、`MIMIC REAL IMAGE MRGL INFERENCE OK 4`、unit 6件OK。
resultは`outputs/mimic_mrgl_inference.json`。画像が増えたら同じcommandで全manifest画像を処理する。
`prepare`はsymlinkのみ作り、既存異内容fileを上書きしない。

## 2026-09-09確認結果

4枚すべてplaceholderなし、`torch.no_grad()`成功、入力/feature/adjacency/outputはfinite。
CXAS valid nodesはmanifest順に19 / 15 / 19 / 17。
画像 `[1,3,256,256]` → ResNet layer4 `[1,2048,8,8]` → ROI features `[1,26,2048]`
→ Spatial/Semantic/Implicit各`[1,26,26]` → output `[1,14]`。
共有symlink 4本、追加画像コピー0 bytes。CXAS cacheとSHA sidecarは新規生成。

既存`precompute_cxas_cache`と解剖mappingは変更せず利用。
非対応7領域と空mask領域はinvalidのまま。計算用ROI sentinelを使ってもvalidityはfalse、
node featuresはゼロ化し、推測bboxは作らない。

MIMIC adapterはdicom/study/subject/local/relative_pathとSHAを検証し、targetを返さない。
CXAS cacheはSHA sidecarで実画像と結び付け、stale cacheを拒否。
モデルはImageNet ResNet-50＋ランダムMRGL graph/head。SemanticはTable V group fallback。
14 outputsはNIH headの配線検証であり、学習済みのMIMIC診断結果ではない。

正当なCAD-Chest/MIMIC CheXpert教師や対応preprocessing artifactは確認できなかった。
NIH/CheXpertがともに14次元でもtaxonomyは異なり、そのままtrainingには使えない。
MIMIC trainingは未実行。既存NIH training/backward/optimizer/inferenceは別途再実行成功。
label欠如をdummy all-zero targetで埋める処理は実装していない。

## 追加コード

- `adapters/mimic_manifest.py`: label-free dataset、manifest整合性・SHA・bbox検証。
- `scripts/prepare_mimic_manifest.py`: CXR_LLM manifestを共有interfaceに変換。
- `scripts/precompute_mimic_cxas_bboxes.py`: 既存CXAS処理のmanifest wrapper。
- `scripts/smoke_infer_mimic_mrgl.py`: 全画像のno_grad・finite・shape assertionsとJSON保存。
- `tests/test_mimic_manifest.py`: labelなし、ID/checksum/重複/欠損、invalid node・stale cacheテスト。

既存NIH adapter、training/inference scripts、model、CXAS mappingは変更していない。
