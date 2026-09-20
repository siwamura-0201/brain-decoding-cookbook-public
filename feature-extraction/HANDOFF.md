# 引継ぎ資料 — brain-decoding-cookbook への feature extraction 追加

作成日: 2026-09-17
対象: fork した `brain-decoding-cookbook-public` / ブランチ `feature-extraction`

---

## 0. 一行サマリ

cookbook に欠けていたパイプライン最上流（画像 → DNN 特徴量）の抽出コードを `feature-extraction/` として追加した。実装と構造検証は完了、**実データでの数値照合のみ未実施**（刺激画像が非配布のため）。

---

## 1. 背景

cookbook のパイプラインは3段階だが、最上流だけコードが存在せず、figshare からの事前抽出済み特徴量のダウンロードで代替されていた。

```
画像 ──(特徴抽出)──▶ features ──▶ [feature-decoding] ──▶ decoded features ──▶ [reconstruction]
       ここが欠落              (submodule)                                   (cookbook 本体)
```

新しい刺激画像で解析を回せるようにするのが目的。

---

## 2. リポジトリ構造の理解（調査済み・再調査不要）

### feature-decoding は DNN を一切動かさない

`requirements.txt` は `bdpy / fastl2lir / hydra-core / matplotlib / numpy<2 / pyyaml / tqdm` のみで **torch も caffe もない**。全スクリプトを `torch|caffe|model_factory|layer_map|FeatureExtractor|prototxt` で grep して 0 件。特徴量は `bdpy.dataform.Features` で `.mat` を読むだけ。

→ **`caffe/VGG19` はコードの仕様ではなくディレクトリ名としてのラベル**。抽出コードをここに置いてはいけない（torch 依存ゼロという設計が壊れる）。

### cookbook 本体と submodule は同一の `data/` を共有する想定

config 上の入出力パスが文字列レベルで一致する。

| | feature-decoding が出力 | reconstruction が入力 |
|---|---|---|
| デコーダ | `./data/feature_decoders/ImageNetTraining/deeprecon_fmriprep_pyfastl2lir_alpha100_allunits/caffe/VGG19` | 同一 |
| decoded features | `./data/decoded_features/ImageNetTest/train_deeprecon_rep5_test_ImageNetTest_fmriprep_pyfastl2lir_alpha100_allunits/caffe/VGG19` | 同一 |

`recon_icnn_image_gd.py` はデコーダ内部の `model/y_mean.mat` も直接読む（バイアス補正）。構造的に依存している。

cookbook はルートの `data/` を各解析ディレクトリへ symlink で配っている（`reconstruction/data -> ../data`、mode 120000）。submodule 側は独自の `data/` を持ち、この仕組みの外にある。hydra は cwd を移動しないため、**どこから実行するかで `./data/` の解決先が変わる**。

### cookbook がコードで名指しするモデルは2つだけ

| モデル | 状態 |
|---|---|
| **VGG19**（`caffe/VGG19`、19層 `conv1_1`…`fc8`） | 現役・主経路 |
| AlexNet（`caffe/bvlc_alexnet_bin`、8層） | レガシー（旧 config スキーマ、スクリプトは `archive/`、`_bin` は二値化特徴量） |

`files.json` には BHscore 用の16ネットワーク分の特徴量もあるが、**それらを読む config は両リポジトリに一本もない**。抽出対象ではない。

### 必要ファイルの入手経路

`data/` で `uv run download.py recon_demo` を実行すると、`data/models/VGG_ILSVRC_19_layers/` に展開される。

```
VGG_ILSVRC_19_layers.pt                                      574 MB  エンコーダ重み
ilsvrc_2012_mean.npy                                         1.5 MB  ImageNet 平均画像
estimated_cnn_feat_std_VGG_ILSVRC_19_layers_..._dof1.mat      53 KB  recon 用（抽出では不要）
```

**刺激画像は配布されていない。** BHscore の README に申請フォーム `https://forms.gle/ujvA34948Xg49jdn9` の記載あり。ラボ内なら既にアクセスがあるかもしれない。

---

## 3. 追加したもの

```
feature-extraction/
├── extract_features.py      6,399 B   抽出本体
├── compare_features.py      3,375 B   配布特徴量との層別照合
├── config/
│   ├── encoder/vgg19.yaml     428 B
│   ├── extract_features_vgg19_ImageNetTest.yaml
│   └── extract_features_vgg19_ImageNetTraining.yaml
├── data   -> ../data              symlink (mode 120000)
├── models -> ../data/models       symlink (mode 120000)
└── README.md                4,442 B
```

### 実行方法

```bash
uv run extract_features.py config/extract_features_vgg19_ImageNetTest.yaml
```

`--override device=cpu batch_size=8` で上書き可（bdpy の `init_hydra_cfg` の CLI は `<config> [-o/--override ...] [-a/--analysis ...]`）。

出力は `<features.path>/<layer>/<image label>.mat`、キー `feat`。全層揃っている画像は forward 前にスキップするので中断・再開可能。

---

## 4. 設計判断と根拠 — **変更する前に必ず読むこと**

| 判断 | 根拠 |
|---|---|
| **cookbook 本体側に置く** | feature-decoding は torch 依存ゼロを保っている。本体には既に torch・encoder config・共有 `data/` がある |
| **encoder config 差し替え式（VGG19 専用にしない）** | `recon_icnn_image_gd.py` と同じ作法。ネットワーク固有の記述をコードに持たない |
| **`layer_mapping` を `FeatureExtractor` の第3引数に渡す** | これをしないとレイヤ名が torch モジュール名（`features[0]`）になり、既存 config・配布特徴量と名前空間がずれて読めなくなる |
| **前処理は Caffe 規約（224×224 強制・BGR・0–255・平均減算）** | 配布特徴量が Caffe 由来。`ToTensor()` で 1/255 スケールすると全く別物になる |
| **出力先は `pytorch/VGG19`（`caffe/VGG19` ではない）** | PyTorch 移植版が Caffe 特徴量を完全再現する保証がまだない。`<framework>/<network>` 規約に従って分離 |
| **`bdpy.dataform.save_feature` を使う** | 先頭 sample 軸を保つ形式を保証。`Features.get_features` は `np.concatenate(axis=0)` するので、この軸を落とすと読めない |
| **encoder config を `reconstruction/` と symlink 共有しない** | あちらは `/home/kiss/data/models_shared/...` というラボ内絶対パスで、`download.py` で入手した人には解決できない（元リポジトリの不具合）。こちらは相対パスで書き直してある |

### 参照した3スクリプトとの関係

| | A: BHscore (Caffe) | B: spurious (PyTorch) | C: ImageNetTraining_v3 | 本実装 |
|---|---|---|---|---|
| リサイズ | 比率無視で入力サイズへ | 短辺224・比率保持 | 比率無視で 224×224 | **C と同じ** |
| レイヤ名 | Caffe blob 名 | torch モジュール名 | Caffe 流（`layer_map` 渡し） | **C と同じ** |
| 設定 | ベタ書き | YAML | ベタ書き | **B 寄り（hydra）** |

C を骨格に、B の config 駆動・再開処理を取り込んだ形。C から変えたのは4点のみ：バッチ forward、forward 前の再開判定、`torch.no_grad()`、`img.convert("RGB")` の戻り値を捨てるバグ修正（P/CMYK 画像が壊れる）。

---

## 5. 検証状況

```
画像 →[読込・リサイズ]→[BGR・平均減算]→ テンソル →[VGG19 forward]→[層の取り出し]→ .mat
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^               ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
       C と bit-identical                             notebook で確認   実測で確認
       ただし A との照合は未実施                       (MSE <= 6.3e-10)
```

### 実施済み

**(a) スクリプト単体**（Python 3.11 + bdpy 0.26 + torch の隔離環境、合成画像と構造的に妥当な `.pt`）

- 19層すべて出力、`Features` で読める。`conv1_1 (1,64,224,224)` / `fc6 (1,4096)` / `fc8 (1,1000)`
- 前処理が参照実装 C と **bit-identical**（RGB・グレースケール、`max|diff|=0`）
- 再開：2回目は全スキップ、モデルをロードせず終了
- `batch_size=1` vs `3`：conv 層は完全一致、fc 層のみ float32 の丸め差（`max|diff|~1e-8`）
- `image_mean_file` / `image_mean` 両経路で出力一致

**(b) 配布特徴量の規約**（fc6 772 KB・conv5_4 18 MB をダウンロードして検査。刺激画像は不要）

```
fc6      shape=(1, 4096)        float32  frac<0 = 0.72〜0.80
conv5_4  shape=(1, 512, 14, 14) float32  frac<0 = 0.93
```

→ **配布特徴量は pre-ReLU**。`layer_map('vgg19')` は Conv2d / Linear モジュールを指すので hook 出力も pre-ReLU。**キー・形状・活性の位置・dtype の4点が一致**することを確認済み。

**(c) Caffe と PyTorch の forward 等価性**（`Convert_VGG_ILSVRC_to_pytorch.ipynb` の cell 26、同一入力テンソル）

層別 MSE は最大 6.3e-10（conv4_2）、fc8 で 6.3e-14。**重み変換とネットワーク構造は差異要因から除外できる。**

**(d) その他**

- torchvision `vgg19` の state_dict は bdpy `VGG19` に `strict=True` でそのまま入る（38 keys、形状一致）
- bdpy `VGG19` の MaxPool は `ceil_mode=False`（Caffe は ceil）だが、**224×224 では全プーリング段の入力辺が偶数なので差は出ない**。`avgpool` も 7×7 入力に対して恒等
- `input_image_shape` は `[H, W, C]`。bdpy の `reconstruct` 内部が `(1, image_size[2], image_size[0], image_size[1])` = (N,C,H,W) と展開することから確定

### 未実施

**配布 `caffe/VGG19` との実データ数値照合。** 刺激画像が非配布のため実施不可能。画像が手に入ったら以下を実行する。

```bash
uv run compare_features.py ./data/features/ImageNetTest/caffe/VGG19 ./data/features/ImageNetTest/pytorch/VGG19
```

残る差異要因は画素値を作る部分のみ。具体的には (1) A の `scipy.misc.imresize` 経路との等価性、(2) Pillow のバージョン差（`resize` のフィルタ実装は歴史的に変更されている）。

---

## 6. 残タスク（優先度順）

1. **README 追記**（最優先）。現 README は「PyTorch 移植版は Caffe 特徴量の完全再現を保証しない」としか書いておらず、5.(b)(c) の実測結果が反映されていない。読み手がリスクの所在を誤解する。あわせて **Caffe prototxt の in-place ReLU 問題**（7章）にも触れておくと、他ネットワークを足す人が同じ罠を踏まない
2. **実データ照合**（刺激画像の入手が前提）。一致すれば `features.name` を `caffe/VGG19` に切り替えてよい
3. **`config/encoder/alexnet.yaml`** の追加（`name: reference_net`、`input_image_shape: [227, 227, 3]`）。構造上は通るが `bvlc_alexnet.pt` が手元になく**未検証**
4. **既知の制限2点を README に明記** — `input_image_shape` の H/W 順序は正方形入力でしか検証されていない／レイヤ名のスラッシュ非対応（7章）
5. **Caffe 参照環境**（2 で有意差が出た場合のみ）。本番抽出器にする必要はなく、50枚程度の参照特徴量を作る**検証器**として置くのが妥当

---

## 7. 落とし穴

### 実装を触るとき

- `ToTensor()` を使うと 1/255 スケールされて配布特徴量と比較不能になる。`ToTensorWithoutScaling` 相当（0–255 保持）が必須
- `layer_mapping` を `FeatureExtractor` に渡し忘れると名前空間が torch モジュール名に変わり、既存 config から読めなくなる
- 保存時に先頭の sample 軸を落とすと `Features.get_features` の `np.concatenate(axis=0)` が壊れる
- `layer_map('vgg19')` は21キー（`relu6`, `relu7` を含む）。config には配布特徴量に合わせて19層だけ列挙してある

### 他ネットワークを足すとき

- 到達可能な範囲は `model_factory` と `layer_map` の積集合、すなわち **`{vgg19, alexnet, reference_net}`** のみ。CORnet / ResNet / Inception は `model_factory` に定義がない
- **Caffe prototxt の in-place ReLU**：標準の prototxt は ReLU を in-place（`top` == `bottom`）で書くため、`net.blobs['conv5_4']` は ReLU 適用**後**の値になる。一方 `layer_map` 経由の hook は適用**前**。同じ層名で全く別物になる。BHscore の `feature_extraction/modify_layer_name.py` はこれを解除するためのツール。今回の VGG19 については配布特徴量が pre-ReLU であることを実測確認済み
- A と C にあった **レイヤ名の `/` から `:` への置換を本実装は持っていない**。bdpy 対応の3ネットワークにスラッシュを含む層名はないので現状は無害だが、GoogLeNet 系（`inception_3a/1x1`）を足すと入れ子ディレクトリになって壊れる

### 環境

- **bdpy は Windows で import できない**。`bdpy/util/info.py` が POSIX 専用の `pwd` を無条件 import している。Windows で検証するなら `pwd.py` の shim が要る（`getpwuid` を返すだけでよい）
- Windows では symlink がプレーンテキストファイルとしてチェックアウトされる（`core.symlinks=false`）。`data` / `models` が機能しないので config のパス調整が必要
- Caffe は現行ディストリでは入手不可。Debian buster に `caffe-cpu` / `python3-caffe-cpu`、bullseye に `caffe` / `python3-caffe`、Debian 12 以降と現行 Ubuntu では削除済み。使うなら `debian:bullseye` 等に固定したコンテナ
- 元の `VGG_ILSVRC_19_layers.caffemodel` は VGG グループの配布 URL で現在も取得可能（HTTP 200 を確認）
- BHscore の Caffe / PyTorch 両スクリプトは `scipy.misc.imresize`（scipy 1.3 以降で削除）と Python 2 的な記述を含み、そのままでは動かない

---

## 8. 検証環境の再現手順

```bash
uv venv --python 3.11 testenv
uv pip install --python testenv/Scripts/python.exe \
    "bdpy==0.26" "hydra-core==1.3.2" "numpy<2" pillow scipy hdf5storage tqdm torch
```

Windows の場合はカレントに `pwd.py` の shim を置く。合成画像と、`bdpy.dl.torch.models.VGG19()` の state_dict をそのまま保存した `.pt` があれば、重み実体なしで全経路を通せる。

---

## 9. 参照

| 対象 | 場所 |
|---|---|
| cookbook 本体 | `KamitaniLab/brain-decoding-cookbook-public` |
| デコーディング（submodule） | `KamitaniLab/feature-decoding` |
| 参照スクリプト A | `KamitaniLab/BHscore` の `feature_extraction/extract_features_caffe.py`（`extract_features_pytorch.py`、`modify_layer_name.py` も同ディレクトリ） |
| 参照スクリプト B | `KamitaniLab/spurious_reconstruction` の `analysis/0_preprocessing/iCNN_image_vgg19_feature_extraction.py` |
| 参照スクリプト C | `KamitaniLab/ImageNetTraining_v3`（private）の `scripts/features/extract_features_images_caffe_vgg19_ImageNet_by_pytorch.py` |
| 重み変換 notebook | `Convert_VGG_ILSVRC_to_pytorch.ipynb`（2020-12、Shirakawa 作成。cookbook とは独立） |
| データ配布 | figshare `brain-decoding-cookbook`（article 21564384）ほか |
| 刺激画像の申請 | `https://forms.gle/ujvA34948Xg49jdn9`（BHscore README 記載） |
