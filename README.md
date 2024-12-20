# InfantCryClassifier

赤ちゃんの泣き声からその原因を予測するCNNモデルです。
[AICのセミナー](https://aic.keio.ac.jp/session_post/acoustic-signal-processing-using-deep-learning/)をきっかけに作成しました。  
主に以下を行っています。
- データ整形
  - 音声の長さ揃え
  - メルスペクトログラム化 (dB to Imageの正規化は、-80〜0の範囲で実施)
- データ拡張
  - 音声データに対してオフライン拡張 (ノイズ付加、ゲイン調整、無音時間の範囲で時間方向に平行移動)
  - 画像データに対してオンライン拡張 (ランダムマスキング)
- CNN訓練
  - ハイパーパラメータチューニング
    - 対象は、CNN設計や学習率、荷重減衰
    - 不均衡データのため、最適化指標にはAUCを用いた
  - 早期終了

## 結果

## データセット

モデルの訓練には、[donateacry-corpus](https://github.com/gveres/donateacry-corpus)のデータを使用しました。
このデータセットには、赤ちゃんの泣き声の音声データと、その原因（空腹、眠気、痛みなど）のラベルが含まれています。

## 主要なファイルの説明

- **visualize.ipynb**:  
  生データ、整形済みデータセット、訓練結果などの可視化を行うためのノートブック

- **build_datasets.py**:  
  以下を行うスクリプト
  - 生データを(train, valid, test) = ()に分割
  - 訓練データについてのみ分類クラス間のデータ数のインバランスを改善しつつデータ拡張
  - その後ImageFolder形式で保存

- **cnn.py**:  
  CNNモデルを定義し、訓練するスクリプト

## 実行方法
```zsh
% git clone https://github.com/kmraven/InfantCryClassifier.git
% cd InfantCryClassifier
% git clone https://github.com/gveres/donateacry-corpus
% python -m venv <YOUR_VENV_NAME>
% pip install -r requirements.txt
% python build_datasets.py
% python cnn.py <GPU_ID_IN_YOUR_ENV>
```
