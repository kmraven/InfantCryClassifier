# InfantCryClassifier

赤ちゃんの泣き声からその原因を予測するCNNモデルです。
[AICのセミナー](https://aic.keio.ac.jp/session_post/acoustic-signal-processing-using-deep-learning/)をきっかけに作成しました。  
主に以下を行っています。
- データ整形
- データ拡張
- メルスペクトログラム化
- CNN訓練 (ハイパラチューニング)

## データセット

モデルの訓練には、[donateacry-corpus](https://github.com/gveres/donateacry-corpus)のデータを使用しました。
このデータセットには、赤ちゃんの泣き声の音声データと、その原因（空腹、眠気、痛みなど）のラベルが含まれています。

## 主要なファイルの説明

- **visualize.ipynb**:  
  生データの可視化や、作成したデータセットの内訳確認など、データの理解を深めるための可視化を行うためのノートブック

- **build_datasets.py**:  
  生データを訓練用と検証用に分割し、それをImageFolder形式で保存するスクリプト

- **build_datasets_aug.py**:  
  生データを訓練用と検証用に分割し、訓練データについてのみ分類クラス間のデータ数のインバランスを改善しつつデータ拡張を行い、その後ImageFolder形式で保存するスクリプト

- **cnn.py**:  
  CNNモデルを定義し、訓練するスクリプト

## 実行方法
```zsh
% git clone https://github.com/kmraven/InfantCryClassifier.git
% cd InfantCryClassifier
% git clone https://github.com/gveres/donateacry-corpus
% python -m venv <YOUR_VENV_NAME>
% pip install -r requirements.txt
% python build_datasets_aug.py
% python cnn.py <GPU_ID_IN_YOUR_ENV>
```