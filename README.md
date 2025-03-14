# 従業員離職予測アプリケーション

## 概要
このアプリケーションは、機械学習を用いて従業員の離職リスクを予測するStreamlitベースのWebアプリケーションです。XGBoostモデルを使用して、従業員の様々な特性から離職の可能性を予測し、リスク評価を提供します。

## 特徴
- 従業員の基本情報入力（年齢、性別、婚姻状況など）
- 職務関連情報の設定（部署、役職、給与など）
- 満足度評価（職務満足度、環境満足度など）
- 詳細なリスク分析と視覚化
- リアルタイムな予測結果の表示

## 必要条件
- Python 3.8以上
- 必要なパッケージ:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - joblib

## インストール方法
```bash
# リポジトリのクローン
git clone [リポジトリURL]

# 必要なパッケージのインストール
pip install -r requirements.txt
```

## 使用方法
```bash
# アプリケーションの起動
streamlit run attrition_predictor_app.py
```

## ファイル構成
- `attrition_predictor_app.py`: メインのStreamlitアプリケーション
- `train_and_save_model.py`: モデルの学習と保存を行うスクリプト
- `xgb_model.joblib`: 学習済みのXGBoostモデル
- `scaler.joblib`: 特徴量のスケーリングに使用する標準化スケーラー

## モデルについて
- XGBoostアルゴリズムを使用
- SMOTEによるクラス不均衡の処理
- ハイパーパラメータの最適化済み
- 特徴量エンジニアリングによる予測精度の向上

## ライセンス
MIT License

## 作者
[あなたの名前]
