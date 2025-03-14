import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score, precision_score, recall_score
from scipy.stats import randint, uniform

# 1. データセットの読み込み
df = pd.read_csv('IBM HR Analytics Employee Attrition & Performance.csv')

# 2. ターゲット変数のエンコード
le_target = LabelEncoder()
df['Attrition'] = le_target.fit_transform(df['Attrition'])

# 3. 特徴量とターゲットに分割
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# 4. トレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. カテゴリカル変数の処理
categorical_columns = X_train.select_dtypes(include=['object']).columns
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# 各カテゴリカル変数に対してLabelEncoderを適用
encoders = {}
for col in categorical_columns:
    encoders[col] = LabelEncoder()
    X_train_encoded[col] = encoders[col].fit_transform(X_train[col])
    X_test_encoded[col] = encoders[col].transform(X_test[col])

# 6. SMOTE によるオーバーサンプリング
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

# SMOTE 適用前後のクラス分布を表示
print("【SMOTE適用前】")
print(y_train.value_counts())
print("\n【SMOTE適用後】")
print(pd.Series(y_train_resampled).value_counts())

# 7. ハイパーパラメータの探索空間を定義
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(range(10, 50, 5)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# 8. カスタムスコアリング指標の定義（離職クラスのF1スコアを最適化）
def attrition_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1)

scoring = {
    'f1': make_scorer(attrition_f1),
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1)
}

# 9. RandomizedSearchCVの設定
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,  # 試行回数
    scoring=scoring,
    refit='f1',  # F1スコアで最適なモデルを選択
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

# 10. ハイパーパラメータの最適化を実行
random_search.fit(X_train_resampled, y_train_resampled)

# 11. 最適なパラメータとスコアを表示
print("\n【最適なパラメータ】")
print(random_search.best_params_)
print("\n【最適なスコア（交差検証の平均）】")
print(f"F1スコア: {random_search.best_score_:.3f}")

# 12. テストセットでの評価
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_encoded)
print("\n【最終的な分類レポート】")
print(classification_report(y_test, y_pred))

# 13. 特徴量の重要度を表示
feature_importance = pd.DataFrame({
    'feature': X_train_encoded.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\n【特徴量の重要度（上位10件）】")
print(feature_importance.head(10))
