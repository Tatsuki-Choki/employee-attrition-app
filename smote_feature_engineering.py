import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score, precision_score, recall_score
from scipy.stats import randint, uniform

# 1. データセットの読み込み
df = pd.read_csv('IBM HR Analytics Employee Attrition & Performance.csv')

# 2. 特徴量エンジニアリング
def create_features(df):
    df = df.copy()
    
    # 2.1 給与関連の特徴量
    # 給与レベルと職位レベルの比率（期待される給与との差）
    df['SalaryToJobLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 1000)
    
    # 勤続年数あたりの給与上昇率の近似
    df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    
    # 2.2 満足度関連の特徴量
    # 総合満足度スコア
    satisfaction_columns = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                          'RelationshipSatisfaction', 'WorkLifeBalance']
    df['OverallSatisfaction'] = df[satisfaction_columns].mean(axis=1)
    
    # 満足度の標準偏差（満足度のばらつき）
    df['SatisfactionStd'] = df[satisfaction_columns].std(axis=1)
    
    # 2.3 キャリア発展関連の特徴量
    # キャリア成長率（昇進の速さ）
    df['CareerProgressRate'] = df['JobLevel'] / (df['YearsAtCompany'] + 1)
    
    # 現在の上司との関係期間の割合
    df['ManagerTimeRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    
    # 2.4 仕事への関与度合い
    # 残業と仕事関与度の組み合わせ
    df['WorkInvolvement'] = df['OverTime'].map({'Yes': 1, 'No': 0}) * df['JobInvolvement']
    
    # 2.5 経験とスキル関連
    # 総経験年数（会社での経験＋前職での経験）
    df['TotalWorkExperience'] = df['YearsAtCompany'] + df['TotalWorkingYears']
    
    # トレーニング受講回数と勤続年数の比率
    df['TrainingRate'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 1)
    
    return df

# 3. 特徴量エンジニアリングの適用
df_engineered = create_features(df)

# 4. ターゲット変数のエンコード
le_target = LabelEncoder()
df_engineered['Attrition'] = le_target.fit_transform(df_engineered['Attrition'])

# 5. カテゴリカル変数の処理
categorical_columns = df_engineered.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    df_engineered[col] = le.fit_transform(df_engineered[col])

# 6. 特徴量とターゲットに分割
X = df_engineered.drop(columns=['Attrition'])
y = df_engineered['Attrition']

# 7. データの標準化
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 8. トレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 9. SMOTE によるオーバーサンプリング
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# SMOTE 適用前後のクラス分布を表示
print("【SMOTE適用前】")
print(y_train.value_counts())
print("\n【SMOTE適用後】")
print(pd.Series(y_train_resampled).value_counts())

# 10. ハイパーパラメータの探索空間を定義
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(range(10, 50, 5)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# 11. カスタムスコアリング指標の定義（離職クラスのF1スコアを最適化）
def attrition_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1)

scoring = {
    'f1': make_scorer(attrition_f1),
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1)
}

# 12. RandomizedSearchCVの設定
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    scoring=scoring,
    refit='f1',
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

# 13. ハイパーパラメータの最適化を実行
random_search.fit(X_train_resampled, y_train_resampled)

# 14. 最適なパラメータとスコアを表示
print("\n【最適なパラメータ】")
print(random_search.best_params_)
print("\n【最適なスコア（交差検証の平均）】")
print(f"F1スコア: {random_search.best_score_:.3f}")

# 15. テストセットでの評価
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\n【最終的な分類レポート】")
print(classification_report(y_test, y_pred))

# 16. 特徴量の重要度を表示
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\n【特徴量の重要度（上位15件）】")
print(feature_importance.head(15))
