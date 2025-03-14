import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# データの読み込みと前処理
def create_features(df):
    df = df.copy()
    
    # 給与関連の特徴量
    df['SalaryToJobLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 1000)
    df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    
    # 満足度関連の特徴量
    satisfaction_columns = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                          'RelationshipSatisfaction', 'WorkLifeBalance']
    df['OverallSatisfaction'] = df[satisfaction_columns].mean(axis=1)
    df['SatisfactionStd'] = df[satisfaction_columns].std(axis=1)
    
    # キャリア発展関連の特徴量
    df['CareerProgressRate'] = df['JobLevel'] / (df['YearsAtCompany'] + 1)
    df['ManagerTimeRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    
    # 仕事への関与度合い
    df['WorkInvolvement'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    
    # 経験とスキル関連
    df['TotalWorkExperience'] = df['YearsAtCompany'] + df['TotalWorkingYears']
    df['TrainingRate'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 1)

    # 特徴量の組み合わせ
    df['StockIncomeInteraction'] = df['StockOptionLevel'] * df['SalaryToJobLevelRatio']
    df['SatisfactionBalanceProduct'] = df['OverallSatisfaction'] * df['WorkLifeBalance']
    df['OvertimeIncomeRatio'] = df['MonthlyIncome'] * df['OverTime'].map({'Yes': 1.5, 'No': 1.0})
    
    expected_job_level = df['TotalWorkingYears'] / 5 + 1
    df['CareerAdvancement'] = df['JobLevel'] - expected_job_level
    
    expected_income = df['JobLevel'] * df['TotalWorkingYears'] * 1000
    df['SalaryContentment'] = df['MonthlyIncome'] / (expected_income + 1)
    
    df['WorkplaceScore'] = (
        df['JobSatisfaction'] * 0.3 +
        df['EnvironmentSatisfaction'] * 0.3 +
        df['RelationshipSatisfaction'] * 0.2 +
        df['WorkLifeBalance'] * 0.2
    )
    
    df['PromotionPotential'] = (
        df['PerformanceRating'] * 0.4 +
        df['JobInvolvement'] * 0.3 +
        (df['TrainingTimesLastYear'] / 3) * 0.3
    )
    
    df['RetentionRiskScore'] = (
        (3 - df['JobSatisfaction']) * 0.2 +
        (df['OverTime'].map({'Yes': 1, 'No': 0})) * 0.2 +
        (1 - df['SalaryContentment']) * 0.2 +
        (3 - df['WorkLifeBalance']) * 0.2 +
        (df['NumCompaniesWorked'] / 10) * 0.2
    )
    
    return df

# メイン処理
if __name__ == "__main__":
    # データの読み込み
    df = pd.read_csv('IBM HR Analytics Employee Attrition & Performance.csv')
    
    # 特徴量エンジニアリングの適用
    df_engineered = create_features(df)
    
    # ターゲット変数のエンコード
    le_target = LabelEncoder()
    df_engineered['Attrition'] = le_target.fit_transform(df_engineered['Attrition'])
    
    # カテゴリカル変数の処理
    categorical_columns = df_engineered.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df_engineered[col] = le.fit_transform(df_engineered[col])
    
    # 特徴量とターゲットに分割
    X = df_engineered.drop(columns=['Attrition'])
    y = df_engineered['Attrition']
    
    # データの標準化
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # トレーニングセットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE によるオーバーサンプリング
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # 最適なパラメータでXGBoostモデルを作成
    model = xgb.XGBClassifier(
        colsample_bytree=0.9921326334864182,
        gamma=0.037673128003064105,
        learning_rate=0.10170910578615455,
        max_depth=9,
        min_child_weight=1,
        n_estimators=355,
        scale_pos_weight=10,
        subsample=0.6677970986744369,
        random_state=42
    )
    
    # モデルの学習
    model.fit(X_train_resampled, y_train_resampled)
    
    # モデルと標準化スケーラーの保存
    joblib.dump(model, 'xgb_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    print("モデルの学習と保存が完了しました。")
