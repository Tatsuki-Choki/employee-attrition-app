import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import joblib

# ページの設定
st.set_page_config(
    page_title="従業員離職予測アプリ",
    page_icon="👥",
    layout="wide"
)

# タイトルとイントロダクション
st.title("従業員離職予測アプリ 🎯")
st.markdown("""
このアプリケーションは、従業員の特性に基づいて離職リスクを予測します。
各項目を入力することで、機械学習モデルが離職の可能性を分析します。
""")

# サイドバーの作成
st.sidebar.header("従業員情報の入力 📝")

# 入力フォームの作成
def get_user_input():
    # 基本情報
    with st.sidebar.expander("基本情報", expanded=True):
        age = st.number_input("年齢", min_value=18, max_value=100, value=30)
        gender = st.selectbox("性別", ["Male", "Female"])
        marital_status = st.selectbox("婚姻状況", ["Single", "Married", "Divorced"])
        education = st.selectbox("教育レベル", [1, 2, 3, 4, 5])
        education_field = st.selectbox("専攻分野", 
                                     ["Life Sciences", "Medical", "Marketing", 
                                      "Technical Degree", "Other", "Human Resources"])
        
    # 職務情報
    with st.sidebar.expander("職務情報", expanded=True):
        department = st.selectbox("部署", 
                                ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("役職", 
                               ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                "Manufacturing Director", "Healthcare Representative", "Manager",
                                "Sales Representative", "Research Director", "Human Resources"])
        job_level = st.slider("職位レベル", 1, 5, 2)
        years_at_company = st.number_input("勤続年数", min_value=0, max_value=40, value=5)
        
    # 給与・評価情報
    with st.sidebar.expander("給与・評価情報", expanded=True):
        daily_rate = st.number_input("日給", min_value=100, max_value=1500, value=800)
        hourly_rate = st.number_input("時給", min_value=30, max_value=100, value=65)
        monthly_income = st.number_input("月収", min_value=2000, max_value=20000, value=5000)
        monthly_rate = monthly_income
        percent_salary_hike = st.slider("昨年の昇給率（%）", 0, 25, 15)
        stock_option_level = st.slider("ストックオプションレベル", 0, 3, 1)
        
    # 満足度・環境
    with st.sidebar.expander("満足度・環境", expanded=True):
        environment_satisfaction = st.slider("環境満足度", 1, 4, 3)
        job_satisfaction = st.slider("職務満足度", 1, 4, 3)
        relationship_satisfaction = st.slider("人間関係満足度", 1, 4, 3)
        work_life_balance = st.slider("ワークライフバランス", 1, 4, 3)
        
    # 業務状況
    with st.sidebar.expander("業務状況", expanded=True):
        business_travel = st.selectbox("出張頻度", 
                                     ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        distance_from_home = st.number_input("通勤距離", min_value=0, max_value=100, value=10)
        overtime = st.selectbox("残業", ["Yes", "No"])
        performance_rating = st.slider("業績評価", 1, 4, 3)
        standard_hours = 80
        
    # その他
    with st.sidebar.expander("その他", expanded=True):
        num_companies_worked = st.number_input("過去の勤務企業数", min_value=0, max_value=10, value=1)
        total_working_years = st.number_input("総勤務年数", min_value=0, max_value=40, value=8)
        training_times_last_year = st.number_input("昨年の研修回数", min_value=0, max_value=10, value=2)
        years_in_current_role = st.number_input("現在の役職での年数", min_value=0, max_value=15, value=2)
        years_since_last_promotion = st.number_input("前回の昇進からの年数", min_value=0, max_value=15, value=1)
        years_with_curr_manager = st.number_input("現在の上司との年数", min_value=0, max_value=15, value=3)
        job_involvement = st.slider("仕事への関与度", 1, 4, 3)
        
    # 固定値の設定
    employee_count = 1
    employee_number = 1
    over18 = 1  # Y を 1 に変換
    standard_hours = 80

    # データの辞書形式での返却
    data = {
        'Age': age,
        'BusinessTravel': business_travel,
        'DailyRate': daily_rate,
        'Department': department,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EducationField': education_field,
        'EmployeeCount': employee_count,
        'EmployeeNumber': employee_number,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'Over18': over18,
        'OverTime': overtime,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StandardHours': standard_hours,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }
    
    return data

    # データの辞書形式での返却
    data = {
        'Age': age,
        'Gender': gender,
        'MaritalStatus': marital_status,
        'Education': education,
        'Department': department,
        'JobRole': job_role,
        'JobLevel': job_level,
        'YearsAtCompany': years_at_company,
        'MonthlyIncome': monthly_income * 10000,  # 万円から円に変換
        'PercentSalaryHike': percent_salary_hike,
        'StockOptionLevel': stock_option_level,
        'JobSatisfaction': job_satisfaction,
        'RelationshipSatisfaction': relationship_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'OverTime': overtime,
        'BusinessTravel': business_travel,
        'DistanceFromHome': distance_from_home,
        'NumCompaniesWorked': num_companies_worked,
        'TrainingTimesLastYear': training_times_last_year,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }
    
    return data

# 特徴量エンジニアリング関数
def create_features(df):
    df = df.copy()
    
    # 給与関連の特徴量
    df['SalaryToJobLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 1000)
    df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    
    # 満足度関連の特徴量
    satisfaction_columns = ['JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
    df['OverallSatisfaction'] = df[satisfaction_columns].mean(axis=1)
    df['SatisfactionStd'] = df[satisfaction_columns].std(axis=1)
    
    # キャリア発展関連の特徴量
    df['CareerProgressRate'] = df['JobLevel'] / (df['YearsAtCompany'] + 1)
    df['ManagerTimeRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    
    # 仕事への関与度合い
    df['WorkInvolvement'] = df['OverTime'].map({'はい': 1, 'いいえ': 0})
    
    # 経験とスキル関連
    df['TotalWorkExperience'] = df['YearsAtCompany']
    df['TrainingRate'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 1)

    # 特徴量の組み合わせ
    df['StockIncomeInteraction'] = df['StockOptionLevel'] * df['SalaryToJobLevelRatio']
    df['SatisfactionBalanceProduct'] = df['OverallSatisfaction'] * df['WorkLifeBalance']
    df['OvertimeIncomeRatio'] = df['MonthlyIncome'] * df['OverTime'].map({'はい': 1.5, 'いいえ': 1.0})
    
    expected_job_level = df['TotalWorkExperience'] / 5 + 1
    df['CareerAdvancement'] = df['JobLevel'] - expected_job_level
    
    expected_income = df['JobLevel'] * df['TotalWorkExperience'] * 1000
    df['SalaryContentment'] = df['MonthlyIncome'] / (expected_income + 1)
    
    df['WorkplaceScore'] = (
        df['JobSatisfaction'] * 0.3 +
        df['RelationshipSatisfaction'] * 0.3 +
        df['WorkLifeBalance'] * 0.4
    )
    
    df['PromotionPotential'] = (
        (df['TrainingTimesLastYear'] / 3) * 0.5 +
        (4 - df['YearsSinceLastPromotion']) * 0.5
    )
    
    df['RetentionRiskScore'] = (
        (5 - df['JobSatisfaction']) * 0.2 +
        (df['OverTime'].map({'はい': 1, 'いいえ': 0})) * 0.2 +
        (1 - df['SalaryContentment']) * 0.2 +
        (5 - df['WorkLifeBalance']) * 0.2 +
        (df['NumCompaniesWorked'] / 10) * 0.2
    )
    
    return df

# メインの処理
def main():
    # ユーザー入力の取得
    user_data = get_user_input()
    
    # データフレームに変換
    input_df = pd.DataFrame([user_data])
    
    # 特徴量エンジニアリングの適用
    input_df = create_features(input_df)
    
    # カテゴリカル変数のエンコーディング
    categorical_mappings = {
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
        'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
        'EducationField': {
            'Life Sciences': 0, 'Medical': 1, 'Marketing': 2,
            'Technical Degree': 3, 'Other': 4, 'Human Resources': 5
        },
        'Gender': {'Male': 0, 'Female': 1},
        'JobRole': {
            'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2,
            'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5,
            'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8
        },
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
        'OverTime': {'Yes': 1, 'No': 0}
    }
    
    # カテゴリカル変数のエンコーディング
    for col, mapping in categorical_mappings.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(mapping)
    
    # 数値型への変換を確認
    numeric_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
                      'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                      'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                      'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                      'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
                      'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                      'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                      'YearsWithCurrManager']
    
    for col in numeric_columns:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col])
    
    # モデルの読み込みと予測
    try:
        model = joblib.load('xgb_model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # データの標準化
        input_scaled = scaler.transform(input_df)
        
        # 予測
        prediction = model.predict_proba(input_scaled)[0]
        
        # 結果の表示
        st.header("予測結果 📊")
        
        # 離職リスクのゲージ表示
        risk_percentage = prediction[1] * 100
        st.subheader("離職リスク")
        st.progress(risk_percentage / 100)
        st.write(f"離職の可能性: {risk_percentage:.1f}%")
        
        # リスクレベルの判定
        if risk_percentage < 20:
            risk_level = "低"
            color = "green"
        elif risk_percentage < 40:
            risk_level = "やや低"
            color = "lightgreen"
        elif risk_percentage < 60:
            risk_level = "中"
            color = "yellow"
        elif risk_percentage < 80:
            risk_level = "やや高"
            color = "orange"
        else:
            risk_level = "高"
            color = "red"
            
        st.markdown(f"リスクレベル: :<span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
        
        # 主要な要因の分析
        st.subheader("主要な要因")
        
        # リスク要因の分析
        risk_factors = []
        if input_df['OverTime'].iloc[0] == 1:
            risk_factors.append("残業が多い")
        if input_df['JobSatisfaction'].iloc[0] <= 2:
            risk_factors.append("職務満足度が低い")
        if input_df['WorkLifeBalance'].iloc[0] <= 2:
            risk_factors.append("ワークライフバランスが悪い")
        if input_df['YearsSinceLastPromotion'].iloc[0] >= 5:
            risk_factors.append("長期間昇進がない")
        if input_df['SalaryToJobLevelRatio'].iloc[0] < 0.8:
            risk_factors.append("給与水準が相対的に低い")
            
        if risk_factors:
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("特に大きなリスク要因は見られません")
        
        # 改善提案
        if risk_percentage >= 40:
            st.subheader("改善提案 💡")
            recommendations = []
            if "残業が多い" in risk_factors:
                recommendations.append("労働時間の適正化と効率的な業務配分の検討")
            if "職務満足度が低い" in risk_factors:
                recommendations.append("キャリア開発機会の提供とジョブローテーションの検討")
            if "ワークライフバランスが悪い" in risk_factors:
                recommendations.append("柔軟な勤務体制の導入と休暇取得の促進")
            if "長期間昇進がない" in risk_factors:
                recommendations.append("キャリアパスの明確化と評価制度の見直し")
            if "給与水準が相対的に低い" in risk_factors:
                recommendations.append("給与体系の見直しと市場水準との比較検討")
                
            for rec in recommendations:
                st.write(f"• {rec}")
                
    except FileNotFoundError:
        st.error("モデルファイルが見つかりません。先にモデルのトレーニングを実行してください。")

if __name__ == "__main__":
    main()
