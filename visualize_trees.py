import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# データの読み込みと前処理
def preprocess_data():
    # データ読み込み
    df = pd.read_csv('IBM HR Analytics Employee Attrition & Performance.csv')
    
    # ターゲット変数のエンコード
    le_target = LabelEncoder()
    df['Attrition'] = le_target.fit_transform(df['Attrition'])
    
    # カテゴリカル変数の処理
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # 特徴量とターゲットに分割
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    # データの標準化
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

# 決定木の可視化
def visualize_decision_tree(X, y, max_depth=3):
    # 決定木モデルの作成と学習
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X, y)
    
    # プロットのサイズ設定
    plt.figure(figsize=(20,10))
    
    # 決定木の可視化
    plot_tree(dt, 
             feature_names=X.columns,
             class_names=['No Attrition', 'Attrition'],
             filled=True,
             rounded=True,
             fontsize=10)
    
    plt.title('Decision Tree Visualization (Max Depth = {})'.format(max_depth))
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()

# ランダムフォレストの一部の木を可視化
def visualize_random_forest_trees(X, y, n_trees=3, max_depth=3):
    # ランダムフォレストモデルの作成と学習
    rf = RandomForestClassifier(n_estimators=n_trees, 
                              max_depth=max_depth,
                              random_state=42)
    rf.fit(X, y)
    
    # 各木を可視化
    for i in range(n_trees):
        plt.figure(figsize=(20,10))
        plot_tree(rf.estimators_[i],
                 feature_names=X.columns,
                 class_names=['No Attrition', 'Attrition'],
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title('Random Forest - Tree {} (Max Depth = {})'.format(i+1, max_depth))
        plt.savefig(f'random_forest_tree_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

# メイン処理
if __name__ == "__main__":
    # データの前処理
    X, y = preprocess_data()
    
    # 決定木の可視化（深さ3まで）
    print("Visualizing Decision Tree...")
    visualize_decision_tree(X, y, max_depth=3)
    
    # ランダムフォレストの3つの木を可視化（深さ3まで）
    print("Visualizing Random Forest Trees...")
    visualize_random_forest_trees(X, y, n_trees=3, max_depth=3)
    
    print("Visualization completed. Check the generated PNG files.")
