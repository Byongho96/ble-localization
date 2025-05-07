import os
import yaml
import pandas as pd
import data_processing as dp
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro, probplot


window_size = 20

def calculate_covariance():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load config
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))
    config['anchors'] = config['anchors']['0421']
    delta = config['delta']
    offset = config['offset']

    anchor_id = 1

    gt_path = os.path.join(base_dir, "../dataset/0421/gt/grid.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/0421/beacons/anchor1/diagonal-grid.csv")
    ms_df = pd.read_csv(ms_path)

    # Preprocess
    position = config['anchors'][anchor_id]['position']   
    orientation = config['anchors'][anchor_id]['orientation']
    
    ms_gt_df = dp.filter_with_position_ground_truth(gt_df, ms_df, offset)
    ms_gt_df = dp.calculate_aoa_ground_truth(ms_gt_df, position, orientation)

    # Rolling features
    ms_gt_df["Azimuth_Error"] = (ms_gt_df["Azimuth"] - ms_gt_df["Azimuth_Real"])
    ms_gt_df["Azimuth_Var"] = ms_gt_df["Azimuth"].rolling(window=window_size).var()
    ms_gt_df["Azimuth_Error_Mean"] = ms_gt_df["Azimuth_Error"].rolling(window=window_size).mean()

    # ms_gt_df["Elevation_Var"] = ms_gt_df["Elevation"].rolling(window=window_size).var()
    # ms_gt_df["Angle_Var"] = (ms_gt_df["Azimuth_Var"] ** 2 + ms_gt_df["Elevation_Var"] ** 2) ** 0.5

    # Azimuth_Var 이 400 이상인 행 제거
    ms_gt_df = ms_gt_df[(ms_gt_df["Azimuth_Var"] >= 75) & (ms_gt_df["Azimuth_Var"] <= 100)]

    # --- 회귀 및 그래프 ---
    df_clean = ms_gt_df.dropna(subset=["Azimuth_Var", "Azimuth_Error_Mean"])
    X = df_clean["Azimuth_Var"].values.reshape(-1, 1)
    Y = df_clean["Azimuth_Error_Mean"].values

    model = LinearRegression().fit(X, Y)
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  
    y_line = model.predict(x_line)

    print(f"Intercept: {model.intercept_}, Slope: {model.coef_[0]}")
    
    plt.scatter(X, Y, alpha=0.01)
    plt.plot(x_line, y_line)
    plt.xlabel("Azimuth_Var")
    plt.ylabel("Azimuth_Error_Mean")
    # plt.title("Azimuth_Error_Var vs Azimuth_Var with Regression Line")
    plt.show()


    # [1] 특정 Azimuth_Var 범위 선택
    target_df = df_clean[(df_clean["Azimuth_Var"] >= 75) & (df_clean["Azimuth_Var"] <= 100)]
    errors = target_df["Azimuth_Error_Mean"].dropna()

    # [2] 분포 시각화: 히스토그램 + KDE
    plt.figure(figsize=(10, 4))
    sns.histplot(errors, kde=True, stat='density', bins=30)
    plt.title("Azimuth_Error_Mean Distribution (Azimuth_Var ∈ [50, 55])")
    plt.xlabel("Azimuth_Error_Mean")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

    # [3] Q-Q plot
    plt.figure(figsize=(6, 6))
    probplot(errors, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Azimuth_Error_Mean")
    plt.grid(True)
    plt.show()

    # [4] 정규성 검정 (Shapiro-Wilk test)
    stat, p_value = shapiro(errors)
    print(f"Shapiro-Wilk Test: stat = {stat:.4f}, p-value = {p_value:.4f}")
    if p_value > 0.05:
        print("✅ 정규분포일 가능성이 있음 (귀무가설 기각 못함)")
    else:
        print("❌ 정규분포가 아님 (귀무가설 기각됨)")

if __name__ == "__main__":
    calculate_covariance()
