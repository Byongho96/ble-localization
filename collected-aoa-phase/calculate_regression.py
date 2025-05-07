import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import os
import yaml
import data_processing as dp
from sklearn.metrics import mean_squared_error, r2_score

window_size = 20

def calculate_exponential_fit():
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
    ms_gt_df["Azimuth_Var"] = ms_gt_df["Azimuth"].rolling(window=window_size).var()
    ms_gt_df["Azimuth_Error_Abs"] = (ms_gt_df["Azimuth_Real"] - ms_gt_df["Azimuth"]).abs()
    ms_gt_df["Error_Mean"] = ms_gt_df["Azimuth_Error_Abs"].rolling(window=window_size).mean()

    # Filter 
    ms_gt_df = ms_gt_df[ms_gt_df["Azimuth_Var"] <= 500].copy()

    
    # Drop rows with NaNs from rolling
    ms_gt_df = ms_gt_df.dropna(subset=["Azimuth_Var", "Azimuth_Error_Abs", "Error_Mean"])

    x = ms_gt_df["Azimuth_Var"].values
    y = ms_gt_df["Error_Mean"].values


    # 로그 함수 모델 정의
    def power_func(x, a, b):
        return a * x ** b

    # ✅ 2. Curve fitting
    popt, _ = curve_fit(power_func, x, y, maxfev=10000)
    a, b = popt

    # ✅ 3. 예측 및 평가
    y_pred = power_func(x, *popt)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Power model: y = {a:.4f} * x^{b:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # 예측 및 시각화
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = power_func(x_fit, a, b)

    print(len(ms_gt_df))

    plt.scatter(x, y, label="Data", alpha=0.01)
    plt.plot(x_fit, y_fit, color="red", label=f"Fitted: y = {a:.2f} * x^{b:.2f}")
    plt.xlabel("Azimuth_Var")
    plt.ylabel("Error_Mean")
    plt.legend()
    plt.grid(True)
    plt.title("Exponential Fit")
    plt.show()


"""
Log
"""
# def log_func(x, a, b):
#     return a * np.log(x) + b

# # 데이터 필터링 (Azimuth_Var <= 500)
# filtered_df = ms_gt_df[ms_gt_df["Azimuth_Var"] <= 500].dropna(subset=["Azimuth_Var", "Error_Mean"])
# x = filtered_df["Azimuth_Var"].values
# y = filtered_df["Error_Mean"].values

# # x가 0 이하인 경우 로그 계산이 안 되므로 필터링
# mask = x > 0
# x_log = x[mask]
# y_log = y[mask]

# # curve fitting
# popt, _ = curve_fit(log_func, x_log, y_log)
# a, b = popt

# # 결과 시각화
# x_fit = np.linspace(min(x_log), max(x_log), 100)
# y_fit = log_func(x_fit, a, b)


"""
Root
"""
# # ✅ Root 함수 정의
# def root_func(x, a, b):
#     return a * np.sqrt(x) + b

# # ✅ 피팅
# popt, _ = curve_fit(root_func, x, y, maxfev=10000)
# a, b = popt

# # ✅ 예측 및 평가
# y_pred = root_func(x, *popt)
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)

if __name__ == "__main__":
    calculate_exponential_fit()