import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import os
import yaml
import data_processing as dp


def fit_piecewise_log_linear(x, y, threshold):
    # 1. 구간 나누기
    mask_left = x <= threshold
    mask_right = x > threshold
    x1, y1 = x[mask_left], y[mask_left]
    x2, y2 = x[mask_right], y[mask_right]

    # 2. 왼쪽 로그 피팅
    def log_func(x, a, b):
        return a * np.log(x + 1) + b

    popt1, _ = curve_fit(log_func, x1, y1, maxfev=10000)

    # 3. 오른쪽 선형 피팅
    def linear_func(x, c, d):
        return c * x + d

    popt2, _ = curve_fit(linear_func, x2, y2, maxfev=10000)

    # 4. 전체 예측
    y_pred = np.empty_like(x)
    y_pred[mask_left] = log_func(x1, *popt1)
    y_pred[mask_right] = linear_func(x2, *popt2)

    # 5. 성능 평가
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return {
        "threshold": threshold,
        "log_params": popt1,
        "linear_params": popt2,
        "mse": mse,
        "r2": r2,
        "y_pred": y_pred
    }


def search_best_piecewise_log_linear(x, y, thresholds):
    best_result = None
    best_mse = np.inf

    for threshold in thresholds:
        try:
            result = fit_piecewise_log_linear(x, y, threshold)
            if result["mse"] < best_mse:
                best_mse = result["mse"]
                best_result = result
        except Exception as e:
            print(f"❌ threshold {threshold}: {e}")
            continue

    return best_result


def plot_piecewise_result(x, y, result):
    threshold = result["threshold"]
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, alpha=0.3, label="Data")
    plt.plot(x, result["y_pred"], color="red", label="Piecewise Log-Linear Fit")
    plt.axvline(threshold, linestyle="--", color="gray", label=f"Threshold = {threshold}")
    plt.xlabel("Azimuth_Var")
    plt.ylabel("Error_Mean")
    plt.title("Piecewise Log-Linear Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


window_size = 20

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load config
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))
    config['anchors'] = config['anchors']['0409']
    delta = config['delta']
    offset = config['offset']

    anchor_id = 1

    gt_path = os.path.join(base_dir, "../dataset/0409/gt/anchor3.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/0409/beacons/anchor3.csv")
    ms_df = pd.read_csv(ms_path)

    # Preprocess
    position = config['anchors'][anchor_id]['position']   
    orientation = config['anchors'][anchor_id]['orientation']

    ms_gt_df = dp.filter_with_position_ground_truth(gt_df, ms_df, offset)
    ms_gt_df = dp.calculate_aoa_ground_truth(ms_gt_df, position, orientation)

    # Rolling features
    ms_gt_df["Azimuth_Error_Abs"] = (ms_gt_df["Azimuth"] - ms_gt_df["Azimuth_Real"]).abs()
    ms_gt_df["Azimuth_Var"] = ms_gt_df["Azimuth"].rolling(window=window_size).var()
    ms_gt_df["Error_Mean"] = ms_gt_df["Azimuth_Error_Abs"].rolling(window=window_size).mean()

    # Filter 
    ms_gt_df = ms_gt_df[ms_gt_df["Azimuth_Var"] <= 1500].copy()

    x = ms_gt_df["Azimuth_Var"].values
    y = ms_gt_df["Error_Mean"].values

    # 2. threshold 후보 설정
    threshold_candidates = np.arange(0, 500, 5)

    # 3. 최적 피팅 수행
    best_result = search_best_piecewise_log_linear(x, y, threshold_candidates)

    # 4. 결과 출력
    if best_result:
        print(f"\n✅ Best threshold: {best_result['threshold']}")
        print(f"Log params (a, b): {best_result['log_params']}")
        print(f"Linear params (c, d): {best_result['linear_params']}")
        print(f"MSE: {best_result['mse']:.4f}")
        print(f"R²: {best_result['r2']:.4f}\n")

        # 5. 시각화
        plot_piecewise_result(x, y, best_result)
    else:
        print("❌ 모델 피팅에 실패했습니다.")