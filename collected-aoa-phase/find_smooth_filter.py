import pandas as pd
import numpy as np
import os
import yaml
import data_processing as dp
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
# 필터 함수 정의
def apply_lowpass_filter(data, cutoff=0.1, order=2):
    if len(data) <= order:
        return data  # 데이터가 너무 짧으면 필터 생략
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, method="gust")

def apply_moving_average(data, window=3):
    return data.rolling(window=window, center=True, min_periods=1).mean()

def apply_median_filter(data, window=3):
    return data.rolling(window=window, center=True, min_periods=1).median()

# MAE 계산 함수
def compute_mae(a, b):
    return np.mean(np.abs(a - b))

# 필터별 MAE 맵 초기화
def create_mae_maps(df):
    xs, ys = df['X_Real'].unique(), df['Y_Real'].unique()
    width, height = xs.max()+1, ys.max()+1
    return {
        'lowpass': np.full((height, width), np.nan),
        'moving_avg': np.full((height, width), np.nan),
        'median': np.full((height, width), np.nan)
    }

# 메인 처리 함수
def process_and_plot(df, window=20, cutoff=0.1, order=2):
    mae_maps = create_mae_maps(df)

    for (x, y), group in df.groupby(['X_Real', 'Y_Real']):
        azimuth = group['Azimuth'].reset_index(drop=True)
        real = group['Azimuth_Real'].reset_index(drop=True)

        if len(azimuth) <= 1:
            continue

        try:
            filtered = {
                'lowpass': apply_lowpass_filter(azimuth, cutoff, order),
                'moving_avg': apply_moving_average(azimuth, window),
                'median': apply_median_filter(azimuth, window),
            }

            for name, filtered_signal in filtered.items():
                mae = compute_mae(filtered_signal, real)
                mae_maps[name][y, x] = mae

        except Exception as e:
            print(f'Error at ({x},{y}): {e}')
            continue

    # 히트맵 시각화
    for name, heatmap in mae_maps.items():
        plt.figure(figsize=(8, 6))
        plt.title(f'MAE Heatmap - {name}', fontsize=14)
        im = plt.imshow(heatmap, origin='lower', cmap='viridis')

        # 컬러바 추가
        plt.colorbar(im, label='MAE')

        # X, Y 축 라벨
        plt.xlabel('X')
        plt.ylabel('Y')

        # 각 셀에 텍스트 표시
        for y in range(heatmap.shape[0]):
            for x in range(heatmap.shape[1]):
                value = heatmap[y, x]
                if not np.isnan(value):
                    plt.text(x, y, f"{value:.1f}", ha='center', va='center', color='blue', fontsize=10, fontweight='bold')

        plt.tight_layout()
    plt.show()

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

    best_filters = process_and_plot(ms_gt_df)

    print(best_filters)  # Display the best filters for each point
    