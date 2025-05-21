import os
import yaml
import pandas as pd
import data_processing as dp
import aoa_filter as af
import visualize as vs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.mixture import GaussianMixture
from scipy.stats import median_abs_deviation, gaussian_kde

def plot_azimuth_distribution(df: pd.DataFrame, column: str = "Azimuth"):
    data = df[column].dropna()

    print("Azimuth_Real",  df["Azimuth_Real"].mean(), '|', "Azimuth", df[column].mean())

    # ─── Histogram with KDE and normal curve ─────────────────────
    mu, std = np.mean(data), np.std(data)
    xmin, xmax = min(data), max(data)
    x = np.linspace(xmin, xmax, 100)

    # print(df, xmin, xmax)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=30, kde=True, stat="density", color="skyblue", label="Azimuth Histogram")
    plt.plot(x, stats.norm.pdf(x, mu, std), 'r--', label=f"Normal PDF\nμ={mu:.2f}, σ={std:.2f}")
    plt.title("Azimuth Histogram with Normal Distribution")
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Density")
    plt.legend()

    # ─── Q-Q Plot ────────────────────────────────────────────────
    plt.subplot(1, 2, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Azimuth")

    plt.tight_layout()
    plt.show()

    # ─── Optional: Shapiro-Wilk test ─────────────────────────────
    stat, p = stats.shapiro(data)
    print(f"[Shapiro-Wilk Test] W={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print("→ 데이터는 정규분포를 따를 가능성이 높습니다.")
    else:
        print("→ 데이터는 정규분포를 따르지 않을 가능성이 높습니다.")


def aoa_kf(dic: dict, delta: int):
    all_anchors_results = {}

    for anchor_id, anchor_df in dic.items():

        anchor_results = {
            'raw': [],
            'maf': [],
            'median': [],
            'low_pass': [],
            '1d_kf': [],
            '2d_kf': [],
        }

        # Filter the data
        for (x, y), point_df in anchor_df.groupby(["X_Real", "Y_Real"]):
            anchor_results['raw'].append(point_df)
            anchor_results['maf'].append(af.aoa_moving_average_filter(point_df))
            anchor_results['median'].append(af.aoa_median_filter(point_df))
            anchor_results['low_pass'].append(af.aoa_low_pass_filter(point_df))
            anchor_results['1d_kf'].append(af.aoa_1d_kalman_filter(point_df, delta))
            anchor_results['2d_kf'].append(af.aoa_2d_kalman_filter(point_df, delta))

        # Concatenate the results
        for key in anchor_results:
            if not anchor_results[key]:
                continue
            anchor_results[key] = pd.concat(anchor_results[key], ignore_index=True)
        
        all_anchors_results[anchor_id] = anchor_results

    # Show the results
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['raw'] for anchor_id, results in all_anchors_results.items()}, 'Elevation_Real', 'Elevation', vmin=0, vmax=15, title="Raw AoA")   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results['maf'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_MAF', vmin=-15, vmax=15, title="MAF AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['median'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_Median', vmin=0, vmax=15, title="Median AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['low_pass'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_LowPass', vmin=0, vmax=15, title="Low Pass AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['1d_kf'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_1d_KF', vmin=0, vmax=15, title="1D KF AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['2d_kf'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_2d_KF', vmin=0, vmax=15, title="2D KF AoA")

def calibration():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))
    config['anchors'] = config['anchors']['0409']
    delta = 500
    offset = config['offset']

    gt_path = os.path.join(base_dir, "../dataset/0409/gt/anchor1.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path_1 = os.path.join(base_dir, "../dataset/0409/beacons/anchor1.csv")
    ms_df_1 = pd.read_csv(ms_path_1)
    # ms_path_2 = os.path.join(base_dir, "../dataset/0414/beacons/rectangular.csv")
    # ms_df_2 = pd.read_csv(ms_path_2)
    # ms_path_3 = os.path.join(base_dir, "../dataset/0414/beacons/rectangular.csv")
    # ms_df_3 = pd.read_csv(ms_path_3)
    # ms_path_4 = os.path.join(base_dir, "../dataset/0317/anchor4/rectangular.csv")
    # ms_df_4 = pd.read_csv(ms_path_4)

    # Group by anchors
    anchors_df_dict = { 
        1: ms_df_1,
        # 2: ms_df_2,
        # 3: ms_df_3,
        # 4: ms_df_4,
    }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        position = config['anchors'][anchor_id]['position']   
        orientation = config['anchors'][anchor_id]['orientation']
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta, filter=True)
        anchor_gt_discretized_aoa_df = dp.calculate_aoa_ground_truth(anchor_gt_discretized_df, position, orientation)

        target_df = anchor_gt_discretized_aoa_df[(anchor_gt_discretized_aoa_df["X_Real"] == 450) & (anchor_gt_discretized_aoa_df["Y_Real"] == 0 )]
        target_df = dp.calculate_aoa_ground_truth(target_df, position, orientation)
        plot_azimuth_distribution(target_df, "Azimuth")

        anchors_df_dict[anchor_id] = anchor_gt_discretized_aoa_df

    aoa_kf(anchors_df_dict, delta)

def mobility():
    pass

if __name__ == '__main__':
    calibration()
    # mobility()