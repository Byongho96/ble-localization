import os
import yaml
import pandas as pd
import data_processing as dp
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    offset = config['offset']

    gt_path = os.path.join(base_dir, "../dataset/calibration/gt/gt_calibration.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/calibration/beacons/beacons_calibration.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }
    anchors_parameters_dict = { anchor_id: {} for anchor_id in anchors_df_dict.keys() }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        position = config['anchors'][anchor_id]['position']   
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_rssi_df = dp.calculate_rssi_and_distance(anchor_gt_df, position)
        rssi_0, n = dp.calculate_log_distance_parameters(anchor_gt_rssi_df)

        anchors_parameters_dict[anchor_id]['rssi_0'] = rssi_0
        anchors_parameters_dict[anchor_id]['n'] = n

        print(f"Anchor {anchor_id}: rssi_0 = {rssi_0}, n = {n}")
    
    return anchors_parameters_dict


def main(anchors_parameters_dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    offset = config['offset']
    delta = config['delta']

    gt_path = os.path.join(base_dir, "../dataset/static/gt/gt_static_south.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/static/beacons/beacons_static_south.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        print(f"Processing Anchor {anchor_id}...")

        position = config['anchors'][anchor_id]['position']
        orientation = config['anchors'][anchor_id]['orientation']   

        rssi_0, n = anchors_parameters_dict[anchor_id]['rssi_0'], anchors_parameters_dict[anchor_id]['n']
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_aoa_df = dp.calculate_aoa_ground_truth(anchor_gt_df, position, orientation)
        anchor_gt_aoa_rssi_df = dp.calculate_rssi_and_distance(anchor_gt_aoa_df, position)
        
        dp.calculate_rssi_estimated_distance(anchor_gt_aoa_rssi_df, rssi_0, n)

        # Calculate distance error
        anchor_gt_aoa_rssi_df['Distance_Error'] = abs(anchor_gt_aoa_rssi_df['Distance'] - anchor_gt_aoa_rssi_df['RSSI_Distance'])

        pearson_corr, pearson_p = pearsonr(anchor_gt_aoa_rssi_df['RSSI'], anchor_gt_aoa_rssi_df['Distance_Error'])
        print(f"RSSI vs AoA 오차 - Pearson: {pearson_corr:.3f}, p-value: {pearson_p:.3f}")

        spearman_corr, spearman_p = spearmanr(anchor_gt_aoa_rssi_df['RSSI'], anchor_gt_aoa_rssi_df['Distance_Error'])
        print(f"RSSI vs AoA 오차 - Spearman: {spearman_corr:.3f}, p-value: {spearman_p:.3f}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=anchor_gt_aoa_rssi_df,
            x='RSSI',
            y='Distance_Error',
            alpha=0.6
        )
        plt.title(f"Anchor {anchor_id} - RSSI vs Distance Error")
        plt.xlabel("RSSI (dBm)")
        plt.ylabel("Distance Error (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

def main_std(anchors_parameters_dict):
    window_size = 100  # rolling window size 조절 가능
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    offset = config['offset']
    delta = config['delta']

    gt_path = os.path.join(base_dir, "../dataset/static/gt/gt_static_south.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/static/beacons/beacons_static_south.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        print(f"Processing Anchor {anchor_id}...")

        position = config['anchors'][anchor_id]['position']
        orientation = config['anchors'][anchor_id]['orientation']   

        rssi_0, n = anchors_parameters_dict[anchor_id]['rssi_0'], anchors_parameters_dict[anchor_id]['n']
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_aoa_df = dp.calculate_aoa_ground_truth(anchor_gt_df, position, orientation)
        anchor_gt_aoa_rssi_df = dp.calculate_rssi_and_distance(anchor_gt_aoa_df, position)
        
        dp.calculate_rssi_estimated_distance(anchor_gt_aoa_rssi_df, rssi_0, n)

        # Distance Error
        anchor_gt_aoa_rssi_df['Distance_Error'] = abs(anchor_gt_aoa_rssi_df['Distance'] - anchor_gt_aoa_rssi_df['RSSI_Distance'])

        # Rolling RSSI 분산 및 거리 오차 평균 계산
        anchor_gt_aoa_rssi_df['RSSI_Var'] = anchor_gt_aoa_rssi_df['RSSI'].rolling(window_size).var()
        anchor_gt_aoa_rssi_df['Distance_Error_Mean'] = anchor_gt_aoa_rssi_df['Distance_Error'].rolling(window_size).mean()

        # NaN 제거 후 상관분석
        df_valid = anchor_gt_aoa_rssi_df.dropna(subset=['RSSI_Var', 'Distance_Error_Mean'])

        pearson_corr, pearson_p = pearsonr(df_valid['RSSI_Var'], df_valid['Distance_Error_Mean'])
        spearman_corr, spearman_p = spearmanr(df_valid['RSSI_Var'], df_valid['Distance_Error_Mean'])

        print(f"[Anchor {anchor_id}] RSSI 분산 vs Distance Error 평균")
        print(f"  - Pearson:  {pearson_corr:.3f}, p = {pearson_p:.3f}")
        print(f"  - Spearman: {spearman_corr:.3f}, p = {spearman_p:.3f}")

        # 시각화
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_valid,
            x='RSSI_Var',
            y='Distance_Error_Mean',
            alpha=0.6
        )
        plt.title(f"Anchor {anchor_id} - RSSI Variance vs Distance Error (window={window_size})")
        plt.xlabel("RSSI Variance")
        plt.ylabel("Distance Error (mean, m)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    anchors_parameters_dict = preprocessing()

    # main(anchors_parameters_dict)
    main_std(anchors_parameters_dict)