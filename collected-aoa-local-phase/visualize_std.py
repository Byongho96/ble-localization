import os
import yaml
import pandas as pd
import data_processing as dp
from scipy.stats import spearmanr, pearsonr
import numpy as np

def plot_variance_heatmaps():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))
    config['anchors'] = config['anchors']['0421']
    delta = config['delta']

    anchor_id = 4

    gt_path = os.path.join(base_dir, "../dataset/0421/gt/diagonal-1.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/0421/beacons/anchor1/diagonal.csv")
    ms_df = pd.read_csv(ms_path)

    # Preprocess the data
    position = config['anchors'][anchor_id]['position']   
    orientation = config['anchors'][anchor_id]['orientation']
    
    gt_df = dp.interpolate_ground_truth(gt_df, delta)
    ms_gt_df = dp.filter_with_position_ground_truth(gt_df, ms_df)

    # drop nan

    ms_gt_df = dp.calculate_aoa_ground_truth(ms_gt_df, position, orientation)
    ms_gt_df = dp.discretize_by_delta(ms_gt_df, delta)
    ms_gt_df['Azimuth_Error'] = np.abs(ms_gt_df['Azimuth'] - ms_gt_df['Azimuth_Real'])

    ms_gt_df = ms_gt_df.dropna(subset=['Azimuth_Error', 'RSSI_Var'])
    # 상관관계 분석
    spearman_corr, _ = spearmanr(ms_gt_df['RSSI_Var'], ms_gt_df['Azimuth_Error'])
    pearson_corr, _ = pearsonr(ms_gt_df['RSSI_Var'], ms_gt_df['Azimuth_Error'])

    print(f"Spearman Correlation: {spearman_corr:.2f}, Pearson Correlation: {pearson_corr:.2f}")
if __name__ == "__main__":
    plot_variance_heatmaps()