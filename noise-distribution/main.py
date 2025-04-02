import os
import yaml
import pandas as pd
import data_processing as dp
import visualize as vs
from LOSClassifier import LOSClassifier 

def aoa():
    TARGET_ANCHOR_ID = 6504
    TARGET_POINTS_1 = [(120, 480), (1080, 480), (540, 420), (660, 420)]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    gt_path = os.path.join(base_dir, "../dataset/calibration/gt/gt_calibration.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/calibration/beacons/beacons_calibration.csv")
    ms_df = pd.read_csv(ms_path)

    # Filter the target anchor and ground truth data
    anchor_df = ms_df[ms_df["AnchorID"] == TARGET_ANCHOR_ID]
    anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df)

    vs.plot_aoa_distribution(anchor_gt_df, TARGET_POINTS_1)
    vs.plot_rssi_distribution(anchor_gt_df, TARGET_POINTS_1, 37)

def los():
    TARGET_ANCHOR_ID = 6501
    TARGET_POINTS_1 = [(720, 240), (840, 240), (960, 240)]


    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    gt_path_1 = os.path.join(base_dir, "../dataset/static/gt/gt_static_east.csv")
    gt_df_1 = pd.read_csv(gt_path_1)

    gt_path_2 = os.path.join(base_dir, "../dataset/static/gt/gt_static_west.csv")
    gt_df_2 = pd.read_csv(gt_path_2)

    ms_path_1 = os.path.join(base_dir, "../dataset/static/beacons/beacons_static_east.csv")
    ms_df_1 = pd.read_csv(ms_path_1)

    ms_path_2 = os.path.join(base_dir, "../dataset/static/beacons/beacons_static_west.csv")
    ms_df_2 = pd.read_csv(ms_path_2)

    # Filter the target anchor and ground truth data
    anchor_df_1 = ms_df_1[ms_df_1["AnchorID"] == TARGET_ANCHOR_ID]
    anchor_gt_df_1 = dp.filter_with_position_ground_truth(gt_df_1, anchor_df_1)

    anchor_df_2 = ms_df_2[ms_df_2["AnchorID"] == TARGET_ANCHOR_ID]
    anchor_gt_df_2 = dp.filter_with_position_ground_truth(gt_df_2, anchor_df_2)

    vs.plot_aoa_distribution(anchor_gt_df_1, TARGET_POINTS_1)
    vs.plot_rssi_distribution(anchor_gt_df_1, TARGET_POINTS_1, 37)

    vs.plot_aoa_distribution(anchor_gt_df_2, TARGET_POINTS_1)
    vs.plot_rssi_distribution(anchor_gt_df_2, TARGET_POINTS_1, 37)

if __name__ == "__main__":
    aoa()
    # los()
    