import os
import yaml
import pandas as pd
import data_processing as dp
import grid_points_localization_filter as gplf
import mobility_localization_filter as mlf
import visualize as vs

def aoa_local_kf(df: pd.DataFrame, anchor_ids: list[int], config: dict, delta: int, threshold: int):
    all_results = {
        'raw': [],
        '2d_kf': [],
        'ekf': [],
        'ukf': [],
        'pf': [],
    }

    all_results['raw'] = gplf.least_squares_triangulation(df, config, anchor_ids)
    all_results['2d_kf'] = gplf.local_2D_kalman_filter(all_results['raw'], delta)
    all_results['ekf'] = gplf.local_extended_kalman_filter(df, config, anchor_ids, delta, threshold)
    all_results['ukf'] = gplf.local_unscented_kalman_filter(df, config, anchor_ids, delta, threshold)
    all_results['pf'] = gplf.local_particle_filter(df, config, anchor_ids, delta, threshold)

    # Show the results
    vs.visualize_distance_error_with_heatmap(all_results['raw'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Raw Local")
    vs.visualize_distance_error_with_heatmap(all_results['2d_kf'], 'X_Real', 'Y_Real', 'X_2D_KF', 'Y_2D_KF', vmin=0, vmax=200, title="2D Kalman Local")
    vs.visualize_distance_error_with_heatmap(all_results['ekf'], 'X_Real', 'Y_Real', 'X_EKF', 'Y_EKF', vmin=0, vmax=200, title="Extended Local")
    vs.visualize_distance_error_with_heatmap(all_results['ukf'], 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=200, title="Unscented Local")
    vs.visualize_distance_error_with_heatmap(all_results['pf'], 'X_Real', 'Y_Real', 'X_PF', 'Y_PF', vmin=0, vmax=200, title="Particle Local")

def aoa_local_mobile_kf(df: pd.DataFrame, anchor_ids: list[int], config: dict, delta: int, threshold: int):
    all_results = {
        'raw': [],
        '2d_kf': [],
        'ekf': [],
        'ukf': [],
        'pf': [],
    }

    all_results['raw'] = mlf.least_squares_triangulation(df, config, anchor_ids)
    all_results['2d_kf'] = mlf.local_2D_kalman_filter(all_results['raw'], delta)
    all_results['ekf'] = mlf.local_extended_kalman_filter(df, config, anchor_ids, delta, threshold)
    all_results['ukf'] = mlf.local_unscented_kalman_filter(df, config, anchor_ids, delta, threshold)
    all_results['pf'] = mlf.local_particle_filter(df, config, anchor_ids, delta, threshold)

    # Show the results
    vs.visualize_distance_error_with_heatmap(all_results['raw'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Raw Local")
    vs.visualize_distance_error_with_heatmap(all_results['2d_kf'], 'X_Real', 'Y_Real', 'X_2D_KF', 'Y_2D_KF', vmin=0, vmax=200, title="2D Kalman Local")
    vs.visualize_distance_error_with_heatmap(all_results['ekf'], 'X_Real', 'Y_Real', 'X_EKF', 'Y_EKF', vmin=0, vmax=200, title="Extended Local")
    vs.visualize_distance_error_with_heatmap(all_results['ukf'], 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=200, title="Unscented Local")
    vs.visualize_distance_error_with_heatmap(all_results['pf'], 'X_Real', 'Y_Real', 'X_PF', 'Y_PF', vmin=0, vmax=200, title="Particle Local")

def calibration():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']
    threshold = config['threshold']

    gt_path = os.path.join(base_dir, "../dataset/calibration/gt/gt_calibration.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/calibration/beacons/beacons_calibration.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }
    anchor_ids = list(anchors_df_dict.keys())

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df)
        anchor_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchors_df_dict[anchor_id] = anchor_discretized_df

    anchors_df = dp.prepare_merged_dataframe(anchors_df_dict)  

    aoa_local_kf(anchors_df, anchor_ids, config, delta, threshold)

def mobility():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load Config
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']
    threshold = config['threshold']

    # Load files
    gt_path = os.path.join(base_dir, "../dataset/mobility/gt/use-case1/gt_mobility_use-case1_run1.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/mobility/beacons/use-case1/beacons_mobility_use-case1_run1.csv")
    ms_df = pd.read_csv(ms_path)

    # Interpolate the ground truth data
    gt_interpolated_df = dp.interpolate_ground_truth(gt_df, delta)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }
    anchor_ids = list(anchors_df_dict.keys())

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_interpolated_df, anchor_df)
        anchor_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchors_df_dict[anchor_id] = anchor_discretized_df

    # Merge the dataframes
    anchors_df = dp.prepare_merged_dataframe(anchors_df_dict)  
    
    aoa_local_mobile_kf(anchors_df, anchor_ids, config, delta, threshold)


if __name__ == '__main__':
    calibration()
    # mobility()