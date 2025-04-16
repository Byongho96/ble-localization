import os
import yaml
import pandas as pd
import data_processing as dp
import aoa_filter as af
import mobility_localization_filter as mlf
import visualize as vs

def aoa_filter(df):
    # df = af.aoa_moving_average_filter(df)
    # filtered_df = af.aoa_median_filter(df)
    # filtered_df = af.aoa_low_pass_filter(df)
    return df

def aoa_local_calibration_kf(df: pd.DataFrame, anchor_ids: list[int], config: dict, delta: int, threshold: int):
    all_results = {
        'raw': [],
        'wls': [],
        'wls_kf': [],
        '2d_kf': [],
        'ekf': [],
        'imm_ekf': [],
        'ukf': [],
        'pf': [],
    }

    for (x, y), point_df in df.groupby(["X_Real", "Y_Real"]):
        all_results['raw'].append(mlf.least_squares_triangulation(point_df, config, anchor_ids))
        # all_results['2d_kf'].append(mlf.local_2D_kalman_filter(all_results['raw'], delta))
        all_results['ekf'].append(mlf.local_extended_kalman_filter(point_df, config, anchor_ids, delta, threshold))
        # all_results['imm_ekf'].append(mlf.local_imm_extended_kalman_filter(point_df, config, anchor_ids, delta, threshold))
        all_results['ukf'].append(mlf.local_unscented_kalman_filter(point_df, config, anchor_ids, delta, threshold))
        all_results['pf'].append(mlf.local_particle_filter(point_df, config, anchor_ids, delta, threshold))

    # Show the results
    vs.visualize_distance_error_with_heatmap(pd.concat(all_results['raw'], ignore_index=True), 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Raw Local")
    # vs.visualize_distance_error_with_heatmap(all_results['2d_kf'], 'X_Real', 'Y_Real', 'X_2D_KF', 'Y_2D_KF', vmin=0, vmax=200, title="2D Kalman Local")
    vs.visualize_distance_error_with_heatmap(pd.concat(all_results['ekf'], ignore_index=True), 'X_Real', 'Y_Real', 'X_EKF', 'Y_EKF', vmin=0, vmax=200, title="Extended Local")
    # vs.visualize_distance_error_with_heatmap(all_results['imm_ekf'], 'X_Real', 'Y_Real', 'X_IMM', 'Y_IMM', vmin=0, vmax=200, title="IMM Extended Local")
    vs.visualize_distance_error_with_heatmap(pd.concat(all_results['ukf'], ignore_index=True), 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=200, title="Unscented Local")
    vs.visualize_distance_error_with_heatmap(pd.concat(all_results['pf'], ignore_index=True), 'X_Real', 'Y_Real', 'X_PF', 'Y_PF', vmin=0, vmax=200, title="Particle Local")

def aoa_local_mobile_kf(df: pd.DataFrame, anchor_ids: list[int], config: dict, delta: int, threshold: int):
    all_results = {
        'raw': [],
        'wls': [],
        'wls_kf': [],
        '2d_kf': [],
        'ekf': [],
        'imm_ekf': [],
        'ukf': [],
        'pf': [],
    }

    all_results['raw'] = mlf.least_squares_triangulation(df, config, anchor_ids)
    all_results['2d_kf'] = mlf.local_2D_kalman_filter(all_results['raw'], delta)
    all_results['wls'] = mlf.weighted_least_squares_triangulation(df, config, anchor_ids)
    all_results['wls_kf'] = mlf.local_2D_kalman_filter(all_results['wls'], delta)
    # all_results['ekf'] = mlf.local_extended_kalman_filter(df, config, anchor_ids, delta, threshold)
    # all_results['imm_ekf'] = mlf.local_imm_extended_kalman_filter(df, config, anchor_ids, delta, threshold)
    # all_results['ukf'] = mlf.local_unscented_kalman_filter(df, config, anchor_ids, delta, threshold)
    # all_results['pf'] = mlf.local_particle_filter(df, config, anchor_ids, delta, threshold)

    # Show the results
    vs.visualize_distance_error_with_heatmap(all_results['raw'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Raw Local")
    vs.visualize_distance_error_with_heatmap(all_results['2d_kf'], 'X_Real', 'Y_Real', 'X_2D_KF', 'Y_2D_KF', vmin=0, vmax=200, title="Raw Local")
    vs.visualize_distance_error_with_heatmap(all_results['wls'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Raw Local")
    vs.visualize_distance_error_with_heatmap(all_results['wls_kf'], 'X_Real', 'Y_Real', 'X_2D_KF', 'Y_2D_KF', vmin=0, vmax=200, title="Raw Local")
    # vs.visualize_distance_error_with_heatmap(all_results['2d_kf'], 'X_Real', 'Y_Real', 'X_2D_KF', 'Y_2D_KF', vmin=0, vmax=200, title="2D Kalman Local")
    # vs.visualize_distance_error_with_heatmap(all_results['ekf'], 'X_Real', 'Y_Real', 'X_EKF', 'Y_EKF', vmin=0, vmax=200, title="Extended Local")
    # vs.visualize_distance_error_with_heatmap(all_results['imm_ekf'], 'X_Real', 'Y_Real', 'X_IMM', 'Y_IMM', vmin=0, vmax=200, title="IMM Extended Local")
    # vs.visualize_distance_error_with_heatmap(all_results['ukf'], 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=200, title="Unscented Local")
    # vs.visualize_distance_error_with_heatmap(all_results['pf'], 'X_Real', 'Y_Real', 'X_PF', 'Y_PF', vmin=0, vmax=200, title="Particle Local")

def calibration():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']
    offset = config['offset']
    threshold = config['threshold']

    gt_path = os.path.join(base_dir, "../dataset/static/gt/gt_static_east.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/static/beacons/beacons_static_east.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }
    anchor_ids = list(anchors_df_dict.keys())

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        position = config['anchors'][anchor_id]['position']   
        orientation = config['anchors'][anchor_id]['orientation']
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchor_gt_discretized_aoa_df = dp.calculate_aoa_ground_truth(anchor_gt_discretized_df, position, orientation)

        # Filter the data
        anchor_gt_discretized_aoa_df =  anchor_gt_discretized_aoa_df[ anchor_gt_discretized_aoa_df['Azimuth_Real'].abs() <= 45]

        anchors_df_dict[anchor_id] = anchor_gt_discretized_aoa_df

    # Merge the dataframes
    anchors_df = dp.prepare_merged_dataframe(anchors_df_dict)  

    aoa_local_calibration_kf(anchors_df, anchor_ids, config, delta, threshold)
def mobility():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load Config
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']
    threshold = config['threshold']

    # Load files
    gt_path = os.path.join(base_dir, "../dataset/mobility/gt/use-case1/gt_mobility_use-case1_run2.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/mobility/beacons/use-case1/beacons_mobility_use-case1_run2.csv")
    ms_df = pd.read_csv(ms_path)

    # Interpolate the ground truth data
    gt_interpolated_df = dp.interpolate_ground_truth(gt_df, delta)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }
    anchor_ids = list(anchors_df_dict.keys())

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        position = config['anchors'][anchor_id]['position']   
        orientation = config['anchors'][anchor_id]['orientation']
        
        anchor_df = aoa_filter(anchor_df)
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_interpolated_df, anchor_df)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchor_gt_discretized_aoa_df = dp.calculate_aoa_ground_truth(anchor_gt_discretized_df, position, orientation)

        # Filter the data
        # anchor_gt_discretized_aoa_df = anchor_gt_discretized_aoa_df[ anchor_gt_discretized_aoa_df['AnchorID'] != 6503 ]
        # anchor_gt_discretized_aoa_df = anchor_gt_discretized_aoa_df.iloc[len(anchor_gt_discretized_aoa_df) // 4:len(anchor_gt_discretized_aoa_df) // 2]
        # anchor_gt_discretized_aoa_df =  anchor_gt_discretized_aoa_df[ anchor_gt_discretized_aoa_df['Azimuth_Real'].abs() <= 45]
        
        anchors_df_dict[anchor_id] = anchor_gt_discretized_aoa_df

    # Merge the dataframes
    anchors_df = dp.prepare_merged_dataframe(anchors_df_dict)  
    
    aoa_local_mobile_kf(anchors_df, anchor_ids, config, delta, threshold)


if __name__ == '__main__':
    calibration()
    # mobility()