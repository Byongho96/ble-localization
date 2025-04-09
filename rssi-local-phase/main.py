import os
import yaml
import pandas as pd
import data_processing as dp
import rssi_filter as rf
import mobility_localization_filter as mlf
import visualize as vs

def rssi_filter(df):
    # df = rf.rssi_moving_average_filter(df)
    # filtered_df = rf.rssi_median_filter(df)
    # filtered_df = rf.rssi_low_pass_filter(df)
    return df

def rssi_local_mobile_kf(df: pd.DataFrame, anchor_ids: list[int], config: dict, delta: int, threshold: int):
    all_results = {
        'ml': [],       
        'raw': [],
        '2d_kf': [],
        'ekf': [],
        'imm_ekf': [],
        'ukf': [],
        'pf': [],
    }

    # all_results['ml'] = mlf.local_multilateration(df, config, parameters, anchor_ids)
    all_results['raw'] = mlf.local_adaptive_multilateration(df, config, anchor_ids)
    all_results['2d_kf'] = mlf.local_2D_kalman_filter(all_results['raw'], delta)
    # all_results['ekf'] = mlf.local_extended_kalman_filter(df, config, anchor_ids, delta, threshold)
    # all_results['imm_ekf'] = mlf.local_imm_extended_kalman_filter(df, config, anchor_ids, delta, threshold)
    # all_results['ukf'] = mlf.local_unscented_kalman_filter(df, config, anchor_ids, delta, threshold)
    # all_results['pf'] = mlf.local_particle_filter(df, config, anchor_ids, delta, threshold)

    # Show the results
    # print(all_results['ml'].head())
    # vs.visualize_distance_error_with_heatmap(all_results['ml'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Raw Local")
    vs.visualize_distance_error_with_heatmap(all_results['raw'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Raw Local")
    vs.visualize_distance_error_with_heatmap(all_results['2d_kf'], 'X_Real', 'Y_Real', 'X_2D_KF', 'Y_2D_KF', vmin=0, vmax=200, title="2D Kalman Local")
    # vs.visualize_distance_error_with_heatmap(all_results['ekf'], 'X_Real', 'Y_Real', 'X_EKF', 'Y_EKF', vmin=0, vmax=200, title="Extended Local")
    # vs.visualize_distance_error_with_heatmap(all_results['imm_ekf'], 'X_Real', 'Y_Real', 'X_IMM', 'Y_IMM', vmin=0, vmax=200, title="IMM Extended Local")
    # vs.visualize_distance_error_with_heatmap(all_results['ukf'], 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=200, title="Unscented Local")
    # vs.visualize_distance_error_with_heatmap(all_results['pf'], 'X_Real', 'Y_Real', 'X_PF', 'Y_PF', vmin=0, vmax=200, title="Particle Local")

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
        
        anchor_df = rssi_filter(anchor_df)
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_interpolated_df, anchor_df)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchor_gt_discretized_rssi_df = dp.calculate_rssi_and_distance(anchor_gt_discretized_df, position)

        anchors_df_dict[anchor_id] = anchor_gt_discretized_rssi_df

        config['anchors'][anchor_id]['rssi_0'] = anchors_parameters_dict[anchor_id]['rssi_0']  
        config['anchors'][anchor_id]['n'] = anchors_parameters_dict[anchor_id]['n']

    # Merge the dataframes
    anchors_df = dp.prepare_merged_dataframe(anchors_df_dict)  
    
    rssi_local_mobile_kf(anchors_df, anchor_ids, config['anchors'], delta, threshold)


if __name__ == '__main__':
    anchors_parameters_dict = preprocessing()
    main(anchors_parameters_dict)