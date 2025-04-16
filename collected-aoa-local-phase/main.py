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
    pass

def calibration():
    TYPE = 'diagonal'

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))

    # Extract type position and orientation from config
    config['anchors'] = config['anchors'][TYPE]

    delta = config['delta']
    offset = config['offset']
    threshold = config['threshold']

    gt_path = os.path.join(base_dir, f"../dataset/0317/gt/{TYPE}.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path_1 = os.path.join(base_dir, f"../dataset/0317/anchor1/{TYPE}.csv")
    ms_df_1 = pd.read_csv(ms_path_1)
    ms_path_2 = os.path.join(base_dir, f"../dataset/0317/anchor2/{TYPE}.csv")
    ms_df_2 = pd.read_csv(ms_path_2)
    ms_path_3 = os.path.join(base_dir, f"../dataset/0317/anchor3/{TYPE}.csv")
    ms_df_3 = pd.read_csv(ms_path_3)
    ms_path_4 = os.path.join(base_dir, f"../dataset/0317/anchor4/{TYPE}.csv")
    ms_df_4 = pd.read_csv(ms_path_4)

    # Group by anchors
    anchors_df_dict = { 
        1: ms_df_1,
        2: ms_df_2,
        3: ms_df_3,
        4: ms_df_4,
    }
    anchor_ids = list(anchors_df_dict.keys())

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        position = config['anchors'][anchor_id]['position']   
        orientation = config['anchors'][anchor_id]['orientation']
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchor_gt_discretized_aoa_df = dp.calculate_aoa_ground_truth(anchor_gt_discretized_df, position, orientation)

        # Filter the data
        # anchor_gt_discretized_aoa_df =  anchor_gt_discretized_aoa_df[ anchor_gt_discretized_aoa_df['Azimuth_Real'].abs() <= 45]

        anchors_df_dict[anchor_id] = anchor_gt_discretized_aoa_df

    # Merge the dataframes
    anchors_df = dp.prepare_merged_dataframe(anchors_df_dict)  

    aoa_local_calibration_kf(anchors_df, anchor_ids, config, delta, threshold)


def mobility():
    pass


if __name__ == '__main__':
    calibration()
    # mobility()