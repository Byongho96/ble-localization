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

from heapq import heappush, heappop

def aoa_local_mobile_kf(df: pd.DataFrame, anchor_ids: list[int], config: dict, delta: int, threshold: int):
    all_results = {}

    # heap = []
    # all_results['ukf'] = mlf.local_unscented_kalman_filter(df, config, anchor_ids, delta, threshold)

    # err, std = vs.get_mean_error(all_results['ukf'], 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=200, title="2D Local")
    # print(f"Initial cost: {err}")

    # for i in range(1, 31):
    #     print(f"i: {i}")
    #     for j in range(1, 11):
    #         # print(f"j: {j}")
    #         for t in range(10, 31):  # 10 ~ 20
    #             k = t / 10
    #             all_results['ukf'] = mlf.local_unscented_kalman_filter(df, config, anchor_ids, delta, threshold, i, j, k)
    #             err, std =vs.get_mean_error(all_results['ukf'], 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=200, title="2D Local")
    #             heappush(heap, (err, i, j, k))

    # for i in range(1, 40):
    #     re = heappop(heap)
    #     print(f"err: {re[0]}, i: {re[1]}, j: {re[2]}, k: {re[3]}")

    # all_results['ls_pre'] = mlf.least_squares_triangulation_prev(df, config, anchor_ids)
    all_results['ls'] = mlf.least_squares_triangulation(df, config, anchor_ids)

    # all_results['wls_pre'] = mlf.weighted_least_squares_triangulation_pre(df, config, anchor_ids)
    all_results['wls'] = mlf.weighted_least_squares_triangulation(df, config, anchor_ids)
    all_results['wls_kf'] = mlf.local_2D_kalman_filter(all_results['wls'], delta)
    all_results['kf-gating'] = mlf.local_2D_kalman_filter_with_gating(all_results['wls'], delta)

    all_results['ekf-pre'] = mlf.local_extended_kalman_filter_pre(df, config, anchor_ids, delta, threshold)
    all_results['ekf'] = mlf.local_extended_kalman_filter(df, config, anchor_ids, delta, threshold)

    # all_results['ukf-pre'] = mlf.local_unscented_kalman_filter_pre(df, config, anchor_ids, delta, threshold)
    # # all_results['ukf'] = mlf.local_unscented_kalman_filter(df, config, anchor_ids, delta, threshold)

    # all_results['pf'] = mlf.local_particle_filter(df, config, anchor_ids, delta, threshold)

    # Show the results
    # vs.visualize_distance_error_with_heatmap(all_results['ls_pre'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Local")
    vs.visualize_distance_error_with_heatmap(all_results['ls'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Local")

    # vs.visualize_distance_error_with_heatmap(all_results['wls_pre'], 'X_Real', 'Y_Real', 'X_WLS', 'Y_WLS', vmin=0, vmax=200, title="Weighted Local")
    vs.visualize_distance_error_with_heatmap(all_results['wls'], 'X_Real', 'Y_Real', 'X_WLS', 'Y_WLS', vmin=0, vmax=200, title="Weighted Local")
    vs.visualize_distance_error_with_heatmap(all_results['wls_kf'], 'X_Real', 'Y_Real', 'X_2D_KF', 'Y_2D_KF', vmin=0, vmax=200, title="2D Local")
    vs.visualize_distance_error_with_heatmap(all_results['kf-gating'], 'X_Real', 'Y_Real', 'X_KF_GA', 'Y_KF_GA', vmin=0, vmax=200, title="2D Local")

    vs.visualize_distance_error_with_heatmap(all_results['ekf-pre'], 'X_Real', 'Y_Real', 'X_EKF', 'Y_EKF', vmin=0, vmax=200, title="Extended Local")
    vs.visualize_distance_error_with_heatmap(all_results['ekf'], 'X_Real', 'Y_Real', 'X_EKF', 'Y_EKF', vmin=0, vmax=200, title="Extended Local")

    # vs.visualize_distance_error_with_heatmap(all_results['ukf-pre'], 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=200, title="Unscented Local")
    # # vs.visualize_distance_error_with_heatmap(all_results['ukf'], 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=200, title="Unscented Local")

    # vs.visualize_distance_error_with_heatmap(all_results['pf'], 'X_Real', 'Y_Real', 'X_PF', 'Y_PF', vmin=0, vmax=200, title="Particle Local")

def calibration():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))

    # Extract type position and orientation from config
    config['anchors'] = config['anchors']['0421']

    delta = config['delta']
    offset = config['offset']
    threshold = config['threshold']

    gt_path = os.path.join(base_dir, f"../dataset/0421/gt/grid.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path_1 = os.path.join(base_dir, f"../dataset/0421/beacons/anchor1/diagonal-grid.csv")
    ms_df_1 = pd.read_csv(ms_path_1)
    ms_path_2 = os.path.join(base_dir, f"../dataset/0421/beacons/anchor2/diagonal-grid.csv")
    ms_df_2 = pd.read_csv(ms_path_2)
    ms_path_3 = os.path.join(base_dir, f"../dataset/0421/beacons/anchor3/diagonal-grid.csv")
    ms_df_3 = pd.read_csv(ms_path_3)
    ms_path_4 = os.path.join(base_dir, f"../dataset/0421/beacons/anchor4/diagonal-grid.csv")
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
        anchor_gt_discretized_aoa_df =  anchor_gt_discretized_aoa_df[ anchor_gt_discretized_aoa_df['Azimuth_Real'].abs() <= 45]

        anchors_df_dict[anchor_id] = anchor_gt_discretized_aoa_df

    # Merge the dataframes
    anchors_df = dp.prepare_merged_dataframe(anchors_df_dict)  

    aoa_local_calibration_kf(anchors_df, anchor_ids, config, delta, threshold)


def mobility():
    TYPE = '0421'

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))

    # Extract type position and orientation from config
    config['anchors'] = config['anchors'][TYPE]

    delta = config['delta']
    offset = config['offset']
    threshold = config['threshold']

    gt_path = os.path.join(base_dir, f"../dataset/{TYPE}/gt/set-3.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path_1 = os.path.join(base_dir, f"../dataset/{TYPE}/beacons/anchor1/diagonal-set.csv")
    ms_df_1 = pd.read_csv(ms_path_1)
    ms_path_2 = os.path.join(base_dir, f"../dataset/{TYPE}/beacons/anchor2/diagonal-set.csv")
    ms_df_2 = pd.read_csv(ms_path_2)
    ms_path_3 = os.path.join(base_dir, f"../dataset/{TYPE}/beacons/anchor3/diagonal-set.csv")
    ms_df_3 = pd.read_csv(ms_path_3)
    ms_path_4 = os.path.join(base_dir, f"../dataset/{TYPE}/beacons/anchor4/diagonal-set.csv")
    ms_df_4 = pd.read_csv(ms_path_4)

    # gt_path = os.path.join(base_dir, f"../dataset/0416/gt/rectangular-mid-reverse.csv")
    # gt_df = pd.read_csv(gt_path)

    # ms_path_1 = os.path.join(base_dir, f"../dataset/0416/beacons/rectangular-mid/anchor1.csv")
    # ms_df_1 = pd.read_csv(ms_path_1)
    # ms_path_2 = os.path.join(base_dir, f"../dataset/0416/beacons/rectangular-mid/anchor2.csv")
    # ms_df_2 = pd.read_csv(ms_path_2)
    # ms_path_3 = os.path.join(base_dir, f"../dataset/0416/beacons/rectangular-mid/anchor3.csv")
    # ms_df_3 = pd.read_csv(ms_path_3)
    # ms_path_4 = os.path.join(base_dir, f"../dataset/0416/beacons/rectangular-mid/anchor4.csv")
    # ms_df_4 = pd.read_csv(ms_path_4)

    # Interpolate the ground truth data
    gt_interpolated_df = dp.interpolate_ground_truth(gt_df, delta)

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
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_interpolated_df, anchor_df, 0)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, gt_df.iloc[0]['StartTimestamp'], delta)
        anchor_gt_discretized_aoa_df = dp.calculate_aoa_ground_truth(anchor_gt_discretized_df, position, orientation)

        # Filter the data
        # anchor_gt_discretized_aoa_df =  anchor_gt_discretized_aoa_df[(anchor_gt_discretized_aoa_df["Time_Bucket"] > 61) | (anchor_gt_discretized_aoa_df["Time_Bucket"] < 60)]

        anchors_df_dict[anchor_id] = anchor_gt_discretized_aoa_df

    # Merge the dataframes
    anchors_df = dp.prepare_merged_dataframe(anchors_df_dict)  

    # save data
    anchors_df.to_csv(os.path.join(base_dir, f"./diagonal-set-aoa.csv"), index=False)

    aoa_local_mobile_kf(anchors_df, anchor_ids, config, delta, threshold)


if __name__ == '__main__':
    # calibration()
    mobility()