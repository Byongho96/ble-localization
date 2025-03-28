import os
import yaml
import pandas as pd
import data_processing as dp
import localization_filter as lf
import visualize as vs

def aoa_local_kf(df: pd.DataFrame, anchor_ids: list[int], config: dict, delta: int):
    all_results = {
        'raw': [],
        '2d_kf': [],
        '3d_kf': [],
        'ekf': [],
        'ukf': [],
        'pf': [],
    }

    # all_results['raw'] = lf.least_squares_triangulation(df, config, anchor_ids)
    # all_results['2d_kf'] = lf.local_2D_kalman_filter(all_results['raw'], delta)
    # all_results['ukf'] = lf.local_unscented_kalman_filter(df, config, anchor_ids, delta)
    all_results['pf'] = lf.local_particle_filter(df, config, anchor_ids, delta)

    # Show the results
    # vs.visualize_distance_error_with_heatmap(all_results['raw'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=200, title="Raw Local")
    # vs.visualize_distance_error_with_heatmap(all_results['2d_kf'], 'X_Real', 'Y_Real', 'X_2D_KF', 'Y_2D_KF', vmin=0, vmax=300, title="2D Kalman Local")
    # vs.visualize_distance_error_with_heatmap(all_results['ukf'], 'X_Real', 'Y_Real', 'X_UKF', 'Y_UKF', vmin=0, vmax=300, title="Unscented Local")
    vs.visualize_distance_error_with_heatmap(all_results['pf'], 'X_Real', 'Y_Real', 'X_PF', 'Y_PF', vmin=0, vmax=300, title="Particle Local")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']

    gt_path = os.path.join(base_dir, "../dataset/calibration/gt/gt_calibration.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/calibration/beacons/beacons_calibration.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df)
        anchor_discretized_df = dp.discretize_grid_points_by_delta(anchor_gt_df, delta)

        anchors_df_dict[anchor_id] = anchor_discretized_df

    anchor_ids = list(config['anchors'].keys())
    anchors_df = dp.prepare_merged_dataframe(anchors_df_dict)  

    aoa_local_kf(anchors_df, anchor_ids, config, delta)

if __name__ == '__main__':
    main()