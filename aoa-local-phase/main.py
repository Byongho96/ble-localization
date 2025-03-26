import os
import yaml
import pandas as pd
import data_processing as dp
import localization_filter as lf
import visualize as vs

def aoa_local_kf(dic: dict, config: dict, delta: int):
    all_results = {
        'raw': [],
        'kf': [],
        # 'ekf': [],
        # 'ukf': [],
        # 'pf': [],
    }

    all_results['raw'] = lf.least_squares_triangulation(dic, config)
    # all_results['kf'] = lf.kalman_filter(dic, config, delta)

    # Show the results
    vs.visualize_distance_error_with_heatmap(all_results['raw'], 'X_Real', 'Y_Real', 'X_LS', 'Y_LS', vmin=0, vmax=300, title="Raw Local")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['kf'] for anchor_id, results in all_anchors_results.items()}, 'X_Real', 'Y_Real', 'X_KF', 'Y_KF', vmin=0, vmax=2, title="KF Local")


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

    aoa_local_kf(anchors_df_dict, config, delta)

if __name__ == '__main__':
    main()