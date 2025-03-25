import os
import yaml
import pandas as pd
import utils as utils

def aoa_kf(df):
    all_results = {}

    for anchor_id, anchor_df in df.groupby("AnchorID"):

        anchor_results = {
            'raw': [],
            'kf': [],
        }

        for (x, y), point_df in anchor_df.groupby(["X_Real", "Y_Real"]):
            
            # Raw AoA
            anchor_results['raw'].append(point_df)

            # EKF for each anchors' AoA
            anchor_results['kf'].append(utils.aoa_1d_kalman_filter(point_df))


        # Concatenate the results
        for key in anchor_results:
            anchor_results[key] = pd.concat(anchor_results[key], ignore_index=True)
        
        all_results[anchor_id] = anchor_results

    # Regroup the results
    results_by_method = {
        'raw': {},
        'kf': {},
    }

    for anchor_id, anchor_results in all_results.items():
        for key in anchor_results:
            results_by_method[key][anchor_id] = anchor_results[key]

    # Show the results
    utils.visualize_all_anchors_with_heatmap(results_by_method['raw'], 'Azimuth_Real', 'Azimuth', 0, 15)
    utils.visualize_all_anchors_with_heatmap(results_by_method['raw'], 'Azimuth_Real', 'Azimuth_KF', 0, 15)


def aoa_local_kf(df):
    for x, y in [1, 2, 3, 4, 5]:
        pass
        # KF for each anchors' AoA
        

        # UKF for each anchors' AoA

        # PK for each anchors' AoA


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))

    gt_path = os.path.join(base_dir, "../dataset/calibration/gt/gt_calibration.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/calibration/beacons/beacons_calibration.csv")
    ms_df = pd.read_csv(ms_path)

    filtered_df = utils.filter_and_match_ground_truth(gt_df, ms_df, config)

    # aoa_kf(filtered_df)
    aoa_local_kf(filtered_df)

if __name__ == '__main__':
    main()