import os
import yaml
import pandas as pd
import data_processing as dp
import rssi_filter as rf
import visualize as vs

def rssi_kf(dic: dict, delta: int):
    all_anchors_results = {}

    for anchor_id, anchor_df in dic.items():

        anchor_results = {
            'raw': [],
            'maf': [],
            'median': [],
            'low_pass': [],
            '1d_kf': [],
            '2d_kf': [],
        }

        # Filter the data
        for (x, y), point_df in anchor_df.groupby(["X_Real", "Y_Real"]):
            anchor_results['raw'].append(point_df)
            anchor_results['maf'].append(rf.rssi_moving_average_filter(point_df))
            anchor_results['median'].append(rf.rssi_median_filter(point_df))
            anchor_results['low_pass'].append(rf.rssi_low_pass_filter(point_df))
            anchor_results['1d_kf'].append(rf.rssi_1d_kalman_filter(point_df, delta))
            anchor_results['2d_kf'].append(rf.rssi_2d_kalman_filter(point_df, delta))

        # Concatenate the results
        for key in anchor_results:
            if not anchor_results[key]:
                continue
            anchor_results[key] = pd.concat(anchor_results[key], ignore_index=True)
        
        all_anchors_results[anchor_id] = anchor_results

    

    # Show the results
    vs.visualize_all_anchors_with_heatmap({anchor_id: results['raw'] for anchor_id, results in all_anchors_results.items()}, 'Distance', 'RSSI_Distance', vmin=0, vmax=300, title="Raw RSSI Distance")   
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['maf'] for anchor_id, results in all_anchors_results.items()}, 'Distance', 'RSSI_Distance_MAF', vmin=0, vmax=300, title="MAF RSSI Distance")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['median'] for anchor_id, results in all_anchors_results.items()}, 'Distance', 'RSSI_Distance_Median', vmin=0, vmax=300, title="Median RSSI Distance")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['low_pass'] for anchor_id, results in all_anchors_results.items()}, 'Distance', 'RSSI_Distance_LowPass', vmin=0, vmax=300, title="Low Pass RSSI Distance")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['1d_kf'] for anchor_id, results in all_anchors_results.items()}, 'Distance', 'RSSI_Distance_1d_KF', vmin=0, vmax=300, title="1D KF RSSI Distance")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['2d_kf'] for anchor_id, results in all_anchors_results.items()}, 'Distance', 'RSSI_Distance_2d_KF', vmin=0, vmax=300, title="2D KF RSSI Distance")

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

def calibration(anchors_parameters_dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    offset = config['offset']
    delta = config['delta']

    gt_path = os.path.join(base_dir, "../dataset/calibration/gt/gt_calibration.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/calibration/beacons/beacons_calibration.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        position = config['anchors'][anchor_id]['position']   
        rssi_0, n = anchors_parameters_dict[anchor_id]['rssi_0'], anchors_parameters_dict[anchor_id]['n']
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchor_gt_discretized_rssi_df = dp.calculate_rssi_and_distance(anchor_gt_discretized_df, position)
        anchor_gt_discretized_rssi_estimated_df = dp.calculate_rssi_estimated_distance(anchor_gt_discretized_rssi_df, rssi_0, n)

        anchors_df_dict[anchor_id] = anchor_gt_discretized_rssi_estimated_df

    rssi_kf(anchors_df_dict, delta)

def mobility(anchors_parameters_dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load Config
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']

    # Load files
    gt_path = os.path.join(base_dir, "../dataset/mobility/gt/use-case1/gt_mobility_use-case1_run1.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/mobility/beacons/use-case1/beacons_mobility_use-case1_run1.csv")
    ms_df = pd.read_csv(ms_path)

    # Interpolate the ground truth data
    gt_interpolated_df = dp.interpolate_ground_truth(gt_df, delta)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        position = config['anchors'][anchor_id]['position']
        rssi_0, n = anchors_parameters_dict[anchor_id]['rssi_0'], anchors_parameters_dict[anchor_id]['n']

        anchor_gt_df = dp.filter_with_position_ground_truth(gt_interpolated_df, anchor_df)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchor_gt_discretized_rssi_df = dp.calculate_rssi_and_distance(anchor_gt_discretized_df, position)
        anchor_gt_discretized_rssi_estimated_df = dp.calculate_rssi_estimated_distance(anchor_gt_discretized_rssi_df, rssi_0, n)

        anchors_df_dict[anchor_id] = anchor_gt_discretized_rssi_estimated_df

    print(anchor_gt_discretized_rssi_estimated_df)
    vs.visualize_all_anchors_with_heatmap(anchors_df_dict, 'Distance', 'RSSI_Distance', vmin=0, vmax=300, title="Raw RSSI Distance")   


if __name__ == '__main__':
    anchors_parameters_dict = preprocessing()

    calibration(anchors_parameters_dict)
    # mobility(anchors_parameters_dict)