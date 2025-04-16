import os
import yaml
import pandas as pd
import data_processing as dp
import rssi_filter as rf
import visualize as vs

def aoa_kf(dic: dict, delta: int):
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
            # anchor_results['maf'].append(rf.rssi_moving_average_filter(point_df))
            # anchor_results['median'].append(rf.rssi_median_filter(point_df))
            # anchor_results['low_pass'].append(rf.rssi_low_pass_filter(point_df))
            # anchor_results['1d_kf'].append(rf.rssi_1d_kalman_filter(point_df, delta))
            # anchor_results['2d_kf'].append(rf.rssi_2d_kalman_filter(point_df, delta))

        # Concatenate the results
        for key in anchor_results:
            if not anchor_results[key]:
                continue
            anchor_results[key] = pd.concat(anchor_results[key], ignore_index=True)
        
        all_anchors_results[anchor_id] = anchor_results

    # Show the results
    all_anchors_results[1]['raw']['Zero'] = 0
    
    vs.visualize_all_anchors_with_heatmap({anchor_id: results['raw'] for anchor_id, results in all_anchors_results.items()}, 'Distance', 'RSSI_Distance', vmin=0, vmax=300, title="Raw AoA")   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results['raw'] for anchor_id, results in all_anchors_results.items()}, 'Zero', 'RSSI', vmin=0, vmax=100, title="Raw AoA")   
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['maf'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_MAF', vmin=0, vmax=15, title="MAF AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['median'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_Median', vmin=0, vmax=15, title="Median AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['low_pass'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_LowPass', vmin=0, vmax=15, title="Low Pass AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['1d_kf'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_1d_KF', vmin=0, vmax=15, title="1D KF AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['2d_kf'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_2d_KF', vmin=0, vmax=15, title="2D KF AoA")

def preprocessing():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))
    offset = config['offset']

    gt_path = os.path.join(base_dir, "../dataset/0409/gt/anchor1.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/0409/beacons/anchor1.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchor_id =1

    position = config['anchors']['0409'][anchor_id]['position']   

    anchors_parameters_dict = {
        1: {
            'rssi_0': None,
            'n': None,
        }
    }
    anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, ms_df, offset)
    anchor_gt_rssi_df = dp.calculate_rssi_and_distance(anchor_gt_df, position)
    rssi_0, n = dp.calculate_log_distance_parameters(anchor_gt_rssi_df)

    anchors_parameters_dict[anchor_id]['rssi_0'] = rssi_0
    anchors_parameters_dict[anchor_id]['n'] = n

    return anchors_parameters_dict


def calibration(anchors_parameters_dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))
    config['anchors'] = config['anchors']['0409']
    delta = config['delta']
    offset = config['offset']

    gt_path = os.path.join(base_dir, "../dataset/0409/gt/anchor1.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path_1 = os.path.join(base_dir, "../dataset/0409/beacons/anchor1.csv")
    ms_df_1 = pd.read_csv(ms_path_1)
    # ms_path_2 = os.path.join(base_dir, "../dataset/0409/beacons/rectangular.csv")
    # ms_df_2 = pd.read_csv(ms_path_2)
    # ms_path_3 = os.path.join(base_dir, "../dataset/0409/beacons/rectangular.csv")
    # ms_df_3 = pd.read_csv(ms_path_3)
    # ms_path_4 = os.path.join(base_dir, "../dataset/0317/anchor4/rectangular.csv")
    # ms_df_4 = pd.read_csv(ms_path_4)

    # Group by anchors
    anchors_df_dict = { 
        1: ms_df_1,
        # 2: ms_df_2,
        # 3: ms_df_3,
        # 4: ms_df_4,
    }

    print(anchors_parameters_dict)
    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        position = config['anchors'][anchor_id]['position']   
        rssi_0, n = anchors_parameters_dict[anchor_id]['rssi_0'], anchors_parameters_dict[anchor_id]['n']

        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchor_gt_discretized_rssi_df = dp.calculate_rssi_and_distance(anchor_gt_discretized_df, position)
        anchor_gt_discretized_rssi_estimated_df = dp.calculate_rssi_estimated_distance(anchor_gt_discretized_rssi_df, rssi_0, n)

        anchors_df_dict[anchor_id] = anchor_gt_discretized_rssi_estimated_df

    aoa_kf(anchors_df_dict, delta)

def mobility():
    pass

if __name__ == '__main__':
    anchors_parameters_dict = preprocessing()

    calibration(anchors_parameters_dict)
    # mobility()