import os
import yaml
import pandas as pd
import data_processing as dp
import aoa_filter as af
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
            anchor_results['maf'].append(af.aoa_moving_average_filter(point_df))
            anchor_results['median'].append(af.aoa_median_filter(point_df))
            anchor_results['low_pass'].append(af.aoa_low_pass_filter(point_df))
            anchor_results['1d_kf'].append(af.aoa_1d_kalman_filter(point_df, delta))
            anchor_results['2d_kf'].append(af.aoa_2d_kalman_filter(point_df, delta))

        # Concatenate the results
        for key in anchor_results:
            if not anchor_results[key]:
                continue
            anchor_results[key] = pd.concat(anchor_results[key], ignore_index=True)
        
        all_anchors_results[anchor_id] = anchor_results

    # Show the results
    vs.visualize_all_anchors_with_heatmap({anchor_id: results['raw'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth', vmin=0, vmax=15, title="Raw AoA")   
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['maf'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_MAF', vmin=0, vmax=15, title="MAF AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['median'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_Median', vmin=0, vmax=15, title="Median AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['low_pass'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_LowPass', vmin=0, vmax=15, title="Low Pass AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['1d_kf'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_1d_KF', vmin=0, vmax=15, title="1D KF AoA")
    # vs.visualize_all_anchors_with_heatmap({anchor_id: results['2d_kf'] for anchor_id, results in all_anchors_results.items()}, 'Azimuth_Real', 'Azimuth_2d_KF', vmin=0, vmax=15, title="2D KF AoA")

def calibration():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']
    offset = config['offset']

    gt_path = os.path.join(base_dir, "../dataset/static/gt/gt_static_east.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/static/beacons/beacons_static_east.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():
        position = config['anchors'][anchor_id]['position']   
        orientation = config['anchors'][anchor_id]['orientation']
    
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchor_gt_discretized_aoa_df = dp.calculate_aoa_ground_truth(anchor_gt_discretized_df, position, orientation)

        anchors_df_dict[anchor_id] = anchor_gt_discretized_aoa_df

    aoa_kf(anchors_df_dict, delta)

def mobility():
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
        orientation = config['anchors'][anchor_id]['orientation']

        anchor_gt_df = dp.filter_with_position_ground_truth(gt_interpolated_df, anchor_df)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)
        anchor_gt_discretized_aoa_df = dp.calculate_aoa_ground_truth(anchor_gt_discretized_df, position, orientation)

        # Add zero column "Zero"
        anchor_gt_discretized_aoa_df["Zero"] = 0

        # Filter the data
        # anchor_gt_discretized_aoa_df = anchor_gt_discretized_aoa_df[ anchor_gt_discretized_aoa_df['AnchorID'] != 6503 ]
        # anchor_gt_discretized_aoa_df =  anchor_gt_discretized_aoa_df[ anchor_gt_discretized_aoa_df['Azimuth_Real'].abs() <= 45]
        
        anchors_df_dict[anchor_id] = anchor_gt_discretized_aoa_df 

    # Merge the dataframes
    vs.visualize_all_anchors_with_heatmap(anchors_df_dict, 'Azimuth_Real', 'Azimuth', vmin=0, vmax=15, title="Raw AoA")   
    vs.visualize_all_anchors_with_heatmap(anchors_df_dict, 'Zero', 'Azimuth_Std', vmin=0, vmax=5, title="Raw AoA Std")   
    vs.visualize_all_anchors_with_heatmap(anchors_df_dict, 'Zero', '1stP_Std', vmin=0, vmax=10, title="Raw AoA Std")   
    vs.visualize_all_anchors_with_heatmap(anchors_df_dict, 'Zero', '2ndP_Std', vmin=0, vmax=10, title="Raw AoA Std")   


if __name__ == '__main__':
    calibration()
    # mobility()