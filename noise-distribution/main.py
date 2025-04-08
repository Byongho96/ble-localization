import os
import yaml
import pandas as pd
import data_processing as dp
import statistical_analysis as sa
import visualize as vs

base_dir = os.path.dirname(os.path.abspath(__file__))

def aoa():
    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']
    offset = config['offset']

    gt_path = os.path.join(base_dir, "../dataset/calibration/gt/gt_calibration.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/calibration/beacons/beacons_calibration.csv")
    ms_df = pd.read_csv(ms_path)

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():   
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)

        anchors_df_dict[anchor_id] = anchor_gt_discretized_df

    # Calculate the statistics
    all_anchors_results = {}

    for anchor_id, anchor_df in anchors_df_dict.items():
        anchor_results = []

        # Filter the data
        for (x, y), point_df in anchor_df.groupby(["X_Real", "Y_Real"]):
            anchor_results.append(sa.calculate_statistics(point_df, 'Azimuth'))

        # Concatenate the results
        all_anchors_results[anchor_id] = pd.concat(anchor_results, ignore_index=True)

    print('df', all_anchors_results   )

    # Show the results
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'Sigma', vmin=0, vmax=10)   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'S', vmin= -5, vmax= 5)   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'K', vmin =0, vmax = 50)   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'Hyper_Skew', vmin= -100, vmax = 300)   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'Peak_Prob', vmin= 0, vmax= 5)   

def rssi():
    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']
    offset = config['offset']

    gt_path = os.path.join(base_dir, "../dataset/static/gt/gt_static_east.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/static/beacons/beacons_static_east.csv")
    ms_df = pd.read_csv(ms_path)

    # Filter with channel
    ms_df = ms_df[ms_df['Channel'] == 37]

    # Group by anchors
    anchors_df_dict = { anchor_id: anchor_df for anchor_id, anchor_df in ms_df.groupby("AnchorID") }

    # Preprocess the data
    for anchor_id, anchor_df in anchors_df_dict.items():   
        anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, anchor_df, offset)
        anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)

        anchors_df_dict[anchor_id] = anchor_gt_discretized_df

    # Calculate the statistics
    all_anchors_results = {}

    for anchor_id, anchor_df in anchors_df_dict.items():
        anchor_results = []

        # Filter the data
        for (x, y), point_df in anchor_df.groupby(["X_Real", "Y_Real"]):
            anchor_results.append(sa.calculate_statistics(point_df, '1stP'))

        # Concatenate the results
        all_anchors_results[anchor_id] = pd.concat(anchor_results, ignore_index=True)


    # Show the results
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'Sigma', vmin=0, vmax=3)   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'S', vmin= -1, vmax = 1)   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'K', vmin =0, vmax = 5)   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'Hyper_Skew', vmin = -30, vmax = 20)   
    vs.visualize_all_anchors_with_heatmap({anchor_id: results for anchor_id, results in all_anchors_results.items()}, 'Peak_Prob', vmin = 0.1, vmax = 0.5)   

def point():
    CHANNEL = 39
    ANCHOR_ID = 6504
    TAG_POINTS = [
        (840, 480), # BAD
        (480, 360) # GOOD
    ]

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../config.yml")))
    delta = config['delta']
    offset = config['offset']

    gt_path = os.path.join(base_dir, "../dataset/calibration/gt/gt_calibration.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/calibration/beacons/beacons_calibration.csv")
    ms_df = pd.read_csv(ms_path)

    # Filter with channel and anchor ID
    ms_df = ms_df[ms_df['Channel'] == CHANNEL]
    ms_df = ms_df[ms_df['AnchorID'] == ANCHOR_ID]

    # Preprocess the data
    anchor_gt_df = dp.filter_with_position_ground_truth(gt_df, ms_df, offset)
    anchor_gt_discretized_df = dp.discretize_by_delta(anchor_gt_df, delta)

    # Show the results
    vs.plot_distribution(anchor_gt_discretized_df, TAG_POINTS, 'Azimuth')   
    vs.plot_distribution(anchor_gt_discretized_df, TAG_POINTS, '1stP')   
    vs.plot_distribution(anchor_gt_discretized_df, TAG_POINTS, '2ndP')   

if __name__ == "__main__":
    # aoa()
    # rssi()
    point()