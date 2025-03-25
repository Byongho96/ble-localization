import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

def filter_and_match_ground_truth(gt_df, ms_df, config):
    '''
    Filter the measurement data to match the ground truth data
    Add the real position and azimuth to the measurement data ["X_Real", "Y_Real", "Azimuth_Real"]
    '''
    filtered_data = []

    for _, row in gt_df.iterrows():
        start_timestamp, end_timestamp, x, y = row

        def compute_azimuth(df_row):
            anchor = df_row["AnchorID"]
            position = config['anchors'][anchor]['position']   
            orientation = config['anchors'][anchor]['orientation']

            azimuth_real = math.degrees(math.atan2(x - position[0], y - position[1])) - orientation
            if azimuth_real < -450:
                azimuth_real += 540
            elif azimuth_real < -270:
                azimuth_real += 360
            elif azimuth_real < -90:
                azimuth_real += 180

            return azimuth_real 

        filtered_df = ms_df[
            (ms_df["Timestamp"] >= start_timestamp) & (ms_df["Timestamp"] <= end_timestamp)
        ].copy()
        filtered_df["X_Real"] = x
        filtered_df["Y_Real"] = y
        filtered_df["Azimuth_Real"] = filtered_df.apply(compute_azimuth, axis=1) # row-wise operation
        filtered_data.append(filtered_df)
        
    result_df = pd.concat(filtered_data, ignore_index=True) if filtered_data else pd.DataFrame()

    return result_df

def aoa_1d_kalman_filter(df):
    '''
    AoA 1D Kalman Filter
    AoA change_rate would be proportional to the RSSI_distance
    '''
    dt = 0.02 # 50 Hz
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([float(df.iloc[0]['Azimuth']), 0.])  # [azimuth, azimuth_rate]
    kf.F = np.array([[1, dt],[0, 1]])
    kf.H = np.array([[1, 0]])
    kf.P *= 1000.
    kf.R *= 5  # Measurement noise : [U-blox C211 5 degree]
    kf.Q *= dt  # Process noise proportional to the dt

    # Run the Kalman Filter
    prev_timestamp = None
    filtered_azimuths = []

    for _, row in df.iterrows():
        timestamp = row["Timestamp"]

        # Update time difference
        if prev_timestamp:
            dt = (timestamp - prev_timestamp) / 1000.0 
        else:
            dt = 0.02
        prev_timestamp = timestamp

        # Update the Kalman Filter
        kf.F = np.array([[1, dt],[0, 1]])
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1.0)

        # Prediction step
        kf.predict()

        # Update step
        z = float(row["Azimuth"])
        kf.update(z)

        # Save the filtered azimuth
        filtered_azimuths.append(kf.x[0])

    df["Azimuth_KF"] = filtered_azimuths
    return df

def visualize_all_anchors_with_heatmap(all_results, gt_column, ms_column, vmin=None, vmax=None):    
    '''
    Visualize the error on each positions with heatmap
    data format: {anchor_id: DataFrame}
    '''
    cols = 2
    rows = math.ceil(len(all_results) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()

    # Mean Error
    for idx, key in enumerate(all_results):
        ax = axes[idx]
        df = all_results[key]

        error_df = abs(df[ms_column] - df[gt_column])

        sc = ax.scatter(df["X_Real"], df["Y_Real"], c=error_df, cmap='coolwarm', s=100, vmin=vmin, vmax=vmax)

        # color bar
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("A_real - A_measurement")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Error map for {key}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()