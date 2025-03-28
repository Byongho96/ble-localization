import math
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from ParticleFilter import ParticleFilter

def least_squares_triangulation(df: pd.DataFrame, config: dict, anchor_ids: list) -> pd.DataFrame:
    """
    Least Squares Triangulation
    
    Parameters:
        df (pd.DataFrame): Index is Time_Bucket, columns are [ f"X_Real", f"Y_Real", f"{anchor_id}_AnchorID", f"{anchor_id}_Azimuth"]
        config (dict): Configuration containing anchor positions and orientations
        anchor_ids (list): List of anchor IDs
    
    Returns:
        pd.DataFrame: DataFrame with new ["Time_Bucket", "X_Real", "Y_Real", "X_LS", "Y_LS"] columns
    """
    estimated_positions = []
    
    for time_bucket, row in df.iterrows():
        H = []
        C = []
        
        for anchor in anchor_ids:
            azimuth = row[f"{anchor}_Azimuth"]
            pos = config["anchors"][anchor]["position"]
            orientation = config["anchors"][anchor]["orientation"]

            aoa_rad = math.radians(90 - azimuth - orientation)
            H.append([-math.tan(aoa_rad), 1])
            C.append([pos[1] - pos[0] * math.tan(aoa_rad)])
        
        H = np.array(H)
        C = np.array(C)
        
        try:
            e = np.linalg.inv(H.T @ H) @ H.T @ C
        except np.linalg.LinAlgError:
            continue
        
        x_LS, y_LS = e[0][0], e[1][0]
        x_real = row["X_Real"]
        y_real = row["Y_Real"]
        estimated_positions.append([time_bucket, x_real, y_real, x_LS, y_LS])
    
    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_LS", "Y_LS"])

def local_2D_kalman_filter(df: pd.DataFrame, dt: float = 0.02) -> pd.DataFrame:
    '''
    Localization 2D Kalman Filter

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["X_LS", "Y_LS"]

    Returns:
        pd.DataFrame: DataFrame with new ["X_2D_KF", "Y_2D_KF"] column
    '''
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([float(df.iloc[0]['X_LS']), 0., float(df.iloc[0]['Y_LS']), 0.])  # [x, x_rate, y, y_rate]
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])
    kf.P *= 500
    kf.R *= 100
    kf.Q = np.diag([1, 0.1, 1, 0.1])  # Process noise

    # Run the Kalman Filter
    filtered_positions = []
    
    for _, row in df.iterrows():
        # Prediction step
        kf.predict()

        # Update step
        z = np.array([float(row["X_LS"]), float(row["Y_LS"])])
        kf.update(z)

        # Save the filtered position
        filtered_positions.append([float(kf.x[0]), float(kf.x[2])])

    df["X_2D_KF"], df["Y_2D_KF"] = zip(*filtered_positions)
    return df

def local_extended_kalman_filter(df: pd.DataFrame, dt: float = 0.02) -> pd.DataFrame:
    pass


def local_unscented_kalman_filter(merged_df: pd.DataFrame, config: dict, anchor_ids: list, dt: float = None) -> pd.DataFrame:
    """
    Local Particle Filter applied for each group based on ("X_Real", "Y_Real").
    
    Parameters:
        merged_df (pd.DataFrame): Result from prepare_merged_dataframe(), with the index as Time_Bucket.
                                  Each anchor's columns are in the format:
                                  ["X_Real", "Y_Real", f"{anchor1_id}_Azimuth}", f"{anchor2_id}_Azimuth}"].
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        
    Returns:
        pd.DataFrame: DataFrame with columns ["Time_Bucket", "X_Real", "Y_Real", "X_UKF", "Y_UKF"].
    """
    print("Start the Unscented Kalman Filter Process")
    
    dt /= 1000 # ms -> sec

    num_anchors = len(anchor_ids)
    anchors_position = np.array([config["anchors"][aid]["position"] for aid in anchor_ids])
    anchors_orientation = np.array([config["anchors"][aid]["orientation"] for aid in anchor_ids])
    
    # fx : State Transition Function
    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition function for the Unscented Kalman Filter.

        [x, y, x_dot, y_dot] -> [x, y, x_dot, y_dot]
        """
        dt2 = 0.5 * dt**2
        F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return F @ x
    
    # hx : Measurement Function
    anchors_orientation_rad = np.deg2rad(anchors_orientation)
    
    def hx(x: np.ndarray) -> np.ndarray:
        pos = x[:2]  # Extract the position
        predicted_angles = []
        for (ax, ay), a_ori_rad in zip(anchors_position, anchors_orientation_rad):
            angle = np.arctan2(pos[0] - ax, pos[1] - ay) - a_ori_rad
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            predicted_angles.append(np.degrees(angle))

        return np.array(predicted_angles)

    # Sigma Points
    points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2, kappa=0)
    
    # Unscented Kalman Filter
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=num_anchors, dt=dt, fx=fx, hx=hx, points=points)
    
    # Measurement Noise
    ukf.R = np.eye(num_anchors) * (6.0** 2) # angle noise std = 6.0
    
    state_bounds = (0, 1200, 0, 600)
    min_x, max_x, min_y, max_y = state_bounds
    
    estimated_positions = []
    THRESHOLD = 100

    # Run the UKF for each group based on ("X_Real", "Y_Real").
    for (x_real, y_real), group_df in merged_df.groupby(["X_Real", "Y_Real"]):
        initial_state = np.array([
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y),
            0.0, 0.0,
            0.0, 0.0
        ])
        ukf.x = initial_state.copy()
        ukf.P = np.eye(6) * 500.0
        
        th = 0
        for time_bucket, row in group_df.iterrows():
            measured_aoa = np.array([row[f"{aid}_Azimuth"] for aid in anchor_ids])

            # Adaptive Process Noise
            vel = np.linalg.norm(ukf.x[2:4])
            if vel < 0.1:  # If the velocity is low
                ukf.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
            else:
                ukf.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])

            ukf.predict()
            ukf.update(measured_aoa)
            x_ukf, y_ukf = ukf.x[0], ukf.x[1]
            
            th += 1
            if th < THRESHOLD:
                continue
            
            estimated_positions.append([time_bucket, x_real, y_real, x_ukf, y_ukf])
            
        print(f"Processed sample in group ({x_real}, {y_real})")
    
    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_UKF", "Y_UKF"])

def local_particle_filter(merged_df: pd.DataFrame, config: dict, anchor_ids: list, dt: float = None) -> pd.DataFrame:
    """
    Local Particle Filter applied for each group based on ("X_Real", "Y_Real").
    
    Parameters:
        merged_df (pd.DataFrame): Result from prepare_merged_dataframe(), with the index as Time_Bucket.
                                  Each anchor's columns are in the format:
                                  ["X_Real", "Y_Real", f"{anchor1_id}_Azimuth}", f"{anchor2_id}_Azimuth}"].
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        
    Returns:
        pd.DataFrame: DataFrame with columns ["Time_Bucket", "X_Real", "Y_Real", "X_PF", "Y_PF"].
    """
    print("Start the Particle Filter Process")

    dt /= 1000 # ms -> sec

    # Prepare arrays for each anchor's position and orientation.
    anchors_position = np.array([config["anchors"][aid]["position"] for aid in anchor_ids])
    anchors_orientation = np.array([config["anchors"][aid]["orientation"] for aid in anchor_ids])
    
    # Assume that the ParticleFilter class is already implemented.
    pf = ParticleFilter(num_particles=1000, state_bounds=(0, 1200, 0, 600), angle_noise_std=6., dt=dt)
    
    estimated_positions = []
    THRESHOLD = 100
    
    # Test each group based on ("X_Real", "Y_Real").
    for (x_real, y_real), group_df in merged_df.groupby(["X_Real", "Y_Real"]):
        pf.initialize_particles()
        
        th = 0  # Threshold for skipping the initial samples

        for time_bucket, row in group_df.iterrows():
            # Extract the measured AoA for each anchor directly.
            measured_aoa = np.array([row[f"{aid}_Azimuth"] for aid in anchor_ids])
            
            # Execute the Particle Filter predict, update, and estimate steps.
            pf.predict() # Put the velocity data here if available.
            pf.update(measured_aoa, anchors_position, anchors_orientation)
            x_pf, y_pf = pf.estimate()

            # Record the estimated position if the threshold is reached.
            th += 1
            if th < THRESHOLD:
                continue

            estimated_positions.append([time_bucket, x_real, y_real, x_pf, y_pf])

        print(f"Processed sample in group ({x_real}, {y_real})")
    
    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_PF", "Y_PF"])
