import math
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
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

def local_2D_kalman_filter(df: pd.DataFrame, dt: int = 20) -> pd.DataFrame:
    '''
    Localization 2D Kalman Filter

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["X_LS", "Y_LS"]

    Returns:
        pd.DataFrame: DataFrame with new ["X_2D_KF", "Y_2D_KF"] column
    '''
    dt /= 1000  # ms -> sec

    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Initial state: [x, y, vx, vy]
    kf.x = np.array([df.iloc[0]["X_LS"], df.iloc[0]["Y_LS"],
                     0.0, 0.0])

    # State Transition Matrix
    kf.F = np.array([
        [1, 0,  dt,  0],
        [0,  1,  0, dt],
        [0,  0,  1,  0],
        [0,  0,  0,  1]])
    
    # Measurement Matrix
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]])
    
    """
    Mean Error : 107 -> 75
    Mean Std : 87 -> 62
    """
    # Initial Covariance Matrix
    kf.P = np.diag([75, 75, 1, 1])
    
    # Process Noise Covariance
    kf.Q = np.diag([19, 19, 7, 7])

    # Measurement Noise Covariance
    kf.R = np.diag([62**2, 62**2])  

    filtered_positions = []

    # Run the Kalman Filter    
    for _, row in df.iterrows():
        # Prediction step
        kf.predict()

        # Update step
        z = np.array([float(row["X_LS"]), float(row["Y_LS"])])
        kf.update(z)

        # Save the filtered position
        filtered_positions.append([float(kf.x[0]), float(kf.x[1])])

    df["X_2D_KF"], df["Y_2D_KF"] = zip(*filtered_positions)
    return df

def local_extended_kalman_filter(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 0) -> pd.DataFrame:
    """
    Extended kalman filter with constant velocity model (state: [x, y, vx, vy]).`
    
    Parameters:
        df (pd.DataFrame): DataFrame with ["X_Real", "Y_Real"] and each anchor's "Azimuth" information.
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        dt (int): Time interval (in milliseconds; converted to seconds internally).
        threshold (int): Threshold for the number of iterations before starting to save results.
    
    Returns:
        pd.DataFrame: Dataframe with ["Time_Bucket", "X_Real", "Y_Real", "X_EKF", "Y_EKF"] columns.
    """
    print("Start the Extended Kalman Filter Process")
    
    dt /= 1000  # ms -> sec
    num_anchors = len(anchor_ids)
    anchors_position = np.array([config["anchors"][aid]["position"] for aid in anchor_ids])
    anchors_orientation = np.array([config["anchors"][aid]["orientation"] for aid in anchor_ids])
    
    # State Transition Function (constant velocity model)
    def fx(x, dt):
        F = np.array([
            [1,  0,  dt, 0],
            [0,  1,  0, dt],
            [0,  0,  1,  0],
            [0,  0,  0,  1]
        ])
        return F @ x
    fx_lambda = lambda x: fx(x, dt) # Capture dt in lambda function
    
    # Jacobian of the state transition function. Same as F in this case
    def FJacobian(x, dt):
        F = np.array([
            [1,  0, dt,  0],
            [0,  1,  0, dt],
            [0,  0,  1,  0],
            [0,  0,  0,  1]
        ])
        return F
    FJacobian_lambda = lambda x: FJacobian(x, dt) # Capture dt in lambda function
    
    # Measurement Function
    anchors_orientation_rad = np.deg2rad(anchors_orientation)
    def hx(x):
        pos = x[:2]
        predicted_angles = []
        for (ax, ay), a_ori_rad in zip(anchors_position, anchors_orientation_rad):
            angle = np.arctan2(pos[0] - ax, pos[1] - ay) - a_ori_rad
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            predicted_angles.append(np.degrees(angle))
        return np.array(predicted_angles)
    
    # Jacobian of the measurement function
    def HJacobian(x):
        H = np.zeros((num_anchors, 4))
        for i, a_pos in enumerate(anchors_position):
            r = x[0] - a_pos[0]
            s = x[1] - a_pos[1]
            denom = r**2 + s**2
            if denom == 0:
                H[i, 0] = 0
                H[i, 1] = 0
            else:
                H[i, 0] = (180/np.pi) * (s / denom)
                H[i, 1] = (180/np.pi) * (-r / denom)
        return H
    
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=num_anchors)

    # Initial state
    initial_state = np.array([
        np.random.uniform(0, 1200),
        np.random.uniform(0, 600),
        0.0,
        0.0
    ])
    ekf.x = initial_state.copy()
    
    # Initial Covariance
    ekf.P *= 500.0
    
    # Measurement Noise
    ekf.R = np.eye(num_anchors) * (6.0 ** 2)
    
    # Set the state transition function and its Jacobian
    ekf.fx = fx_lambda
    ekf.FJacobian = FJacobian_lambda
    
    # Process Noise (pos_noise_std = 10, vel_noise_std = 1)
    ekf.Q = np.diag([10**2, 10**2, 1**2, 1**2])
    
    estimated_positions = []
    th = 0
    
    # Run the EKF
    for time_bucket, row in df.iterrows():
        measured_aoa = np.array([row[f"{aid}_Azimuth"] for aid in anchor_ids])
        
        ekf.predict()
        ekf.update(measured_aoa, HJacobian, hx)
        x_ekf, y_ekf = ekf.x[0], ekf.x[1]

        th += 1
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], x_ekf, y_ekf])
    
    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_EKF", "Y_EKF"])


def local_unscented_kalman_filter(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 0) -> pd.DataFrame:
    """
    Local Unscented Kalman Filter

    Parameters:
        df (pd.DataFrame): DataFrame with ["X_Real", "Y_Real"] and each anchor's "Azimuth" information.
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        dt (int): Time interval (in milliseconds; converted to seconds internally).

    Returns:
        pd.DataFrame: ["Time_Bucket", "X_Real", "Y_Real", "X_UKF", "Y_UKF"] 컬럼 포함 DataFrame.
    """
    print("Start the Unscented Kalman Filter Process")
    
    dt /= 1000  # ms -> sec

    num_anchors = len(anchor_ids)
    anchors_position = np.array([config["anchors"][aid]["position"] for aid in anchor_ids])
    anchors_orientation = np.array([config["anchors"][aid]["orientation"] for aid in anchor_ids])
    
    # fx : State Transition Function
    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        return F @ x

    # hx : Measurement Function
    anchors_orientation_rad = np.deg2rad(anchors_orientation)
    def hx(x: np.ndarray) -> np.ndarray:
        pos = x[:2]
        predicted_angles = []
        for (ax, ay), a_ori_rad in zip(anchors_position, anchors_orientation_rad):
            angle = np.arctan2(pos[0] - ax, pos[1] - ay) - a_ori_rad
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            predicted_angles.append(np.degrees(angle))
        return np.array(predicted_angles)
    
    # Sigma Points
    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
    
    # Unscented Kalman Filter
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=num_anchors, dt=dt, fx=fx, hx=hx, points=points)
    
    # Measurement Noise
    ukf.R = np.eye(num_anchors) * (6.0 ** 2)
    
    # Process Noise (pos_noise_std = 10, vel_noise_std = 1)
    ukf.Q = np.diag([10**2, 10**2, 1**2, 1**2])
    
    # Initial State
    initial_state = np.array([
        np.random.uniform(0, 1200),
        np.random.uniform(0, 600),
        0.0,
        0.0
    ])
    ukf.x = initial_state.copy()

    # Initial Covariance
    ukf.P = np.eye(4) * 500.0
    
    estimated_positions = []
    th = 0

    # Run the UKF
    for time_bucket, row in df.iterrows():
        measured_aoa = np.array([row[f"{aid}_Azimuth"] for aid in anchor_ids])
        
        ukf.predict()
        ukf.update(measured_aoa)
        x_ukf, y_ukf = ukf.x[0], ukf.x[1]

        th += 1
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], x_ukf, y_ukf])
    
    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_UKF", "Y_UKF"])

def local_particle_filter(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 5) -> pd.DataFrame:
    """
    Local Particle Filter applied for each group based on ("X_Real", "Y_Real").
    
    Parameters:
        df (pd.DataFrame): Result from prepare_merged_dataframe(), with the index as Time_Bucket.
                                  Each anchor's columns are in the format:
                                  ["X_Real", "Y_Real", f"{anchor1_id}_Azimuth}", f"{anchor2_id}_Azimuth}"].
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        dt (int): Time interval (in milliseconds; converted to seconds internally).
        threshold (int): Threshold for the number of iterations before starting to save results.
        
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
    
    # noise
    pos_noise_std = 10.
    vel_noise_std = 1.
    
    # Run Particle Filter
    pf.initialize_particles()

    estimated_positions = []
    th = 0

    # Run the Particle Filter
    for time_bucket, row in df.iterrows():
        # Extract the measured AoA for each anchor directly.
        measured_aoa = np.array([row[f"{aid}_Azimuth"] for aid in anchor_ids])
        
        # Execute the Particle Filter predict, update, and estimate steps.
        pf.predict(pos_noise_std, vel_noise_std) # Put the velocity data here if available.
        pf.update(measured_aoa, anchors_position, anchors_orientation)
        x_pf, y_pf = pf.estimate()

        th += 1
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], x_pf, y_pf])

    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_PF", "Y_PF"])
