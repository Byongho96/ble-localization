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


def create_h_function_v(anchors_position: np.ndarray, anchors_orientation: np.ndarray):
    """
    각 앵커의 위치와 방향을 바탕으로 측정 함수(hx)를 생성합니다.
    상태 벡터 x의 처음 두 요소 ([x, y])를 사용하여 각도를 계산합니다.
    """
    anchors_orientation_rad = np.deg2rad(anchors_orientation)
    
    def hx(x: np.ndarray) -> np.ndarray:
        pos = x[:2]  # [x, y]만 사용
        predicted_angles = []
        for (ax, ay), a_ori_rad in zip(anchors_position, anchors_orientation_rad):
            # 앵커에서 태그까지의 각도 계산 (비교: arctan2(pos[0]-ax, pos[1]-ay))
            angle = np.arctan2(pos[0] - ax, pos[1] - ay) - a_ori_rad
            # [-pi, pi]로 정규화 후 degree 단위로 변환
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            predicted_angles.append(np.degrees(angle))
        return np.array(predicted_angles)
    
    return hx

def fx(x: np.ndarray, dt: float) -> np.ndarray:
    """
    4차원 상태 전이 함수 (상수 속도 모델).
    
    x = [x, y, vx, vy]라 할 때,
      x_new = x + vx * dt
      y_new = y + vy * dt
      vx_new = vx
      vy_new = vy
    """
    F = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    return F @ x

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
    
    anchors_position = np.array([config["anchors"][aid]["position"] for aid in anchor_ids])
    anchors_orientation = np.array([config["anchors"][aid]["orientation"] for aid in anchor_ids])
    num_anchors = len(anchor_ids)
    
    # 상태 공간 4차원에 맞춰 hx 함수 생성
    hx = create_h_function_v(anchors_position, anchors_orientation)
    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=num_anchors, dt=dt, fx=fx, hx=hx, points=points)
    
    # 프로세스 노이즈 (위치, 속도에 대한 불확실성 반영)
    ukf.Q = np.diag([1.0, 1.0, 1.0, 1.0])
    angle_noise_std = 6.0
    ukf.R = np.eye(num_anchors) * (angle_noise_std ** 2)
    
    state_bounds = (0, 1200, 0, 600)
    min_x, max_x, min_y, max_y = state_bounds
    
    estimated_positions = []
    THRESHOLD = 200

    # 그룹별 처리 ("X_Real", "Y_Real" 기준)
    for (x_real, y_real), group_df in merged_df.groupby(["X_Real", "Y_Real"]):
        # 초기 상태: 위치는 범위 내 임의값, 속도는 0으로 초기화
        initial_state = np.array([
            600,
            300,
            0.0,
            0.0
        ])
        ukf.x = initial_state.copy()
        ukf.P = np.eye(4) * 500.0
        
        th = 0
        for time_bucket, row in group_df.iterrows():
            measured_aoa = np.array([row[f"{aid}_Azimuth"] for aid in anchor_ids])
            ukf.predict()
            ukf.update(measured_aoa)
            x_ukf, y_ukf = ukf.x[0], ukf.x[1]
            th += 1
            if th < THRESHOLD:
                continue
            estimated_positions.append([time_bucket, x_real, y_real, x_ukf, y_ukf])

            # if (x_real ==420) and (y_real == 180):
            #     print(x_ukf, y_ukf)
            
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

    # Prepare arrays for each anchor's position and orientation.
    anchors_position = np.array([config["anchors"][aid]["position"] for aid in anchor_ids])
    anchors_orientation = np.array([config["anchors"][aid]["orientation"] for aid in anchor_ids])
    
    # Assume that the ParticleFilter class is already implemented.
    pf = ParticleFilter(num_particles=1000, state_bounds=(0, 1200, 0, 600), angle_noise_std=6.)
    
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
