import math
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from ParticleFilter import ParticleFilter

LEAST_ANCHORS = 2

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

    cnt = 0
    
    for time_bucket, row in df.iterrows():
        H = []
        C = []
        valid_anchor_count = 0
        
        for anchor in anchor_ids:
            azimuth = row[f"{anchor}_Azimuth"]
            
            # skip if azimuth is missing (NaN)
            if pd.isna(azimuth):
                continue  

            pos = config["anchors"][anchor]["position"]
            orientation = config["anchors"][anchor]["orientation"]

            aoa_rad = math.radians(90 - azimuth - orientation)
            H.append([-math.tan(aoa_rad), 1])
            C.append([pos[1] - pos[0] * math.tan(aoa_rad)])
            valid_anchor_count += 1
        
        # skip if fewer than 2 valid anchors
        if valid_anchor_count < LEAST_ANCHORS:
            continue  
        cnt += 1
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
    
    print(cnt)

    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_LS", "Y_LS"])

def weighted_least_squares_triangulation(df: pd.DataFrame, config: dict, anchor_ids: list) -> pd.DataFrame:
    """
    Weighted Least Squares Triangulation
    앵커별 Azimuth_Var 에 따라 가중치를 부여합니다.

    Returns:
        pd.DataFrame: ["Time_Bucket","X_Real","Y_Real","X_LS","Y_LS"]
    """
    est = []

    cnt =0

    for time_bucket, row in df.iterrows():
        H, C, W_list = [], [], []
        valid = 0

        for anchor in anchor_ids:
            az = row.get(f"{anchor}_Azimuth")
            var = row.get(f"{anchor}_Azimuth_Var")
            if pd.isna(az) or pd.isna(var):
                continue

            pos = config["anchors"][anchor]["position"]
            ori = config["anchors"][anchor]["orientation"]
            # AoA 라디안 변환
            aoa_rad = math.radians(90 - az - ori)

            # H, C 항 추가
            H.append([-math.tan(aoa_rad), 1.0])
            C.append([pos[1] - pos[0] * math.tan(aoa_rad)])

            # 오차 모델 & 가중치
            # error = 1.3259 * (var ** 0.4069)
            # error = 0.8490 * var ** 0.5 + 0.2458
            error = 0.8514 * var ** 0.5 + 0.2294 if var < 400 else 0.0102 * var + 18.0772

            w = 1.0 / (error ** 2) if error > 0 else 0.0
            W_list.append(w)

            valid += 1

        if valid < LEAST_ANCHORS:
            continue

        cnt += 1

        H = np.array(H)        # shape (valid, 2)
        C = np.array(C)        # shape (valid, 1)
        W = np.diag(W_list)    # shape (valid, valid)

        try:
            # Weighted least squares
            e = np.linalg.inv(H.T @ W @ H) @ (H.T @ W @ C)
        except np.linalg.LinAlgError:
            continue

        x_ls, y_ls = float(e[0]), float(e[1])
        est.append([
            time_bucket,
            row["X_Real"],
            row["Y_Real"],
            x_ls,
            y_ls
        ])

    print(cnt)

    return pd.DataFrame(est, columns=["Time_Bucket","X_Real","Y_Real","X_WLS","Y_WLS"])

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
    Extended Kalman Filter with constant velocity model (state: [x, y, vx, vy]).
    Supports variable number of valid AoA measurements per time step.
    
    Parameters:
        df (pd.DataFrame): DataFrame with ["X_Real", "Y_Real"] and each anchor's "Azimuth" information.
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        dt (int): Time interval (in milliseconds; converted to seconds internally).
        threshold (int): Threshold for the number of iterations before starting to save results.
    
    Returns:
        pd.DataFrame: DataFrame with ["Time_Bucket", "X_Real", "Y_Real", "X_EKF", "Y_EKF"] columns.
    """
    print("Start the Extended Kalman Filter Process")
    
    dt /= 1000  # Convert ms to seconds
    
    # State Transition Function (constant velocity model)
    def fx(x, dt):
        F = np.array([
            [1,  0,  dt, 0],
            [0,  1,  0, dt],
            [0,  0,  1,  0],
            [0,  0,  0,  1]
        ])
        return F @ x
    fx_lambda = lambda x: fx(x, dt)

    # Jacobian of the state transition function (same as F)
    def FJacobian(x, dt):
        return np.array([
            [1,  0, dt,  0],
            [0,  1,  0, dt],
            [0,  0,  1,  0],
            [0,  0,  0,  1]
        ])
    FJacobian_lambda = lambda x: FJacobian(x, dt)

    # Initialize EKF
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=1)  # dim_z will be adjusted dynamically

    # Initial state
    ekf.x = np.array([
        np.random.uniform(0, 1200),
        np.random.uniform(0, 600),
        0.0,
        0.0
    ])

    # Initial covariance
    ekf.P *= 500.0

    # Process noise (pos_noise_std = 10, vel_noise_std = 1)
    ekf.Q = np.diag([10**2, 10**2, 1**2, 1**2])

    # Set state transition functions
    ekf.fx = fx_lambda
    ekf.FJacobian = FJacobian_lambda

    estimated_positions = []
    th = 0

    # Run the EKF
    for time_bucket, row in df.iterrows():
        measured_aoa = []
        valid_positions = []
        valid_orientations = []

        for aid in anchor_ids:
            az = row.get(f"{aid}_Azimuth", np.nan)
            if not pd.isna(az):
                measured_aoa.append(az)
                valid_positions.append(config["anchors"][aid]["position"])
                valid_orientations.append(config["anchors"][aid]["orientation"])

        if len(measured_aoa) < LEAST_ANCHORS:
            continue  # Skip update if fewer than 2 anchors are available

        measured_aoa = np.array(measured_aoa)
        anchors_position_np = np.array(valid_positions)
        print(f"anchors_position_np: {anchors_position_np}")
        anchors_orientation_rad = np.deg2rad(valid_orientations)

        # Measurement function (variable-length)
        def hx(x):
            pos = x[:2]
            predicted_angles = []
            for (ax, ay), a_ori_rad in zip(anchors_position_np, anchors_orientation_rad):
                angle = np.arctan2(pos[0] - ax, pos[1] - ay) - a_ori_rad
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                predicted_angles.append(np.degrees(angle))
            return np.array(predicted_angles)

        # Jacobian of the measurement function
        def HJacobian(x):
            H = np.zeros((len(anchors_position_np), 4))
            for i, (ax, ay) in enumerate(anchors_position_np):
                dx = x[0] - ax
                dy = x[1] - ay
                denom = dx**2 + dy**2
                if denom == 0:
                    H[i, 0] = 0
                    H[i, 1] = 0
                else:
                    H[i, 0] = (180 / np.pi) * (dy / denom)
                    H[i, 1] = (180 / np.pi) * (-dx / denom)
            return H

        # Measurement noise (adjusted dynamically)
        ekf.R = np.eye(len(measured_aoa)) * (6.0 ** 2)

        # Predict and update
        ekf.predict()
        ekf.update(measured_aoa, HJacobian, hx)

        x_ekf, y_ekf = ekf.x[0], ekf.x[1]

        th += 1
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], x_ekf, y_ekf])

    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_EKF", "Y_EKF"])

def local_imm_extended_kalman_filter(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 0) -> pd.DataFrame:
    """
    IMM-EKF for BLE tag localization based on AoA measurements.
    Uses three models:
      - Constant Velocity (CV): state = [x, y, vx, vy]
      - Constant Acceleration (CA): state = [x, y, vx, vy, ax, ay]
      - Stop Model: state = [x, y]
    The final position is computed as the weighted combination (based on likelihood) of each model’s estimate.
    
    Parameters:
        df (pd.DataFrame): DataFrame with ["X_Real", "Y_Real"] and each anchor's "Azimuth" info.
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        dt (int): Time interval in milliseconds.
        threshold (int): Minimum iteration count before saving results.
        
    Returns:
        pd.DataFrame: DataFrame with ["Time_Bucket", "X_Real", "Y_Real", "X_IMM", "Y_IMM"] columns.
    """
    print("Start the IMM-EKF Process")
    dt /= 1000  # Convert ms to seconds
    
    # --- Model definitions ---
    # CV Model: [x, y, vx, vy]
    def fx_cv(x):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        return F @ x

    def FJacobian_cv(x):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    # CA Model: [x, y, vx, vy, ax, ay]
    def fx_ca(x):
        dt2 = 0.5 * dt * dt
        F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1,  0],
            [0, 0, 0, 0, 0,  1]
        ])
        return F @ x

    def FJacobian_ca(x):
        dt2 = 0.5 * dt * dt
        return np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
    
    # Stop Model: [x, y]
    def fx_stop(x):
        return x  # No change

    def FJacobian_stop(x):
        return np.eye(2)
    
    # --- Initialize EKFs for each model ---
    ekf_cv   = ExtendedKalmanFilter(dim_x=4, dim_z=1)
    ekf_ca   = ExtendedKalmanFilter(dim_x=6, dim_z=1)
    ekf_stop = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    
    # 초기 상태 (동일한 시작점 사용)
    init_x = np.random.uniform(0, 1200)
    init_y = np.random.uniform(0, 600)
    ekf_cv.x   = np.array([init_x, init_y, 0.0, 0.0])
    ekf_ca.x   = np.array([init_x, init_y, 0.0, 0.0, 0.0, 0.0])
    ekf_stop.x = np.array([init_x, init_y])
    
    # 공분산 초기화
    ekf_cv.P   *= 500.0
    ekf_ca.P   *= 500.0
    ekf_stop.P *= 500.0
    
    # 프로세스 노이즈
    ekf_cv.Q   = np.diag([10**2, 10**2, 1**2, 1**2])
    ekf_ca.Q   = np.diag([10**2, 10**2, 1**2, 1**2, 0.1**2, 0.1**2])
    ekf_stop.Q = np.diag([10**2, 10**2])
    
    # 상태 전이 함수 설정
    ekf_cv.fx = lambda x: fx_cv(x)
    ekf_cv.FJacobian = lambda x: FJacobian_cv(x)
    ekf_ca.fx = lambda x: fx_ca(x)
    ekf_ca.FJacobian = lambda x: FJacobian_ca(x)
    ekf_stop.fx = lambda x: fx_stop(x)
    ekf_stop.FJacobian = lambda x: FJacobian_stop(x)
    
    # 모델들 및 초기 모델 확률 (동일 가중치)
    model_names = ['cv', 'ca', 'stop']
    ekfs = {'cv': ekf_cv, 'ca': ekf_ca, 'stop': ekf_stop}
    probs = np.array([1/3, 1/3, 1/3])
    
    estimated_positions = []
    th = 0

    # --- IMM-EKF Loop: 각 시간 스텝마다 측정 처리 ---
    for time_bucket, row in df.iterrows():
        measured_aoa = []
        valid_positions = []
        valid_orientations = []
        for aid in anchor_ids:
            az = row.get(f"{aid}_Azimuth", np.nan)
            if not pd.isna(az):
                measured_aoa.append(az)
                valid_positions.append(config["anchors"][aid]["position"])
                valid_orientations.append(config["anchors"][aid]["orientation"])

        if len(measured_aoa) < LEAST_ANCHORS:
            continue  # 충분한 앵커가 없으면 스킵

        measured_aoa = np.array(measured_aoa)
        anchors_position_np = np.array(valid_positions)
        anchors_orientation_rad = np.deg2rad(valid_orientations)
        
        # 공통 측정 함수: 상태 벡터의 첫 2개 값(위치)로부터 예측 각도 계산
        def hx(x):
            pos = x[:2]
            predicted_angles = []
            for (ax, ay), a_ori_rad in zip(anchors_position_np, anchors_orientation_rad):
                angle = np.arctan2(pos[0] - ax, pos[1] - ay) - a_ori_rad
                angle = (angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
                predicted_angles.append(np.degrees(angle))
            return np.array(predicted_angles)
        
        # 측정 함수의 자코비안 (상태 차원에 맞게 패딩)
        def HJacobian(x):
            n_anchors = len(anchors_position_np)
            # 2차원에 대한 기초 자코비안 계산
            H_base = np.zeros((n_anchors, 2))
            for i, (ax, ay) in enumerate(anchors_position_np):
                dx = x[0] - ax
                dy = x[1] - ay
                denom = dx**2 + dy**2
                if denom == 0:
                    H_base[i, :] = 0
                else:
                    H_base[i, 0] = (180 / np.pi) * (dy / denom)
                    H_base[i, 1] = (180 / np.pi) * (-dx / denom)
            # x의 길이에 맞춰 오른쪽에 0 패딩
            dim = len(x)
            H_full = np.hstack((H_base, np.zeros((n_anchors, dim - 2))))
            return H_full
        
        # 측정 노이즈: 앵커 개수에 따라 동적으로 결정
        R = np.eye(len(measured_aoa)) * (6.0 ** 2)
        for name in model_names:
            ekfs[name].R = R

        # Step 1: 각 모델별로 예측
        for name in model_names:
            ekfs[name].predict()
        
        # Step 2: 각 모델별로 업데이트 및 likelihood 계산
        likelihoods = []
        for name in model_names:
            try:
                ekfs[name].update(measured_aoa, HJacobian, hx)
                # innovation과 covariance를 이용한 likelihood 계산
                y = ekfs[name].y
                S = ekfs[name].S
                det_S = np.linalg.det(S)
                if det_S <= 0:
                    det_S = 1e-10
                inv_S = np.linalg.inv(S)
                likelihood = np.exp(-0.5 * (y.T @ inv_S @ y)) / np.sqrt((2 * np.pi)**len(y) * det_S)
            except Exception:
                likelihood = 1e-10
            likelihoods.append(likelihood)
        likelihoods = np.array(likelihoods)
        
        # Step 3: 모델 확률 갱신 (기존 확률과 likelihood 곱 후 정규화)
        probs = probs * likelihoods
        if probs.sum() == 0:
            probs = np.array([1/3, 1/3, 1/3])
        else:
            probs /= probs.sum()
        
        # Step 4: 최종 위치 추정: 각 모델의 추정 위치(x[0:2])의 가중평균
        pos_est = np.zeros(2)
        for i, name in enumerate(model_names):
            pos_est += probs[i] * ekfs[name].x[:2]
        
        th += 1
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], pos_est[0], pos_est[1]])
    
    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_IMM", "Y_IMM"])

def local_unscented_kalman_filter(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 0) -> pd.DataFrame:
    """
    Local Unscented Kalman Filter with variable number of valid AoA measurements per time step.
    
    Parameters:
        df (pd.DataFrame): DataFrame with ["X_Real", "Y_Real"] and each anchor's "Azimuth" information.
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        dt (int): Time interval (in milliseconds; converted to seconds internally).
    
    Returns:
        pd.DataFrame: DataFrame with ["Time_Bucket", "X_Real", "Y_Real", "X_UKF", "Y_UKF"] columns.
    """
    print("Start the Unscented Kalman Filter Process")
    
    dt /= 1000  # Convert ms to seconds

    # fx : State Transition Function (constant velocity)
    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        return F @ x

    # Sigma points generator
    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)

    # UKF instance (measurement dimension will vary later)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=dt, fx=fx, hx=None, points=points)

    # Initial state
    ukf.x = np.array([
        np.random.uniform(0, 1200),
        np.random.uniform(0, 600),
        0.0,
        0.0
    ])

    # Initial covariance
    ukf.P = np.eye(4) * 500.0

    # Process noise (position std = 10, velocity std = 1)
    ukf.Q = np.diag([10**2, 10**2, 1**2, 1**2])

    estimated_positions = []
    th = 0

    # Run the UKF
    for time_bucket, row in df.iterrows():
        measured_aoa = []
        valid_positions = []
        valid_orientations = []

        for aid in anchor_ids:
            az = row.get(f"{aid}_Azimuth", np.nan)
            if not pd.isna(az):
                measured_aoa.append(az)
                valid_positions.append(config["anchors"][aid]["position"])
                valid_orientations.append(config["anchors"][aid]["orientation"])

        if len(measured_aoa) < LEAST_ANCHORS:
            continue  # Require at least 2 valid anchors

        measured_aoa = np.array(measured_aoa)
        anchors_position_np = np.array(valid_positions)
        anchors_orientation_rad = np.deg2rad(valid_orientations)

        # Dynamically redefine hx and R
        def hx(x: np.ndarray) -> np.ndarray:
            pos = x[:2]
            predicted_angles = []
            for (ax, ay, _), a_ori_rad in zip(anchors_position_np, anchors_orientation_rad):
                angle = np.arctan2(pos[0] - ax, pos[1] - ay) - a_ori_rad
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                predicted_angles.append(np.degrees(angle))
            return np.array(predicted_angles)

        ukf.hx = hx
        ukf.R = np.eye(len(measured_aoa)) * (6.0 ** 2)  # Adjust R size to match valid anchors
        ukf.dim_z = len(measured_aoa)  # Update measurement dimension

        ukf.predict()
        ukf.update(measured_aoa)

        x_ukf, y_ukf = ukf.x[0], ukf.x[1]

        th += 1
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], x_ukf, y_ukf])

    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_UKF", "Y_UKF"])

def local_weighted_unscented_kalman_filter(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 0) -> pd.DataFrame:
    """
    Local Unscented Kalman Filter with measurement noise weighted by Azimuth_Var.
    """
    dt /= 1000  # ms → s

    # State transition (constant velocity)
    def fx(x, dt):
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]])
        return F @ x

    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=dt, fx=fx, hx=None, points=points)

    # init
    ukf.x = np.array([0,0,0,0])
    ukf.P = np.eye(4)*500
    ukf.Q = np.diag([10**2,10**2,1**2,1**2])

    est = []
    count = 0

    for tb, row in df.iterrows():
        meas, vars_, poses, oris = [], [], [], []
        for aid in anchor_ids:
            az  = row.get(f"{aid}_Azimuth",    np.nan)
            var = row.get(f"{aid}_Azimuth_Var", np.nan)
            if pd.isna(az) or pd.isna(var):
                continue
            meas.append(az)
            vars_.append(var)
            poses.append(config["anchors"][aid]["position"])
            oris.append(config["anchors"][aid]["orientation"])
        if len(meas) < LEAST_ANCHORS:
            continue

        meas = np.array(meas)
        poses = np.array(poses)
        oris_rad = np.deg2rad(oris)

        # 예측함수
        def hx(x):
            pos = x[:2]
            angs = []
            for (ax, ay, _), ori in zip(poses, oris_rad):
                a = np.arctan2(pos[0]-ax, pos[1]-ay) - ori
                angs.append(np.degrees((a+np.pi)%(2*np.pi)-np.pi))
            return np.array(angs)

        # Azimuth_Var → 에러 모델 → R diagonal
        # errs = 1.3259 * (np.array(vars_) ** 0.4069)
        # errs = 0.8490 * (np.array(vars_) ** 0.5 + 0.2458

        vars_arr = np.array(vars_)
        errs = np.where(
            vars_arr < 400,
            0.8514 * vars_arr ** 0.5 + 0.2294,
            0.0102 * vars_arr + 18.0772
        )
        R_diag = errs**1.1 # 왜 낮춰야 하지?

        ukf.hx     = hx
        ukf.dim_z  = len(meas)
        ukf.R      = np.diag(R_diag)

        ukf.predict()
        ukf.update(meas)

        x_u, y_u = ukf.x[0], ukf.x[1]
        count += 1
        if count > threshold:
            est.append([tb, row["X_Real"], row["Y_Real"], x_u, y_u])

    return pd.DataFrame(est, columns=["Time_Bucket","X_Real","Y_Real","X_WUKF","Y_WUKF"])


def local_particle_filter(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 5) -> pd.DataFrame:
    """
    Local Particle Filter with dynamic number of valid AoA measurements.

    Parameters:
        df (pd.DataFrame): DataFrame with ["X_Real", "Y_Real"] and each anchor's "Azimuth" information.
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        dt (int): Time interval (in milliseconds; converted to seconds internally).
        threshold (int): Threshold before collecting estimated results.

    Returns:
        pd.DataFrame: DataFrame with ["Time_Bucket", "X_Real", "Y_Real", "X_PF", "Y_PF"] columns.
    """
    print("Start the Particle Filter Process")

    dt /= 1000  # Convert ms to seconds

    # Initialize Particle Filter
    pf = ParticleFilter(num_particles=1000, state_bounds=(0, 1200, 0, 600), angle_noise_std=6., dt=dt)

    pos_noise_std = 10.0
    vel_noise_std = 1.0

    pf.initialize_particles()

    estimated_positions = []
    th = 0

    # Run the Particle Filter
    for time_bucket, row in df.iterrows():
        measured_aoa = []
        valid_positions = []
        valid_orientations = []

        for aid in anchor_ids:
            az = row.get(f"{aid}_Azimuth", np.nan)
            if not pd.isna(az):
                measured_aoa.append(az)
                valid_positions.append(config["anchors"][aid]["position"])
                valid_orientations.append(config["anchors"][aid]["orientation"])

        if len(measured_aoa) < LEAST_ANCHORS:
            continue  # Require at least 2 valid anchors

        measured_aoa = np.array(measured_aoa)
        anchors_position_np = np.array(valid_positions)
        anchors_orientation_np = np.array(valid_orientations)

        pf.predict(pos_noise_std, vel_noise_std)
        pf.update(measured_aoa, anchors_position_np, anchors_orientation_np)
        x_pf, y_pf = pf.estimate()

        th += 1
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], x_pf, y_pf])

    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_PF", "Y_PF"])
