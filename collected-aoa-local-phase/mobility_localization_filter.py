import math
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from ParticleFilter import ParticleFilter
from scipy.stats import chi2

LEAST_ANCHORS = 2

def aoa_np_error_model(vars_arr: np.ndarray) -> np.ndarray:
    return np.where(vars_arr < 400,
                      0.8514 * vars_arr ** 0.5 + 0.2294,
                      0.0102 * vars_arr + 25.0772)

def aoa_error_model(vars_arr: float) -> float:
    return 0.8514 * vars_arr ** 0.5 + 0.2294 if vars_arr < 400 else 0.0102 * vars_arr + 18.0772

def least_squares_triangulation_prev(df: pd.DataFrame, config: dict, anchor_ids: list) -> pd.DataFrame:
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
        A = []  # coefficient matrix for least squares
        b = []  # target vector for least squares
        valid_anchor_count = 0

        for anchor in anchor_ids:
            azimuth = row.get(f"{anchor}_Azimuth")

            # Skip this anchor if no AoA value is available
            if pd.isna(azimuth):
                continue

            # Get anchor position and orientation
            pos = np.array(config["anchors"][anchor]["position"][:2]) 
            orientation = config["anchors"][anchor]["orientation"]
            theta = math.radians(90 - azimuth - orientation)

            # Direction vector pointing from anchor in AoA direction
            d = np.array([math.cos(theta), math.sin(theta)])

            # Orthogonal projection matrix to the direction vector
            # This removes the component along the direction d
            P = np.eye(2) - np.outer(d, d)

            # Project the anchor position onto the plane orthogonal to the direction vector
            A.append(P)
            b.append(P @ pos.reshape(-1, 1))
            valid_anchor_count += 1

        if valid_anchor_count < LEAST_ANCHORS:
            continue

        A = np.vstack(A)  # shape (2n, 2)
        b = np.vstack(b)  # shape (2n, 1)

        try:
            # Solve least squares problem: minimize ||Ax - b||
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue

        x_LS, y_LS = x.flatten()
        x_real = row["X_Real"]
        y_real = row["Y_Real"]

        estimated_positions.append([time_bucket, x_real, y_real, x_LS, y_LS])
        cnt += 1

    print(cnt)
    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_LS", "Y_LS"])

def weighted_least_squares_triangulation_pre(df: pd.DataFrame, config: dict, anchor_ids: list) -> pd.DataFrame:
    """
    Weighted Least Squares Triangulation
    앵커별 Azimuth_Var 에 따라 가중치를 부여합니다.

    Returns:
        pd.DataFrame: ["Time_Bucket","X_Real","Y_Real","X_WLS","Y_WLS"]
    """
    est = []
    cnt = 0

    for time_bucket, row in df.iterrows():
        H, C, W_list = [], [], []
        valid = 0

        for anchor in anchor_ids:
            az = row.get(f"{anchor}_Azimuth")
            var = row.get(f"{anchor}_Azimuth_Var")

            if pd.isna(az) or pd.isna(var):
                continue

            pos = np.array(config["anchors"][anchor]["position"][:2])  # 2D position
            orientation = config["anchors"][anchor]["orientation"]

            aoa_rad = math.radians(90 - az - orientation)

            # H, C 항 추가
            H.append([-math.tan(aoa_rad), 1.0])
            C.append([pos[1] - pos[0] * math.tan(aoa_rad)])
    
            error = aoa_error_model(var)
            w = 1.0 / (error ** 2) if error > 0 else 0.0
            W_list.extend([w])  # 2D에서 각 P는 2행이므로 w를 2번

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
    return pd.DataFrame(est, columns=["Time_Bucket", "X_Real", "Y_Real", "X_WLS", "Y_WLS"])

def weighted_least_squares_triangulation(df: pd.DataFrame, config: dict, anchor_ids: list) -> pd.DataFrame:
    """
    Weighted Least Squares Triangulation
    앵커별 Azimuth_Var 에 따라 가중치를 부여합니다.

    Returns:
        pd.DataFrame: ["Time_Bucket","X_Real","Y_Real","X_LS","Y_LS"]
    """
    est = []
    cnt = 0

    print(df)

    for time_bucket, row in df.iterrows():
        A, b, W_list = [], [], []
        valid = 0

        for anchor in anchor_ids:
            az = row.get(f"{anchor}_Azimuth")
            var = row.get(f"{anchor}_Azimuth_Var")
            if pd.isna(az) or pd.isna(var):
                continue

            pos = np.array(config["anchors"][anchor]["position"][:2])  # 2D position
            ori = config["anchors"][anchor]["orientation"]
            theta = math.radians(90 - az - ori)

            d = np.array([math.cos(theta), math.sin(theta)])
            P = np.eye(2) - np.outer(d, d)

            A.append(P)
            b.append(P @ pos.reshape(-1, 1))

            # 가중치 계산 (오차의 역제곱)
            error = aoa_error_model(var)
            w = 1.0 / (error ** 2) if error > 0 else 0.0
            W_list.extend([w, w])  # 2D에서 각 P는 2행이므로 w를 2번

            valid += 1

        if valid < LEAST_ANCHORS:
            continue

        cnt += 1
        A = np.vstack(A)  # shape (2n, 2)
        b = np.vstack(b)  # shape (2n, 1)
        W = np.diag(W_list)  # shape (2n, 2n)

        try:
            x = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ b)
        except np.linalg.LinAlgError:
            continue

        x_ls, y_ls = float(x[0]), float(x[1])
        est.append([
            time_bucket,
            row["X_Real"],
            row["Y_Real"],
            x_ls,
            y_ls
        ])

    print(cnt)
    return pd.DataFrame(est, columns=["Time_Bucket", "X_Real", "Y_Real", "X_WLS", "Y_WLS"])

def local_2D_kalman_filter(df: pd.DataFrame, dt: int = 20) -> pd.DataFrame:
    '''
    Localization 2D Kalman Filter

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["X_WLS", "Y_WLS"]

    Returns:
        pd.DataFrame: DataFrame with new ["X_2D_KF", "Y_2D_KF"] column
    '''
    dt /= 1000  # ms -> sec

    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Initial state: [x, y, vx, vy]
    kf.x = np.array([df.iloc[0]["X_WLS"], df.iloc[0]["Y_WLS"],
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
    Mean Error : 78 -> 55
    Mean Std : 66 -> 46

    // 10, 1, 46
    // 5, 1, 22
    """
    # Initial Covariance Matrix
    kf.P = np.eye(4) * 500.0
    
    # Process Noise Covariance
    kf.Q = np.diag([5 ** 2, 5 ** 2, 1 ** 2, 1 ** 2])

    # Measurement Noise Covariance
    kf.R = np.diag([22 ** 2, 22 ** 2])  

    filtered_positions = []

    # Run the Kalman Filter    
    for _, row in df.iterrows():
        # Prediction step
        kf.predict()

        # Update step
        z = np.array([float(row["X_WLS"]), float(row["Y_WLS"])])
        kf.update(z)

        # Save the filtered position
        filtered_positions.append([float(kf.x[0]), float(kf.x[1])])

    df["X_2D_KF"], df["Y_2D_KF"] = zip(*filtered_positions)
    return df

def local_extended_kalman_filter_pre(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 0) -> pd.DataFrame:
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
        90,
        540,
        0.0,
        0.0
    ])

    """
    // 10, 1, 1.2
    // 13, 10, 1.4
    """
    # Initial covariance
    ekf.P *= 500.0

    # Process noise (pos_noise_std = 10, vel_noise_std = 1)
    ekf.Q = np.diag([13**2, 13**2, 10**2, 10**2])

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
        anchors_orientation_rad = np.deg2rad(valid_orientations)

        # Measurement function (variable-length)
        def hx(x):
            pos = x[:2]
            predicted_angles = []
            for (ax, ay, az), a_ori_rad in zip(anchors_position_np, anchors_orientation_rad):
                angle = np.arctan2(pos[0] - ax, pos[1] - ay) - a_ori_rad
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                predicted_angles.append(np.degrees(angle))
            return np.array(predicted_angles)

        # Jacobian of the measurement function
        def HJacobian(x):
            H = np.zeros((len(anchors_position_np), 4))
            for i, (ax, ay, az) in enumerate(anchors_position_np):
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
        # 측정값에 대한 오차 가중치 계산
        az_vars = np.array([row.get(f"{aid}_Azimuth_Var", np.nan) for aid in anchor_ids])
        az_vars = az_vars[~np.isnan(az_vars)]  # 유효한 분산만 추림

        if len(az_vars) != len(measured_aoa):
            continue  # 가중치와 측정값 개수가 맞지 않으면 스킵

        # 모델 기반으로 각도 오차 표준편차 계산
        az_std_devs = aoa_np_error_model(az_vars)  # 이 값은 degrees 단위 오차 (표준편차)

        # 공분산 행렬 R 설정 (각 측정값에 따른 오차 제곱)
        ekf.R = np.diag(az_std_devs ** 1.4)

        # Predict and update
        ekf.predict()
        ekf.update(measured_aoa, HJacobian, hx)

        x_ekf, y_ekf = ekf.x[0], ekf.x[1]

        th += 1
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], x_ekf, y_ekf])

    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_EKF", "Y_EKF"])


def local_extended_kalman_filter(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 0) -> pd.DataFrame:
    """
    Extended Kalman Filter with constant velocity model (state: [x, y, vx, vy]).
    Includes Mahalanobis gating. If gating fails, only prediction is used and
    threshold is relaxed based on accumulated dt.

    Parameters:
        df (pd.DataFrame): DataFrame with ["X_Real", "Y_Real"] and each anchor's "Azimuth" information.
        config (dict): Configuration containing anchor positions and orientations.
        anchor_ids (list): List of anchor IDs.
        dt (int): Time interval (in milliseconds; converted to seconds internally).
        threshold (int): Threshold for the number of iterations before saving results.

    Returns:
        pd.DataFrame: DataFrame with ["Time_Bucket", "X_Real", "Y_Real", "X_EKF", "Y_EKF"] columns.
    """
    print("Start the Extended Kalman Filter Process")

    dt_sec = dt / 1000  # Convert ms to seconds

    # State transition function (constant velocity)
    def fx(x, delta):
        F = np.array([
            [1, 0, delta, 0],
            [0, 1, 0, delta],
            [0, 0, 1,     0],
            [0, 0, 0,     1]
        ])
        return F @ x

    def FJacobian(x, delta):
        return np.array([
            [1, 0, delta, 0],
            [0, 1, 0, delta],
            [0, 0, 1,     0],
            [0, 0, 0,     1]
        ])

    # Initialize EKF
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=1)  # z dimension will vary

    ekf.x = np.array([540, 90, 0.0, 0.0])  # Initial state [x, y, vx, vy]
    ekf.P *= 500.0                         # Initial covariance
    ekf.Q = np.diag([13**2, 13**2, 10**2, 10**2])  # Process noise

    delta_accum = 0
    th = 0
    estimated_positions = []

    for time_bucket, row in df.iterrows():
        measured_aoa = []
        valid_positions = []
        valid_orientations = []
        az_vars = []

        for aid in anchor_ids:
            az = row.get(f"{aid}_Azimuth", np.nan)
            var = row.get(f"{aid}_Azimuth_Var", np.nan)
            if not pd.isna(az) and not pd.isna(var):
                measured_aoa.append(az)
                valid_positions.append(config["anchors"][aid]["position"])
                valid_orientations.append(config["anchors"][aid]["orientation"])
                az_vars.append(var)

        if len(measured_aoa) < LEAST_ANCHORS:
            continue

        measured_aoa = np.array(measured_aoa)
        az_vars = np.array(az_vars)
        anchors_position_np = np.array(valid_positions)
        anchors_orientation_rad = np.deg2rad(valid_orientations)

        # Define measurement function hx
        def hx(x):
            pos = x[:2]
            predicted_angles = []
            for (ax, ay, _), a_ori_rad in zip(anchors_position_np, anchors_orientation_rad):
                angle = np.arctan2(pos[0] - ax, pos[1] - ay) - a_ori_rad
                angle = (angle + np.pi) % (2 * np.pi) - np.pi  # normalize
                predicted_angles.append(np.degrees(angle))
            return np.array(predicted_angles)

        # Define Jacobian of hx
        def HJacobian(x):
            H = np.zeros((len(anchors_position_np), 4))
            for i, (ax, ay, _) in enumerate(anchors_position_np):
                dx = x[0] - ax
                dy = x[1] - ay
                denom = dx**2 + dy**2
                if denom == 0:
                    continue
                H[i, 0] = (180 / np.pi) * (dy / denom)
                H[i, 1] = (180 / np.pi) * (-dx / denom)
            return H

        # Measurement noise R
        az_std_devs = aoa_np_error_model(az_vars)  # returns std in degrees
        R = np.diag(az_std_devs ** 1.4)
        ekf.R = R

        # Predict step (with current delta)
        ekf.fx = lambda x: fx(x, dt_sec + delta_accum / 1000)
        ekf.FJacobian = lambda x: FJacobian(x, dt_sec + delta_accum / 1000)
        ekf.predict()

        # Compute Mahalanobis distance
        z = measured_aoa
        z_hat = hx(ekf.x)
        y = z - z_hat
        y = (y + 180) % 360 - 180  # wrap-around correction

        try:
            S = R
            D_M = np.sqrt(y.T @ np.linalg.inv(S) @ y)
            threshold_dynamic = 10 + (delta_accum / 1000) * 0.5  # dynamic gating
            print(f"Mahalanobis distance: {D_M}, threshold: {threshold_dynamic}")
            if D_M < threshold_dynamic:
                ekf.update(z, HJacobian, hx)
                delta_accum = 0
            else:
                delta_accum += dt
        except np.linalg.LinAlgError:
            delta_accum += dt

        x_ekf, y_ekf = ekf.x[0], ekf.x[1]
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], x_ekf, y_ekf])
        th += 1

    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_EKF", "Y_EKF"])

def local_unscented_kalman_filter_pre(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 0, i: int = 1, j: int =9, k:float= 1.0) -> pd.DataFrame:
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

    points = MerweScaledSigmaPoints(n=4, alpha=0.001, beta=2, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=dt, fx=fx, hx=None, points=points)

    """ 
    // 1, 7, 1.1
    // 14, 1, 1.2
    """
    ukf.x = np.array([
        90,
        540,
        0.0,
        0.0
    ])

    ukf.P = np.eye( 4 )*500
    ukf.Q = np.diag([i**2, i**2, j**2, j**2])

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

        vars_arr = np.array(vars_)
        errs = aoa_np_error_model(vars_arr)
        R_diag = errs**2
        ukf.hx     = hx
        ukf.dim_z  = len(meas)
        ukf.R      = np.diag(R_diag)

        ukf.predict()
        ukf.update(meas)

        # Mahalanobis distance filtering
        innovation = ukf.y
        S = ukf.S
        d = innovation.T @ np.linalg.inv(S) @ innovation
        maha_thresh = chi2.ppf(0.99, df=ukf.dim_z)

        if d < maha_thresh:
            x_u, y_u = ukf.x[0], ukf.x[1]
        else:
            x_pred = ukf.x_prior
            x_u, y_u = x_pred[0], x_pred[1]

        x_u, y_u = ukf.x[0], ukf.x[1]
        count += 1
        if count > threshold:
            est.append([tb, row["X_Real"], row["Y_Real"], x_u, y_u])

    return pd.DataFrame(est, columns=["Time_Bucket","X_Real","Y_Real","X_UKF","Y_UKF"])

def local_unscented_kalman_filter(df: pd.DataFrame, config: dict, anchor_ids: list, dt: int = 20, threshold: int = 0, i: int = 8, j: int =3, k:float= 1)  -> pd.DataFrame:
    """
    Local Unscented Kalman Filter with measurement noise weighted by Azimuth_Var.
    """
    dt /= 1000  # ms → s

    # State transition (constant velocity along heading)
    def fx(x, dt):
        x_new = x.copy()
        v = x[2]
        theta = x[3]
        x_new[0] += v * np.cos(theta) * dt
        x_new[1] += v * np.sin(theta) * dt
        # v, theta remain unchanged
        return x_new

    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=dt, fx=fx, hx=None, points=points)

    """
    4, 3, 1.0
    8, 3, 1.0
    """
    ukf.x = np.array([90, 540, 0.1, 0.0])  # small initial velocity
    ukf.P = np.eye(4) * 500
    ukf.Q = np.diag([i**2, i**2, j**2, k**2])

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

        # Measurement model
        def hx(x):
            pos = x[:2]
            angs = []
            for (ax, ay, _), ori in zip(poses, oris_rad):
                a = np.arctan2(pos[0] - ax, pos[1] - ay) - ori
                angs.append(np.degrees((a + np.pi) % (2 * np.pi) - np.pi))
            return np.array(angs)

        # Azimuth_Var → error std (in degrees)
        vars_arr = np.array(vars_)
        errs = aoa_np_error_model(vars_arr)
        R_diag = errs ** 1.0

        ukf.hx = hx
        ukf.dim_z = len(meas)
        ukf.R = np.diag(R_diag)

        ukf.predict()
        ukf.update(meas)

        x_u, y_u = ukf.x[0], ukf.x[1]
        count += 1
        if count > threshold:
            est.append([tb, row["X_Real"], row["Y_Real"], x_u, y_u])

    return pd.DataFrame(est, columns=["Time_Bucket", "X_Real", "Y_Real", "X_UKF", "Y_UKF"])



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

    dt /= 1000
    pf = ParticleFilter(num_particles=1000, state_bounds=(0, 600, 0, 600), dt=dt, angle_noise_std=5)

    pf.initialize_particles()

    estimated_positions = []
    th = 0

    for time_bucket, row in df.iterrows():
        measured_aoa = []
        valid_positions = []
        valid_orientations = []
        aoa_vars = []

        for aid in anchor_ids:
            az = row.get(f"{aid}_Azimuth", np.nan)
            az_var = row.get(f"{aid}_Azimuth_Var", np.nan)
            if not pd.isna(az) and not pd.isna(az_var):
                measured_aoa.append(az)
                valid_positions.append(config["anchors"][aid]["position"])
                valid_orientations.append(config["anchors"][aid]["orientation"])
                aoa_vars.append(az_var)

        if len(measured_aoa) < LEAST_ANCHORS:
            continue

        measured_aoa = np.array(measured_aoa)
        anchors_position_np = np.array(valid_positions)
        anchors_orientation_np = np.array(valid_orientations)
        aoa_vars = np.array(aoa_vars)
        aoa_errors = aoa_np_error_model(aoa_vars)

        pf.predict(10, 2)
        pf.update(measured_aoa, anchors_position_np, anchors_orientation_np, aoa_errors)
        x_pf, y_pf = pf.estimate()

        th += 1
        if th > threshold:
            estimated_positions.append([time_bucket, row["X_Real"], row["Y_Real"], x_pf, y_pf])

    return pd.DataFrame(estimated_positions, columns=["Time_Bucket", "X_Real", "Y_Real", "X_PF", "Y_PF"])
