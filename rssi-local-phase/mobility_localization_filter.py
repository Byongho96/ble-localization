import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.optimize import minimize

def rssi_to_distance(rssi, rssi_0, n):
    """RSSI → 거리 변환 모델"""
    return 10 ** ((rssi_0 - rssi) / (10 * n))

def trilateration_least_squares(positions, distances):
    """Least Squares를 이용한 2D 삼변측량"""
    def loss(pos):
        return np.sum((np.linalg.norm(pos - positions, axis=1) - distances) ** 2)
    
    result = minimize(loss, x0=np.mean(positions, axis=0))
    return result.x  # 추정된 [x, y] 위치

def estimate_n(anchor_positions, rssi_values, rssi_0):
    """Compatibility 기반으로 경로 손실 지수 n 추정"""
    def compatibility_error(n):
        distances = rssi_to_distance(rssi_values, rssi_0, n)
        
        def loss(pos):
            return np.sum((np.linalg.norm(pos - anchor_positions, axis=1) - distances) ** 2)
        
        res = minimize(loss, x0=np.mean(anchor_positions, axis=0))
        return loss(res.x)
    
    result = minimize(compatibility_error, x0=2.0, bounds=[(1.5, 5.0)])
    return result.x[0]

def local_adaptive_multilateration(df: pd.DataFrame, config: dict, anchor_ids: list) -> pd.DataFrame:
    """
    Least Squares Triangulation with Dynamic Path Loss Exponent Estimation

    각 시간 단위로, config에 정의된 앵커 정보와 실시간 RSSI를 이용하여
    동적으로 경로 손실 지수를 추정하고, 이를 바탕으로 앵커와의 거리를 계산하여
    최소제곱법(trilateration)으로 위치를 추정합니다.
    
    Parameters:
        df (pd.DataFrame): 인덱스는 Time_Bucket, 컬럼은
                           ["X_Real", "Y_Real", f"{anchor_id}_AnchorID", f"{anchor_id}_RSSI"]
        config (dict): 각 anchor_id에 대해 {"position": [x, y, z], "rssi_0": float, "n": float}
        anchor_ids (list): 사용될 앵커 ID 목록
    
    Returns:
        pd.DataFrame: ["Time_Bucket", "X_Real", "Y_Real", "X_LS", "Y_LS"] 컬럼을 갖는 결과 DataFrame 
    """
    results = []

    # 시간 단위(인덱스)를 그룹화하여 각 시간 프레임에 대해 처리
    for time_bucket, group in df.groupby(df.index):
        anchor_positions = []
        rssi_values = []
        rssi_0_val = np.mean([config[anchor_id]["rssi_0"] for anchor_id in anchor_ids if anchor_id in config])

        for anchor_id in anchor_ids:
            rssi_col = f"{anchor_id}_RSSI"
            if rssi_col in group.columns and pd.notna(group.iloc[0][rssi_col]):
                # config에 정의된 앵커 위치에서 x, y 좌표만 사용
                pos = np.array(config[anchor_id]["position"][:2])
                anchor_positions.append(pos)
                rssi_values.append(group.iloc[0][rssi_col])

        if len(anchor_positions) < 3:
            # 최소 3개의 앵커 정보가 필요함
            continue

        anchor_positions = np.array(anchor_positions)
        rssi_values = np.array(rssi_values)

        # Step 1: 동적 경로손실지수 n 추정 (Compatibility 기반 최적화)
        est_n = estimate_n(anchor_positions, rssi_values, rssi_0_val)

        # Step 2: 추정된 n값을 사용하여 각 앵커까지의 거리 계산
        distances = rssi_to_distance(rssi_values, rssi_0_val, est_n)

        # Step 3: 최소제곱법을 이용하여 2D trilateration 수행
        ls_position = trilateration_least_squares(anchor_positions, distances)

        results.append({
            "Time_Bucket": time_bucket,
            "X_Real": group.iloc[0]["X_Real"],
            "Y_Real": group.iloc[0]["Y_Real"],
            "X_LS": ls_position[0],
            "Y_LS": ls_position[1]
        })

    return pd.DataFrame(results)


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
    kf.P = np.diag([180, 180, 1, 1])
    
    # Process Noise Covariance
    kf.Q = np.diag([19, 19, 7, 7])

    # Measurement Noise Covariance
    kf.R = np.diag([75**2, 75**2])  

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