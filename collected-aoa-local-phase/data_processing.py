import numpy as np
import pandas as pd
from scipy.stats import zscore
def interpolate_ground_truth(gt_df: pd.DataFrame, dt: int) -> pd.DataFrame:
    """
    Interpolates ground truth (gt_df) at fixed dt intervals.
    - If EndTimestamp is present, keep the position constant (stationary).
    - If EndTimestamp is NaN, linearly interpolate to the next point.

    Parameters:
        gt_df (pd.DataFrame): DataFrame with ['StartTimestamp', 'EndTimestamp', 'X', 'Y']
        dt (int): Time step in the same unit as 'StartTimestamp'

    Returns:
        pd.DataFrame: Interpolated DataFrame with ['StartTimestamp', 'EndTimestamp', 'X', 'Y']
    """
    times, x_vals, y_vals = [], [], []

    for _, row in gt_df.iterrows():
        times.append(row['StartTimestamp'])
        x_vals.append(row['X'])
        y_vals.append(row['Y'])
        if pd.notna(row['EndTimestamp']):
            times.append(row['EndTimestamp'])
            x_vals.append(row['X'])
            y_vals.append(row['Y'])

    # Sort by time
    times = np.array(times)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    sort_idx = np.argsort(times)
    times, x_vals, y_vals = times[sort_idx], x_vals[sort_idx], y_vals[sort_idx]

    # Generate new timestamps
    new_timestamps = np.arange(times[0], times[-1] + dt, dt)
    new_x = np.interp(new_timestamps, times, x_vals)
    new_y = np.interp(new_timestamps, times, y_vals)

    return pd.DataFrame({
        'StartTimeISO': pd.to_datetime(new_timestamps, unit='ms').astype(str),
        'StartTimestamp': new_timestamps,
        'EndTimeISO': pd.to_datetime(new_timestamps + dt, unit='ms').astype(str),
        'EndTimestamp': new_timestamps + dt,
        'X': new_x,
        'Y': new_y,
        'Z': 140
    })

def filter_with_position_ground_truth(gt_df: pd.DataFrame, ms_df: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    '''
    Filter the measurement data by the ground truth data.

    Parameters:
        gt_df (pd.DataFrame): Ground truth data with ["StartTimestamp", "EndTimestamp", "X", "Y"]
        ms_df (pd.DataFrame): Measurement data with ["StartTimestamp"]

    Returns:
        pd.DataFrame: Measurement data matched with ground truth and annotated with real positions ["X_Real", "Y_Real"]
    '''
    filtered_data = []

    for row in gt_df.itertuples(index=False):
        start_iso, start_timestamp, end_iso, end_timestamp, x, y, z = row
        mask = (ms_df["Timestamp"] >= start_timestamp + offset) & (ms_df["Timestamp"] <= end_timestamp - offset)
        filtered = ms_df.loc[mask].copy()
        filtered["X_Real"] = x
        filtered["Y_Real"] = y
        filtered_data.append(filtered)
        
    return pd.concat(filtered_data, ignore_index=True) if filtered_data else pd.DataFrame()

def calculate_aoa_ground_truth(df: pd.DataFrame, position: list[float, float, float], orientation: float) -> pd.DataFrame:
    '''
    Calculate the real azimuth from the ground truth data and add it to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["X_Real", "Y_Real"]
        position (list[int, int, int]): [x, y, z] position of the anchor
        orientation (int): Orientation of the reference in degrees

    Returns:
        pd.DataFrame: DataFrame with new ["Azimuth_Real"] column
    '''
    result_df = df.copy()

    # Calculate the azimuth from the ground truth data
    dx = result_df["X_Real"] - position[0] 
    dy = result_df["Y_Real"] - position[1]

    # Calculate the azimuth in the real world
    azimuth_real = np.arctan2(dx, dy) - np.radians(orientation)
    azimuth_real = np.degrees((azimuth_real + np.pi) % (2 * np.pi) - np.pi)
    result_df["Azimuth_Real"] = azimuth_real

    return result_df

def discretize_by_delta(df: pd.DataFrame, min_ts, dt: int = 0) -> pd.DataFrame:
    """
    Discretize the data by time intervals (dt).

    Parameters:
        df (pd.DataFrame): Input DataFrame with ['Timestamp', 'X_Real', 'Y_Real'] columns
        dt (int): Time step in the same unit as 'Timestamp'

    Returns:
        pd.DataFrame: Discretized DataFrame averaged by time bucket. ['Time_Bucket'] column is added.
    """
    if not dt:
        return df.copy()

    df = df.copy()
    # min_ts = df["Timestamp"].min()
    df["Time_Bucket"] = ((df["Timestamp"] - min_ts) // dt).astype(int)

    # 1) 평균
    mean_df = (
        df.groupby("Time_Bucket", as_index=False)
          .mean(numeric_only=True)
    )

    # 2) Azimuth 분산
    var_df = (
        df.groupby("Time_Bucket")["Azimuth"]
          .var()
          .reset_index(name="Azimuth_Var")
    )

    # 3) RSSI 분산
    var_df["RSSI_Var"] = (
        df.groupby("Time_Bucket")["RSSI"]
          .var()
          .reset_index(name="RSSI_Var")["RSSI_Var"]
    )

    # 3) 합치고 Timestamp 갱신
    discretized_df = mean_df.merge(var_df, on="Time_Bucket")
    discretized_df["Timestamp"] = discretized_df["Time_Bucket"] + dt

    return discretized_df

def sliding_window_average(df: pd.DataFrame, dt: int = 200, stride: int = 500) -> pd.DataFrame:
    """
    Apply sliding window with z-score-based outlier removal and compute stats.

    Parameters:
        df (pd.DataFrame): Must include ['Timestamp', 'Azimuth', 'Azimuth_Real', 'RSSI']
        dt (int): Window size in milliseconds
        stride (int): Step size in milliseconds

    Returns:
        pd.DataFrame: Aggregated statistics per sliding window
    """
    df = df.copy()
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    rows = []
    cnt = 0

    for i in range(len(df)):
        now = df.loc[i, "Timestamp"]
        window = df[(df["Timestamp"] >= now - dt) & (df["Timestamp"] <= now)]

        if len(window) >= 3:
            z = zscore(window[["Azimuth"]], nan_policy='omit')
            mask = (np.abs(z) < 3).all(axis=1)
            filtered = window[mask]

            if not filtered.empty:
                rows.append({
                    "Time_Bucket": cnt,
                    "Timestamp": now,
                    "Azimuth": filtered["Azimuth"].mean(),
                    "Azimuth_Var": filtered["Azimuth"].var(ddof=1),
                    "X_Real": filtered["X_Real"].mean(),
                    "Y_Real": filtered["Y_Real"].mean(),
                    "RSSI": filtered["RSSI"].mean(),
                    "RSSI_Var": filtered["RSSI"].var(ddof=1),
                })
                cnt += 1

    return pd.DataFrame(rows)

def lowpass_filter(df: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
    """
    Apply a low-pass filter (exponential moving average) to Azimuth and RSSI.

    Parameters:
        df (pd.DataFrame): Must include ['Timestamp', 'Azimuth', 'Azimuth_Real', 'RSSI', 'X_Real', 'Y_Real']
        alpha (float): Smoothing factor between 0 and 1

    Returns:
        pd.DataFrame: Filtered stats per row with Time_Bucket
    """
    df = df.copy()
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    rows = []
    cnt = 0

    azimuth_ema = None
    rssi_ema = None

    for i in range(len(df)):
        now = df.loc[i, "Timestamp"]
        az = df.loc[i, "Azimuth"]
        rssi = df.loc[i, "RSSI"]

        # EMA 초기화
        if azimuth_ema is None:
            azimuth_ema = az
            rssi_ema = rssi
        else:
            azimuth_ema = alpha * az + (1 - alpha) * azimuth_ema
            rssi_ema = alpha * rssi + (1 - alpha) * rssi_ema

        rows.append({
            "Time_Bucket": cnt,
            "Timestamp": now,
            "Azimuth": azimuth_ema,
            "Azimuth_Var": np.nan,  # EMA는 분산 없음
            "X_Real": df.loc[i, "X_Real"],
            "Y_Real": df.loc[i, "Y_Real"],
            "RSSI": rssi_ema,
            "RSSI_Var": np.nan,
        })
        cnt += 1

    return pd.DataFrame(rows)

def prepare_merged_dataframe(dic: dict) -> pd.DataFrame:
    """
    Merge multiple DataFrames into a single DataFrame by Time_Bucket.

    Parameters:
        dic (dict): Dictionary containing multiple DataFrames with Time_Bucket columns

    Returns:
        pd.DataFrame: Merged DataFrame with prefix added to columns
    """
    dfs = []
    for i, (anchor_id, df) in enumerate(dic.items()):

        df_temp = df.copy().set_index("Time_Bucket")
        if i == 0:
            # For the base anchor, keep "X_Real" and "Y_Real" columns unchanged,
            # and add prefix to the rest of the columns.
            non_xy = [col for col in df_temp.columns if col not in ["X_Real", "Y_Real"]]
            df_prefixed = df_temp[non_xy].add_prefix(f"{anchor_id}_")
            df_temp = pd.concat([df_temp[["X_Real", "Y_Real"]], df_prefixed], axis=1)
        else:
            # For other anchors, drop "X_Real" and "Y_Real" (to avoid duplicates),
            # and add prefix to all remaining columns.
            df_temp = df_temp.drop(columns=["X_Real", "Y_Real"], errors="ignore")
            df_temp = df_temp.add_prefix(f"{anchor_id}_")
        dfs.append(df_temp)
        
    # Merge the DataFrames by Time_Bucket
    merged_df = pd.concat(dfs, axis=1, join="outer")
    merged_df = merged_df.sort_index()
    return merged_df