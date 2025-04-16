import numpy as np
import pandas as pd

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
        'StartTimestamp': new_timestamps,
        'EndTimestamp': new_timestamps + dt,
        'X': new_x,
        'Y': new_y
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
        s_iso, start_timestamp, e_iso, end_timestamp, x, y, z = row
        
        mask = (ms_df["Timestamp"] >= start_timestamp + offset) & (ms_df["Timestamp"] <= end_timestamp - offset)
        filtered = ms_df.loc[mask].copy()
        filtered["X_Real"] = x
        filtered["Y_Real"] = y
        filtered["Z_Real"] = z 
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
    dz = result_df["Z_Real"] - position[2] if "Z_Real" in result_df.columns else 0

    # Calculate the azimuth in the real world
    azimuth_real = np.arctan2(dx, dy) - np.radians(orientation)
    azimuth_real = np.degrees((azimuth_real + np.pi) % (2 * np.pi) - np.pi)
    result_df["Azimuth_Real"] = azimuth_real

    # Calculate the elevation angle
    elevation_angle = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    elevation_angle = np.degrees(elevation_angle)
    result_df["Elevation_Real"] = elevation_angle

    return result_df

def discretize_by_delta(df: pd.DataFrame, dt: int = 0) -> pd.DataFrame:
    """
    Discretize the data by time intervals (dt).

    Parameters:
        df (pd.DataFrame): Input DataFrame with ['Timestamp', 'X_Real', 'Y_Real'] columns
        dt (int): Time step in the same unit as 'Timestamp'

    Returns:
        pd.DataFrame: Discretized DataFrame averaged by time bucket. ['Time_Bucket'] column is added.
    """
    if not dt:
        return df

    df = df.copy()

    # Create time buckets by discretizing the Timestamp column in dt intervals
    df["Time_Bucket"] = (df["Timestamp"] // dt) * dt

    # Compute mean for each unique (Time_Bucket) group
    discretized_df = df.groupby(["Time_Bucket"], as_index=False).mean(numeric_only=True)

    # Compute std for each unique (Time_Bucket) group
    discretized_df["Azimuth_Std"] = df.groupby(["Time_Bucket"])["Azimuth"].std().reset_index(drop=True).fillna(0)
    discretized_df["1stP_Std"] = df.groupby(["Time_Bucket"])["RSSI"].std().reset_index(drop=True).fillna(0)

    discretized_df["Timestamp"] = discretized_df["Time_Bucket"] + dt

    return discretized_df
