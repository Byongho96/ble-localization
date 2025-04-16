import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

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

def filter_with_position_ground_truth(gt_df: pd.DataFrame, ms_df: pd.DataFrame, offset:int = 0) -> pd.DataFrame:
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
        start_timestamp, end_timestamp, x, y = row
        
        mask = (ms_df["Timestamp"] >= start_timestamp + offset) & (ms_df["Timestamp"] <= end_timestamp - offset)
        filtered = ms_df.loc[mask].copy()
        filtered["X_Real"] = x
        filtered["Y_Real"] = y
        filtered_data.append(filtered)
        
    return pd.concat(filtered_data, ignore_index=True) if filtered_data else pd.DataFrame()


def calculate_rssi_and_distance(df: pd.DataFrame, position: list[float, float, float]) -> pd.DataFrame:
    '''
    Calculate the distance from the position and add it to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["X_Real", "Y_Real"]
        position (list): Position of the anchor [x, y]
    
    Returns:
        pd.DataFrame: DataFrame with new ["Distance", "RSSI"] columns
    '''

    df['Distance'] = np.sqrt((position[0] - df['X_Real'])**2 + (position[1] - df['Y_Real'])**2)
    df['RSSI'] = df[['1stP', '2ndP']].max(axis=1) # Use the maximum RSSI value

    # Remove outliers based on RSSI values
    def filter_group(group):
        q1 = group['RSSI'].quantile(0.25)
        q3 = group['RSSI'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return group[(group['RSSI'] >= lower_bound) & (group['RSSI'] <= upper_bound)]
    
    return df.groupby(['X_Real', 'Y_Real'], group_keys=False).apply(filter_group)

def filter_outlier_with_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    '''
    Filter outliers using the Interquartile Range (IQR) method.

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["RSSI"]
        column (str): Column name to filter outliers

    Returns:
        pd.DataFrame: DataFrame without outliers
    '''
    def iqr_filter(group):
        q1 = group["RSSI"].quantile(0.25)
        q3 = group["RSSI"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return group[(group["RSSI"] >= lower_bound) & (group["RSSI"] <= upper_bound)]

    cleaned_df = df.groupby(["X_Real", "Y_Real"], group_keys=False).apply(iqr_filter)
    return cleaned_df.reset_index(drop=True)

def calculate_log_distance_parameters(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the log-distance path loss parameters (rssi_0, n) using the RSSI and distance.

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["Distance", "RSSI"]
    
    Returns:
        tuple: Estimated RSSI at 1 meter (rssi_0) and path loss exponent (n)
    '''

    def log_distance_model(d, rssi_0, n):
        return rssi_0 - 10 * n * np.log10(d)

    # Fit model
    popt, _ = curve_fit(log_distance_model, df['Distance'], df['RSSI'])
    rssi_0_est, n_est = popt

    return rssi_0_est, n_est

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
    discretized_df["1stP_Std"] = df.groupby(["Time_Bucket"])["1stP"].std().reset_index(drop=True).fillna(0)
    discretized_df["2ndP_Std"] = df.groupby(["Time_Bucket"])["2ndP"].std().reset_index(drop=True).fillna(0)


    discretized_df["Timestamp"] = discretized_df["Time_Bucket"] + dt

    return discretized_df


def calculate_rssi_estimated_distance(df: pd.DataFrame, rssi_0: float, n: float) -> pd.DataFrame:
    '''
    Calculate the estimated distance from the RSSI and add it to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["RSSI"]
        rssi_0 (float): RSSI at 1 meter
        n (float): Path loss exponent

    Returns:
        pd.DataFrame: DataFrame with new ["RSSI_Distance"] column
    '''
    df['RSSI_Distance'] = 10 ** ((rssi_0 - df['RSSI']) / (10 * n))
    # print(df['RSSI_Distance'])
    return df