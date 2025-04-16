import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def filter_with_position_ground_truth(gt_df: pd.DataFrame, ms_df: pd.DataFrame, offset: int) -> pd.DataFrame:
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

    return df

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
    discretized_df["Timestamp"] = discretized_df["Time_Bucket"] + dt

    return discretized_df


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