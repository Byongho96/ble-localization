import numpy as np
import pandas as pd

def filter_with_position_ground_truth(gt_df: pd.DataFrame, ms_df: pd.DataFrame) -> pd.DataFrame:
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
        
        mask = (ms_df["Timestamp"] >= start_timestamp) & (ms_df["Timestamp"] <= end_timestamp)
        filtered = ms_df.loc[mask].copy()
        filtered["X_Real"] = x
        filtered["Y_Real"] = y
        filtered_data.append(filtered)
        
    return pd.concat(filtered_data, ignore_index=True) if filtered_data else pd.DataFrame()

def calculate_aoa_ground_truth(df: pd.DataFrame, position: list[int, int, int], orientation: int) -> pd.DataFrame:
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
    azimuth_real = np.degrees(np.arctan2(dx, dy)) - orientation

    # Normalize angle to [-90, 90] range as implied by your original conditions
    azimuth_real = np.where(azimuth_real < -450, azimuth_real + 540, azimuth_real)
    azimuth_real = np.where(azimuth_real < -270, azimuth_real + 360, azimuth_real)
    azimuth_real = np.where(azimuth_real < -90,  azimuth_real + 180, azimuth_real)

    result_df["Azimuth_Real"] = azimuth_real

    return result_df

def discretize_grid_points_by_delta(df: pd.DataFrame, dt: int = 0) -> pd.DataFrame:
    """
    Discretize the data at each (X_Real, Y_Real) grid point by time intervals (dt).

    Parameters:
        df (pd.DataFrame): Input DataFrame with ['Timestamp', 'X_Real', 'Y_Real'] columns
        dt (int): Time step in the same unit as 'Timestamp'

    Returns:
        pd.DataFrame: Discretized DataFrame averaged by time bucket and grid location
    """
    if not dt:
        return df

    df = df.copy()

    # Create time buckets by discretizing the Timestamp column in dt intervals
    df["Time_Bucket"] = (df["Timestamp"] // dt) * dt

    # Compute mean for each unique (X_Real, Y_Real, Time_Bucket) group
    discretized_df = df.groupby(["X_Real", "Y_Real", "Time_Bucket"], as_index=False).mean(numeric_only=True)
    discretized_df["Timestamp"] = discretized_df["Time_Bucket"] + dt
    discretized_df = discretized_df.drop(columns=["Time_Bucket"])

    return discretized_df