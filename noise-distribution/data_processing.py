import pandas as pd

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