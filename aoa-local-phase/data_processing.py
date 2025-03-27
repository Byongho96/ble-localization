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

def discretize_grid_points_by_delta(df: pd.DataFrame, dt: int = 0) -> pd.DataFrame:
    """
    Discretize the data at each (X_Real, Y_Real) grid point by time intervals (dt).

    Parameters:
        df (pd.DataFrame): Input DataFrame with ['Timestamp', 'X_Real', 'Y_Real'] columns
        dt (int): Time step in the same unit as 'Timestamp'

    Returns:
        pd.DataFrame: Discretized DataFrame averaged by time bucket and grid location. ['Time_Bucket'] column is added.
    """
    if not dt:
        return df

    df = df.copy()

    # Create time buckets by discretizing the Timestamp column in dt intervals
    df["Time_Bucket"] = (df["Timestamp"] // dt) * dt

    # Compute mean for each unique (X_Real, Y_Real, Time_Bucket) group
    discretized_df = df.groupby(["X_Real", "Y_Real", "Time_Bucket"], as_index=False).mean(numeric_only=True)
    discretized_df["Timestamp"] = discretized_df["Time_Bucket"] + dt

    return discretized_df


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
    merged_df = pd.concat(dfs, axis=1, join="inner")
    return merged_df