import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

def rssi_moving_average_filter(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    '''
    Moving Average Filter

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["RSSI_Distance"]
        window_size (int): Size of the moving average window
    
    Returns:
        pd.DataFrame: DataFrame with new ["RSSI_Distance_MAF"] column
    '''
    filtered_RSSI_Distances = df["RSSI_Distance"].rolling(window=window_size).mean()
    df["RSSI_Distance_MAF"] = filtered_RSSI_Distances
    return df


def rssi_median_filter(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    '''
    Median Filter
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with ["RSSI_Distance"]
        window_size (int): Size of the median filter window
    
    Returns:
        pd.DataFrame: DataFrame with new ["RSSI_Distance_Median"] column
    '''
    filtered_RSSI_Distances = df["RSSI_Distance"].rolling(window=window_size).median()
    df["RSSI_Distance_Median"] = filtered_RSSI_Distances
    return df

def rssi_low_pass_filter(df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    '''
    Low Pass Filter
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with ["RSSI_Distance"]
        alpha (float): Alpha value for the low pass filter
        
    Returns:
        pd.DataFrame: DataFrame with new ["RSSI_Distance_LowPass"] column
    '''
    filtered_RSSI_Distances = [df.iloc[0]["RSSI_Distance"]]

    for i in range(1, len(df)):
        filtered = alpha * df.iloc[i]["RSSI_Distance"] + (1 - alpha) * filtered_RSSI_Distances[-1]
        filtered_RSSI_Distances.append(filtered)

    df["RSSI_Distance_LowPass"] = filtered_RSSI_Distances
    return df

def rssi_1d_kalman_filter(df: pd.DataFrame, dt: float = 0.02) -> pd.DataFrame:
    '''
    AoA 1D Kalman Filter

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["RSSI_Distance"]

    Returns:
        pd.DataFrame: DataFrame with new ["RSSI_Distance_1d_KF"] column
    '''
    kf = KalmanFilter(dim_x=1, dim_z=1)

    kf.x = np.array([float(df.iloc[0]['RSSI_Distance'])])  # [RSSI_Distance]
    kf.F = np.array([1])
    kf.H = np.array([1])
    kf.P *= 1000.
    kf.R *= 5  # Measurement noise : [U-blox C211 5 degree]
    kf.Q *= dt  # Process noise

    filtered_RSSI_Distances = []

    for _, row in df.iterrows():
        # Prediction step
        kf.predict()

        # Update step
        z = float(row["RSSI_Distance"])
        kf.update(z)

        # Save the filtered RSSI_Distance
        filtered_RSSI_Distances.append(float(kf.x))

    df["RSSI_Distance_1d_KF"] = filtered_RSSI_Distances
    return df

def rssi_2d_kalman_filter(df: pd.DataFrame, dt: int = 0.02) -> pd.DataFrame:
    '''
    AoA 2D Kalman Filter

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["RSSI_Distance"]

    Returns:
        pd.DataFrame: DataFrame with new ["RSSI_Distance_2d_KF"] column
    '''
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([float(df.iloc[0]['RSSI_Distance']), 0.])  # [RSSI_Distance, RSSI_Distance_rate]
    kf.F = np.array([[1, dt],[0, 1]])
    kf.H = np.array([[1, 0]])
    kf.P *= 1000.
    kf.R *= 5  # Measurement noise : [U-blox C211 5 degree]
    kf.Q *= dt  # Process noise proportional to the dt

    # Run the Kalman Filter
    filtered_RSSI_Distances = []

    for _, row in df.iterrows():
        # Prediction step
        kf.predict()

        # Update step
        z = float(row["RSSI_Distance"])
        kf.update(z)

        # Save the filtered RSSI_Distance
        filtered_RSSI_Distances.append(float(kf.x[0]))

    df["RSSI_Distance_2d_KF"] = filtered_RSSI_Distances
    return df
