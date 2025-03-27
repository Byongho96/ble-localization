import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

def aoa_moving_average_filter(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    '''
    Moving Average Filter

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["Azimuth"]
        window_size (int): Size of the moving average window
    
    Returns:
        pd.DataFrame: DataFrame with new ["Azimuth_MAF"] column
    '''
    filtered_azimuths = df["Azimuth"].rolling(window=window_size).mean()
    df["Azimuth_MAF"] = filtered_azimuths
    return df


def aoa_median_filter(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    '''
    Median Filter
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with ["Azimuth"]
        window_size (int): Size of the median filter window
    
    Returns:
        pd.DataFrame: DataFrame with new ["Azimuth_Median"] column
    '''
    filtered_azimuths = df["Azimuth"].rolling(window=window_size).median()
    df["Azimuth_Median"] = filtered_azimuths
    return df

def aoa_low_pass_filter(df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    '''
    Low Pass Filter
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with ["Azimuth"]
        alpha (float): Alpha value for the low pass filter
        
    Returns:
        pd.DataFrame: DataFrame with new ["Azimuth_LowPass"] column
    '''
    filtered_azimuths = [df.iloc[0]["Azimuth"]]

    for i in range(1, len(df)):
        filtered = alpha * df.iloc[i]["Azimuth"] + (1 - alpha) * filtered_azimuths[-1]
        filtered_azimuths.append(filtered)

    df["Azimuth_LowPass"] = filtered_azimuths
    return df

def aoa_1d_kalman_filter(df: pd.DataFrame, dt: float = 0.02) -> pd.DataFrame:
    '''
    AoA 1D Kalman Filter

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["Azimuth"]

    Returns:
        pd.DataFrame: DataFrame with new ["Azimuth_1d_KF"] column
    '''
    kf = KalmanFilter(dim_x=1, dim_z=1)

    kf.x = np.array([float(df.iloc[0]['Azimuth'])])  # [azimuth]
    kf.F = np.array([1])
    kf.H = np.array([1])
    kf.P *= 1000.
    kf.R *= 5  # Measurement noise : [U-blox C211 5 degree]
    kf.Q *= dt  # Process noise

    filtered_azimuths = []

    for _, row in df.iterrows():
        # Prediction step
        kf.predict()

        # Update step
        z = float(row["Azimuth"])
        kf.update(z)

        # Save the filtered azimuth
        filtered_azimuths.append(float(kf.x))

    df["Azimuth_1d_KF"] = filtered_azimuths
    return df

def aoa_2d_kalman_filter(df: pd.DataFrame, dt: int = 0.02) -> pd.DataFrame:
    '''
    AoA 2D Kalman Filter

    Parameters:
        df (pd.DataFrame): Input DataFrame with ["Azimuth"]

    Returns:
        pd.DataFrame: DataFrame with new ["Azimuth_2d_KF"] column
    '''
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([float(df.iloc[0]['Azimuth']), 0.])  # [azimuth, azimuth_rate]
    kf.F = np.array([[1, dt],[0, 1]])
    kf.H = np.array([[1, 0]])
    kf.P *= 1000.
    kf.R *= 5  # Measurement noise : [U-blox C211 5 degree]
    kf.Q *= dt  # Process noise proportional to the dt

    # Run the Kalman Filter
    filtered_azimuths = []

    for _, row in df.iterrows():
        # Prediction step
        kf.predict()

        # Update step
        z = float(row["Azimuth"])
        kf.update(z)

        # Save the filtered azimuth
        filtered_azimuths.append(float(kf.x[0]))

    df["Azimuth_2d_KF"] = filtered_azimuths
    return df
