import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, gaussian_kde
from scipy.signal import find_peaks

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


def modified_z_score_filter(series: pd.Series, z_thresh: float = 3.5) -> pd.Series:
    med = np.median(series)
    mad = median_abs_deviation(series)

    if mad == 0:  # Avoid division by zero
        return series
    
    z_scores = 0.6745 * (series - med) / mad
    return series[np.abs(z_scores) <= z_thresh]

def mode_based_filter(series: pd.Series, mode_window: float = 20.0) -> pd.Series:
    if len(series) < 2:
        return series.iloc[0]
    
    if np.all(series == series.iloc[0]):
        return series.iloc[0]

    # Gaussian kernel density estimation to find the mode
    kde = gaussian_kde(series.dropna())
    x_vals = np.linspace(series.min(), series.max(), 1000)
    densities = kde(x_vals)
    mode_estimate = x_vals[np.argmax(densities)]

    # Filter based on the mode estimate
    return mode_estimate

def filter_by_primary_mode(data: pd.Series, bandwidth=1.0, threshold_std=1.5):
    if len(data) < 2:
        return data
    
    if np.all(data == data.iloc[0]):
        return data

    # KDE로 연속 확률 밀도 함수 추정
    kde = gaussian_kde(data, bw_method=bandwidth / data.std(ddof=1))
    x_vals = np.linspace(data.min(), data.max(), 1000)
    kde_vals = kde(x_vals)

    # 피크 검출
    peaks, _ = find_peaks(kde_vals)
    if len(peaks) == 0:
        raise ValueError("봉우리를 찾을 수 없습니다.")

    # 가장 높은 피크 선택
    primary_peak_idx = peaks[np.argmax(kde_vals[peaks])]
    primary_peak = x_vals[primary_peak_idx]

    # 주 피크 기준 범위 설정 (± threshold_std * 표준편차)
    std_dev = data.std()
    lower_bound = primary_peak - threshold_std * std_dev
    upper_bound = primary_peak + threshold_std * std_dev

    # 필터링
    return data[(data >= lower_bound) & (data <= upper_bound)]

def discretize_by_delta(df: pd.DataFrame, dt: int = 0, filter: bool = True) -> pd.DataFrame:
    """
    Generate a new DataFrame using time-bucketed (non-overlapping) windows.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with required columns
        delta (int): Time window size in milliseconds
        start_time (int | None): Start time for filtering (default is None)
        z_thresh (float): Z-score threshold for outlier removal
        filter (bool): Whether to apply outlier filtering

    Returns:
        pd.DataFrame: Aggregated DataFrame with computed sliding statistics
    """
    # Count the rows where Sequence is not ordered
    if not df["Sequence"].is_monotonic_increasing:
        raise ValueError("The Sequence column is not ordered. Please sort the DataFrame by Sequence before using this function.")


    df = df.sort_values("Timestamp").copy().reset_index(drop=True)
    output_rows = []

    # Determine time range
    start_time = df["Timestamp"].iloc[0]

    end_time = df["Timestamp"].iloc[-1]
    time_bucket = -1
    current_start = start_time

    # Time-bucketed calculation
    while current_start < end_time:
        time_bucket += 1
        current_end = current_start + dt
        window_df = df[(df["Timestamp"] >= current_start) & (df["Timestamp"] < current_end)]

        if len(window_df) < 2:
            current_start = current_end
            continue

        row_data = window_df.mean(numeric_only=True).to_dict()
        row_data["Time_Bucket"] = time_bucket
        row_data["Tag"] = window_df["Tag"].iloc[0]
        row_data["Anchor"] = window_df["Anchor"].iloc[0]
        row_data["Sequence"] = window_df["Sequence"].iloc[-1]
        row_data["Timestamp"] = current_end

        # Calculate the mean and variance for RSSI, Azimuth, and Elevation
        for col in ["RSSI", "Azimuth", "Elevation"]:
            if col in window_df.columns:
                row_data[f"{col}_Var"] = window_df[col].var()
                clean = modified_z_score_filter(window_df[col]) if filter else window_df[col]
                row_data[col] = clean.mean()

        output_rows.append(row_data)
        current_start = current_end

    return pd.DataFrame(output_rows)
