import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rice, rayleigh, kstest

def plot_aoa_distribution(df: pd.DataFrame, points: list[list[int, int]]):
    num_points = len(points)
    fig, axes = plt.subplots(1, num_points, figsize=(5 * num_points, 4), squeeze=False)
    
    for idx, (x, y) in enumerate(points):
        ax = axes[0, idx]
        # 해당 좌표에 해당하는 데이터 필터링
        subset = df[(df['X_Real'] == x) & (df['Y_Real'] == y)]
        
        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(f'Azimuth at ({x}, {y})')
            ax.axis('off')
            continue
        
        # Azimuth 값 시각화
        ax.hist(subset['Azimuth'], bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f'Azimuth at ({x}, {y})')
        ax.set_xlabel('Azimuth (°)')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def plot_rssi_distribution(df: pd.DataFrame, points: list[list[int, int]], channel: int = 37):
    df = df[df['Channel'] == channel]

    num_points = len(points)
    fig, axes = plt.subplots(1, num_points, figsize=(5 * num_points, 4), squeeze=False)
    
    for idx, (x, y) in enumerate(points):
        ax = axes[0, idx]
        # 해당 좌표에 해당하는 데이터 필터링
        subset = df[(df['X_Real'] == x) & (df['Y_Real'] == y)]
        
        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(f'2ndP at ({x}, {y})')
            ax.axis('off')
            continue
        
        # 2ndP 값 시각화
        ax.hist(subset['2ndP'], bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f'2ndP at ({x}, {y})')
        ax.set_xlabel('2ndP (°)')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def visualize_distance_error_with_heatmap(df: pd.DataFrame, x_gt_column: str, y_gt_column: str, error_column: str, vmin:int = None, vmax:int = None, title:str = None):
    '''
    Visualize distance error with a heatmap.

    Parameters:
        df (pd.DataFrame): DataFrame with ['X_Real', 'Y_Real', 'X_LS', 'Y_LS'] columns
        x_gt_column (str): Ground truth x column name
        y_gt_column (str): Ground truth y column name
        x_ms_column (str): Measurement x column name
        y_ms_column (str): Measurement y column name
        vmin (int): Minimum value for the colormap
        vmax (int): Maximum value for the colormap
    
    Returns:
        None
    '''
    # print(f"Visualizing {title}")
    
    error_df = df.copy()

    plt.figure(figsize=(12, 6))
    sc = plt.scatter(error_df[x_gt_column], error_df[y_gt_column], c=error_df[error_column], cmap='coolwarm', s=100, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc)
    cbar.set_label("Distance Error")
    
    plt.xlabel("X")
    plt.xlim(0, 1200)
    plt.ylabel("Y")
    plt.ylim(0, 600)
    # plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.show()
