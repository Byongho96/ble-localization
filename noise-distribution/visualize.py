import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rice, rayleigh, kstest

def plot_distribution(df: pd.DataFrame, points: list[list[int, int]], column_name: str):
    num_points = len(points)
    fig, axes = plt.subplots(1, num_points, figsize=(5 * num_points, 4), squeeze=False)
    
    for idx, (x, y) in enumerate(points):
        ax = axes[0, idx]

        # Filter the point
        subset = df[(df['X_Real'] == x) & (df['Y_Real'] == y)]
        
        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(f'{column_name} at ({x}, {y})')
            ax.axis('off')
            continue
        
        # column_name 값 시각화
        ax.hist(subset[column_name], bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f'{column_name} at ({x}, {y})')
        ax.set_xlabel(f'{column_name}')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def visualize_all_anchors_with_heatmap(all_results: dict, column_name: str, cols:int = 4, vmin:int =None, vmax:int =None):
    '''
    Visualize the results of all anchors with a heatmap.

    Parameters:
        all_results (dict): Dictionary of results with anchor_id as key and DataFrame as value
        gt_column (str): Ground truth column name
        ms_column (str): Measurement column name
        cols (int): Number of columns in the plot

        

    Returns:
        None
    '''
    print(f"Visualizing {column_name}")
    rows = math.ceil(len(all_results) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(30, 5))
    axes = axes.flatten()

    # Mean Error
    for idx, anchor_id in enumerate(all_results):
        ax = axes[idx]
        df = all_results[anchor_id]

        sc = ax.scatter(df["X_Real"], df["Y_Real"], c=df[column_name], cmap='coolwarm', s=100, vmin=vmin, vmax=vmax)

        # color bar
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f"{column_name}")

        ax.set_xlabel("X")
        ax.set_xlim(0, 1200)
        ax.set_ylabel("Y")
        ax.set_ylim(0, 600)
        ax.set_title(f"{column_name} map for {anchor_id}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()