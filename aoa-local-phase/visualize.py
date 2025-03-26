import math
import pandas as pd
import matplotlib.pyplot as plt

def visualize_distance_error_with_heatmap(df: pd.DataFrame, x_gt_column: str, y_gt_column: str, x_ms_column:str, y_ms_column:str, vmin:int =None, vmax:int =None, title:str=None):
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
    print(f"Visualizing {title}")
    
    error_df = df.copy()
    error_df["X_Error"] = abs(df[x_ms_column] - df[x_gt_column])
    error_df["Y_Error"] = abs(df[y_ms_column] - df[y_gt_column])
    error_df["Distance_Error"] = (error_df["X_Error"]**2 + error_df["Y_Error"]**2)**0.5
    
    # Print the mean error and std
    mean_error = error_df["Distance_Error"].mean()
    std_error = error_df["Distance_Error"].std()
    print(f"Mean Error: {mean_error:.2f}, Std Error: {std_error:.2f}")

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(df["X_Real"], df["Y_Real"], c=error_df["Distance_Error"], cmap='coolwarm', s=100, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc)
    cbar.set_label("Distance Error")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.show()
