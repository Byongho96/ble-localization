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
    # error_df = error_df[(error_df['X_Real'] >= 300) & (error_df['X_Real'] <= 900) & (error_df['Y_Real'] >= 180) & (error_df['Y_Real'] <= 420)]
    error_df["X_Error"] = abs(error_df[x_ms_column] - error_df[x_gt_column])
    error_df["Y_Error"] = abs(error_df[y_ms_column] - error_df[y_gt_column])
    error_df["Distance_Error"] = (error_df["X_Error"]**2 + error_df["Y_Error"]**2)**0.5
    
    # Print the mean error and std
    mean_error = error_df["Distance_Error"].mean()
    std_error = error_df["Distance_Error"].std()
    print(f"Mean Error: {mean_error:.2f}, Std Error: {std_error:.2f}")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(error_df[x_ms_column], error_df[y_ms_column], c=error_df["Distance_Error"],
                     cmap='coolwarm', s=100, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc)
    cbar.set_label("Distance Error")

    # 🔽 추가: 추측 위치에서 실제 위치로 선분 그리기
    for _, row in error_df.iterrows():
        plt.plot([row[x_ms_column], row[x_gt_column]],
                 [row[y_ms_column], row[y_gt_column]],
                 color='gray', linestyle='--', linewidth=1)

    plt.xlabel("X")
    plt.xlim(0, 630)
    plt.ylabel("Y")
    plt.ylim(0, 630)
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_distance_error_with_heatmap_timebucket(df: pd.DataFrame, x_gt_column: str, y_gt_column: str, x_ms_column:str, y_ms_column:str, vmin:int =None, vmax:int =None, title:str=None):
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
    # error_df = error_df[(error_df['X_Real'] >= 300) & (error_df['X_Real'] <= 900) & (error_df['Y_Real'] >= 180) & (error_df['Y_Real'] <= 420)]
    error_df["X_Error"] = abs(error_df[x_ms_column] - error_df[x_gt_column])
    error_df["Y_Error"] = abs(error_df[y_ms_column] - error_df[y_gt_column])
    error_df["Distance_Error"] = (error_df["X_Error"]**2 + error_df["Y_Error"]**2)**0.5
    
    # Print the mean error and std
    mean_error = error_df["Distance_Error"].mean()
    std_error = error_df["Distance_Error"].std()
    print(f"Mean Error: {mean_error:.2f}, Std Error: {std_error:.2f}")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(error_df[x_ms_column], error_df[y_ms_column], c=error_df["Distance_Error"],
                     cmap='coolwarm', s=100, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc)
    cbar.set_label("Distance Error")

    # 🔽 추가: 추측 위치에서 실제 위치로 선분 그리기
    for _, row in error_df.iterrows():
        # 선분 그리기
        plt.plot([row[x_ms_column], row[x_gt_column]],
                 [row[y_ms_column], row[y_gt_column]],
                 color='gray', linestyle='--', linewidth=1)
        
        # Time_Bucket 텍스트 표시
        plt.text(row[x_ms_column], row[y_ms_column],
                 str(row["Time_Bucket"]), fontsize=8, ha='center', va='bottom', color='black')

    plt.xlabel("X")
    plt.xlim(0, 630)
    plt.ylabel("Y")
    plt.ylim(0, 630)
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.show()




def visualize_distance_error_with_gt_heatmap(df: pd.DataFrame, x_gt_column: str, y_gt_column: str, x_ms_column:str, y_ms_column:str, vmin:int =None, vmax:int =None, title:str=None):
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
    # error_df = error_df[(error_df['X_Real'] >= 300) & (error_df['X_Real'] <= 900) & (error_df['Y_Real'] >= 180) & (error_df['Y_Real'] <= 420)]
    error_df["X_Error"] = abs(error_df[x_ms_column] - error_df[x_gt_column])
    error_df["Y_Error"] = abs(error_df[y_ms_column] - error_df[y_gt_column])
    error_df["Distance_Error"] = (error_df["X_Error"]**2 + error_df["Y_Error"]**2)**0.5
    
    # Print the mean error and std
    mean_error = error_df["Distance_Error"].mean()
    std_error = error_df["Distance_Error"].std()
    print(f"Mean Error: {mean_error:.2f}, Std Error: {std_error:.2f}")

    plt.figure(figsize=(12, 6))
    sc = plt.scatter(error_df[x_gt_column], error_df[y_gt_column], c=error_df["Distance_Error"], cmap='coolwarm', s=100, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc)
    cbar.set_label("Distance Error")
    
    plt.xlabel("X")
    plt.xlim(0, 1200)
    plt.ylabel("Y")
    plt.ylim(0, 600)
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_mean_error(df: pd.DataFrame, x_gt_column: str, y_gt_column: str, x_ms_column:str, y_ms_column:str, vmin:int =None, vmax:int =None, title:str=None):
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
    
    error_df = df.copy()
    # error_df = error_df[(error_df['X_Real'] >= 300) & (error_df['X_Real'] <= 900) & (error_df['Y_Real'] >= 180) & (error_df['Y_Real'] <= 420)]
    error_df["X_Error"] = abs(error_df[x_ms_column] - error_df[x_gt_column])
    error_df["Y_Error"] = abs(error_df[y_ms_column] - error_df[y_gt_column])
    error_df["Distance_Error"] = (error_df["X_Error"]**2 + error_df["Y_Error"]**2)**0.5
    
    # Print the mean error and std
    mean_error = error_df["Distance_Error"].mean()
    std_error = error_df["Distance_Error"].std()
   
    return mean_error, std_error