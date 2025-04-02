import math
import matplotlib.pyplot as plt

def visualize_all_anchors_with_heatmap(all_results: dict, gt_column: str, ms_column: str, cols:int = 2, vmin:int =None, vmax:int =None, title:str=None):
    '''
    Visualize the results of all anchors with a heatmap.

    Parameters:
        all_results (dict): Dictionary of results with anchor_id as key and DataFrame as value
        gt_column (str): Ground truth column name
        ms_column (str): Measurement column name
        cols (int): Number of columns in the plot
        vmin (int): Minimum value for the colormap
        vmax (int): Maximum value for the colormap

    Returns:
        None
    '''
    print(f"Visualizing {title}")
    rows = math.ceil(len(all_results) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()

    # Mean Error
    for idx, anchor_id in enumerate(all_results):
        ax = axes[idx]
        df = all_results[anchor_id]

        error_df = abs(df[ms_column] - df[gt_column])

        sc = ax.scatter(df["X_Real"], df["Y_Real"], c=error_df, cmap='coolwarm', s=100, vmin=vmin, vmax=vmax)

        # color bar
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("A_real - A_measurement")

        ax.set_xlabel("X")
        ax.set_xlim(0, 1200)
        ax.set_ylabel("Y")
        ax.set_ylim(0, 600)
        ax.set_title(f"Error map for {anchor_id}")
        ax.grid(True)

        # print the mean error and std
        mean_error = error_df.mean()
        std_error = error_df.std()
        print(f"Anchor {anchor_id} - Mean Error: {mean_error:.2f}, Std Error: {std_error:.2f}")

    plt.tight_layout()
    plt.show()