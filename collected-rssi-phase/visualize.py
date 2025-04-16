import math
import matplotlib.pyplot as plt
import pandas as pd

def visualize_all_anchors_with_heatmap(all_results: dict, gt_column: str, ms_column: str, cols: int = 2, vmin: int = None, vmax: int = None, title: str = None):
    '''
    Visualize mean absolute error per (X_Real, Y_Real) grid point for each anchor using heatmap.

    Parameters:
        all_results (dict): Dictionary of anchor_id -> DataFrame
        gt_column (str): Ground truth column name
        ms_column (str): Measurement column name
        cols (int): Number of columns in subplot layout
        vmin (int): Minimum value for colormap (optional)
        vmax (int): Maximum value for colormap (optional)
        title (str): Title to print (optional)

    Returns:
        None
    '''
    if title:
        print(f"Visualizing {title}")

    rows = math.ceil(len(all_results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
    axes = axes.flatten()

    for idx, (anchor_id, df) in enumerate(all_results.items()):
        ax = axes[idx]

        # Compute absolute error
        df["AbsError"] = (df[ms_column] - df[gt_column]).abs()

        # Group by position and compute mean error
        grouped = df.groupby(["X_Real", "Y_Real"])["AbsError"].mean().reset_index()

        # Scatter plot
        sc = ax.scatter(grouped["X_Real"], grouped["Y_Real"], c=grouped["AbsError"],
                        cmap='coolwarm', s=120, vmin=vmin, vmax=vmax)

        # Add text label for error at each point
        for _, row in grouped.iterrows():
            ax.text(row["X_Real"], row["Y_Real"], f"{row['AbsError']:.2f}",
                    ha='center', va='center', fontsize=9, color='black', weight='bold')

        # Colorbar
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("|GT - Measured|")

        ax.set_xlabel("X")
        ax.set_xlim(-100, 640)
        ax.set_ylabel("Y")
        ax.set_ylim(-100, 550)
        ax.set_title(f"Mean Error for Anchor {anchor_id}")
        ax.grid(True)

        # Print stats
        mean_error = grouped["AbsError"].mean()
        std_error = grouped["AbsError"].std()
        print(f"Anchor {anchor_id} - Mean Grid Error: {mean_error:.2f}, Std: {std_error:.2f}")

    # Hide unused subplots if any
    for i in range(len(all_results), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
