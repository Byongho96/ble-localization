import os
import yaml
import pandas as pd
import data_processing as dp
import matplotlib.pyplot as plt
import seaborn as sns

def plot_variance_heatmaps():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))
    config['anchors'] = config['anchors']['0409']
    delta = config['delta']
    offset = config['offset']

    anchor_id = 1

    gt_path = os.path.join(base_dir, "../dataset/0409/gt/anchor1.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/0409/beacons/anchor1.csv")
    ms_df = pd.read_csv(ms_path)

    # Preprocess the data
    position = config['anchors'][anchor_id]['position']   
    orientation = config['anchors'][anchor_id]['orientation']
    
    ms_gt_df = dp.filter_with_position_ground_truth(gt_df, ms_df, offset)

    # Group by grid point and calculate variance
    grouped = ms_gt_df.groupby(['X_Real', 'Y_Real'])
    var_df = grouped.agg({
        'Azimuth': 'var',
        'Elevation': 'var',
        'RSSI': 'var'
    }).reset_index()

    # Metrics to plot
    metrics = ['Azimuth', 'Elevation', 'RSSI']
    rows, cols = 1, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i == 0:
            vmin = 0
            vmax = 400
        elif i == 1:
            vmin = 0
            vmax = 400
        else:
            vmin = 0
            vmax = 50

        ax = axes[i]
        scatter = ax.scatter(
            var_df["X_Real"],
            var_df["Y_Real"],
            c=var_df[metric],
            cmap='coolwarm',
            s=300,
            vmin=vmin,
            vmax=vmax
        )

        # Add color bar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(f'{metric} Variance')

        # Label points with variance values
        for _, row in var_df.iterrows():
            ax.text(
                row["X_Real"],
                row["Y_Real"],
                f'{row[metric]:.2f}',
                color='black',
                ha='center',
                va='center',
                fontsize=9,
            )

        ax.set_xlabel("X")
        ax.set_xlim(-100, 640)
        ax.set_ylabel("Y")
        ax.set_ylim(-100, 550)
        ax.set_title(f"{metric} Variance Map")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_variance_heatmaps()