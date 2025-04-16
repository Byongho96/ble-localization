import os
import yaml
import pandas as pd
import data_processing as dp
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt


window_size = 20

def calculate_covariance():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load config
    config = yaml.safe_load(open(os.path.join(base_dir, "../collected-config.yml")))
    config['anchors'] = config['anchors']['0409']
    delta = config['delta']
    offset = config['offset']

    anchor_id = 1

    gt_path = os.path.join(base_dir, "../dataset/0409/gt/anchor3.csv")
    gt_df = pd.read_csv(gt_path)

    ms_path = os.path.join(base_dir, "../dataset/0409/beacons/anchor3.csv")
    ms_df = pd.read_csv(ms_path)

    # Preprocess
    position = config['anchors'][anchor_id]['position']   
    orientation = config['anchors'][anchor_id]['orientation']
    
    ms_gt_df = dp.filter_with_position_ground_truth(gt_df, ms_df, offset)
    ms_gt_df = dp.calculate_aoa_ground_truth(ms_gt_df, position, orientation)

    # Rolling features
    ms_gt_df["Azimuth_Error_Abs"] = (ms_gt_df["Azimuth"] - ms_gt_df["Azimuth_Real"]).abs()
    ms_gt_df["Azimuth_Var"] = ms_gt_df["Azimuth"].rolling(window=window_size).var()
    ms_gt_df["Error_Mean"] = ms_gt_df["Azimuth_Error_Abs"].rolling(window=window_size).mean()


    # Drop rows with NaNs from rolling
    valid_df = ms_gt_df.dropna(subset=["Azimuth_Var", "Azimuth_Error_Abs", "Error_Mean"])

    # Correlation
    pearson_corr, _ = pearsonr(valid_df["Azimuth_Var"], valid_df["Error_Mean"])
    spearman_corr, _ = spearmanr(valid_df["Azimuth_Var"], valid_df["Error_Mean"])


    # Save Azimuth_Var and Error_Mean to CSV
    output_path = os.path.join(base_dir, "../dataset/test.csv")
    valid_df[["Azimuth_Var", "Error_Mean"]].to_csv(output_path, index=False)

    print(f"✅ Pearson correlation: {pearson_corr:.4f}")
    print(f"✅ Spearman correlation: {spearman_corr:.4f}")

    # 📊 Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pearson - linear scatter with regression
    sns.regplot(
        data=valid_df,
        x="Azimuth_Var",
        y="Error_Mean",
        ax=axes[0],
        scatter_kws={"alpha": 0.1},
        line_kws={"color": "red"},
    )
    axes[0].set_title(f"Pearson (r = {pearson_corr:.2f})")
    axes[0].set_xlabel("Azimuth Variance")
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].grid(True)

    # Spearman - rank-based scatter
    ranks_df = valid_df[["Azimuth_Var", "Error_Mean"]].rank()
    sns.regplot(
        x=ranks_df["Azimuth_Var"],
        y=ranks_df["Error_Mean"],
        ax=axes[1],
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "green"},
    )
    axes[1].set_title(f"Spearman (ρ = {spearman_corr:.2f})")
    axes[1].set_xlabel("Rank(Azimuth Variance)")
    axes[1].set_ylabel("Rank(Mean Error)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calculate_covariance()
