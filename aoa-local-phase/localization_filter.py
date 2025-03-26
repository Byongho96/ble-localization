import math
import numpy as np
import pandas as pd

def least_squares_triangulation(dic: dict, config: dict) -> pd.DataFrame:
    '''
    Least Squares Triangulation

    Parameters:
        dic (dict): Dictionary with anchor_id as key and DataFrame as value
        config (dict): Anchor config containing position and orientation

    Returns:
        pd.DataFrame: DataFrame with new ["X_Real", "Y_Real", "X_LS", "Y_LS"] columns
    '''
    anchor_ids = list(dic.keys())
    base_df = dic[anchor_ids[0]]

    estimated_positions = []

    # Iterate over time buckets in base anchor's DataFrame
    for row in base_df.itertuples(index=False):
        time_bucket = row.Time_Bucket

        # Check if all anchors have this time bucket
        if not all(time_bucket in dic[anchor_id]['Time_Bucket'].values for anchor_id in anchor_ids):
            continue

        # Extract the matched rows from all anchors
        rows = [dic[aid].loc[dic[aid]['Time_Bucket'] == time_bucket].iloc[0] for aid in anchor_ids]

        # Compute AoA for each anchor
        aoas = []
        for r in rows:
            orientation = config["anchors"][r["AnchorID"]]["orientation"]
            azimuth = r["Azimuth"]
            aoa_rad = math.radians(90 - azimuth - orientation)
            aoas.append(aoa_rad)

        # Set up least squares matrix
        H = np.array([[-math.tan(a), 1] for a in aoas])
        c = np.array([[r["Y_Real"] - r["X_Real"] * math.tan(a)] for r, a in zip(rows, aoas)])

        try:
            e = np.linalg.inv(H.T @ H) @ H.T @ c
        except np.linalg.LinAlgError:
            # Skip ill-conditioned cases
            continue

        x_LS, y_LS = e[0][0], e[1][0]
        x_real, y_real = rows[0]["X_Real"], rows[0]["Y_Real"]
        estimated_positions.append([x_real, y_real, x_LS, y_LS])

    

    return pd.DataFrame(estimated_positions, columns=["X_Real", "Y_Real", "X_LS", "Y_LS"])
