import pandas as pd
import pickle as pkl

from utils.baseline_config import (
    FEATURE_FORMAT,
)

import numpy as np


if __name__ == "__main__":
    feature_file = "features/val.pkl"
    print('loading pickle')
    df = pd.read_pickle(feature_file)
    print('pickle loaded, length %d' % df['SEQUENCE'].shape[0])

    gt_features = ['X', 'Y']

    input_feature_idx = [
        FEATURE_FORMAT[feature] for feature in gt_features
    ]

    input_features_data = np.stack(
        df["FEATURES"].values)[:, :, input_feature_idx].astype("float")

    data = input_features_data[:, 20:]

    forecasted_trajectories = {}
    indexes = df['SEQUENCE'].tolist()
    for i in range(len(indexes)):
        seq_id = int(indexes[i])
        forecasted_trajectories[seq_id] = data[i]

    with open('features/val_gt.pkl', "wb") as f:
        pkl.dump(forecasted_trajectories, f)
