import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

from utilities import load_data, shuffle_data, remove_nans, normalize
from data import label, training_columns_forward, training_columns_matching, boundaries


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        help="ROOT file containing the data set",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fraction",
        help="Fraction of ghosts to include in dataset",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output", help="Prefix for the output file", type=str, required=True
    )
    parser.add_argument(
        "--plot", help="Plot the histogram for each feature.", action="store_true"
    )
    parser.add_argument(
        "--normalize", help="Normalize features in [0, 1].", action="store_true"
    )
    parser.add_argument(
        "--track",
        help="Forward or Matching",
        type=str,
        choices=["forward", "matching"],
        required=True,
    )
    return parser.parse_args()


def __main__():
    arguments = command_line()
    dataframe, columns = load_data(arguments.filename)
    print(f"Columns in the table: {len(dataframe)}")
    print(columns)
    if label not in columns:
        print("Missing labels.")
        return
    labels = dataframe[label].astype(int)
    if arguments.track.lower() == "forward":
        training_columns = training_columns_forward
    elif arguments.track.lower() == "matching":
        training_columns = training_columns_matching
    for column in training_columns:
        if column not in columns:
            print("Missing training data.")
            return
    trainining_columns = training_columns
    print(f"Columns for training: {len(trainining_columns)}")
    print(f"Entries in the table: {len(dataframe[label])}")
    data = [dataframe[column] for column in trainining_columns]
    # Remove NaNs
    data, labels = remove_nans(data, labels)
    # Plot histograms and interval
    if arguments.plot:
        for column in training_columns:
            feature = dataframe[column]
            print(f"Feature: {column} ({np.min(feature)}, {np.max(feature)})")
            counts, bins = np.histogram(feature)
            plt.title(f"Feature: {column}")
            plt.stairs(counts, bins, fill=True)
            plt.show()
    # Normalize each feature
    features = {
        "features": [training_columns[feature_id] for feature_id in range(len(data))]
    }
    if arguments.normalize:
        offsets_and_scales = {}
        for feature_id in range(len(data)):
            data_min_max = (
                float(np.min(data[feature_id])),
                float(np.max(data[feature_id])),
            )
            min_max = boundaries.get(training_columns[feature_id], data_min_max)
            offsets_and_scales[training_columns[feature_id]] = (
                min_max[0],
                min_max[1] - min_max[0],
            )
            print(f"Feature: {feature_id} {data_min_max}")
            data[feature_id] = normalize(data[feature_id], min_max)
            print(
                f"Feature: {feature_id} ({np.min(data[feature_id])}, {np.max(data[feature_id])})"
            )
            print()
        features["offsets_and_scales"] = offsets_and_scales
    with open(f"{arguments.output}_features.json", "w") as jf:
        json.dump(features, jf, indent=4)

    # split into real and ghost tracks
    data = np.hstack([data[i].reshape(len(data[0]), 1) for i in range(len(data))])
    data_ghost = data[labels == 1]
    data_real = data[labels == 0]
    print(
        f"Number of ghosts ({len(data_ghost)}) and real tracks ({len(data_real)}) in data set"
    )
    data_ghost = data_ghost[: int(arguments.fraction * len(data_ghost))]
    # select the same number of real tracks as there are ghosts
    rng = np.random.default_rng()
    rng.shuffle(data_real)
    max_train = int(0.6 * len(data_ghost))
    max_validation = int(0.8 * len(data_ghost))
    data_train = np.vstack((data_ghost[:max_train], data_real[:max_train]))
    labels_ghost = np.ones((len(data_ghost[:max_train]), 1), dtype=int)
    labels_real = np.zeros((len(data_real[:max_train]), 1), dtype=int)
    labels_train = np.vstack((labels_ghost, labels_real))
    data_train, labels_train = shuffle_data(rng, data_train, labels_train)
    data_validation = np.vstack(
        (data_ghost[max_train:max_validation], data_real[max_train:max_validation])
    )
    labels_ghost = np.ones((len(data_ghost[max_train:max_validation]), 1), dtype=int)
    labels_real = np.zeros((len(data_real[max_train:max_validation]), 1), dtype=int)
    labels_validation = np.vstack((labels_ghost, labels_real))
    data_validation, labels_validation = shuffle_data(
        rng, data_validation, labels_validation
    )
    data_test = np.vstack(
        (data_ghost[max_validation:], data_real[max_validation : len(data_ghost)])
    )
    labels_ghost = np.ones((len(data_ghost[max_validation:]), 1), dtype=int)
    labels_real = np.zeros(
        (len(data_real[max_validation : len(data_ghost)]), 1), dtype=int
    )
    labels_test = np.vstack((labels_ghost, labels_real))
    data_test, labels_test = shuffle_data(rng, data_test, labels_test)
    # save train, validation, test datasets
    print(f"Training dataset size: {len(data_train)}")
    print(f"Validation dataset size: {len(data_validation)}")
    print(f"Test dataset size: {len(data_test)}")
    np.save(f"{arguments.output}_train_data", data_train)
    np.save(f"{arguments.output}_train_labels", labels_train)
    np.save(f"{arguments.output}_valid_data", data_validation)
    np.save(f"{arguments.output}_valid_labels", labels_validation)
    np.save(f"{arguments.output}_test_data", data_test)
    np.save(f"{arguments.output}_test_labels", labels_test)


if __name__ == "__main__":
    __main__()
