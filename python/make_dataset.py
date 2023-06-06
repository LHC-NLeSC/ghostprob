import argparse
import numpy as np

from utilities import load_data, shuffle_data, remove_nans
from data import label, training_columns


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
        "--output", help="Prefix for the output file", type=str, required=True
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
    # split into real and ghost tracks
    data = np.hstack([data[i].reshape(len(data[0]), 1) for i in range(len(data))])
    data_ghost = data[labels == 1]
    data_real = data[labels == 0]
    print(
        f"Number of ghosts ({len(data_ghost)}) and real tracks ({len(data_real)}) in data set"
    )
    # select the same number of other particles as there are electrons
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
