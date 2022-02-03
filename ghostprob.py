import argparse
import numpy as np
import tensorflow as tf
from ROOT import TFile, RDataFrame


# Columns to use for training
columns = ["p", "z", "x", "y", "tx", "ty", "qop", "first_qop", "best_qop", "best_pt", "kalman_ip", "kalman_ipx",
        "kalman_ipy", "kalman_ip_chi2", "kalman_docaz", "velo_ip", "velo_ipx", "velo_ipy", "velo_ip_chi2", 
        "velo_docaz", "chi2", "chi2V", "chi2UT", "chi2T", "ndof", "ndofV", "ndofT", "nUT", "mcp_p"]

# Bounds for the values
bounds = {"x": (-10., 10.),
          "y": (-10., 10.),
          "tx": (-0.3, 0.3),
          "ty": (-0.3, 0.3),
          "best_pt": (0, 15000),
          "kalman_ip_chi2": (-0.5, 10000.5),
          "kalman_docaz": (-0.5, 25.5),
          "chi2dof": (0, 400),
          "chi2Vdof": (0, 150),
          "chi2Tdof": (0, 150)}


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="File with validator data", type=str, required=True)
    parser.add_argument("--bound", help="Filter entries outside the boundaries", action="store_true")
    return parser.parse_args()


def shuffle_data(rng, data, labels):
    assert(len(data) == len(labels))
    permutation = rng.permutation(len(data))
    return data[permutation], labels[permutation]


def __main__():
    arguments = command_line()

    # Load file into a table
    kalman_file = TFile(arguments.filename)
    df = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file, columns).Define("p", "abs(1.f/best_qop)")

    # Filter out of bounds data
    if arguments.bound:
        for column in columns:
            if column in bounds:
                lower, upper = bounds[column]
                df = df.Filter(f"{column} > {lower} && {column} < {upper}")

    # Convert table to numpy
    np_df = df.AsNumpy()
    print(f"Columns in the table: {len(np_df)}")
    print(f"Columns for training: {len(columns)}")
    print(f"Entries in the table: {len(np_df['p'])}")
    labels = np_df["ghost"].astype(int)
    data = [np_df[column] for column in columns]
    data = np.hstack([data[i].reshape(len(np_df["p"]), 1) for i in range(len(data))])

    # Split into real tracks and ghosts
    data_tracks = data[labels == 0]
    data_ghosts = data[labels == 1]
    
    # Shuffle tracks and select the same number of real tracks and ghosts
    rng = np.random.default_rng()
    rng.shuffle(data_tracks)
    data_tracks = data_tracks[:len(data_ghosts)]
    print(f"Number of real tracks ({len(data_tracks)}) and ghost tracks ({len(data_ghosts)})")

    # Assemble training data set and labels
    data = np.vstack((data_tracks, data_ghosts))
    labels_tracks = np.zeros((len(data_ghosts), 1), dtype=int)
    labels_ghosts = np.ones((len(data_ghosts), 1), dtype=int)
    labels = np.vstack((labels_tracks, labels_ghosts))
    data, labels = shuffle_data(rng, data, labels)

    # Keep 20% of data for testing
    test_point = int(len(data) * 0.8)
    print(f"Training set size: {test_point}")
    print(f"Test set size: {len(data) - test_point}")

    # Model
    features = len(columns)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, input_dim=features, activation="relu"),
        tf.keras.layers.Dense(units=1)
        ])
    model.summary()
    model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"]
            )

    # Training
    num_epochs = 100
    batch_size = 256
    training_history = model.fit(
            data[:test_point],
            labels[:test_point],
            validation_split=0.2,
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=0
            )

    # Evaluation
    loss, accuracy = model.evaluate(data[test_point:], labels[test_point:], verbose=0)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

if __name__ == "__main__":
    __main__()
