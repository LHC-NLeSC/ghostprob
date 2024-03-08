import argparse
import pickle
import numpy as np
import torch
import onnx2torch
import matplotlib.pyplot as plt
from captum import attr

from utilities import load_data, shuffle_data, remove_nans, normalize
from networks import GhostNetwork, GhostNetworkWithNormalization
from data import label, training_columns_forward, training_columns_matching


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filename", help="File containing the data set", type=str, required=True
    )
    parser.add_argument("--nocuda", help="Disable CUDA", action="store_true")
    parser.add_argument(
        "--model",
        help="Name of the file containing the model.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config",
        help="Name of the file containing the model configuration.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--normalize",
        help="Normalize input data before inference.",
        action="store_true",
    )
    parser.add_argument(
        "--track",
        help="Forward or Matching",
        type=str,
        choices=["forward", "matching"],
        required=True,
    )
    parser.add_argument("--int8", help="INT8 quantization.", action="store_true")
    return parser.parse_args()


def __main__():
    arguments = command_line()
    if not arguments.nocuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if ".root" in arguments.filename:
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
                print("Missing data.")
                return
        # create dataset
        data = [dataframe[column] for column in training_columns]
        # Remove NaNs
        data, labels = remove_nans(data, labels)
        # Normalize
        if arguments.normalize:
            for feature_id in range(len(data)):
                data[feature_id] = normalize(data[feature_id])
        # split into ghost and real tracks
        data = np.hstack([data[i].reshape(len(data[0]), 1) for i in range(len(data))])
        data_ghost = data[labels == 1]
        data_real = data[labels == 0]
        print(
            f"Number of ghost ({len(data_ghost)}) and real tracks ({len(data_real)}) in data set"
        )
        # select the same number of other real tracks as there are ghosts
        rng = np.random.default_rng()
        rng.shuffle(data_real)
        data_real = data_real[: len(data_ghost)]
        # create data set
        data = np.vstack((data_ghost, data_real))
        labels_ghost = np.ones((len(data_ghost), 1), dtype=int)
        labels_real = np.zeros((len(data_real), 1), dtype=int)
        labels = np.vstack((labels_ghost, labels_real))
        data, labels = shuffle_data(rng, data, labels)
    else:
        data = np.load(f"{arguments.filename}_test_data.npy")
        labels = np.load(f"{arguments.filename}_test_labels.npy")
    print(f"Test set size: {len(data)}")
    # read model
    num_features = data.shape[1]
    with open(arguments.config, "rb") as file:
        model_config = pickle.load(file)
    if arguments.int8:
        model = torch.load(arguments.model)
    else:
        if "onnx" in arguments.model:
            model = onnx2torch.convert(arguments.model)
        else:
            model = GhostNetwork(
                num_features=num_features,
                l0=model_config["l0"],
                activation=model_config["activation"],
                normalization=model_config["normalization"],
            )
            weights = torch.load(arguments.model)
            model.load_state_dict(weights)
    model.to(device)
    print()
    print(model)
    print()
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    # Integrated Gradients
    interpreter = attr.IntegratedGradients(model)
    attribution = interpreter.attribute(data_tensor)
    plt.bar(
        range(len(training_columns)),
        attribution.mean(axis=0),
        width=0.85,
    )
    plt.xticks(range(len(training_columns)), training_columns, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Attribution")
    plt.title("Integrated Gradients")
    plt.show()
    # Feature Ablation
    interpreter = attr.FeatureAblation(model)
    attribution = interpreter.attribute(data_tensor)
    plt.bar(
        range(len(training_columns)),
        attribution.mean(axis=0),
        width=0.85,
    )
    plt.xticks(range(len(training_columns)), training_columns, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Attribution")
    plt.title("Feature Ablation")
    plt.show()


if __name__ == "__main__":
    __main__()
