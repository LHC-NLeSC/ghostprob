import argparse
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import onnx2torch
import matplotlib.pyplot as plt

from utilities import (
    load_data,
    shuffle_data,
    GhostDataset,
    remove_nans,
    normalize,
    infer_probabilities,
)
from networks import (
    GhostNetwork,
    GhostNetworkWithNormalization,
    GhostNetworkWithManualNormalization,
)
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
    parser.add_argument(
        "--network", help="Network to use", type=int, choices=range(0, 3), default=0
    )
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
        # Shuffle
        data = np.hstack([data[i].reshape(len(data[0]), 1) for i in range(len(data))])
        rng = np.random.default_rng()
        data, labels = shuffle_data(rng, data, labels)
    else:
        data = np.load(f"{arguments.filename}_test_data.npy")
        labels = np.load(f"{arguments.filename}_test_labels.npy")
        print(f"Test set size: {len(data)}")
    test_dataset = GhostDataset(
        torch.tensor(data, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
    )
    # read model
    num_features = data.shape[1]
    with open(arguments.config, "rb") as file:
        model_config = pickle.load(file)
    if "onnx" in arguments.model:
        model = onnx2torch.convert(arguments.model)
    else:
        if arguments.network == 0:
            model = GhostNetwork(
                num_features,
                l0=model_config["l0"],
                activation=model_config["activation"],
            )
        elif arguments.network == 1:
            model = GhostNetworkWithNormalization(
                num_features,
                l0=model_config["l0"],
                activation=model_config["activation"],
                normalization=model_config["normalization"],
            )
        elif arguments.network == 2:
            model = GhostNetworkWithManualNormalization(
                num_features,
                l0=model_config["l0"],
                matching=True,
                activation=model_config["activation"],
                device=device,
            )
        weights = torch.load(arguments.model)
        model.load_state_dict(weights)
    model = model.to(device)
    print()
    print(model)
    print()
    test_dataloader = DataLoader(
        test_dataset, batch_size=model_config["batch"], shuffle=True
    )
    loss_function = nn.BCELoss()
    # Run inference and return probabilities
    probabilities = infer_probabilities(device, model, test_dataloader)
    # Plot histogram of probabilities
    counts, bins = np.histogram(probabilities)
    plt.stairs(counts, bins, fill=True)
    plt.show()


if __name__ == "__main__":
    __main__()
