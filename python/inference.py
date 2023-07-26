import argparse
import pickle
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import onnx2torch

from utilities import load_data, shuffle_data, GhostDataset, testing_loop, remove_nans
from networks import GhostNetwork, GhostNetworkExperiment
from data import label, training_columns


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
    )
    parser.add_argument(
        "--int8", help="Quantize the trained model to INT8", action="store_true"
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
        for column in training_columns:
            if column not in columns:
                print("Missing data.")
                return
        # create dataset
        data = [dataframe[column] for column in training_columns]
        # Remove NaNs
        data, labels = remove_nans(data, labels)
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
    test_dataset = GhostDataset(
        torch.tensor(data, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
    )
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
    test_dataloader = DataLoader(
        test_dataset, batch_size=model_config["batch"], shuffle=True
    )
    # inference
    for threshold in [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]:
        loss_function = nn.BCELoss()
        start_time = perf_counter()
        accuracy, _ = testing_loop(
            device, model, test_dataloader, loss_function, threshold
        )
        end_time = perf_counter()
        print(f"Threshold: {threshold}")
        print(f"Accuracy: {accuracy * 100.0:.2f}%")
        print(f"Inference time: {end_time - start_time:.2f} seconds")
        print()


if __name__ == "__main__":
    __main__()
