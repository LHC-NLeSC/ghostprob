import argparse
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import onnx2torch

from utilities import load_data, shuffle_data, GhostDataset, testing_loop
from networks import GhostNetwork
import data


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
    parser.add_argument("--batch", help="Batch size", type=int, default=512)
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
    if ".root" in arguments.filename:
        dataframe, columns = load_data(arguments.filename)
        print(f"Columns in the table: {len(dataframe)}")
        print(columns)
        if data.label not in columns:
            print("Missing labels.")
            return
        labels = dataframe[data.label].astype(int)
        for column in data.training_columns:
            if column not in columns:
                print("Missing data.")
                return
        # create dataset
        data = [dataframe[column] for column in data.training_columns]
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
        # create training and testing data set
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
        torch.tensor(datadtype=torch.float32, device=device),
        torch.tensor(labelsdtype=torch.float32, device=device),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=arguments.batch, shuffle=True)
    # read model
    num_features = data.shape[1]
    if arguments.int8:
        model = torch.load(arguments.model)
    else:
        if "onnx" in arguments.model:
            model = onnx2torch.convert(arguments.model)
        else:
            model = GhostNetwork(num_features=num_features)
            weights = torch.load(arguments.model)
            model.load_state_dict(weights)
    print(f"Device: {device}")
    model.to(device)
    print()
    print(model)
    print()
    # inference
    loss_function = nn.BCELoss()
    start_time = perf_counter()
    accuracy, loss = testing_loop(model, test_dataloader, loss_function)
    end_time = perf_counter()
    print(f"Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Loss: {loss}")
    print(f"Inference time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    __main__()
