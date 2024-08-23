import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from data import GhostDataset
from networks import (
    GhostNetwork,
    GhostNetworkWithNormalization,
    GhostNetworkWithManualNormalization,
)


normalization_layers = [nn.BatchNorm1d, nn.LazyBatchNorm1d, nn.SyncBatchNorm]
activation_layers = [
    nn.ReLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.LeakyReLU,
    nn.ELU,
    nn.Softmax,
    nn.Softmin,
]


def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        help="NumPy base filename containing dataset",
        type=str,
        required=True,
    )
    # parameters
    parser.add_argument(
        "--network", help="Network to train", type=int, choices=range(0, 3), default=0
    )
    parser.add_argument(
        "--normalization",
        help="Normalization layer (if necessary)",
        type=int,
        choices=range(0, 3),
        default=0,
    )
    parser.add_argument(
        "--activation",
        help="Activation layer (if necessary)",
        type=int,
        choices=range(0, 7),
        default=0,
    )
    parser.add_argument(
        "--optimizer", help="Optimizer", type=int, choices=range(0, 2), default=0
    )
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=256)
    parser.add_argument("--batch", help="Batch size.", type=int, default=2048)
    # misc
    parser.add_argument("--nocuda", help="Disable CUDA", action="store_true")
    parser.add_argument(
        "--save", help="Save the trained model to disk", action="store_true"
    )
    return parser.parse_args()


def __main__():
    arguments = command_line()
    use_cuda = not arguments.nocuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:0")
    else:
        device = torch.device("cpu")
    # load training, validation, and testing data sets
    data_train = np.load(f"{arguments.filename}_train_data.npy")
    labels_train = np.load(f"{arguments.filename}_train_labels.npy")
    print(f"Training set size: {len(data_train)}")
    data_test = np.load(f"{arguments.filename}_test_data.npy")
    labels_test = np.load(f"{arguments.filename}_test_labels.npy")
    print(f"Test set size: {len(data_test)}")
    # model
    num_features = data_train.shape[1]
    if arguments.network == 0:
        model = GhostNetwork(
            num_features,
            activation=activation_layers[arguments.activation],
        )
    elif arguments.network == 1:
        model = GhostNetworkWithNormalization(
            num_features,
            activation=activation_layers[arguments.activation],
            normalization=normalization_layers[arguments.normalization],
        )
    elif arguments.network == 2:
        model = GhostNetworkWithManualNormalization(
            num_features,
            matching=True,
            activation=activation_layers[arguments.activation],
            device=device,
        )
    model = model.to(device)
    # training
    training_dataset = GhostDataset(
        torch.tensor(data_train, dtype=torch.float32, device=device),
        torch.tensor(labels_train, dtype=torch.float32, device=device),
    )
    training_dataloader = DataLoader(training_dataset, batch_size=arguments.batch)
    if arguments.optimizer == 0:
        optimizer = torch.optim.Adam(model.parameters())
    elif arguments.optimizer == 1:
        optimizer = torch.optim.SGD(model.parameters())
    loss_function = nn.BCELoss()
    for _ in range(0, arguments.epochs):
        model.train()
        for x, y in training_dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            optimizer.step()
    # save model
    if arguments.save:
        print("Saving model to disk")
        model = model.to("cpu")
        if arguments.network == 2:
            model.normalization.shift = model.normalization.shift.to("cpu")
            model.normalization.scale = model.normalization.scale.to("cpu")
        print("Saving model to ONNX format")
        dummy_input = torch.randn(arguments.batch, num_features)
        dummy_input = dummy_input.to("cpu")
        torch.onnx.export(
            model,
            dummy_input,
            "ghost_model.onnx",
            input_names=["features"],
            output_names=["probabilities"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "probabilities": {0: "batch_size"},
            },
        )


if __name__ == "__main__":
    __main__()
