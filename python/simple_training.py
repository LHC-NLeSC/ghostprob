import argparse
from collections import namedtuple
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ROOT import TFile, RDataFrame


class GhostDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class GhostNetwork(nn.Module):
    def __init__(self, num_features, l0=32, activation=nn.ReLU):
        super(GhostNetwork, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.layer0 = nn.Linear(num_features, l0)
        self.activation = activation()
        self.output = nn.Linear(l0, 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.layer0(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x


class GhostNetworkWithNormalization(nn.Module):
    def __init__(
        self, num_features, l0=32, activation=nn.ReLU, normalization=nn.BatchNorm1d
    ):
        super(GhostNetworkWithNormalization, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.norm = normalization(num_features, track_running_stats=True)
        self.layer0 = nn.Linear(num_features, l0)
        self.activation = activation()
        self.output = nn.Linear(l0, 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.norm(x)
        x = self.layer0(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x


class NormalizationLayer(nn.Module):
    def __init__(
        self, matching: bool = True, device: torch.device = torch.device("cpu")
    ):
        super(NormalizationLayer, self).__init__()
        self.matching = matching
        if matching:
            self.shift = torch.tensor(
                [
                    12.615314483642578,
                    0.0,
                    -0.00042932009091600776,
                    10.0,
                    3.0,
                    2.0,
                    4.006599192507565e-05,
                    -0.4567539691925049,
                    -0.019999638199806213,
                    -19.99979019165039,
                    -149.963134765625,
                    0.004197417292743921,
                    -0.0004279722925275564,
                ],
                dtype=torch.float32,
                device=device,
            )
            self.scale = torch.tensor(
                [
                    155663427.38468552,
                    25787.57421875,
                    0.000783147057518363,
                    2,
                    20,
                    40,
                    1.9999514701485168,
                    1.02848482131958,
                    0.03999875485897064,
                    39.99972915649414,
                    299.958740234375,
                    1301.8299090280198,
                    0.0007813300180714577,
                ],
                dtype=torch.float32,
                device=device,
            )

    def forward(self, x):
        if self.matching:
            x[:,] = (x[:,] - self.shift) / self.scale
        return x


class GhostNetworkWithManualNormalization(nn.Module):
    def __init__(
        self,
        num_features,
        l0=32,
        matching=True,
        activation=nn.ReLU,
        device: torch.device = torch.device("cpu"),
    ):
        super(GhostNetworkWithManualNormalization, self).__init__()
        self.matching = matching
        self.quant = torch.quantization.QuantStub()
        self.normalization = NormalizationLayer(matching, device)
        self.layer0 = nn.Linear(num_features, l0)
        self.activation = activation()
        self.output = nn.Linear(l0, 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.normalization(x)
        x = self.layer0(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x

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

DataLabels = namedtuple("DataLabels", ["data", "labels"])

label = "ghost"
training_columns_forward = [
    "best_pt",
    "chi2",
    "first_qop",
    "n_scifi",
    "n_velo",
    "ndof",
    "pr_forward_quality",
    "qop",
]
training_columns_matching = [
    "best_pt",
    "chi2",
    "first_qop",
    "n_scifi",
    "n_velo",
    "ndof",
    "pr_match_chi2",
    "pr_match_dtx",
    "pr_match_dty",
    "pr_match_dx",
    "pr_match_dy",
    "pr_seed_chi2",
    "qop",
]
boundaries = {"best_pt": (0, 1e4), "chi2": (0, 400)}


def load_data(filename: str):
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    return dataframe.AsNumpy(), dataframe.GetColumnNames()


def shuffle_data(rng, data, labels):
    assert len(data) == len(labels)
    permutation = rng.permutation(len(data))
    return data[permutation], labels[permutation]


def remove_nans(data, labels):
    corrected_columns = 0
    for _, column in enumerate(data):
        index = np.isfinite(column)
        if len(np.unique(index)) == 2:
            corrected_columns += 1
            for j_col in range(len(data)):
                data[j_col] = data[j_col][index]
            labels = labels[index]
    print(f"Number of columns with NaN: {corrected_columns}")
    return data, labels


def normalize(data, min_max=None) -> np.ndarray:
    if min_max is None:
        minimum = np.min(data)
        maximum = np.max(data)
    else:
        minimum, maximum = min_max
    with np.errstate(divide="ignore"):
        if np.isfinite(np.random.rand(1) / (maximum - minimum)):
            return (data - minimum) / (maximum - minimum)
    return data


def dataset(arguments: argparse.Namespace) -> tuple[DataLabels, DataLabels, DataLabels]:
    dataframe, columns = load_data(arguments.filename)
    print(f"Columns in the table: {len(dataframe)}")
    print(columns)
    if label not in columns:
        raise ValueError(f"Missing labels: {label} ∉ {columns}")
    labels = dataframe[label].astype(int)
    if arguments.track.lower() == "forward":
        training_columns = training_columns_forward
    else:
        training_columns = training_columns_matching
    for column in training_columns:
        if column not in columns:
            raise ValueError(f"Missing training data: {column} ∉ {columns}")
    trainining_columns = training_columns
    print(f"Columns for training: {len(trainining_columns)}")
    print(f"Entries in the table: {len(dataframe[label])}")
    data = [dataframe[column] for column in trainining_columns]

    # Remove NaNs
    data, labels = remove_nans(data, labels)

    # Normalize each feature
    features: dict[str, list[str] | dict[str, tuple[float, float]]] = {
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
    with open("feature-offsets-and-scales.json", "w") as jf:
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

    # train, validation, test datasets
    print(f"Training dataset size: {len(data_train)}")
    print(f"Validation dataset size: {len(data_validation)}")
    print(f"Test dataset size: {len(data_test)}")
    training_set = DataLabels(data_train, labels_train)
    test_set = DataLabels(data_test, labels_test)
    validation_set = DataLabels(data_validation, labels_validation)
    return (training_set, test_set, validation_set)


def train(arguments: argparse.Namespace):
    use_cuda = not arguments.nocuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # load training, validation, and testing data sets
    training_set, test_set, *_ = dataset(arguments)
    data_train, labels_train = training_set
    print(f"Training set size: {len(data_train)}")
    data_test, labels_test = test_set
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
    else:  # arguments.network == 2
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
    else:  # arguments.optimizer == 1
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
        "--normalize", help="Normalize features in [0, 1].", action="store_true"
    )
    parser.add_argument(
        "--track",
        help="Forward or Matching",
        type=str,
        choices=["forward", "matching"],
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
    args = parser.parse_args()
    train(args)
