import os
import tempfile
from ROOT import TFile, RDataFrame
import torch
import numpy as np
from torch.utils.data import DataLoader
from ray import train

from networks import (
    GhostNetwork,
    GhostNetworkWithNormalization,
    GhostNetworkWithManualNormalization,
)

from data import GhostDataset


def load_data(filename: str):
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    return dataframe.AsNumpy(), dataframe.GetColumnNames()


def save_data(filename: str):
    pass


def shuffle_data(rng, data, labels):
    assert len(data) == len(labels)
    permutation = rng.permutation(len(data))
    return data[permutation], labels[permutation]


def select_optimizer(config, model):
    if config["optimizer"] == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning"])
    elif config["optimizer"] == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning"])
    return optimizer


def training_loop(config):
    if config["use_cuda"]:
        device = torch.device(f"cuda:0")
    else:
        device = torch.device("cpu")

    data_train = config["data_train"]
    labels_train = config["labels_train"]
    training_dataset = GhostDataset(
        torch.tensor(data_train, dtype=torch.float32, device=device),
        torch.tensor(labels_train, dtype=torch.float32, device=device),
    )
    data_validation = config["data_validation"]
    labels_validation = config["labels_validation"]
    validation_dataset = GhostDataset(
        torch.tensor(data_validation, dtype=torch.float32, device=device),
        torch.tensor(labels_validation, dtype=torch.float32, device=device),
    )

    training_dataloader = DataLoader(training_dataset, batch_size=int(config["batch"]))
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=int(config["batch"])
    )
    # model
    if config["network"] == 0:
        model = GhostNetwork(
            config["num_features"],
            l0=config["l0"],
            activation=config["activation"],
        )
    elif config["network"] == 1:
        model = GhostNetworkWithNormalization(
            config["num_features"],
            l0=config["l0"],
            activation=config["activation"],
            normalization=config["normalization"],
        )
    elif config["network"] == 2:
        model = GhostNetworkWithManualNormalization(
            config["num_features"],
            l0=config["l0"],
            matching=True,
            activation=config["activation"],
            device=device,
        )
    optimizer = select_optimizer(config, model)
    model = model.to(device)
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "ghost_checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    start_epoch = 0
    num_epochs = config["epochs"]

    def report(epoch, metrics, force=False):
        if (epoch % 10 == 0 and epoch != 0) or force:
            with tempfile.TemporaryDirectory(
                dir=config["tmp_path"]
            ) as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "ghost_checkpoint.pt")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
                checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(metrics=metrics, checkpoint=checkpoint)
        else:
            train.report(metrics=metrics)

    metrics = None
    for epoch in range(start_epoch, num_epochs):
        inner_training_loop(
            model,
            training_dataloader,
            device,
            optimizer,
            config["loss_function"],
        )
        accuracy, loss = testing_loop(
            device,
            model,
            validation_dataloader,
            config["loss_function"],
            config["threshold"],
        )
        metrics = {"loss": loss, "accuracy": accuracy}
        report(epoch, metrics, epoch == num_epochs - 1)


def inner_training_loop(model, dataloader, device, optimizer, loss_function):
    model.train()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()


def testing_loop(device, model, dataloader, loss_function, threshold=0.5):
    correct = 0
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            if len(y.shape) != len(prediction.shape):
                if len(y.shape) > len(prediction.shape):
                    y = y.squeeze(-1)
                else:
                    prediction = prediction.squeeze(-1)
            loss = loss_function(prediction, y)
            epoch_loss = epoch_loss + loss
            correct += ((prediction > threshold).int() == y).sum().item()
    epoch_loss = float(epoch_loss / len(dataloader))
    accuracy = float(correct / len(dataloader.dataset))
    return accuracy, epoch_loss


def testing_accuracy(device, model, dataloader, threshold):
    model.eval()
    # TP, TN, FP, FN
    accuracy = [0, 0, 0, 0]
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            prediction = (prediction > threshold).int()
            for i in range(0, len(prediction)):
                # True Positive
                if prediction[i] == 1 and y.int()[i] == 1:
                    accuracy[0] += 1
                # True Negative
                elif prediction[i] == 0 and y.int()[i] == 0:
                    accuracy[1] += 1
                # False Positive
                elif prediction[i] == 1 and y.int()[i] == 0:
                    accuracy[2] += 1
                # False Negative
                elif prediction[i] == 0 and y.int()[i] == 1:
                    accuracy[3] += 1
    return accuracy


def infer_probabilities(device, model, dataloader):
    probabilities = list()
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x).cpu()
            for probability in prediction:
                probabilities.append(probability)
    return probabilities


def infer_probabilities_ort(ort_session, dataloader):
    probabilities = list()
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    for x, y in dataloader:
        prediction = ort_session.run([output_name], {input_name: x.numpy()})[0]
        for probability in prediction:
            probabilities.append(probability)
    return probabilities


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


def normalize(data, min_max=None) -> np.array:
    if min_max is None:
        minimum = np.min(data)
        maximum = np.max(data)
    else:
        minimum, maximum = min_max
    with np.errstate(divide="ignore"):
        if np.isfinite(np.random.rand(1) / (maximum - minimum)):
            return (data - minimum) / (maximum - minimum)
    return data
