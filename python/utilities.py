from ROOT import TFile, RDataFrame
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ray.air import Checkpoint, session
from ray.tune import CLIReporter

from networks import GhostNetwork, GhostNetworkExperiment


class GhostDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class QuietReporter(CLIReporter):
    def should_report(self, trials, done=False):
        return done


def load_data(filename: str):
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    dataframe = dataframe.Define("p", "abs(1.f/best_qop)")
    return dataframe.AsNumpy(), dataframe.GetColumnNames()


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


def training_loop(
    config,
    num_features,
    device,
    loss_function,
    training_dataset,
    validation_dataset,
    threshold=0.5,
):
    training_dataloader = DataLoader(training_dataset, batch_size=int(config["batch"]))
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=int(config["batch"])
    )
    # model
    model = GhostNetworkExperiment(
        num_features,
        l0=config["l0"],
        activation=config["activation"],
        # normalization=config["normalization"],
    )
    optimizer = select_optimizer(config, model)
    model.to(device)
    # checkpointing
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    num_epochs = config["epochs"]
    for epoch in range(start_epoch, num_epochs):
        inner_training_loop(
            model, training_dataloader, device, optimizer, loss_function
        )
        accuracy, loss = testing_loop(
            device, model, validation_dataloader, loss_function, threshold
        )
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)
        session.report(
            {"loss": loss, "accuracy": accuracy},
            checkpoint=checkpoint,
        )


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
    accuracy = 0.0
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
            epoch_loss = epoch_loss + loss.item()
            accuracy = accuracy + ((prediction > threshold).float() == y).float().mean()
    epoch_loss = epoch_loss / len(dataloader)
    accuracy = accuracy / len(dataloader)
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


def normalize(data: np.array) -> np.array:
    min = np.min(data)
    max = np.max(data)
    with np.errstate(divide='ignore'):
        if np.isfinite(np.random.rand(1) / (max - min)):
            return (data - min) / (max - min)
    return data
