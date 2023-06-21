from ROOT import TFile, RDataFrame
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ray.air import Checkpoint, session
from ray.tune import CLIReporter

from networks import GhostNetwork


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


def training_loop(
    config, num_features, device, loss_function, training_dataset, validation_dataset
):
    training_dataloader = DataLoader(training_dataset, batch_size=int(config["batch"]))
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=int(config["batch"])
    )
    # model
    model = GhostNetwork(num_features, l0=config["l0"])
    if config["optimizer"] == 0:
        if "cuda" in device.type:
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning"], fused=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning"])
    elif config["optimizer"] == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning"])
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
        model.train()
        for x, y in training_dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            optimizer.step()
        accuracy, loss = testing_loop(
            device, model, validation_dataloader, loss_function
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


def testing_loop(device, model, dataloader, loss_function):
    accuracy = 0.0
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = loss_function(prediction, y)
            epoch_loss = epoch_loss + loss.item()
            accuracy = accuracy + (prediction.round() == y).float().mean()
    epoch_loss = epoch_loss / len(dataloader)
    accuracy = accuracy / len(dataloader)
    return accuracy, epoch_loss


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
