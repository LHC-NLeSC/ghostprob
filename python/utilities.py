from ROOT import TFile, RDataFrame
import torch
import numpy as np
from torch.utils.data import Dataset


class GhostDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def load_data(filename: str):
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    dataframe = dataframe.Define("p", "abs(1.f/best_qop)")
    return dataframe.AsNumpy(), dataframe.GetColumnNames()


def shuffle_data(rng, data, labels):
    assert len(data) == len(labels)
    permutation = rng.permutation(len(data))
    return data[permutation], labels[permutation]


def training_loop(model, dataloader, loss_function, optimizer):
    model.train()
    for x, y in dataloader:
        prediction = model(x)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def testing_loop(model, dataloader, loss_function):
    accuracy = 0.0
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            prediction = model(x)
            loss = loss_function(prediction, y)
            epoch_loss = epoch_loss + loss.item()
            accuracy = accuracy + (prediction.round() == y).float().mean()
    epoch_loss = epoch_loss / len(dataloader)
    accuracy = accuracy / len(dataloader)
    return accuracy, epoch_loss


def remove_nans(data, labels):
    for _, column in enumerate(data):
        index = np.isfinite(column)
        if len(np.unique(index)) == 2:
            for j_col in range(len(data)):
                data[j_col] = data[j_col][index]
            labels = labels[index]
    return data, labels
