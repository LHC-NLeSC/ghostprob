import torch
from torch import nn


class GhostNetwork(nn.Module):
    def __init__(self, num_features):
        super(GhostNetwork, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.norm = nn.BatchNorm1d(num_features)
        self.layer0 = nn.Linear(num_features, int((num_features + 1) / 2))
        self.relu = nn.ReLU()
        self.output = nn.Linear(int((num_features + 1) / 2), 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.norm(x)
        x = self.layer0(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x


class RoelOriginalNetwork(nn.Module):
    def __init__(self, num_features):
        super(RoelOriginalNetwork, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.norm = nn.BatchNorm1d(num_features)
        self.layer0 = nn.Linear(num_features, int(1.5 * num_features))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.output = nn.Linear(int(1.5 * num_features), 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.norm(x)
        x = self.layer0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x
