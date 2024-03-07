import torch
from torch import nn


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
