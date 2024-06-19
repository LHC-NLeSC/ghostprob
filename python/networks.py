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


class GhostNetworkWithManualNormalization(nn.Module):
    def __init__(self, num_features, l0=32, matching=True, activation=nn.ReLU):
        super(GhostNetworkWithManualNormalization, self).__init__()
        self.matching = matching
        self.quant = torch.quantization.QuantStub()
        self.layer0 = nn.Linear(num_features, l0)
        self.activation = activation()
        self.output = nn.Linear(l0, 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def normalization(self, x):
        if self.matching:
            x[:, 0] = (x[:, 0] - 12.615314483642578) / 155663427.38468552
            x[:, 1] = x[:, 1] / 25787.57421875
            x[:, 2] = (x[:, 2] + 0.00042932009091600776) / 0.000783147057518363
            x[:, 3] = (x[:, 3] - 10) / 2
            x[:, 4] = (x[:, 4] - 3) / 20
            x[:, 5] = (x[:, 5] - 2) / 40
            x[:, 6] = (x[:, 6] - 4.006599192507565e-05) / 1.9999514701485168
            x[:, 7] = (x[:, 7] + 0.4567539691925049) / 1.02848482131958
            x[:, 8] = (x[:, 8] + 0.019999638199806213) / 0.03999875485897064
            x[:, 9] = (x[:, 9] + 19.99979019165039) / 39.99972915649414
            x[:, 10] = (x[:, 10] + 149.963134765625) / 299.958740234375
            x[:, 11] = (x[:, 11] - 0.004197417292743921) / 1301.8299090280198
            x[:, 12] = (x[:, 12] + 0.0004279722925275564) / 0.0007813300180714577
        else:
            pass
        return x

    def forward(self, x):
        x = self.quant(x)
        x = self.normalization(x)
        x = self.layer0(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x
